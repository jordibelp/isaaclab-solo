#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compare rendered outcome regions with the actual task expressions and UI toggles."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from playwright.sync_api import sync_playwright
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("page", help="Live viewer URL or standalone HTML path")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = Path(__file__).resolve().parent.parents[3] / "source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper"
tree = ast.parse((folder / "borinot_goalkeeper_env.py").read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sample_physics_events")
names = {"returned", "goal_front", "outside", "safely_wide", "landed_in_front", "escaped"}
expressions = [n for n in method.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
code = compile(ast.Module(body=expressions, type_ignores=[]), "actual_task_regions", "exec")
spec = importlib.util.spec_from_file_location("task_math", folder / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)
checks, errors = [], []


def check(name, condition):
    assert bool(condition), name
    checks.append(name)


def inside_boxes(points, boxes):
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        inside |= ((points > box["min"]) & (points < box["max"])).all(-1)
    return inside


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                               args=["--no-sandbox", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1500, "height": 1050})
    page.on("pageerror", lambda error: errors.append(str(error)))
    offline = not args.page.startswith("http")
    page.goto(Path(args.page).resolve().as_uri() if offline else args.page)
    page.wait_for_function("window.viewer && window.viewer.outcomeZones")
    original = page.evaluate("viewer.snapshot()")
    zones = page.evaluate("viewer.outcomeZones()")
    cfg = SimpleNamespace(**original["config"])
    check("all five independently controlled zones", len(zones["zones"]) == 5)
    line = next(n for n in zones["zones"]["returned"]["lines"] if n["name"] == "returned-floor-line")
    vertices = np.asarray(line["vertices"]).reshape(-1, 3)
    check("actual floor line at keeper plane plus clearance distance", np.allclose(vertices[:, 0], cfg.keeper_plane_x + cfg.clear_distance_m, atol=1e-7))
    check("actual floor line spans displayed floor", np.allclose(vertices[:, 1], zones["display_bounds"]["y"], atol=1e-7))
    check("actual floor line above floor to avoid z-fighting", (vertices[:, 2] > 0).all() and (vertices[:, 2] < .02).all())

    rng = np.random.default_rng(42)
    low = [zones["display_bounds"][axis][0] + .0001 for axis in "xyz"]
    high = [zones["display_bounds"][axis][1] - .0001 for axis in "xyz"]
    points = rng.uniform(low, high, (10000, 3))
    velocity = rng.uniform(-2, 2, (10000, 3))
    env = SimpleNamespace(cfg=cfg, _hit=torch.ones(10000, dtype=torch.bool),
                          _flags={"floor": torch.ones(10000, dtype=torch.bool)})
    scope = {"self": env, "position": torch.tensor(points), "velocity": torch.tensor(velocity)}
    exec(code, scope)
    check("10000 wide-region points match real task code", np.array_equal(inside_boxes(points, zones["zones"]["wide"]["boxes"]), scope["safely_wide"].numpy()))
    check("10000 miss-region points match real task code", np.array_equal(inside_boxes(points, zones["zones"]["miss"]["boxes"]), scope["escaped"].numpy()))
    check("10000 return points and speeds match real task code", np.array_equal((points[:, 0] > zones["returned_x_m"]) & (velocity[:, 0] > cfg.clear_speed_m_s), scope["returned"].numpy()))
    floor_box = zones["zones"]["floor"]["boxes"][0]
    check("10000 floor-save X positions match real task code", np.array_equal(points[:, 0] > floor_box["min"][0], scope["landed_in_front"].numpy()))
    check("floor fallback threshold includes source slack", zones["floor_center_height_m"] == cfg.ball_radius_m + cfg.outcome_rules["floor_slack_m"])

    # Check strict inequalities at the return position/speed boundary.
    x, speed = zones["returned_x_m"], cfg.clear_speed_m_s
    scope["position"] = torch.tensor([[x, 0., 1.], [x+1e-6, 0., 1.], [x+1e-6, 0., 1.]], dtype=torch.float64)
    scope["velocity"] = torch.tensor([[speed+1e-6, 0., 0.], [speed, 0., 0.], [speed+1e-6, 0., 0.]], dtype=torch.float64)
    env._hit = torch.ones(3, dtype=torch.bool)
    env._flags["floor"] = torch.ones(3, dtype=torch.bool)
    exec(code, scope)
    check("return boundary is strict for both position and speed", scope["returned"].tolist() == [False, False, True])
    env._hit.zero_()
    exec(code, scope)
    check("floor route needs a prior paddle hit", not scope["landed_in_front"].any())

    goal = zones["zones"]["goal"]["boxes"][0]
    previous = torch.tensor(points.copy())
    current = previous.clone()
    previous[:, 0] = zones["goal_front_x_m"] + .1
    current[:, 0] = zones["goal_front_x_m"] - .1
    actual = task_math.crosses_goal(previous, current, cfg).numpy()
    expected = ((points[:, 1:] > np.asarray(goal["min"])[1:]) & (points[:, 1:] < np.asarray(goal["max"])[1:])).all(-1)
    check("10000 goal-plane crossings match real swept goal detection", np.array_equal(actual, expected))
    check("goal plane uses expanded goal face", abs((goal["min"][0]+goal["max"][0])/2 - scope["goal_front"]) < 1e-7)

    for key in zones["zones"]:
        checkbox = page.locator("#zone-" + key)
        for selected in (False, True, False):
            before = page.evaluate("viewer.outcomeZones()")
            checkbox.set_checked(selected)
            after = page.evaluate("viewer.outcomeZones()")
            check(f"{key} toggle {selected} updates scene visibility", after["zones"][key]["visible"] == selected)
            check(f"{key} toggle {selected} preserves other zones", all(before["zones"][other] == after["zones"][other] for other in before["zones"] if other != key))
    check("all overlays leave robot, shot and source config untouched", page.evaluate("viewer.snapshot()") == original)
    note = page.locator("#outcome-note").text_content()
    check("clearance prerequisites and non-spatial checks explained", all(text in note for text in ("prior racket hit", "tilt", "shot deadline", "ball-centre", "no goal")))
    for key in ("returned", "wide", "goal"):
        page.locator("#zone-" + key).check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(args.output.with_suffix(".png")))
    page.locator('[data-view="top"]').click()
    for key in ("wide", "goal"):
        page.locator("#zone-" + key).uncheck()
    page.screenshot(path=str(args.output.with_name(args.output.stem + "-floor-line.png")))
    page.locator("#zone-miss").check()
    page.locator('[data-view="3d"]').click()
    page.locator('#zones-fit').click()
    check("fit control expands view for miss ceiling", np.linalg.norm(page.evaluate('viewer.snapshot().camera')) > 10)
    page.screenshot(path=str(args.output.with_name(args.output.stem + "-miss.png")))
    page.set_viewport_size({"width": 390, "height": 844})
    for key in zones["zones"]:
        page.locator("#zone-" + key).check()
        page.locator("#zone-" + key).uncheck()
        check(key + " mobile toggle", not page.evaluate("viewer.outcomeZones()")["zones"][key]["visible"])
    check("mobile no horizontal overflow", page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
    check("no JavaScript errors", not errors)
    browser.close()
args.output.write_text(json.dumps({"passed": True, "checks": checks, "errors": errors, "geometry": zones,
                                   "random_points": len(points)}, indent=2) + "\n")
print(f"Outcome-zone checks passed: {len(checks)}; {len(points)} independent points per region")
