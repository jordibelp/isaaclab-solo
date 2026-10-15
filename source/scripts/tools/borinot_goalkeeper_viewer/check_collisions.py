#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check the delivered browser preview against independent convex feasibility tests."""
import argparse
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("html", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
checks, errors, requests, poses = [], [], [], []


def check(name, condition):
    assert condition, name
    checks.append(name)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                               args=["--no-sandbox", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1500, "height": 1050})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: requests.append(r.url) if r.url.startswith(("https:", "http:")) else None)
    page.goto(args.html.resolve().as_uri())
    page.wait_for_function("window.viewer")
    check("offline load without external requests", not requests)
    check("all six USD shapes plus round head and handle", len(page.evaluate("viewer.collisionGeometry()")) == 8)
    check("current config starts with collisions off", page.locator("#collision-mode").input_value() == "off")
    page.locator("#colliders").check()
    page.locator("#collision-mode").select_option("on")
    page.locator("#z-num").fill("1.5")
    initial = page.evaluate("viewer.snapshot()")
    check("default neutral racket clears eligible collision pairs", not any(c["enabled"] for c in initial["collisions"]))
    for diameter_cm in (8, 12.37, 16, 24, 32):
        page.locator("#paddle-diameter-num").fill(str(diameter_cm))
        snapshot = page.evaluate("viewer.snapshot()")
        geometry = page.evaluate("viewer.collisionGeometry()")
        rotation = Rotation.from_quat(snapshot["paddle_quaternion"])
        local = {s["name"]: rotation.inv().apply(np.array(s["vertices"]) - snapshot["paddle"])
                 for s in geometry if s["body"] == "paddle"}
        head, handle = local["paddle_head"], local["paddle_handle"]
        check(f"diameter {diameter_cm} round head radius", np.isclose(np.linalg.norm(head[:, 1:], axis=1).max(), diameter_cm/200))
        check(f"diameter {diameter_cm} thickness", np.isclose(np.ptp(head[:, 0]), initial["config"]["paddle_thickness_m"]))
        handle_y = diameter_cm/200 + initial["config"]["paddle_handle_length_m"]/2 - min(.01, diameter_cm/800)
        check(f"diameter {diameter_cm} handle follows rim", np.isclose(handle[:, 1].mean(), handle_y))
        check(f"diameter {diameter_cm} mount unchanged", np.allclose(snapshot["paddle"], initial["paddle"]))
        check(f"diameter {diameter_cm} source config not mutated", snapshot["config"] == initial["config"])
        check(f"diameter {diameter_cm} copyable config", page.locator("#paddle-config").inner_text().splitlines()[0] == f"paddle_diameter_m = {diameter_cm/100:g}")
    previous = page.evaluate("viewer.snapshot().state")
    for invalid in ("", "0", "999"):
        page.locator("#paddle-diameter-num").fill(invalid)
        check(f"invalid diameter {invalid!r} leaves geometry unchanged", page.evaluate("viewer.snapshot().state") == previous)
    page.locator("#paddle-reset").click()
    check("restore size uses config", page.evaluate("viewer.snapshot().state['paddle-diameter']") == initial["config"]["paddle_diameter_m"]*100)
    slider = page.locator("#paddle-diameter")
    slider.focus()
    slider.press("Home")
    slider.press("ArrowRight")
    check("keyboard slider updates geometry and numeric value", page.locator("#paddle-diameter-num").input_value() == "8.1")
    page.locator("#paddle-reset").click()
    for shoulder, elbow in ((0, 0), (35, -70), (-60, 85)):
        page.locator("#joint0-num").fill(str(shoulder))
        page.locator("#joint1-num").fill(str(elbow))
        page.locator("#paddle-mount-delta-num").fill("0")
        base = page.evaluate("viewer.snapshot()")
        rotation = Rotation.from_quat(base["bodies"]["flying_arm_2__link2"]["quaternion"])
        base_geometry = {s["name"]: np.asarray(s["vertices"]) for s in page.evaluate("viewer.collisionGeometry()")}
        for delta_cm in (-6, 0, 4.37, 12):
            page.locator("#paddle-mount-delta-num").fill(str(delta_cm))
            snapshot = page.evaluate("viewer.snapshot()")
            displacement = rotation.apply([0, -delta_cm/100, 0])
            check(f"mount {shoulder}/{elbow}/{delta_cm} follows calf axis", np.allclose(np.asarray(snapshot["paddle"])-base["paddle"], displacement, atol=1e-7))
            check(f"mount {shoulder}/{elbow}/{delta_cm} preserves face normal", np.allclose(snapshot["paddle_quaternion"], base["paddle_quaternion"]))
            check(f"mount {shoulder}/{elbow}/{delta_cm} leaves body and shot fixed", snapshot["bodies"] == base["bodies"] and snapshot["shot"] == base["shot"])
            for shape in page.evaluate("viewer.collisionGeometry()"):
                expected = base_geometry[shape["name"]] + (displacement if shape["body"] == "paddle" else 0)
                check(f"mount {shoulder}/{elbow}/{delta_cm} hull {shape['name']}", np.allclose(shape["vertices"], expected, atol=1e-7))
            check(f"mount {shoulder}/{elbow}/{delta_cm} copyable config", page.locator("#paddle-config").inner_text().splitlines()[1] == f"paddle_mount_delta_m = {delta_cm/100:g}")
    previous = page.evaluate("viewer.snapshot().state")
    for invalid in ("", "-999", "999"):
        page.locator("#paddle-mount-delta-num").fill(invalid)
        check(f"invalid mount {invalid!r} leaves state unchanged", page.evaluate("viewer.snapshot().state") == previous)
    page.locator("#paddle-diameter-num").fill("12.37")
    page.locator("#mount-reset").click()
    check("mount reset preserves diameter", page.evaluate("viewer.snapshot().state['paddle-diameter']") == 12.37)
    check("mount reset uses config delta", page.evaluate("viewer.snapshot().state['paddle-mount-delta']") == initial["config"]["paddle_mount_delta_m"]*100)
    slider = page.locator("#paddle-mount-delta")
    slider.focus()
    slider.press("Home")
    slider.press("ArrowRight")
    check("mount keyboard slider updates numeric value", page.locator("#paddle-mount-delta-num").input_value() == "-9.9")
    page.locator("#mount-reset").click()
    page.locator("#paddle-reset").click()
    rng = np.random.default_rng(42)
    angles = [[0, 0], [90, 0], [-90, 0], [0, 180], [0, -180], [63, 80], [-63, -80], [30, 0], [0, 60]]
    angles += rng.uniform([-90, -185], [90, 185], (31, 2)).tolist()
    for index, (shoulder, elbow) in enumerate(angles):
        page.locator("#joint0-num").fill(str(shoulder))
        page.locator("#joint1-num").fill(str(elbow))
        snapshot = page.evaluate("viewer.snapshot()")
        geometry = page.evaluate("viewer.collisionGeometry()")
        actual = {tuple(sorted((c["a"], c["b"]))) for c in snapshot["collisions"] if c["overlap"] is True}
        expected = set()
        hulls = [ConvexHull(s["vertices"]).equations for s in geometry]
        for i, a in enumerate(geometry):
            for j in range(i+1, len(geometry)):
                b = geometry[j]
                if a["body"] == b["body"]:
                    continue
                equations = np.vstack([hulls[i], hulls[j]])
                result = linprog(np.zeros(3), A_ub=equations[:, :3], b_ub=-equations[:, 3],
                                 bounds=[(None, None)]*3, method="highs")
                pair = tuple(sorted((a["name"], b["name"])))
                if result.success:
                    expected.add(pair)
                check(f"pose {index} convex feasibility {pair}", (pair in actual) == result.success)
        check(f"pose {index} no indeterminate GJK results", all(c["overlap"] is not None for c in snapshot["collisions"]))
        check(f"pose {index} parent-child pairs filtered", all(not c["enabled"] for c in snapshot["collisions"] if c["parent_child"]))
        poses.append({"angles_deg": [shoulder, elbow], "overlaps": snapshot["collisions"]})
        if index == 5:
            page.locator("#focus").click()
            page.screenshot(path=str(args.output.with_suffix(".png")))
    page.locator("#collision-mode").select_option("off")
    check("off filters every overlap", all(not c["enabled"] for c in page.evaluate("viewer.snapshot().collisions")))
    for view in ("front", "side", "top", "3d"):
        page.locator(f'[data-view="{view}"]').click()
        check("camera "+view, np.isfinite(page.evaluate("viewer.snapshot().camera")).all())
    page.locator("#home").click()
    state = page.evaluate("viewer.snapshot().state")
    page.locator("#joint0-num").fill("999")
    check("invalid joint angle does not change pose", page.evaluate("viewer.snapshot().state") == state)
    page.locator("#home").click()
    page.set_viewport_size({"width": 390, "height": 844})
    check("mobile layout no page overflow", page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
    check("no browser exceptions", not errors)
    check("no external requests during interaction", not requests)
    browser.close()
args.output.write_text(json.dumps({"passed": len(checks), "checks": checks, "poses": poses, "errors": errors}, indent=2)+"\n")
print(f"Passed {len(checks)} independent geometry and browser checks across {len(poses)} poses")
