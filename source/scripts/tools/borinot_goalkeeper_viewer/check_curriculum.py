#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise phase edits, real overlay vertices, exports and ballistic samples in Chrome."""
import argparse
import ast
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("page", help="Standalone HTML path or local viewer URL")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
checks, errors, requests = [], [], []


def check(name, condition):
    assert condition, name
    checks.append(name)


def config_values(text):
    return {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(text).body}


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                               args=["--no-sandbox", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1450, "height": 1000})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: requests.append(r.url))
    local_file = not args.page.startswith("http")
    page.goto(Path(args.page).resolve().as_uri() if local_file else args.page)
    page.wait_for_function("window.viewer")
    snapshot = lambda: page.evaluate("viewer.snapshot()")
    base = snapshot()
    defaults = base["config"]
    original_state = base["state"]
    check("phase count comes from config", page.locator("#level option").count() == len(defaults["shot_half_extents_yz"]))
    check("selected phase comes from config", int(page.locator("#level").input_value()) == defaults["shot_level"])

    def verify_overlay(index, extent):
        overlays = snapshot()["curriculum"]["overlays"]
        for plane, x in (("goal", defaults["goal_center"][0] + defaults["goal_size"][0]/2 + .002),
                         ("keeper", defaults["keeper_plane_x"])):
            actual = next(o for o in overlays if o["phase"] == index and o["plane"] == plane and not o["filled"])
            vertices = list(zip(*[iter(actual["vertices"])]*3))
            check(f"phase {index} {plane} actual X coordinates", all(abs(v[0]-x) < 1e-6 for v in vertices))
            for axis in (1, 2):
                center, half = defaults["target_center_yz"][axis-1], extent[axis-1]
                check(f"phase {index} {plane} axis {axis} actual bounds",
                      abs(min(v[axis] for v in vertices)-(center-half)) < 1e-6 and
                      abs(max(v[axis] for v in vertices)-(center+half)) < 1e-6)
        check(f"phase {index} selected area is filled on goal", sum(o["filled"] for o in overlays) == 1)

    def verify_samples(index, extent, times):
        # Read actual JS sampler output, then compute each trajectory independently.
        samples = page.evaluate("""() => Array.from({length:32},()=>{
            document.getElementById('sample').click();return viewer.snapshot().shot;
        })""")
        for shot in samples:
            t = shot["flightTime"]
            check(f"phase {index} flight in edited range", times[0] <= t <= times[1])
            for axis in (1, 2):
                check(f"phase {index} target axis {axis} in range", abs(shot["target"][axis]-defaults["target_center_yz"][axis-1]) <= extent[axis-1]+1e-12)
            check(f"phase {index} launch distribution", defaults["launch_x_range"][0] <= shot["position"][0] <= defaults["launch_x_range"][1]
                  and abs(shot["position"][1]-shot["target"][1]) <= defaults["launch_y_noise"]
                  and shot["position"][2] == defaults["launch_z"])
            target = [shot["position"][i]+shot["velocity"][i]*t-(.5*9.81*t*t if i == 2 else 0) for i in range(3)]
            check(f"phase {index} independent ballistic target", max(abs(a-b) for a, b in zip(target, shot["target"])) < 1e-10)
        return samples

    for i, extent in enumerate(defaults["shot_half_extents_yz"]):
        page.locator("#level").select_option(str(i))
        verify_overlay(i, extent)
        verify_samples(i, extent, defaults["shot_flight_time_ranges_s"][i])

    page.locator("#spread-y").fill("0.333333")
    page.locator("#spread-z").fill("0.1731")
    page.locator("#flight-max").fill("0.61")
    page.locator("#flight-min").fill("0.61")
    edited = snapshot()["curriculum"]["selected"]
    verify_overlay(edited, [.333333, .1731])
    verify_samples(edited, [.333333, .1731], [.61, .61])
    check("continuous numbers are valid", page.locator("#phase-error").inner_text() == "")
    valid = snapshot()["curriculum"]
    for field, invalid in (("spread-y", ""), ("spread-y", "-1"), ("spread-y", "1e309"),
                           ("flight-min", "0"), ("flight-min", "0.7")):
        page.locator("#level").select_option(str(edited))
        page.locator("#"+field).fill(invalid)
        check(f"invalid {field}={invalid!r} preserves phases and overlays", snapshot()["curriculum"] == valid)
        check(f"invalid {field}={invalid!r} is visible and cannot export", page.locator("#phase-error").inner_text() != "" and page.locator("#phases-copy").is_disabled())
    page.locator("#level").select_option(str(edited))
    page.locator("#phase-add").click()
    added = len(defaults["shot_half_extents_yz"])
    check("new phase beyond original four is selected", snapshot()["curriculum"]["selected"] == added)
    check("new phase copies selected values", snapshot()["curriculum"]["phases"][added] == valid["phases"][edited])
    page.locator("#spread-y").fill("0")
    page.locator("#spread-z").fill("0")
    verify_overlay(added, [0, 0])
    verify_samples(added, [0, 0], [.61, .61])
    check("new phase owns independent values", snapshot()["curriculum"]["phases"][edited] == valid["phases"][edited])
    page.locator("#spread-y").fill("1.6")
    page.locator("#spread-z").fill("1.4")
    check("area beyond goal is flagged and not clipped", "outside the goal" in page.locator("#spread-summary").inner_text())
    verify_overlay(added, [1.6, 1.4])
    page.locator("#spread-all").uncheck()
    check("comparison toggle leaves selected goal and keeper outlines", len(snapshot()["curriculum"]["overlays"]) == 3)
    page.locator("#spread-goal").uncheck()
    check("goal toggle leaves keeper outline", [o["plane"] for o in snapshot()["curriculum"]["overlays"]] == ["keeper"])
    page.locator("#aim").uncheck()
    check("keeper toggle hides remaining overlay", snapshot()["curriculum"]["overlays"] == [])
    page.locator("#spread-goal").check()
    page.locator("#aim").check()
    page.locator("#spread-all").check()
    page.locator("#curriculum-editor details summary").click()
    exported = config_values(page.locator("#curriculum-config").input_value())
    current = snapshot()["curriculum"]
    check("export contains all paired phases and selected level", exported == {
        "shot_half_extents_yz": tuple(tuple(p["extent"]) for p in current["phases"]),
        "shot_flight_time_ranges_s": tuple(tuple(p["flight"]) for p in current["phases"]), "shot_level": added})
    check("export preserves float types for zero/integer distances", all(type(v) is float for row in exported["shot_half_extents_yz"] for v in row))
    page.locator("#phases-copy").click()
    page.wait_for_function("document.getElementById('copy-status').textContent.length > 0")
    check("copy action completes or selects fallback", "Cop" in page.locator("#copy-status").inner_text() or "selected" in page.locator("#copy-status").inner_text())
    page.locator("#phase-remove").click()
    check("remove reindexes selected phase safely", snapshot()["curriculum"]["selected"] == added-1)
    while page.locator("#level option").count() > 1:
        page.locator("#phase-remove").click()
    check("last phase cannot be removed", page.locator("#phase-remove").is_disabled())
    one = config_values(page.locator("#curriculum-config").input_value())
    check("single-phase export is a tuple not a bare pair", len(one["shot_half_extents_yz"]) == 1 and isinstance(one["shot_half_extents_yz"][0], tuple))
    page.locator("#phases-reset").click()
    reset = snapshot()
    check("restore brings back source phases and source level", reset["curriculum"]["selected"] == defaults["shot_level"] and
          [p["extent"] for p in reset["curriculum"]["phases"]] == defaults["shot_half_extents_yz"] and
          [p["flight"] for p in reset["curriculum"]["phases"]] == defaults["shot_flight_time_ranges_s"])
    check("phase edits leave robot and source config untouched", reset["state"] == original_state and reset["config"] == defaults)
    for mode in ("front", "side", "top", "3d"):
        page.locator(f'[data-view="{mode}"]').click()
        check("camera with spread overlay: "+mode, all(math.isfinite(v) for v in snapshot()["camera"]))
    page.locator('[data-view="front"]').click()
    page.locator("#level").select_option(str(len(defaults["shot_half_extents_yz"])-1))
    page.locator("#z-num").fill("1.5")
    page.locator("#area").uncheck()
    page.locator("aside").evaluate("e=>e.scrollTop=0")
    page.screenshot(path=str(args.output.with_suffix(".png")))
    page.set_viewport_size({"width":390, "height":844})
    page.wait_for_timeout(200)
    check("mobile has no horizontal overflow", page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
    page.locator("#phase-add").click()
    page.locator("#spread-y").fill("0.1234")
    check("mobile phase edits update actual values", abs(snapshot()["curriculum"]["phases"][-1]["extent"][0]-.1234) < 1e-12)
    page.locator("#phases-reset").click()
    check("no JavaScript errors", not errors)
    check("no external resources", not any(url.startswith("http") for url in requests) if local_file else all(url.startswith(args.page.rstrip("/")) for url in requests))
    browser.close()

args.output.write_text(json.dumps({"passed": True, "checks": checks, "errors": errors, "requests": requests}, indent=2))
print(f"Browser curriculum checks passed: {len(checks)}")
