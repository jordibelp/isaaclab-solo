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
    check("all six USD shapes plus paddle", len(page.evaluate("viewer.collisionGeometry()")) == 7)
    check("current config starts with collisions off", page.locator("#collision-mode").input_value() == "off")
    page.locator("#colliders").check()
    page.locator("#collision-mode").select_option("on")
    page.locator("#z-num").fill("1.5")
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
            check("folded policy pose has active paddle/base overlap", any(c["enabled"] and c["body_a"] == "base"
                  and c["body_b"] == "paddle" for c in snapshot["collisions"]))
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
