#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check real checkbox interaction, arm-plane geometry and rotated collision hulls."""
import argparse
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("page", help="Live URL or standalone HTML path")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--baseline", type=Path, help="Optional snapshots captured before the feature was added")
args = parser.parse_args()
checks, errors, samples = [], [], []
model = json.loads((Path(__file__).parent / "robot_model.json").read_text())


def check(name, condition):
    assert condition, name
    checks.append(name)


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                               args=["--no-sandbox", "--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1450, "height": 1000})
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(args.page if args.page.startswith("http") else Path(args.page).resolve().as_uri())
    page.wait_for_function("window.viewer")
    snapshot = lambda: page.evaluate("viewer.snapshot()")
    initial = snapshot()
    checkbox = page.get_by_label("Racket face parallel to arm plane", exact=True)
    check("checkbox starts from config", checkbox.is_checked() == initial["config"]["racket_parallel_to_arm_plane"])
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    page.locator("#colliders").check()
    for index, pose in enumerate(((0, 0, 0, 0, -90), (35, -70, 0, 0, -90),
                                  (-60, 85, 0, 0, -90), (20, 45, 15, -25, 33))):
        for field, value in zip(("joint0", "joint1", "roll", "pitch", "yaw"), pose):
            page.locator(f"#{field}-num").fill(str(value))
        checkbox.check()
        parallel = snapshot()
        parallel_hulls = {shape["name"]: shape for shape in page.evaluate("viewer.collisionGeometry()")}
        if baseline:
            old = baseline[index]
            check(f"pose {index} default position parity", np.allclose(parallel["paddle"], old["snapshot"]["paddle"], atol=1e-12))
            check(f"pose {index} default rotation parity", np.allclose(parallel["paddle_quaternion"], old["snapshot"]["paddle_quaternion"], atol=1e-12))
            for hull in old["hulls"]:
                check(f"pose {index} default {hull['name']} parity", np.allclose(parallel_hulls[hull["name"]]["vertices"], hull["vertices"], atol=1e-12))
        for enabled in (False, True, False, True):
            checkbox.set_checked(enabled)
            snap = snapshot()
            rotation = Rotation.from_quat(snap["paddle_quaternion"])
            old_rotation = Rotation.from_quat(parallel["paddle_quaternion"])
            normal, old_normal = rotation.apply([1, 0, 0]), old_rotation.apply([1, 0, 0])
            check(f"pose {index} {enabled} centre unchanged", np.allclose(snap["paddle"], parallel["paddle"], atol=1e-12))
            check(f"pose {index} {enabled} bodies and shot unchanged", snap["bodies"] == parallel["bodies"] and snap["shot"] == parallel["shot"])
            check(f"pose {index} {enabled} source config unchanged", snap["config"] == initial["config"])
            check(f"pose {index} {enabled} handle axis unchanged", np.allclose(rotation.apply([0, 1, 0]), old_rotation.apply([0, 1, 0]), atol=1e-10))
            check(f"pose {index} {enabled} angle between face normals", abs(normal @ old_normal - (1 if enabled else 0)) < 1e-10)
            for joint in model["joints"]:
                parent = Rotation.from_quat(snap["bodies"][joint["parent"]]["quaternion"])
                arm_axis = (parent * Rotation.from_quat(joint["rot0"])).apply(np.eye(3)["XYZ".index(joint["axis"])])
                # USD float32 joint frames leave the two hinge axes slightly skewed.
                check(f"pose {index} {enabled} face vs {joint['name']} sweep plane", abs(abs(normal @ arm_axis) - (1 if enabled else 0)) < 1e-6)
            turn = Rotation.identity() if enabled else Rotation.from_rotvec([0, np.pi/2, 0])
            world_turn = old_rotation * turn * old_rotation.inv()
            for hull in page.evaluate("viewer.collisionGeometry()"):
                vertices = np.array(parallel_hulls[hull["name"]]["vertices"])
                expected = world_turn.apply(vertices - snap["paddle"]) + snap["paddle"] if hull["body"] == "paddle" else vertices
                check(f"pose {index} {enabled} hull {hull['name']} follows orientation", np.allclose(hull["vertices"], expected, atol=1e-9))
            check(f"pose {index} {enabled} config text", f"racket_parallel_to_arm_plane = {enabled}" in page.locator("#paddle-config").inner_text())
            samples.append(snap)
    checkbox.uncheck()
    page.locator("#paddle-diameter-num").fill("12.37")
    page.locator("#paddle-mount-delta-num").fill("4.37")
    page.locator("#paddle-reset").click()
    page.locator("#mount-reset").click()
    check("size and mount restores preserve perpendicular selection", not checkbox.is_checked() and not snapshot()["state"]["racket-parallel"])
    page.locator("#paddle-diameter-num").fill("12.37")
    page.locator("#paddle-mount-delta-num").fill("4.37")
    page.locator("#racket-orientation-reset").click()
    restored = snapshot()
    check("orientation restore uses source config", checkbox.is_checked() == initial["config"]["racket_parallel_to_arm_plane"])
    check("orientation restore preserves size and mount", restored["state"]["paddle-diameter"] == 12.37 and restored["state"]["paddle-mount-delta"] == 4.37)
    page.locator("#paddle-reset").click()
    page.locator("#mount-reset").click()
    page.locator("#home").click()
    page.locator("#focus").click()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for enabled in (True, False):
        checkbox.set_checked(enabled)
        checkbox.scroll_into_view_if_needed()
        page.screenshot(path=str(args.output.with_name(args.output.stem + ("-parallel.png" if enabled else "-perpendicular.png"))))
    page.set_viewport_size({"width": 390, "height": 844})
    checkbox.check()
    checkbox.uncheck()
    check("mobile checkbox changes geometry", not snapshot()["state"]["racket-parallel"])
    check("mobile has no horizontal overflow", page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
    check("no JavaScript errors", not errors)
    browser.close()
args.output.write_text(json.dumps({"passed": True, "checks": checks, "errors": errors, "poses": samples}, indent=2) + "\n")
print(f"Racket orientation browser checks passed: {len(checks)}")
