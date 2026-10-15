#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Isolated Borinot arm bench: fixed base, no policy, ball, goal or task resets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--self-collisions", choices=["on", "off", "compare"], default="compare")
parser.add_argument("--scan", action="store_true", help="Probe a joint grid by imposing poses; not a reachable-motion test.")
parser.add_argument("--grid-deg", type=float, default=10.0)
parser.add_argument("--output", type=Path, help="Save all probe and motion samples as JSON.")
parser.add_argument("--duration", type=float, default=0.0, help="GUI time limit in seconds; 0 runs until closed.")
parser.add_argument("--demo", action="store_true", help="GUI demo: clear motion followed by a blocked target.")
parser.add_argument("--screenshot", type=Path, help="Capture the GUI viewport near the end of duration.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if args.device != "cpu":
    parser.error("This isolated bench uses CPU physics for fresh per-step contact reports. Use --device cpu.")
if not 1 <= args.grid_deg <= 90:
    parser.error("grid-deg must be between 1 and 90")
if args.headless and not args.scan:
    parser.error("Use --scan for a headless run")
if args.screenshot and (args.headless or args.duration <= 0):
    parser.error("screenshot needs a GUI run with a positive duration")
app = AppLauncher(args).app

import numpy as np
import torch
from pxr import Gf, PhysicsSchemaTools, Usd, UsdPhysics
from omni.physx import get_physx_simulation_interface
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg
from isaaclab_tasks.direct.borinot_goalkeeper.paddle import attach_paddle


def main():
    cfg = BorinotGoalkeeperEnvCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/cfg.physics_hz, device=args.device,
                                                            gravity=(0.0, 0.0, 0.0)))
    # We use the low-level callback rather than ContactSensor, so enable reports explicitly.
    sim.carb_settings.set_bool("/physics/disableContactProcessing", False)
    sim.set_camera_view((1.8, -2.4, 1.8), (0.0, 0.0, 1.2))
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)
    stage = sim_utils.get_current_stage()
    modes = [False, True] if args.self_collisions == "compare" else [args.self_collisions == "on"]
    rigs = []
    for i, enabled in enumerate(modes):
        rcfg = cfg.robot.copy()
        root = f"/World/Robot_{'on' if enabled else 'off'}"
        rcfg.prim_path = root
        rcfg.init_state.pos = (0.0, (i-(len(modes)-1)/2)*0.85, 1.2)
        rcfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        rcfg.spawn.articulation_props.enabled_self_collisions = enabled
        # Keep this zero-gravity diagnostic awake while probing static poses.
        rcfg.spawn.articulation_props.sleep_threshold = 0.0
        # Level the authored USD base before creating its world-fixed joint.
        rcfg.spawn.articulation_props.fix_root_link = None
        rcfg.spawn.rigid_props.disable_gravity = True
        rcfg.actuators["arm"].stiffness = cfg.joint_stiffness
        rcfg.actuators["arm"].damping = cfg.joint_damping
        robot = Articulation(rcfg)
        base = stage.GetPrimAtPath(root+"/base")
        authored = base.GetAttribute("xformOp:orient").Get()
        stage.GetPrimAtPath(root).GetAttribute("xformOp:orient").Set(Gf.Quatf(authored.GetInverse()))
        sim_utils.modify_articulation_root_properties(
            root, sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True))
        attach_paddle(stage, root, cfg)
        rigs.append({"enabled": enabled, "root": root, "robot": robot})
    sim.reset()
    sim.carb_settings.set_bool("/physics/disableContactProcessing", False)
    def fresh_contact_stream(rig):
        body_paths = [rig["root"]+"/"+name for name in rig["robot"].body_names]
        rig["contact"] = sim.physics_sim_view.create_rigid_contact_view(
            body_paths, filter_patterns=[body_paths for _ in body_paths], max_contact_data_count=4096)

    for rig in rigs:
        robot = rig["robot"]
        rig["ids"] = robot.find_joints(list(cfg.joint_names), preserve_order=True)[0]
        fresh_contact_stream(rig)
        rig["contact_bodies"] = list(robot.body_names)
        rig["target"] = robot.data.default_joint_pos.clone()
    current_contacts = {}
    paths = {}

    def on_contacts(headers, data):
        for header in headers:
            if not header.num_contact_data:
                continue
            for value in (header.actor0, header.actor1):
                if value not in paths:
                    paths[value] = str(PhysicsSchemaTools.intToSdfPath(value))
            a, b = sorted([paths[header.actor0], paths[header.actor1]])
            key = (a, b)
            entry = current_contacts.setdefault(key, {"normal_impulse": np.zeros(3), "points": 0,
                                                       "minimum_separation_m": float("inf")})
            for index in range(header.contact_data_offset, header.contact_data_offset+header.num_contact_data):
                point = data[index]
                normal, impulse = np.array(point.normal), np.array(point.impulse)
                entry["normal_impulse"] += normal*np.dot(impulse, normal)
                entry["points"] += 1
                entry["minimum_separation_m"] = min(entry["minimum_separation_m"], float(point.separation))

    from isaacsim.core.version import get_version
    report = {"simulator_version": list(get_version()), "sleep_threshold": 0.0,
              "paddle_size_m": list(cfg.paddle_size), "paddle_offset_ee_m": list(cfg.paddle_offset_ee),
              "paddle_quat_ee_wxyz": list(cfg.paddle_quat_ee),
              "bench_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "refresh_kinematics_after_impose": True,
              "contact_source": "per-step immediate PhysX report; cached tensor matrix also retained for comparison",
              "disable_contact_processing": sim.get_setting("/physics/disableContactProcessing"),
              "usd_path": cfg.robot.spawn.usd_path,
              "usd_sha256": hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest(),
              "dt_s": sim.get_physics_dt(), "device": args.device, "gravity": [0, 0, 0],
              "fixed_base": True, "joint_names": list(cfg.joint_names),
              "pd_stiffness": cfg.joint_stiffness, "pd_damping": cfg.joint_damping,
              "effort_limit_nm": cfg.effort_lim_sim, "rigs": [], "probes": [], "motions": []}
    for rig in rigs:
        roots = [p for p in Usd.PrimRange(stage.GetPrimAtPath(rig["root"]))
                 if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        report["rigs"].append({"root": rig["root"], "body_names": rig["robot"].body_names,
                               "fixed_base": rig["robot"].is_fixed_base,
                               "authored_self_collisions": [p.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
                                                            for p in roots]})

    def tick(render=False):
        current_contacts.clear()
        for rig in rigs:
            rig["robot"].set_joint_position_target(rig["target"])
            rig["robot"].write_data_to_sim()
        sim.step(render=render)
        on_contacts(*get_physx_simulation_interface().get_contact_report())
        for rig in rigs:
            rig["robot"].update(sim.get_physics_dt())

    def impose(angles):
        for rig in rigs:
            robot = rig["robot"]
            q = robot.data.default_joint_pos.clone()
            q[:, rig["ids"]] = torch.tensor([angles], device=robot.device, dtype=q.dtype) * math.pi/180
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.reset()
            rig["target"] = q.clone()
    # Refresh the link poses after direct DOF writes.
        # Otherwise joint angles can be new while contact geometry remains at an old pose.
        sim.physics_sim_view.update_articulations_kinematic()

    def sample(rig):
        forces = rig["contact"].get_contact_force_matrix(dt=sim.get_physics_dt()).reshape(
            len(rig["contact_bodies"]), len(rig["robot"].body_names), 3)
        tensor_contacts = []
        for i, a in enumerate(rig["contact_bodies"]):
            for j, b in enumerate(rig["robot"].body_names):
                force = float(torch.linalg.vector_norm(forces[i, j]))
                if a < b and force > 0.001:
                    tensor_contacts.append({"body_a": a, "body_b": b, "normal_force_n": force})
        contacts = []
        for (a, b), entry in current_contacts.items():
            if not (a.startswith(rig["root"]+"/") and b.startswith(rig["root"]+"/")):
                continue
            force = float(np.linalg.norm(entry["normal_impulse"])/sim.get_physics_dt())
            if force > .001:
                contacts.append({"body_a": a.rsplit("/", 1)[-1], "body_b": b.rsplit("/", 1)[-1],
                                 "normal_force_n": force, "point_count": entry["points"],
                                 "minimum_separation_m": entry["minimum_separation_m"]})
        robot = rig["robot"]
        return {"self_collisions": rig["enabled"],
                "joint_deg": (robot.data.joint_pos[0, rig["ids"]]*180/math.pi).tolist(),
                "joint_velocity_rad_s": robot.data.joint_vel[0, rig["ids"]].tolist(),
                "contacts": contacts, "tensor_contacts": tensor_contacts}

    if args.scan:
        def grid(limit):
            values = np.arange(0, limit, args.grid_deg)
            return sorted(set([-limit, limit, *map(float, values), *map(float, -values)]))
        shoulder, elbow = grid(90), grid(185)
        for s in shoulder:
            for e in elbow:
                impose([s, e])
                tick()
                report["probes"].append({"imposed_joint_deg": [float(s), float(e)], "rigs": [sample(r) for r in rigs]})
            print(f"Probed shoulder {s:.1f} deg ({len(report['probes'])} poses)", flush=True)
        # Continuous PD paths begin at neutral. They never teleport between targets.
        for label, destination in [("clear_positive", [30, 0]), ("clear_negative", [-30, 0]),
                                    ("clear_elbow_positive", [0, 60]), ("clear_elbow_negative", [0, -60]),
                                    ("policy_positive", [63, 80]), ("policy_negative", [-63, -80]),
                                    ("shoulder_positive", [90, 0]), ("shoulder_negative", [-90, 0]),
                                    ("fold_positive", [0, 180]), ("fold_negative", [0, -180])]:
            impose([0, 0])
            for _ in range(40):
                tick()
            motion = {"name": label, "destination_deg": destination, "start_body_poses_w": [
                {"self_collisions": r["enabled"], "names": r["robot"].body_names,
                 "position_m": r["robot"].data.body_pos_w[0].tolist(),
                 "quaternion_wxyz": r["robot"].data.body_quat_w[0].tolist()} for r in rigs], "samples": []}
            for step in range(500):
                progress = min(1.0, step/300)
                for rig in rigs:
                    rig["target"][0, rig["ids"]] = torch.tensor(destination, device=args.device)*progress*math.pi/180
                tick()
                motion["samples"].append({"time_s": (step+1)*sim.get_physics_dt(),
                                          "target_deg": [d*progress for d in destination],
                                          "rigs": [sample(r) for r in rigs]})
            report["motions"].append(motion)
            print(label, motion["samples"][-1], flush=True)
    else:
        import omni.ui as ui
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        model_path = Path(__file__).with_name("borinot_goalkeeper_viewer") / "robot_model.json"
        geometry = json.loads(model_path.read_text())
        if geometry["usd_sha256"] != report["usd_sha256"]:
            raise ValueError("Collision outline cache is stale; refresh export_robot.py")
        outlines = geometry["collision_shapes"]
        # The paddle is a true collision shape too.
        vertices = [[x*cfg.paddle_size[0]/2, y*cfg.paddle_size[1]/2, z*cfg.paddle_size[2]/2]
                    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        outlines.append({"body": "paddle", "matrix": np.eye(4).reshape(-1).tolist(), "vertices": vertices,
                         "faces": [[0,1,2],[1,2,3],[4,5,6],[5,6,7],[0,1,4],[1,4,5],
                                   [2,3,6],[3,6,7],[0,2,4],[2,4,6],[1,3,5],[3,5,7]]})
        window = ui.Window("Borinot collision bench", width=465, height=350)
        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Fixed base / zero gravity / actual PhysX contacts", word_wrap=True)
                ui.Label("Compare: OFF at -Y; ON at +Y. Shared PD targets.", word_wrap=True)
                sliders = []
                for label, lo, hi in [("Shoulder (deg)", -90, 90), ("Elbow (deg)", -185, 185)]:
                    ui.Label(label)
                    slider = ui.FloatSlider(min=lo, max=hi, step=0.1)
                    sliders.append(slider.model)
                ui.Button("Target neutral (no teleport)", clicked_fn=lambda: [m.set_value(0.0) for m in sliders])
                demo = ui.CheckBox()
                demo.model.set_value(args.demo)
                ui.Label("Scripted demo (uncheck to control the sliders)")
                overlay = ui.CheckBox()
                overlay.model.set_value(True)
                ui.Label("Collision outlines: green = no report, red = contact. OFF can pass through!")
                status = ui.Label("", word_wrap=True)
                ui.Label("Parent-child contacts stay filtered by PhysX, even with ON.\n"
                         "Do not treat imposed poses or OFF motion as physically safe.", word_wrap=True)
        impose([0, 0])
        started = time.monotonic()
        captured = False
        while app.is_running() and (args.duration <= 0 or time.monotonic()-started < args.duration):
            if demo.model.get_value_as_bool():
                elapsed = time.monotonic()-started
                destination = [30.0, 0.0] if elapsed < 4 else [63.0, 80.0]
                for model, value in zip(sliders, destination):
                    model.set_value(value)
            for rig in rigs:
                rig["target"][0, rig["ids"]] = torch.tensor([m.get_value_as_float() for m in sliders],
                                                            device=args.device)*math.pi/180
            tick(render=True)
            samples = [sample(r) for r in rigs]
            status.text = "\n".join(f"{'ON' if s['self_collisions'] else 'OFF'}: "
                                    f"q={np.round(s['joint_deg'], 1)}; "
                                    + (", ".join(f"{c['body_a']} / {c['body_b']}: {c['normal_force_n']:.2f} N"
                                                 for c in s["contacts"]) or "no reported self contact") for s in samples)
            draw.clear_lines()
            if overlay.model.get_value_as_bool():
                starts, ends, colors = [], [], []
                for rig, live in zip(rigs, samples):
                    robot = rig["robot"]
                    active = {c[k] for c in live["contacts"] for k in ("body_a", "body_b")}
                    for shape in outlines:
                        bid = robot.body_names.index(shape["body"])
                        p = robot.data.body_pos_w[0, bid].cpu().numpy()
                        q = robot.data.body_quat_w[0, bid].cpu().numpy()
                        transform = Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
                        transform.SetTranslateOnly(Gf.Vec3d(*map(float, p)))
                        local = np.array(shape["matrix"]).reshape(4, 4)
                        world = [transform.Transform(Gf.Vec3d(*map(float, (np.array([*v, 1]) @ local)[:3])))
                                 for v in shape["vertices"]]
                        edges = shape.get("edges") or {tuple(sorted((f[i], f[(i+1)%3])))
                                                       for f in shape["faces"] for i in range(3)}
                        for a, b in edges:
                            starts.append(tuple(world[a])); ends.append(tuple(world[b]))
                            colors.append((1, .2, .2, 1) if shape["body"] in active else (.2, 1, .35, 1))
                draw.draw_lines(starts, ends, colors, [1.5]*len(starts))
            if args.screenshot and not captured and time.monotonic()-started > args.duration-2:
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                capture_viewport_to_file(get_active_viewport(), str(args.screenshot))
                captured = True
        report["gui_final"] = samples
        report["gui_target_deg"] = [m.get_value_as_float() for m in sliders]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(f"Saved {args.output}", flush=True)


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    # Match DirectRLEnv.close(): unsubscribe IsaacLab's stop/render loop before closing Kit.
    context = sim_utils.SimulationContext.instance()
    if context is not None:
        context.clear_all_callbacks()
        context.clear_instance()
    app.close(wait_for_replicator=False)
