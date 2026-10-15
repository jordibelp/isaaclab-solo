# SPDX-License-Identifier: BSD-3-Clause
"""Verify round-head/handle contacts, active mass properties, and finite task steps."""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--diameter", type=float, default=0.16)
parser.add_argument("--self-collisions", action="store_true")
parser.add_argument("--mount-delta", type=float, default=0.0)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab_tasks
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg
from isaaclab_tasks.direct.borinot_goalkeeper.paddle import attach_paddle

cfg = BorinotGoalkeeperEnvCfg()
cfg.paddle_diameter_m = args.diameter
cfg.paddle_mount_delta_m = args.mount_delta
cfg.robot.spawn.articulation_props.enabled_self_collisions = args.self_collisions
cfg.scene.num_envs = 8
cfg.sim.device = args.device
cfg.curriculum = False
cfg.seed = 42
cfg.robot_home = (0.0, 0.0, 1.5)  # isolate geometry from local home tuning
cfg.robot.init_state.pos = cfg.robot_home
cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
cfg.hold_arm = True
cfg.ball.spawn.rigid_props.disable_gravity = True  # straight controlled shots
report = {"diameter_m": args.diameter, "mount_delta_m": args.mount_delta, "handle_length_m": cfg.paddle_handle_length_m,
          "self_collisions": args.self_collisions, "checks": [], "shots": {}}


def check(name, condition, detail=None):
    report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
    print(f"CHECK {name}: {bool(condition)} {detail if detail is not None else ''}", flush=True)
    if not condition:
        args.output.write_text(json.dumps(report, indent=2))
        raise AssertionError(name)


env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
try:
    env.reset()
    stage = env.sim.stage
    root = "/World/envs/env_0/Robot/paddle"
    head = UsdGeom.Cylinder(stage.GetPrimAtPath(root + "/geometry/mesh"))
    check("real cylinder radius", abs(head.GetRadiusAttr().Get() - args.diameter/2) < 1e-7)
    check("real cylinder thickness and axis", abs(head.GetHeightAttr().Get() - cfg.paddle_thickness_m) < 1e-7
          and head.GetAxisAttr().Get() == "X")
    shapes = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.CollisionAPI)]
    check("head and handle are both colliders", len(shapes) == 2, [str(p.GetPath()) for p in shapes])
    check("one paddle rigid body, no extra articulation body", len(env.robot.body_names) == 4, env.robot.body_names)
    masses = env.robot.root_physx_view.get_masses()
    check("100g racket including handle in every clone", bool(torch.allclose(masses[:, env.paddle_id], torch.full((8,), .1), atol=1e-6)))
    check("total aircraft mass preserved", abs(env.robot_mass_kg - 2.910538) < 1e-4, env.robot_mass_kg)
    com = env.robot.root_physx_view.get_coms()[0, env.paddle_id, :3]
    inertia = env.robot.root_physx_view.get_inertias()[0, env.paddle_id].reshape(3, 3)
    check("active inertia positive", bool((torch.linalg.eigvalsh(inertia) > 0).all()), inertia.tolist())
    check("handle moves COM toward wrist", bool(com[1] > 0), com.tolist())
    report["mass_kg"] = masses[0].tolist()
    report["paddle_com_m"] = com.tolist()
    report["paddle_inertia_kg_m2"] = inertia.tolist()
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath("/World/envs/env_0/Robot/paddle_fixed_joint"))
    expected_offset = torch.tensor([cfg.paddle_offset_ee[0], cfg.paddle_offset_ee[1]-args.mount_delta,
                                    cfg.paddle_offset_ee[2]], device=env.device)
    check("fixed joint uses shifted attachment frame", torch.allclose(torch.tensor(list(joint.GetLocalPos0Attr().Get()), device=env.device), expected_offset, atol=1e-7))
    # Rejected dimensions must fail before any stage mutation.
    for field, value in (("paddle_diameter_m", 0), ("paddle_thickness_m", -1),
                         ("paddle_handle_width_m", float("nan")), ("paddle_mount_delta_m", float("inf"))):
        invalid = cfg.copy()
        setattr(invalid, field, value)
        try:
            attach_paddle(None, "/unused", invalid)
        except ValueError:
            check("reject invalid " + field, True)
        else:
            check("reject invalid " + field, False)

    actions = torch.zeros(8, cfg.action_space, device=env.device)
    ee_id = env.robot.find_bodies("flying_arm_2__link2")[0][0]
    for shoulder, elbow in ((0., 0.), (.4, -.7), (-.5, .8)):
        joints = env.robot.data.default_joint_pos.clone()
        joints[:, env.joint_ids] = torch.tensor([shoulder, elbow], device=env.device)
        env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.step(actions)
        actual_offset = quat_apply_inverse(env.robot.data.body_quat_w[:, ee_id],
            env.robot.data.body_pos_w[:, env.paddle_id] - env.robot.data.body_pos_w[:, ee_id])
        check(f"live mount follows link at {shoulder}/{elbow}", torch.allclose(actual_offset, expected_offset.expand_as(actual_offset), atol=2e-3), actual_offset.tolist())
    env.reset()
    if args.self_collisions:
        for _ in range(5):
            _, _, _, _, info = env.step(actions)
            check("neutral physical self-collision clearance", not bool(info["goalkeeper_robot_contact_crash"].any()))
    radius = args.diameter/2
    handle_y = radius + cfg.paddle_handle_length_m/2 - min(.01, radius/4)
    diagonal = (radius + cfg.ball_radius_m + .012) / math.sqrt(2)
    cases = (("head centre", 0., 0., True), ("head interior", -radius*.4, radius*.4, True),
             ("handle", handle_y, 0., True), ("outside circular rim", -diagonal, diagonal, False))
    for label, y, z, expected_hit in cases:
        env.reset()
        env.step(actions)
        position, quaternion, _ = env._paddle_pose()
        state = env.ball.data.default_root_state.clone()
        local = torch.tensor([.10, y, z], device=env.device).expand(8, -1)
        state[:, :3] = position + env.scene.env_origins + quat_apply(quaternion, local)
        state[:, 7:10] = quat_apply(quaternion, torch.tensor([-2., 0., 0.], device=env.device).expand(8, -1))
        state[:, 10:] = 0.
        env.ball.write_root_state_to_sim(state)
        env._previous_ball.copy_(state[:, :3] - env.scene.env_origins)
        hits = torch.zeros(8, dtype=torch.bool, device=env.device)
        samples = []
        for _ in range(12):
            obs, reward, _, _, info = env.step(actions)
            check_finite = all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(reward).all())
            if not check_finite:
                check(label + " finite task state", False)
            hits |= info["goalkeeper_had_paddle_hit"]
            samples.append({"paddle_hit": info["goalkeeper_had_paddle_hit"].tolist(),
                            "outcome": info["goalkeeper_outcome"].tolist(),
                            "ball_position_m": (env.ball.data.root_pos_w-env.scene.env_origins).tolist()})
        check(label + " real contact classification", bool((hits == expected_hit).all()), hits.tolist())
        report["shots"][label] = samples
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
finally:
    env.close()
    app.close()
