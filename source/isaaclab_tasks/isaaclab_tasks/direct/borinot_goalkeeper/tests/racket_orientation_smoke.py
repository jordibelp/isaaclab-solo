# SPDX-License-Identifier: BSD-3-Clause
"""Verify Hydra-selected racket orientation in active PhysX, observations and contacts."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
sys.argv = [sys.argv[0], *overrides]
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import UsdPhysics

import isaaclab_tasks
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.curriculum = False
    cfg.seed = 42
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    cfg.hold_arm = True
    cfg.ball.spawn.rigid_props.disable_gravity = True
    report = {"parallel": cfg.racket_parallel_to_arm_plane, "checks": [], "poses": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def check(name, condition):
        passed = bool(condition)
        report["checks"].append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    try:
        env.reset()
        check("Hydra override remains a boolean", type(cfg.racket_parallel_to_arm_plane) is bool)
        ee_id = env.robot.find_bodies("flying_arm_2__link2")[0][0]
        axis = torch.tensor([0., 0., 1.] if cfg.racket_parallel_to_arm_plane else [1., 0., 0.], device=env.device)
        handle = torch.tensor([0., 1., 0.], device=env.device).expand(4, -1)
        offset = torch.tensor([cfg.paddle_offset_ee[0], cfg.paddle_offset_ee[1] - cfg.paddle_mount_delta_m,
                               cfg.paddle_offset_ee[2]], device=env.device)
        joint = UsdPhysics.FixedJoint(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/paddle_fixed_joint"))
        rotation = joint.GetLocalRot0Attr().Get()
        rotation = torch.tensor([rotation.GetReal(), *rotation.GetImaginary()], device=env.device).expand(4, -1)
        check("fixed-joint normal selects requested plane", torch.allclose(quat_apply(rotation, env._x_axis), axis.expand(4, -1), atol=1e-6))
        check("fixed-joint handle axis unchanged", torch.allclose(quat_apply(rotation, handle), handle, atol=1e-6))
        check("fixed-joint mount offset unchanged", torch.allclose(torch.tensor(list(joint.GetLocalPos0Attr().Get()), device=env.device), offset, atol=1e-6))
        masses = env.robot.root_physx_view.get_masses()
        check("racket mass unchanged in every clone", torch.allclose(masses[:, env.paddle_id], torch.full((4,), .1), atol=1e-6))
        check("total robot mass unchanged", abs(env.robot_mass_kg - 2.910538) < 1e-4)

        actions = torch.zeros(4, cfg.action_space, device=env.device)

        def check_pose(label):
            env.sim.forward()
            env.scene.update(env.physics_dt)
            position, quaternion, normal = env._paddle_pose()
            ee_quat = env.robot.data.body_quat_w[:, ee_id]
            relative_normal = quat_apply_inverse(ee_quat, normal)
            relative_offset = quat_apply_inverse(ee_quat, position + env.scene.env_origins - env.robot.data.body_pos_w[:, ee_id])
            check(label + " active face normal", torch.allclose(relative_normal, axis.expand(4, -1), atol=2e-4))
            check(label + " active handle direction", torch.allclose(quat_apply_inverse(ee_quat, quat_apply(quaternion, handle)), handle, atol=2e-4))
            check(label + " active head centre", torch.allclose(relative_offset, offset.expand(4, -1), atol=2e-4))
            obs = env._get_observations()
            expected = quat_apply_inverse(env.robot.data.root_quat_w, normal)
            for group, width in (("policy", cfg.observation_space), ("critic", cfg.state_space)):
                check(label + " " + group + " shape unchanged", obs[group].shape == (4, width))
                check(label + " " + group + " sees physical normal", torch.allclose(obs[group][:, 25:28], expected, atol=1e-6))
            report["poses"].append({"label": label, "position": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
                                    "normal": normal.tolist(), "relative_normal": relative_normal.tolist()})

        joints = env.robot.data.default_joint_pos.clone()
        joints[:, env.joint_ids] = torch.tensor([[0., 0.], [.4, -.7], [-.5, .8], [.7, .5]], device=env.device)
        env.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        root = env.robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        angles = torch.tensor([[0., 0., -1.57079632679], [.2, -.3, .7], [-.3, .2, -.9], [.1, .4, 1.2]], device=env.device)
        root[:, 3:7] = quat_from_euler_xyz(*angles.unbind(-1))
        env.robot.write_root_state_to_sim(root)
        check_pose("bent joints and tilted roots")
        env._reset_idx(torch.tensor([1, 3], device=env.device))
        check_pose("partial reset")
        env.reset()
        for _ in range(16):
            obs, reward, _, _, _ = env.step(actions)
            check("finite rollout", torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values()))
        check_pose("after physics steps")

        # Launch along the actual face normal: this must hit in BOTH orientations.
        env.reset()
        env.step(actions)
        position, quaternion, _ = env._paddle_pose()
        state = env.ball.data.default_root_state.clone()
        state[:, :3] = position + env.scene.env_origins + quat_apply(quaternion, env._x_axis * .10)
        state[:, 7:10] = quat_apply(quaternion, env._x_axis * -2.)
        state[:, 10:] = 0.
        env.ball.write_root_state_to_sim(state)
        env._previous_ball.copy_(state[:, :3] - env.scene.env_origins)
        hits = torch.zeros(4, dtype=torch.bool, device=env.device)
        for _ in range(12):
            _, _, _, _, info = env.step(actions)
            hits |= info["goalkeeper_had_paddle_hit"]
        check("real rotated-head contacts attributed to paddle", hits.all())
        report["passed"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
