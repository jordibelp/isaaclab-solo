# SPDX-License-Identifier: BSD-3-Clause
"""Verify ground-contact filtering through Hydra and real PhysX forces."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.curriculum = False
    cfg.seed = 230999
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    report = {"reset_on_leg_or_racket_contact": cfg.reset_on_leg_or_racket_contact,
              "goal_area_z_m": list(cfg.goal_area_z_m), "checks": [], "contacts": {}}

    def check(name, condition, detail=None):
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        print(f"CHECK {name}: {bool(condition)} {detail if detail is not None else ''}", flush=True)
        if not condition:
            raise AssertionError(name)

    def sample_ground(name, height, joints, expected_floor_column, expected_crash=None):
        env.reset()
        env._ball_active[:] = False
        env._park_ball(torch.arange(env.num_envs, device=env.device))
        root = env.robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins + torch.tensor([0.2, 0.0, height], device=env.device)
        root[:, 7:] = 0.0
        q = env.robot.data.default_joint_pos.clone()
        q[:, env.joint_ids] = torch.tensor(joints, device=env.device)
        env.robot.write_root_state_to_sim(root)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        maximum = torch.zeros(env.num_envs, 3, device=env.device)
        net_maximum = torch.zeros(env.num_envs, 4, device=env.device)
        detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for _ in range(12):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            for flag in env._flags.values():
                flag.zero_()
            floor = env.floor_contact.data.force_matrix_w[:, 0]
            net = env.robot_contact.data.net_forces_w[:, env._contact_ids]
            maximum = torch.maximum(maximum, floor.norm(dim=-1))
            net_maximum = torch.maximum(net_maximum, net.norm(dim=-1))
            contact = (floor[:, expected_floor_column].norm(dim=-1) if expected_floor_column is not None
                       else net[:, 1].norm(dim=-1)) > cfg.robot_crash_force_threshold_n
            if contact.any():
                terminated, _ = env._get_dones()
                expected = cfg.reset_on_leg_or_racket_contact if expected_crash is None else expected_crash
                check(name + " crash attribution", bool((env._flags["crash"] == expected).all()),
                      {"crash": env._flags["crash"].tolist(), "floor_forces_n": floor.norm(dim=-1).tolist(),
                       "net_forces_n": net.norm(dim=-1).tolist(), "area": env._flags["area"].tolist()})
                check(name + " termination", bool((terminated == expected).all()))
                env._get_rewards()
                outcome_reward = env.extras["goalkeeper_reward_terms"]["outcome"]
                expected_reward = cfg.rwd_scale_robot_crash if expected else 0.0
                check(name + " crash reward", bool((outcome_reward == expected_reward).all()))
                detected |= contact
                break
        report["contacts"][name] = {"floor_max_n": maximum.tolist(), "robot_max_n": net_maximum.tolist()}
        check(name + " real contact in every clone", bool(detected.all()))

    try:
        env.reset()
        check("filtered floor contract", env.floor_contact.data.force_matrix_w.shape == (4, 1, 3, 3))
        check("control timing", env.physics_dt == 0.005 and env.step_dt == 0.01)
        check("self collisions enabled", cfg.robot.spawn.articulation_props.enabled_self_collisions)
        sample_ground("racket", 0.515, (0.0, 0.0), 0)
        # A low base must remain fatal even with simultaneous leg/racket contacts.
        sample_ground("base", 0.14, (0.0, 0.0), None, expected_crash=True)

        env.reset()
        tilted = env.robot.data.default_root_state.clone()
        tilted[:, :3] += env.scene.env_origins
        tilted[:, 3:7] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0], device=env.device)
        env.robot.write_root_state_to_sim(tilted)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        terminated, _ = env._get_dones()
        check("tilt reset remains active", bool(terminated.all() and env._flags["tilt"].all()))

        # Default area bounds would still reject a low base without the launch override.
        cfg.goal_area_z_m = (0.55, 2.6)
        env._area_min[2] = 0.55
        low = env.robot.data.default_root_state.clone()
        low[:, :3] += env.scene.env_origins
        low[:, 2] = env.scene.env_origins[:, 2] + 0.50
        env.robot.write_root_state_to_sim(low)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        for flag in env._flags.values():
            flag.zero_()
        env._sample_physics_events()
        check("height bound remains independent", bool(env._flags["area"].all()))

        # Sensor state follows partial resets without clearing other clones.
        before = env.floor_contact._data.force_matrix_w[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial sensor reset isolation", torch.equal(env.floor_contact._data.force_matrix_w[1:], before)
              and not env.floor_contact._data.force_matrix_w[0].any())
        check("finite observations", all(torch.isfinite(v).all() for v in env._get_observations().values()))
        env.reset()
        actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        for _ in range(32):
            obs, reward, terminated, truncated, info = env.step(actions)
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
        check("32 finite full policy steps", True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
