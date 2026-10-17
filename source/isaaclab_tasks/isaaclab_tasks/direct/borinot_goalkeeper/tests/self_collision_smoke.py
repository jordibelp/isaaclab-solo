# SPDX-License-Identifier: BSD-3-Clause
"""Verify self-contact reset filtering with real Hydra overrides and PhysX forces."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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
    cfg.seed = 230999
    cfg.curriculum = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    report = {"self_collision_triggers_reset": cfg.self_collision_triggers_reset,
              "enabled_self_collisions": cfg.enabled_self_collisions, "checks": [], "contacts": {}}

    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        print(f"CHECK {name}: {bool(condition)}", flush=True)
        assert condition, name

    def pose(height, folded):
        env.reset()
        ids = torch.arange(env.num_envs, device=env.device)
        env._ball_active[:] = False
        env._park_ball(ids)
        env._next_launch_steps[:] = 100000
        root = env.robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins + torch.tensor([0.2, 0.0, height], device=env.device)
        root[:, 7:] = 0.0
        q = torch.zeros_like(env.robot.data.default_joint_pos)
        q[folded] = torch.deg2rad(torch.tensor([70., -160.], device=env.device))
        env.robot.write_root_state_to_sim(root)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))

    def advance():
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        for flag in env._flags.values():
            flag.zero_()
        return env._get_dones()

    def forces():
        total = env.robot_contact.data.net_forces_w[:, env._contact_ids][:, env._crash_contact_columns]
        own = torch.zeros_like(total)
        for column, sensor in enumerate(env._self_contact_sensors):
            own[:, column] = sensor.data.force_matrix_w[:, 0].sum(1)
        return total, own

    try:
        check("Hydra preserves physical self-collisions", cfg.robot.spawn.articulation_props.enabled_self_collisions)
        check("extra sensors only on opt-out path", len(env._self_contact_sensors) == (
            0 if cfg.self_collision_triggers_reset else len(cfg.robot_crash_body_names)))
        folded = torch.tensor([0, 2], device=env.device)
        pose(1.5, folded)
        for _ in range(24):
            terminated, truncated = advance()
            if env._flags["crash"][folded].all():
                break
        total, own = forces()
        report["contacts"]["self_only"] = {"net_n": total.tolist(), "self_n": own.tolist()}
        check("real folded-arm/base crash in selected envs", env._flags["crash"].tolist() == [True, False, True, False])
        check("self-contact has no scene or tilt violation", not (env._flags["area"] | env._flags["tilt"]).any())
        check("self-contact reset follows flag", terminated.tolist() == [cfg.self_collision_triggers_reset, False,
                                                                       cfg.self_collision_triggers_reset, False])
        check("self-contact preserves crash outcome", env._outcome.tolist() == [8, 0, 8, 0])
        env._get_rewards()
        check("self-contact preserves crash penalty", torch.equal(
            env.extras["goalkeeper_reward_terms"]["outcome"][folded],
            torch.full((2,), cfg.rwd_scale_robot_crash, device=env.device)))
        if not cfg.self_collision_triggers_reset:
            check("filtered self forces match base net forces", torch.allclose(total, own, atol=1e-4, rtol=1e-5))
            check("self-contact is not a scene crash", not env._flags["external_crash"].any())

        for name, joint_ids in (("base_floor", []), ("self_and_floor", list(range(4)))):
            saved_min, saved_max = env._area_min.clone(), env._area_max.clone()
            # The imposed overlapping pose is a force-attribution fixture, not a
            # reachable rollout. Isolate contact resets from its violent response.
            if name == "self_and_floor":
                env._area_min.fill_(-10.)
                env._area_max.fill_(10.)
                cfg.reset_robot_on_tilt_threshold = False
            pose(0.14 if name == "base_floor" else 0.08, joint_ids)
            for _ in range(24):
                terminated, _ = advance()
                total, own = forces()
                scene = (total - own).norm(dim=-1).amax(-1)
                if (scene > cfg.robot_crash_force_threshold_n).all():
                    break
            report["contacts"][name] = {"net_n": total.tolist(), "self_n": own.tolist()}
            check(f"{name} real scene force", (scene > cfg.robot_crash_force_threshold_n).all())
            check(f"{name} resets", terminated.all())
            cfg.reset_robot_on_robot_crash = False
            check(f"{name} reset requires contact switch", not env._get_dones()[0].any())
            cfg.reset_robot_on_robot_crash = True
            if not cfg.self_collision_triggers_reset:
                check(f"{name} attributed to scene", env._flags["external_crash"].all())
                if name == "self_and_floor":
                    check("simultaneous self and scene forces", (own.norm(dim=-1).amax(-1) > 0.1).any())
            env._area_min.copy_(saved_min)
            env._area_max.copy_(saved_max)
            cfg.reset_robot_on_tilt_threshold = True

        pose(1.5, [])
        root = env.robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        root[:, 3:7] = torch.tensor([2**-0.5, 2**-0.5, 0., 0.], device=env.device)
        env.robot.write_root_state_to_sim(root)
        terminated, _ = advance()
        check("tilt reset unchanged", terminated.all() and env._flags["tilt"].all())

        pose(1.5, [])
        root = env.robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins + torch.tensor([0.95, 0., 1.5], device=env.device)
        env.robot.write_root_state_to_sim(root)
        terminated, _ = advance()
        check("area reset unchanged", terminated.all() and env._flags["area"].all())

        env._flags["external_crash"][:] = True
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial reset clears only selected contact latch", env._flags["external_crash"].tolist() == [False, True, True, True])
        env.reset()
        for _ in range(32):
            obs, reward, *_ = env.step(torch.zeros(4, cfg.action_space, device=env.device))
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
        check("32 finite policy steps", True)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
