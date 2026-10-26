# SPDX-License-Identifier: BSD-3-Clause
"""Verify legacy Borinot delay through its real physics-rate action path."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--torque", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
sys.argv = [sys.argv[0]] + overrides
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-Direct-v0", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 3
    cfg.sim.device = args.device
    cfg.seed = 8
    if args.torque:
        cfg.des_joint_out, cfg.action_space, cfg.observation_space = False, 8, 32
        for actuator in cfg.robot_cfg.actuators.values():
            actuator.stiffness, actuator.damping = 0., 0.
    env = gym.make("Borinot-Direct-v0", cfg=cfg).unwrapped
    try:
        env.reset()
        steps = env._actuator_delay.steps.clone()
        observed = []
        name = "_map_actions" if args.torque else "_map_actions_angle_des"
        original = getattr(env, name)

        def mapped(actions=None):
            observed.append(actions.clone())
            return original(actions)

        setattr(env, name, mapped)
        action = torch.full_like(env.actions, .2)
        with torch.inference_mode():
            for _ in range(8):
                obs, rewards, terminated, truncated, info = env.step(action)
                assert torch.isfinite(rewards).all() and torch.isfinite(obs["policy"]).all()
        assert len(observed) == 8 * cfg.decimation
        for tick, arrived in enumerate(observed):
            expected = torch.where((tick >= steps).unsqueeze(1), action, 0.)
            torch.testing.assert_close(arrived, expected, rtol=0, atol=0)
        torch.testing.assert_close(env.actions, action, rtol=0, atol=0)
        # A physical reset clears selected queues without touching the other environments.
        before = env._actuator_delay._queue.clone()
        with torch.inference_mode():
            env._reset_idx(torch.tensor([1], device=env.device))
        assert not env._actuator_delay._queue[:, 1].any()
        torch.testing.assert_close(env._actuator_delay._queue[:, [0, 2]], before[:, [0, 2]], rtol=0, atol=0)
        report = {"task": "Borinot-Direct-v0", "torque": args.torque, "physics_dt": cfg.sim.dt,
                  "actuator_delay": cfg.actuator_delay, "actuator_delay_range": cfg.actuator_delay_range,
                  "steps": steps.tolist(), "physics_commands_checked": len(observed),
                  "finite_rollout": True, "exact_command_timing": True, "reset_isolation": True}
        args.output.write_text(json.dumps(report, indent=2))
        print("LEGACY_DELAY_OK=" + json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
