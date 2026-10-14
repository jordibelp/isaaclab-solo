# SPDX-License-Identifier: BSD-3-Clause
"""Reload a PPO checkpoint and run a finite deterministic inference rollout."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--control-mode", default="thrust_and_torque_control")
parser.add_argument("--physics-hz", type=int, default=200)
parser.add_argument("--policy-hz", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.direct.borinot_goalkeeper.agents.rsl_rl_ppo_cfg import BorinotGoalkeeperPPORunnerCfg
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg

cfg = BorinotGoalkeeperEnvCfg()
cfg.control_mode = args.control_mode
cfg.physics_hz, cfg.policy_hz = args.physics_hz, args.policy_hz
cfg.scene.num_envs = 16
cfg.sim.device = args.device
cfg.seed = 42
cfg.curriculum = False
env = RslRlVecEnvWrapper(gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg), clip_actions=1.0)
try:
    runner_cfg = BorinotGoalkeeperPPORunnerCfg()
    runner_cfg.device = args.device
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=args.device)
    obs = env.get_observations()
    counts = torch.zeros(10, dtype=torch.long, device=args.device)
    max_action = 0.0
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs)
            assert actions.shape == (16, cfg.action_space) and torch.isfinite(actions).all()
            max_action = max(max_action, float(actions.abs().max()))
            obs, reward, done, info = env.step(actions)
            assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
            codes = info["goalkeeper_outcome"]
            counts += torch.bincount(codes[codes != 0], minlength=10)
    report = {"passed": True, "steps": args.steps, "num_envs": 16,
              "observation_dimension": cfg.observation_space, "action_dimension": cfg.action_space,
              "control_mode": cfg.control_mode, "physics_hz": cfg.physics_hz, "policy_hz": cfg.policy_hz,
              "max_raw_action": max_action,
              "outcomes": counts.tolist(), "checkpoint_name": args.checkpoint.name,
              "checkpoint_path": str(args.checkpoint.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("CHECKPOINT_VALIDATION=" + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
