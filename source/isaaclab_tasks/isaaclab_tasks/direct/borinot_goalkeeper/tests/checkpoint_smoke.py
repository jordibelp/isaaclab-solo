# SPDX-License-Identifier: BSD-3-Clause
"""Reload a PPO checkpoint and run a finite deterministic inference rollout."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0",
                    choices=["Borinot-Direct-v0", "Borinot-Goalkeeper-Direct-v0"])
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
from isaaclab_tasks.utils import load_cfg_from_registry

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
if hasattr(cfg, "control_mode"):
    cfg.control_mode = args.control_mode
    cfg.physics_hz, cfg.policy_hz = args.physics_hz, args.policy_hz
cfg.scene.num_envs = 16
cfg.sim.device = args.device
cfg.seed = 42
if hasattr(cfg, "curriculum"):
    cfg.curriculum = False
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=1.0)
try:
    runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner_cfg.device = args.device
    runner_cfg.algorithm.torch_compile = False  # check portable weights with eager inference
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_optimizer=False)  # inference does not need Adam/AdamW state
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
            if "goalkeeper_outcome" in info:
                codes = info["goalkeeper_outcome"]
                counts += torch.bincount(codes[codes != 0], minlength=10)
    report = {"passed": True, "steps": args.steps, "num_envs": 16,
              "observation_dimension": cfg.observation_space, "action_dimension": cfg.action_space,
              "task": args.task, "control_mode": getattr(cfg, "control_mode", None),
              "physics_hz": 1/env.unwrapped.physics_dt, "policy_hz": 1/env.unwrapped.step_dt,
              "max_raw_action": max_action,
              "outcomes": counts.tolist(), "checkpoint_name": args.checkpoint.name,
              "checkpoint_path": str(args.checkpoint.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("CHECKPOINT_VALIDATION=" + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
