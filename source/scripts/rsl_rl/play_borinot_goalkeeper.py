# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play Borinot goalkeeper PPO weights with a fixed goal/shot camera."""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0", choices=["Borinot-Goalkeeper-Direct-v0"])
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True, help="RSL-RL PPO checkpoint; optimizer state is not restored.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration_s", type=float, default=2000.0, help="Playback duration in simulated seconds.")
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True,
                    help="Pace playback at the policy frequency when possible; never skip physics steps.")
parser.add_argument("--camera", choices=["overview", "front", "side", "free"], default="overview",
                    help="Fixed env-0 view. Free uses env.viewer.eye/lookat and allows mouse navigation.")
parser.add_argument("--video", action="store_true", help="Record the first video_length policy steps.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs < 1 or not math.isfinite(args_cli.duration_s) or args_cli.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive (duration_s must be finite).")
if args_cli.video_length < 1:
    parser.error("video_length must be positive.")
if args_cli.video:
    args_cli.enable_cameras = True

# Evaluation stays at one shot level and uses eager forwards unless explicitly overridden.
defaults = {"env.curriculum": "False", "agent.algorithm.torch_compile": "False",
            "agent.local_redundancy.enabled": "False"}
override_keys = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
sys.argv = [sys.argv[0]] + [f"{key}={value}" for key, value in defaults.items()
                          if key not in override_keys] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401 -- registers the task and its PPO algorithm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


def configure_camera(cfg, mode):
    """Frame the goal and throw corridor, independently of robot motion/yaw."""
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    gx, gy, gz = cfg.goal_center
    launch_x = max(cfg.launch_x_range)
    if mode == "overview":
        cfg.viewer.eye = (launch_x + 2.0, gy - 4.5, gz + 1.6)
        cfg.viewer.lookat = ((gx + launch_x) / 2, gy, gz)
    elif mode == "front":
        cfg.viewer.eye = (launch_x + 3.0, gy, gz)
        cfg.viewer.lookat = cfg.goal_center
    elif mode == "side":
        center_x = (gx + launch_x) / 2
        cfg.viewer.eye = (center_x, gy - 5.0, gz + 0.6)
        cfg.viewer.lookat = (center_x, gy, gz)
    # No per-step camera writes: all views allow ordinary viewport orbit/pan/zoom.
    # 'free' also preserves the user's env.viewer.eye/lookat Hydra settings.


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.configure_control()  # derived action/observation sizes and rates after Hydra
    env_cfg.robot.init_state.pos = tuple(env_cfg.robot_home)
    configure_camera(env_cfg, args_cli.camera)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        video_dir = None
        if args_cli.video:
            video_dir = args_cli.video_dir or (
                Path("logs/rsl_rl/borinot_goalkeeper/play") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir),
                                           step_trigger=lambda step: step == 0,
                                           video_length=args_cli.video_length, disable_logger=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        dt = env.unwrapped.step_dt
        total_steps = math.ceil(args_cli.duration_s / dt)
        if args_cli.video:
            total_steps = min(total_steps, args_cli.video_length)
        counts = torch.zeros(10, dtype=torch.long, device=env.device)
        terminated_episodes = 0
        truncated_episodes = 0
        print(f"[INFO] Goalkeeper: {env_cfg.control_mode}, {env_cfg.physics_hz}/{env_cfg.policy_hz} Hz, "
              f"home={env_cfg.robot_home}, shot_level={env_cfg.shot_level}, curriculum={env_cfg.curriculum}, "
              f"camera={args_cli.camera}, {total_steps} policy steps", flush=True)
        completed_steps = 0
        with torch.inference_mode():
            for _ in range(total_steps):
                if not simulation_app.is_running():
                    break
                start = time.perf_counter()
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("The checkpoint produced non-finite actions.")
                obs, rewards, dones, info = env.step(actions)
                if not torch.isfinite(rewards).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("The environment produced non-finite rewards or observations.")
                runner.alg.policy.reset(dones)
                codes = info["goalkeeper_outcome"]
                counts += torch.bincount(codes[info["goalkeeper_shot_finished"]], minlength=10)
                terminated_episodes += int(env.unwrapped.reset_terminated.sum())
                truncated_episodes += int(env.unwrapped.reset_time_outs.sum())
                completed_steps += 1
                remaining = dt - (time.perf_counter() - start)
                if args_cli.real_time and remaining > 0:
                    time.sleep(remaining)
        names = ("ongoing", "base_hit", "arm_hit", "goal", "ball_floor", "leave_goal_area",
                 "deadline", "save", "robot_crash", "miss")
        report = {"checkpoint_name": Path(checkpoint).name, "checkpoint_path": str(checkpoint),
                  "steps": completed_steps, "simulated_time_s": completed_steps * dt,
                  "num_envs": env.num_envs, "control_mode": env_cfg.control_mode,
                  "physics_hz": env_cfg.physics_hz, "policy_hz": env_cfg.policy_hz,
                  "robot_home": list(env_cfg.robot_home), "shot_level_start": env_cfg.shot_level,
                  "shot_level_end": env.unwrapped.shot_level, "curriculum": env_cfg.curriculum,
                  "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
                  "camera": args_cli.camera, "camera_eye": list(env_cfg.viewer.eye),
                  "camera_lookat": list(env_cfg.viewer.lookat),
                  "completed_outcomes": dict(zip(names, counts.cpu().tolist())),
                  "terminated_episodes": terminated_episodes, "time_limit_episodes": truncated_episodes,
                  "video_dir": str(video_dir) if video_dir else None}
        print("GOALKEEPER_PLAY=" + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
