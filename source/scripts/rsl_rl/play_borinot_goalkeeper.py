# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play Borinot goalkeeper PPO or SAC weights with a match panel and recent-ball physics."""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

TASKS = ("Borinot-Goalkeeper-Direct-v0", "Borinot-Goalkeeper-Simple-Direct-v0")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASKS[0], choices=TASKS)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint; optimizer state is not restored.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--duration_s", type=float, default=2000.0, help="Playback duration in simulated seconds.")
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True,
                    help="Pace playback at the policy frequency when possible; never skip physics steps.")
parser.add_argument("--camera", choices=["overview", "front", "side", "top", "keeper", "follow", "orbit", "free"],
                    default="overview", help="Initial env-0 camera; switch live using the match panel.")
parser.add_argument("--keep-balls", type=int, default=5,
                    help="Keep the last N balls including the live shot (0 disables history, max 20).")
parser.add_argument("--playback-speed", type=float, default=1.0,
                    help="Wall-clock speed; does not change physics/policy rates.")
parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True, help="Show the match panel in GUI mode.")
parser.add_argument("--video", action="store_true", help="Record the first video_length policy steps.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs < 1 or not math.isfinite(args_cli.duration_s) or args_cli.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive (duration_s must be finite).")
if args_cli.video_length < 1:
    parser.error("video_length must be positive.")
if not 0 <= args_cli.keep_balls <= 20:
    parser.error("keep-balls must be between 0 and 20.")
if not math.isfinite(args_cli.playback_speed) or args_cli.playback_speed <= 0:
    parser.error("playback-speed must be positive and finite.")
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401 -- registers the task and its PPO algorithm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_goalkeeper.playback import BorinotGoalkeeperPlayEnv
from isaaclab_tasks.direct.borinot_goalkeeper.playback_ui import PlayControls, camera_pose
from isaaclab_tasks.direct.borinot_goalkeeper_simple.playback import BorinotGoalkeeperSimplePlayEnv


def configure_camera(cfg, mode):
    """Set the initial env-0 view; the match controls handle live camera changes."""
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    pose = camera_pose(cfg, mode)
    if pose is not None:
        cfg.viewer.eye, cfg.viewer.lookat = pose
    # Static presets allow mouse navigation; Free preserves Hydra eye/lookat.


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.device = env_cfg.sim.device
    configure_environment = getattr(agent_cfg, "configure_environment", None)
    if configure_environment is not None:
        configure_environment(env_cfg)
    env_cfg.configure_control()  # derived action/observation sizes and rates after Hydra
    env_cfg.robot.init_state.pos = tuple(env_cfg.robot_home)
    configure_camera(env_cfg, args_cli.camera)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    # Only this entry point constructs the play subclass. Training keeps its
    # original single ball, observations, rewards, collisions and reset rules.
    play_env = BorinotGoalkeeperSimplePlayEnv if args_cli.task == TASKS[1] else BorinotGoalkeeperPlayEnv
    env = play_env(cfg=env_cfg, keep_balls=args_cli.keep_balls, render_mode="rgb_array" if args_cli.video else None)
    controls = None
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
        if agent_cfg.class_name == "OffPolicyRunner":
            # Playback never collects replay; avoid allocating the training-sized buffer.
            agent_cfg.algorithm.replay_buffer_size = env.num_envs
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
            policy_model = runner.alg.actor
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_optimizer=False)
            policy_model = runner.alg.policy
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        controls = PlayControls(env.unwrapped, args_cli.camera, args_cli.playback_speed,
                                show_ui=args_cli.ui and env.unwrapped.sim.has_gui())
        dt = env.unwrapped.step_dt
        total_steps = math.ceil(args_cli.duration_s / dt)
        if args_cli.video:
            total_steps = min(total_steps, args_cli.video_length)
        counts = torch.zeros(10, dtype=torch.long, device=env.device)
        terminated_episodes = 0
        truncated_episodes = 0
        print(f"[INFO] {args_cli.task}: {env_cfg.control_mode}, {env_cfg.physics_hz}/{env_cfg.policy_hz} Hz, "
              f"actuator_delay={env_cfg.actuator_delay} s, actuator_delay_range={env_cfg.actuator_delay_range}, "
              f"home={env_cfg.robot_home}, shot_level={env_cfg.shot_level}, curriculum={env_cfg.curriculum}, "
              f"reset_on_goal={env_cfg.reset_on_goal}, camera={args_cli.camera}, "
              f"keep_balls={args_cli.keep_balls}, {total_steps} policy steps", flush=True)
        completed_steps = 0
        with torch.inference_mode():
            while completed_steps < total_steps:
                if not simulation_app.is_running():
                    break
                start = time.perf_counter()
                if controls.reset_requested:
                    env.unwrapped.reset_borinot()
                    policy_model.reset(torch.ones(env.num_envs, device=env.device, dtype=torch.bool))
                    obs = env.get_observations()
                    controls.reset_requested = False
                    controls.manual_resets += 1
                controls.update(completed_steps * dt)
                if controls.paused:
                    env.unwrapped.sim.render()
                    time.sleep(0.02)
                    continue
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("The checkpoint produced non-finite actions.")
                obs, rewards, dones, info = env.step(actions)
                if not torch.isfinite(rewards).all() or not torch.isfinite(obs["policy"]).all():
                    raise RuntimeError("The environment produced non-finite rewards or observations.")
                policy_model.reset(dones)
                codes = info["goalkeeper_outcome"]
                counts += torch.bincount(codes[info["goalkeeper_shot_finished"]], minlength=10)
                controls.score.record(int(codes[0]), bool(info["goalkeeper_shot_finished"][0]))
                terminated_episodes += int(env.unwrapped.reset_terminated.sum())
                truncated_episodes += int(env.unwrapped.reset_time_outs.sum())
                completed_steps += 1
                remaining = dt / controls.speed - (time.perf_counter() - start)
                if args_cli.real_time and remaining > 0:
                    time.sleep(remaining)
        names = ("ongoing", "base_hit", "arm_hit", "goal", "ball_floor", "leave_goal_area",
                 "deadline", "save", "robot_crash", "miss")
        # The simplified task uses only "goal" and "save" (shot ended without a goal).
        report = {"task": args_cli.task, "checkpoint_name": Path(checkpoint).name, "checkpoint_path": str(checkpoint),
                  "steps": completed_steps, "simulated_time_s": completed_steps * dt,
                  "num_envs": env.num_envs, "control_mode": env_cfg.control_mode,
                  "physics_hz": env_cfg.physics_hz, "policy_hz": env_cfg.policy_hz,
                  "actuator_delay_s": env_cfg.actuator_delay, "actuator_delay_range_s": env_cfg.actuator_delay_range,
                  "robot_home": list(env_cfg.robot_home), "shot_level_start": env_cfg.shot_level,
                  "shot_level_end": env.unwrapped.shot_level, "curriculum": env_cfg.curriculum,
                  "reset_on_goal": env_cfg.reset_on_goal,
                  "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
                  "camera": args_cli.camera, "camera_eye": list(env_cfg.viewer.eye),
                  "camera_lookat": list(env_cfg.viewer.lookat),
                  "camera_end": controls.camera, "keep_balls": args_cli.keep_balls,
                  "playback_speed_end": controls.speed, "manual_resets": controls.manual_resets,
                  "match_score_env_0": {"goals": controls.score.counts[3], "blocked": controls.score.counts[7],
                                        "other": controls.score.other, "shots": controls.score.shots},
                  "completed_outcomes": dict(zip(names, counts.cpu().tolist())),
                  "terminated_episodes": terminated_episodes, "time_limit_episodes": truncated_episodes,
                  "video_dir": str(video_dir) if video_dir else None}
        print("GOALKEEPER_PLAY=" + json.dumps(report), flush=True)
    finally:
        if controls is not None:
            controls.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
