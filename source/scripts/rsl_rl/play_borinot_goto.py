# SPDX-License-Identifier: BSD-3-Clause
"""Play Borinot go-to PPO/SAC weights with a live XYZ goal and deadline, or with training goal sampling."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-goto", choices=("Borinot-goto", "Borinot-GoTo-Direct-v0"))
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--goal_init", "--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Initial XYZ in metres, interpreted with --goal-frame. Default: home (world) or zero (relative).")
parser.add_argument("--goal-frame", choices=("world", "relative"), default="world",
                    help="World position or displacement from each base position along world axes. Also switchable live.")
parser.add_argument("--log_interval_s", type=float, default=1.0,
                    help="Print robot 0 position/velocity/goal/error/time every N simulated seconds; 0 disables it.")
parser.add_argument("--deadline_s", type=float, default=None,
                    help="Seconds per goal. Default: middle of the training deadline range for the goal distance.")
parser.add_argument("--reset-at-deadline", action="store_true",
                    help="Training episodes: reset the robot to home at every deadline and repeat the goal.")
parser.add_argument("--random-goals", action="store_true", help="Use training goal/deadline sampling instead of live goals.")
parser.add_argument("--curriculum-level", type=int, default=None,
                    help="Fixed goal stage. Default: saved checkpoint stage, then environment default.")
parser.add_argument("--camera", choices=("goal", "follow", "chase", "side", "top", "free"), default="goal")
parser.add_argument("--ui", action=argparse.BooleanOptionalAction, default=True)
for axis in ("x", "y", "z"):
    parser.add_argument(f"--{axis}_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help=f"Goal {axis} slider range. Default: home +/- the stage goal box.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or not math.isfinite(args.duration_s) or args.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive and finite.")
if args.goal_init is not None and not all(map(math.isfinite, args.goal_init)):
    parser.error("goal_init must contain three finite values: x y z.")
if args.deadline_s is not None and (not math.isfinite(args.deadline_s) or args.deadline_s <= 0):
    parser.error("deadline_s must be positive and finite.")
if not math.isfinite(args.log_interval_s) or args.log_interval_s < 0:
    parser.error("log_interval_s must be finite and nonnegative.")
if args.random_goals and (args.goal_init is not None or args.deadline_s is not None or args.reset_at_deadline
                         or args.goal_frame != "world"):
    parser.error("--random-goals samples goals and deadlines; do not combine it with live-goal options.")
for axis in ("x", "y", "z"):
    bounds = getattr(args, f"{axis}_range")
    if bounds is not None and (not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]):
        parser.error(f"{axis}_range must contain two finite values with MIN < MAX.")
defaults = {"env.curriculum": "False", "env.goal_debug_vis": "True", "agent.algorithm.torch_compile": "False",
            "agent.local_redundancy.enabled": "False"}
override_keys = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
sys.argv = [sys.argv[0]] + [f"{k}={v}" for k, v in defaults.items() if k not in override_keys] + hydra_args
simulation_app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_cmd_vel.playback_ui import camera_pose
from isaaclab_tasks.direct.borinot_goto.playback import BorinotGoToPlayEnv, checkpoint_goal_level
from isaaclab_tasks.direct.borinot_goto.playback_ui import (
    GoalControls, describe_result, describe_state, flight_state, goal_camera_pose)
from isaaclab_tasks.direct.borinot_goto.task_math import stage_settings


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    checkpoint = retrieve_file_path(str(Path(args.checkpoint).expanduser()))
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_cfg.goal_curriculum_start_level = (args.curriculum_level if args.curriculum_level is not None else
        checkpoint_goal_level(metadata.get("infos") or {}, env_cfg.goal_curriculum_start_level, override_keys))
    del metadata
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg.device = env_cfg.sim.device
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(env_cfg)
    env_cfg.configure_control()
    home = tuple(env_cfg.robot_home)
    relative = args.goal_frame == "relative"
    default_goal = (0.0, 0.0, 0.0) if relative else home
    goal = None if args.random_goals else tuple(args.goal_init if args.goal_init is not None else default_goal)
    difficulty, factor = stage_settings(env_cfg, env_cfg.goal_curriculum_start_level)
    ranges = []
    for index, axis in enumerate("xyz"):
        bounds = getattr(args, f"{axis}_range")
        if bounds is None:
            extent = env_cfg.goal_half_extents_m[difficulty][index]
            # Support both live input modes, including negative relative Z and absolute height.
            bounds = (min(-extent, home[index] - extent), max(extent, home[index] + extent))
            if goal is not None:
                bounds = (min(bounds[0], goal[index]), max(bounds[1], goal[index]))
        if goal is not None and not bounds[0] <= goal[index] <= bounds[1]:
            raise ValueError(f"{axis}_range must include the initial goal.")
        ranges.append(bounds)
    ranges.append((0.1, max(5.0, args.deadline_s or 0.0)))
    env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
    camera_goal = tuple(h + d for h, d in zip(home, goal)) if relative and goal is not None else goal
    pose = goal_camera_pose(home, camera_goal or home) if args.camera == "goal" else camera_pose(args.camera, home)
    if pose is not None:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = pose
    env = RslRlVecEnvWrapper(BorinotGoToPlayEnv(env_cfg, goal=goal, deadline_s=args.deadline_s,
                                                reset_at_deadline=args.reset_at_deadline, goal_relative=relative),
                            clip_actions=agent_cfg.clip_actions)
    play_env = env.unwrapped
    controls = None
    try:
        if agent_cfg.class_name == "OffPolicyRunner":
            agent_cfg.algorithm.replay_buffer_size = env.num_envs
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
            model = runner.alg.actor
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_optimizer=False)
            model = runner.alg.policy
        policy = runner.get_inference_policy(device=env.device)
        controls = GoalControls(play_env, goal or home, args.deadline_s, ranges, args.camera,
                                show_ui=args.ui and not args.random_goals and play_env.sim.has_gui(), relative=relative)
        initial_goal = (play_env._goal_position_w[0] - play_env.scene.env_origins[0]).tolist()
        mode = "random" if args.random_goals else ("episodes" if args.reset_at_deadline else "chain")
        print(f"[INFO] Borinot go-to: {mode} mode, {env_cfg.control_mode}, {env_cfg.physics_hz}/{env_cfg.policy_hz} Hz, "
              f"actuator_delay={env_cfg.actuator_delay} s, actuator_delay_range={env_cfg.actuator_delay_range}, "
              f"goal stage {play_env.goal_curriculum.level} (box +/-{env_cfg.goal_half_extents_m[difficulty]} m, "
              f"deadlines {env_cfg.goal_deadline_range_s[difficulty]} s), goal={goal or 'sampled'} m, "
              f"input_frame={args.goal_frame}, deadline={args.deadline_s or 'auto'}, "
              f"checkpoint={Path(checkpoint).name}", flush=True)
        observation = env.get_observations()
        every_env = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        total_steps = math.ceil(args.duration_s / play_env.step_dt)
        completed = resets = printed = 0
        next_log_s = 0.0
        with torch.inference_mode():
            while completed < total_steps and simulation_app.is_running():
                start = time.perf_counter()
                if controls.reset_requested:
                    observation, _ = env.reset()
                    model.reset(every_env)
                    controls.reset_requested = False
                    controls.manual_resets += 1
                if controls.go_requested:
                    controls.go_requested = False
                    try:
                        play_env.set_play_goal(controls.goal, controls.requested_deadline(), relative=controls.relative)
                    except ValueError as error:
                        controls.message = f"Goal not sent: {error}"
                    else:
                        controls.message = ""
                        if play_env.reset_at_deadline:
                            observation, _ = env.reset()  # a new goal starts a new training-style episode
                            model.reset(every_env)
                        else:
                            observation = env.get_observations()
                controls.update(completed * play_env.step_dt)
                if controls.paused:
                    play_env.sim.render()
                    time.sleep(0.02)
                    continue
                time_s = completed * play_env.step_dt
                if args.log_interval_s > 0 and time_s + 1e-9 >= next_log_s:
                    print(describe_state(flight_state(play_env, time_s)), flush=True)
                    next_log_s = time_s + args.log_interval_s
                action = policy(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite policy action.")
                observation, reward, dones, _ = env.step(action)
                if not (torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in observation.values())):
                    raise RuntimeError("Non-finite rollout.")
                model.reset(dones)
                resets += int(dones.sum())
                completed += 1
                for number, result in enumerate(play_env.play_results[printed:], printed + 1):
                    print(f"[GOAL] t={completed * play_env.step_dt:.2f} s, robot 0: {describe_result(result, number)}",
                          flush=True)
                printed = len(play_env.play_results)
                if args.real_time:
                    time.sleep(max(0.0, play_env.step_dt - (time.perf_counter() - start)))
        print("BORINOT_GOTO_PLAY " + json.dumps({
            "task": args.task, "mode": mode, "control_mode": env_cfg.control_mode, "steps": completed, "finite": True,
            "checkpoint_name": Path(checkpoint).name, "checkpoint_path": checkpoint,
            "physics_hz": env_cfg.physics_hz, "policy_hz": env_cfg.policy_hz,
            "actuator_delay_s": env_cfg.actuator_delay, "actuator_delay_range_s": env_cfg.actuator_delay_range,
            "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
            "curriculum_level": play_env.goal_curriculum.level, "goal_difficulty": difficulty,
            "progress_reward_factor": factor, "initial_goal_m": initial_goal, "deadline_s": args.deadline_s,
            "initial_goal_input_m": goal, "initial_goal_frame": args.goal_frame,
            "final_goal_input_m": None if args.random_goals else controls.goal,
            "final_goal_frame": "relative" if controls.relative else "world",
            "final_state_robot0": flight_state(play_env, completed * play_env.step_dt),
            "totals": play_env.play_totals, "hold_worst_distance_m": play_env.play_hold_worst_distance_m,
            "robot0_results": play_env.play_results[-20:], "robot0_result_count": len(play_env.play_results),
            "resets": resets, "manual_resets": controls.manual_resets, "camera": controls.camera,
            "simulated_s": completed * play_env.step_dt,
        }), flush=True)
    finally:
        if controls is not None:
            controls.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
