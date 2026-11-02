# SPDX-License-Identifier: BSD-3-Clause
"""Play a Borinot go-to PPO/SAC policy with sampled or fixed goal/time commands."""
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
parser.add_argument("--duration_s", type=float, default=20.0)
parser.add_argument("--goal-offset", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"))
parser.add_argument("--deadline_s", type=float, default=None)
parser.add_argument("--curriculum-level", type=int, default=None)
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--camera", choices=("follow", "chase", "side", "top", "free"), default="follow")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or not math.isfinite(args.duration_s) or args.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive and finite.")
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
    env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "env", 0
    pose = camera_pose(args.camera, env_cfg.robot_home)
    if pose is not None:
        env_cfg.viewer.eye, env_cfg.viewer.lookat = pose
    env = RslRlVecEnvWrapper(BorinotGoToPlayEnv(env_cfg, args.goal_offset, args.deadline_s),
                            clip_actions=agent_cfg.clip_actions)
    completed = episodes = successes = 0
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
        observation = env.get_observations()
        total_steps = math.ceil(args.duration_s / env.unwrapped.step_dt)
        with torch.inference_mode():
            while completed < total_steps and simulation_app.is_running():
                start = time.perf_counter()
                action = policy(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite policy action.")
                observation, reward, dones, info = env.step(action)
                if not (torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in observation.values())):
                    raise RuntimeError("Non-finite rollout.")
                model.reset(dones)
                episodes += int(dones.sum())
                successes += int(info["goal"]["success"].sum())
                completed += 1
                if pose is not None and args.camera != "free":
                    root = env.unwrapped.robot.data.root_pos_w[0].cpu().tolist()
                    w, x, y, z = env.unwrapped.robot.data.root_quat_w[0].cpu().tolist()
                    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
                    eye, lookat = camera_pose(args.camera, root, yaw)
                    env.unwrapped.sim.set_camera_view(eye, lookat)
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        print("BORINOT_GOTO_PLAY " + json.dumps({
            "task": args.task, "control_mode": env_cfg.control_mode, "steps": completed, "finite": True,
            "checkpoint_name": Path(checkpoint).name, "checkpoint_path": checkpoint,
            "observation_dimension": env_cfg.observation_space, "action_dimension": env_cfg.action_space,
            "curriculum_level": env.unwrapped.get_curriculum_global_idx(),
            "completed_episodes": episodes, "successes": successes,
            "goal_offset_m": args.goal_offset, "deadline_s": args.deadline_s,
            "simulated_s": completed * env.unwrapped.step_dt,
        }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
