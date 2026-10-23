# SPDX-License-Identifier: BSD-3-Clause
"""Play Borinot-cmd-vel PPO, SAC-CE or SAC-MSE weights without optimizer/replay restoration."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-cmd-vel", choices=("Borinot-cmd-vel", "Borinot-Cmd-Vel-Direct-v0"))
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration_s", type=float, default=30.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or not math.isfinite(args.duration_s) or args.duration_s <= 0:
    parser.error("num_envs and duration_s must be positive and finite.")
defaults = {"env.curriculum": "False", "agent.algorithm.torch_compile": "False",
            "agent.local_redundancy.enabled": "False"}
override_keys = {arg.split("=", 1)[0].lstrip("+") for arg in hydra_args}
sys.argv = [sys.argv[0]] + [f"{key}={value}" for key, value in defaults.items() if key not in override_keys] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl_sac_vendor"))
from rsl_rl_sac.runners import OffPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    agent_cfg.device = env_cfg.sim.device
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(env_cfg)
    env_cfg.configure_control()
    env_cfg.viewer.origin_type, env_cfg.viewer.env_index = "asset_root", 0
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye, env_cfg.viewer.lookat = (3.0, 3.0, 1.5), (0.0, 0.0, 0.0)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        checkpoint = retrieve_file_path(args.checkpoint)
        if agent_cfg.class_name == "OffPolicyRunner":
            agent_cfg.algorithm.replay_buffer_size = env.num_envs
            runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False})
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        observation = env.get_observations()
        total_steps = math.ceil(args.duration_s / env.unwrapped.step_dt)
        completed = episodes = 0
        tracking = torch.zeros(3, device=env.device)
        with torch.inference_mode():
            while completed < total_steps and simulation_app.is_running():
                start = time.perf_counter()
                action = policy(observation)
                observation, reward, dones, info = env.step(action)
                if not (torch.isfinite(action).all() and torch.isfinite(reward).all()
                        and all(torch.isfinite(value).all() for value in observation.values())):
                    raise RuntimeError("Non-finite policy rollout.")
                tracking += torch.stack([info["tracking"][key].mean() for key in ("xyz", "yaw", "xy")])
                episodes += int(dones.sum())
                completed += 1
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - start)))
        print("BORINOT_CMD_VEL_PLAY " + json.dumps({
            "task": args.task, "control_mode": env_cfg.control_mode, "steps": completed,
            "simulated_s": completed * env.unwrapped.step_dt, "completed_episodes": episodes,
            "force_level": env.unwrapped.force_level,
            "mean_tracking_xyz_yaw_xy": (tracking / max(1, completed)).cpu().tolist(),
            "finite": True,
        }), flush=True)
        if completed != total_steps:
            raise RuntimeError(f"Playback stopped after {completed}/{total_steps} requested steps.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
