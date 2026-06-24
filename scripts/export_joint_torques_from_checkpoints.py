# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Run one rollout for each skrl checkpoint and save the applied joint torques as .npy.

For each checkpoint, this script saves:
    [T, num_joints]   where T = round(episode_seconds / step_dt)

The saved tensor is taken from:
    raw_env.scene["robot"].data.applied_torque[env_index]

This corresponds to the torques actually applied by the actuator model.

Example
-------
./isaaclab.sh -p scripts/export_joint_torques_from_checkpoints.py \
    --task="Velocity_Flat_Solo12_Play_v0" \
    --num_envs 1 \
    --headless \
    --episode_seconds 5.0 \
    --cmd_init 1.0 0.0 0.0 \
    --checkpoints \
      /home/jordibelp/IsaacLab/logs/skrl/checkpoints/0323_oneqb2d_best_agent.pt \
      /home/jordibelp/IsaacLab/logs/skrl/checkpoints/0323_mi55fsfd_best_agent.pt \
      /home/jordibelp/IsaacLab/logs/skrl/checkpoints/0323_xr0sxbab_best_agent.pt

Notes
-----
- By default, each .npy is saved next to its checkpoint with the same basename.
- The ordering of columns is the articulation joint ordering printed by the script.
- If the environment terminates before the requested duration, the script stops that rollout
  early by default and saves the shorter array.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Export applied joint torques from one rollout per checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. "
        "If omitted, --algorithm is used to determine the default entry point."
    ),
)
parser.add_argument(
    "--checkpoints",
    type=str,
    nargs="+",
    required=True,
    help="List of checkpoint paths to evaluate.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework used by the trained skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="RL algorithm used for training.",
)
parser.add_argument(
    "--episode_seconds",
    type=float,
    default=5.0,
    help="Requested rollout duration for each checkpoint in seconds.",
)
parser.add_argument(
    "--env_index",
    type=int,
    default=0,
    help="Which environment index to record from.",
)
parser.add_argument(
    "--no_stop_on_done",
    dest="stop_on_done",
    action="store_false",
    default=True,
    help="Do not stop early if the rollout terminates or truncates before episode_seconds.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional output directory. If omitted, each .npy is saved next to its checkpoint.",
)
parser.add_argument(
    "--cmd_init",
    type=float,
    nargs=3,
    default=(1.0, 0.0, 0.0),
    metavar=("VX", "VY", "WZ"),
    help="Fixed command used during evaluation: (vx, vy, wz).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports that need the app already launched
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import borinotIsaacLab.tasks  # noqa: F401


# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install with 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _pin_fixed_velocity_command(env_cfg, vx: float, vy: float, wz: float):
    """Force the manager-based velocity command sampler to always output one fixed command."""
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        return

    base_velocity = env_cfg.commands.base_velocity

    if hasattr(base_velocity, "resampling_time_range"):
        base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(base_velocity, "rel_standing_envs"):
        base_velocity.rel_standing_envs = 0.0
    if hasattr(base_velocity, "rel_heading_envs"):
        base_velocity.rel_heading_envs = 0.0
    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = False

    if hasattr(base_velocity, "ranges"):
        if hasattr(base_velocity.ranges, "lin_vel_x"):
            base_velocity.ranges.lin_vel_x = (vx, vx)
        if hasattr(base_velocity.ranges, "lin_vel_y"):
            base_velocity.ranges.lin_vel_y = (vy, vy)
        if hasattr(base_velocity.ranges, "ang_vel_z"):
            base_velocity.ranges.ang_vel_z = (wz, wz)


def _get_actions_from_outputs(vec_env, outputs):
    """Extract deterministic/mean actions from skrl Runner.act outputs."""
    if hasattr(vec_env, "possible_agents"):
        return {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in vec_env.possible_agents}
    return outputs[-1].get("mean_actions", outputs[0])


def _signal_is_true_for_env(signal, env_index: int) -> bool:
    """Robustly read termination / truncation signals."""
    if signal is None:
        return False
    if isinstance(signal, torch.Tensor):
        if signal.numel() == 1:
            return bool(signal.item())
        return bool(signal[env_index].item())
    if isinstance(signal, np.ndarray):
        if signal.size == 1:
            return bool(signal.item())
        return bool(signal[env_index].item())
    if isinstance(signal, (list, tuple)):
        if len(signal) == 1:
            return bool(signal[0])
        return bool(signal[env_index])
    return bool(signal)


def _resolve_joint_names(robot) -> list[str]:
    if hasattr(robot, "joint_names") and robot.joint_names is not None:
        return list(robot.joint_names)
    if hasattr(robot, "data") and hasattr(robot.data, "joint_names") and robot.data.joint_names is not None:
        return list(robot.data.joint_names)
    return [f"joint_{i}" for i in range(robot.num_joints)]


def _save_joint_names_sidecar(npy_path: str, joint_names: list[str]):
    txt_path = os.path.splitext(npy_path)[0] + "_joint_names.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(joint_names):
            f.write(f"{i}\t{name}\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    # basic setup
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # fixed command for all rollouts
    fixed_vx, fixed_vy, fixed_wz = map(float, args_cli.cmd_init)
    _pin_fixed_velocity_command(env_cfg, fixed_vx, fixed_vy, fixed_wz)

    # make env once and reuse it for all checkpoints
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = float(env.step_dt)
    except AttributeError:
        dt = float(env.unwrapped.step_dt)

    num_steps = int(round(float(args_cli.episode_seconds) / dt))
    if num_steps <= 0:
        raise ValueError(f"episode_seconds={args_cli.episode_seconds} is too small for step_dt={dt}")

    vec_env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    raw_env = env.unwrapped

    # disable logging/checkpoint side effects during evaluation
    if "trainer" in experiment_cfg:
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
    if "agent" in experiment_cfg and "experiment" in experiment_cfg["agent"]:
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(vec_env, experiment_cfg)

    robot = raw_env.scene["robot"]
    joint_names = _resolve_joint_names(robot)

    print("=" * 80)
    print("Joint order used in saved .npy columns:")
    for i, name in enumerate(joint_names):
        print(f"  [{i:02d}] {name}")
    print("=" * 80)
    print(f"Requested rollout duration: {args_cli.episode_seconds:.4f} s")
    print(f"Environment step_dt      : {dt:.6f} s")
    print(f"Recorded steps per policy: {num_steps}")
    print(f"Recorded env index       : {args_cli.env_index}")
    print(f"Fixed command            : (vx, vy, wz) = ({fixed_vx}, {fixed_vy}, {fixed_wz})")

    for checkpoint in args_cli.checkpoints:
        checkpoint_path = os.path.abspath(checkpoint)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print("\n" + "-" * 80)
        print(f"Loading checkpoint: {checkpoint_path}")
        runner.agent.load(checkpoint_path)
        runner.agent.set_running_mode("eval")

        obs, _ = vec_env.reset()
        rollout_torques = []
        done_early = False

        for step_idx in range(num_steps):
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                actions = _get_actions_from_outputs(vec_env, outputs)
                obs, _, terminated, truncated, _ = vec_env.step(actions)

            # Record the torques that were applied during this most recent env step.
            torque_t = robot.data.applied_torque[args_cli.env_index].detach().cpu().numpy().astype(np.float32).copy()
            rollout_torques.append(torque_t)

            term = _signal_is_true_for_env(terminated, args_cli.env_index)
            trunc = _signal_is_true_for_env(truncated, args_cli.env_index)
            if term or trunc:
                done_early = True
                print(
                    f"[WARN] Rollout ended early at step {step_idx + 1}/{num_steps} "
                    f"(terminated={term}, truncated={trunc})."
                )
                if args_cli.stop_on_done:
                    break

        if len(rollout_torques) == 0:
            torque_array = np.zeros((0, len(joint_names)), dtype=np.float32)
        else:
            torque_array = np.stack(rollout_torques, axis=0)

        output_dir = args_cli.output_dir if args_cli.output_dir is not None else os.path.dirname(checkpoint_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, Path(checkpoint_path).stem + ".npy")
        np.save(output_path, torque_array)
        _save_joint_names_sidecar(output_path, joint_names)

        print(f"Saved: {output_path}")
        print(f"Shape: {torque_array.shape}")
        print(f"Early termination: {done_early}")

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
