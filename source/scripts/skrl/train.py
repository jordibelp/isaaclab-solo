# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import wandb
from helpers import _jsonify, _wandb_snapshot
import inspect
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--symmetry-mode",
    type=str,
    default="none",
    choices=["none", "augmentation", "loss", "both"],
    help=(
        "Enable the symmetric PPO variants from arXiv:2403.04359. "
        "'augmentation' duplicates the PPO batch with the quadruped symmetries; "
        "'loss' adds the mirror/symmetry loss; 'both' combines the two."
    ),
)
parser.add_argument(
    "--symmetry-loss-coeff",
    type=float,
    default=1.0e-3,
    help="Mirror/symmetry loss coefficient used when --symmetry-mode is 'loss' or 'both'.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner as BaseRunner

    from symmetry_ppo import SymmetryRunner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner as BaseRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import borinotIsaacLab.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def _get_curriculum_state_from_env(env) -> dict | None:
    raw_env = getattr(env, "unwrapped", None) or getattr(env, "_unwrapped", None)
    if raw_env is None:
        return None

    global_idx = None
    if hasattr(raw_env, "get_curriculum_global_idx"):
        global_idx = raw_env.get_curriculum_global_idx()
    if global_idx is None:
        return None

    velx_low, velx_high = getattr(raw_env.cfg, "command_lin_vel_x_range", (0.0, 0.0))
    force_low, force_high = getattr(raw_env.cfg, "base_push_force_xy_range", (0.0, 0.0))
    return {
        "global_idx": int(global_idx),
        "max_velx_range_idx": int(getattr(raw_env, "_max_velx_range_curriculum_idx", 0)),
        "base_push_force_idx": int(getattr(raw_env, "_base_push_force_curriculum_idx", 0)),
        "command_lin_vel_x_abs": max(abs(float(velx_low)), abs(float(velx_high))),
        "base_push_force_xy_abs": max(abs(float(force_low)), abs(float(force_high))),
    }


def _patch_skrl_agent_curriculum_best_checkpoints(agent, env) -> None:
    if _get_curriculum_state_from_env(env) is None:
        return

    original_post_interaction = agent.post_interaction
    best_reward_by_curriculum_idx: dict[int, float] = {}
    last_curriculum_idx: int | None = None

    def _post_interaction_with_curriculum_best(timestep: int, timesteps: int) -> None:
        nonlocal last_curriculum_idx

        curriculum_state = _get_curriculum_state_from_env(env)
        curriculum_idx = None if curriculum_state is None else curriculum_state["global_idx"]
        if curriculum_idx is None:
            original_post_interaction(timestep, timesteps)
            return

        if last_curriculum_idx is not None and curriculum_idx != last_curriculum_idx:
            print(
                "[INFO]: Curriculum advanced "
                f"{last_curriculum_idx} -> {curriculum_idx}; resetting skrl best-checkpoint tracking."
            )
            agent.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": True, "modules": {}}
        last_curriculum_idx = curriculum_idx

        previous_best = best_reward_by_curriculum_idx.get(curriculum_idx, float("-inf"))
        original_post_interaction(timestep, timesteps)

        best_reward = float(agent.checkpoint_best_modules.get("reward", float("-inf")))
        best_modules = agent.checkpoint_best_modules.get("modules", {})
        if best_reward <= previous_best or not best_modules:
            return

        best_reward_by_curriculum_idx[curriculum_idx] = best_reward
        checkpoint_dir = os.path.join(agent.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        modules = {name: best_modules[name] for name in agent.checkpoint_modules}
        torch.save(modules, os.path.join(checkpoint_dir, "best_model.pt"))
        torch.save(modules, os.path.join(checkpoint_dir, f"best_model_curriculum_idx_{curriculum_idx}.pt"))
        print(
            "[INFO]: Saved skrl curriculum best checkpoints "
            f"(curriculum_idx={curriculum_idx}, reward={best_reward:.4f})."
        )

    agent.post_interaction = _post_interaction_with_curriculum_best


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    symmetry_enabled = args_cli.symmetry_mode != "none"
    if symmetry_enabled:
        if algorithm != "ppo":
            raise ValueError("Symmetry PPO is currently only implemented for PPO on torch.")
        if not args_cli.ml_framework.startswith("torch"):
            raise ValueError("Symmetry PPO is currently only implemented for the torch backend.")
        if args_cli.task != "solo12-v0":
            raise ValueError("The current symmetry transforms are implemented for the Solo12 direct environment only (task='solo12-v0').")

        agent_cfg["agent"]["class"] = "SymmetryPPO"
        agent_cfg["agent"]["symmetry"] = {
            "use_data_augmentation": args_cli.symmetry_mode in ("augmentation", "both"),
            "use_mirror_loss": args_cli.symmetry_mode in ("loss", "both"),
            "mirror_loss_coeff": args_cli.symmetry_loss_coeff,
            "obs_type": "policy",
            "data_augmentation_func": "solo12_symmetry.compute_symmetric_observations_actions",
        }

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    exp_cfg = agent_cfg.get("agent", {}).get("experiment", {})
    base_run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"

    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {base_run_name}")

    configured_experiment_name = agent_cfg["agent"]["experiment"].get("experiment_name", "")
    full_run_name = base_run_name
    if configured_experiment_name:
        full_run_name += f"_{configured_experiment_name}"
    if symmetry_enabled:
        symmetry_tag = f"sym-{args_cli.symmetry_mode}"
        if args_cli.symmetry_mode in ("loss", "both"):
            symmetry_tag += f"-{args_cli.symmetry_loss_coeff:g}"
        full_run_name += f"_{symmetry_tag}"

    log_dir_name = full_run_name
    if exp_cfg.get("wandb", False):
        import wandb

        wandb_kwargs = dict(exp_cfg.get("wandb_kwargs", {}))
        wandb_project = wandb_kwargs.pop("project", None)
        if not wandb_project:
            raise ValueError("W&B is enabled but agent.experiment.wandb_kwargs.project is not set.")

        wandb_name = wandb_kwargs.pop("name", full_run_name)
        wandb_sync_tensorboard = wandb_kwargs.pop("sync_tensorboard", True)
        wandb_config = dict(wandb_kwargs.pop("config", {}) or {})
        wandb_config.update(
            {
                "task": args_cli.task,
                "algorithm": algorithm,
                "ml_framework": args_cli.ml_framework,
                "seed": agent_cfg["seed"],
                "num_envs": env_cfg.scene.num_envs,
                "configured_experiment_name": configured_experiment_name,
                "full_run_name": full_run_name,
                "symmetry_mode": args_cli.symmetry_mode,
                "symmetry_loss_coeff": args_cli.symmetry_loss_coeff,
            }
        )

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=wandb_config,
            sync_tensorboard=wandb_sync_tensorboard,
            **wandb_kwargs,
        )
        log_dir_name = f"{full_run_name}_{wandb_run.id}"

    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir_name)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for  all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    env_cfg_py = inspect.getsourcefile(type(env_cfg))  # Finds config and uploads it
    env_py = inspect.getsourcefile(env.unwrapped.__class__)  # Finds env and upload it (useful mainly for direct)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner_cls = SymmetryRunner if args_cli.ml_framework.startswith("torch") else BaseRunner
    runner = runner_cls(env, agent_cfg)
    if args_cli.ml_framework.startswith("torch"):
        _patch_skrl_agent_curriculum_best_checkpoints(runner.agent, env)
    exp_cfg = agent_cfg.get("agent", {}).get("experiment", {})
    if exp_cfg.get("wandb", False):
        _wandb_snapshot(log_dir, env_cfg, agent_cfg, args_cli, env_cfg_py, env_py)


    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
