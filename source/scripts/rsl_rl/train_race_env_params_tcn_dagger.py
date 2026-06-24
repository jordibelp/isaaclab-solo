# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Solo12 race RMA/DAgger adaptation module for the env-params latent policy.

Phase 1 is the already-trained ``Solo12-Race-ParamsConditionedEnc-Direct-v0`` policy:
    raw obs [current race obs, privileged GT env params]
    -> frozen actor env-param encoder -> 8D latent z
    -> frozen actor MLP -> action

This script implements phase 2:
    history [joint errors, optionally foot IMUs]
    -> randomly initialized TCN adapter -> 8D z_hat
    -> frozen phase-1 actor MLP -> action

Data is aggregated on-policy with the current adapter (DAgger-style), while the
supervision target is the frozen phase-1 teacher latent z = mu(e_gt) on each
visited state. No RL update is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"
for _path in (str(_UPSTREAM_RSL_SCRIPT_DIR),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_WANDB_SOURCE_FILE_REL_PATHS = (
    "source/scripts/rsl_rl/train_race_env_params_tcn_dagger.py",
    "source/scripts/rsl_rl/race_dagger_adapter_policy.py",
    "source/scripts/rsl_rl/solo_race_eval.py",
    "source/scripts/rsl_rl/play_direct_race_0423.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/__init__.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/solo12_race_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/solo12_race_env_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/agents/rsl_rl_ppo_cfg.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/agents/env_params_conditioned_encoder_actor.py",
    "source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race/agents/imu_tcn_actor_critic.py",
)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train a Solo12 race env-params TCN adapter with supervised DAgger.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during data collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in env steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in env steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="DAgger task id with GT params + history observations.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RSL-RL config entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment and adapter init.")
parser.add_argument("--max_iterations", type=int, default=None, help="Alias for --dagger_iterations.")
parser.add_argument(
    "--teacher-checkpoint",
    "--teacher_checkpoint",
    dest="teacher_checkpoint",
    type=str,
    default=None,
    help="Path to the trained ParamsConditionedEnc checkpoint. Falls back to --checkpoint if omitted.",
)
parser.add_argument(
    "--teacher-shared-networks",
    action="store_true",
    default=False,
    help="Set if the phase-1 ParamsConditionedEnc checkpoint was trained with shared actor/critic networks.",
)
parser.add_argument(
    "--adapter-checkpoint",
    "--adapter_checkpoint",
    dest="adapter_checkpoint",
    type=str,
    default=None,
    help="Optional DAgger adapter checkpoint to initialize from before continuing/fine-tuning training.",
)
parser.add_argument(
    "--load-adapter-optimizer",
    "--load_adapter_optimizer",
    dest="load_adapter_optimizer",
    action="store_true",
    default=False,
    help="Also restore the adapter optimizer state from --adapter-checkpoint. By default, fine-tuning uses a fresh optimizer.",
)
parser.add_argument(
    "--adapter-input",
    type=str,
    default="auto",
    choices=["auto", "joint_state", "joint_state_imu"],
    help="Expected adapter input layout. 'auto' infers it from the selected DAgger env config.",
)
parser.add_argument(
    "--history-policy-steps",
    "--history_policy_steps",
    dest="history_policy_steps",
    type=int,
    default=5,
    help=(
        "History window in policy/control steps. The env records history at physics rate, so the adapter TCN "
        "length is decimation * history_policy_steps. Default is 5 policy steps -> T=20 for decimation=4."
    ),
)
parser.add_argument("--dagger_iterations", type=int, default=None, help="Number of DAgger collect/train iterations.")
parser.add_argument(
    "--num_steps_per_iter",
    type=int,
    default=None,
    help="Environment steps collected per DAgger iteration. Defaults to runner num_steps_per_env.",
)
parser.add_argument("--train_epochs", type=int, default=4, help="Supervised epochs over the replay buffer per iteration.")
parser.add_argument("--updates_per_iter", type=int, default=None, help="Override supervised updates per iteration.")
parser.add_argument("--mini_batch_size", type=int, default=8192, help="Supervised mini-batch size.")
parser.add_argument(
    "--policy_eval_every_dag_it",
    "--policy-eval-every-dag-it",
    dest="policy_eval_every_dag_it",
    type=int,
    default=0,
    help="Run closed-loop adapter policy evaluation every N DAgger iterations. Set <=0 to disable.",
)
parser.add_argument(
    "--policy_eval_dagger_iterations",
    dest="policy_eval_every_dag_it",
    type=int,
    default=argparse.SUPPRESS,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--n_env_eval",
    "--n-env-eval",
    dest="n_env_eval",
    type=int,
    default=None,
    help="Number of completed policy-evaluation episodes to collect when periodic DAgger eval is enabled. Defaults to --num_envs / env_cfg.scene.num_envs.",
)
parser.add_argument(
    "--policy-eval-seed",
    "--policy_eval_seed",
    dest="policy_eval_seed",
    type=int,
    default=None,
    help=(
        "Optional holdout seed used before periodic DAgger policy eval resets. "
        "For Solo12 race tasks, this also seeds patch-friction assignments during eval. "
        "Leave unset to evaluate on the current training RNG stream."
    ),
)
parser.add_argument(
    "--policy-eval-friction-seed",
    "--policy_eval_friction_seed",
    dest="policy_eval_friction_seed",
    type=int,
    default=None,
    help=argparse.SUPPRESS,  # Deprecated: --policy-eval-seed now controls eval friction too.
)
parser.add_argument("--learning_rate", type=float, default=3.0e-4, help="Adapter optimizer learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay for the adapter.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Adapter gradient clipping norm.")
parser.add_argument(
    "--dataset_capacity",
    type=int,
    default=262_144,
    help="Maximum aggregated samples kept in the DAgger replay buffer (ring buffer).",
)
parser.add_argument("--save_interval", type=int, default=50, help="Save latest adapter checkpoint every N iterations.")
parser.add_argument("--log_interval", type=int, default=1, help="Print/log every N iterations.")
parser.add_argument(
    "--no-history-normalization",
    action="store_true",
    default=False,
    help="Disable the adapter input EmpiricalNormalization over history vectors.",
)
parser.add_argument(
    "--stochastic-actions",
    action="store_true",
    default=False,
    help="Sample from the frozen actor distribution instead of using deterministic actor means during rollouts.",
)
parser.add_argument(
    "--dagger_experiment_name",
    type=str,
    default="solo12_race_params_dagger",
    help="Log directory experiment name under logs/rsl_rl/.",
)
parser.add_argument("--run-name", type=str, default=None, help="Optional run name suffix for logs/W&B.")
parser.add_argument("--disable-wandb", action="store_true", default=False, help="Disable custom W&B logging.")
parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/team.")
parser.add_argument("--wandb-name", type=str, default=None, help="Optional explicit W&B run name.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None, help=argparse.SUPPRESS)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401
from isaaclab_tasks.direct.solo12_race.agents.env_params_conditioned_encoder_actor import (
    EnvParamsConditionedEncoderActor,
)
from isaaclab_tasks.direct.solo12_race.agents.imu_tcn_actor_critic import FootImuTcnEncoder


class ReplayBuffer:
    """Simple CPU ring buffer for aggregated supervised DAgger samples."""

    def __init__(self, capacity: int, history_dim: int, latent_dim: int) -> None:
        self.capacity = int(capacity)
        self.history = torch.empty((self.capacity, history_dim), dtype=torch.float32, device="cpu")
        self.target_z = torch.empty((self.capacity, latent_dim), dtype=torch.float32, device="cpu")
        self.size = 0
        self.pos = 0

    def add(self, history: torch.Tensor, target_z: torch.Tensor) -> None:
        history = history.detach().to(device="cpu", dtype=torch.float32)
        target_z = target_z.detach().to(device="cpu", dtype=torch.float32)
        num = history.shape[0]
        if num >= self.capacity:
            self.history.copy_(history[-self.capacity :])
            self.target_z.copy_(target_z[-self.capacity :])
            self.size = self.capacity
            self.pos = 0
            return

        end = self.pos + num
        if end <= self.capacity:
            self.history[self.pos : end].copy_(history)
            self.target_z[self.pos : end].copy_(target_z)
        else:
            first = self.capacity - self.pos
            self.history[self.pos :].copy_(history[:first])
            self.target_z[self.pos :].copy_(target_z[:first])
            self.history[: end - self.capacity].copy_(history[first:])
            self.target_z[: end - self.capacity].copy_(target_z[first:])
        self.pos = end % self.capacity
        self.size = min(self.capacity, self.size + num)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(self.size, (batch_size,), device="cpu")
        return self.history[indices].to(device=device), self.target_z[indices].to(device=device)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _resolve_teacher_checkpoint() -> str:
    teacher_checkpoint = args_cli.teacher_checkpoint or args_cli.checkpoint
    if teacher_checkpoint is None:
        raise ValueError("A phase-1 teacher checkpoint is required: pass --teacher-checkpoint /path/to/best_model.pt")
    teacher_checkpoint = os.path.abspath(os.path.expanduser(teacher_checkpoint))
    if not os.path.isfile(teacher_checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_checkpoint}")
    return teacher_checkpoint


def _checkpoint_model_state_dict(checkpoint_path: str, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find a model state_dict in teacher checkpoint: {checkpoint_path}")


def _linear_stack_dims(state_dict: dict[str, torch.Tensor], prefix: str) -> tuple[list[int], int | None]:
    layers: list[tuple[int, int]] = []
    layer_prefix = f"{prefix}."
    for key, value in state_dict.items():
        if not key.startswith(layer_prefix) or not key.endswith(".weight") or value.ndim != 2:
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        layers.append((int(parts[1]), int(value.shape[0])))
    layers.sort(key=lambda item: item[0])
    if not layers:
        return [], None
    out_dims = [out_dim for _, out_dim in layers]
    return out_dims[:-1], out_dims[-1]


def _set_policy_cfg_if_changed(policy_cfg: Any, attr: str, value: Any) -> bool:
    if not hasattr(policy_cfg, attr) or value is None:
        return False
    old_value = getattr(policy_cfg, attr)
    comparable_old = list(old_value) if isinstance(old_value, (list, tuple)) else old_value
    comparable_new = list(value) if isinstance(value, (list, tuple)) else value
    if comparable_old == comparable_new:
        return False
    setattr(policy_cfg, attr, comparable_new)
    print(f"[INFO] Teacher checkpoint overrides policy.{attr}: {comparable_old} -> {comparable_new}", flush=True)
    return True


def _apply_teacher_checkpoint_policy_arch(policy_cfg: Any, checkpoint_path: str) -> dict[str, Any]:
    """Make the frozen teacher model architecture match the checkpoint before instantiation.

    DAgger only trains the adapter, but it still has to reconstruct the frozen phase-1 teacher exactly. This avoids
    manually editing ``rsl_rl_ppo_cfg.py`` or passing Hydra overrides when the teacher checkpoint used a different MLP.
    """

    state_dict = _checkpoint_model_state_dict(checkpoint_path, map_location="cpu")
    actor_hidden_dims, actor_output_dim = _linear_stack_dims(state_dict, "actor")
    critic_hidden_dims, critic_output_dim = _linear_stack_dims(state_dict, "critic")
    encoder_hidden_dims, encoder_output_dim = _linear_stack_dims(state_dict, "actor_env_params_encoder")

    changed = False
    if actor_hidden_dims:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "actor_hidden_dims", actor_hidden_dims)
    if critic_hidden_dims:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "critic_hidden_dims", critic_hidden_dims)
    if encoder_hidden_dims:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "env_params_encoder_hidden_dims", encoder_hidden_dims)
    if encoder_output_dim is not None:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "env_params_latent_dim", int(encoder_output_dim))
    if "log_std" in state_dict:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "noise_std_type", "log")
    elif "std" in state_dict:
        changed |= _set_policy_cfg_if_changed(policy_cfg, "noise_std_type", "scalar")
    changed |= _set_policy_cfg_if_changed(
        policy_cfg,
        "actor_obs_normalization",
        any(key.startswith("actor_obs_normalizer.") for key in state_dict),
    )
    changed |= _set_policy_cfg_if_changed(
        policy_cfg,
        "critic_obs_normalization",
        any(key.startswith("critic_obs_normalizer.") for key in state_dict),
    )

    inferred = {
        "actor_hidden_dims": actor_hidden_dims,
        "actor_output_dim": actor_output_dim,
        "critic_hidden_dims": critic_hidden_dims,
        "critic_output_dim": critic_output_dim,
        "env_params_encoder_hidden_dims": encoder_hidden_dims,
        "env_params_latent_dim": encoder_output_dim,
        "noise_std_type": "log" if "log_std" in state_dict else "scalar" if "std" in state_dict else None,
        "actor_obs_normalization": any(key.startswith("actor_obs_normalizer.") for key in state_dict),
        "critic_obs_normalization": any(key.startswith("critic_obs_normalizer.") for key in state_dict),
        "changed_policy_cfg": changed,
    }
    print(f"[INFO] Inferred teacher checkpoint architecture: {json.dumps(inferred, sort_keys=True)}", flush=True)
    return inferred


def _load_teacher(
    *,
    checkpoint_path: str,
    policy_cfg: Any,
    num_actions: int,
    device: torch.device,
) -> EnvParamsConditionedEncoderActor:
    policy_kwargs = _cfg_to_dict(policy_cfg)
    policy_kwargs.pop("class_name", None)
    if args_cli.teacher_shared_networks:
        if "shared_networks" not in policy_kwargs:
            raise ValueError("--teacher-shared-networks was set, but this policy config has no shared_networks field.")
        policy_kwargs["shared_networks"] = True

    current_obs_dim = int(policy_kwargs.get("current_obs_dim", 57))
    env_params_dim = int(policy_kwargs.get("env_params_dim", 16))
    teacher_obs_dim = current_obs_dim + env_params_dim
    dummy_obs = TensorDict(
        {"policy": torch.zeros((1, teacher_obs_dim), device=device)},
        batch_size=[1],
    )
    teacher = EnvParamsConditionedEncoderActor(
        obs=dummy_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=num_actions,
        **policy_kwargs,
    ).to(device)

    state_dict = _checkpoint_model_state_dict(checkpoint_path, map_location=device)
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _infer_history_layout(env_cfg: Any) -> dict[str, Any]:
    has_joint = bool(getattr(env_cfg, "include_joint_state_history_obs", False))
    has_imu = bool(getattr(env_cfg, "include_foot_imu_obs", False))
    has_gt = bool(getattr(env_cfg, "include_forces_to_gt_obs", False)) or bool(
        getattr(env_cfg, "include_mu_coefs_to_gt_obs", False)
    )
    if not has_gt:
        raise ValueError("DAgger training needs privileged GT env params enabled for teacher latent labels.")
    if has_joint and has_imu:
        layout = {
            "kind": "joint_state_imu",
            "name": "joint-state + foot-IMU",
            "history_len": int(env_cfg.joint_imu_history_length),
            "history_dim": int(env_cfg.joint_imu_history_obs_dim),
            "channels": int(env_cfg.joint_imu_tcn_channels),
            "kernel_size": int(env_cfg.joint_imu_tcn_kernel_size),
            "activation": str(env_cfg.joint_imu_tcn_activation),
        }
    elif has_joint:
        layout = {
            "kind": "joint_state",
            "name": "joint-state",
            "history_len": int(env_cfg.joint_state_history_length),
            "history_dim": int(env_cfg.joint_state_history_obs_dim),
            "channels": int(env_cfg.joint_state_tcn_channels),
            "kernel_size": int(env_cfg.joint_state_tcn_kernel_size),
            "activation": str(env_cfg.joint_state_tcn_activation),
        }
    else:
        raise ValueError("DAgger training needs joint-state history. Use one of the ParamsDagger*TCN task IDs.")

    if args_cli.adapter_input != "auto" and args_cli.adapter_input != layout["kind"]:
        raise ValueError(
            f"--adapter-input={args_cli.adapter_input} does not match selected env layout {layout['kind']}."
        )
    if layout["kind"] == "joint_state_imu":
        layout["history_policy_steps"] = int(getattr(env_cfg, "joint_imu_history_policy_steps"))
    else:
        layout["history_policy_steps"] = int(getattr(env_cfg, "joint_state_history_policy_steps"))
    layout["decimation"] = int(getattr(env_cfg, "decimation", 1))
    layout["flat_dim"] = layout["history_len"] * layout["history_dim"]
    return layout


def _apply_history_policy_steps_override(env_cfg: Any) -> None:
    """Override the DAgger history window before the env and adapter are constructed."""

    if args_cli.history_policy_steps is None:
        return
    history_policy_steps = int(args_cli.history_policy_steps)
    if history_policy_steps < 1:
        raise ValueError(f"--history-policy-steps must be >= 1, got {history_policy_steps}.")

    # Keep all Solo12 race history variants aligned so joint-only and joint+IMU DAgger layouts stay compatible.
    for attr in (
        "foot_imu_history_policy_steps",
        "joint_state_history_policy_steps",
        "joint_imu_history_policy_steps",
    ):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, history_policy_steps)

    # Recompute derived history lengths and observation_space after mutating the config.
    post_init = getattr(env_cfg, "__post_init__", None)
    if callable(post_init):
        post_init()


def _actor_mean_and_std(teacher: EnvParamsConditionedEncoderActor, actor_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = teacher.actor(actor_input)
    if teacher.state_dependent_std:
        if teacher.noise_std_type == "scalar":
            mean, std = torch.unbind(output, dim=-2)
        elif teacher.noise_std_type == "log":
            mean, log_std = torch.unbind(output, dim=-2)
            std = torch.exp(log_std)
        else:
            raise ValueError(f"Unsupported teacher noise_std_type: {teacher.noise_std_type}")
        return mean, std

    if teacher.noise_std_type == "scalar":
        std = teacher.std.expand_as(output)
    elif teacher.noise_std_type == "log":
        std = torch.exp(teacher.log_std).expand_as(output)
    else:
        raise ValueError(f"Unsupported teacher noise_std_type: {teacher.noise_std_type}")
    return output, std


@torch.no_grad()
def _teacher_latent_and_student_action(
    teacher: EnvParamsConditionedEncoderActor,
    teacher_obs_raw: torch.Tensor,
    z_hat: torch.Tensor,
    *,
    stochastic_actions: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_obs = teacher.actor_obs_normalizer(teacher_obs_raw)
    current_obs = teacher_obs[:, : teacher.current_obs_dim]
    gt_env_params = teacher_obs[:, teacher.current_obs_dim : teacher.current_obs_dim + teacher.env_params_dim]
    z_teacher = teacher.actor_env_params_encoder(gt_env_params)
    actor_input = torch.cat((current_obs, z_hat), dim=-1)
    mean, std = _actor_mean_and_std(teacher, actor_input)
    if stochastic_actions:
        actions = mean + torch.randn_like(mean) * std
    else:
        actions = mean
    return z_teacher, actions, mean


def _save_checkpoint(
    *,
    path: str,
    adapter: nn.Module,
    history_normalizer: EmpiricalNormalization | None,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    samples: int,
    best_loss: float,
    teacher_checkpoint: str,
    layout: dict[str, Any],
    dims: dict[str, int],
    selection_metric_name: str | None = None,
    selection_metric_value: float | None = None,
    policy_eval: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "adapter_state_dict": adapter.state_dict(),
        "history_normalizer_state_dict": history_normalizer.state_dict() if history_normalizer is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
        "samples": int(samples),
        "best_loss": float(best_loss),
        "teacher_checkpoint": teacher_checkpoint,
        "layout": layout,
        "dims": dims,
        "args": vars(args_cli),
    }
    if selection_metric_name is not None:
        checkpoint["selection_metric_name"] = selection_metric_name
        checkpoint["selection_metric_value"] = selection_metric_value
    if policy_eval is not None:
        checkpoint["policy_eval"] = policy_eval
    torch.save(checkpoint, path)


def _load_adapter_checkpoint(
    *,
    checkpoint_path: str,
    adapter: nn.Module,
    history_normalizer: EmpiricalNormalization | None,
    optimizer: torch.optim.Optimizer,
    layout: dict[str, Any],
    dims: dict[str, int],
    load_optimizer: bool,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Adapter checkpoint must be a dict with adapter_state_dict: {checkpoint_path}")

    adapter_state = checkpoint.get("adapter_state_dict")
    if adapter_state is None:
        # Allow passing a bare adapter state_dict for quick local experiments.
        adapter_state = checkpoint
    adapter.load_state_dict(adapter_state, strict=True)

    ckpt_layout = checkpoint.get("layout")
    if isinstance(ckpt_layout, dict):
        expected = ("kind", "history_len", "history_dim", "flat_dim")
        mismatches = [
            f"{key}: checkpoint={ckpt_layout.get(key)!r} current={layout.get(key)!r}"
            for key in expected
            if ckpt_layout.get(key) != layout.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Adapter checkpoint layout does not match the current DAgger env/config: " + "; ".join(mismatches)
            )

    ckpt_dims = checkpoint.get("dims")
    if isinstance(ckpt_dims, dict):
        expected = ("teacher_obs_dim", "history_flat_dim", "latent_dim")
        mismatches = [
            f"{key}: checkpoint={ckpt_dims.get(key)!r} current={dims.get(key)!r}"
            for key in expected
            if ckpt_dims.get(key) != dims.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Adapter checkpoint dims do not match the current teacher/env/config: " + "; ".join(mismatches)
            )

    normalizer_state = checkpoint.get("history_normalizer_state_dict")
    if history_normalizer is not None and normalizer_state is not None:
        history_normalizer.load_state_dict(normalizer_state, strict=True)
    elif history_normalizer is not None:
        print("[WARN] Adapter checkpoint has no history normalizer state; starting normalizer fresh.", flush=True)
    elif normalizer_state is not None:
        print("[WARN] Adapter checkpoint contains history normalizer state but --no-history-normalization is set; ignoring it.", flush=True)

    if load_optimizer:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            print("[WARN] --load-adapter-optimizer was set, but checkpoint has no optimizer_state_dict.", flush=True)
        else:
            optimizer.load_state_dict(optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = args_cli.learning_rate
                group["weight_decay"] = args_cli.weight_decay

    print(
        f"[INFO] Initialized adapter from checkpoint: {checkpoint_path} "
        f"(checkpoint iteration={checkpoint.get('iteration', 'unknown')}, optimizer_loaded={load_optimizer})",
        flush=True,
    )
    return checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_wandb_run_files(wandb: Any, log_dir: str) -> list[str]:
    """Attach the source/config files needed to understand and replay this DAgger run."""

    logged_files: list[str] = []
    manifest: list[dict[str, str]] = []

    for rel_path in _WANDB_SOURCE_FILE_REL_PATHS:
        source_path = _REPO_ROOT / rel_path
        if not source_path.is_file():
            print(f"[WARN] W&B source snapshot skipped missing file: {source_path}", flush=True)
            continue
        try:
            wandb.save(str(source_path), base_path=str(_REPO_ROOT), policy="now")
        except Exception as exc:  # pragma: no cover - depends on W&B runtime/network
            print(f"[WARN] W&B source snapshot failed for {source_path}: {exc}", flush=True)
            continue
        logged_files.append(rel_path)
        manifest.append(
            {
                "path": rel_path,
                "sha256": _file_sha256(source_path),
            }
        )

    params_dir = Path(log_dir) / "params"
    manifest_path = params_dir / "wandb_source_files.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    for path in (params_dir / "env.yaml", params_dir / "agent.yaml", params_dir / "dagger_args.json", manifest_path):
        if path.is_file():
            try:
                wandb.save(str(path), base_path=log_dir, policy="now")
            except Exception as exc:  # pragma: no cover - depends on W&B runtime/network
                print(f"[WARN] W&B params snapshot failed for {path}: {exc}", flush=True)

    return logged_files


def _maybe_init_wandb(log_dir: str, run_name: str, config: dict[str, Any]):
    if args_cli.disable_wandb:
        return None
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on training env
        print(f"[WARN] W&B import failed; continuing without W&B: {exc}", flush=True)
        return None

    project = args_cli.log_project_name or "borinotIsaacLab"
    try:
        run = wandb.init(
            project=project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_name or run_name,
            dir=log_dir,
            config=config,
        )
        logged_source_files = _snapshot_wandb_run_files(wandb, log_dir)
        run.config.update({"wandb_source_files": logged_source_files}, allow_val_change=True)
        wandb.define_metric("dagger/iteration")
        for metric_name in (
            "dagger/*",
            "rollout/*",
            "collection/*",
            "optimization/*",
            "buffer/*",
            "policy_eval/*",
            "time/*",
        ):
            wandb.define_metric(metric_name, step_metric="dagger/iteration")
        return run
    except Exception as exc:  # pragma: no cover - depends on user auth/network
        print(f"[WARN] W&B init failed; continuing without W&B: {exc}", flush=True)
        return None


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _to_bool_tensor(value: Any, device: torch.device, num_envs: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bool).reshape(-1)[:num_envs]
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)[:num_envs]


def _safe_bool_metric(raw_env: Any, name: str) -> torch.Tensor:
    try:
        value = getattr(raw_env, name)
        if isinstance(value, torch.Tensor):
            return value.to(device=raw_env.device, dtype=torch.bool)
    except Exception:
        pass
    return torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)


def _gate_progress_ratio(raw_env: Any) -> torch.Tensor:
    try:
        target_count = max(int(raw_env._target_count), 1)
        return torch.clamp(raw_env._current_gate_idx.float() / target_count, max=1.0)
    except Exception:
        return torch.zeros(raw_env.num_envs, dtype=torch.float, device=raw_env.device)


def _log_scalar(extras: dict[str, Any], name: str) -> float | None:
    log = extras.get("log", {}) if isinstance(extras, dict) else {}
    value = log.get(name) if isinstance(log, dict) else None
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().item())
    try:
        return float(value)
    except Exception:
        return None


def _set_policy_eval_env_flags(raw_env: Any) -> list[tuple[str, Any]]:
    """Disable non-policy stochasticity during in-training policy eval and return values to restore."""

    cfg = raw_env.cfg
    saved: list[tuple[str, Any]] = []

    def set_if_present(attr: str, value: Any) -> None:
        if hasattr(cfg, attr):
            saved.append((attr, getattr(cfg, attr)))
            setattr(cfg, attr, value)

    set_if_present("events", None)
    set_if_present("enable_observation_corruption", False)
    set_if_present("enable_reset_pose_randomization", False)
    set_if_present("reset_base_lin_vel_range", (0.0, 0.0))
    set_if_present("reset_base_ang_vel_range", (0.0, 0.0))
    set_if_present("actuation_delay_range", (0, 0))
    # Keep patch friction randomization enabled, matching solo_race_eval.py's default behavior for race eval.
    return saved


def _restore_policy_eval_env_flags(raw_env: Any, saved: list[tuple[str, Any]]) -> None:
    for attr, value in saved:
        setattr(raw_env.cfg, attr, value)


@torch.no_grad()
def _adapter_policy_action(
    *,
    teacher: EnvParamsConditionedEncoderActor,
    adapter: nn.Module,
    history_normalizer: EmpiricalNormalization | None,
    policy_obs: torch.Tensor,
    dims: dict[str, int],
) -> torch.Tensor:
    teacher_obs_raw = policy_obs[:, : dims["teacher_obs_dim"]]
    history_raw = policy_obs[:, dims["history_start"] : dims["history_start"] + dims["history_flat_dim"]]
    adapter_input = history_normalizer(history_raw) if history_normalizer is not None else history_raw
    z_hat = adapter(adapter_input)
    teacher_obs = teacher.actor_obs_normalizer(teacher_obs_raw)
    current_obs = teacher_obs[:, : teacher.current_obs_dim]
    actor_input = torch.cat((current_obs, z_hat), dim=-1)
    action_mean, _ = _actor_mean_and_std(teacher, actor_input)
    return action_mean


def _evaluate_current_adapter_policy(
    *,
    vec_env: RslRlVecEnvWrapper,
    teacher: EnvParamsConditionedEncoderActor,
    adapter: nn.Module,
    history_normalizer: EmpiricalNormalization | None,
    dims: dict[str, int],
    eval_n: int,
    iteration: int,
) -> tuple[dict[str, Any], TensorDict]:
    """Run closed-loop eval with the current adapter in the existing vectorized env.

    The env is reset before eval and left at the post-eval state; the returned obs should be used by the next DAgger
    collection iteration. This avoids constructing a second Isaac env while still measuring complete episodes.
    """

    raw_env = vec_env.unwrapped
    num_envs = int(raw_env.num_envs)
    eval_n = int(eval_n)
    if eval_n <= 0:
        raise ValueError(f"--n_env_eval must be positive when policy eval is enabled, got {eval_n}.")

    adapter_was_training = adapter.training
    teacher_was_training = teacher.training
    normalizer_was_training = history_normalizer.training if history_normalizer is not None else False
    saved_flags = _set_policy_eval_env_flags(raw_env)
    old_friction_seed = getattr(raw_env.cfg, "friction_seed", None)
    old_friction_generator_state = None
    if getattr(raw_env, "_friction_generator", None) is not None:
        old_friction_generator_state = raw_env._friction_generator.get_state()
    eval_friction_seed = args_cli.policy_eval_friction_seed
    if eval_friction_seed is None:
        eval_friction_seed = args_cli.policy_eval_seed
    using_eval_friction_seed = eval_friction_seed is not None and hasattr(raw_env.cfg, "friction_seed")
    using_eval_seed = args_cli.policy_eval_seed is not None
    saved_rng_state = None
    if using_eval_seed:
        saved_rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def restore_eval_randomization_state() -> None:
        nonlocal using_eval_friction_seed, using_eval_seed
        if using_eval_seed and saved_rng_state is not None:
            random.setstate(saved_rng_state["python"])
            np.random.set_state(saved_rng_state["numpy"])
            torch.set_rng_state(saved_rng_state["torch"])
            if saved_rng_state["cuda"] is not None:
                torch.cuda.set_rng_state_all(saved_rng_state["cuda"])
            using_eval_seed = False
        if using_eval_friction_seed:
            raw_env.cfg.friction_seed = old_friction_seed
            raw_env._friction_generator = None
            if old_friction_seed is not None:
                raw_env._friction_generator = torch.Generator(device=raw_env.device)
                if old_friction_generator_state is not None:
                    raw_env._friction_generator.set_state(old_friction_generator_state)
                else:
                    raw_env._friction_generator.manual_seed(int(old_friction_seed))
            using_eval_friction_seed = False

    adapter.eval()
    teacher.eval()
    if history_normalizer is not None:
        history_normalizer.eval()

    try:
        if using_eval_friction_seed:
            raw_env.cfg.friction_seed = int(eval_friction_seed)
            raw_env._friction_generator = None
        if args_cli.policy_eval_seed is not None:
            vec_env.seed(int(args_cli.policy_eval_seed))
        obs, _ = vec_env.reset()
        device = raw_env.device
        episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        episodes: list[dict[str, Any]] = []
        gate_progress_reward_sum = 0.0
        gate_progress_reward_count = 0

        # When eval_n < num_envs, do not stop after the first eval_n resets: that measures the fastest
        # envs, not an unbiased eval set. Instead, record the first episode from a fixed subset of env ids.
        if eval_n <= num_envs:
            eval_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            eval_env_mask[:eval_n] = True
            recorded_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            eval_env_mask = None
            recorded_env_mask = None

        max_steps = int(raw_env.max_episode_length)
        max_total_steps = max_steps * (math.ceil(eval_n / max(num_envs, 1)) + 1) + 5
        eval_t0 = time.time()
        last_step = 0

        for step_idx in range(1, max_total_steps + 1):
            if len(episodes) >= eval_n:
                break

            progress_before_step = _gate_progress_ratio(raw_env)[:num_envs].clone()

            actions = _adapter_policy_action(
                teacher=teacher,
                adapter=adapter,
                history_normalizer=history_normalizer,
                policy_obs=obs["policy"],
                dims=dims,
            )
            obs, _, dones, extras = vec_env.step(actions)
            episode_steps += 1
            last_step = step_idx

            gate_progress_reward = _log_scalar(extras, "RewardsPerStep/gate_progress")
            if gate_progress_reward is not None:
                gate_progress_reward_sum += gate_progress_reward
                gate_progress_reward_count += 1

            done = _to_bool_tensor(dones, device, num_envs)
            if not bool(torch.any(done)):
                continue

            terminated = _safe_bool_metric(raw_env, "reset_terminated")[:num_envs]
            timed_out = _safe_bool_metric(raw_env, "reset_time_outs")[:num_envs]
            finished = done & (progress_before_step >= 1.0)
            floor_terminal = done & terminated & ~finished & ~timed_out

            record_done = done
            if eval_env_mask is not None and recorded_env_mask is not None:
                record_done = done & eval_env_mask & ~recorded_env_mask

            for env_id in record_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if len(episodes) >= eval_n:
                    break
                steps = int(episode_steps[env_id].item())
                did_finish = bool(finished[env_id].item())
                episodes.append(
                    {
                        "finished": did_finish,
                        "timed_out": bool(timed_out[env_id].item()) and not did_finish,
                        "floor_collision_terminal": bool(floor_terminal[env_id].item()) and not did_finish,
                        "steps": steps,
                        "seconds": steps * float(raw_env.step_dt),
                        "finish_time_steps": steps if did_finish else None,
                        "finish_time_seconds": steps * float(raw_env.step_dt) if did_finish else None,
                        "gate_progress_ratio": float(progress_before_step[env_id].item()),
                    }
                )
                if recorded_env_mask is not None:
                    recorded_env_mask[env_id] = True

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            episode_steps[done_ids] = 0

        if len(episodes) < eval_n:
            print(
                f"[WARN] Policy eval at DAgger iter {iteration} collected {len(episodes)}/{eval_n} episodes "
                f"before safety limit {max_total_steps} steps.",
                flush=True,
            )

        finished_episodes = [ep for ep in episodes if ep["finished"]]
        penalized_steps = [float(ep["steps"] if ep["finished"] else max_steps) for ep in episodes]
        success_steps = [float(ep["finish_time_steps"]) for ep in finished_episodes if ep["finish_time_steps"] is not None]
        penalized_seconds = [steps * float(raw_env.step_dt) for steps in penalized_steps]

        finish_time_steps_mean = _mean_or_none(penalized_steps)
        finish_time_seconds_mean = _mean_or_none(penalized_seconds)
        gate_progress_ratio_mean = _mean_or_none([float(ep["gate_progress_ratio"]) for ep in episodes])
        reward_gate_progress = (
            gate_progress_reward_sum / gate_progress_reward_count if gate_progress_reward_count > 0 else None
        )
        elapsed = time.time() - eval_t0
        metrics = {
            "policy_eval/iteration": iteration,
            "policy_eval/n_finished": len(finished_episodes),
            "policy_eval/eval_seed": int(args_cli.policy_eval_seed) if args_cli.policy_eval_seed is not None else -1,
            "policy_eval/friction_seed": int(eval_friction_seed) if eval_friction_seed is not None else -1,
            "policy_eval/Episode/successRate": (len(finished_episodes) / len(episodes)) if episodes else 0.0,
            # Decisive metric: failures are assigned max_episode_length so low means fast and reliable.
            "policy_eval/Episode/finishTimeSteps": finish_time_steps_mean,
            "policy_eval/Episode/finishTimeSeconds": finish_time_seconds_mean,
            "policy_eval/Episode/finishTimeSteps_successOnly": _mean_or_none(success_steps),
            "policy_eval/Episode/gateProgressRatio": gate_progress_ratio_mean,
            "policy_eval/RewardsPerStep/gate_progress": reward_gate_progress,
            "policy_eval/max_episode_length": max_steps,
            "policy_eval/elapsed_s": elapsed,
        }
        printable = {
            k: v
            for k, v in metrics.items()
            if k.startswith("policy_eval/Episode") or k in ("policy_eval/eval_seed", "policy_eval/friction_seed")
        }
        print(f"[POLICY_EVAL] iter={iteration:05d} " + json.dumps(printable, sort_keys=True), flush=True)
        if using_eval_seed or using_eval_friction_seed:
            restore_eval_randomization_state()
            obs, _ = vec_env.reset()
        return metrics, obs
    finally:
        restore_eval_randomization_state()
        _restore_policy_eval_env_flags(raw_env, saved_flags)
        adapter.train(adapter_was_training)
        teacher.train(teacher_was_training)
        if history_normalizer is not None:
            history_normalizer.train(normalizer_was_training)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.teacher_shared_networks and hasattr(agent_cfg.policy, "shared_networks"):
        agent_cfg.policy.shared_networks = True

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.n_env_eval is None:
        args_cli.n_env_eval = int(env_cfg.scene.num_envs)
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    teacher_checkpoint = _resolve_teacher_checkpoint()
    teacher_checkpoint_arch = _apply_teacher_checkpoint_policy_arch(agent_cfg.policy, teacher_checkpoint)
    _apply_history_policy_steps_override(env_cfg)
    layout = _infer_history_layout(env_cfg)
    # Keep CLI/config/checkpoint/W&B metadata aligned with the actual env config used to build observations.
    args_cli.history_policy_steps = int(layout["history_policy_steps"])

    num_iterations = args_cli.dagger_iterations or args_cli.max_iterations or 1000
    num_steps_per_iter = args_cli.num_steps_per_iter or int(getattr(agent_cfg, "num_steps_per_env", 32))
    samples_per_iteration = int(env_cfg.scene.num_envs) * int(num_steps_per_iter)
    if samples_per_iteration > int(args_cli.dataset_capacity):
        print(
            "[WARN] samples collected per DAgger iteration "
            f"({samples_per_iteration} = num_envs {env_cfg.scene.num_envs} * num_steps_per_iter {num_steps_per_iter}) "
            f"exceeds dataset_capacity ({args_cli.dataset_capacity}). The ring buffer will keep only the newest "
            "samples from the current collection, so no previous-iteration data survives. Consider increasing "
            "--dataset_capacity or reducing --num_steps_per_iter.",
            flush=True,
        )
    latent_dim = int(getattr(agent_cfg.policy, "env_params_latent_dim", 8))
    if latent_dim != 8:
        print(f"[WARN] Expected 8D env-param latent, got {latent_dim}D from agent cfg.", flush=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    teacher_tag = Path(teacher_checkpoint).stem
    run_suffix = args_cli.run_name or f"{layout['kind']}_teacher-{teacher_tag}"
    run_name = f"{timestamp}_SL-DAGGER_{run_suffix}"
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.dagger_experiment_name))
    log_dir_name = run_name
    wandb_run_id: str | None = None
    if not args_cli.disable_wandb:
        # Match train.py: pre-generate the W&B run id so the local folder name matches the W&B run id.
        try:
            import wandb

            wandb_run_id = wandb.util.generate_id()
            os.environ["WANDB_RUN_ID"] = wandb_run_id
            os.environ["WANDB_RESUME"] = "allow"
            log_dir_name = f"{run_name}_{wandb_run_id}"
        except Exception as exc:  # pragma: no cover - depends on training env
            print(f"[WARN] W&B run id generation failed; using log dir without run id: {exc}", flush=True)
    log_dir = os.path.join(log_root_path, log_dir_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Logging DAgger experiment in directory: {log_dir}", flush=True)
    print(f"[INFO] Teacher checkpoint: {teacher_checkpoint}", flush=True)
    print(
        f"[INFO] Adapter input: {layout['name']} history "
        f"T={layout['history_len']}, D={layout['history_dim']}, flat={layout['flat_dim']} -> z_hat[{latent_dim}]",
        flush=True,
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "dagger_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args_cli), f, indent=2, sort_keys=True, default=str)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "dagger"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during DAgger collection.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = torch.device(vec_env.unwrapped.device)
    teacher = _load_teacher(
        checkpoint_path=teacher_checkpoint,
        policy_cfg=agent_cfg.policy,
        num_actions=vec_env.num_actions,
        device=device,
    )

    dims = {
        "current_obs_dim": int(teacher.current_obs_dim),
        "env_params_dim": int(teacher.env_params_dim),
        "teacher_obs_dim": int(teacher.current_obs_dim + teacher.env_params_dim),
        "history_start": int(teacher.current_obs_dim + teacher.env_params_dim),
        "history_flat_dim": int(layout["flat_dim"]),
        "latent_dim": int(latent_dim),
    }
    expected_obs_dim = dims["history_start"] + dims["history_flat_dim"]
    first_obs = vec_env.get_observations()
    actual_obs_dim = int(first_obs["policy"].shape[-1])
    if actual_obs_dim != expected_obs_dim:
        raise ValueError(
            f"DAgger env observation dim {actual_obs_dim} != expected {expected_obs_dim} "
            f"([current {dims['current_obs_dim']}, gt {dims['env_params_dim']}, history {dims['history_flat_dim']}])."
        )

    if args_cli.seed is not None:
        torch.manual_seed(int(args_cli.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args_cli.seed))

    adapter = FootImuTcnEncoder(
        history_len=int(layout["history_len"]),
        imu_dim=int(layout["history_dim"]),
        channels=int(layout["channels"]),
        latent_dim=latent_dim,
        kernel_size=int(layout["kernel_size"]),
        activation=str(layout["activation"]),
    ).to(device)
    history_normalizer = None
    if not args_cli.no_history_normalization:
        history_normalizer = EmpiricalNormalization(dims["history_flat_dim"]).to(device)
        history_normalizer.train()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args_cli.learning_rate, weight_decay=args_cli.weight_decay)
    initial_adapter_checkpoint: dict[str, Any] | None = None
    if args_cli.adapter_checkpoint is not None:
        initial_adapter_checkpoint = _load_adapter_checkpoint(
            checkpoint_path=args_cli.adapter_checkpoint,
            adapter=adapter,
            history_normalizer=history_normalizer,
            optimizer=optimizer,
            layout=layout,
            dims=dims,
            load_optimizer=bool(args_cli.load_adapter_optimizer),
            device=device,
        )
    buffer = ReplayBuffer(args_cli.dataset_capacity, dims["history_flat_dim"], latent_dim)
    policy_eval_enabled = int(args_cli.policy_eval_every_dag_it) > 0

    wandb_run = _maybe_init_wandb(
        log_dir,
        run_name,
        {
            "task": args_cli.task,
            "agent": args_cli.agent,
            "policy_model": str(getattr(env_cfg, "policy_model", "")),
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_checkpoint_arch": teacher_checkpoint_arch,
            "adapter_checkpoint": os.path.abspath(os.path.expanduser(args_cli.adapter_checkpoint))
            if args_cli.adapter_checkpoint is not None
            else None,
            "adapter_checkpoint_iteration": initial_adapter_checkpoint.get("iteration")
            if initial_adapter_checkpoint is not None
            else None,
            "load_adapter_optimizer": bool(args_cli.load_adapter_optimizer),
            "layout": layout,
            "dims": dims,
            "num_envs": int(env_cfg.scene.num_envs),
            "episode_length_s": float(getattr(env_cfg, "episode_length_s", 0.0)),
            "decimation": int(getattr(env_cfg, "decimation", 1)),
            "dagger_iterations": num_iterations,
            "num_steps_per_iter": num_steps_per_iter,
            "samples_per_iteration": samples_per_iteration,
            "adapter_input_kind": layout["kind"],
            "history_policy_steps": args_cli.history_policy_steps,
            "history_len": int(layout["history_len"]),
            "history_dim": int(layout["history_dim"]),
            "history_flat_dim": int(layout["flat_dim"]),
            "adapter_tcn_channels": int(layout["channels"]),
            "adapter_tcn_kernel_size": int(layout["kernel_size"]),
            "adapter_tcn_activation": str(layout["activation"]),
            "history_normalization": not bool(args_cli.no_history_normalization),
            "stochastic_actions": bool(args_cli.stochastic_actions),
            "policy_eval_every_dag_it": args_cli.policy_eval_every_dag_it,
            "policy_eval_enabled": policy_eval_enabled,
            "n_env_eval": args_cli.n_env_eval,
            "wandb_run_id": wandb_run_id,
            "train_epochs": args_cli.train_epochs,
            "updates_per_iter": args_cli.updates_per_iter,
            "mini_batch_size": args_cli.mini_batch_size,
            "learning_rate": args_cli.learning_rate,
            "weight_decay": args_cli.weight_decay,
            "max_grad_norm": args_cli.max_grad_norm,
            "dataset_capacity": args_cli.dataset_capacity,
            "buffer_capacity": args_cli.dataset_capacity,
            "actual_obs_dim": actual_obs_dim,
            "expected_obs_dim": expected_obs_dim,
            "include_joint_state_history_obs": bool(getattr(env_cfg, "include_joint_state_history_obs", False)),
            "include_foot_imu_obs": bool(getattr(env_cfg, "include_foot_imu_obs", False)),
            "include_forces_to_gt_obs": bool(getattr(env_cfg, "include_forces_to_gt_obs", False)),
            "include_mu_coefs_to_gt_obs": bool(getattr(env_cfg, "include_mu_coefs_to_gt_obs", False)),
            "wandb_source_files_expected": list(_WANDB_SOURCE_FILE_REL_PATHS),
        },
    )

    obs = first_obs
    total_steps = 0
    best_loss = float("inf")
    best_eval_finish_time_steps = float("inf")
    best_latent_path = os.path.join(
        log_dir,
        "checkpoints",
        "adapter_best_latent_mse.pt" if policy_eval_enabled else "adapter_best.pt",
    )
    start_time = time.time()

    for iteration in range(1, num_iterations + 1):
        adapter.eval()
        reward_sum = 0.0
        done_sum = 0.0
        action_mean_abs_sum = 0.0
        z_hat_abs_sum = 0.0
        z_teacher_abs_sum = 0.0
        new_data_loss_sum = 0.0
        collected = 0

        for _ in range(num_steps_per_iter):
            policy_obs = obs["policy"]
            teacher_obs_raw = policy_obs[:, : dims["teacher_obs_dim"]]
            history_raw = policy_obs[:, dims["history_start"] : dims["history_start"] + dims["history_flat_dim"]]
            if history_normalizer is not None:
                history_normalizer.update(history_raw)
                adapter_input = history_normalizer(history_raw)
            else:
                adapter_input = history_raw

            with torch.no_grad():
                z_hat = adapter(adapter_input)
                z_teacher, actions, action_mean = _teacher_latent_and_student_action(
                    teacher,
                    teacher_obs_raw,
                    z_hat,
                    stochastic_actions=args_cli.stochastic_actions,
                )

            buffer.add(history_raw, z_teacher)
            obs, rewards, dones, _ = vec_env.step(actions)

            new_data_loss_sum += float(F.mse_loss(z_hat, z_teacher).item())
            reward_sum += float(rewards.mean().item())
            done_sum += float(dones.float().mean().item())
            action_mean_abs_sum += float(action_mean.abs().mean().item())
            z_hat_abs_sum += float(z_hat.abs().mean().item())
            z_teacher_abs_sum += float(z_teacher.abs().mean().item())
            collected += policy_obs.shape[0]
            total_steps += policy_obs.shape[0]

        adapter.train()
        if buffer.size < 1:
            raise RuntimeError("No DAgger samples were collected; cannot train adapter.")
        if args_cli.updates_per_iter is None:
            updates = max(1, args_cli.train_epochs * math.ceil(buffer.size / args_cli.mini_batch_size))
        else:
            updates = int(args_cli.updates_per_iter)

        loss_sum = 0.0
        for _ in range(updates):
            history_batch, target_z_batch = buffer.sample(args_cli.mini_batch_size, device)
            if history_normalizer is not None:
                history_batch = history_normalizer(history_batch)
            pred_z = adapter(history_batch)
            loss = F.mse_loss(pred_z, target_z_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args_cli.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(adapter.parameters(), args_cli.max_grad_norm)
            optimizer.step()
            loss_sum += float(loss.item())
        mean_loss = loss_sum / max(updates, 1)
        new_data_loss_before_update = new_data_loss_sum / max(num_steps_per_iter, 1)

        mean_reward_per_step = reward_sum / max(num_steps_per_iter, 1)
        mean_done_per_step = done_sum / max(num_steps_per_iter, 1)
        mean_action_abs = action_mean_abs_sum / max(num_steps_per_iter, 1)
        mean_z_hat_abs = z_hat_abs_sum / max(num_steps_per_iter, 1)
        mean_z_teacher_abs = z_teacher_abs_sum / max(num_steps_per_iter, 1)
        elapsed = time.time() - start_time

        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_checkpoint(
                path=best_latent_path,
                adapter=adapter,
                history_normalizer=history_normalizer,
                optimizer=optimizer,
                iteration=iteration,
                samples=buffer.size,
                best_loss=best_loss,
                teacher_checkpoint=teacher_checkpoint,
                layout=layout,
                dims=dims,
                selection_metric_name="dagger/latent_mse",
                selection_metric_value=best_loss,
            )

        policy_eval_metrics: dict[str, Any] = {}
        should_policy_eval = policy_eval_enabled and (
            iteration % int(args_cli.policy_eval_every_dag_it) == 0 or iteration == num_iterations
        )
        if should_policy_eval:
            policy_eval_metrics, obs = _evaluate_current_adapter_policy(
                vec_env=vec_env,
                teacher=teacher,
                adapter=adapter,
                history_normalizer=history_normalizer,
                dims=dims,
                eval_n=int(args_cli.n_env_eval),
                iteration=iteration,
            )
            eval_finish_time_steps = policy_eval_metrics.get("policy_eval/Episode/finishTimeSteps")
            if eval_finish_time_steps is not None and float(eval_finish_time_steps) < best_eval_finish_time_steps:
                best_eval_finish_time_steps = float(eval_finish_time_steps)
                _save_checkpoint(
                    path=os.path.join(log_dir, "checkpoints", "adapter_best.pt"),
                    adapter=adapter,
                    history_normalizer=history_normalizer,
                    optimizer=optimizer,
                    iteration=iteration,
                    samples=buffer.size,
                    best_loss=best_loss,
                    teacher_checkpoint=teacher_checkpoint,
                    layout=layout,
                    dims=dims,
                    selection_metric_name="policy_eval/Episode/finishTimeSteps",
                    selection_metric_value=best_eval_finish_time_steps,
                    policy_eval=policy_eval_metrics,
                )

        if iteration % args_cli.save_interval == 0 or iteration == num_iterations:
            _save_checkpoint(
                path=os.path.join(log_dir, "checkpoints", "adapter_latest.pt"),
                adapter=adapter,
                history_normalizer=history_normalizer,
                optimizer=optimizer,
                iteration=iteration,
                samples=buffer.size,
                best_loss=best_loss,
                teacher_checkpoint=teacher_checkpoint,
                layout=layout,
                dims=dims,
                selection_metric_name=(
                    "policy_eval/Episode/finishTimeSteps" if policy_eval_enabled else "dagger/latent_mse"
                ),
                selection_metric_value=(best_eval_finish_time_steps if policy_eval_enabled else best_loss),
            )

        metrics = {
            "dagger/iteration": iteration,
            "dagger/num_iterations": num_iterations,
            "dagger/loss": mean_loss,
            "dagger/latent_mse": mean_loss,
            "dagger/best_loss": best_loss,
            "dagger/best_latent_mse": best_loss,
            "dagger/new_data_loss_before_update": new_data_loss_before_update,
            "policy_eval/best_finishTimeSteps": (
                best_eval_finish_time_steps if math.isfinite(best_eval_finish_time_steps) else None
            ),
            "rollout/reward_perStep": mean_reward_per_step,
            "rollout/done_perStep": mean_done_per_step,
            "rollout/action_mean_abs": mean_action_abs,
            "rollout/z_hat_abs": mean_z_hat_abs,
            "rollout/z_teacher_abs": mean_z_teacher_abs,
            "dagger/buffer_size": buffer.size,
            "dagger/collected_samples": collected,
            "dagger/total_env_steps": total_steps,
            "collection/samples": collected,
            "collection/env_steps": collected,
            "collection/samples_per_iteration": samples_per_iteration,
            "collection/steps_per_env": num_steps_per_iter,
            "collection/total_env_steps": total_steps,
            "optimization/updates": updates,
            "optimization/mini_batch_size": args_cli.mini_batch_size,
            "optimization/train_epochs": args_cli.train_epochs,
            "buffer/size": buffer.size,
            "buffer/capacity": buffer.capacity,
            "buffer/fill_fraction": buffer.size / max(buffer.capacity, 1),
            "time/elapsed_s": elapsed,
            "time/elapsed_min": elapsed / 60.0,
        }
        metrics.update(policy_eval_metrics)
        if wandb_run is not None:
            wandb_run.log({key: value for key, value in metrics.items() if value is not None}, step=iteration)
            wandb_run.summary["dagger/best_loss"] = best_loss
            wandb_run.summary["dagger/latest_loss"] = mean_loss
            wandb_run.summary["dagger/latest_new_data_loss_before_update"] = new_data_loss_before_update
            if math.isfinite(best_eval_finish_time_steps):
                wandb_run.summary["policy_eval/best_finishTimeSteps"] = best_eval_finish_time_steps
            wandb_run.summary["rollout/latest_reward_perStep"] = mean_reward_per_step
            wandb_run.summary["rollout/latest_done_perStep"] = mean_done_per_step
            wandb_run.summary["buffer/final_size"] = buffer.size
        if iteration % args_cli.log_interval == 0:
            print(
                f"[DAgger] iter={iteration:05d}/{num_iterations} "
                f"loss={mean_loss:.6f} best={best_loss:.6f} "
                f"reward_perStep={mean_reward_per_step:.3f} done_perStep={mean_done_per_step:.3f} "
                f"buffer={buffer.size} updates={updates} elapsed={elapsed/60.0:.1f}m",
                flush=True,
            )

    _save_checkpoint(
        path=os.path.join(log_dir, "checkpoints", "adapter_latest.pt"),
        adapter=adapter,
        history_normalizer=history_normalizer,
        optimizer=optimizer,
        iteration=num_iterations,
        samples=buffer.size,
        best_loss=best_loss,
        teacher_checkpoint=teacher_checkpoint,
        layout=layout,
        dims=dims,
        selection_metric_name=("policy_eval/Episode/finishTimeSteps" if policy_eval_enabled else "dagger/latent_mse"),
        selection_metric_value=(best_eval_finish_time_steps if policy_eval_enabled else best_loss),
    )
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[INFO] Finished DAgger training in {time.time() - start_time:.1f}s", flush=True)
    if policy_eval_enabled:
        print(
            f"[INFO] Best policy-eval adapter: {os.path.join(log_dir, 'checkpoints', 'adapter_best.pt')} "
            f"(policy_eval/Episode/finishTimeSteps={best_eval_finish_time_steps:.3f})",
            flush=True,
        )
        print(f"[INFO] Best latent-MSE adapter: {best_latent_path}", flush=True)
    else:
        print(f"[INFO] Best adapter: {os.path.join(log_dir, 'checkpoints', 'adapter_best.pt')}", flush=True)
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
