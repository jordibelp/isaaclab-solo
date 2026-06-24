# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the Solo12 base-IMU student history encoder with DAgger-style supervision.

The frozen teacher is a ``solo12-IMU-based-teacher`` checkpoint:
    privileged teacher obs -> teacher MLP encoder -> z
    [z, command] -> frozen actor head -> action

The student env exposes:
    [teacher obs, flattened base-IMU/proprio/action history, command]

This script rolls out the current student encoder through the frozen teacher actor
head, aggregates visited histories, and trains the student TCN to match the frozen
teacher latent. Optionally it also supervises the action mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_UPSTREAM_RSL_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train a Solo12 base-IMU TCN student with DAgger supervision.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during data collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos in env steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in env steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="solo12-IMU-student-dagger", help="DAgger task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RSL-RL config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for env and student init.")
parser.add_argument("--max_iterations", type=int, default=None, help="Alias for --dagger_iterations.")
parser.add_argument(
    "--teacher-checkpoint",
    "--teacher_checkpoint",
    dest="teacher_checkpoint",
    type=str,
    default=None,
    help="Path to a trained solo12-IMU-based-teacher checkpoint. Falls back to --checkpoint.",
)
parser.add_argument(
    "--adapter-checkpoint",
    "--adapter_checkpoint",
    dest="adapter_checkpoint",
    type=str,
    default=None,
    help="Optional previous DAgger adapter checkpoint to continue from.",
)
parser.add_argument("--dagger_iterations", type=int, default=None, help="Number of DAgger collect/train iterations.")
parser.add_argument("--num_steps_per_iter", type=int, default=None, help="Env steps collected per iteration.")
parser.add_argument("--train_epochs", type=int, default=4, help="Supervised epochs per DAgger iteration.")
parser.add_argument("--updates_per_iter", type=int, default=None, help="Override supervised updates per iteration.")
parser.add_argument(
    "--num_mini_batches",
    "--num-mini-batches",
    type=int,
    default=8,
    help="Number of supervised mini-batches per collected DAgger rollout.",
)
parser.add_argument(
    "--mini_batch_size",
    type=int,
    default=None,
    help="Optional explicit supervised mini-batch size. Overrides --num_mini_batches when set.",
)
parser.add_argument("--learning_rate", type=float, default=3.0e-4, help="AdamW learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.0, help="AdamW weight decay.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Student encoder gradient clipping norm.")
parser.add_argument("--dataset_capacity", type=int, default=262_144, help="Replay ring-buffer capacity.")
parser.add_argument("--save_interval", type=int, default=50, help="Save latest checkpoint every N iterations.")
parser.add_argument("--log_interval", type=int, default=1, help="Print every N iterations.")
parser.add_argument("--no-history-normalization", action="store_true", default=False, help="Disable history normalizer.")
parser.add_argument("--stochastic-actions", action="store_true", default=False, help="Sample from frozen actor std.")
parser.add_argument("--supervise-action", action="store_true", default=False, help="Also supervise frozen actor output.")
parser.add_argument("--action-loss-coef", type=float, default=0.25, help="Weight for optional action MSE.")
parser.add_argument("--dagger_experiment_name", type=str, default="solo12_base_imu_dagger", help="Log dir name.")
parser.add_argument("--run-name", type=str, default=None, help="Optional run name suffix.")
parser.add_argument("--disable-wandb", action="store_true", default=False, help="Disable custom W&B logging.")
parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity/team.")
parser.add_argument("--wandb-project", type=str, default="borinotIsaacLab", help="W&B project name.")
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
import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401
from isaaclab_tasks.direct.solo12.agents.base_imu_actor_critic import (
    BaseImuTcnEncoder,
    Solo12BaseImuTeacherActorCritic,
)


class ReplayBuffer:
    def __init__(self, capacity: int, history_dim: int, latent_dim: int, action_dim: int, command_dim: int) -> None:
        self.capacity = int(capacity)
        self.history = torch.empty((self.capacity, history_dim), dtype=torch.float32, device="cpu")
        self.target_z = torch.empty((self.capacity, latent_dim), dtype=torch.float32, device="cpu")
        self.command = torch.empty((self.capacity, command_dim), dtype=torch.float32, device="cpu")
        self.target_action = torch.empty((self.capacity, action_dim), dtype=torch.float32, device="cpu")
        self.size = 0
        self.pos = 0

    def add(
        self,
        history: torch.Tensor,
        target_z: torch.Tensor,
        command: torch.Tensor,
        target_action: torch.Tensor,
    ) -> None:
        tensors = [x.detach().to(device="cpu", dtype=torch.float32) for x in (history, target_z, command, target_action)]
        num = tensors[0].shape[0]
        if num >= self.capacity:
            self.history.copy_(tensors[0][-self.capacity :])
            self.target_z.copy_(tensors[1][-self.capacity :])
            self.command.copy_(tensors[2][-self.capacity :])
            self.target_action.copy_(tensors[3][-self.capacity :])
            self.size = self.capacity
            self.pos = 0
            return
        end = self.pos + num
        targets = (self.history, self.target_z, self.command, self.target_action)
        if end <= self.capacity:
            for target, tensor in zip(targets, tensors):
                target[self.pos : end].copy_(tensor)
        else:
            first = self.capacity - self.pos
            for target, tensor in zip(targets, tensors):
                target[self.pos :].copy_(tensor[:first])
                target[: end - self.capacity].copy_(tensor[first:])
        self.pos = end % self.capacity
        self.size = min(self.capacity, self.size + num)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = torch.randint(self.size, (batch_size,), device="cpu")
        return (
            self.history[indices].to(device=device),
            self.target_z[indices].to(device=device),
            self.command[indices].to(device=device),
            self.target_action[indices].to(device=device),
        )


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _resolve_teacher_checkpoint() -> str:
    checkpoint = args_cli.teacher_checkpoint or args_cli.checkpoint
    if checkpoint is None:
        raise ValueError("Pass --teacher-checkpoint /path/to/teacher.pt for DAgger training.")
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")
    return checkpoint


def _checkpoint_model_state_dict(path: str, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(f"Could not find model state_dict in checkpoint: {path}")


def _load_teacher(
    *,
    checkpoint_path: str,
    policy_cfg: Any,
    num_actions: int,
    teacher_obs_dim: int,
    device: torch.device,
) -> Solo12BaseImuTeacherActorCritic:
    kwargs = _cfg_to_dict(policy_cfg)
    kwargs.pop("class_name", None)
    dummy_obs = TensorDict(
        {
            "policy": torch.zeros((1, teacher_obs_dim), device=device),
            "critic": torch.zeros((1, teacher_obs_dim), device=device),
        },
        batch_size=[1],
    )
    teacher = Solo12BaseImuTeacherActorCritic(
        obs=dummy_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions,
        **kwargs,
    ).to(device)
    teacher.load_state_dict(_checkpoint_model_state_dict(checkpoint_path, map_location=device), strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def _teacher_targets_and_student_action(
    *,
    teacher: Solo12BaseImuTeacherActorCritic,
    adapter: BaseImuTcnEncoder,
    history_normalizer: EmpiricalNormalization | None,
    teacher_obs_raw: torch.Tensor,
    history_raw: torch.Tensor,
    stochastic_actions: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_obs = teacher.actor_obs_normalizer(teacher_obs_raw)
    target_z = teacher.encode_teacher(teacher_obs)
    command = teacher_obs[:, teacher.teacher_encoder_obs_dim : teacher.teacher_encoder_obs_dim + teacher.command_dim]
    history = history_normalizer(history_raw) if history_normalizer is not None else history_raw
    z_hat = adapter(history)
    student_mean = teacher.actor(torch.cat((z_hat, command), dim=-1))
    teacher_action = teacher.actor(torch.cat((target_z, command), dim=-1))
    if stochastic_actions:
        if teacher.noise_std_type == "scalar":
            std = teacher.std.expand_as(student_mean)
        else:
            std = torch.exp(teacher.log_std).expand_as(student_mean)
        action = student_mean + torch.randn_like(student_mean) * std
    else:
        action = student_mean
    return target_z, command, teacher_action, action


def _save_checkpoint(path: str, **payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float, bool)):
        return float(value)
    return None


def _accumulate_env_log(sums: dict[str, float], counts: dict[str, int], extras: dict[str, Any]) -> None:
    log = extras.get("log", {}) if isinstance(extras, dict) else {}
    if not isinstance(log, dict):
        return
    for key, value in log.items():
        scalar = _to_float(value)
        if scalar is None:
            continue
        sums[key] = sums.get(key, 0.0) + scalar
        counts[key] = counts.get(key, 0) + 1


def _mean_env_log(sums: dict[str, float], counts: dict[str, int], prefix: str = "env") -> dict[str, float]:
    return {f"{prefix}/{key}": value / max(1, counts[key]) for key, value in sums.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _prepare_wandb_run_id() -> str | None:
    if args_cli.disable_wandb:
        return None
    existing_run_id = os.environ.get("WANDB_RUN_ID")
    if existing_run_id:
        return existing_run_id
    try:
        import wandb

        wandb_run_id = wandb.util.generate_id()
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        return wandb_run_id
    except Exception as exc:
        print(f"[WARN] W&B run id generation failed; log dir will not include a W&B id: {exc}", flush=True)
        return None


def _maybe_init_wandb(log_dir: str, run_name: str, config: dict[str, Any], run_id: str | None = None):
    if args_cli.disable_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] W&B import failed; continuing without W&B: {exc}", flush=True)
        return None

    project = args_cli.log_project_name or args_cli.wandb_project
    try:
        run = wandb.init(
            project=project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_name or run_name,
            id=run_id,
            dir=log_dir,
            config=_json_safe(config),
        )
        wandb.define_metric("dagger/iteration")
        for metric_name in (
            "dagger/*",
            "rollout/*",
            "collection/*",
            "optimization/*",
            "buffer/*",
            "time/*",
            "env/*",
        ):
            wandb.define_metric(metric_name, step_metric="dagger/iteration")
        for filename in ("env.yaml", "agent.yaml", "dagger_args.json"):
            path = os.path.join(log_dir, "params", filename)
            if os.path.isfile(path):
                wandb.save(path, base_path=log_dir, policy="now")
        return run
    except Exception as exc:
        print(f"[WARN] W&B init failed; continuing without W&B: {exc}", flush=True)
        return None


def _sync_base_imu_policy_cfg_from_env_cfg(env_cfg, agent_cfg) -> None:
    """Keep base-IMU dimensions aligned after Hydra env overrides."""
    refresh_dimensions = getattr(env_cfg, "refresh_base_imu_dimensions", None)
    if callable(refresh_dimensions):
        refresh_dimensions()

    policy_cfg = getattr(agent_cfg, "policy", None)
    if policy_cfg is None:
        return

    env_to_policy_fields = {
        "teacher_encoder_obs_dim": "teacher_encoder_obs_dim",
        "teacher_latent_dim": "teacher_latent_dim",
        "teacher_encoder_hidden_dims": "teacher_encoder_hidden_dims",
        "teacher_critic_obs_dim": "teacher_critic_obs_dim",
        "base_imu_history_length": "history_len",
        "base_imu_history_sample_dim": "history_sample_dim",
        "base_imu_tcn_channels": "tcn_channels",
        "base_imu_tcn_latent_dim": "tcn_latent_dim",
        "base_imu_tcn_kernel_size": "tcn_kernel_size",
        "base_imu_tcn_activation": "tcn_activation",
        "feed_history_encoding_to_critic": "feed_history_encoding_to_critic",
    }
    for env_field, policy_field in env_to_policy_fields.items():
        if hasattr(env_cfg, env_field) and hasattr(policy_cfg, policy_field):
            setattr(policy_cfg, policy_field, getattr(env_cfg, env_field))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    _sync_base_imu_policy_cfg_from_env_cfg(env_cfg, agent_cfg)

    teacher_checkpoint = _resolve_teacher_checkpoint()
    num_iterations = args_cli.dagger_iterations or args_cli.max_iterations or 1000
    num_steps_per_iter = args_cli.num_steps_per_iter or int(getattr(agent_cfg, "num_steps_per_env", 24))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_suffix = args_cli.run_name or f"teacher-{Path(teacher_checkpoint).stem}"
    wandb_run_id = _prepare_wandb_run_id()
    log_dir_name = f"{timestamp}_{run_suffix}"
    if wandb_run_id is not None:
        log_dir_name = f"{log_dir_name}_{wandb_run_id}"
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.dagger_experiment_name, log_dir_name))
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    env_cfg.log_dir = log_dir
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "dagger_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args_cli), f, indent=2, sort_keys=True, default=str)
    wandb_run = _maybe_init_wandb(
        log_dir,
        run_suffix,
        {
            "teacher_checkpoint": teacher_checkpoint,
            "dagger_iterations": int(num_iterations),
            "num_steps_per_iter": int(num_steps_per_iter),
            "num_envs": int(env_cfg.scene.num_envs),
            "dataset_capacity": int(args_cli.dataset_capacity),
            "num_mini_batches": int(args_cli.num_mini_batches),
            "mini_batch_size_override": args_cli.mini_batch_size,
            "train_epochs": int(args_cli.train_epochs),
            "learning_rate": float(args_cli.learning_rate),
            "weight_decay": float(args_cli.weight_decay),
            "stochastic_actions": bool(args_cli.stochastic_actions),
            "supervise_action": bool(args_cli.supervise_action),
            "action_loss_coef": float(args_cli.action_loss_coef),
            "wandb_run_id": wandb_run_id,
            "env": _cfg_to_dict(env_cfg),
            "agent": _cfg_to_dict(agent_cfg),
        },
        run_id=wandb_run_id,
    )

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
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    device = torch.device(vec_env.unwrapped.device)
    raw_env = vec_env.unwrapped
    teacher_obs_dim = int(raw_env.cfg.teacher_critic_obs_dim)
    history_dim = int(raw_env.cfg.base_imu_history_flat_dim)
    command_dim = 3
    policy_obs_dim = int(vec_env.get_observations()["policy"].shape[-1])
    expected_obs_dim = teacher_obs_dim + history_dim + command_dim
    if policy_obs_dim != expected_obs_dim:
        raise ValueError(f"DAgger obs dim {policy_obs_dim} != expected {expected_obs_dim}.")

    teacher = _load_teacher(
        checkpoint_path=teacher_checkpoint,
        policy_cfg=agent_cfg.policy,
        num_actions=vec_env.num_actions,
        teacher_obs_dim=teacher_obs_dim,
        device=device,
    )
    latent_dim = int(teacher.teacher_latent_dim)
    adapter = BaseImuTcnEncoder(
        history_len=int(raw_env.cfg.base_imu_history_length),
        sample_dim=int(raw_env.cfg.base_imu_history_sample_dim),
        channels=int(raw_env.cfg.base_imu_tcn_channels),
        latent_dim=latent_dim,
        kernel_size=int(raw_env.cfg.base_imu_tcn_kernel_size),
        activation=str(raw_env.cfg.base_imu_tcn_activation),
    ).to(device)
    history_normalizer = None
    if not args_cli.no_history_normalization:
        history_normalizer = EmpiricalNormalization(history_dim).to(device)
        history_normalizer.train()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args_cli.learning_rate, weight_decay=args_cli.weight_decay)
    if args_cli.adapter_checkpoint is not None:
        ckpt = torch.load(os.path.abspath(os.path.expanduser(args_cli.adapter_checkpoint)), map_location=device, weights_only=False)
        saved_dims = ckpt.get("dims", {})
        saved_sample_dim = saved_dims.get("history_sample_dim") if isinstance(saved_dims, dict) else None
        if saved_sample_dim is not None and int(saved_sample_dim) != int(raw_env.cfg.base_imu_history_sample_dim):
            raise ValueError(
                "Adapter checkpoint IMU layout does not match the current environment: "
                f"checkpoint history_sample_dim={saved_sample_dim}, current={raw_env.cfg.base_imu_history_sample_dim}."
            )
        adapter.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        if history_normalizer is not None and ckpt.get("history_normalizer_state_dict") is not None:
            history_normalizer.load_state_dict(ckpt["history_normalizer_state_dict"], strict=True)

    buffer = ReplayBuffer(args_cli.dataset_capacity, history_dim, latent_dim, vec_env.num_actions, command_dim)
    obs = vec_env.get_observations()
    best_loss = float("inf")
    start_time = time.time()
    print(f"[INFO] Logging Solo12 base-IMU DAgger run in: {log_dir}", flush=True)
    print(f"[INFO] Teacher checkpoint: {teacher_checkpoint}", flush=True)
    print(
        f"[INFO] Student history: T={raw_env.cfg.base_imu_history_length}, "
        f"D={raw_env.cfg.base_imu_history_sample_dim}, flat={history_dim} -> z[{latent_dim}]",
        flush=True,
    )

    for iteration in range(1, int(num_iterations) + 1):
        adapter.eval()
        reward_sum = 0.0
        done_sum = 0
        collect_loss_sum = 0.0
        collected = 0
        env_log_sums: dict[str, float] = {}
        env_log_counts: dict[str, int] = {}

        for _ in range(int(num_steps_per_iter)):
            policy_obs = obs["policy"]
            teacher_obs_raw = policy_obs[:, :teacher_obs_dim]
            history_raw = policy_obs[:, teacher_obs_dim : teacher_obs_dim + history_dim]
            if history_normalizer is not None:
                history_normalizer.update(history_raw)
            target_z, command_norm, teacher_action, action = _teacher_targets_and_student_action(
                teacher=teacher,
                adapter=adapter,
                history_normalizer=history_normalizer,
                teacher_obs_raw=teacher_obs_raw,
                history_raw=history_raw,
                stochastic_actions=bool(args_cli.stochastic_actions),
            )
            buffer.add(history_raw, target_z, command_norm, teacher_action)
            obs, rewards, dones, extras = vec_env.step(action)
            reward_sum += float(rewards.mean().item())
            done_sum += int(torch.count_nonzero(dones).item())
            _accumulate_env_log(env_log_sums, env_log_counts, extras)
            collect_loss_sum += float(nn.functional.mse_loss(adapter(history_normalizer(history_raw) if history_normalizer is not None else history_raw), target_z).item())
            collected += history_raw.shape[0]

        adapter.train()
        if args_cli.mini_batch_size is not None:
            mini_batch_size = int(args_cli.mini_batch_size)
            if mini_batch_size <= 0:
                raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}.")
            default_updates = max(1, int(args_cli.train_epochs) * max(1, buffer.size // mini_batch_size))
        else:
            num_mini_batches = int(args_cli.num_mini_batches)
            if num_mini_batches <= 0:
                raise ValueError(f"num_mini_batches must be positive, got {num_mini_batches}.")
            train_samples = max(1, min(buffer.size, collected))
            mini_batch_size = max(1, (train_samples + num_mini_batches - 1) // num_mini_batches)
            default_updates = max(1, int(args_cli.train_epochs) * num_mini_batches)

        updates = args_cli.updates_per_iter
        if updates is None:
            updates = default_updates
        loss_sum = 0.0
        latent_loss_sum = 0.0
        action_loss_sum = 0.0
        for _ in range(int(updates)):
            history_batch, z_batch, command_batch, action_batch = buffer.sample(mini_batch_size, device)
            history_input = history_normalizer(history_batch) if history_normalizer is not None else history_batch
            z_hat = adapter(history_input)
            latent_loss = nn.functional.mse_loss(z_hat, z_batch)
            action_loss = torch.zeros((), device=device)
            if args_cli.supervise_action:
                action_hat = teacher.actor(torch.cat((z_hat, command_batch), dim=-1))
                action_loss = nn.functional.mse_loss(action_hat, action_batch)
            loss = latent_loss + float(args_cli.action_loss_coef) * action_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args_cli.max_grad_norm))
            optimizer.step()
            loss_sum += float(loss.item())
            latent_loss_sum += float(latent_loss.item())
            action_loss_sum += float(action_loss.item())

        mean_loss = loss_sum / max(1, int(updates))
        if mean_loss < best_loss:
            best_loss = mean_loss
            _save_checkpoint(
                os.path.join(log_dir, "checkpoints", "adapter_best.pt"),
                adapter_state_dict=adapter.state_dict(),
                history_normalizer_state_dict=history_normalizer.state_dict() if history_normalizer is not None else None,
                optimizer_state_dict=optimizer.state_dict(),
                iteration=iteration,
                best_loss=best_loss,
                teacher_checkpoint=teacher_checkpoint,
                dims={
                    "teacher_obs_dim": teacher_obs_dim,
                    "history_len": int(raw_env.cfg.base_imu_history_length),
                    "history_sample_dim": int(raw_env.cfg.base_imu_history_sample_dim),
                    "history_flat_dim": history_dim,
                    "latent_dim": latent_dim,
                    "command_dim": command_dim,
                },
                imu_layout={
                    "imu_ekf_processed_inputs": bool(raw_env.cfg.imu_ekf_processed_inputs),
                    "use_rotMat_on_imu_encoder": bool(raw_env.cfg.use_rotMat_on_imu_encoder),
                    "base_imu_obs_dim": int(raw_env.cfg.base_imu_obs_dim),
                },
            )

        if iteration % int(args_cli.save_interval) == 0 or iteration == int(num_iterations):
            _save_checkpoint(
                os.path.join(log_dir, "checkpoints", "adapter_latest.pt"),
                adapter_state_dict=adapter.state_dict(),
                history_normalizer_state_dict=history_normalizer.state_dict() if history_normalizer is not None else None,
                optimizer_state_dict=optimizer.state_dict(),
                iteration=iteration,
                best_loss=best_loss,
                teacher_checkpoint=teacher_checkpoint,
                dims={
                    "teacher_obs_dim": teacher_obs_dim,
                    "history_len": int(raw_env.cfg.base_imu_history_length),
                    "history_sample_dim": int(raw_env.cfg.base_imu_history_sample_dim),
                    "history_flat_dim": history_dim,
                    "latent_dim": latent_dim,
                    "command_dim": command_dim,
                },
                imu_layout={
                    "imu_ekf_processed_inputs": bool(raw_env.cfg.imu_ekf_processed_inputs),
                    "use_rotMat_on_imu_encoder": bool(raw_env.cfg.use_rotMat_on_imu_encoder),
                    "base_imu_obs_dim": int(raw_env.cfg.base_imu_obs_dim),
                },
            )

        mean_loss = loss_sum / max(1, int(updates))
        mean_latent_loss = latent_loss_sum / max(1, int(updates))
        mean_action_loss = action_loss_sum / max(1, int(updates))
        mean_collect_loss = collect_loss_sum / max(1, int(num_steps_per_iter))
        mean_reward_per_step = reward_sum / max(1, int(num_steps_per_iter))
        mean_done_per_step = done_sum / max(1, int(num_steps_per_iter))
        elapsed = time.time() - start_time
        metrics = {
            "dagger/iteration": iteration,
            "dagger/num_iterations": int(num_iterations),
            "dagger/loss": mean_loss,
            "dagger/latent_mse": mean_latent_loss,
            "dagger/action_mse": mean_action_loss,
            "dagger/collect_latent_mse": mean_collect_loss,
            "dagger/best_loss": best_loss,
            "rollout/reward_perStep": mean_reward_per_step,
            "rollout/done_perStep": mean_done_per_step,
            "rollout/dones": done_sum,
            "collection/samples": collected,
            "collection/samples_per_iteration": int(num_steps_per_iter) * int(env_cfg.scene.num_envs),
            "collection/steps_per_env": int(num_steps_per_iter),
            "collection/total_samples": iteration * collected,
            "optimization/updates": int(updates),
            "optimization/mini_batch_size": mini_batch_size,
            "optimization/train_epochs": int(args_cli.train_epochs),
            "buffer/size": buffer.size,
            "buffer/capacity": buffer.capacity,
            "buffer/fill_fraction": buffer.size / max(1, buffer.capacity),
            "time/elapsed_s": elapsed,
            "time/elapsed_min": elapsed / 60.0,
        }
        metrics.update(_mean_env_log(env_log_sums, env_log_counts))
        if wandb_run is not None:
            wandb_run.log({key: value for key, value in metrics.items() if value is not None}, step=iteration)
            wandb_run.summary["dagger/best_loss"] = best_loss
            wandb_run.summary["dagger/latest_loss"] = mean_loss
            wandb_run.summary["rollout/latest_reward_perStep"] = mean_reward_per_step
            wandb_run.summary["rollout/latest_done_perStep"] = mean_done_per_step
            wandb_run.summary["buffer/final_size"] = buffer.size

        if iteration % int(args_cli.log_interval) == 0:
            print(
                f"[{iteration:05d}/{int(num_iterations):05d}] "
                f"loss={mean_loss:.6f} latent={mean_latent_loss:.6f} "
                f"action={mean_action_loss:.6f} "
                f"collect_latent={mean_collect_loss:.6f} "
                f"reward={mean_reward_per_step:.3f} dones={done_sum} "
                f"buffer={buffer.size} samples+={collected} batch={mini_batch_size} updates={int(updates)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    vec_env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
    simulation_app.close()
