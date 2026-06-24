# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Headless batch evaluator for trained Solo12 race RSL-RL policies.

Example:
./isaaclab.sh -p source/scripts/rsl_rl/solo_race_eval.py \
  --task="Isaac-Solo12-Race-Direct-v0" \
  --checkpoint "/path/to/best_model.pt" \
  --eval_n 1024 \
  --eval_n_parallel 256 \
  --max_steps 1000 \
  --out_dir logs/rsl_rl/solo12_race_eval

DAgger adapter example:
./isaaclab.sh -p source/scripts/rsl_rl/solo_race_eval.py \
  --task="Solo12-Race-ParamsConditionedEnc-Direct-v0" \
  --checkpoint "/path/to/adapter_best.pt" \
  --dagger-teacher-checkpoint "/path/to/params_conditioned_enc_best_model.pt" \
  --eval_n 1024 \
  --eval_n_parallel 256 \
  --max_steps 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_UPSTREAM_RSL_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _optional_int(value: str) -> int | None:
    if str(value).strip().lower() in {"none", "null", ""}:
        return None
    return int(value)


parser = argparse.ArgumentParser(description="Evaluate a Solo12 race policy and write JSON metrics.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Solo12-Race-Direct-v0",
    help="Race task name, e.g. Isaac-Solo12-Race-Direct-v0 or Isaac-Solo12-Race-IMU-Direct-v0.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--friction-seed",
    "--friction_seed",
    dest="friction_seed",
    type=int,
    default=None,
    help=argparse.SUPPRESS,  # Deprecated: --seed now controls eval friction too.
)
parser.add_argument(
    "--eval_n",
    type=_optional_int,
    default=1024,
    help="Number of one-episode rollouts to evaluate. Use None to default to --eval_n_parallel.",
)
parser.add_argument(
    "--eval_n_parallel",
    "--eval_n_parlel",
    dest="eval_n_parallel",
    type=int,
    default=256,
    help="Number of environments to evaluate in parallel per batch. Alias: --eval_n_parlel.",
)
parser.add_argument(
    "--max_steps",
    type=_optional_int,
    default=1000,
    help="Evaluation episode timeout in env steps. Default: 1000. Pass None to use --episode_length_s instead.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Evaluation episode timeout in seconds. Used only when --max_steps None.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Directory where eval_stats_*.json.gz is written. Defaults to <checkpoint-run-dir>/eval.",
)
parser.add_argument(
    "--eval-output-format",
    "--eval_output_format",
    dest="eval_output_format",
    type=str,
    choices=("json.gz", "json"),
    default="json.gz",
    help="Eval artifact format. Default: json.gz. Use json for an uncompressed, human-readable file.",
)
parser.add_argument(
    "--no-wandb",
    "--no_wandb",
    dest="no_wandb",
    action="store_true",
    default=False,
    help="Disable W&B upload. By default eval metrics and the JSON artifact are uploaded to W&B.",
)
parser.add_argument(
    "--wandb-project",
    "--wandb_project",
    dest="wandb_project",
    type=str,
    default="soloRace_evals",
    help="W&B project used for eval uploads.",
)
parser.add_argument(
    "--wandb-name",
    "--wandb_name",
    dest="wandb_name",
    type=str,
    default=None,
    help="Optional W&B run name for this eval. Defaults to a compact checkpoint/task/seed name.",
)
parser.add_argument(
    "--progress_interval_s",
    type=float,
    default=5.0,
    help="Wall-clock seconds between progress prints during each evaluation batch. Set <=0 to disable.",
)
parser.add_argument(
    "--keep_training_stochasticity",
    action="store_true",
    default=False,
    help=(
        "Keep training-time stochasticity. By default, evaluation disables domain randomization events, observation "
        "corruption, reset velocity randomization, and actuation delay. Race patch friction remains randomized "
        "when the task config enables randomize_fric_coefs."
    ),
)
parser.add_argument(
    "--within-episode-fric-resample",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override Solo12 race within-episode patch-friction resampling during evaluation.",
)
parser.add_argument(
    "--within-episode-fric-resample-time-range",
    type=float,
    nargs=2,
    metavar=("MIN_S", "MAX_S"),
    default=None,
    help="Uniform time range in seconds between within-episode patch-friction resamples.",
)
parser.add_argument(
    "--group-all-patches-single-bucket",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override whether all Solo12 race patches share one sampled friction bucket during evaluation.",
)
parser.add_argument(
    "--slip-angle-delta-deg",
    "--slip_angle_delta_deg",
    dest="slip_angle_delta_deg",
    type=float,
    default=0.5,
    help=(
        "Half-width in degrees around the dynamic friction-cone angle used for slip-contact metrics. "
        "Default: 0.5."
    ),
)
parser.add_argument(
    "--dagger-teacher-checkpoint",
    "--dagger_teacher_checkpoint",
    dest="dagger_teacher_checkpoint",
    type=str,
    default=None,
    help=(
        "Phase-1 ParamsConditionedEnc teacher checkpoint used with a DAgger adapter checkpoint. "
        "If omitted, the script uses the path saved in the adapter checkpoint or searches local exported checkpoints."
    ),
)
parser.add_argument(
    "--dagger-teacher-shared-networks",
    action="store_true",
    default=False,
    help="Set if the phase-1 ParamsConditionedEnc teacher checkpoint was trained with shared actor/critic networks.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required for evaluation")
if args_cli.eval_n is not None and args_cli.eval_n <= 0:
    parser.error("--eval_n must be positive when provided")
if args_cli.eval_n_parallel <= 0:
    parser.error("--eval_n_parallel must be positive")
if args_cli.slip_angle_delta_deg < 0.0:
    parser.error("--slip-angle-delta-deg must be non-negative")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401
from race_dagger_adapter_policy import (  # noqa: E402
    apply_checkpoint_architecture_to_policy_cfg,
    configure_env_cfg_for_dagger_adapter,
    load_dagger_adapter_checkpoint,
    load_dagger_latent_policy,
)


def _disable_training_stochasticity_for_eval(env_cfg, *, preserve_patch_friction_randomization: bool = False) -> list[str]:
    disabled = []

    if getattr(env_cfg, "events", None):
        env_cfg.events = None
        disabled.append("events")

    if hasattr(env_cfg, "enable_observation_corruption") and env_cfg.enable_observation_corruption:
        env_cfg.enable_observation_corruption = False
        disabled.append("observation_corruption")

    if getattr(env_cfg, "enable_reset_pose_randomization", False):
        env_cfg.enable_reset_pose_randomization = False
        disabled.append("reset_pose_randomization")

    if hasattr(env_cfg, "reset_base_lin_vel_range"):
        env_cfg.reset_base_lin_vel_range = (0.0, 0.0)
        disabled.append("reset_base_lin_vel_range")

    if hasattr(env_cfg, "reset_base_ang_vel_range"):
        env_cfg.reset_base_ang_vel_range = (0.0, 0.0)
        disabled.append("reset_base_ang_vel_range")

    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
        disabled.append("actuation_delay_range")

    if (
        hasattr(env_cfg, "randomize_fric_coefs")
        and getattr(env_cfg, "randomize_fric_coefs")
        and not preserve_patch_friction_randomization
    ):
        env_cfg.randomize_fric_coefs = False
        disabled.append("randomize_fric_coefs")

    return disabled


_EVAL_FRICTION_BOOL_FIELDS = (
    "randomize_fric_coefs",
    "randomize_friction_bucket_values",
    "group_all_patches_single_bucket",
    "within_episode_fric_resample",
)
_EVAL_FRICTION_RANGE_FIELDS = ("within_episode_fric_resample_time_range",)


def _find_run_env_yaml(checkpoint_path: str | os.PathLike[str] | None) -> Path | None:
    """Find the training env.yaml associated with a checkpoint path."""
    if checkpoint_path is None:
        return None
    checkpoint = Path(os.path.expanduser(str(checkpoint_path))).resolve()
    candidates = [checkpoint.parent, *checkpoint.parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        env_yaml = candidate / "params" / "env.yaml"
        if env_yaml.is_file():
            return env_yaml
    return None


def _parse_bool_from_env_yaml(lines: list[str], field: str) -> bool | None:
    pattern = re.compile(rf"^{re.escape(field)}:\s*(true|false)\s*(?:#.*)?$", re.IGNORECASE)
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            return match.group(1).lower() == "true"
    return None


def _parse_float_pair_from_env_yaml(lines: list[str], field: str) -> tuple[float, float] | None:
    field_prefix = f"{field}:"
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(field_prefix):
            continue

        values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped[len(field_prefix) :])]
        if len(values) >= 2:
            return values[0], values[1]

        for follow in lines[idx + 1 :]:
            follow_stripped = follow.strip()
            if not follow_stripped or follow_stripped.startswith("#"):
                continue
            if not follow.startswith((" ", "-")) and ":" in follow_stripped:
                break
            if follow_stripped.startswith("-"):
                found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", follow_stripped)
                if found:
                    values.append(float(found[0]))
                    if len(values) >= 2:
                        return values[0], values[1]
    return None


def _load_eval_friction_config_from_env_yaml(env_yaml: Path) -> dict[str, Any]:
    try:
        lines = env_yaml.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[WARN] Could not read training env config {env_yaml}: {exc}", flush=True)
        return {}

    config: dict[str, Any] = {}
    for field in _EVAL_FRICTION_BOOL_FIELDS:
        value = _parse_bool_from_env_yaml(lines, field)
        if value is not None:
            config[field] = value
    for field in _EVAL_FRICTION_RANGE_FIELDS:
        value = _parse_float_pair_from_env_yaml(lines, field)
        if value is not None:
            min_s, max_s = sorted(float(v) for v in value)
            config[field] = (max(0.0, min_s), max_s)
    return config


def _infer_eval_friction_config(*checkpoint_paths: str | os.PathLike[str] | None) -> tuple[dict[str, Any], dict[str, str]]:
    """Infer race-friction eval settings from saved training env.yaml files.

    Paths passed later have higher priority.
    """
    config: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for checkpoint_path in checkpoint_paths:
        env_yaml = _find_run_env_yaml(checkpoint_path)
        if env_yaml is None:
            continue
        loaded = _load_eval_friction_config_from_env_yaml(env_yaml)
        for key, value in loaded.items():
            config[key] = value
            sources[key] = str(env_yaml)
    return config, sources


def _apply_eval_friction_config(env_cfg, config: dict[str, Any], sources: dict[str, str], *, label: str) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    for key, value in config.items():
        if not hasattr(env_cfg, key):
            continue
        setattr(env_cfg, key, value)
        applied[key] = value

    if applied:
        details = ", ".join(f"{key}={value}" for key, value in sorted(applied.items()))
        source_list = sorted({sources.get(key, label) for key in applied})
        print(f"[INFO] Applied Solo12 race friction eval config from {label}: {details}", flush=True)
        for source in source_list:
            print(f"[INFO]   source: {source}", flush=True)
    return applied


def _explicit_eval_friction_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args_cli.within_episode_fric_resample is not None:
        overrides["within_episode_fric_resample"] = bool(args_cli.within_episode_fric_resample)
    if args_cli.within_episode_fric_resample_time_range is not None:
        min_s, max_s = sorted(float(value) for value in args_cli.within_episode_fric_resample_time_range)
        if max_s <= 0.0:
            raise ValueError("--within-episode-fric-resample-time-range must contain at least one positive value.")
        overrides["within_episode_fric_resample_time_range"] = (max(0.0, min_s), max_s)
    if args_cli.group_all_patches_single_bucket is not None:
        overrides["group_all_patches_single_bucket"] = bool(args_cli.group_all_patches_single_bucket)
    return overrides


def _to_bool_tensor(value: Any, device: torch.device, num_envs: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.bool).reshape(-1)[:num_envs]
    return torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)[:num_envs]


def _safe_bool_metric(raw_env, name: str, default: bool = False) -> torch.Tensor:
    try:
        value = getattr(raw_env, name)
        if isinstance(value, torch.Tensor):
            return value.to(device=raw_env.device, dtype=torch.bool)
    except Exception:
        pass
    return torch.full((raw_env.num_envs,), default, device=raw_env.device, dtype=torch.bool)


def _floor_collision_now(raw_env) -> torch.Tensor:
    try:
        return raw_env._compute_filtered_base_contact(raw_env._base_floor_contact_sensor, raw_env.cfg.base_contact_threshold)
    except Exception:
        return torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)


def _pillar_collision_now(raw_env) -> torch.Tensor:
    try:
        return raw_env._compute_filtered_base_contact(raw_env._base_pillar_contact_sensor, raw_env.cfg.base_contact_threshold)
    except Exception:
        return torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)


def _thigh_contact_now(raw_env) -> torch.Tensor:
    try:
        return raw_env._compute_contact_count(raw_env._thigh_body_ids, raw_env.cfg.undesired_contact_threshold) > 0
    except Exception:
        return torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)


def _gate_progress_ratio(raw_env) -> torch.Tensor:
    try:
        target_count = max(int(raw_env._target_count), 1)
        return torch.clamp(raw_env._current_gate_idx.float() / target_count, max=1.0)
    except Exception:
        return torch.zeros(raw_env.num_envs, dtype=torch.float, device=raw_env.device)


def _foot_label_from_body_name(name: str) -> str | None:
    short_name = str(name).split("/")[-1]
    for label in ("FL", "FR", "RL", "RR"):
        if short_name.startswith(f"{label}_"):
            return label
    return None


def _contact_to_robot_perm(raw_env) -> torch.Tensor:
    cached = getattr(raw_env, "_eval_contact_to_robot_perm", None)
    if isinstance(cached, torch.Tensor):
        return cached

    contact_names = list(getattr(raw_env, "_feet_body_names", []) or [])
    robot_names = list(getattr(raw_env, "_feet_robot_body_names", []) or [])
    device = getattr(raw_env, "device", "cpu")
    perm: list[int] | None = None

    if contact_names and robot_names and len(contact_names) == len(robot_names):
        contact_label_to_pos = {
            label: pos
            for pos, name in enumerate(contact_names)
            if (label := _foot_label_from_body_name(str(name))) is not None
        }
        try:
            perm = [contact_label_to_pos[_foot_label_from_body_name(str(name))] for name in robot_names]
        except (KeyError, TypeError):
            perm = None

    if perm is None:
        perm = list(range(len(robot_names) or len(contact_names)))

    tensor = torch.as_tensor(perm, dtype=torch.long, device=device)
    try:
        raw_env._eval_contact_to_robot_perm = tensor
    except Exception:
        pass
    return tensor


def _foot_reaction_sensors(raw_env) -> list[Any]:
    sensors = getattr(raw_env, "_foot_reaction_contact_sensors", None)
    labels = tuple(getattr(raw_env, "_foot_reaction_contact_sensor_labels", ()) or ())
    if not isinstance(sensors, dict) or not labels:
        return []

    ordered_labels: list[str] = []
    for name in list(getattr(raw_env, "_feet_robot_body_names", []) or []):
        label = _foot_label_from_body_name(str(name))
        if label is not None:
            ordered_labels.append(label)
    if not ordered_labels:
        ordered_labels = list(labels)

    return [sensors[label] for label in ordered_labels if label in sensors]


def _slip_angle_foot_labels(raw_env, num_feet: int) -> list[str]:
    labels = []
    for name in list(getattr(raw_env, "_feet_robot_body_names", []) or []):
        label = _foot_label_from_body_name(str(name))
        if label is not None:
            labels.append(label)

    if len(labels) != num_feet or len(set(labels)) != len(labels):
        reaction_labels = [
            str(label) for label in tuple(getattr(raw_env, "_foot_reaction_contact_sensor_labels", ()) or ())
        ]
        if len(reaction_labels) == num_feet and len(set(reaction_labels)) == len(reaction_labels):
            labels = reaction_labels
        else:
            labels = [f"foot_{idx}" for idx in range(num_feet)]

    return labels[:num_feet]


def _sum_filtered_contact_data(data: torch.Tensor, *, num_envs: int) -> torch.Tensor:
    data = torch.nan_to_num(data[:num_envs], nan=0.0)
    if data.ndim == 4:
        return data.sum(dim=(1, 2))
    if data.ndim == 3:
        return data.sum(dim=1)
    return data.reshape(num_envs, -1, 3).sum(dim=1)


def _contact_reaction_forces_w(raw_env, num_envs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    foot_sensors = _foot_reaction_sensors(raw_env)
    if foot_sensors:
        normal_forces = []
        friction_forces = []
        for sensor in foot_sensors:
            data = sensor.data
            normal = getattr(data, "force_matrix_w", None)
            friction = getattr(data, "friction_forces_w", None)
            if normal is None or friction is None:
                return None
            normal_forces.append(_sum_filtered_contact_data(normal, num_envs=num_envs))
            friction_forces.append(_sum_filtered_contact_data(friction, num_envs=num_envs))
        normal_forces_w = torch.stack(normal_forces, dim=1)
        friction_forces_w = torch.stack(friction_forces, dim=1)
        return normal_forces_w + friction_forces_w, normal_forces_w, friction_forces_w

    try:
        data = raw_env._contact_sensor.data
        normal_forces_w = getattr(data, "force_matrix_w", None)
        friction_forces_w = getattr(data, "friction_forces_w", None)
        if normal_forces_w is None or friction_forces_w is None:
            return None
        body_ids = raw_env._feet_body_ids
        normal_forces_w = torch.nan_to_num(normal_forces_w[:num_envs, body_ids, :, :], nan=0.0).sum(dim=2)
        friction_forces_w = torch.nan_to_num(friction_forces_w[:num_envs, body_ids, :, :], nan=0.0).sum(dim=2)
        perm = _contact_to_robot_perm(raw_env)
        normal_forces_w = normal_forces_w[:, perm, :]
        friction_forces_w = friction_forces_w[:, perm, :]
        return normal_forces_w + friction_forces_w, normal_forces_w, friction_forces_w
    except Exception:
        return None


def _foot_reference_positions_w(raw_env) -> torch.Tensor:
    foot_pos_w = raw_env._get_foot_positions_w()
    foot_sensors = _foot_reaction_sensors(raw_env)
    if not foot_sensors:
        return foot_pos_w

    contact_positions = []
    valid_contacts = []
    num_envs = foot_pos_w.shape[0]
    for sensor in foot_sensors:
        contact_pos_w = getattr(sensor.data, "contact_pos_w", None)
        if contact_pos_w is None:
            return foot_pos_w
        contact_pos_w = contact_pos_w[:num_envs]
        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        count = valid.to(dtype=contact_pos_w.dtype).sum(dim=(1, 2))
        summed = torch.where(valid[..., None], contact_pos_w, torch.zeros_like(contact_pos_w)).sum(dim=(1, 2))
        contact_positions.append(summed / count.clamp_min(1.0).unsqueeze(-1))
        valid_contacts.append(count > 0)

    contact_positions_w = torch.stack(contact_positions, dim=1)
    valid_contact = torch.stack(valid_contacts, dim=1)
    return torch.where(valid_contact[..., None], contact_positions_w, foot_pos_w)


def _foot_friction_values(raw_env, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_feet = len(getattr(raw_env, "_feet_body_ids", []))
    try:
        static_default = float(getattr(raw_env.cfg, "friction_static_range")[0])
    except Exception:
        static_default = float(getattr(raw_env.cfg, "gt_obs_default_mu", 1.0))
    try:
        dynamic_default = static_default * float(getattr(raw_env.cfg, "mu_dynamic_static_ratio"))
    except Exception:
        dynamic_default = static_default

    default_static = torch.full((num_envs, num_feet), static_default, device=raw_env.device)
    default_dynamic = torch.full((num_envs, num_feet), dynamic_default, device=raw_env.device)
    if (
        getattr(raw_env, "_patch_friction_static", torch.empty(0)).numel() == 0
        or getattr(raw_env, "_patch_xy_min", torch.empty(0)).numel() == 0
    ):
        return default_static, default_dynamic

    foot_xy = _foot_reference_positions_w(raw_env)[:num_envs, :, :2] - raw_env.scene.env_origins[:num_envs, None, :2]
    inside_patch = torch.logical_and(
        foot_xy[:, :, None, :] >= raw_env._patch_xy_min[None, None, :, :],
        foot_xy[:, :, None, :] <= raw_env._patch_xy_max[None, None, :, :],
    ).all(dim=-1)
    has_patch = inside_patch.any(dim=-1)
    patch_ids = torch.argmax(inside_patch.to(torch.long), dim=-1)
    static_mu = raw_env._patch_friction_static[:num_envs].gather(dim=1, index=patch_ids)
    dynamic_mu = raw_env._patch_friction_dynamic[:num_envs].gather(dim=1, index=patch_ids)
    return torch.where(has_patch, static_mu, default_static), torch.where(has_patch, dynamic_mu, default_dynamic)


def _slip_angle_contact_counts(raw_env, num_envs: int, delta_deg: float) -> dict[str, torch.Tensor] | None:
    contact_reaction = _contact_reaction_forces_w(raw_env, num_envs)
    if contact_reaction is None:
        return None

    contact_forces_w, normal_forces_w, _ = contact_reaction
    contact_mask = torch.linalg.norm(normal_forces_w, dim=-1) > float(getattr(raw_env.cfg, "base_contact_threshold", 1.0))

    horizontal_force = torch.linalg.norm(contact_forces_w[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces_w[..., 2])
    angle_deg = torch.rad2deg(torch.atan2(horizontal_force, torch.clamp(vertical_force, min=1.0e-6)))
    static_mu, dynamic_mu = _foot_friction_values(raw_env, num_envs)
    angle_static_deg = torch.rad2deg(torch.atan(torch.clamp(static_mu, min=0.0)))
    angle_dynamic_deg = torch.rad2deg(torch.atan(torch.clamp(dynamic_mu, min=0.0)))
    delta = float(delta_deg)

    below_dynamic = contact_mask & (angle_deg >= 0.0) & (angle_deg <= angle_dynamic_deg - delta)
    slipping = contact_mask & (angle_deg > angle_dynamic_deg - delta) & (angle_deg <= angle_dynamic_deg + delta)
    dynamic_to_static = contact_mask & (angle_deg > angle_dynamic_deg + delta) & (angle_deg <= angle_static_deg)
    above_static = contact_mask & (angle_deg > angle_static_deg)

    return {
        "contact": contact_mask.sum(dim=1),
        "below_dynamic": below_dynamic.sum(dim=1),
        "slipping": slipping.sum(dim=1),
        "dynamic_to_static": dynamic_to_static.sum(dim=1),
        "above_static": above_static.sum(dim=1),
        "contact_by_foot": contact_mask.to(dtype=torch.long),
        "below_dynamic_by_foot": below_dynamic.to(dtype=torch.long),
        "slipping_by_foot": slipping.to(dtype=torch.long),
        "dynamic_to_static_by_foot": dynamic_to_static.to(dtype=torch.long),
        "above_static_by_foot": above_static.to(dtype=torch.long),
    }


def _pct_from_counts(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * float(numerator) / float(denominator)


def _slip_angle_contact_time_summary(
    episodes: list[dict[str, Any]],
    delta_deg: float,
    foot_labels: list[str] | None = None,
) -> dict[str, Any]:
    contact = int(sum(int(ep.get("slip_angle_contact_samples", 0)) for ep in episodes))
    below_dynamic = int(sum(int(ep.get("slip_angle_below_dynamic_minus_delta_samples", 0)) for ep in episodes))
    slipping = int(sum(int(ep.get("slip_angle_slipping_samples", 0)) for ep in episodes))
    dynamic_to_static = int(sum(int(ep.get("slip_angle_dynamic_plus_delta_to_static_samples", 0)) for ep in episodes))
    above_static = int(sum(int(ep.get("slip_angle_above_static_samples", 0)) for ep in episodes))
    summary = {
        "delta_deg": float(delta_deg),
        "contact_samples": contact,
        "below_dynamic_minus_delta_samples": below_dynamic,
        "slipping_samples": slipping,
        "dynamic_plus_delta_to_static_samples": dynamic_to_static,
        "above_static_samples": above_static,
        "below_dynamic_minus_delta_pct": _pct_from_counts(below_dynamic, contact),
        "slipping_pct": _pct_from_counts(slipping, contact),
        "dynamic_plus_delta_to_static_pct": _pct_from_counts(dynamic_to_static, contact),
        "above_static_pct": _pct_from_counts(above_static, contact),
    }
    for label in foot_labels or []:
        prefix = f"slip_angle_{label}_"
        foot_contact = int(sum(int(ep.get(f"{prefix}contact_samples", 0)) for ep in episodes))
        foot_below_dynamic = int(
            sum(int(ep.get(f"{prefix}below_dynamic_minus_delta_samples", 0)) for ep in episodes)
        )
        foot_slipping = int(sum(int(ep.get(f"{prefix}slipping_samples", 0)) for ep in episodes))
        foot_dynamic_to_static = int(
            sum(int(ep.get(f"{prefix}dynamic_plus_delta_to_static_samples", 0)) for ep in episodes)
        )
        foot_above_static = int(sum(int(ep.get(f"{prefix}above_static_samples", 0)) for ep in episodes))
        summary.update(
            {
                f"{label}_contact_samples": foot_contact,
                f"{label}_below_dynamic_minus_delta_samples": foot_below_dynamic,
                f"{label}_slipping_samples": foot_slipping,
                f"{label}_dynamic_plus_delta_to_static_samples": foot_dynamic_to_static,
                f"{label}_above_static_samples": foot_above_static,
                f"{label}_contact_time_share_pct": _pct_from_counts(foot_contact, contact),
                f"{label}_below_dynamic_minus_delta_pct": _pct_from_counts(foot_below_dynamic, foot_contact),
                f"{label}_slipping_pct": _pct_from_counts(foot_slipping, foot_contact),
                f"{label}_dynamic_plus_delta_to_static_pct": _pct_from_counts(
                    foot_dynamic_to_static,
                    foot_contact,
                ),
                f"{label}_above_static_pct": _pct_from_counts(foot_above_static, foot_contact),
            }
        )
    return summary


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    sorted_values = sorted(float(v) for v in values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "median": _percentile(sorted_values, 0.50),
        "p05": _percentile(sorted_values, 0.05),
        "p95": _percentile(sorted_values, 0.95),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def _safe_wandb_name(text: str, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w .:=+@|/\\[\](),-]", "-", text)
    return text[:max_len].strip(" -_") or "solo-race-eval"


def _output_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".json.gz"):
        return name[: -len(".json.gz")]
    return path.stem


def _default_wandb_name(result: dict[str, Any], output_path: Path) -> str:
    checkpoint_stem = Path(str(result["checkpoint"])).stem
    task = str(result["task"]).replace("Isaac-", "")
    seed = result.get("seed")
    finish_ratio = result.get("finish_ratio")
    if isinstance(finish_ratio, (int, float)) and math.isfinite(float(finish_ratio)):
        score = f"finish={float(finish_ratio):.3f}"
    else:
        score = "finish=nan"
    return _safe_wandb_name(f"{checkpoint_stem} | {task} | seed={seed} | {score} | {_output_stem(output_path)}")


def _write_eval_result(result: dict[str, Any], output_path: Path) -> None:
    if output_path.name.endswith(".json.gz"):
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        return

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _flatten_numeric_metrics(data: Any, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "episodes":
                continue
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(_flatten_numeric_metrics(value, child_prefix))
    elif isinstance(data, bool):
        metrics[prefix] = float(data)
    elif isinstance(data, (int, float)) and math.isfinite(float(data)):
        metrics[prefix] = float(data)
    return metrics


def _log_eval_to_wandb(result: dict[str, Any], output_path: Path) -> None:
    if args_cli.no_wandb:
        return

    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] Could not import wandb; skipping W&B upload: {exc}", flush=True)
        return

    run_name = args_cli.wandb_name or _default_wandb_name(result, output_path)
    config = {key: value for key, value in result.items() if key != "episodes"}
    metrics = _flatten_numeric_metrics(result, prefix="eval")
    episodes = result.get("episodes", [])
    finish_times = [float(ep["finish_time_seconds"]) for ep in episodes if ep.get("finish_time_seconds") is not None]
    gate_progress = [float(ep["gate_progress_ratio"]) for ep in episodes if ep.get("gate_progress_ratio") is not None]

    try:
        run = wandb.init(project=args_cli.wandb_project, name=run_name, config=config)
        run.log(metrics, step=0)
        if finish_times:
            run.log({"eval/finish_time_seconds_success_only/hist": wandb.Histogram(finish_times)}, step=0)
        if gate_progress:
            run.log({"eval/gate_progress_ratio/hist": wandb.Histogram(gate_progress)}, step=0)
        run.summary.update(metrics)
        artifact_name = _safe_wandb_name(_output_stem(output_path)).replace(" ", "_")
        artifact = wandb.Artifact(artifact_name, type="solo_race_eval")
        artifact.add_file(str(output_path))
        run.log_artifact(artifact)
        wandb.save(str(output_path), base_path=str(output_path.parent), policy="now")
        print(f"[INFO] Uploaded eval metrics/artifact to W&B project '{args_cli.wandb_project}' run '{run.name}'.", flush=True)
    except Exception as exc:
        print(f"[WARN] W&B upload failed; JSON is still available at {output_path}: {exc}", flush=True)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def _load_policy(
    vec_env,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    dagger_adapter_checkpoint: dict[str, Any] | None = None,
):
    if dagger_adapter_checkpoint is not None:
        policy, teacher_path = load_dagger_latent_policy(
            adapter_checkpoint_path=resume_path,
            adapter_checkpoint=dagger_adapter_checkpoint,
            teacher_checkpoint_path=args_cli.dagger_teacher_checkpoint,
            policy_cfg=agent_cfg.policy,
            num_actions=vec_env.num_actions,
            device=torch.device(vec_env.unwrapped.device),
            teacher_shared_networks=bool(args_cli.dagger_teacher_shared_networks),
        )
        print(f"[INFO] Loaded DAgger adapter checkpoint from: {resume_path}", flush=True)
        print(f"[INFO] Loaded frozen teacher checkpoint from: {teacher_path}", flush=True)
        return None, policy

    apply_checkpoint_architecture_to_policy_cfg(agent_cfg.policy, resume_path)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading model checkpoint from: {resume_path}", flush=True)
    runner.load(resume_path)
    return runner, runner.get_inference_policy(device=vec_env.unwrapped.device)


def _evaluate_rollouts(
    eval_n: int,
    num_envs: int,
    env_cfg,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
    dagger_adapter_checkpoint: dict[str, Any] | None = None,
) -> dict:
    """Collect eval_n completed episodes from one continuously-running vectorized Isaac env.

    This mirrors the training execution pattern better than launching one simulator per batch: all envs keep running and
    Isaac auto-resets completed envs while we record one episode result per reset.
    """

    env_cfg.scene.num_envs = int(num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    runner, policy = _load_policy(vec_env, agent_cfg, resume_path, dagger_adapter_checkpoint)
    obs = vec_env.get_observations()

    device = raw_env.device
    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    floor_events = torch.zeros(num_envs, dtype=torch.long, device=device)
    pillar_events = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_contact_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_below_dynamic_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_slipping_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_dynamic_to_static_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_above_static_samples = torch.zeros(num_envs, dtype=torch.long, device=device)
    slip_feet_ids = getattr(raw_env, "_feet_robot_body_ids", None)
    if slip_feet_ids is None or len(slip_feet_ids) == 0:
        slip_feet_ids = getattr(raw_env, "_feet_body_ids", [])
    num_slip_feet = len(slip_feet_ids)
    if num_slip_feet == 0:
        num_slip_feet = len(tuple(getattr(raw_env, "_foot_reaction_contact_sensor_labels", ()) or ()))
    slip_foot_labels = _slip_angle_foot_labels(raw_env, num_slip_feet)
    slip_contact_samples_by_foot = torch.zeros((num_envs, num_slip_feet), dtype=torch.long, device=device)
    slip_below_dynamic_samples_by_foot = torch.zeros((num_envs, num_slip_feet), dtype=torch.long, device=device)
    slip_slipping_samples_by_foot = torch.zeros((num_envs, num_slip_feet), dtype=torch.long, device=device)
    slip_dynamic_to_static_samples_by_foot = torch.zeros((num_envs, num_slip_feet), dtype=torch.long, device=device)
    slip_above_static_samples_by_foot = torch.zeros((num_envs, num_slip_feet), dtype=torch.long, device=device)
    prev_floor = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_pillar = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_thigh = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes: list[dict[str, Any]] = []
    warned_slip_angle_metrics = False

    scene = getattr(raw_env, "scene", None)
    scene_update_original = getattr(scene, "update", None)
    if callable(scene_update_original):
        try:
            sim_dt = float(getattr(raw_env.cfg.sim, "dt", raw_env.step_dt))
        except Exception:
            sim_dt = float(raw_env.step_dt)

        def _wrapped_scene_update(*args, **kwargs):
            nonlocal warned_slip_angle_metrics
            result = scene_update_original(*args, **kwargs)
            slip_counts = _slip_angle_contact_counts(raw_env, num_envs, float(args_cli.slip_angle_delta_deg))
            if slip_counts is None:
                if not warned_slip_angle_metrics:
                    print(
                        "[WARN] Slip-angle contact metrics are unavailable because the contact sensor did not expose "
                        "foot contact-force data.",
                        flush=True,
                    )
                    warned_slip_angle_metrics = True
            else:
                slip_contact_samples[:] += slip_counts["contact"].to(dtype=torch.long)
                slip_below_dynamic_samples[:] += slip_counts["below_dynamic"].to(dtype=torch.long)
                slip_slipping_samples[:] += slip_counts["slipping"].to(dtype=torch.long)
                slip_dynamic_to_static_samples[:] += slip_counts["dynamic_to_static"].to(dtype=torch.long)
                slip_above_static_samples[:] += slip_counts["above_static"].to(dtype=torch.long)
                slip_contact_samples_by_foot[:] += slip_counts["contact_by_foot"].to(dtype=torch.long)
                slip_below_dynamic_samples_by_foot[:] += slip_counts["below_dynamic_by_foot"].to(dtype=torch.long)
                slip_slipping_samples_by_foot[:] += slip_counts["slipping_by_foot"].to(dtype=torch.long)
                slip_dynamic_to_static_samples_by_foot[:] += slip_counts["dynamic_to_static_by_foot"].to(
                    dtype=torch.long
                )
                slip_above_static_samples_by_foot[:] += slip_counts["above_static_by_foot"].to(dtype=torch.long)
            return result

        scene.update = _wrapped_scene_update
        rate = (1.0 / sim_dt) if sim_dt > 0.0 else 0.0
        print(
            f"[INFO] Slip-angle contact metrics: sampling per scene.update at ~{rate:.0f} Hz with "
            f"delta={float(args_cli.slip_angle_delta_deg):g} deg.",
            flush=True,
        )
    else:
        print("[WARN] Slip-angle contact metrics unavailable: scene.update is not callable.", flush=True)

    # If eval_n < num_envs, record a fixed subset of env ids rather than the first eval_n resets.
    # The latter is biased toward the fastest episodes.
    if eval_n <= num_envs:
        eval_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        eval_env_mask[:eval_n] = True
        recorded_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    else:
        eval_env_mask = None
        recorded_env_mask = None

    max_steps = int(raw_env.max_episode_length)
    # If every env times out, eval_n episodes need ceil(eval_n / num_envs) full horizons. Add one horizon for safety.
    max_total_steps = max_steps * (math.ceil(eval_n / num_envs) + 1) + 5
    eval_t0 = time.time()
    next_progress_t = eval_t0 + max(float(args_cli.progress_interval_s), 0.0)
    last_step = 0

    for step_idx in range(1, max_total_steps + 1):
        if len(episodes) >= eval_n:
            break

        progress_before_step = _gate_progress_ratio(raw_env)[:num_envs].clone()

        # Count non-terminal collision events from the current sensor state. Terminal floor collisions are also counted
        # from the reset cause below, because Isaac resets completed envs before the script can inspect their sensors again.
        floor_now = _floor_collision_now(raw_env)[:num_envs]
        pillar_now = _pillar_collision_now(raw_env)[:num_envs]
        thigh_now = _thigh_contact_now(raw_env)[:num_envs]
        floor_rising = floor_now & ~prev_floor
        pillar_rising = (pillar_now & ~prev_pillar) | (thigh_now & ~prev_thigh)
        floor_events[floor_rising] += 1
        pillar_events[pillar_rising] += 1
        prev_floor = floor_now
        prev_pillar = pillar_now
        prev_thigh = thigh_now

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            if runner is not None:
                try:
                    runner.alg.actor_critic.reset(dones)
                except Exception:
                    pass
            try:
                policy.actor_critic.reset(dones)
            except Exception:
                pass

        episode_steps += 1
        last_step = step_idx

        done = _to_bool_tensor(dones, device, num_envs)
        if not bool(torch.any(done)):
            _print_progress_if_due(
                step_idx=step_idx,
                total_steps=max_total_steps,
                episodes=episodes,
                eval_n=eval_n,
                num_envs=num_envs,
                eval_t0=eval_t0,
                next_progress_t_ref=[next_progress_t],
                active_progress=_gate_progress_ratio(raw_env)[:num_envs],
            )
            next_progress_t = _print_progress_if_due.next_progress_t
            continue

        terminated = _safe_bool_metric(raw_env, "reset_terminated")[:num_envs]
        timed_out = _safe_bool_metric(raw_env, "reset_time_outs")[:num_envs]
        # Finishing is detected one step after the final gate is passed, so the pre-step progress is the reliable signal.
        finished = done & (progress_before_step >= 1.0)
        floor_terminal = done & terminated & ~finished & ~timed_out
        floor_events[floor_terminal] += 1

        record_done = done
        if eval_env_mask is not None and recorded_env_mask is not None:
            record_done = done & eval_env_mask & ~recorded_env_mask

        done_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
        for env_id in record_done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if len(episodes) >= eval_n:
                break
            steps = int(episode_steps[env_id].item())
            sec = steps * float(raw_env.step_dt)
            did_finish = bool(finished[env_id].item())
            did_timeout = bool(timed_out[env_id].item()) and not did_finish
            did_floor_terminal = bool(floor_terminal[env_id].item()) and not did_finish
            slip_contact = int(slip_contact_samples[env_id].item())
            slip_below_dynamic = int(slip_below_dynamic_samples[env_id].item())
            slip_slipping = int(slip_slipping_samples[env_id].item())
            slip_dynamic_to_static = int(slip_dynamic_to_static_samples[env_id].item())
            slip_above_static = int(slip_above_static_samples[env_id].item())
            slip_by_foot: dict[str, int | float | None] = {}
            for foot_idx, foot_label in enumerate(slip_foot_labels):
                foot_contact = int(slip_contact_samples_by_foot[env_id, foot_idx].item())
                foot_below_dynamic = int(slip_below_dynamic_samples_by_foot[env_id, foot_idx].item())
                foot_slipping = int(slip_slipping_samples_by_foot[env_id, foot_idx].item())
                foot_dynamic_to_static = int(slip_dynamic_to_static_samples_by_foot[env_id, foot_idx].item())
                foot_above_static = int(slip_above_static_samples_by_foot[env_id, foot_idx].item())
                prefix = f"slip_angle_{foot_label}_"
                slip_by_foot.update(
                    {
                        f"{prefix}contact_samples": foot_contact,
                        f"{prefix}below_dynamic_minus_delta_samples": foot_below_dynamic,
                        f"{prefix}slipping_samples": foot_slipping,
                        f"{prefix}dynamic_plus_delta_to_static_samples": foot_dynamic_to_static,
                        f"{prefix}above_static_samples": foot_above_static,
                        f"{prefix}contact_time_share_pct": _pct_from_counts(foot_contact, slip_contact),
                        f"{prefix}below_dynamic_minus_delta_pct": _pct_from_counts(
                            foot_below_dynamic,
                            foot_contact,
                        ),
                        f"{prefix}slipping_pct": _pct_from_counts(foot_slipping, foot_contact),
                        f"{prefix}dynamic_plus_delta_to_static_pct": _pct_from_counts(
                            foot_dynamic_to_static,
                            foot_contact,
                        ),
                        f"{prefix}above_static_pct": _pct_from_counts(foot_above_static, foot_contact),
                    }
                )
            episodes.append(
                {
                    "env_id": int(env_id),
                    "finished": did_finish,
                    "timed_out": did_timeout,
                    "floor_collision_terminal": did_floor_terminal,
                    "steps": steps,
                    "seconds": sec,
                    "finish_time_steps": steps if did_finish else None,
                    "finish_time_seconds": sec if did_finish else None,
                    "gate_progress_ratio": float(progress_before_step[env_id].item()),
                    "floor_collision_events": int(floor_events[env_id].item()),
                    "pillar_collision_events": int(pillar_events[env_id].item()),
                    "slip_angle_contact_samples": slip_contact,
                    "slip_angle_below_dynamic_minus_delta_samples": slip_below_dynamic,
                    "slip_angle_slipping_samples": slip_slipping,
                    "slip_angle_dynamic_plus_delta_to_static_samples": slip_dynamic_to_static,
                    "slip_angle_above_static_samples": slip_above_static,
                    "slip_angle_below_dynamic_minus_delta_pct": _pct_from_counts(slip_below_dynamic, slip_contact),
                    "slip_angle_slipping_pct": _pct_from_counts(slip_slipping, slip_contact),
                    "slip_angle_dynamic_plus_delta_to_static_pct": _pct_from_counts(
                        slip_dynamic_to_static,
                        slip_contact,
                    ),
                    "slip_angle_above_static_pct": _pct_from_counts(slip_above_static, slip_contact),
                    **slip_by_foot,
                }
            )
            if recorded_env_mask is not None:
                recorded_env_mask[env_id] = True

        # Isaac already reset these envs internally; reset this script's per-episode counters to match.
        done_tensor_ids = done.nonzero(as_tuple=False).squeeze(-1)
        episode_steps[done_tensor_ids] = 0
        floor_events[done_tensor_ids] = 0
        pillar_events[done_tensor_ids] = 0
        slip_contact_samples[done_tensor_ids] = 0
        slip_below_dynamic_samples[done_tensor_ids] = 0
        slip_slipping_samples[done_tensor_ids] = 0
        slip_dynamic_to_static_samples[done_tensor_ids] = 0
        slip_above_static_samples[done_tensor_ids] = 0
        slip_contact_samples_by_foot[done_tensor_ids] = 0
        slip_below_dynamic_samples_by_foot[done_tensor_ids] = 0
        slip_slipping_samples_by_foot[done_tensor_ids] = 0
        slip_dynamic_to_static_samples_by_foot[done_tensor_ids] = 0
        slip_above_static_samples_by_foot[done_tensor_ids] = 0
        prev_floor[done_tensor_ids] = False
        prev_pillar[done_tensor_ids] = False
        prev_thigh[done_tensor_ids] = False

        _print_progress_if_due(
            step_idx=step_idx,
            total_steps=max_total_steps,
            episodes=episodes,
            eval_n=eval_n,
            num_envs=num_envs,
            eval_t0=eval_t0,
            next_progress_t_ref=[next_progress_t],
            active_progress=_gate_progress_ratio(raw_env)[:num_envs],
        )
        next_progress_t = _print_progress_if_due.next_progress_t

    if len(episodes) < eval_n:
        print(
            f"[WARN] Evaluation stopped after safety limit {max_total_steps} steps with "
            f"{len(episodes)}/{eval_n} episodes collected.",
            flush=True,
        )

    elapsed = max(time.time() - eval_t0, 1e-6)
    print(
        "[INFO] "
        f"Finished evaluation: episodes={len(episodes)}/{eval_n}, "
        f"finished={sum(bool(ep.get('finished', False)) for ep in episodes)}, "
        f"sim_steps={last_step}, elapsed={elapsed:.1f}s, eps_per_s={len(episodes) / elapsed:.2f}",
        flush=True,
    )

    step_dt = float(raw_env.step_dt)
    if callable(scene_update_original):
        try:
            scene.update = scene_update_original
        except Exception:
            pass
    env.close()
    return {
        "episodes": episodes,
        "step_dt": step_dt,
        "max_episode_length": max_steps,
        "slip_angle_foot_labels": slip_foot_labels,
    }


def _print_progress_if_due(
    *,
    step_idx: int,
    total_steps: int,
    episodes: list[dict[str, Any]],
    eval_n: int,
    num_envs: int,
    eval_t0: float,
    next_progress_t_ref: list[float],
    active_progress: torch.Tensor,
):
    progress_interval_s = float(args_cli.progress_interval_s)
    now = time.time()
    next_progress_t = next_progress_t_ref[0]
    if progress_interval_s <= 0.0 or (now < next_progress_t and len(episodes) < eval_n):
        _print_progress_if_due.next_progress_t = next_progress_t
        return

    finished_count = sum(bool(ep.get("finished", False)) for ep in episodes)
    timeout_count = sum(bool(ep.get("timed_out", False)) for ep in episodes)
    floor_terminal_count = sum(bool(ep.get("floor_collision_terminal", False)) for ep in episodes)
    mean_progress = float(active_progress.mean().item()) if active_progress.numel() > 0 else 0.0
    elapsed = max(now - eval_t0, 1e-6)
    completed_count = len(episodes)
    print(
        "[PROGRESS] "
        f"step={step_idx}/{total_steps} envs={num_envs} "
        f"episodes={completed_count}/{eval_n} "
        f"finished={finished_count} timeout={timeout_count} floor_term={floor_terminal_count} "
        f"gate_progress_mean={mean_progress:.3f} "
        f"elapsed={elapsed:.1f}s eps_per_s={completed_count / elapsed:.2f}",
        flush=True,
    )
    _print_progress_if_due.next_progress_t = now + progress_interval_s


_print_progress_if_due.next_progress_t = 0.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    resume_path = os.path.abspath(args_cli.checkpoint)

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    if args_cli.max_steps is not None:
        step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
        env_cfg.episode_length_s = float(args_cli.max_steps) * step_dt
    elif args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    eval_friction_seed = args_cli.friction_seed if args_cli.friction_seed is not None else agent_cfg.seed
    if eval_friction_seed is not None:
        if hasattr(env_cfg, "friction_seed"):
            env_cfg.friction_seed = int(eval_friction_seed)
            print(f"[INFO] Using race patch-friction seed: {env_cfg.friction_seed}", flush=True)
        elif args_cli.friction_seed is not None:
            print("[WARN] --friction-seed was provided, but this task config has no friction_seed field.", flush=True)

    dagger_adapter_checkpoint = load_dagger_adapter_checkpoint(resume_path)
    teacher_checkpoint_for_config = None
    if dagger_adapter_checkpoint is not None:
        teacher_checkpoint_for_config = args_cli.dagger_teacher_checkpoint or dagger_adapter_checkpoint.get("teacher_checkpoint")

    # For DAgger adapters, prefer the frozen teacher's race-friction config when available: the teacher checkpoint
    # defines the policy family being adapted, and explicit CLI flags below can still override it.
    inferred_config, inferred_sources = _infer_eval_friction_config(resume_path, teacher_checkpoint_for_config)
    _apply_eval_friction_config(env_cfg, inferred_config, inferred_sources, label="training env.yaml")
    explicit_overrides = _explicit_eval_friction_overrides()
    _apply_eval_friction_config(
        env_cfg,
        explicit_overrides,
        {key: "command line" for key in explicit_overrides},
        label="command line",
    )

    if not args_cli.keep_training_stochasticity:
        disabled = _disable_training_stochasticity_for_eval(
            env_cfg,
            preserve_patch_friction_randomization=bool(getattr(env_cfg, "randomize_fric_coefs", False)),
        )
        if disabled:
            print("[INFO] Disabled training-time stochasticity for eval: " + ", ".join(disabled), flush=True)

    if dagger_adapter_checkpoint is not None:
        configure_env_cfg_for_dagger_adapter(env_cfg, dagger_adapter_checkpoint)
        layout = dagger_adapter_checkpoint.get("layout", {})
        print(
            "[INFO] Detected DAgger adapter checkpoint: "
            f"layout={layout.get('kind')} T={layout.get('history_len')} D={layout.get('history_dim')}",
            flush=True,
        )

    eval_n = int(args_cli.eval_n_parallel if args_cli.eval_n is None else args_cli.eval_n)
    eval_n_parallel = min(int(args_cli.eval_n_parallel), eval_n)
    out_dir = Path(args_cli.out_dir) if args_cli.out_dir is not None else Path(os.path.dirname(os.path.dirname(resume_path))) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(env_cfg, "contact_sensor"):
        env_cfg.contact_sensor.track_contact_points = True
        env_cfg.contact_sensor.track_friction_forces = True
        env_cfg.contact_sensor.max_contact_data_count_per_prim = max(
            int(getattr(env_cfg.contact_sensor, "max_contact_data_count_per_prim", 4)),
            8,
        )

    print(
        f"[INFO] Evaluating {eval_n} episodes with one vectorized env of {eval_n_parallel} parallel envs; "
        f"max_steps={args_cli.max_steps}, episode_length_s={env_cfg.episode_length_s}",
        flush=True,
    )

    eval_result = _evaluate_rollouts(eval_n, eval_n_parallel, env_cfg, agent_cfg, resume_path, dagger_adapter_checkpoint)
    all_episodes = eval_result["episodes"]
    step_dt = eval_result["step_dt"]
    max_episode_length = eval_result["max_episode_length"]
    slip_angle_foot_labels = eval_result.get("slip_angle_foot_labels", [])

    finished = [ep for ep in all_episodes if ep.get("finished")]
    finish_steps = [float(ep["finish_time_steps"]) for ep in finished if ep.get("finish_time_steps") is not None]
    finish_seconds = [float(ep["finish_time_seconds"]) for ep in finished if ep.get("finish_time_seconds") is not None]
    # Success-only times answer "how fast when it completes?". Penalized times answer "how good is this policy overall?".
    # For model selection/reporting, use the penalized metric: failures receive the episode timeout.
    penalized_steps = [
        float(ep["steps"] if ep.get("finished") else max_episode_length)
        for ep in all_episodes
    ]
    penalized_seconds = [steps * step_dt for steps in penalized_steps]

    result = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "dagger_adapter": dagger_adapter_checkpoint is not None,
        "dagger_teacher_checkpoint": str(dagger_adapter_checkpoint.get("teacher_checkpoint"))
        if dagger_adapter_checkpoint is not None
        else None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_n": eval_n,
        "eval_n_parallel": eval_n_parallel,
        "seed": agent_cfg.seed,
        "friction_seed": getattr(env_cfg, "friction_seed", None),
        "randomize_fric_coefs": getattr(env_cfg, "randomize_fric_coefs", None),
        "randomize_friction_bucket_values": getattr(env_cfg, "randomize_friction_bucket_values", None),
        "group_all_patches_single_bucket": getattr(env_cfg, "group_all_patches_single_bucket", None),
        "within_episode_fric_resample": getattr(env_cfg, "within_episode_fric_resample", None),
        "within_episode_fric_resample_time_range": getattr(env_cfg, "within_episode_fric_resample_time_range", None),
        "requested_max_steps": args_cli.max_steps,
        "episode_length_s": float(env_cfg.episode_length_s),
        "step_dt": step_dt,
        "max_episode_length": max_episode_length,
        "num_episodes": len(all_episodes),
        "num_finished": len(finished),
        "finish_ratio": (len(finished) / len(all_episodes)) if all_episodes else math.nan,
        "finish_time_steps": _summarize(finish_steps),
        "finish_time_seconds": _summarize(finish_seconds),
        "finish_time_steps_success_only": _summarize(finish_steps),
        "finish_time_seconds_success_only": _summarize(finish_seconds),
        "finish_time_steps_penalized": _summarize(penalized_steps),
        "finish_time_seconds_penalized": _summarize(penalized_seconds),
        "gate_progress_ratio": _summarize([float(ep["gate_progress_ratio"]) for ep in all_episodes]),
        "collisions": {
            "floor_events_total": int(sum(int(ep.get("floor_collision_events", 0)) for ep in all_episodes)),
            "pillar_events_total": int(sum(int(ep.get("pillar_collision_events", 0)) for ep in all_episodes)),
            "episodes_with_floor_collision": int(sum(int(ep.get("floor_collision_events", 0)) > 0 for ep in all_episodes)),
            "episodes_with_pillar_collision": int(sum(int(ep.get("pillar_collision_events", 0)) > 0 for ep in all_episodes)),
            "floor_collision_terminal_episodes": int(sum(bool(ep.get("floor_collision_terminal", False)) for ep in all_episodes)),
        },
        "slip_angle_contact_time": _slip_angle_contact_time_summary(
            all_episodes,
            float(args_cli.slip_angle_delta_deg),
            list(slip_angle_foot_labels),
        ),
        "episodes": all_episodes,
    }

    checkpoint_stem = Path(resume_path).stem
    output_suffix = ".json.gz" if args_cli.eval_output_format == "json.gz" else ".json"
    output_path = out_dir / f"eval_stats_{checkpoint_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_suffix}"
    _write_eval_result(result, output_path)
    print(f"[INFO] Wrote eval stats: {output_path}", flush=True)
    _log_eval_to_wandb(result, output_path)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "seed",
                    "friction_seed",
                    "randomize_fric_coefs",
                    "randomize_friction_bucket_values",
                    "group_all_patches_single_bucket",
                    "within_episode_fric_resample",
                    "within_episode_fric_resample_time_range",
                    "num_episodes",
                    "num_finished",
                    "finish_ratio",
                    "finish_time_steps",
                    "finish_time_seconds",
                    "finish_time_steps_penalized",
                    "finish_time_seconds_penalized",
                    "collisions",
                    "slip_angle_contact_time",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
