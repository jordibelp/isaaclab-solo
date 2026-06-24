# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare two trained Solo12 race policies on plasticity metrics.

This evaluates two checkpoints (e.g. a vanilla-backprop teacher vs a CBP teacher) under the
*same* fixed friction/eval condition and reports, for each:

  (a) race finish time         - how fast the race is completed (success-only and penalized).
  (b) % of dormant units       - the dormant-neuron metric of Sokar et al. 2023
                                 ("The Dormant Neuron Phenomenon in Deep RL"): a unit i in a
                                 layer is tau-dormant if  s_i = E|h_i| / mean_j E|h_j|  <= tau.
                                 tau=0 reproduces the Nature CBP paper's "dead units" count;
                                 for the ELU actor here the informative thresholds are 0.025/0.1.
  (c) rank of the representation - both the *stable rank*  (sum sigma_i^2 / sigma_1^2) and the
                                 *effective rank*  exp(-sum p_i log p_i), p_i = sigma_i/sum sigma,
                                 of the post-activation feature matrix. The Nature CBP paper
                                 (https://www.nature.com/articles/s41586-024-07711-7) plots the
                                 effective rank; "stable rank" is reported alongside it.

Hidden features are captured *post-activation* using the same helper continual backprop uses
(continual_backprop.cbp_specs_for_sequential_mlp), which hooks the unique Linear layer and
re-applies the activation. This is robust to the reused ELU module instance in rsl_rl MLPs.

Example (matches the friction condition in play_direct_race_0423.py):

./isaaclab.sh -p source/scripts/rsl_rl/solo_race_plasticity_eval.py \
  --task="Solo12-Race-ParamsConditionedEnc-Direct-v0" \
  --checkpoint-a "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0620_e2p2ooga_teacher_model_45800.pt" \
  --checkpoint-b "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0620_teacher_diqtj6fh_model_46900.pt" \
  --label-a vanilla --label-b cbp \
  --eval_n 64 --eval_n_parallel 64 --max_steps 3000 --seed 271828 \
  env.mu_dynamic_static_ratio=0.20 "env.friction_static_range=[1.5,1.5]" \
  env.enable_observation_corruption=False \
  env.physics_dt=0.00125 env.sim.dt=0.00125 env.decimation=4 env.sim.render_interval=4
"""

from __future__ import annotations

import argparse
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


def _parse_thresholds(text: str) -> list[float]:
    values = [float(v) for v in re.split(r"[,\s]+", str(text).strip()) if v != ""]
    if not values:
        raise argparse.ArgumentTypeError("--dormant-thresholds must contain at least one number")
    return sorted({max(0.0, v) for v in values})


parser = argparse.ArgumentParser(description="Compare two Solo12 race policies on plasticity metrics.")
parser.add_argument(
    "--task",
    type=str,
    default="Solo12-Race-ParamsConditionedEnc-Direct-v0",
    help="Race task name.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment + agent seed (paired across both checkpoints).")

parser.add_argument("--checkpoint-a", "--checkpoint_a", dest="checkpoint_a", type=str, required=True, help="First checkpoint.")
parser.add_argument("--checkpoint-b", "--checkpoint_b", dest="checkpoint_b", type=str, required=True, help="Second checkpoint.")
parser.add_argument("--label-a", "--label_a", dest="label_a", type=str, default=None, help="Short label for checkpoint A.")
parser.add_argument("--label-b", "--label_b", dest="label_b", type=str, default=None, help="Short label for checkpoint B.")

parser.add_argument(
    "--eval_n",
    type=_optional_int,
    default=64,
    help="Number of one-episode rollouts to evaluate per checkpoint. None defaults to --eval_n_parallel.",
)
parser.add_argument(
    "--eval_n_parallel",
    dest="eval_n_parallel",
    type=int,
    default=64,
    help="Number of environments simulated in parallel.",
)
parser.add_argument(
    "--max_steps",
    type=_optional_int,
    default=3000,
    help="Per-episode timeout in env (policy) steps. Pass None to use --episode_length_s instead.",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Per-episode timeout in seconds. Used only when --max_steps None.",
)
parser.add_argument(
    "--keep_training_stochasticity",
    action="store_true",
    default=False,
    help="Keep training-time stochasticity. By default DR events, observation corruption, reset-velocity "
    "randomization, and actuation delay are disabled for evaluation.",
)

# Plasticity-analysis options.
parser.add_argument(
    "--dormant-thresholds",
    "--dormant_thresholds",
    dest="dormant_thresholds",
    type=_parse_thresholds,
    default="0.0,0.025,0.1",
    help="Comma-separated tau thresholds for the Sokar dormant-unit score. tau=0 == dead units.",
)
parser.add_argument(
    "--rank-sample-cap",
    "--rank_sample_cap",
    dest="rank_sample_cap",
    type=int,
    default=8192,
    help="Max post-activation feature rows kept (reservoir-sampled) per layer for the SVD/rank estimate.",
)
parser.add_argument(
    "--include-critic",
    "--include_critic",
    dest="include_critic",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also analyze the critic hidden layers (runs an extra critic forward per step).",
)
parser.add_argument(
    "--warmup-steps",
    "--warmup_steps",
    dest="warmup_steps",
    type=int,
    default=0,
    help="Skip this many initial policy steps before collecting features (lets episodes leave the start pose).",
)
parser.add_argument(
    "--cbp-util-type",
    "--cbp_util_type",
    dest="cbp_util_type",
    type=str,
    default="contribution",
    choices=("contribution", "weight"),
    help="Which CBP utility to report per unit ('contribution' = out_weight_mag * mean|feature|).",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Directory for the JSON report. Defaults to <checkpoint-A-run-dir>/plasticity_eval.",
)

# Friction-condition overrides (kept for parity with solo_race_eval; usually set via hydra env.* instead).
parser.add_argument("--within-episode-fric-resample", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--within-episode-fric-resample-time-range", type=float, nargs=2, metavar=("MIN_S", "MAX_S"), default=None)
parser.add_argument("--group-all-patches-single-bucket", action=argparse.BooleanOptionalAction, default=None)

# DAgger adapter options (forwarded to the shared loader; teacher checkpoints don't need these).
parser.add_argument("--dagger-teacher-checkpoint", "--dagger_teacher_checkpoint", dest="dagger_teacher_checkpoint", type=str, default=None)
parser.add_argument("--dagger-teacher-shared-networks", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.eval_n is not None and args_cli.eval_n <= 0:
    parser.error("--eval_n must be positive when provided")
if args_cli.eval_n_parallel <= 0:
    parser.error("--eval_n_parallel must be positive")
if args_cli.rank_sample_cap < 8:
    parser.error("--rank-sample-cap must be at least 8")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
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
from continual_backprop import CBPSpec, cbp_specs_for_sequential_mlp  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# Env-config helpers (mirrors solo_race_eval.py so the eval condition matches the batch evaluator).
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# Per-episode race metric helpers (copied from solo_race_eval.py to keep this script standalone).
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# Policy loading (mirrors solo_race_eval._load_policy).
# ----------------------------------------------------------------------------------------------------------------------
def _load_policy(vec_env, agent_cfg: RslRlBaseRunnerCfg, resume_path: str, dagger_adapter_checkpoint: dict[str, Any] | None):
    """Return (runner, callable_policy). For DAgger adapters runner is None."""
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
    runner.eval_mode()
    return runner, runner.get_inference_policy(device=vec_env.unwrapped.device)


def _resolve_module(runner, policy) -> torch.nn.Module:
    """Return the underlying actor-critic nn.Module exposing `.actor` (and maybe `.critic`)."""
    if runner is not None:
        return runner.alg.policy
    for attr in ("actor_critic", "module", "policy"):
        candidate = getattr(policy, attr, None)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "actor"):
            return candidate
    if isinstance(policy, torch.nn.Module) and hasattr(policy, "actor"):
        return policy
    raise RuntimeError(
        "Could not locate an actor-critic module exposing `.actor`. This tool targets the standard "
        "EnvParamsConditionedEncoderActor teacher checkpoints; DAgger adapters are not supported for the "
        "dormancy/rank analysis."
    )


# ----------------------------------------------------------------------------------------------------------------------
# Plasticity feature capture + metrics.
# ----------------------------------------------------------------------------------------------------------------------
class _ReservoirSampler:
    """Bounded, ~unbiased sample of streamed feature rows for the SVD/rank estimate (Algorithm R)."""

    def __init__(self, capacity: int, dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.device = device
        self.buffer = torch.empty((self.capacity, self.dim), dtype=torch.float32, device=device)
        self.n_filled = 0
        self.seen = 0

    def add(self, rows: torch.Tensor) -> None:
        rows = rows.detach().to(self.buffer.device, torch.float32)
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        m = rows.shape[0]
        i = 0
        if self.n_filled < self.capacity:
            take = min(self.capacity - self.n_filled, m)
            self.buffer[self.n_filled : self.n_filled + take] = rows[:take]
            self.n_filled += take
            self.seen += take
            i = take
        if i >= m:
            return
        rest = rows[i:]
        r = rest.shape[0]
        positions = self.seen + torch.arange(1, r + 1, device=self.buffer.device, dtype=torch.float32)
        accept = torch.rand(r, device=self.buffer.device) < (self.capacity / positions)
        n_accept = int(accept.sum().item())
        if n_accept > 0:
            slots = torch.randint(0, self.capacity, (n_accept,), device=self.buffer.device)
            self.buffer[slots] = rest[accept]
        self.seen += r

    def get(self) -> torch.Tensor:
        return self.buffer[: self.n_filled]


class _FeatureCollector:
    """Capture post-activation hidden features for one CBP feature group and accumulate plasticity stats."""

    def __init__(self, spec: CBPSpec, rank_sample_cap: int, device: torch.device) -> None:
        self.spec = spec
        self.name = spec.name
        self.device = device
        self.hidden_size = int(spec.input_layer.out_features)
        self.feature_transform = tuple(spec.feature_transform)
        # running per-unit accumulators (exact over all collected rows)
        self.count = 0
        self.sum_abs = torch.zeros(self.hidden_size, dtype=torch.float64, device=device)
        self.sum_val = torch.zeros(self.hidden_size, dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros(self.hidden_size, dtype=torch.float64, device=device)
        self.reservoir = _ReservoirSampler(rank_sample_cap, self.hidden_size, device)
        self._last_features: torch.Tensor | None = None
        self._handle = spec.capture_layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return
        features = output
        if self.feature_transform:
            with torch.no_grad():
                for module in self.feature_transform:
                    features = module(features)
        self._last_features = features.detach()

    def consume(self) -> None:
        """Fold the most recent captured features into the accumulators, then clear them."""
        features = self._last_features
        self._last_features = None
        if features is None:
            return
        flat = features.reshape(-1, features.shape[-1]).to(self.device)
        flat64 = flat.to(torch.float64)
        self.count += int(flat.shape[0])
        self.sum_abs += flat64.abs().sum(dim=0)
        self.sum_val += flat64.sum(dim=0)
        self.sum_sq += (flat64 * flat64).sum(dim=0)
        self.reservoir.add(flat)

    def remove_hook(self) -> None:
        try:
            self._handle.remove()
        except Exception:
            pass


def _rank_metrics(features: torch.Tensor) -> dict[str, Any]:
    """Stable rank, effective rank, approximate rank, numerical rank of a feature matrix [n, d]."""
    n, d = int(features.shape[0]), int(features.shape[1])
    out: dict[str, Any] = {"n_samples": n, "dim": d}
    if n < 2:
        for key in ("stable_rank", "effective_rank", "approx_rank_99", "numerical_rank"):
            out[key] = None
        return out
    x = features.to(torch.float64).cpu()
    try:
        sigma = torch.linalg.svdvals(x)
    except Exception:
        sigma = torch.linalg.svdvals(x + 1e-9 * torch.randn_like(x))
    sigma = torch.clamp(sigma, min=0.0)
    sigma = sigma[sigma > 0]
    if sigma.numel() == 0:
        out.update({"stable_rank": 0.0, "effective_rank": 0.0, "approx_rank_99": 0, "numerical_rank": 0})
        return out
    sigma_sq = sigma * sigma
    sigma_max_sq = float(sigma_sq[0].item())
    # stable rank = ||X||_F^2 / sigma_max^2
    out["stable_rank"] = float((sigma_sq.sum() / max(sigma_max_sq, 1e-30)).item())
    # effective rank = exp(entropy of normalized singular values)  (Roy & Vetterli; the Nature CBP measure)
    p = sigma / sigma.sum()
    entropy = float(-(p * torch.log(p)).sum().item())
    out["effective_rank"] = float(math.exp(entropy))
    # approximate rank: # of components to reach 99% of the squared-singular-value energy
    energy = torch.cumsum(sigma_sq, dim=0) / sigma_sq.sum()
    out["approx_rank_99"] = int(torch.searchsorted(energy, torch.tensor(0.99, dtype=energy.dtype)).item()) + 1
    # numerical rank: singular values above relative tolerance
    tol = float(sigma[0].item()) * 1e-3
    out["numerical_rank"] = int((sigma > tol).sum().item())
    return out


def _layer_metrics(collector: _FeatureCollector, thresholds: list[float], cbp_util_type: str) -> dict[str, Any]:
    n = collector.count
    d = collector.hidden_size
    metrics: dict[str, Any] = {"name": collector.name, "hidden_size": d, "n_samples": n}
    if n == 0:
        return metrics

    mean_abs = (collector.sum_abs / n)
    mean_val = (collector.sum_val / n)
    var = torch.clamp(collector.sum_sq / n - mean_val * mean_val, min=0.0)
    std = torch.sqrt(var)

    # Sokar dormant-neuron score: s_i = E|h_i| / mean_j E|h_j|.
    denom = float(mean_abs.mean().item())
    if denom <= 0.0:
        score = torch.zeros_like(mean_abs)  # everything dead
        dormant_all = True
    else:
        score = mean_abs / denom
        dormant_all = False
    dormant_fraction: dict[str, float] = {}
    dormant_counts: dict[str, int] = {}
    for tau in thresholds:
        if dormant_all:
            count = d
        else:
            count = int((score <= tau).sum().item())
        dormant_counts[f"{tau:g}"] = count
        dormant_fraction[f"{tau:g}"] = count / d if d > 0 else float("nan")
    metrics["dormant_fraction"] = dormant_fraction
    metrics["dormant_counts"] = dormant_counts

    # ELU-aware diagnostics: near-constant (no information) and saturated-negative (ELU stuck near -1, ~zero gradient).
    near_constant = int((std <= 1e-3).sum().item())
    saturated_negative = int(((mean_val < -0.9) & (std <= 0.05)).sum().item())
    metrics["near_constant_fraction"] = near_constant / d
    metrics["saturated_negative_fraction"] = saturated_negative / d

    metrics["mean_abs_activation"] = float(mean_abs.mean().item())
    metrics["activation_std_mean"] = float(std.mean().item())

    # CBP utility readout (per-unit), to relate to the units CBP would actually replace.
    out_w = collector.spec.output_layer
    if isinstance(out_w, torch.nn.Linear):
        out_weight_mag = out_w.weight.detach().abs().mean(dim=0).to(torch.float64).to(mean_abs.device)
    else:
        out_weight_mag = torch.ones_like(mean_abs)
    if cbp_util_type == "weight":
        util = out_weight_mag
    else:  # contribution
        util = out_weight_mag * mean_abs
    util_max = float(util.max().item()) if util.numel() else 0.0
    metrics["utility"] = {
        "type": cbp_util_type,
        "mean": float(util.mean().item()),
        "median": float(util.median().item()),
        "min": float(util.min().item()),
        "frac_below_1pct_of_max": float((util < 0.01 * util_max).float().mean().item()) if util_max > 0 else float("nan"),
    }

    # Weight-magnitude signal from the Nature paper (growth of incoming weights).
    in_w = collector.spec.input_layer
    if isinstance(in_w, torch.nn.Linear):
        metrics["input_weight_abs_mean"] = float(in_w.weight.detach().abs().mean().item())

    # Rank of the representation (post-activation feature matrix), uncentered and centered.
    sample = collector.reservoir.get()
    metrics["rank"] = {
        "uncentered": _rank_metrics(sample),
        "centered": _rank_metrics(sample - sample.mean(dim=0, keepdim=True)) if sample.shape[0] >= 2 else _rank_metrics(sample),
    }
    return metrics


def _build_feature_specs(module: torch.nn.Module, include_critic: bool) -> list[CBPSpec]:
    """Hidden-feature groups for the actor (and optionally critic), deduped by Linear identity."""
    specs = list(cbp_specs_for_sequential_mlp("actor", getattr(module, "actor")))
    if include_critic:
        critic = getattr(module, "critic", None)
        if critic is not None and critic is not getattr(module, "actor", None):
            seen = {id(s.input_layer) for s in specs}
            for spec in cbp_specs_for_sequential_mlp("critic", critic):
                if id(spec.input_layer) not in seen:
                    specs.append(spec)
    if not specs:
        raise RuntimeError("No hidden feature groups found on the actor MLP; cannot compute plasticity metrics.")
    return specs


def _aggregate_overall_dormancy(group_metrics: list[dict[str, Any]], thresholds: list[float], prefix: str) -> dict[str, float]:
    """Network-wide dormant fraction = total dormant units / total units, over the selected groups."""
    overall: dict[str, float] = {}
    for tau in thresholds:
        key = f"{tau:g}"
        total_units = 0
        total_dormant = 0
        for gm in group_metrics:
            if not gm.get("name", "").startswith(prefix):
                continue
            total_units += int(gm.get("hidden_size", 0))
            total_dormant += int(gm.get("dormant_counts", {}).get(key, 0))
        overall[key] = (total_dormant / total_units) if total_units > 0 else float("nan")
    return overall


# ----------------------------------------------------------------------------------------------------------------------
# Rollout: collect finished episodes + plasticity features for one checkpoint on the shared env.
# ----------------------------------------------------------------------------------------------------------------------
def _rollout_collect(
    *,
    vec_env,
    raw_env,
    policy,
    runner,
    module,
    collectors: list[_FeatureCollector],
    include_critic: bool,
    eval_n: int,
    num_envs: int,
    warmup_steps: int,
    label: str,
) -> dict[str, Any]:
    obs = vec_env.get_observations()
    device = raw_env.device

    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    floor_events = torch.zeros(num_envs, dtype=torch.long, device=device)
    pillar_events = torch.zeros(num_envs, dtype=torch.long, device=device)
    prev_floor = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_pillar = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_thigh = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes: list[dict[str, Any]] = []

    if eval_n <= num_envs:
        eval_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        eval_env_mask[:eval_n] = True
        recorded_env_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    else:
        eval_env_mask = None
        recorded_env_mask = None

    max_steps = int(raw_env.max_episode_length)
    max_total_steps = max_steps * (math.ceil(eval_n / num_envs) + 1) + 5
    t0 = time.time()
    next_progress_t = t0 + 5.0

    for step_idx in range(1, max_total_steps + 1):
        if len(episodes) >= eval_n:
            break

        progress_before_step = _gate_progress_ratio(raw_env)[:num_envs].clone()

        floor_now = _floor_collision_now(raw_env)[:num_envs]
        pillar_now = _pillar_collision_now(raw_env)[:num_envs]
        thigh_now = _thigh_contact_now(raw_env)[:num_envs]
        floor_rising = floor_now & ~prev_floor
        pillar_rising = (pillar_now & ~prev_pillar) | (thigh_now & ~prev_thigh)
        floor_events[floor_rising] += 1
        pillar_events[pillar_rising] += 1
        prev_floor, prev_pillar, prev_thigh = floor_now, pillar_now, thigh_now

        with torch.inference_mode():
            actions = policy(obs)  # runs the actor -> actor capture hooks fire for the current obs
            if include_critic:
                try:
                    module.evaluate(obs)  # runs the critic -> critic capture hooks fire
                except Exception:
                    pass
            # Fold features for the current obs into the plasticity accumulators (after warmup).
            if step_idx > warmup_steps:
                for collector in collectors:
                    collector.consume()
            else:
                for collector in collectors:
                    collector._last_features = None

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
        done = _to_bool_tensor(dones, device, num_envs)
        if bool(torch.any(done)):
            terminated = _safe_bool_metric(raw_env, "reset_terminated")[:num_envs]
            timed_out = _safe_bool_metric(raw_env, "reset_time_outs")[:num_envs]
            finished = done & (progress_before_step >= 1.0)
            floor_terminal = done & terminated & ~finished & ~timed_out
            floor_events[floor_terminal] += 1

            record_done = done
            if eval_env_mask is not None and recorded_env_mask is not None:
                record_done = done & eval_env_mask & ~recorded_env_mask

            for env_id in record_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if len(episodes) >= eval_n:
                    break
                steps = int(episode_steps[env_id].item())
                sec = steps * float(raw_env.step_dt)
                did_finish = bool(finished[env_id].item())
                did_timeout = bool(timed_out[env_id].item()) and not did_finish
                did_floor_terminal = bool(floor_terminal[env_id].item()) and not did_finish
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
                    }
                )
                if recorded_env_mask is not None:
                    recorded_env_mask[env_id] = True

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            episode_steps[done_ids] = 0
            floor_events[done_ids] = 0
            pillar_events[done_ids] = 0
            prev_floor[done_ids] = False
            prev_pillar[done_ids] = False
            prev_thigh[done_ids] = False

        now = time.time()
        if now >= next_progress_t:
            finished_count = sum(bool(ep.get("finished", False)) for ep in episodes)
            samples = collectors[0].count if collectors else 0
            print(
                f"[{label}] step={step_idx}/{max_total_steps} episodes={len(episodes)}/{eval_n} "
                f"finished={finished_count} feat_samples={samples} elapsed={now - t0:.1f}s",
                flush=True,
            )
            next_progress_t = now + 5.0

    elapsed = max(time.time() - t0, 1e-6)
    print(
        f"[{label}] done: episodes={len(episodes)}/{eval_n}, "
        f"finished={sum(bool(ep.get('finished', False)) for ep in episodes)}, "
        f"feat_samples={collectors[0].count if collectors else 0}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "episodes": episodes,
        "step_dt": float(raw_env.step_dt),
        "max_episode_length": max_steps,
    }


def _finish_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    episodes = rollout["episodes"]
    step_dt = rollout["step_dt"]
    max_episode_length = rollout["max_episode_length"]
    finished = [ep for ep in episodes if ep.get("finished")]
    finish_seconds = [float(ep["finish_time_seconds"]) for ep in finished if ep.get("finish_time_seconds") is not None]
    finish_steps = [float(ep["finish_time_steps"]) for ep in finished if ep.get("finish_time_steps") is not None]
    penalized_steps = [float(ep["steps"] if ep.get("finished") else max_episode_length) for ep in episodes]
    penalized_seconds = [s * step_dt for s in penalized_steps]
    return {
        "num_episodes": len(episodes),
        "num_finished": len(finished),
        "finish_ratio": (len(finished) / len(episodes)) if episodes else float("nan"),
        "finish_time_seconds_success_only": _summarize(finish_seconds),
        "finish_time_steps_success_only": _summarize(finish_steps),
        "finish_time_seconds_penalized": _summarize(penalized_seconds),
        "gate_progress_ratio": _summarize([float(ep["gate_progress_ratio"]) for ep in episodes]),
        "floor_events_total": int(sum(int(ep.get("floor_collision_events", 0)) for ep in episodes)),
        "pillar_events_total": int(sum(int(ep.get("pillar_collision_events", 0)) for ep in episodes)),
    }


def _evaluate_one(
    *,
    label: str,
    checkpoint: str,
    vec_env,
    raw_env,
    agent_cfg: RslRlBaseRunnerCfg,
    eval_n: int,
    num_envs: int,
    thresholds: list[float],
    include_critic: bool,
    warmup_steps: int,
    rank_sample_cap: int,
    cbp_util_type: str,
    seed: int | None,
) -> dict[str, Any]:
    # Re-seed and reset so both checkpoints face matched starting conditions.
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed))
    try:
        vec_env.reset()
    except Exception as exc:
        print(f"[WARN] vec_env.reset() failed ({exc}); continuing from current state.", flush=True)

    dagger_adapter_checkpoint = load_dagger_adapter_checkpoint(checkpoint)
    runner, policy = _load_policy(vec_env, agent_cfg, checkpoint, dagger_adapter_checkpoint)
    module = _resolve_module(runner, policy)
    module.eval()

    specs = _build_feature_specs(module, include_critic)
    device = raw_env.device
    collectors = [_FeatureCollector(spec, rank_sample_cap, device) for spec in specs]
    print(
        f"[{label}] analyzing {len(collectors)} hidden feature groups: "
        + ", ".join(f"{c.name}({c.hidden_size})" for c in collectors),
        flush=True,
    )

    try:
        rollout = _rollout_collect(
            vec_env=vec_env,
            raw_env=raw_env,
            policy=policy,
            runner=runner,
            module=module,
            collectors=collectors,
            include_critic=include_critic,
            eval_n=eval_n,
            num_envs=num_envs,
            warmup_steps=warmup_steps,
            label=label,
        )
        group_metrics = [_layer_metrics(c, thresholds, cbp_util_type) for c in collectors]
    finally:
        for c in collectors:
            c.remove_hook()

    actor_groups = [g for g in group_metrics if g["name"].startswith("actor")]
    representation_layer = actor_groups[-1]["name"] if actor_groups else (group_metrics[-1]["name"] if group_metrics else None)
    result = {
        "label": label,
        "checkpoint": os.path.abspath(checkpoint),
        "finish": _finish_summary(rollout),
        "overall_actor_dormant_fraction": _aggregate_overall_dormancy(group_metrics, thresholds, prefix="actor"),
        "representation_layer": representation_layer,
        "groups": group_metrics,
    }
    if include_critic and any(g["name"].startswith("critic") for g in group_metrics):
        result["overall_critic_dormant_fraction"] = _aggregate_overall_dormancy(group_metrics, thresholds, prefix="critic")

    # Free the runner/policy before the next checkpoint to avoid holding two graphs of optimizer state.
    del runner, policy, module, collectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Comparison report.
# ----------------------------------------------------------------------------------------------------------------------
def _fmt(value: Any, nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "   --   "
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _delta(a: Any, b: Any, nd: int = 3) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b):
        return f"{(b - a):+.{nd}f}"
    return "   --   "


def _print_comparison(result_a: dict[str, Any], result_b: dict[str, Any], thresholds: list[float]) -> None:
    la, lb = result_a["label"], result_b["label"]
    fa, fb = result_a["finish"], result_b["finish"]

    def row(name: str, a: Any, b: Any, nd: int = 3) -> str:
        return f"  {name:<40} {_fmt(a, nd):>10} {_fmt(b, nd):>10} {_delta(a, b, nd):>10}"

    print("\n" + "=" * 76, flush=True)
    print(f"  PLASTICITY COMPARISON   A='{la}'   B='{lb}'   (Delta = B - A)", flush=True)
    print("=" * 76, flush=True)
    print(f"  {'metric':<40} {la:>10} {lb:>10} {'Delta':>10}", flush=True)
    print("-" * 76, flush=True)

    print("  [race performance]", flush=True)
    print(row("episodes / finished", f"{fa['num_finished']}/{fa['num_episodes']}", f"{fb['num_finished']}/{fb['num_episodes']}"), flush=True)
    print(row("finish ratio", fa["finish_ratio"], fb["finish_ratio"]), flush=True)
    print(row("finish time [s] (success mean)", fa["finish_time_seconds_success_only"]["mean"], fb["finish_time_seconds_success_only"]["mean"]), flush=True)
    print(row("finish time [s] (success median)", fa["finish_time_seconds_success_only"]["median"], fb["finish_time_seconds_success_only"]["median"]), flush=True)
    print(row("finish time [s] (penalized mean)", fa["finish_time_seconds_penalized"]["mean"], fb["finish_time_seconds_penalized"]["mean"]), flush=True)

    print("  [dormant units - actor, network-wide fraction]", flush=True)
    for tau in thresholds:
        key = f"{tau:g}"
        label = f"dormant frac (tau={key})" + ("  [=dead]" if tau == 0.0 else "")
        print(row(label, result_a["overall_actor_dormant_fraction"].get(key), result_b["overall_actor_dormant_fraction"].get(key), 4), flush=True)

    rep_a = next((g for g in result_a["groups"] if g["name"] == result_a["representation_layer"]), None)
    rep_b = next((g for g in result_b["groups"] if g["name"] == result_b["representation_layer"]), None)
    if rep_a and rep_b and "rank" in rep_a and "rank" in rep_b:
        print(f"  [representation rank - last actor hidden layer '{result_a['representation_layer']}']", flush=True)
        print(row("stable rank (uncentered)", rep_a["rank"]["uncentered"]["stable_rank"], rep_b["rank"]["uncentered"]["stable_rank"], 2), flush=True)
        print(row("effective rank (uncentered)", rep_a["rank"]["uncentered"]["effective_rank"], rep_b["rank"]["uncentered"]["effective_rank"], 2), flush=True)
        print(row("stable rank (centered)", rep_a["rank"]["centered"]["stable_rank"], rep_b["rank"]["centered"]["stable_rank"], 2), flush=True)
        print(row("effective rank (centered)", rep_a["rank"]["centered"]["effective_rank"], rep_b["rank"]["centered"]["effective_rank"], 2), flush=True)
        print(row("approx rank @99% (uncentered)", rep_a["rank"]["uncentered"]["approx_rank_99"], rep_b["rank"]["uncentered"]["approx_rank_99"], 0), flush=True)

    print("  [per-layer detail: dormant frac (tau=0.1) | eff.rank(unc) | mean|W_in|]", flush=True)
    for ga in result_a["groups"]:
        gb = next((g for g in result_b["groups"] if g["name"] == ga["name"]), None)
        if gb is None:
            continue
        da = ga.get("dormant_fraction", {}).get("0.1")
        db = gb.get("dormant_fraction", {}).get("0.1")
        era = ga.get("rank", {}).get("uncentered", {}).get("effective_rank")
        erb = gb.get("rank", {}).get("uncentered", {}).get("effective_rank")
        wa = ga.get("input_weight_abs_mean")
        wb = gb.get("input_weight_abs_mean")
        print(
            f"    {ga['name']:<26}(d={ga['hidden_size']:<4}) "
            f"dorm {_fmt(da, 3)}/{_fmt(db, 3)}  "
            f"erank {_fmt(era, 1)}/{_fmt(erb, 1)}  "
            f"|W| {_fmt(wa, 4)}/{_fmt(wb, 4)}",
            flush=True,
        )
    print("=" * 76 + "\n", flush=True)


# ----------------------------------------------------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.seed = seed
    env_cfg.seed = seed
    if seed is not None and hasattr(env_cfg, "friction_seed"):
        env_cfg.friction_seed = int(seed)

    if args_cli.max_steps is not None:
        step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
        env_cfg.episode_length_s = float(args_cli.max_steps) * step_dt
    elif args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)

    # Optional explicit friction-eval overrides (parity with solo_race_eval; hydra env.* usually preferred).
    if args_cli.within_episode_fric_resample is not None and hasattr(env_cfg, "within_episode_fric_resample"):
        env_cfg.within_episode_fric_resample = bool(args_cli.within_episode_fric_resample)
    if args_cli.group_all_patches_single_bucket is not None and hasattr(env_cfg, "group_all_patches_single_bucket"):
        env_cfg.group_all_patches_single_bucket = bool(args_cli.group_all_patches_single_bucket)
    if args_cli.within_episode_fric_resample_time_range is not None and hasattr(env_cfg, "within_episode_fric_resample_time_range"):
        lo, hi = sorted(float(v) for v in args_cli.within_episode_fric_resample_time_range)
        env_cfg.within_episode_fric_resample_time_range = (max(0.0, lo), hi)

    if not args_cli.keep_training_stochasticity:
        disabled = _disable_training_stochasticity_for_eval(
            env_cfg, preserve_patch_friction_randomization=bool(getattr(env_cfg, "randomize_fric_coefs", False))
        )
        if disabled:
            print("[INFO] Disabled training-time stochasticity for eval: " + ", ".join(disabled), flush=True)

    eval_n = int(args_cli.eval_n_parallel if args_cli.eval_n is None else args_cli.eval_n)
    num_envs = int(args_cli.eval_n_parallel)
    env_cfg.scene.num_envs = num_envs

    label_a = args_cli.label_a or Path(args_cli.checkpoint_a).stem
    label_b = args_cli.label_b or Path(args_cli.checkpoint_b).stem
    if label_a == label_b:
        label_a, label_b = f"{label_a}#A", f"{label_b}#B"

    print(
        f"[INFO] Plasticity eval: A='{label_a}' vs B='{label_b}'; task={args_cli.task}; "
        f"num_envs={num_envs}; eval_n={eval_n}; seed={seed}; "
        f"mu_dynamic_static_ratio={getattr(env_cfg, 'mu_dynamic_static_ratio', None)}; "
        f"friction_static_range={getattr(env_cfg, 'friction_static_range', None)}; "
        f"obs_corruption={getattr(env_cfg, 'enable_observation_corruption', None)}",
        flush=True,
    )

    # Build ONE env and evaluate both checkpoints on it (paired, identical conditions).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    thresholds = list(args_cli.dormant_thresholds)
    common = dict(
        vec_env=vec_env,
        raw_env=raw_env,
        agent_cfg=agent_cfg,
        eval_n=eval_n,
        num_envs=num_envs,
        thresholds=thresholds,
        include_critic=bool(args_cli.include_critic),
        warmup_steps=int(args_cli.warmup_steps),
        rank_sample_cap=int(args_cli.rank_sample_cap),
        cbp_util_type=str(args_cli.cbp_util_type),
        seed=seed,
    )
    result_a = _evaluate_one(label=label_a, checkpoint=os.path.abspath(args_cli.checkpoint_a), **common)
    result_b = _evaluate_one(label=label_b, checkpoint=os.path.abspath(args_cli.checkpoint_b), **common)
    env.close()

    _print_comparison(result_a, result_b, thresholds)

    out_dir = (
        Path(args_cli.out_dir)
        if args_cli.out_dir is not None
        else Path(os.path.dirname(os.path.dirname(os.path.abspath(args_cli.checkpoint_a)))) / "plasticity_eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task,
        "seed": seed,
        "num_envs": num_envs,
        "eval_n": eval_n,
        "max_steps": args_cli.max_steps,
        "episode_length_s": float(env_cfg.episode_length_s),
        "dormant_thresholds": thresholds,
        "rank_sample_cap": int(args_cli.rank_sample_cap),
        "include_critic": bool(args_cli.include_critic),
        "warmup_steps": int(args_cli.warmup_steps),
        "friction": {
            "mu_dynamic_static_ratio": getattr(env_cfg, "mu_dynamic_static_ratio", None),
            "friction_static_range": list(getattr(env_cfg, "friction_static_range", []) or []),
            "randomize_fric_coefs": getattr(env_cfg, "randomize_fric_coefs", None),
            "enable_observation_corruption": getattr(env_cfg, "enable_observation_corruption", None),
        },
        "A": result_a,
        "B": result_b,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"plasticity_{Path(args_cli.checkpoint_a).stem}_vs_{Path(args_cli.checkpoint_b).stem}_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote plasticity report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
