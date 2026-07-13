# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL training script for Solo12 / borinot workflows.

Mirrors the borinot skrl train workflow as closely as possible while using RSL-RL.
Adds:
- borinot-style W&B init naming/config
- code/config snapshot artifact upload
- Solo12 symmetry mode toggles (none / augmentation / loss / both)
- round-wise observation permutations for plasticity-loss experiments
- paper-style plasticity mitigation (LayerNorm, regenerative L2, shrink+perturb)
- run naming compatible with the current skrl conventions
"""

import argparse
import copy
import inspect
import importlib.metadata as metadata
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_UPSTREAM_RSL_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"
_UPSTREAM_SKRL_HELPERS_DIR = Path(__file__).resolve().parents[1] / "skrl"
for _path in (str(_UPSTREAM_RSL_SCRIPT_DIR), str(_UPSTREAM_SKRL_HELPERS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from helpers import _wandb_snapshot  # isort: skip


parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable-periodic-eval-video",
    action="store_true",
    default=False,
    help="Disable Solo12 race periodic inference videos during training.",
)
parser.add_argument(
    "--periodic-eval-video-interval-minutes",
    type=float,
    default=15.0,
    help=(
        "Training-time interval between Solo12 race inference videos. "
        "Uses RSL-RL collection+learn time, not video/upload time. Set <=0 to disable."
    ),
)
parser.add_argument(
    "--periodic-eval-video-episodes",
    type=int,
    default=2,
    help="Number of env-0 inference episodes to record for each periodic Solo12 race video.",
)
parser.add_argument(
    "--periodic-eval-video-speed",
    type=float,
    default=0.5,
    help="Playback speed multiplier for periodic Solo12 race videos; 0.5 writes half-speed videos.",
)
parser.add_argument(
    "--periodic-eval-video-env-index",
    type=int,
    default=0,
    help="Deprecated: detached periodic eval videos now launch a one-env play process and record env 0.",
)
parser.add_argument(
    "--periodic-eval-video-max-steps",
    type=int,
    default=0,
    help=(
        "Optional safety cap for each periodic eval rollout in policy steps. "
        "The default 0 records until the requested number of env-0 episodes completes."
    ),
)
parser.add_argument(
    "--within-episode-fric-resample",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override Solo12 race within-episode patch-friction resampling during training.",
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
    help="Override whether all Solo12 race patches share one sampled friction bucket.",
)
parser.add_argument(
    "--simple_video",
    "--simple-video",
    "--periodic-eval-simple-video",
    action="store_true",
    dest="periodic_eval_simple_video",
    default=False,
    help="Upload lightweight top-down state-space periodic eval videos instead of Isaac RGB videos.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Override the RSL-RL runner config run_name without editing the task config file.",
)
parser.add_argument(
    "--shared_networks",
    action="store_true",
    default=False,
    help=(
        "Use shared actor/critic network parameters when the selected policy supports it. "
        "For encoded policies, actor and critic encoders are shared too."
    ),
)
parser.add_argument(
    "--reuse-mlp",
    action="store_true",
    default=False,
    help=(
        "Initialize compatible actor/critic MLP weights and observation normalizer stats from --checkpoint. "
        "This skips TCN encoders, optimizer state, and iteration resume."
    ),
)
parser.add_argument(
    "--reuse-mlp-source-history-kind",
    type=str,
    default="auto",
    choices=["auto", "foot_imu", "joint_state"],
    help=(
        "When --reuse-mlp adapts 24D TCN-history normalizer stats into a 48D joint+IMU history, "
        "choose whether the checkpoint history is foot_imu or joint_state. 'auto' uses the checkpoint path/name "
        "when possible and otherwise assumes foot_imu."
    ),
)
parser.add_argument(
    "--init-from-dagger-adapter",
    "--init_from_dagger_adapter",
    dest="init_from_dagger_adapter",
    type=str,
    default=None,
    help=(
        "Initialize a solo12-IMU-student-rl policy from a Solo12 base-IMU DAgger adapter checkpoint. "
        "If --checkpoint points to an adapter checkpoint, this is inferred automatically."
    ),
)
parser.add_argument(
    "--use-cbp",
    "--cbp-enable",
    "--cbp_enable",
    dest="use_cbp",
    action="store_true",
    default=False,
    help="Enable Continual Backpropagation neuron replacement for RSL-RL actor/critic MLPs.",
)
parser.add_argument(
    "--cbp-replacement-rate",
    "--cbp_replacement_rate",
    type=float,
    default=1.0e-4,
    help="CBP replacement rate per mature hidden unit and optimizer step.",
)
parser.add_argument(
    "--cbp-maturity-threshold",
    "--cbp_maturity_threshold",
    type=int,
    default=10_000,
    help="Number of optimizer steps before a hidden unit is eligible for CBP replacement.",
)
parser.add_argument(
    "--cbp-decay-rate",
    "--cbp_decay_rate",
    type=float,
    default=0.99,
    help="Exponential decay rate for CBP feature-utility estimates.",
)
parser.add_argument(
    "--cbp-util-type",
    "--cbp_util_type",
    type=str,
    choices=(
        "contribution",
        "zero_contribution",
        "adaptable_contribution",
        "weight",
        "adaptation",
        "feature_by_input",
        "random",
    ),
    default="contribution",
    help="CBP utility score used to choose mature hidden units for replacement.",
)
parser.add_argument(
    "--cbp-init",
    "--cbp_init",
    type=str,
    choices=("default", "xavier", "lecun", "kaiming"),
    default="kaiming",
    help="Initialization bound used when CBP resets incoming weights.",
)
parser.add_argument(
    "--cbp-accumulate",
    "--cbp_accumulate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Accumulate fractional CBP replacements across optimizer steps.",
)
parser.add_argument(
    "--plasticity-metrics",
    "--plasticity_metrics",
    dest="plasticity_metrics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Log plasticity diagnostics (feature rank, %% dormant units, weight norm, gradient kurtosis; "
        "arXiv:2506.03404 Appendix B) for the actor and critic MLPs as W&B/TensorBoard scalars."
    ),
)
parser.add_argument(
    "--plasticity-metrics-interval",
    "--plasticity_metrics_interval",
    dest="plasticity_metrics_interval",
    type=int,
    default=10,
    help="Learning iterations between plasticity-metric measurements.",
)
parser.add_argument(
    "--plasticity-metrics-sample-cap",
    "--plasticity_metrics_sample_cap",
    dest="plasticity_metrics_sample_cap",
    type=int,
    default=None,
    help=(
        "Maximum number of current environment observations used for activation-based plasticity metrics. "
        "Defaults to the training environment count."
    ),
)
parser.add_argument(
    "--plasticity-loss-exp",
    "--plasticity_loss_exp",
    dest="plasticity_loss_exp",
    action="store_true",
    default=False,
    help=(
        "Run the round-wise observation permutation experiment from arXiv:2405.19153. "
        "Round duration and count come from env.duration_plasticity_exp_iteration and "
        "env.num_plasticity_exp_iterations."
    ),
)
parser.add_argument(
    "--plasticity-loss-exp-reset-all",
    "--plasticity_loss_exp_reset_all",
    dest="plasticity_loss_exp_reset_all",
    action="store_true",
    default=False,
    help=(
        "Reset the full actor-critic and optimizer at every observation-permutation boundary. "
        "Requires --plasticity-loss-exp and provides the paper's reset-all control."
    ),
)
parser.add_argument(
    "--plasticity-exp-first-layer-only",
    "--plasticity_exp_first_layer_only",
    dest="plasticity_exp_first_layer_only",
    action="store_true",
    default=False,
    help=(
        "After the first observation-permutation phase, freeze the continued actor-critic except "
        "for the actor and critic input Linear layers. Requires --plasticity-loss-exp."
    ),
)
parser.add_argument(
    "--plasticity-mitigation-strategy",
    "--plasticity_mitigation_strategy",
    type=str,
    default="none",
    help=(
        "Paper-style mitigation strategy: none, layernorm, regenerative-l2, "
        "regenerative-l2-layernorm, shrink-perturb, soft-shrink-perturb, or "
        "soft-shrink-perturb-layernorm. The boundary shrink-perturb strategy requires "
        "--plasticity-loss-exp; continuous strategies and LayerNorm may also be used in ordinary runs."
    ),
)
parser.add_argument(
    "--plasticity-regen-l2-coef",
    "--plasticity_regen_l2_coef",
    type=float,
    default=1.0e-2,
    help="Coefficient for the paper's global unsquared L2 distance to initialization (default: 0.01).",
)
parser.add_argument(
    "--plasticity-soft-sp-beta",
    "--plasticity_soft_sp_beta",
    type=float,
    default=1.0e-6,
    help="Fresh-initialization mixture beta applied after every optimizer step by soft shrink+perturb.",
)
parser.add_argument(
    "--plasticity-sp-beta",
    "--plasticity_sp_beta",
    type=float,
    default=0.5,
    help="Fresh-initialization mixture beta applied at each distribution shift by shrink+perturb.",
)
parser.add_argument(
    "--plasticity-mitigation-seed",
    "--plasticity_mitigation_seed",
    type=int,
    default=None,
    help="Independent fresh-initialization RNG seed. Defaults deterministically to the agent seed plus an offset.",
)
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--symmetry-mode",
    type=str,
    default="none",
    choices=["none", "augmentation", "loss", "both"],
    help=(
        "Enable the symmetric PPO variants from arXiv:2403.04359. "
        "'augmentation' enables symmetry data augmentation; "
        "'loss' enables mirror loss; 'both' combines the two."
    ),
)
parser.add_argument(
    "--symmetry-loss-coeff",
    type=float,
    default=1.0e-3,
    help="Mirror/symmetry loss coefficient used when --symmetry-mode is 'loss' or 'both'.",
)
parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="Optional W&B entity/team. If omitted, uses the one from the runner config if available.",
)
parser.add_argument(
    "--wandb-name",
    type=str,
    default=None,
    help="Optional explicit W&B run name. Defaults to the generated full run name.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _is_solo12_race_task_name(task_name: str | None) -> bool:
    return bool(task_name) and "Solo12-Race" in str(task_name)


_SOLO12_DIRECT_SYMMETRY_TASKS = {
    "solo12-v0",
    "solo12-two-feet",
    "Isaac-Solo12-Laas-Direct-v0",
    "solo12-IMU-based-teacher",
    "solo12-IMU-student-rl",
    "Isaac-Solo12-BaseIMU-Teacher-Direct-v0",
    "Isaac-Solo12-BaseIMU-StudentRL-Direct-v0",
}


def _sync_base_imu_policy_cfg_from_env_cfg(env_cfg, agent_cfg) -> None:
    """Keep base-IMU actor-critic dimensions aligned with Hydra env overrides."""
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


def _plasticity_permutation_cfg(env_cfg) -> tuple[int, int]:
    """Read and validate the round schedule requested through the environment config."""

    required_fields = ("duration_plasticity_exp_iteration", "num_plasticity_exp_iterations")
    missing = [name for name in required_fields if not hasattr(env_cfg, name)]
    if missing:
        raise ValueError(
            "--plasticity-loss-exp requires an environment config with " + ", ".join(missing) + "."
        )
    duration = int(env_cfg.duration_plasticity_exp_iteration)
    num_rounds = int(env_cfg.num_plasticity_exp_iterations)
    if duration < 1 or num_rounds < 1:
        raise ValueError(
            "Plasticity experiment duration/count must be positive, got "
            f"duration={duration}, num_rounds={num_rounds}."
        )
    return duration, num_rounds


def _periodic_eval_video_requested(parsed_args) -> bool:
    return (
        _is_solo12_race_task_name(getattr(parsed_args, "task", None))
        and not bool(getattr(parsed_args, "disable_periodic_eval_video", False))
        and float(getattr(parsed_args, "periodic_eval_video_interval_minutes", 0.0)) > 0.0
        and int(getattr(parsed_args, "periodic_eval_video_episodes", 0)) > 0
    )


PERIODIC_EVAL_VIDEO_REQUESTED = _periodic_eval_video_requested(args_cli)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    raise SystemExit(1)

import logging

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import plasticity_metrics
import plasticity_mitigation
import observation_permutation
import solo12_rnd
from continual_backprop import build_continual_backprop_manager, collect_actor_critic_cbp_specs
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlSymmetryCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import borinotIsaacLab.tasks  # noqa: F401
import solo12_symmetry
import solo12_race_symmetry

try:
    from isaaclab_tasks.direct.solo12_race.agents.shared_actor_critic import SharedActorCritic
    import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

    rsl_rl_on_policy_runner.SharedActorCritic = SharedActorCritic
except Exception:
    SharedActorCritic = None

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _patch_rsl_rl_wandb_writer_for_single_stream() -> None:
    """Keep TensorBoard files local while sending a single metric stream to W&B.

    Upstream ``rsl_rl.utils.wandb_utils.WandbSummaryWriter`` does both of these for every scalar:
    1. writes the scalar to TensorBoard via ``SummaryWriter.add_scalar``
    2. writes the scalar directly to W&B via ``wandb.log``

    If W&B tensorboard syncing is enabled (explicitly or through local/default W&B settings),
    the same metric reaches W&B twice and the workspace ends up with duplicate panels such as
    multiple ``Train/mean_reward`` charts. We keep the local TensorBoard event files, but force
    the W&B run created by this script to disable TensorBoard syncing.
    """

    from torch.utils.tensorboard import SummaryWriter
    from rsl_rl.utils import wandb_utils as rsl_wandb_utils

    if getattr(rsl_wandb_utils.WandbSummaryWriter, "_borinot_single_stream_patch", False):
        return

    import wandb

    class SingleStreamWandbSummaryWriter(rsl_wandb_utils.WandbSummaryWriter):
        _borinot_single_stream_patch = True

        def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
            SummaryWriter.__init__(self, log_dir, flush_secs)

            run_name = os.environ.get("BORINOT_WANDB_NAME") or cfg.get("run_name") or os.path.split(log_dir)[-1]
            project = cfg.get("wandb_project")
            if not project:
                raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

            entity = cfg.get("wandb_entity") or os.environ.get("WANDB_USERNAME")

            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                id=os.environ.get("WANDB_RUN_ID"),
                resume=os.environ.get("WANDB_RESUME", "allow"),
                sync_tensorboard=False,
                settings=wandb.Settings(sync_tensorboard=False),
            )

            wandb.config.update({"log_dir": log_dir})

    rsl_wandb_utils.WandbSummaryWriter = SingleStreamWandbSummaryWriter


def _resolve_resume_path(log_root_path: str, agent_cfg: RslRlBaseRunnerCfg, checkpoint_arg: str | None) -> str:
    """Resolve a checkpoint either from an explicit filesystem path or the usual run/checkpoint selectors."""

    if checkpoint_arg:
        expanded_checkpoint = Path(checkpoint_arg).expanduser()
        if expanded_checkpoint.is_file():
            return str(expanded_checkpoint.resolve())

        looks_like_path = checkpoint_arg.startswith(("~", ".", "/")) or "/" in checkpoint_arg or "\\" in checkpoint_arg
        if looks_like_path:
            raise FileNotFoundError(f"Checkpoint path does not exist: {expanded_checkpoint}")

    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _infer_input_checkpoint_name(checkpoint_path: str | None) -> str | None:
    """Build a compact W&B config label for the checkpoint used to initialize training."""

    if checkpoint_path is None:
        return None

    checkpoint = Path(checkpoint_path).expanduser()
    checkpoint_stem = checkpoint.stem
    run_dir_name = checkpoint.parent.name
    parent_run_id = run_dir_name.rsplit("_", 1)[-1].strip()
    if not checkpoint_stem:
        return None
    if re.fullmatch(r"[A-Za-z0-9]{7,8}", parent_run_id):
        return f"{parent_run_id}_{checkpoint_stem}"
    return checkpoint_stem


def _safe_run_dir_name(run_name: str) -> str:
    """Return a single filesystem path component for a generated run folder."""

    return re.sub(r"[\\/]+", "-", run_name).strip()


def _get_curriculum_state_from_runner(runner) -> dict | None:
    raw_env = getattr(runner.env, "unwrapped", None)
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


def _patch_runner_save_with_cbp(runner, cbp_manager) -> None:
    """Patch RSL-RL checkpoint saving so CBP state is uploaded with the model."""

    if getattr(runner, "_borinot_cbp_save_patch", False):
        return

    def _save_with_cbp(path: str, infos: dict | None = None) -> None:
        save_infos = infos
        if isinstance(save_infos, dict):
            save_infos = dict(save_infos)
            save_infos["continual_backprop"] = cbp_manager.summary()
        elif save_infos is None:
            save_infos = {"continual_backprop": cbp_manager.summary()}

        saved_dict = {
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": runner.current_learning_iteration,
            "infos": save_infos,
            "continual_backprop_state_dict": cbp_manager.state_dict(),
        }
        if hasattr(runner.alg, "rnd") and runner.alg.rnd:
            saved_dict["rnd_state_dict"] = runner.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = runner.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if getattr(runner, "logger_type", None) in ["neptune", "wandb"] and not getattr(runner, "disable_logs", False):
            runner.writer.save_model(path, runner.current_learning_iteration)

    runner.save = _save_with_cbp
    runner._borinot_cbp_save_patch = True


def _restore_cbp_state_from_checkpoint(cbp_manager, checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = checkpoint.get("continual_backprop_state_dict")
    if isinstance(state, dict):
        report = cbp_manager.load_state_dict(state)
        print(
            "[INFO]: Restored continual backprop state from: "
            f"{checkpoint_path} "
            f"(optimizer_steps={report['optimizer_steps']}, "
            f"groups={report['groups_loaded']}/{report['groups_total']}, "
            f"age_tensors={report['age_tensors_loaded']}/{report['groups_total']} exact"
            f", fallback={report['age_tensors_from_optimizer_steps']})."
        )
    else:
        infos = checkpoint.get("infos", {})
        cbp_summary = infos.get("continual_backprop", {}) if isinstance(infos, dict) else {}
        optimizer_steps = cbp_summary.get("optimizer_steps", None) if isinstance(cbp_summary, dict) else None
        if optimizer_steps is not None:
            report = cbp_manager.initialize_ages_from_optimizer_steps(int(optimizer_steps))
            print(
                "[INFO]: Checkpoint has only a continual backprop summary, not exact per-neuron state; "
                f"initialized CBP ages from optimizer_steps={report['optimizer_steps']} for "
                f"{report['age_tensors_from_optimizer_steps']}/{report['groups_total']} groups. "
                "Future checkpoints will save exact per-neuron ages."
            )
        else:
            print("[INFO]: Checkpoint has no continual backprop state; starting CBP ages/utilities from scratch.")


def _patch_runner_save_with_plasticity_mitigation(runner, controller) -> None:
    """Patch RSL-RL checkpoint saving with exact mitigation reference/RNG state."""

    if getattr(runner, "_borinot_plasticity_mitigation_save_patch", False):
        return

    def _save_with_plasticity_mitigation(path: str, infos: dict | None = None) -> None:
        save_infos = infos
        if isinstance(save_infos, dict):
            save_infos = dict(save_infos)
            save_infos["plasticity_mitigation"] = controller.summary()
        elif save_infos is None:
            save_infos = {"plasticity_mitigation": controller.summary()}

        saved_dict = {
            "model_state_dict": runner.alg.policy.state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": runner.current_learning_iteration,
            "infos": save_infos,
            "plasticity_mitigation_state_dict": controller.state_dict(),
        }
        if hasattr(runner.alg, "rnd") and runner.alg.rnd:
            saved_dict["rnd_state_dict"] = runner.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = runner.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if getattr(runner, "logger_type", None) in ["neptune", "wandb"] and not getattr(
            runner, "disable_logs", False
        ):
            runner.writer.save_model(path, runner.current_learning_iteration)

    runner.save = _save_with_plasticity_mitigation
    runner._borinot_plasticity_mitigation_save_patch = True


def _restore_plasticity_mitigation_state(controller, checkpoint_path: str | None) -> bool:
    """Restore the initialization reference, intervention RNG, and counters when available."""

    if checkpoint_path is None:
        return False
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = checkpoint.get("plasticity_mitigation_state_dict")
    if not isinstance(state, dict):
        print(
            "[WARN]: Resume checkpoint has no plasticity mitigation state. Treating the loaded policy "
            "as the start of a new mitigation run (new L2 reference/RNG stream).",
            flush=True,
        )
        return False
    controller.load_state_dict(state)
    print(
        "[INFO]: Restored plasticity mitigation state from "
        f"{checkpoint_path} (strategy={controller.spec.name}, "
        f"optimizer_steps={controller.optimizer_steps}, "
        f"soft_events={controller.soft_perturbations}, "
        f"boundary_events={controller.boundary_perturbations}).",
        flush=True,
    )
    return True


def _sanitize_policy_action_std(runner) -> None:
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    sanitize = getattr(policy, "sanitize_action_std_", None)
    if callable(sanitize):
        sanitize()


def _policy_action_noise_param_ids(policy) -> set[int]:
    return {
        id(param)
        for name, param in (policy.named_parameters() if policy is not None else ())
        if name.rsplit(".", 1)[-1] in ("std", "log_std")
    }


def _split_action_noise_optimizer_group(optimizer: torch.optim.Optimizer, noise_param_ids: set[int]) -> int:
    """Keep action-noise std/log_std params in decay-free groups, without duplicating groups."""

    if not noise_param_ids:
        return 0

    excluded_groups = []
    moved_params = 0
    for group in optimizer.param_groups:
        params = list(group["params"])
        excluded = [p for p in params if id(p) in noise_param_ids]
        if not excluded:
            continue

        kept = [p for p in params if id(p) not in noise_param_ids]
        if kept:
            group["params"] = kept
            excluded_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": excluded})
            moved_params += len(excluded)
        else:
            group["weight_decay"] = 0.0

    for group in excluded_groups:
        group["weight_decay"] = 0.0
        optimizer.add_param_group(group)

    return moved_params


def _checkpoint_optimizer_param_group_count(checkpoint_path: str | None) -> int | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return None
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        return None
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(param_groups, list):
        return None
    return len(param_groups)


def _match_optimizer_param_groups_to_checkpoint(runner, checkpoint_path: str | None) -> None:
    """Pre-shape optimizer groups so RSL-RL can restore optimizer state from recent checkpoints."""

    target_groups = _checkpoint_optimizer_param_group_count(checkpoint_path)
    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    if target_groups is None or optimizer is None or len(optimizer.param_groups) == target_groups:
        return

    policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)
    moved_params = _split_action_noise_optimizer_group(optimizer, _policy_action_noise_param_ids(policy))
    if len(optimizer.param_groups) == target_groups:
        print(
            "[INFO]: Matched RSL-RL optimizer param groups to checkpoint before resume "
            f"({target_groups} groups; moved {moved_params} action-noise std parameter(s))."
        )


def _apply_agent_weight_decay_to_optimizer(runner, agent_cfg: RslRlBaseRunnerCfg) -> None:
    weight_decay = float(getattr(agent_cfg, "weight_decay", 0.0) or 0.0)
    if weight_decay < 0.0:
        raise ValueError(f"agent.weight_decay must be non-negative, got {weight_decay}.")

    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    if optimizer is None:
        if weight_decay == 0.0:
            return
        raise ValueError("agent.weight_decay was set, but the selected RSL-RL runner has no optimizer.")

    # The action-noise std/log_std is exploration state, not a network weight: decaying it
    # toward zero silently shrinks exploration, so it must stay in a decay-free group.
    policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)
    noise_param_ids = _policy_action_noise_param_ids(policy)

    previous_values = {float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups}
    _split_action_noise_optimizer_group(optimizer, noise_param_ids)
    for group in optimizer.param_groups:
        if any(id(p) in noise_param_ids for p in group["params"]):
            group["weight_decay"] = 0.0
        else:
            group["weight_decay"] = weight_decay

    if weight_decay != 0.0 or previous_values != {0.0}:
        print(
            "[INFO]: Set RSL-RL optimizer weight_decay="
            f"{weight_decay:g} on "
            f"{sum(not any(id(p) in noise_param_ids for p in group['params']) for group in optimizer.param_groups)} "
            f"parameter group(s); kept {len(noise_param_ids)} action-noise std parameter(s) decay-free."
        )


def _apply_agent_adam_betas_to_optimizer(runner, agent_cfg: RslRlBaseRunnerCfg) -> None:
    beta1 = float(getattr(agent_cfg, "adam_beta1", 0.9))
    beta2 = float(getattr(agent_cfg, "adam_beta2", 0.999))
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"agent.adam_beta1 must be in [0, 1), got {beta1}.")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"agent.adam_beta2 must be in [0, 1), got {beta2}.")

    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    if optimizer is None:
        if (beta1, beta2) == (0.9, 0.999):
            return
        raise ValueError(
            "agent.adam_beta1/agent.adam_beta2 were set, but the selected RSL-RL runner has no optimizer."
        )

    betas = (beta1, beta2)
    previous_values = {
        tuple(float(beta) for beta in group.get("betas", (0.9, 0.999))) for group in optimizer.param_groups
    }
    for group in optimizer.param_groups:
        group["betas"] = betas

    if betas != (0.9, 0.999) or previous_values != {(0.9, 0.999)}:
        print(
            "[INFO]: Set RSL-RL optimizer Adam betas="
            f"({beta1:g}, {beta2:g}) on {len(optimizer.param_groups)} parameter group(s)."
        )


def _attach_continual_backprop_to_runner(runner, parsed_args):
    if not bool(getattr(parsed_args, "use_cbp", False)):
        return None
    if not isinstance(runner, OnPolicyRunner):
        raise ValueError("--use-cbp currently supports the RSL-RL OnPolicyRunner / PPO path only.")
    if not hasattr(runner.alg, "optimizer"):
        raise ValueError("--use-cbp expected runner.alg.optimizer to exist.")

    specs = collect_actor_critic_cbp_specs(runner.alg.policy)
    cbp_manager = build_continual_backprop_manager(
        specs,
        replacement_rate=float(parsed_args.cbp_replacement_rate),
        maturity_threshold=int(parsed_args.cbp_maturity_threshold),
        decay_rate=float(parsed_args.cbp_decay_rate),
        util_type=str(parsed_args.cbp_util_type),
        init=str(parsed_args.cbp_init),
        accumulate=bool(parsed_args.cbp_accumulate),
    )

    optimizer = runner.alg.optimizer
    original_step = optimizer.step

    def _step_with_cbp(*step_args, **step_kwargs):
        result = original_step(*step_args, **step_kwargs)
        cbp_manager.after_optimizer_step(optimizer)
        return result

    optimizer.step = _step_with_cbp
    optimizer._cbp_manager = cbp_manager
    runner._borinot_cbp_manager = cbp_manager
    _patch_runner_save_with_cbp(runner, cbp_manager)

    group_names = ", ".join(group.name for group in cbp_manager.groups)
    print(
        "[INFO]: Continual Backpropagation enabled "
        f"(replacement_rate={cbp_manager.replacement_rate:g}, "
        f"maturity_threshold={cbp_manager.maturity_threshold}, "
        f"decay_rate={cbp_manager.decay_rate:g}, util_type={cbp_manager.util_type}, "
        f"init={cbp_manager.init}, accumulate={cbp_manager.accumulate})."
    )
    print(f"[INFO]: CBP feature groups: {group_names}")
    return cbp_manager


def _log_cbp_stats(runner, iteration: int) -> None:
    cbp_manager = getattr(runner, "_borinot_cbp_manager", None)
    if cbp_manager is None or getattr(runner, "writer", None) is None:
        return

    last_replacements = cbp_manager.last_replacements
    total_replacements_by_group = cbp_manager.total_replacements_by_group
    runner.writer.add_scalar("CBP/optimizer_steps", cbp_manager.optimizer_steps, iteration)
    runner.writer.add_scalar("CBP/replacements_last_update", sum(last_replacements.values()), iteration)
    runner.writer.add_scalar("CBP/replacements_total", sum(total_replacements_by_group.values()), iteration)
    for group_name, total_replacements in total_replacements_by_group.items():
        runner.writer.add_scalar(f"CBP/replacements_total/{group_name}", total_replacements, iteration)


def _attach_plasticity_metrics_to_runner(runner, parsed_args) -> None:
    """Track plasticity diagnostics (arXiv:2506.03404 Appendix B) for the actor/critic MLPs."""

    if not bool(getattr(parsed_args, "plasticity_metrics", False)):
        return

    policy = getattr(runner.alg, "policy", None)
    groups = {
        name: module
        for name in ("actor", "critic")
        if isinstance(module := getattr(policy, name, None), torch.nn.Module)
    }
    if not groups or not hasattr(runner.alg, "optimizer"):
        print("[WARN]: Plasticity metrics disabled: policy has no actor/critic modules or no optimizer.", flush=True)
        return

    sample_cap = getattr(parsed_args, "plasticity_metrics_sample_cap", None)
    if sample_cap is None:
        sample_cap = getattr(getattr(runner, "env", None), "num_envs", None)
    if sample_cap is None:
        sample_cap = plasticity_metrics.DEFAULT_SAMPLE_CAP
    sample_cap = int(sample_cap)
    if sample_cap < 1:
        raise ValueError("--plasticity-metrics-sample-cap must be a positive integer.")

    grad_capture = plasticity_metrics.GradKurtosisCapture(
        {name: list(module.parameters()) for name, module in groups.items()}
    )
    grad_capture.wrap_optimizer(runner.alg.optimizer)
    runner._borinot_plasticity_groups = groups
    runner._borinot_plasticity_grad_capture = grad_capture
    runner._borinot_plasticity_activation_ok = dict.fromkeys(groups, True)
    runner._borinot_plasticity_sample_cap = sample_cap
    print(
        "[INFO]: Plasticity metrics enabled for "
        f"{', '.join(groups)} (every {int(parsed_args.plasticity_metrics_interval)} iterations; "
        f"sample_cap={sample_cap}).",
        flush=True,
    )


def _log_plasticity_metrics(runner, locs: dict, interval: int) -> None:
    groups = getattr(runner, "_borinot_plasticity_groups", None)
    grad_capture = getattr(runner, "_borinot_plasticity_grad_capture", None)
    if not groups or getattr(runner, "writer", None) is None:
        return

    it = int(locs.get("it", 0))
    interval = max(1, int(interval))
    if (it + 1) % interval == 0:
        # Captured during the next iteration's PPO update, logged at that iteration below.
        grad_capture.arm()
    if it % interval != 0:
        return

    policy = runner.alg.policy
    obs = locs.get("obs")
    activation_ok = runner._borinot_plasticity_activation_ok
    sample_cap = int(getattr(runner, "_borinot_plasticity_sample_cap", plasticity_metrics.DEFAULT_SAMPLE_CAP))
    forward_fns = {
        "actor": (lambda: policy.act_inference(obs)),
        "critic": (lambda: policy.evaluate(obs)),
    }
    for name, module in groups.items():
        scalars = {"weight_norm": plasticity_metrics.weight_norm(module.parameters())}
        kurtosis = grad_capture.last.get(name)
        if kurtosis is not None:
            scalars["grad_kurtosis"] = kurtosis
        forward_fn = forward_fns.get(name)
        if obs is not None and forward_fn is not None and activation_ok.get(name, False):
            try:
                activations = plasticity_metrics.collect_hidden_activations(module, forward_fn, sample_cap=sample_cap)
                scalars.update(plasticity_metrics.activation_plasticity_metrics(activations))
            except Exception as exc:
                activation_ok[name] = False
                print(
                    f"[WARN]: Plasticity activation metrics disabled for '{name}' "
                    f"({type(exc).__name__}: {exc}); weight norm and gradient kurtosis stay enabled.",
                    flush=True,
                )
        for key, value in scalars.items():
            if value == value:  # skip NaN
                runner.writer.add_scalar(f"Plasticity/{name}/{key}", value, it)


def _infer_checkpoint_history_dim(checkpoint_state: dict) -> int | None:
    """Infer the checkpoint TCN input dimension from the first actor encoder convolution."""

    conv_weight = checkpoint_state.get("actor_imu_encoder.conv_stack.0.conv.weight")
    if isinstance(conv_weight, torch.Tensor) and conv_weight.ndim == 3:
        return int(conv_weight.shape[1])
    return None


def _resolve_source_history_kind(checkpoint_path: str, source_history_kind: str) -> str:
    """Resolve the semantic meaning of a 24D source history when adapting to joint+IMU history."""

    if source_history_kind != "auto":
        return source_history_kind

    path_lower = checkpoint_path.lower()
    if "joint" in path_lower and "imu" not in path_lower:
        return "joint_state"
    return "foot_imu"


def _checkpoint_has_dagger_adapter_state(checkpoint_path: str | None) -> bool:
    if checkpoint_path is None:
        return False
    expanded_checkpoint = Path(checkpoint_path).expanduser()
    if not expanded_checkpoint.is_file():
        return False
    try:
        checkpoint = torch.load(expanded_checkpoint, weights_only=False, map_location="cpu")
    except Exception:
        return False
    return isinstance(checkpoint, dict) and isinstance(checkpoint.get("adapter_state_dict"), dict)


def _resolve_existing_model_path(path: str) -> str:
    expanded = Path(path).expanduser()
    candidates = [expanded]
    path_str = str(expanded)
    cluster_prefix = "/home/jbeltran/IsaacLab"
    local_prefix = "/home/jordibelp/IsaacLab"
    if path_str.startswith(cluster_prefix):
        candidates.append(Path(local_prefix + path_str[len(cluster_prefix) :]))
    elif path_str.startswith(local_prefix):
        candidates.append(Path(cluster_prefix + path_str[len(local_prefix) :]))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Checkpoint path does not exist: {expanded}")


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


def _copy_normalizer_state(target_state, target_prefix: str, source_state, source_prefix: str) -> list[str]:
    copied = []
    for suffix in ("_mean", "_var", "_std", "count"):
        target_key = f"{target_prefix}.{suffix}"
        source_key = f"{source_prefix}.{suffix}"
        if target_key in target_state and source_key in source_state:
            source_value = source_state[source_key].to(device=target_state[target_key].device)
            if tuple(source_value.shape) == tuple(target_state[target_key].shape):
                target_state[target_key] = source_value.clone()
                copied.append(target_key)
    return copied


def _copy_dagger_actor_normalizer_state(target_state, adapter_checkpoint, teacher_state, policy) -> list[str]:
    history_state = adapter_checkpoint.get("history_normalizer_state_dict")
    if not isinstance(history_state, dict):
        return []

    target_prefix = "actor_obs_normalizer"
    teacher_prefix = "actor_obs_normalizer"
    command_dim = int(getattr(policy, "command_dim", 3))
    history_dim = int(getattr(policy, "history_flat_dim"))
    copied = []

    for suffix in ("_mean", "_var", "_std"):
        target_key = f"{target_prefix}.{suffix}"
        if target_key not in target_state:
            continue
        target_value = target_state[target_key].clone()
        history_value = history_state.get(suffix)
        teacher_value = teacher_state.get(f"{teacher_prefix}.{suffix}")
        if not isinstance(history_value, torch.Tensor) or not isinstance(teacher_value, torch.Tensor):
            continue
        if history_value.shape[-1] != history_dim or teacher_value.shape[-1] < command_dim:
            continue
        target_value[:, :history_dim] = history_value.to(device=target_value.device)
        target_value[:, history_dim : history_dim + command_dim] = teacher_value[:, -command_dim:].to(
            device=target_value.device
        )
        target_state[target_key] = target_value
        copied.append(target_key)

    target_count_key = f"{target_prefix}.count"
    if target_count_key in target_state:
        count_values = []
        history_count = history_state.get("count")
        teacher_count = teacher_state.get(f"{teacher_prefix}.count")
        if torch.is_tensor(history_count):
            count_values.append(history_count.to(device=target_state[target_count_key].device))
        if torch.is_tensor(teacher_count):
            count_values.append(teacher_count.to(device=target_state[target_count_key].device))
        if count_values:
            target_state[target_count_key] = torch.stack([value.reshape(()) for value in count_values]).max()
            copied.append(target_count_key)

    return copied


def _initialize_student_from_dagger_adapter(runner, adapter_checkpoint_path: str) -> None:
    policy = runner.alg.policy
    if not hasattr(policy, "actor_history_encoder"):
        raise RuntimeError("DAgger adapter initialization requires a base-IMU student policy.")

    try:
        map_location = next(policy.parameters()).device
    except StopIteration:
        map_location = "cpu"

    adapter_checkpoint = torch.load(adapter_checkpoint_path, weights_only=False, map_location=map_location)
    adapter_state = adapter_checkpoint.get("adapter_state_dict")
    if not isinstance(adapter_state, dict):
        raise RuntimeError(f"Checkpoint is not a DAgger adapter checkpoint: {adapter_checkpoint_path}")

    saved_dims = adapter_checkpoint.get("dims", {})
    saved_sample_dim = saved_dims.get("history_sample_dim") if isinstance(saved_dims, dict) else None
    current_sample_dim = int(getattr(policy, "history_sample_dim"))
    if saved_sample_dim is not None and int(saved_sample_dim) != current_sample_dim:
        raise RuntimeError(
            "DAgger adapter IMU layout is incompatible with the current student config: "
            f"checkpoint history_sample_dim={saved_sample_dim}, current={current_sample_dim}. "
            "Use matching imu_ekf_processed_inputs/use_rotMat_on_imu_encoder settings or retrain the adapter."
        )

    teacher_checkpoint = adapter_checkpoint.get("teacher_checkpoint")
    if not isinstance(teacher_checkpoint, str):
        raise RuntimeError(f"DAgger adapter checkpoint does not contain teacher_checkpoint: {adapter_checkpoint_path}")
    teacher_checkpoint = _resolve_existing_model_path(teacher_checkpoint)
    teacher_state = _checkpoint_model_state_dict(teacher_checkpoint, map_location=map_location)

    target_state = policy.state_dict()
    copied_adapter = []
    copied_actor = []
    copied_critic = []
    copied_std = []
    skipped = []

    for key, value in adapter_state.items():
        for prefix in ("actor_history_encoder", "critic_history_encoder"):
            target_key = f"{prefix}.{key}"
            if target_key not in target_state:
                continue
            if tuple(value.shape) != tuple(target_state[target_key].shape):
                skipped.append((target_key, tuple(value.shape), tuple(target_state[target_key].shape)))
                continue
            target_state[target_key] = value.to(device=target_state[target_key].device).clone()
            copied_adapter.append(target_key)

    for source_prefix, bucket in (("actor.", copied_actor), ("critic.", copied_critic)):
        for key, value in teacher_state.items():
            if not key.startswith(source_prefix) or key not in target_state:
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped.append((key, tuple(value.shape), tuple(target_state[key].shape)))
                continue
            target_state[key] = value.to(device=target_state[key].device).clone()
            bucket.append(key)

    for key in ("std", "log_std"):
        value = teacher_state.get(key)
        if torch.is_tensor(value) and key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            target_state[key] = value.to(device=target_state[key].device).clone()
            copied_std.append(key)

    copied_actor_norm = _copy_dagger_actor_normalizer_state(target_state, adapter_checkpoint, teacher_state, policy)
    copied_critic_norm = _copy_normalizer_state(target_state, "critic_obs_normalizer", teacher_state, "critic_obs_normalizer")

    if not copied_adapter:
        raise RuntimeError(f"No adapter tensors could be loaded into the student policy from: {adapter_checkpoint_path}")
    if not copied_actor:
        raise RuntimeError(f"No teacher actor tensors could be loaded from: {teacher_checkpoint}")
    if not copied_critic:
        raise RuntimeError(f"No teacher critic tensors could be loaded from: {teacher_checkpoint}")

    policy.load_state_dict(target_state, strict=True)
    print(
        "[INFO]: Initialized base-IMU student RL policy from DAgger adapter:\n"
        f"  adapter: {adapter_checkpoint_path}\n"
        f"  teacher: {teacher_checkpoint}\n"
        f"  adapter tensors: {len(copied_adapter)}\n"
        f"  actor tensors: {len(copied_actor)}\n"
        f"  critic tensors: {len(copied_critic)}\n"
        f"  std tensors: {len(copied_std)}\n"
        f"  actor normalizer tensors: {len(copied_actor_norm)}\n"
        f"  critic normalizer tensors: {len(copied_critic_norm)}",
        flush=True,
    )
    if skipped:
        preview = ", ".join(
            f"{key}: adapter/teacher{src_shape}->student{dst_shape}" for key, src_shape, dst_shape in skipped[:8]
        )
        suffix = " ..." if len(skipped) > 8 else ""
        print(f"[WARN]: Skipped {len(skipped)} tensors due to shape mismatch: {preview}{suffix}", flush=True)


def _adapt_normalizer_buffer(
    policy,
    key: str,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    checkpoint_path: str,
    source_history_dim: int | None,
    source_history_kind: str,
) -> torch.Tensor | None:
    """Adapt observation normalizer buffers when TCN history shape changes.

    Shared current-observation stats are copied exactly. History stats are copied only when
    their channel layout is clear. Normalizer counts are always reset so the running stats
    adapt quickly to the new observation distribution.
    """

    if key.endswith(".count"):
        return torch.zeros_like(target)

    if tuple(source.shape) == tuple(target.shape):
        return source
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != 1 or target.shape[0] != 1:
        return None

    current_obs_dim = int(getattr(policy, "current_obs_dim", 0))
    target_history_dim = int(getattr(policy, "imu_dim", 0))
    if current_obs_dim <= 0 or target_history_dim <= 0:
        return None
    if source.shape[1] < current_obs_dim or target.shape[1] < current_obs_dim:
        return None

    adapted = target.clone()
    adapted[:, :current_obs_dim] = source[:, :current_obs_dim]

    source_history = source[:, current_obs_dim:]
    target_history = adapted[:, current_obs_dim:]
    if source_history.numel() == 0 or target_history.numel() == 0:
        return adapted
    if source_history_dim is None or source_history_dim <= 0:
        return adapted
    if source_history.shape[1] % source_history_dim != 0 or target_history.shape[1] % target_history_dim != 0:
        return adapted

    source_by_step = source_history.reshape(1, -1, source_history_dim)
    target_history_len = target_history.shape[1] // target_history_dim
    per_source_channel = source_by_step.mean(dim=1, keepdim=True)

    if source_history_dim == target_history_dim:
        adapted[:, current_obs_dim:] = per_source_channel.repeat(1, target_history_len, 1).reshape(1, -1)
        print(
            f"[INFO]: Adapted {key} from shape {tuple(source.shape)} to {tuple(target.shape)} "
            "by copying current-observation stats and repeating per-history-channel stats.",
            flush=True,
        )
        return adapted

    if source_history_dim == 24 and target_history_dim == 48:
        resolved_kind = _resolve_source_history_kind(checkpoint_path, source_history_kind)
        target_by_step = adapted[:, current_obs_dim:].reshape(1, target_history_len, target_history_dim)
        if resolved_kind == "joint_state":
            target_by_step[:, :, :24] = per_source_channel.repeat(1, target_history_len, 1)
        else:
            target_by_step[:, :, 24:48] = per_source_channel.repeat(1, target_history_len, 1)
        print(
            f"[INFO]: Adapted {key} from 24D {resolved_kind} history stats to 48D joint+IMU history stats; "
            "the other 24D half keeps fresh default normalizer stats.",
            flush=True,
        )
        return adapted

    print(
        f"[WARN]: Could not safely map history stats for {key}: "
        f"source_history_dim={source_history_dim}, target_history_dim={target_history_dim}. "
        "Copied only current-observation stats.",
        flush=True,
    )
    return adapted


def _reuse_actor_critic_mlp_weights(runner, checkpoint_path: str, source_history_kind: str = "auto") -> None:
    """Load compatible actor/critic MLP tensors and observation normalizer stats from a checkpoint."""

    policy = runner.alg.policy
    try:
        map_location = next(policy.parameters()).device
    except StopIteration:
        map_location = "cpu"

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
    checkpoint_state = checkpoint.get("model_state_dict")
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError(f"Checkpoint does not contain a usable model_state_dict: {checkpoint_path}")

    target_state = policy.state_dict()
    source_history_dim = _infer_checkpoint_history_dim(checkpoint_state)
    reusable_prefixes = ("actor.", "critic.")
    normalizer_prefixes = ("actor_obs_normalizer.", "critic_obs_normalizer.")
    reused_mlp = []
    reused_normalizer = []
    skipped_shape = []
    skipped_missing = []

    for key, value in checkpoint_state.items():
        if not key.startswith(reusable_prefixes + normalizer_prefixes):
            continue
        if key not in target_state:
            skipped_missing.append(key)
            continue

        target_value = target_state[key]
        adapted_value = value
        if key.startswith(normalizer_prefixes):
            adapted_value = _adapt_normalizer_buffer(
                policy,
                key,
                value,
                target_value,
                checkpoint_path=checkpoint_path,
                source_history_dim=source_history_dim,
                source_history_kind=source_history_kind,
            )
        if adapted_value is None or tuple(adapted_value.shape) != tuple(target_value.shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue

        target_state[key] = adapted_value
        if key.startswith(reusable_prefixes):
            reused_mlp.append(key)
        else:
            reused_normalizer.append(key)

    if not reused_mlp:
        raise RuntimeError(
            f"No compatible actor/critic MLP tensors could be reused from checkpoint: {checkpoint_path}"
        )

    policy.load_state_dict(target_state, strict=True)
    print(
        f"[INFO]: Reused {len(reused_mlp)} actor/critic MLP tensors and {len(reused_normalizer)} "
        f"observation-normalizer tensors from checkpoint: {checkpoint_path}. "
        "Reset normalizer counts; skipped TCN encoders, optimizer state, and learning iteration.",
        flush=True,
    )
    if skipped_shape:
        preview = ", ".join(
            f"{key}: checkpoint{src_shape}->current{dst_shape}" for key, src_shape, dst_shape in skipped_shape[:8]
        )
        suffix = " ..." if len(skipped_shape) > 8 else ""
        print(f"[WARN]: Skipped {len(skipped_shape)} tensors due to shape mismatch: {preview}{suffix}", flush=True)
    if skipped_missing:
        preview = ", ".join(skipped_missing[:8])
        suffix = " ..." if len(skipped_missing) > 8 else ""
        print(f"[WARN]: Skipped {len(skipped_missing)} tensors missing in current policy: {preview}{suffix}", flush=True)


class PeriodicEvalVideoLauncher:
    """Launch one-env play subprocesses for periodic eval videos without touching the training env."""

    def __init__(
        self,
        runner,
        log_dir: str,
        *,
        task: str,
        hydra_args: list[str],
        interval_s: float,
        episodes: int,
        speed: float,
        max_steps: int,
        seed: int | None,
        simple_video: bool,
        group_all_patches_single_bucket: bool | None,
        within_episode_fric_resample: bool,
        within_episode_fric_resample_time_range: tuple[float, float] | None,
        wandb_project: str | None,
        wandb_entity: str | None,
        wandb_run_id: str | None,
    ) -> None:
        self.runner = runner
        self.log_dir = Path(log_dir)
        self.task = str(task)
        self.hydra_args = [str(arg) for arg in hydra_args]
        self.interval_s = float(interval_s)
        self.next_eval_time_s = float(interval_s)
        self.episodes = max(1, int(episodes))
        self.speed = max(1.0e-6, float(speed))
        self.max_steps = int(max_steps)
        self.seed = None if seed is None else int(seed)
        self.simple_video = bool(simple_video)
        self.group_all_patches_single_bucket = group_all_patches_single_bucket
        self.within_episode_fric_resample = bool(within_episode_fric_resample)
        self.within_episode_fric_resample_time_range = within_episode_fric_resample_time_range
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_id = wandb_run_id
        self.checkpoint_dir = self.log_dir / "periodic_eval_checkpoints"
        self.video_dir = self.log_dir / "videos" / "periodic_eval"
        self.child_log_dir = self.log_dir / "periodic_eval_logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.child_log_dir.mkdir(parents=True, exist_ok=True)
        self.children: list[tuple[subprocess.Popen, Path]] = []

    def maybe_record(self, locs: dict) -> bool:
        if self.interval_s <= 0.0 or getattr(self.runner, "disable_logs", False):
            return False
        if getattr(self.runner, "tot_time", 0.0) + 1.0e-6 < self.next_eval_time_s:
            self._reap_children()
            return False

        while self.next_eval_time_s <= getattr(self.runner, "tot_time", 0.0) + 1.0e-6:
            self.next_eval_time_s += self.interval_s

        self._reap_children()
        active_children = [child for child, _ in self.children if child.poll() is None]
        if active_children:
            print(
                f"[WARN]: Skipping periodic eval video launch because {len(active_children)} previous "
                "video process is still running.",
                flush=True,
            )
            return False

        self._launch(locs)
        return False

    def _launch(self, locs: dict) -> None:
        iteration = int(locs.get("it", getattr(self.runner, "current_learning_iteration", 0)))
        total_time_s = int(getattr(self.runner, "tot_time", 0.0))
        checkpoint_path = self.checkpoint_dir / f"periodic_eval_it{iteration:06d}_t{total_time_s:06d}s.pt"
        video_path = self.video_dir / f"periodic_eval_it{iteration:06d}_t{total_time_s:06d}s.mp4"
        child_log_path = self.child_log_dir / f"periodic_eval_it{iteration:06d}_t{total_time_s:06d}s.log"

        try:
            self.runner.save(
                str(checkpoint_path),
                infos={
                    "periodic_eval_video": True,
                    "periodic_eval_iteration": iteration,
                    "periodic_eval_total_time_s": float(getattr(self.runner, "tot_time", 0.0)),
                    "periodic_eval_total_timesteps": getattr(self.runner, "tot_timesteps", None),
                },
            )
        except Exception as exc:
            print(f"[WARN]: Could not save periodic eval checkpoint: {type(exc).__name__}: {exc}", flush=True)
            return

        max_steps = self.max_steps
        if max_steps <= 0:
            raw_env = self.runner.env.unwrapped
            max_steps = self.episodes * int(getattr(raw_env, "max_episode_length", 1))
        episode_length_s = self._episode_length_s()

        cmd = self._build_command(
            checkpoint_path=checkpoint_path,
            video_path=video_path,
            child_log_path=child_log_path,
            iteration=iteration,
            max_steps=max_steps,
            episode_length_s=episode_length_s,
        )
        print(
            "[INFO]: Launching detached periodic eval video process; "
            f"checkpoint={checkpoint_path}, video={video_path}, log={child_log_path}",
            flush=True,
        )
        print(f"[INFO]: Periodic eval command: {shlex.join(cmd)}", flush=True)

        env = os.environ.copy()
        if self.wandb_run_id:
            env["WANDB_RUN_ID"] = self.wandb_run_id
            env["WANDB_RESUME"] = "allow"
        try:
            out = child_log_path.open("a", encoding="utf-8", buffering=1)
            child = subprocess.Popen(
                cmd,
                cwd=str(_THIS_DIR.parents[2]),
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            out.close()
            self.children.append((child, child_log_path))
        except Exception as exc:
            print(f"[WARN]: Could not launch periodic eval video process: {type(exc).__name__}: {exc}", flush=True)

    def _build_command(
        self,
        *,
        checkpoint_path: Path,
        video_path: Path,
        child_log_path: Path,
        iteration: int,
        max_steps: int,
        episode_length_s: float,
    ) -> list[str]:
        isaaclab_sh = _THIS_DIR.parents[2] / "isaaclab.sh"
        cmd = [
            str(isaaclab_sh),
            "-p",
            str(_THIS_DIR / "play_direct_race_0423.py"),
            f"--task={self.task}",
            "--checkpoint",
            str(checkpoint_path),
            "--num_envs",
            "1",
            "--headless",
            "--periodic_eval_video",
            "--periodic_eval_video_output",
            str(video_path),
            "--periodic_eval_video_episodes",
            str(self.episodes),
            "--periodic_eval_video_speed",
            f"{self.speed:g}",
            "--periodic_eval_video_max_steps",
            str(max(1, int(max_steps))),
            "--episode_length_s",
            f"{episode_length_s:g}",
            "--duration_s",
            f"{max(1, int(max_steps)) * max(1.0e-6, self._step_dt()):g}",
            "--wandb_video_step",
            str(iteration),
            "--wandb_video_key",
            "EvalVideo/periodic",
        ]
        if not self.simple_video:
            cmd.append("--enable_cameras")
        else:
            cmd.append("--simple_video")
        if self.group_all_patches_single_bucket is not None:
            cmd.append(
                "--group-all-patches-single-bucket"
                if self.group_all_patches_single_bucket
                else "--no-group-all-patches-single-bucket"
            )
        if self.within_episode_fric_resample:
            cmd.append("--within-episode-fric-resample")
            if self.within_episode_fric_resample_time_range is not None:
                min_s, max_s = self.within_episode_fric_resample_time_range
                cmd.extend(["--within-episode-fric-resample-time-range", f"{min_s:g}", f"{max_s:g}"])
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])
        if self.wandb_project and self.wandb_run_id:
            cmd.extend(
                [
                    "--wandb_upload_video",
                    "--wandb_project",
                    str(self.wandb_project),
                    "--wandb_run_id",
                    str(self.wandb_run_id),
                ]
            )
            if self.wandb_entity:
                cmd.extend(["--wandb_entity", str(self.wandb_entity)])
        cmd.extend(self.hydra_args)
        return cmd

    def _step_dt(self) -> float:
        try:
            return float(self.runner.env.unwrapped.step_dt)
        except Exception:
            return 1.0 / 50.0

    def _episode_length_s(self) -> float:
        raw_env = self.runner.env.unwrapped
        try:
            return float(raw_env.max_episode_length) * self._step_dt()
        except Exception:
            return 20.0

    def _reap_children(self) -> None:
        still_tracked = []
        for child, log_path in self.children:
            return_code = child.poll()
            if return_code is None:
                still_tracked.append((child, log_path))
                continue
            if return_code == 0:
                print(f"[INFO]: Periodic eval video process finished successfully; log={log_path}", flush=True)
            else:
                print(
                    f"[WARN]: Periodic eval video process exited with code {return_code}; log={log_path}",
                    flush=True,
                )
        self.children = still_tracked


def _patch_runner_learn_with_periodic_eval_video(runner, recorder) -> None:
    """Patch RSL-RL's learn loop so periodic eval launches can happen after logging."""

    if getattr(runner, "_borinot_periodic_eval_video_patch", False):
        return

    def _learn_with_periodic_eval_video(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        from rsl_rl.utils import store_code_state

        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            env_state_changed = recorder.maybe_record(locals())
            if env_state_changed:
                obs = self.env.get_observations().to(self.device)
                self.train_mode()
                cur_reward_sum.zero_()
                cur_episode_length.zero_()
                if self.alg.rnd:
                    cur_ereward_sum.zero_()
                    cur_ireward_sum.zero_()

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    runner.learn = _learn_with_periodic_eval_video.__get__(runner, type(runner))
    runner._borinot_periodic_eval_video_patch = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    mitigation_spec = plasticity_mitigation.resolve_strategy(args_cli.plasticity_mitigation_strategy)
    args_cli.plasticity_mitigation_strategy = mitigation_spec.name
    if args_cli.plasticity_loss_exp_reset_all and not args_cli.plasticity_loss_exp:
        raise ValueError("--plasticity-loss-exp-reset-all requires --plasticity-loss-exp.")
    if args_cli.plasticity_exp_first_layer_only and not args_cli.plasticity_loss_exp:
        raise ValueError("--plasticity-exp-first-layer-only requires --plasticity-loss-exp.")
    if args_cli.plasticity_exp_first_layer_only and args_cli.plasticity_loss_exp_reset_all:
        raise ValueError(
            "--plasticity-exp-first-layer-only and --plasticity-loss-exp-reset-all are mutually exclusive."
        )
    if args_cli.plasticity_exp_first_layer_only and args_cli.use_cbp:
        raise ValueError(
            "--plasticity-exp-first-layer-only cannot be combined with --use-cbp because CBP would "
            "modify frozen hidden layers."
        )
    if args_cli.plasticity_exp_first_layer_only and args_cli.shared_networks:
        raise ValueError(
            "--plasticity-exp-first-layer-only currently supports separate actor/critic MLPs, not --shared-networks."
        )
    if mitigation_spec.name != "none" and args_cli.plasticity_loss_exp_reset_all:
        raise ValueError(
            "--plasticity-mitigation-strategy and --plasticity-loss-exp-reset-all select different "
            "intervention conditions; run them as separate experiments."
        )
    if mitigation_spec.name != "none" and args_cli.plasticity_exp_first_layer_only:
        raise ValueError(
            "--plasticity-mitigation-strategy and --plasticity-exp-first-layer-only select different "
            "intervention conditions; run them as separate experiments."
        )
    if mitigation_spec.name != "none" and args_cli.use_cbp:
        raise ValueError(
            "--plasticity-mitigation-strategy and --use-cbp are separate mitigation conditions and cannot "
            "be combined in one controlled run."
        )
    if mitigation_spec.name != "none" and args_cli.shared_networks:
        raise ValueError(
            "Paper-style plasticity mitigation currently targets separate actor/critic MLPs, not --shared_networks."
        )
    if mitigation_spec.name != "none" and agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("Paper-style plasticity mitigation currently supports the OnPolicyRunner / PPO path only.")
    if mitigation_spec.boundary_shrink_perturb and not args_cli.plasticity_loss_exp:
        raise ValueError("The shrink-perturb strategy requires --plasticity-loss-exp distribution boundaries.")
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.shared_networks:
        if SharedActorCritic is None:
            raise RuntimeError("--shared_networks requires the Solo12 race SharedActorCritic module to import cleanly.")
        if not hasattr(agent_cfg.policy, "shared_networks"):
            raise ValueError(
                "--shared_networks was requested, but this policy config has no shared_networks field. "
                "Use a Solo12 race policy config that supports shared networks."
            )
        agent_cfg.policy.shared_networks = True
        if getattr(agent_cfg.policy, "class_name", None) == "ActorCritic":
            agent_cfg.policy.class_name = "SharedActorCritic"
    if args_cli.reuse_mlp and args_cli.checkpoint is None:
        raise ValueError("--reuse-mlp requires --checkpoint.")
    dagger_adapter_checkpoint_arg = args_cli.init_from_dagger_adapter
    if dagger_adapter_checkpoint_arg is None and _checkpoint_has_dagger_adapter_state(args_cli.checkpoint):
        dagger_adapter_checkpoint_arg = args_cli.checkpoint
        print(
            "[INFO]: --checkpoint points to a DAgger adapter checkpoint; "
            "initializing the base-IMU student policy instead of resuming RSL-RL optimizer state."
        )
    if args_cli.reuse_mlp:
        if agent_cfg.resume:
            print("[INFO]: --reuse-mlp was provided; disabling full RSL-RL resume and optimizer restore.")
        agent_cfg.resume = False
    elif dagger_adapter_checkpoint_arg is not None:
        if agent_cfg.resume:
            print("[INFO]: DAgger adapter initialization requested; disabling full RSL-RL resume.")
        agent_cfg.resume = False
    elif args_cli.checkpoint is not None and not agent_cfg.resume:
        print("[INFO]: --checkpoint was provided; enabling resume automatically.")
        agent_cfg.resume = True
    if args_cli.plasticity_loss_exp_reset_all and (
        args_cli.reuse_mlp or dagger_adapter_checkpoint_arg is not None
    ):
        raise ValueError(
            "--plasticity-loss-exp-reset-all is a from-scratch control and cannot be combined with "
            "--reuse-mlp or DAgger-adapter initialization."
        )

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    plasticity_exp_duration = None
    plasticity_exp_num_rounds = None
    if args_cli.plasticity_loss_exp:
        plasticity_exp_duration, plasticity_exp_num_rounds = _plasticity_permutation_cfg(env_cfg)
        scheduled_iterations = plasticity_exp_duration * plasticity_exp_num_rounds
        if args_cli.max_iterations is None:
            agent_cfg.max_iterations = scheduled_iterations
        elif int(args_cli.max_iterations) != scheduled_iterations:
            print(
                "[WARN]: --max_iterations overrides the full plasticity experiment duration "
                f"({int(args_cli.max_iterations)} requested vs {scheduled_iterations} configured).",
                flush=True,
            )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.within_episode_fric_resample is not None and hasattr(env_cfg, "within_episode_fric_resample"):
        env_cfg.within_episode_fric_resample = bool(args_cli.within_episode_fric_resample)
    if args_cli.within_episode_fric_resample_time_range is not None and hasattr(
        env_cfg, "within_episode_fric_resample_time_range"
    ):
        min_s, max_s = sorted(float(value) for value in args_cli.within_episode_fric_resample_time_range)
        if max_s <= 0.0:
            raise ValueError("--within-episode-fric-resample-time-range must contain at least one positive value.")
        env_cfg.within_episode_fric_resample_time_range = (max(0.0, min_s), max_s)
    if args_cli.group_all_patches_single_bucket is not None and hasattr(env_cfg, "group_all_patches_single_bucket"):
        env_cfg.group_all_patches_single_bucket = bool(args_cli.group_all_patches_single_bucket)

    if args_cli.use_cbp and args_cli.distributed:
        raise ValueError("--use-cbp is currently single-process only; CBP replacement events are not synchronized across ranks.")

    if args_cli.plasticity_loss_exp and args_cli.distributed:
        raise ValueError(
            "--plasticity-loss-exp is currently single-process only so every environment and optimizer "
            "uses exactly the same permutation boundary."
        )

    if mitigation_spec.name != "none" and args_cli.distributed:
        raise ValueError(
            "Paper-style plasticity mitigation is currently single-process only so fresh-parameter samples "
            "and regenerative references remain exactly synchronized."
        )

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. Please use GPU device (e.g., --device cuda)."
        )

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    _sync_base_imu_policy_cfg_from_env_cfg(env_cfg, agent_cfg)
    rnd_setup = solo12_rnd.configure_solo12_rnd(env_cfg, agent_cfg)
    if rnd_setup.enabled:
        print(
            "[INFO]: Solo12 RND enabled: "
            f"beta={rnd_setup.beta:g}, curiosity_state=projected_gravity+joint_position+foot_contacts "
            "(19D, clean simulator state), state_normalization=yes, reward_normalization=no, "
            "target=[5], predictor=[5, 5], output=1.",
            flush=True,
        )

    symmetry_enabled = args_cli.symmetry_mode != "none"
    symmetry_fn = None
    if symmetry_enabled:
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError("Symmetry mode currently requires the OnPolicyRunner / PPO path in RSL-RL.")
        if args_cli.task in _SOLO12_DIRECT_SYMMETRY_TASKS:
            symmetry_fn = solo12_symmetry.compute_symmetric_observations_actions
        elif args_cli.task in {
            "Isaac-Solo12-Race-Direct-v0",
            "Isaac-Solo12-Race-IMU-Direct-v0",
            "Isaac-Solo12-Race-JointStateTCN-Direct-v0",
            "Isaac-Solo12-Race-JointStates_IMU_TCN-Direct-v0",
            "Isaac-Solo12-Race-ParamsConditioned-Direct-v0",
            "Isaac-Solo12-Race-ParamsConditionedEnc-Direct-v0",
            "Solo12-Race-ParamsConditionedEnc-Direct-v0",
            "Isaac-Solo12-Race-EvalCamera-Direct-v0",
            "Isaac-Solo12-Race-IMU-EvalCamera-Direct-v0",
            "Isaac-Solo12-Race-JointStateTCN-EvalCamera-Direct-v0",
            "Isaac-Solo12-Race-JointStates_IMU_TCN-EvalCamera-Direct-v0",
            "Isaac-Solo12-Race-Vision-Direct-v0",
            "Isaac-Solo12-Race-Vision-IMU-Direct-v0",
        }:
            symmetry_fn = solo12_race_symmetry.compute_symmetric_observations_actions
        else:
            raise ValueError(
                "Symmetry mode is implemented for Solo12 direct locomotion/base-IMU tasks "
                "and the Isaac-Solo12-Race-* direct tasks only."
            )
        configured_symmetry_fn = (
            observation_permutation.compute_permutation_aware_symmetry
            if args_cli.plasticity_loss_exp
            else symmetry_fn
        )
        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=args_cli.symmetry_mode in ("augmentation", "both"),
            use_mirror_loss=args_cli.symmetry_mode in ("loss", "both"),
            mirror_loss_coeff=args_cli.symmetry_loss_coeff,
            data_augmentation_func=configured_symmetry_fn,
        )

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    base_run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Run timestamp: {base_run_name}")

    full_run_name = base_run_name
    if agent_cfg.run_name:
        full_run_name += f"_{agent_cfg.run_name}"
    if symmetry_enabled:
        symmetry_tag = f"sym-{args_cli.symmetry_mode}"
        if args_cli.symmetry_mode in ("loss", "both"):
            symmetry_tag += f"-{args_cli.symmetry_loss_coeff:g}"
        full_run_name += f"_{symmetry_tag}"
    if args_cli.use_cbp:
        full_run_name += "_cbp"
    if args_cli.plasticity_loss_exp:
        full_run_name += "_permute-plasticity"
    if args_cli.plasticity_loss_exp_reset_all:
        full_run_name += "_reset-all"
    if args_cli.plasticity_exp_first_layer_only:
        full_run_name += "_first-layer-only"
    if mitigation_spec.name != "none":
        full_run_name += f"_mit-{mitigation_spec.name}"
    if rnd_setup.enabled:
        full_run_name += f"_rnd-{rnd_setup.beta:g}"
    if getattr(agent_cfg.policy, "shared_networks", False):
        full_run_name += "_shared-networks"

    log_dir_name = full_run_name
    if agent_cfg.logger == "wandb":
        # Let rsl_rl own wandb initialization to avoid config collisions inside its writer.
        # We still pre-generate the run id so the local folder name matches the W&B run id.
        import wandb

        wandb_project = agent_cfg.wandb_project
        if not wandb_project:
            raise ValueError("W&B is enabled but agent_cfg.wandb_project is empty.")
        wandb_run_id = wandb.util.generate_id()
        os.environ["WANDB_RUN_ID"] = wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["BORINOT_WANDB_NAME"] = args_cli.wandb_name or full_run_name
        log_dir_name = f"{full_run_name}_{wandb_run_id}"
        agent_cfg.run_name = full_run_name
        if args_cli.log_project_name:
            agent_cfg.wandb_project = args_cli.log_project_name

    safe_log_dir_name = _safe_run_dir_name(log_dir_name)
    if safe_log_dir_name != log_dir_name:
        print(f"[INFO]: Sanitized run folder name: {safe_log_dir_name}")
        log_dir_name = safe_log_dir_name

    log_dir = os.path.join(log_root_path, log_dir_name)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning("IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported.")

    env_cfg.log_dir = log_dir
    # Keep the large training env in normal non-rgb mode.  Periodic eval videos temporarily
    # switch render_mode to rgb_array only while recording, which avoids the huge render-product
    # initialization cost seen on 12k-env cluster jobs.
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    env_cfg_py = inspect.getsourcefile(type(env_cfg))
    env_py = inspect.getsourcefile(env.unwrapped.__class__)
    agent_cfg_py = inspect.getsourcefile(type(agent_cfg))
    symmetry_py = inspect.getsourcefile(solo12_symmetry) if symmetry_enabled else None

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    should_resume = (agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation") and not args_cli.reuse_mlp
    resume_path = None
    if should_resume or args_cli.reuse_mlp or dagger_adapter_checkpoint_arg is not None:
        resume_path = _resolve_resume_path(
            log_root_path,
            agent_cfg,
            dagger_adapter_checkpoint_arg if dagger_adapter_checkpoint_arg is not None else args_cli.checkpoint,
        )
    input_checkpoint_name = _infer_input_checkpoint_name(resume_path)

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
    if agent_cfg.logger == "wandb":
        _patch_rsl_rl_wandb_writer_for_single_stream()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.plasticity_loss_exp:
        env = observation_permutation.ObservationPermutationVecEnv(
            env,
            round_duration=plasticity_exp_duration,
            num_rounds=plasticity_exp_num_rounds,
            seed=agent_cfg.seed,
            symmetry_fn=symmetry_fn,
        )
        print(
            "[INFO]: Plasticity-loss permutation experiment enabled "
            f"({plasticity_exp_num_rounds} mappings x {plasticity_exp_duration} learning iterations; "
            f"{env.total_learning_iterations} configured iterations total; seed={agent_cfg.seed}).",
            flush=True,
        )

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    layer_norm_result = None
    if mitigation_spec.layer_norm:
        layer_norm_result = plasticity_mitigation.insert_actor_critic_layer_norm(runner)
        print(
            "[INFO]: Inserted paper-style LayerNorm before every actor/critic hidden activation "
            f"({layer_norm_result.layer_count} layers, "
            f"{layer_norm_result.parameter_count} affine parameters).",
            flush=True,
        )

    runner.add_git_repo_to_log(__file__)
    reset_all_initial_policy_state = (
        observation_permutation.clone_state_dict(runner.alg.policy.state_dict())
        if args_cli.plasticity_loss_exp_reset_all
        else None
    )
    loaded_checkpoint_infos = None
    if args_cli.reuse_mlp:
        print(f"[INFO]: Reusing actor/critic MLP weights from checkpoint: {resume_path}")
        _reuse_actor_critic_mlp_weights(runner, resume_path, args_cli.reuse_mlp_source_history_kind)
    elif dagger_adapter_checkpoint_arg is not None:
        print(f"[INFO]: Initializing base-IMU student policy from DAgger adapter checkpoint: {resume_path}")
        _initialize_student_from_dagger_adapter(runner, resume_path)
    elif should_resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        _match_optimizer_param_groups_to_checkpoint(runner, resume_path)
        loaded_checkpoint_infos = runner.load(resume_path)
    if args_cli.plasticity_loss_exp:
        env.set_learning_iteration(runner.current_learning_iteration)
    if resume_path is not None:
        _sanitize_policy_action_std(runner)
    _apply_agent_weight_decay_to_optimizer(runner, agent_cfg)
    _apply_agent_adam_betas_to_optimizer(runner, agent_cfg)
    mitigation_controller = None
    if mitigation_spec.name != "none":
        if float(getattr(agent_cfg, "weight_decay", 0.0) or 0.0) != 0.0:
            print(
                "[WARN]: agent.weight_decay is nonzero alongside a paper-style plasticity strategy. "
                "This is allowed, but it adds ordinary L2/AdamW decay as a separate intervention.",
                flush=True,
            )
        mitigation_seed = args_cli.plasticity_mitigation_seed
        if mitigation_seed is None:
            mitigation_seed = int(agent_cfg.seed) + 1_618_033
        mitigation_controller = plasticity_mitigation.PlasticityMitigationController(
            runner,
            strategy=mitigation_spec,
            regenerative_l2_coef=float(args_cli.plasticity_regen_l2_coef),
            soft_shrink_perturb_beta=float(args_cli.plasticity_soft_sp_beta),
            shrink_perturb_beta=float(args_cli.plasticity_sp_beta),
            seed=int(mitigation_seed),
            learning_rate=float(agent_cfg.algorithm.learning_rate),
            layer_norm_count=0 if layer_norm_result is None else layer_norm_result.layer_count,
        )
        if should_resume and not args_cli.reuse_mlp:
            _restore_plasticity_mitigation_state(mitigation_controller, resume_path)
        _patch_runner_save_with_plasticity_mitigation(runner, mitigation_controller)
        print(
            "[INFO]: Plasticity mitigation enabled: "
            f"strategy={mitigation_spec.name}, target={mitigation_controller.target_parameter_count}/"
            f"{mitigation_controller.total_policy_parameter_count} policy parameters "
            f"({100.0 * mitigation_controller.target_parameter_fraction:.2f}%), "
            f"regen_coef={float(args_cli.plasticity_regen_l2_coef):g}, "
            f"soft_beta={float(args_cli.plasticity_soft_sp_beta):g}, "
            f"boundary_beta={float(args_cli.plasticity_sp_beta):g}, seed={int(mitigation_seed)}. "
            "Action-noise std/log_std and empirical-normalizer buffers are excluded.",
            flush=True,
        )
    cbp_manager = _attach_continual_backprop_to_runner(runner, args_cli)
    reset_all_initial_cbp_state = (
        copy.deepcopy(cbp_manager.state_dict())
        if args_cli.plasticity_loss_exp_reset_all and cbp_manager is not None
        else None
    )
    if cbp_manager is not None and should_resume and not args_cli.reuse_mlp:
        _restore_cbp_state_from_checkpoint(cbp_manager, resume_path)
    _attach_plasticity_metrics_to_runner(runner, args_cli)
    reset_all_controller = None
    if args_cli.plasticity_loss_exp_reset_all:
        reset_all_controller = observation_permutation.ResetAllController(
            runner,
            initial_policy_state=reset_all_initial_policy_state,
            learning_rate=float(agent_cfg.algorithm.learning_rate),
            initial_cbp_state=reset_all_initial_cbp_state,
        )
        print(
            "[INFO]: Plasticity reset-all control enabled: actor/critic layers, observation normalizers, "
            "action-noise state, optimizer state, learning rate, and CBP state (when enabled) will reset "
            "at every permutation boundary.",
            flush=True,
        )
    first_layer_only_controller = None
    if args_cli.plasticity_exp_first_layer_only:
        first_layer_only_controller = observation_permutation.FirstLayerOnlyController(runner)
        if env.round_index > 0:
            first_layer_only_controller.activate()
        print(
            "[INFO]: Plasticity first-layer-only control enabled: phase 1 trains the full actor-critic; "
            "after its first permutation boundary, only "
            f"{', '.join(first_layer_only_controller.trainable_parameter_names)} remain trainable "
            f"({first_layer_only_controller.trainable_parameter_count}/"
            f"{first_layer_only_controller.total_parameter_count} parameters; "
            f"active now={'yes' if first_layer_only_controller.active else 'no'}).",
            flush=True,
        )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "plasticity_mitigation.yaml"),
        {
            "strategy": mitigation_spec.name,
            "regenerative_l2_coef": float(args_cli.plasticity_regen_l2_coef),
            "soft_shrink_perturb_beta": float(args_cli.plasticity_soft_sp_beta),
            "shrink_perturb_beta": float(args_cli.plasticity_sp_beta),
            "seed": (
                None
                if mitigation_controller is None
                else int(mitigation_controller.seed)
            ),
            "target_parameter_count": (
                0
                if mitigation_controller is None
                else int(mitigation_controller.target_parameter_count)
            ),
            "layer_norm_count": 0 if layer_norm_result is None else int(layer_norm_result.layer_count),
            "excludes_action_noise_and_observation_normalizers": True,
        },
    )

    if agent_cfg.logger == "wandb":
        snapshot_uploaded = False

        def _upload_snapshot_once():
            nonlocal snapshot_uploaded
            if snapshot_uploaded:
                return
            _wandb_snapshot(
                log_dir=log_dir,
                env_cfg=env_cfg,
                agent_cfg=agent_cfg,
                args_cli=args_cli,
                env_cfg_py=env_cfg_py,
                env_py=env_py,
                extra_files=[
                    path
                    for path in [
                        agent_cfg_py,
                        symmetry_py,
                        __file__,
                        _THIS_DIR / "continual_backprop.py",
                        _THIS_DIR / "plasticity_metrics.py",
                        _THIS_DIR / "plasticity_mitigation.py" if mitigation_spec.name != "none" else None,
                        _THIS_DIR / "observation_permutation.py" if args_cli.plasticity_loss_exp else None,
                    ]
                    if path
                ],
            )
            if input_checkpoint_name is not None:
                import wandb

                if wandb.run is not None:
                    wandb.run.config.update(
                        {
                            "input_checkpoint_name": input_checkpoint_name,
                            "input_checkpoint_path": resume_path,
                            # Kept for older dashboards/scripts that already used this field.
                            "parent_model": input_checkpoint_name,
                        },
                        allow_val_change=True,
                    )
            snapshot_uploaded = True

        if hasattr(runner, "_prepare_logging_writer"):
            original_prepare_logging_writer = runner._prepare_logging_writer

            def _prepare_logging_writer_with_snapshot():
                original_prepare_logging_writer()
                if getattr(runner, "writer", None) is None:
                    return
                _upload_snapshot_once()

            runner._prepare_logging_writer = _prepare_logging_writer_with_snapshot
        elif hasattr(getattr(runner, "logger", None), "init_logging_writer"):
            original_init_logging_writer = runner.logger.init_logging_writer

            def _init_logging_writer_with_snapshot():
                original_init_logging_writer()
                _upload_snapshot_once()

            runner.logger.init_logging_writer = _init_logging_writer_with_snapshot
        else:
            print("[WARN] Could not hook into the runner's W&B initialization; skipping code snapshot upload.")

    best_mean_reward = float("-inf")
    carry_best_from_checkpoint = False
    if resume_path is not None:
        carry_best_from_checkpoint = os.path.abspath(os.path.dirname(resume_path)) == os.path.abspath(log_dir)

    if isinstance(loaded_checkpoint_infos, dict):
        previous_best_reward = loaded_checkpoint_infos.get("best_model_value")
        if isinstance(previous_best_reward, (int, float)) and carry_best_from_checkpoint:
            best_mean_reward = float(previous_best_reward)
        elif isinstance(previous_best_reward, (int, float)) and resume_path is not None:
            print(
                "[INFO]: Resuming from a checkpoint in a different run directory; resetting best-model tracking "
                f"for this new run (loaded best Train/mean_reward={float(previous_best_reward):.4f} from {resume_path})."
            )

    original_log = runner.log
    best_model_path = os.path.join(log_dir, "best_model.pt")
    best_mean_reward_by_curriculum_idx: dict[int, float] = {}
    last_curriculum_idx: int | None = None
    mean_reward_history: deque[float] = deque(maxlen=50)

    def _log_with_best_model(*log_args, **log_kwargs):
        nonlocal best_mean_reward, last_curriculum_idx
        original_log(*log_args, **log_kwargs)

        if runner.log_dir is None or getattr(runner, "disable_logs", False):
            return
        if not log_args:
            return

        locs = log_args[0]
        if getattr(runner, "writer", None) is not None:
            _log_plasticity_metrics(runner, locs, args_cli.plasticity_metrics_interval)
        rewbuffer = locs.get("rewbuffer")
        if rewbuffer is None or len(rewbuffer) == 0:
            return

        mean_reward = float(statistics.mean(rewbuffer))
        mean_reward_history.append(mean_reward)
        mean_reward_smooth = float(statistics.median(mean_reward_history))

        if getattr(runner, "writer", None) is not None:
            runner.writer.add_scalar("Train/mean_reward_smooth", mean_reward_smooth, locs["it"])
            if getattr(runner, "logger_type", None) != "wandb":
                runner.writer.add_scalar("Train/mean_reward_smooth/time", mean_reward_smooth, runner.tot_time)
            _log_cbp_stats(runner, locs["it"])

        curriculum_state = _get_curriculum_state_from_runner(runner)
        curriculum_idx = None if curriculum_state is None else curriculum_state["global_idx"]
        if curriculum_idx is not None and getattr(runner, "writer", None) is not None:
            runner.writer.add_scalar("Curriculum/global_idx", curriculum_idx, locs["it"])

        if curriculum_idx is None:
            previous_best_reward = best_mean_reward
        else:
            if last_curriculum_idx is not None and curriculum_idx != last_curriculum_idx:
                print(
                    "[INFO]: Curriculum advanced "
                    f"{last_curriculum_idx} -> {curriculum_idx}; resetting best-model tracking for the new stage."
                )
            last_curriculum_idx = curriculum_idx
            previous_best_reward = best_mean_reward_by_curriculum_idx.get(curriculum_idx, float("-inf"))

        if mean_reward <= previous_best_reward:
            return

        if curriculum_idx is None:
            best_mean_reward = mean_reward
            curriculum_model_path = None
        else:
            best_mean_reward_by_curriculum_idx[curriculum_idx] = mean_reward
            curriculum_model_path = os.path.join(log_dir, f"best_model_curriculum_idx_{curriculum_idx}.pt")

        best_infos = {
            "best_model_metric": "Train/mean_reward",
            "best_model_value": mean_reward,
            "best_model_iteration": locs.get("it"),
            "best_model_total_timesteps": getattr(runner, "tot_timesteps", None),
            "best_model_total_time": getattr(runner, "tot_time", None),
            "source_checkpoint": resume_path,
        }
        if curriculum_state is not None:
            best_infos.update(
                {
                    "best_model_curriculum_idx": curriculum_idx,
                    "curriculum_global_idx": curriculum_state["global_idx"],
                    "curriculum_max_velx_range_idx": curriculum_state["max_velx_range_idx"],
                    "curriculum_base_push_force_idx": curriculum_state["base_push_force_idx"],
                    "curriculum_command_lin_vel_x_abs": curriculum_state["command_lin_vel_x_abs"],
                    "curriculum_base_push_force_xy_abs": curriculum_state["base_push_force_xy_abs"],
                }
            )
            runner.save(curriculum_model_path, infos=best_infos)
        runner.save(best_model_path, infos=best_infos)

        destination = best_model_path if curriculum_model_path is None else f"{best_model_path} and {curriculum_model_path}"
        print(
            f"[INFO]: Saved new best model to {destination} "
            f"(iteration={locs.get('it')}, Train/mean_reward={mean_reward:.4f}, "
            f"Train/mean_reward_smooth={mean_reward_smooth:.4f})"
        )

    runner.log = _log_with_best_model

    if mitigation_controller is not None:
        log_before_mitigation_metrics = runner.log

        def _log_with_mitigation_metrics(*log_args, **log_kwargs):
            log_before_mitigation_metrics(*log_args, **log_kwargs)
            if not log_args or getattr(runner, "writer", None) is None:
                return
            iteration = int(log_args[0]["it"])
            for name, value in mitigation_controller.logging_values().items():
                runner.writer.add_scalar(f"PlasticityMitigation/{name}", value, iteration)

        runner.log = _log_with_mitigation_metrics

    if args_cli.plasticity_loss_exp:
        log_before_permutation_update = runner.log

        def _log_with_permutation_update(*log_args, **log_kwargs):
            log_before_permutation_update(*log_args, **log_kwargs)
            if not log_args:
                return

            locs = log_args[0]
            iteration = int(locs["it"])
            permutation_boundary = (
                (iteration + 1) % env.round_duration == 0
                and env.round_index < env.num_rounds - 1
            )
            first_layer_freeze_event = bool(
                permutation_boundary
                and first_layer_only_controller is not None
                and not first_layer_only_controller.active
            )
            if getattr(runner, "writer", None) is not None:
                for name, value in env.logging_values(iteration).items():
                    runner.writer.add_scalar(f"PlasticityExperiment/{name}", value, iteration)
                runner.writer.add_scalar(
                    "PlasticityExperiment/reset_all_enabled",
                    int(reset_all_controller is not None),
                    iteration,
                )
                runner.writer.add_scalar(
                    "PlasticityExperiment/network_resets_total",
                    0 if reset_all_controller is None else reset_all_controller.reset_count,
                    iteration,
                )
                runner.writer.add_scalar(
                    "PlasticityExperiment/network_reset_event",
                    int(permutation_boundary and reset_all_controller is not None),
                    iteration,
                )
                runner.writer.add_scalar(
                    "PlasticityExperiment/first_layer_only_enabled",
                    int(first_layer_only_controller is not None),
                    iteration,
                )
                runner.writer.add_scalar(
                    "PlasticityExperiment/first_layer_only_active",
                    int(first_layer_only_controller is not None and first_layer_only_controller.active),
                    iteration,
                )
                runner.writer.add_scalar(
                    "PlasticityExperiment/first_layer_only_freeze_event",
                    int(first_layer_freeze_event),
                    iteration,
                )
                if first_layer_only_controller is not None:
                    runner.writer.add_scalar(
                        "PlasticityExperiment/first_layer_trainable_parameter_fraction",
                        first_layer_only_controller.trainable_parameter_fraction,
                        iteration,
                    )
                runner.writer.add_scalar(
                    "PlasticityMitigation/boundary_event",
                    int(permutation_boundary and mitigation_spec.boundary_shrink_perturb),
                    iteration,
                )

            if env.finish_learning_iteration(iteration, locs["obs"]):
                if reset_all_controller is not None:
                    reset_all_controller.reset()
                mitigation_applied = (
                    mitigation_controller.on_distribution_shift()
                    if mitigation_controller is not None
                    else False
                )
                first_layer_activated = (
                    first_layer_only_controller.activate()
                    if first_layer_only_controller is not None
                    else False
                )
                print(
                    "[INFO]: Plasticity experiment switched observation mapping after learning iteration "
                    f"{iteration} (next permutation index={env.round_index}, "
                    f"permutations seen={env.round_index + 1}/{env.num_rounds}, "
                    f"network reset={'yes' if reset_all_controller is not None else 'no'}, "
                    f"mitigation event={'yes' if mitigation_applied else 'no'}, "
                    f"first-layer-only activated={'yes' if first_layer_activated else 'no'}).",
                    flush=True,
                )

        runner.log = _log_with_permutation_update

    if PERIODIC_EVAL_VIDEO_REQUESTED:
        if not isinstance(runner, OnPolicyRunner):
            print("[WARN]: Periodic eval videos are currently supported only for OnPolicyRunner; disabling.", flush=True)
        elif args_cli.distributed:
            print("[WARN]: Periodic eval videos are disabled for distributed training to avoid rank desync.", flush=True)
        else:
            periodic_eval_recorder = PeriodicEvalVideoLauncher(
                runner,
                log_dir,
                task=args_cli.task,
                hydra_args=hydra_args,
                interval_s=float(args_cli.periodic_eval_video_interval_minutes) * 60.0,
                episodes=int(args_cli.periodic_eval_video_episodes),
                speed=float(args_cli.periodic_eval_video_speed),
                max_steps=int(args_cli.periodic_eval_video_max_steps),
                seed=args_cli.seed,
                simple_video=bool(args_cli.periodic_eval_simple_video),
                group_all_patches_single_bucket=(
                    bool(env_cfg.group_all_patches_single_bucket)
                    if hasattr(env_cfg, "group_all_patches_single_bucket")
                    else None
                ),
                within_episode_fric_resample=bool(getattr(env_cfg, "within_episode_fric_resample", False)),
                within_episode_fric_resample_time_range=getattr(
                    env_cfg, "within_episode_fric_resample_time_range", None
                ),
                wandb_project=agent_cfg.wandb_project if agent_cfg.logger == "wandb" else None,
                wandb_entity=args_cli.wandb_entity or getattr(agent_cfg, "wandb_entity", None),
                wandb_run_id=os.environ.get("WANDB_RUN_ID") if agent_cfg.logger == "wandb" else None,
            )
            _patch_runner_learn_with_periodic_eval_video(runner, periodic_eval_recorder)
            estimated_iterations = max(
                1,
                int(round(float(args_cli.periodic_eval_video_interval_minutes) * 60.0 * 0.145)),
            )
            print(
                "[INFO]: Detached periodic Solo12 race eval videos enabled: "
                f"every {float(args_cli.periodic_eval_video_interval_minutes):g} training minutes "
                f"(~{estimated_iterations} iterations using recent cluster timing), "
                f"launching one-env play subprocesses for {int(args_cli.periodic_eval_video_episodes)} episodes, "
                f"{float(args_cli.periodic_eval_video_speed):g}x playback speed, "
                f"mode={'simple top-down' if bool(args_cli.periodic_eval_simple_video) else 'RGB'}, "
                f"within-episode friction resample={bool(getattr(env_cfg, 'within_episode_fric_resample', False))}.",
                flush=True,
            )

    num_learning_iterations = int(agent_cfg.max_iterations)
    if args_cli.plasticity_loss_exp and should_resume:
        num_learning_iterations = max(0, num_learning_iterations - int(runner.current_learning_iteration))
        print(
            "[INFO]: Resuming permutation experiment with "
            f"{num_learning_iterations} learning iterations remaining.",
            flush=True,
        )
    if num_learning_iterations > 0:
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
    else:
        print("[INFO]: Plasticity permutation schedule is already complete; no training iterations remain.")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
