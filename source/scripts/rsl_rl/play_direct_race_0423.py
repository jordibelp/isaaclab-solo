# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Solo12 race policy with RSL-RL.

Example:
./isaaclab.sh -p source/scripts/rsl_rl/play_direct_race_0423.py \
  --task="Isaac-Solo12-Race-Direct-v0" \
  --checkpoint "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0423_c3zhc10j_best_model.pt" \
  --num_envs 1 --duration_s 2000 --episode_length_s 80 --headless

DAgger adapter example:
./isaaclab.sh -p source/scripts/rsl_rl/play_direct_race_0423.py \
  --task="Solo12-Race-ParamsConditionedEnc-Direct-v0" \
  --checkpoint "/path/to/adapter_best.pt" \
  --dagger-teacher-checkpoint "/path/to/params_conditioned_enc_best_model.pt" \
  --num_envs 1 --duration_s 2000 --episode_length_s 80
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_UPSTREAM_RSL_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "reinforcement_learning" / "rsl_rl"
_UPSTREAM_SKRL_HELPERS_DIR = Path(__file__).resolve().parents[1] / "skrl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _import_race_source_sync_helpers():
    """Import optional W&B source-swap helpers lazily.

    The default play path should keep the pre-AppLauncher environment as close
    as possible to the known-good non-race play script. These helpers are only
    needed when --swap_wandb_files is explicitly requested.
    """

    if str(_UPSTREAM_SKRL_HELPERS_DIR) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM_SKRL_HELPERS_DIR))
    from helpers import (  # noqa: PLC0415
        restore_solo12_race_sources,
        solo12_race_sources_need_restore,
        sync_solo12_race_sources_from_wandb,
    )

    return restore_solo12_race_sources, solo12_race_sources_need_restore, sync_solo12_race_sources_from_wandb


def _detect_primary_screen_size(default: tuple[int, int] = (2560, 1600)) -> tuple[int, int]:
    """Detect the active desktop size before Isaac/Kit starts."""

    try:
        output = subprocess.check_output(["xrandr", "--current"], text=True, stderr=subprocess.DEVNULL, timeout=1.0)
        match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return default


def _detect_requested_window_size() -> tuple[int, int]:
    """Return the optional large play-window size.

    Keep this lazy: probing desktop geometry before Kit starts has been flaky on
    some Linux desktop sessions, and the default play path should look like a
    normal Isaac Sim launch.
    """

    screen_width, screen_height = _detect_primary_screen_size()
    return screen_width, max(720, screen_height - 40)


_REQUESTED_WINDOW_SIZE: tuple[int, int] | None = None


def _optional_positive_int(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer or None, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer or None, got {value!r}")
    return parsed


def _optional_positive_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive number or None, got {value!r}") from exc
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive number or None, got {value!r}")
    return parsed

parser = argparse.ArgumentParser(description="Play a trained Solo12 race policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=400, help="Recorded video length in steps.")
parser.add_argument(
    "--periodic_eval_video",
    "--periodic-eval-video",
    action="store_true",
    default=False,
    help="Record a dedicated one-env periodic training eval video and optionally upload it to W&B.",
)
parser.add_argument(
    "--periodic_eval_video_output",
    "--periodic-eval-video-output",
    type=str,
    default=None,
    help="Output MP4 path for --periodic_eval_video.",
)
parser.add_argument(
    "--periodic_eval_video_episodes",
    "--periodic-eval-video-episodes",
    type=int,
    default=2,
    help="Completed env-0 episodes to include in --periodic_eval_video.",
)
parser.add_argument(
    "--periodic_eval_video_speed",
    "--periodic-eval-video-speed",
    type=float,
    default=0.5,
    help="Playback speed multiplier for --periodic_eval_video. 0.5 writes half-speed MP4s.",
)
parser.add_argument(
    "--periodic_eval_video_max_steps",
    "--periodic-eval-video-max-steps",
    type=int,
    default=0,
    help="Safety cap in policy steps for --periodic_eval_video. 0 uses duration_s.",
)
parser.add_argument(
    "--simple_video",
    "--simple-video",
    "--periodic_eval_simple_video",
    "--periodic-eval-simple-video",
    action="store_true",
    default=False,
    help="For --periodic_eval_video, write the lightweight top-down state-space visualization instead of RGB.",
)
parser.add_argument(
    "--wandb_upload_video",
    "--wandb-upload-video",
    action="store_true",
    default=False,
    help="Upload the periodic eval video to W&B after recording.",
)
parser.add_argument("--wandb_project", "--wandb-project", type=str, default="borinotIsaacLab")
parser.add_argument("--wandb_entity", "--wandb-entity", type=str, default="jordibelp")
parser.add_argument("--wandb_run_id", "--wandb-run-id", type=str, default=None)
parser.add_argument(
    "--wandb_video_step",
    "--wandb-video-step",
    type=int,
    default=None,
    help="Training iteration recorded as EvalVideo/trigger_iteration metadata; not used as explicit W&B step.",
)
parser.add_argument("--wandb_video_key", "--wandb-video-key", type=str, default="EvalVideo/periodic")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Solo12-Race-Direct-v0",
    help="Race task name, e.g. Isaac-Solo12-Race-Direct-v0 or Isaac-Solo12-Race-IMU-Direct-v0.",
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--friction_seed",
    type=int,
    default=None,
    help=argparse.SUPPRESS,  # Deprecated: --seed now controls play friction too.
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument(
    "--play-speed",
    "--play_speed",
    type=float,
    default=1.0,
    help=(
        "Interactive playback speed multiplier. 1.0 follows sim time in wall time, 0.5 is half speed, "
        "and 1.25 is 25%% faster. Can also be changed from the Isaac UI during play."
    ),
)
parser.add_argument("--duration_s", type=float, default=120.0, help="How long to run the policy.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=80.0,
    help="Play-only environment episode timeout in seconds. This overrides the task config during inference.",
)
parser.add_argument(
    "--keep_training_stochasticity",
    action="store_true",
    default=False,
    help=(
        "Keep training-time stochasticity during play. By default, play disables domain randomization events, "
        "observation corruption, reset velocity randomization, and actuation delay. Race patch friction remains randomized "
        "when the task config enables randomize_fric_coefs."
    ),
)
parser.add_argument(
    "--training_wandb_project",
    type=str,
    default="borinotIsaacLab",
    help="W&B project that contains the training run used to produce the checkpoint.",
)
parser.add_argument(
    "--training_wandb_entity",
    type=str,
    default="jordibelp",
    help="W&B entity/user that owns the training run used to produce the checkpoint.",
)
parser.add_argument(
    "--training_wandb_run_id",
    type=str,
    default=None,
    help="Explicit W&B training run id. If omitted, infer it from the checkpoint filename.",
)
parser.add_argument(
    "--swap_wandb_files",
    action="store_true",
    default=False,
    help="Temporarily swap Solo12 race env/cfg source files from the checkpoint's W&B training snapshot.",
)
parser.add_argument(
    "--force_training_source_sync",
    action="store_true",
    default=False,
    help="Force --swap_wandb_files even if the checkpoint is not in logs/skrl/checkpoints.",
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
parser.add_argument(
    "--print_patch_friction",
    action="store_true",
    default=False,
    help="Print the active per-patch friction coefficients for env 0.",
)
parser.add_argument(
    "--no_friction_popup",
    action="store_true",
    default=False,
    help="Disable the interactive popup that shows patch friction when a colored patch prim is selected.",
)
parser.add_argument(
    "--no_foot_friction_popup",
    action="store_true",
    default=False,
    help="Disable the live popup that shows the friction coefficient below each foot.",
)
parser.add_argument(
    "--visualize-slip",
    "--visualize_slip",
    action="store_true",
    default=False,
    help="Show a live per-foot friction-cone/slip visualization during interactive play.",
)
parser.add_argument(
    "--visualize-slip-speed-threshold",
    "--visualize_slip_speed_threshold",
    type=float,
    default=0.03,
    help=(
        "Friction-opposed contact-speed threshold in m/s used by --visualize-slip to split OK contact from slip."
    ),
)
parser.add_argument(
    "--visualize-slip-log",
    "--visualize_slip_log",
    dest="visualize_slip_log",
    type=str,
    nargs="?",
    const="auto",
    default="auto",
    help=(
        "For --visualize-slip, write a physics-rate (per-substep) per-foot CSV time series of "
        "friction-opposed speed and friction-cone angles for offline angle-vs-time plots. Pass a path to "
        "override the default <log_dir>/slip_logs/<checkpoint>_<timestamp>.csv, or 'none' to disable."
    ),
)
parser.add_argument(
    "--contact-stats-ui-viz",
    "--contact_stats_ui_viz",
    dest="contact_stats_ui_viz",
    type=str,
    choices=("max", "mean", "median"),
    default="max",
    help=(
        "For --visualize-slip, how the live UI summarizes a foot's friction-opposed speed: 'max' (worst substep in the "
        "current policy step, default), or 'mean'/'median' of the speed over the whole ongoing contact. Use mean/median "
        "to suppress the brief high speed measured right at touchdown. The CSV log always keeps raw per-substep values."
    ),
)
parser.add_argument(
    "--viz-superior-fric-markers",
    "--viz_superior_fric_markers",
    action="store_true",
    default=False,
    help=(
        "For --visualize-slip, draw the single friction-status marker set in the higher body-relative positions "
        "instead of close to the feet/floor."
    ),
)
parser.add_argument(
    "--viz-air-points",
    "--viz_air_points",
    action="store_true",
    default=False,
    help=(
        "For --visualize-slip, show airborne feet as air/no-contact rows. By default, each foot keeps showing its "
        "last contact measurement until contact is detected again."
    ),
)
parser.add_argument(
    "--generate-slip-plots",
    "--generate_slip_plots",
    dest="generate_slip_plots",
    action="store_true",
    default=False,
    help=(
        "Run headless, record the per-substep slip CSV (implies --headless and --visualize-slip), then save per-foot "
        "slip plots (PNG) next to the CSV when done and open them interactively for zooming."
    ),
)
parser.add_argument(
    "--slip-plots-output",
    "--slip_plots_output",
    dest="slip_plots_output",
    type=str,
    default=None,
    help="For --generate-slip-plots, override the PNG output path. Defaults to the slip CSV path with a .png suffix.",
)
parser.add_argument(
    "--slip-plots-no-window",
    "--slip_plots_no_window",
    dest="slip_plots_no_window",
    action="store_true",
    default=False,
    help="For --generate-slip-plots, only save the PNG/CSV and skip auto-opening the interactive matplotlib window.",
)
parser.add_argument(
    "--slip-plots-min-samples",
    "--slip_plots_min_samples",
    dest="slip_plots_min_samples",
    type=int,
    default=1,
    help=(
        "For --generate-slip-plots, ignore contact episodes with fewer than this many physics samples in the "
        "generated PNG/interactive plot. Use 2 or 3 to hide one-sample contact fringe outliers."
    ),
)
parser.add_argument(
    "--disable_slider_friction",
    "--disable-slider-friction",
    action="store_true",
    default=False,
    help=(
        "Disable the play-mode friction slider. When group_all_patches_single_bucket=True, grouped patch friction "
        "then keeps the normal random bucket sampling on reset."
    ),
)
parser.add_argument(
    "--within-episode-fric-resample",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override Solo12 race within-episode patch-friction resampling during play.",
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
    help="Override whether all Solo12 race patches share one sampled friction bucket at startup.",
)
parser.add_argument(
    "--keep_isaac_panels",
    action="store_true",
    default=False,
    help="Keep the default Isaac editor panels visible. By default play hides them so the viewport fills the window.",
)
parser.add_argument(
    "--disable_window_resize",
    action="store_true",
    default=True,
    help="Use Isaac Sim's default startup window size. This is the default safe path.",
)
parser.add_argument(
    "--enable_window_resize",
    dest="disable_window_resize",
    action="store_false",
    help="Opt into best-effort large-window resizing after Isaac Sim has started.",
)
parser.add_argument(
    "--light_rig",
    type=str,
    default="Default",
    help="Viewport light rig to apply in interactive mode. Use an empty string to keep Isaac's current lighting.",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=True,
    help="Keep a side-follow camera attached to the robot yaw frame.",
)
parser.add_argument(
    "--no_follow_camera",
    action="store_true",
    default=False,
    help="Disable the follow camera.",
)
parser.add_argument(
    "--camera_eye_b",
    type=float,
    nargs=3,
    default=(0.0, -2.6, 1.1),
    metavar=("X", "Y", "Z"),
    help="Camera eye offset in the robot yaw frame (body-like frame).",
)
parser.add_argument(
    "--camera_lookat_b",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.35),
    metavar=("X", "Y", "Z"),
    help="Camera look-at offset in the robot yaw frame.",
)
parser.add_argument(
    "--free_cam",
    action="store_true",
    default=False,
    help="Start with a fixed overhead camera and leave it free for manual viewport control.",
)
parser.add_argument(
    "--free_cam_eye",
    type=float,
    nargs=3,
    default=(4.5, 0.0, 9.0),
    metavar=("X", "Y", "Z"),
    help="World-frame eye position for --free_cam.",
)
parser.add_argument(
    "--free_cam_lookat",
    type=float,
    nargs=3,
    default=(4.5, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="World-frame look-at point for --free_cam.",
)
parser.add_argument(
    "--free_cam_heading_deg",
    type=float,
    default=-90.0,
    help="Absolute top-down free-camera heading in degrees. This sets the default orientation without accumulating rotations.",
)
parser.add_argument(
    "--tsne_encoding_viz",
    "--tsne-encoding-viz",
    action="store_true",
    default=False,
    help=(
        "Record a 2D latent-encoding projection video during play. "
        "Supports env-param encoder z, DAgger adapter z_hat, and TCN latents when available."
    ),
)
parser.add_argument(
    "--tsne_viz_interval_s",
    type=float,
    default=0.1,
    help="Sim-time interval between latent samples/MP4 frames when --tsne_encoding_viz is enabled.",
)
parser.add_argument(
    "--tsne_viz_dir",
    type=str,
    default=None,
    help="Output directory for latent projection video/CSV. Defaults to <log_dir>/encoding_viz/<checkpoint>_<timestamp>.",
)
parser.add_argument(
    "--tsne_num_episodes",
    "--tsne-num-episodes",
    "--tsne_episodes",
    "--tsne-episodes",
    dest="tsne_num_episodes",
    type=_optional_positive_int,
    nargs="?",
    const=3,
    default=None,
    help=(
        "When set with --tsne_encoding_viz, stop after this many completed episodes for --tsne_viz_env_index. "
        "Use --tsne_num_episodes 5 for a short recording; omit it or pass None to keep --duration_s behavior. "
        "The older --tsne_episodes name is kept as an alias."
    ),
)
parser.add_argument(
    "--tsne_viz_video_fps",
    "--tsne-viz-video-fps",
    type=_optional_positive_float,
    default=None,
    help=(
        "FPS for the encoding visualization MP4. Default matches --tsne_viz_interval_s, "
        "so a 0.1s interval records a 10 FPS video."
    ),
)
parser.add_argument(
    "--tsne_viz_max_points",
    type=int,
    default=5000,
    help="Maximum latent samples to keep in the final fixed projection video.",
)
parser.add_argument(
    "--tsne_viz_method",
    type=str,
    choices=("tsne", "pca"),
    default="tsne",
    help="2D projection method for the live snapshots. PCA is cheaper and temporally steadier; t-SNE is the default.",
)
parser.add_argument(
    "--tsne_viz_perplexity",
    type=float,
    default=30.0,
    help="Requested t-SNE perplexity. It is automatically reduced when there are few samples.",
)
parser.add_argument(
    "--tsne_viz_max_iter",
    type=int,
    default=350,
    help="Maximum t-SNE optimization iterations per snapshot. Lower values are faster but rougher.",
)
parser.add_argument(
    "--tsne_viz_env_index",
    type=int,
    default=0,
    help="Environment index whose latent encoding is visualized.",
)
parser.add_argument(
    "--no-reset-samples-tsne",
    action="store_true",
    default=False,
    help="If set, accumulate t-SNE samples across episodes. Otherwise, reset samples each episode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _hydra_override_value(key: str) -> str | None:
    for raw_arg in hydra_args:
        if "=" not in raw_arg:
            continue
        raw_key, raw_value = raw_arg.split("=", 1)
        if raw_key.lstrip("+~") == key:
            return raw_value
    return None


def _hydra_bool_override(key: str) -> bool | None:
    value = _hydra_override_value(key)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


if args_cli.checkpoint is None:
    parser.error("--checkpoint is required for play")

if bool(args_cli.generate_slip_plots):
    # Plot generation is an offline batch path: force headless + slip logging so the per-substep
    # CSV is produced without spinning up the interactive viewport, then save/open the plots when done.
    args_cli.headless = True
    args_cli.visualize_slip = True
    print("[INFO] --generate-slip-plots: forcing --headless and --visualize-slip for offline slip plotting.", flush=True)

if args_cli.video or (args_cli.periodic_eval_video and not bool(args_cli.simple_video)):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Do not add or pass window/viewport sizing into SimulationApp startup. On some desktop sessions,
# Kit can segfault before opening the app when these keys are present in the launcher config.
# Apply the large-window preference later with a best-effort runtime resize instead.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if not bool(args_cli.headless) and bool(args_cli.disable_window_resize):
    print("[INFO] Using Isaac Sim default startup window size (--disable_window_resize).", flush=True)
elif not bool(args_cli.headless):
    _REQUESTED_WINDOW_SIZE = _detect_requested_window_size()
    print(
        f"[INFO] Will request large Isaac window after startup: "
        f"window={int(_REQUESTED_WINDOW_SIZE[0])}x{int(_REQUESTED_WINDOW_SIZE[1])}",
        flush=True,
    )

if args_cli.swap_wandb_files:
    _, _, sync_solo12_race_sources_from_wandb = _import_race_source_sync_helpers()
    source_sync_run_id = sync_solo12_race_sources_from_wandb(
        checkpoint_path=args_cli.checkpoint,
        task=args_cli.task,
        entity=args_cli.training_wandb_entity,
        project=args_cli.training_wandb_project,
        run_id=args_cli.training_wandb_run_id,
        force=args_cli.force_training_source_sync,
    )
else:
    source_sync_run_id = None

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
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


def _try_resize_app_window(width: int, height: int):
    """Best-effort runtime resize for Kit's OS window while keeping omni.ui overlays visible."""

    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set_int("/app/window/x", 0)
        settings.set_int("/app/window/y", 0)
        settings.set_int("/app/window/width", int(width))
        settings.set_int("/app/window/height", int(height))
        settings.set_int("/app/renderer/resolution/width", int(width))
        settings.set_int("/app/renderer/resolution/height", int(height))
    except Exception:
        pass

    try:
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        for method_name in ("set_window_size", "resize"):
            method = getattr(app_window, method_name, None)
            if callable(method):
                method(int(width), int(height))
                break
        for method_name in ("set_window_pos", "move"):
            method = getattr(app_window, method_name, None)
            if callable(method):
                method(0, 0)
                break
    except Exception:
        pass


def _try_resize_requested_app_window():
    """Best-effort optional large-window resize after Kit has started."""

    if bool(args_cli.disable_window_resize) or _REQUESTED_WINDOW_SIZE is None:
        return
    _try_resize_app_window(int(_REQUESTED_WINDOW_SIZE[0]), int(_REQUESTED_WINDOW_SIZE[1]))


_PLAY_VIEW_LAYOUT_APPLIED = False


def _apply_play_view_layout():
    """Hide editor panels so the viewport fills the Isaac window while custom popups stay visible."""

    global _PLAY_VIEW_LAYOUT_APPLIED
    if _PLAY_VIEW_LAYOUT_APPLIED:
        return

    try:
        import omni.ui as ui

        # These are the default editor panes visible in the full Isaac Sim layout. Hiding them lets the viewport
        # reclaim the application area without using F11, so floating omni.ui popups remain visible.
        panel_names = (
            "Stage",
            "Layer",
            "Property",
            "Properties",
            "Content",
            "Content Browser",
            "Console",
            "Render Settings",
            "Simulation Settings",
            "Semantics Schema Editor",
            "Isaac Lab",
        )
        for name in panel_names:
            try:
                ui.Workspace.show_window(name, False)
                continue
            except Exception:
                pass

            window = ui.Workspace.get_window(name)
            if window is None:
                continue
            try:
                window.collapsed = True
            except Exception:
                pass
            try:
                window.visible = False
            except Exception:
                pass

        _PLAY_VIEW_LAYOUT_APPLIED = True
    except Exception:
        pass


def _apply_viewport_light_rig(light_rig: str):
    """Apply Isaac's viewport light-rig menu option, e.g. Light Rigs -> Default."""

    if not light_rig:
        return

    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/exts/omni.kit.viewport.menubar.lighting/defaultRig", light_rig)
        settings.set("/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled", True)

        import omni.kit.actions.core

        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_rig")
        if action is not None:
            action.execute(light_rig)
            print(f"[INFO] Viewport light rig set to: {light_rig}", flush=True)
            return
    except Exception as exc:
        print(f"[WARN] Could not set viewport light rig to {light_rig!r}: {type(exc).__name__}: {exc}", flush=True)


class SideFollowCamera:
    """Camera that tracks the robot root and keeps a side view in the robot yaw frame."""

    def __init__(
        self,
        raw_env,
        env_index: int = 0,
        eye_b=(0.0, -2.6, 1.1),
        lookat_b=(0.0, 0.0, 0.35),
        *,
        camera_path: str = "/World/RaceFollowCamera",
        activate_viewport: bool = True,
    ):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.eye_b = np.asarray(eye_b, dtype=np.float64)
        self.lookat_b = np.asarray(lookat_b, dtype=np.float64)
        self.asset = raw_env._robot
        self.camera_path = str(camera_path)
        self.perspective_path = "/OmniverseKit_Persp"
        self.activate_viewport = bool(activate_viewport)
        self._warned_camera_prim_failure = False
        self.last_eye_w: np.ndarray | None = None
        self.last_lookat_w: np.ndarray | None = None
        self.raw_env.cfg.viewer.cam_prim_path = self.camera_path
        self._ensure_camera_prim()
        # If an RGB render product was already created for another camera, force IsaacLab/Replicator to rebuild it
        # against this follow camera on the next capture.
        try:
            _detach_rgb_render_product(self.raw_env)
        except Exception:
            pass

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _yaw_rot(yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def update(self):
        self._ensure_camera_prim()
        root_pos_w = self.asset.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
        root_quat_w = self.asset.data.root_quat_w[self.env_index].detach().cpu()
        yaw = self._quat_wxyz_to_yaw(root_quat_w)
        rot = self._yaw_rot(yaw)

        eye_w = root_pos_w + rot @ self.eye_b
        lookat_w = root_pos_w + rot @ self.lookat_b
        self.last_eye_w = np.asarray(eye_w, dtype=np.float64)
        self.last_lookat_w = np.asarray(lookat_w, dtype=np.float64)
        self._set_camera_pose_usd(eye_w, lookat_w)
        try:
            self.raw_env.sim.set_camera_view(eye=eye_w, target=lookat_w, camera_prim_path=self.camera_path)
        except TypeError:
            self.raw_env.sim.set_camera_view(eye=eye_w, target=lookat_w)
        except Exception:
            # Headless render products do not require an active viewport; the USD camera pose above is enough.
            pass
        if self.activate_viewport:
            self._set_active_camera(self.camera_path)

    def deactivate(self):
        if self.activate_viewport:
            self._set_active_camera(self.perspective_path)

    def _ensure_camera_prim(self) -> None:
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            camera = UsdGeom.Camera.Define(stage, self.camera_path)
            prim = camera.GetPrim()
            camera.CreateFocalLengthAttr(12.0).Set(12.0)
            camera.CreateFocusDistanceAttr(3.0).Set(3.0)
            coi_attr = prim.GetProperty("omni:kit:centerOfInterest")
            if not coi_attr or not coi_attr.IsValid():
                prim.CreateAttribute(
                    "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
                ).Set(Gf.Vec3d(0.0, 0.0, -3.0))
            self.raw_env.cfg.viewer.cam_prim_path = self.camera_path
        except Exception as exc:
            if not self._warned_camera_prim_failure:
                print(
                    f"[WARN] Could not create follow-camera prim {self.camera_path}; using default viewport camera: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_camera_prim_failure = True
            self.camera_path = self.perspective_path
            self.raw_env.cfg.viewer.cam_prim_path = self.camera_path

    def _set_camera_pose_usd(self, eye_w: np.ndarray, lookat_w: np.ndarray) -> None:
        """Set the USD camera transform directly so headless render products do not need a viewport."""

        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.camera_path)
            if not prim or not prim.IsValid():
                return

            eye = np.asarray(eye_w, dtype=np.float64)
            target = np.asarray(lookat_w, dtype=np.float64)
            forward = target - eye
            up = Gf.Vec3d(0.0, 0.0, 1.0)
            if float(np.linalg.norm(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64)))) <= 1.0e-9:
                up = Gf.Vec3d(0.0, 1.0, 0.0)
            eye_gf = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
            target_gf = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
            transform = Gf.Matrix4d().SetLookAt(eye_gf, target_gf, up).GetInverse().GetOrthonormalized()

            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(transform)
        except Exception as exc:
            if not self._warned_camera_prim_failure:
                print(
                    f"[WARN] Could not set USD camera pose for {self.camera_path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_camera_prim_failure = True

    @staticmethod
    def _set_active_camera(camera_path: str):
        try:
            from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name

            viewport = get_viewport_from_window_name("Viewport") or get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
        except Exception:
            pass


class StaticCamera:
    """Camera with a fixed world-frame view that can be manually adjusted afterwards."""

    def __init__(self, raw_env, eye_w=(4.5, 0.0, 9.0), lookat_w=(4.5, 0.0, 0.0), heading_deg: float = 90.0):
        self.raw_env = raw_env
        self.eye_w = np.asarray(eye_w, dtype=np.float64)
        self.lookat_w = np.asarray(lookat_w, dtype=np.float64)
        self.heading_deg = float(heading_deg)
        self.camera_path = "/World/RaceFreeCamera"
        self.perspective_path = "/OmniverseKit_Persp"

    def update(self):
        self._ensure_camera_prim()
        self._set_active_camera(self.camera_path)

    def deactivate(self):
        self._set_active_camera(self.perspective_path)

    def _ensure_camera_prim(self):
        """Create/update a dedicated free camera with an absolute top-down heading.

        Using `/OmniverseKit_Persp` plus `set_camera_view()` lets Kit choose/reset the top-down roll. A dedicated
        camera prim with explicit transform ops makes the rotated overhead orientation deterministic and non-cumulative.
        """

        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            camera = UsdGeom.Camera.Define(stage, self.camera_path)
            prim = camera.GetPrim()

            camera.CreateFocalLengthAttr(8.5).Set(8.5)
            camera.CreateFocusDistanceAttr(float(abs(self.eye_w[2] - self.lookat_w[2]))).Set(
                float(abs(self.eye_w[2] - self.lookat_w[2]))
            )
            coi_attr = prim.GetProperty("omni:kit:centerOfInterest")
            if not coi_attr or not coi_attr.IsValid():
                prim.CreateAttribute(
                    "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
                ).Set(Gf.Vec3d(0.0, 0.0, -float(abs(self.eye_w[2] - self.lookat_w[2]))))

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            rotate_op = xform.AddRotateXYZOp()
            translate_op.Set(Gf.Vec3d(*[float(v) for v in self.eye_w]))
            rotate_op.Set(Gf.Vec3f(0.0, 0.0, float(self.heading_deg)))
        except Exception as exc:
            print(f"[WARN] Could not set dedicated free camera: {type(exc).__name__}: {exc}", flush=True)
            self.raw_env.sim.set_camera_view(eye=self.eye_w, target=self.lookat_w)

    @staticmethod
    def _set_active_camera(camera_path: str):
        try:
            from omni.kit.viewport.utility import get_active_viewport, get_viewport_from_window_name

            viewport = get_viewport_from_window_name("Viewport") or get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
        except Exception as exc:
            print(f"[WARN] Could not set active viewport camera to {camera_path}: {type(exc).__name__}: {exc}", flush=True)


def _set_window_bounds(window, *, x: int, y: int, width: int, height: int):
    """Best-effort initial placement for omni.ui windows."""

    for attr, value in (("position_x", x), ("position_y", y), ("width", width), ("height", height)):
        try:
            setattr(window, attr, value)
        except Exception:
            pass


def _build_camera_window(camera_state: dict, follow_camera: SideFollowCamera | None, free_camera: StaticCamera):
    import omni.ui as ui

    title_model = ui.SimpleStringModel(f"Current: {camera_state['mode']}")

    def _set_mode(mode: str):
        if mode == "follow" and follow_camera is None:
            mode = "free"
        camera_state["mode"] = mode
        title_model.set_value(f"Current: {mode}")
        if mode == "follow":
            free_camera.deactivate()
            follow_camera.update()
        else:
            free_camera.update()

    win = ui.Window("Race camera", width=360, height=190, visible=True)
    _set_window_bounds(win, x=20, y=80, width=360, height=190)
    with win.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Switch viewport camera mode", height=22)
            ui.StringField(title_model, read_only=True, height=28)
            with ui.HStack(spacing=8, height=36):
                ui.Button("Follow robot", clicked_fn=lambda: _set_mode("follow"))
                ui.Button("Free camera", clicked_fn=lambda: _set_mode("free"))
            ui.Button("Reset episode", height=36, clicked_fn=lambda: camera_state.__setitem__("reset_requested", True))

    return win, title_model


def _build_source_swap_window(source_sync_run_id: str | None):
    """Build a small manual restore button for the temporary training source-file swap."""
    import omni.ui as ui

    restore_solo12_race_sources, solo12_race_sources_need_restore, _ = _import_race_source_sync_helpers()

    if source_sync_run_id is None:
        initial_status = "No W&B training source swap is active."
    elif solo12_race_sources_need_restore():
        initial_status = f"Training env/cfg files active from run {source_sync_run_id}."
    else:
        initial_status = "Local env/cfg files are already restored."

    status_model = ui.SimpleStringModel(initial_status)

    def _restore_sources():
        if restore_solo12_race_sources():
            status_model.set_value("Restored local env/cfg files and removed .bak backups.")
        else:
            status_model.set_value("No active env/cfg .bak swap to restore.")

    win = ui.Window("Race source files", width=430, height=140, visible=True)
    _set_window_bounds(win, x=20, y=285, width=430, height=140)
    with win.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Temporary training source-file swap", height=22)
            ui.StringField(status_model, read_only=True, height=28)
            ui.Button("env and cfg swap files", height=36, clicked_fn=_restore_sources)
            ui.Label("Click before exiting to put your working files back.", height=22)

    return win, status_model


class RaceMetricsPopup:
    """Small live dashboard for race completion time, progress, and collision helpers."""

    def __init__(self, raw_env, env_index: int = 0):
        import omni.ui as ui

        self.raw_env = raw_env
        self.env_index = int(env_index)
        self._last_display_text = None
        self._prev_episode_step = -1
        self._prev_floor_collision = False
        self._prev_pillar_collision = False
        self._prev_thigh_contact = False
        self._floor_collision_events = 0
        self._pillar_collision_events = 0
        self._last_finish_steps = None
        self._last_finish_seconds = None

        self._progress_model = ui.SimpleStringModel("Progress: --")
        self._time_model = ui.SimpleStringModel("Time: --")
        self._last_finish_model = ui.SimpleStringModel("Last finish: --")
        self._collision_model = ui.SimpleStringModel("Collisions: --")

        self.window = ui.Window("Race metrics", width=430, height=175, visible=True)
        _set_window_bounds(self.window, x=840, y=80, width=430, height=175)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Race metrics", height=22)
                ui.StringField(self._progress_model, read_only=True, height=28)
                ui.StringField(self._time_model, read_only=True, height=28)
                ui.StringField(self._last_finish_model, read_only=True, height=28)
                ui.StringField(self._collision_model, read_only=True, height=28)

    def update(self, dones=None):
        step = self._current_episode_step()
        if step < self._prev_episode_step:
            self._reset_episode_counters()
        self._prev_episode_step = step

        finished_now = self._is_finished()
        if dones is not None and self._done_for_env(dones):
            self._capture_last_finish_from_extras(finished_now=finished_now)

        floor_collision = self._floor_collision_now()
        pillar_collision = self._pillar_collision_now()
        thigh_contact = self._thigh_contact_count_now() > 0
        if floor_collision and not self._prev_floor_collision:
            self._floor_collision_events += 1
        if (pillar_collision and not self._prev_pillar_collision) or (thigh_contact and not self._prev_thigh_contact):
            self._pillar_collision_events += 1
        self._prev_floor_collision = floor_collision
        self._prev_pillar_collision = pillar_collision
        self._prev_thigh_contact = thigh_contact

        progress, gate_idx, target_count = self._progress()
        seconds = step * float(getattr(self.raw_env, "step_dt", 0.0))
        status = "FINISHED" if finished_now else "running"
        last_finish = "Last finish: --"
        if self._last_finish_steps is not None and self._last_finish_seconds is not None:
            last_finish = f"Last finish: {self._last_finish_seconds:.2f}s / {int(self._last_finish_steps)} steps"

        self._set_text(
            f"Progress: {100.0 * progress:.1f}%  gate {gate_idx}/{target_count}  {status}",
            f"Current time: {seconds:.2f}s / {int(step)} steps",
            last_finish,
            f"Collisions: floor={self._floor_collision_events} pillar={self._pillar_collision_events}",
        )

    def _reset_episode_counters(self):
        self._floor_collision_events = 0
        self._pillar_collision_events = 0
        self._prev_floor_collision = False
        self._prev_pillar_collision = False
        self._prev_thigh_contact = False

    def _current_episode_step(self) -> int:
        try:
            return int(self.raw_env.episode_length_buf[self.env_index].item())
        except Exception:
            return 0

    def _progress(self) -> tuple[float, int, int]:
        try:
            target_count = int(getattr(self.raw_env, "_target_count", 0))
            gate_idx = int(self.raw_env._current_gate_idx[self.env_index].item())
            progress = 0.0 if target_count <= 0 else min(1.0, gate_idx / target_count)
            return progress, gate_idx, target_count
        except Exception:
            return 0.0, 0, 0

    def _is_finished(self) -> bool:
        progress, _, _ = self._progress()
        return progress >= 1.0

    def _done_for_env(self, dones) -> bool:
        try:
            return bool(dones[self.env_index].item())
        except Exception:
            try:
                return bool(dones)
            except Exception:
                return False

    def _capture_last_finish_from_extras(self, *, finished_now: bool):
        try:
            log = self.raw_env.extras.get("log", {})
            success_rate = float(log.get("Episode/successRate", 1.0 if finished_now else 0.0))
            if success_rate < 1.0:
                return
            steps = log.get("Episode/finishTimeSteps")
            seconds = log.get("Episode/finishTimeSeconds")
            if steps is not None and seconds is not None:
                self._last_finish_steps = float(steps)
                self._last_finish_seconds = float(seconds)
                return
        except Exception:
            if not finished_now:
                return
        self._last_finish_steps = float(self._current_episode_step())
        self._last_finish_seconds = self._last_finish_steps * float(getattr(self.raw_env, "step_dt", 0.0))

    def _floor_collision_now(self) -> bool:
        try:
            value = self.raw_env._compute_filtered_base_contact(
                self.raw_env._base_floor_contact_sensor, self.raw_env.cfg.base_contact_threshold
            )
            return bool(value[self.env_index].item())
        except Exception:
            return False

    def _pillar_collision_now(self) -> bool:
        try:
            value = self.raw_env._compute_filtered_base_contact(
                self.raw_env._base_pillar_contact_sensor, self.raw_env.cfg.base_contact_threshold
            )
            return bool(value[self.env_index].item())
        except Exception:
            return False

    def _thigh_contact_count_now(self) -> int:
        try:
            value = self.raw_env._compute_contact_count(
                self.raw_env._thigh_body_ids, self.raw_env.cfg.undesired_contact_threshold
            )
            return int(value[self.env_index].item())
        except Exception:
            return 0

    def _set_text(self, progress_text: str, time_text: str, last_finish_text: str, collision_text: str):
        display_text = (progress_text, time_text, last_finish_text, collision_text)
        if display_text == self._last_display_text:
            return
        self._last_display_text = display_text
        self._progress_model.set_value(progress_text)
        self._time_model.set_value(time_text)
        self._last_finish_model.set_value(last_finish_text)
        self._collision_model.set_value(collision_text)


class PatchFrictionSelectionPopup:
    """Small UI panel that shows the friction values for the selected race patch prim."""

    def __init__(self, raw_env, env_index: int = 0):
        import omni.ui as ui
        import omni.usd

        self.raw_env = raw_env
        self.env_index = int(env_index)
        self._selection = omni.usd.get_context().get_selection()
        self._last_display_text = None
        self._patch_model = ui.SimpleStringModel("Click a colored patch in the viewport")
        self._friction_model = ui.SimpleStringModel("mu_static / mu_dynamic will appear here")
        self._hint_model = ui.SimpleStringModel("Tip: click the tile cube/patch prim, not the robot.")

        self.window = ui.Window("Patch friction", width=430, height=160, visible=True)
        _set_window_bounds(self.window, x=400, y=80, width=430, height=160)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Selected friction patch", height=22)
                ui.StringField(self._patch_model, read_only=True, height=28)
                ui.StringField(self._friction_model, read_only=True, height=28)
                ui.StringField(self._hint_model, read_only=True, height=28)

    @staticmethod
    def _patch_name_from_path(path: str) -> str | None:
        for part in reversed(path.split("/")):
            if re.fullmatch(r"patch_\d+", part):
                return part
        return None

    @staticmethod
    def _env_index_from_path(path: str, default: int) -> int:
        for part in path.split("/"):
            if part.startswith("env_") and part[4:].isdigit():
                return int(part[4:])
        return default

    def update(self):
        selected_paths = self._selection.get_selected_prim_paths()
        if not selected_paths:
            self._set_text(
                "Click a colored patch in the viewport",
                "mu_static / mu_dynamic will appear here",
                "Tip: click the tile cube/patch prim, not the robot.",
            )
            return

        selected_path = selected_paths[-1]
        patch_name = self._patch_name_from_path(selected_path)
        if patch_name is None:
            self._set_text(
                "Selected prim is not a friction patch",
                "",
                selected_path,
            )
            return

        env_index = self._env_index_from_path(selected_path, self.env_index)
        try:
            summary = self.raw_env.get_patch_friction_summary(env_index=env_index)
        except Exception as exc:
            self._set_text(patch_name, "Could not read patch friction", f"{type(exc).__name__}: {exc}")
            return

        patch_summary = {str(item["patch"]): item for item in summary}.get(patch_name)
        if patch_summary is None:
            self._set_text(patch_name, "Patch exists in USD, but no friction data was found", selected_path)
            return

        self._set_text(
            f"{patch_name}  |  env {env_index}  |  bucket {int(patch_summary['bucket']):03d}",
            f"mu_static={float(patch_summary['static']):.3f}    mu_dynamic={float(patch_summary['dynamic']):.3f}",
            selected_path,
        )

    def _set_text(self, patch_text: str, friction_text: str, hint_text: str):
        display_text = (patch_text, friction_text, hint_text)
        if display_text == self._last_display_text:
            return
        self._last_display_text = display_text
        self._patch_model.set_value(patch_text)
        self._friction_model.set_value(friction_text)
        self._hint_model.set_value(hint_text)


class FootFrictionPopup:
    """Small live UI panel that shows the patch friction coefficient under each foot."""

    _DISPLAY_ORDER = ("FR", "FL", "RR", "RL")

    def __init__(self, raw_env, env_index: int = 0):
        import omni.ui as ui

        self.raw_env = raw_env
        self.env_index = int(env_index)
        self._last_display_text = None
        self._models = {label: ui.SimpleStringModel(f"{label}: --") for label in self._DISPLAY_ORDER}
        self._hint_model = ui.SimpleStringModel("mu_static below each foot")

        self.window = ui.Window("Foot friction", width=260, height=190, visible=True)
        _set_window_bounds(self.window, x=20, y=465, width=260, height=190)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Foot patch friction", height=22)
                for label in self._DISPLAY_ORDER:
                    ui.StringField(self._models[label], read_only=True, height=28)
                ui.StringField(self._hint_model, read_only=True, height=28)

    def update(self):
        try:
            mu_by_label, foot_order = self._read_foot_mu_by_label()
        except Exception as exc:
            self._set_text({}, f"Could not read foot mu: {type(exc).__name__}: {exc}")
            return

        missing = [label for label in self._DISPLAY_ORDER if label not in mu_by_label]
        hint = "mu_static below each foot"
        if missing:
            hint = "foot order: " + ", ".join(foot_order)
        self._set_text(mu_by_label, hint)

    def _read_foot_mu_by_label(self) -> tuple[dict[str, float], list[str]]:
        if not hasattr(self.raw_env, "_get_gt_patch_mu_obs"):
            raise AttributeError("raw env has no _get_gt_patch_mu_obs()")

        with torch.inference_mode():
            mu = self.raw_env._get_gt_patch_mu_obs()[self.env_index].detach().cpu().tolist()

        foot_names = self._foot_body_names()
        foot_order = foot_names if foot_names else [f"foot_{index}" for index in range(len(mu))]
        mu_by_label = {}
        for index, value in enumerate(mu):
            label = self._label_from_foot_name(foot_names[index] if index < len(foot_names) else "")
            if label is not None:
                mu_by_label[label] = float(value)
        return mu_by_label, foot_order

    def _foot_body_names(self) -> list[str]:
        robot = getattr(self.raw_env, "_robot", None)
        find_bodies = getattr(robot, "find_bodies", None)
        if callable(find_bodies):
            try:
                _, body_names = find_bodies(".*_calf")
                if body_names:
                    return [str(body_name) for body_name in body_names]
            except Exception:
                pass

        foot_ids = self._ids_to_list(getattr(self.raw_env, "_feet_robot_body_ids", []))
        body_names = list(getattr(robot, "body_names", []) or [])
        if foot_ids and body_names:
            try:
                return [str(body_names[int(body_id)]) for body_id in foot_ids]
            except Exception:
                pass

        foot_ids = self._ids_to_list(getattr(self.raw_env, "_feet_body_ids", []))
        body_names = list(getattr(getattr(self.raw_env, "_contact_sensor", None), "body_names", []) or [])
        if foot_ids and body_names:
            try:
                return [str(body_names[int(body_id)]) for body_id in foot_ids]
            except Exception:
                pass

        return []

    @staticmethod
    def _ids_to_list(ids) -> list[int]:
        if isinstance(ids, torch.Tensor):
            return [int(value) for value in ids.detach().cpu().tolist()]
        return [int(value) for value in ids]

    @staticmethod
    def _label_from_foot_name(name: str) -> str | None:
        match = re.search(r"\b(FR|FL|RR|RL)_", name)
        return match.group(1) if match else None

    def _set_text(self, mu_by_label: dict[str, float], hint_text: str):
        display_text = tuple(mu_by_label.get(label) for label in self._DISPLAY_ORDER), hint_text
        if display_text == self._last_display_text:
            return
        self._last_display_text = display_text
        for label in self._DISPLAY_ORDER:
            if label in mu_by_label:
                self._models[label].set_value(f"{label}:  mu_static={mu_by_label[label]:.3f}")
            else:
                self._models[label].set_value(f"{label}:  --")
        self._hint_model.set_value(hint_text)


class SlipConeVisualizer:
    """Live per-foot friction-cone status with UI and viewport dot markers."""

    _DISPLAY_ORDER = ("FR", "FL", "RR", "RL")
    _SCHEMATIC_ORDER = ("FL", "FR", "RL", "RR")
    _OVERHEAD_OFFSETS_B = {
        "FL": np.array([0.20, 0.14, 0.48], dtype=np.float64),
        "FR": np.array([0.20, -0.14, 0.48], dtype=np.float64),
        "RL": np.array([-0.20, 0.14, 0.48], dtype=np.float64),
        "RR": np.array([-0.20, -0.14, 0.48], dtype=np.float64),
    }
    _COLORS = {
        "air": (0.65, 0.65, 0.65, 1.0),
        "white": (0.95, 0.95, 0.95, 0.9),
        "yellow": (1.00, 0.92, 0.00, 1.0),
        "yg1": (0.88, 0.93, 0.00, 1.0),
        "yg2": (0.75, 0.94, 0.00, 1.0),
        "yg3": (0.62, 0.95, 0.02, 1.0),
        "yg4": (0.49, 0.96, 0.05, 1.0),
        "yg5": (0.36, 0.97, 0.08, 1.0),
        "yg6": (0.23, 0.98, 0.11, 1.0),
        "yg7": (0.11, 0.99, 0.15, 1.0),
        "green": (0.00, 1.00, 0.18, 1.0),
        "red": (1.00, 0.00, 0.00, 1.0),
    }
    _ANGLE_GRADIENT_KEYS = ("yellow", "yg1", "yg2", "yg3", "yg4", "yg5", "yg6", "yg7", "green")
    _UI_COLOR_KEYS = ("air", "white", *_ANGLE_GRADIENT_KEYS, "red")
    _VIEWPORT_DOT_COLOR_KEYS = ("white", *_ANGLE_GRADIENT_KEYS, "red")

    def __init__(
        self,
        raw_env,
        env_index: int = 0,
        *,
        speed_threshold: float = 0.03,
        superior_markers: bool = False,
        viz_air_points: bool = False,
        physics_rate_csv_path: str | None = None,
        contact_stat: str = "max",
        headless: bool = False,
    ):
        self.headless = bool(headless)

        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.speed_threshold = max(0.0, float(speed_threshold))
        self.superior_markers = bool(superior_markers)
        self.viz_air_points = bool(viz_air_points)
        self.contact_stat = str(contact_stat).strip().lower()
        if self.contact_stat not in {"max", "mean", "median"}:
            self.contact_stat = "max"
        self._last_display_text = None
        self._last_contact_states: dict[str, dict[str, object]] = {}
        self._foot_link_body_ids_cache: torch.Tensor | None = None
        self._contact_to_robot_perm_cache: torch.Tensor | None = None
        # Physics-rate (per-substep) sampler state. The sampler hooks raw_env.scene.update so
        # foot velocity + cone angle are captured at the full sim rate instead of once per
        # policy step, which is what the live UI and the CSV time series consume.
        self._physics_sampler_active = False
        self._step_samples: dict[str, list[dict[str, object]]] = {}
        self._global_substep = 0
        self._sim_dt = 1.0 / 200.0
        self._decimation = 1
        self._foot_prev_contact: dict[str, bool] = {}
        self._foot_contact_counter: dict[str, int] = {}
        # Per-foot friction-opposed speed buffer for the *current* ongoing contact, used by the
        # mean/median UI aggregation so the touchdown-instant speed spike does not dominate.
        self._contact_speed_buffer: dict[str, list[float]] = {}
        self._contact_active_id: dict[str, int | None] = {}
        self._orig_scene_update = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        self._sampler_rows_written = 0
        self._warned_sampler = False
        self._draw = None
        self._warned_debug_draw = False
        self._warned_update = False
        self._fields: dict[str, object] = {}
        self._marker_rects: dict[str, dict[str, object]] = {}
        self._row_rects: dict[str, dict[str, object]] = {}
        self._marker_visualizers: dict[str, object] = {}
        self.window = None

        # Headless plot-logging only needs the physics-rate sampler that writes the CSV; skip the
        # omni.ui dashboard and the viewport dot markers entirely so this works with --headless.
        if not self.headless:
            import omni.ui as ui

            self._models = {label: ui.SimpleStringModel(f"{label}: --") for label in self._DISPLAY_ORDER}
            self._schematic_model = ui.SimpleStringModel("")
            self._hint_model = ui.SimpleStringModel("")

            self.window = ui.Window("Foot slip cones", width=760, height=360, visible=True)
            _set_window_bounds(self.window, x=20, y=600, width=760, height=360)
            with self.window.frame:
                with ui.VStack(spacing=6, height=0):
                    ui.Label("Foot slip cone status", height=22)
                    ui.StringField(self._schematic_model, read_only=True, height=28)
                    self._build_robot_schematic(ui)
                    for label in self._DISPLAY_ORDER:
                        self._build_status_row(ui, label)
                    ui.StringField(self._hint_model, read_only=True, height=28)
            self._set_schematic_colors({})
            self._create_viewport_dot_markers()
        self._install_physics_sampler(physics_rate_csv_path)

    def _build_robot_schematic(self, ui):
        with ui.VStack(spacing=3, height=104):
            with ui.HStack(height=30):
                self._build_foot_marker(ui, "FL")
                ui.Spacer(width=110)
                self._build_foot_marker(ui, "FR")
            with ui.HStack(height=30):
                ui.Spacer(width=84)
                ui.Rectangle(
                    width=150,
                    height=28,
                    style={
                        "background_color": 0x30202020,
                        "border_color": 0xFFAAAAAA,
                        "border_width": 1.0,
                        "margin": 0.0,
                    },
                )
                ui.Spacer(width=84)
            with ui.HStack(height=30):
                self._build_foot_marker(ui, "RL")
                ui.Spacer(width=110)
                self._build_foot_marker(ui, "RR")

    def _build_foot_marker(self, ui, label: str):
        with ui.HStack(width=90, height=28, spacing=5):
            ui.Label(label, width=24, height=24)
            with ui.ZStack(width=34, height=24):
                rects: dict[str, object] = {}
                for color_key in self._UI_COLOR_KEYS:
                    rect = ui.Rectangle(
                        width=30,
                        height=20,
                        style=self._rect_style(color_key),
                    )
                    rects[color_key] = rect
                self._marker_rects[label] = rects

    def _build_status_row(self, ui, label: str):
        with ui.HStack(height=28, spacing=6):
            with ui.ZStack(width=34, height=24):
                rects: dict[str, object] = {}
                for color_key in self._UI_COLOR_KEYS:
                    rect = ui.Rectangle(
                        width=30,
                        height=20,
                        style=self._rect_style(color_key),
                    )
                    rects[color_key] = rect
                self._row_rects[label] = rects
            self._fields[label] = ui.StringField(
                self._models[label],
                read_only=True,
                height=28,
                style=self._field_style("air"),
            )

    def _set_schematic_colors(self, states: dict[str, dict[str, object]]):
        schematic_parts = []
        for label in self._SCHEMATIC_ORDER:
            color_key = str(states.get(label, {}).get("color_key", "air"))
            if color_key not in self._COLORS:
                color_key = "air"
            schematic_parts.append(f"{label}:{color_key.upper()}")
            for rect_key, rect in self._marker_rects.get(label, {}).items():
                try:
                    rect.visible = rect_key == color_key
                except Exception:
                    pass
            for rect_key, rect in self._row_rects.get(label, {}).items():
                try:
                    rect.visible = rect_key == color_key
                except Exception:
                    pass
            field = self._fields.get(label)
            if field is not None:
                try:
                    field.style = self._field_style(color_key)
                except Exception:
                    pass
        self._schematic_model.set_value("   ".join(schematic_parts))

    @classmethod
    def _ui_color(cls, color_key: str) -> int:
        red, green, blue, alpha = cls._COLORS.get(color_key, cls._COLORS["white"])
        return (
            (int(round(alpha * 255.0)) << 24)
            | (int(round(blue * 255.0)) << 16)
            | (int(round(green * 255.0)) << 8)
            | int(round(red * 255.0))
        )

    @classmethod
    def _rect_style(cls, color_key: str) -> dict[str, float | int]:
        return {
            "background_color": cls._ui_color(color_key),
            "border_color": 0xFFFFFFFF,
            "border_width": 1.0,
            "border_radius": 9.0,
            "margin": 0.0,
        }

    @classmethod
    def _field_style(cls, color_key: str) -> dict[str, dict[str, float | int]]:
        red, green, blue, _ = cls._COLORS.get(color_key, cls._COLORS["air"])
        # Darken the fill a little so white text stays readable while the state color is obvious.
        fill = (
            (0xFF << 24)
            | (int(round(min(1.0, blue * 0.55 + 0.08) * 255.0)) << 16)
            | (int(round(min(1.0, green * 0.55 + 0.08) * 255.0)) << 8)
            | int(round(min(1.0, red * 0.55 + 0.08) * 255.0))
        )
        return {
            "StringField": {
                "background_color": cls._ui_color(color_key),
                "color": 0xFFFFFFFF,
                "border_color": 0xFFFFFFFF,
                "border_width": 1.0,
                "margin": 0.0,
            },
            "Field": {
                "background_color": fill,
                "color": 0xFFFFFFFF,
                "border_color": 0xFFFFFFFF,
                "border_width": 1.0,
                "margin": 0.0,
            },
            "Rectangle": {
                "background_color": fill,
                "color": 0xFFFFFFFF,
                "border_color": 0xFFFFFFFF,
                "border_width": 1.0,
                "margin": 0.0,
            },
        }

    def _create_viewport_dot_markers(self):
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            markers = {}
            for color_key in self._VIEWPORT_DOT_COLOR_KEYS:
                red, green, blue, _ = self._COLORS[color_key]
                markers[color_key] = sim_utils.SphereCfg(
                    radius=0.095,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(float(red), float(green), float(blue)),
                        emissive_color=(float(red), float(green), float(blue)),
                    ),
                )
            for color_key, marker in markers.items():
                self._marker_visualizers[color_key] = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path=f"/World/Visuals/foot_slip_dots_env_{self.env_index}_{color_key}",
                        markers={color_key: marker},
                    )
                )
        except Exception as exc:
            self._marker_visualizers = {}
            print(f"[WARN] Could not initialize slip-cone viewport dots: {type(exc).__name__}: {exc}", flush=True)

    def close(self):
        self.clear()
        self._uninstall_physics_sampler()
        for visualizer in self._marker_visualizers.values():
            try:
                visualizer.set_visibility(False)
            except Exception:
                pass

    def _uninstall_physics_sampler(self) -> None:
        """Restore the original scene.update and close the CSV log."""
        if self._orig_scene_update is not None:
            scene = getattr(self.raw_env, "scene", None)
            if scene is not None:
                try:
                    scene.update = self._orig_scene_update
                except Exception:
                    pass
            self._orig_scene_update = None
        self._physics_sampler_active = False
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            if self._csv_path:
                print(
                    f"[INFO] Slip viz: wrote {self._sampler_rows_written} physics-rate rows to {self._csv_path}.",
                    flush=True,
                )
            self._csv_file = None
            self._csv_writer = None

    def clear(self):
        if self._draw is None:
            return
        try:
            self._draw.clear_lines()
        except Exception:
            pass

    def update(self):
        # Headless logging has no UI/viewport to refresh; the CSV is written by the physics-rate
        # sampler hook, so there is nothing to do per policy step here.
        if self.headless:
            return
        try:
            states = self._read_states()
            self._set_text(states)
            self._draw_viewport_dots(states)
            self.clear()
        except Exception as exc:
            if not self._warned_update:
                print(f"[WARN] Could not update slip-cone visualization: {type(exc).__name__}: {exc}", flush=True)
                self._warned_update = True
            self._set_error_text(f"{type(exc).__name__}: {exc}")
            self.clear()

    def _compute_current_foot_states(self) -> dict[str, dict[str, object]]:
        """Compute each foot's slip state from the *current* simulation buffers.

        Reads instantaneous reaction forces, friction, foot velocity, and friction-cone
        angles, all aligned to robot foot order. When called from the physics-rate hook this
        runs once per physics substep. No air/held caching is applied here.
        """
        contact_forces_w, normal_forces_w, friction_forces_w = self._contact_reaction_forces_w()
        contact_forces_w = contact_forces_w[self.env_index].detach()
        normal_forces_w = normal_forces_w[self.env_index].detach()
        friction_forces_w = friction_forces_w[self.env_index].detach()
        force_norm = torch.linalg.norm(contact_forces_w, dim=-1)
        normal_force_norm = torch.linalg.norm(normal_forces_w, dim=-1)
        friction_force_norm = torch.linalg.norm(friction_forces_w, dim=-1)
        contact_mask = self._instant_contact_mask(normal_force_norm)
        static_mu, dynamic_mu = self._foot_friction_values()
        foot_pos_w = self._foot_reference_positions_w()[self.env_index].detach()
        foot_vel_w = self._foot_velocities_w()[self.env_index].detach()
        foot_vel_origin_w = self._foot_origin_velocities_w()[self.env_index].detach()
        root_quat_w = self.raw_env._robot.data.root_quat_w[self.env_index].detach().cpu()
        yaw = SideFollowCamera._quat_wxyz_to_yaw(root_quat_w)
        world_to_footprint = SideFollowCamera._yaw_rot(yaw).T

        foot_names = _foot_body_names_for_overlay(self.raw_env)
        states: dict[str, dict[str, object]] = {}
        for index in range(contact_forces_w.shape[0]):
            name = foot_names[index] if index < len(foot_names) else ""
            label = FootFrictionPopup._label_from_foot_name(name) or f"foot_{index}"
            force_w = contact_forces_w[index]
            normal_norm_value = float(normal_force_norm[index].item())
            friction_norm_value = float(friction_force_norm[index].item())
            friction_force = friction_forces_w[index].detach().cpu().numpy().astype(np.float64)
            friction_force_footprint = world_to_footprint @ friction_force
            friction_force_xy_norm = float(np.linalg.norm(friction_force_footprint[:2]))
            if friction_force_xy_norm > 1.0e-9:
                friction_direction_footprint = friction_force_footprint[:2] / friction_force_xy_norm
            else:
                friction_direction_footprint = np.zeros(2, dtype=np.float64)
            angle_rad = torch.atan2(friction_force_norm[index], normal_force_norm[index])
            angle_deg = float(torch.rad2deg(angle_rad).item())
            effective_mu = friction_norm_value / max(normal_norm_value, 1.0e-6)
            mu_static = float(static_mu[index].item())
            mu_dynamic = float(dynamic_mu[index].item())
            angle_static_deg = math.degrees(math.atan(max(0.0, mu_static)))
            angle_dynamic_deg = math.degrees(math.atan(max(0.0, mu_dynamic)))
            foot_vel_tensor = foot_vel_w[index]
            foot_vel = foot_vel_tensor.detach().cpu().numpy().astype(np.float64)
            tangential_speed_xy = float(np.linalg.norm(foot_vel[:2]))
            tangential_speed = self._friction_axis_speed(
                foot_vel_tensor,
                friction_forces_w[index],
                friction_force_norm[index],
            )
            foot_vel_origin = foot_vel_origin_w[index].detach().cpu().numpy().astype(np.float64)
            tangential_speed_origin = float(np.linalg.norm(foot_vel_origin[:2]))
            rho_static = friction_norm_value / max(mu_static * normal_norm_value, 1.0e-6)
            rho_dynamic = friction_norm_value / max(mu_dynamic * normal_norm_value, 1.0e-6)
            contact = bool(contact_mask[index].item())
            color_key, status = self._classify(
                contact=contact,
                angle_rad=float(angle_rad.item()),
                angle_dynamic_rad=math.atan(max(0.0, mu_dynamic)),
                angle_static_rad=math.atan(max(0.0, mu_static)),
                tangential_speed=tangential_speed,
            )
            states[label] = {
                "label": label,
                "contact": contact,
                "status": status,
                "color_key": color_key,
                "force_w": force_w.detach().cpu().numpy().astype(np.float64),
                "normal_force_w": normal_forces_w[index].detach().cpu().numpy().astype(np.float64),
                "friction_force_w": friction_force,
                "friction_direction_footprint": friction_direction_footprint,
                "foot_pos_w": foot_pos_w[index].detach().cpu().numpy().astype(np.float64),
                "foot_vel_w": foot_vel,
                "angle_deg": angle_deg,
                "angle_dynamic_deg": angle_dynamic_deg,
                "angle_static_deg": angle_static_deg,
                "effective_mu": effective_mu,
                "rho_static": rho_static,
                "rho_dynamic": rho_dynamic,
                "mu_static": mu_static,
                "mu_dynamic": mu_dynamic,
                "tangential_speed": tangential_speed,
                "tangential_speed_xy": tangential_speed_xy,
                "tangential_speed_origin": tangential_speed_origin,
                "force_norm": float(force_norm[index].item()),
                "normal_force_norm": normal_norm_value,
                "friction_force_norm": friction_norm_value,
            }
        return states

    @staticmethod
    def _friction_axis_speed(
        foot_vel_w: torch.Tensor,
        friction_force_w: torch.Tensor,
        friction_force_norm: torch.Tensor,
    ) -> float:
        """Velocity component opposed by the PhysX friction direction.

        The old slip-speed diagnostic used ``||v_xy||``. That is only approximately tied to the
        friction cone on flat ground, while the cone angle itself is computed from PhysX's
        normal/friction force decomposition. Since the measured friction force is already a
        tangential vector, project directly onto that axis. In sliding contact, friction opposes
        relative motion, so only ``-dot(v, friction_dir)`` is slip speed.
        """
        friction_norm = float(friction_force_norm.item())
        if friction_norm <= 1.0e-9:
            return 0.0

        friction_dir = friction_force_w / friction_force_norm.clamp_min(1.0e-9)
        return float(torch.clamp(-torch.dot(foot_vel_w, friction_dir), min=0.0).item())

    @staticmethod
    def _select_display_sample(samples: list[dict[str, object]]) -> dict[str, object]:
        """Pick the worst-case sample for a foot over the substeps of one policy step.

        Prefer the in-contact substep with the highest friction-opposed speed so a brief touchdown
        slip is not aliased away; if the foot never contacted during the step, show the latest
        substep so airborne feet still update.
        """
        contact_samples = [sample for sample in samples if sample.get("contact")]
        if contact_samples:
            return max(contact_samples, key=lambda sample: float(sample.get("tangential_speed", 0.0)))
        return samples[-1]

    def _aggregate_display_sample(self, label: str, samples: list[dict[str, object]]) -> dict[str, object]:
        """Build the per-foot display state, summarizing speed per the selected stat.

        ``max`` keeps the worst substep in the current policy step. ``mean``/``median`` replace
        the shown speed with the mean/median friction-opposed speed over the *whole ongoing contact*
        and re-color against the friction cone, so a brief touchdown speed spike is averaged out.
        """
        if self.contact_stat == "max":
            return self._select_display_sample(samples)

        contact_samples = [sample for sample in samples if sample.get("contact")]
        if not contact_samples:
            return samples[-1]

        display_state = dict(contact_samples[-1])
        buffer = self._contact_speed_buffer.get(label)
        if buffer:
            if self.contact_stat == "median":
                aggregated_speed = float(statistics.median(buffer))
            else:
                aggregated_speed = float(statistics.fmean(buffer))
        else:
            aggregated_speed = float(display_state.get("tangential_speed", 0.0))

        display_state["tangential_speed"] = aggregated_speed
        display_state["display_stat"] = self.contact_stat
        color_key, status = self._classify(
            contact=True,
            angle_rad=math.radians(float(display_state.get("angle_deg", 0.0))),
            angle_dynamic_rad=math.radians(float(display_state.get("angle_dynamic_deg", 0.0))),
            angle_static_rad=math.radians(float(display_state.get("angle_static_deg", 0.0))),
            tangential_speed=aggregated_speed,
        )
        display_state["color_key"] = color_key
        display_state["status"] = status
        return display_state

    def _read_states(self) -> dict[str, dict[str, object]]:
        # Consume the physics-rate samples collected since the previous UI update, summarized
        # per foot by the selected stat. Fall back to an instantaneous read when inactive.
        per_foot_samples = self._drain_step_samples()
        if per_foot_samples:
            current_states = {
                label: self._aggregate_display_sample(label, samples) for label, samples in per_foot_samples.items()
            }
        else:
            current_states = self._compute_current_foot_states()

        states: dict[str, dict[str, object]] = {}
        for label, current_state in current_states.items():
            if current_state.get("contact"):
                self._last_contact_states[label] = current_state
                states[label] = current_state
            elif self.viz_air_points:
                states[label] = current_state
            else:
                cached_state = self._last_contact_states.get(label)
                if cached_state is not None:
                    held_state = dict(cached_state)
                    held_state["held"] = True
                    states[label] = held_state
        return states

    def _drain_step_samples(self) -> dict[str, list[dict[str, object]]]:
        """Return and clear the physics-rate samples accumulated since the last UI update."""
        if not self._physics_sampler_active:
            return {}
        samples = self._step_samples
        self._step_samples = {}
        return samples

    def _install_physics_sampler(self, csv_path: str | None) -> None:
        """Hook ``raw_env.scene.update`` so feet are sampled every physics substep.

        ``scene.update`` is called once per physics substep right after the physics step and
        after the contact-sensor/robot buffers are refreshed, so wrapping it samples fresh data
        at the full physics rate instead of the policy rate.
        """
        scene = getattr(self.raw_env, "scene", None)
        update_fn = getattr(scene, "update", None)
        if not callable(update_fn):
            print(
                "[WARN] Slip viz: scene.update unavailable; sampling at policy rate only (no physics-rate log).",
                flush=True,
            )
            return

        cfg = getattr(self.raw_env, "cfg", None)
        try:
            self._sim_dt = float(cfg.sim.dt)
        except Exception:
            self._sim_dt = 1.0 / 200.0
        try:
            self._decimation = max(1, int(cfg.decimation))
        except Exception:
            self._decimation = 1

        if csv_path:
            try:
                directory = os.path.dirname(csv_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                self._csv_file = open(csv_path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(self._csv_header())
                self._csv_path = csv_path
            except Exception as exc:
                print(f"[WARN] Slip viz: could not open CSV log {csv_path}: {type(exc).__name__}: {exc}", flush=True)
                self._csv_file = None
                self._csv_writer = None
                self._csv_path = None

        self._orig_scene_update = update_fn

        def _wrapped_scene_update(*args, **kwargs):
            result = self._orig_scene_update(*args, **kwargs)
            self._on_physics_substep()
            return result

        scene.update = _wrapped_scene_update
        self._physics_sampler_active = True
        rate = (1.0 / self._sim_dt) if self._sim_dt > 0 else 0.0
        contact_threshold = float(getattr(getattr(self.raw_env, "cfg", None), "base_contact_threshold", 1.0))
        print(
            f"[INFO] Slip viz: a foot is flagged contact=1 when its filtered normal force exceeds "
            f"base_contact_threshold={contact_threshold:.3g} N (the CSV keeps raw normal_force so you can re-threshold).",
            flush=True,
        )
        message = f"[INFO] Slip viz: physics-rate sampler active at ~{rate:.0f} Hz (decimation {self._decimation})."
        if self._csv_path:
            message += f" Writing per-foot time series to {self._csv_path}."
        print(message, flush=True)

    def _on_physics_substep(self) -> None:
        """Sample every foot once per physics substep: log to CSV and aggregate for the UI."""
        if not self._physics_sampler_active:
            return
        try:
            states = self._compute_current_foot_states()
        except Exception as exc:
            if not self._warned_sampler:
                print(f"[WARN] Slip viz: physics-rate sample failed: {type(exc).__name__}: {exc}", flush=True)
                self._warned_sampler = True
            return

        substep = self._global_substep
        self._global_substep += 1
        sim_time_s = substep * self._sim_dt
        policy_step = substep // self._decimation
        substep_in_step = substep % self._decimation

        for label, state in states.items():
            contact = bool(state.get("contact"))
            prev_contact = self._foot_prev_contact.get(label, False)
            if contact and not prev_contact:
                self._foot_contact_counter[label] = self._foot_contact_counter.get(label, -1) + 1
            self._foot_prev_contact[label] = contact
            contact_id = self._foot_contact_counter.get(label, -1) if contact else -1

            state = dict(state)
            state["contact_id"] = contact_id

            # The per-step sample buffers below feed the live UI aggregation only, and they are
            # drained in update(). Headless logging never calls update(), so skip them to avoid
            # growing them without bound across the whole run.
            if not self.headless:
                self._step_samples.setdefault(label, []).append(state)

                if contact:
                    # Reset the running buffer when a new contact starts, then accumulate this
                    # contact's per-substep speeds for the mean/median UI aggregation.
                    if self._contact_active_id.get(label) != contact_id:
                        self._contact_active_id[label] = contact_id
                        self._contact_speed_buffer[label] = []
                    buffer = self._contact_speed_buffer.setdefault(label, [])
                    buffer.append(float(state.get("tangential_speed", 0.0)))
                    if len(buffer) > 8192:
                        del buffer[: len(buffer) - 8192]

            if self._csv_writer is not None:
                self._write_csv_row(policy_step, substep_in_step, sim_time_s, label, contact_id, state)

        if self._csv_file is not None and (self._sampler_rows_written % 800) == 0:
            try:
                self._csv_file.flush()
            except Exception:
                pass

    @staticmethod
    def _csv_header() -> list[str]:
        return [
            "policy_step",
            "substep",
            "policy_action_update",
            "sim_time_s",
            "foot",
            "contact",
            "contact_id",
            "tangential_speed",
            "tangential_speed_xy",
            "tangential_speed_origin",
            "vx",
            "vy",
            "vz",
            "contact_x",
            "contact_y",
            "contact_z",
            "normal_force_x",
            "normal_force_y",
            "normal_force_z",
            "friction_force_x",
            "friction_force_y",
            "friction_force_z",
            "friction_direction_footprint_x",
            "friction_direction_footprint_y",
            "angle_deg",
            "angle_dyn_deg",
            "angle_static_deg",
            "effective_mu",
            "mu_static",
            "mu_dynamic",
            "normal_force",
            "friction_force",
            "force_norm",
            "rho_static",
            "rho_dynamic",
            "status",
            "color_key",
        ]

    def _write_csv_row(
        self,
        policy_step: int,
        substep_in_step: int,
        sim_time_s: float,
        label: str,
        contact_id: int,
        state: dict[str, object],
    ) -> None:
        vel = state.get("foot_vel_w")
        if vel is None:
            vel = np.zeros(3, dtype=np.float64)
        normal_force = state.get("normal_force_w")
        if normal_force is None:
            normal_force = np.zeros(3, dtype=np.float64)
        friction_force = state.get("friction_force_w")
        if friction_force is None:
            friction_force = np.zeros(3, dtype=np.float64)
        friction_direction_footprint = state.get("friction_direction_footprint")
        if friction_direction_footprint is None:
            friction_direction_footprint = np.zeros(2, dtype=np.float64)
        contact_pos = state.get("foot_pos_w")
        if contact_pos is None:
            contact_pos = np.zeros(3, dtype=np.float64)
        self._csv_writer.writerow(
            [
                int(policy_step),
                int(substep_in_step),
                int(substep_in_step == 0),
                f"{sim_time_s:.6f}",
                label,
                int(bool(state.get("contact"))),
                int(contact_id),
                f"{float(state.get('tangential_speed', 0.0)):.6f}",
                f"{float(state.get('tangential_speed_xy', 0.0)):.6f}",
                f"{float(state.get('tangential_speed_origin', 0.0)):.6f}",
                f"{float(vel[0]):.6f}",
                f"{float(vel[1]):.6f}",
                f"{float(vel[2]):.6f}",
                f"{float(contact_pos[0]):.6f}",
                f"{float(contact_pos[1]):.6f}",
                f"{float(contact_pos[2]):.6f}",
                f"{float(normal_force[0]):.6f}",
                f"{float(normal_force[1]):.6f}",
                f"{float(normal_force[2]):.6f}",
                f"{float(friction_force[0]):.6f}",
                f"{float(friction_force[1]):.6f}",
                f"{float(friction_force[2]):.6f}",
                f"{float(friction_direction_footprint[0]):.6f}",
                f"{float(friction_direction_footprint[1]):.6f}",
                f"{float(state.get('angle_deg', 0.0)):.4f}",
                f"{float(state.get('angle_dynamic_deg', 0.0)):.4f}",
                f"{float(state.get('angle_static_deg', 0.0)):.4f}",
                f"{float(state.get('effective_mu', 0.0)):.6f}",
                f"{float(state.get('mu_static', 0.0)):.6f}",
                f"{float(state.get('mu_dynamic', 0.0)):.6f}",
                f"{float(state.get('normal_force_norm', 0.0)):.6f}",
                f"{float(state.get('friction_force_norm', 0.0)):.6f}",
                f"{float(state.get('force_norm', 0.0)):.6f}",
                f"{float(state.get('rho_static', 0.0)):.6f}",
                f"{float(state.get('rho_dynamic', 0.0)):.6f}",
                str(state.get("status", "")),
                str(state.get("color_key", "")),
            ]
        )
        self._sampler_rows_written += 1

    def _foot_reaction_sensors(self) -> list[object]:
        sensors = getattr(self.raw_env, "_foot_reaction_contact_sensors", None)
        labels = tuple(getattr(self.raw_env, "_foot_reaction_contact_sensor_labels", ()) or ())
        if not isinstance(sensors, dict) or not labels:
            return []

        ordered_labels: list[str] = []
        for name in list(getattr(self.raw_env, "_feet_robot_body_names", []) or []):
            label = FootFrictionPopup._label_from_foot_name(str(name))
            if label is not None:
                ordered_labels.append(label)
        if not ordered_labels:
            ordered_labels = list(labels)

        return [sensors[label] for label in ordered_labels if label in sensors]

    @staticmethod
    def _sum_filtered_contact_data(data: torch.Tensor) -> torch.Tensor:
        data = torch.nan_to_num(data, nan=0.0)
        if data.ndim == 4:
            return data.sum(dim=(1, 2))
        if data.ndim == 3:
            return data.sum(dim=1)
        return data.reshape(data.shape[0], -1, 3).sum(dim=1)

    def _contact_reaction_forces_w(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        foot_sensors = self._foot_reaction_sensors()
        if foot_sensors:
            normal_forces = []
            friction_forces = []
            for sensor in foot_sensors:
                data = sensor.data
                normal = getattr(data, "force_matrix_w", None)
                friction = getattr(data, "friction_forces_w", None)
                if normal is None or friction is None:
                    raise RuntimeError("per-foot contact sensor does not provide normal + friction forces")
                normal_forces.append(self._sum_filtered_contact_data(normal))
                friction_forces.append(self._sum_filtered_contact_data(friction))
            normal_forces_w = torch.stack(normal_forces, dim=1)
            friction_forces_w = torch.stack(friction_forces, dim=1)
            return normal_forces_w + friction_forces_w, normal_forces_w, friction_forces_w

        sensor = self.raw_env._contact_sensor
        data = sensor.data
        body_ids = self.raw_env._feet_body_ids

        normal_forces_w = getattr(data, "force_matrix_w", None)
        if normal_forces_w is not None:
            normal_forces_w = torch.nan_to_num(normal_forces_w[:, body_ids, :, :], nan=0.0).sum(dim=2)
        else:
            normal_forces_w = data.net_forces_w[:, body_ids, :]

        friction_forces_w = getattr(data, "friction_forces_w", None)
        if friction_forces_w is None:
            raise RuntimeError("contact sensor does not provide friction_forces_w")
        if friction_forces_w.ndim == 4:
            friction_forces_w = torch.nan_to_num(friction_forces_w[:, body_ids, :, :], nan=0.0).sum(dim=2)
        else:
            friction_forces_w = torch.nan_to_num(friction_forces_w[:, body_ids, :], nan=0.0)

        # Reorder from contact-sensor foot order into robot foot order so that the forces
        # below pair with the robot-ordered velocities/positions/labels used in _read_states.
        perm = self._contact_to_robot_perm()
        normal_forces_w = normal_forces_w[:, perm, :]
        friction_forces_w = friction_forces_w[:, perm, :]

        return normal_forces_w + friction_forces_w, normal_forces_w, friction_forces_w

    def _instant_contact_mask(self, normal_force_norm: torch.Tensor) -> torch.Tensor:
        threshold = float(getattr(getattr(self.raw_env, "cfg", None), "base_contact_threshold", 1.0))
        return normal_force_norm > threshold

    def _foot_friction_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_feet = len(getattr(self.raw_env, "_feet_body_ids", []))
        try:
            static_default = float(getattr(self.raw_env.cfg, "friction_static_range")[0])
        except Exception:
            static_default = float(getattr(self.raw_env.cfg, "gt_obs_default_mu", 1.0))
        try:
            dynamic_default = static_default * float(getattr(self.raw_env.cfg, "mu_dynamic_static_ratio"))
        except Exception:
            dynamic_default = static_default
        default_static = torch.full((num_feet,), static_default, device=self.raw_env.device)
        default_dynamic = torch.full((num_feet,), dynamic_default, device=self.raw_env.device)
        if (
            getattr(self.raw_env, "_patch_friction_static", torch.empty(0)).numel() == 0
            or getattr(self.raw_env, "_patch_xy_min", torch.empty(0)).numel() == 0
        ):
            return default_static, default_dynamic

        foot_xy = self._foot_reference_positions_w()[self.env_index, :, :2]
        foot_xy = foot_xy - self.raw_env.scene.env_origins[self.env_index, :2]
        inside_patch = torch.logical_and(
            foot_xy[:, None, :] >= self.raw_env._patch_xy_min[None, :, :],
            foot_xy[:, None, :] <= self.raw_env._patch_xy_max[None, :, :],
        ).all(dim=-1)
        has_patch = inside_patch.any(dim=-1)
        patch_ids = torch.argmax(inside_patch.to(torch.long), dim=-1)
        env_ids = torch.full_like(patch_ids, self.env_index)
        static_mu = self.raw_env._patch_friction_static[env_ids, patch_ids]
        dynamic_mu = self.raw_env._patch_friction_dynamic[env_ids, patch_ids]
        return torch.where(has_patch, static_mu, default_static), torch.where(has_patch, dynamic_mu, default_dynamic)

    def _contact_points_w(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        foot_sensors = self._foot_reaction_sensors()
        if foot_sensors:
            contact_positions = []
            valid_contacts = []
            for sensor in foot_sensors:
                contact_pos_w = getattr(sensor.data, "contact_pos_w", None)
                if contact_pos_w is None:
                    return None
                valid = torch.isfinite(contact_pos_w).all(dim=-1)
                count = valid.to(dtype=contact_pos_w.dtype).sum(dim=(1, 2))
                summed = torch.where(valid[..., None], contact_pos_w, torch.zeros_like(contact_pos_w)).sum(dim=(1, 2))
                contact_positions.append(summed / count.clamp_min(1.0).unsqueeze(-1))
                valid_contacts.append(count > 0)
            return torch.stack(contact_positions, dim=1), torch.stack(valid_contacts, dim=1)

        contact_pos_w = getattr(self.raw_env._contact_sensor.data, "contact_pos_w", None)
        if contact_pos_w is None:
            return None

        body_ids = self.raw_env._feet_body_ids
        # Gather the feet in contact-sensor order, then reorder into robot foot order so the
        # returned contact points align with the robot-ordered foot positions/velocities.
        contact_pos_w = contact_pos_w[:, body_ids][:, self._contact_to_robot_perm()]
        if contact_pos_w.ndim == 3:
            valid = torch.isfinite(contact_pos_w).all(dim=-1)
            return contact_pos_w, valid
        if contact_pos_w.ndim != 4:
            return None

        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        count = valid.to(dtype=contact_pos_w.dtype).sum(dim=2)
        summed = torch.where(valid[..., None], contact_pos_w, torch.zeros_like(contact_pos_w)).sum(dim=2)
        contact_points_w = summed / count.clamp_min(1.0).unsqueeze(-1)
        return contact_points_w, count > 0

    def _foot_reference_positions_w(self) -> torch.Tensor:
        foot_pos_w = self.raw_env._get_foot_positions_w()
        contact_data = self._contact_points_w()
        if contact_data is None:
            return foot_pos_w

        contact_points_w, valid_contact = contact_data
        return torch.where(valid_contact[..., None], contact_points_w, foot_pos_w)

    def _contact_to_robot_perm(self) -> torch.Tensor:
        """Permutation that reorders contact-sensor-ordered feet into robot-body order.

        The contact sensor and the robot articulation index the feet in *independent*
        body-index spaces whose order is not guaranteed to match. The visualizer reads
        reaction forces from the contact sensor (``_feet_body_ids`` order) but velocities,
        positions, and labels from the robot (``_feet_robot_body_ids`` order). If those two
        orders differ, a stance foot's force gets paired with a swing foot's velocity, which
        shows up as "in contact but moving several m/s". This builds a per-foot permutation,
        matched by the FL/FR/RL/RR label, so every contact-sourced tensor can be reordered
        into the robot foot order before pairing.
        """

        if self._contact_to_robot_perm_cache is not None:
            return self._contact_to_robot_perm_cache

        contact_names = list(getattr(self.raw_env, "_feet_body_names", []) or [])
        robot_names = list(getattr(self.raw_env, "_feet_robot_body_names", []) or [])
        device = getattr(self.raw_env, "device", "cpu")

        perm: list[int] | None = None
        if robot_names and len(contact_names) == len(robot_names):
            contact_label_to_pos: dict[str, int] = {}
            for pos, name in enumerate(contact_names):
                label = FootFrictionPopup._label_from_foot_name(str(name))
                if label is not None:
                    contact_label_to_pos[label] = pos
            try:
                perm = [contact_label_to_pos[FootFrictionPopup._label_from_foot_name(str(name))] for name in robot_names]
            except (KeyError, TypeError):
                perm = None

        if perm is None:
            perm = list(range(len(robot_names) or len(contact_names)))

        self._contact_to_robot_perm_cache = torch.as_tensor(perm, dtype=torch.long, device=device)
        aligned = perm == list(range(len(perm)))
        pairing = ", ".join(
            f"{FootFrictionPopup._label_from_foot_name(str(robot_names[i])) or '?'}"
            f"[robot:{robot_names[i]} <- contact:{contact_names[perm[i]]}]"
            for i in range(len(perm))
            if i < len(robot_names) and perm[i] < len(contact_names)
        )
        if aligned:
            print(f"[INFO] Slip viz: contact-sensor and robot foot orders already match ({pairing}).", flush=True)
        else:
            print(
                "[WARN] Slip viz: contact-sensor foot order differed from robot foot order; "
                f"realigning by label so forces/velocities pair correctly ({pairing}).",
                flush=True,
            )
        return self._contact_to_robot_perm_cache

    def _foot_link_body_ids(self) -> torch.Tensor | None:
        """Resolve the dedicated ``*_foot`` rigid-body ids on the robot articulation.

        These bodies sit at the feet, so their world linear velocity is the foot velocity
        directly, without propagating the calf link velocity across the calf-to-foot offset.
        Ids are ordered to match the calf-based foot order used by the contact reaction
        forces, friction lookup, positions, and labels.
        """

        if self._foot_link_body_ids_cache is not None:
            return self._foot_link_body_ids_cache if self._foot_link_body_ids_cache.numel() else None

        robot = getattr(self.raw_env, "_robot", None)
        body_names = list(getattr(robot, "body_names", []) or [])
        calf_names = list(getattr(self.raw_env, "_feet_robot_body_names", []) or [])
        foot_ids: list[int] = []
        for calf_name in calf_names:
            foot_name = str(calf_name).replace("_calf", "_foot")
            try:
                foot_ids.append(body_names.index(foot_name))
            except ValueError:
                foot_ids = []
                break

        device = getattr(self.raw_env, "device", "cpu")
        if foot_ids:
            self._foot_link_body_ids_cache = torch.as_tensor(foot_ids, dtype=torch.long, device=device)
            foot_names = [str(c).replace("_calf", "_foot") for c in calf_names]
            print(
                f"[INFO] Slip viz: measuring foot velocity directly from *_foot bodies {foot_names}.",
                flush=True,
            )
        else:
            self._foot_link_body_ids_cache = torch.empty(0, dtype=torch.long, device=device)
            print(
                "[WARN] Slip viz: no *_foot rigid bodies found on the robot; "
                "falling back to calf link velocity propagation.",
                flush=True,
            )
        return self._foot_link_body_ids_cache if self._foot_link_body_ids_cache.numel() else None

    def _foot_velocities_w(self) -> torch.Tensor:
        """World velocity of each foot at its *contact point* (the slip-relevant material point).

        Stick vs. slip is decided by the velocity of the point of the foot that touches the
        ground, not by the velocity of the foot body origin. A foot that rolls or pivots over a
        planted contact (touchdown, push-off) keeps a near-zero contact-point speed while its
        body origin sweeps through several m/s, so reading the body-origin linear velocity alone
        reports spurious "slip" exactly where PhysX is still applying static-band friction.

        We therefore evaluate the rigid-body velocity of the foot at the contact reference point:
        ``v_contact = v_origin + omega x (contact_point - origin)``. ``contact_point`` is the
        sensor contact position when valid (else the geometric foot tip), so the ``omega x r``
        lever follows the contact as it migrates across the foot collider during a roll.
        """
        return self._foot_velocities_at_w(self._foot_reference_positions_w())

    def _foot_origin_velocities_w(self) -> torch.Tensor:
        """Raw foot-body-origin linear velocity, kept only as a diagnostic baseline.

        This is the pre-fix quantity (foot link origin velocity, no ``omega x r`` term). Logged
        next to the corrected contact-point speed so the size of the correction is visible.
        """
        robot = self.raw_env._robot
        data = robot.data
        foot_body_ids = self._foot_link_body_ids()
        if foot_body_ids is not None:
            lin_vel_w = getattr(data, "body_link_lin_vel_w", None)
            if lin_vel_w is not None:
                return lin_vel_w[:, foot_body_ids, :]
            link_vel_w = getattr(data, "body_link_vel_w", None)
            if link_vel_w is not None:
                return link_vel_w[:, foot_body_ids, :3]
        body_ids = self.raw_env._feet_robot_body_ids
        try:
            return data.body_link_vel_w[:, body_ids, :3]
        except AttributeError:
            return data.root_lin_vel_w[:, None, :].expand(-1, len(body_ids), -1)

    def _foot_velocities_at_w(self, point_w: torch.Tensor) -> torch.Tensor:
        """Rigid-body velocity of each foot evaluated at ``point_w`` (env-batched, robot order).

        Uses the ``*_foot`` body when present, otherwise propagates the calf link motion. The
        ``omega x (point - origin)`` term is what turns the body-origin velocity into the velocity
        of the material point at ``point_w``.
        """
        robot = self.raw_env._robot
        data = robot.data
        foot_body_ids = self._foot_link_body_ids()
        body_ids = foot_body_ids if foot_body_ids is not None else self.raw_env._feet_robot_body_ids

        origin_vel_w = getattr(data, "body_link_lin_vel_w", None)
        origin_pos_w = getattr(data, "body_link_pos_w", None)
        origin_ang_vel_w = getattr(data, "body_link_ang_vel_w", None)
        if origin_vel_w is not None and origin_pos_w is not None and origin_ang_vel_w is not None:
            v0 = origin_vel_w[:, body_ids, :]
            p0 = origin_pos_w[:, body_ids, :]
            w0 = origin_ang_vel_w[:, body_ids, :]
            return v0 + torch.linalg.cross(w0, point_w - p0, dim=-1)

        try:
            link_vel_w = data.body_link_vel_w[:, body_ids, :]
            link_pos_w = data.body_link_pos_w[:, body_ids, :]
            return link_vel_w[..., :3] + torch.linalg.cross(link_vel_w[..., 3:6], point_w - link_pos_w, dim=-1)
        except AttributeError:
            pass

        try:
            com_pos_w = data.body_com_pos_w[:, body_ids, :]
            com_lin_vel_w = data.body_com_lin_vel_w[:, body_ids, :]
            com_ang_vel_w = data.body_com_ang_vel_w[:, body_ids, :]
            return com_lin_vel_w + torch.linalg.cross(com_ang_vel_w, point_w - com_pos_w, dim=-1)
        except AttributeError:
            return data.root_lin_vel_w[:, None, :].expand(-1, len(body_ids), -1)

    def _classify(
        self,
        *,
        contact: bool,
        angle_rad: float,
        angle_dynamic_rad: float,
        angle_static_rad: float,
        tangential_speed: float,
    ) -> tuple[str, str]:
        if not contact:
            return "air", "air: no contact"

        slipping = tangential_speed > self.speed_threshold
        if angle_rad < angle_dynamic_rad:
            return "white", "alpha < alpha_dyn"
        if angle_rad > angle_static_rad:
            return "red", "alpha > alpha_static"
        if slipping:
            return "red", "slip in cone band"
        return self._gradient_color_key(angle_rad, angle_dynamic_rad, angle_static_rad), "no slip in cone band"

    @classmethod
    def _gradient_color_key(cls, angle_rad: float, angle_dynamic_rad: float, angle_static_rad: float) -> str:
        span = max(angle_static_rad - angle_dynamic_rad, 1.0e-6)
        t = max(0.0, min(1.0, (angle_rad - angle_dynamic_rad) / span))
        index = int(round(t * (len(cls._ANGLE_GRADIENT_KEYS) - 1)))
        return cls._ANGLE_GRADIENT_KEYS[index]

    @classmethod
    def _display_color_label(cls, color_key: str) -> str:
        if color_key in cls._ANGLE_GRADIENT_KEYS:
            if color_key == "yellow" or color_key == "green":
                return color_key
            return "y->g"
        return color_key

    def _set_error_text(self, message: str):
        for label in self._DISPLAY_ORDER:
            self._models[label].set_value(f"{label}: --")
        self._set_schematic_colors({})
        self._hint_model.set_value(f"Slip cone unavailable: {message}")

    def _set_text(self, states: dict[str, dict[str, object]]):
        display_text = tuple(
            (
                label,
                states.get(label, {}).get("status"),
                states.get(label, {}).get("color_key"),
                states.get(label, {}).get("angle_deg"),
                states.get(label, {}).get("angle_dynamic_deg"),
                states.get(label, {}).get("angle_static_deg"),
                states.get(label, {}).get("rho_static"),
                states.get(label, {}).get("rho_dynamic"),
                states.get(label, {}).get("tangential_speed"),
                states.get(label, {}).get("held"),
            )
            for label in self._DISPLAY_ORDER
        )
        if display_text == self._last_display_text:
            return
        self._last_display_text = display_text
        self._set_schematic_colors(states)

        for label in self._DISPLAY_ORDER:
            state = states.get(label)
            if state is None:
                self._models[label].set_value(f"{label}: --")
                continue
            status = str(state["status"])
            if bool(state.get("held", False)):
                status = f"{status} (held)"
            angle = float(state["angle_deg"])
            dyn_angle = float(state["angle_dynamic_deg"])
            static_angle = float(state["angle_static_deg"])
            rho_static = float(state["rho_static"])
            rho_dynamic = float(state["rho_dynamic"])
            speed = float(state["tangential_speed"])
            color_name = self._display_color_label(str(state["color_key"]))
            self._models[label].set_value(
                f"{label}: {color_name:>6} | a={angle:5.1f}° | dyn={dyn_angle:5.1f}° "
                f"stat={static_angle:5.1f}° | rho_d={rho_dynamic:.2f} "
                f"rho_s={rho_static:.2f} | vc={speed:.3f} | {status}"
            )
        speed_desc = "worst substep/step" if self.contact_stat == "max" else f"{self.contact_stat} over contact"
        self._hint_model.set_value(
            f"Colors use alpha vs alpha_dyn/alpha_static; vc is friction-opposed contact speed ({speed_desc}); "
            f"red in cone band if vc > {self.speed_threshold:.3f}; "
            f"air rows {'shown' if self.viz_air_points else 'hold last contact'}"
        )

    def _draw_viewport_dots(self, states: dict[str, dict[str, object]]):
        if not self._marker_visualizers:
            return

        positions_by_color: dict[str, list[np.ndarray]] = {
            color_key: [] for color_key in self._VIEWPORT_DOT_COLOR_KEYS
        }
        root_pos_w, yaw_rot = self._root_schematic_pose() if self.superior_markers else (None, None)
        for label in self._DISPLAY_ORDER:
            state = states.get(label)
            if state is None:
                continue
            foot = np.asarray(state["foot_pos_w"], dtype=np.float32)
            color_key = str(state.get("color_key", "air"))
            if color_key not in positions_by_color:
                continue

            marker_position = foot + np.array([0.0, 0.0, 0.13], dtype=np.float32)
            if self.superior_markers:
                offset_b = self._OVERHEAD_OFFSETS_B.get(label)
                if root_pos_w is not None and yaw_rot is not None and offset_b is not None:
                    marker_position = (root_pos_w + yaw_rot @ offset_b).astype(np.float32)

            positions_by_color[color_key].append(marker_position)

        for color_key, visualizer in self._marker_visualizers.items():
            positions = positions_by_color.get(color_key, [])
            try:
                if positions:
                    visualizer.set_visibility(True)
                    visualizer.visualize(translations=np.stack(positions, axis=0))
                else:
                    visualizer.set_visibility(False)
            except Exception as exc:
                if not self._warned_debug_draw:
                    print(f"[WARN] Could not draw slip-cone viewport dots: {type(exc).__name__}: {exc}", flush=True)
                    self._warned_debug_draw = True

    def _root_schematic_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            root_pos_w = self.raw_env._robot.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
            root_quat_w = self.raw_env._robot.data.root_quat_w[self.env_index].detach().cpu()
            yaw = SideFollowCamera._quat_wxyz_to_yaw(root_quat_w)
            return root_pos_w, SideFollowCamera._yaw_rot(yaw)
        except Exception:
            return None, None


class PlaySpeedControl:
    """Small interactive UI that throttles the play loop against wall time."""

    _MIN_SPEED = 0.05
    _MAX_SPEED = 3.0

    def __init__(self, speed_state: dict[str, float], *, initial_speed: float = 1.0):
        import omni.ui as ui

        self.speed_state = speed_state
        self._subscriptions = []
        self._updating_model = False

        initial_speed = self._clip_speed(initial_speed)
        self.speed_state["speed"] = initial_speed
        self._speed_model = ui.SimpleFloatModel(initial_speed, min=self._MIN_SPEED, max=self._MAX_SPEED)
        self._status_model = ui.SimpleStringModel("")

        self.window = ui.Window("Play speed", width=380, height=160, visible=True)
        _set_window_bounds(self.window, x=840, y=265, width=380, height=160)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Play speed", height=22)
                with ui.HStack(spacing=8, height=32):
                    for speed in (0.25, 0.5, 1.0, 1.25, 2.0):
                        ui.Button(f"{speed:g}x", clicked_fn=lambda value=speed: self.set_speed(value))
                with ui.HStack(spacing=8, height=32):
                    ui.Label("speed", width=52)
                    ui.FloatField(self._speed_model, width=90)
                    ui.FloatSlider(self._speed_model, min=self._MIN_SPEED, max=self._MAX_SPEED, width=190)
                ui.StringField(self._status_model, read_only=True, height=28)

        def _on_speed_change(model):
            if self._updating_model:
                return
            try:
                self.set_speed(float(model.get_value_as_float()), update_model=False)
            except Exception:
                return

        if hasattr(self._speed_model, "subscribe_value_changed_fn"):
            self._subscriptions.append(self._speed_model.subscribe_value_changed_fn(_on_speed_change))
        else:
            self._speed_model.add_value_changed_fn(_on_speed_change)
            self._subscriptions.append(_on_speed_change)
        self._update_status()

    def set_speed(self, speed: float, *, update_model: bool = True):
        speed = self._clip_speed(speed)
        self.speed_state["speed"] = speed
        if update_model:
            self._updating_model = True
            try:
                self._speed_model.set_value(speed)
            finally:
                self._updating_model = False
        self._update_status()

    def get_speed(self) -> float:
        return self._clip_speed(self.speed_state.get("speed", 1.0))

    def close(self):
        try:
            self.window.visible = False
        except Exception:
            pass

    def _update_status(self):
        speed = self.get_speed()
        if speed < 1.0:
            text = f"{speed:.2f}x playback; sim is slowed down on screen"
        elif speed > 1.0:
            text = f"{speed:.2f}x playback; sim is sped up if rendering can keep up"
        else:
            text = "1.00x playback; sim time follows wall time"
        self._status_model.set_value(text)

    @classmethod
    def _clip_speed(cls, speed: float) -> float:
        if not math.isfinite(float(speed)):
            return 1.0
        return min(cls._MAX_SPEED, max(cls._MIN_SPEED, float(speed)))


class PatchFrictionSlider:
    """Play-only control for race patch friction grouping, live slider apply, and immediate resampling."""

    def __init__(self, raw_env, env_index: int = 0):
        import omni.ui as ui
        import omni.usd

        self.raw_env = raw_env
        self.env_index = int(env_index)
        self._selection = omni.usd.get_context().get_selection()
        self._original_sample_and_apply_track_layout = getattr(raw_env, "_sample_and_apply_track_layout", None)
        if not callable(self._original_sample_and_apply_track_layout):
            raise RuntimeError("raw env has no race patch layout sampler")

        cfg = getattr(raw_env, "cfg", None)
        if not getattr(cfg, "randomize_fric_coefs", False):
            raise RuntimeError("randomize_fric_coefs is False, so no friction bucket sampling is active")
        if not callable(getattr(raw_env, "_apply_patch_materials", None)):
            raise RuntimeError("raw env has no patch-material applier")
        if len(getattr(raw_env, "_patch_rel_paths", []) or []) == 0:
            raise RuntimeError("no friction patches were found")

        bucket_values = getattr(raw_env, "_patch_material_bucket_values", None)
        if not isinstance(bucket_values, torch.Tensor) or bucket_values.ndim < 2 or bucket_values.shape[0] < 2:
            raise RuntimeError("at least two friction buckets are required")
        bucket_values_cpu = bucket_values.detach().float().cpu()
        self._bucket_static = bucket_values_cpu[:, 0].numpy()
        self._bucket_dynamic = bucket_values_cpu[:, 1].numpy()
        self._range_min, self._range_max = self._friction_range_from_cfg()
        if self._range_max <= self._range_min:
            raise RuntimeError(
                f"friction_static_range is degenerate ({self._range_min:g}, {self._range_max:g}); slider is not useful"
            )

        self._subscriptions = []
        self._last_display_text = None
        self._last_applied_bucket_idx: int | None = None
        self._last_applied_env_count = 0
        self._warned_apply_failure = False
        self._setting_slider_model = False
        self._applying_slider_change = False
        # The slider is a manual/fixed-friction reset override.  Automatic within-episode randomization must own
        # reset-time sampling while it is enabled; otherwise every reset snaps back to the selected slider bucket and
        # looks like randomization is broken.
        self._manual_slider_reset_enabled = not self._within_episode_resample_enabled()

        initial_bucket_idx = self._nearest_bucket_idx(self._current_static_mu_or_default())
        self._selected_bucket_idx = initial_bucket_idx
        initial_mu = float(self._bucket_static[initial_bucket_idx])

        self._slider_model = ui.SimpleFloatModel(initial_mu, min=self._range_min, max=self._range_max)
        self._mode_model = ui.SimpleStringModel("")
        self._slider_mode_model = ui.SimpleStringModel("")
        self._pending_model = ui.SimpleStringModel("")
        self._selected_model = ui.SimpleStringModel("")
        self._active_model = ui.SimpleStringModel("")
        self._hint_model = ui.SimpleStringModel("")
        self._resample_model = ui.SimpleStringModel("")
        self._status_model = ui.SimpleStringModel("")

        self.window = ui.Window("Race friction control", width=560, height=370, visible=True)
        _set_window_bounds(self.window, x=400, y=250, width=560, height=370)
        with self.window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Patch friction control", height=22)
                with ui.HStack(spacing=8, height=32):
                    ui.Button("Toggle slider", clicked_fn=self._toggle_slider_mode)
                    ui.Button("Toggle within-episode", clicked_fn=self._toggle_within_episode_resample)
                with ui.HStack(spacing=8, height=32):
                    ui.Button("Grouped sampling", clicked_fn=lambda: self._set_grouping(True))
                    ui.Button("Per-patch sampling", clicked_fn=lambda: self._set_grouping(False))
                with ui.HStack(spacing=8, height=32):
                    ui.Label("mu_static", width=80)
                    ui.FloatField(self._slider_model, width=110)
                    ui.FloatSlider(self._slider_model, min=self._range_min, max=self._range_max, width=280)
                with ui.HStack(spacing=8, height=34):
                    ui.Button("Apply now", clicked_fn=self._apply_selected_bucket_now)
                    ui.Button("Resample now", clicked_fn=self._resample_now)
                with ui.HStack(spacing=8, height=34):
                    ui.Button("Within-episode ON", clicked_fn=lambda: self._set_within_episode_resample(True))
                    ui.Button("Within-episode OFF", clicked_fn=lambda: self._set_within_episode_resample(False))
                ui.StringField(self._mode_model, read_only=True, height=26)
                ui.StringField(self._slider_mode_model, read_only=True, height=26)
                ui.StringField(self._pending_model, read_only=True, height=26)
                ui.StringField(self._selected_model, read_only=True, height=26)
                ui.StringField(self._active_model, read_only=True, height=26)
                ui.StringField(self._hint_model, read_only=True, height=26)
                ui.StringField(self._resample_model, read_only=True, height=26)
                ui.StringField(self._status_model, read_only=True, height=26)

        def _on_slider_change(model):
            if self._setting_slider_model:
                return
            try:
                requested_mu = float(model.get_value_as_float())
            except Exception:
                return
            self._selected_bucket_idx = self._nearest_bucket_idx(requested_mu)
            snapped_mu = float(self._bucket_static[self._selected_bucket_idx])
            if abs(snapped_mu - requested_mu) > 1.0e-7:
                self._setting_slider_model = True
                try:
                    self._slider_model.set_value(snapped_mu)
                finally:
                    self._setting_slider_model = False
            if self._slider_controls_reset():
                self._apply_selected_bucket_now(source="slider")
                return
            self.update(force=True)

        if hasattr(self._slider_model, "subscribe_value_changed_fn"):
            self._subscriptions.append(self._slider_model.subscribe_value_changed_fn(_on_slider_change))
        else:
            self._slider_model.add_value_changed_fn(_on_slider_change)
            self._subscriptions.append(_on_slider_change)

        # Keep the grouped-slider reset override only while automatic randomization is off.  In per-patch mode, or
        # when within-episode randomization is on, reset sampling falls back to the env's normal random sampler.
        raw_env._sample_and_apply_track_layout = self._sample_and_apply_track_layout
        self.update(force=True)

    def close(self):
        if callable(self._original_sample_and_apply_track_layout):
            try:
                self.raw_env._sample_and_apply_track_layout = self._original_sample_and_apply_track_layout
            except Exception:
                pass

    def update(self, *, force: bool = False):
        grouped = self._grouped()
        auto_resample = self._within_episode_resample_enabled()
        slider_reset = self._slider_controls_reset()
        bucket_idx = int(self._selected_bucket_idx)
        pending_static, pending_dynamic = self._bucket_values(bucket_idx)
        mode_text = "Sampler: grouped all patches share one random bucket" if grouped else "Sampler: per-patch independent random buckets"
        slider_mode_text = (
            f"Slider reset: {'🟢 ON' if slider_reset else '🔴 OFF'}  |  "
            f"Within-episode: {'🟢 ON' if auto_resample else '🔴 OFF'}"
        )
        slider_role = "live + reset bucket" if slider_reset else "Apply-now only"
        pending_text = (
            f"Slider ({slider_role}): bucket={bucket_idx:03d}/{len(self._bucket_static) - 1:03d}  "
            f"mu_static={pending_static:.3f}  mu_dynamic={pending_dynamic:.3f}"
        )

        selected = self._selected_patch_info()
        if selected is None:
            selected_text = "Selected patch: -- (click a colored patch for per-patch Apply now)"
            selected_patch_idx = None
            selected_env_index = self.env_index
        else:
            patch_name, selected_patch_idx, selected_env_index, _ = selected
            selected_text = f"Selected patch: {patch_name}  |  env {selected_env_index}"

        active_info = self._current_patch_info(env_index=selected_env_index, patch_idx=selected_patch_idx)
        if active_info is None:
            active_text = f"Active env {selected_env_index}: --"
        else:
            active_patch, active_bucket, active_static, active_dynamic = active_info
            active_text = (
                f"Active {active_patch} env {selected_env_index}: bucket={active_bucket:03d}  "
                f"mu_static={active_static:.3f}  mu_dynamic={active_dynamic:.3f}"
            )
        if self._last_applied_bucket_idx is not None:
            active_text += f"  |  last apply: {self._last_applied_env_count} env(s)"

        if auto_resample:
            if grouped:
                hint_text = "Within-episode ON: resets/timed resamples pick one random bucket for all patches."
            else:
                hint_text = "Within-episode ON: resets/timed resamples pick independent random buckets per patch."
        elif slider_reset:
            hint_text = "Slider ON: slider/input changes are applied immediately and reset uses the slider bucket."
        elif grouped:
            hint_text = "Slider OFF: reset/resample use the grouped random sampler."
        else:
            hint_text = "Slider OFF: reset/resample use per-patch random buckets; Apply now updates only selected patch."
        resample_text = self._within_episode_resample_text()

        display_text = (mode_text, slider_mode_text, pending_text, selected_text, active_text, hint_text, resample_text)
        if not force and display_text == self._last_display_text:
            return
        self._last_display_text = display_text
        self._mode_model.set_value(mode_text)
        self._slider_mode_model.set_value(slider_mode_text)
        self._pending_model.set_value(pending_text)
        self._selected_model.set_value(selected_text)
        self._active_model.set_value(active_text)
        self._hint_model.set_value(hint_text)
        self._resample_model.set_value(resample_text)

    def _sample_and_apply_track_layout(self, env_ids, *, update_usd: bool = True):
        try:
            env_ids = self._env_ids_tensor(env_ids)
            if self._slider_controls_reset():
                bucket_idx = int(np.clip(self._selected_bucket_idx, 0, len(self._bucket_static) - 1))
                with torch.inference_mode():
                    if hasattr(self.raw_env, "apply_uniform_patch_friction_bucket"):
                        self.raw_env.apply_uniform_patch_friction_bucket(bucket_idx, env_ids=env_ids, update_usd=update_usd)
                    else:
                        num_patches = len(getattr(self.raw_env, "_patch_rel_paths", []) or [])
                        bucket_ids = torch.full(
                            (len(env_ids), num_patches), bucket_idx, device=self.raw_env.device, dtype=torch.long
                        )
                        self.raw_env._apply_patch_materials(bucket_ids, env_ids=env_ids, update_usd=update_usd)
                self._last_applied_bucket_idx = bucket_idx
                self._last_applied_env_count = int(len(env_ids))
            else:
                self._call_original_sampler(env_ids, update_usd=update_usd)
                self._last_applied_bucket_idx = None
                self._last_applied_env_count = int(len(env_ids))
            self.update(force=True)
        except Exception as exc:
            if not self._warned_apply_failure:
                print(
                    f"[WARN] Friction control failed to apply selected bucket; falling back to random sampler: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_apply_failure = True
            return self._call_original_sampler(env_ids, update_usd=update_usd)

    def _set_grouping(self, grouped: bool):
        try:
            if hasattr(self.raw_env, "set_patch_friction_grouping"):
                self.raw_env.set_patch_friction_grouping(bool(grouped))
            else:
                self.raw_env.cfg.group_all_patches_single_bucket = bool(grouped)
            if self._within_episode_resample_enabled():
                with torch.inference_mode():
                    self._resample_all_envs_now(update_usd=True)
                    self._schedule_all_envs_for_auto_resample()
                self._status_model.set_value("Sampling mode changed; within-episode randomization resampled now.")
            elif self._slider_controls_reset():
                self._status_model.set_value("Sampling mode changed, but slider reset is ON until toggled off.")
            else:
                self._status_model.set_value("Sampling mode changed; reset/resample uses the new sampler mode.")
        except Exception as exc:
            self._status_model.set_value(f"Could not change grouping: {type(exc).__name__}: {exc}")
        self.update(force=True)

    def _apply_selected_bucket_now(self, *, source: str = "button"):
        if source == "slider" and self._applying_slider_change:
            return
        bucket_idx = int(np.clip(self._selected_bucket_idx, 0, len(self._bucket_static) - 1))
        selected = self._selected_patch_info()
        env_index = self.env_index if selected is None else int(selected[2])
        try:
            if source == "slider":
                self._applying_slider_change = True
            with torch.inference_mode():
                if self._slider_controls_reset() or self._grouped():
                    if hasattr(self.raw_env, "apply_uniform_patch_friction_bucket"):
                        self.raw_env.apply_uniform_patch_friction_bucket(bucket_idx, env_ids=[env_index], update_usd=True)
                    else:
                        num_patches = len(getattr(self.raw_env, "_patch_rel_paths", []) or [])
                        bucket_ids = torch.full((1, num_patches), bucket_idx, device=self.raw_env.device, dtype=torch.long)
                        self.raw_env._apply_patch_materials(bucket_ids, env_ids=torch.tensor([env_index], device=self.raw_env.device), update_usd=True)
                    suffix = " Auto resampling may overwrite it." if self._within_episode_resample_enabled() else ""
                    verb = "Live-applied" if source == "slider" else "Applied"
                    self._status_model.set_value(
                        f"{verb} bucket {bucket_idx:03d} to all patches in env {env_index}.{suffix}"
                    )
                else:
                    if selected is None:
                        self._status_model.set_value("Select a patch first, then Apply now in per-patch mode.")
                        self.update(force=True)
                        return
                    patch_name, patch_idx, env_index, _ = selected
                    if hasattr(self.raw_env, "apply_single_patch_friction_bucket"):
                        self.raw_env.apply_single_patch_friction_bucket(env_index, patch_idx, bucket_idx, update_usd=True)
                    else:
                        bucket_ids = self.raw_env._patch_bucket_ids[env_index : env_index + 1].clone()
                        bucket_ids[:, patch_idx] = bucket_idx
                        self.raw_env._apply_patch_materials(bucket_ids, env_ids=torch.tensor([env_index], device=self.raw_env.device), update_usd=True)
                    suffix = " Auto resampling may overwrite it." if self._within_episode_resample_enabled() else ""
                    verb = "Live-applied" if source == "slider" else "Applied"
                    self._status_model.set_value(f"{verb} bucket {bucket_idx:03d} to {patch_name} in env {env_index}.{suffix}")
            self._last_applied_bucket_idx = bucket_idx
            self._last_applied_env_count = 1
        except Exception as exc:
            action = "Live slider apply" if source == "slider" else "Apply now"
            self._status_model.set_value(f"{action} failed: {type(exc).__name__}: {exc}")
        finally:
            if source == "slider":
                self._applying_slider_change = False
        self.update(force=True)

    def _resample_now(self):
        selected = self._selected_patch_info()
        env_index = self.env_index if selected is None else int(selected[2])
        try:
            with torch.inference_mode():
                if hasattr(self.raw_env, "resample_patch_friction"):
                    self.raw_env.resample_patch_friction(env_ids=[env_index], update_usd=True)
                else:
                    self._call_original_sampler(torch.tensor([env_index], device=self.raw_env.device), update_usd=True)
            if self._within_episode_resample_enabled():
                self._schedule_all_envs_for_auto_resample()
            mode = "grouped" if self._grouped() else "per-patch"
            suffix = " Timer restarted." if self._within_episode_resample_enabled() else ""
            self._status_model.set_value(f"Resampled {mode} patch friction for env {env_index} now.{suffix}")
            self._last_applied_bucket_idx = None
            self._last_applied_env_count = 1
        except Exception as exc:
            self._status_model.set_value(f"Resample now failed: {type(exc).__name__}: {exc}")
        self.update(force=True)

    def _toggle_slider_mode(self):
        self._set_slider_mode(not self._slider_controls_reset())

    def _toggle_within_episode_resample(self):
        self._set_within_episode_resample(not self._within_episode_resample_enabled())

    def _set_slider_mode(self, enabled: bool):
        try:
            enabled = bool(enabled)
            self._manual_slider_reset_enabled = enabled
            if enabled:
                with torch.inference_mode():
                    if hasattr(self.raw_env, "set_within_episode_patch_friction_resampling"):
                        self.raw_env.set_within_episode_patch_friction_resampling(False)
                    else:
                        self.raw_env.cfg.within_episode_fric_resample = False
                        if hasattr(self.raw_env, "_next_friction_resample_time_s"):
                            self.raw_env._next_friction_resample_time_s[:] = float("inf")
                self._status_model.set_value("Slider reset enabled; applying selected bucket now.")
                self._apply_selected_bucket_now(source="slider")
            else:
                self._status_model.set_value("Slider reset disabled; resets use the random sampler.")
        except Exception as exc:
            self._status_model.set_value(f"Could not toggle slider mode: {type(exc).__name__}: {exc}")
        self.update(force=True)

    def _set_within_episode_resample(self, enabled: bool):
        try:
            enabled = bool(enabled)
            if enabled:
                self._manual_slider_reset_enabled = False
            with torch.inference_mode():
                if hasattr(self.raw_env, "set_within_episode_patch_friction_resampling"):
                    self.raw_env.set_within_episode_patch_friction_resampling(
                        enabled,
                        update_usd_on_resample=True,
                        resample_now=enabled,
                    )
                else:
                    self.raw_env.cfg.within_episode_fric_resample = enabled
                    self.raw_env.cfg.within_episode_fric_resample_update_usd = bool(enabled)
                    if not enabled and hasattr(self.raw_env, "_next_friction_resample_time_s"):
                        self.raw_env._next_friction_resample_time_s[:] = float("inf")
                    elif enabled:
                        self._resample_all_envs_now(update_usd=True)
                        self._schedule_all_envs_for_auto_resample()
            if enabled:
                self._last_applied_bucket_idx = None
                self._last_applied_env_count = int(getattr(self.raw_env, "num_envs", 0))
                self._status_model.set_value(
                    "Within-episode randomization enabled; resampled now, colors update on timed resamples, slider reset OFF."
                )
            else:
                self._status_model.set_value("Within-episode randomization disabled; slider state unchanged.")
        except Exception as exc:
            self._status_model.set_value(f"Could not toggle within-episode resampling: {type(exc).__name__}: {exc}")
        self.update(force=True)

    def _resample_all_envs_now(self, *, update_usd: bool):
        env_ids = torch.arange(self.raw_env.num_envs, device=self.raw_env.device, dtype=torch.long)
        if hasattr(self.raw_env, "resample_patch_friction"):
            self.raw_env.resample_patch_friction(env_ids=env_ids, update_usd=update_usd)
        else:
            self._call_original_sampler(env_ids, update_usd=update_usd)

    def _schedule_all_envs_for_auto_resample(self):
        if hasattr(self.raw_env, "_schedule_next_patch_friction_resample"):
            self.raw_env._schedule_next_patch_friction_resample(
                torch.arange(self.raw_env.num_envs, device=self.raw_env.device, dtype=torch.long)
            )

    def _call_original_sampler(self, env_ids, *, update_usd: bool):
        try:
            return self._original_sample_and_apply_track_layout(env_ids, update_usd=update_usd)
        except TypeError:
            return self._original_sample_and_apply_track_layout(env_ids)

    def _env_ids_tensor(self, env_ids) -> torch.Tensor:
        if hasattr(self.raw_env, "_env_ids_tensor"):
            return self.raw_env._env_ids_tensor(env_ids)
        if env_ids is None:
            return torch.arange(self.raw_env.num_envs, device=self.raw_env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.raw_env.device, dtype=torch.long).reshape(-1)
        if isinstance(env_ids, slice):
            return torch.arange(self.raw_env.num_envs, device=self.raw_env.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], device=self.raw_env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.raw_env.device, dtype=torch.long).reshape(-1)

    def _grouped(self) -> bool:
        return bool(getattr(getattr(self.raw_env, "cfg", None), "group_all_patches_single_bucket", False))

    def _within_episode_resample_enabled(self) -> bool:
        return bool(getattr(getattr(self.raw_env, "cfg", None), "within_episode_fric_resample", False))

    def _slider_controls_reset(self) -> bool:
        return bool(self._manual_slider_reset_enabled) and not self._within_episode_resample_enabled()

    def _selected_patch_info(self) -> tuple[str, int, int, str] | None:
        try:
            selected_paths = self._selection.get_selected_prim_paths()
        except Exception:
            selected_paths = []
        if not selected_paths:
            return None
        selected_path = selected_paths[-1]
        patch_name = PatchFrictionSelectionPopup._patch_name_from_path(selected_path)
        if patch_name is None:
            return None
        patch_idx = None
        if hasattr(self.raw_env, "get_patch_index_from_name"):
            patch_idx = self.raw_env.get_patch_index_from_name(patch_name)
        if patch_idx is None:
            rel_paths = [str(path).rstrip("/") for path in (getattr(self.raw_env, "_patch_rel_paths", []) or [])]
            rel_names = [path.split("/")[-1] for path in rel_paths]
            if patch_name in rel_names:
                patch_idx = rel_names.index(patch_name)
        if patch_idx is None:
            patch_names = [str(name) for name in (getattr(self.raw_env, "_patch_names", []) or [])]
            if patch_name not in patch_names:
                return None
            patch_idx = patch_names.index(patch_name)
        env_index = PatchFrictionSelectionPopup._env_index_from_path(selected_path, self.env_index)
        return patch_name, int(patch_idx), int(env_index), selected_path

    def _within_episode_resample_text(self) -> str:
        enabled = self._within_episode_resample_enabled()
        try:
            low, high = getattr(self.raw_env.cfg, "within_episode_fric_resample_time_range")
            low = float(low)
            high = float(high)
        except Exception:
            low, high = 0.0, 0.0
        state = "ON" if enabled else "OFF"
        reset_owner = "slider" if self._slider_controls_reset() else "random sampler"
        return f"Within-episode resample: {state}  range={low:.2f}-{high:.2f}s  reset={reset_owner}"

    def _friction_range_from_cfg(self) -> tuple[float, float]:
        try:
            low, high = getattr(self.raw_env.cfg, "friction_static_range")
            low = float(low)
            high = float(high)
        except Exception:
            low = float(np.nanmin(self._bucket_static))
            high = float(np.nanmax(self._bucket_static))
        if not (math.isfinite(low) and math.isfinite(high)):
            low = float(np.nanmin(self._bucket_static))
            high = float(np.nanmax(self._bucket_static))
        if high < low:
            low, high = high, low
        return low, high

    def _nearest_bucket_idx(self, requested_mu: float) -> int:
        values = np.asarray(self._bucket_static, dtype=np.float64)
        requested_mu = float(np.clip(requested_mu, self._range_min, self._range_max))
        errors = np.abs(values - requested_mu)
        errors = np.where(np.isfinite(errors), errors, np.inf)
        return int(np.argmin(errors))

    def _current_static_mu_or_default(self) -> float:
        active_info = self._current_patch_info(env_index=self.env_index, patch_idx=0)
        if active_info is not None:
            return float(active_info[2])
        return 0.5 * (self._range_min + self._range_max)

    def _current_patch_info(self, *, env_index: int, patch_idx: int | None) -> tuple[str, int, float, float] | None:
        try:
            summary = self.raw_env.get_patch_friction_summary(env_index=env_index)
            if not summary:
                return None
            if patch_idx is None:
                patch_idx = 0
            if patch_idx < 0 or patch_idx >= len(summary):
                return None
            patch = summary[patch_idx]
            return str(patch["patch"]), int(patch["bucket"]), float(patch["static"]), float(patch["dynamic"])
        except Exception:
            return None

    def _bucket_values(self, bucket_idx: int) -> tuple[float, float]:
        bucket_idx = int(np.clip(bucket_idx, 0, len(self._bucket_static) - 1))
        return float(self._bucket_static[bucket_idx]), float(self._bucket_dynamic[bucket_idx])

def _as_rgb_uint8_frame(frame) -> np.ndarray | None:
    if frame is None:
        return None
    if isinstance(frame, (tuple, list)):
        frame = next((item for item in frame if item is not None), None)
        if frame is None:
            return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    try:
        array = np.asarray(frame)
    except Exception:
        return None
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        return None
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.shape[-1] != 3:
        return None
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = 255.0 * array
        array = np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _rgb_frame_looks_empty(frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return True
    try:
        frame_f = np.asarray(frame, dtype=np.float32)
        max_value = float(np.nanmax(frame_f))
        mean_value = float(np.nanmean(frame_f))
        std_value = float(np.nanstd(frame_f))
        # Black buffers are common while RTX/Replicator warms up.  In this headless path we have also seen
        # all-white LdrColor buffers when the render product is not fully wired to the camera AOV yet.  Treat
        # both near-constant extremes as invalid so we can retry/fall back instead of accepting a white race view.
        nearly_black = max_value <= 2.0 and mean_value <= 1.0
        nearly_white = mean_value >= 253.0 and std_value <= 2.0
        nearly_constant = std_value <= 0.5
        return nearly_black or nearly_white or nearly_constant
    except Exception:
        return True


def _render_rgb_warmup(raw_env, *, frames: int = 4) -> None:
    """Give RTX/replicator render products a few ticks before reading RGB data.

    This helper is used while a policy rollout is in progress.  Keep it physics-step-free: synchronizing the scene
    to the renderer is OK, but do not call the Kit app directly here.  A bare ``omni.kit.app.get_app().update()`` can
    advance Isaac while playSimulations is enabled, which made detached training-time eval videos show repeated early
    falls even though the same checkpoint played correctly without recording.
    """

    for _ in range(max(0, int(frames))):
        try:
            raw_env.scene.write_data_to_sim()
        except Exception:
            pass
        try:
            raw_env.sim.forward()
        except Exception:
            pass
        try:
            raw_env.sim.render()
        except Exception:
            pass
        try:
            # Some Replicator render products need one extra Kit update after attachment before their buffers become
            # readable.  Keep physics playback disabled around it; the previous bare app.update() here could advance
            # the simulation between policy steps and poison periodic eval rollouts.
            raw_env.sim.set_setting("/app/player/playSimulations", False)
            import omni.kit.app

            omni.kit.app.get_app().update()
        except Exception:
            pass
        finally:
            try:
                raw_env.sim.set_setting("/app/player/playSimulations", True)
            except Exception:
                pass


def _detach_rgb_render_product(raw_env) -> None:
    annotator = getattr(raw_env, "_rgb_annotator", None)
    render_product = getattr(raw_env, "_render_product", None)
    if annotator is not None and render_product is not None:
        try:
            annotator.detach([render_product])
        except Exception:
            try:
                annotator.detach(render_product)
            except Exception as exc:
                print(f"[WARN] Could not detach RGB annotator: {type(exc).__name__}: {exc}", flush=True)
    for attr in ("_rgb_annotator", "_render_product"):
        if hasattr(raw_env, attr):
            try:
                delattr(raw_env, attr)
            except Exception:
                pass

    capture = getattr(raw_env, "_openclaw_rgb_capture", None)
    if isinstance(capture, dict):
        capture_render_product = capture.get("render_product")
        capture_annotators = list(capture.get("annotators", []))
        legacy_annotator = capture.get("annotator")
        if legacy_annotator is not None:
            capture_annotators.append(("legacy", legacy_annotator))
        if capture_render_product is not None:
            for _, capture_annotator in capture_annotators:
                if capture_annotator is None:
                    continue
                try:
                    capture_annotator.detach([capture_render_product])
                except Exception:
                    try:
                        capture_annotator.detach(capture_render_product)
                    except Exception:
                        pass
        capture_render_product = capture.get("render_product")
        destroy = getattr(capture_render_product, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass
    if hasattr(raw_env, "_openclaw_rgb_capture"):
        try:
            delattr(raw_env, "_openclaw_rgb_capture")
        except Exception:
            pass


def _capture_rgb_via_replicator(raw_env, *, camera_path: str | None = None) -> np.ndarray | None:
    """Capture RGB from the configured camera with Replicator annotators.

    Isaac Lab's ``env.render()`` uses the ``rgb`` annotator.  On this Isaac Sim setup that path sometimes returns
    empty buffers during headless warmup, while ``LdrColor`` can occasionally return an all-white frame before its
    render product is ready.  Try both annotators and only accept frames that contain real image variation.
    """

    try:
        import omni.replicator.core as rep
    except Exception:
        return None

    try:
        resolution = tuple(int(v) for v in raw_env.cfg.viewer.resolution)
    except Exception:
        resolution = (960, 720)
    if camera_path is None:
        camera_path = str(getattr(raw_env.cfg.viewer, "cam_prim_path", "/OmniverseKit_Persp"))

    capture = getattr(raw_env, "_openclaw_rgb_capture", None)
    if not isinstance(capture, dict) or capture.get("camera_path") != camera_path or capture.get("resolution") != resolution:
        if isinstance(capture, dict):
            capture_render_product = capture.get("render_product")
            capture_annotators = list(capture.get("annotators", []))
            legacy_annotator = capture.get("annotator")
            if legacy_annotator is not None:
                capture_annotators.append(("legacy", legacy_annotator))
            if capture_render_product is not None:
                for _, capture_annotator in capture_annotators:
                    if capture_annotator is None:
                        continue
                    try:
                        capture_annotator.detach([capture_render_product])
                    except Exception:
                        try:
                            capture_annotator.detach(capture_render_product)
                        except Exception:
                            pass
        try:
            try:
                render_product = rep.create.render_product(camera_path, resolution, force_new=True)
            except TypeError:
                render_product = rep.create.render_product(camera_path, resolution)
            annotators = []
            for annotator_name in ("rgb", "LdrColor"):
                try:
                    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name, device="cpu")
                except TypeError:
                    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
                except Exception:
                    continue
                try:
                    annotator.attach([render_product])
                except Exception:
                    try:
                        annotator.attach(render_product)
                    except Exception:
                        continue
                annotators.append((annotator_name, annotator))
            capture = {
                "camera_path": camera_path,
                "resolution": resolution,
                "render_product": render_product,
                "annotators": annotators,
            }
            setattr(raw_env, "_openclaw_rgb_capture", capture)
        except Exception:
            return None

    last_frame = None
    for _ in range(2):
        _render_rgb_warmup(raw_env, frames=1)
        for _, annotator in capture.get("annotators", []):
            try:
                frame = _as_rgb_uint8_frame(annotator.get_data())
            except Exception:
                continue
            if frame is None:
                continue
            last_frame = frame
            if not _rgb_frame_looks_empty(frame):
                return frame
    return last_frame


def _capture_rgb_via_ldrcolor(raw_env, *, camera_path: str | None = None) -> np.ndarray | None:
    """Backward-compatible wrapper for older call sites."""

    return _capture_rgb_via_replicator(raw_env, camera_path=camera_path)


def _capture_rgb_via_viewport_file(raw_env) -> np.ndarray | None:
    """Capture the active viewport to a temporary PNG without advancing physics.

    In headless Isaac Sim 5.1, ad-hoc Replicator render products can stay blank with ``LdrColorSD`` warnings.
    The Kit viewport capture path is slower, but it uses the already-active follow camera and has proven less brittle.
    Keep ``/app/player/playSimulations`` disabled while waiting for the capture so the recorder cannot insert extra
    physics updates between policy steps.
    """

    try:
        import asyncio
        import tempfile

        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport, next_viewport_frame_async
    except Exception:
        return None

    try:
        viewport = get_active_viewport()
    except Exception:
        viewport = None
    if viewport is None:
        return None

    capture_dir = getattr(raw_env, "_openclaw_viewport_capture_dir", None)
    if capture_dir is None:
        capture_dir = Path(tempfile.mkdtemp(prefix="periodic_eval_viewport_"))
        setattr(raw_env, "_openclaw_viewport_capture_dir", capture_dir)
    counter = int(getattr(raw_env, "_openclaw_viewport_capture_counter", 0))
    setattr(raw_env, "_openclaw_viewport_capture_counter", counter + 1)
    capture_path = Path(capture_dir) / f"frame_{counter:06d}.png"

    async def _capture_once():
        try:
            await next_viewport_frame_async(viewport)
        except Exception:
            pass
        capture = capture_viewport_to_file(viewport, file_path=str(capture_path))
        if capture is None:
            return None
        return await capture.wait_for_result(completion_frames=2)

    _render_rgb_warmup(raw_env, frames=1)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return None

    try:
        raw_env.sim.set_setting("/app/player/playSimulations", False)
        result = loop.run_until_complete(_capture_once())
        try:
            import omni.kit.renderer_capture

            omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        except Exception:
            pass
    except Exception:
        result = None
    finally:
        try:
            raw_env.sim.set_setting("/app/player/playSimulations", True)
        except Exception:
            pass

    if not result or not capture_path.exists():
        return None
    try:
        from PIL import Image

        frame = np.asarray(Image.open(capture_path).convert("RGB"), dtype=np.uint8)
        return _as_rgb_uint8_frame(frame)
    except Exception:
        return None
    finally:
        try:
            capture_path.unlink(missing_ok=True)
        except Exception:
            pass


def _render_periodic_eval_state_frame(
    raw_env,
    *,
    timestep: int,
    episodes_done: int,
    target_episodes: int,
    width: int = 960,
    height: int = 720,
) -> np.ndarray | None:
    """Render a deterministic top-down race-state frame when Isaac RGB capture is unavailable.

    This fallback is intentionally pure Python/PIL and reads only simulator state. It keeps training-time periodic eval
    videos useful in headless runs without calling Kit/Replicator between policy steps.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    try:
        robot = raw_env._robot
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        root_quat = robot.data.root_quat_w[0].detach().cpu()
        yaw = SideFollowCamera._quat_wxyz_to_yaw(root_quat)
    except Exception:
        return None

    def _tensor_xy(name: str):
        value = getattr(raw_env, name, None)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return None if value is None else np.asarray(value, dtype=np.float64)

    waypoints = _tensor_xy("_track_waypoints_w")
    patch_min = _tensor_xy("_patch_xy_min")
    patch_max = _tensor_xy("_patch_xy_max")
    if waypoints is None or waypoints.size == 0:
        waypoints = np.asarray([[0.0, 0.0], [max(4.0, root_pos[0] + 2.0), 0.0]], dtype=np.float64)
    waypoints = np.asarray(waypoints[:, :2], dtype=np.float64)

    xy_sets = [waypoints, root_pos[None, :2]]
    if patch_min is not None and patch_max is not None and patch_min.size and patch_max.size:
        xy_sets.extend([patch_min[:, :2], patch_max[:, :2]])
    all_xy = np.concatenate(xy_sets, axis=0)
    min_xy = np.nanmin(all_xy, axis=0) - np.array([1.2, 1.2])
    max_xy = np.nanmax(all_xy, axis=0) + np.array([1.2, 1.2])
    span = np.maximum(max_xy - min_xy, 1.0)
    scale = min((width - 120) / float(span[0]), (height - 120) / float(span[1]))
    center = 0.5 * (min_xy + max_xy)

    def world_to_px(xy):
        xy = np.asarray(xy, dtype=np.float64)
        px = width * 0.5 + (xy[..., 0] - center[0]) * scale
        py = height * 0.5 - (xy[..., 1] - center[1]) * scale
        return np.stack([px, py], axis=-1)

    image = Image.new("RGB", (width, height), (20, 24, 30))
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Grid.
    grid_step = 1.0
    x0 = math.floor(min_xy[0] / grid_step) * grid_step
    x1 = math.ceil(max_xy[0] / grid_step) * grid_step
    y0 = math.floor(min_xy[1] / grid_step) * grid_step
    y1 = math.ceil(max_xy[1] / grid_step) * grid_step
    x = x0
    while x <= x1 + 1.0e-9:
        p0 = world_to_px([x, y0])
        p1 = world_to_px([x, y1])
        draw.line([tuple(p0), tuple(p1)], fill=(255, 255, 255, 18), width=1)
        x += grid_step
    y = y0
    while y <= y1 + 1.0e-9:
        p0 = world_to_px([x0, y])
        p1 = world_to_px([x1, y])
        draw.line([tuple(p0), tuple(p1)], fill=(255, 255, 255, 18), width=1)
        y += grid_step

    # Friction patches.
    patch_buckets = getattr(raw_env, "_patch_bucket_ids", None)
    if isinstance(patch_buckets, torch.Tensor) and patch_buckets.numel() > 0:
        patch_buckets = patch_buckets[0].detach().cpu().numpy()
    else:
        patch_buckets = None
    if patch_min is not None and patch_max is not None:
        for patch_idx, (mn, mx) in enumerate(zip(patch_min[:, :2], patch_max[:, :2])):
            p0 = world_to_px(mn)
            p1 = world_to_px(mx)
            bucket = 0 if patch_buckets is None or patch_idx >= len(patch_buckets) else int(patch_buckets[patch_idx])
            color_t = max(0.0, min(1.0, bucket / max(1, int(getattr(raw_env.cfg, "friction_num_buckets", 1000)) - 1)))
            color = (int(40 + 190 * color_t), int(120 - 60 * color_t), int(230 - 180 * color_t), 88)
            draw.rectangle([tuple(np.minimum(p0, p1)), tuple(np.maximum(p0, p1))], fill=color, outline=(255, 255, 255, 55))

    # Track and gates.
    wp_px = world_to_px(waypoints)
    if len(wp_px) >= 2:
        draw.line([tuple(p) for p in wp_px], fill=(95, 210, 255, 210), width=4)
    try:
        gate_idx = int(raw_env._current_gate_idx[0].detach().cpu().item())
    except Exception:
        gate_idx = 0
    for idx, p in enumerate(wp_px):
        radius = 7 if idx != gate_idx + 1 else 11
        fill = (255, 210, 80, 255) if idx == gate_idx + 1 else (220, 235, 245, 220)
        draw.ellipse((p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius), fill=fill)
        draw.text((p[0] + 8, p[1] - 8), str(idx), fill=(235, 240, 245, 210), font=small_font)

    # Robot body as oriented triangle plus body dot.
    robot_px = world_to_px(root_pos[:2])
    heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    side = np.array([-heading[1], heading[0]], dtype=np.float64)
    tri_w = np.stack(
        [
            root_pos[:2] + 0.34 * heading,
            root_pos[:2] - 0.24 * heading + 0.18 * side,
            root_pos[:2] - 0.24 * heading - 0.18 * side,
        ],
        axis=0,
    )
    tri_px = world_to_px(tri_w)
    draw.polygon([tuple(p) for p in tri_px], fill=(255, 115, 90, 245), outline=(255, 255, 255, 240))
    draw.ellipse((robot_px[0] - 4, robot_px[1] - 4, robot_px[0] + 4, robot_px[1] + 4), fill=(255, 255, 255, 255))

    try:
        speed_xy = float(torch.linalg.norm(robot.data.root_lin_vel_w[0, :2]).detach().cpu().item())
    except Exception:
        speed_xy = float("nan")
    lines = [
        "Inference eval — state fallback",
        f"episode {min(episodes_done + 1, target_episodes)}/{target_episodes} | step {int(timestep)}",
        f"gate_idx={gate_idx} | speed_xy={speed_xy:.2f} m/s | yaw={math.degrees(yaw):.0f}°",
    ]
    lines.extend(_foot_friction_overlay_lines(raw_env, env_index=0))
    box_h = 14 + len(lines) * 22
    draw.rectangle((14, 14, 520, 14 + box_h), fill=(0, 0, 0, 165), outline=(255, 255, 255, 80))
    y_text = 25
    for line in lines:
        draw.text((26, y_text), line, fill=(245, 248, 250, 255), font=font)
        y_text += 22
    return np.asarray(image, dtype=np.uint8)


def _capture_inference_camera_frame(raw_env, *, env_index: int = 0) -> np.ndarray | None:
    """Read an RGB frame from Solo12 race IsaacLab camera sensors when they are enabled.

    This is a useful fallback for headless encoding visualizations because IsaacLab's ``TiledCamera`` path owns the
    RTX sensor lifecycle, while ad-hoc render products can return blank buffers on some Isaac Sim launches.
    Prefer the side camera because it follows the robot; fall back to overhead if that is the only available view.
    """

    for camera_attr in ("_side_camera", "_overhead_camera"):
        camera = getattr(raw_env, camera_attr, None)
        if camera is None:
            continue
        try:
            camera.update(float(getattr(raw_env, "step_dt", 0.02)))
        except Exception:
            pass
        try:
            frame = camera.data.output.get("rgb")
        except Exception:
            continue
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        try:
            frame_array = np.asarray(frame)
        except Exception:
            continue
        if frame_array.ndim == 4:
            frame_array = frame_array[min(max(0, int(env_index)), frame_array.shape[0] - 1)]
        frame_array = _as_rgb_uint8_frame(frame_array)
        if frame_array is None:
            continue
        if not _rgb_frame_looks_empty(frame_array):
            return frame_array
    return None


def _foot_body_names_for_overlay(raw_env) -> list[str]:
    robot = getattr(raw_env, "_robot", None)
    find_bodies = getattr(robot, "find_bodies", None)
    if callable(find_bodies):
        try:
            _, body_names = find_bodies(".*_calf")
            if body_names:
                return [str(body_name) for body_name in body_names]
        except Exception:
            pass

    for attr, source in (
        ("_feet_robot_body_ids", robot),
        ("_feet_body_ids", getattr(raw_env, "_contact_sensor", None)),
    ):
        ids = getattr(raw_env, attr, [])
        if isinstance(ids, torch.Tensor):
            ids = [int(value) for value in ids.detach().cpu().reshape(-1).tolist()]
        else:
            try:
                ids = [int(value) for value in ids]
            except TypeError:
                ids = [int(ids)] if ids else []
        body_names = list(getattr(source, "body_names", []) or [])
        if ids and body_names:
            try:
                return [str(body_names[body_id]) for body_id in ids]
            except Exception:
                pass

    return []


def _foot_friction_overlay_lines(raw_env, env_index: int) -> list[str]:
    if not hasattr(raw_env, "_get_gt_patch_mu_obs"):
        return ["Foot μ: unavailable"]
    try:
        with torch.inference_mode():
            values = raw_env._get_gt_patch_mu_obs()[int(env_index)].detach().cpu().tolist()
    except Exception as exc:
        return [f"Foot μ: error {type(exc).__name__}"]

    foot_names = _foot_body_names_for_overlay(raw_env)
    mu_by_label: dict[str, float] = {}
    for index, value in enumerate(values):
        name = foot_names[index] if index < len(foot_names) else ""
        label = FootFrictionPopup._label_from_foot_name(name)
        if label is not None:
            mu_by_label[label] = float(value)

    return [f"{label} μ={mu_by_label[label]:.3f}" if label in mu_by_label else f"{label} μ=--" for label in ("FR", "FL", "RR", "RL")]


def _draw_video_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        font = ImageFont.load_default()
        padding = 8
        line_gap = 4
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_width = max(box[2] - box[0] for box in boxes)
        text_height = sum(box[3] - box[1] for box in boxes) + line_gap * max(0, len(lines) - 1)
        x0, y0 = 12, 12
        x1 = x0 + text_width + 2 * padding
        y1 = y0 + text_height + 2 * padding
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 165), outline=(255, 255, 255, 110))
        y = y0 + padding
        for line, box in zip(lines, boxes):
            draw.text((x0 + padding, y), line, fill=(255, 255, 255, 255), font=font)
            y += (box[3] - box[1]) + line_gap
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return frame


class PeriodicEvalVideoWriter:
    """Write a half-speed follow-camera MP4 and optionally upload it to a W&B run."""

    def __init__(
        self,
        *,
        output_path: str | os.PathLike,
        dt: float,
        speed: float,
        target_episodes: int,
        wandb_upload: bool,
        wandb_project: str,
        wandb_entity: str | None,
        wandb_run_id: str | None,
        wandb_step: int | None,
        wandb_key: str,
        simple_video: bool = False,
    ) -> None:
        import imageio.v2 as imageio

        self.path = Path(output_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(round((1.0 / max(float(dt), 1.0e-9)) * max(float(speed), 1.0e-6))))
        self.target_episodes = max(1, int(target_episodes))
        self.wandb_upload = bool(wandb_upload)
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_id = wandb_run_id
        self.wandb_step = wandb_step
        self.wandb_key = wandb_key
        self.frame_count = 0
        self.rgb_failure_count = 0
        self.use_state_fallback = bool(simple_video)
        self.writer = imageio.get_writer(str(self.path), fps=self.fps, codec="libx264", quality=8, macro_block_size=16)
        print(
            f"[INFO] Periodic eval MP4 recording: path={self.path}, fps={self.fps}, "
            f"target_episodes={self.target_episodes}, mode={'simple' if self.use_state_fallback else 'rgb'}",
            flush=True,
        )

    def append(self, render_env, raw_env, *, timestep: int, episodes_done: int) -> None:
        frame = None
        if not self.use_state_fallback:
            frame = _capture_inference_camera_frame(raw_env, env_index=0)
            if _rgb_frame_looks_empty(frame):
                frame = _capture_rgb_via_viewport_file(raw_env)
            if _rgb_frame_looks_empty(frame):
                for _ in range(2):
                    _render_rgb_warmup(raw_env, frames=1)
                    candidate = _capture_rgb_via_ldrcolor(raw_env)
                    if candidate is not None and not _rgb_frame_looks_empty(candidate):
                        frame = candidate
                        break
                    for render_target in (render_env, raw_env):
                        render = getattr(render_target, "render", None)
                        if not callable(render):
                            continue
                        try:
                            candidate = _as_rgb_uint8_frame(render())
                        except Exception:
                            candidate = None
                        if candidate is not None and not _rgb_frame_looks_empty(candidate):
                            frame = candidate
                            break
                    if frame is not None and not _rgb_frame_looks_empty(frame):
                        break
            if _rgb_frame_looks_empty(frame):
                self.rgb_failure_count += 1
                if self.rgb_failure_count >= 5:
                    self.use_state_fallback = True
                    print(
                        "[WARN] Headless RGB capture stayed empty; switching periodic eval video to state-space fallback.",
                        flush=True,
                    )
            else:
                self.rgb_failure_count = 0

        if self.use_state_fallback or _rgb_frame_looks_empty(frame):
            frame = _render_periodic_eval_state_frame(
                raw_env,
                timestep=timestep,
                episodes_done=episodes_done,
                target_episodes=self.target_episodes,
            )
        if frame is None or _rgb_frame_looks_empty(frame):
            if self.frame_count == 0:
                print("[WARN] Could not generate periodic eval video frames yet.", flush=True)
            return

        lines = [
            "Inference eval",
            f"episode {min(episodes_done + 1, self.target_episodes)}/{self.target_episodes} | step {int(timestep)}",
        ]
        lines.extend(_foot_friction_overlay_lines(raw_env, env_index=0))
        frame = _as_rgb_uint8_frame(_draw_video_overlay(frame, lines))
        if frame is not None:
            self.writer.append_data(frame)
            self.frame_count += 1

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            print(f"[INFO] Periodic eval MP4 wrote {self.frame_count} frames to: {self.path}", flush=True)
        if self.frame_count > 0 and self.wandb_upload:
            self._upload_to_wandb()

    def _upload_to_wandb(self) -> None:
        # Upload from a clean lightweight Python subprocess instead of the active Isaac/Kit process.
        # On cluster jobs, wandb.init() inside the rendering process can timeout even though a plain
        # Python uploader succeeds immediately.
        upload_script = r'''
import os
import sys
import wandb

path, project, entity, run_id, key, step_text, fps_text = sys.argv[1:]
settings = wandb.Settings(init_timeout=300, sync_tensorboard=False, disable_git=True)
init_kwargs = {"project": project, "resume": "allow", "settings": settings}
if entity:
    init_kwargs["entity"] = entity
if run_id:
    init_kwargs["id"] = run_id
run = wandb.init(**init_kwargs)
payload = {
    key: wandb.Video(path, fps=int(float(fps_text)), format="mp4"),
    "EvalVideo/frames": int(os.environ.get("BORINOT_VIDEO_FRAMES", "0")),
    "EvalVideo/playback_fps": int(float(fps_text)),
}
if step_text:
    payload["EvalVideo/trigger_iteration"] = int(step_text)
wandb.log(payload)
wandb.finish()
print(f"uploaded periodic eval video to run={run.id} key={key}", flush=True)
'''
        cmd = [
            sys.executable,
            "-c",
            upload_script,
            str(self.path),
            str(self.wandb_project),
            str(self.wandb_entity or ""),
            str(self.wandb_run_id or ""),
            str(self.wandb_key),
            "" if self.wandb_step is None else str(int(self.wandb_step)),
            str(int(self.fps)),
        ]
        env = os.environ.copy()
        for key in (
            "WANDB_SERVICE",
            "WANDB_RUN_ID",
            "WANDB_RESUME",
            "WANDB_NAME",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_DIR",
            "WANDB_CONFIG_PATH",
            "WANDB_SWEEP_ID",
        ):
            env.pop(key, None)
        env["BORINOT_VIDEO_FRAMES"] = str(int(self.frame_count))
        env["WANDB_DISABLE_GIT"] = "true"
        try:
            result = subprocess.run(
                cmd,
                cwd=str(_THIS_DIR.parents[2]),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
            if result.stdout:
                print(result.stdout.rstrip(), flush=True)
            if result.returncode == 0:
                print(
                    f"[INFO] Uploaded periodic eval video to W&B run={self.wandb_run_id} key={self.wandb_key}",
                    flush=True,
                )
                return
            print(f"[WARN] W&B video uploader exited with code {result.returncode}.", flush=True)
        except Exception as exc:
            print(f"[WARN] W&B video uploader subprocess failed: {type(exc).__name__}: {exc}", flush=True)

        # Fallback for local/non-cluster runs where direct wandb.init is fine.
        try:
            import wandb

            init_kwargs = {
                "project": self.wandb_project,
                "resume": "allow",
                "settings": wandb.Settings(init_timeout=300, sync_tensorboard=False, disable_git=True),
            }
            if self.wandb_entity:
                init_kwargs["entity"] = self.wandb_entity
            if self.wandb_run_id:
                init_kwargs["id"] = self.wandb_run_id
            run = wandb.init(**init_kwargs)
            payload = {
                self.wandb_key: wandb.Video(str(self.path), fps=self.fps, format="mp4"),
                "EvalVideo/frames": self.frame_count,
                "EvalVideo/playback_fps": self.fps,
            }
            if self.wandb_step is not None:
                # Do not use this as W&B's explicit history step: the async video process may finish after
                # training has already logged newer steps, and W&B can drop out-of-order media rows.
                payload["EvalVideo/trigger_iteration"] = int(self.wandb_step)
            wandb.log(payload)
            wandb.finish()
            print(
                f"[INFO] Uploaded periodic eval video to W&B run={run.id if run is not None else self.wandb_run_id} "
                f"key={self.wandb_key}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] Could not upload periodic eval video to W&B: {type(exc).__name__}: {exc}", flush=True)


def _disable_training_stochasticity_for_play(
    env_cfg,
    *,
    preserve_patch_friction_randomization: bool = False,
    preserve_within_episode_friction_resampling: bool = False,
) -> list[str]:
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

    if (
        hasattr(env_cfg, "within_episode_fric_resample")
        and getattr(env_cfg, "within_episode_fric_resample")
        and not preserve_within_episode_friction_resampling
    ):
        env_cfg.within_episode_fric_resample = False
        disabled.append("within_episode_fric_resample")

    return disabled


def _manual_reset_all_envs(raw_env, vec_env):
    with torch.inference_mode():
        env_ids = torch.arange(raw_env.num_envs, dtype=torch.int64, device=raw_env.device)
        raw_env._reset_idx(env_ids)
        raw_env.scene.write_data_to_sim()
        raw_env.sim.forward()

        if raw_env.sim.has_rtx_sensors() and raw_env.cfg.num_rerenders_on_reset > 0:
            for _ in range(raw_env.cfg.num_rerenders_on_reset):
                raw_env.sim.render()
        elif raw_env.sim.has_gui():
            raw_env.sim.render()

        return vec_env.get_observations()


def _compute_summary(raw_env) -> dict[str, float]:
    summary = {}
    if hasattr(raw_env, "_current_gate_idx"):
        summary["gate_idx"] = float(raw_env._current_gate_idx.float().mean().item())
    if hasattr(raw_env, "_robot"):
        summary["root_speed_xy"] = float(torch.norm(raw_env._robot.data.root_lin_vel_b[:, :2], dim=1).mean().item())
        summary["root_ang_vel_z"] = float(raw_env._robot.data.root_ang_vel_b[:, 2].mean().item())
    return summary


def _race_env_finished(raw_env, env_index: int = 0) -> bool:
    try:
        target_count = int(getattr(raw_env, "_target_count", 0))
        gate_idx = int(raw_env._current_gate_idx[env_index].item())
        return target_count > 0 and gate_idx >= target_count
    except Exception:
        return False


def _completed_race_episode(raw_env, dones, env_index: int = 0) -> bool:
    if _race_env_finished(raw_env, env_index=env_index):
        return True

    try:
        done_for_env = bool(dones.reshape(-1)[env_index].item()) if isinstance(dones, torch.Tensor) else bool(dones)
    except Exception:
        done_for_env = False
    if not done_for_env:
        return False

    try:
        log = raw_env.extras.get("log", {})
        success_rate = log.get("Episode/successRate")
        if success_rate is not None:
            return float(success_rate) >= 1.0
        finish_count = log.get("Episode_Termination/finish")
        if finish_count is not None:
            return int(finish_count) > 0
    except Exception:
        pass
    return False


def _resolve_slip_csv_path(args_cli, log_dir: str, resume_path: str) -> str | None:
    """Resolve the --visualize-slip CSV path shared by interactive and headless slip logging.

    Returns ``None`` when slip logging is explicitly disabled, except under --generate-slip-plots
    where a CSV is mandatory and the default auto path is used instead.
    """
    slip_log_arg = args_cli.visualize_slip_log
    disabled = slip_log_arg is None or str(slip_log_arg).strip().lower() in {"", "none", "null", "off"}
    if disabled:
        if not bool(getattr(args_cli, "generate_slip_plots", False)):
            return None
        slip_log_arg = "auto"
    if str(slip_log_arg).strip().lower() == "auto":
        checkpoint_stem = Path(resume_path).stem
        return os.path.join(log_dir, "slip_logs", f"{checkpoint_stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return str(slip_log_arg)


def _format_slip_plot_float(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _format_slip_plot_range(value) -> str:
    if isinstance(value, (str, bytes)):
        return str(value)
    try:
        values = list(value)
    except TypeError:
        return str(value)
    if len(values) >= 2:
        return f"[{_format_slip_plot_float(values[0])}, {_format_slip_plot_float(values[1])}]"
    return "[" + ", ".join(_format_slip_plot_float(item) for item in values) + "]"


def _slip_plot_friction_title(env_cfg) -> str | None:
    pieces = []
    friction_static_range = getattr(env_cfg, "friction_static_range", None)
    if friction_static_range is not None:
        pieces.append(r"$\mu_s \in " + _format_slip_plot_range(friction_static_range) + "$")
    mu_dynamic_static_ratio = getattr(env_cfg, "mu_dynamic_static_ratio", None)
    if mu_dynamic_static_ratio is not None:
        pieces.append(r"$\mu_d = " + _format_slip_plot_float(mu_dynamic_static_ratio) + r"\,\mu_s$")
    physics_dt = getattr(env_cfg.sim, "dt", None) if getattr(env_cfg, "sim", None) is not None else None
    if physics_dt is not None:
        pieces.append("physics_dt=" + _format_slip_plot_float(physics_dt))
    decimation = getattr(env_cfg, "decimation", None)
    if decimation is not None:
        pieces.append("decimation=" + _format_slip_plot_float(decimation))
    return ", ".join(pieces) if pieces else None


def _generate_slip_plots_after_run(args_cli, slip_csv_path: str | None, env_cfg=None) -> None:
    """Save per-foot slip plots from the recorded CSV and optionally open them interactively.

    The PNG is rendered with the Agg backend in-process (safe inside Kit). The interactive
    matplotlib window is launched as a detached subprocess so it survives Isaac Sim shutdown
    and never blocks teardown.
    """
    if not bool(getattr(args_cli, "generate_slip_plots", False)):
        return
    if not slip_csv_path or not os.path.isfile(slip_csv_path):
        print(f"[WARN] --generate-slip-plots: no slip CSV found at {slip_csv_path!r}; skipping plot generation.", flush=True)
        return

    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))

    png_path = args_cli.slip_plots_output or (os.path.splitext(slip_csv_path)[0] + ".png")
    min_samples = max(1, int(getattr(args_cli, "slip_plots_min_samples", 1)))
    title_extra = _slip_plot_friction_title(env_cfg) if env_cfg is not None else None
    try:
        import slip_plots  # noqa: PLC0415

        slip_plots.save_slip_figure(slip_csv_path, png_path, min_samples=min_samples, title_extra=title_extra)
        polar_paths = slip_plots.save_polar_slip_figures(
            slip_csv_path,
            os.path.splitext(png_path)[0],
            title_extra=title_extra,
        )
        print(f"[INFO] --generate-slip-plots: saved slip data to {slip_csv_path}", flush=True)
        print(f"[INFO] --generate-slip-plots: saved slip plot to {png_path}", flush=True)
        print(
            "[INFO] --generate-slip-plots: saved polar reaction-force plots:\n    " + "\n    ".join(polar_paths),
            flush=True,
        )
    except Exception as exc:
        print(f"[WARN] --generate-slip-plots: could not save slip plot: {type(exc).__name__}: {exc}", flush=True)
        return

    slip_plots_script = os.path.join(str(_THIS_DIR), "slip_plots.py")
    repo_root = Path(__file__).resolve().parents[3]
    interactive_args = [slip_csv_path, "--min-samples", str(min_samples)]
    if title_extra:
        interactive_args.extend(("--title-extra", title_extra))
    interactive_cmd = " ".join(
        shlex.quote(str(part))
        for part in [repo_root / "isaaclab.sh", "-p", os.path.relpath(slip_plots_script, repo_root), *interactive_args]
    )
    if bool(args_cli.slip_plots_no_window):
        print(f"[INFO] --generate-slip-plots: open interactively (zoom/pan) with:\n    {interactive_cmd}", flush=True)
        return
    if not os.environ.get("DISPLAY"):
        print(
            "[INFO] --generate-slip-plots: no DISPLAY detected; skipping the interactive window. "
            f"Open it later with:\n    {interactive_cmd}",
            flush=True,
        )
        return

    try:
        subprocess.Popen(
            [sys.executable, slip_plots_script, *interactive_args],
            start_new_session=True,
        )
        print(
            "[INFO] --generate-slip-plots: opening the interactive slip plot in a separate window "
            f"(zoom/pan). Reopen later with:\n    {interactive_cmd}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[WARN] --generate-slip-plots: could not open the interactive window ({type(exc).__name__}: {exc}). "
            f"Open it manually with:\n    {interactive_cmd}",
            flush=True,
        )


def _print_patch_friction_summary(raw_env, env_index: int = 0):
    if not hasattr(raw_env, "get_patch_friction_summary"):
        print("[INFO] Patch-friction summary is not available for this environment.", flush=True)
        return

    summary = raw_env.get_patch_friction_summary(env_index=env_index)
    if not summary:
        print("[INFO] No friction patches were found in the current race scene.", flush=True)
        return

    print("[INFO] Active patch friction coefficients (env 0):", flush=True)
    for item in summary:
        print(
            "  - {patch}: bucket={bucket:03d}, mu_static={static:.3f}, mu_dynamic={dynamic:.3f}, "
            "physics={physics_material}, visual={visual_material}".format(**item),
            flush=True,
        )


class EncodingLatentExtractor:
    """Extract the actor-side latent encoding used by race policies during play."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.module = self._find_latent_module(policy)
        self.source_name = self._source_name(self.module)

    @staticmethod
    def _find_latent_module(policy):
        candidates = [policy]
        bound_self = getattr(policy, "__self__", None)
        if bound_self is not None:
            candidates.append(bound_self)
        actor_critic = getattr(policy, "actor_critic", None)
        if actor_critic is not None:
            candidates.append(actor_critic)

        for candidate in candidates:
            if candidate is None:
                continue
            if getattr(candidate, "is_dagger_adapter_policy", False):
                return candidate
            if hasattr(candidate, "actor_env_params_encoder"):
                return candidate
            if hasattr(candidate, "actor_imu_encoder"):
                return candidate
        return None

    @staticmethod
    def _source_name(module) -> str:
        if module is None:
            return "unsupported latent source"
        if getattr(module, "is_dagger_adapter_policy", False):
            try:
                latent_dim = int(module.dims.get("latent_dim", 0))
            except Exception:
                latent_dim = None
            suffix = f" ({latent_dim}D)" if latent_dim else ""
            return "DAgger adapter z_hat" + suffix
        if hasattr(module, "actor_env_params_encoder"):
            latent_dim = int(getattr(module, "env_params_latent_dim", 0) or 0)
            suffix = f" ({latent_dim}D)" if latent_dim else ""
            return "env-param encoder z" + suffix
        if hasattr(module, "actor_imu_encoder"):
            latent_dim = int(getattr(module, "tcn_latent_dim", 0) or 0)
            suffix = f" ({latent_dim}D)" if latent_dim else ""
            return f"{getattr(module, 'history_name', 'history')} TCN latent" + suffix
        return "unsupported latent source"

    @staticmethod
    def _policy_obs(obs):
        try:
            return obs["policy"]
        except Exception:
            return obs

    def extract(self, obs, env_index: int = 0) -> np.ndarray | None:
        if self.module is None:
            return None
        with torch.inference_mode():
            try:
                if getattr(self.module, "is_dagger_adapter_policy", False):
                    z = self._extract_dagger_adapter_z(obs)
                elif hasattr(self.module, "actor_env_params_encoder"):
                    z = self._extract_env_params_z(obs)
                elif hasattr(self.module, "actor_imu_encoder"):
                    z = self._extract_tcn_z(obs)
                else:
                    return None
            except Exception as exc:
                print(f"[WARN] Could not extract latent encoding for visualization: {type(exc).__name__}: {exc}", flush=True)
                return None

        if z is None or z.numel() == 0:
            return None
        env_index = min(max(0, int(env_index)), int(z.shape[0]) - 1)
        return z[env_index].detach().float().cpu().reshape(-1).numpy()

    def _extract_dagger_adapter_z(self, obs) -> torch.Tensor:
        policy_obs = self._policy_obs(obs)
        history_start = int(self.module.dims["history_start"])
        history_flat_dim = int(self.module.dims["history_flat_dim"])
        history_raw = policy_obs[:, history_start : history_start + history_flat_dim]
        if self.module.history_normalizer is not None:
            adapter_input = self.module.history_normalizer(history_raw)
        else:
            adapter_input = history_raw
        return self.module.adapter(adapter_input)

    def _extract_env_params_z(self, obs) -> torch.Tensor:
        actor_obs = self.module.get_actor_obs(obs)
        actor_obs = self.module.actor_obs_normalizer(actor_obs)
        current_obs_dim = int(self.module.current_obs_dim)
        env_params_dim = int(self.module.env_params_dim)
        env_params = actor_obs[:, current_obs_dim : current_obs_dim + env_params_dim]
        return self.module.actor_env_params_encoder(env_params)

    def _extract_tcn_z(self, obs) -> torch.Tensor:
        actor_obs = self.module.get_actor_obs(obs)
        actor_obs = self.module.actor_obs_normalizer(actor_obs)
        history = actor_obs[:, int(self.module.current_obs_dim) :]
        return self.module.actor_imu_encoder(history)


class EncodingVizRecorder:
    """Collect latent samples, then render a stable fixed-projection MP4 at shutdown."""

    def __init__(
        self,
        *,
        policy,
        render_env,
        raw_env,
        log_dir: str,
        checkpoint_path: str,
        output_dir: str | None,
        interval_s: float,
        dt: float,
        max_points: int,
        method: str,
        perplexity: float,
        max_iter: int,
        env_index: int = 0,
        video_fps: float | None = None,
    ) -> None:
        self.extractor = EncodingLatentExtractor(policy)
        self.render_env = render_env
        self.raw_env = raw_env
        self.env_index = min(max(0, int(env_index)), int(raw_env.num_envs) - 1)
        self.interval_s = max(float(dt), float(interval_s))
        self.snapshot_step_interval = max(1, int(round(self.interval_s / float(dt))))
        self.max_points = max(2, int(max_points))
        self.method = str(method)
        self.perplexity = max(1.0, float(perplexity))
        self.max_iter = max(250, int(max_iter))
        self.out_dir = self._resolve_output_dir(output_dir, log_dir=log_dir, checkpoint_path=checkpoint_path)
        self.video_path = self.out_dir / "encoding_viz.mp4"
        self.latest_png_path = self.out_dir / "latest.png"
        self.video_fps = max(1.0e-6, float(video_fps) if video_fps is not None else 1.0 / self.interval_s)
        self.csv_path = self.out_dir / "encoding_samples.csv"
        self._csv_file = None
        self._csv_header_written = False
        self._next_snapshot_step = 0
        self._snapshot_index = 0
        self._video_writer = None
        self._frames_written = 0
        self._first_image_reported = False
        self._disabled = False
        self._warned_no_latent = False
        self._warned_tsne_unavailable = False
        self._warned_tsne_failure = False
        self._warned_render_failure = False
        self._warned_video_failure = False
        self._projection_rendered = False
        self.z_history: list[np.ndarray] = []
        self.metadata_history: list[dict[str, float | int | None]] = []
        self.all_z_history: list[np.ndarray] = []
        self.all_metadata_history: list[dict[str, float | int | None]] = []

        self._Figure = None
        self._FigureCanvasAgg = None
        self._matplotlib_image = None
        self._setup_matplotlib()
        self._TSNE = self._load_tsne_class() if self.method == "tsne" else None

        print(
            "[INFO] Encoding visualization enabled: "
            f"source={self.extractor.source_name}, method={self.method}, interval={self.interval_s:.3f}s, "
            f"max_points={self.max_points}, out_dir={self.out_dir}, video={self.video_path}, "
            f"latest_png={self.latest_png_path}, csv={self.csv_path}",
            flush=True,
        )

    @staticmethod
    def _resolve_output_dir(output_dir: str | None, *, log_dir: str, checkpoint_path: str) -> Path:
        if output_dir:
            out_dir = Path(output_dir).expanduser()
        else:
            checkpoint_stem = Path(checkpoint_path).stem
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path(log_dir).expanduser() / "encoding_viz" / f"{checkpoint_stem}_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir.resolve()

    def _setup_matplotlib(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            import matplotlib.image as mpl_image

            self._Figure = Figure
            self._FigureCanvasAgg = FigureCanvasAgg
            self._matplotlib_image = mpl_image
        except Exception as exc:
            print(
                "[WARN] --tsne_encoding_viz could not import matplotlib; "
                f"MP4/preview rendering is disabled, but CSV latent logging will continue: {type(exc).__name__}: {exc}",
                flush=True,
            )

    @staticmethod
    def _load_tsne_class():
        try:
            from sklearn.manifold import TSNE

            return TSNE
        except Exception:
            return None

    def maybe_record(self, *, timestep: int, sim_time_s: float, obs, force: bool = False) -> None:
        if self._disabled:
            return
        timestep = int(timestep)
        if not force and timestep < self._next_snapshot_step:
            return
        while self._next_snapshot_step <= timestep:
            self._next_snapshot_step += self.snapshot_step_interval
        self.record(timestep=timestep, sim_time_s=float(sim_time_s), obs=obs)

    def record(self, *, timestep: int, sim_time_s: float, obs) -> None:
        z = self.extractor.extract(obs, env_index=self.env_index)
        if z is None:
            if not self._warned_no_latent:
                print(
                    "[WARN] --tsne_encoding_viz was requested, but this policy does not expose a supported "
                    "actor-side latent encoder. Supported sources: EnvParamsConditionedEncoderActor, "
                    "DAggerLatentPolicy, and ActorCriticFootImuTcn.",
                    flush=True,
                )
                self._warned_no_latent = True
            self._disabled = True
            return

        metadata = self._read_metadata(timestep=timestep, sim_time_s=sim_time_s)
        self._append_sample(z, metadata)
        self._write_csv_row(z, metadata)
        self._snapshot_index += 1
        if not self._first_image_reported:
            print(
                f"[INFO] First encoding visualization sample recorded; final stable MP4 will be rendered at shutdown: "
                f"video={self.video_path}, latest_png={self.latest_png_path}",
                flush=True,
            )
            self._first_image_reported = True

    def _append_sample(self, z: np.ndarray, metadata: dict[str, float | int | None]) -> None:
        z_array = np.asarray(z, dtype=np.float32)
        self.z_history.append(z_array)
        self.metadata_history.append(metadata)
        if len(self.z_history) > self.max_points:
            self.z_history = self.z_history[-self.max_points :]
            self.metadata_history = self.metadata_history[-self.max_points :]
        self.all_z_history.append(z_array)
        self.all_metadata_history.append(metadata)
        if len(self.all_z_history) > self.max_points:
            self.all_z_history = self.all_z_history[-self.max_points :]
            self.all_metadata_history = self.all_metadata_history[-self.max_points :]

    def reset_samples(self) -> None:
        """Start a fresh latent-projection window, keeping the MP4/CSV outputs continuous."""

        self.z_history.clear()
        self.metadata_history.clear()

    def _write_csv_row(self, z: np.ndarray, metadata: dict[str, float | int | None]) -> None:
        if self._csv_file is None:
            self._csv_file = self.csv_path.open("w", encoding="utf-8", buffering=1)
        if not self._csv_header_written:
            columns = ["snapshot", "step", "sim_time_s", "episode_time_s", "gate_idx", "progress"]
            columns.extend(f"z{i}" for i in range(len(z)))
            self._csv_file.write(",".join(columns) + "\n")
            self._csv_header_written = True

        values = [
            str(self._snapshot_index + 1),
            str(int(metadata["step"])),
            f"{float(metadata['sim_time_s']):.6f}",
            f"{float(metadata['episode_time_s']):.6f}",
            str(int(metadata["gate_idx"])) if metadata["gate_idx"] is not None else "",
            f"{float(metadata['progress']):.6f}" if metadata["progress"] is not None else "",
        ]
        values.extend(f"{float(value):.8g}" for value in z)
        self._csv_file.write(",".join(values) + "\n")

    def _read_metadata(self, *, timestep: int, sim_time_s: float) -> dict[str, float | int | None]:
        gate_idx = None
        progress = None
        try:
            gate_idx = int(self.raw_env._current_gate_idx[self.env_index].item())
            target_count = int(getattr(self.raw_env, "_target_count", 0))
            progress = None if target_count <= 0 else min(1.0, gate_idx / target_count)
        except Exception:
            pass

        episode_time_s = sim_time_s
        try:
            episode_time_s = float(self.raw_env.episode_length_buf[self.env_index].item()) * float(self.raw_env.step_dt)
        except Exception:
            pass

        return {
            "step": int(timestep),
            "sim_time_s": float(sim_time_s),
            "episode_time_s": float(episode_time_s),
            "gate_idx": gate_idx,
            "progress": progress,
        }

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, str]:
        points = np.asarray(points, dtype=np.float32)
        points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        if points.shape[0] < 2:
            return np.zeros((points.shape[0], 2), dtype=np.float32), "waiting for samples"

        if self.method == "tsne" and self._TSNE is not None and points.shape[0] >= 8:
            try:
                perplexity = min(self.perplexity, max(2.0, (points.shape[0] - 1) / 3.0))
                kwargs = {
                    "n_components": 2,
                    "perplexity": perplexity,
                    "learning_rate": 200.0,
                    "init": "pca",
                    "random_state": 0,
                }
                tsne_params = inspect.signature(self._TSNE).parameters
                if "max_iter" in tsne_params:
                    kwargs["max_iter"] = self.max_iter
                else:
                    kwargs["n_iter"] = self.max_iter
                return self._TSNE(**kwargs).fit_transform(points).astype(np.float32), "t-SNE"
            except Exception as exc:
                if not self._warned_tsne_failure:
                    print(
                        f"[WARN] t-SNE projection failed once; falling back to PCA for this run: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self._warned_tsne_failure = True
                self.method = "pca"

        if self.method == "tsne" and self._TSNE is None and not self._warned_tsne_unavailable:
            print("[WARN] scikit-learn is not available; using PCA for encoding visualization.", flush=True)
            self._warned_tsne_unavailable = True
        return self._pca_projection(points), "PCA"

    @staticmethod
    def _pca_projection(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            components = vh[:2].T
            proj = centered @ components
        except Exception:
            proj = np.zeros((points.shape[0], 2), dtype=np.float32)
        if proj.shape[1] < 2:
            proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])))
        return proj.astype(np.float32)

    def _render_plot(
        self,
        *,
        frame_index: int,
        projection: np.ndarray,
        projection_name: str,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        color_values: np.ndarray,
        color_vmax: float,
    ) -> np.ndarray:
        frame_index = min(max(0, int(frame_index)), projection.shape[0] - 1)
        visible_projection = projection[: frame_index + 1]
        visible_colors = color_values[: frame_index + 1]
        latest = self.all_metadata_history[frame_index]

        fig = self._Figure(figsize=(6.4, 5.0), dpi=150, facecolor="white")
        canvas = self._FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if visible_projection.shape[0] >= 2:
            ax.plot(
                visible_projection[:, 0],
                visible_projection[:, 1],
                color="0.35",
                alpha=0.25,
                linewidth=1.0,
                zorder=1,
            )
        scatter = ax.scatter(
            visible_projection[:, 0],
            visible_projection[:, 1],
            c=visible_colors,
            cmap="viridis",
            vmin=0.0,
            vmax=float(color_vmax),
            s=24,
            alpha=0.72,
            linewidths=0.0,
            zorder=2,
        )
        ax.scatter(
            visible_projection[-1:, 0],
            visible_projection[-1:, 1],
            c="red",
            s=135,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="current",
            zorder=3,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("episode time (s): start → finish")
        gate_text = "--" if latest["gate_idx"] is None else str(int(latest["gate_idx"]))
        progress_text = "--" if latest["progress"] is None else f"{100.0 * float(latest['progress']):.1f}%"
        ax.set_title(
            f"{self.extractor.source_name} → fixed 2D {projection_name}\n"
            f"sample={frame_index + 1}/{len(self.all_z_history)} | sim={float(latest['sim_time_s']):.2f}s | "
            f"episode={float(latest['episode_time_s']):.2f}s/{float(color_vmax):.2f}s | "
            f"gate={gate_text} | progress={progress_text}"
        )
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.tight_layout()
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        return np.asarray(rgba[..., :3], dtype=np.uint8)

    @staticmethod
    def _projection_limits(projection: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        limits = []
        for values in (projection[:, 0], projection[:, 1]):
            low = float(np.min(values))
            high = float(np.max(values))
            if not math.isfinite(low) or not math.isfinite(high) or abs(high - low) < 1e-6:
                limits.append((-1.0, 1.0))
                continue
            margin = 0.08 * (high - low)
            limits.append((low - margin, high + margin))
        return limits[0], limits[1]

    def _episode_time_color_values(self) -> tuple[np.ndarray, float]:
        values = np.asarray(
            [float(metadata.get("episode_time_s") or 0.0) for metadata in self.all_metadata_history],
            dtype=np.float32,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        observed_episode_s = float(np.max(values)) if values.size else 0.0
        color_vmax = max(observed_episode_s, 1.0e-6)
        return np.clip(values, 0.0, color_vmax), color_vmax

    def _render_final_video(self) -> None:
        if self._projection_rendered or self._Figure is None or self._FigureCanvasAgg is None or not self.all_z_history:
            return

        points = np.stack(self.all_z_history, axis=0)
        projection, projection_name = self._project(points)
        xlim, ylim = self._projection_limits(projection)
        color_values, color_vmax = self._episode_time_color_values()
        print(
            f"[INFO] Rendering stable encoding visualization MP4 from {len(self.all_z_history)} fixed samples "
            f"({projection_name}, fixed axes, color=episode time).",
            flush=True,
        )
        for frame_index in range(len(self.all_z_history)):
            frame = self._render_plot(
                frame_index=frame_index,
                projection=projection,
                projection_name=projection_name,
                xlim=xlim,
                ylim=ylim,
                color_values=color_values,
                color_vmax=color_vmax,
            )
            self._append_video_frame(frame)
            if frame_index == len(self.all_z_history) - 1:
                self._save_rgb(self.latest_png_path, frame)
        self._projection_rendered = True

    @staticmethod
    def _sanitize_rgb(frame) -> np.ndarray | None:
        if frame is None:
            return None
        if isinstance(frame, (tuple, list)):
            frame = next((item for item in frame if item is not None), None)
            if frame is None:
                return None
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        try:
            array = np.asarray(frame)
        except Exception:
            return None
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3 or array.size == 0:
            return None
        if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.shape[-1] != 3:
            return None
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if float(np.nanmax(array)) <= 1.0:
                array = 255.0 * array
            array = np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _save_rgb(self, path: Path, image: np.ndarray) -> None:
        try:
            from PIL import Image

            Image.fromarray(image).save(path)
        except Exception:
            if self._matplotlib_image is not None:
                self._matplotlib_image.imsave(path, image)

    def _open_video_writer(self):
        if self._video_writer is not None:
            return self._video_writer
        try:
            import imageio.v2 as imageio

            self._video_writer = imageio.get_writer(
                str(self.video_path), fps=float(self.video_fps), codec="libx264", quality=8, macro_block_size=16
            )
            print(f"[INFO] Encoding visualization MP4 recording: path={self.video_path}, fps={self.video_fps:.3f}", flush=True)
        except Exception as exc:
            if not self._warned_video_failure:
                print(
                    f"[WARN] Could not open MP4 writer for --tsne_encoding_viz: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._warned_video_failure = True
            self._video_writer = None
        return self._video_writer

    def _append_video_frame(self, image: np.ndarray) -> None:
        writer = self._open_video_writer()
        if writer is None:
            return
        frame = self._sanitize_rgb(image)
        if frame is None:
            return
        writer.append_data(self._pad_to_macroblock(frame))
        self._frames_written += 1

    @staticmethod
    def _pad_to_macroblock(image: np.ndarray, block: int = 16) -> np.ndarray:
        height, width = image.shape[:2]
        padded_height = int(math.ceil(height / block) * block)
        padded_width = int(math.ceil(width / block) * block)
        if padded_height == height and padded_width == width:
            return np.ascontiguousarray(image)
        padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        padded[:height, :width] = image
        return np.ascontiguousarray(padded)

    def close(self) -> None:
        self._render_final_video()
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        if self._snapshot_index > 0:
            print(
                f"[INFO] Encoding visualization wrote {self._frames_written} stable MP4 frames to {self.video_path}; "
                f"latest preview: {self.latest_png_path}; CSV: {self.csv_path}",
                flush=True,
            )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed

    play_friction_seed = args_cli.friction_seed if args_cli.friction_seed is not None else env_cfg.seed
    if play_friction_seed is not None:
        if hasattr(env_cfg, "friction_seed"):
            env_cfg.friction_seed = int(play_friction_seed)
            print(f"[INFO] Using race patch-friction seed: {env_cfg.friction_seed}", flush=True)
        elif args_cli.friction_seed is not None:
            print("[WARN] --friction_seed was provided, but this task config has no friction_seed field.", flush=True)

    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    explicit_within_episode_resample = (
        args_cli.within_episode_fric_resample is True
        or _hydra_bool_override("env.within_episode_fric_resample") is True
    )
    if not args_cli.keep_training_stochasticity:
        disabled = _disable_training_stochasticity_for_play(
            env_cfg,
            preserve_patch_friction_randomization=bool(getattr(env_cfg, "randomize_fric_coefs", False)),
            preserve_within_episode_friction_resampling=explicit_within_episode_resample,
        )
        if disabled:
            print("[INFO] Disabled training-time stochasticity for play: " + ", ".join(disabled), flush=True)

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
    if hasattr(env_cfg, "within_episode_fric_resample"):
        print(
            "[INFO] Effective patch-friction resampling: "
            f"within_episode_fric_resample={bool(getattr(env_cfg, 'within_episode_fric_resample'))}, "
            f"time_range={getattr(env_cfg, 'within_episode_fric_resample_time_range', None)}, "
            f"group_all_patches_single_bucket={getattr(env_cfg, 'group_all_patches_single_bucket', None)}, "
            f"randomize_fric_coefs={getattr(env_cfg, 'randomize_fric_coefs', None)}",
            flush=True,
        )
    resume_path = os.path.abspath(args_cli.checkpoint)
    dagger_adapter_checkpoint = load_dagger_adapter_checkpoint(resume_path)
    if dagger_adapter_checkpoint is not None:
        configure_env_cfg_for_dagger_adapter(env_cfg, dagger_adapter_checkpoint)
        layout = dagger_adapter_checkpoint.get("layout", {})
        print(
            "[INFO] Detected DAgger adapter checkpoint: "
            f"layout={layout.get('kind')} T={layout.get('history_len')} D={layout.get('history_dim')}",
            flush=True,
        )
    if bool(args_cli.visualize_slip) and hasattr(env_cfg, "contact_sensor"):
        env_cfg.contact_sensor.track_contact_points = True
        env_cfg.contact_sensor.track_friction_forces = True
        env_cfg.contact_sensor.max_contact_data_count_per_prim = max(
            int(getattr(env_cfg.contact_sensor, "max_contact_data_count_per_prim", 4)),
            8,
        )
        foot_reaction_update_period = float(
            getattr(
                env_cfg,
                "foot_reaction_sensor_update_period_s",
                getattr(env_cfg, "foot_reaction_contact_sensor_update_period_s", 0.0),
            )
        )
        if foot_reaction_update_period <= 0.0:
            foot_reaction_update_period = float(
                getattr(env_cfg, "foot_reaction_contact_sensor_update_period_s", 0.0)
            )
        if foot_reaction_update_period <= 0.0:
            foot_reaction_update_period = float(getattr(env_cfg, "physics_dt", getattr(env_cfg.sim, "dt", 0.0)))
        print(
            "[INFO] Slip visualization enabled contact-point and contact-reaction tracking "
            "(normal + friction), with reaction-angle/contact-speed coloring. "
            f"foot_reaction_sensor_update_period_s={foot_reaction_update_period:g}",
            flush=True,
        )

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_direct_race_0423_rsl"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    interactive = not bool(args_cli.headless)
    camera_state = {"mode": "free" if bool(args_cli.free_cam) else "follow", "reset_requested": False}
    follow_camera_enabled = bool(args_cli.follow_camera) and not bool(args_cli.no_follow_camera) and (
        interactive or (bool(args_cli.periodic_eval_video) and not bool(args_cli.simple_video))
    )
    follow_camera_controller = None
    free_camera_controller = None
    camera_window = None
    source_swap_window = None
    friction_popup = None
    friction_slider = None
    foot_friction_popup = None
    slip_cone_visualizer = None
    slip_csv_path = None
    metrics_popup = None
    play_speed_state = {"speed": PlaySpeedControl._clip_speed(float(args_cli.play_speed))}
    play_speed_control = None
    if interactive:
        _try_resize_requested_app_window()
        if not bool(args_cli.keep_isaac_panels):
            _apply_play_view_layout()
        _apply_viewport_light_rig(str(args_cli.light_rig).strip())
        free_camera_controller = StaticCamera(
            raw_env,
            eye_w=tuple(map(float, args_cli.free_cam_eye)),
            lookat_w=tuple(map(float, args_cli.free_cam_lookat)),
            heading_deg=float(args_cli.free_cam_heading_deg),
        )
        if follow_camera_enabled:
            follow_camera_controller = SideFollowCamera(
                raw_env,
                env_index=0,
                eye_b=tuple(map(float, args_cli.camera_eye_b)),
                lookat_b=tuple(map(float, args_cli.camera_lookat_b)),
            )
        if follow_camera_controller is None:
            camera_state["mode"] = "free"
        camera_window = _build_camera_window(camera_state, follow_camera_controller, free_camera_controller)
        if source_sync_run_id is not None:
            source_swap_window = _build_source_swap_window(source_sync_run_id)

        if camera_state["mode"] == "free" or follow_camera_controller is None:
            free_camera_controller.update()
            print("[INFO] Free camera enabled: initialized an overhead view and left the viewport unlocked.", flush=True)
        else:
            follow_camera_controller.update()

        metrics_popup = RaceMetricsPopup(raw_env, env_index=0)
        print("[INFO] Race metrics popup enabled: showing progress, race time, finish time, and collision helpers.", flush=True)
        play_speed_control = PlaySpeedControl(play_speed_state, initial_speed=float(args_cli.play_speed))
        print(
            "[INFO] Play speed control enabled: throttle display playback with the UI speed multiplier.",
            flush=True,
        )

        if bool(args_cli.disable_slider_friction):
            if getattr(getattr(raw_env, "cfg", None), "group_all_patches_single_bucket", False):
                print(
                    "[INFO] Race friction slider disabled by --disable_slider_friction; "
                    "grouped patch friction keeps random bucket sampling on reset.",
                    flush=True,
                )
        else:
            try:
                friction_slider = PatchFrictionSlider(raw_env, env_index=0)
                print(
                    "[INFO] Race friction control enabled: toggle grouped/per-patch mode, apply the slider now, "
                    "or resample patch friction without waiting for reset.",
                    flush=True,
                )
            except RuntimeError as exc:
                print(f"[INFO] Race friction slider inactive: {exc}.", flush=True)
            except Exception as exc:
                print(f"[WARN] Could not create race friction slider: {type(exc).__name__}: {exc}", flush=True)

        if not bool(args_cli.no_friction_popup) and hasattr(raw_env, "get_patch_friction_summary"):
            friction_popup = PatchFrictionSelectionPopup(raw_env, env_index=0)
            print("[INFO] Patch friction popup enabled: click a colored patch prim to inspect mu_static/mu_dynamic.", flush=True)
        if not bool(args_cli.no_foot_friction_popup) and hasattr(raw_env, "_get_gt_patch_mu_obs"):
            foot_friction_popup = FootFrictionPopup(raw_env, env_index=0)
            print("[INFO] Foot friction popup enabled: showing live per-foot mu_static values.", flush=True)
        if bool(args_cli.visualize_slip):
            try:
                slip_csv_path = _resolve_slip_csv_path(args_cli, log_dir, resume_path)
                slip_cone_visualizer = SlipConeVisualizer(
                    raw_env,
                    env_index=0,
                    speed_threshold=float(args_cli.visualize_slip_speed_threshold),
                    superior_markers=bool(args_cli.viz_superior_fric_markers),
                    viz_air_points=bool(args_cli.viz_air_points),
                    physics_rate_csv_path=slip_csv_path,
                    contact_stat=str(args_cli.contact_stats_ui_viz),
                )
                print(
                    "[INFO] Foot slip visualization enabled: coloring by cone angle and contact speed.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[WARN] Could not create slip-cone visualization: {type(exc).__name__}: {exc}", flush=True)
    elif args_cli.periodic_eval_video:
        if bool(args_cli.simple_video):
            camera_state["mode"] = "simple"
            print("[INFO] Headless periodic eval using simple top-down state-space video; Isaac RGB cameras disabled.", flush=True)
        elif follow_camera_enabled:
            recording_env_index = 0
            follow_camera_controller = SideFollowCamera(
                raw_env,
                env_index=recording_env_index,
                eye_b=tuple(map(float, args_cli.camera_eye_b)),
                lookat_b=tuple(map(float, args_cli.camera_lookat_b)),
                camera_path="/World/RaceFollowCamera",
                activate_viewport=True,
            )
            camera_state["mode"] = "follow"
            follow_camera_controller.update()
            _render_rgb_warmup(raw_env, frames=4)
            print(f"[INFO] Headless recording follow camera enabled for env {recording_env_index}.", flush=True)
        else:
            camera_state["mode"] = "free"
            print("[WARN] Headless recording follow camera is disabled; RGB capture will use the default camera.", flush=True)
    elif not interactive:
        if bool(args_cli.visualize_slip):
            try:
                slip_csv_path = _resolve_slip_csv_path(args_cli, log_dir, resume_path)
                slip_cone_visualizer = SlipConeVisualizer(
                    raw_env,
                    env_index=0,
                    speed_threshold=float(args_cli.visualize_slip_speed_threshold),
                    superior_markers=bool(args_cli.viz_superior_fric_markers),
                    viz_air_points=bool(args_cli.viz_air_points),
                    physics_rate_csv_path=slip_csv_path,
                    contact_stat=str(args_cli.contact_stats_ui_viz),
                    headless=True,
                )
                print(
                    "[INFO] Headless slip logging enabled: recording the per-substep slip CSV (no live UI).",
                    flush=True,
                )
            except Exception as exc:
                print(f"[WARN] Could not create headless slip logger: {type(exc).__name__}: {exc}", flush=True)
        print("[WARN] Running headless: follow camera is disabled.")

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
    else:
        apply_checkpoint_architecture_to_policy_cfg(agent_cfg.policy, resume_path)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        # Play mode never steps the optimizer; skipping it avoids param-group mismatches with
        # checkpoints trained under agent.weight_decay (std params split into a decay-free group).
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    obs = vec_env.get_observations()
    if interactive:
        _try_resize_requested_app_window()
        if not bool(args_cli.keep_isaac_panels):
            _apply_play_view_layout()
        _apply_viewport_light_rig(str(args_cli.light_rig).strip())
        if camera_state["mode"] == "follow" and follow_camera_controller is not None:
            follow_camera_controller.update()
        elif free_camera_controller is not None:
            free_camera_controller.update()
    if args_cli.print_patch_friction or bool(getattr(env_cfg, "randomize_fric_coefs", False)):
        _print_patch_friction_summary(raw_env, env_index=0)

    tsne_env_index = min(max(0, int(args_cli.tsne_viz_env_index)), int(raw_env.num_envs) - 1)
    encoding_viz_recorder = None
    if args_cli.tsne_encoding_viz:
        encoding_viz_recorder = EncodingVizRecorder(
            policy=policy,
            render_env=env,
            raw_env=raw_env,
            log_dir=log_dir,
            checkpoint_path=resume_path,
            output_dir=args_cli.tsne_viz_dir,
            interval_s=float(args_cli.tsne_viz_interval_s),
            dt=float(dt),
            max_points=int(args_cli.tsne_viz_max_points),
            method=str(args_cli.tsne_viz_method),
            perplexity=float(args_cli.tsne_viz_perplexity),
            max_iter=int(args_cli.tsne_viz_max_iter),
            env_index=tsne_env_index,
            video_fps=args_cli.tsne_viz_video_fps,
        )
        if args_cli.tsne_num_episodes is not None:
            print(
                f"[INFO] Encoding visualization will stop after {int(args_cli.tsne_num_episodes)} completed episodes "
                f"for env {tsne_env_index} (duration_s remains a safety cap).",
                flush=True,
            )
        encoding_viz_recorder.maybe_record(timestep=0, sim_time_s=0.0, obs=obs, force=True)

    tsne_target_episodes = args_cli.tsne_num_episodes if args_cli.tsne_encoding_viz else None
    tsne_episodes_done = 0
    tsne_sample_reset = not bool(args_cli.no_reset_samples_tsne)
    if args_cli.tsne_encoding_viz:
        reset_mode = "also keeping the live window across episodes" if not tsne_sample_reset else "resetting the live window each episode"
        print(
            f"[INFO] Encoding visualization sample mode: final MP4/CSV keep all retained samples; {reset_mode}.",
            flush=True,
        )

    periodic_video_recorder = None
    periodic_eval_target_episodes = max(1, int(args_cli.periodic_eval_video_episodes))
    periodic_eval_episodes_done = 0
    if args_cli.periodic_eval_video:
        output_path = args_cli.periodic_eval_video_output
        if output_path is None:
            checkpoint_stem = Path(resume_path).stem
            output_path = os.path.join(
                log_dir,
                "videos",
                "periodic_eval",
                f"{checkpoint_stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
            )
        periodic_video_recorder = PeriodicEvalVideoWriter(
            output_path=output_path,
            dt=float(dt),
            speed=float(args_cli.periodic_eval_video_speed),
            target_episodes=periodic_eval_target_episodes,
            wandb_upload=bool(args_cli.wandb_upload_video),
            wandb_project=str(args_cli.wandb_project),
            wandb_entity=args_cli.wandb_entity,
            wandb_run_id=args_cli.wandb_run_id or os.environ.get("WANDB_RUN_ID"),
            wandb_step=args_cli.wandb_video_step,
            wandb_key=str(args_cli.wandb_video_key),
            simple_video=bool(args_cli.simple_video),
        )

    target_steps = max(1, int(round(float(args_cli.duration_s) / float(dt))))
    if args_cli.periodic_eval_video and int(args_cli.periodic_eval_video_max_steps) > 0:
        target_steps = int(args_cli.periodic_eval_video_max_steps)
    gate_history = []
    speed_history = []
    ang_vel_history = []

    for timestep in range(target_steps):
        loop_t0 = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            try:
                policy.actor_critic.reset(dones)
            except AttributeError:
                pass

        stop_after_success = bool(args_cli.generate_slip_plots) and _completed_race_episode(raw_env, dones, env_index=0)

        if camera_state.get("reset_requested", False):
            camera_state["reset_requested"] = False
            print("[INFO] Manual reset requested from UI.", flush=True)
            if metrics_popup is not None:
                metrics_popup._reset_episode_counters()
            obs = _manual_reset_all_envs(raw_env, vec_env)
            try:
                policy.actor_critic.reset(torch.ones(raw_env.num_envs, dtype=torch.bool, device=raw_env.device))
            except AttributeError:
                pass

        if hasattr(raw_env, "_current_gate_idx"):
            gate_history.append(float(raw_env._current_gate_idx.float().mean().item()))
        if hasattr(raw_env, "_robot"):
            speed_history.append(float(torch.norm(raw_env._robot.data.root_lin_vel_b[:, :2], dim=1).mean().item()))
            ang_vel_history.append(float(raw_env._robot.data.root_ang_vel_b[:, 2].mean().item()))

        if isinstance(dones, torch.Tensor) and dones.any():
            obs = vec_env.get_observations()
            try:
                policy.actor_critic.reset(dones)
            except AttributeError:
                pass

        if camera_state["mode"] == "follow" and follow_camera_controller is not None:
            follow_camera_controller.update()
        if friction_popup is not None:
            friction_popup.update()
        if friction_slider is not None:
            friction_slider.update()
        if foot_friction_popup is not None:
            foot_friction_popup.update()
        if slip_cone_visualizer is not None:
            slip_cone_visualizer.update()
        if metrics_popup is not None:
            metrics_popup.update(dones=dones)
        if interactive and timestep < 120 and timestep % 10 == 0:
            _try_resize_requested_app_window()
        if encoding_viz_recorder is not None:
            encoding_viz_recorder.maybe_record(timestep=timestep + 1, sim_time_s=(timestep + 1) * float(dt), obs=obs)
        if periodic_video_recorder is not None:
            periodic_video_recorder.append(
                env,
                raw_env,
                timestep=timestep + 1,
                episodes_done=periodic_eval_episodes_done,
            )

        if stop_after_success:
            print(
                f"[INFO] Race task completed after {(timestep + 1) * float(dt):.3f}s; "
                "stopping early for --generate-slip-plots.",
                flush=True,
            )
            break

        if args_cli.tsne_encoding_viz and isinstance(dones, torch.Tensor):
            if bool(dones.reshape(-1)[tsne_env_index].item()):
                tsne_episodes_done += 1
                if tsne_target_episodes is not None and tsne_episodes_done >= int(tsne_target_episodes):
                    print(
                        f"[INFO] Encoding visualization completed {tsne_episodes_done}/{int(tsne_target_episodes)} "
                        f"episodes; stopping play. Output: {encoding_viz_recorder.out_dir}",
                        flush=True,
                    )
                    break
                if tsne_sample_reset and encoding_viz_recorder is not None:
                    encoding_viz_recorder.reset_samples()

        if args_cli.periodic_eval_video and isinstance(dones, torch.Tensor) and bool(dones.reshape(-1)[0].item()):
            periodic_eval_episodes_done += 1
            if periodic_eval_episodes_done >= periodic_eval_target_episodes:
                print(
                    f"[INFO] Periodic eval completed {periodic_eval_episodes_done}/{periodic_eval_target_episodes} episodes.",
                    flush=True,
                )
                break

        target_wall_dt = float(dt) / max(PlaySpeedControl._clip_speed(play_speed_state.get("speed", 1.0)), 1.0e-6)
        sleep_time = target_wall_dt - (time.time() - loop_t0)
        if (interactive or args_cli.real_time) and sleep_time > 0:
            time.sleep(sleep_time)

    summary = _compute_summary(raw_env)
    if gate_history:
        print(f"[RESULT] median_gate_idx={statistics.median(gate_history):.3f}")
    if speed_history:
        print(f"[RESULT] median_root_speed_xy={statistics.median(speed_history):.3f}")
    if ang_vel_history:
        print(f"[RESULT] median_root_ang_vel_z={statistics.median(ang_vel_history):.3f}")
    for key, value in summary.items():
        print(f"[RESULT] current_{key}={value:.3f}")

    if encoding_viz_recorder is not None:
        encoding_viz_recorder.close()
    if periodic_video_recorder is not None:
        periodic_video_recorder.close()
    if args_cli.video or args_cli.periodic_eval_video:
        _detach_rgb_render_product(raw_env)
    if friction_slider is not None:
        friction_slider.close()
    if play_speed_control is not None:
        play_speed_control.close()
    if slip_cone_visualizer is not None:
        slip_cone_visualizer.close()

    # close() above flushes and closes the slip CSV, so the file is complete before we plot it.
    _generate_slip_plots_after_run(args_cli, slip_csv_path, env_cfg=env_cfg)

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
