# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL translation of play_direct_0325 for the direct Solo12 environment."""

"""
Example usage: 
./isaaclab.sh -p source/scripts/rsl_rl/play_direct_0325.py  --task="solo12-v0"  --checkpoint "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0406_kemck4qa_best_model.pt"  --num_envs 1  --duration_s 2000  --cmd_init 0.5 0.0 0.0 --episode_length_s 80

"""


import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_ISAACLAB_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
_RSL_RL_SAC_VENDOR_DIR = _ISAACLAB_ROOT / "source" / "rsl_rl_sac_vendor"
for _path in (str(_UPSTREAM_RSL_SCRIPT_DIR), str(_RSL_RL_SAC_VENDOR_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

SequenceAnalysisRecorder = None
checkpoint_label_from_path = None
parse_record_sequence = None
record_sequence_command_at = None
record_sequence_total_s = None


def _solo12_policy_inference_search_paths() -> list[Path]:
    candidates = []
    env_path = os.environ.get("SOLO12_POLICY_INFERENCE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            _ISAACLAB_ROOT / "source" / "solo12_policy_inference",
            _ISAACLAB_ROOT / "source" / "solo12-velocity-cmd-policy-inference",
            Path.home() / "hidro" / "solo_ws" / "src" / "solo12-velocity-cmd-policy-inference",
        ]
    )
    return candidates


def _load_sequence_analysis_helpers() -> None:
    global SequenceAnalysisRecorder
    global checkpoint_label_from_path
    global parse_record_sequence
    global record_sequence_command_at
    global record_sequence_total_s

    if parse_record_sequence is not None:
        return

    checked_paths = []
    for candidate in _solo12_policy_inference_search_paths():
        candidate = candidate.resolve()
        checked_paths.append(str(candidate))
        if (candidate / "solo12_policy_inference").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    try:
        from solo12_policy_inference.sequence_analysis import (  # isort: skip
            SequenceAnalysisRecorder as _SequenceAnalysisRecorder,
            checkpoint_label_from_path as _checkpoint_label_from_path,
            parse_record_sequence as _parse_record_sequence,
            record_sequence_command_at as _record_sequence_command_at,
            record_sequence_total_s as _record_sequence_total_s,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "solo12_policy_inference":
            raise
        raise ModuleNotFoundError(
            "--record-sequence needs the optional 'solo12_policy_inference' package. "
            "Install/source the Solo12 ROS inference workspace, put the repo under IsaacLab/source, "
            "or set SOLO12_POLICY_INFERENCE_PATH to its checkout root. Checked: " + ", ".join(checked_paths)
        ) from exc

    SequenceAnalysisRecorder = _SequenceAnalysisRecorder
    checkpoint_label_from_path = _checkpoint_label_from_path
    parse_record_sequence = _parse_record_sequence
    record_sequence_command_at = _record_sequence_command_at
    record_sequence_total_s = _record_sequence_total_s


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _consume_record_sequence_hydra_override(args_cli, hydra_args: list[str]) -> list[str]:
    filtered_args = []
    for arg in hydra_args:
        if arg.startswith("record_sequence:="):
            args_cli.record_sequence = _truthy(arg.split(":=", 1)[1])
        elif arg.startswith("record_sequence="):
            args_cli.record_sequence = _truthy(arg.split("=", 1)[1])
        else:
            filtered_args.append(arg)
    return filtered_args


parser = argparse.ArgumentParser(description="Play a direct Solo12 checkpoint with a fixed command using RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=400, help="Recorded video length in steps.")
parser.add_argument(
    "--record-sequence",
    action="store_true",
    default=False,
    help="Run the fixed 35 s Solo12 command-sequence comparison and save/upload telemetry plots.",
)
parser.add_argument(
    "--track-cmds",
    type=str,
    default="",
    help=(
        "Run a command-tracking experiment and save error plots. Commands are semicolon-separated "
        "'vx vy wz' rows; each command runs for 5 s. Example: '0.5 0 0; -0.5 0 0; 0 0.3 0'."
    ),
)
parser.add_argument(
    "--tracking-wandb-project",
    type=str,
    default="solo12-two-feet-exp",
    help="W&B project used automatically for --track-cmds evaluation artifacts.",
)
parser.add_argument("--tracking-wandb-entity", type=str, default=None, help="Optional W&B entity for --track-cmds.")
parser.add_argument("--tracking-wandb-name", type=str, default=None, help="Optional W&B run name for --track-cmds.")
parser.add_argument(
    "--no-tracking-wandb",
    action="store_true",
    default=False,
    help="Keep --track-cmds artifacts local without uploading to W&B.",
)
parser.add_argument(
    "--record-sequence-commands",
    type=str,
    default="",
    help=(
        "Command sequence for --record-sequence. Use semicolon-separated 'vx vy wz duration_s' rows, "
        "e.g. '0.5 0 0 5; 1.2 0 0 5'. Empty uses the default 35 s sequence."
    ),
)
parser.add_argument(
    "--analysis_output_dir",
    type=str,
    default=None,
    help="Output root for --record-sequence telemetry artifacts. Defaults under the checkpoint log dir.",
)
parser.add_argument(
    "--analysis_wandb_project",
    type=str,
    default="analysis-solo-inference",
    help="W&B project for --record-sequence telemetry plots.",
)
parser.add_argument("--analysis_wandb_entity", type=str, default=None, help="Optional W&B entity for analysis upload.")
parser.add_argument("--analysis_wandb_name", type=str, default=None, help="Optional W&B run name for analysis upload.")
parser.add_argument(
    "--no_analysis_wandb",
    action="store_true",
    default=False,
    help="Disable automatic W&B upload for --record-sequence telemetry artifacts.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="solo12-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument(
    "--dagger-teacher-checkpoint",
    "--dagger_teacher_checkpoint",
    dest="dagger_teacher_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional teacher checkpoint override for solo12 base-IMU DAgger adapter checkpoints. "
        "If omitted, the teacher path saved inside the adapter checkpoint is used."
    ),
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument(
    "--cmd",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 0.0),
    metavar=("VX", "VY", "WZ"),
    help="Fixed command (vx vy wz) applied to all environments.",
)
parser.add_argument("--duration_s", type=float, default=5.0, help="How long to run the policy.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=40.0,
    help="Play-only environment episode timeout in seconds. This overrides the task config during inference.",
)
parser.add_argument(
    "--cmd_init",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 0.0),
    metavar=("VX", "VY", "WZ"),
    help="Initial UI command shown in the window: (vx, vy, wz).",
)
parser.add_argument(
    "--vx_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Live vx slider range. Defaults to env_cfg.command_lin_vel_x_range.",
)
parser.add_argument(
    "--vy_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Live vy slider range. Defaults to env_cfg.command_lin_vel_y_range.",
)
parser.add_argument(
    "--wz_range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
    help="Live wz slider range [rad/s]. Defaults to env_cfg.command_ang_vel_z_range.",
)
parser.add_argument(
    "--ui_title",
    type=str,
    default="Solo12 direct live command control (vx, vy, wz)",
    help="Title of the Omni.UI command window.",
)
parser.add_argument(
    "--apply-force-ui",
    action="store_true",
    default=False,
    help="Show an Omni.UI panel for point forces on the Solo12 base (3-D direction, surface point, and pulse duration).",
)
parser.add_argument(
    "--force-ui-max-magnitude",
    type=float,
    default=None,
    help=(
        "Maximum force magnitude shown by the apply-force UI slider [N]. Defaults to the largest configured "
        "base_push_force_xy_range / forces_applied_to_base_curriculum value."
    ),
)
parser.add_argument(
    "--force-ui-initial-magnitude",
    type=float,
    default=0.0,
    help="Initial selected body-frame force magnitude in the apply-force UI [N].",
)
parser.add_argument(
    "--force-ui-initial-angle-deg",
    type=float,
    default=0.0,
    help="Initial selected body-frame XY force direction angle in degrees. 0 is +body-x, 90 is +body-y.",
)
parser.add_argument(
    "--force-ui-initial-elevation-deg",
    type=float,
    default=0.0,
    help="Initial force elevation in degrees. 0 is in body XY; +90 is +body-z.",
)
parser.add_argument(
    "--force-ui-initial-duration-s",
    type=float,
    default=0.25,
    help="Initial pulse duration selected in the force UI [s].",
)
parser.add_argument(
    "--force_transmited_through_joints_reward_scale",
    type=float,
    default=None,
    help="Override the environment reward scale for force_transmited_through_joints.",
)
parser.add_argument(
    "--draw-helper-plane-at",
    "--draw_helper_plane_at",
    dest="draw_helper_plane_at",
    type=float,
    default=None,
    metavar="HEIGHT_M",
    help=(
        "Draw a translucent horizontal plane at this height [m] above the local terrain below env 0's "
        "front feet. The plane follows the robot horizontally. Default: disabled."
    ),
)
parser.add_argument(
    "--chase_camera",
    "--follow_camera",
    dest="follow_camera",
    action="store_true",
    default=True,
    help="Enable scripted viewport cameras attached to the robot yaw frame.",
)
parser.add_argument(
    "--no_chase_camera",
    "--no_follow_camera",
    dest="no_follow_camera",
    action="store_true",
    default=False,
    help="Disable scripted viewport cameras.",
)
parser.add_argument(
    "--camera_mode",
    type=str,
    choices=("chase", "follow", "side", "free"),
    default="chase",
    help=(
        "Initial interactive viewport camera mode. 'chase' tracks Solo12 from behind; "
        "'follow'/'side' tracks Solo12 from the side; 'free' leaves the viewport camera user-controlled."
    ),
)
parser.add_argument(
    "--camera_eye_b",
    type=float,
    nargs=3,
    default=(-2.6, 0.0, 1.0),
    metavar=("X", "Y", "Z"),
    help="Chase camera eye offset in the robot yaw frame.",
)
parser.add_argument(
    "--camera_lookat_b",
    type=float,
    nargs=3,
    default=(0.45, 0.0, 0.35),
    metavar=("X", "Y", "Z"),
    help="Chase camera look-at offset in the robot yaw frame.",
)
parser.add_argument(
    "--side_camera_eye_b",
    type=float,
    nargs=3,
    default=(0.0, -2.6, 1.1),
    metavar=("X", "Y", "Z"),
    help="Side-follow camera eye offset in the robot yaw frame.",
)
parser.add_argument(
    "--side_camera_lookat_b",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.35),
    metavar=("X", "Y", "Z"),
    help="Side-follow camera look-at offset in the robot yaw frame.",
)
parser.add_argument("--wandb", action="store_true", default=False, help="Log metrics to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="borinotIsaacLab_inference", help="W&B project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team.")
parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name.")
parser.add_argument("--verbose_play", action="store_true", default=False, help="Print per-step telemetry to the terminal.")
parser.add_argument(
    "--keep_training_stochasticity",
    action="store_true",
    default=False,
    help=(
        "Keep training-time stochasticity during play. By default, play disables domain randomization events, "
        "observation corruption, reset velocity randomization, and action delay for cleaner evaluation."
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
    "--disable_training_gain_sync",
    action="store_true",
    default=False,
    help="Disable the automatic W&B lookup that syncs Solo12 actuator KP/KD before inference.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
hydra_args = _consume_record_sequence_hydra_override(args_cli, hydra_args)
RECORD_SEQUENCE = None


def _parse_tracking_commands(value: str) -> list[tuple[float, float, float]]:
    commands = []
    for index, row in enumerate(value.split(";"), start=1):
        row = row.strip()
        if not row:
            continue
        fields = row.split()
        if len(fields) != 3:
            raise ValueError(f"--track-cmds row {index} must contain exactly vx vy wz, got: {row!r}")
        command = tuple(float(field) for field in fields)
        if not all(math.isfinite(component) for component in command):
            raise ValueError(f"--track-cmds row {index} contains a non-finite value: {row!r}")
        commands.append(command)
    if not commands:
        raise ValueError("--track-cmds must contain at least one vx vy wz command")
    return commands


def _evaluation_duration_s(requested_duration_s: float, tracking_commands) -> float:
    return (len(tracking_commands) * TRACKING_COMMAND_DURATION_S
            if tracking_commands else float(requested_duration_s))


TRACKING_COMMAND_DURATION_S = 5.0
TRACKING_COMMANDS = _parse_tracking_commands(args_cli.track_cmds) if args_cli.track_cmds.strip() else []

if args_cli.checkpoint is None and not args_cli.use_pretrained_checkpoint:
    parser.error("either --checkpoint or --use_pretrained_checkpoint is required")

if args_cli.record_sequence:
    _load_sequence_analysis_helpers()
    RECORD_SEQUENCE = parse_record_sequence(args_cli.record_sequence_commands)
    if args_cli.video:
        print("[WARN] --record-sequence now records telemetry plots; ignoring --video.", flush=True)
    args_cli.video = False
    args_cli.duration_s = record_sequence_total_s(RECORD_SEQUENCE)
    args_cli.cmd_init = RECORD_SEQUENCE[0][0]
    args_cli.camera_mode = "chase"
    args_cli.follow_camera = True
    args_cli.no_follow_camera = False

if TRACKING_COMMANDS:
    if args_cli.record_sequence:
        parser.error("--track-cmds and --record-sequence cannot be used together")
    args_cli.duration_s = len(TRACKING_COMMANDS) * TRACKING_COMMAND_DURATION_S
    args_cli.cmd_init = TRACKING_COMMANDS[0]

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import wandb
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl_sac.runners import OffPolicyRunner
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401
from isaaclab_tasks.direct.solo12.agents.base_imu_actor_critic import (
    BaseImuTcnEncoder,
    Solo12BaseImuTeacherActorCritic,
)
from isaaclab_tasks.direct.solo12.solo12_env import _world_velocity_in_heading_frame_xy


@dataclass
class LiveCommandState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    reset_requested: bool = False

    def __post_init__(self):
        self._lock = threading.Lock()

    def set_vx(self, value: float):
        with self._lock:
            self.vx = float(value)

    def set_vy(self, value: float):
        with self._lock:
            self.vy = float(value)

    def set_wz(self, value: float):
        with self._lock:
            self.wz = float(value)

    def set_command(self, vx: float, vy: float, wz: float):
        with self._lock:
            self.vx = float(vx)
            self.vy = float(vy)
            self.wz = float(wz)

    def request_reset(self):
        with self._lock:
            self.reset_requested = True

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = bool(self.reset_requested)
            self.reset_requested = False
            return requested

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            return float(self.vx), float(self.vy), float(self.wz)


@dataclass
class BodyFrameForceState:
    selected_magnitude: float = 0.0
    selected_angle_deg: float = 0.0
    selected_elevation_deg: float = 0.0
    selected_point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    selected_duration_s: float = 0.25
    dt: float = 0.02

    def __post_init__(self):
        self._lock = threading.Lock()
        self._active_force: tuple[float, float, float, float, float, float, float, float] | None = None
        self._pulse_steps_left: int | None = 0
        self._clear_requested = False

    def set_magnitude(self, value: float):
        with self._lock:
            self.selected_magnitude = max(0.0, float(value))

    def set_angle_deg(self, value: float):
        with self._lock:
            self.selected_angle_deg = float(value)

    def set_elevation_deg(self, value: float):
        with self._lock:
            self.selected_elevation_deg = float(np.clip(value, -90.0, 90.0))

    def set_point_axis(self, axis: int, value: float):
        with self._lock:
            point = list(self.selected_point_b)
            point[axis] = float(value)
            self.selected_point_b = tuple(point)

    def set_point(self, point_b):
        with self._lock:
            self.selected_point_b = tuple(map(float, point_b))

    def set_duration_s(self, value: float):
        with self._lock:
            self.selected_duration_s = max(float(self.dt), float(value))

    def _selected_locked(self):
        magnitude = max(0.0, float(self.selected_magnitude))
        azimuth_deg = float(self.selected_angle_deg)
        elevation_deg = float(self.selected_elevation_deg)
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        horizontal = magnitude * math.cos(elevation)
        force = (
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            magnitude * math.sin(elevation),
        )
        return (*force, *self.selected_point_b, magnitude, azimuth_deg, elevation_deg)

    def apply_selected(self, continuous: bool = False):
        with self._lock:
            fx_b, fy_b, fz_b, px_b, py_b, pz_b, _magnitude, azimuth_deg, elevation_deg = self._selected_locked()
            self._active_force = (fx_b, fy_b, fz_b, px_b, py_b, pz_b, azimuth_deg, elevation_deg)
            self._pulse_steps_left = None if continuous else max(1, int(round(self.selected_duration_s / self.dt)))
            self._clear_requested = False

    def request_clear(self):
        with self._lock:
            self._active_force = None
            self._pulse_steps_left = 0
            self._clear_requested = True

    def tick(self):
        with self._lock:
            if self._active_force is None or self._pulse_steps_left is None:
                return
            self._pulse_steps_left -= 1
            if self._pulse_steps_left <= 0:
                self._active_force = None
                self._clear_requested = True

    def consume_clear_request(self) -> bool:
        with self._lock:
            requested = bool(self._clear_requested)
            self._clear_requested = False
            return requested

    def get_selected(self):
        with self._lock:
            return self._selected_locked()

    def get_active_force(self):
        with self._lock:
            return self._active_force


@dataclass
class CameraModeState:
    mode: str = "chase"

    def __post_init__(self):
        self._lock = threading.Lock()
        self.mode = self._normalize(self.mode)

    @staticmethod
    def _normalize(mode: str) -> str:
        mode = str(mode).lower()
        if mode == "side":
            return "follow"
        if mode not in ("chase", "follow", "free"):
            return "chase"
        return mode

    def set_mode(self, mode: str):
        with self._lock:
            self.mode = self._normalize(mode)

    def toggle(self):
        with self._lock:
            modes = ("chase", "follow", "free")
            current = self._normalize(self.mode)
            self.mode = modes[(modes.index(current) + 1) % len(modes)]

    def is_chase(self) -> bool:
        with self._lock:
            return self._normalize(self.mode) == "chase"

    def is_follow(self) -> bool:
        with self._lock:
            return self._normalize(self.mode) == "follow"

    def scripted_mode(self) -> str | None:
        with self._lock:
            mode = self._normalize(self.mode)
            return mode if mode in ("chase", "follow") else None

    def label(self) -> str:
        with self._lock:
            return self._normalize(self.mode)


def _infer_wandb_run_id_from_checkpoint(checkpoint_path: str) -> str | None:
    stem = Path(checkpoint_path).stem.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    skip_tokens = {"best", "agent", "model", "lastagent", "last", "overnight", "checkpoint"}
    for token in tokens:
        if token in skip_tokens:
            continue
        if token.isdigit():
            continue
        if 7 <= len(token) <= 8 and any(ch.isalpha() for ch in token):
            return token
    return None


def _fetch_training_run_config_from_wandb(entity: str, project: str, run_id: str, max_attempts: int = 3) -> tuple[str, dict]:
    api = wandb.Api(timeout=30)
    projects = [project]
    legacy_dagger_project = "solo12_base_imu_dagger"
    if legacy_dagger_project not in projects:
        projects.append(legacy_dagger_project)

    errors = []
    for project_name in projects:
        run_path = f"{entity}/{project_name}/{run_id}"
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                run = api.run(run_path)
                config = dict(run.config or {})
                if not config:
                    raise RuntimeError(f"W&B run '{run_path}' returned an empty config.")
                if project_name != project:
                    print(f"[INFO] Found training W&B run in fallback project '{project_name}': {run_path}", flush=True)
                return run_path, config
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts:
                    sleep_s = float(attempt)
                    print(
                        f"[WARN] Failed to fetch W&B config for '{run_path}' on attempt {attempt}/{max_attempts}: {exc}. "
                        f"Retrying in {sleep_s:.1f}s...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
        errors.append(f"{run_path}: {last_error}")
    raise RuntimeError(
        f"Failed to fetch W&B config for run id '{run_id}' from projects {projects}: " + " | ".join(errors)
    )


def _actuator_gain_to_float(value, field_name: str) -> float:
    if isinstance(value, dict):
        if not value:
            raise RuntimeError(f"Actuator {field_name} dictionary is empty.")
        gains = [_actuator_gain_to_float(item, field_name) for item in value.values()]
        first_gain = gains[0]
        if any(not math.isclose(gain, first_gain, rel_tol=0.0, abs_tol=1.0e-12) for gain in gains[1:]):
            raise RuntimeError(f"Actuator {field_name} has per-joint values; expected one shared gain.")
        return first_gain
    return float(value)


def _sync_gain_preserving_shape(current_value, synced_value: float):
    synced_value = float(synced_value)
    if isinstance(current_value, dict):
        return {key: synced_value for key in current_value}
    return synced_value


def _hydra_override_keys(raw_args: list[str]) -> set[str]:
    keys = set()
    for raw_arg in raw_args:
        if "=" not in raw_arg:
            continue
        key = raw_arg.split("=", 1)[0].strip().lstrip("+~")
        if key.endswith(":"):
            key = key[:-1]
        keys.add(key)
    return keys


def _capture_cli_actuator_gain_overrides(env_cfg, raw_hydra_args: list[str]) -> dict[str, float]:
    """Capture explicit CLI gain overrides before W&B sync can modify the actuator config."""
    override_keys = _hydra_override_keys(raw_hydra_args)
    if not override_keys.intersection(
        {
            "env.kp",
            "env.kd",
            "env.robot.actuators.legs.stiffness",
            "env.robot.actuators.legs.damping",
        }
    ):
        return {}

    legs_actuator = env_cfg.robot.actuators.get("legs")
    if legs_actuator is None:
        raise RuntimeError("Solo12 env config does not contain a 'legs' actuator for CLI KP/KD overrides.")

    overrides = {}
    if "env.kp" in override_keys:
        overrides["stiffness"] = float(env_cfg.kp)
    elif "env.robot.actuators.legs.stiffness" in override_keys:
        overrides["stiffness"] = _actuator_gain_to_float(legs_actuator.stiffness, "stiffness")

    if "env.kd" in override_keys:
        overrides["damping"] = float(env_cfg.kd)
    elif "env.robot.actuators.legs.damping" in override_keys:
        overrides["damping"] = _actuator_gain_to_float(legs_actuator.damping, "damping")

    return overrides


def _apply_cli_actuator_gain_overrides(env_cfg, overrides: dict[str, float]) -> None:
    """Apply explicit CLI gains after W&B sync so command-line values take precedence."""
    if not overrides:
        return

    legs_actuator = env_cfg.robot.actuators.get("legs")
    if legs_actuator is None:
        raise RuntimeError("Solo12 env config does not contain a 'legs' actuator for CLI KP/KD overrides.")

    messages = []
    if "stiffness" in overrides:
        previous = _actuator_gain_to_float(legs_actuator.stiffness, "stiffness")
        value = float(overrides["stiffness"])
        legs_actuator.stiffness = _sync_gain_preserving_shape(legs_actuator.stiffness, value)
        if hasattr(env_cfg, "kp"):
            env_cfg.kp = value
        messages.append(f"stiffness(KP) {previous:g} -> {value:g}")

    if "damping" in overrides:
        previous = _actuator_gain_to_float(legs_actuator.damping, "damping")
        value = float(overrides["damping"])
        legs_actuator.damping = _sync_gain_preserving_shape(legs_actuator.damping, value)
        if hasattr(env_cfg, "kd"):
            env_cfg.kd = value
        messages.append(f"damping(KD) {previous:g} -> {value:g}")

    print("[INFO] Applied Solo12 actuator gains from CLI overrides: " + ", ".join(messages), flush=True)


def _extract_training_kp_kd_from_wandb_config(run_config: dict) -> tuple[float, float]:
    env_cfg = run_config.get("env_cfg")
    if env_cfg is None:
        env_cfg = run_config.get("env")
    if not isinstance(env_cfg, dict):
        raise RuntimeError("W&B run config does not contain a usable 'env_cfg' or 'env' dictionary.")

    robot_cfg = env_cfg.get("robot")
    if not isinstance(robot_cfg, dict):
        raise RuntimeError("W&B env_cfg does not contain a usable 'robot' dictionary.")

    actuator_cfgs = robot_cfg.get("actuators")
    if not isinstance(actuator_cfgs, dict):
        raise RuntimeError("W&B env_cfg.robot does not contain a usable 'actuators' dictionary.")

    legs_cfg = actuator_cfgs.get("legs")
    if not isinstance(legs_cfg, dict):
        raise RuntimeError("W&B env_cfg.robot.actuators does not contain a usable 'legs' dictionary.")

    if "stiffness" not in legs_cfg or "damping" not in legs_cfg:
        raise RuntimeError("W&B env_cfg.robot.actuators.legs does not contain both 'stiffness' and 'damping'.")

    return (
        _actuator_gain_to_float(legs_cfg["stiffness"], "stiffness"),
        _actuator_gain_to_float(legs_cfg["damping"], "damping"),
    )


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _off_policy_inference_runner_cfg(agent_cfg: Any, num_envs: int) -> dict[str, Any]:
    """Return a runner config whose replay storage is minimal for actor-only inference."""
    runner_cfg = copy.deepcopy(_cfg_to_dict(agent_cfg))
    if "algorithm" not in runner_cfg:
        raise ValueError("Off-policy agent config is missing its algorithm section.")
    runner_cfg["algorithm"]["replay_buffer_size"] = max(int(num_envs), 1)
    return runner_cfg


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


def _load_dagger_adapter_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any] | None:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "adapter_state_dict" in checkpoint and "teacher_checkpoint" in checkpoint:
        return checkpoint
    return None


def _resolve_dagger_teacher_checkpoint(adapter_checkpoint: dict[str, Any]) -> str:
    if args_cli.dagger_teacher_checkpoint is not None:
        candidates = [args_cli.dagger_teacher_checkpoint]
    else:
        saved_path = str(adapter_checkpoint["teacher_checkpoint"])
        candidates = [saved_path]
        remote_prefix = "/home/jbeltran/IsaacLab"
        if saved_path.startswith(remote_prefix):
            candidates.append(str(_ISAACLAB_ROOT) + saved_path[len(remote_prefix) :])
        if saved_path.startswith("~/IsaacLab/"):
            candidates.append(str(_ISAACLAB_ROOT / saved_path[len("~/IsaacLab/") :]))

    expanded_candidates = [os.path.abspath(os.path.expanduser(candidate)) for candidate in candidates]
    for candidate in expanded_candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find the teacher checkpoint needed by this DAgger adapter. "
        "Pass --dagger-teacher-checkpoint explicitly or copy it locally. Tried: "
        + ", ".join(expanded_candidates)
    )


def _load_base_imu_teacher_policy(
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


class BaseImuDaggerAdapterPolicy:
    def __init__(
        self,
        *,
        teacher: Solo12BaseImuTeacherActorCritic,
        adapter: BaseImuTcnEncoder,
        history_normalizer: EmpiricalNormalization | None,
        teacher_obs_dim: int,
        history_dim: int,
        device: torch.device,
    ) -> None:
        self.teacher = teacher
        self.adapter = adapter
        self.history_normalizer = history_normalizer
        self.teacher_obs_dim = int(teacher_obs_dim)
        self.history_dim = int(history_dim)
        self.device = device
        self.actor_critic = teacher

    @torch.no_grad()
    def __call__(self, obs) -> torch.Tensor:
        policy_obs = obs["policy"] if isinstance(obs, (dict, TensorDict)) else obs
        policy_obs = policy_obs.to(self.device)
        teacher_obs_raw = policy_obs[:, : self.teacher_obs_dim]
        history_raw = policy_obs[:, self.teacher_obs_dim : self.teacher_obs_dim + self.history_dim]
        teacher_obs = self.teacher.actor_obs_normalizer(teacher_obs_raw)
        command = teacher_obs[
            :, self.teacher.teacher_encoder_obs_dim : self.teacher.teacher_encoder_obs_dim + self.teacher.command_dim
        ]
        history = self.history_normalizer(history_raw) if self.history_normalizer is not None else history_raw
        z = self.adapter(history)
        return self.teacher.actor(torch.cat((z, command), dim=-1))


def _build_base_imu_dagger_adapter_policy(
    *,
    adapter_checkpoint: dict[str, Any],
    vec_env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    resume_path: str,
) -> BaseImuDaggerAdapterPolicy:
    device = torch.device(vec_env.unwrapped.device)
    raw_env = vec_env.unwrapped
    teacher_obs_dim = int(raw_env.cfg.teacher_critic_obs_dim)
    history_dim = int(raw_env.cfg.base_imu_history_flat_dim)
    saved_dims = adapter_checkpoint.get("dims", {})
    saved_sample_dim = saved_dims.get("history_sample_dim") if isinstance(saved_dims, dict) else None
    if saved_sample_dim is not None and int(saved_sample_dim) != int(raw_env.cfg.base_imu_history_sample_dim):
        raise ValueError(
            "DAgger adapter IMU layout does not match the play environment: "
            f"checkpoint history_sample_dim={saved_sample_dim}, current={raw_env.cfg.base_imu_history_sample_dim}. "
            "Use matching imu_ekf_processed_inputs/use_rotMat_on_imu_encoder settings."
        )
    command_dim = 3
    policy_obs_dim = int(vec_env.get_observations()["policy"].shape[-1])
    expected_obs_dim = teacher_obs_dim + history_dim + command_dim
    if policy_obs_dim != expected_obs_dim:
        raise ValueError(f"DAgger obs dim {policy_obs_dim} != expected {expected_obs_dim}.")

    dims = adapter_checkpoint.get("dims", {})
    if isinstance(dims, dict):
        if int(dims.get("teacher_obs_dim", teacher_obs_dim)) != teacher_obs_dim:
            raise ValueError(
                f"Adapter teacher obs dim {dims.get('teacher_obs_dim')} != env teacher obs dim {teacher_obs_dim}. "
                "Check env.include_foot_height_obs and the teacher checkpoint used for DAgger."
            )
        if int(dims.get("history_flat_dim", history_dim)) != history_dim:
            raise ValueError(f"Adapter history dim {dims.get('history_flat_dim')} != env history dim {history_dim}.")

    teacher_checkpoint = _resolve_dagger_teacher_checkpoint(adapter_checkpoint)
    teacher = _load_base_imu_teacher_policy(
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
    adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()

    history_normalizer = None
    normalizer_state = adapter_checkpoint.get("history_normalizer_state_dict")
    if normalizer_state is not None:
        history_normalizer = EmpiricalNormalization(history_dim).to(device)
        history_normalizer.load_state_dict(normalizer_state)
        history_normalizer.eval()

    print(f"[INFO] Loading DAgger adapter checkpoint from: {resume_path}", flush=True)
    print(f"[INFO] Loading frozen teacher checkpoint from: {teacher_checkpoint}", flush=True)
    return BaseImuDaggerAdapterPolicy(
        teacher=teacher,
        adapter=adapter,
        history_normalizer=history_normalizer,
        teacher_obs_dim=teacher_obs_dim,
        history_dim=history_dim,
        device=device,
    )


def _build_command_window(
    cmd_state: LiveCommandState,
    vx_range,
    vy_range,
    wz_range,
    title: str,
    camera_state: CameraModeState | None = None,
):
    import omni.ui as ui

    vx_min, vx_max = map(float, vx_range)
    vy_min, vy_max = map(float, vy_range)
    wz_min, wz_max = map(float, wz_range)

    win_height = 305 if camera_state is not None else 255
    win = ui.Window(title, width=440, height=win_height, visible=True)
    keepalive = []

    with win.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Solo12 live command", height=22)
            ui.Label("Direct env command is directly (vx, vy, wz).", height=18)

            def add_row(label: str, getter, setter, vmin: float, vmax: float):
                with ui.HStack(spacing=8, height=28):
                    ui.Label(label, width=90)
                    model = ui.SimpleFloatModel(getter(), min=vmin, max=vmax)
                    ui.FloatField(model, width=120)
                    ui.FloatSlider(model, min=vmin, max=vmax, width=180)

                    def _on_change(m):
                        setter(m.get_value_as_float())

                    if hasattr(model, "subscribe_value_changed_fn"):
                        keepalive.append(model.subscribe_value_changed_fn(_on_change))
                    else:
                        model.add_value_changed_fn(_on_change)
                        keepalive.append(_on_change)

            add_row("lin vel x [m/s]", lambda: cmd_state.get()[0], cmd_state.set_vx, vx_min, vx_max)
            add_row("lin vel y [m/s]", lambda: cmd_state.get()[1], cmd_state.set_vy, vy_min, vy_max)
            add_row("ang vel z [rad/s]", lambda: cmd_state.get()[2], cmd_state.set_wz, wz_min, wz_max)
            with ui.HStack(spacing=8, height=30):
                ui.Label("manual reset", width=90)
                ui.Button("Reset env", width=120, clicked_fn=cmd_state.request_reset)
                ui.Label("Reset all active envs now.", height=18)
            if camera_state is not None:
                camera_mode_model = ui.SimpleStringModel(f"Camera: {camera_state.label()}")

                def _toggle_camera():
                    camera_state.toggle()
                    camera_mode_model.set_value(f"Camera: {camera_state.label()}")

                with ui.HStack(spacing=8, height=30):
                    ui.Label("camera", width=90)
                    ui.Button("Next camera", width=150, clicked_fn=_toggle_camera)
                    ui.StringField(camera_mode_model, read_only=True, height=26)
                ui.Label("Modes cycle through chase, follow, and free.", height=18)
            ui.Label("Tip: click a number to type an exact value and press Enter.", height=18)

    return win, keepalive


def _build_body_force_window(force_state: BodyFrameForceState, max_magnitude: float, point_half_extents):
    import omni.ui as ui

    max_magnitude = max(1.0e-6, float(max_magnitude))
    keepalive = []

    selected_model = ui.SimpleStringModel("")
    active_model = ui.SimpleStringModel("Active force: none")

    def _selected_text() -> str:
        fx, fy, fz, px, py, pz, magnitude, azimuth, elevation = force_state.get_selected()
        return f"Selected: {magnitude:.1f} N, F_b=[{fx:.1f},{fy:.1f},{fz:.1f}], p_b=[{px:.3f},{py:.3f},{pz:.3f}] m"

    def _active_text() -> str:
        active = force_state.get_active_force()
        if active is None:
            return "Active force: none"
        fx, fy, fz, px, py, pz, _azimuth, _elevation = active
        magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
        return f"Active: {magnitude:.1f} N, F_b=[{fx:.1f},{fy:.1f},{fz:.1f}], p_b=[{px:.3f},{py:.3f},{pz:.3f}] m"

    def _refresh_selected():
        selected_model.set_value(_selected_text())

    def _pulse():
        force_state.apply_selected(continuous=False)
        active_model.set_value(_active_text())

    def _hold():
        force_state.apply_selected(continuous=True)
        active_model.set_value(_active_text())

    def _clear():
        force_state.request_clear()
        active_model.set_value("Active force: none")

    half_x, half_y, half_z = map(float, point_half_extents)
    win = ui.Window("Solo12 point-force robustness control", width=590, height=510, visible=True)
    with win.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Point force in the moving Solo12 body frame", height=22)
            ui.Label("Direction: azimuth 0 = +x, 90 = +y; elevation +90 = +z.", height=18)

            magnitude_model = ui.SimpleFloatModel(
                max(0.0, float(force_state.selected_magnitude)), min=0.0, max=max_magnitude
            )
            angle_model = ui.SimpleFloatModel(float(force_state.selected_angle_deg), min=-180.0, max=180.0)
            elevation_model = ui.SimpleFloatModel(float(force_state.selected_elevation_deg), min=-90.0, max=90.0)
            duration_model = ui.SimpleFloatModel(float(force_state.selected_duration_s), min=force_state.dt, max=3.0)
            point_models = [
                ui.SimpleFloatModel(float(force_state.selected_point_b[0]), min=-half_x, max=half_x),
                ui.SimpleFloatModel(float(force_state.selected_point_b[1]), min=-half_y, max=half_y),
                ui.SimpleFloatModel(float(force_state.selected_point_b[2]), min=-half_z, max=half_z),
            ]

            with ui.HStack(spacing=8, height=30):
                ui.Label("magnitude [N]", width=105)
                ui.FloatField(magnitude_model, width=110)
                ui.FloatSlider(magnitude_model, min=0.0, max=max_magnitude, width=270)
            with ui.HStack(spacing=8, height=30):
                ui.Label("azimuth [deg]", width=105)
                ui.FloatField(angle_model, width=110)
                ui.FloatSlider(angle_model, min=-180.0, max=180.0, width=270)
            with ui.HStack(spacing=8, height=30):
                ui.Label("elevation [deg]", width=105)
                ui.FloatField(elevation_model, width=110)
                ui.FloatSlider(elevation_model, min=-90.0, max=90.0, width=270)
            with ui.HStack(spacing=8, height=30):
                ui.Label("pulse time [s]", width=105)
                ui.FloatField(duration_model, width=110)
                ui.FloatSlider(duration_model, min=force_state.dt, max=3.0, width=270)

            ui.Label("Application point p_b [m] (box surface presets below)", height=20)
            for axis, (label, model, limit) in enumerate(zip(("point x", "point y", "point z"), point_models, (half_x, half_y, half_z))):
                with ui.HStack(spacing=8, height=28):
                    ui.Label(label, width=105)
                    ui.FloatField(model, width=110)
                    ui.FloatSlider(model, min=-limit, max=limit, width=270)

            def _on_magnitude_change(model):
                force_state.set_magnitude(model.get_value_as_float())
                _refresh_selected()

            def _on_angle_change(model):
                force_state.set_angle_deg(model.get_value_as_float())
                _refresh_selected()

            def _on_elevation_change(model):
                force_state.set_elevation_deg(model.get_value_as_float())
                _refresh_selected()

            def _on_duration_change(model):
                force_state.set_duration_s(model.get_value_as_float())

            def _point_callback(axis):
                def callback(model):
                    force_state.set_point_axis(axis, model.get_value_as_float())
                    _refresh_selected()
                return callback

            callbacks = [(magnitude_model, _on_magnitude_change), (angle_model, _on_angle_change),
                         (elevation_model, _on_elevation_change), (duration_model, _on_duration_change)]
            callbacks.extend((model, _point_callback(axis)) for axis, model in enumerate(point_models))
            for model, callback in callbacks:
                if hasattr(model, "subscribe_value_changed_fn"):
                    keepalive.append(model.subscribe_value_changed_fn(callback))
                else:
                    model.add_value_changed_fn(callback)
                    keepalive.append(callback)

            with ui.HStack(spacing=8, height=34):
                ui.Button("Pulse", clicked_fn=_pulse)
                ui.Button("Hold", clicked_fn=_hold)
                ui.Button("Release", clicked_fn=_clear)
                ui.Button("+X", width=48, clicked_fn=lambda: angle_model.set_value(0.0))
                ui.Button("+Y", width=48, clicked_fn=lambda: angle_model.set_value(90.0))
                ui.Button("-X", width=48, clicked_fn=lambda: angle_model.set_value(180.0))
                ui.Button("-Y", width=48, clicked_fn=lambda: angle_model.set_value(-90.0))

            def _set_point(point):
                for model, value in zip(point_models, point):
                    model.set_value(float(value))

            with ui.HStack(spacing=5, height=30):
                ui.Label("surface", width=65)
                ui.Button("front", clicked_fn=lambda: _set_point((half_x, 0, 0)))
                ui.Button("rear", clicked_fn=lambda: _set_point((-half_x, 0, 0)))
                ui.Button("left", clicked_fn=lambda: _set_point((0, half_y, 0)))
                ui.Button("right", clicked_fn=lambda: _set_point((0, -half_y, 0)))
                ui.Button("top", clicked_fn=lambda: _set_point((0, 0, half_z)))
                ui.Button("center", clicked_fn=lambda: _set_point((0, 0, 0)))

            ui.StringField(selected_model, read_only=True, height=26)
            ui.StringField(active_model, read_only=True, height=26)
            ui.Label("Pulse applies for the selected simulation time; Hold stays active until Release.", height=18)

    _refresh_selected()
    return win, keepalive, selected_model, active_model


def _apply_live_direct_command(raw_env, cmd_state: LiveCommandState, vx_limits, vy_limits, wz_limits, env_ids=slice(None)):
    vx, vy, wz = cmd_state.get()
    vx = float(np.clip(vx, *map(float, vx_limits)))
    vy = float(np.clip(vy, *map(float, vy_limits)))
    wz = float(np.clip(wz, *map(float, wz_limits)))
    raw_env._commands[env_ids, 0] = vx
    raw_env._commands[env_ids, 1] = vy
    raw_env._commands[env_ids, 2] = wz
    if hasattr(raw_env, "_command_steps_left"):
        raw_env._command_steps_left[env_ids] = max(1, int(1e9))


def _resolve_base_wrench_body_ids(raw_env) -> list[int]:
    if hasattr(raw_env, "_base_wrench_body_ids"):
        return list(raw_env._base_wrench_body_ids)
    body_ids, _ = raw_env._robot.find_bodies("base")
    return list(body_ids)


def _set_manual_body_frame_base_force(raw_env, force_b):
    fx_b, fy_b, fz_b, px_b, py_b, pz_b, _azimuth_deg, _elevation_deg = force_b
    body_ids = _resolve_base_wrench_body_ids(raw_env)
    forces = torch.zeros((raw_env.num_envs, len(body_ids), 3), device=raw_env.device)
    forces[..., 0] = float(fx_b)
    forces[..., 1] = float(fy_b)
    forces[..., 2] = float(fz_b)

    application_points = None
    if hasattr(raw_env, "_base_push_application_points_b"):
        application_points = torch.empty_like(forces)
        application_points[..., 0] = float(px_b)
        application_points[..., 1] = float(py_b)
        application_points[..., 2] = float(pz_b)
        raw_env._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            positions=application_points,
            body_ids=body_ids,
            is_global=False,
        )
    else:
        raw_env._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=body_ids,
            is_global=False,
        )

    if hasattr(raw_env, "_base_push_forces_b") and raw_env._base_push_forces_b.shape == forces.shape:
        raw_env._base_push_forces_b[:] = forces
        if application_points is not None:
            raw_env._base_push_application_points_b[:] = application_points
        else:
            raw_env._base_push_torques_b[:] = 0.0
        raw_env._base_push_steps_left[:] = 2**30
        raw_env._base_push_steps_until_next[:] = 2**30


def _clear_manual_body_frame_base_force(raw_env):
    if hasattr(raw_env, "_base_push_forces_b"):
        raw_env._base_push_forces_b[:] = 0.0
        if hasattr(raw_env, "_base_push_application_points_b"):
            raw_env._base_push_application_points_b[:] = 0.0
        else:
            raw_env._base_push_torques_b[:] = 0.0
        raw_env._base_push_steps_left[:] = 0
        raw_env._base_push_steps_until_next[:] = 2**30
    raw_env._robot.permanent_wrench_composer.reset()


def _sync_manual_body_frame_base_force(raw_env, force_state: BodyFrameForceState | None):
    if force_state is None:
        return None
    if force_state.consume_clear_request():
        _clear_manual_body_frame_base_force(raw_env)
    active_force = force_state.get_active_force()
    if active_force is not None:
        _set_manual_body_frame_base_force(raw_env, active_force)
    return active_force


def _infer_force_ui_max_magnitude(env_cfg) -> float:
    candidates = []
    if hasattr(env_cfg, "base_push_force_xy_range"):
        candidates.extend(abs(float(value)) for value in env_cfg.base_push_force_xy_range)
    if hasattr(env_cfg, "forces_applied_to_base_curriculum"):
        candidates.extend(abs(float(value)) for value in env_cfg.forces_applied_to_base_curriculum)
    return max(candidates) if candidates else 1.0


def _disable_training_stochasticity_for_play(env_cfg) -> list[str]:
    disabled = []

    if getattr(env_cfg, "events", None):
        env_cfg.events = None
        disabled.append("events")

    if hasattr(env_cfg, "enable_observation_corruption") and env_cfg.enable_observation_corruption:
        env_cfg.enable_observation_corruption = False
        disabled.append("observation_corruption")

    if hasattr(env_cfg, "reset_base_lin_vel_range"):
        env_cfg.reset_base_lin_vel_range = (0.0, 0.0)
        disabled.append("reset_base_lin_vel_range")

    if hasattr(env_cfg, "reset_base_ang_vel_range"):
        env_cfg.reset_base_ang_vel_range = (0.0, 0.0)
        disabled.append("reset_base_ang_vel_range")

    if hasattr(env_cfg, "flexed_initial_joint_pos_noise_range"):
        env_cfg.flexed_initial_joint_pos_noise_range = (0.0, 0.0)
        disabled.append("flexed_initial_joint_pos_noise_range")

    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
        disabled.append("actuation_delay_range")

    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)
        disabled.append("base_push_interval_range_s")

    if hasattr(env_cfg, "forces_applied_to_base_curriculum") and env_cfg.forces_applied_to_base_curriculum:
        env_cfg.forces_applied_to_base_curriculum = []
        disabled.append("forces_applied_to_base_curriculum")

    if hasattr(env_cfg, "max_velx_range_curriculum") and env_cfg.max_velx_range_curriculum:
        env_cfg.max_velx_range_curriculum = []
        disabled.append("max_velx_range_curriculum")

    return disabled


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


def _manual_reset_all_envs(raw_env, vec_env):
    # During play we step the env under torch.inference_mode(), which means some env buffers
    # (for example raw_env._actions) become inference tensors. Resetting those buffers must
    # happen under inference_mode as well, otherwise PyTorch raises on in-place updates.
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


def _get_tracked_base_velocity(raw_env):
    """Return the planar velocity and yaw rate in the frame used by the environment rewards."""
    if raw_env.cfg.track_commands_in_world_heading_frame:
        lin_vel_xy = _world_velocity_in_heading_frame_xy(
            raw_env._robot.data.root_lin_vel_w, raw_env._robot.data.root_quat_w
        )
        yaw_rate = raw_env._robot.data.root_ang_vel_w[:, 2]
    else:
        lin_vel_xy = raw_env._robot.data.root_lin_vel_b[:, :2]
        yaw_rate = raw_env._robot.data.root_ang_vel_b[:, 2]
    return lin_vel_xy, yaw_rate


def _compute_reward_terms(raw_env):
    commands = raw_env._commands
    robot = raw_env._robot
    joint_ids = raw_env._joint_ids

    tracked_lin_vel_xy, tracked_yaw_rate = _get_tracked_base_velocity(raw_env)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - tracked_lin_vel_xy), dim=1)
    lin_vel_xy_error_norm = torch.linalg.vector_norm(commands[:, :2] - tracked_lin_vel_xy, dim=1)
    ang_vel_z_error_abs = torch.abs(commands[:, 2] - tracked_yaw_rate)
    force_transmited_through_joints = raw_env._compute_force_transmited_through_joints()

    track_lin_vel_xy_exp = (
        torch.exp(-lin_vel_error / raw_env.cfg.tracking_std**2)
        * raw_env.cfg.track_lin_vel_xy_reward_scale
        * raw_env.step_dt
    )
    force_reward = (
        force_transmited_through_joints
        * raw_env.cfg.force_transmited_through_joints_reward_scale
        * raw_env.step_dt
    )
    ratio = force_reward / torch.clamp(track_lin_vel_xy_exp, min=1.0e-8)

    joint_torque = torch.sum(torch.square(robot.data.applied_torque[:, joint_ids]), dim=1)

    return {
        "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
        "lin_vel_xy_error_norm": lin_vel_xy_error_norm,
        "ang_vel_z_error_abs": ang_vel_z_error_abs,
        "force_transmited_through_joints": force_reward,
        "force_transmited_through_joints_raw": force_transmited_through_joints,
        "ratio": ratio,
        "joint_torque_sq": joint_torque,
    }


def _save_command_tracking_plots(
    output_dir: Path,
    times_s: list[float],
    commands: list[tuple[float, float, float]],
    tracked_lin_vel_xy: list[tuple[float, float]],
    tracked_yaw_rate: list[float],
    joint_names: list[str],
    joint_positions: list[list[float]],
    joint_position_targets: list[list[float]],
    joint_physical_position_limits: np.ndarray,
    joint_soft_position_limits: np.ndarray,
) -> list[Path]:
    """Save command-tracking plots with physical and task soft joint limits."""
    import csv

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times_s)
    command_samples = np.asarray(
        [commands[min(int(max(t - 1.0e-9, 0.0) / TRACKING_COMMAND_DURATION_S), len(commands) - 1)] for t in times]
    )
    lin_vel = np.asarray(tracked_lin_vel_xy)
    yaw_rate = np.asarray(tracked_yaw_rate)
    lin_error = np.linalg.norm(command_samples[:, :2] - lin_vel, axis=1)
    yaw_error = np.abs(command_samples[:, 2] - yaw_rate)
    signed_xy_error = lin_vel - command_samples[:, :2]
    signed_wz_error = yaw_rate - command_samples[:, 2]
    command_norm = np.linalg.norm(command_samples[:, :2], axis=1)
    along_command_error = np.divide(
        np.sum(signed_xy_error * command_samples[:, :2], axis=1), command_norm,
        out=np.zeros_like(command_norm), where=command_norm > 1.0e-9,
    )

    csv_path = output_dir / "command_tracking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_s", "cmd_vx", "cmd_vy", "cmd_wz", "velocity_vx", "velocity_vy", "yaw_rate_wz",
             "vxy_error_norm", "wz_error_abs", "vx_error_signed", "vy_error_signed", "wz_error_signed",
             "vxy_error_along_command"]
        )
        writer.writerows(np.column_stack((times, command_samples, lin_vel, yaw_rate, lin_error, yaw_error,
                                          signed_xy_error, signed_wz_error, along_command_error)))

    def decorate_cells(ax):
        for boundary in np.arange(1, len(commands)) * TRACKING_COMMAND_DURATION_S:
            ax.axvline(boundary, color="0.25", linewidth=1.0, alpha=0.65)
        ax.set_xlim(0.0, len(commands) * TRACKING_COMMAND_DURATION_S)
        ax.grid(axis="y", alpha=0.22)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.8), constrained_layout=True)
    ax.plot(times, lin_error, color="#e69f00", linewidth=1.7)
    decorate_cells(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\|\Delta v_{xy}\|$ [m/s]")
    ax.set_title("Planar command-tracking error (base-footprint frame)")
    y0, y1 = ax.get_ylim()
    y_span = max(y1 - y0, 1.0e-6)
    max_cmd = max(max(math.hypot(vx, vy) for vx, vy, _ in commands), 1.0e-6)
    for index, (vx, vy, _) in enumerate(commands):
        center_t = (index + 0.5) * TRACKING_COMMAND_DURATION_S
        scale = 0.18 / max_cmd
        # Match the requested footprint glyph: +vx points up and +vy points left.
        dx = -vy * TRACKING_COMMAND_DURATION_S * scale
        dy = vx * y_span * scale
        center_y = y0 + 0.82 * y_span
        ax.annotate(
            "",
            xy=(center_t + dx, center_y + dy),
            xytext=(center_t - dx, center_y - dy),
            arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.8, "alpha": 0.65},
        )
        ax.text(center_t, y0 + 0.97 * y_span, f"({vx:g}, {vy:g})", ha="center", va="top", fontsize=8, color="#a51f1f")
    lin_path = output_dir / "vxy_tracking_error.png"
    fig.savefig(lin_path, dpi=180)
    plt.close(fig)

    paths = [csv_path, lin_path]

    def save_signed_error_plot(series, ylabel, title, filename, color):
        fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.4), constrained_layout=True)
        ax.plot(times, series, color=color, linewidth=1.5)
        ax.axhline(0.0, color="0.2", linewidth=1.0)
        decorate_cells(ax)
        ax.set(xlabel="Time [s]", ylabel=ylabel, title=title)
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    save_signed_error_plot(signed_xy_error[:, 0], r"$v_x-v_x^{cmd}$ [m/s]",
                           "Signed x-velocity tracking error (base-footprint frame)", "vx_tracking_error_signed.png", "#d55e00")
    save_signed_error_plot(signed_xy_error[:, 1], r"$v_y-v_y^{cmd}$ [m/s]",
                           "Signed y-velocity tracking error (base-footprint frame)", "vy_tracking_error_signed.png", "#009e73")
    save_signed_error_plot(along_command_error, r"$(v_{xy}-v_{xy}^{cmd})\cdot\hat{v}_{xy}^{cmd}$ [m/s]",
                           "Planar tracking error projected along command direction", "vxy_tracking_error_along_command.png", "#cc79a7")
    save_signed_error_plot(signed_wz_error, r"$\omega_z-\omega_z^{cmd}$ [rad/s]",
                           "Signed yaw-rate tracking error (world z)", "wz_tracking_error_signed.png", "#0072b2")
    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.4), constrained_layout=True)
    ax.plot(times, yaw_rate, label=r"$\omega_z$", color="#0072b2", linewidth=1.5)
    ax.plot(times, command_samples[:, 2], label=r"$\omega_z^{cmd}$", color="#d55e00", linewidth=1.3, linestyle="--")
    decorate_cells(ax)
    ax.legend()
    ax.set(xlabel="Time [s]", ylabel="Yaw rate [rad/s]", title="Measured and commanded yaw rate (world z)")
    wz_compare_path = output_dir / "wz_tracking_actual_vs_command.png"
    fig.savefig(wz_compare_path, dpi=180)
    plt.close(fig)
    paths.append(wz_compare_path)
    if any(abs(wz) > 1.0e-9 for _, _, wz in commands):
        fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.8), constrained_layout=True)
        ax.plot(times, yaw_error, color="#0072b2", linewidth=1.7)
        decorate_cells(ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"$|\Delta \omega_z|$ [rad/s]")
        ax.set_title("Yaw-rate command-tracking error (world vertical axis)")
        y0, y1 = ax.get_ylim()
        y_span = max(y1 - y0, 1.0e-6)
        for index, (_, _, wz) in enumerate(commands):
            center_t = (index + 0.5) * TRACKING_COMMAND_DURATION_S
            glyph = "↶" if wz > 0 else "↷" if wz < 0 else "·"
            ax.text(center_t, y0 + 0.86 * y_span, glyph, ha="center", va="center", fontsize=22, color="#d62728", alpha=0.65)
            ax.text(center_t, y0 + 0.97 * y_span, f"{wz:g} rad/s", ha="center", va="top", fontsize=8, color="#a51f1f")
        yaw_path = output_dir / "wz_tracking_error.png"
        fig.savefig(yaw_path, dpi=180)
        plt.close(fig)
        paths.append(yaw_path)

    q = np.asarray(joint_positions)
    q_des = np.asarray(joint_position_targets)
    physical_limits = np.asarray(joint_physical_position_limits)
    soft_limits = np.asarray(joint_soft_position_limits)
    if q.shape != q_des.shape or q.shape != (len(times), len(joint_names)):
        raise ValueError(
            f"Joint telemetry shapes must be ({len(times)}, {len(joint_names)}), "
            f"got q={q.shape} and q_des={q_des.shape}."
        )
    expected_limit_shape = (len(joint_names), 2)
    if physical_limits.shape != expected_limit_shape:
        raise ValueError(
            f"Physical joint limits must have shape {expected_limit_shape}, got {physical_limits.shape}."
        )
    if soft_limits.shape != expected_limit_shape:
        raise ValueError(f"Soft joint limits must have shape {expected_limit_shape}, got {soft_limits.shape}.")

    side_groups = {
        "left": [index for index, name in enumerate(joint_names) if name.startswith(("FL_", "RL_"))],
        "right": [index for index, name in enumerate(joint_names) if name.startswith(("FR_", "RR_"))],
    }
    if any(not indices for indices in side_groups.values()):
        midpoint = (len(joint_names) + 1) // 2
        side_groups = {"first_half": list(range(midpoint)), "second_half": list(range(midpoint, len(joint_names)))}

    for side, joint_indices in side_groups.items():
        fig, axes = plt.subplots(
            len(joint_indices),
            1,
            figsize=(max(10.0, 1.45 * len(commands)), 2.15 * len(joint_indices)),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, joint_index in zip(axes, joint_indices):
            ax.plot(times, q[:, joint_index], label=r"$q$", color="#0072b2", linewidth=1.25)
            ax.plot(times, q_des[:, joint_index], label=r"$q_{des}$", color="#d55e00", linewidth=1.1, alpha=0.9)
            ax.axhline(
                soft_limits[joint_index, 0],
                label="soft lower",
                color="#cc79a7",
                linestyle="--",
                linewidth=1.15,
            )
            ax.axhline(
                soft_limits[joint_index, 1],
                label="soft upper",
                color="#009e73",
                linestyle="--",
                linewidth=1.15,
            )
            ax.axhline(
                physical_limits[joint_index, 0],
                label="hard lower",
                color="0.45",
                linestyle=":",
                linewidth=0.9,
            )
            ax.axhline(
                physical_limits[joint_index, 1],
                label="hard upper",
                color="0.45",
                linestyle=":",
                linewidth=0.9,
            )
            decorate_cells(ax)
            ax.set_ylabel("rad")
            ax.set_title(joint_names[joint_index], loc="left", fontsize=10)
        axes[0].legend(ncol=3, loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(
            f"Joint position and executed target vs soft/hard limits — {side.replace('_', ' ')}",
            fontsize=13,
        )
        joint_path = output_dir / f"joint_position_vs_limits_{side}.png"
        fig.savefig(joint_path, dpi=180)
        plt.close(fig)
        paths.append(joint_path)
    return paths


def _summarize_command_tracking(
    times_s: list[float],
    commands: list[tuple[float, float, float]],
    tracked_lin_vel_xy: list[tuple[float, float]],
    tracked_yaw_rate: list[float],
    resets: int,
) -> dict[str, float | int]:
    """Return the simulator-independent tracking statistics shared with the MuJoCo runner."""
    times = np.asarray(times_s)
    command_samples = np.asarray(
        [commands[min(int(max(t - 1.0e-9, 0.0) / TRACKING_COMMAND_DURATION_S), len(commands) - 1)] for t in times]
    )
    lin_vel = np.asarray(tracked_lin_vel_xy)
    yaw_rate = np.asarray(tracked_yaw_rate)
    vxy_error = np.linalg.norm(command_samples[:, :2] - lin_vel, axis=1)
    wz_error = np.abs(command_samples[:, 2] - yaw_rate)
    signed_xy_error = lin_vel - command_samples[:, :2]
    signed_wz_error = yaw_rate - command_samples[:, 2]
    command_norm = np.linalg.norm(command_samples[:, :2], axis=1)
    along_command_error = np.divide(
        np.sum(signed_xy_error * command_samples[:, :2], axis=1), command_norm,
        out=np.zeros_like(command_norm), where=command_norm > 1.0e-9,
    )

    def distribution(prefix: str, values: np.ndarray, unit: str) -> dict[str, float]:
        return {
            f"{prefix}_mean_{unit}": float(values.mean()),
            f"{prefix}_median_{unit}": float(np.median(values)),
            f"{prefix}_p05_{unit}": float(np.percentile(values, 5)),
            f"{prefix}_p95_{unit}": float(np.percentile(values, 95)),
            f"{prefix}_std_{unit}": float(values.std()),
            f"{prefix}_max_{unit}": float(values.max()),
        }

    return {
        "samples": len(times),
        "duration_s": float(times[-1]),
        "resets": int(resets),
        "failures": int(resets),
        **distribution("vxy_error", vxy_error, "mps"),
        **distribution("wz_error", wz_error, "radps"),
        **distribution("vx_error_signed", signed_xy_error[:, 0], "mps"),
        **distribution("vy_error_signed", signed_xy_error[:, 1], "mps"),
        **distribution("wz_error_signed", signed_wz_error, "radps"),
        **distribution("vxy_error_along_command", along_command_error, "mps"),
    }


def _log_command_tracking_csv_to_wandb(run, csv_path: Path) -> int:
    """Log Isaac command tracking using the same scalar schema as the MuJoCo evaluator."""
    import csv

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return 0
    metric_names = [
        "tracking/command_vx_mps", "tracking/command_vy_mps", "tracking/command_wz_radps",
        "tracking/velocity_vx_mps", "tracking/velocity_vy_mps", "tracking/velocity_wz_radps",
        "tracking/error_vx_abs_mps", "tracking/error_vy_abs_mps", "tracking/error_vxy_norm_mps",
        "tracking/error_wz_abs_radps", "tracking/error_vx_signed_mps", "tracking/error_vy_signed_mps",
        "tracking/error_wz_signed_radps", "tracking/error_vxy_along_command_mps",
    ]
    run.define_metric("tracking/time_s")
    for name in metric_names:
        run.define_metric(name, step_metric="tracking/time_s")
    for row in rows:
        cmd_vx, cmd_vy, cmd_wz = (float(row[key]) for key in ("cmd_vx", "cmd_vy", "cmd_wz"))
        vx, vy, wz = (float(row[key]) for key in ("velocity_vx", "velocity_vy", "yaw_rate_wz"))
        vx_error, vy_error, wz_error = vx - cmd_vx, vy - cmd_vy, wz - cmd_wz
        command_norm = math.hypot(cmd_vx, cmd_vy)
        run.log({
            "tracking/time_s": float(row["time_s"]),
            "tracking/command_vx_mps": cmd_vx, "tracking/command_vy_mps": cmd_vy,
            "tracking/command_wz_radps": cmd_wz, "tracking/velocity_vx_mps": vx,
            "tracking/velocity_vy_mps": vy, "tracking/velocity_wz_radps": wz,
            "tracking/error_vx_abs_mps": abs(vx_error), "tracking/error_vy_abs_mps": abs(vy_error),
            "tracking/error_vxy_norm_mps": float(row["vxy_error_norm"]),
            "tracking/error_wz_abs_radps": float(row["wz_error_abs"]),
            "tracking/error_vx_signed_mps": vx_error, "tracking/error_vy_signed_mps": vy_error,
            "tracking/error_wz_signed_radps": wz_error,
            "tracking/error_vxy_along_command_mps": ((vx_error * cmd_vx + vy_error * cmd_vy) / command_norm
                                                       if command_norm > 1.0e-9 else 0.0),
        })
    return len(rows)


def _save_and_upload_tracking_metadata(
    *,
    output_dir: Path,
    artifact_paths: list[Path],
    summary: dict[str, float | int],
    resume_path: str,
    env_cfg,
    agent_cfg,
    dt: float,
) -> tuple[Path, Path, str | None]:
    """Persist complete Isaac tracking metadata and optionally upload it to the shared W&B project."""
    config = {
        "task": args_cli.task,
        "simulator": "isaac_sim",
        "checkpoint": str(Path(resume_path).resolve()),
        "commands": [list(command) for command in TRACKING_COMMANDS],
        "command_duration_s": TRACKING_COMMAND_DURATION_S,
        "duration_s": float(summary["duration_s"]),
        "episode_length_s": float(env_cfg.episode_length_s),
        "num_envs": int(args_cli.num_envs),
        "policy_dt": float(dt),
        "kp": float(env_cfg.kp),
        "kd": float(env_cfg.kd),
        "env_config": env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else str(env_cfg),
        "agent_config": agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else str(agent_cfg),
    }
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    run_url = None
    if not args_cli.no_tracking_wandb:
        try:
            run_name = args_cli.tracking_wandb_name or f"isaac-sim-{Path(resume_path).stem}-{output_dir.name[:15]}"
            tracking_run = wandb.init(
                project=args_cli.tracking_wandb_project,
                entity=args_cli.tracking_wandb_entity,
                name=run_name,
                config={key: value for key, value in config.items() if key not in {"env_config", "agent_config"}},
                job_type="sim-to-sim-eval",
            )
            tracking_run.summary.update(summary)
            csv_path = next(path for path in artifact_paths if path.name == "command_tracking.csv")
            _log_command_tracking_csv_to_wandb(tracking_run, csv_path)
            plot_paths = [path for path in artifact_paths if path.suffix == ".png"]
            if plot_paths:
                tracking_run.log({f"plots/{path.stem}": wandb.Image(str(path)) for path in plot_paths})
            artifact = wandb.Artifact(
                f"isaac-sim-{Path(resume_path).stem}-{output_dir.name[:15]}",
                type="sim-to-sim-evaluation",
                metadata=summary,
            )
            for path in [config_path, summary_path, *artifact_paths]:
                artifact.add_file(str(path), name=path.name)
            tracking_run.log_artifact(artifact)
            run_url = tracking_run.url
            tracking_run.finish()
        except Exception as exc:
            print(f"[WARN] Tracking W&B upload failed; local artifacts are complete: {exc}", flush=True)
    return config_path, summary_path, run_url


class ChaseCamera:
    """Viewport camera that follows a smoothed horizontal robot heading.

    Heading is measured from the projected body-y axis instead of Euler yaw.  The
    lateral axis remains horizontal while Solo12 stands on two feet, avoiding the
    +/-180 degree Euler-yaw flips that occur when body pitch approaches 90 degrees.
    """

    def __init__(self, raw_env, env_index: int = 0, eye_b=(-2.6, 0.0, 1.0), lookat_b=(0.45, 0.0, 0.35)):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.eye_b = np.asarray(eye_b, dtype=np.float64)
        self.lookat_b = np.asarray(lookat_b, dtype=np.float64)
        self.asset = raw_env._robot
        self._anchor_w = None
        self._heading = None
        self._smoothing_time_s = 0.20

    @staticmethod
    def _stable_heading_from_quat(quat_wxyz: torch.Tensor, previous: float | None = None) -> float:
        w, x, y, z = [float(v) for v in quat_wxyz]
        # Horizontal projection of body +y. Unlike Euler yaw, this remains
        # well-conditioned when the robot's body +x points nearly vertically.
        lateral_x = 2.0 * (x * y - w * z)
        lateral_y = 1.0 - 2.0 * (x * x + z * z)
        if math.hypot(lateral_x, lateral_y) > 1.0e-4:
            heading = math.atan2(lateral_y, lateral_x) - 0.5 * math.pi
        else:
            # Rare fallback for a near-vertical lateral axis.
            forward_x = 1.0 - 2.0 * (y * y + z * z)
            forward_y = 2.0 * (x * y + w * z)
            if math.hypot(forward_x, forward_y) <= 1.0e-4:
                return 0.0 if previous is None else previous
            heading = math.atan2(forward_y, forward_x)

        if previous is not None:
            heading = previous + math.atan2(math.sin(heading - previous), math.cos(heading - previous))
        return heading

    @staticmethod
    def _yaw_rot(yaw: float) -> np.ndarray:
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def update(self):
        root_pos_w = self.asset.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
        root_quat_w = self.asset.data.root_quat_w[self.env_index].detach().cpu()
        heading = self._stable_heading_from_quat(root_quat_w, self._heading)

        dt = float(getattr(self.raw_env, "step_dt", 1.0 / 60.0))
        alpha = 1.0 - math.exp(-max(dt, 1.0e-6) / self._smoothing_time_s)
        if self._anchor_w is None:
            self._anchor_w = root_pos_w
            self._heading = heading
        else:
            self._anchor_w = self._anchor_w + alpha * (root_pos_w - self._anchor_w)
            self._heading = self._heading + alpha * (heading - self._heading)

        rot = self._yaw_rot(self._heading)

        eye_w = self._anchor_w + rot @ self.eye_b
        lookat_w = self._anchor_w + rot @ self.lookat_b
        self.raw_env.sim.set_camera_view(eye=eye_w, target=lookat_w)


class SideFollowCamera(ChaseCamera):
    """Viewport camera that tracks the robot root from the side in the robot yaw frame."""

    def __init__(self, raw_env, env_index: int = 0, eye_b=(0.0, -2.6, 1.1), lookat_b=(0.0, 0.0, 0.35)):
        super().__init__(raw_env, env_index=env_index, eye_b=eye_b, lookat_b=lookat_b)


def _update_active_camera(cameras: dict[str, ChaseCamera] | None, camera_state: CameraModeState | None):
    if cameras is None or camera_state is None:
        return
    active_mode = camera_state.scripted_mode()
    if active_mode is None:
        return
    camera = cameras.get(active_mode)
    if camera is not None:
        camera.update()


class BodyForceArrowVisualizer:
    """Best-effort viewport arrow for the active body-frame base force."""

    def __init__(self, raw_env, env_index: int = 0):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.asset = raw_env._robot
        self._warned = False
        try:
            try:
                from isaacsim.core.utils.extensions import enable_extension

                enable_extension("isaacsim.util.debug_draw")
            except Exception:
                pass
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._draw = None

    @staticmethod
    def _quat_apply_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        q_w = quat_wxyz[0]
        q_xyz = quat_wxyz[1:4]
        uv = np.cross(q_xyz, vec)
        uuv = np.cross(q_xyz, uv)
        return vec + 2.0 * (q_w * uv + uuv)

    def clear(self):
        if self._draw is not None:
            try:
                self._draw.clear_lines()
            except Exception:
                pass

    def update(self, active_force):
        if self._draw is None:
            return
        self.clear()
        if active_force is None:
            return

        fx_b, fy_b, fz_b, px_b, py_b, pz_b, _azimuth_deg, _elevation_deg = active_force
        force_b = np.array([fx_b, fy_b, fz_b], dtype=np.float64)
        magnitude = float(np.linalg.norm(force_b))
        if magnitude <= 1.0e-6:
            return

        try:
            root_pos_w = self.asset.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
            root_quat_w = self.asset.data.root_quat_w[self.env_index].detach().cpu().numpy().astype(np.float64)
            direction_w = self._quat_apply_wxyz(root_quat_w, force_b / magnitude)
            direction_w /= max(float(np.linalg.norm(direction_w)), 1.0e-6)
            point_b = np.array([px_b, py_b, pz_b], dtype=np.float64)
            start = root_pos_w + self._quat_apply_wxyz(root_quat_w, point_b)
            arrow_length = min(1.75, 0.25 + 0.05 * magnitude)
            end = start + arrow_length * direction_w
            side = np.cross(direction_w, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            if np.linalg.norm(side) < 1.0e-6:
                side = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            side /= max(float(np.linalg.norm(side)), 1.0e-6)
            head_len = min(0.25, 0.25 * arrow_length)
            head_left = end - head_len * direction_w + 0.45 * head_len * side
            head_right = end - head_len * direction_w - 0.45 * head_len * side
            starts = [tuple(start.tolist()), tuple(head_left.tolist()), tuple(head_right.tolist())]
            ends = [tuple(end.tolist()), tuple(end.tolist()), tuple(end.tolist())]
            colors = [(1.0, 0.15, 0.05, 1.0), (1.0, 0.15, 0.05, 1.0), (1.0, 0.15, 0.05, 1.0)]
            sizes = [6.0, 4.0, 4.0]
            self._draw.draw_lines(starts, ends, colors, sizes)
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Could not update debug force arrow: {type(exc).__name__}: {exc}", flush=True)
                self._warned = True


class HelperHeightPlaneVisualizer:
    """Translucent, non-physical height guide that follows one robot horizontally."""

    def __init__(self, raw_env, height_m: float, env_index: int = 0):
        self.raw_env = raw_env
        self.height_m = float(height_m)
        self.env_index = int(env_index)
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Solo12HelperHeightPlane",
            markers={
                "plane": sim_utils.CuboidCfg(
                    size=(2.4, 2.4, 0.004),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.65, 1.0),
                        emissive_color=(0.02, 0.13, 0.2),
                        opacity=0.16,
                    ),
                )
            },
        )
        self._markers = VisualizationMarkers(cfg)

    def update(self):
        position = self.raw_env._robot.data.root_pos_w[self.env_index : self.env_index + 1].clone()
        foot_positions_w = self.raw_env._get_foot_positions_w()
        terrain_z = self.raw_env._get_terrain_height_below_feet(foot_positions_w)
        front_terrain_z = terrain_z[self.env_index, self.raw_env._front_feet_robot_indices].mean()
        position[:, 2] = front_terrain_z + self.height_m
        self._markers.visualize(position)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed
    if env_cfg.seed is not None and hasattr(env_cfg, "friction_seed"):
        env_cfg.friction_seed = int(env_cfg.seed)
        print(f"[INFO] Using race patch-friction seed: {env_cfg.friction_seed}", flush=True)

    if args_cli.draw_helper_plane_at is not None and not math.isfinite(args_cli.draw_helper_plane_at):
        raise ValueError(f"--draw-helper-plane-at must be finite, got {args_cli.draw_helper_plane_at}.")

    vx_range = args_cli.vx_range if args_cli.vx_range is not None else env_cfg.command_lin_vel_x_range
    vy_range = args_cli.vy_range if args_cli.vy_range is not None else env_cfg.command_lin_vel_y_range
    wz_range = args_cli.wz_range if args_cli.wz_range is not None else env_cfg.command_ang_vel_z_range

    vx_ui_min, vx_ui_max = tuple(map(float, vx_range))
    vy_ui_min, vy_ui_max = tuple(map(float, vy_range))
    wz_ui_min, wz_ui_max = tuple(map(float, wz_range))
    init_vx = float(np.clip(args_cli.cmd_init[0], vx_ui_min, vx_ui_max))
    init_vy = float(np.clip(args_cli.cmd_init[1], vy_ui_min, vy_ui_max))
    init_wz = float(np.clip(args_cli.cmd_init[2], wz_ui_min, wz_ui_max))

    env_cfg.command_resampling_time_s = 1.0e9
    env_cfg.standing_env_prob = 0.0
    env_cfg.command_lin_vel_x_range = (init_vx, init_vx)
    env_cfg.command_lin_vel_y_range = (init_vy, init_vy)
    env_cfg.command_ang_vel_z_range = (init_wz, init_wz)
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    force_ui_max_magnitude = None
    if args_cli.apply_force_ui:
        force_ui_max_magnitude = (
            float(args_cli.force_ui_max_magnitude)
            if args_cli.force_ui_max_magnitude is not None
            else _infer_force_ui_max_magnitude(env_cfg)
        )
        if force_ui_max_magnitude <= 0.0:
            raise RuntimeError(f"--force-ui-max-magnitude must be positive, got {force_ui_max_magnitude}.")
        if hasattr(env_cfg, "base_push_interval_range_s"):
            env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)
        if hasattr(env_cfg, "forces_applied_to_base_curriculum"):
            env_cfg.forces_applied_to_base_curriculum = []
        print(
            f"[INFO] Manual body-frame base force UI enabled. Slider max={force_ui_max_magnitude:g} N; "
            "automatic base-push curriculum is disabled for this play session.",
            flush=True,
        )
    if args_cli.force_transmited_through_joints_reward_scale is not None:
        env_cfg.force_transmited_through_joints_reward_scale = float(args_cli.force_transmited_through_joints_reward_scale)
    if not args_cli.keep_training_stochasticity:
        disabled = _disable_training_stochasticity_for_play(env_cfg)
        if disabled:
            print(
                "[INFO] Disabled training-time stochasticity for play: " + ", ".join(disabled),
                flush=True,
            )
    _sync_base_imu_policy_cfg_from_env_cfg(env_cfg, agent_cfg)
    cli_actuator_gain_overrides = _capture_cli_actuator_gain_overrides(env_cfg, hydra_args)

    training_wandb_run_path = None
    training_kp = None
    training_kd = None
    if not args_cli.disable_training_gain_sync:
        training_run_id = args_cli.training_wandb_run_id or _infer_wandb_run_id_from_checkpoint(args_cli.checkpoint)
        if training_run_id is None:
            raise RuntimeError(
                "Could not infer the training W&B run id from the checkpoint filename. "
                "Pass --training_wandb_run_id explicitly or use --disable_training_gain_sync to skip this check."
            )

        training_wandb_run_path, training_run_config = _fetch_training_run_config_from_wandb(
            entity=args_cli.training_wandb_entity,
            project=args_cli.training_wandb_project,
            run_id=training_run_id,
        )
        training_kp, training_kd = _extract_training_kp_kd_from_wandb_config(training_run_config)

        legs_actuator = env_cfg.robot.actuators.get("legs")
        if legs_actuator is None:
            raise RuntimeError("Solo12 env config does not contain a 'legs' actuator to sync KP/KD into.")

        previous_kp = _actuator_gain_to_float(legs_actuator.stiffness, "stiffness")
        previous_kd = _actuator_gain_to_float(legs_actuator.damping, "damping")
        legs_actuator.stiffness = _sync_gain_preserving_shape(legs_actuator.stiffness, training_kp)
        legs_actuator.damping = _sync_gain_preserving_shape(legs_actuator.damping, training_kd)
        print(
            f"[INFO] Synced Solo12 actuator gains from W&B run '{training_wandb_run_path}': "
            f"stiffness(KP) {previous_kp:g} -> {training_kp:g}, "
            f"damping(KD) {previous_kd:g} -> {training_kd:g}",
            flush=True,
        )

    _apply_cli_actuator_gain_overrides(env_cfg, cli_actuator_gain_overrides)

    resume_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.record_sequence:
        sequence_total_s = record_sequence_total_s(RECORD_SEQUENCE)
        args_cli.duration_s = sequence_total_s
        args_cli.video_length = max(1, int(round(sequence_total_s / float(dt))))
        print(
            "[INFO] Recording fixed command sequence: "
            + ", ".join(f"{cmd} for {duration_s:g}s" for cmd, duration_s in RECORD_SEQUENCE),
            flush=True,
        )

    if args_cli.video:
        video_subdir = "play_direct_0325_rsl_record_sequence" if args_cli.record_sequence else "play_direct_0325_rsl"
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", video_subdir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    sequence_recorder = None
    if args_cli.record_sequence:
        analysis_output_root = args_cli.analysis_output_dir or os.path.join(log_dir, "analysis", "play_direct_0325_rsl")
        checkpoint_label = checkpoint_label_from_path(resume_path)
        analysis_run_name = args_cli.analysis_wandb_name or f"direct_rsl_{args_cli.task}_record_sequence"
        sequence_recorder = SequenceAnalysisRecorder(
            run_label=analysis_run_name,
            joint_names=[str(name) for name in env_cfg.joint_names],
            output_root=analysis_output_root,
            wandb_project=args_cli.analysis_wandb_project,
            wandb_entity=args_cli.analysis_wandb_entity,
            wandb_name=analysis_run_name,
            enable_wandb=not bool(args_cli.no_analysis_wandb),
            metadata={
                "source": "direct_rsl_play",
                "task": args_cli.task,
                "checkpoint": resume_path,
                "checkpoint_label": checkpoint_label,
                "record_sequence_total_s": record_sequence_total_s(RECORD_SEQUENCE),
                "dt": float(dt),
                "args": vars(args_cli),
                "training_wandb_run_path": training_wandb_run_path,
                "training_actuator_stiffness_kp": training_kp,
                "training_actuator_damping_kd": training_kd,
            },
            checkpoint_label=checkpoint_label,
            command_sequence=RECORD_SEQUENCE,
        )

    interactive = not bool(args_cli.headless)
    cmd_state = LiveCommandState(vx=init_vx, vy=init_vy, wz=init_wz)
    camera_toggle_enabled = bool(args_cli.follow_camera) and not bool(args_cli.no_follow_camera)
    camera_state = CameraModeState(mode=str(args_cli.camera_mode).lower())
    ui_window = None
    ui_keepalive = None
    force_state = None
    force_ui_window = None
    force_ui_keepalive = None
    force_arrow = None
    helper_height_plane = None
    if interactive:
        ui_window, ui_keepalive = _build_command_window(
            cmd_state,
            (vx_ui_min, vx_ui_max),
            (vy_ui_min, vy_ui_max),
            (wz_ui_min, wz_ui_max),
            args_cli.ui_title,
            camera_state if camera_toggle_enabled else None,
        )
        if args_cli.apply_force_ui:
            initial_force_magnitude = float(np.clip(args_cli.force_ui_initial_magnitude, 0.0, force_ui_max_magnitude))
            force_state = BodyFrameForceState(
                selected_magnitude=initial_force_magnitude,
                selected_angle_deg=float(args_cli.force_ui_initial_angle_deg),
                selected_elevation_deg=float(args_cli.force_ui_initial_elevation_deg),
                selected_duration_s=float(args_cli.force_ui_initial_duration_s),
                dt=float(dt),
            )
            point_half_extents = getattr(raw_env, "_base_push_application_half_extents", (0.2, 0.1, 0.05))
            if torch.is_tensor(point_half_extents):
                point_half_extents = point_half_extents.detach().cpu().tolist()
            force_ui_window, force_ui_keepalive, _, _ = _build_body_force_window(
                force_state,
                max_magnitude=force_ui_max_magnitude,
                point_half_extents=point_half_extents,
            )
    else:
        print("[WARN] Running headless: live Omni.UI window and scripted cameras are disabled.")
        if args_cli.apply_force_ui:
            print("[WARN] --apply-force-ui was requested but no UI can be shown in headless mode.", flush=True)

    scripted_camera = camera_toggle_enabled and interactive
    cameras = None
    if scripted_camera:
        cameras = {
            "chase": ChaseCamera(
                raw_env,
                env_index=0,
                eye_b=tuple(map(float, args_cli.camera_eye_b)),
                lookat_b=tuple(map(float, args_cli.camera_lookat_b)),
            ),
            "follow": SideFollowCamera(
                raw_env,
                env_index=0,
                eye_b=tuple(map(float, args_cli.side_camera_eye_b)),
                lookat_b=tuple(map(float, args_cli.side_camera_lookat_b)),
            ),
        }
        _update_active_camera(cameras, camera_state)
    if force_state is not None:
        force_arrow = BodyForceArrowVisualizer(raw_env, env_index=0)
    if args_cli.draw_helper_plane_at is not None:
        helper_height_plane = HelperHeightPlaneVisualizer(raw_env, args_cli.draw_helper_plane_at, env_index=0)
        helper_height_plane.update()
        print(
            f"[INFO] Drawing helper plane at {args_cli.draw_helper_plane_at:g} m above the local terrain "
            "below env 0's front feet.",
            flush=True,
        )

    dagger_adapter_checkpoint = _load_dagger_adapter_checkpoint(resume_path)
    if dagger_adapter_checkpoint is not None:
        policy = _build_base_imu_dagger_adapter_policy(
            adapter_checkpoint=dagger_adapter_checkpoint,
            vec_env=vec_env,
            agent_cfg=agent_cfg,
            resume_path=resume_path,
        )
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    elif agent_cfg.class_name == "OffPolicyRunner":
        inference_runner_cfg = _off_policy_inference_runner_cfg(agent_cfg, raw_env.num_envs)
        training_replay_buffer_size = int(agent_cfg.algorithm.replay_buffer_size)
        print(
            "[INFO] SAC inference uses minimal replay storage: "
            f"{training_replay_buffer_size:,} -> {inference_runner_cfg['algorithm']['replay_buffer_size']:,} "
            "transitions; critic/training state is released after actor loading.",
            flush=True,
        )
        runner = OffPolicyRunner(vec_env, inference_runner_cfg, log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading SAC model checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
            raise ValueError(
                "The selected --agent is SAC/OffPolicyRunner, but the checkpoint is not an RSL-RL SAC checkpoint "
                f"(missing 'actor_state_dict'): {resume_path}"
            )
        runner.load(
            resume_path,
            load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": True, "rnd": False},
            map_location=agent_cfg.device,
        )
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    run = None
    if args_cli.wandb:
        run_name = args_cli.wandb_name or f"play_direct_0325_rsl_{os.path.basename(resume_path)}"
        run = wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=run_name,
            config={
                "task": args_cli.task,
                "checkpoint": resume_path,
                "cmd": [init_vx, init_vy, init_wz],
                "duration_s": float(args_cli.duration_s),
                "episode_length_s": float(env_cfg.episode_length_s),
                "num_envs": int(args_cli.num_envs),
                "force_transmited_through_joints_reward_scale": float(env_cfg.force_transmited_through_joints_reward_scale),
                "training_wandb_run_path": training_wandb_run_path,
                "training_actuator_stiffness_kp": float(training_kp) if training_kp is not None else None,
                "training_actuator_damping_kd": float(training_kd) if training_kd is not None else None,
                "apply_force_ui": bool(args_cli.apply_force_ui),
                "force_ui_max_magnitude": float(force_ui_max_magnitude) if force_ui_max_magnitude is not None else None,
            },
        )

    _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
    obs = vec_env.get_observations()
    active_body_force = _sync_manual_body_frame_base_force(raw_env, force_state)
    if force_arrow is not None:
        force_arrow.update(active_body_force)
    if helper_height_plane is not None:
        helper_height_plane.update()
    evaluation_duration_s = _evaluation_duration_s(args_cli.duration_s, TRACKING_COMMANDS)
    target_steps = max(1, int(round(evaluation_duration_s / float(dt))))
    ratio_over_time = []
    track_over_time = []
    force_over_time = []
    force_raw_over_time = []
    lin_vel_error_over_time = []
    ang_vel_error_over_time = []
    root_lin_vel_x_over_time = []
    root_ang_vel_z_over_time = []
    tracking_times_s = []
    tracking_lin_vel_xy = []
    tracking_yaw_rate = []
    tracking_joint_positions = []
    tracking_joint_position_targets = []
    tracking_resets = 0

    for timestep in range(target_steps):
        loop_t0 = time.time()
        if args_cli.record_sequence:
            cmd_state.set_command(*record_sequence_command_at(RECORD_SEQUENCE, timestep * float(dt)))
        elif TRACKING_COMMANDS:
            command_index = min(
                int(timestep * float(dt) / TRACKING_COMMAND_DURATION_S), len(TRACKING_COMMANDS) - 1
            )
            cmd_state.set_command(*TRACKING_COMMANDS[command_index])

        if cmd_state.consume_reset_request():
            print("[INFO] Manual reset requested from the UI; returning env(s) to the reset pose.", flush=True)
            obs = _manual_reset_all_envs(raw_env, vec_env)
            try:
                policy.actor_critic.reset(
                    torch.ones(raw_env.num_envs, device=vec_env.unwrapped.device, dtype=torch.long)
                )
            except AttributeError:
                pass
            except Exception as exc:
                print(f"[WARN] Could not reset policy hidden state after manual env reset: {exc}", flush=True)
            _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
            obs = vec_env.get_observations()
            active_body_force = _sync_manual_body_frame_base_force(raw_env, force_state)
            if force_arrow is not None:
                force_arrow.update(active_body_force)
            _update_active_camera(cameras, camera_state)
            if helper_height_plane is not None:
                helper_height_plane.update()
            continue

        _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
        active_body_force = _sync_manual_body_frame_base_force(raw_env, force_state)
        if force_arrow is not None:
            force_arrow.update(active_body_force)
        _update_active_camera(cameras, camera_state)
        if helper_height_plane is not None:
            helper_height_plane.update()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            if TRACKING_COMMANDS:
                tracking_resets += int(torch.count_nonzero(dones).item())
            try:
                policy.actor_critic.reset(dones)
            except AttributeError:
                pass

        if sequence_recorder is not None:
            q_desired = getattr(raw_env, "_delayed_processed_actions", None)
            if q_desired is None:
                q_desired = getattr(raw_env, "_processed_actions", None)
            if q_desired is None:
                q_desired = torch.full_like(raw_env._robot.data.joint_pos[:, raw_env._joint_ids], float("nan"))
            sequence_recorder.record(
                time_s=(timestep + 1) * float(dt),
                command=raw_env._commands[0],
                q=raw_env._robot.data.joint_pos[0, raw_env._joint_ids],
                q_desired=q_desired[0],
                torque=raw_env._robot.data.applied_torque[0, raw_env._joint_ids],
            )

        terms = _compute_reward_terms(raw_env)

        track_mean = terms["track_lin_vel_xy_exp"].mean().item()
        force_mean = terms["force_transmited_through_joints"].mean().item()
        force_raw_mean = terms["force_transmited_through_joints_raw"].mean().item()
        ratio_mean = terms["ratio"].mean().item()
        torque_mean = terms["joint_torque_sq"].mean().item()
        lin_vel_error_mean = terms["lin_vel_xy_error_norm"].mean().item()
        ang_vel_error_mean = terms["ang_vel_z_error_abs"].mean().item()
        tracked_base_vel, tracked_base_yaw_rate = _get_tracked_base_velocity(raw_env)
        base_vel = tracked_base_vel.mean(dim=0)
        base_ang_vel_z = tracked_base_yaw_rate.mean().item()
        if active_body_force is None:
            force_ui_fx_b = 0.0
            force_ui_fy_b = 0.0
            force_ui_magnitude = 0.0
            force_ui_angle_deg = 0.0
            force_ui_active = 0.0
        else:
            (force_ui_fx_b, force_ui_fy_b, force_ui_fz_b, force_ui_px_b, force_ui_py_b, force_ui_pz_b,
             force_ui_angle_deg, force_ui_elevation_deg) = active_body_force
            force_ui_magnitude = math.sqrt(
                force_ui_fx_b * force_ui_fx_b + force_ui_fy_b * force_ui_fy_b + force_ui_fz_b * force_ui_fz_b
            )
            force_ui_active = 1.0

        ratio_over_time.append(ratio_mean)
        track_over_time.append(track_mean)
        force_over_time.append(force_mean)
        force_raw_over_time.append(force_raw_mean)
        lin_vel_error_over_time.append(lin_vel_error_mean)
        ang_vel_error_over_time.append(ang_vel_error_mean)
        root_lin_vel_x_over_time.append(base_vel[0].item())
        root_ang_vel_z_over_time.append(base_ang_vel_z)
        if TRACKING_COMMANDS:
            tracking_times_s.append((timestep + 1) * float(dt))
            tracking_lin_vel_xy.append((base_vel[0].item(), base_vel[1].item()))
            tracking_yaw_rate.append(base_ang_vel_z)
            q_desired = getattr(raw_env, "_delayed_processed_actions", None)
            if q_desired is None:
                q_desired = getattr(raw_env, "_processed_actions", None)
            if q_desired is None:
                q_desired = torch.full_like(
                    raw_env._robot.data.joint_pos[:, raw_env._joint_ids], float("nan")
                )
            tracking_joint_positions.append(
                raw_env._robot.data.joint_pos[0, raw_env._joint_ids].detach().cpu().tolist()
            )
            tracking_joint_position_targets.append(q_desired[0].detach().cpu().tolist())

        log_data = {
            "play/step": timestep,
            "play/time_s": (timestep + 1) * dt,
            "reward_terms/track_lin_vel_xy_exp": track_mean,
            "reward_terms/force_transmited_through_joints": force_mean,
            "reward_terms/force_transmited_through_joints_raw": force_raw_mean,
            "reward_terms/ratio_force_over_track": ratio_mean,
            "telemetry/joint_torque_sq": torque_mean,
            "telemetry/cmd_lin_vel_x": raw_env._commands[:, 0].mean().item(),
            "telemetry/cmd_lin_vel_y": raw_env._commands[:, 1].mean().item(),
            "telemetry/cmd_ang_vel_z": raw_env._commands[:, 2].mean().item(),
            "telemetry/lin_vel_xy_error_norm": lin_vel_error_mean,
            "telemetry/ang_vel_z_error_abs": ang_vel_error_mean,
            "telemetry/root_lin_vel_x": base_vel[0].item(),
            "telemetry/root_lin_vel_y": base_vel[1].item(),
            "telemetry/root_ang_vel_z": base_ang_vel_z,
            "force_ui/active": force_ui_active,
            "force_ui/fx_b": float(force_ui_fx_b),
            "force_ui/fy_b": float(force_ui_fy_b),
            "force_ui/fz_b": float(force_ui_fz_b) if active_body_force is not None else 0.0,
            "force_ui/px_b": float(force_ui_px_b) if active_body_force is not None else 0.0,
            "force_ui/py_b": float(force_ui_py_b) if active_body_force is not None else 0.0,
            "force_ui/pz_b": float(force_ui_pz_b) if active_body_force is not None else 0.0,
            "force_ui/magnitude": float(force_ui_magnitude),
            "force_ui/angle_deg": float(force_ui_angle_deg),
            "force_ui/elevation_deg": float(force_ui_elevation_deg) if active_body_force is not None else 0.0,
        }

        if run is not None:
            wandb.log(log_data)
        elif args_cli.verbose_play:
            print(log_data)

        if isinstance(dones, torch.Tensor) and dones.any():
            obs = vec_env.get_observations()
            _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
            obs = vec_env.get_observations()
            active_body_force = _sync_manual_body_frame_base_force(raw_env, force_state)
            if force_arrow is not None:
                force_arrow.update(active_body_force)
            _update_active_camera(cameras, camera_state)

        if force_state is not None:
            force_state.tick()

        _update_active_camera(cameras, camera_state)

        sleep_time = dt - (time.time() - loop_t0)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    final_ratio = statistics.median(ratio_over_time)
    median_track = statistics.median(track_over_time)
    median_force = statistics.median(force_over_time)
    median_force_raw = statistics.median(force_raw_over_time)
    median_lin_vel_error = statistics.median(lin_vel_error_over_time)
    median_ang_vel_error = statistics.median(ang_vel_error_over_time)
    median_root_lin_vel_x = statistics.median(root_lin_vel_x_over_time)
    median_root_ang_vel_z = statistics.median(root_ang_vel_z_over_time)
    print("[RESULT] median_ratio_force_over_track=", final_ratio)
    print("[RESULT] median_track_lin_vel_xy_exp=", median_track)
    print("[RESULT] median_force_transmited_through_joints=", median_force)
    print("[RESULT] median_force_transmited_through_joints_raw=", median_force_raw)
    print("[RESULT] median_lin_vel_xy_error_norm=", median_lin_vel_error)
    print("[RESULT] median_ang_vel_z_error_abs=", median_ang_vel_error)
    print("[RESULT] median_root_lin_vel_x=", median_root_lin_vel_x)
    print("[RESULT] median_root_ang_vel_z=", median_root_ang_vel_z)
    print("[RESULT] current_force_transmited_through_joints_reward_scale=", env_cfg.force_transmited_through_joints_reward_scale)

    if run is not None:
        wandb.log(
            {
                "summary/median_ratio_force_over_track": final_ratio,
                "summary/median_track_lin_vel_xy_exp": median_track,
                "summary/median_force_transmited_through_joints": median_force,
                "summary/median_force_transmited_through_joints_raw": median_force_raw,
                "summary/median_lin_vel_xy_error_norm": median_lin_vel_error,
                "summary/median_ang_vel_z_error_abs": median_ang_vel_error,
                "summary/median_root_lin_vel_x": median_root_lin_vel_x,
                "summary/median_root_ang_vel_z": median_root_ang_vel_z,
                "summary/current_force_transmited_through_joints_reward_scale": float(env_cfg.force_transmited_through_joints_reward_scale),
            }
        )
        wandb.finish()

    if sequence_recorder is not None:
        sequence_recorder.finish(env_config=env_cfg, agent_config=agent_cfg.to_dict())

    if TRACKING_COMMANDS:
        checkpoint_stem = Path(resume_path).stem
        output_dir = (
            _ISAACLAB_ROOT
            / "logs"
            / "rsl_rl"
            / "cmd_tracking"
            / f"{time.strftime('%Y%m%d_%H%M%S')}_{checkpoint_stem}"
        )
        paths = _save_command_tracking_plots(
            output_dir,
            tracking_times_s,
            TRACKING_COMMANDS,
            tracking_lin_vel_xy,
            tracking_yaw_rate,
            [str(name) for name in env_cfg.joint_names],
            tracking_joint_positions,
            tracking_joint_position_targets,
            raw_env._robot.data.joint_pos_limits[0, raw_env._joint_ids].detach().cpu().numpy(),
            raw_env._joint_soft_pos_limits[0].detach().cpu().numpy(),
        )
        tracking_summary = _summarize_command_tracking(
            tracking_times_s,
            TRACKING_COMMANDS,
            tracking_lin_vel_xy,
            tracking_yaw_rate,
            tracking_resets,
        )
        config_path, summary_path, tracking_run_url = _save_and_upload_tracking_metadata(
            output_dir=output_dir,
            artifact_paths=paths,
            summary=tracking_summary,
            resume_path=resume_path,
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            dt=float(dt),
        )
        print("[RESULT] Command-tracking artifacts:", flush=True)
        for path in [config_path, summary_path, *paths]:
            print(f"  {path}", flush=True)
        print("[RESULT] Command-tracking summary:", json.dumps(tracking_summary, sort_keys=True), flush=True)
        if tracking_run_url:
            print(f"[RESULT] W&B tracking run: {tracking_run_url}", flush=True)

    if force_arrow is not None:
        force_arrow.clear()
    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
