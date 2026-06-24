# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a direct Solo12 skrl checkpoint with a fixed command and log reward-term ratios to W&B.

Example:
./isaaclab.sh -p source/scripts/skrl/play_direct_0325.py \
    --task="solo12-v0" \
    --num_envs 1 \
    --checkpoint "/home/jordibelp/IsaacLab/logs/skrl/checkpoints/0323_bq05v1pq_best_agent.pt" \
    --cmd 1 1 0 \
    --duration_s 5 \
    --headless
"""

import argparse
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a direct Solo12 checkpoint with a fixed command.")
parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=400, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="solo12-v0", help="Task name.")
parser.add_argument("--agent", type=str, default=None, help="RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
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
    default=None,
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
    "--force_transmited_through_joints_reward_scale",
    type=float,
    default=None,
    help="Override the environment reward scale for force_transmited_through_joints.",
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
    "--camera_mode",
    type=str,
    choices=("follow", "free"),
    default="follow",
    help="Initial interactive viewport camera mode. 'follow' tracks Solo12; 'free' leaves the viewport camera user-controlled.",
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
parser.add_argument("--wandb", action="store_true", default=False, help="Log metrics to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="borinotIsaacLab_inference", help="W&B project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team.")
parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name.")
parser.add_argument(
    "--training_wandb_run_id",
    type=str,
    default=None,
    help="Accepted for CLI compatibility with the RSL-RL play script. SKRL play does not use it.",
)
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

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import skrl
import torch
import wandb
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401
import cat_envs.tasks  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install with 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner as SkrlRunner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner as SkrlRunner

try:
    from cat_envs.tasks.utils.skrl.runner import Runner as CatSkrlRunner
except Exception:
    CatSkrlRunner = None

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@dataclass
class LiveCommandState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

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

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            return float(self.vx), float(self.vy), float(self.wz)


@dataclass
class CameraModeState:
    follow_enabled: bool = True

    def __post_init__(self):
        self._lock = threading.Lock()

    def toggle(self):
        with self._lock:
            self.follow_enabled = not self.follow_enabled

    def is_follow(self) -> bool:
        with self._lock:
            return bool(self.follow_enabled)

    def label(self) -> str:
        return "follow" if self.is_follow() else "free"


def _build_command_window(
    cmd_state: LiveCommandState,
    vx_range,
    vy_range,
    wz_range,
    title: str,
    camera_state: CameraModeState | None = None,
):
    """Create a small Omni.UI window with live vx / vy / wz controls."""
    import omni.ui as ui

    vx_min, vx_max = map(float, vx_range)
    vy_min, vy_max = map(float, vy_range)
    wz_min, wz_max = map(float, wz_range)

    win_height = 265 if camera_state is not None else 215
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
            if camera_state is not None:
                camera_mode_model = ui.SimpleStringModel(f"Camera: {camera_state.label()}")

                def _toggle_camera():
                    camera_state.toggle()
                    camera_mode_model.set_value(f"Camera: {camera_state.label()}")

                with ui.HStack(spacing=8, height=30):
                    ui.Label("camera", width=90)
                    ui.Button("Toggle follow/free", width=150, clicked_fn=_toggle_camera)
                    ui.StringField(camera_mode_model, read_only=True, height=26)
                ui.Label("Free mode stops scripted camera updates; use the viewport navigation normally.", height=18)
            ui.Label("Tip: click a number to type an exact value and press Enter.", height=18)

    return win, keepalive


def _apply_live_direct_command(raw_env, cmd_state: LiveCommandState, vx_limits, vy_limits, wz_limits, env_ids=slice(None)):
    vx, vy, wz = cmd_state.get()
    vx = float(np.clip(vx, *map(float, vx_limits)))
    vy = float(np.clip(vy, *map(float, vy_limits)))
    wz = float(np.clip(wz, *map(float, wz_limits)))
    if hasattr(raw_env, "_commands"):
        raw_env._commands[env_ids, 0] = vx
        raw_env._commands[env_ids, 1] = vy
        raw_env._commands[env_ids, 2] = wz
        if hasattr(raw_env, "_command_steps_left"):
            raw_env._command_steps_left[env_ids] = max(1, int(1e9))
        return

    command_manager = getattr(raw_env, "command_manager", None)
    if command_manager is None:
        raise RuntimeError("Could not find direct _commands or manager-based command_manager on environment.")

    term = command_manager.get_term("base_velocity")
    command = command_manager.get_command("base_velocity")
    command[env_ids, 0] = vx
    command[env_ids, 1] = vy
    command[env_ids, 2] = wz
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[env_ids, 0] = vx
        term.vel_command_b[env_ids, 1] = vy
        term.vel_command_b[env_ids, 2] = wz
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[env_ids] = False
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[env_ids] = False
    # CaT's velocity command has an intra-episode stochastic resampler. Make it
    # effectively inert during play; the script writes the command every policy step.
    if hasattr(term, "max_episode_length_s"):
        term.max_episode_length_s = 1.0e12


def _get_command_ranges_from_cfg(env_cfg):
    if hasattr(env_cfg, "command_lin_vel_x_range"):
        return env_cfg.command_lin_vel_x_range, env_cfg.command_lin_vel_y_range, env_cfg.command_ang_vel_z_range

    command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    ranges = getattr(command_cfg, "ranges", None)
    if ranges is None:
        raise RuntimeError("Could not resolve velocity command ranges from env_cfg.")
    return ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z


def _set_fixed_command_in_cfg(env_cfg, vx: float, vy: float, wz: float):
    if hasattr(env_cfg, "command_resampling_time_s"):
        env_cfg.command_resampling_time_s = 1.0e9
    if hasattr(env_cfg, "standing_env_prob"):
        env_cfg.standing_env_prob = 0.0
    if hasattr(env_cfg, "command_lin_vel_x_range"):
        env_cfg.command_lin_vel_x_range = (vx, vx)
        env_cfg.command_lin_vel_y_range = (vy, vy)
        env_cfg.command_ang_vel_z_range = (wz, wz)
        return

    command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    ranges = getattr(command_cfg, "ranges", None)
    if ranges is None:
        raise RuntimeError("Could not configure fixed velocity command on env_cfg.")
    ranges.lin_vel_x = (vx, vx)
    ranges.lin_vel_y = (vy, vy)
    ranges.ang_vel_z = (wz, wz)
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(command_cfg, "rel_standing_envs"):
        command_cfg.rel_standing_envs = 0.0
    if hasattr(command_cfg, "rel_heading_envs"):
        command_cfg.rel_heading_envs = 0.0
    if hasattr(command_cfg, "heading_command"):
        command_cfg.heading_command = False


def _get_robot(raw_env):
    if hasattr(raw_env, "_robot"):
        return raw_env._robot
    return raw_env.scene["robot"]


def _get_command_tensor(raw_env):
    if hasattr(raw_env, "_commands"):
        return raw_env._commands
    return raw_env.command_manager.get_command("base_velocity")


def _get_reward_step_term(raw_env, name: str):
    reward_manager = getattr(raw_env, "reward_manager", None)
    if reward_manager is None:
        return None
    try:
        term_idx = reward_manager.active_terms.index(name)
    except ValueError:
        return None
    return reward_manager._step_reward[:, term_idx] * raw_env.step_dt


def _disable_training_stochasticity_for_play(env_cfg) -> list[str]:
    disabled = []

    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        disabled_events = []
        for name, value in vars(events_cfg).items():
            if name.startswith("_") or value is None:
                continue
            setattr(events_cfg, name, None)
            disabled_events.append(name)
        if disabled_events:
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

    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
        disabled.append("actuation_delay_range")

    return disabled


def _compute_reward_terms(raw_env):
    commands = _get_command_tensor(raw_env)
    robot = _get_robot(raw_env)
    joint_ids = getattr(raw_env, "_joint_ids", slice(None))

    lin_vel_error = torch.sum(torch.square(commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
    lin_vel_xy_error_norm = torch.linalg.vector_norm(commands[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
    ang_vel_z_error_abs = torch.abs(commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
    if hasattr(raw_env, "_compute_force_transmited_through_joints"):
        force_transmited_through_joints = raw_env._compute_force_transmited_through_joints()
    else:
        force_transmited_through_joints = torch.zeros_like(lin_vel_xy_error_norm)

    track_lin_vel_xy_exp = _get_reward_step_term(raw_env, "track_lin_vel_xy_exp")
    if track_lin_vel_xy_exp is None:
        tracking_std = float(getattr(raw_env.cfg, "tracking_std", math.sqrt(0.25)))
        reward_scale = float(getattr(raw_env.cfg, "track_lin_vel_xy_reward_scale", 1.0))
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / tracking_std**2) * reward_scale * raw_env.step_dt
    force_reward = (
        force_transmited_through_joints
        * float(getattr(raw_env.cfg, "force_transmited_through_joints_reward_scale", 0.0))
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


class SideFollowCamera:
    """Camera that tracks the robot root and keeps a side view in the robot yaw frame."""

    def __init__(self, raw_env, env_index: int = 0, eye_b=(0.0, -2.6, 1.1), lookat_b=(0.0, 0.0, 0.35)):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.eye_b = np.asarray(eye_b, dtype=np.float64)
        self.lookat_b = np.asarray(lookat_b, dtype=np.float64)
        self.asset = _get_robot(raw_env)

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
        root_pos_w = self.asset.data.root_pos_w[self.env_index].detach().cpu().numpy().astype(np.float64)
        root_quat_w = self.asset.data.root_quat_w[self.env_index].detach().cpu()
        yaw = self._quat_wxyz_to_yaw(root_quat_w)
        rot = self._yaw_rot(yaw)

        eye_w = root_pos_w + rot @ self.eye_b
        lookat_w = root_pos_w + rot @ self.lookat_b
        self.raw_env.sim.set_camera_view(eye=eye_w, target=lookat_w)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)

    if args_cli.seed is not None:
        experiment_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed
    if env_cfg.seed is not None and hasattr(env_cfg, "friction_seed"):
        env_cfg.friction_seed = int(env_cfg.seed)
        print(f"[INFO] Using race patch-friction seed: {env_cfg.friction_seed}", flush=True)

    default_vx_range, default_vy_range, default_wz_range = _get_command_ranges_from_cfg(env_cfg)
    vx_range = args_cli.vx_range if args_cli.vx_range is not None else default_vx_range
    vy_range = args_cli.vy_range if args_cli.vy_range is not None else default_vy_range
    wz_range = args_cli.wz_range if args_cli.wz_range is not None else default_wz_range

    vx_ui_min, vx_ui_max = tuple(map(float, vx_range))
    vy_ui_min, vy_ui_max = tuple(map(float, vy_range))
    wz_ui_min, wz_ui_max = tuple(map(float, wz_range))
    init_vx = float(np.clip(args_cli.cmd_init[0], vx_ui_min, vx_ui_max))
    init_vy = float(np.clip(args_cli.cmd_init[1], vy_ui_min, vy_ui_max))
    init_wz = float(np.clip(args_cli.cmd_init[2], wz_ui_min, wz_ui_max))

    _set_fixed_command_in_cfg(env_cfg, init_vx, init_vy, init_wz)
    if args_cli.force_transmited_through_joints_reward_scale is not None:
        env_cfg.force_transmited_through_joints_reward_scale = float(args_cli.force_transmited_through_joints_reward_scale)
    if not args_cli.keep_training_stochasticity:
        disabled = _disable_training_stochasticity_for_play(env_cfg)
        if disabled:
            print(
                "[INFO] Disabled training-time stochasticity for play: " + ", ".join(disabled),
                flush=True,
            )
    # env_cfg.terrain.terrain_type = "usd"
    # env_cfg.terrain.usd_path = "/home/jordibelp/IsaacLab/source/borinotIsaacLab/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"

    resume_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_direct_0325"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    vec_env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    raw_env = env.unwrapped

    interactive = not bool(args_cli.headless)
    cmd_state = LiveCommandState(vx=init_vx, vy=init_vy, wz=init_wz)
    camera_toggle_enabled = bool(args_cli.follow_camera) and not bool(args_cli.no_follow_camera)
    camera_state = CameraModeState(follow_enabled=(str(args_cli.camera_mode).lower() == "follow"))
    ui_window = None
    ui_keepalive = None
    if interactive:
        ui_window, ui_keepalive = _build_command_window(
            cmd_state,
            (vx_ui_min, vx_ui_max),
            (vy_ui_min, vy_ui_max),
            (wz_ui_min, wz_ui_max),
            args_cli.ui_title,
            camera_state if camera_toggle_enabled else None,
        )
    else:
        print("[WARN] Running headless: live Omni.UI window and camera follow are disabled.")

    follow_camera = camera_toggle_enabled and interactive
    camera = None
    if follow_camera:
        camera = SideFollowCamera(
            raw_env,
            env_index=0,
            eye_b=tuple(map(float, args_cli.camera_eye_b)),
            lookat_b=tuple(map(float, args_cli.camera_lookat_b)),
        )
        if camera_state.is_follow():
            camera.update()

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    # Never let the underlying skrl training config auto-start W&B during play.
    # If the user explicitly passes --wandb, this script handles that manually below.
    experiment_cfg["agent"]["experiment"]["wandb"] = False
    experiment_cfg["agent"]["experiment"]["wandb_kwargs"] = {}

    runner_cls = (
        CatSkrlRunner
        if isinstance(raw_env, ManagerBasedRLEnv) and "cat" in args_cli.task.lower() and CatSkrlRunner is not None
        else SkrlRunner
    )
    print(f"[INFO] Using SKRL runner: {runner_cls.__module__}.{runner_cls.__name__}", flush=True)
    runner = runner_cls(vec_env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    run = None
    if args_cli.wandb:
        run_name = args_cli.wandb_name or f"play_direct_0325_{os.path.basename(resume_path)}"
        run = wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=run_name,
            config={
                "task": args_cli.task,
                "checkpoint": resume_path,
                "cmd": [init_vx, init_vy, init_wz],
                "duration_s": float(args_cli.duration_s),
                "num_envs": int(args_cli.num_envs),
                "force_transmited_through_joints_reward_scale": float(
                    getattr(env_cfg, "force_transmited_through_joints_reward_scale", 0.0)
                ),
            },
        )

    obs, _ = vec_env.reset()
    _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
    target_steps = max(1, int(round(float(args_cli.duration_s) / float(dt))))
    cumulative = {
        "track_lin_vel_xy_exp": 0.0,
        "force_transmited_through_joints": 0.0,
        "force_transmited_through_joints_raw": 0.0,
    }
    ratio_over_time = []
    track_over_time = []
    force_over_time = []
    force_raw_over_time = []
    lin_vel_error_over_time = []
    ang_vel_error_over_time = []
    root_lin_vel_x_over_time = []
    root_ang_vel_z_over_time = []

    for timestep in range(target_steps):
        loop_t0 = time.time()

        _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(vec_env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in vec_env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = vec_env.step(actions)

        terms = _compute_reward_terms(raw_env)

        track_mean = terms["track_lin_vel_xy_exp"].mean().item()
        force_mean = terms["force_transmited_through_joints"].mean().item()
        force_raw_mean = terms["force_transmited_through_joints_raw"].mean().item()
        ratio_mean = terms["ratio"].mean().item()
        torque_mean = terms["joint_torque_sq"].mean().item()
        lin_vel_error_mean = terms["lin_vel_xy_error_norm"].mean().item()
        ang_vel_error_mean = terms["ang_vel_z_error_abs"].mean().item()
        robot = _get_robot(raw_env)
        command_tensor = _get_command_tensor(raw_env)
        base_vel = robot.data.root_lin_vel_b[:, :2].mean(dim=0)
        base_ang_vel_z = robot.data.root_ang_vel_b[:, 2].mean().item()

        cumulative["track_lin_vel_xy_exp"] += track_mean
        cumulative["force_transmited_through_joints"] += force_mean
        cumulative["force_transmited_through_joints_raw"] += force_raw_mean
        ratio_over_time.append(ratio_mean)
        track_over_time.append(track_mean)
        force_over_time.append(force_mean)
        force_raw_over_time.append(force_raw_mean)
        lin_vel_error_over_time.append(lin_vel_error_mean)
        ang_vel_error_over_time.append(ang_vel_error_mean)
        root_lin_vel_x_over_time.append(base_vel[0].item())
        root_ang_vel_z_over_time.append(base_ang_vel_z)

        log_data = {
            "play/step": timestep,
            "play/time_s": (timestep + 1) * dt,
            "reward_terms/track_lin_vel_xy_exp": track_mean,
            "reward_terms/force_transmited_through_joints": force_mean,
            "reward_terms/force_transmited_through_joints_raw": force_raw_mean,
            "reward_terms/ratio_force_over_track": ratio_mean,
            "telemetry/joint_torque_sq": torque_mean,
            "telemetry/cmd_lin_vel_x": command_tensor[:, 0].mean().item(),
            "telemetry/cmd_lin_vel_y": command_tensor[:, 1].mean().item(),
            "telemetry/cmd_ang_vel_z": command_tensor[:, 2].mean().item(),
            "telemetry/lin_vel_xy_error_norm": lin_vel_error_mean,
            "telemetry/ang_vel_z_error_abs": ang_vel_error_mean,
            "telemetry/root_lin_vel_x": base_vel[0].item(),
            "telemetry/root_lin_vel_y": base_vel[1].item(),
            "telemetry/root_ang_vel_z": base_ang_vel_z,
        }

        if run is not None:
            wandb.log(log_data)
        elif args_cli.verbose_play:
            print(log_data)

        if isinstance(terminated, torch.Tensor) and terminated.any():
            obs, _ = vec_env.reset()
            _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
            if camera is not None and camera_state.is_follow():
                camera.update()
        elif isinstance(truncated, torch.Tensor) and truncated.any():
            obs, _ = vec_env.reset()
            _apply_live_direct_command(raw_env, cmd_state, (vx_ui_min, vx_ui_max), (vy_ui_min, vy_ui_max), (wz_ui_min, wz_ui_max))
            if camera is not None and camera_state.is_follow():
                camera.update()

        if camera is not None and camera_state.is_follow():
            camera.update()

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
    force_scale = float(getattr(env_cfg, "force_transmited_through_joints_reward_scale", 0.0))
    print("[RESULT] current_force_transmited_through_joints_reward_scale=", force_scale)

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
                "summary/current_force_transmited_through_joints_reward_scale": force_scale,
            }
        )
        wandb.finish()

    vec_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
