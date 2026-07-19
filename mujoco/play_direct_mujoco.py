#!/usr/bin/env python3
"""Run an IsaacLab Solo12 RSL-RL policy in MuJoCo for sim-to-sim evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch


PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
# play_direct_0325.py --disable_training_gain_sync plays the solo12-two-feet cfg gains (kp=15, kd=0.5),
# while W&B run q3a68133 trained with kp=9, kd=0.2. Default to the played gains; --kp/--kd override.
DEFAULT_KP = 15.0
DEFAULT_KD = 0.5
TRAINING_KP = 9.0
TRAINING_KD = 0.2
EFFORT_LIMIT = 2.65
OBS_NORM_EPS = 1.0e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
COMMAND_DURATION_S = 5.0
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
SAFE_Q = np.array((0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8))


def parse_commands(value: str) -> list[tuple[float, float, float]]:
    commands = []
    for index, raw in enumerate(value.split(";"), start=1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(f"command {index} must contain vx vy wz, got {raw!r}")
        commands.append(tuple(float(field) for field in fields))
    if not commands:
        raise argparse.ArgumentTypeError("--track-cmds must contain at least one command")
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--track-cmds", type=parse_commands, default=None,
                        help="Scripted command sequence (5 s each). Omit to drive the robot live with sliders/keys.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration_s", type=float, default=2000.0)
    parser.add_argument("--episode_length_s", type=float, default=80.0)
    parser.add_argument("--cmd_init", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output-dir", type=Path,
                        help="Output root (default logs/mujoco/cmd_tracking); model/timestamp are appended.")
    parser.add_argument("--realtime", action="store_true", help="Pace the interactive viewer at wall-clock speed.")
    parser.add_argument("--task", default="solo12-two-feet")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--kp", type=float, default=None,
                        help=f"PD stiffness (default: env.kp override, else {DEFAULT_KP:g} = Isaac play with "
                             f"--disable_training_gain_sync; training used {TRAINING_KP:g}).")
    parser.add_argument("--kd", type=float, default=None,
                        help=f"PD damping (default: env.kd override, else {DEFAULT_KD:g} = Isaac play with "
                             f"--disable_training_gain_sync; training used {TRAINING_KD:g}).")
    parser.add_argument("--camera", choices=("free", "side", "front"), default="side",
                        help="Initial viewer camera; press C in the viewer to cycle.")
    parser.add_argument("--show-viewer-ui", action="store_true",
                        help="Show MuJoCo's left/right helper panels (hidden by default).")
    parser.add_argument("--force-ui-max", type=float, default=10.0,
                        help="Maximum magnitude of the base-force slider in live mode [N].")
    parser.add_argument("--wandb-project", default="solo12-two-feet-exp")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--no-wandb", action="store_true", help="Keep artifacts local without uploading to W&B.")
    return parser


# solo12-two-feet training command ranges; env.command_*_range CLI overrides widen the slider/clip limits.
DEFAULT_COMMAND_RANGES = ((-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5))
_ENV_RANGE_KEYS = {
    "env.command_lin_vel_x_range": 0,
    "env.command_lin_vel_y_range": 1,
    "env.command_ang_vel_z_range": 2,
}


def consume_env_overrides(unknown: list[str]) -> tuple[dict, list[str]]:
    """Pull the env.* overrides MuJoCo understands (kp/kd/command ranges) out of the Isaac CLI tail."""
    import ast

    overrides: dict = {}
    ignored: list[str] = []
    for token in unknown:
        key, _, raw = token.partition("=")
        try:
            if key in ("env.kp", "env.kd"):
                overrides[key.removeprefix("env.")] = float(raw)
            elif key in _ENV_RANGE_KEYS:
                low, high = ast.literal_eval(raw)
                overrides[key.removeprefix("env.")] = (float(low), float(high))
            else:
                ignored.append(token)
        except (ValueError, SyntaxError):
            ignored.append(token)
    return overrides, ignored


class Policy(torch.nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        weight_keys = sorted(
            (key for key in state if key.startswith("actor.") and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        self.layers = torch.nn.ModuleList()
        for key in weight_keys:
            out_dim, in_dim = state[key].shape
            layer = torch.nn.Linear(in_dim, out_dim)
            layer.weight.data.copy_(state[key])
            layer.bias.data.copy_(state[key.replace(".weight", ".bias")])
            self.layers.append(layer)
        self.register_buffer("obs_mean", state["actor_obs_normalizer._mean"].reshape(-1))
        self.register_buffer("obs_std", state["actor_obs_normalizer._std"].reshape(-1))
        self.eval()

    @torch.inference_mode()
    def forward(self, observation: np.ndarray) -> np.ndarray:
        x = (torch.from_numpy(observation).float() - self.obs_mean) / (self.obs_std + OBS_NORM_EPS)
        for layer in self.layers[:-1]:
            x = torch.nn.functional.elu(layer(x))
        return self.layers[-1](x).numpy()


def quat_rotation(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ))


class Solo12Mujoco:
    def __init__(self, model_path: Path, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.base_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if self.model.geom_bodyid[geom_id] == self.base_id and self.model.geom_contype[geom_id] != 0
        }
        self.set_gains(kp, kd)
        self.action = np.zeros(12)
        self.reset()

    def set_gains(self, kp: float, kd: float) -> None:
        """Mirror PhysX implicit PD drives: position servo with implicit damping, force-limited."""
        self.kp, self.kd = float(kp), float(kd)
        self.model.actuator_gainprm[self.actuator_ids, 0] = self.kp
        self.model.actuator_biasprm[self.actuator_ids, 1] = -self.kp
        self.model.actuator_biasprm[self.actuator_ids, 2] = -self.kd
        self.model.actuator_forcerange[self.actuator_ids, 0] = -EFFORT_LIMIT
        self.model.actuator_forcerange[self.actuator_ids, 1] = EFFORT_LIMIT

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = (0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0)  # wxyz quat: yaw = pi, as in Isaac reset
        self.data.qpos[self.joint_qpos] = SAFE_Q
        self.data.ctrl[self.actuator_ids] = SAFE_Q
        self.action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

    def _base_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame (angular, linear-at-COM) base velocity, matching PhysX root com velocities."""
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        return velocity[:3], velocity[3:]

    def observation(self, command: tuple[float, float, float]) -> np.ndarray:
        # IsaacLab contract: root_com_{lin,ang}_vel_b = R_link^T @ world COM velocity;
        # mjOBJ_BODY with flg_local=1 would use the inertial (ximat) frame instead -- do not use it.
        rotation_wb = quat_rotation(self.data.qpos[3:7])
        ang_vel_w, lin_vel_w = self._base_velocity_world()
        lin_vel_b = rotation_wb.T @ lin_vel_w
        ang_vel_b = rotation_wb.T @ ang_vel_w
        gravity_b = rotation_wb.T @ np.array((0.0, 0.0, -1.0))
        q = self.data.qpos[self.joint_qpos]
        qd = self.data.qvel[self.joint_dof]
        return np.concatenate((lin_vel_b, ang_vel_b, gravity_b, command, q - SAFE_Q, qd, self.action))

    def tracked_velocity(self) -> tuple[float, float, float]:
        """Planar velocity in the gravity-aligned heading frame plus world yaw rate (env reward frame)."""
        rotation_wb = quat_rotation(self.data.qpos[3:7])
        lateral = rotation_wb[:2, 1]
        norm = np.linalg.norm(lateral)
        if norm < np.finfo(float).eps:
            lateral = np.array((0.0, 1.0))
        else:
            lateral /= norm
        forward = np.array((lateral[1], -lateral[0]))
        ang_vel_w, lin_vel_w = self._base_velocity_world()
        velocity_xy = lin_vel_w[:2]
        return float(velocity_xy @ forward), float(velocity_xy @ lateral), float(ang_vel_w[2])

    def step(self, action: np.ndarray) -> None:
        self.action = np.asarray(action).copy()
        self.data.ctrl[self.actuator_ids] = SAFE_Q + ACTION_SCALE * self.action
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

    def base_hit_ground(self) -> bool:
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if self.ground_id in pair and pair & self.base_geom_ids:
                return True
        return False

    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def gravity_x_b(self) -> float:
        """Base-frame gravity x: ~0 on four feet, ~-0.96 in the upright two-feet stance."""
        return float(quat_rotation(self.data.qpos[3:7]).T[0] @ np.array((0.0, 0.0, -1.0)))


def prepare_viewer_environment() -> None:
    """Prefer XWayland for the GLFW viewer: the Wayland backend spams GLFW/GLib warnings."""
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        os.environ.pop("WAYLAND_DISPLAY")
        print("[INFO] Wayland session detected; using XWayland for the MuJoCo viewer.", flush=True)
    warnings.filterwarnings("ignore", message=".*Wayland.*")


def save_results(output_dir: Path, rows: list[tuple], commands: list[tuple[float, float, float]]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "command_tracking.csv"
    header = ("time_s", "cmd_vx", "cmd_vy", "cmd_wz", "velocity_vx", "velocity_vy", "yaw_rate_wz",
              "vxy_error_norm", "wz_error_abs", "reset", "base_height_m", "gravity_x_b")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    values = np.asarray(rows, dtype=float)
    times, errors = values[:, 0], values[:, 7]

    def decorate(ax):
        for boundary in np.arange(1, len(commands)) * COMMAND_DURATION_S:
            ax.axvline(boundary, color="0.25", linewidth=1.0, alpha=0.65)
        for reset_t in values[values[:, 9] > 0.5, 0]:
            ax.axvline(reset_t, color="#9467bd", linewidth=1.1, linestyle="--", alpha=0.8)
        ax.set_xlim(0.0, len(commands) * COMMAND_DURATION_S); ax.grid(axis="y", alpha=0.22)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.8), constrained_layout=True)
    ax.plot(times, errors, color="#e69f00", linewidth=1.7); decorate(ax)
    ax.set(xlabel="Time [s]", ylabel=r"$\|\Delta v_{xy}\|$ [m/s]", title="MuJoCo planar command-tracking error (base-footprint frame)")
    y0, y1 = ax.get_ylim(); span = max(y1 - y0, 1e-6); max_cmd = max(max(math.hypot(x, y) for x, y, _ in commands), 1e-6)
    for index, (vx, vy, _) in enumerate(commands):
        center = (index + 0.5) * COMMAND_DURATION_S; scale = 0.18 / max_cmd
        dx, dy = -vy * COMMAND_DURATION_S * scale, vx * span * scale
        ax.annotate("", xy=(center + dx, y0 + 0.82*span + dy), xytext=(center - dx, y0 + 0.82*span - dy), arrowprops={"arrowstyle":"->", "color":"#d62728", "lw":1.8, "alpha":0.65})
        ax.text(center, y0 + 0.97*span, f"({vx:g}, {vy:g})", ha="center", va="top", fontsize=8, color="#a51f1f")
    vxy_path = output_dir / "vxy_tracking_error.png"; fig.savefig(vxy_path, dpi=180); plt.close(fig)
    paths = [csv_path, vxy_path]
    if any(abs(wz) > 1e-9 for _, _, wz in commands):
        fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.4), constrained_layout=True)
        ax.plot(times, values[:, 8], color="#0072b2", linewidth=1.7); decorate(ax)
        ax.set(xlabel="Time [s]", ylabel=r"$|\Delta \omega_z|$ [rad/s]", title="MuJoCo yaw-rate command-tracking error (world z)")
        wz_path = output_dir / "wz_tracking_error.png"; fig.savefig(wz_path, dpi=180); plt.close(fig); paths.append(wz_path)
    return paths


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_meshes_sha256(model_path: Path) -> str | None:
    """Digest of the visual mesh set referenced by the MJCF.

    The archived `solo12.xml` is no longer standalone -- it references `meshes/*.obj`, which are
    versioned in-repo rather than uploaded per run. The meshes never affect physics, so recording
    their digest (alongside `git_revision`) is enough to pin exactly which visuals were used.
    """
    mesh_dir = model_path.parent / "meshes"
    files = sorted(mesh_dir.glob("*.obj"))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_directory(output_root: Path, checkpoint_stem: str, timestamp: str) -> Path:
    run_dir = output_root / checkpoint_stem / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / checkpoint_stem / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def summarize_rows(rows: list[tuple]) -> dict[str, float | int]:
    values = np.asarray(rows, dtype=float)
    vxy_error = values[:, 7]
    wz_error = values[:, 8]
    resets = int(values[:, 9].sum())
    return {
        "samples": len(rows),
        "duration_s": float(values[-1, 0]),
        "vxy_error_mean_mps": float(vxy_error.mean()),
        "vxy_error_median_mps": float(np.median(vxy_error)),
        "vxy_error_p05_mps": float(np.percentile(vxy_error, 5)),
        "vxy_error_p95_mps": float(np.percentile(vxy_error, 95)),
        "vxy_error_std_mps": float(vxy_error.std()),
        "vxy_error_max_mps": float(vxy_error.max()),
        "wz_error_mean_radps": float(wz_error.mean()),
        "wz_error_median_radps": float(np.median(wz_error)),
        "wz_error_p05_radps": float(np.percentile(wz_error, 5)),
        "wz_error_p95_radps": float(np.percentile(wz_error, 95)),
        "wz_error_std_radps": float(wz_error.std()),
        "wz_error_max_radps": float(wz_error.max()),
        "resets": resets,
        "failures": resets,
        "min_base_height_m": float(values[:, 10].min()),
        "min_gravity_x_b": float(values[:, 11].min()),
    }


def upload_to_wandb(
    *, project: str, entity: str | None, run_name: str, config: dict, summary: dict,
    artifact_paths: list[Path], plot_paths: list[Path], artifact_name: str,
) -> str | None:
    """Upload a completed evaluation while keeping local results authoritative."""
    try:
        import wandb

        run = wandb.init(project=project, entity=entity, name=run_name, config=config, job_type="sim-to-sim-eval")
        run.summary.update(summary)
        image_log = {f"plots/{path.stem}": wandb.Image(str(path)) for path in plot_paths}
        if image_log:
            run.log(image_log)
        artifact = wandb.Artifact(artifact_name, type="sim-to-sim-evaluation", metadata=summary)
        for path in artifact_paths:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        run_url = run.url
        run.finish()
        return run_url
    except Exception as exc:
        print(f"[WARN] W&B upload failed; local artifacts are complete: {exc}", flush=True)
        return None


def launch_viewer(sim: Solo12Mujoco, cmd_state, force_state, camera, show_viewer_ui: bool = False):
    prepare_viewer_environment()
    from mujoco import viewer as mujoco_viewer
    import interactive

    return mujoco_viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=interactive.make_key_callback(cmd_state, force_state, camera),
        show_left_ui=show_viewer_ui,
        show_right_ui=show_viewer_ui,
    )


def run_live(args, sim: Solo12Mujoco, policy: Policy, command_ranges) -> None:
    """Drive the robot with sliders/keyboard, mirroring the Isaac live command + force UI."""
    import interactive

    cmd_state = interactive.LiveCommandState(*args.cmd_init)
    force_state = interactive.BodyForceState(dt=POLICY_DT)
    camera = interactive.FollowCamera(dt=POLICY_DT, mode=args.camera)
    viewer = launch_viewer(sim, cmd_state, force_state, camera, args.show_viewer_ui)
    panel = interactive.start_control_panel(cmd_state, force_state, camera, command_ranges, args.force_ui_max)
    print(f"[INFO] Live mode: {'slider panel + ' if panel else ''}{interactive.KEYBOARD_HELP}", flush=True)
    print("[INFO] Live mode paces at wall-clock speed; close the viewer window to stop.", flush=True)

    command = cmd_state.get_clipped(command_ranges)
    observation = sim.observation(command)
    episode_elapsed = 0.0
    for _ in range(int(round(args.duration_s / POLICY_DT))):
        wall_start = time.perf_counter()
        command = cmd_state.get_clipped(command_ranges)
        if cmd_state.consume_reset_request():
            print("[INFO] Manual reset requested; returning the robot to the reset pose.", flush=True)
            sim.reset(); episode_elapsed = 0.0
            observation = sim.observation(command)
        active_force = force_state.get_active_force()
        interactive.apply_body_force(sim, active_force)
        action = policy(observation)
        sim.step(action)
        episode_elapsed += POLICY_DT
        if sim.base_hit_ground() or episode_elapsed >= args.episode_length_s:
            print(f"[INFO] Reset ({'base contact' if sim.base_hit_ground() else 'timeout'})", flush=True)
            sim.reset(); episode_elapsed = 0.0
        observation = sim.observation(command)
        camera.update(sim.data.qpos[:3], sim.data.qpos[3:7])
        with viewer.lock():
            camera.apply(viewer.cam)
            interactive.update_force_arrow(viewer.user_scn, sim, active_force)
        viewer.sync()
        if not viewer.is_running():
            break
        time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - wall_start)))
    viewer.close()
    print("[INFO] Live session finished; no artifacts are recorded in live mode.", flush=True)


def run_tracking(args, sim: Solo12Mujoco, policy: Policy, checkpoint: Path, model_path: Path,
                 env_overrides: dict, ignored_overrides: list[str]) -> None:
    commands = args.track_cmds
    duration = len(commands) * COMMAND_DURATION_S
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir or Path("logs/mujoco/cmd_tracking")
    output_dir = create_run_directory(output_root, checkpoint.stem, timestamp)
    config = {
        "task": args.task,
        "simulator": "mujoco",
        "mujoco_version": mujoco.__version__,
        "started_at": started_at.isoformat(),
        "timestamp": timestamp,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "model_xml": str(model_path),
        "visual_meshes_sha256": visual_meshes_sha256(model_path),
        "git_revision": git_revision(),
        "commands": [list(command) for command in commands],
        "command_duration_s": COMMAND_DURATION_S,
        "duration_s": duration,
        "episode_length_s": args.episode_length_s,
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "policy_dt": POLICY_DT,
        "action_scale": ACTION_SCALE,
        "kp": sim.kp,
        "kd": sim.kd,
        "effort_limit_nm": EFFORT_LIMIT,
        "model_mass_kg": float(sim.model.body_mass.sum()),
        "headless": args.headless,
        "realtime": args.realtime,
        "env_overrides": {key: list(value) if isinstance(value, tuple) else value
                          for key, value in env_overrides.items()},
        "ignored_isaaclab_overrides": ignored_overrides,
    }
    config_path = output_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Running {len(commands)} commands for {duration:g} s; output={output_dir}")
    viewer = None
    camera = None
    cmd_state = None
    if not args.headless:
        import interactive

        cmd_state = interactive.LiveCommandState()
        camera = interactive.FollowCamera(dt=POLICY_DT, mode=args.camera)
        viewer = launch_viewer(sim, cmd_state, None, camera, args.show_viewer_ui)
        print(f"[INFO] Viewer keys: R=reset robot, C=cycle camera (starting on '{camera.mode}').", flush=True)
    rows = []
    episode_elapsed = 0.0
    reset_next_sample = False
    # Match play_direct_0325.py: the observation served to the policy carries the command from
    # the previous loop iteration (the first observation carries commands[0]).
    observation = sim.observation(commands[0])
    for step in range(int(round(duration / POLICY_DT))):
        t = step * POLICY_DT
        command = commands[min(int(t / COMMAND_DURATION_S), len(commands) - 1)]
        wall_start = time.perf_counter()
        if cmd_state is not None and cmd_state.consume_reset_request():
            print("[INFO] Manual reset requested; returning the robot to the reset pose.", flush=True)
            sim.reset(); episode_elapsed = 0.0; reset_next_sample = True
            observation = sim.observation(command)
        action = policy(observation)
        sim.step(action)
        episode_elapsed += POLICY_DT
        vx, vy, wz = sim.tracked_velocity()
        rows.append((t + POLICY_DT, *command, vx, vy, wz, math.hypot(command[0]-vx, command[1]-vy),
                     abs(command[2]-wz), int(reset_next_sample), sim.base_height(), sim.gravity_x_b()))
        reset_next_sample = False
        if sim.base_hit_ground() or episode_elapsed >= args.episode_length_s:
            print(f"[INFO] Reset at t={t + POLICY_DT:.3f} s ({'base contact' if sim.base_hit_ground() else 'timeout'})")
            sim.reset(); episode_elapsed = 0.0; reset_next_sample = True
        observation = sim.observation(command)
        if viewer is not None:
            camera.update(sim.data.qpos[:3], sim.data.qpos[3:7])
            with viewer.lock():
                camera.apply(viewer.cam)
            viewer.sync()
            if not viewer.is_running(): break
        if args.realtime:
            time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - wall_start)))
    if viewer is not None: viewer.close()
    result_paths = save_results(output_dir, rows, commands)
    summary = summarize_rows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for path in [config_path, summary_path, *result_paths]:
        print(f"[INFO] Saved {path}")
    if not args.no_wandb:
        run_name = args.wandb_name or f"mujoco-{checkpoint.stem}-{timestamp}"
        run_url = upload_to_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=run_name,
            config=config,
            summary=summary,
            artifact_paths=[config_path, summary_path, model_path, *result_paths],
            plot_paths=[path for path in result_paths if path.suffix == ".png"],
            artifact_name=f"{checkpoint.stem}-{timestamp}",
        )
        if run_url:
            print(f"[INFO] W&B run: {run_url}")


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if args.task != "solo12-two-feet" or args.num_envs != 1:
        raise ValueError("MuJoCo sim-to-sim currently supports --task=solo12-two-feet --num_envs=1 only")
    env_overrides, ignored = consume_env_overrides(unknown)
    if env_overrides:
        print("[INFO] Applying overrides:", " ".join(f"{k}={v}" for k, v in env_overrides.items()), flush=True)
    if ignored:
        print("[INFO] Ignoring IsaacLab-only overrides:", " ".join(ignored), flush=True)
    if args.track_cmds is None and args.headless:
        raise SystemExit("Live slider mode needs the viewer; pass --track-cmds for headless runs.")

    kp = args.kp if args.kp is not None else env_overrides.get("kp", DEFAULT_KP)
    kd = args.kd if args.kd is not None else env_overrides.get("kd", DEFAULT_KD)
    for name, cli_value in (("kp", args.kp), ("kd", args.kd)):
        env_value = env_overrides.get(name)
        if cli_value is not None and env_value is not None:
            if math.isclose(cli_value, env_value):
                print(f"[INFO] Duplicate {name}: --{name} and env.{name} both request {cli_value:g}.", flush=True)
            else:
                print(
                    f"[WARN] Conflicting {name}: --{name}={cli_value:g} overrides env.{name}={env_value:g}.",
                    flush=True,
                )
    command_ranges = tuple(
        env_overrides.get(key.removeprefix("env."), DEFAULT_COMMAND_RANGES[index])
        for key, index in _ENV_RANGE_KEYS.items()
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    policy = Policy(checkpoint)
    model_path = Path(__file__).with_name("solo12.xml").resolve()
    sim = Solo12Mujoco(model_path, kp=kp, kd=kd)
    print(f"[INFO] MuJoCo {mujoco.__version__}; model mass={sim.model.body_mass.sum():.6f} kg")
    print(f"[INFO] PD gains kp={sim.kp:g} kd={sim.kd:g} "
          f"(Isaac play w/ --disable_training_gain_sync: {DEFAULT_KP:g}/{DEFAULT_KD:g}; "
          f"training run: {TRAINING_KP:g}/{TRAINING_KD:g})", flush=True)

    if args.track_cmds is None:
        run_live(args, sim, policy, command_ranges)
    else:
        run_tracking(args, sim, policy, checkpoint, model_path, env_overrides, ignored)


if __name__ == "__main__":
    main()
