#!/usr/bin/env python3
"""Run an IsaacLab Solo12 RSL-RL policy in MuJoCo for sim-to-sim evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch


PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
KP = 9.0
KD = 0.2
EFFORT_LIMIT = 2.65
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
    parser.add_argument("--track-cmds", required=True, type=parse_commands)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration_s", type=float, default=2000.0)
    parser.add_argument("--episode_length_s", type=float, default=80.0)
    parser.add_argument("--cmd_init", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--realtime", action="store_true", help="Pace the interactive viewer at wall-clock speed.")
    parser.add_argument("--task", default="solo12-two-feet")
    parser.add_argument("--num_envs", type=int, default=1)
    return parser


class Policy(torch.nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        dims = (48, 256, 128, 64, 12)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        for index, layer in enumerate(self.layers):
            state_index = 2 * index
            layer.weight.data.copy_(state[f"actor.{state_index}.weight"])
            layer.bias.data.copy_(state[f"actor.{state_index}.bias"])
        self.register_buffer("obs_mean", state["actor_obs_normalizer._mean"].reshape(-1))
        self.register_buffer("obs_std", state["actor_obs_normalizer._std"].reshape(-1))
        self.eval()

    @torch.inference_mode()
    def forward(self, observation: np.ndarray) -> np.ndarray:
        x = (torch.from_numpy(observation).float() - self.obs_mean) / self.obs_std.clamp_min(1.0e-8)
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
    def __init__(self, model_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.base_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision")
        self.action = np.zeros(12)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = (0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0)  # yaw = pi
        self.data.qpos[self.joint_qpos] = SAFE_Q
        self.action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

    def observation(self, command: tuple[float, float, float]) -> np.ndarray:
        rotation_wb = quat_rotation(self.data.qpos[3:7])
        velocity_b = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity_b, 1
        )
        ang_vel_b, lin_vel_b = velocity_b[:3], velocity_b[3:]
        gravity_b = rotation_wb.T @ np.array((0.0, 0.0, -1.0))
        q = self.data.qpos[self.joint_qpos]
        qd = self.data.qvel[self.joint_dof]
        return np.concatenate((lin_vel_b, ang_vel_b, gravity_b, command, q - SAFE_Q, qd, self.action))

    def tracked_velocity(self) -> tuple[float, float, float]:
        rotation_wb = quat_rotation(self.data.qpos[3:7])
        lateral = rotation_wb[:2, 1]
        norm = np.linalg.norm(lateral)
        if norm < np.finfo(float).eps:
            lateral = np.array((0.0, 1.0))
        else:
            lateral /= norm
        forward = np.array((lateral[1], -lateral[0]))
        velocity_w = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity_w, 0
        )
        velocity_xy = velocity_w[3:5]
        return float(velocity_xy @ forward), float(velocity_xy @ lateral), float(velocity_w[2])

    def step(self, action: np.ndarray) -> None:
        self.action = np.asarray(action).copy()
        target = SAFE_Q + ACTION_SCALE * self.action
        for _ in range(DECIMATION):
            q = self.data.qpos[self.joint_qpos]
            qd = self.data.qvel[self.joint_dof]
            torque = np.clip(KP * (target - q) - KD * qd, -EFFORT_LIMIT, EFFORT_LIMIT)
            self.data.ctrl[self.actuator_ids] = torque
            mujoco.mj_step(self.model, self.data)

    def base_hit_ground(self) -> bool:
        for contact in self.data.contact:
            if {contact.geom1, contact.geom2} == {self.ground_id, self.base_geom_id}:
                return True
        return False


def save_results(output_dir: Path, rows: list[tuple], commands: list[tuple[float, float, float]]) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "command_tracking.csv"
    header = ("time_s", "cmd_vx", "cmd_vy", "cmd_wz", "velocity_vx", "velocity_vy", "yaw_rate_wz", "vxy_error_norm", "wz_error_abs", "reset")
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


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if args.task != "solo12-two-feet" or args.num_envs != 1:
        raise ValueError("MuJoCo sim-to-sim currently supports --task=solo12-two-feet --num_envs=1 only")
    if unknown:
        print("[INFO] Ignoring IsaacLab-only overrides:", " ".join(unknown), flush=True)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    commands = args.track_cmds
    duration = len(commands) * COMMAND_DURATION_S
    output_dir = args.output_dir or Path("logs/mujoco/cmd_tracking") / checkpoint.stem
    policy = Policy(checkpoint)
    sim = Solo12Mujoco(Path(__file__).with_name("solo12.xml"))
    print(f"[INFO] MuJoCo {mujoco.__version__}; model mass={sim.model.body_mass.sum():.6f} kg")
    print(f"[INFO] Running {len(commands)} commands for {duration:g} s; output={output_dir}")
    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(sim.model, sim.data)
    rows = []
    episode_elapsed = 0.0
    reset_next_sample = False
    for step in range(int(round(duration / POLICY_DT))):
        t = step * POLICY_DT
        command = commands[min(int(t / COMMAND_DURATION_S), len(commands) - 1)]
        wall_start = time.perf_counter()
        action = policy(sim.observation(command))
        sim.step(action)
        episode_elapsed += POLICY_DT
        vx, vy, wz = sim.tracked_velocity()
        rows.append((t + POLICY_DT, *command, vx, vy, wz, math.hypot(command[0]-vx, command[1]-vy), abs(command[2]-wz), int(reset_next_sample)))
        reset_next_sample = False
        if sim.base_hit_ground() or episode_elapsed >= args.episode_length_s:
            print(f"[INFO] Reset at t={t + POLICY_DT:.3f} s ({'base contact' if sim.base_hit_ground() else 'timeout'})")
            sim.reset(); episode_elapsed = 0.0; reset_next_sample = True
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running(): break
        if args.realtime:
            time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - wall_start)))
    if viewer is not None: viewer.close()
    for path in save_results(output_dir, rows, commands): print(f"[INFO] Saved {path}")


if __name__ == "__main__":
    main()
