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

from solo12_imu_policy import DEFAULT_CONTRACT, TASK_MODELS, EncodedPolicy


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
IMU_OFFSET_B = np.array((0.01, 0.0, 0.0))  # base-link origin, not COM
IMU_GYRO_CLIP = math.radians(900.0)
IMU_EKF_ACC_CLIP = 32767.0 / 2048.0


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
    parser.add_argument("--spawn-z", type=float, default=0.35,
                        help="Base height used for the initial spawn and every reset [m] (default: 0.35).")
    parser.add_argument("--output-dir", type=Path,
                        help="Output root (default logs/mujoco/cmd_tracking); model/timestamp are appended.")
    parser.add_argument("--realtime", action="store_true", help="Pace the interactive viewer at wall-clock speed.")
    parser.add_argument("--task", choices=("solo12-two-feet", "solo12-v0", *TASK_MODELS), default="solo12-two-feet")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Local frozen teacher for a DAgger adapter whose saved teacher path moved.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--kp", type=float, default=None,
                        help=f"PD stiffness (default: env.kp override, else {DEFAULT_KP:g} for two feet, 9 for solo12-v0).")
    parser.add_argument("--kd", type=float, default=None,
                        help=f"PD damping (default: env.kd override, else {DEFAULT_KD:g} for two feet, 0.2 for solo12-v0).")
    parser.add_argument("--bringup_clip_rad", "--bringup-clip-rad", type=float, default=0.0,
                        help="Clip q_des - q before position PD [rad]; 0 disables clipping.")
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
_ENV_JOINT_LIMIT_KEYS = {
    f"env.joint_physical_limit_{name}"
    for name in ("hip", "thigh", "calf", "front_thigh", "rear_thigh", "front_calf", "rear_calf")
}
_ENV_ASYMMETRIC_LIMIT_KEYS = {"env.use_asymmetric_thigh_limits", "env.use_asymmetric_calf_limits"}
_ENV_OBSERVATION_BOOL_KEYS = {
    "env.base_obs_from_imu", "env.remove_root_lin_vel_b_from_actor_obs",
    "env.remove_root_lin_vel_b_from_obs", "env.base_imu_clip", "env.imu_ekf_processed_inputs",
    "env.include_foot_height_obs", "env.teacher_include_base_force", "env.apply_tanh_to_latent",
    "env.use_rotMat_on_imu_encoder", "env.action_obs_is_last_executed", "env.base_imu_history_action_is_last_executed",
}
_ENV_IMU_CLIP_KEYS = {"env.base_imu_gyro_clip", "env.base_imu_ekf_acc_clip"}


def consume_env_overrides(unknown: list[str]) -> tuple[dict, list[str]]:
    """Pull supported actuator, observation and command settings out of the Isaac CLI tail."""
    import ast

    overrides: dict = {}
    ignored: list[str] = []
    for token in unknown:
        key, _, raw = token.partition("=")
        if key in _ENV_JOINT_LIMIT_KEYS:
            try:
                low, high = map(float, ast.literal_eval(raw))
                if not (math.isfinite(low) and math.isfinite(high) and low < high):
                    raise ValueError
            except (TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"{key} requires finite [lower, upper] degrees with lower < upper.") from exc
            overrides[key.removeprefix("env.")] = (low, high)
            continue
        if key in _ENV_ASYMMETRIC_LIMIT_KEYS | _ENV_OBSERVATION_BOOL_KEYS:
            if raw.lower() not in ("true", "false"):
                raise ValueError(f"{key} requires True or False.")
            overrides[key.removeprefix("env.")] = raw.lower() == "true"
            continue
        if key in ("env.include_lin_acc_steps_in_imu_obs", "env.base_imu_history_policy_steps"):
            try:
                steps = int(raw)
                if steps < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"{key} requires a non-negative integer.") from exc
            overrides[key.removeprefix("env.")] = steps
            continue
        if key in ("env.teacher_foot_force_obs", "env.base_imu_tcn_activation"):
            overrides[key.removeprefix("env.")] = raw.strip("'\"")
            continue
        if key == "agent.policy.activation":
            overrides["policy_activation"] = raw.strip("'\"")
            continue
        if key in _ENV_IMU_CLIP_KEYS:
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} requires a finite positive value.")
            overrides[key.removeprefix("env.")] = value
            continue
        try:
            if key in ("env.kp", "env.kd", "env.effort_limit_sim"):
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
    """Actor-only inference; SAC critic type, bins, and symlog support do not affect loading."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in payload:
            state = payload["model_state_dict"]
            layer_prefix = "actor."
            normalizer_prefix = "actor_obs_normalizer"
            self.checkpoint_format = "ppo"
            self.activation = torch.nn.functional.elu
        elif "actor_state_dict" in payload and "action_range" in payload["actor_state_dict"]:
            state = payload["actor_state_dict"]
            layer_prefix = "mlp."
            normalizer_prefix = "obs_normalizer"
            self.checkpoint_format = "sac"
            self.activation = torch.nn.functional.silu
        else:
            raise ValueError(
                f"Unsupported checkpoint layout in {checkpoint}: expected PPO model_state_dict "
                "or SAC actor_state_dict."
            )
        weight_keys = sorted(
            (key for key in state if key.startswith(layer_prefix) and key.endswith(".weight")),
            key=lambda key: int(key.split(".")[1]),
        )
        self.layers = torch.nn.ModuleList()
        for key in weight_keys:
            out_dim, in_dim = state[key].shape
            layer = torch.nn.Linear(in_dim, out_dim)
            layer.weight.data.copy_(state[key])
            layer.bias.data.copy_(state[key.replace(".weight", ".bias")])
            self.layers.append(layer)
        self.register_buffer("obs_mean", state[f"{normalizer_prefix}._mean"].reshape(-1))
        self.register_buffer("obs_std", state[f"{normalizer_prefix}._std"].reshape(-1))
        self.input_dim = self.obs_mean.numel()
        if self.checkpoint_format == "sac":
            self.register_buffer("action_bias", state["action_bias"].reshape(-1))
            self.register_buffer("action_range", state["action_range"].reshape(-1))
        self.eval()

    @torch.inference_mode()
    def forward(self, observation: np.ndarray) -> np.ndarray:
        x = (torch.from_numpy(observation).float() - self.obs_mean) / (self.obs_std + OBS_NORM_EPS)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        output = self.layers[-1](x)
        if self.checkpoint_format == "sac":
            mean = output[..., : self.action_range.numel()]
            output = self.action_range * torch.tanh(mean) + self.action_bias
        return output.numpy()


def quat_rotation(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ))


class Solo12Mujoco:
    def __init__(self, model_path: Path, kp: float = DEFAULT_KP, kd: float = DEFAULT_KD,
                 spawn_z: float = 0.35, env_overrides: dict | None = None, bringup_clip_rad: float = 0.0,
                 policy_model: str = "simple_mlp"):
        if not math.isfinite(bringup_clip_rad) or bringup_clip_rad < 0.0:
            raise ValueError("bringup_clip_rad must be finite and non-negative (0 disables clipping).")
        self.bringup_clip_rad = float(bringup_clip_rad)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.joint_dof = np.array([self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self.actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        # Keep legacy XML limits unless explicitly overridden. SAC's saved action
        # scaling already encodes the training soft limits; do not rescale the actor.
        self.policy_model = policy_model
        self.encoded_imu = policy_model in TASK_MODELS.values()
        overrides = {**(DEFAULT_CONTRACT if self.encoded_imu else {}), **(env_overrides or {})}
        self.base_obs_from_imu = overrides.get("base_obs_from_imu", False)
        self.include_root_lin_vel = not (
            overrides.get("remove_root_lin_vel_b_from_actor_obs", False)
            or overrides.get("remove_root_lin_vel_b_from_obs", False)
        )
        self.imu_acc_steps = overrides.get("include_lin_acc_steps_in_imu_obs", 0)
        if self.imu_acc_steps and not (self.base_obs_from_imu or self.encoded_imu):
            raise ValueError("env.include_lin_acc_steps_in_imu_obs requires env.base_obs_from_imu=True.")
        if self.base_obs_from_imu and not overrides.get("imu_ekf_processed_inputs", True):
            raise ValueError("IMU playback requires env.imu_ekf_processed_inputs=True (gravity-free acceleration).")
        self.base_imu_clip = overrides.get("base_imu_clip", True)
        self.base_imu_gyro_clip = overrides.get("base_imu_gyro_clip", IMU_GYRO_CLIP)
        self.base_imu_ekf_acc_clip = overrides.get("base_imu_ekf_acc_clip", IMU_EKF_ACC_CLIP)
        self._imu_lin_acc_history = np.zeros((self.imu_acc_steps, 3))
        self._teacher_lin_acc_history = np.zeros((self.imu_acc_steps if self.encoded_imu else 0, 3))
        self.history_len = overrides.get("history_len", 13) if policy_model in ("base_imu_student_rl", "base_imu_student_dagger") else 0
        self._base_imu_history = np.zeros((self.history_len, 45))
        self.include_foot_height_obs = overrides.get("include_foot_height_obs", False)
        self.teacher_include_base_force = overrides.get("teacher_include_base_force", False)
        self.teacher_foot_force_obs = overrides.get("teacher_foot_force_obs", "none")
        self._base_push_force_b = np.zeros(3)
        self._calf_forces_w = np.zeros((4, 3))
        self.effort_limit = float(overrides.get("effort_limit_sim", EFFORT_LIMIT))
        if not math.isfinite(self.effort_limit) or self.effort_limit <= 0.0:
            raise ValueError("env.effort_limit_sim must be finite and positive.")
        for name, actuator_id in zip(JOINT_NAMES, self.actuator_ids):
            joint_type = name.split("_")[1]
            if overrides.get(f"use_asymmetric_{joint_type}_limits", False):
                side = "front" if name.startswith(("FL_", "FR_")) else "rear"
                joint_type = f"{side}_{joint_type}"
            limits = overrides.get(f"joint_physical_limit_{joint_type}")
            if limits is not None:
                joint_id = self.model.actuator_trnid[actuator_id, 0]
                self.model.jnt_range[joint_id] = np.deg2rad(limits)
                self.model.jnt_limited[joint_id] = True
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.calf_body_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
                                       for leg in ("FL", "FR", "RL", "RR")])
        self._observation_data = mujoco.MjData(self.model) if self.encoded_imu and self.include_foot_height_obs else None
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.base_geom_ids = {
            geom_id for geom_id in range(self.model.ngeom)
            if self.model.geom_bodyid[geom_id] == self.base_id and self.model.geom_contype[geom_id] != 0
        }
        self.set_gains(kp, kd)
        self.spawn_z = float(spawn_z)
        self.action = np.zeros(12)
        self._base_contact_latched = False
        self.reset()

    def set_gains(self, kp: float, kd: float) -> None:
        """Mirror PhysX implicit PD drives: position servo with implicit damping, force-limited."""
        self.kp, self.kd = float(kp), float(kd)
        self.model.actuator_gainprm[self.actuator_ids, 0] = self.kp
        self.model.actuator_biasprm[self.actuator_ids, 1] = -self.kp
        self.model.actuator_biasprm[self.actuator_ids, 2] = -self.kd
        self.model.actuator_forcerange[self.actuator_ids, 0] = -self.effort_limit
        self.model.actuator_forcerange[self.actuator_ids, 1] = self.effort_limit

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = (0.0, 0.0, self.spawn_z, 0.0, 0.0, 0.0, 1.0)
        self.data.qpos[self.joint_qpos] = SAFE_Q
        self.data.ctrl[self.actuator_ids] = SAFE_Q
        self.action.fill(0.0)
        self._base_contact_latched = False
        mujoco.mj_forward(self.model, self.data)
        self._imu_lin_acc_history.fill(0.0)
        self._teacher_lin_acc_history.fill(0.0)
        self._base_imu_history.fill(0.0)
        self._base_push_force_b.fill(0.0)
        self._calf_forces_w.fill(0.0)
        _, angular, linear = self._base_imu_kinematics()
        self._previous_imu_com_velocity = linear.copy()
        self._previous_imu_ang_velocity = angular.copy()

    def _base_imu_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Endpoint rotation and world COM velocities without changing MuJoCo's dynamics caches.

        Free-joint translation velocity is world-aligned at the link origin; angular
        velocity is link-aligned. mj_step's derived body caches still describe the
        pre-integration state, so read the integrated qpos/qvel for IMU sampling.
        """
        rotation = quat_rotation(self.data.qpos[3:7])
        angular = rotation @ self.data.qvel[3:6]
        linear = self.data.qvel[:3] + np.cross(angular, rotation @ self.model.body_ipos[self.base_id])
        return rotation, angular, linear

    def _record_base_imu_sample(self) -> None:
        """Record clean gravity-free IMU acceleration after each 5 ms physics step."""
        if not self.encoded_imu and (not self.base_obs_from_imu or not self.imu_acc_steps):
            return
        rotation, angular, linear = self._base_imu_kinematics()
        com_acceleration = (linear - self._previous_imu_com_velocity) / PHYSICS_DT
        angular_acceleration = (angular - self._previous_imu_ang_velocity) / PHYSICS_DT
        lever = rotation @ (IMU_OFFSET_B - self.model.body_ipos[self.base_id])
        acceleration = com_acceleration + np.cross(angular_acceleration, lever)
        acceleration += np.cross(angular, np.cross(angular, lever))
        acceleration_b = rotation.T @ acceleration
        if self._teacher_lin_acc_history.size:
            origin_lever = -rotation @ self.model.body_ipos[self.base_id]
            origin_acc = com_acceleration + np.cross(angular_acceleration, origin_lever)
            origin_acc += np.cross(angular, np.cross(angular, origin_lever))
            self._teacher_lin_acc_history[:-1] = self._teacher_lin_acc_history[1:]
            self._teacher_lin_acc_history[-1] = rotation.T @ origin_acc
        if self.base_imu_clip:
            acceleration_b = acceleration_b.clip(-self.base_imu_ekf_acc_clip, self.base_imu_ekf_acc_clip)
        if self.base_obs_from_imu and self.imu_acc_steps:
            self._imu_lin_acc_history[:-1] = self._imu_lin_acc_history[1:]
            self._imu_lin_acc_history[-1] = acceleration_b
        if self.history_len:
            gyro = rotation.T @ angular
            if self.base_imu_clip:
                gyro = gyro.clip(-self.base_imu_gyro_clip, self.base_imu_gyro_clip)
            sample = np.concatenate((self.data.qpos[self.joint_qpos] - SAFE_Q, self.data.qvel[self.joint_dof],
                                     gyro, acceleration_b, rotation.T @ np.array((0., 0., -1.)), self.action))
            self._base_imu_history[:-1] = self._base_imu_history[1:]
            self._base_imu_history[-1] = sample
        self._previous_imu_com_velocity = linear.copy()
        self._previous_imu_ang_velocity = angular.copy()

    def _base_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame (angular, linear-at-COM) base velocity, matching PhysX root com velocities."""
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
        return velocity[:3], velocity[3:]

    def observation(self, command: tuple[float, float, float]) -> np.ndarray:
        if self.history_len:
            return np.concatenate((self._base_imu_history.reshape(-1), command))
        if self.policy_model == "base_imu_teacher":
            return self._teacher_observation(command)
        # IsaacLab contract: root_com_{lin,ang}_vel_b = R_link^T @ world COM velocity;
        # mjOBJ_BODY with flg_local=1 would use the inertial (ximat) frame instead -- do not use it.
        rotation_wb = quat_rotation(self.data.qpos[3:7])
        ang_vel_w, lin_vel_w = self._base_velocity_world()
        lin_vel_b = rotation_wb.T @ lin_vel_w
        ang_vel_b = rotation_wb.T @ ang_vel_w
        gravity_b = rotation_wb.T @ np.array((0.0, 0.0, -1.0))
        q = self.data.qpos[self.joint_qpos]
        qd = self.data.qvel[self.joint_dof]
        if self.base_obs_from_imu:
            rotation_wb, angular, linear = self._base_imu_kinematics()
            lin_vel_b = rotation_wb.T @ linear
            ang_vel_b = rotation_wb.T @ angular
            if self.base_imu_clip:
                ang_vel_b = ang_vel_b.clip(-self.base_imu_gyro_clip, self.base_imu_gyro_clip)
        terms = [lin_vel_b] if self.include_root_lin_vel else []
        terms.extend((ang_vel_b, gravity_b))
        if self.base_obs_from_imu:
            terms.append(self._imu_lin_acc_history.reshape(-1))
        terms.extend((command, q - SAFE_Q, qd, self.action))
        return np.concatenate(terms)

    def _teacher_observation(self, command) -> np.ndarray:
        rotation, angular, linear = self._base_imu_kinematics()
        terms = [rotation.T @ np.array((0., 0., -1.)), rotation.T @ linear, rotation.T @ angular]
        if self.include_foot_height_obs:
            # Query endpoint link positions separately; preserve the completed-step contact/dynamics caches.
            self._observation_data.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, self._observation_data)
            terms.append(self._observation_data.xpos[self.calf_body_ids, 2].copy())
        terms.extend((self.action, self.data.qpos[self.joint_qpos] - SAFE_Q, self.data.qvel[self.joint_dof],
                      self._teacher_lin_acc_history.reshape(-1)))
        if self.teacher_include_base_force:
            terms.append(self._base_push_force_b)
        if self.teacher_foot_force_obs == "norm":
            terms.append(np.linalg.norm(self._calf_forces_w, axis=-1))
        elif self.teacher_foot_force_obs == "vector":
            terms.append((self._calf_forces_w @ rotation).reshape(-1))
        return np.concatenate((*terms, command))

    def _record_calf_contacts(self) -> None:
        """Net completed-step contact forces on calf bodies, including non-ground contacts."""
        self._calf_forces_w.fill(0.)
        for index, contact in enumerate(self.data.contact):
            body1, body2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if body1 not in self.calf_body_ids and body2 not in self.calf_body_ids:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force_w = contact.frame.reshape(3, 3).T @ wrench[:3]
            for foot, body in enumerate(self.calf_body_ids):
                if body == body1:
                    self._calf_forces_w[foot] -= force_w
                if body == body2:
                    self._calf_forces_w[foot] += force_w

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
        if self.encoded_imu:
            self._base_push_force_b = quat_rotation(self.data.qpos[3:7]).T @ (
                self.data.qfrc_applied[:3] + self.data.xfrc_applied[self.base_id, :3])
        self.action = np.asarray(action).copy()
        requested_target = SAFE_Q + ACTION_SCALE * self.action
        self.data.ctrl[self.actuator_ids] = requested_target
        self._base_contact_latched = False
        for _ in range(DECIMATION):
            if self.bringup_clip_rad > 0.0:
                q = self.data.qpos[self.joint_qpos]
                self.data.ctrl[self.actuator_ids] = q + np.clip(
                    requested_target - q, -self.bringup_clip_rad, self.bringup_clip_rad
                )
            mujoco.mj_step(self.model, self.data)
            self._record_base_imu_sample()
            self._base_contact_latched |= self._base_hit_ground_now()
        if self.policy_model == "base_imu_teacher":
            self._record_calf_contacts()

    def _base_hit_ground_now(self) -> bool:
        for contact in self.data.contact:
            pair = {contact.geom1, contact.geom2}
            if self.ground_id in pair and pair & self.base_geom_ids:
                return True
        return False

    def base_hit_ground(self) -> bool:
        """Return whether the base touched ground during the latest policy step."""
        return self._base_contact_latched or self._base_hit_ground_now()

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
    values = np.asarray(rows, dtype=float)
    signed_xy_error = values[:, 4:6] - values[:, 1:3]
    signed_wz_error = values[:, 6] - values[:, 3]
    command_norm = np.linalg.norm(values[:, 1:3], axis=1)
    along_command_error = np.divide(
        np.sum(signed_xy_error * values[:, 1:3], axis=1),
        command_norm,
        out=np.zeros_like(command_norm),
        where=command_norm > 1.0e-9,
    )
    csv_rows = np.column_stack((values, signed_xy_error, signed_wz_error, along_command_error))
    csv_path = output_dir / "command_tracking.csv"
    header = ("time_s", "cmd_vx", "cmd_vy", "cmd_wz", "velocity_vx", "velocity_vy", "yaw_rate_wz",
              "vxy_error_norm", "wz_error_abs", "reset", "base_height_m", "gravity_x_b",
              "vx_error_signed", "vy_error_signed", "wz_error_signed", "vxy_error_along_command")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(csv_rows)
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

    def save_signed_error_plot(series, ylabel, title, filename, color):
        fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.4), constrained_layout=True)
        ax.plot(times, series, color=color, linewidth=1.5); ax.axhline(0.0, color="0.2", linewidth=1.0)
        decorate(ax); ax.set(xlabel="Time [s]", ylabel=ylabel, title=title)
        path = output_dir / filename; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    save_signed_error_plot(signed_xy_error[:, 0], r"$v_x-v_x^{cmd}$ [m/s]",
                           "Signed x-velocity tracking error (base-footprint frame)", "vx_tracking_error_signed.png", "#d55e00")
    save_signed_error_plot(signed_xy_error[:, 1], r"$v_y-v_y^{cmd}$ [m/s]",
                           "Signed y-velocity tracking error (base-footprint frame)", "vy_tracking_error_signed.png", "#009e73")
    save_signed_error_plot(along_command_error, r"$(v_{xy}-v_{xy}^{cmd})\cdot\hat{v}_{xy}^{cmd}$ [m/s]",
                           "Planar tracking error projected along command direction", "vxy_tracking_error_along_command.png", "#cc79a7")
    save_signed_error_plot(signed_wz_error, r"$\omega_z-\omega_z^{cmd}$ [rad/s]",
                           "Signed yaw-rate tracking error (world z)", "wz_tracking_error_signed.png", "#0072b2")
    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(commands)), 4.4), constrained_layout=True)
    ax.plot(times, values[:, 6], label=r"$\omega_z$", color="#0072b2", linewidth=1.5)
    ax.plot(times, values[:, 3], label=r"$\omega_z^{cmd}$", color="#d55e00", linewidth=1.3, linestyle="--")
    decorate(ax); ax.legend(); ax.set(xlabel="Time [s]", ylabel="Yaw rate [rad/s]",
                                      title="Measured and commanded yaw rate (world z)")
    wz_compare_path = output_dir / "wz_tracking_actual_vs_command.png"
    fig.savefig(wz_compare_path, dpi=180); plt.close(fig); paths.append(wz_compare_path)
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
    signed_xy_error = values[:, 4:6] - values[:, 1:3]
    signed_wz_error = values[:, 6] - values[:, 3]
    command_norm = np.linalg.norm(values[:, 1:3], axis=1)
    along_command_error = np.divide(
        np.sum(signed_xy_error * values[:, 1:3], axis=1), command_norm,
        out=np.zeros_like(command_norm), where=command_norm > 1.0e-9,
    )
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
        "vx_error_signed_mean_mps": float(signed_xy_error[:, 0].mean()),
        "vx_error_signed_std_mps": float(signed_xy_error[:, 0].std()),
        "vy_error_signed_mean_mps": float(signed_xy_error[:, 1].mean()),
        "vy_error_signed_std_mps": float(signed_xy_error[:, 1].std()),
        "wz_error_signed_mean_radps": float(signed_wz_error.mean()),
        "wz_error_signed_std_radps": float(signed_wz_error.std()),
        "vxy_error_along_command_mean_mps": float(along_command_error.mean()),
        "vxy_error_along_command_std_mps": float(along_command_error.std()),
        "resets": resets,
        "failures": resets,
        "min_base_height_m": float(values[:, 10].min()),
        "min_gravity_x_b": float(values[:, 11].min()),
    }


def tracking_history_metrics(row: dict[str, str]) -> dict[str, float]:
    values = {key: float(value) for key, value in row.items()}
    vx_error = values["velocity_vx"] - values["cmd_vx"]
    vy_error = values["velocity_vy"] - values["cmd_vy"]
    wz_error = values["yaw_rate_wz"] - values["cmd_wz"]
    command_norm = math.hypot(values["cmd_vx"], values["cmd_vy"])
    along_command_error = ((vx_error * values["cmd_vx"] + vy_error * values["cmd_vy"]) / command_norm
                           if command_norm > 1.0e-9 else 0.0)
    return {
        "tracking/time_s": values["time_s"],
        "tracking/command_vx_mps": values["cmd_vx"],
        "tracking/command_vy_mps": values["cmd_vy"],
        "tracking/command_wz_radps": values["cmd_wz"],
        "tracking/velocity_vx_mps": values["velocity_vx"],
        "tracking/velocity_vy_mps": values["velocity_vy"],
        "tracking/velocity_wz_radps": values["yaw_rate_wz"],
        "tracking/error_vx_abs_mps": abs(values["cmd_vx"] - values["velocity_vx"]),
        "tracking/error_vy_abs_mps": abs(values["cmd_vy"] - values["velocity_vy"]),
        "tracking/error_vx_signed_mps": vx_error,
        "tracking/error_vy_signed_mps": vy_error,
        "tracking/error_vxy_norm_mps": values["vxy_error_norm"],
        "tracking/error_wz_abs_radps": values["wz_error_abs"],
        "tracking/error_wz_signed_radps": wz_error,
        "tracking/error_vxy_along_command_mps": along_command_error,
        "tracking/reset": values["reset"],
        "tracking/base_height_m": values["base_height_m"],
        "tracking/gravity_x_b": values["gravity_x_b"],
    }


def log_tracking_csv(run, csv_path: Path) -> int:
    """Log every evaluation sample as comparable W&B scalar history."""
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return 0
    metric_names = tracking_history_metrics(rows[0])
    run.define_metric("tracking/time_s")
    for name in metric_names:
        if name != "tracking/time_s":
            run.define_metric(name, step_metric="tracking/time_s")
    for row in rows:
        run.log(tracking_history_metrics(row))
    return len(rows)


def upload_to_wandb(
    *, project: str, entity: str | None, run_name: str, config: dict, summary: dict,
    artifact_paths: list[Path], plot_paths: list[Path], artifact_name: str,
    history_path: Path | None = None,
) -> str | None:
    """Upload a completed evaluation while keeping local results authoritative."""
    try:
        import wandb

        run = wandb.init(project=project, entity=entity, name=run_name, config=config, job_type="sim-to-sim-eval")
        run.summary.update(summary)
        if history_path is not None:
            log_tracking_csv(run, history_path)
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
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "model_xml": str(model_path),
        "visual_meshes_sha256": visual_meshes_sha256(model_path),
        "git_revision": git_revision(),
        "commands": [list(command) for command in commands],
        "command_duration_s": COMMAND_DURATION_S,
        "duration_s": duration,
        "episode_length_s": args.episode_length_s,
        "spawn_z": sim.spawn_z,
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "policy_dt": POLICY_DT,
        "action_scale": ACTION_SCALE,
        "observation_contract": {
            "input_dim": int(policy.input_dim),
            "policy_model": sim.policy_model,
            "saved_contract": getattr(policy, "contract", None),
            "history_len": sim.history_len,
            "history_sample_dim": 45 if sim.history_len else 0,
            "teacher_acceleration": "unclipped gravity-free acceleration at base link origin",
            "teacher_contact_forces": "net MuJoCo calf-body contacts from final physics substep",
            "teacher_checkpoint": getattr(policy, "teacher_checkpoint", None),
            "base_obs_from_imu": sim.base_obs_from_imu,
            "include_root_lin_vel": sim.include_root_lin_vel,
            "include_lin_acc_steps_in_imu_obs": sim.imu_acc_steps,
            "imu_offset_pos_b_m": IMU_OFFSET_B.tolist(),
            "imu_acceleration": "gravity-free; world COM/angular velocity differences at physics_dt + rigid lever arm",
            "base_imu_clip": sim.base_imu_clip,
            "base_imu_gyro_clip_radps": sim.base_imu_gyro_clip,
            "base_imu_ekf_acc_clip_mps2": sim.base_imu_ekf_acc_clip,
            "acceleration_history_order": "oldest first, XYZ per sample; zeroed on reset",
            "noise_enabled": False,
        },
        "kp": sim.kp,
        "kd": sim.kd,
        "effort_limit_nm": sim.effort_limit,
        "bringup_clip_rad": sim.bringup_clip_rad,
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
            history_path=result_paths[0],
        )
        if run_url:
            print(f"[INFO] W&B run: {run_url}")


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if not math.isfinite(args.bringup_clip_rad) or args.bringup_clip_rad < 0.0:
        raise SystemExit("--bringup_clip_rad must be finite and non-negative (0 disables clipping)")
    if args.num_envs != 1:
        raise ValueError("MuJoCo sim-to-sim supports --num_envs=1 only")
    env_overrides, ignored = consume_env_overrides(unknown)
    if env_overrides:
        print("[INFO] Applying overrides:", " ".join(f"{k}={v}" for k, v in env_overrides.items()), flush=True)
    if ignored:
        print("[INFO] Ignoring IsaacLab-only overrides:", " ".join(ignored), flush=True)
    if args.track_cmds is None and args.headless:
        raise SystemExit("Live slider mode needs the viewer; pass --track-cmds for headless runs.")

    default_kp, default_kd = (9.0, 0.2) if args.task == "solo12-v0" else (DEFAULT_KP, DEFAULT_KD)
    kp = args.kp if args.kp is not None else env_overrides.get("kp", default_kp)
    kd = args.kd if args.kd is not None else env_overrides.get("kd", default_kd)
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
    policy = (EncodedPolicy(checkpoint, args.task, env_overrides, args.teacher_checkpoint)
              if args.task in TASK_MODELS else Policy(checkpoint))
    print(f"[INFO] Loaded {policy.checkpoint_format.upper()} actor for deterministic inference (critic not used).")
    if isinstance(policy, EncodedPolicy):
        print(f"[INFO] Policy={policy.policy_model}; apply_tanh_to_latent={policy.contract['apply_tanh_to_latent']}; "
              "production encoder/normalization restored.", flush=True)
    model_path = Path(__file__).with_name("solo12.xml").resolve()
    sim = Solo12Mujoco(model_path, kp=kp, kd=kd, spawn_z=args.spawn_z,
                     env_overrides=getattr(policy, "contract", env_overrides),
                     bringup_clip_rad=args.bringup_clip_rad, policy_model=TASK_MODELS.get(args.task, "simple_mlp"))
    observation_size = sim.observation(tuple(args.cmd_init)).size
    if observation_size != policy.input_dim:
        raise ValueError(
            f"Observation contract mismatch: checkpoint expects {policy.input_dim} inputs, "
            f"but playback supplies {observation_size}. Set env.base_obs_from_imu, "
            "env.remove_root_lin_vel_b_from_actor_obs and env.include_lin_acc_steps_in_imu_obs "
            "to match training; input size alone cannot identify their meanings."
        )
    print(f"[INFO] Observation inputs={observation_size}; IMU={sim.base_obs_from_imu}; "
          f"base linear velocity={sim.include_root_lin_vel}; acceleration samples={sim.imu_acc_steps} "
          f"at {1 / PHYSICS_DT:g} Hz, oldest first; clean signals (no training noise).", flush=True)
    print(f"[INFO] MuJoCo {mujoco.__version__}; model mass={sim.model.body_mass.sum():.6f} kg")
    print(f"[INFO] Bring-up position-error clipping: {sim.bringup_clip_rad:g} rad (0 = disabled); "
          f"effort limit={sim.effort_limit:g} Nm.", flush=True)
    print(f"[INFO] PD gains kp={sim.kp:g} kd={sim.kd:g} "
          f"(Isaac play w/ --disable_training_gain_sync: {DEFAULT_KP:g}/{DEFAULT_KD:g}; "
          f"training run: {TRAINING_KP:g}/{TRAINING_KD:g})", flush=True)

    if args.track_cmds is None:
        run_live(args, sim, policy, command_ranges)
    else:
        run_tracking(args, sim, policy, checkpoint, model_path, env_overrides, ignored)


if __name__ == "__main__":
    main()
