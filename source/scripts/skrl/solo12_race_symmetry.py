# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry transforms for the Solo12 race direct environment.

These transforms are intentionally conservative. They mirror the robot-centric proprioception,
actions, target direction vectors, closest-pillar vectors, privileged per-foot environment
parameters, and IMU/history channels, while leaving scalar race progress features untouched.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from tensordict import TensorDict
except Exception:  # pragma: no cover
    TensorDict = None

__all__ = [
    "compute_left_right_symmetric_observations_actions",
    "compute_symmetric_observations_actions",
]

_VECTOR_REFLECT_X = torch.tensor([1.0, -1.0, 1.0])
_VECTOR_REFLECT_Y = torch.tensor([-1.0, 1.0, 1.0])
_PSEUDOVECTOR_REFLECT_X = torch.tensor([-1.0, 1.0, -1.0])
_PSEUDOVECTOR_REFLECT_Y = torch.tensor([1.0, -1.0, -1.0])

_LEFT_RIGHT_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long)
_FRONT_BACK_PERM = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=torch.long)
_LEFT_RIGHT_SIGN = torch.tensor([-1.0, 1.0, 1.0] * 4)
_FRONT_BACK_SIGN = torch.tensor([1.0, -1.0, -1.0] * 4)
_LEFT_RIGHT_FOOT_PERM = torch.tensor([1, 0, 3, 2], dtype=torch.long)
_FRONT_BACK_FOOT_PERM = torch.tensor([2, 3, 0, 1], dtype=torch.long)

# Local sensor-frame -> base-frame rotations for the foot IMU channels, ordered FL, FR, RL, RR.
# The current Solo12 race USD mounts each foot IMU with identity offset orientation on calf frames whose axes are
# authored consistently with the base frame. Keeping the matrices explicit makes the intended operation clear:
# local foot IMU vector -> base frame -> mirror -> mirrored foot local frame. If a future asset rotates the sensor
# frames, update these matrices instead of changing the symmetry logic.
_FOOT_IMU_LOCAL_TO_BASE = torch.eye(3).repeat(4, 1, 1)


class RaceObservationSlices:
    def __init__(
        self,
        *,
        include_lin_vel: bool,
        include_imu: bool,
        layout: str,
        include_c_close: bool = False,
        include_gt_forces: bool = False,
        include_gt_mu: bool = False,
        history_kind: str | None = None,
        history_len: int = 0,
        history_sample_dim: int = 24,
    ):
        offset = 0
        self.layout = layout
        self.history_kind = history_kind
        self.lin_vel = None
        self.c_close0 = None
        self.c_close1 = None
        if include_lin_vel:
            self.lin_vel = slice(offset, offset + 3)
            offset += 3
        self.ang_vel = slice(offset, offset + 3)
        offset += 3
        self.gravity = slice(offset, offset + 3)
        offset += 3
        self.joint_pos = slice(offset, offset + 12)
        offset += 12
        self.joint_vel = slice(offset, offset + 12)
        offset += 12
        self.actions = slice(offset, offset + 12)
        offset += 12
        if layout == "gate_progress":
            self.target_pos = slice(offset, offset + 3)
            offset += 3
            self.target_tangent = slice(offset, offset + 3)
            offset += 3
            self.progress = slice(offset, offset + 1)
            offset += 1
            self.gate = slice(offset, offset + 1)
            offset += 1
        elif layout in ("gate_vectors", "gate_vectors_idx"):
            self.r_c1 = slice(offset, offset + 3)
            offset += 3
            self.r_c2 = slice(offset, offset + 3)
            offset += 3
            self.r_c3 = None
            self.r_c4 = None
            self.gate_idx = slice(offset, offset + 1)
            offset += 1
        elif layout == "gate_vectors_next":
            self.r_c1 = slice(offset, offset + 3)
            offset += 3
            self.r_c2 = slice(offset, offset + 3)
            offset += 3
            self.r_c3 = slice(offset, offset + 3)
            offset += 3
            self.r_c4 = slice(offset, offset + 3)
            offset += 3
            self.gate_idx = None
            if include_c_close:
                self.c_close0 = slice(offset, offset + 3)
                offset += 3
                self.c_close1 = slice(offset, offset + 3)
                offset += 3
        else:
            raise ValueError(f"Unsupported race observation layout: {layout}")
        self.gt_forces = None
        self.gt_mu = None
        if include_gt_forces:
            self.gt_forces = slice(offset, offset + 12)
            offset += 12
        if include_gt_mu:
            self.gt_mu = slice(offset, offset + 4)
            offset += 4
        self.imu = []
        if include_imu:
            for _ in range(4):
                self.imu.append(slice(offset, offset + 3))
                offset += 3
                self.imu.append(slice(offset, offset + 3))
                offset += 3
        self.history = []
        for _ in range(history_len):
            self.history.append(slice(offset, offset + history_sample_dim))
            offset += history_sample_dim
        self.obs_dim = offset


def _as_device_tensor(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    return values.to(device=device)


def _history_kind_from_env(env: Any | None) -> str | None:
    if env is None:
        return None
    env = getattr(env, "unwrapped", env)
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return None
    include_joint_state_history = getattr(cfg, "include_joint_state_history_obs", False)
    include_foot_imu = getattr(cfg, "include_foot_imu_obs", False)
    if include_joint_state_history and include_foot_imu:
        return "joint_state_foot_imu"
    if include_joint_state_history:
        return "joint_state"
    if include_foot_imu:
        return "foot_imu"
    return None


def _race_cfg_from_env(env: Any | None) -> Any | None:
    if env is None:
        return None
    env = getattr(env, "unwrapped", env)
    return getattr(env, "cfg", None)


def _gt_obs_flags_from_env(env: Any | None) -> tuple[bool, bool]:
    cfg = _race_cfg_from_env(env)
    if cfg is None:
        return False, False
    return (
        bool(getattr(cfg, "include_forces_to_gt_obs", False)),
        bool(getattr(cfg, "include_mu_coefs_to_gt_obs", False)),
    )


def _history_slices(
    obs_dim: int,
    base_dim: int,
    *,
    include_lin_vel: bool,
    layout: str,
    env: Any | None,
    include_c_close: bool = False,
    include_gt_forces: bool = False,
    include_gt_mu: bool = False,
):
    history_dim = obs_dim - base_dim
    if history_dim <= 0 or history_dim % 24 != 0:
        return None
    history_kind = _history_kind_from_env(env)
    if history_kind is None:
        raise ValueError(f"Observation size {obs_dim} looks like a race TCN history, but the env history kind is unknown.")
    history_sample_dim = 48 if history_kind == "joint_state_foot_imu" else 24
    if history_dim % history_sample_dim != 0:
        return None
    return RaceObservationSlices(
        include_lin_vel=include_lin_vel,
        include_imu=False,
        layout=layout,
        include_c_close=include_c_close,
        include_gt_forces=include_gt_forces,
        include_gt_mu=include_gt_mu,
        history_kind=history_kind,
        history_len=history_dim // history_sample_dim,
        history_sample_dim=history_sample_dim,
    )


def _get_slices(obs_dim: int, env: Any | None = None) -> RaceObservationSlices:
    cfg = _race_cfg_from_env(env)
    if cfg is not None:
        include_lin_vel = bool(getattr(cfg, "include_root_lin_vel_b_obs", True))
        include_gt_forces, include_gt_mu = _gt_obs_flags_from_env(env)
        gt_dim = (12 if include_gt_forces else 0) + (4 if include_gt_mu else 0)
        include_c_close = not bool(getattr(cfg, "remove_c_close_vectors_from_observation", False))
        base_obs_dim = int(
            getattr(cfg, "base_observation_dim", (57 if include_lin_vel else 54) + (6 if include_c_close else 0))
        )
        base_dim = base_obs_dim + gt_dim
        if obs_dim == base_dim:
            return RaceObservationSlices(
                include_lin_vel=include_lin_vel,
                include_imu=False,
                layout="gate_vectors_next",
                include_c_close=include_c_close,
                include_gt_forces=include_gt_forces,
                include_gt_mu=include_gt_mu,
            )
        history_slices = _history_slices(
            obs_dim,
            base_dim,
            include_lin_vel=include_lin_vel,
            layout="gate_vectors_next",
            env=env,
            include_c_close=include_c_close,
            include_gt_forces=include_gt_forces,
            include_gt_mu=include_gt_mu,
        )
        if history_slices is not None:
            return history_slices

    if obs_dim == 50:
        return RaceObservationSlices(include_lin_vel=False, include_imu=False, layout="gate_progress")
    if obs_dim == 53:
        return RaceObservationSlices(include_lin_vel=True, include_imu=False, layout="gate_progress")
    if obs_dim == 74:
        return RaceObservationSlices(include_lin_vel=False, include_imu=True, layout="gate_progress")
    if obs_dim == 77:
        return RaceObservationSlices(include_lin_vel=True, include_imu=True, layout="gate_progress")
    if obs_dim == 49:
        return RaceObservationSlices(include_lin_vel=False, include_imu=False, layout="gate_vectors_idx")
    if obs_dim == 52:
        return RaceObservationSlices(include_lin_vel=True, include_imu=False, layout="gate_vectors_idx")
    if obs_dim == 54:
        return RaceObservationSlices(include_lin_vel=False, include_imu=False, layout="gate_vectors_next")
    if obs_dim == 57:
        return RaceObservationSlices(include_lin_vel=True, include_imu=False, layout="gate_vectors_next")
    if obs_dim == 60:
        return RaceObservationSlices(
            include_lin_vel=False, include_imu=False, layout="gate_vectors_next", include_c_close=True
        )
    if obs_dim == 63:
        return RaceObservationSlices(
            include_lin_vel=True, include_imu=False, layout="gate_vectors_next", include_c_close=True
        )
    if obs_dim == 79:
        return RaceObservationSlices(
            include_lin_vel=True,
            include_imu=False,
            layout="gate_vectors_next",
            include_c_close=True,
            include_gt_forces=True,
            include_gt_mu=True,
        )
    if obs_dim == 73:
        return RaceObservationSlices(include_lin_vel=False, include_imu=True, layout="gate_vectors_idx")
    if obs_dim == 76:
        return RaceObservationSlices(include_lin_vel=True, include_imu=True, layout="gate_vectors_idx")
    history_slices = _history_slices(
        obs_dim, 63, include_lin_vel=True, layout="gate_vectors_next", env=env, include_c_close=True
    )
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(
        obs_dim, 60, include_lin_vel=False, layout="gate_vectors_next", env=env, include_c_close=True
    )
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 57, include_lin_vel=True, layout="gate_vectors_next", env=env)
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 54, include_lin_vel=False, layout="gate_vectors_next", env=env)
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 52, include_lin_vel=True, layout="gate_vectors_idx", env=env)
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 49, include_lin_vel=False, layout="gate_vectors_idx", env=env)
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 53, include_lin_vel=True, layout="gate_progress", env=env)
    if history_slices is not None:
        return history_slices
    history_slices = _history_slices(obs_dim, 50, include_lin_vel=False, layout="gate_progress", env=env)
    if history_slices is not None:
        return history_slices
    raise ValueError(f"Unsupported race observation size: {obs_dim}")


def _transform_joint_data_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    return joint_data[..., _as_device_tensor(_LEFT_RIGHT_PERM, joint_data.device)] * _as_device_tensor(
        _LEFT_RIGHT_SIGN, joint_data.device
    )


def _transform_joint_data_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    return joint_data[..., _as_device_tensor(_FRONT_BACK_PERM, joint_data.device)] * _as_device_tensor(
        _FRONT_BACK_SIGN, joint_data.device
    )


def _transform_foot_imu_data(
    imu_data: torch.Tensor,
    *,
    foot_perm: torch.Tensor,
    gyro_sign: torch.Tensor,
    acc_sign: torch.Tensor,
) -> torch.Tensor:
    imu_data = imu_data.reshape(*imu_data.shape[:-1], 4, 6)
    foot_perm = _as_device_tensor(foot_perm, imu_data.device)
    local_to_base = _FOOT_IMU_LOCAL_TO_BASE.to(device=imu_data.device, dtype=imu_data.dtype)

    def _mirror_local_vectors(local_vectors: torch.Tensor, base_sign: torch.Tensor) -> torch.Tensor:
        # Output foot i receives source foot foot_perm[i]. Convert that source local vector into the base frame,
        # apply the base-frame reflection/pseudoreflection, then express it in output foot i's local frame.
        source_local = local_vectors[..., foot_perm, :]
        source_to_base = local_to_base[foot_perm]
        target_to_base = local_to_base
        source_base = torch.einsum("fij,...fj->...fi", source_to_base, source_local)
        mirrored_base = source_base * _as_device_tensor(base_sign, imu_data.device).to(dtype=imu_data.dtype)
        return torch.einsum("fji,...fj->...fi", target_to_base, mirrored_base)

    gyro = _mirror_local_vectors(imu_data[..., :3], gyro_sign)
    acc = _mirror_local_vectors(imu_data[..., 3:6], acc_sign)
    return torch.cat((gyro, acc), dim=-1).reshape(*imu_data.shape[:-2], 24)


def _transform_foot_vector_data(
    foot_vector_data: torch.Tensor,
    *,
    foot_perm: torch.Tensor,
    vector_sign: torch.Tensor,
) -> torch.Tensor:
    foot_vector_data = foot_vector_data.reshape(*foot_vector_data.shape[:-1], 4, 3)
    foot_perm = _as_device_tensor(foot_perm, foot_vector_data.device)
    transformed = foot_vector_data[..., foot_perm, :].clone()
    transformed *= _as_device_tensor(vector_sign, foot_vector_data.device)
    return transformed.reshape(*transformed.shape[:-2], 12)


def _transform_foot_scalar_data(foot_scalar_data: torch.Tensor, *, foot_perm: torch.Tensor) -> torch.Tensor:
    foot_scalar_data = foot_scalar_data.reshape(*foot_scalar_data.shape[:-1], 4)
    foot_perm = _as_device_tensor(foot_perm, foot_scalar_data.device)
    return foot_scalar_data[..., foot_perm]


def _transform_history_reflect_x(history_sample: torch.Tensor, history_kind: str | None) -> torch.Tensor:
    if history_kind == "joint_state":
        return torch.cat(
            (
                _transform_joint_data_left_right(history_sample[..., :12]),
                _transform_joint_data_left_right(history_sample[..., 12:24]),
            ),
            dim=-1,
        )
    if history_kind == "foot_imu":
        return _transform_foot_imu_data(
            history_sample,
            foot_perm=_LEFT_RIGHT_FOOT_PERM,
            gyro_sign=_PSEUDOVECTOR_REFLECT_X,
            acc_sign=_VECTOR_REFLECT_X,
        )
    if history_kind == "joint_state_foot_imu":
        return torch.cat(
            (
                _transform_history_reflect_x(history_sample[..., :24], "joint_state"),
                _transform_history_reflect_x(history_sample[..., 24:48], "foot_imu"),
            ),
            dim=-1,
        )
    raise ValueError(f"Unsupported race history kind: {history_kind}")


def _transform_history_reflect_y(history_sample: torch.Tensor, history_kind: str | None) -> torch.Tensor:
    if history_kind == "joint_state":
        return torch.cat(
            (
                _transform_joint_data_front_back(history_sample[..., :12]),
                _transform_joint_data_front_back(history_sample[..., 12:24]),
            ),
            dim=-1,
        )
    if history_kind == "foot_imu":
        return _transform_foot_imu_data(
            history_sample,
            foot_perm=_FRONT_BACK_FOOT_PERM,
            gyro_sign=_PSEUDOVECTOR_REFLECT_Y,
            acc_sign=_VECTOR_REFLECT_Y,
        )
    if history_kind == "joint_state_foot_imu":
        return torch.cat(
            (
                _transform_history_reflect_y(history_sample[..., :24], "joint_state"),
                _transform_history_reflect_y(history_sample[..., 24:48], "foot_imu"),
            ),
            dim=-1,
        )
    raise ValueError(f"Unsupported race history kind: {history_kind}")


def transform_actions_reflect_x(actions: torch.Tensor) -> torch.Tensor:
    return _transform_joint_data_left_right(actions)


def transform_actions_reflect_y(actions: torch.Tensor) -> torch.Tensor:
    return _transform_joint_data_front_back(actions)


def transform_actions_rotate_180(actions: torch.Tensor) -> torch.Tensor:
    return transform_actions_reflect_y(transform_actions_reflect_x(actions))


def transform_policy_obs_reflect_x(obs: torch.Tensor, env: Any | None = None) -> torch.Tensor:
    slices = _get_slices(obs.shape[-1], env)
    obs = obs.clone()
    if slices.lin_vel is not None:
        obs[:, slices.lin_vel] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    obs[:, slices.ang_vel] *= _as_device_tensor(_PSEUDOVECTOR_REFLECT_X, obs.device)
    obs[:, slices.gravity] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    obs[:, slices.joint_pos] = _transform_joint_data_left_right(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_left_right(obs[:, slices.joint_vel])
    obs[:, slices.actions] = _transform_joint_data_left_right(obs[:, slices.actions])
    if slices.layout == "gate_progress":
        obs[:, slices.target_pos] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
        obs[:, slices.target_tangent] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    else:
        for gate_vector_slice in (slices.r_c1, slices.r_c2, slices.r_c3, slices.r_c4, slices.c_close0, slices.c_close1):
            if gate_vector_slice is not None:
                obs[:, gate_vector_slice] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    if slices.gt_forces is not None:
        obs[:, slices.gt_forces] = _transform_foot_vector_data(
            obs[:, slices.gt_forces], foot_perm=_LEFT_RIGHT_FOOT_PERM, vector_sign=_VECTOR_REFLECT_X
        )
    if slices.gt_mu is not None:
        obs[:, slices.gt_mu] = _transform_foot_scalar_data(obs[:, slices.gt_mu], foot_perm=_LEFT_RIGHT_FOOT_PERM)
    for imu_slice in slices.imu:
        obs[:, imu_slice] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    for history_slice in slices.history:
        obs[:, history_slice] = _transform_history_reflect_x(obs[:, history_slice], slices.history_kind)
    return obs


def transform_policy_obs_reflect_y(obs: torch.Tensor, env: Any | None = None) -> torch.Tensor:
    slices = _get_slices(obs.shape[-1], env)
    obs = obs.clone()
    if slices.lin_vel is not None:
        obs[:, slices.lin_vel] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    obs[:, slices.ang_vel] *= _as_device_tensor(_PSEUDOVECTOR_REFLECT_Y, obs.device)
    obs[:, slices.gravity] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    obs[:, slices.joint_pos] = _transform_joint_data_front_back(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_front_back(obs[:, slices.joint_vel])
    obs[:, slices.actions] = _transform_joint_data_front_back(obs[:, slices.actions])
    if slices.layout == "gate_progress":
        obs[:, slices.target_pos] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
        obs[:, slices.target_tangent] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    else:
        for gate_vector_slice in (slices.r_c1, slices.r_c2, slices.r_c3, slices.r_c4, slices.c_close0, slices.c_close1):
            if gate_vector_slice is not None:
                obs[:, gate_vector_slice] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    if slices.gt_forces is not None:
        obs[:, slices.gt_forces] = _transform_foot_vector_data(
            obs[:, slices.gt_forces], foot_perm=_FRONT_BACK_FOOT_PERM, vector_sign=_VECTOR_REFLECT_Y
        )
    if slices.gt_mu is not None:
        obs[:, slices.gt_mu] = _transform_foot_scalar_data(obs[:, slices.gt_mu], foot_perm=_FRONT_BACK_FOOT_PERM)
    for imu_slice in slices.imu:
        obs[:, imu_slice] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    for history_slice in slices.history:
        obs[:, history_slice] = _transform_history_reflect_y(obs[:, history_slice], slices.history_kind)
    return obs


def transform_policy_obs_rotate_180(obs: torch.Tensor, env: Any | None = None) -> torch.Tensor:
    return transform_policy_obs_reflect_y(transform_policy_obs_reflect_x(obs, env), env)


def _extract_policy_obs_tensor(obs: Any, obs_type: str) -> tuple[torch.Tensor, bool]:
    if isinstance(obs, torch.Tensor):
        return obs, False
    if TensorDict is not None and isinstance(obs, TensorDict):
        if obs_type not in obs.keys(include_nested=False):
            raise KeyError(f"Expected obs TensorDict to contain key '{obs_type}', got keys {list(obs.keys())}")
        return obs[obs_type], True
    raise TypeError(f"Expected obs to be a torch.Tensor or TensorDict, got {type(obs)!r}")


def _pack_policy_obs(obs_template: Any, obs_policy_aug: torch.Tensor, obs_type: str, was_tensordict: bool):
    if not was_tensordict:
        return obs_policy_aug
    batch_dim = obs_policy_aug.shape[0]
    data = {}
    for key in obs_template.keys(include_nested=False):
        value = obs_template[key]
        if key == obs_type:
            data[key] = obs_policy_aug
        else:
            repeat_factor = batch_dim // value.shape[0]
            data[key] = torch.cat([value] * repeat_factor, dim=0)
    if TensorDict is None:
        raise RuntimeError("TensorDict support expected but tensordict import is unavailable.")
    return TensorDict(source=data, batch_size=[batch_dim], device=obs_policy_aug.device)


@torch.no_grad()
def compute_left_right_symmetric_observations_actions(
    env: Any = None,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Augment race observations/actions with identity and left-right reflection only."""
    obs_aug = None
    actions_aug = None
    if obs is not None:
        obs_tensor, was_tensordict = _extract_policy_obs_tensor(obs, obs_type)
        obs_aug = torch.cat([obs_tensor, transform_policy_obs_reflect_x(obs_tensor, env)], dim=0)
        obs_aug = _pack_policy_obs(obs, obs_aug, obs_type, was_tensordict)
    if actions is not None:
        actions_aug = torch.cat([actions, transform_actions_reflect_x(actions)], dim=0)
    return obs_aug, actions_aug


@torch.no_grad()
def compute_symmetric_observations_actions(
    env: Any = None,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Apply the four quadruped symmetries to race observations/actions."""
    obs_aug = None
    actions_aug = None
    if obs is not None:
        obs_tensor, was_tensordict = _extract_policy_obs_tensor(obs, obs_type)
        obs_aug = torch.cat(
            [
                obs_tensor,
                transform_policy_obs_reflect_x(obs_tensor, env),
                transform_policy_obs_reflect_y(obs_tensor, env),
                transform_policy_obs_rotate_180(obs_tensor, env),
            ],
            dim=0,
        )
        obs_aug = _pack_policy_obs(obs, obs_aug, obs_type, was_tensordict)
    if actions is not None:
        actions_aug = torch.cat(
            [
                actions,
                transform_actions_reflect_x(actions),
                transform_actions_reflect_y(actions),
                transform_actions_rotate_180(actions),
            ],
            dim=0,
        )
    return obs_aug, actions_aug
