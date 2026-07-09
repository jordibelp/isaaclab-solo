# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Symmetry transforms for the Solo12 direct-locomotion observation/action spaces.

The policy observation layout for ``solo12-v0`` supports two variants:

With base linear velocity included (48 dims):
- 0:3   -> base linear velocity in body frame
- 3:6   -> base angular velocity in body frame
- 6:9   -> projected gravity in body frame
- 9:12  -> velocity commands ``[v_x, v_y, w_z]``
- 12:24 -> joint position offsets (12 DoF)
- 24:36 -> joint velocities (12 DoF)
- 36:48 -> previous/raw actions (12 DoF)

Without base linear velocity included (45 dims):
- 0:3   -> base angular velocity in body frame
- 3:6   -> projected gravity in body frame
- 6:9   -> velocity commands ``[v_x, v_y, w_z]``
- 9:21  -> joint position offsets (12 DoF)
- 21:33 -> joint velocities (12 DoF)
- 33:45 -> previous/raw actions (12 DoF)

The base-IMU teacher and student RSL tasks add two more layouts:

Teacher critic/policy observation without foot heights (48 dims):
- 0:3   -> projected gravity in body frame
- 3:6   -> base linear velocity in body frame
- 6:9   -> base angular velocity in body frame
- 9:21  -> previous/raw actions (12 DoF)
- 21:33 -> joint position offsets (12 DoF)
- 33:45 -> joint velocities (12 DoF)
- 45:48 -> velocity commands ``[v_x, v_y, w_z]``

Teacher critic/policy observation with foot heights (52 dims):
- 0:9   -> same gravity / base linear velocity / base angular velocity terms
- 9:13  -> foot heights in FL, FR, RL, RR order
- 13:52 -> same action / joint / command terms shifted by 4

Student policy observation (dynamic with history length and IMU layout):
- 0:N   -> flattened base-IMU history
- N:N+3 -> velocity commands ``[v_x, v_y, w_z]``

Each student history sample starts with:
- 0:12  -> joint position offsets (12 DoF)
- 12:24 -> joint velocities (12 DoF)
- 24:27 -> base gyro in body frame
- 27:30 -> base linear acceleration/specific force in body frame

It then uses one of three layouts:
- raw specific force: 30:42 actions (42 dims total)
- EKF + projected gravity: 30:33 projected gravity, 33:45 actions (45 dims total)
- EKF + rotation matrix: 30:39 body-to-world matrix, 39:51 actions (51 dims total)

The joint order is:

- FL = indices [0, 1, 2]
- FR = indices [3, 4, 5]
- RL = indices [6, 7, 8]
- RR = indices [9, 10, 11]

We implement the four quadruped symmetries used in the paper:
identity, reflect-x, reflect-y and rotate-180. In practice:

- reflect-x: mirror with respect to the x-axis (left-right swap)
- reflect-y: mirror with respect to the y-axis (front-back swap)
- rotate-180: composition of reflect-x and reflect-y
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from tensordict import TensorDict
except Exception:  # pragma: no cover - optional import for skrl-only use
    TensorDict = None

__all__ = [
    "OBSERVATION_SIZE",
    "ACTION_SIZE",
    "NUM_SYMMETRY_TRANSFORMS",
    "compute_symmetric_observations_actions",
]

OBSERVATION_SIZE = 48
OBSERVATION_SIZE_WITHOUT_LIN_VEL = 45
BASE_IMU_TEACHER_OBSERVATION_SIZE = 48
BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS = 52
BASE_IMU_RAW_HISTORY_SAMPLE_SIZE = 42
BASE_IMU_PROJECTED_GRAVITY_HISTORY_SAMPLE_SIZE = 45
BASE_IMU_ROTATION_MATRIX_HISTORY_SAMPLE_SIZE = 51
BASE_IMU_HISTORY_SAMPLE_SIZES = (
    BASE_IMU_RAW_HISTORY_SAMPLE_SIZE,
    BASE_IMU_PROJECTED_GRAVITY_HISTORY_SAMPLE_SIZE,
    BASE_IMU_ROTATION_MATRIX_HISTORY_SAMPLE_SIZE,
)
BASE_IMU_COMMAND_SIZE = 3
ACTION_SIZE = 12
NUM_SYMMETRY_TRANSFORMS = 4

_VECTOR_REFLECT_X = torch.tensor([1.0, -1.0, 1.0])
_VECTOR_REFLECT_Y = torch.tensor([-1.0, 1.0, 1.0])
_PSEUDOVECTOR_REFLECT_X = torch.tensor([-1.0, 1.0, -1.0])
_PSEUDOVECTOR_REFLECT_Y = torch.tensor([1.0, -1.0, -1.0])
_COMMAND_REFLECT_X = torch.tensor([1.0, -1.0, -1.0])
_COMMAND_REFLECT_Y = torch.tensor([-1.0, 1.0, -1.0])

# FL, FR, RL, RR ordering.
_LEFT_RIGHT_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.long)
_FRONT_BACK_PERM = torch.tensor([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=torch.long)
_LEFT_RIGHT_SIGN = torch.tensor([-1.0, 1.0, 1.0] * 4)
_FRONT_BACK_SIGN = torch.tensor([1.0, -1.0, -1.0] * 4)
_LEFT_RIGHT_FOOT_PERM = torch.tensor([1, 0, 3, 2], dtype=torch.long)
_FRONT_BACK_FOOT_PERM = torch.tensor([2, 3, 0, 1], dtype=torch.long)


class Solo12ObservationSlices:
    def __init__(self, *, include_lin_vel: bool):
        offset = 0
        self.lin_vel = None
        if include_lin_vel:
            self.lin_vel = slice(offset, offset + 3)
            offset += 3
        self.ang_vel = slice(offset, offset + 3)
        offset += 3
        self.gravity = slice(offset, offset + 3)
        offset += 3
        self.commands = slice(offset, offset + 3)
        offset += 3
        self.joint_pos = slice(offset, offset + 12)
        offset += 12
        self.joint_vel = slice(offset, offset + 12)
        offset += 12
        self.previous_actions = slice(offset, offset + 12)
        offset += 12
        self.obs_dim = offset


class Solo12BaseImuTeacherObservationSlices:
    def __init__(self, *, include_foot_heights: bool):
        offset = 0
        self.gravity = slice(offset, offset + 3)
        offset += 3
        self.lin_vel = slice(offset, offset + 3)
        offset += 3
        self.ang_vel = slice(offset, offset + 3)
        offset += 3
        self.foot_heights = None
        if include_foot_heights:
            self.foot_heights = slice(offset, offset + 4)
            offset += 4
        self.previous_actions = slice(offset, offset + 12)
        offset += 12
        self.joint_pos = slice(offset, offset + 12)
        offset += 12
        self.joint_vel = slice(offset, offset + 12)
        offset += 12
        self.commands = slice(offset, offset + 3)
        offset += 3
        self.obs_dim = offset


def _get_observation_slices(obs_dim: int) -> Solo12ObservationSlices:
    if obs_dim == OBSERVATION_SIZE:
        return Solo12ObservationSlices(include_lin_vel=True)
    if obs_dim == OBSERVATION_SIZE_WITHOUT_LIN_VEL:
        return Solo12ObservationSlices(include_lin_vel=False)
    raise ValueError(
        f"Expected Solo12 policy observations with size {OBSERVATION_SIZE} or {OBSERVATION_SIZE_WITHOUT_LIN_VEL}, got {obs_dim}"
    )


def _get_base_imu_teacher_observation_slices(obs_dim: int) -> Solo12BaseImuTeacherObservationSlices:
    if obs_dim == BASE_IMU_TEACHER_OBSERVATION_SIZE:
        return Solo12BaseImuTeacherObservationSlices(include_foot_heights=False)
    if obs_dim == BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS:
        return Solo12BaseImuTeacherObservationSlices(include_foot_heights=True)
    raise ValueError(
        "Expected Solo12 base-IMU teacher observations with size "
        f"{BASE_IMU_TEACHER_OBSERVATION_SIZE} or {BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS}, got {obs_dim}"
    )


def _is_base_imu_student_observation_size(obs_dim: int, history_sample_dim: int | None = None) -> bool:
    history_flat_dim = obs_dim - BASE_IMU_COMMAND_SIZE
    sample_dims = (history_sample_dim,) if history_sample_dim is not None else BASE_IMU_HISTORY_SAMPLE_SIZES
    matches = [
        sample_dim
        for sample_dim in sample_dims
        if history_flat_dim >= 2 * sample_dim and history_flat_dim % sample_dim == 0
    ]
    return len(matches) == 1


def _get_policy_model(env: Any) -> str | None:
    visited = set()
    candidates = [env]
    while candidates:
        current = candidates.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        cfg = getattr(current, "cfg", None)
        policy_model = getattr(cfg, "policy_model", None)
        if policy_model is not None:
            return str(policy_model)
        candidates.extend(
            getattr(current, name, None)
            for name in ("unwrapped", "env", "_env", "venv")
            if getattr(current, name, None) is not current
        )
    return None


def _get_base_imu_history_sample_dim(env: Any) -> int | None:
    visited = set()
    candidates = [env]
    while candidates:
        current = candidates.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        cfg = getattr(current, "cfg", None)
        sample_dim = getattr(cfg, "base_imu_history_sample_dim", None)
        if sample_dim is not None:
            return int(sample_dim)
        candidates.extend(
            getattr(current, name, None)
            for name in ("unwrapped", "env", "_env", "venv")
            if getattr(current, name, None) is not current
        )
    return None


def _uses_front_back_symmetry(env: Any) -> bool:
    visited = set()
    candidates = [env]
    while candidates:
        current = candidates.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        cfg = getattr(current, "cfg", None)
        if cfg is not None and hasattr(cfg, "front_back_asymetry"):
            return not bool(getattr(cfg, "front_back_asymetry"))
        candidates.extend(
            getattr(current, name, None)
            for name in ("unwrapped", "env", "_env", "venv")
            if getattr(current, name, None) is not current
        )
    return True


def _as_device_tensor(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    return values.to(device=device)

# transform angle left -> right
def _transform_joint_data_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data = joint_data.clone()
    perm = _as_device_tensor(_LEFT_RIGHT_PERM, joint_data.device)
    sign = _as_device_tensor(_LEFT_RIGHT_SIGN, joint_data.device)
    return joint_data[..., perm] * sign


def _transform_joint_data_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data = joint_data.clone()
    perm = _as_device_tensor(_FRONT_BACK_PERM, joint_data.device)
    sign = _as_device_tensor(_FRONT_BACK_SIGN, joint_data.device)
    return joint_data[..., perm] * sign


def _transform_foot_data_left_right(foot_data: torch.Tensor) -> torch.Tensor:
    perm = _as_device_tensor(_LEFT_RIGHT_FOOT_PERM, foot_data.device)
    return foot_data[..., perm]


def _transform_foot_data_front_back(foot_data: torch.Tensor) -> torch.Tensor:
    perm = _as_device_tensor(_FRONT_BACK_FOOT_PERM, foot_data.device)
    return foot_data[..., perm]


def transform_actions_reflect_x(actions: torch.Tensor) -> torch.Tensor:
    return _transform_joint_data_left_right(actions)


def transform_actions_reflect_y(actions: torch.Tensor) -> torch.Tensor:
    return _transform_joint_data_front_back(actions)


def transform_actions_rotate_180(actions: torch.Tensor) -> torch.Tensor:
    return transform_actions_reflect_y(transform_actions_reflect_x(actions))


def _transform_vector_by_sign(data: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return data * _as_device_tensor(sign, data.device).to(dtype=data.dtype)


def transform_policy_obs_reflect_x(obs: torch.Tensor) -> torch.Tensor:
    slices = _get_observation_slices(obs.shape[-1])

    obs = obs.clone()
    if slices.lin_vel is not None:
        obs[:, slices.lin_vel] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    obs[:, slices.ang_vel] *= _as_device_tensor(_PSEUDOVECTOR_REFLECT_X, obs.device)
    obs[:, slices.gravity] *= _as_device_tensor(_VECTOR_REFLECT_X, obs.device)
    obs[:, slices.commands] *= _as_device_tensor(_COMMAND_REFLECT_X, obs.device)
    obs[:, slices.joint_pos] = _transform_joint_data_left_right(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_left_right(obs[:, slices.joint_vel])
    obs[:, slices.previous_actions] = _transform_joint_data_left_right(obs[:, slices.previous_actions])
    return obs


def transform_policy_obs_reflect_y(obs: torch.Tensor) -> torch.Tensor:
    slices = _get_observation_slices(obs.shape[-1])

    obs = obs.clone()
    if slices.lin_vel is not None:
        obs[:, slices.lin_vel] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    obs[:, slices.ang_vel] *= _as_device_tensor(_PSEUDOVECTOR_REFLECT_Y, obs.device)
    obs[:, slices.gravity] *= _as_device_tensor(_VECTOR_REFLECT_Y, obs.device)
    obs[:, slices.commands] *= _as_device_tensor(_COMMAND_REFLECT_Y, obs.device)
    obs[:, slices.joint_pos] = _transform_joint_data_front_back(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_front_back(obs[:, slices.joint_vel])
    obs[:, slices.previous_actions] = _transform_joint_data_front_back(obs[:, slices.previous_actions])
    return obs


def transform_policy_obs_rotate_180(obs: torch.Tensor) -> torch.Tensor:
    return transform_policy_obs_reflect_y(transform_policy_obs_reflect_x(obs))


def transform_base_imu_teacher_obs_reflect_x(obs: torch.Tensor) -> torch.Tensor:
    slices = _get_base_imu_teacher_observation_slices(obs.shape[-1])

    obs = obs.clone()
    obs[:, slices.gravity] = _transform_vector_by_sign(obs[:, slices.gravity], _VECTOR_REFLECT_X)
    obs[:, slices.lin_vel] = _transform_vector_by_sign(obs[:, slices.lin_vel], _VECTOR_REFLECT_X)
    obs[:, slices.ang_vel] = _transform_vector_by_sign(obs[:, slices.ang_vel], _PSEUDOVECTOR_REFLECT_X)
    if slices.foot_heights is not None:
        obs[:, slices.foot_heights] = _transform_foot_data_left_right(obs[:, slices.foot_heights])
    obs[:, slices.previous_actions] = _transform_joint_data_left_right(obs[:, slices.previous_actions])
    obs[:, slices.joint_pos] = _transform_joint_data_left_right(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_left_right(obs[:, slices.joint_vel])
    obs[:, slices.commands] = _transform_vector_by_sign(obs[:, slices.commands], _COMMAND_REFLECT_X)
    return obs


def transform_base_imu_teacher_obs_reflect_y(obs: torch.Tensor) -> torch.Tensor:
    slices = _get_base_imu_teacher_observation_slices(obs.shape[-1])

    obs = obs.clone()
    obs[:, slices.gravity] = _transform_vector_by_sign(obs[:, slices.gravity], _VECTOR_REFLECT_Y)
    obs[:, slices.lin_vel] = _transform_vector_by_sign(obs[:, slices.lin_vel], _VECTOR_REFLECT_Y)
    obs[:, slices.ang_vel] = _transform_vector_by_sign(obs[:, slices.ang_vel], _PSEUDOVECTOR_REFLECT_Y)
    if slices.foot_heights is not None:
        obs[:, slices.foot_heights] = _transform_foot_data_front_back(obs[:, slices.foot_heights])
    obs[:, slices.previous_actions] = _transform_joint_data_front_back(obs[:, slices.previous_actions])
    obs[:, slices.joint_pos] = _transform_joint_data_front_back(obs[:, slices.joint_pos])
    obs[:, slices.joint_vel] = _transform_joint_data_front_back(obs[:, slices.joint_vel])
    obs[:, slices.commands] = _transform_vector_by_sign(obs[:, slices.commands], _COMMAND_REFLECT_Y)
    return obs


def transform_base_imu_teacher_obs_rotate_180(obs: torch.Tensor) -> torch.Tensor:
    return transform_base_imu_teacher_obs_reflect_y(transform_base_imu_teacher_obs_reflect_x(obs))


def _transform_base_imu_history_left_right(history: torch.Tensor) -> torch.Tensor:
    history = history.clone()
    history[..., 0:12] = _transform_joint_data_left_right(history[..., 0:12])
    history[..., 12:24] = _transform_joint_data_left_right(history[..., 12:24])
    history[..., 24:27] = _transform_vector_by_sign(history[..., 24:27], _PSEUDOVECTOR_REFLECT_X)
    history[..., 27:30] = _transform_vector_by_sign(history[..., 27:30], _VECTOR_REFLECT_X)
    action_start = _transform_base_imu_orientation_left_right(history)
    history[..., action_start : action_start + 12] = _transform_joint_data_left_right(
        history[..., action_start : action_start + 12]
    )
    return history


def _transform_base_imu_history_front_back(history: torch.Tensor) -> torch.Tensor:
    history = history.clone()
    history[..., 0:12] = _transform_joint_data_front_back(history[..., 0:12])
    history[..., 12:24] = _transform_joint_data_front_back(history[..., 12:24])
    history[..., 24:27] = _transform_vector_by_sign(history[..., 24:27], _PSEUDOVECTOR_REFLECT_Y)
    history[..., 27:30] = _transform_vector_by_sign(history[..., 27:30], _VECTOR_REFLECT_Y)
    action_start = _transform_base_imu_orientation_front_back(history)
    history[..., action_start : action_start + 12] = _transform_joint_data_front_back(
        history[..., action_start : action_start + 12]
    )
    return history


def _transform_rotation_matrix_reflection(flat_rotation: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    rotation = flat_rotation.reshape(*flat_rotation.shape[:-1], 3, 3)
    reflection = torch.diag(_as_device_tensor(sign, flat_rotation.device).to(dtype=flat_rotation.dtype))
    return (reflection @ rotation @ reflection).reshape_as(flat_rotation)


def _transform_base_imu_orientation_left_right(history: torch.Tensor) -> int:
    sample_dim = history.shape[-1]
    if sample_dim == BASE_IMU_RAW_HISTORY_SAMPLE_SIZE:
        return 30
    if sample_dim == BASE_IMU_PROJECTED_GRAVITY_HISTORY_SAMPLE_SIZE:
        history[..., 30:33] = _transform_vector_by_sign(history[..., 30:33], _VECTOR_REFLECT_X)
        return 33
    if sample_dim == BASE_IMU_ROTATION_MATRIX_HISTORY_SAMPLE_SIZE:
        history[..., 30:39] = _transform_rotation_matrix_reflection(history[..., 30:39], _VECTOR_REFLECT_X)
        return 39
    raise ValueError(f"Unsupported Solo12 base-IMU history sample size: {sample_dim}.")


def _transform_base_imu_orientation_front_back(history: torch.Tensor) -> int:
    sample_dim = history.shape[-1]
    if sample_dim == BASE_IMU_RAW_HISTORY_SAMPLE_SIZE:
        return 30
    if sample_dim == BASE_IMU_PROJECTED_GRAVITY_HISTORY_SAMPLE_SIZE:
        history[..., 30:33] = _transform_vector_by_sign(history[..., 30:33], _VECTOR_REFLECT_Y)
        return 33
    if sample_dim == BASE_IMU_ROTATION_MATRIX_HISTORY_SAMPLE_SIZE:
        history[..., 30:39] = _transform_rotation_matrix_reflection(history[..., 30:39], _VECTOR_REFLECT_Y)
        return 39
    raise ValueError(f"Unsupported Solo12 base-IMU history sample size: {sample_dim}.")


def transform_base_imu_student_obs_reflect_x(
    obs: torch.Tensor, history_sample_dim: int | None = None
) -> torch.Tensor:
    obs = obs.clone()
    history_flat_dim = obs.shape[-1] - BASE_IMU_COMMAND_SIZE
    history_sample_dim = history_sample_dim or next(
        (size for size in BASE_IMU_HISTORY_SAMPLE_SIZES if history_flat_dim % size == 0), None
    )
    if history_sample_dim is None or history_flat_dim <= 0 or history_flat_dim % history_sample_dim != 0:
        raise ValueError(
            "Expected Solo12 base-IMU student observations with flattened history plus 3 commands, "
            f"got size {obs.shape[-1]}."
        )
    history_len = history_flat_dim // history_sample_dim
    history = obs[:, :history_flat_dim].reshape(obs.shape[0], history_len, history_sample_dim)
    obs[:, :history_flat_dim] = _transform_base_imu_history_left_right(history).reshape(obs.shape[0], -1)
    obs[:, -3:] = _transform_vector_by_sign(obs[:, -3:], _COMMAND_REFLECT_X)
    return obs


def transform_base_imu_student_obs_reflect_y(
    obs: torch.Tensor, history_sample_dim: int | None = None
) -> torch.Tensor:
    obs = obs.clone()
    history_flat_dim = obs.shape[-1] - BASE_IMU_COMMAND_SIZE
    history_sample_dim = history_sample_dim or next(
        (size for size in BASE_IMU_HISTORY_SAMPLE_SIZES if history_flat_dim % size == 0), None
    )
    if history_sample_dim is None or history_flat_dim <= 0 or history_flat_dim % history_sample_dim != 0:
        raise ValueError(
            "Expected Solo12 base-IMU student observations with flattened history plus 3 commands, "
            f"got size {obs.shape[-1]}."
        )
    history_len = history_flat_dim // history_sample_dim
    history = obs[:, :history_flat_dim].reshape(obs.shape[0], history_len, history_sample_dim)
    obs[:, :history_flat_dim] = _transform_base_imu_history_front_back(history).reshape(obs.shape[0], -1)
    obs[:, -3:] = _transform_vector_by_sign(obs[:, -3:], _COMMAND_REFLECT_Y)
    return obs


def transform_base_imu_student_obs_rotate_180(
    obs: torch.Tensor, history_sample_dim: int | None = None
) -> torch.Tensor:
    return transform_base_imu_student_obs_reflect_y(
        transform_base_imu_student_obs_reflect_x(obs, history_sample_dim), history_sample_dim
    )


def _transform_obs_reflect_x(
    obs: torch.Tensor, policy_model: str | None, history_sample_dim: int | None
) -> torch.Tensor:
    if policy_model == "base_imu_teacher":
        return transform_base_imu_teacher_obs_reflect_x(obs)
    if policy_model == "base_imu_student_rl" and _is_base_imu_student_observation_size(
        obs.shape[-1], history_sample_dim
    ):
        return transform_base_imu_student_obs_reflect_x(obs, history_sample_dim)
    if policy_model is None and _is_base_imu_student_observation_size(obs.shape[-1], history_sample_dim):
        return transform_base_imu_student_obs_reflect_x(obs, history_sample_dim)
    if policy_model is None and obs.shape[-1] == BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS:
        return transform_base_imu_teacher_obs_reflect_x(obs)
    if policy_model == "base_imu_student_rl" and obs.shape[-1] in (
        BASE_IMU_TEACHER_OBSERVATION_SIZE,
        BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS,
    ):
        return transform_base_imu_teacher_obs_reflect_x(obs)
    return transform_policy_obs_reflect_x(obs)


def _transform_obs_reflect_y(
    obs: torch.Tensor, policy_model: str | None, history_sample_dim: int | None
) -> torch.Tensor:
    if policy_model == "base_imu_teacher":
        return transform_base_imu_teacher_obs_reflect_y(obs)
    if policy_model == "base_imu_student_rl" and _is_base_imu_student_observation_size(
        obs.shape[-1], history_sample_dim
    ):
        return transform_base_imu_student_obs_reflect_y(obs, history_sample_dim)
    if policy_model is None and _is_base_imu_student_observation_size(obs.shape[-1], history_sample_dim):
        return transform_base_imu_student_obs_reflect_y(obs, history_sample_dim)
    if policy_model is None and obs.shape[-1] == BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS:
        return transform_base_imu_teacher_obs_reflect_y(obs)
    if policy_model == "base_imu_student_rl" and obs.shape[-1] in (
        BASE_IMU_TEACHER_OBSERVATION_SIZE,
        BASE_IMU_TEACHER_OBSERVATION_SIZE_WITH_FOOT_HEIGHTS,
    ):
        return transform_base_imu_teacher_obs_reflect_y(obs)
    return transform_policy_obs_reflect_y(obs)


def _transform_obs_rotate_180(
    obs: torch.Tensor, policy_model: str | None, history_sample_dim: int | None
) -> torch.Tensor:
    return _transform_obs_reflect_y(
        _transform_obs_reflect_x(obs, policy_model, history_sample_dim), policy_model, history_sample_dim
    )


def _augment_obs_tensor(
    obs: torch.Tensor,
    policy_model: str | None,
    history_sample_dim: int | None,
    use_front_back_symmetry: bool,
) -> torch.Tensor:
    transforms = (
        (obs, _transform_obs_reflect_x(obs, policy_model, history_sample_dim))
        if not use_front_back_symmetry
        else (
            obs,
            _transform_obs_reflect_x(obs, policy_model, history_sample_dim),
            _transform_obs_reflect_y(obs, policy_model, history_sample_dim),
            _transform_obs_rotate_180(obs, policy_model, history_sample_dim),
        )
    )
    return torch.cat(
        transforms,
        dim=0,
    )


def _extract_policy_obs_tensor(obs: Any, obs_type: str) -> tuple[torch.Tensor, bool]:
    if isinstance(obs, torch.Tensor):
        return obs, False
    if TensorDict is not None and isinstance(obs, TensorDict):
        if obs_type not in obs.keys(include_nested=False):
            raise KeyError(f"Expected obs TensorDict to contain key '{obs_type}', got keys {list(obs.keys())}")
        return obs[obs_type], True
    raise TypeError(f"Expected obs to be a torch.Tensor or TensorDict, got {type(obs)!r}")


def _pack_policy_obs(
    obs_template: Any,
    obs_policy_aug: torch.Tensor,
    obs_type: str,
    was_tensordict: bool,
    policy_model: str | None,
    history_sample_dim: int | None,
    use_front_back_symmetry: bool,
):
    if not was_tensordict:
        return obs_policy_aug

    batch_dim = obs_policy_aug.shape[0]
    data = {}
    for key in obs_template.keys(include_nested=False):
        value = obs_template[key]
        if key == obs_type:
            data[key] = obs_policy_aug
        elif isinstance(value, torch.Tensor) and value.ndim == 2:
            data[key] = _augment_obs_tensor(
                value, policy_model, history_sample_dim, use_front_back_symmetry
            )
        else:
            repeat_factor = batch_dim // value.shape[0]
            data[key] = torch.cat([value] * repeat_factor, dim=0)

    if TensorDict is None:
        raise RuntimeError("TensorDict support expected but tensordict import is unavailable.")
    return TensorDict(source=data, batch_size=[batch_dim], device=obs_policy_aug.device)


@torch.no_grad()
def compute_symmetric_observations_actions(
    env: Any = None,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Apply quadruped symmetries to Solo12 observations and/or actions.

    Args:
        env: Isaac Lab env or wrapper. Used to disambiguate legacy 48D observations from base-IMU teacher 48D observations.
        obs: Policy observation tensor of shape ``[batch, 48]``, ``[batch, 45]``, or base-IMU student ``[batch, 549]``.
        actions: Action tensor of shape ``[batch, 12]``.
        obs_type: Present for API compatibility. Only ``"policy"`` is supported.

    Returns:
        A tuple ``(obs_aug, actions_aug)`` where each non-``None`` tensor has the batch dimension
        multiplied by 4 in the order ``[identity, reflect_x, reflect_y, rotate_180]``. If
        ``env.cfg.front_back_asymetry`` is true, only ``[identity, reflect_x]`` is used.
    """
    if obs_type != "policy":
        raise ValueError(f"Solo12 symmetry only supports obs_type='policy'. Got: {obs_type}")

    obs_aug = None
    policy_model = _get_policy_model(env)
    history_sample_dim = _get_base_imu_history_sample_dim(env)
    use_front_back_symmetry = _uses_front_back_symmetry(env)
    if obs is not None:
        obs_policy, was_tensordict = _extract_policy_obs_tensor(obs, obs_type)
        obs_policy_aug = _augment_obs_tensor(
            obs_policy, policy_model, history_sample_dim, use_front_back_symmetry
        )
        obs_aug = _pack_policy_obs(
            obs,
            obs_policy_aug,
            obs_type,
            was_tensordict,
            policy_model,
            history_sample_dim,
            use_front_back_symmetry,
        )

    actions_aug = None
    if actions is not None:
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"Expected actions to be a torch.Tensor, got {type(actions)!r}")
        if actions.shape[-1] != ACTION_SIZE:
            raise ValueError(f"Expected Solo12 actions with size {ACTION_SIZE}, got {actions.shape[-1]}")
        transforms = (
            (actions, transform_actions_reflect_x(actions))
            if not use_front_back_symmetry
            else (
                actions,
                transform_actions_reflect_x(actions),
                transform_actions_reflect_y(actions),
                transform_actions_rotate_180(actions),
            )
        )
        actions_aug = torch.cat(transforms, dim=0)

    return obs_aug, actions_aug
