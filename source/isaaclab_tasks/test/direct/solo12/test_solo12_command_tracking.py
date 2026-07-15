# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

import isaaclab.utils.math as math_utils
from isaaclab_tasks.direct.solo12.solo12_env import _sample_reset_root_rpy, _world_velocity_in_heading_frame_xy
from isaaclab_tasks.direct.solo12.solo12_env_cfg import (
    TWO_FEET_INITIAL_JOINT_POS,
    Solo12EnvCfg,
    Solo12TwoFeetEnvCfg,
)


def test_heading_frame_tracking_is_enabled_only_for_two_feet_config():
    assert Solo12EnvCfg().track_commands_in_world_heading_frame is False
    assert Solo12TwoFeetEnvCfg().track_commands_in_world_heading_frame is True


def test_world_velocity_in_heading_frame_ignores_pitch_and_vertical_velocity():
    roll = torch.zeros(2)
    pitch = torch.full((2,), math.pi / 2)
    yaw = torch.tensor((0.0, math.pi / 2))
    root_quat_w = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    root_lin_vel_w = torch.tensor(((0.4, 0.2, 3.0), (-0.2, 0.4, -3.0)))

    actual = _world_velocity_in_heading_frame_xy(root_lin_vel_w, root_quat_w)

    torch.testing.assert_close(actual, torch.tensor(((0.4, 0.2), (0.4, 0.2))), atol=1.0e-6, rtol=0.0)


def test_world_velocity_in_heading_frame_uses_inverse_heading_rotation():
    yaw = torch.tensor((math.pi / 4,))
    root_quat_w = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), yaw)
    root_lin_vel_w = torch.tensor(((1.0, 0.0, 0.0),))

    actual = _world_velocity_in_heading_frame_xy(root_lin_vel_w, root_quat_w)

    expected = torch.tensor(((math.sqrt(0.5), -math.sqrt(0.5)),))
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)


def test_two_feet_task_defaults_to_safe_reset_profile():
    standard = Solo12EnvCfg()
    two_feet = Solo12TwoFeetEnvCfg()

    assert standard.initial_position == "safe"
    assert standard.reset_root_height is None
    assert standard.reset_root_pitch == 0.0
    assert standard.reset_root_rpy_noise == (0.0, 0.0, 0.0)

    assert two_feet.initial_position == "safe"
    assert two_feet.reset_x_pos == 0.5
    assert two_feet.reset_y_pos == 0.5
    assert two_feet.reset_root_height is None
    assert two_feet.reset_root_pitch == 0.0
    assert two_feet.reset_root_rpy_noise == (0.0, 0.0, 0.0)
    assert two_feet.flexed_initial_joint_pos_noise_range == (-0.07, 0.07)
    assert two_feet.reset_base_lin_vel_range == (-0.3, 0.3)
    assert two_feet.reset_base_ang_vel_range == (-0.1, 0.1)


def test_two_feet_initial_position_activates_complete_reset_profile():
    two_feet = Solo12TwoFeetEnvCfg()
    two_feet.initial_position = "two_feet"
    two_feet.refresh_runtime_dependent_config()

    assert two_feet.initial_joint_pos_by_name["two_feet"] == TWO_FEET_INITIAL_JOINT_POS
    assert TWO_FEET_INITIAL_JOINT_POS["FL_thigh_joint"] == 1.1
    assert TWO_FEET_INITIAL_JOINT_POS["FR_thigh_joint"] == 1.1
    assert TWO_FEET_INITIAL_JOINT_POS["RL_thigh_joint"] == 0.6
    assert TWO_FEET_INITIAL_JOINT_POS["RR_thigh_joint"] == 0.6
    assert two_feet.reset_x_pos == 0.0
    assert two_feet.reset_y_pos == 0.0
    assert two_feet.reset_root_height == 0.53
    assert two_feet.reset_root_pitch == math.radians(-73.32)
    assert two_feet.reset_root_rpy_noise == tuple(math.radians(value) for value in (3.0, 5.0, 5.0))
    assert two_feet.flexed_initial_joint_pos_noise_range == (-0.05, 0.05)
    assert two_feet.reset_base_lin_vel_range == (0.0, 0.0)
    assert two_feet.reset_base_ang_vel_range == (0.0, 0.0)


def test_two_feet_reset_profile_remains_configurable():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.initial_position = "two_feet"
    cfg.two_feet_reset_root_height = 0.6
    cfg.two_feet_reset_root_pitch = -1.0
    cfg.two_feet_joint_pos_noise_range = (-0.02, 0.02)
    cfg.refresh_runtime_dependent_config()

    assert cfg.reset_root_height == 0.6
    assert cfg.reset_root_pitch == -1.0
    assert cfg.flexed_initial_joint_pos_noise_range == (-0.02, 0.02)


def test_root_orientation_noise_stays_inside_accepted_ranges():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.initial_position = "two_feet"
    cfg.refresh_runtime_dependent_config()
    nominal = (cfg.reset_root_roll, cfg.reset_root_pitch, cfg.reset_yaw)
    torch.manual_seed(7)

    samples = _sample_reset_root_rpy(10_000, nominal, cfg.reset_root_rpy_noise, "cpu")
    offsets = samples - torch.tensor(nominal)
    limits = torch.tensor(cfg.reset_root_rpy_noise)

    assert torch.all(offsets.abs() <= limits + 1.0e-7)
    assert torch.all(offsets.amin(dim=0) < -0.95 * limits)
    assert torch.all(offsets.amax(dim=0) > 0.95 * limits)
