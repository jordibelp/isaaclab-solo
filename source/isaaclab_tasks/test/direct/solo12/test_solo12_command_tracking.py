# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

import isaaclab.utils.math as math_utils
from isaaclab_tasks.direct.solo12.solo12_env import _world_velocity_in_heading_frame_xy
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg, Solo12TwoFeetEnvCfg


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
