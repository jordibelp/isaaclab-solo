# SPDX-License-Identifier: BSD-3-Clause
"""Physical-unit velocity errors, independent of reward widths and weights."""
import math

import pytest
import torch

from test_task import curriculum_env, method, quaternion, reward_env, task_math


@pytest.mark.parametrize("std", [.02, .5, 2.])
@pytest.mark.parametrize("rpy", [(0., 0., 0.), (.6, -.8, 1.2)])
def test_errors_use_heading_frame_euclidean_norm_without_std_scaling(std, rpy):
    q = quaternion(*rpy)
    c, s = math.cos(rpy[2]), math.sin(rpy[2])
    # Desired heading velocity (4, 6, 15), actual (1, 2, 3): error (3, 4, 12).
    command = torch.tensor([[4., 6., 15., .7]], dtype=q.dtype)
    measured_w = torch.tensor([[c - 2*s, s + 2*c, 3.]], dtype=q.dtype)
    angular_w = torch.tensor([[10., -20., .7]], dtype=q.dtype)
    xyz, yaw, xy, xy_error, xyz_error = task_math.tracking_metrics(
        command, measured_w, angular_w, q, std, .5)
    torch.testing.assert_close(xy_error, torch.tensor([5.], dtype=q.dtype))
    torch.testing.assert_close(xyz_error, torch.tensor([13.], dtype=q.dtype))
    torch.testing.assert_close(xyz, torch.tensor([math.exp(-169/std**2)], dtype=q.dtype))
    torch.testing.assert_close(xy, torch.tensor([math.exp(-25/std**2)], dtype=q.dtype))
    torch.testing.assert_close(yaw, torch.ones(1, dtype=q.dtype))


def test_reward_path_accumulates_mean_norm_not_rms_or_norm_of_mean():
    env = reward_env()
    env._commands[:, :3] = torch.tensor([[3., 4., 12.], [0., 0., -2.]])
    get_rewards = method("_get_rewards")
    first = get_rewards(env)
    torch.testing.assert_close(env.extras["tracking_error_m_s"]["v_xy"], torch.tensor([5., 0.]))
    torch.testing.assert_close(env.extras["tracking_error_m_s"]["v_xyz"], torch.tensor([13., 2.]))
    assert "log" not in env.extras  # sparse episode packets stay sparse
    # Second sample: perfect in env 0, opposite Z error in env 1.
    # Means are (2.5, 6.5) and (0, 2), not RMS or norm of the mean error.
    env._commands.zero_()
    env._commands[1, 2] = 2.
    get_rewards(env)
    torch.testing.assert_close(env._episode_velocity_error_integral / (2*env.step_dt),
                               torch.tensor([[2.5, 6.5], [0., 2.]]))
    expected_first = .015 * torch.exp(-torch.tensor([169., 4.]) / .25) + .005
    torch.testing.assert_close(first, expected_first, rtol=0, atol=1e-9)


def test_episode_errors_use_each_completed_duration_and_ignore_unfinished_envs():
    env = curriculum_env()
    env.cfg.curriculum = False
    # Durations: 1 s, 2 s, 10 s, and an unstarted episode excluded from logging.
    env._episode_steps[:] = torch.tensor([10, 20, 100, 0])
    env._episode_velocity_error_integral[:] = torch.tensor([[2., 3.], [8., 12.], [900., 900.], [500., 500.]])
    method("_update_curriculum")(env, torch.tensor([0, 1, 3]))
    torch.testing.assert_close(env.extras["log"]["Tracking/v_xy_error_m_s"], torch.tensor(3.))
    torch.testing.assert_close(env.extras["log"]["Tracking/v_xyz_error_m_s"], torch.tensor(4.5))


def test_logging_does_not_change_legacy_rewards_or_curriculum():
    env = curriculum_env()
    reference = curriculum_env()
    env._episode_velocity_error_integral.fill_(1000.)
    for candidate in (env, reference):
        method("_update_curriculum")(candidate, torch.tensor([0, 1]))
    assert env.force_level == reference.force_level == 1
    for key, value in reference.extras["log"].items():
        if key not in ("Tracking/v_xy_error_m_s", "Tracking/v_xyz_error_m_s"):
            torch.testing.assert_close(torch.as_tensor(env.extras["log"][key]), torch.as_tensor(value))
