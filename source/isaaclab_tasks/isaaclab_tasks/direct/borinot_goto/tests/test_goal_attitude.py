# SPDX-License-Identifier: BSD-3-Clause
"""Goal-facing reset checks against an independent rotation-matrix construction."""
import importlib.util

import pytest
import torch

from test_goto import FOLDER, config, fake_env, matrix, task_math


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flat_goal_facing_quaternion_matches_column_construction(dtype):
    torch.manual_seed(41)
    start = 100 * torch.rand(10008, 3, dtype=dtype)
    delta = torch.randn_like(start)
    delta[:8] = start.new_tensor([[1, 0, 2], [0, 1, -4], [-1, 0, 0], [0, -1, 1],
                                  [1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]])
    goal = start + delta
    q = task_math.goal_facing_quat(start, goal)
    x_w = goal - start
    x_w[:, 2] = 0
    x_w /= x_w.norm(dim=-1, keepdim=True)
    z_w = torch.zeros_like(x_w)
    z_w[:, 2] = 1
    expected = torch.stack((x_w, torch.cross(z_w, x_w, dim=-1), z_w), dim=-1)
    torch.testing.assert_close(matrix(q), expected, rtol=0, atol=5e-7)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(len(q), dtype=dtype))
    assert not q[:, 1:3].any()  # exactly flat, not just a matching projected yaw


def test_vertical_or_coincident_goal_has_deterministic_flat_world_x_fallback():
    start = torch.tensor([[4., -8., 3.]]).repeat(3, 1)
    goal = start + torch.tensor([[0., 0., 2.], [0., 0., -2.], [0., 0., 0.]])
    state = torch.get_rng_state()
    q = task_math.goal_facing_quat(start, goal)
    assert torch.equal(q, torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1))
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("ids", [torch.tensor([1]), torch.tensor([1, 0]), torch.tensor([], dtype=torch.long)])
def test_reset_uses_new_goal_once_and_skips_all_attitude_noise(ids):
    env = fake_env(config(init_base_attitude_points_to_goal=True, robot_home_quat=(.5, .5, .5, .5),
                          reset_rpy_noise_degs=(30., 40., 170.)))
    before_goal, before_steps = env._goal_position_w.clone(), env._goal_deadline_steps.clone()
    root = torch.zeros(len(ids), 13, dtype=torch.float64)
    root[:, :3] = env.robot.data.root_pos_w[ids] + torch.tensor([.3, -.7, 1.])
    torch.manual_seed(15)
    expected_goal, expected_steps = task_math.sample_goals(root[:, :3], env.cfg, 0, env.step_dt)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(15)
    env._sample_reset_attitude(root, ids)
    assert torch.equal(torch.get_rng_state(), expected_rng)  # no hidden RPY draws
    torch.testing.assert_close(env._goal_position_w[ids], expected_goal.to(before_goal.dtype))
    assert torch.equal(env._goal_deadline_steps[ids], expected_steps)
    torch.testing.assert_close(root[:, 3:7], task_math.goal_facing_quat(root[:, :3], expected_goal))
    kept = torch.tensor([i for i in range(env.num_envs) if i not in ids], dtype=torch.long)
    assert torch.equal(env._goal_position_w[kept], before_goal[kept])
    assert torch.equal(env._goal_deadline_steps[kept], before_steps[kept])


def test_default_off_delegates_to_the_unchanged_home_attitude_sampler():
    assert config().init_base_attitude_points_to_goal is False
    env = fake_env()
    spec = importlib.util.spec_from_file_location("reset_math", FOLDER.parents[3] / "isaaclab/isaaclab/utils/math.py")
    math_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_utils)
    from test_task import method
    env.__class__.__mro__[1]._sample_reset_attitude = method(
        "_sample_reset_attitude", quat_from_euler_xyz=math_utils.quat_from_euler_xyz, quat_mul=math_utils.quat_mul)
    root = torch.zeros(2, 13, dtype=torch.float64)
    root[:, 3:7] = torch.tensor(env.cfg.robot_home_quat)
    torch.manual_seed(88)
    rpy = (2 * torch.rand(2, 3) - 1) * torch.deg2rad(torch.tensor(env.cfg.reset_rpy_noise_degs))
    delta = math_utils.quat_from_euler_xyz(*rpy.unbind(-1))
    expected = math_utils.quat_mul(root[:, 3:7], delta)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(88)
    before = env._goal_position_w.clone()
    env._sample_reset_attitude(root, torch.arange(2))
    assert torch.equal(root[:, 3:7], expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(env._goal_position_w, before)
