# SPDX-License-Identifier: BSD-3-Clause
"""Run with ordinary pytest. No simulator launch or package-wide import needed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("goalkeeper_math", Path(__file__).parents[1] / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


@pytest.fixture
def cfg():
    return SimpleNamespace(
        keeper_plane_x=0.1147006, target_center_yz=(-0.0803284, 1.1292502),
        shot_half_extents_yz=((0.045, 0.045), (0.20, 0.12), (0.40, 0.22), (0.65, 0.32)),
        shot_flight_time_ranges_s=((0.42, 0.50), (0.35, 0.50), (0.30, 0.46), (0.25, 0.42)),
        launch_x_range=(1.9, 2.1), launch_z=1.55, launch_y_noise=0.15,
        goal_center=(-0.45, 0.0, 1.2), goal_size=(0.04, 2.4, 2.4), ball_radius_m=0.03,
        curriculum=True, curriculum_min_steps=3000, curriculum_min_episodes=1024,
        curriculum_success_threshold=0.7,
    )


@pytest.mark.parametrize("level", range(4))
def test_shots_hit_sampled_target_and_threaten_goal(cfg, level):
    torch.manual_seed(42 + level)
    position, velocity, target, time_s = task_math.sample_shots(10000, "cpu", cfg, level)
    at_hit = task_math.ballistic_state(position, velocity, time_s)
    torch.testing.assert_close(at_hit, target, atol=1e-6, rtol=1e-5)
    goal_x = cfg.goal_center[0] + cfg.goal_size[0] / 2 + cfg.ball_radius_m
    goal_time = (goal_x - position[:, 0]) / velocity[:, 0]
    at_goal = task_math.ballistic_state(position, velocity, goal_time)
    assert (at_goal[:, 1].abs() < cfg.goal_size[1] / 2 - cfg.ball_radius_m).all()
    assert (at_goal[:, 2] > cfg.ball_radius_m + 0.005).all(), "Unblocked shots must not hit the floor first."
    assert (at_goal[:, 2] < cfg.goal_size[2] - cfg.ball_radius_m).all()


def test_prediction_and_outgoing_mask():
    p = torch.tensor([[2., 0., 1.5], [2., 0., 1.5]])
    v = torch.tensor([[-4., 0., 1.], [4., 0., 1.]])
    target, time_s, valid = task_math.predict_intercept(p, v, 0.)
    assert valid.tolist() == [True, False]
    assert time_s[0] == 0.5
    assert target[0, 2].item() == pytest.approx(1.5 + .5 - .5 * 9.81 * .5**2)


@pytest.mark.parametrize("previous,current,expected", [
    ([0., 0., 1.], [-1., 0., 1.], True),
    ([0., 2., 1.], [-1., 2., 1.], False),
    ([-1., 0., 1.], [0., 0., 1.], False),
    ([0., 1.22, 1.], [-1., 1.22, 1.], True),
    ([0., 0., 3.], [-1., 0., 3.], False),
])
def test_swept_goal_crossing(cfg, previous, current, expected):
    assert task_math.crosses_goal(torch.tensor([previous]), torch.tensor([current]), cfg).item() == expected


def test_failure_precedence_and_codes():
    flags = torch.eye(9, dtype=torch.bool)
    codes = task_math.terminal_outcome(*flags.unbind(1))
    assert codes.tolist() == [1, 2, 8, 3, 4, 5, 9, 6, 7]
    all_flags = [torch.ones(1, dtype=torch.bool)] * 9
    assert task_math.terminal_outcome(*all_flags).item() == 1
    no_base = [torch.zeros(1, dtype=torch.bool), *all_flags[1:]]
    assert task_math.terminal_outcome(*no_base).item() == 3


@pytest.mark.parametrize("age,n,rate,expected", [
    (2999, 1024, .9, False), (3000, 1023, .9, False), (3000, 1024, .7, False),
    (3000, 1024, .7001, True), (4000, 0, 1., False),
])
def test_curriculum_boundary(cfg, age, n, rate, expected):
    assert task_math.promotion_ready(age, n, rate, cfg) == expected


def test_disabled_curriculum(cfg):
    cfg.curriculum = False
    assert not task_math.promotion_ready(10000, 10000, 1.0, cfg)


def test_goal_parallel_progress_ignores_x_and_passive_ball_motion():
    before = torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    after = torch.tensor([[2., 0., 1.], [0., .2, 1.], [0., -.2, 1.], [0., 0., 1.]])
    target = torch.tensor([[4., 1., 1.], [4., 1., 1.], [4., 1., 1.], [8., 2., 1.]])
    result = task_math.plane_progress(before, after, target, .5)
    torch.testing.assert_close(result, torch.tensor([0., .1, -.1, 0.]))
