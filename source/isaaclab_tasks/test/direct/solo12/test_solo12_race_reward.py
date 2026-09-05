# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12_race.reward_utils import dense_reaction_force_reward
from isaaclab_tasks.direct.solo12_race.solo12_race_env import (
    Solo12RaceEnv,
    _straight_track_start_to_end_distance_m,
)


def _force(alpha: float, azimuth: float, magnitude: float = 100.0) -> list[float]:
    tangent = magnitude * math.sin(alpha)
    return [
        tangent * math.cos(azimuth),
        tangent * math.sin(azimuth),
        magnitude * math.cos(alpha),
    ]


def test_dense_reaction_force_reward_matches_angle_and_direction_definition():
    mu_static = torch.full((1, 4), 1.0)
    mu_dynamic = torch.full((1, 4), 0.5)
    alpha_static = math.atan(1.0)
    alpha_dynamic = math.atan(0.5)
    alpha_midpoint = 0.5 * (alpha_static + alpha_dynamic)

    forces = torch.tensor(
        [[
            _force(alpha_dynamic, 0.0),
            _force(alpha_midpoint, 0.0),
            _force(alpha_static, math.pi / 2),
            _force(alpha_static, math.pi),
        ]]
    )
    forward_axes = torch.tensor([[[1.0, 0.0, 0.0]] * 4])

    reward = dense_reaction_force_reward(forces, forward_axes, mu_static, mu_dynamic, contact_threshold=1.0)

    # Per-foot terms are 0, +0.5, 0, and -1 respectively.
    torch.testing.assert_close(reward, torch.tensor([-0.5]), atol=1.0e-6, rtol=0.0)


def test_dense_reaction_force_reward_clips_magnitude_and_ignores_invalid_contacts():
    mu_static = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mu_dynamic = torch.tensor([[0.5, 0.5, 1.0, 0.5]])
    forces = torch.tensor(
        [[
            _force(math.atan(2.0), 0.0),
            [0.0, 0.0, 0.5],
            _force(math.atan(1.0), 0.0),
            [10.0, 0.0, -10.0],
        ]]
    )
    forward_axes = torch.tensor([[[1.0, 0.0, 0.0]] * 4])

    reward = dense_reaction_force_reward(forces, forward_axes, mu_static, mu_dynamic, contact_threshold=1.0)

    # Only the first foot is valid: its angle is above alpha_static and therefore clips to 1.
    torch.testing.assert_close(reward, torch.tensor([1.0]), atol=1.0e-6, rtol=0.0)


def test_race_physical_joint_limits_are_written_from_config():
    env = object.__new__(Solo12RaceEnv)
    env._is_closed = True
    env._joint_ids = list(range(6))
    env.cfg = SimpleNamespace(
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
        ],
        joint_physical_limit_hip=(-40.0, 45.0),
        joint_physical_limit_thigh=(-80.0, 85.0),
        joint_physical_limit_calf=(-160.0, 165.0),
    )
    captured = {}
    env._robot = SimpleNamespace(
        data=SimpleNamespace(joint_pos_limits=torch.zeros(2, 6, 2)),
        write_joint_position_limit_to_sim=lambda limits, joint_ids: captured.update(
            limits=limits.clone(), joint_ids=joint_ids
        ),
    )

    env._configure_joint_position_limits()

    expected_degrees = torch.tensor(
        [
            [-40.0, 45.0],
            [-80.0, 85.0],
            [-160.0, 165.0],
            [-40.0, 45.0],
            [-80.0, 85.0],
            [-160.0, 165.0],
        ]
    )
    torch.testing.assert_close(torch.rad2deg(captured["limits"]), expected_degrees.expand(2, -1, -1))
    assert captured["joint_ids"] == env._joint_ids


def test_straight_track_start_to_end_distance_is_planar_and_straight_only():
    waypoints = torch.tensor([[1.0, 2.0, 5.0], [2.0, 3.0, -4.0], [4.0, 6.0, 10.0]])

    assert _straight_track_start_to_end_distance_m("straightSimple", waypoints) == 5.0
    assert _straight_track_start_to_end_distance_m("simple", waypoints) is None


def _make_backward_force_curriculum_env(stages=(1.0, 1.7), threshold=0.6, initial_force=0.0):
    env = object.__new__(Solo12RaceEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        backward_force=initial_force,
        backward_force_curriculum=stages,
        backward_force_curriculum_sr_threshold=threshold,
        race_scene="straightSimple",
    )
    env._configure_backward_force_curriculum()
    return env


def test_backward_force_curriculum_advances_one_stage_per_threshold_crossing():
    env = _make_backward_force_curriculum_env()

    assert env.current_backward_force == 0.0
    assert env.update_backward_force_curriculum(0.6) is False
    assert env.current_backward_force == 0.0

    assert env.update_backward_force_curriculum(0.6001) is True
    assert env.current_backward_force == 1.0
    assert env.update_backward_force_curriculum(0.9) is True
    assert env.current_backward_force == 1.7
    assert env.update_backward_force_curriculum(1.0) is False
    assert env.current_backward_force == 1.7


def test_empty_backward_force_curriculum_keeps_configured_force():
    env = _make_backward_force_curriculum_env(stages=(), initial_force=2.5)

    assert env.current_backward_force == 2.5
    assert env.update_backward_force_curriculum(1.0) is False
    assert env.current_backward_force == 2.5
