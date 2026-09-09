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
from isaaclab_tasks.direct.solo12_race.solo12_race_env_cfg import Solo12RaceEnvCfg


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


def _make_patch_boundary_env() -> Solo12RaceEnv:
    env = object.__new__(Solo12RaceEnv)
    env._is_closed = True
    num_envs = 4
    env.sim = SimpleNamespace(device="cpu")
    env._patch_xy_min = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    env._patch_xy_max = torch.tensor([[1.0, 1.0], [1.0, 2.0]])
    env.scene = SimpleNamespace(
        num_envs=num_envs,
        env_origins=torch.tensor([[10.0, 20.0, 0.0]] * num_envs),
    )
    local_root_pos = torch.tensor(
        [
            [0.5, 0.5, 0.4],
            [0.5, 1.5, 0.4],
            [1.01, 0.5, 0.4],
            [-0.01, 1.5, 0.4],
        ]
    )
    env._robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=local_root_pos + env.scene.env_origins))
    env.cfg = SimpleNamespace(
        base_contact_threshold=1.0,
        penalty_leaving_patches=-20.0,
        reset_on_leaving_patches=True,
        leaving_patches_single_feet_outside=False,
        apply_penalty_leaving_patches_and_reset_only_after_seconds=0.0,
        race_scene="straightSimple",
        sim=SimpleNamespace(dt=0.02),
        decimation=1,
        episode_length_s=2.0,
    )
    env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
    env._current_gate_idx = torch.zeros(env.num_envs, dtype=torch.long)
    env._target_count = 8
    env._base_floor_contact_sensor = object()
    env._compute_filtered_base_contact = lambda sensor, threshold: torch.zeros(env.num_envs, dtype=torch.bool)
    return env


def test_leaving_patch_config_defaults_enable_penalty_and_reset():
    cfg = Solo12RaceEnvCfg()

    assert cfg.penalty_leaving_patches == -20.0
    assert cfg.reset_on_leaving_patches is True
    assert cfg.leaving_patches_single_feet_outside is False
    assert cfg.apply_penalty_leaving_patches_and_reset_only_after_seconds == 1.0


def test_finish_time_assigns_maximum_duration_to_unsuccessful_episodes():
    env = object.__new__(Solo12RaceEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.002), decimation=10, episode_length_s=20.0)
    env.episode_length_buf = torch.tensor([275, 40, 999])
    env_ids = torch.tensor([0, 1, 2])
    episode_finished = torch.tensor([True, False, False])

    finish_time_steps = env._compute_finish_time_steps(env_ids, episode_finished)

    assert math.isclose(finish_time_steps, (275 + 1000 + 1000) / 3, rel_tol=1.0e-6)


def test_base_outside_patches_gets_penalty_and_terminates():
    env = _make_patch_boundary_env()

    expected_outside = torch.tensor([False, False, True, True])
    torch.testing.assert_close(env._compute_outside_patches(), expected_outside)
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), expected_outside.float() * -20.0)

    terminated, time_out = env._get_dones()
    torch.testing.assert_close(terminated, expected_outside)
    assert not torch.any(time_out)


def test_straight_track_enclosing_boundary_bridges_internal_patch_gaps():
    env = _make_patch_boundary_env()
    env._patch_xy_min = torch.tensor([[0.0, 0.0], [0.0, 1.1]])
    env._patch_xy_max = torch.tensor([[1.0, 0.9], [1.0, 2.0]])
    local_root_pos = torch.tensor(
        [
            [0.5, 0.5, 0.4],
            [0.5, 1.0, 0.4],
            [0.5, 1.5, 0.4],
            [1.01, 1.0, 0.4],
        ]
    )
    env._robot.data.root_pos_w = local_root_pos + env.scene.env_origins

    expected_outside = torch.tensor([False, False, False, True])
    torch.testing.assert_close(env._compute_outside_patches(), expected_outside)
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), expected_outside.float() * -20.0)
    terminated, _ = env._get_dones()
    torch.testing.assert_close(terminated, expected_outside)

    env.cfg.race_scene = "simple_zigzag"
    torch.testing.assert_close(
        env._compute_outside_patches(), torch.tensor([False, True, False, True])
    )


def test_leaving_patch_reset_can_be_disabled_without_disabling_penalty():
    env = _make_patch_boundary_env()
    env.cfg.reset_on_leaving_patches = False

    terminated, _ = env._get_dones()

    assert not torch.any(terminated)
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), torch.tensor([0.0, 0.0, -20.0, -20.0]))


def test_leaving_patch_penalty_and_reset_start_after_configured_grace_period():
    env = _make_patch_boundary_env()
    env.cfg.apply_penalty_leaving_patches_and_reset_only_after_seconds = 1.0
    env.episode_length_buf = torch.tensor([50, 50, 49, 50])

    torch.testing.assert_close(env._compute_leaving_patches_penalty(), torch.tensor([0.0, 0.0, 0.0, -20.0]))
    terminated, _ = env._get_dones()
    torch.testing.assert_close(terminated, torch.tensor([False, False, False, True]))


def test_missing_patches_do_not_penalize_or_terminate():
    env = _make_patch_boundary_env()
    env._patch_xy_min = torch.empty(0, 2)
    env._patch_xy_max = torch.empty(0, 2)

    assert not torch.any(env._compute_outside_patches())
    assert not torch.any(env._compute_leaving_patches_penalty())
    terminated, _ = env._get_dones()
    assert not torch.any(terminated)


def _set_boundary_test_feet(env):
    # All bases stay inside. Each environment has a different offending foot;
    # the first has only lifted feet, which must remain valid.
    env._robot.data.root_pos_w[:] = env.scene.env_origins + torch.tensor([0.5, 0.5, 0.4])
    local_feet = torch.tensor([[[0.5, 0.5, 0.0]] * 4] * env.num_envs)
    local_feet[0, :, 2] = 2.0
    local_feet[1, 0, 0] = -0.01
    local_feet[2, 1, 0] = 1.01
    local_feet[2, 1, 2] = 2.0  # An airborne foot outside XY still violates the boundary.
    local_feet[3, 3, 1] = 2.01
    env._feet_robot_body_ids = list(range(4))
    env._feet_robot_body_to_foot_offsets_b = torch.tensor([[0.0, 0.0, -0.2]] * 4)
    env._robot.data.body_pos_w = (
        local_feet + env.scene.env_origins[:, None, :] - env._feet_robot_body_to_foot_offsets_b
    )
    env._robot.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4] * env.num_envs)


def test_any_foot_outside_xy_catches_hack_but_legacy_mode_ignores_feet():
    env = _make_patch_boundary_env()
    _set_boundary_test_feet(env)
    assert not torch.any(env._compute_leaving_patches_penalty())
    assert not torch.any(env._get_dones()[0])

    env.cfg.leaving_patches_single_feet_outside = True
    outside = torch.tensor([False, True, True, True])
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), outside.float() * -20.0)
    torch.testing.assert_close(env._get_dones()[0], outside)

    env.cfg.apply_penalty_leaving_patches_and_reset_only_after_seconds = 1.0
    env.episode_length_buf = torch.tensor([50, 49, 50, 51])
    outside = torch.tensor([False, False, True, True])
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), outside.float() * -20.0)
    torch.testing.assert_close(env._get_dones()[0], outside)

    env.cfg.reset_on_leaving_patches = False
    assert not torch.any(env._get_dones()[0])
    torch.testing.assert_close(env._compute_leaving_patches_penalty(), outside.float() * -20.0)


def test_any_foot_boundary_allows_straight_patch_seams_and_edges():
    env = _make_patch_boundary_env()
    _set_boundary_test_feet(env)
    env.cfg.leaving_patches_single_feet_outside = True
    env._patch_xy_min = torch.tensor([[0.0, 0.0], [0.0, 1.1]])
    env._patch_xy_max = torch.tensor([[1.0, 0.9], [1.0, 2.0]])
    # Put all four feet at corners of the outer boundary, then one at an internal seam.
    local_feet = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
    env._robot.data.body_pos_w = (
        local_feet[None, :, :] + env.scene.env_origins[:, None, :] - env._feet_robot_body_to_foot_offsets_b
    )
    env._robot.data.body_pos_w[1, 2, :2] = env.scene.env_origins[1, :2] + torch.tensor([0.5, 1.0])
    assert not torch.any(env._compute_leaving_patches_penalty())
    assert not torch.any(env._get_dones()[0])

    env.cfg.race_scene = "simple_zigzag"
    torch.testing.assert_close(env._get_dones()[0], torch.tensor([False, True, False, False]))

    env._patch_xy_min = torch.empty(0, 2)
    env._patch_xy_max = torch.empty(0, 2)
    assert not torch.any(env._compute_leaving_patches_penalty())
    assert not torch.any(env._get_dones()[0])
