import math
from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

import isaaclab.utils.math as math_utils

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env, _episode_reward_ratios
from isaaclab_tasks.direct.solo12.solo12_env_cfg import SAFE_INITIAL_JOINT_POS, Solo12EnvCfg, Solo12TwoFeetEnvCfg


def _bare_env(cfg: Solo12TwoFeetEnvCfg) -> Solo12Env:
    env = Solo12Env.__new__(Solo12Env)
    env.cfg = cfg
    env._is_closed = True
    env._two_feet_curriculum_phase = 1
    env._base_push_force_curriculum_idx = 0
    env._curriculum_event_randomization_active = False
    env._refresh_tricky_terrain_origins = lambda *args, **kwargs: None
    env.event_manager = SimpleNamespace(
        available_modes=("curriculum_startup",),
        calls=[],
    )
    env.event_manager.apply = lambda **kwargs: env.event_manager.calls.append(kwargs)
    env._set_two_feet_curriculum_phase(1)
    return env


def test_two_feet_sac_profile_matches_requested_phase_table():
    cfg = Solo12TwoFeetEnvCfg()

    assert cfg.curriculum_profile == "two_feet_sac"
    assert cfg.command_lin_vel_x_range == (-0.5, 0.5)
    assert cfg.command_lin_vel_y_range == (-0.3, 0.3)
    assert (cfg.kp, cfg.kd) == (9.0, 0.2)
    assert cfg.base_filtered_pairs == ("hip",)
    assert cfg.two_feet_above_height_reward_scale_curriculum == (1.7, 1.2, 1.5, 1.5, 1.5)
    assert cfg.track_lin_vel_xy_reward_scale_curriculum == (1.2, 1.6, 1.5, 1.5, 1.5)
    assert cfg.forces_applied_to_base_curriculum_by_phase == (0.0, 0.0, 0.0, 5.0, 8.0)
    assert cfg.tricky_terrain_curriculum == (False, False, True, True, True)
    assert cfg.include_events_randomization_curriculum == (False, False, True, True, True)


def test_curriculum_initialization_preserves_configured_start_by_default():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)

    assert cfg.skip_curriculum is False
    assert env._initial_two_feet_curriculum_phase() == 1
    assert env._initial_curriculum_level_idx((0.5, 1.0, 1.5)) == 0


def test_skip_curriculum_selects_and_applies_final_two_feet_profile():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.skip_curriculum = True
    env = _bare_env(cfg)

    final_phase = env._initial_two_feet_curriculum_phase()
    env._set_two_feet_curriculum_phase(final_phase)

    assert final_phase == 5
    assert env._initial_curriculum_level_idx((0.5, 1.0, 1.5)) == 2
    assert env.get_curriculum_global_idx() == 4
    assert cfg.two_feet_above_height_reward_scale == 1.5
    assert cfg.track_lin_vel_xy_reward_scale == 1.5
    assert cfg.two_feet_above_height_alpha == 25.0
    assert cfg.actuation_delay_range == (0, 3)
    assert cfg.tricky_terrain_curriculum[final_phase - 1] is True
    assert cfg.opposite_direction_cmd_prob == 0.05
    assert cfg.front_back_asymetry is True
    assert cfg.base_push_force_xy_range == (-8.0, 8.0)
    assert cfg.base_push_force_z_range == (-8.0, 8.0)
    assert env._curriculum_event_randomization_active is True


def test_curriculum_reward_ratio_uses_fixed_maximum_episode_horizon():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)
    env.episode_length_buf = torch.tensor([10, 20])
    scale = cfg.two_feet_above_height_reward_scale_curriculum[0]
    env._episode_sums = {"two_feet_above_height": torch.tensor([scale * 3.0, scale * 7.0])}

    assert env._episode_reward_ratio("two_feet_above_height", torch.tensor([0, 1])) == pytest.approx(0.5)


def test_exact_ratio_threshold_advances_integrated_curriculum():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)
    env.episode_length_buf = torch.tensor([10, 20])
    scale = cfg.two_feet_above_height_reward_scale_curriculum[0]
    env._episode_sums = {
        "two_feet_above_height": torch.full((2,), scale * env.max_episode_length_s * 0.7),
    }

    env._update_two_feet_curriculum(torch.tensor([0, 1]))

    assert env._two_feet_curriculum_phase == 2
    assert env._curriculum_last_reward_ratio == pytest.approx(0.7)


def test_logged_episode_reward_ratios_divide_by_scale_and_fixed_horizon():
    env_ids = torch.tensor([0, 1])
    ratios = _episode_reward_ratios(
        {
            "positive": torch.tensor([1.0, 3.0]),
            "penalty": torch.tensor([-4.0, -8.0]),
            "disabled": torch.tensor([9.0, 9.0]),
        },
        {"positive": 2.0, "penalty": -4.0, "disabled": 0.0},
        env_ids,
        max_episode_length_s=10.0,
        step_dt=0.02,
    )

    assert ratios == pytest.approx({"positive": 0.1, "penalty": 0.15})


def test_soft_joint_limit_episode_ratio_accounts_for_missing_dt_factor():
    ratios = _episode_reward_ratios(
        {"soft_qlim_penalty": torch.tensor([-500.0])},
        {"soft_qlim_penalty": -1.0},
        torch.tensor([0]),
        max_episode_length_s=10.0,
        step_dt=0.02,
    )

    assert ratios == pytest.approx({"soft_qlim_penalty": 1.0})


def test_startup_events_are_deferred_for_early_phases():
    cfg = Solo12TwoFeetEnvCfg()

    cfg.prepare_curriculum_event_randomization()

    assert cfg.events.physics_material.mode == "curriculum_startup"
    assert cfg.events.base_com.mode == "curriculum_startup"


def test_phase_three_enables_terrain_delay_and_startup_randomization():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)

    env._set_two_feet_curriculum_phase(3)

    assert cfg.two_feet_above_height_alpha == 25.0
    assert cfg.track_lin_vel_xy_reward_scale == 1.5
    assert cfg.actuation_delay_range == (0, 3)
    assert env._curriculum_event_randomization_active is True
    assert env.event_manager.calls == [{"mode": "curriculum_startup"}]


def test_force_phases_apply_five_then_eight_newtons_and_vertical_range():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)

    env._set_two_feet_curriculum_phase(4)
    assert cfg.base_push_force_xy_range == (-5.0, 5.0)
    assert cfg.base_push_force_z_range == (-8.0, 8.0)
    assert cfg.opposite_direction_cmd_prob == 0.05

    env._set_two_feet_curriculum_phase(5)
    assert cfg.base_push_force_xy_range == (-8.0, 8.0)
    assert cfg.base_push_force_z_range == (-8.0, 8.0)


def test_curriculum_phase_sets_airborne_reset_probability():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.twofeet_airborne_reset_prob = 0.25
    env = _bare_env(cfg)

    # An empty per-phase list keeps the scalar in every phase.
    env._set_two_feet_curriculum_phase(3)
    assert cfg.twofeet_airborne_reset_prob == 0.25

    cfg.twofeet_airborne_reset_prob_curriculum = (0.5, 0.4, 0.3, 0.2, 0.0)
    env._validate_two_feet_curriculum_config()
    env._set_two_feet_curriculum_phase(3)
    assert cfg.twofeet_airborne_reset_prob == 0.3
    env._set_two_feet_curriculum_phase(5)
    assert cfg.twofeet_airborne_reset_prob == 0.0


@pytest.mark.parametrize(
    ("values", "match"),
    [((0.3, 0.3, 0.3, 0.3), "empty or have 5 values"), ((0.3, 0.3, 1.5, 0.3, 0.3), "between 0 and 1")],
)
def test_curriculum_validation_rejects_bad_airborne_reset_probabilities(values, match):
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)
    cfg.twofeet_airborne_reset_prob_curriculum = values

    with pytest.raises(ValueError, match=match):
        env._validate_two_feet_curriculum_config()


REAR_THIGHS = [7, 10]


def _bare_reset_env(cfg: Solo12TwoFeetEnvCfg, num_envs: int) -> Solo12Env:
    env = Solo12Env.__new__(Solo12Env)
    env.cfg = cfg
    env._is_closed = True
    env.sim = SimpleNamespace(device="cpu")
    env._joint_ids = list(range(len(cfg.joint_names)))
    env._rear_thigh_joint_idx = [cfg.joint_names.index(name) for name in ("RL_thigh_joint", "RR_thigh_joint")]
    env._twofeet_airborne_start = torch.zeros(num_envs, dtype=torch.bool)
    limits = torch.tensor([-3.0, 3.0]).repeat(num_envs, len(cfg.joint_names), 1)
    limits[:, REAR_THIGHS, 1] = math.radians(125.0)
    env._joint_soft_pos_limits = limits
    return env


def _regular_reset_state(num_resets: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    root_pose = torch.zeros(num_resets, 7)
    root_pose[:, 2] = 0.35
    root_pose[:, 3] = 1.0
    root_velocity = torch.full((num_resets, 6), 0.3)
    joint_pos = torch.tensor(list(SAFE_INITIAL_JOINT_POS.values())).repeat(num_resets, 1)
    return root_pose, root_velocity, joint_pos


def test_airborne_reset_is_off_by_default_and_draws_no_random_numbers():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_reset_env(cfg, num_envs=2)
    env._twofeet_airborne_start[:] = True
    root_pose, root_velocity, joint_pos = _regular_reset_state(2)
    expected = [tensor.clone() for tensor in (root_pose, root_velocity, joint_pos)]
    rng_state = torch.get_rng_state()

    env._apply_twofeet_airborne_reset(torch.tensor([0, 1]), torch.zeros(2), root_pose, root_velocity, joint_pos)

    assert cfg.twofeet_airborne_reset_prob == 0.0
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert not env._twofeet_airborne_start.any()
    for tensor, value in zip((root_pose, root_velocity, joint_pos), expected):
        assert torch.equal(tensor, value)


def test_airborne_reset_pivots_base_about_rear_hips_and_turns_rear_thighs():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.twofeet_airborne_reset_prob = 1.0
    cfg.twofeet_airborne_reset_tilt_range = (30.0, 30.0)
    cfg.twofeet_airborne_reset_drop_height_range = (0.05, 0.05)
    env = _bare_reset_env(cfg, num_envs=3)
    env_ids = torch.tensor([0, 2])
    yaw = torch.tensor([math.pi, 0.5])
    root_pose, root_velocity, joint_pos = _regular_reset_state(2)
    regular_joint_pos = joint_pos.clone()

    env._apply_twofeet_airborne_reset(env_ids, yaw, root_pose, root_velocity, joint_pos)

    # 30 degrees from the vertical is 60 degrees nose-up, along the regular heading.
    pitch_up = math.radians(60.0)
    base_x_w = math_utils.quat_apply(root_pose[:, 3:7], torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
    horizontal = math.cos(pitch_up)
    expected_x_w = torch.stack(
        (horizontal * torch.cos(yaw), horizontal * torch.sin(yaw), torch.full((2,), math.sin(pitch_up))), dim=1
    )
    assert torch.allclose(base_x_w, expected_x_w, atol=1e-6)
    # The rear hips keep their regular height; the base center rises around them, plus the drop.
    assert torch.allclose(root_pose[:, 2], torch.full((2,), 0.35 + 0.1946 * math.sin(pitch_up) + 0.05))
    assert torch.equal(root_pose[:, :2], torch.zeros(2, 2))
    # Only the rear thighs change: they turn with the base, so the rear legs keep their world pose.
    expected_joint_pos = regular_joint_pos.clone()
    expected_joint_pos[:, REAR_THIGHS] += pitch_up
    assert torch.allclose(joint_pos, expected_joint_pos)
    assert torch.all(root_velocity.abs() <= 0.1)
    assert env._twofeet_airborne_start.tolist() == [True, False, True]


def test_airborne_reset_clamps_rear_thighs_to_soft_limits():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.twofeet_airborne_reset_prob = 1.0
    # Leaning 60 degrees backward needs -22.9 + 150 = 127.1 degrees of rear thigh; the soft limit is 125.
    cfg.twofeet_airborne_reset_tilt_range = (-60.0, -60.0)
    env = _bare_reset_env(cfg, num_envs=1)
    root_pose, root_velocity, joint_pos = _regular_reset_state(1)

    env._apply_twofeet_airborne_reset(torch.tensor([0]), torch.zeros(1), root_pose, root_velocity, joint_pos)

    assert torch.allclose(joint_pos[0, REAR_THIGHS], torch.full((2,), math.radians(125.0)))


def _bare_velx_force_env(cfg: Solo12EnvCfg) -> Solo12Env:
    env = Solo12Env.__new__(Solo12Env)
    env.cfg = cfg
    env._is_closed = True
    env._max_velx_range_curriculum_values = env._parse_max_velx_range_curriculum()
    env._base_push_force_curriculum_values = env._parse_base_push_force_curriculum()
    env._max_velx_range_curriculum_idx = 0
    env._base_push_force_curriculum_idx = 0
    env._base_push_mean_reward_smooth = None
    env._base_push_last_curriculum_step = 0
    env.common_step_counter = env.max_episode_length
    env._refresh_tricky_terrain_origins = lambda *args, **kwargs: None
    return env


def test_max_episode_reward_sums_bounded_positive_scales_over_episode():
    cfg = Solo12EnvCfg()
    env = _bare_velx_force_env(cfg)

    # track_lin_vel_xy 1.5 + track_ang_vel_z 0.75 over 20 s; unbounded/negative terms are excluded.
    assert env._max_episode_reward() == pytest.approx(45.0)


def test_velx_force_curriculum_advances_at_max_reward_ratio():
    cfg = Solo12EnvCfg()
    cfg.max_velx_range_curriculum = [1.0, 1.5]
    cfg.forces_applied_to_base_curriculum = [10.0, 13.0]
    env = _bare_velx_force_env(cfg)
    threshold = cfg.forces_curriculum_threshold_reward_max_ratio * env._max_episode_reward()

    env._update_base_push_force_curriculum(torch.full((4,), threshold - 0.01, dtype=torch.float64))
    assert env._max_velx_range_curriculum_idx == 0

    env._base_push_mean_reward_smooth = None
    env._update_base_push_force_curriculum(torch.full((4,), threshold, dtype=torch.float64))
    assert env._max_velx_range_curriculum_idx == 1
    assert cfg.command_lin_vel_x_range == (-1.5, 1.5)


def test_velx_force_curriculum_rejects_ratio_outside_unit_interval():
    cfg = Solo12EnvCfg()
    cfg.forces_curriculum_threshold_reward_max_ratio = 1.5

    with pytest.raises(ValueError, match="forces_curriculum_threshold_reward_max_ratio"):
        _bare_velx_force_env(cfg)
