from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12TwoFeetEnvCfg


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
    return env


def test_two_feet_sac_profile_matches_requested_phase_table():
    cfg = Solo12TwoFeetEnvCfg()

    assert cfg.curriculum_profile == "two_feet_sac"
    assert cfg.command_lin_vel_x_range == (-0.5, 0.5)
    assert cfg.command_lin_vel_y_range == (-0.3, 0.3)
    assert (cfg.kp, cfg.kd) == (9.0, 0.2)
    assert cfg.base_filtered_pairs == ("hip",)
    assert cfg.two_feet_above_height_reward_scale_curriculum == (1.5, 1.2, 1.5, 1.5, 1.5)
    assert cfg.track_lin_vel_xy_reward_scale_curriculum == (1.2, 1.5, 1.5, 1.5, 1.5)
    assert cfg.forces_applied_to_base_curriculum_by_phase == (0.0, 0.0, 0.0, 5.0, 8.0)
    assert cfg.tricky_terrain_curriculum == (False, False, True, True, True)
    assert cfg.include_events_randomization_curriculum == (False, False, True, True, True)


def test_curriculum_reward_ratio_is_scale_and_episode_length_independent():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)
    env.episode_length_buf = torch.tensor([10, 20])
    scale = cfg.two_feet_above_height_reward_scale_curriculum[0]
    env._episode_sums = {
        "two_feet_above_height": scale * env.step_dt * env.episode_length_buf * torch.tensor([0.6, 0.8])
    }

    assert env._episode_reward_ratio("two_feet_above_height", torch.tensor([0, 1])) == pytest.approx(0.7)


def test_exact_ratio_threshold_advances_integrated_curriculum():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)
    env.episode_length_buf = torch.tensor([10, 20])
    scale = cfg.two_feet_above_height_reward_scale_curriculum[0]
    env._episode_sums = {
        "two_feet_above_height": scale * env.step_dt * env.episode_length_buf * 0.7,
    }

    env._update_two_feet_curriculum(torch.tensor([0, 1]))

    assert env._two_feet_curriculum_phase == 2
    assert env._curriculum_last_reward_ratio == pytest.approx(0.7)


def test_startup_events_are_deferred_for_early_phases():
    cfg = Solo12TwoFeetEnvCfg()

    cfg.prepare_curriculum_event_randomization()

    assert cfg.events.physics_material.mode == "curriculum_startup"
    assert cfg.events.base_com.mode == "curriculum_startup"


def test_phase_three_enables_terrain_delay_and_startup_randomization():
    cfg = Solo12TwoFeetEnvCfg()
    env = _bare_env(cfg)

    env._set_two_feet_curriculum_phase(3)

    assert cfg.two_feet_above_height_alpha == 20.0
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
