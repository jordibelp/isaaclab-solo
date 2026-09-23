import math
from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg
from isaaclab_tasks.direct.solo12_backflip.solo12_backflip_env import (
    Solo12BackflipEnv,
    _backward_pitch,
    _completed_backflips,
)
from isaaclab_tasks.direct.solo12_backflip.solo12_backflip_env_cfg import Solo12BackflipEnvCfg


def _gravity_after_backward_rotation(angle: torch.Tensor) -> torch.Tensor:
    """Gravity direction in the base frame after a nose-up rotation by ``angle`` about the base -y axis."""
    return torch.stack((-torch.sin(angle), torch.zeros_like(angle), -torch.cos(angle)), dim=1)


def _bare_env(cfg: Solo12BackflipEnvCfg, num_envs: int = 4) -> Solo12BackflipEnv:
    env = Solo12BackflipEnv.__new__(Solo12BackflipEnv)
    env.cfg = cfg
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env._curriculum_event_randomization_active = False
    env.event_manager = SimpleNamespace(available_modes=("curriculum_startup",), calls=[])
    env.event_manager.apply = lambda **kwargs: env.event_manager.calls.append(kwargs)
    env._set_backflip_phase(cfg.backflip_curriculum_start_phase)
    return env


def test_defaults_keep_only_task_reward_and_collision_penalties():
    cfg = Solo12BackflipEnvCfg()

    assert cfg.backflip_ang_vel_reward_scale > 0.0
    assert cfg.base_collision_terminal_penalty == -10.0
    assert cfg.undesired_contact_reward_scale == -2.25
    for name in (
        "action_rate_reward_scale",
        "joint_torque_reward_scale",
        "foot_contact_reward_scale",
        "soft_qlim_penalty_reward_scale",
        "track_lin_vel_xy_reward_scale",
        "track_ang_vel_z_reward_scale",
        "base_tilt_penalty_reward_scale",
        "two_feet_above_height_reward_scale",
        "three_or_more_feet_contact_penalty_reward_scale",
    ):
        assert getattr(cfg, name) == 0.0, name
    assert cfg.command_lin_vel_x_range == cfg.command_lin_vel_y_range == cfg.command_ang_vel_z_range == (0.0, 0.0)
    # Left-right augmentation only: a front-back mirror would turn the backflip into a frontflip.
    assert cfg.front_back_asymetry is True
    assert cfg.observation_space == Solo12EnvCfg().observation_space


def test_backward_pitch_is_positive_for_nose_up_rotation():
    angles = torch.tensor([0.0, 0.3, math.pi / 2, 3.0])
    assert torch.allclose(_backward_pitch(_gravity_after_backward_rotation(angles)), angles, atol=1e-6)


def test_completed_backflips_tolerates_a_short_landing():
    rotation = torch.tensor([-1.0, 1.5 * math.pi, 2.0 * math.pi - 0.5, 4.0 * math.pi + 0.3])
    assert _completed_backflips(rotation).tolist() == [0.0, 0.0, 1.0, 2.0]


def test_rotation_tracker_unwraps_two_backflips_and_skips_side_poses():
    cfg = Solo12BackflipEnvCfg()
    env = _bare_env(cfg, num_envs=1)
    env._backflip_rotation = torch.zeros(1)
    env._backflip_pitch = torch.zeros(1)
    env._backflip_pitch_valid = torch.zeros(1, dtype=torch.bool)
    gravity = torch.zeros(1, 3)
    env._robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))

    for angle in torch.linspace(0.0, 4.0 * math.pi, 81):
        gravity[:] = _gravity_after_backward_rotation(angle.reshape(1))
        env._update_backflip_rotation()
    assert env._backflip_rotation.item() == pytest.approx(4.0 * math.pi, abs=1e-4)

    # Lying on the side leaves the x-z angle undefined; it must not add rotation.
    gravity[:] = torch.tensor([[0.0, -1.0, 0.0]])
    env._update_backflip_rotation()
    gravity[:] = _gravity_after_backward_rotation(torch.tensor([math.pi]))
    env._update_backflip_rotation()
    assert env._backflip_rotation.item() == pytest.approx(4.0 * math.pi, abs=1e-4)


def test_phases_schedule_randomization_delay_and_pushes():
    cfg = Solo12BackflipEnvCfg()
    env = _bare_env(cfg)

    assert cfg.actuation_delay_range == (0, 0)
    assert cfg.base_push_force_xy_range == (0.0, 0.0)
    assert env.event_manager.calls == []

    env._set_backflip_phase(2)
    assert cfg.actuation_delay_range == (0, 3)
    assert env.event_manager.calls == [{"mode": "curriculum_startup"}]

    env._set_backflip_phase(5)
    assert cfg.base_push_force_xy_range == (-8.0, 8.0)
    assert cfg.base_push_force_z_range == (-8.0, 8.0)
    assert env.get_curriculum_global_idx() == env.get_curriculum_max_global_idx() == 4


def test_curriculum_advances_after_a_full_window_of_episodes():
    cfg = Solo12BackflipEnvCfg()
    env = _bare_env(cfg, num_envs=4)
    threshold = cfg.backflip_curriculum_advance_thresholds[0]

    env._update_backflip_curriculum(torch.full((3,), threshold))
    assert env._backflip_phase == 1

    env._update_backflip_curriculum(torch.full((1,), threshold - 1.0))
    assert env._backflip_phase == 1
    assert env._last_window_mean_backflips == pytest.approx(threshold - 0.25)

    env._update_backflip_curriculum(torch.full((4,), threshold))
    assert env._backflip_phase == 2


def test_startup_events_wait_for_the_randomization_phase():
    cfg = Solo12BackflipEnvCfg()
    cfg.prepare_curriculum_event_randomization()
    assert cfg.events.physics_material.mode == "curriculum_startup"
    assert cfg.events.base_com.mode == "curriculum_startup"

    cfg = Solo12BackflipEnvCfg()
    cfg.backflip_curriculum = False
    cfg.prepare_curriculum_event_randomization()
    assert cfg.events.physics_material.mode == "startup"
