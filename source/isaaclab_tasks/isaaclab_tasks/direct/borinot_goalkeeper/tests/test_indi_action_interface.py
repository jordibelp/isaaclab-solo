# SPDX-License-Identifier: BSD-3-Clause
"""PPO physical references, SAC normalized references, and final actuator limits."""
import math
from types import SimpleNamespace

import pytest
import torch

from test_control import method
from test_indi import fixture, home_quat, indi


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("selected", ["geom", "method_1", "method_2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_controller_receives_ppo_si_references_or_sac_scaled_references(monkeypatch, sac, selected, dtype):
    ctrl = fixture(dtype=dtype)
    ctrl.task.sac_action_mapping = sac
    ctrl.task.indi_tilt_method = selected
    # Deliberately nondefault scales: only SAC should use them.
    ctrl.accel_scale = torch.tensor([3., 5., 7.], dtype=dtype)
    ctrl.rate_scale = torch.tensor([4., 6., 8.], dtype=dtype)
    ctrl.joint_scale = torch.tensor([.7, 1.3], dtype=dtype)
    ctrl.cfg.heading_range_degs = 135.
    ctrl.task.kp, ctrl.task.kd, ctrl.cfg.joint_ki = .01, 0., 0.
    actions = torch.tensor([
        [24., -32., 48., 6., -7., 9., 4.2, .1, -.1],
        [-24., 32., -48., -6., 7., -9., -4.2, -.1, .1],
        [0.] * 9,
        [1.7, -2.3, 3.1, 4.4, -5.5, 6.6, -2.7, 2.3, -3.1],
    ], dtype=dtype)
    before, seen = actions.clone(), {}
    original_thrust, original_attitude = indi.thrust_attitude, indi.attitude_acceleration

    def thrust(acceleration, heading, mass):
        seen['acceleration'], seen['heading'] = acceleration.clone(), heading.clone()
        return original_thrust(acceleration, heading, mass)

    def attitude(quaternion, rotation, rates, reference_rates, cfg, selected):
        seen['rates'] = reference_rates.clone()
        return original_attitude(quaternion, rotation, rates, reference_rates, cfg, selected)

    monkeypatch.setattr(indi, 'thrust_attitude', thrust)
    monkeypatch.setattr(indi, 'attitude_acceleration', attitude)
    zeros = torch.zeros(4, 3, dtype=dtype)
    thrusts, efforts = ctrl.step(actions, home_quat(ctrl), zeros, zeros[:, :2], zeros[:, :2])
    expected = actions.clamp(-1, 1) if sac else actions
    torch.testing.assert_close(seen['acceleration'], expected[:, :3] * ctrl.accel_scale if sac else expected[:, :3])
    torch.testing.assert_close(seen['rates'], expected[:, 3:6] * ctrl.rate_scale if sac else expected[:, 3:6])
    torch.testing.assert_close(seen['heading'], ctrl.home_yaw + expected[:, 6] *
                               (math.radians(ctrl.cfg.heading_range_degs) if sac else 1.))
    target = expected[:, 7:9] * ctrl.joint_scale if sac else expected[:, 7:9]
    torch.testing.assert_close(efforts, ctrl.task.kp * target)
    torch.testing.assert_close(actions, before, rtol=0, atol=0)
    assert torch.isfinite(ctrl.alpha_desired).all() and torch.isfinite(thrusts).all()
    assert thrusts.min() >= 0 and thrusts.max() <= ctrl.allocator.cap + 1e-5
    assert efforts.abs().max() <= ctrl.task.effort_lim_sim


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("filename", [
    "borinot_goalkeeper_env.py",
    "../borinot_goalkeeper_simple/borinot_goalkeeper_simple_env.py",
    "../borinot_cmd_vel/borinot_cmd_vel_env.py",
])
def test_actual_prephysics_preserves_ppo_actions_and_only_clips_sac(filename, sac):
    ctrl = fixture(n=2)
    ctrl.task.sac_action_mapping = sac
    ctrl.task.action_clamp = .01  # the non-INDI fallback must not touch PPO INDI
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
    env = SimpleNamespace(
        cfg=ctrl.task, extras={}, _episode_steps=zeros(2), _next_launch_steps=torch.ones(2),
        _ball_active=torch.ones(2, dtype=torch.bool), _shot_steps=zeros(2), _outcome=zeros(2),
        _previous_actuator_commands=zeros(2, 8), _actuator_commands=zeros(2, 8),
        actions=zeros(2, 9), _previous_paddle=zeros(2, 3), _paddle_pose=lambda: (zeros(2, 3),),
        ball=SimpleNamespace(data=SimpleNamespace(root_pos_w=zeros(2, 3), root_lin_vel_w=zeros(2, 3))),
        scene=SimpleNamespace(env_origins=zeros(2, 3)), _old_progress_active=torch.zeros(2, dtype=torch.bool),
        _hit=torch.zeros(2, dtype=torch.bool), _new_hit=zeros(2), _flags={}, _events={},
        _progress_active=torch.zeros(2, dtype=torch.bool), _touched=torch.zeros(2, dtype=torch.bool),
        _self_contact_bodies=zeros(2), _command_steps_left=torch.ones(2), _update_pushes=lambda: None,
        _power_sum_w=zeros(2), _power_samples=0, _physics_samples=0)
    raw = torch.tensor([[24., -32., 48., 6., -7., 9., 4.2, 2.3, -3.1],
                        [-.2, .3, -.4, .5, -.6, .7, -.8, .09, -.1]], dtype=torch.float64)
    before = raw.clone()
    method(filename, '_pre_physics_step')(env, raw)
    torch.testing.assert_close(env.actions, raw.clamp(-1, 1) if sac else raw, rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


@pytest.mark.parametrize("sac", [False, True])
def test_final_thrust_and_joint_effort_remain_bounded_under_extreme_requests(sac):
    ctrl = fixture()
    ctrl.task.sac_action_mapping = sac
    actions = torch.tensor([[1000., -2000., 3000., 100., -200., 300., 30., 20., -30.]],
                           dtype=torch.float64).expand(4, -1).clone()
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    for _ in range(4):
        thrust, effort = ctrl.step(actions, home_quat(ctrl), zeros, zeros[:, :2], zeros[:, :2])
        assert torch.isfinite(thrust).all() and torch.isfinite(effort).all()
        assert thrust.min() >= 0 and thrust.max() <= ctrl.allocator.cap + 1e-9
        actual_thrust = ctrl.task.kf * ctrl.rotor_speed.square()
        assert actual_thrust.min() >= 0 and actual_thrust.max() <= ctrl.allocator.cap + 1e-9
        assert effort.abs().max() <= ctrl.task.effort_lim_sim
    ctrl.task.hold_arm = True
    _, effort = ctrl.step(actions, home_quat(ctrl), zeros, zeros[:, :2], zeros[:, :2])
    assert not effort.any()
