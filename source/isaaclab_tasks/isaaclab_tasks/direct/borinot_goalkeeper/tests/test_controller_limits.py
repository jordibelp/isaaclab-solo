# SPDX-License-Identifier: BSD-3-Clause
"""Independent checks of controller demands before actuator saturation."""
import numpy as np
import pytest
import torch

from test_control import MODES, action_env, method
from test_indi import fixture, home_quat


def test_indi_exposes_unprojected_rotor_request_and_unclipped_pid_output():
    ctrl = fixture()
    ctrl.task.sac_action_mapping = False
    actions = torch.zeros(4, 9, dtype=torch.float64)
    actions[:, 2] = 100.
    actions[:, 3] = 30.
    actions[:, 7:] = torch.tensor([2., -2.])
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    thrust, effort = ctrl.step(actions, home_quat(ctrl), zeros, zeros[:, :2], zeros[:, :2])
    requested = ctrl.requested_rotor_thrust
    oracle = np.linalg.lstsq(ctrl.allocator.allocation.numpy(), ctrl.requested_wrench.numpy().T, rcond=None)[0].T
    np.testing.assert_allclose(requested.numpy(), oracle, atol=1e-10, rtol=1e-12)
    assert requested.max() > ctrl.allocator.cap
    assert ctrl.requested_joint_effort.abs().max() > ctrl.task.effort_lim_sim
    assert thrust.min() >= 0 and thrust.max() <= ctrl.allocator.cap + 1e-10
    assert effort.abs().max() <= ctrl.task.effort_lim_sim


def test_pd_cost_uses_computed_not_clipped_effort_and_includes_final_substep():
    env, _ = action_env(MODES[1])
    env._action_limit_costs = torch.zeros(3, 3, dtype=torch.float64)
    env._power_sum_w = torch.zeros(3, 4, dtype=torch.float64)
    env._power_samples = 0
    env.robot.data.applied_torque = torch.full((3, 2), env.cfg.effort_lim_sim, dtype=torch.float64)
    env.robot.data.computed_torque = torch.tensor([[1.1, -1.1]], dtype=torch.float64).expand(3, -1)*env.cfg.effort_lim_sim
    env.robot.data.joint_vel = torch.zeros(3, 2, dtype=torch.float64)
    sample = method('borinot_goalkeeper_env.py', '_sample_power')
    for index in range(env.cfg.decimation):
        env._physics_samples = index + 1
        sample(env)
    torch.testing.assert_close(env._action_limit_costs[:, 1], torch.full((3,), .02, dtype=torch.float64), atol=1e-14, rtol=0)
    before = env._action_limit_costs.clone()
    sample(env)
    torch.testing.assert_close(env._action_limit_costs, before, atol=0, rtol=0)


@pytest.mark.parametrize('decimation', [1, 2, 4])
def test_changing_pd_demands_are_averaged_as_costs_not_averaged_before_squaring(decimation):
    env, _ = action_env(MODES[1])
    env.cfg.decimation = decimation
    env._power_sum_w = torch.zeros(3, 4, dtype=torch.float64)
    env._power_samples = 0
    env.robot.data.applied_torque = torch.zeros(3, 2, dtype=torch.float64)
    env.robot.data.joint_vel = torch.zeros(3, 2, dtype=torch.float64)
    sample = method('borinot_goalkeeper_env.py', '_sample_power')
    wanted = 0.
    for i in range(decimation):
        excess = (i+1)*.1
        env.robot.data.computed_torque = torch.full((3, 2), (1+excess)*env.cfg.effort_lim_sim, dtype=torch.float64)
        env._physics_samples = i+1
        sample(env)
        wanted += 2*excess**2/decimation
    torch.testing.assert_close(env._action_limit_costs[:, 1], torch.full((3,), wanted, dtype=torch.float64), atol=1e-14, rtol=0)


def test_sac_indi_can_have_physical_excess_with_bounded_policy_actions():
    ctrl = fixture()
    ctrl.task.sac_action_mapping = True
    actions = torch.zeros(4, 9, dtype=torch.float64)
    actions[:, 7:] = torch.tensor([1., -1.])
    quaternion = torch.tensor([[2**-.5, 2**-.5, 0., 0.]], dtype=torch.float64).expand(4, -1)
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    thrust, effort = ctrl.step(actions, quaternion, zeros, zeros[:, :2], zeros[:, :2])
    assert ((ctrl.requested_rotor_thrust < 0) | (ctrl.requested_rotor_thrust > ctrl.allocator.cap)).any()
    assert ctrl.requested_joint_effort.abs().max() > ctrl.task.effort_lim_sim
    assert thrust.min() >= 0 and thrust.max() <= ctrl.allocator.cap+1e-10
    assert effort.abs().max() <= ctrl.task.effort_lim_sim


def test_indi_new_request_buffers_clear_only_selected_environments():
    ctrl = fixture()
    actions = torch.full((4, 9), 4., dtype=torch.float64)
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    ctrl.step(actions, home_quat(ctrl), zeros, zeros[:, :2], zeros[:, :2])
    old = [value.clone() for value in (ctrl.requested_wrench, ctrl.requested_rotor_thrust, ctrl.requested_joint_effort)]
    ctrl.reset(torch.tensor([1]))
    for value, before in zip((ctrl.requested_wrench, ctrl.requested_rotor_thrust, ctrl.requested_joint_effort), old):
        assert not value[1].any()
        torch.testing.assert_close(value[[0, 2, 3]], before[[0, 2, 3]], rtol=0, atol=0)


@pytest.mark.parametrize('remove_cap', [False, True])
def test_pi_cost_scores_unclipped_allocation_without_altering_commands(remove_cap):
    from test_control import relative_limit_cost
    env, _ = action_env(MODES[2], motor_time_constant_s=0.)
    env.cfg.remove_thrust_clipping = remove_cap
    env.actions[:, 0] = 10.
    env.actions[:, 1] = -10.
    error = env.actions[:, 1:4]*env._rate_scale
    integral = (env._rate_integral+error*env.physics_dt).clamp(-.5, .5)
    torque = env._kp*error+env._ki*integral
    collective = env._hover_total*(1+env.cfg.collective_action_scale*env.actions[:, :1])
    raw = torch.cat((collective, torque), -1) @ env._allocation_inverse.T
    cap = env._hover_per_rotor*env.cfg.thrust_to_weight
    wanted = relative_limit_cost(raw, 0., None if remove_cap else cap, cap)/env.cfg.decimation
    method('borinot_goalkeeper_env.py', '_apply_action')(env)
    torch.testing.assert_close(env._action_limit_costs[:, 0], wanted, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(env._rotor_thrust, raw.clamp(0., None if remove_cap else cap), atol=1e-14, rtol=1e-14)
