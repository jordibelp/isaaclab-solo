# SPDX-License-Identifier: BSD-3-Clause
"""Hybrid INDI mechanics, causality, filter and task-interface contracts."""
import importlib

import numpy as np
import pytest
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation
import torch

from test_control import angles, method, raw_action_limit_costs
from test_model_wrench import controller as nominal_controller
from test_model_wrench_yawless import controller as yawless_controller
from test_wrench import wrench


def hybrid(n=8, yawless=False, sac=False, dtype=torch.float64, ip=0.0):
    module = importlib.import_module(wrench.__package__ + ".borinot_model_wrench_indi")
    old = (yawless_controller if yawless else nominal_controller)(n=n, sac=sac, dtype=dtype)
    cfg = old.task
    cfg.control_mode += "_indi"
    cfg.model_wrench_indi = module.ModelWrenchIndiCfg()
    cfg.indi.rotor_spin_inertia_kg_m2 = ip
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    cls = module.ModelWrenchYawlessIndiController if yawless else module.ModelWrenchIndiController
    return cls(cfg, old.allocator.allocation, old.model, old.hover), old


@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("sac", [False, True])
def test_first_tick_is_exact_nominal_and_interface_is_unchanged(yawless, sac):
    ctrl, old = hybrid(yawless=yawless, sac=sac)
    torch.manual_seed(51)
    q, dq, w = torch.randn(8, 2).double(), torch.randn(8, 2).double(), torch.randn(8, 3).double()
    quat = torch.tensor(Rotation.random(8, random_state=52).as_quat(scalar_first=True))
    action = torch.randn(8, ctrl.task.action_space).double()
    expected = old.step(action, quat, w, q, dq)
    actual = ctrl.step(action, quat, w, q, dq, torch.randn(8, 3).double())
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(ctrl.rotor_speed, old.rotor_speed, atol=0, rtol=0)
    assert not ctrl.disturbance_b.any()
    assert ctrl.task.action_space == old.task.action_space
    assert ctrl.task.observation_space == old.task.observation_space


@pytest.mark.parametrize("ip", [0.0, 2e-5])
@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-12), (torch.float32, 3e-5)])
def test_independent_forward_model_residual_sign_and_realized_torque(yawless, dtype, tol, ip):
    ctrl, _ = hybrid(yawless=yawless, dtype=dtype, ip=ip)
    torch.manual_seed(5)
    q, dq, rates = torch.randn(8, 2, dtype=dtype)*.2, torch.randn(8, 2, dtype=dtype)*.1, torch.randn(8, 3, dtype=dtype)*.1
    quat = torch.tensor(Rotation.random(8, random_state=9).as_quat(scalar_first=True), dtype=dtype)
    rotation = wrench.rotation_from_quat(quat)
    velocity = torch.randn(8, 3, dtype=dtype)
    actions = torch.randn(8, ctrl.task.action_space, dtype=dtype)*.03
    previous_speed = ctrl.rotor_speed.clone()
    _, effort = ctrl.step(actions, quat, rates, q, dq, velocity)
    realized = wrench.matvec(ctrl.allocator.allocation, ctrl.task.kf*ctrl.rotor_speed.square())
    realized[:, 3] += ((ctrl.rotor_speed-previous_speed)/ctrl.dt
                      * torch.tensor(ctrl.task.rotor_spin, dtype=dtype) * ip).sum(-1)
    torch.testing.assert_close(ctrl.applied_wrench, realized, atol=tol, rtol=tol)
    disturbance = torch.tensor([[.03, -.04, .02]], dtype=dtype).repeat(8, 1)
    rhs = torch.cat((torch.zeros(8, 2, dtype=dtype), realized[:, :1], realized[:, 1:]+disturbance, effort), -1)
    # Full eight-row forward solve, independent of the controller's reduced row.
    z = torch.linalg.solve(ctrl.mass_matrix, (rhs-ctrl.bias).unsqueeze(-1)).squeeze(-1)
    next_velocity = velocity+ctrl.dt*wrench.matvec(rotation, z[:, :3])
    next_rates = rates+ctrl.dt*z[:, 3:6]
    # Change references radically. They must not contaminate the old interval residual.
    ctrl.step(actions*10, quat, next_rates, q, dq, next_velocity)
    torch.testing.assert_close(ctrl.raw_disturbance_w, wrench.matvec(rotation, disturbance), atol=tol, rtol=tol)
    b, _ = butter(2, ctrl.task.model_wrench_indi.filter_cutoff_hz, fs=ctrl.task.physics_hz)
    torch.testing.assert_close(ctrl.disturbance_b, b[0]*disturbance, atol=tol, rtol=tol)
    torch.testing.assert_close(ctrl.requested_wrench[:, 1:], ctrl.nominal_wrench[:, 1:]-ctrl.disturbance_b, atol=tol, rtol=tol)
    torch.testing.assert_close(ctrl.requested_wrench[:, :1], ctrl.nominal_wrench[:, :1], atol=0, rtol=0)


@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("field,value", [
    ("rotor_spin_inertia_kg_m2", -1e-5), ("rotor_spin_inertia_kg_m2", float("nan")),
    ("rotor_spin_inertia_kg_m2", float("inf")), ("nonlinear_iterations", 0),
    ("nonlinear_iterations", True), ("nonlinear_iterations", 1.5),
])
def test_shared_inertia_config_is_validated_for_hybrids(yawless, field, value):
    ctrl, _ = hybrid(yawless=yawless)
    setattr(ctrl.task.indi, field, value)
    with pytest.raises(ValueError, match=field):
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(ctrl.task)


@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-8), (torch.float32, 2e-5)])
@pytest.mark.parametrize("lag", [1.0, 0.12])
def test_nonzero_inertia_allocation_lag_and_reset(yawless, dtype, tol, lag):
    ctrl, _ = hybrid(yawless=yawless, dtype=dtype, ip=2e-5)
    ctrl.motor_alpha = lag
    torch.manual_seed(123)
    previous = ctrl.rotor_speed.clone()
    feasible_speed = previous*(.95+.1*torch.rand_like(previous))
    g2 = torch.zeros(4, 6, dtype=dtype)
    g2[3] = torch.tensor(ctrl.task.rotor_spin, dtype=dtype)*2e-5
    target = (wrench.matvec(ctrl.allocator.allocation, ctrl.task.kf*feasible_speed.square())
              + ((feasible_speed-previous)/ctrl.dt) @ g2.T)
    ctrl.requested_wrench.copy_(target)
    commands = ctrl._allocate_rotors()
    independent = (wrench.matvec(ctrl.allocator.allocation, commands)
                   + ((ctrl.command_speed-previous)/ctrl.dt) @ g2.T)
    torch.testing.assert_close(independent, target, atol=tol, rtol=tol)
    torch.testing.assert_close(ctrl.allocation_residual, independent-target, atol=tol, rtol=tol)
    torch.testing.assert_close(ctrl.rotor_speed, previous+lag*((commands/ctrl.task.kf).sqrt()-previous),
                               atol=tol, rtol=tol)
    expected_transient = ((ctrl.rotor_speed-previous)/ctrl.dt) @ g2[1:].T
    torch.testing.assert_close(ctrl.transient_torque, expected_transient, atol=tol, rtol=tol)
    assert ctrl.transient_torque[:, 2].abs().max() > 1e-4
    assert (commands >= 0).all() and (commands <= ctrl.allocator.cap).all()
    keep = ctrl.command_speed[1:].clone(), ctrl.transient_torque[1:].clone()
    ctrl.reset(torch.tensor([0]))
    assert not ctrl.transient_torque[0].any()
    torch.testing.assert_close(ctrl.command_speed[0], ctrl.rotor_speed[0], atol=0, rtol=0)
    torch.testing.assert_close(ctrl.command_speed[1:], keep[0], atol=0, rtol=0)
    torch.testing.assert_close(ctrl.transient_torque[1:], keep[1], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["dsam_indi", "model_wrench_indi", "model_wrench_yawless_indi"])
def test_actual_action_method_applies_inertia_torque_to_plant(mode):
    from types import SimpleNamespace
    from test_control import action_env

    env, calls = action_env("thrust_and_torque_control")
    env.cfg.control_mode = mode
    # Use a nonzero fixture torque so dropping/doubling it cannot pass by roundoff.
    controller = SimpleNamespace(rotor_speed=torch.full((3, 6), 500., dtype=torch.float64),
        transient_torque=torch.tensor([[0., 0., .17]], dtype=torch.float64).repeat(3, 1))
    env.cfg.rwd_scale_thrust_limits = env.cfg.rwd_scale_torque_limits = 0.
    commands = env.cfg.kf*controller.rotor_speed.square()
    effort = torch.zeros(3, 2, dtype=torch.float64)
    if mode == "dsam_indi":
        env._indi = controller
        env.robot.data.root_quat_w = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        env.robot.data.joint_pos = env.robot.data.joint_vel = effort
        controller.step = lambda *args: (commands, effort)
    else:
        env._wrench = controller
        env._step_wrench = lambda _: (commands, effort)
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    expected = commands @ env._allocation[1:].T + controller.transient_torque
    torch.testing.assert_close(calls["wrench"]["torques"][:, 0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("yawless", [False, True])
def test_rotating_body_interval_and_common_filter_against_scipy(yawless):
    ctrl, _ = hybrid(n=1, yawless=yawless)
    q = torch.zeros(1, 2).double()
    velocity, omega_w = torch.zeros(1, 3).double(), torch.tensor([[.3, -.2, .4]]).double()
    quat = torch.tensor([[1., 0., 0., 0.]]).double()
    action = torch.zeros(1, ctrl.task.action_space).double()
    ctrl.step(action, quat, omega_w, q, q, velocity)
    raw_samples = []
    for k in range(30):
        # Forward response to the previously APPLIED wrench and an independent world torque.
        prev_rotation = ctrl.previous_rotation.clone()
        disturbance_w = torch.tensor([[.02*np.sin(k*.3), -.03, .01]]).double()
        rhs = torch.cat((torch.zeros(1, 2).double(), ctrl.applied_wrench[:, :1],
                         ctrl.applied_wrench[:, 1:]+wrench.matvec(prev_rotation.mT, disturbance_w),
                         ctrl.previous_joint_effort), -1)
        z = torch.linalg.solve(ctrl.mass_matrix, (rhs-ctrl.bias).unsqueeze(-1)).squeeze(-1)
        velocity = velocity+ctrl.dt*wrench.matvec(prev_rotation, z[:, :3])
        omega_w = omega_w+ctrl.dt*wrench.matvec(prev_rotation, z[:, 3:6])
        r = Rotation.from_rotvec([.04*k, -.02*k, .03*k]).as_matrix()
        quat = torch.tensor(Rotation.from_matrix(r).as_quat(scalar_first=True))[None]
        rates = omega_w @ torch.tensor(r)
        ctrl.step(action, quat, rates, q, q, velocity)
        raw_samples.append(disturbance_w.numpy()[0])
        b, a = butter(2, ctrl.task.model_wrench_indi.filter_cutoff_hz, fs=ctrl.task.physics_hz)
        expected = lfilter(b, a, np.asarray(raw_samples), axis=0)[-1]
        np.testing.assert_allclose(ctrl.disturbance_b.numpy()[0], r.T@expected, atol=3e-13, rtol=3e-12)


@pytest.mark.parametrize("yawless", [False, True])
def test_partial_reset_discards_old_interval_without_touching_other_environments(yawless):
    ctrl, _ = hybrid(yawless=yawless)
    q, w = torch.zeros(8, 2).double(), torch.zeros(8, 3).double()
    quat = torch.tensor([[1., 0., 0., 0.]]).double().repeat(8, 1)
    action = torch.zeros(8, ctrl.task.action_space).double()
    ctrl.step(action, quat, w, q, q, w)
    ctrl.step(action, quat, w+.1, q, q, w+.1)
    kept = ctrl.residual_filter.z1[1:].clone()
    ctrl.reset(torch.tensor([0]))
    assert not ctrl.measurement_ready[0]
    assert not ctrl.disturbance_b[0].any()
    torch.testing.assert_close(ctrl.residual_filter.z1[1:], kept, atol=0, rtol=0)
    ctrl.step(action, quat, w+100, q, q, w+100)
    assert not ctrl.raw_disturbance_w[0].any()
    assert not ctrl.disturbance_b[0].any()


@pytest.mark.parametrize("yawless", [False, True])
def test_invalid_filter_and_yaw_pair_rejected(yawless):
    ctrl, _ = hybrid(yawless=yawless)
    for value in (0., -1., float("nan"), ctrl.task.physics_hz/2):
        ctrl.task.model_wrench_indi.filter_cutoff_hz = value
        with pytest.raises(ValueError, match="filter_cutoff_hz"):
            method("borinot_goalkeeper_env_cfg.py", "configure_control")(ctrl.task)
    ctrl.task.model_wrench_indi.filter_cutoff_hz = 12.
    if yawless:
        ctrl.task.predict_yaw_as_sincos = True
        with pytest.raises(ValueError, match="yaw"):
            method("borinot_goalkeeper_env_cfg.py", "configure_control")(ctrl.task)


@pytest.mark.parametrize("yawless", [False, True])
@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("bits", range(8))
def test_angle_pairs_and_reference_limit_costs_match_nominal(yawless, sac, bits):
    if yawless and bits & 1:
        pytest.skip("Yawless modes intentionally have no yaw channel.")
    ctrl, old = hybrid(yawless=yawless, sac=sac)
    for c in (ctrl, old):
        c.task.predict_yaw_as_sincos = bool(bits & 1)
        c.task.predict_thigh_joint_pos_as_sincos = bool(bits & 2)
        c.task.predict_calf_joint_pos_as_sincos = bool(bits & 4)
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(c.task)
    assert angles.angle_action_slots(ctrl.task) == angles.angle_action_slots(old.task)
    action = torch.randn(8, ctrl.task.action_space).double()*2
    quat = torch.tensor([[1., 0., 0., 0.]]).double().repeat(8, 1)
    w, q = torch.zeros(8, 3).double(), torch.zeros(8, 2).double()
    actual = ctrl.step(action, quat, w, q, q, w)
    expected = old.step(action, quat, w, q, q)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(raw_action_limit_costs(action, ctrl.task, ctrl.hover),
                               raw_action_limit_costs(action, old.task, old.hover), atol=0, rtol=0)


@pytest.mark.parametrize("yawless", [False, True])
def test_zero_mismatch_stays_zero_for_changing_motion_and_current_arm_commands(yawless):
    ctrl, old = hybrid(yawless=yawless)
    q, rates, velocity = torch.zeros(8, 2).double(), torch.zeros(8, 3).double(), torch.zeros(8, 3).double()
    quat = torch.tensor([[1., 0., 0., 0.]]).double().repeat(8, 1)
    action = torch.zeros(8, ctrl.task.action_space).double()
    ctrl.step(action, quat, rates, q, q, velocity)
    old.step(action, quat, rates, q, q)
    for k in range(12):
        rhs = torch.cat((torch.zeros(8, 2).double(), ctrl.applied_wrench, ctrl.previous_joint_effort), -1)
        z = torch.linalg.solve(ctrl.mass_matrix, (rhs-ctrl.bias).unsqueeze(-1)).squeeze(-1)
        velocity = velocity+ctrl.dt*z[:, :3]
        rates = rates+ctrl.dt*z[:, 3:6]
        action[:, -2:] = .03*np.sin(k*.2)
        actual = ctrl.step(action, quat, rates, q, q, velocity)
        expected = old.step(action, quat, rates, q, q)
        assert ctrl.disturbance_b.abs().max() < 1e-14
        # Fixed-iteration allocation can magnify sub-femtonewton-metre roundoff.
        torch.testing.assert_close(actual[0], expected[0], atol=2e-11, rtol=2e-11)
