# SPDX-License-Identifier: BSD-3-Clause
"""Independent Newton/Euler, filter and allocation checks for six-action CBRT."""
import importlib

import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.signal import butter, lfilter, lfilter_zi
import torch

from test_control import config, method
from test_sac import algorithm_class
from test_wrench import package

cbrt = importlib.import_module(package.__name__+".borinot_cbrt")


def fixture(mode="cbrt", dtype=torch.float64, n=4, sac=False):
    cfg = config()
    cfg.control_mode, cfg.sac_action_mapping, cfg.cbrt = mode, sac, cbrt.CbrtControllerCfg()
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    pos = torch.tensor(cfg.rotor_pos_b, dtype=dtype)
    a = torch.stack((torch.ones(6, dtype=dtype), pos[:, 1]-.00244, -pos[:, 0],
                     torch.tensor(cfg.rotor_spin, dtype=dtype)*cfg.km_over_kf))
    inertia = torch.tensor([[.027, .001, -.002], [.001, .029, .003], [-.002, .003, .04]], dtype=dtype)
    return cbrt.CbrtController(cfg, n, a, inertia, 2.94*9.81/6)


@pytest.mark.parametrize("mode", ["cbrt", "cbrt_indi", "cbtr", "cbtr_indi"])
@pytest.mark.parametrize("sac", [False, True])
def test_config_aliases_action_sizes_and_explicit_pd(mode, sac):
    c = fixture(mode, sac=sac)
    assert c.task.control_mode == {"cbtr": "cbrt", "cbtr_indi": "cbrt_indi"}.get(mode, mode)
    assert (c.task.action_space, c.task.observation_space, c.task.state_space) == (6, 48, 60)
    assert c.task.robot.actuators["arm"].stiffness == c.task.robot.actuators["arm"].damping == 0
    assert c.task.robot.actuators["arm"].armature == .00036207
    upper, lower = algorithm_class()._compute_action_scaling(type("E", (), {"cfg": c.task})(), "cpu") if sac else (None, None)
    if sac:
        torch.testing.assert_close(upper, torch.ones(6))
        torch.testing.assert_close(lower, torch.ones(6))


@pytest.mark.parametrize("field,value", [("rate_gain", (20.,)), ("rate_gain", (1., -1., 1.)),
    ("allocation_weights", (1., 0., 1., 1.)), ("allocation_iterations", 0), ("allocation_iterations", True),
    ("allocation_iterations", 1.5), ("inertia_scale", 0), ("inertia_scale", float('nan')),
    ("filter_cutoff_hz", 100.), ("filter_cutoff_hz", float('nan'))])
def test_config_invalid_values_fail_before_simulation(field, value):
    c = fixture("cbrt_indi")
    setattr(c.cfg, field, value)
    with pytest.raises(ValueError):
        c.cfg.validate(c.task)


def test_plain_mode_does_not_require_or_instantiate_indi_filters():
    c = fixture()
    c.cfg.filter_cutoff_hz = float('nan')  # unused in the plain path
    c.cfg.validate(c.task)
    assert not hasattr(c, "gyro_filter") and not hasattr(c, "rotor_filter")


@pytest.mark.parametrize("mode", ["cbrt", "cbrt_indi"])
@pytest.mark.parametrize("sac", [False, True])
def test_rate_acceleration_full_euler_pd_and_motor_lag(mode, sac):
    c = fixture(mode, sac=sac)
    actions = torch.tensor([[.6, -.7, .8, .2, .1, -.2]], dtype=torch.float64).repeat(4, 1)
    if not sac:
        actions[:, 3] = 30.
    rates = torch.tensor([[.2, -.4, .5]], dtype=actions.dtype).repeat(4, 1)
    pos, vel = torch.ones(4, 2, dtype=actions.dtype)*.15, torch.ones(4, 2, dtype=actions.dtype)*-.3
    speed = c.rotor_speed.clone()
    before = actions.clone()
    commands, effort = c.step(actions, rates, pos, vel)
    reference = actions[:, :3]*(c.rate_scale if sac else 1)
    alpha = np.asarray(c.cfg.rate_gain)*(reference-rates).numpy()
    np.testing.assert_allclose(c.alpha_desired, alpha, atol=1e-13)
    for i in range(4):
        I, w = c.inertia.numpy(), rates[i].numpy()
        expected = I.dot(alpha[i])+np.cross(w, I.dot(w))
        np.testing.assert_allclose(c.requested_wrench[i, 1:], expected, atol=1e-13)
    target = actions[:, 4:6]*(c.joint_scale if sac else 1)
    expected_pd = c.task.kp*(target-pos)-c.task.kd*vel
    torch.testing.assert_close(c.requested_joint_effort, expected_pd)
    torch.testing.assert_close(effort, expected_pd.clamp(-c.task.effort_lim_sim, c.task.effort_lim_sim))
    torch.testing.assert_close(c.rotor_speed, speed+c.motor_alpha*((commands/c.task.kf).sqrt()-speed))
    torch.testing.assert_close(actions, before)


@pytest.mark.parametrize("mode", ["cbrt", "cbrt_indi"])
def test_sac_collective_endpoints_and_physical_ppo_request(mode):
    c = fixture(mode, n=3, sac=True)
    actions = torch.zeros(3, 6, dtype=torch.float64)
    actions[:, 3] = torch.tensor([-2., 0., 2.])
    zeros = torch.zeros(3, 3, dtype=actions.dtype)
    c.step(actions, zeros, zeros[:, :2], zeros[:, :2])
    expected = torch.tensor([0., c.hover_total, 6*c.allocator.cap], dtype=actions.dtype)
    torch.testing.assert_close(c.requested_wrench[:, 0], expected)
    c.task.sac_action_mapping = False
    actions[:, 0], actions[:, 3] = 8., -15.
    c.step(actions, zeros, zeros[:, :2], zeros[:, :2])
    torch.testing.assert_close(c.requested_wrench[:, 0], torch.full((3,), -15., dtype=actions.dtype))
    torch.testing.assert_close(c.alpha_desired[:, 0], torch.full((3,), 160., dtype=actions.dtype))


@pytest.mark.parametrize("mode", ["cbrt", "cbrt_indi"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_allocation_stage_count_and_scipy_objective(mode, dtype):
    c = fixture(mode, dtype, n=20)
    c.ready.fill_(True)
    actions = torch.randn(20, 6, generator=torch.Generator().manual_seed(31), dtype=dtype)
    actions[:, 3] *= 100
    zeros = torch.zeros(20, 3, dtype=dtype)
    calls, original = [], c.allocator.thrust
    def traced(wrench):
        calls.append(wrench.clone())
        return original(wrench)
    c.allocator.thrust = traced
    commands, _ = c.step(actions, zeros, zeros[:, :2], zeros[:, :2])
    assert len(calls) == (2 if mode == "cbrt_indi" else 1)
    torch.testing.assert_close(calls[0], c.requested_wrench)
    if c.use_indi:
        torch.testing.assert_close(calls[1], c.desired_wrench)
    assert commands.min() >= 0 and commands.max() <= c.allocator.cap
    targets = (c.desired_wrench/c.allocator.scale*c.allocator.weight).double().numpy()
    matrices = c.allocator.a.double().numpy()
    for i in range(20):
        oracle = lsq_linear(matrices[i], targets[i], bounds=(0, 1), tol=1e-13, max_iter=1000)
        cost = np.linalg.norm(matrices[i]@(commands[i]/c.allocator.cap).double().numpy()-targets[i])**2
        assert cost <= 2*oracle.cost+3e-6


def test_second_stage_uses_matched_measurements_and_feasible_alpha_against_scipy():
    c = fixture("cbrt_indi", n=1)
    n = 30
    times = np.arange(n)*c.dt
    rates = np.stack((.4*np.sin(10*times), -.2*np.sin(7*times), .3*np.sin(5*times)), -1)
    speeds = np.sqrt(c.hover/c.task.kf)*(1+np.sin(13*times)[:, None]*np.array([.1,-.2,.15,-.1,.05,-.08]))
    b, a = butter(2, c.cfg.filter_cutoff_hz, fs=c.task.physics_hz)
    filtered_rates = lfilter(b, a, rates, axis=0, zi=lfilter_zi(b,a)[:,None]*rates[0])[0]
    filtered_speeds = lfilter(b, a, speeds, axis=0, zi=lfilter_zi(b,a)[:,None]*speeds[0])[0]
    alpha_measured = np.vstack((np.zeros((1,3)), np.diff(filtered_rates, axis=0)/c.dt))
    action = torch.tensor([[.5, -.8, .6, 31., 0., 0.]], dtype=torch.float64)
    zeros = torch.zeros(1, 2, dtype=action.dtype)
    for i in range(n):
        c.rotor_speed.copy_(torch.from_numpy(speeds[i:i+1]))
        c.step(action, torch.from_numpy(rates[i:i+1]), zeros, zeros)
        np.testing.assert_allclose(c.angular_acceleration[0], alpha_measured[i], atol=1e-12)
        torque_f = c.allocator.allocation[0,1:].numpy().dot(c.task.kf*filtered_speeds[i]**2)
        np.testing.assert_allclose(c.filtered_torque[0], torque_f, atol=2e-13)
        if i:
            I = c.inertia.numpy()
            gyro = np.cross(rates[i], I.dot(rates[i]))
            alpha_star = np.linalg.solve(I, c.nominal_wrench[0,1:].numpy()-gyro)
            expected = torque_f+I.dot(alpha_star-alpha_measured[i])
            np.testing.assert_allclose(c.desired_wrench[0,1:], expected, atol=2e-12)
            np.testing.assert_allclose(c.desired_wrench[0,0], c.nominal_wrench[0,0], atol=1e-12)
    # A correction based on just-computed first-stage torque would erase this measured difference.
    assert not torch.allclose(c.filtered_torque, c.nominal_wrench[:, 1:], atol=1e-3)


@pytest.mark.parametrize("mode", ["cbrt", "cbrt_indi"])
def test_partial_reset_filters_and_neutral_hold(mode):
    c = fixture(mode)
    c.task.hold_arm = True
    actions = torch.ones(4, 6, dtype=torch.float64)*.4
    actions[:,3] = c.hover_total
    rates = torch.ones(4,3,dtype=actions.dtype)*.2
    pos, vel = rates[:,:2]*.5, rates[:,:2]*.1
    c.step(actions, rates, pos, vel)
    previous = c.rotor_speed[1:].clone()
    c.reset(torch.tensor([0]))
    torch.testing.assert_close(c.rotor_speed[1:], previous)
    assert not c.ready[0] and c.ready[1:].all()
    commands, effort = c.step(actions, rates, pos, vel)
    torch.testing.assert_close(effort, (-c.task.kp*pos-c.task.kd*vel).clamp(-c.task.effort_lim_sim,c.task.effort_lim_sim))
    if c.use_indi:
        torch.testing.assert_close(c.angular_acceleration[0], torch.zeros(3,dtype=actions.dtype))
        torch.testing.assert_close(commands[0], c.stage_one_thrust[0])
