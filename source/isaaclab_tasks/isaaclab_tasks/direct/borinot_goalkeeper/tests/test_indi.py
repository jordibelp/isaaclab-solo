# SPDX-License-Identifier: BSD-3-Clause
"""Numerical oracles for frame conventions, filters, allocation and controller state."""
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.spatial.transform import Rotation
import torch

from test_control import action_env, config, method

SPEC = importlib.util.spec_from_file_location("borinot_indi_test", Path(__file__).resolve().parents[1] / "indi.py")
indi = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = indi
SPEC.loader.exec_module(indi)


def fixture(dtype=torch.float64, n=4, hz=300, ip=0.0):
    cfg = config()
    cfg.control_mode, cfg.physics_hz, cfg.policy_hz = "indi", hz, hz/2
    cfg.indi = indi.IndiControllerCfg(rotor_spin_inertia_kg_m2=ip)
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    pos = torch.tensor(cfg.rotor_pos_b, dtype=dtype)
    allocation = torch.stack((torch.ones(6, dtype=dtype), pos[:, 1]-.00244, -pos[:, 0],
                              torch.tensor(cfg.rotor_spin, dtype=dtype)*cfg.km_over_kf))
    inertia = torch.diag(torch.tensor([.02724642, .02685468, .03980717], dtype=dtype))
    return indi.IndiController(cfg, n, allocation, inertia, 2.91, 2.91*9.81/6)


def home_quat(controller):
    return controller.rotor_speed.new_tensor(controller.task.robot_home_quat).expand(controller.rotor_speed.shape[0], -1)


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("hz", [200, 300, 400])
def test_config_resolves_new_mode_without_changing_default_timing(sac, hz):
    cfg = config()
    cfg.control_mode, cfg.indi, cfg.sac_action_mapping = "indi", indi.IndiControllerCfg(), sac
    cfg.physics_hz, cfg.policy_hz = hz, hz/2
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (9, 53, 65)
    assert cfg.sim.dt == 1/hz and cfg.decimation == 2
    assert cfg.robot.actuators["arm"].stiffness == cfg.robot.actuators["arm"].damping == 0


@pytest.mark.parametrize("field,value", [("filter_cutoff_hz", 150), ("filter_cutoff_hz", float('nan')),
    ("acceleration_scale", [1, 2]), ("allocation_weights", [1, 0, 1, 1]),
    ("rotor_spin_inertia_kg_m2", -1e-5), ("joint_ki", -1), ("allocation_iterations", 0),
    ("allocation_iterations", 1.5), ("joint_target_limit_degs", [100, 90])])
def test_invalid_indi_settings_fail_before_simulator(field, value):
    ctrl = fixture()
    setattr(ctrl.cfg, field, value)
    with pytest.raises(ValueError):
        ctrl.cfg.validate(ctrl.task)


def test_unbounded_rotors_are_rejected_only_in_indi():
    ctrl = fixture()
    ctrl.task.remove_thrust_clipping = True
    with pytest.raises(ValueError, match="physical rotor bounds"):
        ctrl.cfg.validate(ctrl.task)


@pytest.mark.parametrize("hz", [200, 300, 400])
def test_matched_butterworth_against_scipy_and_partial_reset(hz):
    torch.manual_seed(21)
    initial = torch.tensor([[2., -1.], [3., 8.]], dtype=torch.float64)
    filt = indi.Butterworth(initial, 12, 1/hz)
    b, a = butter(2, 12, fs=hz)
    values = torch.randn(150, 2, 2, dtype=torch.float64)
    expected = np.empty(values.shape)
    for env in range(2):
        for channel in range(2):
            expected[:, env, channel], _ = lfilter(b, a, values[:, env, channel],
                                                  zi=lfilter_zi(b, a)*initial[env, channel].item())
    actual = torch.stack([filt.update(value) for value in values])
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    other = (filt.z1[1].clone(), filt.z2[1].clone())
    filt.reset(torch.tensor([0]), initial[:1])
    torch.testing.assert_close(filt.z1[1], other[0])
    torch.testing.assert_close(filt.z2[1], other[1])
    for _ in range(5):
        torch.testing.assert_close(filt.update(initial)[0], initial[0], atol=1e-13, rtol=0)


def test_thrust_decomposition_zero_force_and_heading_singularity():
    acceleration = torch.tensor([[0., 0., 0.], [0., 0., -9.81], [1., 0., -9.81]], dtype=torch.float64)
    thrust, rotation = indi.thrust_attitude(acceleration, torch.zeros(3, dtype=torch.float64), 3.)
    torch.testing.assert_close(thrust[:, 0], torch.tensor([29.43, 0., 3.], dtype=torch.float64))
    torch.testing.assert_close(rotation.mT @ rotation, torch.eye(3, dtype=torch.float64).expand(3, -1, -1))
    assert torch.linalg.det(rotation).min() > .99999
    torch.testing.assert_close(rotation[2, :, 2], torch.tensor([1., 0., 0.], dtype=torch.float64))


def test_attitude_error_matches_independent_quaternion_formula_and_global_yaw_invariance():
    rng = np.random.default_rng(4)
    current = Rotation.random(100, random_state=rng)
    desired = Rotation.random(100, random_state=rng)
    q = torch.from_numpy(current.as_quat(scalar_first=True))
    rd = torch.from_numpy(desired.as_matrix())
    qe = (current.inv()*desired).as_quat(scalar_first=True)
    w, x, y, z = qe.T
    denominator = np.sqrt(w*w + z*z)
    cfg = indi.IndiControllerCfg()
    expected = cfg.tilt_gain*np.column_stack(((w*x-y*z)/denominator, (w*y+x*z)/denominator, np.zeros(100)))
    expected[:, 2] = cfg.yaw_gain*np.sign(w)*z/denominator
    zeros = torch.zeros(100, 3, dtype=torch.float64)
    actual = indi.attitude_acceleration(q, rd, zeros, zeros, cfg)
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    yaw = Rotation.from_euler('z', 1.73)
    changed = indi.attitude_acceleration(torch.from_numpy((yaw*current).as_quat(scalar_first=True)),
        torch.from_numpy((yaw*desired).as_matrix()), zeros, zeros, cfg)
    torch.testing.assert_close(actual, changed, atol=1e-10, rtol=0)
    torch.testing.assert_close(actual, indi.attitude_acceleration(-q, rd, zeros, zeros, cfg))


def test_attitude_antipode_is_finite_with_deterministic_tilt_direction():
    cfg = indi.IndiControllerCfg()
    q = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
    rd = torch.diag(torch.tensor([1., -1., -1.], dtype=torch.float64))[None]
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    actual = indi.attitude_acceleration(q, rd, zeros, zeros, cfg)
    torch.testing.assert_close(actual, torch.tensor([[150., 0., 0.]], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_box_allocator_matches_independent_bounded_lsq_and_preserves_tilt_priority(dtype):
    allocator = fixture(dtype).allocator
    generator = torch.Generator().manual_seed(12)
    request = torch.randn(60, 4, generator=generator, dtype=dtype)*torch.tensor([40., 5., 5., 1.])
    solution = allocator.thrust(request)
    assert solution.min() >= 0 and solution.max() <= allocator.cap
    matrix = allocator.a.double().numpy()
    targets = (request/allocator.scale*allocator.weight).double().numpy()
    ours = (solution/allocator.cap).double().numpy()
    costs = ((ours@matrix.T - targets)**2).sum(-1)
    optimal = np.array([np.linalg.norm(matrix@lsq_linear(matrix, target, bounds=(0, 1), tol=1e-12).x-target)**2
                        for target in targets])
    assert np.max(costs - optimal) < 1e-4
    # A feasible but uneven rotor command need not have a feasible minimum-norm inverse.
    force = torch.rand(60, 6, generator=generator, dtype=dtype)*allocator.cap
    wanted = force@allocator.allocation.T
    obtained = allocator.thrust(wanted)@allocator.allocation.T
    assert (obtained-wanted).abs().max() < .025


@pytest.mark.parametrize("ip", [0., 1e-6, 1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equation16_six_rotor_solver_residual_and_bounds(ip, dtype):
    allocator = fixture(dtype, ip=ip).allocator
    generator = torch.Generator().manual_seed(13)
    speed = (.2+.6*torch.rand(20, 6, generator=generator, dtype=dtype))*allocator.speed_max
    previous = (speed+.03*allocator.speed_max*torch.randn(20, 6, generator=generator, dtype=dtype)).clamp(0, allocator.speed_max)
    wanted = (allocator.kf*speed.square())@allocator.allocation.T + ((speed-previous)/allocator.dt)@allocator.g2.T
    solution = allocator.speed(wanted, previous)
    achieved = (allocator.kf*solution.square())@allocator.allocation.T + ((solution-previous)/allocator.dt)@allocator.g2.T
    assert torch.isfinite(solution).all() and solution.min() >= 0 and solution.max() <= allocator.speed_max
    assert (achieved - wanted).abs().max() < .025


def test_joint_pid_antiwindup_and_physics_rate_feedback():
    ctrl = fixture(n=4)
    actions = torch.zeros(4, 9, dtype=torch.float64)
    actions[:, 7] = .01
    zeros, joints = torch.zeros(4, 3, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64)
    _, effort = ctrl.step(actions, home_quat(ctrl), zeros, joints, joints)
    error = math.pi/2*.01
    expected = ctrl.task.kp*error + ctrl.cfg.joint_ki*ctrl.dt*error
    assert effort[0, 0].item() == pytest.approx(expected, abs=1e-14)
    _, changed = ctrl.step(actions, home_quat(ctrl), zeros, joints+.005, joints+.1)
    assert not torch.equal(effort, changed)  # same held policy target, fresh physics feedback
    ctrl.joint_integral.zero_()
    actions[:, 7:] = 1
    for _ in range(10):
        _, effort = ctrl.step(actions, home_quat(ctrl), zeros, joints, joints)
    assert effort.abs().max().item() == ctrl.task.effort_lim_sim
    assert not ctrl.joint_integral.any()  # no integration further into positive saturation
    ctrl.task.hold_arm = True
    _, effort = ctrl.step(actions, home_quat(ctrl), zeros, joints, joints)
    assert not effort.any()


@pytest.mark.parametrize("tc", [0., .04])
def test_speed_lag_not_thrust_lag_and_matched_reset(tc):
    ctrl = fixture()
    ctrl.motor_alpha = 1 if tc == 0 else -math.expm1(-ctrl.dt/tc)
    actions = torch.zeros(4, 9, dtype=torch.float64)
    actions[:, 2] = .4
    speed_before = ctrl.rotor_speed.clone()
    zeros, joints = torch.zeros(4, 3, dtype=torch.float64), torch.zeros(4, 2, dtype=torch.float64)
    commands, _ = ctrl.step(actions, home_quat(ctrl), zeros, joints, joints)
    expected_speed = speed_before + ctrl.motor_alpha*((commands/ctrl.task.kf).sqrt()-speed_before)
    torch.testing.assert_close(ctrl.rotor_speed, expected_speed)
    preserved = [buffer[0].clone() for buffer in (ctrl.rotor_speed, ctrl.command_speed, ctrl.gyro_filter.z1,
                                                 ctrl.rotor_filter.z2, ctrl.joint_integral)]
    ctrl.reset(torch.tensor([1, 3]))
    for buffer, before in zip((ctrl.rotor_speed, ctrl.command_speed, ctrl.gyro_filter.z1,
                               ctrl.rotor_filter.z2, ctrl.joint_integral), preserved):
        torch.testing.assert_close(buffer[0], before, rtol=0, atol=0)
    assert not ctrl.angular_acceleration[[1, 3]].any()
    torch.testing.assert_close(ctrl.observation()[[1, 3]], torch.tensor([[1.]*6+[0.]*2]*2, dtype=torch.float64))
    ctrl.step(torch.zeros_like(actions), home_quat(ctrl), zeros, joints, joints)
    assert ctrl.angular_acceleration[[1, 3]].abs().max() < 1e-12


def test_actual_apply_method_uses_indi_each_substep_and_no_implicit_pd():
    ctrl = fixture()
    env, calls = action_env("thrust_and_torque_control")
    env.cfg = ctrl.task
    env._indi = ctrl
    env.num_envs, env.actions = 4, torch.zeros(4, 9, dtype=torch.float64)
    env._rotor_thrust = torch.full((4, 6), ctrl.hover, dtype=torch.float64)
    env._actuator_commands = torch.zeros(4, 8, dtype=torch.float64)
    env._base_com_offset = torch.zeros(4, 3, dtype=torch.float64)
    env.robot.data.root_quat_w = home_quat(ctrl)
    env.robot.data.root_ang_vel_b = torch.zeros(4, 3, dtype=torch.float64)
    env.robot.data.joint_pos = env.robot.data.joint_vel = torch.zeros(4, 2, dtype=torch.float64)
    env.actions[:, 7] = .01
    apply = method("borinot_goalkeeper_env.py", "_apply_action")
    apply(env)
    first_effort = calls['effort'].clone()
    env.robot.data.joint_vel = torch.ones(4, 2, dtype=torch.float64)
    apply(env)
    assert not torch.equal(first_effort, calls['effort'])
    assert 'position' not in calls and 'velocity' not in calls


def test_closed_loop_rejects_constant_roll_disturbance_without_an_arm_model():
    ctrl = fixture(n=1)
    ctrl.home_yaw = 0
    rate, angle = 0., 0.
    action = torch.zeros(1, 9, dtype=torch.float64)
    joints = torch.zeros(1, 2, dtype=torch.float64)
    peak = 0.
    for tick in range(900):
        q = torch.tensor([[math.cos(angle/2), math.sin(angle/2), 0., 0.]], dtype=torch.float64)
        rates = torch.tensor([[rate, 0., 0.]], dtype=torch.float64)
        ctrl.step(action, q, rates, joints, joints)
        torque = (ctrl.task.kf*ctrl.rotor_speed.square())@ctrl.allocator.allocation.T
        disturbance = .05 if tick >= 300 else 0
        rate += ctrl.dt*(torque[0, 1].item()+disturbance)/ctrl.inertia[0, 0].item()
        angle += ctrl.dt*rate
        peak = max(peak, abs(angle))
    assert peak < .10 and abs(angle) < .005 and abs(rate) < .01
