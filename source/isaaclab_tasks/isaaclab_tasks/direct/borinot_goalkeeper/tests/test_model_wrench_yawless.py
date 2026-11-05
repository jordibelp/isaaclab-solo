# SPDX-License-Identifier: BSD-3-Clause
"""Independent SO(3) geometry and complete yawless inverse-dynamics checks."""
import importlib
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from test_control import angles, method, raw_action_limit_costs
from test_model_wrench import controller as old_controller
from test_wrench import wrench

yawless = importlib.import_module(wrench.__package__ + ".borinot_model_wrench_yawless")


def controller(n=8, sac=False, dtype=torch.float64, pairs=(False, False)):
    old = old_controller(n, sac, dtype)
    cfg = old.task
    cfg.control_mode = "model_wrench_yawless"
    cfg.predict_thigh_joint_pos_as_sincos, cfg.predict_calf_joint_pos_as_sincos = pairs
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    return yawless.ModelWrenchYawlessController(cfg, old.allocator.allocation, old.model, old.hover)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 3e-13), (torch.float32, 2e-6)])
def test_shortest_rotation_against_scipy_and_all_heading_completions(dtype, tol):
    rng = np.random.default_rng(4)
    current = Rotation.random(256, random_state=rng).as_matrix()
    direction = rng.normal(size=(256, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    error = yawless.minimal_tilt_error(torch.tensor(direction, dtype=dtype), torch.tensor(current, dtype=dtype))
    delta = Rotation.from_rotvec(error.double().numpy())
    target = current @ delta.as_matrix()
    np.testing.assert_allclose(target[:, :, 2], direction, atol=tol, rtol=tol)
    theta = np.arctan2(np.linalg.norm(np.cross(current[:, :, 2], direction), axis=-1),
                       np.sum(current[:, :, 2]*direction, axis=-1))
    np.testing.assert_allclose(error.double().norm(dim=-1), theta, atol=tol, rtol=tol)
    assert not error[:, 2].any()
    # Independently enumerate the complete SO(2) fibre, not just Euler yaw=0.
    for heading in np.linspace(-math.pi, math.pi, 37):
        alternative = target @ Rotation.from_rotvec([0., 0., heading]).as_matrix()
        distance = Rotation.from_matrix(current.transpose(0, 2, 1) @ alternative).magnitude()
        assert np.all(distance + 2*tol >= theta)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-14), (torch.float32, 2e-6)])
def test_identity_zero_force_tiny_tilt_and_both_sides_of_pi(dtype, tol):
    angles_to_test = np.array([0., 1e-9, -1e-9, .2, math.pi-1e-4, math.pi+1e-4, math.pi])
    direction = np.stack((np.sin(angles_to_test), np.zeros_like(angles_to_test), np.cos(angles_to_test)), -1)
    r = torch.eye(3, dtype=dtype).repeat(len(direction), 1, 1)
    e = yawless.minimal_tilt_error(torch.tensor(direction, dtype=dtype), r)
    turned = Rotation.from_rotvec(e.double().numpy()).apply([0., 0., 1.])
    np.testing.assert_allclose(turned, direction, atol=tol, rtol=tol)
    torch.testing.assert_close(yawless.minimal_tilt_error(torch.zeros_like(e), r), torch.zeros_like(e))
    # At the exact antipode choose a body-x half-turn; no unique continuous answer exists.
    torch.testing.assert_close(e[-1], torch.tensor([math.pi, 0., 0.], dtype=dtype), atol=tol, rtol=tol)
    assert torch.isfinite(e).all()


def test_geometry_equivariant_under_arbitrary_world_rotation():
    rng = np.random.default_rng(91)
    r = torch.tensor(Rotation.random(64, random_state=rng).as_matrix())
    f = torch.tensor(rng.normal(size=(64, 3)))
    world = torch.tensor(Rotation.random(random_state=rng).as_matrix())
    torch.testing.assert_close(yawless.minimal_tilt_error(f @ world.mT, world @ r),
                               yawless.minimal_tilt_error(f, r), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("pairs", [(False, False), (True, False), (False, True), (True, True)])
def test_interface_and_joint_pairs(sac, pairs):
    ctrl = controller(sac=sac, pairs=pairs)
    scalar = controller(sac=sac)
    cfg = ctrl.task
    assert cfg.action_space == 8 + sum(pairs)
    assert (cfg.observation_space, cfg.state_space) == (50 + sum(pairs), 62 + sum(pairs))
    assert cfg.robot.actuators["arm"].stiffness == cfg.robot.actuators["arm"].damping == 0
    assert angles.angle_action_slots(cfg) == tuple(6+i for i, enabled in enumerate(pairs) if enabled)
    action = torch.tensor([[.1, -.2, .3, .4, -.3, .2, .15, -.1]]).double().repeat(8, 1)
    physical = action.clone()
    if sac:
        physical[:, -2:] *= ctrl.joint_scale
    parts, start = [], 0
    for slot in angles.angle_action_slots(cfg):
        parts += [action[:, start:slot], torch.stack((physical[:, slot].cos(), physical[:, slot].sin()), -1)*3]
        start = slot + 1
    encoded = torch.cat(parts + [action[:, start:]], -1)
    quat = torch.tensor([[1., 0., 0., 0.]]).double().repeat(8, 1)
    w, q = torch.zeros(8, 3).double(), torch.zeros(8, 2).double()
    scalar.step(action, quat, w, q, q)
    ctrl.step(encoded, quat, w, q, q)
    torch.testing.assert_close(ctrl.requested_wrench, scalar.requested_wrench)
    torch.testing.assert_close(ctrl.requested_joint_effort, scalar.requested_joint_effort)


def test_rejects_yaw_pair_flag_instead_of_adding_a_ghost_channel():
    ctrl = controller()
    ctrl.task.predict_yaw_as_sincos = True
    with pytest.raises(ValueError, match="yaw"):
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(ctrl.task)


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 2e-12), (torch.float32, 2e-4)])
def test_all_eight_dynamics_rows_rate_law_limits_and_one_motor_update(sac, dtype, tol):
    ctrl = controller(sac=sac, dtype=dtype)
    rng = np.random.default_rng(82)
    q, dq, w = (torch.tensor(rng.normal(size=(8, k)), dtype=dtype) for k in [2, 2, 3])
    quat = torch.tensor(Rotation.random(8, random_state=rng).as_quat(scalar_first=True), dtype=dtype)
    action = torch.tensor(rng.normal(size=(8, 8)), dtype=dtype)
    action[:, -2:] *= 20
    before, speed = action.clone(), ctrl.rotor_speed.clone()
    commands, effort = ctrl.step(action, quat, w, q, dq)
    z = torch.cat((ctrl.predicted_acceleration_b, ctrl.alpha_desired, ctrl.predicted_joint_acceleration), -1)
    expected = torch.cat((torch.zeros(8, 2, dtype=dtype), ctrl.requested_wrench, effort), -1)
    # Independent full 8x8 solve, not another Schur-complement implementation.
    independent = np.linalg.solve(ctrl.mass_matrix.double().numpy(),
                                  (expected-ctrl.bias).double().numpy()[..., None])[..., 0]
    np.testing.assert_allclose(z, independent, atol=tol, rtol=tol)
    torch.testing.assert_close(wrench.matvec(ctrl.mass_matrix, z)+ctrl.bias, expected, atol=tol, rtol=tol)
    u = action.clamp(-1, 1) if sac else action
    ref = u[:, 3:6] * ctrl.rate_scale if sac else u[:, 3:6]
    error = yawless.minimal_tilt_error(ctrl.desired_force_w, yawless.rotation_from_quat(quat))
    torch.testing.assert_close(ctrl.alpha_desired, ctrl.cfg.tilt_gain*error +
                               action.new_tensor(ctrl.cfg.rate_gain)*(ref-w))
    # No heading stiffness, but yaw-axis rate control is deliberately retained.
    torch.testing.assert_close(ctrl.alpha_desired[:, 2], ctrl.cfg.rate_gain[2]*(ref[:, 2]-w[:, 2]))
    assert effort.abs().max() <= ctrl.task.effort_lim_sim
    assert commands.min() >= 0 and commands.max() <= ctrl.allocator.cap
    torch.testing.assert_close(ctrl.rotor_speed, speed+ctrl.motor_alpha*((commands/ctrl.task.kf).sqrt()-speed))
    torch.testing.assert_close(action, before)
    acc = u[:, :3]*ctrl.accel_scale if sac else u[:, :3]
    torch.testing.assert_close(ctrl.predicted_acceleration_b[:, 2],
                               wrench.matvec(yawless.rotation_from_quat(quat).mT, acc)[:, 2], atol=tol, rtol=tol)


def test_yaw_gain_heading_settings_and_quaternion_sign_have_no_effect():
    left, right = controller(), controller()
    right.cfg.yaw_gain = 1e6
    right.cfg.heading_range_degs = 3.0
    right.cfg.tilt_method = "method_1"  # New mode always uses the direct minimal-axis error.
    action = torch.randn(8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    quat = torch.tensor(Rotation.random(8, random_state=17).as_quat(scalar_first=True))
    w, q = torch.zeros(8, 3).double(), torch.zeros(8, 2).double()
    a = left.step(action, quat, w, q, q)
    b = right.step(action, -quat, w, q, q)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_partial_reset_clears_only_selected_controller_state():
    ctrl = controller()
    action, w, q = torch.zeros(8, 8).double(), torch.zeros(8, 3).double(), torch.zeros(8, 2).double()
    quat = torch.tensor([[1., 0., 0., 0.]]).double().repeat(8, 1)
    ctrl.step(action, quat, w, q, q)
    saved = {name: value.clone() for name, value in vars(ctrl).items()
             if isinstance(value, torch.Tensor) and value.ndim and len(value) == 8}
    ctrl.reset(torch.tensor([0, 3]))
    for name, value in saved.items():
        torch.testing.assert_close(getattr(ctrl, name)[[1, 2, 4, 5, 6, 7]], value[[1, 2, 4, 5, 6, 7]])
    assert not ctrl.mass_matrix[[0, 3]].any()
    ctrl.step(action, quat, w, q, q)
    assert ctrl.ready.all() and ctrl.mass_matrix[[0, 3]].any()


def test_matches_existing_geometric_tilt_when_only_its_yaw_gain_is_zero():
    new, old = controller(), old_controller()
    old.cfg.yaw_gain = 0.0  # A test instance only; production defaults stay unchanged.
    rng = np.random.default_rng(83)
    action = torch.tensor(rng.normal(size=(8, 8)))
    heading = torch.tensor(rng.normal(size=(8, 1)))
    old_action = torch.cat((action[:, :3], heading, action[:, 3:]), -1)
    quat = torch.tensor(Rotation.random(8, random_state=rng).as_quat(scalar_first=True))
    w, q, dq = (torch.tensor(rng.normal(size=(8, k))) for k in (3, 2, 2))
    new.step(action, quat, w, q, dq)
    old.step(old_action, quat, w, q, dq)
    torch.testing.assert_close(new.alpha_desired, old.alpha_desired, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(new.requested_wrench, old.requested_wrench, atol=2e-12, rtol=2e-12)


def test_full_controller_world_yaw_equivariance_without_a_heading_action():
    left, right = controller(), controller()
    rng = np.random.default_rng(11)
    action = torch.tensor(rng.normal(size=(8, 8)))
    rot = Rotation.random(8, random_state=rng)
    world = Rotation.from_euler("z", 1.4)
    other = action.clone()
    other[:, :3] = action[:, :3] @ torch.tensor(world.as_matrix()).mT
    w, q, dq = (torch.tensor(rng.normal(size=(8, k))) for k in (3, 2, 2))
    left.step(action, torch.tensor(rot.as_quat(scalar_first=True)), w, q, dq)
    right.step(other, torch.tensor((world*rot).as_quat(scalar_first=True)), w, q, dq)
    torch.testing.assert_close(left.requested_wrench, right.requested_wrench, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("sac", [False, True])
def test_soft_limits_follow_yawless_layout_and_hold_arm(sac):
    ctrl = controller(sac=sac, pairs=(True, True))
    cfg = ctrl.task
    cfg.rwd_scale_action_limits = -1.0
    cfg.rwd_scale_thrust_limits = cfg.rwd_scale_torque_limits = 0.0
    action = torch.full((8, 10), 3.0).double()
    cost = raw_action_limit_costs(action, cfg, ctrl.hover)[:, 2]
    torch.testing.assert_close(cost, torch.full_like(cost, 24.0 if sac else 0.0))
    cfg.hold_arm = True
    torch.testing.assert_close(raw_action_limit_costs(action, cfg, ctrl.hover)[:, 2], cost)
