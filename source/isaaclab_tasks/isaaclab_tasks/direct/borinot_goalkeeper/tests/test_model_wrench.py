# SPDX-License-Identifier: BSD-3-Clause
"""Independent Pinocchio mechanics oracle and model-wrench integration contracts."""
import importlib
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from test_control import config, method
from test_wrench import wrench

model_module = importlib.import_module(wrench.__package__ + ".borinot_model_wrench")


def description(tool=True, armature=.00036207):
    """Declared synthetic geometry: offset hip, skew axes and non-diagonal inertias."""
    bodies = [dict(name="base", mass=2.3, com=[.018, -.013, .007],
                   inertia=[[.027, .001, -.002], [.001, .029, .003], [-.002, .003, .04]])]
    for i in range(1, 4 if tool else 3):
        bodies.append(dict(name=f"link{i}", parent=i-1, dof=i-1 if i < 3 else -1,
            mass=[.17, .12, .1][i-1], com=[.005, -.075, .012],
            inertia=[[.0008, .00003, 0.], [.00003, .0004, .00001], [0., .00001, .0006]],
            p0=[.021, -.13 if i > 1 else .02, -.015], p1=[.01, -.006, .004],
            r0=Rotation.from_euler('xyz', [.13*i, -.11, .4*i]).as_matrix().tolist(),
            r1=Rotation.from_euler('xyz', [-.12, .18*i, -.17]).as_matrix().tolist(),
            axis=[0., 1., 0.] if i == 2 else [1., 0., 0.]))
    return dict(format_version=1, bodies=bodies, armature=[armature]*2)


def pinocchio_model(desc):
    pin = pytest.importorskip("pinocchio")
    model = pin.Model()
    base_id = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "base")
    ids, transforms = [base_id], [pin.SE3.Identity()]
    for i, b in enumerate(desc["bodies"]):
        if i:
            p = b["parent"]
            t0 = pin.SE3(np.asarray(b["r0"]), np.asarray(b["p0"]))
            t1 = pin.SE3(np.asarray(b["r1"]), np.asarray(b["p1"]))
            if b["dof"] >= 0:
                jid = model.addJoint(ids[p], pin.JointModelRevoluteUnaligned(np.asarray(b["axis"])),
                                     transforms[p]*t0, b["name"])
                transform = t1.inverse()
            else:
                jid, transform = ids[p], transforms[p]*t0*t1.inverse()
            ids.append(jid)
            transforms.append(transform)
        model.appendBodyToJoint(ids[i], pin.Inertia(b["mass"], np.asarray(b["com"]), np.asarray(b["inertia"])), transforms[i])
    return pin, model


@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-13), (torch.float32, 5e-6)])
def test_mass_and_bias_against_independent_pinocchio(tool, dtype, tolerance):
    desc = description(tool)
    ours = model_module.FloatingArmModel(desc, dtype=dtype)
    pin, other = pinocchio_model(desc)
    rng = np.random.default_rng(123)
    n = 32
    q, dq, omega, gravity = (rng.normal(size=(n, size))*scale for size, scale in [(2, 1.), (2, 2.), (3, 1.2), (3, 5.)])
    matrix, bias = ours.dynamics(*(torch.tensor(x, dtype=dtype) for x in (q, dq, omega, gravity)))
    centre = np.asarray(desc["bodies"][0]["com"])
    shift = np.eye(8)
    shift[:3, 3:6] = np.array([[0, -centre[2], centre[1]], [centre[2], 0, -centre[0]], [-centre[1], centre[0], 0]])
    data = other.createData()
    for i in range(n):
        qp = np.r_[np.zeros(3), [0, 0, 0, 1], q[i]]
        vp = shift @ np.r_[np.zeros(3), omega[i], dq[i]]
        other.gravity.linear = gravity[i]
        mp = pin.crba(other, data, qp).copy()
        # Installed Pinocchio 2.x has no model.armature field. Add this declared
        # constant generalized inertia after its independent rigid-body CRBA.
        mp[6:, 6:] += np.diag(desc["armature"])
        hp = pin.rnea(other, data, qp, vp, np.zeros(8)).copy()
        np.testing.assert_allclose(matrix[i], shift.T@mp@shift, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(bias[i], shift.T@hp, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(matrix, matrix.mT, atol=tolerance, rtol=tolerance)
    assert (torch.linalg.eigvalsh(matrix) > 0).all()


def controller(n=8, sac=False, dtype=torch.float64):
    cfg = config()
    cfg.control_mode, cfg.sac_action_mapping = "model_wrench", sac
    cfg.wrench = wrench.WrenchControllerCfg()
    cfg.sim.gravity = (0., 0., -9.81)
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    model = model_module.FloatingArmModel(description(), dtype=dtype)
    pos = torch.tensor(cfg.rotor_pos_b, dtype=dtype)-model.bodies[0]["com"]
    allocation = torch.stack((torch.ones(6, dtype=dtype), pos[:, 1], -pos[:, 0],
        torch.tensor(cfg.rotor_spin, dtype=dtype)*cfg.km_over_kf)).repeat(n, 1, 1)
    return model_module.ModelWrenchController(cfg, allocation, model, model.total_mass*9.81/6)


@pytest.mark.parametrize("sac", [False, True])
def test_new_mode_resolves_and_has_same_interface_as_wrench(sac):
    ctrl = controller(sac=sac)
    assert (ctrl.task.action_space, ctrl.task.observation_space, ctrl.task.state_space) == (9, 51, 63)
    assert ctrl.task.robot.actuators["arm"].stiffness == ctrl.task.robot.actuators["arm"].damping == 0


@pytest.mark.parametrize("sac", [False, True])
def test_full_eight_equations_and_four_channel_inversion(sac):
    ctrl = controller(sac=sac)
    rng = np.random.default_rng(54)
    q, dq, omega = (torch.tensor(rng.normal(size=(8, k)), dtype=torch.float64) for k in [2, 2, 3])
    quaternion = torch.tensor(Rotation.random(8, random_state=rng).as_quat(scalar_first=True))
    action = torch.tensor(rng.normal(size=(8, 9)), dtype=torch.float64)
    action[:, 7:] *= 5  # prove the dynamics uses SATURATED effort, not the PD request
    before = action.clone()
    old_speed = ctrl.rotor_speed.clone()
    command, effort = ctrl.step(action, quaternion, omega, q, dq)
    z = torch.cat((ctrl.predicted_acceleration_b, ctrl.alpha_desired, ctrl.predicted_joint_acceleration), -1)
    lhs = wrench.matvec(ctrl.mass_matrix, z)+ctrl.bias
    expected = torch.cat((torch.zeros(8, 2, dtype=q.dtype), ctrl.requested_wrench[:, :1], ctrl.requested_wrench[:, 1:], effort), -1)
    torch.testing.assert_close(lhs, expected, atol=3e-13, rtol=3e-13)
    rot = model_module.rotation_from_quat(quaternion)
    a = action[:, :3].clamp(-1, 1)*ctrl.accel_scale if sac else action[:, :3]
    torch.testing.assert_close(ctrl.predicted_acceleration_b[:, 2], wrench.matvec(rot.mT, a)[:, 2], atol=1e-13, rtol=1e-13)
    assert effort.abs().max() <= ctrl.task.effort_lim_sim
    assert command.min() >= 0 and command.max() <= ctrl.allocator.cap
    torch.testing.assert_close(ctrl.rotor_speed, old_speed+ctrl.motor_alpha*((command/ctrl.task.kf).sqrt()-old_speed))
    torch.testing.assert_close(action, before)


def test_static_and_freefall_gravity_limits():
    model = model_module.FloatingArmModel(description(False, 0), dtype=torch.float64)
    q, w = torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64)
    gravity = torch.tensor([[0., 0., -9.81]], dtype=q.dtype)
    matrix, bias = model.dynamics(q, q, w, gravity)
    # A locked, static arm needs total weight. Joint rows specify the holding torque.
    torch.testing.assert_close(bias[:, :3], -model.total_mass*gravity)
    freefall = torch.cat((gravity, w, q), -1)
    torch.testing.assert_close(wrench.matvec(matrix, freefall)+bias, torch.zeros(1, 8, dtype=q.dtype), atol=1e-13, rtol=0)


def test_partial_reset_and_fresh_first_tick():
    ctrl = controller()
    action = torch.zeros(8, 9, dtype=torch.float64)
    quaternion = torch.tensor([[1., 0., 0., 0.]], dtype=action.dtype).repeat(8, 1)
    w, q = torch.zeros(8, 3, dtype=action.dtype), torch.zeros(8, 2, dtype=action.dtype)
    first, _ = ctrl.step(action, quaternion, w, q, q)
    speed = ctrl.rotor_speed.clone()
    ctrl.reset(torch.tensor([0]))
    torch.testing.assert_close(ctrl.rotor_speed[1:], speed[1:])
    assert not ctrl.mass_matrix[0].any()
    ctrl.step(action, quaternion, w, q, q)
    assert ctrl.mass_matrix[0].any() and ctrl.ready.all()
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("flags", [(False, False, False), (True, False, True), (False, True, True), (True, True, True)])
def test_angle_pairs_match_scalar_references(sac, flags):
    scalar_ctrl = controller(sac=sac)
    pair_ctrl = controller(sac=sac)
    cfg = pair_ctrl.task
    cfg.predict_yaw_as_sincos, cfg.predict_thigh_joint_pos_as_sincos, cfg.predict_calf_joint_pos_as_sincos = flags
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    scalar = torch.tensor([[.1, -.1, .2, .15, .2, -.3, .1, .1, -.1]], dtype=torch.float64).repeat(8, 1)
    physical = scalar.clone()
    if sac:
        physical[:, 3] = scalar[:, 3]*math.radians(cfg.wrench.heading_range_degs)
        physical[:, 7:9] = scalar[:, 7:9]*scalar_ctrl.joint_scale
    parts, start = [], 0
    angles = importlib.import_module(wrench.__package__+".borinot_angles")
    for slot in angles.angle_action_slots(cfg):
        parts.extend((scalar[:, start:slot], torch.stack((physical[:, slot].cos(), physical[:, slot].sin()), -1)))
        start = slot+1
    parts.append(scalar[:, start:])
    encoded = torch.cat(parts, -1)
    quaternion = torch.tensor([[1., 0., 0., 0.]], dtype=scalar.dtype).repeat(8, 1)
    zero3, zero2 = torch.zeros(8, 3, dtype=scalar.dtype), torch.zeros(8, 2, dtype=scalar.dtype)
    scalar_ctrl.step(scalar, quaternion, zero3, zero2, zero2)
    pair_ctrl.step(encoded, quaternion, zero3, zero2, zero2)
    torch.testing.assert_close(pair_ctrl.requested_wrench, scalar_ctrl.requested_wrench)
    torch.testing.assert_close(pair_ctrl.requested_joint_effort, scalar_ctrl.requested_joint_effort)


def test_common_world_rotation_changes_only_world_vectors():
    left, right = controller(), controller()
    rng = np.random.default_rng(12)
    action = torch.tensor(rng.normal(size=(8, 9)), dtype=torch.float64)
    q, dq, omega = (torch.tensor(rng.normal(size=(8, k)), dtype=torch.float64) for k in [2, 2, 3])
    rot = Rotation.random(8, random_state=rng)
    yaw = Rotation.from_euler('z', .72)
    quat = torch.tensor(rot.as_quat(scalar_first=True))
    other_quat = torch.tensor((yaw*rot).as_quat(scalar_first=True))
    other_action = action.clone()
    other_action[:, :3] = action[:, :3] @ torch.tensor(yaw.as_matrix()).mT
    other_action[:, 3] += .72
    left.step(action, quat, omega, q, dq)
    right.step(other_action, other_quat, omega, q, dq)
    torch.testing.assert_close(right.requested_wrench, left.requested_wrench)
    torch.testing.assert_close(right.predicted_joint_acceleration, left.predicted_joint_acceleration)
    torch.testing.assert_close(right.desired_force_w, left.desired_force_w @ torch.tensor(yaw.as_matrix()).mT)
