# SPDX-License-Identifier: BSD-3-Clause
"""Independent mechanics and SciPy allocation oracles; no simulator import."""
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import torch

from test_control import config, method

folder = Path(__file__).resolve().parents[3] / "controllers"
package = ModuleType("wrench_test_controllers")
package.__path__ = [str(folder)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__+".borinot_wrench", folder/"borinot_wrench.py")
wrench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrench
spec.loader.exec_module(wrench)


def fixture(dtype=torch.float64, n=4, sac=False):
    cfg = config()
    cfg.control_mode, cfg.sac_action_mapping = "wrench", sac
    cfg.wrench = wrench.WrenchControllerCfg()
    cfg.sim.gravity = (0., 0., -9.81)
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    pos = torch.tensor(cfg.rotor_pos_b, dtype=dtype)
    a = torch.stack((torch.ones(6, dtype=dtype), pos[:, 1], -pos[:, 0],
                     torch.tensor(cfg.rotor_spin, dtype=dtype)*cfg.km_over_kf)).repeat(n, 1, 1)
    # Different actual CoMs, masses and non-diagonal positive-definite inertias.
    a[:, 1] -= torch.linspace(-.02, .02, n, dtype=dtype)[:, None]
    a[:, 2] += torch.linspace(-.01, .01, n, dtype=dtype)[:, None]
    mass = torch.linspace(2.3, 2.8, n, dtype=dtype)
    inertia = torch.tensor([[.027, .001, -.002], [.001, .029, .003], [-.002, .003, .04]], dtype=dtype).repeat(n, 1, 1)
    return wrench.WrenchController(cfg, a, inertia, mass, 2.91*9.81/6)


@pytest.mark.parametrize("sac", [False, True])
def test_config_and_pd_physical_properties(sac):
    ctrl = fixture(sac=sac)
    cfg = ctrl.task
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (9, 51, 63)
    actuator = cfg.robot.actuators["arm"]
    assert actuator.stiffness == actuator.damping == 0
    assert actuator.armature == .00036207
    assert (cfg.kp, cfg.kd, cfg.physics_hz, cfg.policy_hz) == (9., .2, 200, 100)


@pytest.mark.parametrize("field,value", [("allocation_weights", (1, 0, 1, 1)),
    ("allocation_iterations", 0), ("allocation_iterations", True), ("allocation_iterations", 2.5),
    ("tilt_method", "bad"), ("rate_gain", (1., float('nan'), 1.)), ("tilt_gain", -1),
    ("joint_target_limit_degs", (90,)), ("heading_range_degs", float('inf'))])
def test_invalid_config(field, value):
    ctrl = fixture()
    setattr(ctrl.cfg, field, value)
    with pytest.raises(ValueError):
        ctrl.cfg.validate(ctrl.task)


def test_rotated_hip_sign_frame_and_lever_arm_against_spatial_oracle():
    rng = np.random.default_rng(21)
    n = 30
    rb, rc, rj = (Rotation.random(n, random_state=rng).as_matrix() for _ in range(3))
    pc, pj, pb = (rng.normal(size=(n, 3)) for _ in range(3))
    raw = rng.normal(size=(n, 6))
    # Independent virtual force/couple in world coordinates, then change reference point.
    f, m = [], []
    for i in range(n):
        fh = -rc[i].dot(rj[i].dot(raw[i, :3]))
        mh = -rc[i].dot(rj[i].dot(raw[i, 3:]))
        f.append(fh)
        m.append(rb[i].T.dot(mh + np.cross(pc[i]+rc[i].dot(pj[i])-pb[i], fh)))
    actual = wrench.leg_on_base_wrench(*(torch.from_numpy(v) for v in (raw, rc, pc, rj, pj, rb, pb)))
    np.testing.assert_allclose(actual[0], f, atol=2e-14)
    np.testing.assert_allclose(actual[1], m, atol=2e-14)
    # Change the world coordinates; body moment is invariant, world force rotates.
    change = Rotation.from_rotvec([.6, -.4, .7]).as_matrix()
    shifted = wrench.leg_on_base_wrench(*(torch.from_numpy(v) for v in (
        raw, change@rc, pc@change.T, rj, pj, change@rb, pb@change.T)))
    np.testing.assert_allclose(shifted[0], np.asarray(f)@change.T, atol=2e-14)
    np.testing.assert_allclose(shifted[1], m, atol=2e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bounded_allocator_matches_scipy_for_feasible_and_infeasible_requests(dtype):
    ctrl = fixture(dtype, 96)
    allocator = ctrl.allocator
    generator = torch.Generator().manual_seed(43)
    feasible = torch.rand(48, 6, generator=generator, dtype=dtype)*allocator.cap
    request = torch.cat((wrench.matvec(allocator.allocation[:48], feasible),
                         torch.randn(48, 4, generator=generator, dtype=dtype)*torch.tensor([60., 8., 8., 2.], dtype=dtype)))
    result = allocator.thrust(request)
    assert result.min() >= 0 and result.max() <= allocator.cap
    np.testing.assert_allclose(wrench.matvec(allocator.allocation[:48], result[:48]), request[:48], atol=2e-4, rtol=2e-5)
    targets = (request/allocator.scale*allocator.weight).double().numpy()
    matrices = allocator.a.double().numpy()
    for i in range(96):
        oracle = lsq_linear(matrices[i], targets[i], bounds=(0, 1), tol=1e-13, max_iter=1000)
        cost = np.linalg.norm(matrices[i]@(result[i]/allocator.cap).double().numpy()-targets[i])**2
        assert cost <= 2*oracle.cost + 3e-6
    clipped = wrench.matvec(allocator.inverse, request).clamp(0, allocator.cap)
    bounded_cost = ((wrench.matvec(allocator.a, result/allocator.cap)-torch.from_numpy(targets).to(dtype))**2).sum(-1)
    clipped_cost = ((wrench.matvec(allocator.a, clipped/allocator.cap)-torch.from_numpy(targets).to(dtype))**2).sum(-1)
    assert (bounded_cost <= clipped_cost+3e-6).all()
    assert (clipped_cost-bounded_cost).max() > .1



def test_force_attitude_zero_collinear_and_downward():
    force = torch.tensor([[0., 0., 0.], [4., 0., 0.], [0., 0., -7.]], dtype=torch.float64)
    r = torch.tensor(Rotation.from_euler('y', [25., 0., 0.], degrees=True).as_matrix())
    thrust, rd = wrench.force_attitude(force, torch.zeros(3, dtype=torch.float64), r)
    torch.testing.assert_close(rd.mT@rd, torch.eye(3, dtype=r.dtype).repeat(3, 1, 1))
    torch.testing.assert_close(torch.linalg.det(rd), torch.ones(3, dtype=r.dtype))
    torch.testing.assert_close(rd[0, :, 2], r[0, :, 2])
    torch.testing.assert_close(rd[:, :, 2]*thrust, force)


@pytest.mark.parametrize("sac", [False, True])
def test_full_base_equations_and_pd_action_contract(sac):
    ctrl = fixture(n=4, sac=sac)
    ctrl.ready.fill_(True)
    actions = torch.tensor([[.2, -.1, .3, .4, .5, -.6, .7, .8, -.9]], dtype=torch.float64).repeat(4, 1)
    before = actions.clone()
    q = torch.tensor(Rotation.from_euler('xyz', [.3, -.2, .1]).as_quat(scalar_first=True)).repeat(4, 1)
    rate = torch.tensor([[.2, .3, -.4]], dtype=torch.float64).repeat(4, 1)
    pos, vel = torch.ones(4, 2, dtype=torch.float64)*.1, torch.ones(4, 2, dtype=torch.float64)*-.3
    force, moment = torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
    command, effort = ctrl.step(actions, q, rate, pos, vel, force, moment)
    acceleration = actions[:, :3]*(ctrl.accel_scale if sac else 1)
    target = actions[:, 7:9]*(ctrl.joint_scale if sac else 1)
    expected_force = ctrl.mass[:, None]*(acceleration-ctrl.gravity_w)-force
    torch.testing.assert_close(ctrl.desired_force_w, expected_force)
    # Newton/Euler identities reconstructed independently from controller alpha.
    for i in range(4):
        I, w, alpha = ctrl.inertia[i].numpy(), rate[i].numpy(), ctrl.alpha_desired[i].numpy()
        np.testing.assert_allclose(ctrl.requested_wrench[i, 1:]+moment[i], I.dot(alpha)+np.cross(w, I.dot(w)), atol=1e-12)
    torch.testing.assert_close(ctrl.requested_wrench[:, 0], expected_force.norm(dim=-1))
    torch.testing.assert_close(ctrl.requested_joint_effort, ctrl.task.kp*(target-pos)-ctrl.task.kd*vel)
    assert effort.abs().max() <= ctrl.task.effort_lim_sim
    assert command.min() >= 0 and command.max() <= ctrl.allocator.cap
    torch.testing.assert_close(actions, before)
    # Rate channels are [4:7], heading is [3]. Pure rate perturbation changes alpha by K_omega*delta.
    previous = ctrl.alpha_desired.clone()
    actions[:, 4:7] += .05
    ctrl.step(actions, q, rate, pos, vel, force, moment)
    expected_delta = .05*torch.tensor(ctrl.cfg.rate_gain, dtype=q.dtype)*(ctrl.rate_scale if sac else 1)
    torch.testing.assert_close(ctrl.alpha_desired-previous, expected_delta.expand(4, -1))


def test_reset_ignores_stale_impact_and_preserves_other_environments():
    ctrl = fixture()
    ctrl.ready.fill_(True)
    ctrl.rotor_speed *= .7
    other = ctrl.rotor_speed[1:].clone()
    ctrl.reset(torch.tensor([0]))
    torch.testing.assert_close(ctrl.rotor_speed[1:], other)
    assert not ctrl.ready[0] and ctrl.ready[1:].all()
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    actions = torch.zeros(4, 9, dtype=torch.float64)
    q = torch.tensor(ctrl.task.robot_home_quat, dtype=torch.float64).repeat(4, 1)
    huge = torch.full_like(zeros, 1e6)
    commands, _ = ctrl.step(actions, q, zeros, zeros[:, :2], zeros[:, :2], huge, huge)
    torch.testing.assert_close(commands[0], torch.full((6,), ctrl.hover, dtype=q.dtype))
    assert not ctrl.external_force_w[0].any() and not ctrl.external_moment_b[0].any()
    assert ctrl.ready.all()
