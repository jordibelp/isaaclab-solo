# SPDX-License-Identifier: BSD-3-Clause
"""Method selection and independent pure-tilt/twist geometry checks."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from test_control import config, method
from test_indi import fixture, indi


METHODS = ("geom", "method_1", "method_2")


def oracle(error, selected):
    tau = error.as_rotvec()
    yaw = np.zeros_like(tau)
    if selected != "geom":
        yaw[:, 2] = tau[:, 2]
        tilt = ((error * Rotation.from_rotvec(-yaw)).as_rotvec()
                if selected == "method_1" else tau-yaw)
        return tilt, yaw
    # Independent geometry: shortest rotation between the two thrust axes.
    z = error.as_matrix()[:, :, 2]
    cross = np.column_stack((-z[:, 1], z[:, 0], np.zeros(len(z))))
    sine = np.linalg.norm(cross, axis=-1)
    angle = np.arctan2(sine, z[:, 2])
    scale = np.divide(angle, sine, out=np.ones_like(angle), where=sine > 1e-14)
    tilt = cross*scale[:, None]
    yaw = (Rotation.from_rotvec(tilt).inv()*error).as_rotvec()
    return tilt, yaw


def test_default_uses_geometric_tilt_and_twist_for_notebook_example():
    q = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
    error = Rotation.from_quat([[.5, -.5, .5, .5]])  # scipy xyzw
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    actual = indi.attitude_acceleration(q, torch.from_numpy(error.as_matrix()), zeros, zeros, indi.IndiControllerCfg())
    np.testing.assert_allclose(actual, [[150*math.pi/2, 0., 3*math.pi/2]], atol=1e-12)


@pytest.mark.parametrize("selected", METHODS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_selected_vectors_and_feedback_match_independent_geometry(selected, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(20261024)
    current, desired = Rotation.random(2000, random_state=rng), Rotation.random(2000, random_state=rng)
    error = current.inv()*desired
    expected_tilt, expected_yaw = oracle(error, selected)
    q_error = torch.tensor(error.as_quat(scalar_first=True), dtype=dtype, device=device)
    tilt, yaw = indi.attitude_error_vectors(q_error, selected)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
    np.testing.assert_allclose(tilt.cpu(), expected_tilt, atol=tolerance, rtol=0)
    np.testing.assert_allclose(yaw.cpu(), expected_yaw, atol=tolerance, rtol=0)
    negative = indi.attitude_error_vectors(-q_error, selected)
    for actual, changed in zip((tilt, yaw), negative):
        torch.testing.assert_close(actual, changed, atol=tolerance, rtol=0)
    if selected == "geom":
        assert tilt[:, 2].abs().max() < tolerance
        assert yaw[:, :2].abs().max() == 0
        # Compose both rotations; tilt alone must reproduce the thrust direction.
        reconstructed = indi.quaternion_multiply(indi.quaternion_exp(tilt), indi.quaternion_exp(yaw))
        np.testing.assert_allclose(indi.rotation_from_quat(reconstructed).cpu(), error.as_matrix(), atol=tolerance, rtol=0)
        np.testing.assert_allclose(indi.rotation_from_quat(indi.quaternion_exp(tilt))[:, :, 2].cpu(),
                                   error.as_matrix()[:, :, 2], atol=tolerance, rtol=0)
    rates, reference = rng.normal(size=(2, 2000, 3))
    cfg = indi.IndiControllerCfg()
    q = torch.tensor(current.as_quat(scalar_first=True), dtype=dtype, device=device)
    rd = torch.tensor(desired.as_matrix(), dtype=dtype, device=device)
    r, rr = torch.tensor(rates, dtype=dtype, device=device), torch.tensor(reference, dtype=dtype, device=device)
    alpha = indi.attitude_acceleration(q, rd, r, rr, cfg, selected)
    expected = cfg.tilt_gain*expected_tilt + cfg.yaw_gain*expected_yaw + np.asarray(cfg.rate_gain)*(reference-rates)
    np.testing.assert_allclose(alpha.cpu(), expected, atol=8e-4 if dtype == torch.float32 else 5e-11, rtol=0)
    world = Rotation.from_rotvec([.7, -.4, 1.3])
    changed = indi.attitude_acceleration(
        torch.tensor((world*current).as_quat(scalar_first=True), dtype=dtype, device=device),
        torch.tensor((world*desired).as_matrix(), dtype=dtype, device=device), r, rr, cfg, selected)
    # Near inverted thrust, float32 pose rounding is amplified by the geometric
    # split. 0.0015 rad/s^2 is 1e-5 rad at gain 150 (0.0006 degrees).
    torch.testing.assert_close(alpha, changed, atol=1.5e-3 if dtype == torch.float32 else 5e-11, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometric_identity_and_inverted_thrust_axes_are_finite(dtype):
    errors = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.],
                           [0., .6, -.8, 0.]], dtype=dtype)
    tilt, yaw = indi.attitude_error_vectors(errors, "geom")
    assert torch.isfinite(tilt).all() and torch.isfinite(yaw).all()
    assert not yaw.any() and not tilt[:, 2].any()
    torch.testing.assert_close(tilt[0], torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(tilt[1:].norm(dim=-1), torch.full((3,), math.pi, dtype=dtype))
    torch.testing.assert_close(indi.rotation_from_quat(indi.quaternion_exp(tilt)),
                               indi.rotation_from_quat(errors), atol=5e-7 if dtype == torch.float32 else 1e-14, rtol=0)


@pytest.mark.parametrize("selected", METHODS)
def test_controller_step_routes_selected_method(selected):
    ctrl = fixture()
    ctrl.task.indi_tilt_method = selected
    actions = torch.zeros(4, 9, dtype=torch.float64)
    q = torch.from_numpy(Rotation.from_rotvec([[.8, -.6, 1.2]]*4).as_quat(scalar_first=True))
    zeros = torch.zeros(4, 3, dtype=torch.float64)
    ctrl.step(actions, q, zeros, zeros[:, :2], zeros[:, :2])
    desired = Rotation.from_euler('z', [ctrl.home_yaw]*4)
    tilt, yaw = oracle(Rotation.from_quat(q.numpy(), scalar_first=True).inv()*desired, selected)
    np.testing.assert_allclose(ctrl.alpha_desired, ctrl.cfg.tilt_gain*tilt + ctrl.cfg.yaw_gain*yaw, atol=1e-12)


@pytest.mark.parametrize("filename", ["borinot_goalkeeper_env_cfg.py",
    "../borinot_goalkeeper_simple/borinot_goalkeeper_simple_env_cfg.py"])
def test_invalid_method_is_rejected_by_both_config_resolvers(filename):
    cfg = config()
    assert cfg.indi_tilt_method == "geom"
    cfg.control_mode, cfg.indi = "indi", indi.IndiControllerCfg()
    cfg.indi_tilt_method = "typo"
    with pytest.raises(ValueError, match="indi_tilt_method"):
        method(filename, "configure_control")(cfg)
    with pytest.raises(ValueError, match="indi_tilt_method"):
        indi.attitude_error_vectors(torch.tensor([[1., 0., 0., 0.]]), "typo")
