# SPDX-License-Identifier: BSD-3-Clause
"""Independent SO(3) oracles for the requested full-Log z projection."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from test_indi import indi


def oracle(current, desired, rates, reference, cfg):
    error = current.inv() * desired
    tau_z = np.zeros_like(rates)
    tau_z[:, 2] = error.as_rotvec()[:, 2]
    residual = error * Rotation.from_rotvec(-tau_z)
    return (cfg.tilt_gain * residual.as_rotvec() + cfg.yaw_gain * tau_z
            + np.asarray(cfg.rate_gain) * (reference - rates))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_feedback_matches_scipy_full_log_projection(dtype):
    rng = np.random.default_rng(20261023)
    current = Rotation.random(2000, random_state=rng)
    desired = Rotation.random(2000, random_state=rng)
    rates, reference = rng.normal(size=(2, 2000, 3))
    cfg = indi.IndiControllerCfg()
    q = torch.tensor(current.as_quat(scalar_first=True), dtype=dtype)
    rd = torch.tensor(desired.as_matrix(), dtype=dtype)
    actual = indi.attitude_acceleration(q, rd, torch.tensor(rates, dtype=dtype),
                                        torch.tensor(reference, dtype=dtype), cfg, "method_1")
    np.testing.assert_allclose(actual, oracle(current, desired, rates, reference, cfg),
                               atol=2e-4 if dtype == torch.float32 else 2e-11, rtol=2e-6)
    # A change of world frame must not change a current-body-axis error.
    world = Rotation.from_rotvec([.7, -.4, 1.3])
    changed = indi.attitude_acceleration(
        torch.tensor((world * current).as_quat(scalar_first=True), dtype=dtype),
        torch.tensor((world * desired).as_matrix(), dtype=dtype),
        torch.tensor(rates, dtype=dtype), torch.tensor(reference, dtype=dtype), cfg, "method_1")
    torch.testing.assert_close(actual, changed, atol=3e-4 if dtype == torch.float32 else 2e-11,
                               rtol=2e-6)
    torch.testing.assert_close(actual, indi.attitude_acceleration(
        -q, rd, torch.tensor(rates, dtype=dtype), torch.tensor(reference, dtype=dtype), cfg, "method_1"))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("angle", [0., 1e-9, -.01, math.pi * 2/3, math.pi-1e-6, math.pi+1e-6])
def test_pure_axis_feedback_uses_full_shortest_angle(axis, angle):
    vector = np.zeros((1, 3))
    vector[0, axis] = angle
    cfg = indi.IndiControllerCfg()
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    actual = indi.attitude_acceleration(torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
        torch.from_numpy(Rotation.from_rotvec(vector).as_matrix()), zeros, zeros, cfg)
    expected = Rotation.from_rotvec(vector).as_rotvec()
    expected *= cfg.yaw_gain if axis == 2 else cfg.tilt_gain
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_mixed_error_is_not_the_old_geometric_tilt_yaw_split():
    current = Rotation.identity(1)
    desired = Rotation.from_euler("x", [90], degrees=True) * Rotation.from_euler("z", [90], degrees=True)
    cfg = indi.IndiControllerCfg()
    zeros = torch.zeros(1, 3, dtype=torch.float64)
    actual = indi.attitude_acceleration(torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
        torch.from_numpy(desired.as_matrix()), zeros, zeros, cfg, "method_1")
    np.testing.assert_allclose(actual, oracle(current, desired, zeros.numpy(), zeros.numpy(), cfg), atol=1e-12)
    geometric = np.array([[cfg.tilt_gain*math.pi/2, 0., cfg.yaw_gain*math.pi/2]])
    assert np.linalg.norm(actual.numpy() - geometric) > 10


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_log_exp_and_matrix_conversion_at_identity_and_pi(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(81)
    axes = rng.normal(size=(500, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    vectors = np.concatenate((np.zeros((1, 3)), np.eye(3)*1e-10,
        axes*np.linspace(0, math.pi-1e-4, len(axes))[:, None],
        np.eye(3)*math.pi, -np.eye(3)*math.pi, axes*(math.pi-1e-5), axes*(math.pi+1e-5)))
    rotation = Rotation.from_rotvec(vectors)
    expected = rotation.as_rotvec()
    v = torch.tensor(vectors, dtype=dtype, device=device)
    q = indi.quaternion_exp(v)
    tolerance = 1e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(len(q), dtype=dtype, device=device), atol=tolerance, rtol=0)
    np.testing.assert_allclose(indi.rotation_from_quat(q).cpu(), rotation.as_matrix(), atol=tolerance, rtol=0)
    log = indi.quaternion_log(q)
    # The pi-axis sign is not unique; compare its represented rotation there.
    reconstructed = indi.rotation_from_quat(indi.quaternion_exp(log))
    np.testing.assert_allclose(reconstructed.cpu(), rotation.as_matrix(), atol=tolerance, rtol=0)
    assert log.norm(dim=-1).max() <= math.pi + tolerance
    torch.testing.assert_close(log, indi.quaternion_log(-q), atol=tolerance, rtol=0)
    # Compare principal vectors off the ambiguous pi boundary, not only round trips.
    off_pi = np.abs(np.linalg.norm(vectors, axis=-1)-math.pi) > 1e-6
    np.testing.assert_allclose(log.cpu().numpy()[off_pi], expected[off_pi], atol=tolerance, rtol=0)
    matrices = torch.tensor(rotation.as_matrix(), dtype=dtype, device=device)
    from_matrix = indi.quaternion_from_rotation(matrices)
    torch.testing.assert_close(indi.rotation_from_quat(from_matrix), matrices, atol=tolerance, rtol=0)
    # Exact pi inputs with an exactly zero scalar must also be sign-invariant.
    exact = torch.tensor([[0., -1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., -1.]], dtype=dtype, device=device)
    torch.testing.assert_close(indi.quaternion_log(exact), math.pi*torch.eye(3, dtype=dtype, device=device), atol=tolerance, rtol=0)
    torch.testing.assert_close(indi.quaternion_log(exact), indi.quaternion_log(-exact), atol=tolerance, rtol=0)
