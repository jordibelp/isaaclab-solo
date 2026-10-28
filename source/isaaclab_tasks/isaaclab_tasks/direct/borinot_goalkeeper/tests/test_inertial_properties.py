# SPDX-License-Identifier: BSD-3-Clause
"""Independent first-moment/energy checks for shared Borinot inertial overrides."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("borinot_inertial_properties", ROOT / "borinot/inertial_properties.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


@pytest.mark.parametrize("point_mass", [0., .005, .02])
def test_motor_merge_preserves_first_moment_and_kinetic_energy(point_mass):
    base_mass = 2.647
    base_com = torch.tensor([0., .00244, .0202], dtype=torch.float64)
    base_inertia = torch.diag(torch.tensor([.02724, .02685, .0398], dtype=torch.float64))
    points = torch.tensor([[0., -.185, .05], [0., .185, .05], [.1602146997, .0925, .05],
                           [-.1602146997, -.0925, .05], [.1602146997, -.0925, .05],
                           [-.1602146997, .0925, .05]], dtype=torch.float64)
    mass, com, inertia = model.merge_point_masses(base_mass, base_com, base_inertia, point_mass, points)
    assert mass == pytest.approx(base_mass + 6 * point_mass)
    torch.testing.assert_close(mass * com, base_mass * base_com + point_mass * points.sum(0), rtol=0, atol=1e-15)
    generator = torch.Generator().manual_seed(7)
    omega, velocity = torch.randn(2, 100, 3, dtype=torch.float64, generator=generator)
    base_velocity = velocity + torch.linalg.cross(omega, base_com.expand_as(omega))
    expected = .5 * base_mass * base_velocity.square().sum(-1)
    expected += .5 * torch.einsum('ni,ij,nj->n', omega, base_inertia, omega)
    for point in points:
        point_velocity = velocity + torch.linalg.cross(omega, point.expand_as(omega))
        expected += .5 * point_mass * point_velocity.square().sum(-1)
    combined_velocity = velocity + torch.linalg.cross(omega, com.expand_as(omega))
    actual = .5 * mass * combined_velocity.square().sum(-1) + .5 * torch.einsum('ni,ij,nj->n', omega, inertia, omega)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=2e-14)
    assert torch.linalg.eigvalsh(inertia).min() > 0
    if point_mass == 0:
        torch.testing.assert_close(com, base_com, rtol=0, atol=0)
        torch.testing.assert_close(inertia, base_inertia, rtol=0, atol=0)


def test_point_mass_changes_two_transverse_axes_not_the_displacement_axis():
    # Base CoM at origin and one point on Y: extra Ixx = Izz; no extra Iyy.
    mass, com, inertia = model.merge_point_masses(1., torch.zeros(3), torch.eye(3), .1, [[0., .2, 0.]])
    assert inertia[0, 0] > 1 and inertia[2, 2] > 1
    assert inertia[1, 1] == 1
    assert inertia[0, 0] == inertia[2, 2]


def test_ideal_point_at_existing_com_adds_mass_but_no_own_com_inertia():
    center = torch.tensor([0., -.08, -.021], dtype=torch.float64)
    original = torch.diag(torch.tensor([3.2e-5, 1e-6, 3.2e-5], dtype=torch.float64))
    mass, com, inertia = model.merge_point_masses(.015, center, original, .1, center[None])
    assert mass == pytest.approx(.115)
    torch.testing.assert_close(com, center, rtol=0, atol=1e-16)
    torch.testing.assert_close(inertia, original, rtol=0, atol=1e-16)


@pytest.mark.parametrize('mass', [-.01, float('nan'), float('inf')])
def test_invalid_motor_mass_is_rejected(mass):
    with pytest.raises(ValueError):
        model.merge_point_masses(1., [0, 0, 0], torch.eye(3), mass, [[0., 0., 0.]])
