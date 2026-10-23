# SPDX-License-Identifier: BSD-3-Clause
"""Independent mass, first-moment and kinetic-energy oracles for the calf payload."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("payload_task_math", FOLDER / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


@pytest.mark.parametrize("payload", [0.0, 0.001, 0.1, 0.15, 0.5])
def test_point_payload_preserves_total_mass_first_moment_and_kinetic_energy(payload):
    mass, com, inertia = task_math.calf_payload_properties(.015, payload)
    shaft_com = torch.tensor([0., -.08, -.021], dtype=torch.float64)
    tip = torch.tensor([0., -.136, -.00495], dtype=torch.float64)
    assert mass == pytest.approx(.015 + payload)
    torch.testing.assert_close(mass * com, .015 * shaft_com + payload * tip, atol=1e-16, rtol=0)
    assert torch.linalg.eigvalsh(inertia).min() > 0
    torch.testing.assert_close(inertia, inertia.T)
    torch.manual_seed(123)
    omega, origin_velocity = torch.randn(100, 3, dtype=torch.float64), torch.randn(100, 3, dtype=torch.float64)
    shaft_inertia = torch.diag(torch.tensor([.015*.16**2/12, 1e-6, .015*.16**2/12], dtype=torch.float64))
    # Compute each component's kinetic energy from velocities about the link
    # origin. This oracle does not repeat the composite parallel-axis formula.
    shaft_velocity = origin_velocity + torch.linalg.cross(omega, shaft_com.expand_as(omega))
    tip_velocity = origin_velocity + torch.linalg.cross(omega, tip.expand_as(omega))
    expected = (.5*.015*shaft_velocity.square().sum(-1) + .5*payload*tip_velocity.square().sum(-1)
                + .5*torch.einsum("ni,ij,nj->n", omega, shaft_inertia, omega))
    combined_velocity = origin_velocity + torch.linalg.cross(omega, com.expand_as(omega))
    actual = .5*mass*combined_velocity.square().sum(-1) + .5*torch.einsum("ni,ij,nj->n", omega, inertia, omega)
    torch.testing.assert_close(actual, expected, atol=1e-15, rtol=1e-14)


def test_zero_payload_is_the_bare_shaft_not_the_original_usd_ballast():
    mass, com, inertia = task_math.calf_payload_properties(.015, 0.)
    assert mass == .015
    assert com.tolist() == pytest.approx([0., -.08, -.021])
    assert inertia.diag().tolist() == pytest.approx([.000032, .000001, .000032])
    assert not (inertia - torch.diag(inertia.diag())).any()


@pytest.mark.parametrize("shaft,payload", [(0., 0.), (-.1, 0.), (.015, -.1), (.015, math.nan),
                                           (.015, math.inf), (math.inf, 0.)])
def test_invalid_mass_is_rejected(shaft, payload):
    with pytest.raises(ValueError):
        task_math.calf_payload_properties(shaft, payload)
