# SPDX-License-Identifier: BSD-3-Clause
"""Numerical oracles for the simulator-independent flight model."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("flight_randomization", Path(__file__).parents[1] / "flight_randomization.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def config(**overrides):
    return SimpleNamespace(**(dict(
        motor_time_constant_range_s=(.035, .045), thrust_shared_gain_delta=.10,
        thrust_rotor_gain_delta=.03, km_gain_delta=.10, wind_enabled=True,
        gust_interval_range_s=(.8, 1.6), gust_duration_range_s=(.2, .5),
        wind_steady_max_m_s=2.0, wind_gust_max_m_s=2.0, external_moment_max_nm=.04,
        wind_drag_n_per_m_s_squared=.08) | overrides))


@pytest.mark.parametrize("mode", ["model_wrench_indi", "PI_rotors_PD_joint_pos_control"])
def test_step_response_uses_nominal_commands_and_actual_plant_coefficients(mode):
    cfg = config(motor_time_constant_range_s=(.04, .04))
    task = SimpleNamespace(flight_randomization=cfg, kf=4e-6, km=5e-8,
                           rotor_spin=(1, -1, 1, -1, 1, -1), control_mode=mode)
    rotor = model.RotorPlant(task, 6, .005, "cpu", reset_thrust=4, nominal_tau=.04)
    rotor.kf = torch.tensor([[1.13, .97, 1., 1., 1., 1.]]) * task.kf
    rotor.km = 1.10 * task.km
    command = torch.tensor([[9., 4., 4., 4., 4., 4.]]).expand(6, -1)
    for _ in range(8):
        previous = rotor.speed.clone()
        thrust, yaw, transient = rotor.step(command, rotor.spin.float()[None])
    if mode.startswith("PI"):
        expected = ((4 + (command - 4) * (1 - torch.exp(torch.tensor(-1.)))) / task.kf).sqrt()
    else:
        expected = 1000 + ((command / task.kf).sqrt() - 1000) * (1 - torch.exp(torch.tensor(-1.)))
    torch.testing.assert_close(rotor.speed, expected)
    torch.testing.assert_close(thrust, rotor.kf * expected.square())
    torch.testing.assert_close(yaw, (1.10 * task.km * expected.square() * rotor.spin).sum(-1))
    torch.testing.assert_close(transient[:, 0], ((expected - previous) * rotor.spin).sum(-1) / .005, atol=.1, rtol=0)
    torch.testing.assert_close(rotor.power(), (rotor.km * expected**3).sum(-1))


@pytest.mark.parametrize("name,value", [
    ("thrust_shared_gain_delta", .98),
    ("km_gain_delta", 1.0),
    ("wind_drag_n_per_m_s_squared", -1.0),
    ("gust_duration_range_s", (.2, .9)),
    ("motor_time_constant_range_s", (.045, .035)),
])
def test_invalid_flight_randomization_rejected(name, value):
    model.validate_flight_randomization(config())
    with pytest.raises(ValueError):
        model.validate_flight_randomization(config(**{name: value}))


def test_wind_drag_opposes_air_relative_motion_and_pulses_expire():
    wind = model.WindDisturbances(config(gust_duration_range_s=(.2, .2)), 32, .1, "cpu")
    wind.reset(torch.arange(32))
    wind.steady[:] = torch.tensor([2., 0., 0.])
    wind.until_next[:] = 0
    velocity = torch.tensor([[3., 0., 0.]]).expand(32, -1)
    for _ in range(2):
        wind.update()
        assert wind.weight.gt(0).all()
        force, moment = wind.wrench_world(velocity)
        assert moment.norm(dim=-1).max() <= .04 and moment.norm(dim=-1).max() > 0
        air = wind.steady + wind.weight * wind.gust - velocity
        assert (force * air).sum(dim=-1).ge(0).all() and not wind.gust[:, 2].any()
    wind.update()
    force, moment = wind.wrench_world(velocity)
    torch.testing.assert_close(force, torch.tensor([[-.08, 0., 0.]]).expand(32, -1))
    assert not moment.any()
