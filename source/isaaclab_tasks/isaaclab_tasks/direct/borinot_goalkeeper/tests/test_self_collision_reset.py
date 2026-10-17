# SPDX-License-Identifier: BSD-3-Clause
"""Self-contact can retain its crash penalty without ending the robot episode."""
from types import SimpleNamespace

import pytest
import torch

from test_control import config, method
from test_continuous import env as continuous_env


def test_self_collision_reset_is_default_on():
    assert config().self_collision_triggers_reset is True


@pytest.mark.parametrize("reset_self", [False, True])
@pytest.mark.parametrize("reset_contact", [False, True])
def test_self_reset_is_independent_of_scene_tilt_area_and_reward(continuous_env, reset_self, reset_contact):
    env = continuous_env
    env.cfg.self_collision_triggers_reset = reset_self
    env.cfg.reset_robot_on_robot_crash = reset_contact
    env._flags["external_crash"] = torch.zeros(9, dtype=torch.bool)
    # Self only; scene only; both; self + tilt; self + area; legal ball hit.
    env._flags["crash"][:5] = True
    env._flags["external_crash"][[1, 2]] = True
    env._flags["tilt"][3] = True
    env._flags["area"][4] = True
    env._flags["base"][5] = True
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [reset_self and reset_contact, reset_contact, reset_contact,
                                  True, True, False, False, False, False]
    assert env._outcome[:6].tolist() == [8, 8, 8, 8, 8, 1]
    env._get_rewards()
    assert env.extras["goalkeeper_reward_terms"]["outcome"][:5].tolist() == [env.cfg.rwd_scale_robot_crash] * 5
    assert not truncated.any()


def test_filtered_self_forces_leave_simultaneous_scene_forces_and_latch():
    # Self only; simultaneous scene; below threshold; cancelling self/scene forces.
    forces = torch.tensor([[[0., 0., 4.], [2., 0., 0.]],
                           [[0., 0., 7.], [2., 0., 0.]],
                           [[0., 0., 0.05], [0., 0., 0.]],
                           [[0., 0., 0.], [0., 0., 0.]]])
    self_forces = forces.clone()
    self_forces[1, 0, 2] -= 3.
    self_forces[3, 0, 2] = -2.
    flags = {name: torch.zeros(4, dtype=torch.bool) for name in ("crash", "external_crash")}
    env = SimpleNamespace(cfg=SimpleNamespace(self_collision_triggers_reset=False,
                          robot_crash_force_threshold_n=0.1), _flags=flags,
                          _crash_contact_columns=[1, 0],
                          _self_contact_sensors=[SimpleNamespace(data=SimpleNamespace(
                              force_matrix_w=self_forces[:, column, None, None, :])) for column in [1, 0]])
    sample = method("borinot_goalkeeper_env.py", "_sample_robot_crashes")
    sample(env, forces)
    assert flags["crash"].tolist() == [True, True, False, False]
    assert flags["external_crash"].tolist() == [False, True, False, True]
    # Later clean substeps must not erase the earlier impact.
    for sensor in env._self_contact_sensors:
        sensor.data.force_matrix_w.zero_()
    sample(env, torch.zeros_like(forces))
    assert flags["external_crash"].tolist() == [False, True, False, True]


def test_default_crash_detection_needs_no_extra_sensor():
    env = SimpleNamespace(cfg=SimpleNamespace(self_collision_triggers_reset=True,
                          robot_crash_force_threshold_n=0.1),
                          _flags={"crash": torch.zeros(2, dtype=torch.bool)}, _crash_contact_columns=[1])
    forces = torch.tensor([[[3., 0., 0.], [0., 0., 0.05]], [[0., 0., 0.], [0., 0., 2.]]])
    method("borinot_goalkeeper_env.py", "_sample_robot_crashes")(env, forces)
    assert env._flags["crash"].tolist() == [False, True]
