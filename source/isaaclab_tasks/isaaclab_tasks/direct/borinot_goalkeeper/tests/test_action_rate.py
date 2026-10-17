# SPDX-License-Identifier: BSD-3-Clause
"""Check post-clipping actuator differences against independently computed costs."""
import pytest
import torch

from test_control import MODES, action_env, method
from test_rewards import get_rewards, make_env


def reward_env():
    env = make_env()
    n = len(env.actions)
    env.cfg.effort_lim_sim = 2.7
    env.cfg.rwd_scale_action_rate = -0.001
    env.joint_ids = [0, 1]
    env.robot.data.applied_torque = torch.zeros(n, 2)
    env._actuator_commands = torch.zeros(n, 8)
    env._previous_actuator_commands = torch.zeros(n, 8)
    return env


@pytest.mark.parametrize("group", ["rotors", "torques", "both"])
def test_equal_fractional_group_changes_have_equal_weight(group):
    env = reward_env()
    if group in ("rotors", "both"):
        env._actuator_commands[:, :6] = 0.1
    if group in ("torques", "both"):
        env.robot.data.applied_torque[:] = 0.1 * (2 * env.cfg.effort_lim_sim)
    expected_cost = 0.02 if group == "both" else 0.01
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, -0.001 * expected_cost * env.step_dt), atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("index,expected_cost", [(0, 1/6), (5, 1/6), (6, 1/2), (7, 1/2)])
def test_one_full_span_change_is_averaged_within_its_group(index, expected_cost):
    env = reward_env()
    if index < 6:
        env._actuator_commands[:, index] = 1.0
    else:
        env._previous_actuator_commands[:, index] = -0.5
        env.robot.data.applied_torque[:, index - 6] = env.cfg.effort_lim_sim
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, -0.001 * expected_cost * env.step_dt), atol=1e-10, rtol=1e-6)


def test_saturated_raw_action_excursions_do_not_change_cost():
    env = reward_env()
    env.actions.fill_(8.0)
    env._actuator_commands[:, :6] = 1.0
    env._previous_actuator_commands[:, :6] = 1.0
    env.robot.data.applied_torque[:] = env.cfg.effort_lim_sim
    env._previous_actuator_commands[:, 6:] = 0.5
    assert not get_rewards(env).any()


@pytest.mark.parametrize("policy_hz", [60, 100, 200])
def test_policy_dt_applies_once_and_repeated_reward_reads_are_stable(policy_hz):
    env = reward_env()
    env.step_dt = 1 / policy_hz
    env._actuator_commands[:, :6] = 0.3
    env.robot.data.applied_torque[:] = 0.4 * (2 * env.cfg.effort_lim_sim)
    expected = -0.001 * (0.3**2 + 0.4**2) / policy_hz
    for _ in range(2):
        reward = get_rewards(env)
        torch.testing.assert_close(reward, torch.full_like(reward, expected), atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize("remove_cap", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_tracks_thrust_after_mapping_and_clipping_before_motor_lag(mode, remove_cap):
    env, _ = action_env(mode)
    env.cfg.remove_thrust_clipping = remove_cap
    env.cfg.thrust_to_weight = 3.5
    env._actuator_commands = torch.zeros(3, 8, dtype=torch.float64)
    env.actions.zero_()
    if mode == MODES[2]:
        env.actions[:, 0] = 6.0
        # Equal allocation with zero body-rate error.
        expected = (1 + env.cfg.collective_action_scale * 6.0) / 3.5
        expected = expected if remove_cap else min(expected, 1.0)
        expected = torch.full((3, 6), expected, dtype=torch.float64)
    else:
        env.actions[:, :6] = torch.tensor([-3., -1., 0., 2.5, 3., 8.])
        expected = torch.tensor([0., 0., 1., 3.5, 4., 9.], dtype=torch.float64)
        if not remove_cap:
            expected.clamp_max_(3.5)
        expected = (expected / 3.5).expand(3, -1)
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(env._actuator_commands[:, :6], expected)
    assert not torch.allclose(env._rotor_thrust / (env._hover_per_rotor * 3.5), expected)


@pytest.mark.parametrize("mode", MODES[:2])
def test_sac_thrust_mapping_uses_the_same_physical_normalization(mode):
    env, _ = action_env(mode)
    env.cfg.sac_action_mapping = True
    env.cfg.thrust_to_weight = 3.5
    env._sac_joint_offset = torch.zeros(2)
    env._actuator_commands = torch.zeros(3, 8, dtype=torch.float64)
    env.actions[:, :6] = torch.tensor([[-1.], [0.], [1.]])
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    expected = torch.tensor([[0.], [1/3.5], [1.]], dtype=torch.float64).expand(3, 6)
    torch.testing.assert_close(env._actuator_commands[:, :6], expected)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_thrust_normalization_range_is_rejected(value):
    env, _ = action_env(MODES[0])
    env.cfg.thrust_to_weight = value
    with pytest.raises(ValueError, match="thrust_to_weight"):
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(env.cfg)
