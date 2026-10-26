# SPDX-License-Identifier: BSD-3-Clause
"""Physical pre-clipping requests, episode reductions and exact soft-cost alignment."""
import importlib.util
import sys

import pytest
import torch

from test_control import FOLDER, config, raw_action_limit_costs


@pytest.fixture
def metrics_class(monkeypatch):
    # Load the pure tensor helper without importing the simulator-owned task package.
    limits_name = "isaaclab_tasks.direct.borinot.action_limits"
    spec = importlib.util.spec_from_file_location(limits_name, FOLDER.parent / "borinot/action_limits.py")
    limits = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(limits)
    monkeypatch.setitem(sys.modules, limits_name, limits)
    spec = importlib.util.spec_from_file_location("desired_actuation_test", FOLDER / "desired_actuation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DesiredActuationMetrics


@pytest.mark.parametrize("sac", [False, True])
@pytest.mark.parametrize("remove_cap", [False, True])
@pytest.mark.parametrize("action_clamp", [.25, 10.])
def test_requests_and_episode_costs_match_real_penalty_inputs(metrics_class, sac, remove_cap, action_clamp):
    cfg = config()
    cfg.sac_action_mapping, cfg.remove_thrust_clipping, cfg.action_clamp = sac, remove_cap, action_clamp
    cfg.thrust_to_weight, cfg.effort_lim_sim = 3., 2.7
    metrics = metrics_class(cfg, 5., 3, "cpu")
    actions = torch.tensor([[-3., 0., 1., 2., 4., -.5, -8.1, 5.4],
                            [0., 1., 2., 3., 4., 5., .5, -.25],
                            [1., 0., -2., 3., 0., 2., 0., 0.]])
    before = actions.clone()
    expected = torch.zeros(3, 3)
    for raw in (actions, -actions):
        metrics.update(raw)
        expected += raw_action_limit_costs(raw, cfg, 5.)
    packet = metrics.packet(torch.arange(3))
    for index, group in enumerate(("thrust", "torque")):
        torch.testing.assert_close(packet[f"Desired_Actuation/{group}_relative_excess_episode_sum"],
                                   expected[:, index].mean())
        torch.testing.assert_close(packet[f"Desired_Actuation/{group}_relative_excess_mean_per_step"],
                                   expected[:, index].mean() / 2)
    key = f"Desired_Torque/{cfg.joint_names[0]}"
    scale = cfg.effort_lim_sim if sac else 1.
    torch.testing.assert_close(packet[f"{key}/mean_nm"], torch.tensor(0.))
    torch.testing.assert_close(packet[f"{key}/rms_nm"], actions[:, 6].abs().mean() * scale)
    torch.testing.assert_close(actions, before)


def test_signed_physical_values_rms_extrema_and_reset_isolation(metrics_class):
    cfg = config()
    cfg.thrust_to_weight = 3.
    metrics = metrics_class(cfg, 5., 3, "cpu")
    raw = torch.zeros(3, 8)
    raw[0, 0], raw[0, 6] = -2., -5.4  # -5 N and -5.4 N*m, not applied/clipped values.
    metrics.update(raw)
    raw[0, 0], raw[0, 6] = 3., 2.7  # +20 N and +2.7 N*m.
    metrics.update(raw)
    packet = metrics.packet(torch.tensor([0]))
    for stat, expected in (("mean_n", 7.5), ("rms_n", (212.5)**.5),
                           ("min_n", -5.), ("max_n", 20.), ("outside_limits_fraction", 1.)):
        torch.testing.assert_close(packet[f"Desired_Thrust/rotor_1/{stat}"], torch.tensor(expected))
    torque = f"Desired_Torque/{cfg.joint_names[0]}"
    torch.testing.assert_close(packet[f"{torque}/mean_nm"], torch.tensor(-1.35))
    torch.testing.assert_close(packet[f"{torque}/outside_limits_fraction"], torch.tensor(.5))
    other = metrics.packet(torch.tensor([1, 2]))
    metrics.reset(torch.tensor([0]))
    raw.zero_()
    metrics.update(raw)
    fresh = metrics.packet(torch.tensor([0]))
    assert fresh["Desired_Thrust/rotor_1/min_n"] == 5.
    assert fresh[f"{torque}/rms_nm"] == 0.
    for key, value in other.items():
        torch.testing.assert_close(metrics.packet(torch.tensor([1, 2]))[key], value)


def test_logging_survives_zero_reward_weights_and_uses_completed_episode_lengths(metrics_class):
    cfg = config()
    cfg.rwd_scale_thrust_limits = cfg.rwd_scale_torque_limits = 0.
    metrics = metrics_class(cfg, 5., 2, "cpu")
    raw = torch.zeros(2, 8)
    raw[:, 6] = 2 * cfg.effort_lim_sim
    metrics.update(raw)
    metrics.reset(torch.tensor([1]))
    metrics.update(raw)
    packet = metrics.packet(torch.arange(2))
    assert packet["Desired_Actuation/torque_relative_excess_episode_sum"] == 1.5
    assert packet["Desired_Actuation/torque_relative_excess_mean_per_step"] == 1.


def test_actual_episode_log_packet_keeps_reward_and_actuation_cohorts_equal(metrics_class):
    from test_curriculum_checkpoint import make_env, update_curriculum
    cfg = config()
    env = make_env(n=3)
    env._desired_actuation = metrics_class(cfg, 5., 3, "cpu")
    raw = torch.zeros(3, 8)
    raw[:, 6] = torch.tensor([5.4, 0., 8.1])
    env._desired_actuation.update(raw)
    cost = raw_action_limit_costs(raw, cfg, 5.)[:, 1]
    env._episode_terms["torque_limits"] = cfg.rwd_scale_torque_limits * cost
    env._episode_steps[1] = 0  # initial/non-episode reset must not enter either metric.
    update_curriculum(env, torch.arange(3))
    packet = env.extras["log"]
    torch.testing.assert_close(packet["Episode_Reward/torque_limits"],
        cfg.rwd_scale_torque_limits * packet["Desired_Actuation/torque_relative_excess_episode_sum"])
    assert "Episode/return" in packet and "Curriculum/selected_stage" in packet
    torch.testing.assert_close(packet[f"Desired_Torque/{cfg.joint_names[0]}/mean_nm"], torch.tensor(6.75))
