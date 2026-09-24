# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical checks for the categorical-critic spread and estimation-error tooling.

Run with the command in SAC_DISTRIBUTIONAL_CRITIC.md from env_isaaclab.
"""

import numpy as np
import pytest
import torch
from q_spread_plots import Run, discounted_returns
from rsl_rl_sac.models import SACActorModel
from tensordict import TensorDict

GAMMA = 0.5


def returns(rewards, dones, time_outs, bonus=None, gamma=GAMMA, boot=None):
    """Single-env helper: pass plain lists, get 1-D arrays back."""
    shape = (len(rewards), 1)

    def column(values):
        return None if values is None else np.array(values, dtype=np.float64).reshape(shape)

    out = discounted_returns(
        column(rewards),
        column(dones),
        column(time_outs),
        gamma,
        column(bonus),
        None if boot is None else {"plain": column(boot), "soft": column(boot)},
    )
    return {key: value.reshape(-1) for key, value in out.items()}


def test_terminated_episode_return_is_complete():
    out = returns([1.0, 1.0, 1.0], [0, 0, 1], [0, 0, 0])
    # 1 + 0.5 * (1 + 0.5 * 1) walked backwards from the terminal step.
    np.testing.assert_allclose(out["plain"], [1.75, 1.5, 1.0])
    # A real termination has no missing tail, whatever the remaining step count.
    np.testing.assert_allclose(out["observed"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["remaining"], [2.0, 1.0, 0.0])


def test_timeout_leaves_a_discounted_tail_unobserved():
    out = returns([1.0, 1.0, 1.0], [0, 0, 1], [0, 0, 1])
    np.testing.assert_allclose(out["plain"], [1.75, 1.5, 1.0])
    # The cut tail is worth gamma ** (remaining + 1); the last step sees least of its return.
    np.testing.assert_allclose(out["bootstrap_weight"], [0.125, 0.25, 0.5])
    np.testing.assert_allclose(out["observed"], [0.875, 0.75, 0.5])


def test_log_ending_mid_episode_counts_as_truncated():
    out = returns([1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(out["plain"], [1.75, 1.5, 1.0])
    np.testing.assert_allclose(out["observed"], [0.875, 0.75, 0.5])


def test_returns_do_not_leak_across_an_episode_boundary():
    out = returns([1.0, 1.0, 5.0, 5.0], [0, 1, 0, 1], [0, 0, 0, 0])
    # Step 1 ends its episode, so the 5.0 rewards must not reach step 0.
    np.testing.assert_allclose(out["plain"], [1.5, 1.0, 7.5, 5.0])
    np.testing.assert_allclose(out["remaining"], [1.0, 0.0, 1.0, 0.0])


def test_entropy_bonus_starts_one_step_after_the_evaluated_action():
    bonus = [100.0, 2.0, 4.0]
    out = returns([0.0, 0.0, 0.0], [0, 0, 1], [0, 0, 0], bonus=bonus)
    # Q(s_t, a_t) already conditions on a_t, so bonus[0] must never appear.
    expected_1 = GAMMA * bonus[2]
    expected_0 = GAMMA * (expected_1 + bonus[1])
    np.testing.assert_allclose(out["soft"], [expected_0, expected_1, 0.0])
    np.testing.assert_allclose(out["plain"], [0.0, 0.0, 0.0])


def test_envs_are_independent():
    rewards = np.array([[1.0, 10.0], [1.0, 10.0]])
    dones = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = discounted_returns(rewards, dones, np.zeros_like(dones), GAMMA, None)
    # Env 0 ends at step 0; env 1 runs through both steps.
    np.testing.assert_allclose(out["plain"][:, 0], [1.0, 1.0])
    np.testing.assert_allclose(out["plain"][:, 1], [15.0, 10.0])


def test_executed_action_logp_matches_the_sampled_log_probability():
    torch.manual_seed(3)
    obs = TensorDict({"policy": torch.randn(16, 4)}, batch_size=[16])
    actor = SACActorModel(obs, {"actor": ["policy"]}, "actor", 3, hidden_dims=[16], init_noise_std=0.4)
    actor.action_range.fill_(2.5)
    actor.action_bias.fill_(0.25)
    actor.log_action_range.fill_(float(np.log(2.5) * 3))

    action, log_prob = actor.sample_action_logp(obs)
    # Recovering the latent through atanh must reproduce the log-probability of that action.
    torch.testing.assert_close(actor.executed_action_logp(action), log_prob, atol=2e-4, rtol=2e-4)


def test_executed_action_logp_scores_the_deterministic_action():
    torch.manual_seed(5)
    obs = TensorDict({"policy": torch.randn(16, 4)}, batch_size=[16])
    actor = SACActorModel(obs, {"actor": ["policy"]}, "actor", 3, hidden_dims=[16], init_noise_std=0.4)

    mean_action = actor(obs)
    mean_log_prob = actor.executed_action_logp(mean_action)
    sampled_log_prob = actor.executed_action_logp(actor(obs, stochastic_output=True))
    assert mean_log_prob.shape == (16, 1)
    assert torch.isfinite(mean_log_prob).all()
    # The distribution mode is the most likely action, so it must not score below a sample.
    assert mean_log_prob.mean() > sampled_log_prob.mean()


def test_timeout_bootstraps_with_the_next_state_value():
    boot = [0.0, 0.0, 8.0]
    out = returns([1.0, 1.0, 1.0], [0, 0, 1], [0, 0, 1], boot=boot)
    # The cut episode closes on V(s') = 8 instead of 0, so the tail is not thrown away.
    np.testing.assert_allclose(out["plain"][2], 1.0 + GAMMA * 8.0)
    np.testing.assert_allclose(out["plain"][1], 1.0 + GAMMA * (1.0 + GAMMA * 8.0))
    np.testing.assert_allclose(out["plain"][0], 1.0 + GAMMA * out["plain"][1])


def test_termination_ignores_the_bootstrap_value():
    boot = [0.0, 0.0, 99.0]
    terminated = returns([1.0, 1.0, 1.0], [0, 0, 1], [0, 0, 0], boot=boot)
    # A real terminal state is worth zero, whatever the critic would have predicted.
    np.testing.assert_allclose(terminated["plain"], [1.75, 1.5, 1.0])
    np.testing.assert_allclose(terminated["observed"], [1.0, 1.0, 1.0])


def test_bootstrapping_removes_the_decay_toward_a_timeout():
    """A critic that is exactly right should show no error, even next to the cut."""
    steps, gamma, reward = 40, 0.9, 1.0
    value = reward / (1 - gamma)  # The true value of an endless stream of 1.0 rewards.
    # Q(s,a) = r + gamma * V = V for every step of this stationary problem.
    out = returns(
        [reward] * steps, [0] * (steps - 1) + [1], [0] * (steps - 1) + [1],
        gamma=gamma, boot=[value] * steps,
    )
    np.testing.assert_allclose(out["plain"], np.full(steps, value))
    # Without the bootstrap the same log sags toward the timeout.
    naive = returns([reward] * steps, [0] * (steps - 1) + [1], [0] * (steps - 1) + [1], gamma=gamma)
    assert naive["plain"][-1] == pytest.approx(reward)
    assert naive["plain"][0] < value


def test_log_ending_mid_episode_also_bootstraps():
    out = returns([1.0, 1.0, 1.0], [0, 0, 0], [0, 0, 0], boot=[0.0, 0.0, 4.0])
    np.testing.assert_allclose(out["plain"][2], 1.0 + GAMMA * 4.0)


def test_soft_bootstrap_is_used_for_the_soft_return_only():
    shape = (2, 1)
    out = discounted_returns(
        np.zeros(shape), np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]), GAMMA,
        np.full(shape, 0.0),
        {"plain": np.array([[0.0], [10.0]]), "soft": np.array([[0.0], [6.0]])},
    )
    np.testing.assert_allclose(out["plain"][1], [GAMMA * 10.0])
    np.testing.assert_allclose(out["soft"][1], [GAMMA * 6.0])


@pytest.mark.parametrize("stored", [None, "min", "mean"])
def test_estimate_combines_the_twin_critics_as_training_did(tmp_path, stored):
    # (steps, envs, twin critics): Q1 differs from Q2 in both directions.
    q = np.array([[[1.0, 3.0], [-2.0, 0.5]], [[4.0, 4.5], [0.0, -6.0]]], dtype=np.float32)
    extra = {} if stored is None else {"q_reduction_method": stored}
    np.savez_compressed(
        tmp_path / "log.npz",
        q=q,
        probs=np.full((*q.shape, 5), 0.2, dtype=np.float32),
        done=np.zeros(q.shape[:2], dtype=np.float32),
        value_support=np.linspace(-2.0, 2.0, 5),
        gamma=GAMMA,
        reward=np.ones(q.shape[:2], dtype=np.float32),
        time_out=np.zeros(q.shape[:2], dtype=np.float32),
        **extra,
    )
    run = Run(tmp_path / "log.npz", None, drop_first=0)
    # Logs written before the key existed come from "min" training.
    assert run.q_reduction == (stored or "min")
    expected = (q[..., 0] + q[..., 1]) / 2 if stored == "mean" else np.minimum(q[..., 0], q[..., 1])
    np.testing.assert_array_equal(run.q_combined, expected)
    np.testing.assert_allclose(run.error(), expected - run.target_return()[0])


@pytest.mark.parametrize("gamma", [0.9, 0.99])
def test_observed_fraction_matches_the_geometric_tail(gamma):
    steps = 50
    out = returns([1.0] * steps, [0] * steps, [0] * steps, gamma=gamma)
    # Early steps see almost all of their return; the final step sees almost none of it.
    np.testing.assert_allclose(out["observed"][0], 1 - gamma**steps)
    np.testing.assert_allclose(out["observed"][-1], 1 - gamma)
    assert np.all(np.diff(out["observed"]) < 0)
