# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BatchNorm critic as in XQC (arXiv:2509.25174): one batch for (s, a) and (s', a'), running statistics for the actor."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl_sac.algorithms import SAC

NUM_ENVS = 16


def _obs():
    return TensorDict({"policy": torch.randn(NUM_ENVS, 6)}, batch_size=[NUM_ENVS])


def build(monkeypatch, **critic):
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    cfg = dict(
        num_steps_per_env=4,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[16, 16], activation="elu", obs_normalization=True),
        critic=dict(class_name="SACCriticModel", hidden_dims=[16, 16], activation="elu", obs_normalization=True,
                    layer_norm=True, batch_norm=True, distributional_loss="two_hot", distributional_num_bins=51,
                    distributional_symlog_limit=5.0, **critic),
        algorithm=dict(class_name="SAC", replay_buffer_size=NUM_ENVS * 32, num_mini_batches=3, mini_batch_size=8,
                       gamma=0.97, n_steps=3, policy_frequency=1, actor_learning_rate=1e-3,
                       critic_learning_rate=1e-3, alpha_learning_rate=1e-3),
    )
    obs = _obs()
    alg = SAC.construct_algorithm(obs, SimpleNamespace(num_actions=2, num_envs=NUM_ENVS), cfg, "cpu")
    for _ in range(12):
        alg.act(obs)
        obs = _obs()
        alg.process_env_step(obs, torch.randn(NUM_ENVS), (torch.rand(NUM_ENVS) < 0.2).float(), {})
    alg.train_mode()
    return alg


def _running_means(critic):
    return [module.running_mean.clone() for module in critic.modules() if isinstance(module, nn.BatchNorm1d)]


def test_batch_norm_replaces_layer_norm_in_every_q_network(monkeypatch):
    critic = build(monkeypatch).critic
    for network in (critic.critic1, critic.critic2, critic.critic1_target, critic.critic2_target):
        kinds = [type(module) for module in network]
        assert nn.LayerNorm not in kinds
        # A bias in front of BatchNorm is redundant; XQC omits it too. The output layer keeps its bias.
        assert [module.bias is None for module in network if isinstance(module, nn.Linear)] == [True, True, False]
        # Linear -> BatchNorm -> activation for every hidden layer, as in XQC's pre-activation block.
        assert kinds[:3] == [nn.Linear, nn.BatchNorm1d, nn.ELU] and kinds[3:6] == kinds[:3]
        norms = [module for module in network if isinstance(module, nn.BatchNorm1d)]
        assert [(norm.momentum, norm.eps) for norm in norms] == [(0.01, 1e-3)] * 2


def test_paired_outputs_normalize_both_halves_with_one_batch(monkeypatch):
    critic = build(monkeypatch).critic
    obs, next_obs = _obs(), _obs()
    actions, next_actions = torch.randn(NUM_ENVS, 2), torch.randn(NUM_ENVS, 2) + 3.0
    with torch.no_grad():
        nn.init.normal_(critic.critic1[-1].weight)  # the categorical head starts at zero for every input
        ((current, following),) = critic.paired_outputs((critic.critic1,), obs, actions, next_obs, next_actions)
        q_input = torch.cat((critic.get_latent(obs), actions), dim=-1)
        next_q_input = torch.cat((critic.get_latent(next_obs), next_actions), dim=-1)
        torch.testing.assert_close(torch.cat((current, following)), critic.critic1(torch.cat((q_input, next_q_input))))
        # A separate pass would normalize (s', a') with its own statistics.
        assert not torch.allclose(following, critic.critic1(next_q_input))


def test_update_shares_next_actions_and_scores_the_actor_with_running_statistics(monkeypatch):
    alg = build(monkeypatch)
    critic = alg.critic
    calls = []
    paired_outputs = critic.paired_outputs

    def record(networks, obs, actions, next_obs, next_actions):
        calls.append((networks, next_actions, torch.is_grad_enabled()))
        return paired_outputs(networks, obs, actions, next_obs, next_actions)

    actor_objective = alg._actor_objective_fn
    actor_modes = []

    def check_actor_objective(obs):
        actor_modes.append(critic.training)
        before = _running_means(critic)
        result = actor_objective(obs)
        assert all(torch.equal(old, new) for old, new in zip(before, _running_means(critic)))
        return result

    monkeypatch.setattr(critic, "paired_outputs", record)
    monkeypatch.setattr(alg, "_actor_objective_fn", check_actor_objective)
    before = _running_means(critic)
    alg.update()

    # Per mini-batch: the target critics without gradients, then the online critics on the same next actions.
    assert len(calls) == 2 * alg.num_mini_batches
    for (target, target_actions, target_grad), (online, online_actions, online_grad) in zip(calls[::2], calls[1::2]):
        assert target == (critic.critic1_target, critic.critic2_target) and not target_grad
        assert online == (critic.critic1, critic.critic2) and online_grad
        assert online_actions is target_actions
    assert actor_modes == [False] * alg.num_mini_batches
    assert critic.training
    assert all(not torch.equal(old, new) for old, new in zip(before, _running_means(critic)))
