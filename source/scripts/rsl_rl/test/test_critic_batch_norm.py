# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchNorm in the PPO value network (agent.critic.batch_norm), on real RSL-RL PPO updates without Isaac Sim.

Run from the IsaacLab root in env_isaaclab with ``PYTHONPATH=source/scripts/rsl_rl:source/rsl_rl_sac_vendor``.
"""

import critic_batch_norm
import pytest
import torch
import weight_normalization
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict
from torch import nn

NUM_ENVS = 16


def _obs():
    return TensorDict({"policy": torch.randn(NUM_ENVS, 5)}, batch_size=[NUM_ENVS])


def _policy():
    return ActorCritic(_obs(), {"policy": ["policy"], "critic": ["policy"]}, num_actions=2,
                       actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])


def test_insert_puts_batch_norm_before_each_hidden_activation_and_keeps_linear_keys():
    policy = _policy()
    keys = set(policy.state_dict())
    assert not critic_batch_norm.in_checkpoint(policy.state_dict())

    parameters = critic_batch_norm.insert(policy.critic)

    assert len(parameters) == 4
    blocks = [module for module in policy.critic if isinstance(module, nn.Sequential)]
    assert [(type(block[0]), block[0].num_features, type(block[1])) for block in blocks] == [
        (nn.BatchNorm1d, 16, nn.ELU), (nn.BatchNorm1d, 8, nn.ELU)
    ]
    assert keys < set(policy.state_dict()) and critic_batch_norm.in_checkpoint(policy.state_dict())
    assert not any(isinstance(module, nn.BatchNorm1d) for module in policy.actor.modules())


def test_insert_rejects_non_mlp_value_network():
    with pytest.raises(ValueError, match="Sequential MLP"):
        critic_batch_norm.insert(nn.Linear(5, 1))
    with pytest.raises(ValueError, match="hidden Linear"):
        critic_batch_norm.insert(nn.Sequential(nn.Linear(5, 1)))


def test_ppo_update_trains_the_batch_norm_value_network_and_play_can_load_it():
    torch.manual_seed(0)
    policy = _policy()
    ppo = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-2)
    ppo.init_storage("rl", NUM_ENVS, 8, _obs(), [2])
    parameters = critic_batch_norm.insert(policy.critic)
    ppo.optimizer.param_groups[0]["params"].extend(parameters)
    # The combination with weight normalization projects the same hidden Linear layers.
    weight_normalization.attach(type("Runner", (), {"alg": ppo})())
    norms = [module for module in policy.critic.modules() if isinstance(module, nn.BatchNorm1d)]
    before = [(norm.weight.detach().clone(), norm.running_mean.clone()) for norm in norms]

    policy.train()
    obs = _obs()
    with torch.inference_mode():  # like OnPolicyRunner's rollouts, with the policy in training mode
        for _ in range(8):
            ppo.act(obs)
            obs = _obs()
            ppo.process_env_step(obs, torch.randn(NUM_ENVS), (torch.rand(NUM_ENVS) < 0.2).float(), {})
        ppo.compute_returns(obs)
    ppo.update()

    for norm, (weight, running_mean) in zip(norms, before):
        assert not torch.equal(norm.weight, weight) and not torch.equal(norm.running_mean, running_mean)
    assert all(torch.isfinite(p).all() for p in policy.parameters())

    # play_direct_0325.py: detect BatchNorm in the checkpoint, insert it, then load strictly.
    state = policy.state_dict()
    played = _policy()
    if critic_batch_norm.in_checkpoint(state):
        critic_batch_norm.insert(played.critic)
    played.load_state_dict(state)
    played.eval(), policy.eval()
    with torch.no_grad():
        test_obs = _obs()
        torch.testing.assert_close(played.act_inference(test_obs), policy.act_inference(test_obs))
        torch.testing.assert_close(played.evaluate(test_obs), policy.evaluate(test_obs))
