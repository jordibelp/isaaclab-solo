# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""XQC-style weight normalization on real SAC and PPO updates, without Isaac Sim.

Run from the IsaacLab root in env_isaaclab with ``PYTHONPATH=source/scripts/rsl_rl:source/rsl_rl_sac_vendor``.
"""

from types import SimpleNamespace

import torch
import weight_normalization as wn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl_sac.algorithms import SAC
from tensordict import TensorDict
from torch import nn

NUM_ENVS = 16


def _unit_error(layers: list[nn.Linear]) -> float:
    """Largest distance of a unit's (weights, bias) norm from 1."""
    rows = [torch.cat((layer.weight, layer.bias.unsqueeze(1)), dim=1).norm(dim=1) for layer in layers]
    return (torch.cat(rows) - 1).abs().max().item()


def _obs(width=6):
    return TensorDict({"policy": torch.randn(NUM_ENVS, width)}, batch_size=[NUM_ENVS])


def _sac(monkeypatch):
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    cfg = dict(
        num_steps_per_env=4,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[16, 16], activation="swish", obs_normalization=True),
        critic=dict(class_name="SACCriticModel", hidden_dims=[16, 16], activation="swish", obs_normalization=True,
                    layer_norm=True, distributional_loss="two_hot", distributional_num_bins=51,
                    distributional_symlog_limit=5.0),
        algorithm=dict(class_name="SAC", replay_buffer_size=NUM_ENVS * 32, num_mini_batches=4, mini_batch_size=8,
                       gamma=0.97, n_steps=3, policy_frequency=1, actor_learning_rate=1e-2,
                       critic_learning_rate=1e-2, alpha_learning_rate=1e-3, actor_optimizer="adamw",
                       critic_optimizer="adamw"),
    )
    obs = _obs()
    alg = SAC.construct_algorithm(obs, SimpleNamespace(num_actions=2, num_envs=NUM_ENVS), cfg, "cpu")
    for _ in range(12):
        alg.act(obs)
        obs = _obs()
        alg.process_env_step(obs, torch.randn(NUM_ENVS), (torch.rand(NUM_ENVS) < 0.2).float(), {})
    return alg


def test_projection_matches_official_xqc_norm_dense_layer():
    layer = nn.Linear(5, 3)
    # xqc/networks/common.py stores a Flax kernel as (in, out) and normalizes over the input axis.
    kernel, bias = layer.weight.detach().T.clone(), layer.bias.detach().clone()
    norm = torch.cat((kernel, bias.unsqueeze(0)), dim=0).norm(dim=0, keepdim=True)
    wn.project_to_unit_sphere([layer])
    torch.testing.assert_close(layer.weight.T, kernel / norm)
    torch.testing.assert_close(layer.bias, bias / norm.squeeze(0))


def test_sac_keeps_hidden_units_on_unit_sphere_and_output_layers_free(monkeypatch):
    torch.manual_seed(0)
    alg = _sac(monkeypatch)
    hidden = {
        "actor": wn.hidden_linear_layers(alg.actor.mlp),
        "critic": wn.hidden_linear_layers(alg.critic.critic1) + wn.hidden_linear_layers(alg.critic.critic2),
    }
    assert _unit_error(hidden["actor"] + hidden["critic"]) > 0.1  # PyTorch's init is not on the sphere

    assert wn.attach(SimpleNamespace(alg=alg)) == {"actor": 2, "critic": 4}
    assert _unit_error(hidden["actor"] + hidden["critic"]) < 1e-6
    critic = alg.critic
    for online, target in ((critic.critic1, critic.critic1_target), (critic.critic2, critic.critic2_target)):
        for name, value in online.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value)

    before = [p.detach().clone() for p in alg.actor_parameters + alg.critic_parameters]
    for _ in range(3):
        alg.update()
    after = alg.actor_parameters + alg.critic_parameters
    assert all(torch.isfinite(p).all() for p in after)
    assert all(not torch.equal(old, new) for old, new in zip(before, after) if new.dim() == 2)
    assert _unit_error(hidden["actor"] + hidden["critic"]) < 1e-5

    # The mean head starts near zero and the categorical logits at zero; projection would destroy both.
    outputs = [alg.actor.mlp[-2], alg.critic.critic1[-1], alg.critic.critic2[-1]]
    assert all(isinstance(layer, nn.Linear) for layer in outputs)
    assert max(layer.weight.norm(dim=1).max().item() for layer in outputs) < 0.5


def test_ppo_keeps_hidden_units_on_unit_sphere_and_output_layers_free():
    torch.manual_seed(0)
    obs = _obs(5)
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy"]}, num_actions=2,
                         actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
    ppo = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-2)
    ppo.init_storage("rl", NUM_ENVS, 8, obs, [2])
    hidden = wn.hidden_linear_layers(policy.actor) + wn.hidden_linear_layers(policy.critic)
    outputs = [policy.actor[-1], policy.critic[-1]]

    assert wn.attach(SimpleNamespace(alg=ppo)) == {"policy": 4}
    assert _unit_error(hidden) < 1e-6
    output_norms = [layer.weight.norm(dim=1).clone() for layer in outputs]

    for _ in range(8):
        ppo.act(obs)
        obs = _obs(5)
        ppo.process_env_step(obs, torch.randn(NUM_ENVS), (torch.rand(NUM_ENVS) < 0.2).float(), {})
    ppo.compute_returns(obs)
    ppo.update()

    assert _unit_error(hidden) < 1e-5
    for layer, norms in zip(outputs, output_norms):
        assert isinstance(layer, nn.Linear) and torch.isfinite(layer.weight).all()
        assert not torch.allclose(layer.weight.norm(dim=1), norms)  # trained and not rescaled to 1
        assert (layer.weight.norm(dim=1) - 1).abs().max() > 0.1
