# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that opt-in PPO compilation preserves parameters and gradients."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import critic_batch_norm
import weight_normalization as wn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[4]


def _solo12_ppo_class():
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/solo12_ppo.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12PPO"]
    namespace = {"PPO": PPO, "torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Solo12PPO"]


@pytest.mark.parametrize("weight_normalization", [False, True])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_compiled_ppo_forwards_keep_checkpoint_keys_and_gradients(monkeypatch, weight_normalization, batch_norm):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn: original_compile(fn, backend="eager"))
    torch.manual_seed(42)
    obs = TensorDict({"policy": torch.randn(8, 5)}, batch_size=[8])
    policy = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        num_actions=2,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
    )
    eager_policy = copy.deepcopy(policy)
    eager_ppo = PPO(eager_policy)
    compiled_ppo = _solo12_ppo_class()(policy, torch_compile=True)
    if batch_norm:
        for ppo in (eager_ppo, compiled_ppo):
            ppo.optimizer.param_groups[0]["params"].extend(critic_batch_norm.insert(ppo.policy.critic))
    original_keys = policy.state_dict().keys()
    if weight_normalization:
        for ppo in (eager_ppo, compiled_ppo):
            wn.attach(SimpleNamespace(alg=ppo))

    assert policy.state_dict().keys() == eager_policy.state_dict().keys() == original_keys
    for candidate in (eager_policy, policy):
        loss = candidate.actor(obs["policy"]).square().mean() + candidate.critic(obs["policy"]).square().mean()
        loss.backward()
    for eager_param, compiled_param in zip(eager_policy.parameters(), policy.parameters()):
        torch.testing.assert_close(compiled_param.grad, eager_param.grad)
    for ppo in (eager_ppo, compiled_ppo):
        ppo.optimizer.step()
        if weight_normalization:
            for layer in wn.hidden_linear_layers(ppo.policy.actor) + wn.hidden_linear_layers(ppo.policy.critic):
                rows = torch.cat((layer.weight, layer.bias.unsqueeze(1)), dim=1)
                torch.testing.assert_close(rows.norm(dim=1), torch.ones(layer.out_features))
    for eager_param, compiled_param in zip(eager_policy.parameters(), policy.parameters()):
        torch.testing.assert_close(compiled_param, eager_param)
