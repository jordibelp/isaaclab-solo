# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check that opt-in PPO compilation preserves parameters and gradients."""

import ast
import copy
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[4]


def _solo12_ppo_class():
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/rsl_rl_ppo_cfg.py"
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12PPO"]
    namespace = {"PPO": PPO, "torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Solo12PPO"]


def test_compiled_ppo_forwards_keep_checkpoint_keys_and_gradients(monkeypatch):
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
    _solo12_ppo_class()(policy, torch_compile=True)

    assert policy.state_dict().keys() == eager_policy.state_dict().keys()
    for candidate in (eager_policy, policy):
        loss = candidate.actor(obs["policy"]).square().mean() + candidate.critic(obs["policy"]).square().mean()
        loss.backward()
    for eager_param, compiled_param in zip(eager_policy.parameters(), policy.parameters()):
        torch.testing.assert_close(compiled_param.grad, eager_param.grad)
