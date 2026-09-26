# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Check Hydra agent-level optimizer overrides without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]


def _train_helpers():
    tree = ast.parse((ROOT / "source/scripts/rsl_rl/train.py").read_text())
    wanted = {
        "_agent_policy_optimizers",
        "_configure_sac_optimizer",
        "_policy_action_noise_param_ids",
        "_split_action_noise_optimizer_group",
        "_apply_agent_weight_decay_to_optimizer",
        "_apply_agent_adam_betas_to_optimizer",
    }
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted
    namespace = {"torch": torch, "RslRlBaseRunnerCfg": object}
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted})


train = _train_helpers()


def _cfg(beta1=0.85, beta2=0.95, weight_decay=0.01):
    return SimpleNamespace(adam_beta1=beta1, adam_beta2=beta2, weight_decay=weight_decay)


@pytest.mark.parametrize("name,expected", [("adam", torch.optim.Adam), ("adamW", torch.optim.AdamW)])
def test_sac_optimizer_ablation_keeps_betas_and_decay(name, expected):
    cfg = _cfg()
    cfg.optimizer = name
    cfg.algorithm = SimpleNamespace(actor_optimizer="adam", critic_optimizer="adam")
    train._configure_sac_optimizer(cfg)
    assert cfg.algorithm.actor_optimizer == cfg.algorithm.critic_optimizer == name.lower()

    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    alpha = torch.nn.Parameter(torch.zeros(()))
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            actor_optimizer=expected(actor.parameters()),
            critic_optimizer=expected(critic.parameters()),
            alpha_optimizer=torch.optim.Adam([alpha]),
        )
    )
    train._apply_agent_weight_decay_to_optimizer(runner, cfg)
    train._apply_agent_adam_betas_to_optimizer(runner, cfg)
    for optimizer in (runner.alg.actor_optimizer, runner.alg.critic_optimizer):
        assert isinstance(optimizer, expected)
        assert optimizer.param_groups[0]["betas"] == (0.85, 0.95)
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert runner.alg.alpha_optimizer.param_groups[0]["weight_decay"] == 0.0


def test_invalid_sac_optimizer_fails():
    cfg = SimpleNamespace(optimizer="sgd", algorithm=SimpleNamespace())
    with pytest.raises(ValueError, match="agent.optimizer"):
        train._configure_sac_optimizer(cfg)


def _sac_runner():
    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    alpha = torch.nn.Parameter(torch.zeros(()))
    alg = SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()),
        alpha_optimizer=torch.optim.Adam([alpha]),
    )
    return SimpleNamespace(alg=alg)


def test_sac_agent_overrides_actor_and_critic_but_not_temperature():
    runner = _sac_runner()
    cfg = _cfg()
    train._apply_agent_weight_decay_to_optimizer(runner, cfg)
    train._apply_agent_adam_betas_to_optimizer(runner, cfg)

    for optimizer in (runner.alg.actor_optimizer, runner.alg.critic_optimizer):
        assert all(group["betas"] == (0.85, 0.95) for group in optimizer.param_groups)
        assert all(group["weight_decay"] == 0.01 for group in optimizer.param_groups)
    assert runner.alg.alpha_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert runner.alg.alpha_optimizer.param_groups[0]["weight_decay"] == 0.0


def test_sac_default_overrides_leave_optimizer_defaults_unchanged():
    runner = _sac_runner()
    train._apply_agent_weight_decay_to_optimizer(runner, _cfg(weight_decay=0.0))
    train._apply_agent_adam_betas_to_optimizer(runner, _cfg(beta1=0.9, beta2=0.999))
    for optimizer in (runner.alg.actor_optimizer, runner.alg.critic_optimizer):
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
        assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_sac_beta2_only_keeps_default_beta1():
    runner = _sac_runner()
    train._apply_agent_adam_betas_to_optimizer(runner, _cfg(beta1=0.9, beta2=0.95))
    for optimizer in (runner.alg.actor_optimizer, runner.alg.critic_optimizer):
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_ppo_action_noise_stays_decay_free():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.log_std = torch.nn.Parameter(torch.zeros(2))

    policy = Policy()
    runner = SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters())))
    train._apply_agent_weight_decay_to_optimizer(runner, _cfg())
    train._apply_agent_adam_betas_to_optimizer(runner, _cfg())
    assert len(runner.alg.optimizer.param_groups) == 2
    groups = {id(group["params"][0]): group for group in runner.alg.optimizer.param_groups}
    assert groups[id(policy.weight)]["weight_decay"] == 0.01
    assert groups[id(policy.log_std)]["weight_decay"] == 0.0
    assert all(group["betas"] == (0.85, 0.95) for group in groups.values())


@pytest.mark.parametrize("field,value", [("adam_beta1", -0.1), ("adam_beta2", 1.0), ("weight_decay", -0.01)])
def test_invalid_agent_optimizer_override_fails(field, value):
    runner = _sac_runner()
    cfg = _cfg()
    setattr(cfg, field, value)
    fn = train._apply_agent_weight_decay_to_optimizer if field == "weight_decay" else train._apply_agent_adam_betas_to_optimizer
    with pytest.raises(ValueError, match=f"agent.{field}"):
        fn(runner, cfg)
