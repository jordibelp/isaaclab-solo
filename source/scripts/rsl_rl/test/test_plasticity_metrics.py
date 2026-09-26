# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plasticity-metric grouping and namespace checks without starting Isaac Sim.

Run from env_isaaclab:

    PYTHONPATH=source/scripts/rsl_rl:source/rsl_rl_sac_vendor python -m pytest \
        source/scripts/rsl_rl/test/test_plasticity_metrics.py -q
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import plasticity_metrics as pm
import pytest
import torch
from rsl_rl.modules import ActorCritic
from rsl_rl_sac.models import SACActorModel, SACCriticModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[4]
HIDDEN_DIMS = [8, 5]
NUM_HIDDEN_LAYERS = len(HIDDEN_DIMS)
PER_LAYER_ACTIVATION_METRICS = ("dormant_pct", "dormant_tau_pct", "feature_rank", "feature_rank_frac", "feature_num")
SUMMARY_ACTIVATION_METRICS = (
    "dormant_pct",
    "dormant_tau_pct",
    "feature_rank",
    "feature_rank_frac_median",
    "feature_rank_frac_mean",
    "feature_rank_frac_min",
    "feature_num",
)


def _train_helpers():
    """Compile train.py's plasticity helpers on their own, without importing Isaac Sim."""
    tree = ast.parse((ROOT / "source/scripts/rsl_rl/train.py").read_text())
    wanted = ("_attach_plasticity_metrics_to_runner", "_log_plasticity_metrics")
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == set(wanted)
    namespace = {"torch": torch, "plasticity_metrics": pm}
    exec(compile(ast.Module(body=body, type_ignores=[]), "train.py", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in wanted})


train = _train_helpers()


@pytest.fixture
def parsed_args():
    return SimpleNamespace(
        plasticity_metrics=True,
        plasticity_metrics_interval=1,
        plasticity_metrics_sample_cap=16,
    )


def ppo():
    obs = TensorDict({"policy": torch.randn(23, 7)}, batch_size=[23])
    policy = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["policy"]},
        3,
        actor_hidden_dims=HIDDEN_DIMS,
        critic_hidden_dims=HIDDEN_DIMS,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    return SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=optimizer), device="cpu"), obs


def sac():
    obs = TensorDict({"policy": torch.randn(23, 7)}, batch_size=[23])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 3, hidden_dims=HIDDEN_DIMS, state_dependent_std=True)
    critic = SACCriticModel(obs, groups, "critic", 1, hidden_dims=HIDDEN_DIMS, num_actions=3)
    alg = SimpleNamespace(
        actor=actor,
        critic=critic,
        actor_optimizer=torch.optim.Adam(actor.parameters(), lr=1.0e-3),
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=1.0e-3),
    )
    return SimpleNamespace(alg=alg, device="cpu"), obs


def logged(factory, parsed_args):
    runner, obs = factory()
    scalars = {}
    runner.writer = SimpleNamespace(add_scalar=lambda key, value, step: scalars.__setitem__(key, value))
    train._attach_plasticity_metrics_to_runner(runner, parsed_args)
    train._log_plasticity_metrics(runner, {"it": 0, "obs": obs}, parsed_args.plasticity_metrics_interval)
    return runner, obs, scalars


def test_sac_logs_each_twin_critic_separately(parsed_args):
    runner, _, scalars = logged(sac, parsed_args)

    assert set(runner._borinot_plasticity_groups) == {"actor", "critic1", "critic2"}
    # The pooled "critic" group is gone: a single collapsing Q network is now visible.
    assert not any(key.startswith(("Plasticity/summary/critic/", "Plasticity/per_layer/critic/")) for key in scalars)
    for net in ("actor", "critic1", "critic2"):
        assert f"Plasticity/summary/{net}/weight_norm" in scalars
        for metric in SUMMARY_ACTIVATION_METRICS:
            assert f"Plasticity/summary/{net}/{metric}" in scalars
        assert f"Plasticity/summary/{net}/feature_rank_frac" not in scalars
        for metric in PER_LAYER_ACTIVATION_METRICS:
            for layer in range(NUM_HIDDEN_LAYERS):
                assert f"Plasticity/per_layer/{net}/{metric}/layer_{layer:02d}" in scalars
    assert all(math.isfinite(value) for value in scalars.values())
    # Independently initialised twins must not report identical weight norms.
    assert scalars["Plasticity/summary/critic1/weight_norm"] != scalars["Plasticity/summary/critic2/weight_norm"]


def test_sac_per_layer_widths_match_hidden_dims(parsed_args):
    _, _, scalars = logged(sac, parsed_args)

    for net in ("actor", "critic1", "critic2"):
        widths = [scalars[f"Plasticity/per_layer/{net}/feature_num/layer_{i:02d}"] for i in range(NUM_HIDDEN_LAYERS)]
        assert widths == [float(dim) for dim in HIDDEN_DIMS]
        assert scalars[f"Plasticity/summary/{net}/feature_num"] == float(HIDDEN_DIMS[-1])


def test_ppo_groups_actor_and_critic(parsed_args):
    runner, _, scalars = logged(ppo, parsed_args)

    assert set(runner._borinot_plasticity_groups) == {"actor", "critic"}
    for net in ("actor", "critic"):
        assert f"Plasticity/summary/{net}/dormant_tau_pct" in scalars
        assert f"Plasticity/per_layer/{net}/dormant_tau_pct/layer_00" in scalars


@pytest.mark.parametrize("factory", [ppo, sac])
def test_gradient_kurtosis_reported_per_network(factory, parsed_args):
    runner, obs, scalars = logged(factory, parsed_args)
    groups = runner._borinot_plasticity_groups

    # Arm, then run one real optimizer step per distinct optimizer so the wrapped
    # step sees live gradients. PPO's actor/critic and SAC's twin critics share an
    # optimizer, which must still be wrapped exactly once.
    for capture in runner._borinot_plasticity_grad_captures.values():
        capture.arm()
    for module in groups.values():
        sum(p.square().sum() for p in module.parameters()).backward()
    for optimizer in vars(runner.alg).values():
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.step()

    train._log_plasticity_metrics(runner, {"it": 0, "obs": obs}, parsed_args.plasticity_metrics_interval)
    for net in groups:
        assert math.isfinite(scalars[f"Plasticity/summary/{net}/grad_kurtosis"])


def test_activation_metrics_pool_per_layer_values():
    dormant_layer = torch.tensor([[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])  # 2 of 4 units dormant
    live_layer = torch.ones(2, 6)  # 0 of 6 units dormant
    summary, per_layer = pm.activation_plasticity_metrics([dormant_layer, live_layer])

    assert per_layer["dormant_pct"] == [50.0, 0.0]
    assert per_layer["dormant_tau_pct"] == [50.0, 0.0]
    # Pooled over units (2/10), not averaged over layers (which would give 25%).
    assert summary["dormant_pct"] == 20.0
    assert summary["dormant_tau_pct"] == 20.0
    # Both layers are rank 1 (duplicated rows), so the fraction is 1/width.
    assert per_layer["feature_rank"] == [1.0, 1.0]
    assert per_layer["feature_rank_frac"] == pytest.approx([0.25, 1.0 / 6.0])
    assert summary["feature_rank"] == 1.0
    assert summary["feature_rank_frac_median"] == pytest.approx(0.5 * (0.25 + 1.0 / 6.0))
    assert summary["feature_rank_frac_mean"] == pytest.approx(0.5 * (0.25 + 1.0 / 6.0))
    assert summary["feature_rank_frac_min"] == pytest.approx(1.0 / 6.0)


def test_rank_fraction_summaries_distinguish_mean_median_and_min(monkeypatch):
    ranks_by_width = {4: 4.0, 8: 2.0, 16: 8.0}
    monkeypatch.setattr(pm, "feature_rank", lambda act, **kwargs: ranks_by_width[act.shape[-1]])
    activations = [torch.ones(16, width) for width in ranks_by_width]

    summary, per_layer = pm.activation_plasticity_metrics(activations)

    assert per_layer["feature_rank_frac"] == [1.0, 0.25, 0.5]
    assert summary["feature_rank_frac_median"] == 0.5
    assert summary["feature_rank_frac_mean"] == pytest.approx(1.75 / 3.0)
    assert summary["feature_rank_frac_min"] == 0.25
    assert "feature_rank_frac" not in summary


def test_no_activations_yields_no_metrics():
    assert pm.activation_plasticity_metrics([]) == ({}, {})
