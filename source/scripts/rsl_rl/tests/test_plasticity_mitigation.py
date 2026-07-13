from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plasticity_mitigation import (  # noqa: E402
    PlasticityMitigationController,
    insert_actor_critic_layer_norm,
    resolve_strategy,
)


class TinyActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Match RSL-RL's MLP construction: one stateless activation instance is
        # reused at every hidden slot in each branch.
        actor_activation = nn.ELU()
        critic_activation = nn.ELU()
        self.actor = nn.Sequential(
            nn.Linear(4, 8),
            actor_activation,
            nn.Linear(8, 6),
            actor_activation,
            nn.Linear(6, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, 8),
            critic_activation,
            nn.Linear(8, 6),
            critic_activation,
            nn.Linear(6, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(2))


def make_runner(*, learning_rate: float = 3.0e-3):
    policy = TinyActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    algorithm = SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=learning_rate, rnd=None)
    return SimpleNamespace(alg=algorithm, current_learning_iteration=0, logger_type=None, disable_logs=True)


def target_state(controller: PlasticityMitigationController) -> dict[str, torch.Tensor]:
    return {target.name: target.parameter.detach().clone() for target in controller.targets}


def test_strategy_aliases_resolve_to_canonical_compositions() -> None:
    assert resolve_strategy("ln").name == "layernorm"
    assert resolve_strategy("regenerative-regularization").name == "regenerative-l2"
    assert resolve_strategy("regenerative-l2+layernorm").name == "regenerative-l2-layernorm"
    assert resolve_strategy("soft-sp-layernorm").name == "soft-shrink-perturb-layernorm"
    with pytest.raises(ValueError, match="Unknown plasticity mitigation strategy"):
        resolve_strategy("mystery-method")


def test_layer_norm_is_inserted_before_each_hidden_activation_and_optimized() -> None:
    runner = make_runner()
    old_parameter_ids = {id(parameter) for parameter in runner.alg.policy.parameters()}
    old_linear_keys = {
        key for key in runner.alg.policy.state_dict() if key.startswith(("actor.", "critic."))
    }
    result = insert_actor_critic_layer_norm(runner)

    assert result.layer_count == 4
    assert result.parameter_count == 2 * ((8 + 8) + (6 + 6))
    for branch_name in ("actor", "critic"):
        branch = getattr(runner.alg.policy, branch_name)
        assert isinstance(branch[0], nn.Linear)
        assert isinstance(branch[1], nn.Sequential)
        assert isinstance(branch[1][0], nn.LayerNorm)
        assert isinstance(branch[1][1], nn.ELU)
        assert isinstance(branch[2], nn.Linear)
        assert isinstance(branch[3], nn.Sequential)
        assert isinstance(branch[3][0], nn.LayerNorm)
        assert isinstance(branch[3][1], nn.ELU)
        assert isinstance(branch[4], nn.Linear)

    optimized_ids = {
        id(parameter)
        for group in runner.alg.optimizer.param_groups
        for parameter in group["params"]
    }
    new_policy_ids = {id(parameter) for parameter in runner.alg.policy.parameters()}
    assert new_policy_ids - old_parameter_ids
    assert new_policy_ids <= optimized_ids
    assert old_linear_keys <= set(runner.alg.policy.state_dict())
    assert runner.alg.policy.actor(torch.ones(3, 4)).shape == (3, 2)


def test_soft_shrink_perturb_runs_after_every_optimizer_step_with_paper_formula() -> None:
    torch.manual_seed(3)
    runner = make_runner(learning_rate=0.0)
    controller = PlasticityMitigationController(
        runner,
        strategy="soft-shrink-perturb",
        soft_shrink_perturb_beta=0.2,
        seed=17,
        learning_rate=0.0,
    )
    before = target_state(controller)
    sampler_state = copy.deepcopy(controller.sampler.state_dict())
    fresh = controller.sampler.sample()
    controller.sampler.load_state_dict(sampler_state)
    log_std_before = runner.alg.policy.log_std.detach().clone()

    runner.alg.optimizer.zero_grad()
    loss = runner.alg.policy.actor(torch.ones(2, 4)).sum() + runner.alg.policy.critic(torch.ones(2, 4)).sum()
    loss.backward()
    runner.alg.optimizer.step()

    for target in controller.targets:
        expected = 0.8 * before[target.name] + 0.2 * fresh[target.name]
        torch.testing.assert_close(target.parameter, expected)
    torch.testing.assert_close(runner.alg.policy.log_std, log_std_before)
    assert controller.optimizer_steps == 1
    assert controller.soft_perturbations == 1
    assert controller.last_perturbation_l2 > 0.0


def test_boundary_shrink_perturb_clears_adam_and_restores_learning_rate() -> None:
    torch.manual_seed(5)
    runner = make_runner(learning_rate=3.0e-3)
    controller = PlasticityMitigationController(
        runner,
        strategy="shrink-perturb",
        shrink_perturb_beta=0.5,
        seed=23,
        learning_rate=3.0e-3,
    )

    runner.alg.optimizer.zero_grad()
    loss = runner.alg.policy.actor(torch.ones(2, 4)).square().sum()
    loss.backward()
    runner.alg.optimizer.step()
    assert runner.alg.optimizer.state
    before = target_state(controller)
    sampler_state = copy.deepcopy(controller.sampler.state_dict())
    fresh = controller.sampler.sample()
    controller.sampler.load_state_dict(sampler_state)
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 1.0e-5
    runner.alg.learning_rate = 1.0e-5

    assert controller.on_distribution_shift()

    for target in controller.targets:
        torch.testing.assert_close(target.parameter, 0.5 * before[target.name] + 0.5 * fresh[target.name])
    assert not runner.alg.optimizer.state
    assert runner.alg.learning_rate == pytest.approx(3.0e-3)
    assert all(group["lr"] == pytest.approx(3.0e-3) for group in runner.alg.optimizer.param_groups)
    assert all(parameter.grad is None for parameter in runner.alg.policy.parameters())
    assert controller.boundary_perturbations == 1


def test_regenerative_l2_adds_exact_global_unsquared_norm_gradient_before_clipping() -> None:
    torch.manual_seed(7)
    runner = make_runner(learning_rate=0.0)
    coefficient = 0.01
    controller = PlasticityMitigationController(
        runner,
        strategy="regenerative-l2",
        regenerative_l2_coef=coefficient,
        seed=29,
        learning_rate=0.0,
    )
    with torch.no_grad():
        for index, target in enumerate(controller.targets, start=1):
            target.parameter.add_(0.01 * index)

    squared = sum(
        (target.parameter.detach() - controller.initial_parameters[target.name]).square().sum()
        for target in controller.targets
    )
    distance = squared.sqrt()
    runner.alg.optimizer.zero_grad()
    zero_ppo_loss = sum((target.parameter * 0.0).sum() for target in controller.targets)
    zero_ppo_loss.backward()

    for target in controller.targets:
        expected = coefficient * (
            target.parameter.detach() - controller.initial_parameters[target.name]
        ) / distance
        torch.testing.assert_close(target.parameter.grad, expected)
    assert controller.last_regenerative_distance == pytest.approx(float(distance.item()))
    assert controller.last_regenerative_penalty == pytest.approx(coefficient * float(distance.item()))


@pytest.mark.parametrize("strategy", ["soft-shrink-perturb", "shrink-perturb"])
def test_shrink_perturb_rng_and_counters_resume_exactly(strategy: str) -> None:
    torch.manual_seed(11)
    first_runner = make_runner(learning_rate=0.0)
    first = PlasticityMitigationController(first_runner, strategy=strategy, seed=101, learning_rate=0.0)
    first._apply_shrink_perturb(0.1)
    first.optimizer_steps = 17
    first.soft_perturbations = 13
    first.boundary_perturbations = 2
    saved = copy.deepcopy(first.state_dict())

    torch.manual_seed(12)
    second_runner = make_runner(learning_rate=0.0)
    second = PlasticityMitigationController(second_runner, strategy=strategy, seed=999, learning_rate=0.0)
    second.load_state_dict(saved)

    first_fresh = first.sampler.sample()
    second_fresh = second.sampler.sample()
    assert first_fresh.keys() == second_fresh.keys()
    for name in first_fresh:
        torch.testing.assert_close(first_fresh[name], second_fresh[name])
    assert second.optimizer_steps == 17
    assert second.soft_perturbations == 13
    assert second.boundary_perturbations == 2
    assert second.seed == first.seed == 101


def test_regenerative_reference_is_restored_in_checkpoint_state() -> None:
    torch.manual_seed(19)
    first_runner = make_runner()
    first = PlasticityMitigationController(first_runner, strategy="regenerative-l2", seed=31)
    saved = copy.deepcopy(first.state_dict())

    torch.manual_seed(20)
    second_runner = make_runner()
    second = PlasticityMitigationController(second_runner, strategy="regenerative-l2", seed=31)
    second.load_state_dict(saved)

    for name in first.initial_parameters:
        torch.testing.assert_close(second.initial_parameters[name], first.initial_parameters[name])


def test_action_noise_parameter_is_not_an_intervention_target() -> None:
    runner = make_runner()
    controller = PlasticityMitigationController(runner, strategy="soft-shrink-perturb", seed=37)
    assert "log_std" not in controller.target_names
    assert all(name.startswith(("actor.", "critic.")) for name in controller.target_names)
