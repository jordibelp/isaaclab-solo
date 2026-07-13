# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plasticity-mitigation interventions for RSL-RL actor/critic networks.

The defaults reproduce the interventions evaluated in:
    Juliani & Ash, "A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning"
    (arXiv:2405.19153).

The intervention target is deliberately the actor/critic network layers. Separate
continuous-action exploration parameters (``std``/``log_std``) and empirical
observation-normalizer buffers are not included.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import nn


CANONICAL_STRATEGIES = (
    "none",
    "layernorm",
    "regenerative-l2",
    "regenerative-l2-layernorm",
    "shrink-perturb",
    "soft-shrink-perturb",
    "soft-shrink-perturb-layernorm",
)

_STRATEGY_ALIASES = {
    "": "none",
    "off": "none",
    "regenerative-regularization": "regenerative-l2",
    "regenerative-regularization-layernorm": "regenerative-l2-layernorm",
    "regenerative-l2+layernorm": "regenerative-l2-layernorm",
    "regen-l2": "regenerative-l2",
    "regen-l2-layernorm": "regenerative-l2-layernorm",
    "soft-shrink-perturb+layernorm": "soft-shrink-perturb-layernorm",
    "soft-sp": "soft-shrink-perturb",
    "soft-sp-layernorm": "soft-shrink-perturb-layernorm",
    "sp": "shrink-perturb",
    "ln": "layernorm",
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    layer_norm: bool = False
    regenerative_l2: bool = False
    boundary_shrink_perturb: bool = False
    soft_shrink_perturb: bool = False


_STRATEGY_SPECS = {
    "none": StrategySpec("none"),
    "layernorm": StrategySpec("layernorm", layer_norm=True),
    "regenerative-l2": StrategySpec("regenerative-l2", regenerative_l2=True),
    "regenerative-l2-layernorm": StrategySpec(
        "regenerative-l2-layernorm", layer_norm=True, regenerative_l2=True
    ),
    "shrink-perturb": StrategySpec("shrink-perturb", boundary_shrink_perturb=True),
    "soft-shrink-perturb": StrategySpec("soft-shrink-perturb", soft_shrink_perturb=True),
    "soft-shrink-perturb-layernorm": StrategySpec(
        "soft-shrink-perturb-layernorm", layer_norm=True, soft_shrink_perturb=True
    ),
}


def resolve_strategy(name: str | None) -> StrategySpec:
    """Return a canonical strategy specification, accepting a few readable aliases."""

    normalized = "none" if name is None else str(name).strip().lower().replace("_", "-")
    normalized = _STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in _STRATEGY_SPECS:
        choices = ", ".join(CANONICAL_STRATEGIES)
        raise ValueError(f"Unknown plasticity mitigation strategy '{name}'. Choose one of: {choices}.")
    return _STRATEGY_SPECS[normalized]


@dataclass(frozen=True)
class LayerNormInsertionResult:
    layer_count: int
    parameter_count: int
    parameter_names: tuple[str, ...]


def _sequential_with_hidden_layer_norm(branch: nn.Module, branch_name: str) -> tuple[nn.Sequential, int]:
    if not isinstance(branch, nn.Sequential):
        raise ValueError(
            f"LayerNorm mitigation requires policy.{branch_name} to be an nn.Sequential MLP, "
            f"got {type(branch).__name__}."
        )
    if any(isinstance(module, nn.LayerNorm) for module in branch.modules()):
        raise ValueError(f"policy.{branch_name} already contains LayerNorm; refusing to insert it twice.")

    # Read the registered slots directly instead of using children(). RSL-RL's
    # MLP deliberately reuses one stateless activation instance at several
    # Sequential indices, and PyTorch's traversal helpers deduplicate such
    # shared modules. ``_modules.items()`` preserves the actual forward order
    # and every repeated activation slot.
    module_items = list(branch._modules.items())
    modules = [module for _, module in module_items]
    linear_positions = [index for index, module in enumerate(modules) if isinstance(module, nn.Linear)]
    if len(linear_positions) < 2:
        raise ValueError(
            f"LayerNorm mitigation requires at least one hidden Linear layer in policy.{branch_name}."
        )
    transformed = OrderedDict(module_items)
    inserted = 0
    for linear_position in linear_positions[:-1]:
        linear = modules[linear_position]
        activation_position = linear_position + 1
        if activation_position >= len(modules) or isinstance(modules[activation_position], nn.Linear):
            raise ValueError(
                f"LayerNorm mitigation expected an activation after hidden Linear index {linear_position} "
                f"in policy.{branch_name}."
            )
        # Wrap the existing activation at its original Sequential index. This
        # preserves every Linear state-dict key (actor.0, actor.2, ...) and puts
        # LayerNorm immediately before the activation, as in the paper code.
        activation_name = module_items[activation_position][0]
        transformed[activation_name] = nn.Sequential(
            nn.LayerNorm(
                linear.out_features,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            ),
            modules[activation_position],
        )
        inserted += 1

    result = nn.Sequential(transformed)
    result.train(branch.training)
    return result, inserted


def insert_actor_critic_layer_norm(runner: Any) -> LayerNormInsertionResult:
    """Insert LayerNorm before each actor/critic hidden activation.

    RSL-RL has already created its optimizer when this function runs. Existing
    Linear modules are reused, and the new affine LayerNorm parameters are added
    to the optimizer's first parameter group so checkpoint ordering is stable.
    """

    policy = runner.alg.policy
    optimizer = runner.alg.optimizer
    if getattr(optimizer, "_plasticity_layer_norm_inserted", False):
        raise RuntimeError("Plasticity LayerNorm has already been inserted into this optimizer.")

    new_parameters: list[nn.Parameter] = []
    new_parameter_names: list[str] = []
    layer_count = 0
    for branch_name in ("actor", "critic"):
        branch = getattr(policy, branch_name, None)
        transformed, inserted = _sequential_with_hidden_layer_norm(branch, branch_name)
        setattr(policy, branch_name, transformed)
        layer_count += inserted
        for module_name, module in transformed.named_modules():
            if not isinstance(module, nn.LayerNorm):
                continue
            for parameter_name, parameter in module.named_parameters(recurse=False):
                new_parameters.append(parameter)
                full_module_name = f"{branch_name}.{module_name}" if module_name else branch_name
                new_parameter_names.append(f"{full_module_name}.{parameter_name}")

    if not new_parameters or not optimizer.param_groups:
        raise RuntimeError("LayerNorm insertion did not create optimizer-visible parameters.")
    optimizer.param_groups[0]["params"].extend(new_parameters)
    optimizer._plasticity_layer_norm_inserted = True
    return LayerNormInsertionResult(
        layer_count=layer_count,
        parameter_count=sum(parameter.numel() for parameter in new_parameters),
        parameter_names=tuple(new_parameter_names),
    )


@dataclass(frozen=True)
class _TargetParameter:
    name: str
    parameter: nn.Parameter
    owner: nn.Module
    local_name: str


def _collect_actor_critic_targets(policy: nn.Module) -> tuple[_TargetParameter, ...]:
    targets: list[_TargetParameter] = []
    seen: set[int] = set()
    for branch_name in ("actor", "critic"):
        branch = getattr(policy, branch_name, None)
        if not isinstance(branch, nn.Module):
            raise ValueError(
                f"Plasticity mitigation requires a torch module named policy.{branch_name}."
            )
        for module_name, module in branch.named_modules():
            for local_name, parameter in module.named_parameters(recurse=False):
                if id(parameter) in seen:
                    raise ValueError(
                        "Plasticity mitigation found a parameter shared by actor/critic targets; "
                        "shared-network policies need an explicit intervention topology."
                    )
                seen.add(id(parameter))
                prefix = f"{branch_name}.{module_name}" if module_name else branch_name
                targets.append(_TargetParameter(f"{prefix}.{local_name}", parameter, module, local_name))
    if not targets:
        raise ValueError("Plasticity mitigation found no actor/critic parameters.")
    return tuple(targets)


class _FreshInitializationSampler:
    """Sample fresh parameters from the actor/critic layer initialization distributions."""

    def __init__(self, targets: tuple[_TargetParameter, ...], seed: int) -> None:
        self.targets = targets
        self.seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}
        for target in targets:
            if not isinstance(target.owner, (nn.Linear, nn.LayerNorm)):
                raise ValueError(
                    "Shrink+perturb currently supports Linear and LayerNorm actor/critic parameters; "
                    f"{target.name} belongs to {type(target.owner).__name__}."
                )
            key = str(target.parameter.device)
            if key not in self._generators:
                generator = torch.Generator(device=target.parameter.device)
                generator.manual_seed(self.seed + 104_729 * len(self._generators))
                self._generators[key] = generator

    def _sample_target(self, target: _TargetParameter) -> torch.Tensor:
        parameter = target.parameter
        generator = self._generators[str(parameter.device)]
        value = torch.empty_like(parameter, memory_format=torch.preserve_format)
        if isinstance(target.owner, nn.Linear):
            if target.local_name == "weight":
                nn.init.kaiming_uniform_(value, a=math.sqrt(5), generator=generator)
            elif target.local_name == "bias":
                fan_in = target.owner.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(value, -bound, bound, generator=generator)
            else:
                raise ValueError(f"Unsupported Linear parameter for shrink+perturb: {target.name}.")
        elif isinstance(target.owner, nn.LayerNorm):
            if target.local_name == "weight":
                nn.init.ones_(value)
            elif target.local_name == "bias":
                nn.init.zeros_(value)
            else:
                raise ValueError(f"Unsupported LayerNorm parameter for shrink+perturb: {target.name}.")
        return value

    def sample(self) -> dict[str, torch.Tensor]:
        return {target.name: self._sample_target(target) for target in self.targets}

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "generator_states": {key: generator.get_state().cpu() for key, generator in self._generators.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        generator_states = state_dict.get("generator_states", {})
        if not isinstance(generator_states, dict):
            raise ValueError("Invalid shrink+perturb generator state.")
        missing = sorted(set(self._generators) - set(generator_states))
        if missing:
            raise ValueError(f"Shrink+perturb checkpoint is missing generator state for: {missing}.")
        for key, generator in self._generators.items():
            generator.set_state(generator_states[key].cpu())


class PlasticityMitigationController:
    """Apply a selected paper-style intervention to an RSL-RL runner."""

    STATE_VERSION = 1

    def __init__(
        self,
        runner: Any,
        *,
        strategy: str | StrategySpec,
        regenerative_l2_coef: float = 1.0e-2,
        soft_shrink_perturb_beta: float = 1.0e-6,
        shrink_perturb_beta: float = 0.5,
        seed: int = 0,
        learning_rate: float | None = None,
        layer_norm_count: int = 0,
    ) -> None:
        self.runner = runner
        self.policy = runner.alg.policy
        self.optimizer = runner.alg.optimizer
        self.spec = resolve_strategy(strategy.name if isinstance(strategy, StrategySpec) else strategy)
        if self.spec.name == "none":
            raise ValueError("Do not construct PlasticityMitigationController for strategy 'none'.")

        self.regenerative_l2_coef = float(regenerative_l2_coef)
        self.soft_shrink_perturb_beta = float(soft_shrink_perturb_beta)
        self.shrink_perturb_beta = float(shrink_perturb_beta)
        self.seed = int(seed)
        self.learning_rate = float(
            getattr(runner.alg, "learning_rate", 0.0) if learning_rate is None else learning_rate
        )
        self.layer_norm_count = int(layer_norm_count)
        self._validate_hyperparameters()

        self.targets = _collect_actor_critic_targets(self.policy)
        self.target_names = tuple(target.name for target in self.targets)
        self.target_parameter_count = sum(target.parameter.numel() for target in self.targets)
        self.total_policy_parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        self.initial_parameters = (
            {target.name: target.parameter.detach().clone() for target in self.targets}
            if self.spec.regenerative_l2
            else {}
        )
        self.sampler = (
            _FreshInitializationSampler(self.targets, self.seed)
            if self.spec.soft_shrink_perturb or self.spec.boundary_shrink_perturb
            else None
        )

        self.optimizer_steps = 0
        self.soft_perturbations = 0
        self.boundary_perturbations = 0
        self.last_regenerative_distance = 0.0
        self.last_regenerative_penalty = 0.0
        self.last_perturbation_l2 = 0.0
        self._current_regenerative_gradient_scale = 0.0
        self._gradient_hook_handles: list[Any] = []

        if self.spec.regenerative_l2:
            self._register_regenerative_gradient_hooks()
            self._wrap_optimizer_zero_grad()
        if self.spec.regenerative_l2 or self.spec.soft_shrink_perturb:
            self._wrap_optimizer_step()
        self.optimizer._plasticity_mitigation_controller = self

    def _validate_hyperparameters(self) -> None:
        if self.spec.regenerative_l2 and self.regenerative_l2_coef <= 0.0:
            raise ValueError("--plasticity-regen-l2-coef must be positive for regenerative L2.")
        active_betas = []
        if self.spec.soft_shrink_perturb:
            active_betas.append(("--plasticity-soft-sp-beta", self.soft_shrink_perturb_beta))
        if self.spec.boundary_shrink_perturb:
            active_betas.append(("--plasticity-sp-beta", self.shrink_perturb_beta))
        for name, value in active_betas:
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")

    @property
    def target_parameter_fraction(self) -> float:
        return self.target_parameter_count / self.total_policy_parameter_count

    @torch.no_grad()
    def parameter_distance_from_initial(self) -> float:
        if not self.initial_parameters:
            return 0.0
        squared = torch.zeros((), device=self.targets[0].parameter.device, dtype=torch.float64)
        for target in self.targets:
            reference = self.initial_parameters[target.name].to(target.parameter.device)
            squared += (target.parameter.detach().double() - reference.double()).square().sum()
        return float(squared.sqrt().item())

    @torch.no_grad()
    def target_parameter_l2(self) -> float:
        squared = torch.zeros((), device=self.targets[0].parameter.device, dtype=torch.float64)
        for target in self.targets:
            squared += target.parameter.detach().double().square().sum()
        return float(squared.sqrt().item())

    def _prepare_regenerative_gradient(self) -> None:
        squared = torch.zeros((), device=self.targets[0].parameter.device, dtype=torch.float64)
        for target in self.targets:
            reference = self.initial_parameters[target.name].to(target.parameter.device)
            squared += (target.parameter.detach().double() - reference.double()).square().sum()
        distance = float(squared.sqrt().item())
        self.last_regenerative_distance = distance
        self.last_regenerative_penalty = self.regenerative_l2_coef * distance
        self._current_regenerative_gradient_scale = (
            self.regenerative_l2_coef / distance if distance > 0.0 else 0.0
        )

    def _register_regenerative_gradient_hooks(self) -> None:
        for target in self.targets:
            def _add_regenerative_gradient(gradient, *, _target=target):
                scale = self._current_regenerative_gradient_scale
                if scale == 0.0:
                    return gradient
                reference = self.initial_parameters[_target.name].to(_target.parameter.device)
                return gradient + scale * (_target.parameter.detach() - reference)

            self._gradient_hook_handles.append(target.parameter.register_hook(_add_regenerative_gradient))

    def _wrap_optimizer_zero_grad(self) -> None:
        if getattr(self.optimizer, "_plasticity_regenerative_zero_grad_wrapped", False):
            raise RuntimeError("Regenerative L2 optimizer.zero_grad is already wrapped.")
        original_zero_grad = self.optimizer.zero_grad

        def _zero_grad_with_regenerative_reference(_optimizer, *args, **kwargs):
            result = original_zero_grad(*args, **kwargs)
            self._prepare_regenerative_gradient()
            return result

        self.optimizer.zero_grad = MethodType(_zero_grad_with_regenerative_reference, self.optimizer)
        self.optimizer._plasticity_regenerative_zero_grad_wrapped = True

    def _wrap_optimizer_step(self) -> None:
        if getattr(self.optimizer, "_plasticity_mitigation_step_wrapped", False):
            raise RuntimeError("Plasticity mitigation optimizer.step is already wrapped.")
        original_step = self.optimizer.step

        def _step_with_mitigation(_optimizer, *args, **kwargs):
            result = original_step(*args, **kwargs)
            self.optimizer_steps += 1
            if self.spec.soft_shrink_perturb:
                self._apply_shrink_perturb(self.soft_shrink_perturb_beta)
                self.soft_perturbations += 1
            return result

        self.optimizer.step = MethodType(_step_with_mitigation, self.optimizer)
        self.optimizer._plasticity_mitigation_step_wrapped = True

    @torch.no_grad()
    def _apply_shrink_perturb(self, beta: float) -> None:
        if self.sampler is None:
            raise RuntimeError("Shrink+perturb requested without a fresh-initialization sampler.")
        fresh = self.sampler.sample()
        alpha = 1.0 - float(beta)
        delta_squared = torch.zeros((), device=self.targets[0].parameter.device, dtype=torch.float64)
        for target in self.targets:
            before = target.parameter.detach().clone()
            target.parameter.mul_(alpha).add_(fresh[target.name], alpha=float(beta))
            delta_squared += (target.parameter.detach().double() - before.double()).square().sum()
        self.last_perturbation_l2 = float(delta_squared.sqrt().item())

    def on_distribution_shift(self) -> bool:
        """Apply boundary shrink+perturb and fresh optimizer state, if selected."""

        if not self.spec.boundary_shrink_perturb:
            return False
        self._apply_shrink_perturb(self.shrink_perturb_beta)
        self.optimizer.state.clear()
        for parameter in self.policy.parameters():
            parameter.grad = None
        self.runner.alg.learning_rate = self.learning_rate
        self.optimizer.defaults["lr"] = self.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.boundary_perturbations += 1
        return True

    def logging_values(self) -> dict[str, float | int]:
        return {
            "enabled": 1,
            "layernorm_enabled": int(self.spec.layer_norm),
            "regenerative_l2_enabled": int(self.spec.regenerative_l2),
            "soft_shrink_perturb_enabled": int(self.spec.soft_shrink_perturb),
            "boundary_shrink_perturb_enabled": int(self.spec.boundary_shrink_perturb),
            "target_parameter_fraction": self.target_parameter_fraction,
            "target_parameter_l2": self.target_parameter_l2(),
            "regenerative_distance_from_init": (
                self.parameter_distance_from_initial() if self.spec.regenerative_l2 else 0.0
            ),
            "regenerative_penalty": self.last_regenerative_penalty,
            "optimizer_steps": self.optimizer_steps,
            "soft_perturbations_total": self.soft_perturbations,
            "boundary_perturbations_total": self.boundary_perturbations,
            "last_perturbation_l2": self.last_perturbation_l2,
            "layernorm_layers": self.layer_norm_count,
            "regenerative_l2_coef": self.regenerative_l2_coef if self.spec.regenerative_l2 else 0.0,
            "soft_sp_beta": self.soft_shrink_perturb_beta if self.spec.soft_shrink_perturb else 0.0,
            "sp_beta": self.shrink_perturb_beta if self.spec.boundary_shrink_perturb else 0.0,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.spec.name,
            "regenerative_l2_coef": self.regenerative_l2_coef,
            "soft_shrink_perturb_beta": self.soft_shrink_perturb_beta,
            "shrink_perturb_beta": self.shrink_perturb_beta,
            "seed": self.seed,
            "layer_norm_count": self.layer_norm_count,
            "target_parameter_count": self.target_parameter_count,
            "target_parameter_fraction": self.target_parameter_fraction,
            "optimizer_steps": self.optimizer_steps,
            "soft_perturbations": self.soft_perturbations,
            "boundary_perturbations": self.boundary_perturbations,
        }

    def state_dict(self) -> dict[str, Any]:
        state = {
            "state_version": self.STATE_VERSION,
            **self.summary(),
            "target_names": self.target_names,
            "initial_parameters": {
                name: value.detach().cpu().clone() for name, value in self.initial_parameters.items()
            },
            "sampler_state_dict": self.sampler.state_dict() if self.sampler is not None else None,
            "last_regenerative_distance": self.last_regenerative_distance,
            "last_regenerative_penalty": self.last_regenerative_penalty,
            "last_perturbation_l2": self.last_perturbation_l2,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_version = int(state_dict.get("state_version", 0))
        if state_version > self.STATE_VERSION:
            raise ValueError(
                f"Plasticity mitigation checkpoint version {state_version} is newer than supported "
                f"version {self.STATE_VERSION}."
            )
        checkpoint_strategy = resolve_strategy(state_dict.get("strategy", "none")).name
        if checkpoint_strategy != self.spec.name:
            raise ValueError(
                "Plasticity mitigation checkpoint strategy mismatch: "
                f"checkpoint={checkpoint_strategy}, requested={self.spec.name}."
            )
        checkpoint_names = tuple(state_dict.get("target_names", ()))
        if checkpoint_names != self.target_names:
            raise ValueError("Plasticity mitigation checkpoint actor/critic parameter layout does not match.")

        active_hyperparameters = []
        if self.spec.regenerative_l2:
            active_hyperparameters.append(("regenerative_l2_coef", self.regenerative_l2_coef))
        if self.spec.soft_shrink_perturb:
            active_hyperparameters.append(("soft_shrink_perturb_beta", self.soft_shrink_perturb_beta))
        if self.spec.boundary_shrink_perturb:
            active_hyperparameters.append(("shrink_perturb_beta", self.shrink_perturb_beta))
        for key, requested in active_hyperparameters:
            saved = float(state_dict.get(key, requested))
            if not math.isclose(saved, requested, rel_tol=0.0, abs_tol=1.0e-15):
                raise ValueError(
                    f"Plasticity mitigation checkpoint {key}={saved:g} does not match requested {requested:g}."
                )

        saved_initial = state_dict.get("initial_parameters", {})
        if self.spec.regenerative_l2:
            if not isinstance(saved_initial, dict) or set(saved_initial) != set(self.target_names):
                raise ValueError("Regenerative-L2 checkpoint is missing its initial parameter reference.")
            for target in self.targets:
                value = saved_initial[target.name]
                if tuple(value.shape) != tuple(target.parameter.shape):
                    raise ValueError(f"Regenerative-L2 reference shape mismatch for {target.name}.")
                self.initial_parameters[target.name] = value.to(target.parameter.device).clone()

        sampler_state = state_dict.get("sampler_state_dict")
        if self.sampler is not None:
            if not isinstance(sampler_state, dict):
                raise ValueError("Shrink+perturb checkpoint is missing sampler state.")
            self.sampler.load_state_dict(sampler_state)

        self.seed = int(state_dict.get("seed", self.seed))
        if self.sampler is not None:
            self.sampler.seed = self.seed

        self.optimizer_steps = int(state_dict.get("optimizer_steps", 0))
        self.soft_perturbations = int(state_dict.get("soft_perturbations", 0))
        self.boundary_perturbations = int(state_dict.get("boundary_perturbations", 0))
        self.last_regenerative_distance = float(state_dict.get("last_regenerative_distance", 0.0))
        self.last_regenerative_penalty = float(state_dict.get("last_regenerative_penalty", 0.0))
        self.last_perturbation_l2 = float(state_dict.get("last_perturbation_l2", 0.0))
