# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl PPO variant with symmetry augmentation / mirror loss for Solo12."""

from __future__ import annotations

import copy
import importlib
import itertools
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.runner.torch import Runner


SymmetryPPO_DEFAULT_CONFIG = copy.deepcopy(PPO_DEFAULT_CONFIG)
SymmetryPPO_DEFAULT_CONFIG["symmetry"] = {
    "use_data_augmentation": False,
    "use_mirror_loss": False,
    "mirror_loss_coeff": 0.0,
    "data_augmentation_func": None,
    "obs_type": "policy",
}


def _resolve_callable(path_or_callable: Any) -> Any:
    if path_or_callable is None or callable(path_or_callable):
        return path_or_callable
    if not isinstance(path_or_callable, str):
        raise TypeError(f"Expected callable or import path string, got {type(path_or_callable)!r}")

    module_name, _, attr_name = path_or_callable.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(
            "Symmetry data_augmentation_func must be an import path like "
            "'solo12_symmetry.compute_symmetric_observations_actions'"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class SymmetryPPO(PPO):
    """PPO with optional symmetry-based data augmentation and mirror loss.

    The implementation mirrors the paper's three variants via a single config block:

    - augmentation only
    - mirror loss only
    - both
    """

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | None = None,
        action_space: int | tuple[int] | None = None,
        device: str | torch.device | None = None,
        cfg: Mapping[str, Any] | None = None,
        symmetry_env: Any | None = None,
    ) -> None:
        merged_cfg = copy.deepcopy(SymmetryPPO_DEFAULT_CONFIG)
        if cfg is not None:
            merged_cfg.update(cfg)

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=merged_cfg,
        )

        symmetry_cfg = copy.deepcopy(self.cfg.get("symmetry", {}) or {})
        self._use_data_augmentation = bool(symmetry_cfg.get("use_data_augmentation", False))
        self._use_mirror_loss = bool(symmetry_cfg.get("use_mirror_loss", False))
        self._mirror_loss_coeff = float(symmetry_cfg.get("mirror_loss_coeff", 0.0) or 0.0)
        self._symmetry_obs_type = str(symmetry_cfg.get("obs_type", "policy"))
        self._symmetry_data_augmentation_func = _resolve_callable(symmetry_cfg.get("data_augmentation_func"))
        self._symmetry_env = symmetry_env

        if (self._use_data_augmentation or self._use_mirror_loss) and self._symmetry_data_augmentation_func is None:
            raise ValueError(
                "Symmetry PPO is enabled, but no symmetry data_augmentation_func was provided in cfg['symmetry']."
            )
        if self._use_mirror_loss and self._mirror_loss_coeff <= 0.0:
            logger.warning(
                "Mirror loss is enabled but mirror_loss_coeff <= 0. The run will behave like augmentation-only / vanilla PPO."
            )

    def _augment_batch(self, obs: torch.Tensor | None, actions: torch.Tensor | None):
        return self._symmetry_data_augmentation_func(
            env=self._symmetry_env,
            obs=obs,
            actions=actions,
            obs_type=self._symmetry_obs_type,
        )

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step with optional symmetry augmentation / mirror loss."""

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_symmetry_loss = 0.0
        cumulative_symmetry_mae = 0.0

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for (
                sampled_states_raw,
                sampled_actions_raw,
                sampled_log_prob_raw,
                sampled_values_raw,
                sampled_returns_raw,
                sampled_advantages_raw,
            ) in sampled_batches:
                original_batch_size = sampled_states_raw.shape[0]
                sampled_actions = sampled_actions_raw
                sampled_log_prob = sampled_log_prob_raw
                sampled_values = sampled_values_raw
                sampled_returns = sampled_returns_raw
                sampled_advantages = sampled_advantages_raw

                if self._use_data_augmentation:
                    sampled_states_raw, sampled_actions = self._augment_batch(sampled_states_raw, sampled_actions_raw)
                    num_aug = int(sampled_states_raw.shape[0] / original_batch_size)
                    sampled_log_prob = sampled_log_prob.repeat(num_aug, 1)
                    sampled_values = sampled_values.repeat(num_aug, 1)
                    sampled_returns = sampled_returns.repeat(num_aug, 1)
                    sampled_advantages = sampled_advantages.repeat(num_aug, 1)
                else:
                    num_aug = 1

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states_raw, train=not epoch)
                    original_states = sampled_states[:original_batch_size]

                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
                    next_log_prob_original = next_log_prob[:original_batch_size]

                    with torch.no_grad():
                        ratio_for_kl = next_log_prob_original - sampled_log_prob_raw
                        kl_divergence = ((torch.exp(ratio_for_kl) - 1) - ratio_for_kl).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    if self._entropy_loss_scale:
                        entropy = self.policy.get_entropy(role="policy")
                        entropy_loss = -self._entropy_loss_scale * entropy[:original_batch_size].mean()
                    else:
                        entropy_loss = torch.zeros((), device=self.device)

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    symmetry_loss = torch.zeros((), device=self.device)
                    symmetry_mae = torch.zeros((), device=self.device)
                    if self._use_mirror_loss and self._mirror_loss_coeff > 0.0:
                        _, _, original_outputs = self.policy.act({"states": original_states}, role="policy")
                        mean_actions_original = original_outputs["mean_actions"]
                        symmetric_states_raw, symmetric_target_actions = self._augment_batch(
                            sampled_states_raw[:original_batch_size], mean_actions_original
                        )
                        symmetric_states = self._state_preprocessor(symmetric_states_raw, train=False)
                        _, _, symmetric_outputs = self.policy.act({"states": symmetric_states}, role="policy")
                        mean_actions_symmetric = symmetric_outputs["mean_actions"]
                        symmetry_mae = F.l1_loss(mean_actions_symmetric, symmetric_target_actions)
                        symmetry_loss = self._mirror_loss_coeff * F.mse_loss(
                            mean_actions_symmetric, symmetric_target_actions
                        )

                total_loss = policy_loss + entropy_loss + value_loss + symmetry_loss
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy_loss += entropy_loss.item() if self._entropy_loss_scale else 0.0
                cumulative_symmetry_loss += symmetry_loss.item() if self._use_mirror_loss else 0.0
                cumulative_symmetry_mae += symmetry_mae.item() if self._use_mirror_loss else 0.0

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        normalizer = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / normalizer)
        self.track_data("Loss / Value loss", cumulative_value_loss / normalizer)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / normalizer)
        if self._use_mirror_loss:
            self.track_data("Loss / Symmetry loss", cumulative_symmetry_loss / normalizer)
            self.track_data("Symmetry / Action mean absolute error", cumulative_symmetry_mae / normalizer)
        self.track_data("Symmetry / Num transforms", float(num_aug))
        self.track_data("Symmetry / Data augmentation enabled", float(self._use_data_augmentation))
        self.track_data("Symmetry / Mirror loss enabled", float(self._use_mirror_loss))
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])


class SymmetryRunner(Runner):
    """skrl Runner that knows how to instantiate :class:`SymmetryPPO`."""

    def _component(self, name: str):
        lowered = name.lower()
        if lowered in ["symmetryppo", "symmetry_ppo", "symmetryppo_default_config", "symmetry_ppo_default_config"]:
            return SymmetryPPO_DEFAULT_CONFIG if "default_config" in lowered else SymmetryPPO
        return super()._component(name)

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Any:
        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if agent_class not in ["symmetryppo", "symmetry_ppo"]:
            return super()._generate_agent(env, cfg, models)

        if isinstance(env, MultiAgentEnvWrapper):
            raise NotImplementedError("SymmetryPPO is only implemented for single-agent environments.")

        device = env.device
        num_envs = env.num_envs
        observation_space = env.observation_space
        action_space = env.action_space

        if "memory" not in cfg:
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}

        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")

        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        memory = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        agent_cfg = copy.deepcopy(self._component("symmetryppo_default_config"))
        agent_cfg.update(self._process_cfg(cfg["agent"]))
        agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": observation_space, "device": device})
        agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})

        return SymmetryPPO(
            models=models["agent"],
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=agent_cfg,
            symmetry_env=env,
        )
