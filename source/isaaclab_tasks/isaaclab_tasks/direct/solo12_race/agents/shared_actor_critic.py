# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.networks.mlp import resolve_nn_activation
from tensordict import TensorDict
from torch.distributions import Normal


_DEFAULT_MIN_ACTION_STD = 1.0e-6
_DEFAULT_MAX_ACTION_STD = 13.0


def make_shared_actor_critic_mlps(
    input_dim: int,
    num_actions: int,
    hidden_dims: tuple[int] | list[int],
    activation: str,
    *,
    state_dependent_std: bool = False,
) -> tuple[nn.Sequential, nn.Sequential]:
    """Build actor/critic MLPs with one shared hidden trunk and separate output heads."""

    hidden_dims = [input_dim if dim == -1 else int(dim) for dim in hidden_dims]
    trunk_layers: list[nn.Module] = []
    feature_dim = input_dim
    for hidden_dim in hidden_dims:
        trunk_layers.append(nn.Linear(feature_dim, hidden_dim))
        trunk_layers.append(resolve_nn_activation(activation))
        feature_dim = hidden_dim

    shared_trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
    actor_output_dim: int | tuple[int, int] = (2, num_actions) if state_dependent_std else num_actions
    if isinstance(actor_output_dim, int):
        actor = nn.Sequential(shared_trunk, nn.Linear(feature_dim, actor_output_dim))
    else:
        actor = nn.Sequential(
            shared_trunk,
            nn.Linear(feature_dim, actor_output_dim[0] * actor_output_dim[1]),
            nn.Unflatten(dim=-1, unflattened_size=actor_output_dim),
        )
    critic = nn.Sequential(shared_trunk, nn.Linear(feature_dim, 1))
    return actor, critic


def init_state_dependent_actor_std(
    actor: nn.Sequential,
    num_actions: int,
    *,
    noise_std_type: str,
    init_noise_std: float,
) -> None:
    """Initialize the std half of a state-dependent actor output."""

    output_layer = actor[-2] if isinstance(actor[-1], nn.Unflatten) else actor[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError(f"Expected actor output layer to be Linear, got {type(output_layer).__name__}.")
    torch.nn.init.zeros_(output_layer.weight[num_actions:])
    if noise_std_type == "scalar":
        torch.nn.init.constant_(output_layer.bias[num_actions:], init_noise_std)
    elif noise_std_type == "log":
        torch.nn.init.constant_(output_layer.bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7)))
    else:
        raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")


class SharedActorCritic(nn.Module):
    """RSL-RL actor-critic with optional shared actor/critic hidden layers.

    When ``shared_networks`` is false this behaves like the upstream flat ActorCritic.
    When true, actor and critic use a shared MLP trunk and separate action/value output heads.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        critic_hidden_dims: tuple[int] | list[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        shared_networks: bool = False,
        min_action_std: float = _DEFAULT_MIN_ACTION_STD,
        max_action_std: float | None = _DEFAULT_MAX_ACTION_STD,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("SharedActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs]))
        super().__init__()

        self.obs_groups = obs_groups
        self.shared_networks = bool(shared_networks)
        self.state_dependent_std = state_dependent_std
        self.min_action_std = max(float(min_action_std), _DEFAULT_MIN_ACTION_STD)
        self.max_action_std = None if max_action_std is None else float(max_action_std)
        if self.max_action_std is not None and self.max_action_std < self.min_action_std:
            raise ValueError(
                f"max_action_std must be >= min_action_std, got "
                f"{self.max_action_std:g} < {self.min_action_std:g}."
            )
        self._action_std_sanitize_warned = False

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        if self.shared_networks and num_actor_obs != num_critic_obs:
            raise ValueError(
                f"shared_networks=True requires actor and critic observation dims to match, "
                f"got actor={num_actor_obs}, critic={num_critic_obs}."
            )

        if self.shared_networks:
            if list(actor_hidden_dims) != list(critic_hidden_dims):
                print(
                    "[WARN]: shared_networks=True uses actor_hidden_dims as the shared trunk; "
                    "critic_hidden_dims is ignored.",
                    flush=True,
                )
            self.actor, self.critic = make_shared_actor_critic_mlps(
                num_actor_obs,
                num_actions,
                actor_hidden_dims,
                activation,
                state_dependent_std=state_dependent_std,
            )
        else:
            from rsl_rl.networks import MLP

            if state_dependent_std:
                self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
            else:
                self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
            self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        if self.shared_networks and critic_obs_normalization and actor_obs_normalization:
            self.critic_obs_normalizer = self.actor_obs_normalizer
        elif critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if state_dependent_std:
            init_state_dependent_actor_std(
                self.actor, num_actions, noise_std_type=noise_std_type, init_noise_std=init_noise_std
            )
        else:
            if noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _max_std_for(self, tensor: torch.Tensor) -> float:
        if self.max_action_std is not None:
            return self.max_action_std
        return math.sqrt(float(torch.finfo(tensor.dtype).max))

    def _sanitize_std_tensor(self, std: torch.Tensor) -> torch.Tensor:
        max_std = self._max_std_for(std)
        return torch.nan_to_num(std, nan=self.min_action_std, posinf=max_std, neginf=self.min_action_std).clamp(
            min=self.min_action_std, max=max_std
        )

    def _sanitize_log_std_tensor(self, log_std: torch.Tensor) -> torch.Tensor:
        min_log_std = math.log(self.min_action_std)
        max_log_std = math.log(self._max_std_for(log_std))
        return torch.nan_to_num(log_std, nan=0.0, posinf=max_log_std, neginf=min_log_std).clamp(
            min=min_log_std, max=max_log_std
        )

    def sanitize_action_std_(self) -> bool:
        """Keep non-state-dependent action std parameters finite and positive."""

        if self.state_dependent_std:
            return False
        if self.noise_std_type == "scalar" and hasattr(self, "std"):
            parameter = self.std
            repaired = self._sanitize_std_tensor(parameter.detach())
        elif self.noise_std_type == "log" and hasattr(self, "log_std"):
            parameter = self.log_std
            repaired = self._sanitize_log_std_tensor(parameter.detach())
        else:
            return False

        changed = not torch.equal(parameter.detach(), repaired)
        if changed:
            with torch.no_grad():
                parameter.copy_(repaired)
            if not self._action_std_sanitize_warned:
                max_std = self.max_action_std if self.max_action_std is not None else float("inf")
                print(
                    "[WARN]: Clamped policy action std parameter to finite range "
                    f"[{self.min_action_std:g}, {max_std:g}] before sampling.",
                    flush=True,
                )
                self._action_std_sanitize_warned = True
        return changed

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for obs_group in groups:
            assert len(obs[obs_group].shape) == 2, "SharedActorCritic expects flat observations."
            dim += obs[obs_group].shape[-1]
        return dim

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
                std = self._sanitize_std_tensor(std)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                log_std = self._sanitize_log_std_tensor(log_std)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(obs)
            self.sanitize_action_std_()
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization and self.critic_obs_normalizer is not self.actor_obs_normalizer:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    @staticmethod
    def _adapt_race_obs_tensor(
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        zero_missing_columns: bool,
    ) -> torch.Tensor | None:
        target_dim = 63
        old_current_dim = 57
        old_gate_idx_dim = 52
        old_shared_dim = 51  # old layout up to and including c1/c2; old gate_idx was at column 51.
        source_dim = source.shape[-1]
        if target.shape[-1] != target_dim or source_dim not in (old_current_dim, old_gate_idx_dim):
            return None

        adapted = torch.zeros_like(target) if zero_missing_columns else target.clone()
        copy_dim = old_shared_dim if source_dim == old_gate_idx_dim else old_current_dim
        adapted[..., :copy_dim] = source[..., :copy_dim]
        return adapted

    def _adapt_input_tensor(self, key: str, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if (
            key.startswith(("actor.", "critic."))
            and key.endswith(".weight")
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0]
        ):
            return self._adapt_race_obs_tensor(source, target, zero_missing_columns=True)

        if (
            key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer."))
            and key.rsplit(".", 1)[-1] in {"_mean", "_var", "_std"}
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0] == 1
        ):
            return self._adapt_race_obs_tensor(source, target, zero_missing_columns=False)

        if key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer.")) and key.endswith(".count"):
            return torch.zeros_like(target)

        return None

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        target_state = self.state_dict()
        exact_match = True
        adapted_keys: list[str] = []
        skipped_keys: list[str] = []

        for key, source_value in state_dict.items():
            if key not in target_state:
                skipped_keys.append(key)
                exact_match = False
                continue

            target_value = target_state[key]
            if tuple(source_value.shape) == tuple(target_value.shape):
                target_state[key] = source_value
                continue

            adapted_value = self._adapt_input_tensor(key, source_value, target_value)
            if adapted_value is None:
                skipped_keys.append(key)
                exact_match = False
                continue

            target_state[key] = adapted_value
            adapted_keys.append(key)
            exact_match = False

        if exact_match:
            super().load_state_dict(state_dict, strict=strict)
            return True

        if any(key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer.")) for key in adapted_keys):
            for key in ("actor_obs_normalizer.count", "critic_obs_normalizer.count"):
                if key in target_state:
                    target_state[key] = torch.zeros_like(target_state[key])

        super().load_state_dict(target_state, strict=True)
        if adapted_keys:
            print(
                "[INFO]: Warm-started SharedActorCritic from a checkpoint with a different race observation "
                f"layout; adapted {len(adapted_keys)} tensors and zero-initialized new cClose/cClose1 inputs.",
                flush=True,
            )
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            suffix = " ..." if len(skipped_keys) > 8 else ""
            print(f"[WARN]: Skipped {len(skipped_keys)} incompatible checkpoint tensors: {preview}{suffix}", flush=True)
        return False
