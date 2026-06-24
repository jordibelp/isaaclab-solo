# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization, MLP
from tensordict import TensorDict
from torch.distributions import Normal

from .shared_actor_critic import init_state_dependent_actor_std, make_shared_actor_critic_mlps


class EnvParamsConditionedEncoderActor(nn.Module):
    """Actor-critic that encodes privileged GT env params before the policy/value heads.

    The env still exposes the full flat observation:
        [current race obs, GT env params]

    Internally, actor and critic normalize the full observation, encode only the GT env
    params with a small MLP, and feed the heads with:
        [current race obs, env-param encoding]
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = (256, 128, 64),
        critic_hidden_dims: tuple[int] | list[int] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        current_obs_dim: int = 63,
        env_params_dim: int = 16,
        env_params_encoder_hidden_dims: tuple[int] | list[int] = (64, 32),
        env_params_latent_dim: int = 8,
        env_params_encoder_activation: str = "elu",
        shared_networks: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "EnvParamsConditionedEncoderActor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.current_obs_dim = int(current_obs_dim)
        self.env_params_dim = int(env_params_dim)
        self.env_params_latent_dim = int(env_params_latent_dim)
        self.state_dependent_std = state_dependent_std
        self.shared_networks = bool(shared_networks)

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        inferred_current_obs_dim = num_actor_obs - self.env_params_dim
        if inferred_current_obs_dim > 0 and inferred_current_obs_dim != self.current_obs_dim:
            print(
                f"[INFO]: Inferred encoded params-conditioned current_obs_dim={inferred_current_obs_dim} "
                f"from env observation shape; policy config had current_obs_dim={self.current_obs_dim}.",
                flush=True,
            )
            self.current_obs_dim = inferred_current_obs_dim
        expected_obs_dim = self.current_obs_dim + self.env_params_dim
        if num_actor_obs != expected_obs_dim:
            raise ValueError(
                f"Actor observation dim {num_actor_obs} != expected params-conditioned dim {expected_obs_dim}."
            )
        if num_critic_obs != expected_obs_dim:
            raise ValueError(
                f"Critic observation dim {num_critic_obs} != expected params-conditioned dim {expected_obs_dim}."
            )

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if self.shared_networks and critic_obs_normalization and actor_obs_normalization:
            self.critic_obs_normalizer = self.actor_obs_normalizer
        elif critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        encoder_kwargs = {
            "input_dim": self.env_params_dim,
            "output_dim": self.env_params_latent_dim,
            "hidden_dims": list(env_params_encoder_hidden_dims),
            "activation": env_params_encoder_activation,
        }
        self.actor_env_params_encoder = MLP(**encoder_kwargs)
        if self.shared_networks:
            self.critic_env_params_encoder = self.actor_env_params_encoder
        else:
            self.critic_env_params_encoder = MLP(**encoder_kwargs)

        head_input_dim = self.current_obs_dim + self.env_params_latent_dim
        if self.shared_networks:
            if list(actor_hidden_dims) != list(critic_hidden_dims):
                print(
                    "[WARN]: shared_networks=True uses actor_hidden_dims as the shared trunk; "
                    "critic_hidden_dims is ignored.",
                    flush=True,
                )
            self.actor, self.critic = make_shared_actor_critic_mlps(
                head_input_dim,
                num_actions,
                actor_hidden_dims,
                activation,
                state_dependent_std=state_dependent_std,
            )
        elif state_dependent_std:
            self.actor = MLP(head_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(head_input_dim, num_actions, actor_hidden_dims, activation)
        if not self.shared_networks:
            self.critic = MLP(head_input_dim, 1, critic_hidden_dims, activation)
        print(f"Actor env-param encoder: {self.actor_env_params_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic env-param encoder: {self.critic_env_params_encoder}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            init_state_dependent_actor_std(
                self.actor, num_actions, noise_std_type=self.noise_std_type, init_noise_std=init_noise_std
            )
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for obs_group in groups:
            assert len(obs[obs_group].shape) == 2, "EnvParamsConditionedEncoderActor expects flat observations."
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

    def _encode(self, obs: torch.Tensor, encoder: MLP) -> torch.Tensor:
        current_obs = obs[:, : self.current_obs_dim]
        env_params = obs[:, self.current_obs_dim : self.current_obs_dim + self.env_params_dim]
        return torch.cat((current_obs, encoder(env_params)), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        actor_input = self._encode(obs, self.actor_env_params_encoder)
        if self.state_dependent_std:
            mean_and_std = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = self.actor(actor_input)
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        actor_input = self._encode(obs, self.actor_env_params_encoder)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        critic_input = self._encode(obs, self.critic_env_params_encoder)
        return self.critic(critic_input)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization and self.critic_obs_normalizer is not self.actor_obs_normalizer:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def _source_current_obs_dim(self, source_dim: int) -> int | None:
        for current_dim in (self.current_obs_dim, 57):
            if source_dim in (
                current_dim,
                current_dim + self.env_params_dim,
                current_dim + self.env_params_latent_dim,
                current_dim + 64,
            ):
                return current_dim
            history_dim = source_dim - current_dim
            if history_dim > 0 and (history_dim % 24 == 0 or history_dim % 48 == 0):
                return current_dim
        return None

    def _adapt_raw_obs_layout_tensor(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        zero_missing_columns: bool,
    ) -> torch.Tensor | None:
        target_dim = self.current_obs_dim + self.env_params_dim
        if target.shape[-1] != target_dim:
            return None

        old_shared_dim = 51  # old layout up to and including c1/c2; old gate_idx was at column 51.
        old_env_params_start = 52
        source_dim = source.shape[-1]
        if source_dim < old_shared_dim:
            return None

        adapted = torch.zeros_like(target) if zero_missing_columns else target.clone()
        source_current_dim = self._source_current_obs_dim(source_dim)
        if source_current_dim is not None:
            copy_dim = min(source_current_dim, self.current_obs_dim, target_dim)
            adapted[..., :copy_dim] = source[..., :copy_dim]
            if source_dim == source_current_dim + self.env_params_dim:
                adapted[..., self.current_obs_dim : target_dim] = source[
                    ..., source_current_dim : source_current_dim + self.env_params_dim
                ]
            return adapted

        adapted[..., :old_shared_dim] = source[..., :old_shared_dim]
        if source_dim == old_env_params_start + self.env_params_dim:
            adapted[..., self.current_obs_dim : target_dim] = source[
                ..., old_env_params_start : old_env_params_start + self.env_params_dim
            ]
        return adapted

    def _adapt_head_input_tensor(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        target_dim = self.current_obs_dim + self.env_params_latent_dim
        if source.ndim != 2 or target.ndim != 2 or target.shape[-1] != target_dim or source.shape[0] != target.shape[0]:
            return None

        old_shared_dim = 51  # old layout up to and including c1/c2; old gate_idx was at column 51.
        source_dim = source.shape[-1]
        if source_dim < old_shared_dim:
            return None

        adapted = torch.zeros_like(target)
        source_current_dim = self._source_current_obs_dim(source_dim)
        if source_current_dim is not None:
            copy_dim = min(source_current_dim, self.current_obs_dim)
            adapted[..., :copy_dim] = source[..., :copy_dim]
            if source_dim == source_current_dim + self.env_params_latent_dim:
                adapted[..., self.current_obs_dim : target_dim] = source[
                    ..., source_current_dim : source_current_dim + self.env_params_latent_dim
                ]
        else:
            adapted[..., :old_shared_dim] = source[..., :old_shared_dim]
        return adapted

    def _adapt_input_tensor(self, key: str, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if key in ("actor.0.weight", "critic.0.weight"):
            return self._adapt_head_input_tensor(source, target)

        if (
            key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer."))
            and key.rsplit(".", 1)[-1] in {"_mean", "_var", "_std"}
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0] == 1
        ):
            return self._adapt_raw_obs_layout_tensor(source, target, zero_missing_columns=False)

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
                "[INFO]: Warm-started EnvParamsConditionedEncoderActor from a checkpoint with a different "
                f"observation/model layout; adapted {len(adapted_keys)} tensors and zero-initialized new "
                "cClose/cClose1 inputs.",
                flush=True,
            )
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            suffix = " ..." if len(skipped_keys) > 8 else ""
            print(f"[WARN]: Skipped {len(skipped_keys)} incompatible checkpoint tensors: {preview}{suffix}", flush=True)
        return False
