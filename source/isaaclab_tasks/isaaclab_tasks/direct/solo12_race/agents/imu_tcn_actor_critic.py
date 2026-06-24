# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.networks import EmpiricalNormalization, MLP
from tensordict import TensorDict
from torch.distributions import Normal

from .shared_actor_critic import init_state_dependent_actor_std, make_shared_actor_critic_mlps


class CausalConv1d(nn.Module):
    """1D causal convolution using left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class FootImuTcnEncoder(nn.Module):
    """Encode a flattened foot-IMU history into a latent vector."""

    def __init__(
        self,
        *,
        history_len: int,
        imu_dim: int,
        channels: int,
        latent_dim: int,
        kernel_size: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.imu_dim = imu_dim
        self.activation = activation

        self.conv_stack = nn.Sequential(
            CausalConv1d(imu_dim, channels, kernel_size=kernel_size, dilation=1, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1, stride=2),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=2, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1, stride=2),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=4, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1, stride=2),
            self._make_activation(activation),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, imu_dim, history_len)
            conv_output_dim = int(self.conv_stack(dummy).numel())

        self.to_latent = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_output_dim, latent_dim),
            nn.Tanh(),
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "elu":
            return nn.ELU()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported TCN activation: {name}")

    def forward(self, flat_history: torch.Tensor) -> torch.Tensor:
        history = flat_history.reshape(flat_history.shape[0], self.history_len, self.imu_dim)
        history = history.transpose(1, 2)
        return self.to_latent(self.conv_stack(history))


class ActorCriticFootImuTcn(nn.Module):
    """RSL-RL actor-critic with a TCN encoder over foot-IMU history.

    The env exposes a flat policy observation:
        [current proprio/race obs, flattened IMU history]

    The actor and critic each encode the IMU history with their own TCN and then feed
    [current obs, imu latent] through the usual MLP heads. Hidden dimensions are kept
    configurable so the IMU policy can reuse the baseline [256, 128, 64] head.
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
        history_len: int | None = None,
        history_dim: int | None = None,
        history_name: str = "IMU",
        imu_history_len: int = 20,
        imu_dim: int = 24,
        tcn_channels: int = 32,
        tcn_latent_dim: int = 64,
        tcn_kernel_size: int = 5,
        tcn_activation: str = "elu",
        shared_networks: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticFootImuTcn.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if history_len is not None:
            imu_history_len = history_len
        if history_dim is not None:
            imu_dim = history_dim

        self.obs_groups = obs_groups
        self.current_obs_dim = current_obs_dim
        self.history_name = history_name
        self.imu_history_len = imu_history_len
        self.imu_dim = imu_dim
        self.imu_history_flat_dim = imu_history_len * imu_dim
        self.tcn_latent_dim = tcn_latent_dim
        self.state_dependent_std = state_dependent_std
        self.shared_networks = bool(shared_networks)

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        inferred_current_obs_dim = num_actor_obs - self.imu_history_flat_dim
        if inferred_current_obs_dim > 0 and inferred_current_obs_dim != self.current_obs_dim:
            print(
                f"[INFO]: Inferred {history_name} TCN current_obs_dim={inferred_current_obs_dim} "
                f"from env observation shape; policy config had current_obs_dim={self.current_obs_dim}.",
                flush=True,
            )
            self.current_obs_dim = inferred_current_obs_dim
        expected_obs_dim = self.current_obs_dim + self.imu_history_flat_dim
        if num_actor_obs != expected_obs_dim:
            raise ValueError(
                f"Actor observation dim {num_actor_obs} != expected {history_name} TCN dim {expected_obs_dim}."
            )
        if num_critic_obs != expected_obs_dim:
            raise ValueError(
                f"Critic observation dim {num_critic_obs} != expected {history_name} TCN dim {expected_obs_dim}."
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
            "history_len": imu_history_len,
            "imu_dim": imu_dim,
            "channels": tcn_channels,
            "latent_dim": tcn_latent_dim,
            "kernel_size": tcn_kernel_size,
            "activation": tcn_activation,
        }
        self.actor_imu_encoder = FootImuTcnEncoder(**encoder_kwargs)
        if self.shared_networks:
            self.critic_imu_encoder = self.actor_imu_encoder
        else:
            self.critic_imu_encoder = FootImuTcnEncoder(**encoder_kwargs)

        head_input_dim = self.current_obs_dim + tcn_latent_dim
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
        print(f"Actor {history_name} TCN: {self.actor_imu_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic {history_name} TCN: {self.critic_imu_encoder}")
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
            assert len(obs[obs_group].shape) == 2, "ActorCriticFootImuTcn expects flat observations from the env."
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

    def _encode(self, obs: torch.Tensor, encoder: FootImuTcnEncoder) -> torch.Tensor:
        current_obs = obs[:, : self.current_obs_dim]
        imu_history = obs[:, self.current_obs_dim :]
        return torch.cat((current_obs, encoder(imu_history)), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        actor_input = self._encode(obs, self.actor_imu_encoder)
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
        actor_input = self._encode(obs, self.actor_imu_encoder)
        if self.state_dependent_std:
            return self.actor(actor_input)[..., 0, :]
        return self.actor(actor_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        critic_input = self._encode(obs, self.critic_imu_encoder)
        return self.critic(critic_input)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization and self.critic_obs_normalizer is not self.actor_obs_normalizer:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def _source_current_obs_dim(self, source_dim: int, trailing_dim: int) -> int | None:
        for current_dim in (self.current_obs_dim, 57, 52):
            if source_dim == current_dim + trailing_dim:
                return current_dim
        return None

    @staticmethod
    def _copy_current_obs_columns(adapted: torch.Tensor, source: torch.Tensor, source_current_dim: int) -> None:
        old_shared_dim = 51  # old layout up to and including c1/c2; old gate_idx was at column 51.
        copy_dim = old_shared_dim if source_current_dim == 52 else source_current_dim
        adapted[..., :copy_dim] = source[..., :copy_dim]

    def _adapt_head_input_tensor(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        target_dim = self.current_obs_dim + self.tcn_latent_dim
        if source.ndim != 2 or target.ndim != 2 or target.shape[-1] != target_dim or source.shape[0] != target.shape[0]:
            return None

        source_current_dim = self._source_current_obs_dim(source.shape[-1], self.tcn_latent_dim)
        if source_current_dim is None:
            return None

        adapted = torch.zeros_like(target)
        self._copy_current_obs_columns(adapted, source, source_current_dim)
        adapted[..., self.current_obs_dim : target_dim] = source[
            ..., source_current_dim : source_current_dim + self.tcn_latent_dim
        ]
        return adapted

    def _adapt_raw_obs_tensor(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        zero_missing_columns: bool,
    ) -> torch.Tensor | None:
        target_dim = self.current_obs_dim + self.imu_history_flat_dim
        if target.shape[-1] != target_dim:
            return None

        source_current_dim = self._source_current_obs_dim(source.shape[-1], self.imu_history_flat_dim)
        if source_current_dim is None:
            return None

        adapted = torch.zeros_like(target) if zero_missing_columns else target.clone()
        self._copy_current_obs_columns(adapted, source, source_current_dim)
        adapted[..., self.current_obs_dim : target_dim] = source[
            ..., source_current_dim : source_current_dim + self.imu_history_flat_dim
        ]
        return adapted

    def _adapt_input_tensor(self, key: str, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if key.startswith(("actor.", "critic.")) and key.endswith(".weight"):
            return self._adapt_head_input_tensor(source, target)

        if (
            key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer."))
            and key.rsplit(".", 1)[-1] in {"_mean", "_var", "_std"}
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0] == 1
        ):
            return self._adapt_raw_obs_tensor(source, target, zero_missing_columns=False)

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
                f"[INFO]: Warm-started ActorCriticFootImuTcn from an older {self.history_name} checkpoint; "
                f"adapted {len(adapted_keys)} tensors, preserved the history encoder/latent columns, and "
                "zero-initialized new cClose/cClose1 inputs.",
                flush=True,
            )
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            suffix = " ..." if len(skipped_keys) > 8 else ""
            print(f"[WARN]: Skipped {len(skipped_keys)} incompatible checkpoint tensors: {preview}{suffix}", flush=True)
        return False
