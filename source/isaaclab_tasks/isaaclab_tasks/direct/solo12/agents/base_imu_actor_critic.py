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


class BaseImuTcnEncoder(nn.Module):
    """Encode base-IMU/proprio/action history into a latent vector."""

    def __init__(
        self,
        *,
        history_len: int,
        sample_dim: int,
        channels: int,
        latent_dim: int,
        kernel_size: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.sample_dim = int(sample_dim)

        self.conv_stack = nn.Sequential(
            CausalConv1d(sample_dim, channels, kernel_size=kernel_size, dilation=1, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=1, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=2, stride=1),
            self._make_activation(activation),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=4, stride=1),
            self._make_activation(activation),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, sample_dim, history_len)
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
        history = flat_history.reshape(flat_history.shape[0], self.history_len, self.sample_dim)
        return self.to_latent(self.conv_stack(history.transpose(1, 2)))


class _GaussianActorMixin:
    is_recurrent: bool = False

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

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for obs_group in groups:
            assert len(obs[obs_group].shape) == 2, "Solo12 base-IMU models expect flat observations."
            dim += obs[obs_group].shape[-1]
        return dim

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def _set_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)


class Solo12BaseImuTeacherActorCritic(nn.Module, _GaussianActorMixin):
    """Teacher policy: privileged state -> MLP encoder z, then [z, command] -> actor."""

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
        teacher_encoder_obs_dim: int = 45,
        teacher_latent_dim: int = 32,
        teacher_encoder_hidden_dims: tuple[int] | list[int] = (256, 128, 64),
        teacher_encoder_activation: str = "elu",
        command_dim: int = 3,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Solo12BaseImuTeacherActorCritic.__init__ got unexpected arguments, ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise ValueError("Solo12BaseImuTeacherActorCritic does not support state_dependent_std yet.")
        super().__init__()
        self.obs_groups = obs_groups
        self.teacher_encoder_obs_dim = int(teacher_encoder_obs_dim)
        self.teacher_latent_dim = int(teacher_latent_dim)
        self.command_dim = int(command_dim)

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        expected_dim = self.teacher_encoder_obs_dim + self.command_dim
        if num_actor_obs != expected_dim:
            raise ValueError(f"Actor obs dim {num_actor_obs} != expected teacher dim {expected_dim}.")
        if num_critic_obs != expected_dim:
            raise ValueError(f"Critic obs dim {num_critic_obs} != expected teacher dim {expected_dim}.")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        self.teacher_encoder = MLP(
            self.teacher_encoder_obs_dim,
            self.teacher_latent_dim,
            list(teacher_encoder_hidden_dims),
            teacher_encoder_activation,
        )
        self.actor = MLP(self.teacher_latent_dim + self.command_dim, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Teacher encoder: {self.teacher_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def encode_teacher(self, obs: torch.Tensor) -> torch.Tensor:
        teacher_input = obs[:, : self.teacher_encoder_obs_dim]
        return self.teacher_encoder(teacher_input)

    def _actor_input(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encode_teacher(obs)
        command = obs[:, self.teacher_encoder_obs_dim : self.teacher_encoder_obs_dim + self.command_dim]
        return torch.cat((z, command), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._set_distribution(self.actor(self._actor_input(actor_obs)))

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(self._actor_input(actor_obs))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


class Solo12BaseImuStudentActorCritic(nn.Module, _GaussianActorMixin):
    """Student policy: base-IMU/proprio/action history -> TCN z, then [z, command] -> actor."""

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
        history_len: int = 13,
        history_sample_dim: int = 42,
        teacher_critic_obs_dim: int = 48,
        command_dim: int = 3,
        tcn_channels: int = 64,
        tcn_latent_dim: int = 32,
        tcn_kernel_size: int = 5,
        tcn_activation: str = "elu",
        feed_history_encoding_to_critic: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "Solo12BaseImuStudentActorCritic.__init__ got unexpected arguments, ignored: "
                + str([key for key in kwargs])
            )
        if state_dependent_std:
            raise ValueError("Solo12BaseImuStudentActorCritic does not support state_dependent_std yet.")
        super().__init__()
        self.obs_groups = obs_groups
        self.history_len = int(history_len)
        self.history_sample_dim = int(history_sample_dim)
        self.history_flat_dim = self.history_len * self.history_sample_dim
        self.teacher_critic_obs_dim = int(teacher_critic_obs_dim)
        self.command_dim = int(command_dim)
        self.tcn_latent_dim = int(tcn_latent_dim)
        self.feed_history_encoding_to_critic = bool(feed_history_encoding_to_critic)

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        expected_actor_dim = self.history_flat_dim + self.command_dim
        expected_critic_dim = self.teacher_critic_obs_dim + (
            self.history_flat_dim if self.feed_history_encoding_to_critic else 0
        )
        if num_actor_obs != expected_actor_dim:
            raise ValueError(f"Actor obs dim {num_actor_obs} != expected student dim {expected_actor_dim}.")
        if num_critic_obs != expected_critic_dim:
            raise ValueError(f"Critic obs dim {num_critic_obs} != expected student critic dim {expected_critic_dim}.")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()

        encoder_kwargs = {
            "history_len": self.history_len,
            "sample_dim": self.history_sample_dim,
            "channels": tcn_channels,
            "latent_dim": self.tcn_latent_dim,
            "kernel_size": tcn_kernel_size,
            "activation": tcn_activation,
        }
        self.actor_history_encoder = BaseImuTcnEncoder(**encoder_kwargs)
        self.critic_history_encoder = BaseImuTcnEncoder(**encoder_kwargs) if self.feed_history_encoding_to_critic else None
        self.actor = MLP(self.tcn_latent_dim + self.command_dim, num_actions, actor_hidden_dims, activation)
        critic_input_dim = self.teacher_critic_obs_dim + (
            self.tcn_latent_dim if self.feed_history_encoding_to_critic else 0
        )
        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)
        print(f"Actor base-IMU TCN: {self.actor_history_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def encode_history(self, flat_history: torch.Tensor, *, critic: bool = False) -> torch.Tensor:
        if critic and self.critic_history_encoder is not None:
            return self.critic_history_encoder(flat_history)
        return self.actor_history_encoder(flat_history)

    def _actor_input(self, obs: torch.Tensor) -> torch.Tensor:
        history = obs[:, : self.history_flat_dim]
        command = obs[:, self.history_flat_dim : self.history_flat_dim + self.command_dim]
        return torch.cat((self.encode_history(history), command), dim=-1)

    def _critic_input(self, obs: torch.Tensor) -> torch.Tensor:
        critic_base = obs[:, : self.teacher_critic_obs_dim]
        if not self.feed_history_encoding_to_critic:
            return critic_base
        history = obs[:, self.teacher_critic_obs_dim : self.teacher_critic_obs_dim + self.history_flat_dim]
        return torch.cat((critic_base, self.encode_history(history, critic=True)), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._set_distribution(self.actor(self._actor_input(actor_obs)))

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(self._actor_input(actor_obs))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(self._critic_input(critic_obs))

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))
