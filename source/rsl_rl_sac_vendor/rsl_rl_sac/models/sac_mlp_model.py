# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl_sac.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl_sac.utils import unpad_trajectories

from .mlp_model import MLPModel


class SACActorModel(MLPModel):
    """SAC actor model with Tanh-squashed Gaussian output distribution.

    Inherits observation-group resolution, empirical normalization, and ``get_latent()`` from :class:`MLPModel`.
    Overrides the forward pass to always produce Tanh-squashed, scaled actions and provides
    ``sample_action_logp()`` for training with the corrected log-probability.

    .. note::
        TODO (future): Add support for recurrent SAC models (e.g., ``SACActorRNNModel(RNNModel)``)
        analogous to how PPO has both ``MLPModel`` and ``RNNModel``. This would require adding
        ``masks``/``hidden_state`` handling and extending the replay buffer to store hidden states.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        state_dependent_std: bool = True,
        layer_norm: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        **kwargs,
    ) -> None:
        """Initialize the SAC actor model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor").
            output_dim: Dimension of the action space.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize observations.
            init_noise_std: Initial standard deviation for the log-std output head.
            state_dependent_std: If True, predict log standard deviation from the
                actor network. If False, learn one state-independent log-standard-
                deviation parameter per action.
            layer_norm: Whether to apply layer normalization in MLP hidden layers.
            log_std_min: Minimum value for log standard deviation clamping.
            log_std_max: Maximum value for log standard deviation clamping.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            stochastic=True,
            init_noise_std=init_noise_std,
            noise_std_type="log",
            state_dependent_std=state_dependent_std,
            layer_norm=layer_norm,
        )

        self.output_dim = output_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Custom actor weight initialization so that initial actions are near zero.
        # This overrides the parent's default init for state_dependent_std and is critical
        # for SAC stability — it prevents the initial policy from saturating the Tanh.
        last_linear = None
        for module in reversed(self.mlp):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is not None and self.state_dependent_std:
            # Mean output head → near-zero initial actions
            torch.nn.init.normal_(last_linear.weight[:output_dim], mean=0.0, std=1e-3)
            torch.nn.init.zeros_(last_linear.bias[:output_dim])
            # Log-std output head → log(init_noise_std)
            torch.nn.init.zeros_(last_linear.weight[output_dim:])
            torch.nn.init.constant_(last_linear.bias[output_dim:], torch.log(torch.tensor(init_noise_std + 1e-7)))
        elif last_linear is not None:
            # State-independent log_std is initialized by MLPModel. Initialize the
            # complete (mean-only) output layer with the paper's mean-head scheme.
            torch.nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
            torch.nn.init.zeros_(last_linear.bias)

        # Precomputed action scaling buffers — populated by SAC.construct_algorithm()
        # after construction via .copy_(). Using register_buffer ensures .to(device) moves them.
        self.register_buffer("action_bias", torch.zeros(output_dim))
        self.register_buffer("action_range", torch.ones(output_dim))
        self.register_buffer("log_action_range", torch.zeros(1))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning Tanh-squashed, scaled actions.

        Args:
            obs: Observation dictionary.
            masks: Optional masks for unpadding trajectories (unused for non-recurrent).
            hidden_state: Optional hidden state (unused for non-recurrent).
            stochastic_output: If True, sample from the distribution; otherwise use the mean.
            actions: Unused. Accepted for interface compatibility with MLPModel.

        Returns:
            Scaled actions after Tanh squashing.
        """
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        self._update_distribution(latent)
        if stochastic_output:
            x_t = self.distribution.rsample()
        else:
            x_t = self.distribution.mean
        return self._squash_and_scale(x_t)

    def sample_action_logp(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its corrected log-probability.

        The log-probability includes corrections for Tanh squashing and action scaling.

        Args:
            obs: Observation dictionary.

        Returns:
            Tuple of (scaled_action, log_prob).
        """
        latent = self.get_latent(obs)
        self._update_distribution(latent)
        x_t = self.distribution.rsample()
        tanh_x = torch.tanh(x_t)
        action = self.action_range * tanh_x + self.action_bias

        # Log-probability with Tanh Jacobian correction and action scale correction
        log_prob = self.distribution.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - tanh_x.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob -= self.log_action_range

        return action, log_prob

    def _update_distribution(self, latent: torch.Tensor) -> None:
        """Update the Gaussian distribution with log-std clamping."""
        output = self.mlp(latent)
        if self.state_dependent_std:
            mean, log_std = torch.unbind(output, dim=-2)
        else:
            mean = output
            log_std = self.log_std.expand_as(mean)
        std = log_std.clamp(self.log_std_min, self.log_std_max).exp()
        self.distribution = Normal(mean, std)

    def _squash_and_scale(self, x_t: torch.Tensor) -> torch.Tensor:
        """Apply Tanh squashing and affine action scaling using precomputed buffers."""
        return self.action_range * torch.tanh(x_t) + self.action_bias

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchSACActorModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxSACActorModel(self, verbose)


class SACCriticModel(MLPModel):
    """SAC critic model with twin Q-networks and frozen target networks.

    Inherits observation-group resolution, empirical normalization, and ``update_normalization()``
    from :class:`MLPModel`. Overrides the MLP with twin Q-networks that take concatenated
    (observation, action) inputs.

    Target networks are frozen copies (``requires_grad=False``) of the online Q-networks.
    They are never updated by gradient descent — only by Polyak averaging via
    ``soft_update_target_networks(tau)``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        obs_normalization: bool = False,
        num_actions: int = 0,
        layer_norm: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the SAC critic model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "critic").
            output_dim: Dimension of the Q-value output (typically 1).
            hidden_dims: Hidden dimensions of the Q-network MLPs.
            activation: Activation function of the MLPs.
            obs_normalization: Whether to normalize observations.
            num_actions: Dimension of the action space (concatenated with observations).
            layer_norm: Whether to apply layer normalization in MLP hidden layers.
        """
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            stochastic=False,
            layer_norm=layer_norm,
        )

        self.num_actions = num_actions

        # Override parent's MLP — critic input is obs_dim + num_actions
        q_input_dim = self.obs_dim + num_actions
        self.mlp = None  # type: ignore[assignment]

        # Twin Q-networks
        self.critic1 = MLP(q_input_dim, output_dim, hidden_dims, activation, layer_norm=layer_norm)
        self.critic2 = MLP(q_input_dim, output_dim, hidden_dims, activation, layer_norm=layer_norm)

        # Frozen target networks — never updated by gradient descent
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning Q1 value.

        Args:
            obs: Observation dictionary.
            masks: Optional masks for unpadding trajectories.
            hidden_state: Optional hidden state (unused for non-recurrent).
            stochastic_output: Unused for the critic.
            actions: Action tensor to concatenate with observations.

        Returns:
            Q1 value estimate.
        """
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        q_input = torch.cat([latent, actions], dim=-1)
        return self.critic1(q_input)

    def evaluate_all_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1 and Q2 for the given observations and actions.

        Args:
            obs: Observation dictionary.
            actions: Action tensor.

        Returns:
            Tuple of (Q1, Q2) value estimates.
        """
        latent = self.get_latent(obs)
        latent = torch.cat([latent, actions], dim=-1)
        return self.critic1(latent), self.critic2(latent)

    def evaluate_all_target_q(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute target Q1 and target Q2 for the given observations and actions.

        Uses the frozen target networks (never updated by gradient descent).

        Args:
            obs: Observation dictionary.
            actions: Action tensor.

        Returns:
            Tuple of (Q1_target, Q2_target) value estimates.
        """
        latent = self.get_latent(obs)
        latent = torch.cat([latent, actions], dim=-1)
        return self.critic1_target(latent), self.critic2_target(latent)

    def init_target_networks(self) -> None:
        """Initialize the target networks with the current critic network parameters."""
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def soft_update_target_networks(self, tau: float) -> None:
        """Soft-update the target networks using Polyak averaging.

        New target parameters are computed as: ``target = tau * online + (1 - tau) * target``.

        Args:
            tau: Interpolation parameter for soft updates.
        """
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


##############################################
# Export helpers — JIT and ONNX for SAC Actor
##############################################


class _TorchSACActorModel(nn.Module):
    """Exportable SAC actor model for JIT.

    Includes obs normalization, MLP forward, Tanh squashing, and action scaling.
    """

    def __init__(self, model: SACActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std
        self.action_bias = model.action_bias.clone()
        self.action_range = model.action_range.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        output = self.mlp(x)
        mean = output[..., 0, :] if self.state_dependent_std else output
        return self.action_range * torch.tanh(mean) + self.action_bias

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxSACActorModel(nn.Module):
    """Exportable SAC actor model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: SACActorModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.state_dependent_std = model.state_dependent_std
        self.register_buffer("action_bias", model.action_bias.clone())
        self.register_buffer("action_range", model.action_range.clone())
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        output = self.mlp(x)
        mean = output[..., 0, :] if self.state_dependent_std else output
        return self.action_range * torch.tanh(mean) + self.action_bias

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
