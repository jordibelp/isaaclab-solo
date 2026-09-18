# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl_sac.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl_sac.utils import unpad_trajectories

from .mlp_model import MLPModel

# An atom counts as "used" above this probability, matching the threshold the eval plots use.
_ACTIVE_ATOM_PROB = 0.1

#: Log keys for :meth:`SACCriticModel.distribution_stats`, in the order it stacks them.
DISTRIBUTION_STAT_NAMES = (
    "CriticDist/symlog_mean",
    "CriticDist/symlog_std_within_state",
    "CriticDist/symlog_std_across_states",
    "CriticDist/symlog_q05",
    "CriticDist/symlog_q50",
    "CriticDist/symlog_q95",
    "CriticDist/symlog_q05_q95_width",
    "CriticDist/active_atoms_p10",
    "CriticDist/effective_atoms",
    "CriticDist/edge_mass",
)


def symlog_distribution_stats(probs: torch.Tensor, atoms: torch.Tensor) -> dict[str, torch.Tensor]:
    """Describe where a categorical value distribution sits on its support, in symlog units.

    Training logs and the offline eval plots share this definition so their numbers are
    comparable. ``probs`` is ``(..., num_atoms)`` and ``atoms`` is the matching symlog grid;
    every returned tensor has shape ``probs.shape[:-1]``.

    * ``mean``/``std``: probability-weighted center and width of one state-action's distribution.
    * ``q05``/``q50``/``q95``: atom-resolution quantiles, robust when the mass is two-hot.
    * ``active_atoms``: atoms above :data:`_ACTIVE_ATOM_PROB`, the "which categories are used" count.
    * ``effective_atoms``: ``exp(entropy)``, the same question without a probability threshold.
    * ``edge_mass``: mass on the two outermost atoms, which is where a too-small support shows up.
    """
    mean = (probs * atoms).sum(-1)
    variance = (probs * (atoms - mean.unsqueeze(-1)).square()).sum(-1)
    levels = probs.new_tensor([0.05, 0.5, 0.95]).expand(*probs.shape[:-1], 3)
    indices = torch.searchsorted(probs.cumsum(-1).contiguous(), levels.contiguous())
    quantiles = atoms[indices.clamp(max=atoms.numel() - 1)]
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(-1)
    return {
        "mean": mean,
        "std": variance.sqrt(),
        "q05": quantiles[..., 0],
        "q50": quantiles[..., 1],
        "q95": quantiles[..., 2],
        "q05_q95_width": quantiles[..., 2] - quantiles[..., 0],
        "active_atoms": (probs > _ACTIVE_ATOM_PROB).sum(-1).float(),
        "effective_atoms": entropy.exp(),
        "edge_mass": probs[..., 0] + probs[..., -1],
    }


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

    @torch.no_grad()
    def initialize_mean_head_for_action(self, target_action: torch.Tensor) -> None:
        """Center the initial latent Gaussian on a requested scaled action.

        For ``action = action_range * tanh(latent) + action_bias``, the
        required latent mean is ``atanh((target_action - action_bias) /
        action_range)``. Only the mean-head bias is changed; its small random
        weights and the log-standard-deviation initialization are preserved.
        """
        target_action = torch.as_tensor(target_action, device=self.action_bias.device, dtype=self.action_bias.dtype)
        if target_action.shape != self.action_bias.shape:
            raise ValueError(
                f"Expected target_action shape {tuple(self.action_bias.shape)}, got {tuple(target_action.shape)}."
            )
        if torch.any(self.action_range <= 0.0):
            raise ValueError("SAC action ranges must be strictly positive before centering the mean head.")

        normalized_target = (target_action - self.action_bias) / self.action_range
        if torch.any(normalized_target <= -1.0) or torch.any(normalized_target >= 1.0):
            raise ValueError("The requested initial SAC action must lie strictly inside every action bound.")
        latent_mean = torch.atanh(normalized_target)

        last_linear = next(module for module in reversed(self.mlp) if isinstance(module, nn.Linear))
        if self.state_dependent_std:
            last_linear.bias[: self.output_dim].copy_(latent_mean)
        else:
            last_linear.bias.copy_(latent_mean)

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
        return action, self._corrected_log_prob(x_t, tanh_x)

    def _corrected_log_prob(self, x_t: torch.Tensor, tanh_x: torch.Tensor) -> torch.Tensor:
        """Log-probability with Tanh Jacobian correction and action scale correction."""
        log_prob = self.distribution.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - tanh_x.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob - self.log_action_range

    def executed_action_logp(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of already-scaled actions under the last :meth:`forward` distribution.

        Used by evaluation tooling to recover SAC's entropy bonus for actions that were
        actually stepped, including deterministic ones. The pre-squash latent is recovered
        with ``atanh``, so precision degrades for actions pinned against their bounds.
        """
        tanh_x = ((actions - self.action_bias) / self.action_range).clamp(-1 + 1e-6, 1 - 1e-6)
        return self._corrected_log_prob(torch.atanh(tanh_x), tanh_x)

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
        distributional_critic_ce: bool = False,
        distributional_num_bins: int = 255,
        distributional_symlog_limit: float = 8.0,
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
            distributional_critic_ce: Fit scalar TD targets with symexp two-hot cross entropy.
            distributional_num_bins: Odd number of categorical atoms, including zero.
            distributional_symlog_limit: Symmetric log-space bound for the raw-unit support.
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
        self.distributional_critic_ce = distributional_critic_ce
        if distributional_critic_ce:
            if output_dim != 1:
                raise ValueError("Two-hot SAC requires scalar Q-values (output_dim=1).")
            if distributional_num_bins < 3 or distributional_num_bins % 2 != 1:
                raise ValueError("distributional_num_bins must be odd and at least 3.")
            if not math.isfinite(distributional_symlog_limit) or not 0 < distributional_symlog_limit <= 80:
                raise ValueError("distributional_symlog_limit must be finite and in (0, 80] for float32.")
            # Build exact +/- pairs and an exact zero atom, avoiding linspace roundoff.
            positive = torch.linspace(0, distributional_symlog_limit, distributional_num_bins // 2 + 1).expm1()
            self.register_buffer("value_support", torch.cat((-positive[1:].flip(0), positive)))

        # Override parent's MLP — critic input is obs_dim + num_actions
        q_input_dim = self.obs_dim + num_actions
        self.mlp = None  # type: ignore[assignment]

        # Twin Q-networks
        head_dim = distributional_num_bins if distributional_critic_ce else output_dim
        self.critic1 = MLP(q_input_dim, head_dim, hidden_dims, activation, layer_norm=layer_norm)
        self.critic2 = MLP(q_input_dim, head_dim, hidden_dims, activation, layer_norm=layer_norm)
        if distributional_critic_ce:
            # Dreamer initialization: random logits on this wide support imply enormous Q.
            for network in (self.critic1, self.critic2):
                nn.init.zeros_(network[-1].weight)
                nn.init.zeros_(network[-1].bias)

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
        return self.q_from_output(self.critic1(q_input))

    def q_from_output(self, output: torch.Tensor) -> torch.Tensor:
        """Decode the raw-unit mean, not symexp(E[symlog(Q)]); keep action gradients."""
        if not self.distributional_critic_ce:
            return output
        probs = output.float().softmax(dim=-1)
        mid = self.value_support.numel() // 2
        # Sum opposite atoms together so a symmetric distribution gives exactly zero,
        # even with wide supports. Algebraically this is E[B]. The default limit
        # is 8, not Dreamer's 20: SAC also differentiates this mean through actions,
        # where enormous +/- atoms cause float32 cancellation in head backprop.
        return ((probs[..., mid + 1 :] - probs[..., :mid].flip(-1)) * self.value_support[mid + 1 :]).sum(
            dim=-1, keepdim=True
        )

    def two_hot(self, targets: torch.Tensor) -> torch.Tensor:
        """Project detached scalar targets onto adjacent atoms in *raw reward units*.

        Preserves the clipped target's expectation. This is a scalar Bellman-target
        regression, not a C51 distributional Bellman projection.
        """
        support = self.value_support
        targets = targets.detach().to(dtype=support.dtype).clamp(support[0], support[-1])
        upper = torch.searchsorted(support, targets.contiguous()).clamp(1, support.numel() - 1)
        lower = upper - 1
        upper_weight = (targets - support[lower]) / (support[upper] - support[lower])
        labels = targets.new_zeros(*targets.shape[:-1], support.numel())
        labels.scatter_add_(-1, lower, 1 - upper_weight)
        labels.scatter_add_(-1, upper, upper_weight)
        return labels

    def critic_outputs(self, obs: TensorDict, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw twin-critic head outputs: scalar Q values, or categorical logits under CE."""
        latent = torch.cat([self.get_latent(obs), actions], dim=-1)
        return self.critic1(latent), self.critic2(latent)

    def losses_from_outputs(
        self, output1: torch.Tensor, output2: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin critic losses; the default branch retains the original scalar MSE."""
        if not self.distributional_critic_ce:
            return nn.functional.mse_loss(output1, targets), nn.functional.mse_loss(output2, targets)
        labels = self.two_hot(targets)
        loss1 = -(labels * output1.float().log_softmax(-1)).sum(-1).mean()
        loss2 = -(labels * output2.float().log_softmax(-1)).sum(-1).mean()
        return loss1, loss2

    def td_losses(
        self, obs: TensorDict, actions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin critic losses; the default branch retains the original scalar MSE."""
        return self.losses_from_outputs(*self.critic_outputs(obs, actions), targets)

    def support_symlog(self) -> torch.Tensor:
        """Support atoms in the symlog units the ``distributional_symlog_limit`` flag sets."""
        return self.value_support.sign() * self.value_support.abs().log1p()

    @torch.no_grad()
    def distribution_stats(self, output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
        """Summarize which atoms the twin-critic mixture occupies, in symlog units.

        The mixture ``(p1 + p2) / 2`` is used, so critic disagreement widens the reported
        spread exactly as it widens the range of encodable values. These describe the
        categorical *representation*, not a calibrated return distribution: the critics are
        trained on scalar Bellman targets, so the width is fit error plus target spread, not
        an uncertainty estimate.
        """
        probs = 0.5 * (output1.float().softmax(-1) + output2.float().softmax(-1))
        per_sample = symlog_distribution_stats(probs, self.support_symlog())
        center = per_sample["mean"]
        return torch.stack([
            center.mean(),
            per_sample["std"].mean(),
            center.std(unbiased=False),
            per_sample["q05"].mean(),
            per_sample["q50"].mean(),
            per_sample["q95"].mean(),
            per_sample["q05_q95_width"].mean(),
            per_sample["active_atoms"].mean(),
            per_sample["effective_atoms"].mean(),
            per_sample["edge_mass"].mean(),
        ])

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
        return self.q_from_output(self.critic1(latent)), self.q_from_output(self.critic2(latent))

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
        return self.q_from_output(self.critic1_target(latent)), self.q_from_output(self.critic2_target(latent))

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
