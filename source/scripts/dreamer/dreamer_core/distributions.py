# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DreamerV3 output distributions and transforms.

These match the numerics used by DreamerV3 (danijar) and the efficient R2-Dreamer
reproduction: symlog/symexp transforms, unimix straight-through one-hot
categoricals, symexp two-hot regression for reward/value, and a bounded normal
actor distribution for continuous control.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import distributions as torchd


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_symexp_bins(num_bins: int, symlog_range: float, device: torch.device | None = None) -> torch.Tensor:
    """Bin centers spaced uniformly in symlog space (DreamerV3 two-hot support)."""
    if num_bins % 2 == 1:
        half = torch.linspace(-symlog_range, 0.0, (num_bins - 1) // 2 + 1, dtype=torch.float32, device=device)
        half = symexp(half)
        return torch.cat((half, -half[:-1].flip(0)), dim=0)
    half = torch.linspace(-symlog_range, 0.0, num_bins // 2, dtype=torch.float32, device=device)
    half = symexp(half)
    return torch.cat((half, -half.flip(0)), dim=0)


def unimix_logits(logits: torch.Tensor, unimix_ratio: float) -> torch.Tensor:
    """Blend a small uniform mixture into a categorical, returned as log-probs."""
    if unimix_ratio <= 0.0:
        return logits
    probs = F.softmax(logits.float(), dim=-1)
    uniform = unimix_ratio / probs.shape[-1]
    probs = probs * (1.0 - unimix_ratio) + uniform
    return torch.log(probs)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """Straight-through one-hot categorical with a uniform mixture (unimix)."""

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.0):
        logits = unimix_logits(logits.float(), unimix_ratio)
        super().__init__(logits=logits)

    @property
    def mode(self) -> torch.Tensor:
        mode = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[-1]).to(self.logits.dtype)
        # Straight-through: forward the hard mode, backprop through the logits.
        return mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=()) -> torch.Tensor:  # noqa: D401 - torch signature
        # Straight-through Gumbel-Softmax sample (hard forward, soft gradient).
        return F.gumbel_softmax(self.logits, tau=1.0, hard=True, dim=-1)


def categorical_kl(logits_left: torch.Tensor, logits_right: torch.Tensor, unimix_ratio: float = 0.0) -> torch.Tensor:
    """KL between the same unimixed categoricals used for RSSM sampling."""
    logits_left = unimix_logits(logits_left, unimix_ratio)
    logits_right = unimix_logits(logits_right, unimix_ratio)
    logp_left = torch.log_softmax(logits_left, dim=-1)
    logp_right = torch.log_softmax(logits_right, dim=-1)
    prob = torch.softmax(logits_left, dim=-1)
    return (prob * (logp_left - logp_right)).sum(-1)


class TwoHotSymexp:
    """Symexp two-hot categorical regression head (reward & value).

    Predicts a distribution over fixed symexp-spaced bins; the point estimate is
    the expected bin value.  The loss is cross-entropy against the two-hot
    encoding of the (symlog-warped) target.
    """

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor):
        self.logits = logits.float()
        self.bins = bins.to(device=logits.device, dtype=torch.float32)
        assert self.logits.shape[-1] == self.bins.shape[0]

    def mean(self) -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        # Pair the symmetric +/- bins before reducing (as r2dreamer does): the outer
        # bins reach +/-4.9e8, so a naive dot product leaves O(0.1-1) fp32 cancellation
        # noise in the expectation - larger than a typical per-step locomotion reward.
        n = self.bins.shape[0]
        weighted = probs * self.bins
        if n % 2 == 1:
            m = (n - 1) // 2
            paired = weighted[..., :m].flip(-1) + weighted[..., m + 1 :]
            return weighted[..., m] + paired.sum(-1)
        m = n // 2
        paired = weighted[..., :m].flip(-1) + weighted[..., m:]
        return paired.sum(-1)

    # Alias used interchangeably by the agent.
    mode = mean

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=torch.float32)
        bins = self.bins
        n = bins.shape[0]
        below = (torch.searchsorted(bins, target.detach(), right=True) - 1).clamp(0, n - 1)
        above = (below + 1).clamp(0, n - 1)
        equal = below == above
        dist_below = torch.where(equal, torch.ones_like(target), (bins[below] - target).abs())
        dist_above = torch.where(equal, torch.ones_like(target), (bins[above] - target).abs())
        total = dist_below + dist_above
        weight_below = dist_above / total
        weight_above = dist_below / total
        twohot = (
            F.one_hot(below, n).to(self.logits.dtype) * weight_below.unsqueeze(-1)
            + F.one_hot(above, n).to(self.logits.dtype) * weight_above.unsqueeze(-1)
        )
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (twohot * log_pred).sum(-1)


class SymlogMSE:
    """Gaussian-in-symlog-space observation head (proprioceptive reconstruction)."""

    def __init__(self, mode: torch.Tensor):
        self._mode = mode.float()

    def mode(self) -> torch.Tensor:  # noqa: D401
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Negative sum-of-squared error against the symlog-warped target.
        distance = (self._mode - symlog(value.float())) ** 2
        return -distance.sum(-1)


class BoundedNormal:
    """DreamerV3 continuous actor: tanh mean, sigmoid-bounded std, soft-clipped samples.

    The soft clip ``x / max(1, |x|)`` keeps actions in roughly ``[-1, 1]`` while
    remaining differentiable and, unlike a tanh squash, keeping a simple diagonal
    Gaussian log-prob (matching DreamerV3's `bounded_normal`).
    """

    def __init__(self, params: torch.Tensor, min_std: float, max_std: float):
        mean, std = torch.chunk(params.float(), 2, dim=-1)
        mean = torch.tanh(mean)
        std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
        self._dist = torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)

    @staticmethod
    def _soft_clip(x: torch.Tensor) -> torch.Tensor:
        return x / torch.clip(torch.abs(x), min=1.0).detach()

    @property
    def mode(self) -> torch.Tensor:
        return self._soft_clip(self._dist.mean)

    def rsample(self, sample_shape=()) -> torch.Tensor:
        return self._soft_clip(self._dist.rsample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(value)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()
