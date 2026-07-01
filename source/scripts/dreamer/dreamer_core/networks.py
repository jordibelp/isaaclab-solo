# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Network building blocks for the high-performance DreamerV3 core.

Proprioceptive-only: MLP encoder/decoder + symexp two-hot heads.  Uses RMSNorm
and SiLU (faster and more stable than LayerNorm), a block-diagonal linear layer
for the recurrent core, and DreamerV3's truncated-normal weight init.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init

from . import distributions as dists


def weight_init_(module: nn.Module, fan_type: str = "in") -> None:
    """DreamerV3 init: truncated normal scaled by fan, zero bias, unit norm gains."""
    if isinstance(module, nn.RMSNorm):
        if module.weight is not None:
            with torch.no_grad():
                module.weight.fill_(1.0)
        return
    weight = getattr(module, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    fan_in, fan_out = nn_init._calculate_fan_in_and_fan_out(weight)
    fan = {"avg": (fan_in + fan_out) / 2, "in": fan_in, "out": fan_out}[fan_type]
    std = 1.1368 * math.sqrt(1.0 / fan)
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


def rms_norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-4, dtype=torch.float32)


def make_mlp(in_dim: int, hidden_dims: list[int], out_dim: int | None = None) -> nn.Sequential:
    """[Linear, RMSNorm, SiLU] x len(hidden) (+ optional final Linear).

    Kept as a flat ``nn.Sequential`` so Continual-Backprop can discover the
    hidden feature groups.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(dim, hidden, bias=True))
        layers.append(rms_norm(hidden))
        layers.append(nn.SiLU())
        dim = hidden
    if out_dim is not None:
        layers.append(nn.Linear(dim, out_dim, bias=True))
    net = nn.Sequential(*layers)
    net.apply(weight_init_)
    return net


class BlockLinear(nn.Module):
    """Block-diagonal linear layer (DreamerV3 blocked recurrent core).

    Splits the feature dim into ``blocks`` independent groups, giving a
    block-diagonal weight matrix: fewer FLOPs/params than a dense layer at the
    same width, which is a large part of the RSSM speedup.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int):
        super().__init__()
        assert in_ch % blocks == 0 and out_ch % blocks == 0
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        # Layout (O/G, I/G, G) cooperates with torch fan-in/out init.
        self.weight = nn.Parameter(torch.empty(out_ch // blocks, in_ch // blocks, blocks))
        self.bias = nn.Parameter(torch.empty(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class TwoHotHead(nn.Module):
    """Symexp two-hot regression head used for reward, value and critic."""

    def __init__(self, in_dim: int, hidden_dims: list[int], num_bins: int, symlog_range: float, outscale: float = 0.0):
        super().__init__()
        self.num_bins = int(num_bins)
        self.net = make_mlp(in_dim, hidden_dims, self.num_bins)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.mul_(outscale)
            last.bias.zero_()
        self.register_buffer("bins", dists.make_symexp_bins(self.num_bins, symlog_range))

    def forward(self, features: torch.Tensor) -> dists.TwoHotSymexp:
        return dists.TwoHotSymexp(self.net(features), self.bins)

    def pred(self, features: torch.Tensor) -> torch.Tensor:
        return self(features).mean()


class MLPDecoderHead(nn.Module):
    """Symlog-MSE proprioceptive observation decoder."""

    def __init__(self, in_dim: int, hidden_dims: list[int], obs_dim: int):
        super().__init__()
        self.net = make_mlp(in_dim, hidden_dims, obs_dim)

    def forward(self, features: torch.Tensor) -> dists.SymlogMSE:
        return dists.SymlogMSE(self.net(features))


class ContinueHead(nn.Module):
    """Bernoulli continuation head (predicts 1 - is_terminal)."""

    def __init__(self, in_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = make_mlp(in_dim, hidden_dims, 1)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class Actor(nn.Module):
    """Bounded-normal continuous actor (DreamerV3)."""

    def __init__(self, in_dim: int, action_dim: int, hidden_dims: list[int], min_std: float, max_std: float,
                 outscale: float = 0.01):
        super().__init__()
        self.action_dim = int(action_dim)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.net = make_mlp(in_dim, hidden_dims, 2 * action_dim)
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.weight.mul_(outscale)

    def forward(self, features: torch.Tensor) -> dists.BoundedNormal:
        return dists.BoundedNormal(self.net(features), self.min_std, self.max_std)


class ReturnEMA(nn.Module):
    """EMA of return percentiles for actor-return normalization (DreamerV3)."""

    def __init__(self, rate: float, limit: float, low_pct: float, high_pct: float):
        super().__init__()
        self.rate = float(rate)
        self.limit = float(limit)
        self.register_buffer("_range", torch.tensor([low_pct / 100.0, high_pct / 100.0], dtype=torch.float32))
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def __call__(self, returns: torch.Tensor, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        flat = returns.detach().reshape(-1).float()
        if update and flat.numel() > 0:
            quantiles = torch.quantile(flat, self._range.to(flat.device))
            self.low.mul_(1.0 - self.rate).add_(quantiles[0], alpha=self.rate)
            self.high.mul_(1.0 - self.rate).add_(quantiles[1], alpha=self.rate)
        offset = self.low
        scale = (self.high - self.low).clamp_min(self.limit)
        return offset.detach(), scale.detach()
