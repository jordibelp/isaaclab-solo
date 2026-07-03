# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Recurrent State-Space Model with a block-diagonal GRU core (DreamerV3).

The deterministic recurrence uses DreamerV3's blocked GRU: inputs are projected
and normalized, split into ``blocks`` independent groups, and combined with a
block-diagonal linear layer.  Stochastic states are discrete (one-hot
categorical) with a unimix mixture and straight-through gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import distributions as dists
from .networks import BlockLinear, rms_norm, weight_init_


@dataclass
class RSSMState:
    stoch: torch.Tensor  # (B, stoch, discrete)
    deter: torch.Tensor  # (B, deter)


class BlockGRU(nn.Module):
    """DreamerV3 blocked deterministic transition (GRU-style gating)."""

    def __init__(self, deter: int, flat_stoch: int, action_dim: int, hidden: int, blocks: int, dyn_layers: int):
        super().__init__()
        self.blocks = int(blocks)
        self._in_deter = nn.Sequential(nn.Linear(deter, hidden), rms_norm(hidden), nn.SiLU())
        self._in_stoch = nn.Sequential(nn.Linear(flat_stoch, hidden), rms_norm(hidden), nn.SiLU())
        self._in_action = nn.Sequential(nn.Linear(action_dim, hidden), rms_norm(hidden), nn.SiLU())

        self._hidden = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(int(dyn_layers)):
            self._hidden.add_module(f"lin_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._hidden.add_module(f"norm_{i}", rms_norm(deter))
            self._hidden.add_module(f"act_{i}", nn.SiLU())
            in_ch = deter
        self._gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.apply(weight_init_)

    def _to_groups(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _from_groups(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch: torch.Tensor, deter: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch = action.shape[0]
        stoch = stoch.reshape(batch, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()

        x = torch.cat([self._in_deter(deter), self._in_stoch(stoch), self._in_action(action)], dim=-1)
        # Broadcast the shared inputs across blocks and fuse with per-block deter.
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self._from_groups(torch.cat([self._to_groups(deter), x], dim=-1))

        x = self._hidden(x)
        x = self._gru(x)

        reset, cand, update = (self._from_groups(g) for g in torch.chunk(self._to_groups(x), 3, dim=-1))
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter


class RSSM(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        action_dim: int,
        deter: int,
        stoch: int,
        discrete: int,
        hidden: int,
        blocks: int,
        obs_layers: int,
        img_layers: int,
        dyn_layers: int,
        unimix_ratio: float,
    ):
        super().__init__()
        self.deter = int(deter)
        self.stoch = int(stoch)
        self.discrete = int(discrete)
        self.flat_stoch = self.stoch * self.discrete
        self.feat_size = self.flat_stoch + self.deter
        self.unimix_ratio = float(unimix_ratio)

        self._gru = BlockGRU(self.deter, self.flat_stoch, action_dim, hidden, blocks, dyn_layers)

        # Posterior: q(stoch | deter, embed).
        obs_hidden = [hidden] * int(obs_layers)
        self._obs_net = _logit_mlp(self.deter + embed_dim, obs_hidden, self.flat_stoch)
        # Prior: p(stoch | deter).
        img_hidden = [hidden] * int(img_layers)
        self._img_net = _logit_mlp(self.deter, img_hidden, self.flat_stoch)

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        stoch = torch.zeros(batch_size, self.stoch, self.discrete, device=device)
        deter = torch.zeros(batch_size, self.deter, device=device)
        return RSSMState(stoch=stoch, deter=deter)

    def get_dist(self, logit: torch.Tensor) -> dists.OneHotDist:
        return dists.OneHotDist(logit, unimix_ratio=self.unimix_ratio)

    def _to_logits(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(*flat.shape[:-1], self.stoch, self.discrete)

    def obs_step(
        self, state: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor, is_first: torch.Tensor
    ) -> tuple[RSSMState, torch.Tensor]:
        """Posterior step conditioned on an observation embedding."""
        keep = (~is_first).to(embed.dtype)
        stoch = state.stoch * keep.reshape(-1, 1, 1)
        deter = state.deter * keep.reshape(-1, 1)
        prev_action = prev_action * keep.reshape(-1, 1)

        deter = self._gru(stoch, deter, prev_action)
        logit = self._to_logits(self._obs_net(torch.cat([deter, embed], dim=-1)))
        stoch = self.get_dist(logit).rsample()
        return RSSMState(stoch=stoch, deter=deter), logit

    def img_step(self, state: RSSMState, prev_action: torch.Tensor) -> tuple[RSSMState, torch.Tensor]:
        """Prior step (imagination, no observation)."""
        deter = self._gru(state.stoch, state.deter, prev_action)
        logit = self._to_logits(self._img_net(deter))
        stoch = self.get_dist(logit).rsample()
        return RSSMState(stoch=stoch, deter=deter), logit

    def prior_logit(self, deter: torch.Tensor) -> torch.Tensor:
        return self._to_logits(self._img_net(deter))

    def observe(
        self, embed: torch.Tensor, actions: torch.Tensor, initial: RSSMState, is_first: torch.Tensor
    ) -> tuple[RSSMState, torch.Tensor]:
        """Roll out the posterior over a (B, T) sequence."""
        length = actions.shape[1]
        state = initial
        stochs, deters, logits = [], [], []
        for t in range(length):
            state, logit = self.obs_step(state, actions[:, t], embed[:, t], is_first[:, t])
            stochs.append(state.stoch)
            deters.append(state.deter)
            logits.append(logit)
        post = RSSMState(stoch=torch.stack(stochs, 1), deter=torch.stack(deters, 1))
        return post, torch.stack(logits, 1)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        stoch = state.stoch.reshape(*state.stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch, state.deter], dim=-1)

    def kl_loss(self, post_logit: torch.Tensor, prior_logit: torch.Tensor, free: float) -> tuple[torch.Tensor, torch.Tensor]:
        dyn = dists.categorical_kl(post_logit.detach(), prior_logit, self.unimix_ratio).sum(-1)
        rep = dists.categorical_kl(post_logit, prior_logit.detach(), self.unimix_ratio).sum(-1)
        return torch.clip(dyn, min=free), torch.clip(rep, min=free)


def _logit_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(dim, hidden))
        layers.append(rms_norm(hidden))
        layers.append(nn.SiLU())
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    net = nn.Sequential(*layers)
    net.apply(weight_init_)
    return net
