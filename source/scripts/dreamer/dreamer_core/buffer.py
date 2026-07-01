# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-resident sequence replay buffer with latent (replay-context) caching.

Transitions from all vectorized envs are stored in a single ring buffer that can
live on the GPU, so collection is a plain in-place tensor write with no
host<->device traffic (the main throughput win for Isaac Lab's on-GPU envs).

Each stored step also carries the RSSM posterior latent produced when the action
was chosen.  Sampling returns the latent at the first step of every sampled slice
as an RSSM warm-start ("replay context"), and :meth:`update_latents` writes the
freshly recomputed posteriors back so future samples start from an up-to-date
state instead of a cold zero state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SequenceReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        command_dim: int,
        action_dim: int,
        stoch: int,
        discrete: int,
        deter: int,
        device: torch.device,
        storage_device: torch.device | None = None,
        recent_fraction: float = 0.0,
    ):
        self.num_envs = int(num_envs)
        self.capacity_steps = max(2, int(capacity) // self.num_envs)
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.command_dim = int(command_dim)
        self.recent_fraction = float(recent_fraction)

        self.stoch_vars = int(stoch)
        self.discrete = int(discrete)
        cap, n = self.capacity_steps, self.num_envs
        sd = self.storage_device
        self.obs = torch.empty(cap, n, obs_dim, dtype=torch.float32, device=sd)
        self.command = torch.empty(cap, n, command_dim, dtype=torch.float32, device=sd)
        self.action = torch.empty(cap, n, action_dim, dtype=torch.float32, device=sd)
        self.reward = torch.empty(cap, n, dtype=torch.float32, device=sd)
        self.is_first = torch.empty(cap, n, dtype=torch.bool, device=sd)
        self.is_terminal = torch.empty(cap, n, dtype=torch.bool, device=sd)
        self.is_last = torch.empty(cap, n, dtype=torch.bool, device=sd)
        # Posterior latents are cached compactly: discrete stochs are one-hot, so we
        # keep only the argmax class index (int16); deter is kept in fp16.  This makes
        # the replay-context cache ~100x smaller (a few hundred MB for a 2M buffer).
        self.stoch_idx = torch.zeros(cap, n, stoch, dtype=torch.int16, device=sd)
        self.deter = torch.zeros(cap, n, deter, dtype=torch.float16, device=sd)

        self.pos = 0
        self.filled = 0
        self.total = 0

    # ------------------------------------------------------------------ add
    @torch.no_grad()
    def add(
        self,
        *,
        obs: torch.Tensor,
        command: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
        is_terminal: torch.Tensor,
        is_last: torch.Tensor,
        stoch: torch.Tensor,
        deter: torch.Tensor,
    ) -> None:
        i = self.pos
        self.obs[i].copy_(obs, non_blocking=True)
        if self.command_dim:
            self.command[i].copy_(command, non_blocking=True)
        self.action[i].copy_(action, non_blocking=True)
        self.reward[i].copy_(reward, non_blocking=True)
        self.is_first[i].copy_(is_first, non_blocking=True)
        self.is_terminal[i].copy_(is_terminal, non_blocking=True)
        self.is_last[i].copy_(is_last, non_blocking=True)
        self.stoch_idx[i].copy_(stoch.argmax(dim=-1).to(torch.int16), non_blocking=True)
        self.deter[i].copy_(deter.to(torch.float16), non_blocking=True)
        self.pos = (self.pos + 1) % self.capacity_steps
        self.filled = min(self.filled + 1, self.capacity_steps)
        self.total += self.num_envs

    def can_sample(self, seq_len: int, min_steps: int) -> bool:
        return self.total >= min_steps and self.filled >= seq_len

    @property
    def _oldest(self) -> int:
        return self.pos if self.filled == self.capacity_steps else 0

    # --------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Return a batch of ``seq_len``-long slices plus writeback indices.

        The returned dict holds tensors shaped ``(B, seq_len - 1, ...)`` (the last
        ``seq_len - 1`` steps of each slice) together with ``initial_stoch`` /
        ``initial_deter`` (the latent at slice start) and ``prev_action`` (the
        action one step back).  The second element is ``(phys_times, env_ids)``
        addressing the returned steps for :meth:`update_latents`.
        """
        max_start = self.filled - seq_len
        env_ids = torch.randint(0, self.num_envs, (batch_size,), device=self.storage_device)
        starts = torch.randint(0, max_start + 1, (batch_size,), device=self.storage_device)
        if self.recent_fraction > 0.0:
            n_recent = int(round(batch_size * self.recent_fraction))
            if n_recent > 0:
                # Bias a fraction of slices toward the most recent steps.
                lo = max(0, max_start - seq_len + 1)
                starts[:n_recent] = torch.randint(lo, max_start + 1, (n_recent,), device=self.storage_device)

        offsets = torch.arange(seq_len, device=self.storage_device)
        logical = starts[:, None] + offsets[None, :]  # (B, L)
        phys = (self._oldest + logical) % self.capacity_steps
        env_col = env_ids[:, None]

        def gather(t: torch.Tensor) -> torch.Tensor:
            return t[phys, env_col].to(self.device, non_blocking=True)

        obs = gather(self.obs)
        command = gather(self.command) if self.command_dim else obs.new_zeros((batch_size, seq_len, 0))
        action = gather(self.action)
        reward = gather(self.reward)
        is_first = gather(self.is_first)
        is_terminal = gather(self.is_terminal)
        is_last = gather(self.is_last)
        stoch0_idx = self.stoch_idx[phys[:, 0], env_ids].to(self.device, non_blocking=True).long()
        stoch0 = F.one_hot(stoch0_idx, self.discrete).float()
        deter0 = self.deter[phys[:, 0], env_ids].to(self.device, non_blocking=True).float()

        batch = {
            "obs": obs[:, 1:],
            "command": command[:, 1:] if self.command_dim else command[:, 1:],
            "prev_action": action[:, :-1],
            "reward": reward[:, 1:],
            "is_first": is_first[:, 1:],
            "is_terminal": is_terminal[:, 1:],
            "is_last": is_last[:, 1:],
            "initial_stoch": stoch0,
            "initial_deter": deter0,
        }
        writeback = (phys[:, 1:], env_ids)
        return batch, writeback

    @torch.no_grad()
    def update_latents(self, writeback: tuple[torch.Tensor, torch.Tensor], stoch: torch.Tensor, deter: torch.Tensor) -> None:
        """Write freshly recomputed posterior latents back into storage."""
        phys, env_ids = writeback
        env_col = env_ids[:, None]
        self.stoch_idx[phys, env_col] = stoch.argmax(dim=-1).to(torch.int16).to(self.storage_device, non_blocking=True)
        self.deter[phys, env_col] = deter.to(torch.float16).to(self.storage_device, non_blocking=True)
