# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from .replay_buffer import ReplayBuffer


class MixedReplayBuffer:
    """Draw every mini-batch from a frozen offline buffer and the live online buffer.

    This is the "retained replay" part of the recipe in *What Matters for Simulation to
    Online Reinforcement Learning on Real Robots* (arXiv:2602.20220). Each mini-batch takes
    a fixed share of its samples from data recorded while pretraining and the rest from data
    collected in the new environment. The pretraining data anchors the critic while the
    policy meets the new dynamics, and the share is annealed to zero so the final policy is
    fitted on new-environment data only.

    The paper writes the mixture with ``alpha`` as the *online* share and anneals it up to 1.
    Here the same schedule is expressed as ``offline_fraction = 1 - alpha`` annealed down to
    0, because that is what the configuration names.

    New transitions go to the online buffer only; the offline buffer is never written to.
    """

    def __init__(
        self,
        online: ReplayBuffer,
        offline: ReplayBuffer,
        initial_offline_fraction: float = 0.5,
        final_offline_fraction: float = 0.0,
        anneal_iterations: int = 1,
    ) -> None:
        """Initialize the mixture.

        Args:
            online: Buffer that receives the transitions collected during this run.
            offline: Read-only buffer holding the retained pretraining transitions.
            initial_offline_fraction: Share of each mini-batch taken from ``offline`` at the
                first iteration. The paper uses 0.5.
            final_offline_fraction: Share reached at the end of the anneal. The paper uses 0.
            anneal_iterations: Number of iterations over which the share moves from the
                initial to the final value. Set both fractions equal for a constant mixture.
        """
        for name, value in (
            ("initial_offline_fraction", initial_offline_fraction),
            ("final_offline_fraction", final_offline_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        if anneal_iterations < 1:
            raise ValueError(f"anneal_iterations must be at least 1, got {anneal_iterations}.")

        online_obs = {key: tuple(value.shape[2:]) for key, value in online.observations.items()}
        offline_obs = {key: tuple(value.shape[2:]) for key, value in offline.observations.items()}
        if online_obs != offline_obs:
            raise ValueError(
                "Offline replay data does not match the current observation layout: "
                f"offline={offline_obs}, online={online_obs}. The snapshot was recorded with a "
                "different observation configuration, so the two cannot be mixed."
            )
        if tuple(online.actions.shape[2:]) != tuple(offline.actions.shape[2:]):
            raise ValueError(
                "Offline replay data does not match the current action layout: "
                f"offline={tuple(offline.actions.shape[2:])}, online={tuple(online.actions.shape[2:])}."
            )

        self.online = online
        self.offline = offline
        self.initial_offline_fraction = float(initial_offline_fraction)
        self.final_offline_fraction = float(final_offline_fraction)
        self.anneal_iterations = int(anneal_iterations)
        self.offline_fraction = self.initial_offline_fraction

    def set_iteration(self, iteration: int) -> float:
        """Move the offline share along its linear schedule and return the new value.

        ``iteration`` counts iterations completed in the current run, so a resumed run does
        not skip straight to the end of the anneal.
        """
        progress = min(max(iteration, 0) / self.anneal_iterations, 1.0)
        self.offline_fraction = self.initial_offline_fraction + progress * (
            self.final_offline_fraction - self.initial_offline_fraction
        )
        return self.offline_fraction

    def add_transition(self, transition: ReplayBuffer.Transition) -> None:
        """Store a transition collected in the current environment."""
        self.online.add_transition(transition)

    def clear(self) -> None:
        """Drop the online transitions. The retained offline data is kept."""
        self.online.clear()

    def save_snapshot(self, path) -> dict:
        """Write the online transitions so a later run can retain them in turn."""
        return self.online.save_snapshot(path)

    def mini_batch_generator(self, num_mini_batch, mini_batch_size, num_epochs=1):
        """Yield mini-batches whose composition follows the current offline share."""
        # The valid-index grids are the expensive part of sampling and do not change while a
        # generator is being consumed, so build them once here and reuse them per batch.
        online_indices = self.online._generate_valid_indices()
        offline_indices = self.offline._generate_valid_indices()

        num_offline = int(round(self.offline_fraction * mini_batch_size))
        if online_indices is None or len(online_indices[0]) == 0:
            # Nothing online to sample yet (for example an n-step window longer than the data
            # collected so far). Fall back to the retained data rather than failing.
            num_offline = mini_batch_size
        num_online = mini_batch_size - num_offline

        for _ in range(num_epochs):
            for _ in range(num_mini_batch):
                if num_offline == 0:
                    yield self.online._generate_batch(online_indices, num_online)
                elif num_online == 0:
                    yield self.offline._generate_batch(offline_indices, num_offline)
                else:
                    online_batch = self.online._generate_batch(online_indices, num_online)
                    offline_batch = self.offline._generate_batch(offline_indices, num_offline)
                    yield [torch.cat((a, b), dim=0) for a, b in zip(online_batch, offline_batch)]
