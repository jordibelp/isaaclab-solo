# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Round-wise observation permutations for RSL-RL plasticity-loss experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class ObservationPermutationVecEnv(VecEnv):
    """Permute flat observation features with one mapping per training round.

    Each observation group gets its own permutation. A mapping is shared by the
    full batch of parallel environments and remains fixed for the whole round.
    """

    def __init__(
        self,
        env: VecEnv,
        *,
        round_duration: int,
        num_rounds: int,
        seed: int,
        symmetry_fn: Callable | None = None,
    ) -> None:
        if round_duration < 1:
            raise ValueError(f"Plasticity permutation round_duration must be positive, got {round_duration}.")
        if num_rounds < 1:
            raise ValueError(f"Plasticity permutation num_rounds must be positive, got {num_rounds}.")

        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.round_duration = int(round_duration)
        self.num_rounds = int(num_rounds)
        self.seed = int(seed)
        self.symmetry_fn = symmetry_fn
        self.round_index = 0
        self._permutations: dict[str, torch.Tensor] = {}
        self._inverse_permutations: dict[str, torch.Tensor] = {}
        self._feature_dims: dict[str, int] = {}

        # Discover observation groups immediately so the runner's first query is
        # already shuffled. The first random mapping is the paper's first round.
        initial_obs = self.env.get_observations()
        self._initialize_from_observations(initial_obs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cfg(self) -> object:
        return self.env.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    @property
    def total_learning_iterations(self) -> int:
        return self.round_duration * self.num_rounds

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = self.env.reset()
        return self.permute_observations(obs), extras

    def get_observations(self) -> TensorDict:
        return self.permute_observations(self.env.get_observations())

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, dones, extras = self.env.step(actions)
        return self.permute_observations(obs), rewards, dones, extras

    def _initialize_from_observations(self, obs: TensorDict) -> None:
        feature_dims = {
            str(key): int(value.shape[-1])
            for key, value in obs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[-1] > 1
        }
        if not feature_dims:
            raise ValueError(
                "The plasticity permutation experiment requires at least one rank-2 observation tensor."
            )
        self._feature_dims = dict(sorted(feature_dims.items()))
        self._set_round_permutations(0)

    def _permutation_for(self, key_index: int, feature_dim: int, round_index: int) -> torch.Tensor:
        identity = torch.arange(feature_dim)
        previous = None
        for current_round in range(round_index + 1):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 1_000_003 * current_round + 9_176 * key_index)
            permutation = torch.randperm(feature_dim, generator=generator)
            # Avoid identity in the first round and guarantee that every later
            # round changes its mapping, including for tiny feature dimensions.
            if feature_dim > 1 and (
                (current_round == 0 and torch.equal(permutation, identity))
                or (previous is not None and torch.equal(permutation, previous))
            ):
                permutation = torch.roll(permutation, shifts=1)
            previous = permutation
        return permutation.to(self.device)

    def _set_round_permutations(self, round_index: int) -> None:
        if not 0 <= round_index < self.num_rounds:
            raise ValueError(f"Permutation round must be in [0, {self.num_rounds - 1}], got {round_index}.")

        permutations = {}
        inverses = {}
        for key_index, (key, feature_dim) in enumerate(self._feature_dims.items()):
            permutation = self._permutation_for(key_index, feature_dim, round_index)
            permutations[key] = permutation
            inverses[key] = torch.argsort(permutation)
        self._permutations = permutations
        self._inverse_permutations = inverses
        self.round_index = round_index

    def _map_observations(self, obs: TensorDict, mappings: dict[str, torch.Tensor]) -> TensorDict:
        result = obs.clone()
        for key, mapping in mappings.items():
            if key not in result.keys(include_nested=False):
                raise KeyError(f"Observation group '{key}' disappeared during the permutation experiment.")
            value = result[key]
            if value.ndim != 2 or value.shape[-1] != len(mapping):
                raise ValueError(
                    f"Observation group '{key}' changed shape from feature dimension {len(mapping)} "
                    f"to {tuple(value.shape)}."
                )
            result[key] = value.index_select(-1, mapping.to(value.device))
        return result

    def permute_observations(self, obs: TensorDict) -> TensorDict:
        return self._map_observations(obs, self._permutations)

    def unpermute_observations(self, obs: TensorDict) -> TensorDict:
        return self._map_observations(obs, self._inverse_permutations)

    def finish_learning_iteration(self, iteration: int, live_obs: TensorDict) -> bool:
        """Advance the mapping after a complete PPO update and re-encode ``live_obs`` in place."""

        next_iteration = int(iteration) + 1
        next_round = min(next_iteration // self.round_duration, self.num_rounds - 1)
        if next_round == self.round_index:
            return False

        canonical_obs = self.unpermute_observations(live_obs)
        self._set_round_permutations(next_round)
        remapped_obs = self.permute_observations(canonical_obs)
        # RSL-RL creates the live rollout observation inside torch.inference_mode().
        # It must therefore also be re-encoded under inference mode at the boundary.
        with torch.inference_mode():
            for key in self._feature_dims:
                live_obs[key].copy_(remapped_obs[key])
        return True

    def set_learning_iteration(self, iteration: int) -> None:
        """Select the deterministic round mapping needed when training resumes."""

        round_index = min(max(0, int(iteration)) // self.round_duration, self.num_rounds - 1)
        self._set_round_permutations(round_index)

    def logging_values(self, iteration: int) -> dict[str, float | int]:
        return {
            "permutation_index": self.round_index,
            "permutations_seen": self.round_index + 1,
            "round_step": int(iteration) % self.round_duration,
            "round_progress": (int(iteration) % self.round_duration + 1) / self.round_duration,
        }


@torch.no_grad()
def compute_permutation_aware_symmetry(
    env: Any = None,
    obs: TensorDict | torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[TensorDict | torch.Tensor | None, torch.Tensor | None]:
    """Apply semantic Solo12 symmetry around the current feature permutation."""

    if not isinstance(env, ObservationPermutationVecEnv):
        raise TypeError("Permutation-aware symmetry requires ObservationPermutationVecEnv.")
    if env.symmetry_fn is None:
        raise RuntimeError("No base Solo12 symmetry function was configured.")

    semantic_obs = env.unpermute_observations(obs) if isinstance(obs, TensorDict) else obs
    symmetric_obs, symmetric_actions = env.symmetry_fn(
        env=env,
        obs=semantic_obs,
        actions=actions,
        obs_type=obs_type,
    )
    if isinstance(symmetric_obs, TensorDict):
        symmetric_obs = env.permute_observations(symmetric_obs)
    return symmetric_obs, symmetric_actions
