# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Round-wise observation permutations for RSL-RL plasticity-loss experiments."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


def clone_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Clone a module state without retaining references to live tensors."""

    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


class FirstLayerOnlyController:
    """Freeze a continued actor-critic except for its two input layers."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.policy = runner.alg.policy
        self._input_layers = {
            "actor": self._first_linear(self.policy, "actor"),
            "critic": self._first_linear(self.policy, "critic"),
        }
        self._trainable_parameter_ids = {
            id(parameter)
            for layer in self._input_layers.values()
            for parameter in layer.parameters(recurse=False)
        }
        self.trainable_parameter_names = tuple(
            name
            for name, parameter in self.policy.named_parameters()
            if id(parameter) in self._trainable_parameter_ids
        )
        if not self.trainable_parameter_names:
            raise RuntimeError("First-layer-only mode found no actor/critic input-layer parameters.")

        self.total_parameter_count = sum(parameter.numel() for parameter in self.policy.parameters())
        self.trainable_parameter_count = sum(
            parameter.numel()
            for parameter in self.policy.parameters()
            if id(parameter) in self._trainable_parameter_ids
        )
        self._normalizers = tuple(
            module
            for name in ("actor_obs_normalizer", "critic_obs_normalizer")
            if isinstance((module := getattr(self.policy, name, None)), torch.nn.Module)
        )
        self.active = False
        self._patch_runner_train_mode()

    @staticmethod
    def _first_linear(policy: torch.nn.Module, branch_name: str) -> torch.nn.Linear:
        branch = getattr(policy, branch_name, None)
        if not isinstance(branch, torch.nn.Module):
            raise ValueError(
                f"First-layer-only mode requires a policy with a torch module named '{branch_name}'."
            )
        for module in branch.modules():
            if isinstance(module, torch.nn.Linear):
                return module
        raise ValueError(f"First-layer-only mode found no Linear layer in policy.{branch_name}.")

    @property
    def trainable_parameter_fraction(self) -> float:
        return self.trainable_parameter_count / self.total_parameter_count

    def _freeze_normalizers(self) -> None:
        if self.active:
            for normalizer in self._normalizers:
                normalizer.eval()

    def _patch_runner_train_mode(self) -> None:
        original_train_mode = getattr(self.runner, "train_mode", None)
        if not callable(original_train_mode):
            return

        def _train_mode_with_frozen_normalizers(*args, **kwargs):
            result = original_train_mode(*args, **kwargs)
            self._freeze_normalizers()
            return result

        self.runner.train_mode = _train_mode_with_frozen_normalizers

    def activate(self) -> bool:
        """Freeze all but the actor/critic input layers; return whether this changed state."""

        if self.active:
            return False
        for parameter in self.policy.parameters():
            parameter.requires_grad_(id(parameter) in self._trainable_parameter_ids)
            parameter.grad = None
        self.active = True
        self._freeze_normalizers()
        return True


class ResetAllController:
    """Implement the paper's fresh-network reset-all control between rounds."""

    def __init__(
        self,
        runner: Any,
        *,
        initial_policy_state: dict[str, Any] | None = None,
        learning_rate: float | None = None,
        initial_cbp_state: dict[str, Any] | None = None,
    ) -> None:
        if getattr(runner.alg, "rnd", None):
            raise ValueError("Reset-all does not yet support RND auxiliary networks.")
        self.runner = runner
        self.policy = runner.alg.policy
        self.optimizer = runner.alg.optimizer
        self.initial_policy_state = clone_state_dict(
            self.policy.state_dict() if initial_policy_state is None else initial_policy_state
        )
        self.learning_rate = float(
            runner.alg.learning_rate if learning_rate is None else learning_rate
        )
        self.initial_group_lrs = [self.learning_rate for _ in self.optimizer.param_groups]
        self.cbp_manager = getattr(runner, "_borinot_cbp_manager", None)
        if initial_cbp_state is None and self.cbp_manager is not None:
            initial_cbp_state = self.cbp_manager.state_dict()
        self.initial_cbp_state = copy.deepcopy(initial_cbp_state)
        self.reset_count = 0

    @staticmethod
    def _reset_leaf_modules(module: torch.nn.Module) -> int:
        reset_count = 0
        with torch.no_grad():
            for child in module.modules():
                if child is module or any(child.children()):
                    continue
                reset_parameters = getattr(child, "reset_parameters", None)
                if callable(reset_parameters):
                    reset_parameters()
                    reset_count += 1
        return reset_count

    def reset(self) -> None:
        # Restore non-layer state (normalizers, exploration parameters, and any
        # custom buffers), then freshly sample every resettable leaf layer.
        # RSL updates empirical-normalizer buffers inside inference mode, so
        # those buffers must also be restored under inference mode.
        with torch.inference_mode():
            self.policy.load_state_dict(self.initial_policy_state)
            reset_modules = self._reset_leaf_modules(self.policy)
        if reset_modules == 0:
            raise RuntimeError("Reset-all found no resettable actor-critic layers.")

        self.optimizer.state.clear()
        for parameter in self.policy.parameters():
            parameter.grad = None
        self.runner.alg.learning_rate = self.learning_rate
        self.optimizer.defaults["lr"] = self.learning_rate
        for group, initial_lr in zip(self.optimizer.param_groups, self.initial_group_lrs, strict=True):
            group["lr"] = initial_lr

        if self.cbp_manager is not None and self.initial_cbp_state is not None:
            self.cbp_manager.load_state_dict(copy.deepcopy(self.initial_cbp_state))
            for group in self.cbp_manager.groups:
                group._last_features = None

        self.reset_count += 1


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
