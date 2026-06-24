# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from .shared_actor_critic import SharedActorCritic


class EnvParamsConditionedActor(SharedActorCritic):
    """Flat actor-critic that consumes current race obs plus privileged GT env params.

    The env observation layout is:
        [current race obs, GT env params]

    This intentionally stays as close as possible to the standard RSL-RL ActorCritic.
    The custom class mainly validates the privileged-input layout and can warm-start
    from older MLP or TCN checkpoints by copying compatible observation columns
    into the larger first layer and initializing newly-added input columns to zero.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        current_obs_dim: int = 63,
        env_params_dim: int = 16,
        **kwargs: dict[str, Any],
    ) -> None:
        self.current_obs_dim = int(current_obs_dim)
        self.env_params_dim = int(env_params_dim)
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        num_actor_obs = self._obs_dim(obs, obs_groups["policy"])
        num_critic_obs = self._obs_dim(obs, obs_groups["critic"])
        inferred_current_obs_dim = num_actor_obs - self.env_params_dim
        if inferred_current_obs_dim > 0 and inferred_current_obs_dim != self.current_obs_dim:
            print(
                f"[INFO]: Inferred params-conditioned current_obs_dim={inferred_current_obs_dim} "
                f"from env observation shape; policy config had current_obs_dim={self.current_obs_dim}.",
                flush=True,
            )
            self.current_obs_dim = inferred_current_obs_dim
        expected_obs_dim = self.current_obs_dim + self.env_params_dim
        if num_actor_obs != expected_obs_dim:
            raise ValueError(
                f"Actor observation dim {num_actor_obs} != expected params-conditioned dim {expected_obs_dim}."
            )
        if num_critic_obs != expected_obs_dim:
            raise ValueError(
                f"Critic observation dim {num_critic_obs} != expected params-conditioned dim {expected_obs_dim}."
            )

    @staticmethod
    def _obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for obs_group in groups:
            assert len(obs[obs_group].shape) == 2, "EnvParamsConditionedActor expects flat observations from the env."
            dim += obs[obs_group].shape[-1]
        return dim

    def _source_current_obs_dim(self, source_dim: int) -> int | None:
        for current_dim in (self.current_obs_dim, 57):
            if source_dim in (current_dim, current_dim + self.env_params_dim, current_dim + 64):
                return current_dim
            history_dim = source_dim - current_dim
            if history_dim > 0 and (history_dim % 24 == 0 or history_dim % 48 == 0):
                return current_dim
        return None

    def _adapt_obs_layout_tensor(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        zero_missing_columns: bool,
    ) -> torch.Tensor | None:
        target_dim = self.current_obs_dim + self.env_params_dim
        if target.shape[-1] != target_dim:
            return None

        old_shared_dim = 51  # old layout up to and including c1/c2; old gate_idx was at column 51.
        old_env_params_start = 52
        old_env_params_dim = self.env_params_dim
        source_dim = source.shape[-1]
        if source_dim < old_shared_dim:
            return None

        adapted = torch.zeros_like(target) if zero_missing_columns else target.clone()
        source_current_dim = self._source_current_obs_dim(source_dim)
        if source_current_dim is not None:
            copy_dim = min(source_current_dim, self.current_obs_dim)
            adapted[..., :copy_dim] = source[..., :copy_dim]
            if source_dim == source_current_dim + self.env_params_dim:
                adapted[..., self.current_obs_dim : target_dim] = source[
                    ..., source_current_dim : source_current_dim + self.env_params_dim
                ]
            return adapted

        adapted[..., :old_shared_dim] = source[..., :old_shared_dim]
        if source_dim == old_env_params_start + old_env_params_dim:
            adapted[..., self.current_obs_dim : target_dim] = source[
                ..., old_env_params_start : old_env_params_start + old_env_params_dim
            ]
        return adapted

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        target_state = self.state_dict()
        exact_match = True
        adapted_keys: list[str] = []
        skipped_keys: list[str] = []

        for key, source_value in state_dict.items():
            if key not in target_state:
                skipped_keys.append(key)
                exact_match = False
                continue

            target_value = target_state[key]
            if tuple(source_value.shape) == tuple(target_value.shape):
                target_state[key] = source_value
                continue

            adapted_value = self._adapt_input_tensor(key, source_value, target_value)
            if adapted_value is None:
                skipped_keys.append(key)
                exact_match = False
                continue

            target_state[key] = adapted_value
            adapted_keys.append(key)
            exact_match = False

        if exact_match:
            super().load_state_dict(state_dict, strict=strict)
            return True

        if any(key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer.")) for key in adapted_keys):
            for key in ("actor_obs_normalizer.count", "critic_obs_normalizer.count"):
                if key in target_state:
                    target_state[key] = torch.zeros_like(target_state[key])

        super().load_state_dict(target_state, strict=True)
        if adapted_keys:
            print(
                "[INFO]: Warm-started EnvParamsConditionedActor from a checkpoint with a different observation "
                f"layout; adapted {len(adapted_keys)} tensors and zero-initialized new cClose/cClose1 inputs.",
                flush=True,
            )
        if skipped_keys:
            preview = ", ".join(skipped_keys[:8])
            suffix = " ..." if len(skipped_keys) > 8 else ""
            print(f"[WARN]: Skipped {len(skipped_keys)} incompatible checkpoint tensors: {preview}{suffix}", flush=True)
        return False

    def _adapt_input_tensor(self, key: str, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        if key in ("actor.0.weight", "critic.0.weight") and source.ndim == 2 and target.ndim == 2:
            if source.shape[0] != target.shape[0]:
                return None
            return self._adapt_obs_layout_tensor(source, target, zero_missing_columns=True)

        if (
            key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer."))
            and key.rsplit(".", 1)[-1] in {"_mean", "_var", "_std"}
            and source.ndim == 2
            and target.ndim == 2
            and source.shape[0] == target.shape[0] == 1
        ):
            return self._adapt_obs_layout_tensor(source, target, zero_missing_columns=False)

        if key.startswith(("actor_obs_normalizer.", "critic_obs_normalizer.")) and key.endswith(".count"):
            return torch.zeros_like(target)

        return None
