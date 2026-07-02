# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deferred-reset patch for Direct Isaac Lab envs (DreamerV3 data alignment).

Isaac Lab auto-resets an env inside ``step()`` as soon as it terminates, so the
observation returned at a done step is already the *next* episode's initial
observation and the true terminal observation is lost.  DreamerV3 (danijar's
reference and the r2dreamer reproduction) instead stores the terminal
observation together with the reward and terminal flags that *arrive* with it,
and resets one step later.

This patch (adapted from r2dreamer's IsaacLab wrapper, Direct-env variant)
defers the reset by one step:

1. ``step(action, done)`` runs the parent ``step()`` with all ``_reset_idx``
   calls intercepted, so envs that terminate this step keep their terminal
   observation in the returned obs.
2. Envs the caller flags as ``done`` (the previous step's terminations) are
   explicitly reset *now*: their action was zeroed for the (discarded) junk
   step, ``_reset_idx`` runs, this step's termination flags are cleared for
   them, and their rows in the returned obs are replaced with fresh initial
   observations.

The caller is responsible for treating the flagged envs' transitions as
``is_first`` (fresh obs, zero reward).
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv


def patch_deferred_reset(env: DirectRLEnv) -> None:
    """Swap ``env``'s class for a subclass whose ``step()`` defers resets by one step."""
    if not isinstance(env, DirectRLEnv):
        raise TypeError(
            f"Deferred-reset patch supports DirectRLEnv tasks only, got {type(env).__name__}. "
            "ManagerBased envs need r2dreamer's observation-history rollback (see ~/r2dreamer/envs/isaaclab.py)."
        )

    original_cls = type(env)

    class _DeferredResetEnv(original_cls):
        # True only while the parent step() runs; gates the _reset_idx interception.
        _block_reset: bool = False

        def _reset_idx(self, env_ids):
            if not self._block_reset:
                super()._reset_idx(env_ids)

        def step(self, action: torch.Tensor, done: torch.Tensor | None = None):
            # Zero actions for envs being reset so their (discarded) junk step stays neutral.
            if done is not None:
                action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)

            self._block_reset = True
            try:
                result = super().step(action)
            finally:
                self._block_reset = False

            if done is not None and done.any():
                reset_ids = done.nonzero(as_tuple=False).squeeze(-1)

                # Reset first: _reset_idx implementations (e.g. Solo12) read the current
                # termination flags for episode logging.
                original_cls._reset_idx(self, reset_ids)
                if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                    for _ in range(self.cfg.num_rerenders_on_reset):
                        self.sim.render()

                # Clear this step's done flags for the freshly reset envs. The tensors in
                # ``result`` alias these buffers, so the caller sees the cleared flags.
                self.reset_terminated[reset_ids] = False
                self.reset_time_outs[reset_ids] = False
                self.reset_buf[reset_ids] = False

                # Replace the reset envs' rows with their initial observations.
                initial_obs = original_cls._get_observations(self)
                for key, value in initial_obs.items():
                    self.obs_buf[key][reset_ids] = value[reset_ids]

            return result

    _DeferredResetEnv.__name__ = f"DeferredReset{original_cls.__name__}"
    _DeferredResetEnv.__qualname__ = _DeferredResetEnv.__name__
    env.__class__ = _DeferredResetEnv
