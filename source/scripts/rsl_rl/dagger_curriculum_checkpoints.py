# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-pure episode-return selection for Solo12 DAgger adapters."""

import math
from collections import deque

import torch


def curriculum_state(raw_env) -> dict | None:
    global_idx = raw_env.get_curriculum_global_idx() if hasattr(raw_env, "get_curriculum_global_idx") else None
    phase = None
    if bool(getattr(raw_env.cfg, "curriculum_two_feet", False)):
        phase = int(raw_env._two_feet_curriculum_phase)
    if global_idx is None and phase is None:
        return None
    return {"global_idx": None if global_idx is None else int(global_idx), "two_feet_phase": phase}


class CurriculumBestTracker:
    """Keep a separate last-100-completed-episodes window and best for each stage.

    Call ``update`` after every step, supplying the stage on both sides of the
    auto-reset. Call ``pop_improvements`` once per rollout, before optimization,
    so saved weights are the adapter that collected the scored episodes.
    """

    def __init__(self):
        self.returns = None
        self.pure = None
        self.windows = {}
        self.best = {}
        self.touched = {}

    def update(self, rewards, dones, before, after, excluded=None):
        if before is None:
            return
        if self.returns is None:
            self.returns = torch.zeros_like(rewards)
            self.pure = torch.ones_like(dones, dtype=torch.bool)
        dones = dones.bool()
        self.returns += rewards
        eligible = dones & self.pure
        if excluded is not None:
            eligible &= ~excluded
        key = (before["global_idx"], before["two_feet_phase"])
        completed = self.returns[eligible].tolist()
        if completed:
            self.windows.setdefault(key, deque(maxlen=100)).extend(x for x in completed if math.isfinite(x))
            self.touched[key] = before
        self.returns[dones] = 0
        self.pure[dones] = True
        if before != after:
            # Non-reset environments now straddle two stages; ignore that episode.
            self.pure &= dones

    def pop_improvements(self) -> list[dict]:
        improvements = []
        for key, state in self.touched.items():
            window = self.windows[key]
            if not window:
                continue
            mean_return = sum(window) / len(window)
            if mean_return <= self.best.get(key, -math.inf):
                continue
            self.best[key] = mean_return
            idx, phase = key
            suffix = []
            if idx is not None:
                suffix.append(f"curriculum_idx_{idx}")
            if phase is not None:
                suffix.append(f"two_feet_phase_{phase}")
            improvements.append({
                "filename": f"adapter_best_{'_'.join(suffix)}.pt",
                "curriculum_state": state,
                "best_model_metric": "rollout/episode_return",
                "best_model_value": mean_return,
                "best_model_episode_count": len(window),
            })
        self.touched.clear()
        return improvements
