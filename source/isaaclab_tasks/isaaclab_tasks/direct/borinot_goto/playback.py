# SPDX-License-Identifier: BSD-3-Clause
"""Play-only goal ownership. The registered training environment is unchanged."""
import math

import torch

from .borinot_goto_env import BorinotGoToEnv
from .task_math import stage_settings, training_deadline_interval


class BorinotGoToPlayEnv(BorinotGoToEnv):
    """Keep a live goal across deadlines and resets.

    ``goal`` is env-local XYZ in metres, the frame of ``cfg.robot_home`` (z is
    height). ``None`` keeps the training sampler and its reset at every
    deadline. With a goal, a clean deadline ends the segment, not the flight:
    a hold segment with the same goal starts from the current state, and
    ``set_play_goal`` starts a new goal from the current state. Safety failures
    still reset the robot to home, then the goal is flown again.
    ``reset_at_deadline=True`` restores training episodes for the live goal;
    callers reset the environment after ``set_play_goal`` in that mode.
    ``deadline_s=None`` uses the middle of the training deadline range for each
    environment's distance. Holds use the shortest training deadline: the
    position-only stage reward lets the policy loop away and come back, and the
    loop grows with the time left.
    """

    def __init__(self, cfg, goal=None, deadline_s=None, reset_at_deadline=False, goal_relative=False, **kwargs):
        self.play_goal = self.play_deadline_s = None
        self.play_goal_relative = bool(goal_relative)
        self._play_goal_targets = None
        self.reset_at_deadline = bool(reset_at_deadline) or goal is None
        super().__init__(cfg, **kwargs)
        self._segment_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._segment_start_w = self.robot.data.root_pos_w.clone()
        self.play_results = []  # environment 0: every goal/sampled outcome and failed holds, oldest first
        self.play_totals = dict.fromkeys(("goals", "goal_successes", "goal_failures",
                                          "holds", "hold_successes", "hold_failures"), 0)
        self.play_hold_worst_distance_m = 0.0
        if goal is not None:
            # Applied by the first reset; robot state is not valid before it.
            self.play_goal, self.play_deadline_s = self._validated_goal(goal, deadline_s, self.play_goal_relative)

    @property
    def play_difficulty(self):
        return stage_settings(self.cfg, self.goal_curriculum.level)[0]

    def set_play_goal(self, goal, deadline_s=None, relative=False):
        """Snapshot each robot's target; relative displacements use world-aligned axes.

        Resolve once per Go, not on every tick or safety reset. An invalid
        request keeps the previous goal, clock and coordinate mode.
        """
        value, deadline = self._validated_goal(goal, deadline_s, relative)
        targets = self._resolve_goal(value, relative)
        self.play_goal, self.play_deadline_s = value, deadline
        self.play_goal_relative, self._play_goal_targets = bool(relative), targets
        if not self.reset_at_deadline:
            self._start_segment(self.robot._ALL_INDICES, hold=False)

    def _validated_goal(self, goal, deadline_s, relative=False):
        if len(goal) != 3 or not all(math.isfinite(v) for v in goal):
            raise ValueError("Goal needs three finite metres: x y z.")
        if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s <= 0):
            raise ValueError("Deadline must be positive and finite.")
        tensor = torch.tensor(goal, dtype=self._goal_position_w.dtype, device=self.device)
        if not relative:
            self._validate_targets(tensor[None])
        return tensor, None if deadline_s is None else float(deadline_s)

    def _validate_targets(self, targets):
        low, high = self.cfg.flight_height_range_m
        if not bool(((targets[:, 2] > low) & (targets[:, 2] < high)).all()):
            raise ValueError(f"Goal z must lie strictly inside the flight heights {low:g}..{high:g} m.")
        if not bool((targets[:, :2].abs() < self.cfg.flight_xy_limit_m).all()):
            raise ValueError(f"Goal x and y must lie strictly inside +/-{self.cfg.flight_xy_limit_m:g} m.")

    def _resolve_goal(self, value, relative):
        targets = (self.robot.data.root_pos_w - self.scene.env_origins + value if relative
                   else value.expand(self.num_envs, -1).clone())
        self._validate_targets(targets)
        return targets

    def _play_deadline_steps(self, distance, hold):
        if hold or self.play_deadline_s is None:
            low, high = training_deadline_interval(self.cfg, self.play_difficulty, distance)
            seconds = low if hold else (low + high) / 2
        else:
            seconds = torch.full_like(distance, self.play_deadline_s)
        # Round up to policy ticks without float noise turning 1.75 s into 176 ticks.
        return torch.ceil(seconds / self.step_dt - 1e-4).long().clamp_min(1)

    def _start_segment(self, env_ids, hold):
        """Restart the goal clock from the current state. Holds keep the current goal."""
        position = self.robot.data.root_pos_w[env_ids]
        if not hold:
            if self._play_goal_targets is None:
                self._play_goal_targets = self._resolve_goal(self.play_goal, self.play_goal_relative)
            self._goal_position_w[env_ids] = self._play_goal_targets[env_ids] + self.scene.env_origins[env_ids]
            self._segment_start_w[env_ids] = position
        distance = (self._goal_position_w[env_ids] - position).norm(dim=-1)
        self._goal_deadline_steps[env_ids] = self._play_deadline_steps(distance, hold)
        # The goal clock is the episode clock: remaining time and t_norm restart.
        self._episode_steps[env_ids] = 0
        self._initial_goal_distance[env_ids] = distance
        self._previous_goal_distance[env_ids] = distance
        self._minimum_goal_distance[env_ids] = distance
        self._first_arrival_s[env_ids] = torch.where(distance <= self.cfg.goal_success_tolerance_m, 0.0, -1.0)
        for buffer in (self._goal_failed, self._goal_deadline_reached, self._goal_finished, self._goal_success):
            buffer[env_ids] = False
        self._segment_hold[env_ids] = hold

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)  # samples a training goal; a live goal replaces it below
        if self.play_goal is not None:
            self._start_segment(env_ids, hold=False)
        else:
            self._segment_start_w[env_ids] = self.robot.data.root_pos_w[env_ids]
            self._segment_hold[env_ids] = False

    def _get_dones(self):
        terminated, time_outs = super()._get_dones()
        if self.reset_at_deadline:
            return terminated, time_outs
        return self._goal_failed.clone(), time_outs  # only safety failures reset the robot

    def _get_rewards(self):
        reward = super()._get_rewards()
        ended = self._goal_finished.nonzero().flatten()
        if len(ended):
            self._record_outcomes(ended)
            if not self.reset_at_deadline:
                held = ended[~self._goal_failed[ended]]
                if len(held):
                    self._start_segment(held, hold=True)
        return reward

    def _record_outcomes(self, env_ids):
        """Pre-reset outcomes; robot state is still the final state of the segment."""
        info = self.extras["goal"]
        hold, success, failed = self._segment_hold[env_ids], info["success"][env_ids], info["failed"][env_ids]
        distance = info["distance_m"][env_ids]
        for prefix, mask in (("goal", ~hold), ("hold", hold)):
            self.play_totals[f"{prefix}s"] += int(mask.sum())
            self.play_totals[f"{prefix}_successes"] += int((success & mask).sum())
            self.play_totals[f"{prefix}_failures"] += int((failed & mask).sum())
        if hold.any():
            self.play_hold_worst_distance_m = max(self.play_hold_worst_distance_m, float(distance[hold].max()))
        if env_ids[0] != 0 or (hold[0] and not failed[0]):
            return
        origin = self.scene.env_origins[0]
        arrival = float(self._first_arrival_s[0])
        self.play_results.append({
            "kind": "hold" if hold[0] else ("sampled" if self.play_goal is None else "goal"),
            "goal_m": (self._goal_position_w[0] - origin).tolist(),
            "start_m": (self._segment_start_w[0] - origin).tolist(),
            "position_m": (self.robot.data.root_pos_w[0] - origin).tolist(),
            "velocity_m_s": self.robot.data.root_lin_vel_w[0].tolist(),
            "deadline_s": int(self._goal_deadline_steps[0]) * self.step_dt,
            "elapsed_s": int(self._episode_steps[0]) * self.step_dt,
            "distance_m": float(distance[0]),
            "speed_m_s": float(self.robot.data.root_lin_vel_w[0].norm()),
            "min_distance_m": float(self._minimum_goal_distance[0]),
            "first_arrival_s": None if arrival < 0 else arrival,
            "success": bool(success[0]), "failed": bool(failed[0])})


def checkpoint_goal_level(infos, fallback, override_keys):
    if "env.goal_curriculum_start_level" in override_keys:
        return fallback
    for key in ("best_model_curriculum_idx", "curriculum_global_idx"):
        if infos.get(key) is not None:
            return int(infos[key])
    return fallback
