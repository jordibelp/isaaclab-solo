# SPDX-License-Identifier: BSD-3-Clause
"""Simulator-free sampling and curriculum for the go-to task."""
import math

import torch


def stage_settings(cfg, level):
    """Increase geometry first; only then fade the dense reward."""
    difficulty = min(level, len(cfg.goal_half_extents_m) - 1)
    fade = max(0, level - (len(cfg.goal_half_extents_m) - 1))
    factor = cfg.progress_reward_factors[fade] if cfg.progress_reward_curriculum else 1.0
    return difficulty, factor


def max_curriculum_level(cfg):
    return len(cfg.goal_half_extents_m) - 1 + (
        len(cfg.progress_reward_factors) - 1 if cfg.progress_reward_curriculum else 0)


def deadline_floor(cfg, difficulty, distance):
    """Shortest deadline the sampler assigns to a goal distance (tensor in, tensor out).

    The floor is a sampling heuristic, not a plant reachability certificate.
    It combines a cruise-speed bound and a triangular rest-to-rest profile.
    """
    return torch.maximum(distance / cfg.goal_reference_speed_m_s[difficulty],
                         2 * torch.sqrt(distance / cfg.goal_reference_accel_m_s2[difficulty]))


def training_deadline_interval(cfg, difficulty, distance):
    """Deadlines the sampler can assign to this distance, before rounding up to policy ticks."""
    floor = deadline_floor(cfg, difficulty, distance)
    low, high = cfg.goal_deadline_range_s[difficulty]
    return floor.clamp_min(low), floor.clamp_min(high)


def sample_goals(start_position_w, cfg, difficulty, step_dt):
    """Uniform box offsets and quantized deadlines with a distance-dependent floor."""
    size = start_position_w.shape[0]
    extent = start_position_w.new_tensor(cfg.goal_half_extents_m[difficulty])
    offset = (2 * torch.rand(size, 3, device=start_position_w.device) - 1) * extent
    distance = offset.norm(dim=-1)
    low, high = cfg.goal_deadline_range_s[difficulty]
    requested = low + (high - low) * torch.rand(size, device=start_position_w.device)
    floor = deadline_floor(cfg, difficulty, distance)
    deadline_steps = torch.ceil(torch.maximum(requested, floor) / step_dt).long().clamp_min(1)
    return start_position_w + offset, deadline_steps


def validate_goal_config(cfg):
    stages = len(cfg.goal_half_extents_m)
    if stages < 1:
        raise ValueError("goal_half_extents_m needs at least one difficulty.")
    for name in ("goal_deadline_range_s", "goal_reference_speed_m_s", "goal_reference_accel_m_s2"):
        if len(getattr(cfg, name)) != stages:
            raise ValueError(f"{name} must have one entry per goal difficulty.")
    for extent in cfg.goal_half_extents_m:
        if len(extent) != 3 or any(not math.isfinite(v) or v < 0 for v in extent) or not any(extent):
            raise ValueError("Each goal_half_extents_m entry needs three nonnegative finite values, not all zero.")
    for a, b in zip(cfg.goal_half_extents_m, cfg.goal_half_extents_m[1:]):
        if any(x > y for x, y in zip(a, b)):
            raise ValueError("Goal half-extents must be non-decreasing on every axis.")
    for bounds in cfg.goal_deadline_range_s:
        if len(bounds) != 2 or not all(math.isfinite(v) and v > 0 for v in bounds) or bounds[0] > bounds[1]:
            raise ValueError("Each goal_deadline_range_s entry needs two ordered positive finite seconds.")
    for name in ("goal_reference_speed_m_s", "goal_reference_accel_m_s2"):
        if any(not math.isfinite(v) or v <= 0 for v in getattr(cfg, name)):
            raise ValueError(f"{name} must contain positive finite values.")
    factors = cfg.progress_reward_factors
    if (not factors or factors[0] != 1.0 or factors[-1] != 0.0
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in factors)
            or any(a < b for a, b in zip(factors, factors[1:]))):
        raise ValueError("progress_reward_factors must decrease from 1.0 to 0.0.")
    for name in ("curriculum", "skip_curriculum", "progress_reward_curriculum", "use_terminal_speed_penalty",
                 "use_terminal_success_bonus", "goal_debug_vis"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be a boolean.")
    if (not isinstance(cfg.goal_curriculum_start_level, int) or isinstance(cfg.goal_curriculum_start_level, bool)
            or not 0 <= cfg.goal_curriculum_start_level <= max_curriculum_level(cfg)):
        raise ValueError("goal_curriculum_start_level is outside the goal/fade curriculum.")
    if not math.isfinite(cfg.curriculum_success_threshold) or not 0 <= cfg.curriculum_success_threshold <= 1:
        raise ValueError("curriculum_success_threshold must lie in [0,1].")
    for name in ("curriculum_min_steps", "curriculum_min_episodes", "curriculum_window_episodes"):
        v = getattr(cfg, name)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if cfg.curriculum_min_episodes > cfg.curriculum_window_episodes:
        raise ValueError("curriculum_min_episodes cannot exceed curriculum_window_episodes.")
    for name in ("goal_success_tolerance_m", "failure_distance_floor_m"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if cfg.rwd_scale_terminal_distance >= 0 or cfg.rwd_scale_progress < 0:
        raise ValueError("Use a negative terminal-distance scale and a nonnegative progress scale.")
    for name in ("rwd_scale_terminal_distance", "rwd_scale_progress", "rwd_scale_terminal_speed", "rwd_scale_terminal_success"):
        if not math.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} must be finite.")
    if cfg.rwd_scale_terminal_speed > 0 or cfg.rwd_scale_terminal_success < 0:
        raise ValueError("Terminal speed must penalize; terminal success must reward.")
    # No clipping of goals: reject an unsafe sampling box instead of silently
    # putting a probability mass on a safety boundary.
    extent = cfg.goal_half_extents_m[-1]
    low, high = cfg.flight_height_range_m
    if (cfg.robot_home[2] - cfg.reset_z_noise_m - extent[2] <= low
            or cfg.robot_home[2] + cfg.reset_z_noise_m + extent[2] >= high):
        raise ValueError("The largest goal Z box and reset noise must lie strictly inside flight_height_range_m.")
    if any(abs(cfg.robot_home[i]) + cfg.reset_xy_noise_m + extent[i] >= cfg.flight_xy_limit_m for i in (0, 1)):
        raise ValueError("The largest goal XY box and reset noise must lie strictly inside flight_xy_limit_m.")


class SuccessCurriculum:
    """Episode-count-weighted, stage-pure rolling evidence; at most one promotion per batch."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.max_level = max_curriculum_level(cfg)
        self.level = self.max_level if cfg.skip_curriculum else cfg.goal_curriculum_start_level
        self.window = torch.zeros(cfg.curriculum_window_episodes, device=device)
        self.cursor = self.count = self.start_tick = 0

    def observe(self, successes, episode_levels, tick):
        current = episode_levels == self.level
        values = successes[current].float()
        eligible = len(values)
        if eligible:
            self.count += eligible
            values = values[-len(self.window):]
            indices = (torch.arange(len(values), device=self.window.device) + self.cursor) % len(self.window)
            self.window[indices] = values
            self.cursor = (self.cursor + len(values)) % len(self.window)
        retained = min(self.count, len(self.window))
        success_count = self.window.sum()
        rate = success_count / max(1, retained)
        decision = {"evaluated_stage": self.level, "window_success_count": success_count.clone(),
                    "window_completed_count": retained, "window_success_rate": rate.clone(),
                    "new_eligible_count": eligible, "stage_age_steps": tick - self.start_tick}
        if (self.cfg.curriculum and eligible and self.level < self.max_level
                and retained >= self.cfg.curriculum_min_episodes
                and tick - self.start_tick >= self.cfg.curriculum_min_steps
                and rate >= self.cfg.curriculum_success_threshold):
            self.level += 1
            self.start_tick = tick
            self.cursor = self.count = 0
            self.window.zero_()
        decision["selected_stage"] = self.level
        return decision
