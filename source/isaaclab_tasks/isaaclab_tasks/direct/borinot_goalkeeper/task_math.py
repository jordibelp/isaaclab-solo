# SPDX-License-Identifier: BSD-3-Clause
"""Tensor-only shot geometry. These functions also run without Isaac Sim."""
import math

import torch


def validate_reset_grid(cfg):
    """Reject an enabled grid whose root positions/noise can leave the allowed box."""
    if not isinstance(cfg.reset_grid_start_level, int) or cfg.reset_grid_start_level < 0:
        raise ValueError("reset_grid_start_level must be a non-negative integer.")
    for name in ("reset_grid_y_offsets_m", "reset_grid_z_offsets_m"):
        values = getattr(cfg, name)
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} must be a non-empty sequence of finite offsets.")
    if not all(math.isfinite(v) and v >= 0 for v in (cfg.reset_xy_noise_m, cfg.reset_z_noise_m)):
        raise ValueError("Reset noise must be finite and non-negative.")
    if not cfg.reset_grid_enabled:
        return
    for axis, offsets, noise, bounds in (
        (0, (0.0,), cfg.reset_xy_noise_m, cfg.goal_area_x_m),
        (1, (0.0, *cfg.reset_grid_y_offsets_m), cfg.reset_xy_noise_m, cfg.goal_area_y_m),
        (2, (0.0, *cfg.reset_grid_z_offsets_m), cfg.reset_z_noise_m, cfg.goal_area_z_m),
    ):
        lo, hi = cfg.robot_home[axis] + min(offsets) - noise, cfg.robot_home[axis] + max(offsets) + noise
        if not math.isfinite(lo + hi) or lo < bounds[0] - 1e-9 or hi > bounds[1] + 1e-9:
            raise ValueError(f"Reset grid/home plus noise exceeds goal_area_{'xyz'[axis]}_m: "
                             f"[{lo:.4f}, {hi:.4f}] not inside {bounds}. Adjust home, offsets, noise or bounds.")


def sample_robot_positions(n, device, cfg, level):
    """Environment-local base positions; index -1 is centre-only, otherwise z-row/y-column."""
    position = torch.tensor(cfg.robot_home, device=device, dtype=torch.float32).expand(n, -1).clone()
    indices = torch.full((n,), -1, dtype=torch.long, device=device)
    if cfg.reset_grid_enabled and level >= cfg.reset_grid_start_level:
        y = torch.tensor(cfg.reset_grid_y_offsets_m, device=device)
        z = torch.tensor(cfg.reset_grid_z_offsets_m, device=device)
        indices = torch.randint(len(y) * len(z), (n,), device=device)
        position[:, 1] += y[indices % len(y)]
        position[:, 2] += z[indices // len(y)]
    position[:, :2] += (2 * torch.rand(n, 2, device=device) - 1) * cfg.reset_xy_noise_m
    position[:, 2] += (2 * torch.rand(n, device=device) - 1) * cfg.reset_z_noise_m
    return position, indices


def ballistic_state(position, velocity, time_s, gravity=9.81):
    time_s = time_s.unsqueeze(-1)
    result = position + velocity * time_s
    result[..., 2] -= 0.5 * gravity * time_s.squeeze(-1).square()
    return result


def predict_intercept(position, velocity, plane_x, horizon_s=1.0):
    """Ball center at a fixed defense plane; no drag or post-impact prediction."""
    incoming = (velocity[..., 0] < -0.1) & (position[..., 0] > plane_x)
    time_s = (plane_x - position[..., 0]) / velocity[..., 0].clamp(max=-0.1)
    valid = incoming & (time_s >= 0.0) & (time_s <= horizon_s)
    time_s = time_s.clamp(0.0, horizon_s)
    return ballistic_state(position, velocity, time_s), time_s, valid


def sample_shots(n, device, cfg, level):
    """Aim at the paddle plane. Every unblocked sample also threatens the goal."""
    target = torch.empty(n, 3, device=device)
    target[:, 0] = cfg.keeper_plane_x
    extent = torch.tensor(cfg.shot_half_extents_yz[level], device=device)
    center = torch.tensor(cfg.target_center_yz, device=device)
    target[:, 1:] = center + (2 * torch.rand(n, 2, device=device) - 1) * extent
    time_lo, time_hi = cfg.shot_flight_time_ranges_s[level]
    time_s = torch.empty(n, device=device).uniform_(time_lo, time_hi)
    position = torch.empty_like(target)
    position[:, 0].uniform_(*cfg.launch_x_range)
    position[:, 1] = target[:, 1] + (2 * torch.rand(n, device=device) - 1) * cfg.launch_y_noise
    position[:, 2] = cfg.launch_z
    velocity = (target - position) / time_s[:, None]
    velocity[:, 2] += 0.5 * 9.81 * time_s
    return position, velocity, target, time_s


def crosses_goal(previous, current, cfg):
    """Swept leading-face crossing, including edge overlap by the ball radius."""
    front = cfg.goal_center[0] + cfg.goal_size[0] / 2 + cfg.ball_radius_m
    crossing = (previous[:, 0] > front) & (current[:, 0] <= front)
    alpha = ((previous[:, 0] - front) / (previous[:, 0] - current[:, 0]).clamp_min(1e-8)).clamp(0, 1)
    at_plane = previous + alpha[:, None] * (current - previous)
    yz_center = torch.tensor(cfg.goal_center[1:], device=current.device)
    yz_extent = torch.tensor(cfg.goal_size[1:], device=current.device) / 2 + cfg.ball_radius_m
    overlaps = ((at_plane[:, 1:] - yz_center).abs() <= yz_extent).all(-1)
    return crossing & overlaps


def plane_progress(previous_paddle, current_paddle, target, scale):
    """Both distances use the same target, as in the Solo12 race gate reward.

    A moving ball alone cannot earn a motionless paddle a progress reward.
    """
    previous_distance = torch.linalg.vector_norm(previous_paddle[:, 1:] - target[:, 1:], dim=-1)
    current_distance = torch.linalg.vector_norm(current_paddle[:, 1:] - target[:, 1:], dim=-1)
    return scale * (previous_distance - current_distance)


def terminal_outcome(base_hit, arm_hit, robot_crash, goal, floor, left_area, missed, deadline, successful):
    """One terminal label/penalty. Safety failures take precedence over a save."""
    code = torch.zeros_like(base_hit, dtype=torch.long)
    # Later entries take precedence. A goal/base hit cannot be discounted to an
    # arm/floor penalty, nor can any failure also earn the terminal save bonus.
    for index, flag in ((7, successful), (6, deadline), (9, missed), (4, floor),
                        (2, arm_hit), (5, left_area), (8, robot_crash), (3, goal), (1, base_hit)):
        code = torch.where(flag, index, code)
    return code


def promotion_ready(age_steps, completed, window_rate, cfg):
    return (cfg.curriculum and age_steps >= cfg.curriculum_min_steps
            and completed >= cfg.curriculum_min_episodes
            and window_rate > cfg.curriculum_success_threshold)
