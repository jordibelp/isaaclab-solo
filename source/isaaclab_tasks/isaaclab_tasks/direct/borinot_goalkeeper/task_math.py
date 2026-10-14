# SPDX-License-Identifier: BSD-3-Clause
"""Tensor-only shot geometry. These functions also run without Isaac Sim."""
import torch


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
