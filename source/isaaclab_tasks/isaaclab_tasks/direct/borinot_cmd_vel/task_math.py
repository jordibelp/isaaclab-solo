# SPDX-License-Identifier: BSD-3-Clause
"""Simulator-free frame, reward, and disturbance math."""
import math

import torch


def heading_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    """Heading-to-world R=[x,y,z], x=(cos(yaw),sin(yaw),0), y=z cross x.

    Input is a unit body-to-world Hamilton quaternion (wxyz). At exactly vertical
    body x, yaw is undefined; select zero there. The default 70-degree tilt reset
    keeps rollouts away from that singularity. In particular, Log(R).z is NOT yaw.
    """
    w, x, y, z = quaternion.unbind(-1)
    sin_yaw, cos_yaw = 2 * (w * z + x * y), 1 - 2 * (y.square() + z.square())
    singular = sin_yaw.square() + cos_yaw.square() < 1e-12
    yaw = torch.atan2(torch.where(singular, 0.0, sin_yaw), torch.where(singular, 1.0, cos_yaw))
    c, s = yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)


def tracking_kernels(commands, velocity_w, angular_velocity_w, quaternion, linear_std, angular_std):
    """Return XYZ, world-Z yaw, and curriculum-only XY kernels, all in [0,1]."""
    velocity_h = torch.bmm(heading_rotation(quaternion).transpose(1, 2), velocity_w.unsqueeze(-1)).squeeze(-1)
    error = (commands[:, :3] - velocity_h).square()
    linear = torch.exp(-error.sum(-1) / linear_std**2)
    angular = torch.exp(-(commands[:, 3] - angular_velocity_w[:, 2]).square() / angular_std**2)
    planar = torch.exp(-error[:, :2].sum(-1) / linear_std**2)
    return linear, angular, planar


def sample_vertical_pushes(limits_n, radius_m, half_height_m):
    """Independent uniform magnitudes/signs, area-uniform points on a base disk.

    Returns signed WORLD-Z force magnitudes and BODY-LINK application points.
    Upward forces enter the bottom face; downward forces enter the top face.
    """
    samples = torch.rand((len(limits_n), 4), device=limits_n.device, dtype=limits_n.dtype)
    force_z = limits_n * samples[:, 0] * torch.where(samples[:, 1] < 0.5, -1.0, 1.0)
    radius, angle = radius_m * samples[:, 2].sqrt(), 2 * math.pi * samples[:, 3]
    points = torch.stack((radius * angle.cos(), radius * angle.sin(),
                          torch.where(force_z >= 0, -half_height_m, half_height_m)), -1)
    return force_z, points


def sample_steps(count, seconds_range, step_dt, device):
    """Round duration endpoints up to policy ticks; include both integer endpoints."""
    low, high = (max(1, math.ceil(value / step_dt)) for value in seconds_range)
    return torch.randint(low, high + 1, (count,), device=device)
