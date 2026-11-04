# SPDX-License-Identifier: BSD-3-Clause
"""Simulator-free frame, reward, and disturbance math."""
import math

import torch

# Distal-most vertex of the source calf mesh, in flying_arm_2__link2 axes (m).
# Not the racket's offset/standoff. Keep this tied to borinotFlat.usd geometry.
CALF_TIP_B = (0.0, -0.136, -0.00495)
SHAFT_COM_B = (0.0, -0.08, -0.021)


def command_force_curriculum_stages(command_limits, force_limits):
    """Return (XY half-range or None, force limit) for each combined stage.

    Command stages are force-free. Share the final command stage with an
    initial zero-force stage, so it must pass before any nonzero force starts.
    Empty command limits preserve the configured independent X/Y ranges.
    """
    if (any(not math.isfinite(v) or v < 0 for v in command_limits)
            or any(a > b for a, b in zip(command_limits, command_limits[1:]))):
        raise ValueError("command_lin_vel_lim_curriculum must contain non-negative finite, non-decreasing limits.")
    if not command_limits:
        return [(None, force) for force in force_limits]
    commands = command_limits[:-1] if force_limits[0] == 0 else command_limits
    return ([(limit, 0.0) for limit in commands]
            + [(command_limits[-1], force) for force in force_limits])


def calf_payload_properties(shaft_mass_kg, payload_kg):
    """Merge an ideal point payload into the existing 160 mm rod shaft model.

    Return total mass, link-frame CoM and full inertia ABOUT that CoM. The point
    has no intrinsic inertia, but its offset contributes via the parallel-axis
    theorem. The original USD's 115 g distal mass is replaced, not added again.
    """
    if not math.isfinite(shaft_mass_kg) or shaft_mass_kg <= 0:
        raise ValueError("Shaft mass must be finite and positive.")
    if not math.isfinite(payload_kg) or payload_kg < 0:
        raise ValueError("Point payload must be finite and non-negative.")
    shaft_com = torch.tensor(SHAFT_COM_B, dtype=torch.float64)
    tip = torch.tensor(CALF_TIP_B, dtype=torch.float64)
    total = shaft_mass_kg + payload_kg
    com = (shaft_mass_kg * shaft_com + payload_kg * tip) / total
    transverse = shaft_mass_kg * 0.16**2 / 12
    inertia = torch.diag(torch.tensor((transverse, 1.0e-6, transverse), dtype=torch.float64))
    for mass, position in ((shaft_mass_kg, shaft_com), (payload_kg, tip)):
        delta = position - com
        inertia += mass * (delta.square().sum() * torch.eye(3, dtype=torch.float64) - torch.outer(delta, delta))
    return total, com, inertia


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
    return tracking_metrics(commands, velocity_w, angular_velocity_w, quaternion, linear_std, angular_std)[:3]


def tracking_metrics(commands, velocity_w, angular_velocity_w, quaternion, linear_std, angular_std):
    """Return XYZ/yaw/XY kernels followed by raw XY/XYZ error norms in m/s.

    Errors use the same heading frame as the kernels, before width scaling.
    Compute them directly: inverting exp would lose errors when it underflows.
    """
    velocity_h = torch.bmm(heading_rotation(quaternion).transpose(1, 2), velocity_w.unsqueeze(-1)).squeeze(-1)
    error = (commands[:, :3] - velocity_h).square()
    linear = torch.exp(-error.sum(-1) / linear_std**2)
    angular = torch.exp(-(commands[:, 3] - angular_velocity_w[:, 2]).square() / angular_std**2)
    planar = torch.exp(-error[:, :2].sum(-1) / linear_std**2)
    return linear, angular, planar, error[:, :2].sum(-1).sqrt(), error.sum(-1).sqrt()


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
