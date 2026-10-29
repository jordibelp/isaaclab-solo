# SPDX-License-Identifier: BSD-3-Clause
"""Base-only inverse dynamics with the measured hip wrench. Pure batched PyTorch.

Action order differs from INDI: [a_world(3), yaw_world(1), omega_body(3), q_leg(2)].
See direct/borinot/WRENCH_CONTROL.md for frames, timing and model assumptions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .borinot_indi import INDI_TILT_METHODS, attitude_acceleration, rotation_from_quat


@dataclass
class WrenchControllerCfg:
    allocation_weights: tuple = (1.0, 10.0, 10.0, 1.0)  # T, tau_x, tau_y, tau_z
    allocation_iterations: int = 128
    tilt_method: str = "geom"
    tilt_gain: float = 150.0
    yaw_gain: float = 3.0
    rate_gain: tuple = (20.0, 20.0, 8.0)
    acceleration_scale: tuple = (8.0, 8.0, 8.0)  # SAC only, world m/s^2
    heading_range_degs: float = 180.0  # SAC only, around home yaw
    joint_target_limit_degs: tuple = (90.0, 90.0)  # SAC only

    def validate(self, task):
        if self.tilt_method not in INDI_TILT_METHODS:
            raise ValueError(f"wrench.tilt_method must be one of {INDI_TILT_METHODS}.")
        for name, size in (("allocation_weights", 4), ("rate_gain", 3),
                           ("acceleration_scale", 3), ("joint_target_limit_degs", 2)):
            values = getattr(self, name)
            if len(values) != size or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"wrench.{name} requires {size} finite positive values.")
        for name in ("tilt_gain", "yaw_gain", "heading_range_degs"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"wrench.{name} must be finite and positive.")
        if (isinstance(self.allocation_iterations, bool) or not isinstance(self.allocation_iterations, int)
                or self.allocation_iterations < 1):
            raise ValueError("wrench.allocation_iterations must be a positive integer.")
        if task.remove_thrust_clipping:
            raise ValueError("wrench requires physical rotor bounds: remove_thrust_clipping=False.")
        for name in ("kp", "kd", "motor_time_constant_s"):
            if not math.isfinite(getattr(task, name)) or getattr(task, name) < 0:
                raise ValueError(f"wrench requires finite non-negative {name}.")
        for i, name in enumerate(("thigh", "calf")):
            low, high = getattr(task, f"joint_physical_limit_{name}")
            limit = self.joint_target_limit_degs[i]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"joint_physical_limit_{name} requires ordered finite bounds.")
            if task.sac_action_mapping and not low <= -limit < limit <= high:
                raise ValueError(f"wrench SAC joint targets must fit inside joint_physical_limit_{name}.")


def matvec(matrix, vector):
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def leg_on_base_wrench(raw, child_rotation_w, child_position_w, joint_rotation_child,
                       joint_position_child, base_rotation_w, base_com_position_w):
    """PhysX parent-on-child [F,M] at child joint -> leg-on-base F_world, M_baseCoM_body.

    raw uses CHILD JOINT axes, not child link or parent link axes. The USD
    localRot1/localPos1 locate that frame in the proximal thigh link.
    """
    joint_rotation_w = child_rotation_w @ joint_rotation_child
    force_w = -matvec(joint_rotation_w, raw[:, :3])
    hip_w = child_position_w + matvec(child_rotation_w, joint_position_child)
    moment_w = (-matvec(joint_rotation_w, raw[:, 3:])
                + torch.linalg.cross(hip_w - base_com_position_w, force_w))
    return force_w, matvec(base_rotation_w.mT, moment_w)


def force_attitude(force_w, heading, current_rotation):
    """Desired world rotor force -> nonnegative collective and R_WB.

    This is a desired direction, not an instantaneous 3-D force actuator.
    With zero force keep the current thrust axis; yaw feedback remains active.
    """
    magnitude = force_w.norm(dim=-1, keepdim=True)
    z = torch.where(magnitude > 1e-6, force_w / magnitude.clamp_min(1e-6), current_rotation[:, :, 2])
    heading_x = torch.stack((heading.cos(), heading.sin(), torch.zeros_like(heading)), -1)
    heading_y = torch.stack((-heading.sin(), heading.cos(), torch.zeros_like(heading)), -1)
    y = torch.linalg.cross(z, heading_x)
    y = torch.where(y.norm(dim=-1, keepdim=True) > 1e-6, y, torch.linalg.cross(z, heading_y))
    y = torch.nn.functional.normalize(y, dim=-1)
    return magnitude, torch.stack((torch.linalg.cross(y, z), y, z), -1)


class WrenchAllocator:
    """Per-environment 4x6 mixer about the actual base CoM, with bounded least squares.

    Fixed-budget accelerated projected gradient; no host solver or per-env loop.
    Report residuals rather than treating an infeasible request as an equality.
    """

    def __init__(self, allocation, cap, cfg):
        self.allocation, self.cap, self.cfg = allocation, cap, cfg
        self.inverse = torch.linalg.pinv(allocation)
        self.scale = allocation.abs().sum(-1) * cap
        self.weight = allocation.new_tensor(cfg.allocation_weights).sqrt()
        self.a = allocation * cap / self.scale[..., None] * self.weight[None, :, None]
        self.weighted_inverse = torch.linalg.pinv(self.a)
        self.hessian = self.a.mT @ self.a
        self.lipschitz = torch.linalg.eigvalsh(self.hessian)[:, -1:]

    def thrust(self, wrench):
        target = wrench / self.scale * self.weight
        x = matvec(self.weighted_inverse, target).clamp(0, 1)
        linear = matvec(self.a.mT, target)
        y, t = x, 1.0
        for _ in range(self.cfg.allocation_iterations):
            next_x = (y - (matvec(self.hessian, y) - linear) / self.lipschitz).clamp(0, 1)
            next_t = (1 + math.sqrt(1 + 4*t*t)) / 2
            y = next_x + ((t - 1) / next_t) * (next_x - x)
            x, t = next_x, next_t
        return x * self.cap


class WrenchController:
    def __init__(self, task, allocation, inertia, mass, hover):
        self.task, self.cfg, self.dt = task, task.wrench, 1 / task.physics_hz
        self.mass, self.inertia, self.hover = mass, inertia, hover
        self.gravity_w = allocation.new_tensor(task.sim.gravity)
        self.allocator = WrenchAllocator(allocation, hover * task.thrust_to_weight, self.cfg)
        self.reset_thrust = min(hover, self.allocator.cap)
        n = len(mass)
        self.rotor_speed = allocation.new_full((n, 6), math.sqrt(self.reset_thrust / task.kf))
        self.ready = torch.zeros(n, dtype=torch.bool, device=allocation.device)
        self.external_force_w = allocation.new_zeros((n, 3))
        self.external_moment_b = allocation.new_zeros((n, 3))
        self.desired_force_w = allocation.new_zeros((n, 3))
        self.alpha_desired = allocation.new_zeros((n, 3))
        self.requested_wrench = allocation.new_zeros((n, 4))
        self.requested_rotor_thrust = allocation.new_zeros((n, 6))
        self.requested_joint_effort = allocation.new_zeros((n, 2))
        self.allocation_residual = allocation.new_zeros((n, 4))
        self.accel_scale = allocation.new_tensor(self.cfg.acceleration_scale)
        self.rate_scale = allocation.new_tensor(task.body_rate_scale)
        self.joint_scale = allocation.new_tensor(self.cfg.joint_target_limit_degs) * (math.pi / 180)
        w, x, y, z = task.robot_home_quat
        self.home_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self.motor_alpha = 1.0 if task.motor_time_constant_s == 0 else -math.expm1(-self.dt / task.motor_time_constant_s)

    def reset(self, ids):
        self.ready[ids] = False  # incoming wrench is stale until one new physics step completes
        self.rotor_speed[ids] = math.sqrt(self.reset_thrust / self.task.kf)
        for buffer in (self.external_force_w, self.external_moment_b, self.desired_force_w,
                       self.alpha_desired, self.requested_wrench, self.requested_rotor_thrust,
                       self.requested_joint_effort, self.allocation_residual):
            buffer[ids] = 0

    def observation(self):
        return self.task.kf * self.rotor_speed.square() / self.hover

    def step(self, actions, quaternion, rates, joint_pos, joint_vel, force_w, moment_b):
        if self.task.sac_action_mapping:
            u = actions.clamp(-1, 1)
            acceleration, heading = u[:, :3] * self.accel_scale, self.home_yaw + math.radians(self.cfg.heading_range_degs)*u[:, 3]
            reference_rates, target = u[:, 4:7]*self.rate_scale, u[:, 7:9]*self.joint_scale
        else:
            acceleration, heading, reference_rates, target = actions[:, :3], actions[:, 3], actions[:, 4:7], actions[:, 7:9]
        self.external_force_w.copy_(torch.where(self.ready[:, None], force_w, 0.0))
        self.external_moment_b.copy_(torch.where(self.ready[:, None], moment_b, 0.0))
        self.desired_force_w.copy_(self.mass[:, None]*(acceleration-self.gravity_w)-self.external_force_w)
        collective, rotation = force_attitude(self.desired_force_w, heading, rotation_from_quat(quaternion))
        self.alpha_desired.copy_(attitude_acceleration(quaternion, rotation, rates, reference_rates, self.cfg, self.cfg.tilt_method))
        torque = (matvec(self.inertia, self.alpha_desired)
                  + torch.linalg.cross(rates, matvec(self.inertia, rates)) - self.external_moment_b)
        request = torch.cat((collective, torque), -1)
        # One warm-up tick after reset: hold reset hover, never cancel a previous episode's impact.
        hover = torch.full_like(self.rotor_speed, self.reset_thrust)
        request = torch.where(self.ready[:, None], request, matvec(self.allocator.allocation, hover))
        self.requested_wrench.copy_(request)
        self.requested_rotor_thrust.copy_(matvec(self.allocator.inverse, request))
        commands = torch.where(self.ready[:, None], self.allocator.thrust(request), hover)
        self.allocation_residual.copy_(matvec(self.allocator.allocation, commands) - request)
        self.rotor_speed.add_(self.motor_alpha*((commands / self.task.kf).sqrt() - self.rotor_speed))
        self.ready.fill_(True)
        if self.task.hold_arm:
            target = torch.zeros_like(target)
        self.requested_joint_effort.copy_(self.task.kp*(target-joint_pos) - self.task.kd*joint_vel)
        return commands, self.requested_joint_effort.clamp(-self.task.effort_lim_sim, self.task.effort_lim_sim)
