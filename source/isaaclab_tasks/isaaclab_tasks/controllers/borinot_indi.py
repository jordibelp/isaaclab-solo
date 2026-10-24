# SPDX-License-Identifier: BSD-3-Clause
"""Shared Borinot INDI + joint PID. Pure PyTorch; no task/simulator/solver imports."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


INDI_TILT_METHODS = ("geom", "method_1", "method_2")
# Definitions and comparison (geom = method 3):
# /home/jordibelp/Desktop/emmaDocs/2026-10-23-tau-log-exp-comparison.ipynb


@dataclass
class IndiControllerCfg:
    # SAC normalized-action mapping only. PPO sends unclipped physical references.
    acceleration_scale: tuple = (8.0, 8.0, 8.0)  # world m/s^2 per normalized SAC action
    heading_range_degs: float = 180.0  # SAC heading offset half-range from robot_home_quat yaw
    joint_target_limit_degs: tuple = (90.0, 90.0)  # SAC joint target half-ranges
    tilt_gain: float = 150.0  # 1/s^2, multiplying Log(q_error * inverse(q_z)) in radians
    yaw_gain: float = 3.0  # 1/s^2, multiplying the selected twist/log-z vector
    rate_gain: tuple = (20.0, 20.0, 8.0)  # 1/s
    filter_cutoff_hz: float = 12.0
    allocation_weights: tuple = (1.0, 10.0, 10.0, 1.0)  # squared, capacity-normalized wrench error
    allocation_iterations: int = 96
    nonlinear_iterations: int = 6
    rotor_spin_inertia_kg_m2: float = 0.0  # NOT identified: default omits G2 in controller AND physics
    inertia_scale: float = 1.0  # fixed nominal base inertia; arm reactions remain disturbances
    joint_ki: float = 0.5  # N*m/(rad*s); kp/kd use the task's existing fields
    joint_integral_limit: float = 0.3  # rad*s, with conditional anti-windup

    def validate(self, task):
        if task.indi_tilt_method not in INDI_TILT_METHODS:
            raise ValueError(f"Unknown indi_tilt_method {task.indi_tilt_method!r}; choose one of {INDI_TILT_METHODS}.")
        for name, size in (("acceleration_scale", 3), ("rate_gain", 3),
                           ("allocation_weights", 4), ("joint_target_limit_degs", 2)):
            values = getattr(self, name)
            if len(values) != size or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"indi.{name} requires {size} finite positive values.")
        for name in ("heading_range_degs", "tilt_gain", "yaw_gain", "inertia_scale", "joint_integral_limit"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"indi.{name} must be finite and positive.")
        for name in ("rotor_spin_inertia_kg_m2", "joint_ki"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"indi.{name} must be finite and non-negative.")
        for name in ("allocation_iterations", "nonlinear_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"indi.{name} must be a positive integer.")
        if not 0 < self.filter_cutoff_hz < task.physics_hz / 2:
            raise ValueError("indi.filter_cutoff_hz must lie below the physics Nyquist frequency.")
        if task.remove_thrust_clipping:
            raise ValueError("indi requires remove_thrust_clipping=False: allocation uses physical rotor bounds.")
        if not all(math.isfinite(v) and v >= 0 for v in (task.kp, task.kd, task.motor_time_constant_s)):
            raise ValueError("indi requires finite non-negative kp, kd and motor_time_constant_s.")
        for i, name in enumerate(("thigh", "calf")):
            low, high = getattr(task, f"joint_physical_limit_{name}")
            limit = self.joint_target_limit_degs[i]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"joint_physical_limit_{name} must contain ordered finite degrees.")
            if task.sac_action_mapping and not low <= -limit < limit <= high:
                raise ValueError(f"indi SAC joint targets must fit inside joint_physical_limit_{name}.")


def rotation_from_quat(q):
    """Active body-to-world rotation for Isaac's scalar-first unit quaternions."""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)), -1).reshape(-1, 3, 3)


def thrust_attitude(acceleration, heading, mass):
    """World acceleration (gravity excluded) + heading -> collective N and desired R_WB."""
    specific_force = acceleration.clone()
    specific_force[:, 2] += 9.81
    magnitude = specific_force.norm(dim=-1, keepdim=True)
    up = torch.zeros_like(specific_force)
    up[:, 2] = 1
    z = torch.where(magnitude > 1e-6, specific_force / magnitude.clamp_min(1e-6), up)
    heading_x = torch.stack((heading.cos(), heading.sin(), torch.zeros_like(heading)), -1)
    y = torch.linalg.cross(z, heading_x)
    # At thrust/heading collinearity, select the perpendicular horizontal heading instead.
    heading_y = torch.stack((-heading.sin(), heading.cos(), torch.zeros_like(heading)), -1)
    y = torch.where(y.norm(dim=-1, keepdim=True) > 1e-6, y, torch.linalg.cross(z, heading_y))
    y = torch.nn.functional.normalize(y, dim=-1)
    x = torch.linalg.cross(y, z)
    return mass * magnitude, torch.stack((x, y, z), -1)


def quaternion_from_rotation(rotation):
    """Scalar-first unit quaternion, including at pi; largest component is positive.

    Each row below is 4*q_i*q. Select the largest q_i^2 before normalizing,
    avoiding division by a small scalar component near a 180-degree rotation.
    """
    r = rotation
    a, b, c = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    wx, wy, wz = r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]
    xy, xz, yz = r[..., 0, 1]+r[..., 1, 0], r[..., 0, 2]+r[..., 2, 0], r[..., 1, 2]+r[..., 2, 1]
    squares = torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1)
    candidates = torch.stack((torch.stack((squares[..., 0], wx, wy, wz), -1),
                              torch.stack((wx, squares[..., 1], xy, xz), -1),
                              torch.stack((wy, xy, squares[..., 2], yz), -1),
                              torch.stack((wz, xz, yz, squares[..., 3]), -1)), -2)
    index = squares.argmax(-1)[..., None, None].expand(*squares.shape[:-1], 1, 4)
    return torch.nn.functional.normalize(candidates.gather(-2, index).squeeze(-2), dim=-1)


def quaternion_log(quaternion):
    """Principal SO(3) Log: full-angle rotation vector in radians, NOT half-angle log.

    q and -q give the same result. At exactly pi, choose the representative
    with its largest-magnitude vector component positive (the axis is not unique).
    """
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, v = q[..., :1], q[..., 1:]
    largest = v.gather(-1, v.abs().argmax(-1, keepdim=True))
    flip = (w < 0) | ((w == 0) & (largest < 0))
    q = torch.where(flip, -q, q)
    w, v = q[..., :1], q[..., 1:]
    norm = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w)
    # theta/sin(theta/2) -> 2 at identity. Keep both torch.where branches finite.
    scale = torch.where(norm > 1e-6, angle / norm.clamp_min(1e-6), 2 + norm.square()/3)
    return v * scale


def quaternion_exp(vector):
    """SO(3) Exp of a full-angle rotation vector to a scalar-first unit quaternion."""
    angle = vector.norm(dim=-1, keepdim=True)
    # torch.sinc(x)=sin(pi*x)/(pi*x), including its exact limit at zero.
    return torch.cat(((angle/2).cos(), .5*torch.sinc(angle/(2*math.pi))*vector), -1)


def quaternion_multiply(left, right):
    """Hamilton product for scalar-first quaternions."""
    lw, lv, rw, rv = left[..., :1], left[..., 1:], right[..., :1], right[..., 1:]
    return torch.cat((lw*rw - (lv*rv).sum(-1, keepdim=True),
                      lw*rv + rw*lv + torch.linalg.cross(lv, rv)), -1)


def attitude_error_vectors(error, method="geom"):
    """Return full-angle (tilt, twist) vectors in current body axes.

    geom: geometric swing/twist split, with q_z=normalize((w,0,0,z)).
    method_1: Log(q_error*Exp(-project_z(Log(q_error)))); retains residual z.
    method_2: project_xy(Log(q_error)), with project_z(Log(q_error)) for twist.
    See the notebook path above for the three definitions.
    """
    if method == "geom":
        error = torch.nn.functional.normalize(error, dim=-1)
        w, z = error[..., :1], error[..., 3:]
        norm = torch.hypot(w, z)
        epsilon = torch.finfo(error.dtype).eps
        twist = torch.cat((w, torch.zeros_like(error[..., 1:3]), z), -1) / norm.clamp_min(epsilon)
        identity = torch.zeros_like(error)
        identity[..., 0] = 1
        # At inverted thrust axes the twist is undefined. Choose zero twist;
        # the error itself gives a finite half-turn tilt, up to roundoff.
        twist = torch.where(norm > epsilon, twist, identity)
        inverse = torch.cat((twist[..., :1], -twist[..., 1:]), -1)
        return quaternion_log(quaternion_multiply(error, inverse)), quaternion_log(twist)
    if method not in INDI_TILT_METHODS:
        raise ValueError(f"Unknown indi_tilt_method {method!r}; choose one of {INDI_TILT_METHODS}.")
    tau = quaternion_log(error)
    tau_z = torch.cat((torch.zeros_like(tau[..., :2]), tau[..., 2:]), -1)
    if method == "method_2":
        return tau-tau_z, tau_z
    # Exp(-tau_z) = inverse(Exp(tau_z)) for these unit quaternions.
    residual = quaternion_multiply(error, quaternion_exp(-tau_z))
    return quaternion_log(residual), tau_z


def attitude_acceleration(quaternion, desired_rotation, rates, reference_rates, cfg, method="geom"):
    """Selectable full-angle attitude feedback in CURRENT BODY axes; alpha_r=0."""
    error = quaternion_from_rotation(rotation_from_quat(quaternion).mT @ desired_rotation)
    tilt, twist = attitude_error_vectors(error, method)
    return (cfg.tilt_gain * tilt + cfg.yaw_gain * twist
            + rates.new_tensor(cfg.rate_gain) * (reference_rates - rates))  # alpha_r=0


class Butterworth:
    """Second-order Butterworth, bilinear transform, transposed direct form II."""

    def __init__(self, initial, cutoff_hz, dt):
        k = math.tan(math.pi * cutoff_hz * dt)
        norm = 1 / (1 + math.sqrt(2)*k + k*k)
        self.b0 = k*k*norm
        self.b1, self.b2 = 2*self.b0, self.b0
        self.a1, self.a2 = 2*(k*k - 1)*norm, (1 - math.sqrt(2)*k + k*k)*norm
        self.z1, self.z2 = torch.empty_like(initial), torch.empty_like(initial)
        self.reset(slice(None), initial)

    def reset(self, ids, value):
        # Steady-state initialization prevents a reset impulse in either derivative.
        self.z1[ids] = (1 - self.b0) * value
        self.z2[ids] = (self.b2 - self.a2) * value

    def update(self, value):
        output = self.b0 * value + self.z1
        self.z1.copy_(self.b1*value - self.a1*output + self.z2)
        self.z2.copy_(self.b2*value - self.a2*output)
        return output


def box_qp(hessian, linear, initial, lipschitz, iterations):
    """Fixed-budget accelerated projected gradient on 0<=x<=1 (fully batched)."""
    x = initial.clamp(0, 1)
    y, t = x, 1.0
    for _ in range(iterations):
        gradient = y @ hessian - linear
        next_x = (y - gradient / lipschitz).clamp(0, 1)
        next_t = (1 + math.sqrt(1 + 4*t*t)) / 2
        y = next_x + ((t - 1) / next_t) * (next_x - x)
        x, t = next_x, next_t
    return x


class RotorAllocator:
    """Six bounded rotors; normalized wrench least squares, not inverse-then-clip."""

    def __init__(self, allocation, cap, kf, spin, dt, cfg):
        self.allocation, self.cap, self.kf, self.dt, self.cfg = allocation, cap, kf, dt, cfg
        self.speed_max = math.sqrt(cap / kf)
        self.scale = allocation.abs().sum(-1) * cap
        self.weight = allocation.new_tensor(cfg.allocation_weights).sqrt()
        self.a = allocation * cap / self.scale[:, None] * self.weight[:, None]
        self.inverse = torch.linalg.pinv(self.a)
        self.hessian = self.a.T @ self.a
        self.lipschitz = torch.linalg.eigvalsh(self.hessian)[-1]
        self.g2 = torch.zeros_like(allocation)
        self.g2[3] = spin * cfg.rotor_spin_inertia_kg_m2
        self.b = self.g2 * (self.speed_max / dt) / self.scale[:, None] * self.weight[:, None]

    def thrust(self, wrench):
        target = wrench / self.scale * self.weight
        initial = target @ self.inverse.T
        u = box_qp(self.hessian, target @ self.a, initial, self.lipschitz, self.cfg.allocation_iterations)
        return u * self.cap

    def speed(self, wrench, previous_command):
        speed = (self.thrust(wrench) / self.kf).sqrt()
        if self.cfg.rotor_spin_inertia_kg_m2 == 0:
            return speed
        # Eq. (16) generalized from four equations/four rotors to four equations/six rotors.
        # Bounded, damped Gauss-Newton in normalized speed. Solve only 4x4 systems,
        # using solve_ex to avoid a CUDA->CPU error-check synchronization each iteration.
        x, previous = speed / self.speed_max, previous_command / self.speed_max
        target = wrench / self.scale * self.weight
        for _ in range(self.cfg.nonlinear_iterations):
            value = x.square() @ self.a.T + (x - previous) @ self.b.T
            jacobian = 2*self.a[None] * x[:, None, :] + self.b[None]
            residual = value - target
            gradient = (jacobian.mT @ residual.unsqueeze(-1)).squeeze(-1)
            free = ((x > 1e-6) & (x < 1 - 1e-6)) | ((x <= 1e-6) & (gradient < 0)) | ((x >= 1-1e-6) & (gradient > 0))
            jacobian = jacobian * free[:, None, :]
            normal = jacobian @ jacobian.mT + 1e-6*torch.eye(4, device=x.device, dtype=x.dtype)
            solution, _ = torch.linalg.solve_ex(normal, residual.unsqueeze(-1))
            delta = -(jacobian.mT @ solution).squeeze(-1)
            candidate = (x + delta).clamp(0, 1)
            # Keep only improving steps; infeasible wrench requests remain explicitly residuals.
            best, cost = x, (value - target).square().sum(-1)
            for step in (1.0, 0.5, 0.25, 0.125):
                trial = x + step * (candidate - x)
                residual = trial.square() @ self.a.T + (trial - previous) @ self.b.T - target
                trial_cost = residual.square().sum(-1)
                best = torch.where((trial_cost < cost)[:, None], trial, best)
                cost = torch.minimum(cost, trial_cost)
            x = best
        return x * self.speed_max


class IndiController:
    def __init__(self, task, num_envs, allocation, inertia, mass, hover):
        self.cfg, self.task, self.dt = task.indi, task, 1 / task.physics_hz
        self.inertia = inertia * task.indi.inertia_scale
        self.inertia_inverse = torch.linalg.inv(self.inertia)
        self.mass, self.hover = mass, hover
        self.allocator = RotorAllocator(allocation, hover * task.thrust_to_weight, task.kf,
                                        allocation.new_tensor(task.rotor_spin), self.dt, self.cfg)
        zeros = allocation.new_zeros((num_envs, 3))
        self.rotor_speed = allocation.new_full((num_envs, 6), math.sqrt(hover / task.kf))
        self.command_speed = self.rotor_speed.clone()
        self.gyro_filter = Butterworth(zeros, self.cfg.filter_cutoff_hz, self.dt)
        self.rotor_filter = Butterworth(self.rotor_speed, self.cfg.filter_cutoff_hz, self.dt)
        self.filtered_rates, self.filtered_speeds = zeros.clone(), self.rotor_speed.clone()
        self.angular_acceleration = zeros.clone()
        self.alpha_desired, self.alpha_feasible = zeros.clone(), zeros.clone()
        self.nominal_wrench, self.desired_wrench = allocation.new_zeros((num_envs, 4)), allocation.new_zeros((num_envs, 4))
        self.allocation_residual = allocation.new_zeros((num_envs, 4))
        self.transient_torque = zeros.clone()
        self.joint_integral = allocation.new_zeros((num_envs, 2))
        self.accel_scale = allocation.new_tensor(self.cfg.acceleration_scale)
        self.rate_scale = allocation.new_tensor(task.body_rate_scale)
        self.joint_scale = allocation.new_tensor(self.cfg.joint_target_limit_degs) * (math.pi / 180)
        w, x, y, z = task.robot_home_quat
        self.home_yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        self.motor_alpha = 1.0 if task.motor_time_constant_s == 0 else -math.expm1(-self.dt / task.motor_time_constant_s)

    def reset(self, ids):
        self.rotor_speed[ids] = math.sqrt(self.hover / self.task.kf)
        self.command_speed[ids] = self.rotor_speed[ids]
        self.filtered_speeds[ids] = self.rotor_speed[ids]
        for buffer in (self.filtered_rates, self.angular_acceleration, self.alpha_desired, self.alpha_feasible,
                       self.nominal_wrench, self.desired_wrench, self.allocation_residual,
                       self.transient_torque, self.joint_integral):
            buffer[ids] = 0
        self.gyro_filter.reset(ids, self.filtered_rates[ids])
        self.rotor_filter.reset(ids, self.rotor_speed[ids])

    def observation(self):
        return torch.cat((self.task.kf * self.rotor_speed.square() / self.hover,
                          self.joint_integral / self.cfg.joint_integral_limit), -1)

    def step(self, actions, quaternion, rates, joint_pos, joint_vel):
        if self.task.sac_action_mapping:
            u = actions.clamp(-1, 1)
            acceleration, reference_rates = u[:, :3] * self.accel_scale, u[:, 3:6] * self.rate_scale
            heading = self.home_yaw + math.radians(self.cfg.heading_range_degs) * u[:, 6]
            target = u[:, 7:9] * self.joint_scale
        else:
            # PPO: world m/s^2, body rad/s, heading offset rad, absolute joint rad.
            # Do not bound references here; allocation and PID bound actuator outputs.
            acceleration, reference_rates = actions[:, :3], actions[:, 3:6]
            heading, target = self.home_yaw + actions[:, 6], actions[:, 7:9]
        collective, desired_rotation = thrust_attitude(acceleration, heading, self.mass)
        self.alpha_desired.copy_(attitude_acceleration(quaternion, desired_rotation, rates,
                                                      reference_rates, self.cfg, self.task.indi_tilt_method))
        gyro = torch.linalg.cross(rates, rates @ self.inertia.T)
        requested = torch.cat((collective, self.alpha_desired @ self.inertia.T + gyro), -1)
        nominal = self.allocator.thrust(requested) @ self.allocator.allocation.T
        self.nominal_wrench.copy_(nominal)
        self.alpha_feasible.copy_((nominal[:, 1:] - gyro) @ self.inertia_inverse.T)
        filtered_rates = self.gyro_filter.update(rates)
        filtered_speeds = self.rotor_filter.update(self.rotor_speed)
        self.angular_acceleration.copy_((filtered_rates - self.filtered_rates) / self.dt)
        torque_f = (self.task.kf * filtered_speeds.square()) @ self.allocator.allocation[1:].T
        torque_f += ((filtered_speeds - self.filtered_speeds) / self.dt) @ self.allocator.g2[1:].T
        self.filtered_rates.copy_(filtered_rates)
        self.filtered_speeds.copy_(filtered_speeds)
        self.desired_wrench[:, :1] = nominal[:, :1]
        self.desired_wrench[:, 1:] = torque_f + (self.alpha_feasible - self.angular_acceleration) @ self.inertia.T
        command = self.allocator.speed(self.desired_wrench, self.command_speed)
        predicted = self.task.kf * command.square() @ self.allocator.allocation.T
        predicted += ((command - self.command_speed) / self.dt) @ self.allocator.g2.T
        self.allocation_residual.copy_(predicted - self.desired_wrench)
        self.command_speed.copy_(command)
        next_speed = self.rotor_speed + self.motor_alpha * (command - self.rotor_speed)
        self.transient_torque.copy_(((next_speed - self.rotor_speed) / self.dt) @ self.allocator.g2[1:].T)
        self.rotor_speed.copy_(next_speed)

        if self.task.hold_arm:
            target = torch.zeros_like(target)
        error = target - joint_pos
        integral = (self.joint_integral + self.dt*error).clamp(-self.cfg.joint_integral_limit, self.cfg.joint_integral_limit)
        effort = self.task.kp*error - self.task.kd*joint_vel + self.cfg.joint_ki*integral
        winding_up = ((effort > self.task.effort_lim_sim) & (error > 0)) | ((effort < -self.task.effort_lim_sim) & (error < 0))
        self.joint_integral.copy_(torch.where(winding_up, self.joint_integral, integral))
        effort = (self.task.kp*error - self.task.kd*joint_vel + self.cfg.joint_ki*self.joint_integral).clamp(
            -self.task.effort_lim_sim, self.task.effort_lim_sim)
        return self.task.kf * command.square(), effort
