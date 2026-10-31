# SPDX-License-Identifier: BSD-3-Clause
"""Collective/body-rate control, optionally with a second INDI solve.

Scalar layout: [omega_body(3), collective_thrust(1), q_leg(2)]. Joint pairs are optional.
No attitude loop or hip sensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .borinot_angles import clamp_scalar_actions, decode_angle_actions, joint_angle_targets

from .borinot_indi import Butterworth
from .borinot_wrench import WrenchAllocator, matvec


@dataclass
class CbrtControllerCfg:
    rate_gain: tuple = (20.0, 20.0, 8.0)  # 1/s; alpha_des = K_omega (omega_des - omega)
    inertia_scale: float = 1.0  # nominal base-only inertia, not randomized clone properties
    allocation_weights: tuple = (1.0, 10.0, 10.0, 1.0)  # T, tau_x, tau_y, tau_z
    allocation_iterations: int = 128
    filter_cutoff_hz: float = 12.0  # matched gyro/realised-speed filters; cbrt_indi only
    joint_target_limit_degs: tuple = (90.0, 90.0)  # normalized SAC references only

    def validate(self, task):
        for name, size, allow_zero in (("rate_gain", 3, True), ("allocation_weights", 4, False),
                                       ("joint_target_limit_degs", 2, False)):
            values = getattr(self, name)
            if len(values) != size or any(not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero) for v in values):
                raise ValueError(f"cbrt.{name} requires {size} finite {'non-negative' if allow_zero else 'positive'} values.")
        if not math.isfinite(self.inertia_scale) or self.inertia_scale <= 0:
            raise ValueError("cbrt.inertia_scale must be finite and positive.")
        if isinstance(self.allocation_iterations, bool) or not isinstance(self.allocation_iterations, int) or self.allocation_iterations < 1:
            raise ValueError("cbrt.allocation_iterations must be a positive integer.")
        if task.control_mode == "cbrt_indi" and not 0 < self.filter_cutoff_hz < task.physics_hz/2:
            raise ValueError("cbrt.filter_cutoff_hz must be positive and below the physics Nyquist frequency.")
        if task.remove_thrust_clipping:
            raise ValueError("CBRT requires physical rotor bounds: remove_thrust_clipping=False.")
        for name in ("kp", "kd", "motor_time_constant_s"):
            if not math.isfinite(getattr(task, name)) or getattr(task, name) < 0:
                raise ValueError(f"CBRT requires finite non-negative {name}.")
        if task.sac_action_mapping and (len(task.body_rate_scale) != 3 or
                any(not math.isfinite(v) or v <= 0 for v in task.body_rate_scale)):
            raise ValueError("CBRT SAC body_rate_scale requires three finite positive values.")
        for i, name in enumerate(("thigh", "calf")):
            low, high = getattr(task, f"joint_physical_limit_{name}")
            limit = self.joint_target_limit_degs[i]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"joint_physical_limit_{name} requires ordered finite bounds.")
            if task.sac_action_mapping and not low <= -limit < limit <= high:
                raise ValueError(f"CBRT SAC targets must fit inside joint_physical_limit_{name}.")


class CbrtController:
    def __init__(self, task, num_envs, allocation, inertia, hover):
        self.task, self.cfg, self.dt = task, task.cbrt, 1/task.physics_hz
        self.use_indi = task.control_mode == "cbrt_indi"
        self.inertia = inertia*self.cfg.inertia_scale
        self.inertia_inverse = torch.linalg.inv(self.inertia)
        self.hover, self.hover_total = hover, 6*hover
        self.allocator = WrenchAllocator(allocation.expand(num_envs, -1, -1), hover*task.thrust_to_weight, self.cfg)
        self.reset_thrust = min(hover, self.allocator.cap)
        self.rotor_speed = allocation.new_full((num_envs, 6), math.sqrt(self.reset_thrust/task.kf))
        self.stage_one_thrust = torch.zeros_like(self.rotor_speed)
        self.requested_rotor_thrust = torch.zeros_like(self.rotor_speed)
        self.requested_joint_effort = allocation.new_zeros((num_envs, 2))
        self.requested_wrench = allocation.new_zeros((num_envs, 4))
        self.nominal_wrench, self.desired_wrench = torch.zeros_like(self.requested_wrench), torch.zeros_like(self.requested_wrench)
        self.allocation_residual = torch.zeros_like(self.requested_wrench)
        self.alpha_desired = allocation.new_zeros((num_envs, 3))
        self.alpha_feasible, self.angular_acceleration = torch.zeros_like(self.alpha_desired), torch.zeros_like(self.alpha_desired)
        self.filtered_rates, self.filtered_torque = torch.zeros_like(self.alpha_desired), torch.zeros_like(self.alpha_desired)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=allocation.device)
        if self.use_indi:
            self.gyro_filter = Butterworth(self.filtered_rates, self.cfg.filter_cutoff_hz, self.dt)
            self.rotor_filter = Butterworth(self.rotor_speed, self.cfg.filter_cutoff_hz, self.dt)
        self.rate_gain = allocation.new_tensor(self.cfg.rate_gain)
        self.rate_scale = allocation.new_tensor(task.body_rate_scale)
        self.joint_scale = allocation.new_tensor(self.cfg.joint_target_limit_degs)*(math.pi/180)
        self.motor_alpha = 1.0 if task.motor_time_constant_s == 0 else -math.expm1(-self.dt/task.motor_time_constant_s)

    def reset(self, ids):
        self.ready[ids] = False
        self.rotor_speed[ids] = math.sqrt(self.reset_thrust/self.task.kf)
        for buffer in (self.stage_one_thrust, self.requested_rotor_thrust, self.requested_joint_effort,
                       self.requested_wrench, self.nominal_wrench, self.desired_wrench, self.allocation_residual,
                       self.alpha_desired, self.alpha_feasible, self.angular_acceleration,
                       self.filtered_rates, self.filtered_torque):
            buffer[ids] = 0
        if self.use_indi:
            self.gyro_filter.reset(ids, self.filtered_rates[ids])
            self.rotor_filter.reset(ids, self.rotor_speed[ids])

    def observation(self):
        return self.task.kf*self.rotor_speed.square()/self.hover

    def step(self, actions, rates, joint_pos, joint_vel):
        if self.task.sac_action_mapping:
            u = decode_angle_actions(clamp_scalar_actions(actions, self.task, 1.0), self.task)
            reference_rates = u[:, :3]*self.rate_scale
            collective = self.hover_total*(1+torch.where(u[:, 3:4] >= 0,
                (self.task.thrust_to_weight-1)*u[:, 3:4], u[:, 3:4]))
            target = joint_angle_targets(u, self.task, self.joint_scale)
        else:
            actions = decode_angle_actions(actions, self.task)
            reference_rates, collective, target = actions[:, :3], actions[:, 3:4], actions[:, 4:6]
        self.alpha_desired.copy_(self.rate_gain*(reference_rates-rates))
        gyro = torch.linalg.cross(rates, rates@self.inertia.T)
        request = torch.cat((collective, self.alpha_desired@self.inertia.T+gyro), -1)
        self.requested_wrench.copy_(request)
        self.requested_rotor_thrust.copy_(matvec(self.allocator.inverse, request))
        first = self.allocator.thrust(request)
        self.stage_one_thrust.copy_(first)
        self.nominal_wrench.copy_(matvec(self.allocator.allocation, first))
        self.alpha_feasible.copy_((self.nominal_wrench[:, 1:]-gyro)@self.inertia_inverse.T)
        self.desired_wrench.copy_(request)
        commands = first
        if self.use_indi:
            # Initialise fresh-reset gyro states to the current rate without a CPU/GPU sync.
            for state, factor in ((self.gyro_filter.z1, 1-self.gyro_filter.b0),
                                  (self.gyro_filter.z2, self.gyro_filter.b2-self.gyro_filter.a2)):
                state.copy_(torch.where(self.ready[:, None], state, factor*rates))
            filtered_rates = self.gyro_filter.update(rates)
            self.angular_acceleration.copy_(torch.where(self.ready[:, None],
                (filtered_rates-self.filtered_rates)/self.dt, 0.0))
            self.filtered_rates.copy_(filtered_rates)
            # Use realised, lagged speeds, NEVER the just-computed first-stage commands.
            filtered_speeds = self.rotor_filter.update(self.rotor_speed)
            self.filtered_torque.copy_(matvec(self.allocator.allocation[:, 1:], self.task.kf*filtered_speeds.square()))
            self.desired_wrench[:, :1] = self.nominal_wrench[:, :1]
            self.desired_wrench[:, 1:] = self.filtered_torque+(self.alpha_feasible-self.angular_acceleration)@self.inertia.T
            refined = self.allocator.thrust(self.desired_wrench)
            # No completed acceleration sample exists on the first post-reset tick.
            commands = torch.where(self.ready[:, None], refined, first)
            self.desired_wrench.copy_(torch.where(self.ready[:, None], self.desired_wrench, request))
        self.allocation_residual.copy_(matvec(self.allocator.allocation, commands)-self.desired_wrench)
        self.rotor_speed.add_(self.motor_alpha*((commands/self.task.kf).sqrt()-self.rotor_speed))
        self.ready.fill_(True)
        if self.task.hold_arm:
            target = torch.zeros_like(target)
        self.requested_joint_effort.copy_(self.task.kp*(target-joint_pos)-self.task.kd*joint_vel)
        return commands, self.requested_joint_effort.clamp(-self.task.effort_lim_sim, self.task.effort_lim_sim)
