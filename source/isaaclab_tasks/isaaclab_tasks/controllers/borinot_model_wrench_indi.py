# SPDX-License-Identifier: BSD-3-Clause
"""Arm-aware model prediction plus a causal, rotation-only INDI residual.

The acceleration estimate uses completed-interval world CoM velocity and gyro
increments. Every product uses the saved interval-start model and bounded joint
effort; the applied rotor torque is saved AFTER the single motor-lag update.
Filter the complete residual in fixed world axes, then return it to current body
axes. No hip wrench, policy acceleration, or randomized model readback is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .borinot_indi import Butterworth, invert_rotor_speed, rotation_from_quat
from .borinot_model_wrench import ModelWrenchController
from .borinot_model_wrench_yawless import ModelWrenchYawlessController
from .borinot_wrench import matvec


@dataclass
class ModelWrenchIndiCfg:
    filter_cutoff_hz: float = 12.0

    def validate(self, task):
        if not math.isfinite(self.filter_cutoff_hz) or not 0 < self.filter_cutoff_hz < task.physics_hz / 2:
            raise ValueError("model_wrench_indi.filter_cutoff_hz must be positive and below physics Nyquist.")
        inertia = task.indi.rotor_spin_inertia_kg_m2
        if not math.isfinite(inertia) or inertia < 0:
            raise ValueError("indi.rotor_spin_inertia_kg_m2 must be finite and non-negative.")
        iterations = task.indi.nonlinear_iterations
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("indi.nonlinear_iterations must be a positive integer.")


class _ModelWrenchIndi:
    """Shared correction for the heading and heading-free nominal controllers."""

    def __init__(self, task, allocation, model, hover):
        super().__init__(task, allocation, model, hover)
        n = len(allocation)
        self.measurement_ready = torch.zeros(n, dtype=torch.bool, device=allocation.device)
        self.previous_rotation = allocation.new_zeros((n, 3, 3))
        self.previous_velocity_w = allocation.new_zeros((n, 3))
        self.previous_omega_w = allocation.new_zeros((n, 3))
        self.previous_effective_row = allocation.new_zeros((n, 3, 6))
        self.previous_effective_bias = allocation.new_zeros((n, 3))
        self.previous_joint_effort = allocation.new_zeros((n, 2))
        self.applied_wrench = allocation.new_zeros((n, 4))
        self.nominal_wrench = allocation.new_zeros((n, 4))
        self.raw_disturbance_w = allocation.new_zeros((n, 3))
        self.disturbance_b = allocation.new_zeros((n, 3))
        self.measured_acceleration_b = allocation.new_zeros((n, 6))
        self.residual_filter = Butterworth(self.raw_disturbance_w, task.model_wrench_indi.filter_cutoff_hz, self.dt)
        self.command_speed = self.rotor_speed.clone()
        self.transient_torque = allocation.new_zeros((n, 3))
        self.g2 = allocation.new_zeros((4, 6))
        self.g2[3] = allocation.new_tensor(task.rotor_spin) * task.indi.rotor_spin_inertia_kg_m2
        self.speed_max = math.sqrt(self.allocator.cap / task.kf)
        self.speed_effectiveness = (self.g2 * (self.speed_max/self.dt)
                                   / self.allocator.scale[..., None] * self.allocator.weight[:, None])

    def reset(self, ids):
        super().reset(ids)
        self.command_speed[ids] = self.rotor_speed[ids]
        self.measurement_ready[ids] = False
        for buffer in (self.previous_rotation, self.previous_velocity_w, self.previous_omega_w,
                       self.previous_effective_row, self.previous_effective_bias, self.previous_joint_effort,
                       self.applied_wrench, self.nominal_wrench, self.raw_disturbance_w,
                       self.disturbance_b, self.measured_acceleration_b, self.transient_torque):
            buffer[ids] = 0
        self.residual_filter.reset(ids, self.raw_disturbance_w[ids])

    def _correct_model_torque(self, thrust, torque, effective, effective_bias):
        # Called before allocation by either nominal law. Keep its instantaneous
        # arm feedforward; subtract only the filtered observed model mismatch.
        self.nominal_wrench.copy_(torch.cat((thrust[:, None], torque), -1))
        self.previous_effective_row.copy_(effective[:, 3:])
        self.previous_effective_bias.copy_(effective_bias[:, 3:])
        return torque-self.disturbance_b

    def _allocate_rotors(self):
        if self.task.indi.rotor_spin_inertia_kg_m2 == 0:
            return super()._allocate_rotors()
        self.requested_rotor_thrust.copy_(matvec(self.allocator.inverse, self.requested_wrench))
        initial = (self.allocator.thrust(self.requested_wrench)/self.task.kf).sqrt()/self.speed_max
        command = self.speed_max * invert_rotor_speed(
            initial, self.command_speed/self.speed_max, self.allocator.a, self.speed_effectiveness,
            self.requested_wrench/self.allocator.scale*self.allocator.weight, self.task.indi.nonlinear_iterations)
        commands = self.task.kf*command.square()
        predicted = (matvec(self.allocator.allocation, commands)
                     + ((command-self.command_speed)/self.dt) @ self.g2.T)
        self.allocation_residual.copy_(predicted-self.requested_wrench)
        self.command_speed.copy_(command)
        next_speed = self.rotor_speed + self.motor_alpha*(command-self.rotor_speed)
        self.transient_torque.copy_(((next_speed-self.rotor_speed)/self.dt) @ self.g2[1:].T)
        self.rotor_speed.copy_(next_speed)
        return commands

    def step(self, actions, quaternion, rates, joint_pos, joint_vel, velocity_w):
        rotation = rotation_from_quat(quaternion)
        omega_w = matvec(rotation, rates)
        increments_w = torch.cat(((velocity_w-self.previous_velocity_w)/self.dt,
                                  (omega_w-self.previous_omega_w)/self.dt), -1)
        previous = self.previous_rotation.mT
        acceleration_b = torch.cat((matvec(previous, increments_w[:, :3]),
                                    matvec(previous, increments_w[:, 3:])), -1)
        self.measured_acceleration_b.copy_(torch.where(self.measurement_ready[:, None], acceleration_b, 0.0))
        raw_b = (matvec(self.previous_effective_row, self.measured_acceleration_b)
                 + self.previous_effective_bias-self.applied_wrench[:, 1:])
        self.raw_disturbance_w.copy_(torch.where(self.measurement_ready[:, None],
            matvec(self.previous_rotation, raw_b), 0.0))
        self.disturbance_b.copy_(matvec(rotation.mT, self.residual_filter.update(self.raw_disturbance_w)))
        commands, effort = super().step(actions, quaternion, rates, joint_pos, joint_vel)
        self.previous_rotation.copy_(rotation)
        self.previous_velocity_w.copy_(velocity_w)
        self.previous_omega_w.copy_(omega_w)
        self.previous_joint_effort.copy_(effort)
        # Reconstruct rotor-only torque, never the arm reaction or command wrench.
        self.applied_wrench.copy_(matvec(self.allocator.allocation, self.task.kf*self.rotor_speed.square()))
        self.applied_wrench[:, 1:] += self.transient_torque
        self.measurement_ready.fill_(True)
        return commands, effort


class ModelWrenchIndiController(_ModelWrenchIndi, ModelWrenchController):
    """model_wrench action contract plus rotational INDI residual feedback."""


class ModelWrenchYawlessIndiController(_ModelWrenchIndi, ModelWrenchYawlessController):
    """model_wrench_yawless action contract plus rotational INDI feedback."""
