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

from .borinot_indi import Butterworth, rotation_from_quat
from .borinot_model_wrench import ModelWrenchController
from .borinot_model_wrench_yawless import ModelWrenchYawlessController
from .borinot_wrench import matvec


@dataclass
class ModelWrenchIndiCfg:
    filter_cutoff_hz: float = 12.0

    def validate(self, task):
        if not math.isfinite(self.filter_cutoff_hz) or not 0 < self.filter_cutoff_hz < task.physics_hz / 2:
            raise ValueError("model_wrench_indi.filter_cutoff_hz must be positive and below physics Nyquist.")


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

    def reset(self, ids):
        super().reset(ids)
        self.measurement_ready[ids] = False
        for buffer in (self.previous_rotation, self.previous_velocity_w, self.previous_omega_w,
                       self.previous_effective_row, self.previous_effective_bias, self.previous_joint_effort,
                       self.applied_wrench, self.nominal_wrench, self.raw_disturbance_w,
                       self.disturbance_b, self.measured_acceleration_b):
            buffer[ids] = 0
        self.residual_filter.reset(ids, self.raw_disturbance_w[ids])

    def _correct_model_torque(self, thrust, torque, effective, effective_bias):
        # Called before allocation by either nominal law. Keep its instantaneous
        # arm feedforward; subtract only the filtered observed model mismatch.
        self.nominal_wrench.copy_(torch.cat((thrust[:, None], torque), -1))
        self.previous_effective_row.copy_(effective[:, 3:])
        self.previous_effective_bias.copy_(effective_bias[:, 3:])
        return torque-self.disturbance_b

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
        # This model-wrench plant has G2=0: no unidentified rotor spin inertia.
        # Reconstruct rotor-only torque, never the arm reaction or command wrench.
        self.applied_wrench.copy_(matvec(self.allocator.allocation, self.task.kf*self.rotor_speed.square()))
        self.measurement_ready.fill_(True)
        return commands, effort


class ModelWrenchIndiController(_ModelWrenchIndi, ModelWrenchController):
    """model_wrench action contract plus rotational INDI residual feedback."""


class ModelWrenchYawlessIndiController(_ModelWrenchIndi, ModelWrenchYawlessController):
    """model_wrench_yawless action contract plus rotational INDI feedback."""
