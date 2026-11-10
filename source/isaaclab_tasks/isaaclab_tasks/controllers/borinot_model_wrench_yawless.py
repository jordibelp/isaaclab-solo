# SPDX-License-Identifier: BSD-3-Clause
"""Model-wrench inverse dynamics with a shortest-axis tilt error and no heading.

Scalar policy order: [a_world(3), omega_body(3), q_leg(2)]. Body-z rate is
still commandable; yaw ANGLE has no target or restoring gain. Joint (cos,sin)
pairs may expand the last two slots. The original model controller is untouched.
"""
from __future__ import annotations

import math

import torch

from .borinot_angles import clamp_scalar_actions, decode_angle_actions, joint_angle_targets
from .borinot_indi import rotation_from_quat
from .borinot_model_wrench import ModelWrenchController
from .borinot_wrench import matvec


def minimal_tilt_error(force_w, rotation):
    """Log(R.T @ R_nearest) in CURRENT BODY axes; full angle, identically zero z.

    R_nearest minimizes principal SO(3) distance subject to R_nearest*e3=d.
    In body axes v=e3 cross (R.T*d), theta=atan2(norm(v), (R.T*d).z).
    At a zero virtual force keep the current axis. At the antipode, within
    one machine epsilon in cross-product norm, choose a body-x pi turn.
    The antipodal choice cannot be both unique and globally continuous.
    """
    magnitude = force_w.norm(dim=-1, keepdim=True)
    direction = matvec(rotation.mT, force_w / magnitude.clamp_min(1e-6))
    v = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), -1)
    sine = v.norm(dim=-1, keepdim=True)
    cosine = direction[:, 2:3].clamp(-1, 1)
    epsilon = torch.finfo(force_w.dtype).eps
    theta = torch.atan2(sine, cosine)
    scale = torch.where(sine > epsilon, theta / sine.clamp_min(epsilon), 1.0)
    error = scale*v
    half_turn = torch.zeros_like(error)
    half_turn[:, 0] = math.pi
    error = torch.where((sine <= epsilon) & (cosine < 0), half_turn, error)
    return torch.where(magnitude > 1e-6, error, 0.0)


class ModelWrenchYawlessController(ModelWrenchController):
    """Reuse calibration, buffers, reset and allocation; replace only the reference law.

    The explicit step keeps ModelWrenchController.step byte-for-byte unchanged.
    Model elimination, constrained inversion, allocation and one speed-lag update
    use the same operations as that baseline. See MODEL_WRENCH_YAWLESS_CONTROL.md.
    """

    def step(self, actions, quaternion, rates, joint_pos, joint_vel):
        if self.task.sac_action_mapping:
            u = decode_angle_actions(clamp_scalar_actions(actions, self.task, 1.0), self.task)
            acceleration = u[:, :3]*self.accel_scale
            reference_rates = u[:, 3:6]*self.rate_scale
            target = joint_angle_targets(u, self.task, self.joint_scale)
        else:
            u = decode_angle_actions(actions, self.task)
            acceleration, reference_rates, target = u[:, :3], u[:, 3:6], u[:, 6:8]
        if self.task.hold_arm:
            target = torch.zeros_like(target)
        self.requested_joint_effort.copy_(self.task.kp*(target-joint_pos)-self.task.kd*joint_vel)
        effort = self.requested_joint_effort.clamp(-self.task.effort_lim_sim, self.task.effort_lim_sim)
        rotation = rotation_from_quat(quaternion)
        gravity_b = matvec(rotation.mT, self.gravity_w)
        desired_a = matvec(rotation.mT, acceleration)
        matrix, bias = self.model.dynamics(joint_pos, joint_vel, rates, gravity_b)
        self.mass_matrix.copy_(matrix)
        self.bias.copy_(bias)
        solved = torch.linalg.solve(matrix[:, 6:, 6:], torch.cat(
            (matrix[:, 6:, :6], (effort-bias[:, 6:]).unsqueeze(-1)), -1))
        effective = matrix[:, :6, :6] - matrix[:, :6, 6:] @ solved[:, :, :6]
        effective_bias = bias[:, :6] + matvec(matrix[:, :6, 6:], solved[:, :, 6])
        # Same explicit virtual-force direction choice as model_wrench: default
        # alpha=0, or the previous desired alpha when that family option is enabled.
        self.desired_force_w.copy_(self._virtual_force_w(rotation, effective, effective_bias, desired_a))
        tilt = minimal_tilt_error(self.desired_force_w, rotation)
        self.alpha_desired.copy_(self.cfg.tilt_gain*tilt
            + rates.new_tensor(self.cfg.rate_gain)*(reference_rates-rates))
        self._remember_alpha_desired()
        # Same current-thrust-axis inverse dynamics; no projection removes gyro,
        # arm coupling or the torque needed to realize the requested body-z rate.
        c = matvec(effective[:, :3, 3:], self.alpha_desired) + effective_bias[:, :3]
        axis = torch.zeros_like(c)
        axis[:, 2] = 1
        response = torch.linalg.solve(effective[:, :3, :3], torch.stack((axis, c), -1))
        thrust = (desired_a[:, 2]+response[:, 2, 1])/response[:, 2, 0]
        actual_a = response[:, :, 0]*thrust[:, None] - response[:, :, 1]
        torque = (matvec(effective[:, 3:, :3], actual_a)
                  + matvec(effective[:, 3:, 3:], self.alpha_desired) + effective_bias[:, 3:])
        self.predicted_acceleration_b.copy_(actual_a)
        base_a = torch.cat((actual_a, self.alpha_desired), -1)
        self.predicted_joint_acceleration.copy_(solved[:, :, 6]-matvec(solved[:, :, :6], base_a))
        corrected = self._correct_model_torque(thrust, torque, effective, effective_bias)
        self.requested_wrench.copy_(torch.cat((thrust[:, None], corrected), -1))
        self.external_force_w.copy_(matvec(rotation, self.mass[:, None]*(actual_a-gravity_b)-axis*thrust[:, None]))
        self.external_moment_b.copy_(matvec(self.inertia, self.alpha_desired)
            + torch.linalg.cross(rates, matvec(self.inertia, rates))-torque)
        commands = self._allocate_rotors()
        self.ready.fill_(True)
        return commands, effort
