# SPDX-License-Identifier: BSD-3-Clause
"""Contact-free floating-base inverse dynamics. No simulator or load-cell inputs.

Runtime inputs: base attitude/gyro, joint encoders, and the policy references.
Model parameters are fixed calibration data, not randomized plant readbacks.
See direct/borinot/MODEL_WRENCH_CONTROL.md and the accompanying derivation.
"""
from __future__ import annotations

import math

import torch

from .borinot_angles import clamp_scalar_actions, decode_angle_actions, joint_angle_targets
from .borinot_indi import attitude_acceleration, rotation_from_quat
from .borinot_wrench import WrenchController, force_attitude, matvec


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


class FloatingArmModel:
    """Small rigid tree with two revolute DOFs and optional fixed tool bodies.

    Coordinates: physical linear acceleration of the BASE CoM in base axes,
    physical base angular acceleration in base axes, then two joint accelerations.
    The first three entries are NOT d(body-frame linear velocity)/dt.
    Each body's inertia is about its own CoM, in that body's link axes.
    Joint transforms follow USD: parent*T0*Rot(axis,q)*inverse(T1).
    """

    def __init__(self, description, *, device="cpu", dtype=torch.float32):
        self.description = description
        self.tensor = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
        self.eye = torch.eye(3, device=device, dtype=dtype)
        self.bodies = []
        dofs = []
        for i, spec in enumerate(description["bodies"]):
            body = dict(spec)
            if not math.isfinite(body["mass"]) or body["mass"] <= 0:
                raise ValueError("Model body masses must be finite and positive.")
            for key in ("com", "inertia"):
                body[key] = self.tensor(body[key])
            inertia = body["inertia"]
            if (body["com"].shape != (3,) or inertia.shape != (3, 3)
                    or not torch.isfinite(body["com"]).all() or not torch.isfinite(inertia).all()
                    or not torch.allclose(inertia, inertia.mT, atol=1e-7, rtol=0)
                    or not (torch.linalg.eigvalsh(inertia) > 0).all()):
                raise ValueError("Model requires finite CoMs and symmetric positive-definite own-CoM inertias.")
            if i:
                if not 0 <= body["parent"] < i or body["dof"] not in (-1, 0, 1):
                    raise ValueError("Model bodies must be in parent-first order with DOFs -1, 0, or 1.")
                for key in ("p0", "r0", "p1", "r1", "axis"):
                    body[key] = self.tensor(body[key])
                if body["dof"] >= 0:
                    dofs.append(body["dof"])
                    if not torch.allclose(body["axis"].norm(), self.tensor(1.), atol=1e-6):
                        raise ValueError("Revolute model axes must have unit length.")
                body["axis_skew"] = skew(body["axis"])
                body["ancestors"] = self.bodies[body["parent"]]["ancestors"] + (
                    [body["dof"]] if body["dof"] >= 0 else [])
            else:
                body["ancestors"] = []
            self.bodies.append(body)
        if sorted(dofs) != [0, 1]:
            raise ValueError("FloatingArmModel requires exactly the two configured revolute joints.")
        self.armature = self.tensor(description["armature"])
        if self.armature.shape != (2,) or not torch.isfinite(self.armature).all() or (self.armature < 0).any():
            raise ValueError("Model armature must contain two finite nonnegative generalized joint inertias.")
        self.total_mass = sum(b["mass"] for b in self.bodies)

    def kinematics(self, q, dq, omega):
        """Return link CoM Jacobians and zero-generalized-acceleration biases.

        Biases retain base centripetal, joint centripetal, and Coriolis terms.
        All returned vectors/tensors use base axes. No numerical differentiation.
        """
        n = len(q)
        eye = self.eye.expand(n, -1, -1)
        zero = torch.zeros_like(omega)
        axes, anchors, states = {}, {}, []
        for i, b in enumerate(self.bodies):
            if not i:
                rotation = eye
                position = -b["com"].expand(n, -1)
                w, alpha = omega, zero
                acc = torch.linalg.cross(w, torch.linalg.cross(w, position))
            else:
                parent = states[b["parent"]]
                p, rp, wp, ap, bp = (parent[k] for k in ("origin", "rotation", "omega", "alpha_bias", "origin_bias"))
                joint_rotation = rp @ b["r0"]
                offset = matvec(rp, b["p0"])
                anchor = p + offset
                anchor_acc = bp + torch.linalg.cross(ap, offset) + torch.linalg.cross(wp, torch.linalg.cross(wp, offset))
                j = b["dof"]
                if j >= 0:
                    axis = matvec(joint_rotation, b["axis"])
                    axes[j], anchors[j] = axis, anchor
                    k = b["axis_skew"]
                    angle = q[:, j, None, None]
                    turn = eye + angle.sin()*k + (1-angle.cos())*(k@k)
                    rotation = joint_rotation @ turn @ b["r1"].mT
                    relative_w = axis*dq[:, j:j+1]
                    w = wp + relative_w
                    alpha = ap + torch.linalg.cross(wp, relative_w)
                else:
                    rotation = joint_rotation @ b["r1"].mT
                    w, alpha = wp, ap
                child_to_joint = matvec(rotation, b["p1"])
                position = anchor - child_to_joint
                acc = (anchor_acc - torch.linalg.cross(alpha, child_to_joint)
                       - torch.linalg.cross(w, torch.linalg.cross(w, child_to_joint)))
            offset = matvec(rotation, b["com"])
            com = position + offset
            bias = acc + torch.linalg.cross(alpha, offset) + torch.linalg.cross(w, torch.linalg.cross(w, offset))
            jv = torch.cat((eye, -skew(com), q.new_zeros((n, 3, 2))), -1)
            jw = torch.cat((q.new_zeros((n, 3, 3)), eye, q.new_zeros((n, 3, 2))), -1)
            for j in b["ancestors"]:
                jv[:, :, 6+j] = torch.linalg.cross(axes[j], com-anchors[j])
                jw[:, :, 6+j] = axes[j]
            states.append(dict(origin=position, rotation=rotation, omega=w, alpha_bias=alpha,
                               origin_bias=acc, com=com, linear_bias=bias, jv=jv, jw=jw,
                               inertia=rotation @ b["inertia"] @ rotation.mT))
        return states

    def dynamics(self, q, dq, omega, gravity_b):
        """M(q) z + h(q,dq,omega,R) = [F_B, tau_B, tau_j], z=[a_C,alpha,ddq]."""
        matrix, bias = q.new_zeros((len(q), 8, 8)), q.new_zeros((len(q), 8))
        for body, state in zip(self.bodies, self.kinematics(q, dq, omega)):
            jv, jw, inertia = (state[k] for k in ("jv", "jw", "inertia"))
            force = body["mass"]*(state["linear_bias"]-gravity_b)
            moment = matvec(inertia, state["alpha_bias"]) + torch.linalg.cross(state["omega"], matvec(inertia, state["omega"]))
            matrix += body["mass"]*(jv.mT@jv) + jw.mT@inertia@jw
            bias += matvec(jv.mT, force) + matvec(jw.mT, moment)
        # PhysX's scalar generalized armature is not a new rigid link or full motor-rotor model.
        matrix[:, 6:, 6:] += torch.diag(self.armature)
        return matrix, bias


class ModelWrenchController(WrenchController):
    """Predict from bounded PD effort, eliminate ddq, invert four feasible channels.

    Same policy channels/gains/allocator/speed lag as WrenchController. Unlike
    that measured baseline, no stale-sensor warmup tick is necessary after reset.
    """

    def __init__(self, task, allocation, model, hover):
        self.model = model
        n = len(allocation)
        base = model.bodies[0]
        super().__init__(task, allocation, base["inertia"].expand(n, -1, -1),
                         allocation.new_full((n,), base["mass"]), hover)
        self.mass_matrix = allocation.new_zeros((n, 8, 8))
        self.bias = allocation.new_zeros((n, 8))
        self.predicted_acceleration_b = allocation.new_zeros((n, 3))
        self.predicted_joint_acceleration = allocation.new_zeros((n, 2))

    def reset(self, ids):
        super().reset(ids)
        for buffer in (self.mass_matrix, self.bias, self.predicted_acceleration_b, self.predicted_joint_acceleration):
            buffer[ids] = 0

    def step(self, actions, quaternion, rates, joint_pos, joint_vel):
        if self.task.sac_action_mapping:
            u = decode_angle_actions(clamp_scalar_actions(actions, self.task, 1.0), self.task)
            heading_scale = 1.0 if self.task.predict_yaw_as_sincos else math.radians(self.cfg.heading_range_degs)
            acceleration, heading = u[:, :3]*self.accel_scale, self.home_yaw+heading_scale*u[:, 3]
            reference_rates = u[:, 4:7]*self.rate_scale
            target = joint_angle_targets(u, self.task, self.joint_scale)
        else:
            u = decode_angle_actions(actions, self.task)
            acceleration, heading, reference_rates, target = u[:, :3], u[:, 3], u[:, 4:7], u[:, 7:9]
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
        # Joint row: ddq = solve(Mqq,tau-hq) - solve(Mqq,Mqb)*a_base.
        solved = torch.linalg.solve(matrix[:, 6:, 6:], torch.cat(
            (matrix[:, 6:, :6], (effort-bias[:, 6:]).unsqueeze(-1)), -1))
        effective = matrix[:, :6, :6] - matrix[:, :6, 6:] @ solved[:, :, :6]
        effective_bias = bias[:, :6] + matvec(matrix[:, :6, 6:], solved[:, :, 6])
        # Explicit direction reference: virtual alpha=0, not a coupled fixed-point solve.
        self.desired_force_w.copy_(matvec(rotation, matvec(effective[:, :3, :3], desired_a)+effective_bias[:, :3]))
        _, target_rotation = force_attitude(self.desired_force_w, heading, rotation)
        self.alpha_desired.copy_(attitude_acceleration(
            quaternion, target_rotation, rates, reference_rates, self.cfg, self.cfg.tilt_method))
        # Actual force is T*e3. Keep a_parallel=e3^T*a_des, allow lateral acceleration
        # to follow mechanics, and solve torque for the requested angular acceleration.
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
        self.requested_wrench.copy_(torch.cat((thrust[:, None], torque), -1))
        # Diagnostic arm-on-base reaction about BASE CoM, not a second plant force.
        self.external_force_w.copy_(matvec(rotation, self.mass[:, None]*(actual_a-gravity_b)-axis*thrust[:, None]))
        self.external_moment_b.copy_(matvec(self.inertia, self.alpha_desired)
            + torch.linalg.cross(rates, matvec(self.inertia, rates))-torque)
        self.requested_rotor_thrust.copy_(matvec(self.allocator.inverse, self.requested_wrench))
        commands = self.allocator.thrust(self.requested_wrench)
        self.allocation_residual.copy_(matvec(self.allocator.allocation, commands)-self.requested_wrench)
        self.rotor_speed.add_(self.motor_alpha*((commands/self.task.kf).sqrt()-self.rotor_speed))
        self.ready.fill_(True)
        return commands, effort
