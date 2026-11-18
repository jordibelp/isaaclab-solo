# SPDX-License-Identifier: BSD-3-Clause
"""Borinot observation adapter. Rewards, actions, reset and plant remain task-owned."""
import inspect

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_box_plus
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env import BorinotCmdVelEnv
from isaaclab_tasks.direct.borinot_goto.borinot_goto_env import BorinotGoToEnv


def attitude6(quat):
    # First two columns of R_WB, column-major (body x then body y in world).
    return matrix_from_quat(quat)[:, :, :2].transpose(1, 2).flatten(1)


class AdaptiveMeasurements:
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        n = self.num_envs
        self._history = torch.zeros(n, 1 if cfg.policy_model == "base_imu_teacher" else cfg.base_imu_history_length,
                                    27, device=self.device)
        self._clean = torch.zeros(n, 19, device=self.device)
        self._measured_quat = torch.zeros(n, 4, device=self.device)
        self._last_disturbance = torch.zeros(n, 6, device=self.device)
        self._static = self._read_startup_properties()
        self._gravity_w = torch.tensor(cfg.sim.gravity, device=self.device)
        self._sample_after_reset = None

    def _read_startup_properties(self):
        data, view = self.robot.data, self.robot.root_physx_view
        masses = view.get_masses().to(self.device)
        base = masses[:, self.base_id:self.base_id + 1]
        inertia = view.get_inertias()[:, self.base_id].to(self.device)
        plant = self._rotor_plant

        def columns(value, width):
            return torch.as_tensor(value, device=self.device).expand(self.num_envs, width)

        friction = [getattr(data, name, None) for name in (
            "joint_friction_coeff", "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff")]
        fr = [f[:, self.joint_ids] if f is not None else torch.zeros_like(self.actions[:, :2]) for f in friction]
        return torch.cat((masses.sum(1, keepdim=True), base, inertia[:, [0, 4, 8, 1, 2, 5]],
                          self._base_com_b, columns(plant.kf / self.cfg.kf if plant else 1., 6),
                          columns(plant.km / self.cfg.km if plant else 1., 1),
                          columns(plant.alpha if plant else self._motor_alpha, 6), *fr), -1)

    def _apply_action(self):
        super()._apply_action()
        # Same pre-integration state, gust envelope and CoM as the wrench composer.
        quat = self.robot.data.root_quat_w
        vertical = torch.zeros_like(self._base_push_points_b)
        vertical[:, 2] = self._base_push_force_z_w
        force = quat_apply_inverse(quat, vertical)
        moment = torch.linalg.cross(self._base_push_points_b - self._base_com_b, force)
        if self._wind is not None:
            wind, torque = self._wind.wrench_world(self.robot.data.root_lin_vel_w)
            force += quat_apply_inverse(quat, wind)
            moment += quat_apply_inverse(quat, torque)
        self._last_disturbance.copy_(torch.cat((force, moment), -1))

    def _measure(self, ids):
        d = self.robot.data
        quat = d.root_quat_w[ids]
        gyro = d.root_ang_vel_b[ids]
        acc = quat_apply_inverse(quat, d.body_lin_acc_w[ids, self.base_id] - self._gravity_w)
        q = d.joint_pos[ids][:, self.joint_ids]
        dq = d.joint_vel[ids][:, self.joint_ids]
        pos = d.root_pos_w[ids] - self.scene.env_origins[ids]
        self._clean[ids] = torch.cat((gyro, acc, q, dq, attitude6(quat), pos), -1)
        sg, sa, sq, sd, sr, sp = self.cfg.sensor_std
        mq = quat_box_plus(quat, sr * torch.randn_like(gyro))
        self._measured_quat[ids] = mq
        sample = torch.cat((gyro + sg * torch.randn_like(gyro), acc + sa * torch.randn_like(acc),
                            q + sq * torch.randn_like(q), dq + sd * torch.randn_like(dq),
                            attitude6(mq), pos + sp * torch.randn_like(pos), self.actions[ids]), -1)
        self._history[ids, -1] = sample

    def _sample_physics_events(self):
        super()._sample_physics_events()
        if self._history.shape[1] > 1:
            self._history[:, :-1] = self._history[:, 1:].clone()
        self._measure(slice(None))

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        self._history[ids] = 0
        self._last_disturbance[ids] = 0
        # First valid pose is available after the simulator's reset/forward.
        self._sample_after_reset = ids

    def _get_observations(self):
        if self._sample_after_reset is not None:
            self._measure(self._sample_after_reset)
            self._sample_after_reset = None
        current = self._history[:, -1]
        cmd, clean_cmd = self._adaptation_command(current)
        bypass = torch.cat((current, cmd), -1)
        privileged = torch.cat((self._clean, self.robot.data.root_lin_vel_b, self._static,
                                self._actuator_delay.seconds[:, None], self._last_disturbance,
                                self._rotor_thrust / self._hover_per_rotor), -1)
        critic = torch.cat((privileged, current, clean_cmd), -1)
        if self.cfg.policy_model == "base_imu_student_rl":
            policy = torch.cat((self._history.flatten(1), bypass), -1)
        else:
            policy = torch.cat((privileged, bypass), -1)
            if self.cfg.policy_model == "base_imu_student_dagger":
                policy = torch.cat((policy, self._history.flatten(1), bypass), -1)
        return {"policy": policy, "critic": critic}

    def get_code_snapshot_files(self):
        return super().get_code_snapshot_files() + [inspect.getsourcefile(AdaptiveMeasurements)]


class GoToAdaptiveEnv(AdaptiveMeasurements, BorinotGoToEnv):
    def _adaptation_command(self, current):
        target = self._goal_position_w - self.scene.env_origins
        delta = quat_apply_inverse(self._measured_quat, target - current[:, 16:19])
        exact = quat_apply_inverse(self.robot.data.root_quat_w, self._goal_position_w - self.robot.data.root_pos_w)
        left = (self._goal_deadline_steps - self._episode_steps).clamp_min(0)
        clocks = torch.stack((left / self._goal_deadline_steps, left * self.step_dt), -1)
        if self.cfg.goal_debug_vis:
            self._draw_goal_markers()
        return torch.cat((delta, clocks), -1), torch.cat((exact, clocks), -1)


class VelocityAdaptiveEnv(AdaptiveMeasurements, BorinotCmdVelEnv):
    def _adaptation_command(self, current):
        ids = (self._command_steps_left <= 0).nonzero().flatten()
        if len(ids):
            self._resample_commands(ids)
        return self._commands, self._commands
