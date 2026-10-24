# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import inspect

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_mul
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env import BorinotGoalkeeperEnv
from isaaclab_tasks.direct.borinot_goalkeeper.events import configure_randomization
from isaaclab_tasks.controllers.borinot_indi import IndiController
from isaaclab_tasks.direct.borinot_goalkeeper.paddle import attach_paddle

from .borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg
from .end_effector import configure_end_effector
from .task_math import sample_steps, sample_vertical_pushes, tracking_kernels


class BorinotCmdVelEnv(DirectRLEnv):
    """Free-flight velocity tracking with shared controls and a selectable tail payload."""
    cfg: BorinotCmdVelEnvCfg
    _sample_power = BorinotGoalkeeperEnv._sample_power
    _init_indi = BorinotGoalkeeperEnv._init_indi

    def __init__(self, cfg: BorinotCmdVelEnvCfg, render_mode=None, **kwargs):
        cfg.configure_control()  # after Hydra, before spaces/physics
        configure_randomization(cfg)
        # The shared racket randomizer must not resolve a body that is absent.
        if not cfg.arm_holds_racket and cfg.events is not None:
            cfg.events.racket_mass = None
        self.contact_body_names = (("paddle",) if cfg.arm_holds_racket else ()) + (
            "base", "flying_arm_2__link1", "flying_arm_2__link2")
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, _ = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        self.base_id = self.robot.find_bodies("base")[0][0]
        self._base_contact_column = self.contact_body_names.index("base")
        if any(sensor.contact_physx_view.filter_count != len(self.contact_body_names) - 1
               for sensor in self._body_contact_sensors):
            raise RuntimeError("Borinot contact filters must resolve all other robot bodies, excluding the source.")
        self._events = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        for name in ("base_collision", "leave_area", "tilt")}
        self._self_contact_bodies = torch.zeros(self.num_envs, len(self.contact_body_names),
                                               dtype=torch.bool, device=self.device)
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._rate_integral = torch.zeros(self.num_envs, 3, device=self.device)
        self._joint_scale = torch.tensor(cfg.joint_target_scale, device=self.device)
        if cfg.sac_action_mapping:
            self._sac_joint_offset = torch.tensor(cfg.sac_joint_pos_offset, device=self.device)
            self._sac_joint_soft_limits = torch.deg2rad(torch.tensor(
                [cfg.joint_soft_limit_thigh, cfg.joint_soft_limit_calf], device=self.device))
            hard_limits = torch.deg2rad(torch.tensor(
                [cfg.joint_physical_limit_thigh, cfg.joint_physical_limit_calf], device=self.device))
            self.robot.write_joint_position_limit_to_sim(
                hard_limits.expand(self.num_envs, -1, -1).clone(), joint_ids=self.joint_ids)
        self._rate_scale = torch.tensor(cfg.body_rate_scale, device=self.device)
        self._kp = torch.tensor(cfg.rate_kp, device=self.device)
        self._ki = torch.tensor(cfg.rate_ki, device=self.device)
        # Nominal calibration, never environment 0's sampled mass/CoM.
        self.robot_mass_kg = float(self.robot.data.default_mass[0].sum())
        self._hover_total = self.robot_mass_kg * 9.81
        self._hover_per_rotor = self._hover_total / 6
        cfg.thrust_to_weight = min(cfg.thrust_to_weight, cfg.rotor_thrust_limit_n / self._hover_per_rotor)
        self._base_com_b = self.robot.root_physx_view.get_coms()[:, self.base_id, :3].to(self.device)
        nominal_com = getattr(self, "_nominal_base_com", self._base_com_b[0]).to(self.device)
        self._base_com_offset = self._base_com_b - nominal_com
        rotor_pos = torch.tensor(cfg.rotor_pos_b, device=self.device) - nominal_com
        self._allocation = torch.stack((torch.ones(6, device=self.device), rotor_pos[:, 1], -rotor_pos[:, 0],
                                        torch.tensor(cfg.rotor_spin, device=self.device) * cfg.km_over_kf))
        self._allocation_inverse = torch.linalg.pinv(self._allocation)
        self._rotor_thrust = torch.full((self.num_envs, 6), self._hover_per_rotor, device=self.device)
        self._actuator_commands = torch.zeros(self.num_envs, 8, device=self.device)
        self._actuator_commands[:, :6] = 1.0 / cfg.thrust_to_weight
        self._previous_actuator_commands = self._actuator_commands.clone()
        self._motor_alpha = 1.0 if cfg.motor_time_constant_s == 0 else 1 - math.exp(-self.physics_dt / cfg.motor_time_constant_s)
        if cfg.control_mode == "indi":
            self._init_indi()
        self._power_sum_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._power_samples = self._physics_samples = 0

        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._command_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_ranges = torch.tensor((cfg.command_lin_vel_x_range, cfg.command_lin_vel_y_range,
                                            cfg.command_lin_vel_z_range, cfg.command_ang_vel_z_range), device=self.device)
        self._base_push_steps_left = torch.zeros_like(self._command_steps_left)
        self._base_push_steps_until_next = torch.zeros_like(self._command_steps_left)
        self._base_push_force_z_w = torch.zeros(self.num_envs, device=self.device)
        self._base_push_points_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self._force_limits = torch.tensor(cfg.forces_applied_to_base_curriculum, device=self.device)
        self.force_level = len(self._force_limits) - 1 if cfg.skip_curriculum else cfg.curriculum_start_level
        self._episode_stage = torch.full_like(self._command_steps_left, self.force_level)
        self._episode_steps = torch.zeros_like(self._command_steps_left)
        self._episode_xy_integral = torch.zeros(self.num_envs, device=self.device)
        self._episode_energy_j = torch.zeros_like(self._episode_xy_integral)
        self._episode_terms = {name: torch.zeros_like(self._episode_xy_integral) for name in (
            "track_lin_vel_xyz_exp", "track_ang_vel_z_exp", "action_rate", "energy",
            "base_collision", "leave_area", "self_collision", "tilt")}
        self._curriculum_window = torch.zeros(cfg.curriculum_window_episodes, device=self.device)
        self._window_cursor = self._stage_episodes = self._stage_start_step = 0
        self.curriculum_stage_pure_episodes = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        configure_end_effector(sim_utils.get_current_stage(), "/World/envs/env_0/Robot", self.cfg)
        ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.0))
        ground.func("/World/Ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot
        # Filtered reporting is one-to-many: one source body, explicit other-body
        # targets. A source-inclusive wildcard can silently hide self forces.
        self._body_contact_sensors = []
        for name in self.contact_body_names:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}", update_period=0.0, history_length=0,
                filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{other}"
                                        for other in self.contact_body_names if other != name]))
            self._body_contact_sensors.append(sensor)
            self.scene.sensors[f"robot_contact_{name}"] = sensor
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)

    def _resample_commands(self, env_ids):
        values = torch.rand((len(env_ids), 4), device=self.device)
        values = self._command_ranges[:, 0] + values * (self._command_ranges[:, 1] - self._command_ranges[:, 0])
        values[torch.rand(len(env_ids), device=self.device) < self.cfg.standing_env_prob] = 0.0
        self._commands[env_ids] = values
        self._command_steps_left[env_ids] = max(1, math.ceil(self.cfg.command_resampling_time_s / self.step_dt))

    def _update_pushes(self):
        # Start-to-start intervals. Duration cannot exceed the minimum interval.
        start_ids = (self._base_push_steps_until_next <= 0).nonzero().flatten()
        if len(start_ids):
            limits = self._force_limits[self._episode_stage[start_ids]]
            force, points = sample_vertical_pushes(limits, self.cfg.base_push_application_radius_m,
                                                    self.cfg.base_push_application_half_height_m)
            self._base_push_force_z_w[start_ids] = force
            self._base_push_points_b[start_ids] = points
            self._base_push_steps_left[start_ids] = sample_steps(
                len(start_ids), self.cfg.base_push_duration_range_s, self.step_dt, self.device)
            self._base_push_steps_until_next[start_ids] = sample_steps(
                len(start_ids), self.cfg.base_push_interval_range_s, self.step_dt, self.device)
        self._base_push_force_z_w[self._base_push_steps_left <= 0] = 0.0
        self._base_push_steps_left.sub_(1).clamp_min_(0)
        self._base_push_steps_until_next.sub_(1)

    def _pre_physics_step(self, actions):
        self.extras.pop("log", None)  # keep episode-only log packets for RSL
        for event in self._events.values():
            event.zero_()
        self._self_contact_bodies.zero_()
        self._episode_steps += 1
        self._command_steps_left.sub_(1)
        self._update_pushes()
        self._previous_actuator_commands.copy_(self._actuator_commands)
        if self.cfg.control_mode == "indi":
            actions = actions.clamp(-1.0, 1.0)
        elif not self.cfg.sac_action_mapping and not self.cfg.remove_thrust_clipping:
            actions = actions.clamp(-self.cfg.action_clamp, self.cfg.action_clamp)
        self.actions.copy_(actions)
        self._power_sum_w.zero_()
        self._power_samples = self._physics_samples = 0

    def _sample_physics_events(self):
        """Latch completed-substep events until this policy interval is scored."""
        for column, sensor in enumerate(self._body_contact_sensors):
            data = sensor.data
            own_forces = data.force_matrix_w[:, 0]  # [env, other robot body, world XYZ]
            # Count each body once per interval, even across repeated substep contacts.
            # Pairwise norms also retain opposing self forces that cancel in the net.
            self._self_contact_bodies[:, column] |= (
                own_forces.norm(dim=-1) > self.cfg.self_collision_force_threshold_n).any(-1)
            if column == self._base_contact_column:
                external = data.net_forces_w[:, 0] - own_forces.sum(1)
                self._events["base_collision"] |= external.norm(dim=-1) > self.cfg.base_collision_force_threshold_n
        position = self.robot.data.root_pos_w - self.scene.env_origins
        low, high = self.cfg.flight_height_range_m
        self._events["leave_area"] |= ((position[:, 2] < low) | (position[:, 2] > high)
                                      | (position[:, :2].abs() > self.cfg.flight_xy_limit_m).any(-1))
        self._events["tilt"] |= (self.robot.data.projected_gravity_b[:, 2]
                                > -math.cos(math.radians(self.cfg.robot_tilt_threshold_degs)))

    def _apply_action(self):
        BorinotGoalkeeperEnv._apply_action(self)
        force_w = torch.zeros(self.num_envs, 3, device=self.device)
        force_w[:, 2] = self._base_push_force_z_w
        force_b = quat_apply_inverse(self.robot.data.root_quat_w, force_w)
        # Explicit moment about each actual sampled CoM. Composer positions use
        # the link origin, while PhysX applies the resulting torque about the CoM.
        moment_b = torch.linalg.cross(self._base_push_points_b - self._base_com_b, force_b)
        self.robot.permanent_wrench_composer.add_forces_and_torques(
            forces=force_b[:, None], torques=moment_b[:, None], body_ids=[self.base_id], is_global=False)

    def _get_dones(self):
        self._sample_power()  # include the final completed physics substep, before any automatic reset
        self._sample_physics_events()
        failed = self._events["leave_area"].clone()
        if self.cfg.reset_robot_on_robot_crash:
            failed |= self._events["base_collision"]
        if self.cfg.reset_robot_on_tilt_threshold:
            failed |= self._events["tilt"]
        if self.cfg.self_collision_triggers_reset:
            failed |= self._self_contact_bodies.any(-1)
        # A real 10-second task clock, independent of runner episode-counter randomization.
        return failed, self._episode_steps >= self.max_episode_length

    def _get_rewards(self):
        linear, angular, planar = tracking_kernels(
            self._commands, self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w,
            self.robot.data.root_quat_w, self.cfg.tracking_lin_std, self.cfg.tracking_ang_std)
        self._actuator_commands[:, 6:] = self.robot.data.applied_torque[:, self.joint_ids] / (2 * self.cfg.effort_lim_sim)
        delta = self._actuator_commands - self._previous_actuator_commands
        action_rate = delta[:, :6].square().mean(-1) + delta[:, 6:].square().mean(-1)
        power = self._power_sum_w / max(1, self._power_samples)
        terms = {
            "track_lin_vel_xyz_exp": self.cfg.rwd_scale_lin_vel * linear * self.step_dt,
            "track_ang_vel_z_exp": self.cfg.rwd_scale_ang_vel * angular * self.step_dt,
            "action_rate": self.cfg.rwd_scale_action_rate * action_rate * self.step_dt,
            "energy": self.cfg.rwd_scale_energy * power.sum(-1),
            "base_collision": self.cfg.rwd_scale_base_collision * self._events["base_collision"].float(),
            "leave_area": self.cfg.rwd_scale_leave_area * self._events["leave_area"].float(),
            "self_collision": self.cfg.rwd_scale_self_collision * self._self_contact_bodies.sum(-1) * self.step_dt,
            "tilt": self.cfg.rwd_scale_tilt * self._events["tilt"].float(),
        }
        for name, value in terms.items():
            self._episode_terms[name] += value
        self._episode_xy_integral += planar * self.step_dt
        self._episode_energy_j += self._power_sum_w.sum(-1) * self.physics_dt
        self.extras["tracking"] = {"xyz": linear, "yaw": angular, "xy": planar}
        self.extras["reward_terms"] = terms  # keep terminal contributions available after automatic reset
        self.extras["power_mean_w"] = {name: power[:, i] for i, name in enumerate(
            ("rotor_mechanical", "rotor_dissipative", "arm_mechanical", "arm_copper_loss"))}
        return sum(terms.values())

    def _get_observations(self):
        # Expose the next command BEFORE the policy acts on it. Resampling in
        # _pre_physics_step would score one action against an unseen command.
        command_ids = (self._command_steps_left <= 0).nonzero().flatten()
        if len(command_ids):
            self._resample_commands(command_ids)
        data, q = self.robot.data, self.robot.data.root_quat_w
        parts = [data.root_lin_vel_b, data.root_ang_vel_b]
        if self.cfg.obs_include_base_lin_acc:
            acceleration_b = quat_apply_inverse(q, data.body_lin_acc_w[:, self.base_id])
            parts.append(acceleration_b * (self._episode_steps > 0)[:, None])
        parts += [data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids], self.actions,
                  quat_apply_inverse(q, self._gravity), matrix_from_quat(q).flatten(1), self._commands]
        if self.cfg.control_mode == "PI_rotors_PD_joint_pos_control":
            parts += [self._rotor_thrust / self._hover_per_rotor, self._rate_integral]
        elif self.cfg.control_mode == "indi":
            parts.append(self._indi.observation())
        observation = torch.cat(parts, -1)
        return {"policy": observation, "critic": observation}

    def _update_curriculum(self, env_ids):
        completed = env_ids[self._episode_steps[env_ids] > 0]
        if not len(completed):
            return
        duration = self._episode_steps[completed] * self.step_dt
        # Normalize by the full horizon so brief failed flights cannot promote.
        score = (self._episode_xy_integral[completed] / self.cfg.episode_length_s).clamp(0, 1)
        current = self._episode_stage[completed] == self.force_level
        stale = (~current).nonzero().flatten()
        self.curriculum_stage_pure_episodes = (len(completed) - 1 - stale[-1].item() if len(stale)
                                             else self.curriculum_stage_pure_episodes + len(completed))
        values = score[current]
        if len(values):
            self._stage_episodes += len(values)
            values = values[-len(self._curriculum_window):]
            indices = (torch.arange(len(values), device=self.device) + self._window_cursor) % len(self._curriculum_window)
            self._curriculum_window[indices] = values
            self._window_cursor = (self._window_cursor + len(values)) % len(self._curriculum_window)
        retained = min(self._stage_episodes, len(self._curriculum_window))
        mean = self._curriculum_window.sum() / max(1, retained)
        old_level = self.force_level
        if (self.cfg.curriculum and self.force_level < len(self._force_limits) - 1
                and self._stage_episodes >= self.cfg.curriculum_min_episodes
                and self.common_step_counter - self._stage_start_step >= self.cfg.curriculum_min_steps
                and mean > self.cfg.curriculum_tracking_threshold):
            self.force_level += 1
            self._stage_start_step = self.common_step_counter
            self._stage_episodes = self._window_cursor = self.curriculum_stage_pure_episodes = 0
            self._curriculum_window.zero_()
        self.extras["log"] = {
            "Episode/return": sum(v[completed] for v in self._episode_terms.values()).mean(),
            "Episode/duration_s": duration.mean(),
            "Episode/energy_j": self._episode_energy_j[completed].mean(),
            "Tracking/xy_mean": (self._episode_xy_integral[completed] / duration).mean(),
            "Curriculum/collection_stage": self._episode_stage[completed].float().mean(),
            "Curriculum/evaluated_stage": old_level,
            "Curriculum/selected_stage": self.force_level,
            "Curriculum/stage_pure_xy_tracking": mean.clone(),
            "Curriculum/window_completed_episodes": retained,
            "Curriculum/force_limit_n": self._force_limits[self.force_level],
        }
        self.extras["log"].update({f"Episode_Reward/{name}": value[completed].mean()
                                  for name, value in self._episode_terms.items()})
        self.extras["log"].update({f"Power/{name}_mean_w": value[completed].mean()
                                  for name, value in self.extras.get("power_mean_w", {}).items()})

    def get_curriculum_global_idx(self):
        return self.force_level

    def get_curriculum_max_global_idx(self):
        return len(self._force_limits) - 1

    def get_code_snapshot_files(self):
        """Archive shared plant/control helpers alongside this task's source."""
        return [inspect.getsourcefile(obj) for obj in (
            BorinotGoalkeeperEnv, configure_randomization, attach_paddle, configure_end_effector,
            IndiController, tracking_kernels)]

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._update_curriculum(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] = torch.tensor(self.cfg.robot_home, device=self.device) + self.scene.env_origins[env_ids]
        root[:, :2] += torch.empty(n, 2, device=self.device).uniform_(-self.cfg.reset_xy_noise_m, self.cfg.reset_xy_noise_m)
        root[:, 2] += torch.empty(n, device=self.device).uniform_(-self.cfg.reset_z_noise_m, self.cfg.reset_z_noise_m)
        root[:, 3:7] = torch.tensor(self.cfg.robot_home_quat, device=self.device)
        if any(self.cfg.reset_rpy_noise_degs):
            half_ranges = torch.deg2rad(torch.tensor(self.cfg.reset_rpy_noise_degs, device=self.device))
            rpy = (2 * torch.rand(n, 3, device=self.device) - 1) * half_ranges
            delta = quat_from_euler_xyz(*rpy.unbind(-1))
            # Right multiplication: small local offsets about the nominal home attitude.
            root[:, 3:7] = quat_mul(root[:, 3:7], delta)
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                           torch.zeros_like(self.robot.data.default_joint_vel[env_ids]), env_ids=env_ids)
        joint_zero = torch.zeros(n, len(self.joint_ids), device=self.device)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[env_ids][:, self.joint_ids],
                                             joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(joint_zero, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_effort_target(joint_zero, joint_ids=self.joint_ids, env_ids=env_ids)
        for buffer in (self.actions, self._rate_integral, self._episode_steps, self._episode_xy_integral,
                       self._episode_energy_j, self._base_push_steps_left, self._base_push_force_z_w,
                       self._base_push_points_b, self._power_sum_w):
            buffer[env_ids] = 0
        for value in self._episode_terms.values():
            value[env_ids] = 0
        for event in self._events.values():
            event[env_ids] = False
        self._self_contact_bodies[env_ids] = False
        self._rotor_thrust[env_ids] = self._hover_per_rotor
        self._episode_stage[env_ids] = self.force_level
        self._base_push_steps_until_next[env_ids] = sample_steps(
            n, self.cfg.base_push_interval_range_s, self.step_dt, self.device)
        for buffer in (self._actuator_commands, self._previous_actuator_commands):
            buffer[env_ids, :6], buffer[env_ids, 6:] = 1.0 / self.cfg.thrust_to_weight, 0.0
        if self.cfg.control_mode == "indi":
            self._indi.reset(env_ids)
        self._resample_commands(env_ids)
