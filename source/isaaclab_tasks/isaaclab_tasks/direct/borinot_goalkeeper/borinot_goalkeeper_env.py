# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import inspect

import torch
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse
from isaaclab_tasks.controllers.borinot_indi import IndiController
from isaaclab_tasks.direct.borinot.action_limits import raw_action_limit_costs, relative_limit_cost

from .borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg
from .desired_actuation import DesiredActuationMetrics
from .events import configure_randomization
from .paddle import attach_paddle
from .task_math import (
    crosses_goal, plane_progress, predict_intercept, promotion_ready, sample_robot_positions, sample_shots,
    sample_shot_delay_steps, terminal_outcome, validate_reset_grid, validate_shot_schedule,
)


class BorinotGoalkeeperEnv(DirectRLEnv):
    """Continuous one-ball shots; only robot safety failures terminate an episode.

    Outcome codes: 0 ongoing, 1 base hit, 2 arm hit, 3 goal, 4 live-ball floor,
    5 leave goal area, 6 deadline, 7 save, 8 robot collision/crash, 9 miss/escape.
    """

    cfg: BorinotGoalkeeperEnvCfg
    contact_body_names = ("paddle", "base", "flying_arm_2__link1", "flying_arm_2__link2")

    def get_code_snapshot_files(self):
        """Retain the shared implementation when archiving a training run."""
        return [inspect.getsourcefile(obj) for obj in (
            BorinotGoalkeeperEnv, IndiController, raw_action_limit_costs, DesiredActuationMetrics)]

    def __init__(self, cfg: BorinotGoalkeeperEnvCfg, render_mode=None, **kwargs):
        cfg.configure_control()
        configure_randomization(cfg)
        validate_reset_grid(cfg)
        validate_shot_schedule(cfg)
        if not 0 <= cfg.shot_level < len(cfg.shot_half_extents_yz):
            raise ValueError("shot_level must index shot_half_extents_yz.")
        if not 1 <= cfg.curriculum_min_shots <= cfg.curriculum_window_shots:
            raise ValueError("Require 1 <= curriculum_min_shots <= curriculum_window_shots.")
        if not cfg.robot_crash_body_names or set(cfg.robot_crash_body_names) - set(self.contact_body_names):
            raise ValueError(f"robot_crash_body_names must select from {self.contact_body_names}.")
        if cfg.robot_crash_force_threshold_n <= 0:
            raise ValueError("robot_crash_force_threshold_n must be positive.")
        if cfg.curriculum_min_steps < 1 or cfg.motor_time_constant_s < 0:
            raise ValueError("Curriculum dwell must be positive and motor time constant must be non-negative.")
        super().__init__(cfg, render_mode, **kwargs)
        self.joint_ids, _ = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        self.base_id = self.robot.find_bodies("base")[0][0]
        self.paddle_id = self.robot.find_bodies("paddle")[0][0]
        self._contact_ids = self.robot_contact.find_bodies(list(self.contact_body_names), preserve_order=True)[0]
        self._crash_contact_columns = [self.contact_body_names.index(name) for name in cfg.robot_crash_body_names]
        if self.ball_contact.contact_physx_view.filter_count != 6:
            raise RuntimeError("Expected separate paddle, base, arm1, arm2, goal, and floor contact filters.")

        self._home = torch.tensor(cfg.robot_home, device=self.device)
        self._area_min = torch.tensor(
            [cfg.goal_area_x_m[0], cfg.goal_area_y_m[0], cfg.goal_area_z_m[0]], device=self.device
        )
        self._area_max = torch.tensor(
            [cfg.goal_area_x_m[1], cfg.goal_area_y_m[1], cfg.goal_area_z_m[1]], device=self.device
        )
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        self._gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._action_limit_costs = torch.zeros(self.num_envs, 3, device=self.device)
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

        # Fixed action scale/cap: do not let environment 0's random mass retune all rotors.
        self.robot_mass_kg = float(self.robot.data.default_mass[0].sum())
        self._hover_total = self.robot_mass_kg * 9.81
        self._hover_per_rotor = self._hover_total / 6
        # Resolve once so clipping, SAC endpoints and command normalization use the same physical limit.
        cfg.thrust_to_weight = min(cfg.thrust_to_weight, cfg.rotor_thrust_limit_n / self._hover_per_rotor)
        rotor_pos = torch.tensor(cfg.rotor_pos_b, device=self.device)
        base_coms = self.robot.root_physx_view.get_coms()[:, self.base_id, :3].to(self.device)
        base_com = getattr(self, "_nominal_base_com", base_coms[0]).to(self.device)
        self._base_com_offset = base_coms - base_com
        rotor_pos = rotor_pos - base_com
        self._allocation = torch.stack((torch.ones(6, device=self.device), rotor_pos[:, 1], -rotor_pos[:, 0],
                                        torch.tensor(cfg.rotor_spin, device=self.device) * cfg.km_over_kf))
        self._allocation_inverse = torch.linalg.pinv(self._allocation)
        self._rotor_thrust = torch.full((self.num_envs, 6), self._hover_per_rotor, device=self.device)
        # Normalized post-clipping commands: thrust / F_max, torque / (2*tau_max).
        # Reset starts at hover thrust and zero joint effort, not zero thrust.
        self._actuator_commands = torch.zeros(self.num_envs, 8, device=self.device)
        self._actuator_commands[:, :6] = 1.0 / cfg.thrust_to_weight
        self._previous_actuator_commands = self._actuator_commands.clone()
        self._motor_alpha = (
            1.0 if cfg.motor_time_constant_s == 0 else 1 - math.exp(-self.physics_dt / cfg.motor_time_constant_s)
        )
        self._desired_actuation = (
            DesiredActuationMetrics(cfg, self._hover_per_rotor, self.num_envs, self.device)
            if cfg.control_mode == "thrust_and_torque_control" and not cfg.hold_arm else None)
        if cfg.control_mode == "indi":
            self._init_indi()
        # Columns: rotor mechanical, rotor dissipative, arm mechanical, arm copper loss.
        self._power_sum_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._power_samples = 0

        self._hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._clear = torch.zeros_like(self._hit)
        self._new_hit = torch.zeros_like(self._hit)
        self._flags = {
            name: torch.zeros_like(self._hit) for name in ("base", "arm", "crash", "external_crash", "tilt", "goal", "floor", "area", "miss")
        }
        self._center_quality = torch.zeros(self.num_envs, device=self.device)
        self._recovery_time = torch.zeros(self.num_envs, device=self.device)
        self._previous_paddle = torch.zeros(self.num_envs, 3, device=self.device)
        self._old_progress_active = torch.zeros_like(self._hit)
        self._previous_ball = torch.zeros(self.num_envs, 3, device=self.device)
        self._outcome = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.outcome_counts = torch.zeros(10, dtype=torch.long, device=self.device)
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._episode_terms = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("plane_progress", "outcome", "floor_after_save", "optional_rates", "optional_hit", "energy",
                         "thrust_limits", "torque_limits", "action_limits")
        }
        self._shot_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_steps = torch.zeros_like(self._shot_steps)
        self._ball_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._next_launch_steps = torch.zeros_like(self._shot_steps)
        self._shot_stage = torch.full_like(self._shot_steps, cfg.shot_level)
        self._episode_shots = torch.zeros_like(self._shot_steps)
        self._episode_saves = torch.zeros_like(self._shot_steps)
        self._episode_paddle_hits = torch.zeros_like(self._shot_steps)
        self._episode_stage = torch.full((self.num_envs,), cfg.shot_level, device=self.device, dtype=torch.long)
        self.spawn_grid_index = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.shot_level = cfg.shot_level
        self._stage_start_step = 0
        self._curriculum_window = torch.zeros(cfg.curriculum_window_shots, device=self.device)
        self._window_cursor = 0
        self._stage_shots = 0
        # Newest consecutive completed episodes that started at the current level. Training only
        # scores a stage's best checkpoint once the whole reward buffer comes from that stage.
        self.curriculum_stage_pure_episodes = 0
        # PhysX property setters use CPU buffers; retain them rather than querying every reset.
        self._ball_masses = self.ball.root_physx_view.get_masses()
        self._ball_inertias = self.ball.root_physx_view.get_inertias()
        self._ball_materials = self.ball.root_physx_view.get_material_properties()
        self._ball_parameters = torch.zeros(self.num_envs, 2, device=self.device)
        self._physics_samples = 0

    def _init_indi(self):
        # default_inertia is nominal, about the base CoM and expressed in base-link axes.
        # Never tune the common controller to one clone's startup-randomized physics.
        inertia = self.robot.data.default_inertia[0, self.base_id].reshape(3, 3).to(self.device)
        self._indi = IndiController(self.cfg, self.num_envs, self._allocation, inertia,
                                    self.robot_mass_kg, self._hover_per_rotor)
        hard_limits = torch.deg2rad(torch.tensor(
            [self.cfg.joint_physical_limit_thigh, self.cfg.joint_physical_limit_calf], device=self.device))
        self.robot.write_joint_position_limit_to_sim(
            hard_limits.expand(self.num_envs, -1, -1).clone(), joint_ids=self.joint_ids)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        stage = sim_utils.get_current_stage()
        attach_paddle(stage, "/World/envs/env_0/Robot", self.cfg)
        self.ball = RigidObject(self.cfg.ball)
        ball_api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath("/World/envs/env_0/Ball"))
        ball_api.CreateEnableSpeculativeCCDAttr().Set(True)
        goal_cfg = sim_utils.CuboidCfg(
            size=self.cfg.goal_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.20, 0.28)),
        )
        goal_cfg.func("/World/envs/env_0/Goal", goal_cfg, translation=self.cfg.goal_center)
        ground_cfg = sim_utils.CuboidCfg(
            size=(self.cfg.scene.env_spacing - 0.5, self.cfg.scene.env_spacing - 0.5, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
        )
        ground_cfg.func("/World/envs/env_0/Floor", ground_cfg, translation=(0.0, 0.0, -0.05))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        self.ball_contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Ball", update_period=0.0, history_length=0,
            filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{name}" for name in self.contact_body_names]
                                  + ["/World/envs/env_.*/Goal", "/World/envs/env_.*/Floor"],
        ))
        self.robot_contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=0,
        ))
        self.scene.sensors["ball_contacts"] = self.ball_contact
        self.scene.sensors["robot_contacts"] = self.robot_contact
        # Filtered reporting is one-to-many: one source sensor per monitored body.
        # The default path needs no additional sensors or force subtraction.
        self._self_contact_sensors = []
        if not self.cfg.self_collision_triggers_reset:
            for name in self.cfg.robot_crash_body_names:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{name}", update_period=0.0, history_length=0,
                    filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{other}"
                                            for other in self.contact_body_names if other != name],
                ))
                self._self_contact_sensors.append(sensor)
                self.scene.sensors[f"self_contacts_{name}"] = sensor
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)

    def _paddle_pose(self):
        position = self.robot.data.body_pos_w[:, self.paddle_id] - self.scene.env_origins
        quaternion = self.robot.data.body_quat_w[:, self.paddle_id]
        return position, quaternion, quat_apply(quaternion, self._x_axis)

    def _interception(self):
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        target, time_s, valid = predict_intercept(ball_pos, self.ball.data.root_lin_vel_w, self.cfg.keeper_plane_x)
        valid &= self._ball_active
        paddle_pos = self._paddle_pose()[0]
        distance = torch.linalg.vector_norm(target - paddle_pos, dim=-1)
        approaching = valid & ~self._hit
        potential = torch.exp(-distance.square() / self.cfg.intercept_sigma_m**2) * approaching
        return target, time_s, valid, potential

    def _pre_physics_step(self, actions):
        self._action_limit_costs.copy_(raw_action_limit_costs(actions, self.cfg, self._hover_per_rotor))
        if self._desired_actuation is not None:
            self._desired_actuation.update(actions)
        self.extras.pop("log", None)
        self._episode_steps += 1
        self._next_launch_steps.sub_(1).clamp_min_(0)
        launch_ids = ((~self._ball_active) & (self._next_launch_steps == 0)).nonzero().flatten()
        if len(launch_ids):
            self._launch_ball(launch_ids)
        self._shot_steps += self._ball_active
        self._outcome.zero_()
        self._previous_actuator_commands.copy_(self._actuator_commands)
        if self.cfg.control_mode == "indi":
            if self.cfg.sac_action_mapping:
                actions = actions.clamp(-1.0, 1.0)
        elif not self.cfg.sac_action_mapping and not self.cfg.remove_thrust_clipping:
            if self.cfg.control_mode == "thrust_and_torque_control" and not self.cfg.hold_arm:
                # PPO joint actions are unbounded N*m; retain only the rotor input clamp.
                actions = torch.cat((actions[:, :6].clamp(-self.cfg.action_clamp, self.cfg.action_clamp),
                                     actions[:, 6:]), dim=-1)
            else:
                actions = actions.clamp(-self.cfg.action_clamp, self.cfg.action_clamp)
        self.actions.copy_(actions)
        self._previous_paddle.copy_(self._paddle_pose()[0])
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        self._old_progress_active.copy_(self._ball_active & ~self._hit & (self.ball.data.root_lin_vel_w[:, 0] < -0.1)
                                       & (ball_pos[:, 0] > self.cfg.goal_center[0]))
        self._new_hit.zero_()
        for flag in self._flags.values():
            flag.zero_()
        self._physics_samples = 0
        self._power_sum_w.zero_()
        self._power_samples = 0

    def _apply_action(self):
        self._sample_power()
        # Inspect every completed physics substep, including impacts between policy ticks.
        if self._physics_samples:
            self._sample_physics_events()
        self._physics_samples += 1
        if self.cfg.control_mode == "indi":
            commands, joint_effort = self._indi.step(
                self.actions, self.robot.data.root_quat_w, self.robot.data.root_ang_vel_b,
                self.robot.data.joint_pos[:, self.joint_ids], self.robot.data.joint_vel[:, self.joint_ids])
            if self.cfg.rwd_scale_thrust_limits:
                cap = self._hover_per_rotor * self.cfg.thrust_to_weight
                self._action_limit_costs[:, 0] += relative_limit_cost(
                    self._indi.requested_rotor_thrust, 0., cap, cap) / self.cfg.decimation
            if self.cfg.rwd_scale_torque_limits:
                limit = self.cfg.effort_lim_sim
                self._action_limit_costs[:, 1] += relative_limit_cost(
                    self._indi.requested_joint_effort, -limit, limit, limit) / self.cfg.decimation
        elif self.cfg.control_mode == "PI_rotors_PD_joint_pos_control":
            rates = self.robot.data.root_ang_vel_b
            error = self.actions[:, 1:4] * self._rate_scale - rates
            self._rate_integral.add_(error * self.physics_dt).clamp_(-0.5, 0.5)
            torque = self._kp * error + self._ki * self._rate_integral
            collective = self._hover_total * (1 + self.cfg.collective_action_scale * self.actions[:, :1])
            commands = torch.cat((collective, torque), dim=-1) @ self._allocation_inverse.T
            if self.cfg.rwd_scale_thrust_limits:
                cap = self._hover_per_rotor * self.cfg.thrust_to_weight
                self._action_limit_costs[:, 0] += relative_limit_cost(
                    commands, 0., None if self.cfg.remove_thrust_clipping else cap, cap) / self.cfg.decimation
        else:
            if self.cfg.sac_action_mapping:
                u = self.actions[:, :6].clamp(-1.0, 1.0)
                commands = self._hover_per_rotor * (1 + torch.where(u >= 0, (self.cfg.thrust_to_weight - 1) * u, u))
            else:
                commands = self._hover_per_rotor * (1 + self.cfg.rotor_action_scale * self.actions[:, :6])
        # Rotors only push: thrust is always >= 0. The flag removes only the upper thrust cap.
        cap = self._hover_per_rotor * self.cfg.thrust_to_weight
        commands.clamp_(min=0.0, max=None if self.cfg.remove_thrust_clipping else cap)
        self._actuator_commands[:, :6] = commands / cap
        if self.cfg.control_mode == "indi":
            # INDI models lag on speed, not thrust. Other modes keep their original thrust lag.
            self._rotor_thrust.copy_(self.cfg.kf * self._indi.rotor_speed.square())
        elif self.cfg.motor_time_constant_s == 0:
            self._rotor_thrust.copy_(commands)
        else:
            self._rotor_thrust.add_(self._motor_alpha * (commands - self._rotor_thrust))
        wrench = self._rotor_thrust @ self._allocation.T
        if self.cfg.control_mode == "indi":
            wrench[:, 1:] += self._indi.transient_torque
        # Forces act at the fixed rotor positions, about each environment's actual CoM.
        wrench[:, 1] -= self._base_com_offset[:, 1] * wrench[:, 0]
        wrench[:, 2] += self._base_com_offset[:, 0] * wrench[:, 0]
        force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        force[:, 0, 2] = wrench[:, 0]
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force, torques=wrench[:, None, 1:4], body_ids=[self.base_id], is_global=False,
        )
        if self.cfg.control_mode == "indi":
            # Implicit gains are zero: the software PID runs once per physics substep.
            self.robot.set_joint_effort_target(joint_effort, joint_ids=self.joint_ids)
        elif self.cfg.control_mode == "thrust_and_torque_control" and not self.cfg.hold_arm:
            # PPO outputs N*m directly. SAC retains normalized -1 / 0 / +1 -> -limit / 0 / +limit.
            effort = self.actions[:, 6:8]
            if self.cfg.sac_action_mapping:
                effort = effort * self.cfg.effort_lim_sim
            self.robot.set_joint_effort_target(effort, joint_ids=self.joint_ids)
        else:
            targets = self.actions[:, -2:] * self._joint_scale
            if self.cfg.sac_action_mapping:
                targets = targets + self._sac_joint_offset
            if self.cfg.hold_arm:
                targets = torch.zeros_like(targets)
            self.robot.set_joint_position_target(targets, joint_ids=self.joint_ids)
            self.robot.set_joint_velocity_target(torch.zeros_like(targets), joint_ids=self.joint_ids)
            self.robot.set_joint_effort_target(torch.zeros_like(targets), joint_ids=self.joint_ids)

    def _sample_power(self):
        """Sample completed substeps using lagged thrust and the applied PD torque estimate."""
        if self._power_samples == self._physics_samples:
            return
        # PD is implicit in PhysX. computed_torque is its unsaturated PD demand
        # estimate from this completed substep, including the final substep.
        # Direct torques and explicit INDI PID are already scored elsewhere.
        use_pd = self.cfg.control_mode != "indi" and (
            self.cfg.control_mode != "thrust_and_torque_control" or self.cfg.hold_arm)
        if use_pd and self.cfg.rwd_scale_torque_limits:
            limit = self.cfg.effort_lim_sim
            self._action_limit_costs[:, 1] += relative_limit_cost(
                self.robot.data.computed_torque[:, self.joint_ids], -limit, limit, limit) / self.cfg.decimation
        omega = torch.sqrt(self._rotor_thrust.clamp_min(0.0) / self.cfg.kf)
        rotor_mechanical_w = (self.cfg.km * omega**3).sum(-1)
        self._power_sum_w[:, 0] += rotor_mechanical_w
        self._power_sum_w[:, 1] += rotor_mechanical_w * (1.0 / self.cfg.p_el_motor_efficiency - 1.0)
        # Implicit-actuator effort is a clipped PD estimate, not measured PhysX torque.
        torque = self.robot.data.applied_torque[:, self.joint_ids]
        dq = self.robot.data.joint_vel[:, self.joint_ids]
        # Signed joint power; negative values represent ideal regenerative transfer.
        self._power_sum_w[:, 2] += (torque * dq).sum(-1)
        self._power_sum_w[:, 3] += self.cfg.arm_copper_loss_coefficient * torque.square().sum(-1)
        self._power_samples += 1

    def _sample_physics_events(self):
        position = self.ball.data.root_pos_w - self.scene.env_origins
        velocity = self.ball.data.root_lin_vel_w
        matrix = self.ball_contact.data.force_matrix_w[:, 0]
        contact = (torch.linalg.vector_norm(matrix, dim=-1) > self.cfg.contact_force_threshold_n) & self._ball_active[:, None]
        fresh_hit = contact[:, 0] & ~self._hit
        paddle_pos, paddle_quat, _ = self._paddle_pose()
        relative = quat_apply_inverse(paddle_quat, position - paddle_pos)
        center_sigma = 0.75 * self.cfg.paddle_diameter_m / 2
        center_quality = torch.exp(-relative[:, 1:].square().sum(-1) / center_sigma**2)
        self._center_quality = torch.where(fresh_hit, center_quality, self._center_quality)
        self._new_hit |= fresh_hit
        self._hit |= contact[:, 0]
        self._flags["base"] |= contact[:, 1]
        self._flags["arm"] |= contact[:, 2:4].any(-1)
        self._flags["goal"] |= self._ball_active & (contact[:, 4] | crosses_goal(self._previous_ball, position, self.cfg))
        self._flags["floor"] |= self._ball_active & (contact[:, 5] | (position[:, 2] <= self.cfg.ball_radius_m + 0.001)) & ~self._clear

        # Robot net forces include legitimate paddle/ball impact. Remove the
        # equal-and-opposite ball normal force before checking other collisions.
        other_contact = self.robot_contact.data.net_forces_w[:, self._contact_ids] + matrix[:, :4]
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        gravity_b = quat_apply_inverse(self.robot.data.root_quat_w, self._gravity)
        self._flags["area"] |= ((root_pos < self._area_min) | (root_pos > self._area_max)).any(-1)
        self._flags["tilt"] |= gravity_b[:, 2] > -math.cos(math.radians(self.cfg.robot_tilt_threshold_degs))
        self._sample_robot_crashes(other_contact)

        returned = (position[:, 0] > self.cfg.keeper_plane_x + self.cfg.clear_distance_m) & (
            velocity[:, 0] > self.cfg.clear_speed_m_s
        )
        goal_front = self.cfg.goal_center[0] + self.cfg.goal_size[0] / 2 + self.cfg.ball_radius_m
        outside = (
            (position[:, 1] - self.cfg.goal_center[1]).abs() > self.cfg.goal_size[1] / 2 + self.cfg.ball_radius_m
        ) | (
            (position[:, 2] - self.cfg.goal_center[2]).abs() > self.cfg.goal_size[2] / 2 + self.cfg.ball_radius_m
        )
        safely_wide = (position[:, 0] < goal_front) & outside
        landed_in_front = self._hit & self._flags["floor"] & (position[:, 0] > goal_front)
        failed = (self._flags["base"] | self._flags["arm"] | self._flags["goal"]
                  | (self._flags["floor"] & ~landed_in_front) | self._flags["crash"]
                  | self._flags["tilt"] | self._flags["area"])
        # A paddle deflection onto the floor in front of the goal is a save, but
        # still pays the floor penalty. No-contact floor events never earn a save.
        self._clear |= self._ball_active & self._hit & (returned | safely_wide | landed_in_front) & ~failed
        escaped = (position[:, 0].abs() > 3.0) | (position[:, 1].abs() > 3.0) | (position[:, 2] > 4.5)
        self._flags["miss"] |= self._ball_active & escaped & ~self._clear
        self._previous_ball.copy_(position)

    def _sample_robot_crashes(self, other_contact):
        """Keep crash rewards unchanged; optionally exclude self forces from resets."""
        self._flags["crash"] |= (
            torch.linalg.vector_norm(other_contact[:, self._crash_contact_columns], dim=-1) > self.cfg.robot_crash_force_threshold_n
        ).any(-1)
        if not self.cfg.self_collision_triggers_reset:
            scene_forces = other_contact[:, self._crash_contact_columns].clone()
            for column, sensor in enumerate(self._self_contact_sensors):
                scene_forces[:, column] -= sensor.data.force_matrix_w[:, 0].sum(1)
            self._flags["external_crash"] |= (
                torch.linalg.vector_norm(scene_forces, dim=-1) > self.cfg.robot_crash_force_threshold_n
            ).any(-1)

    def _recovery_geometry(self):
        position = self.robot.data.root_pos_w - self.scene.env_origins
        distance = torch.linalg.vector_norm(position - self._home, dim=-1)
        speed = torch.linalg.vector_norm(self.robot.data.root_lin_vel_w, dim=-1)
        angular_speed = torch.linalg.vector_norm(self.robot.data.root_ang_vel_b, dim=-1)
        gravity_b = quat_apply_inverse(self.robot.data.root_quat_w, self._gravity)
        tilt = torch.acos((-gravity_b[:, 2]).clamp(-1, 1))
        joint_error = self.robot.data.joint_pos[:, self.joint_ids].abs().amax(-1)
        stable = ((distance < self.cfg.recovery_position_tolerance_m) & (speed < self.cfg.recovery_speed_m_s)
                  & (angular_speed < self.cfg.recovery_angular_speed_rad_s) & (tilt < self.cfg.recovery_tilt_rad)
                  & (joint_error < self.cfg.recovery_joint_tolerance_rad))
        score = torch.exp(-(distance / 0.35).square() - (speed / 0.5).square()
                          - (angular_speed / 0.8).square() - (tilt / 0.35).square() - (joint_error / 0.5).square())
        return stable, score

    def _get_dones(self):
        self._sample_physics_events()  # final physics substep of this policy transition
        success = self._clear & self._ball_active
        if self.cfg.require_recovery_for_save:
            stable = self._recovery_geometry()[0] & self._clear
            self._recovery_time = torch.where(stable, self._recovery_time + self.step_dt, 0.0)
            success = success & (self._recovery_time >= self.cfg.recovery_dwell_s)
        deadline = self._ball_active & (self._shot_steps * self.step_dt >= self.cfg.shot_timeout_s) & ~success
        live_floor = self._flags["floor"] & ~self._clear
        crash = self._flags["crash"] | self._flags["tilt"]
        self._outcome = terminal_outcome(self._flags["base"], self._flags["arm"], crash,
                                        self._flags["goal"], live_floor, self._flags["area"],
                                        self._flags["miss"], deadline, success)
        contact_crash = self._flags["crash" if self.cfg.self_collision_triggers_reset else "external_crash"]
        reset_crash = ((contact_crash & self.cfg.reset_robot_on_robot_crash)
                       | (self._flags["tilt"] & self.cfg.reset_robot_on_tilt_threshold))
        # Other ball results end only the shot; goal resets are independently optional.
        terminated = reset_crash | self._flags["area"] | (self._flags["goal"] & self.cfg.reset_on_goal)
        time_limit = self._episode_steps >= self.max_episode_length
        return terminated | (time_limit & self.cfg.is_finite_horizon), time_limit & ~terminated & (not self.cfg.is_finite_horizon)

    def _get_rewards(self):
        self._sample_power()  # final completed substep, before automatic resets
        mean_power_w = self._power_sum_w / self.cfg.decimation
        target = (self._interception()[0] if self.cfg.progress_use_prediction
                  else self.ball.data.root_pos_w - self.scene.env_origins)
        progress = plane_progress(self._previous_paddle, self._paddle_pose()[0], target, self.cfg.rwd_scale_progress)
        progress *= self._old_progress_active
        rate = torch.zeros_like(progress)
        if self.cfg.rwd_scale_intercept or self.cfg.rwd_scale_alignment:
            potential = self._interception()[3]
            rate += self.cfg.rwd_scale_intercept * potential
            if self.cfg.rwd_scale_alignment:
                incoming = -self.ball.data.root_lin_vel_w
                direction = incoming / torch.linalg.vector_norm(incoming, dim=-1, keepdim=True).clamp_min(0.1)
                alignment = (self._paddle_pose()[2] * direction).sum(-1).square()
                rate += self.cfg.rwd_scale_alignment * potential * alignment
        if self.cfg.rwd_scale_recovery:
            rate += self.cfg.rwd_scale_recovery * self._recovery_geometry()[1] * self._clear
        if self.cfg.rwd_scale_action_rate:
            # Implicit-actuator effort is a clipped estimate, not measured PhysX torque.
            self._actuator_commands[:, 6:] = (
                self.robot.data.applied_torque[:, self.joint_ids] / (2 * self.cfg.effort_lim_sim))
            delta = self._actuator_commands - self._previous_actuator_commands
            rate += self.cfg.rwd_scale_action_rate * (delta[:, :6].square().mean(-1) + delta[:, 6:].square().mean(-1))
        if self.cfg.rwd_scale_body_angular_velocity:
            rate += self.cfg.rwd_scale_body_angular_velocity * self.robot.data.root_ang_vel_b.square().sum(-1)
        if self.cfg.rwd_scale_body_linear_velocity:
            rate += self.cfg.rwd_scale_body_linear_velocity * self.robot.data.root_lin_vel_w.square().sum(-1)
        if self.cfg.rwd_scale_thrust:
            rate += self.cfg.rwd_scale_thrust * (self._rotor_thrust / self._hover_per_rotor - 1).square().mean(-1)
        event_values = torch.tensor([0.0, self.cfg.rwd_scale_ball_base_collision, self.cfg.rwd_scale_ball_arm_collision,
                                     self.cfg.rwd_scale_goal, self.cfg.rwd_scale_ball_floor,
                                     self.cfg.rwd_scale_leave_goal_area,
                                     self.cfg.rwd_scale_deadline, self.cfg.rwd_scale_save, self.cfg.rwd_scale_robot_crash,
                                     self.cfg.rwd_scale_miss], device=self.device)
        hit_reward = (self.cfg.rwd_scale_hit + self.cfg.rwd_scale_centered_hit * self._center_quality)
        hit_reward *= self._new_hit & ((self._outcome == 0) | (self._outcome == 7))
        floor_after_save = self._flags["floor"] & self._clear & ((self._outcome == 0) | (self._outcome == 7))
        terms = {"plane_progress": progress, "outcome": event_values[self._outcome],
                 "floor_after_save": self.cfg.rwd_scale_ball_floor * floor_after_save,
                 "optional_rates": rate * self.step_dt, "optional_hit": hit_reward,
                 "energy": self.cfg.rwd_scale_energy * mean_power_w.sum(-1),
                 "thrust_limits": self.cfg.rwd_scale_thrust_limits * self._action_limit_costs[:, 0],
                 "torque_limits": self.cfg.rwd_scale_torque_limits * self._action_limit_costs[:, 1],
                 "action_limits": self.cfg.rwd_scale_action_limits * self._action_limit_costs[:, 2]}
        reward = sum(terms.values())
        self._episode_rewards += reward
        for name, term in terms.items():
            self._episode_terms[name] += term
        # Copy before auto-reset so validation/evaluation sees the transition outcome.
        self.extras["goalkeeper_outcome"] = self._outcome.clone()
        self.extras["goalkeeper_paddle_hit"] = self._new_hit.clone()
        self.extras["goalkeeper_had_paddle_hit"] = self._hit.clone()
        self.extras["goalkeeper_robot_contact_crash"] = self._flags["crash"].clone()
        self.extras["goalkeeper_tilt_threshold"] = self._flags["tilt"].clone()
        self.extras["goalkeeper_reward_terms"] = {name: term.clone() for name, term in terms.items()}
        self.extras["power_mean_w"] = {
            "rotor_mechanical": mean_power_w[:, 0].clone(),
            "rotor_dissipative": mean_power_w[:, 1].clone(),
            "rotor_electrical": mean_power_w[:, :2].sum(-1),
            "arm_mechanical": mean_power_w[:, 2].clone(),
            "arm_copper_loss": mean_power_w[:, 3].clone(),
            "mechanical": mean_power_w[:, 0] + mean_power_w[:, 2],
            "dissipative": mean_power_w[:, 1] + mean_power_w[:, 3],
            "total_estimate": mean_power_w.sum(-1),
        }
        finished = self._ball_active & (self._outcome != 0)
        self.extras["goalkeeper_shot_finished"] = finished.clone()
        self.extras["goalkeeper_shot_stage"] = self._shot_stage.clone()
        ids = finished.nonzero().flatten()
        if len(ids):
            self._record_shots(ids)
            self._ball_active[ids] = False
            if self.cfg.shot_resampling_mode == "resolution":
                self._next_launch_steps[ids] = sample_shot_delay_steps(len(ids), self.device, self.cfg, self.step_dt)
            self._park_ball(ids)
        return reward

    def _get_observations(self):
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        quaternion = self.robot.data.root_quat_w
        paddle_pos, _, normal = self._paddle_pose()
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        target, time_s, valid, _ = self._interception()
        common = [quat_apply_inverse(quaternion, self._home - root_pos),  # base position wrt home on the base frame
                  matrix_from_quat(quaternion).flatten(start_dim=1),  # base-to-world rotation matrix, row-major (9)
                  self.robot.data.root_lin_vel_b, # linear vel
                  self.robot.data.root_ang_vel_b, # angular vel
                  self.robot.data.joint_pos[:, self.joint_ids] / self._joint_scale, # joints
                  self.robot.data.joint_vel[:, self.joint_ids] / self.cfg.velocity_lim_sim, # joint vels
                  quat_apply_inverse(quaternion, paddle_pos - root_pos), # vector base->padle expressed base frame. 
                  quat_apply_inverse(quaternion, normal) # normal vector of the racket expressed base frame
                  ]
        ball_parts = [quat_apply_inverse(quaternion, ball_pos - root_pos), # vector base->ball in base frame 
                      quat_apply_inverse(quaternion, self.ball.data.root_lin_vel_w - self.robot.data.root_lin_vel_w), # velocitat relative de la pilota respecte base (base frame); smart choice 
                      quat_apply_inverse(quaternion, target - paddle_pos),# distance racket to plane crossing point (obs_actor_includes_ball_plane_intersection_point)
                      time_s[:, None], valid[:, None].float() #  predicted time until crossing plane, clipped to 0–1 seconds. (obs_actor_includes_time_ball_collides_plane)
                      ]
        ball_parts = [value * self._ball_active[:, None] for value in ball_parts]
        # obs_actor_includes_phase
        phase = torch.stack((self._hit.float(), # if hit the ball with raquet 
                             self._clear.float(), # if ball is considered cleared: past the x=0.3 line
                             (self._shot_steps * self.step_dt / self.cfg.shot_timeout_s).clamp_max(1), # time left before timeout
                             (self._recovery_time / self.cfg.recovery_dwell_s).clamp_max(1),
                             self._ball_active.float(), 
                             self._next_launch_steps * self.step_dt / max(self.cfg.shot_resampling_time_range_s[1], self.step_dt), # next launch time seconds
                             self._episode_steps * self.step_dt / self.cfg.episode_length_s), dim=-1 # episode progress
                             )
        tail = [self.actions]
        if self.cfg.control_mode == "PI_rotors_PD_joint_pos_control":
            # Direct modes already command each rotor in the action; PI hides thrusts and its integral state.
            tail += [self._rotor_thrust / self._hover_per_rotor, self._rate_integral]
        elif self.cfg.control_mode == "indi":
            tail += [self._indi.observation()]
        # The flags simplify only the actor; the critic keeps every privileged input.
        point, time = self.cfg.obs_actor_includes_ball_plane_intersection_point, self.cfg.obs_actor_includes_time_ball_collides_plane
        optional = [(point, ball_parts[2]), (time, ball_parts[3]),
                    (point or time, ball_parts[4]),  # prediction-valid flag
                    (self.cfg.obs_actor_includes_phase, phase)]
        policy = common + ball_parts[:2] + [value for used, value in optional if used] + tail
        if self.cfg.obs_actor_includes_ball_params:
            policy.append(self._ball_parameters)
        critic = common + ball_parts + [phase] + tail + [self._ball_parameters]
        if self.cfg.critic_obs_includes_rotor_omega:
            # Applied thrust retains motor lag, unlike the latest policy command.
            critic.append(torch.sqrt(self._rotor_thrust.clamp_min(0.0) / self.cfg.kf))
        return {"policy": torch.cat(policy, dim=-1),
                "critic": torch.cat(critic, dim=-1)}

    def _update_curriculum(self, env_ids):
        """Log full episodes and gate stage-pure episode-return checkpoints."""
        completed = env_ids[self._episode_steps[env_ids] > 0]
        if not len(completed):
            return
        self.extras["log"] = {"Episode/return": self._episode_rewards[completed].mean(),
                              "Episode/shots": self._episode_shots[completed].float().mean(),
                              "Episode/saves": self._episode_saves[completed].float().mean(),
                              "Episode/duration_s": (self._episode_steps[completed] * self.step_dt).mean(),
                              "Goalkeeper/cumulative_save_fraction": (
                                  self.outcome_counts[7] / self.outcome_counts.sum().clamp_min(1)
                              ),
                              "Goalkeeper/paddle_contact_fraction": self._episode_paddle_hits[completed].sum() / self._episode_shots[completed].sum().clamp_min(1),
                              "Goalkeeper/ball_clearance_fraction": self._episode_saves[completed].sum() / self._episode_shots[completed].sum().clamp_min(1),
                              "Curriculum/collection_stage": self._episode_stage[completed].float().mean()}
        # RSL selects log keys from the first packet: preserve episode-only packets.
        # Per-environment interval means remain available on every step in extras.
        self.extras["log"].update({f"Power/{name}_mean_w": value.mean()
                                  for name, value in self.extras["power_mean_w"].items()})
        self.extras["log"].update({f"Episode_Reward/{name}": value[completed].mean()
                                  for name, value in self._episode_terms.items()})
        if self._desired_actuation is not None:
            self.extras["log"].update(self._desired_actuation.packet(completed))
        current = self._episode_stage[completed] == self.shot_level
        stale = (~current).nonzero()
        if len(stale):
            self.curriculum_stage_pure_episodes = len(completed) - 1 - stale[-1].item()
        else:
            self.curriculum_stage_pure_episodes += len(completed)
        retained = min(self._stage_shots, len(self._curriculum_window))
        self.extras["log"].update({"Curriculum/selected_stage": self.shot_level,
                                  "Curriculum/stage_pure_save_fraction": self._curriculum_window.sum() / max(1, retained),
                                  "Curriculum/window_completed_shots": retained,
                                  "Curriculum/age_policy_steps": self.common_step_counter - self._stage_start_step})

    def _record_shots(self, completed):
        """Count each shot once. In-flight old-stage shots cannot promote a new stage."""
        completed = completed[self._outcome[completed] != 0]
        if not len(completed):
            return
        codes = self._outcome[completed]
        self.outcome_counts += torch.bincount(codes, minlength=10)
        self._episode_shots[completed] += 1
        self._episode_saves[completed] += (codes == 7).long()
        self._episode_paddle_hits[completed] += self._hit[completed].long()
        eligible = completed[self._shot_stage[completed] == self.shot_level]
        if len(eligible):
            outcomes = (self._outcome[eligible] == 7).float()
            self._stage_shots += len(outcomes)
            outcomes = outcomes[-len(self._curriculum_window):]
            indices = (torch.arange(len(outcomes), device=self.device) + self._window_cursor) % len(
                self._curriculum_window
            )
            self._curriculum_window[indices] = outcomes
            self._window_cursor = (self._window_cursor + len(outcomes)) % len(self._curriculum_window)
            retained = min(self._stage_shots, len(self._curriculum_window))
            rate = self._curriculum_window.sum() / retained
            if self.shot_level < len(self.cfg.shot_half_extents_yz) - 1 and promotion_ready(
                self.common_step_counter - self._stage_start_step, self._stage_shots, float(rate), self.cfg
            ):
                self.shot_level += 1
                self._stage_start_step = self.common_step_counter
                self._curriculum_window.zero_()
                self._window_cursor = 0
                self._stage_shots = 0
                self.curriculum_stage_pure_episodes = 0

    def get_curriculum_global_idx(self) -> int:
        """Shot level; train.py keeps one best_model_curriculum_idx_<level>.pt per level."""
        return self.shot_level

    def _park_ball(self, env_ids):
        """Remove a resolved ball without writing any robot/controller state."""
        state = self.ball.data.default_root_state[env_ids].clone()
        state[:, :3] = torch.tensor(self.cfg.ball_park_position, device=self.device) + self.scene.env_origins[env_ids]
        state[:, 7:] = 0.0
        self.ball.write_root_state_to_sim(state, env_ids)
        self.ball_contact.reset(env_ids)
        self._previous_ball[env_ids] = torch.tensor(self.cfg.ball_park_position, device=self.device)

    def _launch_ball(self, env_ids):
        """Re-use the single ball; per-shot resets never teleport the robot."""
        n = len(env_ids)
        position, velocity, _, _ = sample_shots(n, self.device, self.cfg, self.shot_level)
        state = self.ball.data.default_root_state[env_ids].clone()
        state[:, :3] = position + self.scene.env_origins[env_ids]
        state[:, 7:10] = velocity
        state[:, 10:] = 0.0
        self.ball.write_root_state_to_sim(state, env_ids)
        self.ball_contact.reset(env_ids)
        self._previous_ball[env_ids] = position
        ids_cpu = env_ids.cpu()
        masses = torch.empty(n).uniform_(*self.cfg.ball_mass_range_kg)
        self._ball_masses[ids_cpu, 0] = masses
        self._ball_inertias[ids_cpu] = torch.eye(3).reshape(1, 9) * (0.4 * masses * self.cfg.ball_radius_m**2)[:, None]
        self._ball_materials[ids_cpu, :, 2] = torch.empty(n, 1).uniform_(*self.cfg.ball_restitution_range)
        self._ball_parameters[env_ids, 0] = masses.to(self.device)
        self._ball_parameters[env_ids, 1] = self._ball_materials[ids_cpu, 0, 2].to(self.device)
        self.ball.root_physx_view.set_masses(self._ball_masses, ids_cpu)
        self.ball.root_physx_view.set_inertias(self._ball_inertias, ids_cpu)
        self.ball.root_physx_view.set_material_properties(self._ball_materials, ids_cpu)
        for buffer in (self._hit, self._clear, self._new_hit, self._center_quality, self._shot_steps, self._recovery_time):
            buffer[env_ids] = 0
        self._shot_stage[env_ids] = self.shot_level
        self._next_launch_steps[env_ids] = sample_shot_delay_steps(n, self.device, self.cfg, self.step_dt)
        self._ball_active[env_ids] = True

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._update_curriculum(env_ids)
        if self._desired_actuation is not None:
            self._desired_actuation.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        position, self.spawn_grid_index[env_ids] = sample_robot_positions(n, self.device, self.cfg, self.shot_level)
        root[:, :3] = position + self.scene.env_origins[env_ids]
        root[:, 3:7] = torch.tensor(self.cfg.robot_home_quat, device=self.device)
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos[env_ids],
            torch.zeros_like(self.robot.data.default_joint_vel[env_ids]), env_ids=env_ids,
        )
        joint_zero = torch.zeros(n, len(self.joint_ids), device=self.device)
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[env_ids][:, self.joint_ids], joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(joint_zero, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_effort_target(joint_zero, joint_ids=self.joint_ids, env_ids=env_ids)
        for buffer in (self.actions, self._action_limit_costs, self._rate_integral, self._recovery_time,
                       self._hit, self._clear, self._new_hit, self._center_quality, self._outcome,
                       self._episode_rewards, self._episode_steps, self._episode_shots,
                       self._episode_saves, self._episode_paddle_hits, self._shot_steps):
            buffer[env_ids] = 0
        for flag in self._flags.values():
            flag[env_ids] = False
        for value in self._episode_terms.values():
            value[env_ids] = 0
        self._rotor_thrust[env_ids] = self._hover_per_rotor
        if self.cfg.control_mode == "indi":
            self._indi.reset(env_ids)
        for buffer in (self._actuator_commands, self._previous_actuator_commands):
            buffer[env_ids, :6] = 1.0 / self.cfg.thrust_to_weight
            buffer[env_ids, 6:] = 0.0
        self._power_sum_w[env_ids] = 0.0
        self._episode_stage[env_ids] = self.shot_level
        self._launch_ball(env_ids)
