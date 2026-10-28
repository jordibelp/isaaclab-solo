# SPDX-License-Identifier: BSD-3-Clause
"""Borinot goalkeeper with a summed reward: task + integrity + desirability."""
from __future__ import annotations

import math

import torch
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from isaaclab_tasks.direct.borinot_goalkeeper.actuator_delay import ActuatorDelay, actuator_delay_bounds
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env import BorinotGoalkeeperEnv
from isaaclab_tasks.direct.borinot.action_limits import raw_action_limit_costs
from isaaclab_tasks.direct.borinot.inertial_properties import apply_inertial_properties, validate_inertial_properties
from isaaclab_tasks.direct.borinot_goalkeeper.events import configure_randomization
from isaaclab_tasks.direct.borinot_goalkeeper.paddle import attach_paddle
from isaaclab_tasks.direct.borinot_goalkeeper.task_math import (
    crosses_goal, plane_progress, predict_intercept, promotion_ready, sample_robot_positions, sample_shot_delay_steps,
    sample_shots, validate_reset_grid, validate_shot_schedule,
)

from .borinot_goalkeeper_simple_env_cfg import BorinotGoalkeeperSimpleEnvCfg

# The original task's goal and save codes, so the shared player and score panel work unchanged.
OUTCOME_GOAL = 3
OUTCOME_SAVE = 7  # the shot ended without a goal
EVENTS = ("hit_racket", "hit_arm", "hit_base", "goal", "base_crash", "self_collision", "leave_area", "tilt")
REWARD_TERMS = {
    "task": ("progress", "hit_racket", "hit_arm", "hit_base", "goal"),
    "integrity": ("base_crash", "self_collision", "leave_area", "tilt"),
    "desirability": ("action_rate", "energy", "thrust_limits", "torque_limits", "action_limits"),
}


class BorinotGoalkeeperSimpleEnv(DirectRLEnv):
    """Continuous one-ball shots; the reward is the plain sum of task, integrity and desirability terms.

    A shot ends when its ball scores, passes the goal plane, leaves the tracking region or times out.
    Only the robot reset switches and the time limit end an episode.
    """

    cfg: BorinotGoalkeeperSimpleEnvCfg
    # Rotor/arm control and the power model are shared, unchanged, with the original task.
    _apply_action = BorinotGoalkeeperEnv._apply_action
    _sample_power = BorinotGoalkeeperEnv._sample_power
    _init_indi = BorinotGoalkeeperEnv._init_indi
    get_code_snapshot_files = BorinotGoalkeeperEnv.get_code_snapshot_files

    def __init__(self, cfg: BorinotGoalkeeperSimpleEnvCfg, render_mode=None, **kwargs):
        validate_inertial_properties(cfg.inertial_properties)
        cfg.configure_control()
        configure_randomization(cfg)
        validate_reset_grid(cfg)
        validate_shot_schedule(cfg)
        if not 0 <= cfg.shot_level < len(cfg.shot_half_extents_yz):
            raise ValueError("shot_level must index shot_half_extents_yz.")
        if not 1 <= cfg.curriculum_min_shots <= cfg.curriculum_window_shots:
            raise ValueError("Require 1 <= curriculum_min_shots <= curriculum_window_shots.")
        if cfg.robot_crash_force_threshold_n <= 0:
            raise ValueError("robot_crash_force_threshold_n must be positive.")
        if cfg.curriculum_min_steps < 1 or cfg.motor_time_constant_s < 0:
            raise ValueError("Curriculum dwell must be positive and motor time constant must be non-negative.")
        actuator_delay_bounds(cfg)  # fail before the simulator starts
        super().__init__(cfg, render_mode, **kwargs)
        for sensor, filters in ((self.ball_contact, 5), (self.base_contact, 4), (self.paddle_contact, 1)):
            if sensor.contact_physx_view.filter_count != filters:
                raise RuntimeError("A contact filter did not resolve; check the robot body names.")
        self.joint_ids, _ = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        self.base_id = self.robot.find_bodies("base")[0][0]
        self.paddle_id = self.robot.find_bodies("paddle")[0][0]
        self._home = torch.tensor(cfg.robot_home, device=self.device)
        self._area_min = torch.tensor(
            [cfg.goal_area_x_m[0], cfg.goal_area_y_m[0], cfg.goal_area_z_m[0]], device=self.device
        )
        self._area_max = torch.tensor(
            [cfg.goal_area_x_m[1], cfg.goal_area_y_m[1], cfg.goal_area_z_m[1]], device=self.device
        )
        self._x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        self._gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, -1)

        # Control state read by the shared _apply_action() and _sample_power().
        self.actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._actuator_delay = ActuatorDelay(cfg, self.num_envs, cfg.action_space, self.device)
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
        # Keep action scaling nominal; each robot still has its sampled physical mass.
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
        if cfg.control_mode == "indi":
            self._init_indi()
        # Columns: rotor mechanical, rotor dissipative, arm mechanical, arm copper loss.
        self._power_sum_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._power_samples = 0
        self._physics_samples = 0
        self._episode_energy_j = torch.zeros(self.num_envs, device=self.device)
        self._unpaid_energy_j = torch.zeros(self.num_envs, device=self.device)  # energy option: since the last payment

        # One ball per environment. Each ball pays its first robot contact and its goal at most once.
        self._ball_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._touched = torch.zeros_like(self._ball_active)  # first robot contact already happened
        self._ball_done = torch.zeros_like(self._ball_active)  # scored, passed the goal plane or escaped
        self._progress_active = torch.zeros_like(self._ball_active)
        self._events = {name: torch.zeros_like(self._ball_active) for name in EVENTS}  # current action interval
        self._previous_paddle = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_ball = torch.zeros(self.num_envs, 3, device=self.device)
        self._outcome = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._shot_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_steps = torch.zeros_like(self._shot_steps)
        self._next_launch_steps = torch.zeros_like(self._shot_steps)
        self._shot_stage = torch.full_like(self._shot_steps, cfg.shot_level)
        # PhysX property setters use CPU buffers; retain them rather than querying every launch.
        self._ball_masses = self.ball.root_physx_view.get_masses()
        self._ball_inertias = self.ball.root_physx_view.get_inertias()
        self._ball_materials = self.ball.root_physx_view.get_material_properties()
        self._ball_parameters = torch.zeros(self.num_envs, 2, device=self.device)

        # Episode statistics and curriculum.
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._episode_terms = {f"{group}/{name}": torch.zeros(self.num_envs, device=self.device)
                               for group, names in REWARD_TERMS.items() for name in names}
        self._episode_counts = {name: torch.zeros_like(self._shot_steps)
                                for name in ("shots", "saves", "goals", "hit_racket", "hit_arm", "hit_base")}
        self.outcome_counts = torch.zeros(10, dtype=torch.long, device=self.device)
        self._episode_stage = torch.full_like(self._shot_steps, cfg.shot_level)
        self.spawn_grid_index = torch.full_like(self._shot_steps, -1)
        self.shot_level = cfg.shot_level
        self._stage_start_step = 0
        self._curriculum_window = torch.zeros(cfg.curriculum_window_shots, device=self.device)
        self._window_cursor = 0
        self._stage_shots = 0
        # Newest consecutive completed episodes that started at the current level. Training only
        # scores a stage's best checkpoint once the whole reward buffer comes from that stage.
        self.curriculum_stage_pure_episodes = 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        stage = sim_utils.get_current_stage()
        attach_paddle(stage, "/World/envs/env_0/Robot", self.cfg)
        apply_inertial_properties(stage, "/World/envs/env_0/Robot", self.cfg.inertial_properties)
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
        # Filtered contact forces: one source body per sensor; force columns follow the filter order.
        env, robot = "/World/envs/env_.*/", "/World/envs/env_.*/Robot/"
        self.ball_contact = self._contact_sensor("ball", env + "Ball", [
            robot + "paddle", robot + "flying_arm_2__link1", robot + "flying_arm_2__link2", robot + "base", env + "Goal"])
        # Base against the scene (crash), then against the two robot bodies it is not jointed to.
        self.base_contact = self._contact_sensor("base", robot + "base", [
            env + "Floor", env + "Goal", robot + "flying_arm_2__link2", robot + "paddle"])
        # The last unjointed robot pair. PhysX never reports jointed parent-child pairs.
        self.paddle_contact = self._contact_sensor("paddle", robot + "paddle", [robot + "flying_arm_2__link1"])
        light = sim_utils.DomeLightCfg(intensity=1800.0)
        light.func("/World/Light", light)

    def _contact_sensor(self, name, prim_path, filters):
        sensor = ContactSensor(ContactSensorCfg(
            prim_path=prim_path, update_period=0.0, history_length=0, filter_prim_paths_expr=filters))
        self.scene.sensors[f"{name}_contacts"] = sensor
        return sensor

    def _paddle_pose(self):
        position = self.robot.data.body_pos_w[:, self.paddle_id] - self.scene.env_origins
        quaternion = self.robot.data.body_quat_w[:, self.paddle_id]
        return position, quaternion, quat_apply(quaternion, self._x_axis)

    def _pre_physics_step(self, actions):
        self._action_limit_costs.copy_(raw_action_limit_costs(actions, self.cfg, self._hover_per_rotor))
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
        # Progress compares racket positions across this interval, only before the ball's first contact.
        self._previous_paddle.copy_(self._paddle_pose()[0])
        self._progress_active.copy_(self._ball_active & ~self._touched)
        for event in self._events.values():
            event.zero_()
        self._physics_samples = 0
        self._power_sum_w.zero_()
        self._power_samples = 0

    def _sample_physics_events(self):
        """Record this interval's events after every completed physics substep."""
        ball = self.ball.data.root_pos_w - self.scene.env_origins
        live = self._ball_active & ~self._ball_done
        touching = (torch.linalg.vector_norm(self.ball_contact.data.force_matrix_w[:, 0], dim=-1)
                    > self.cfg.contact_force_threshold_n)
        racket, arm, base = touching[:, 0], touching[:, 1:3].any(-1), touching[:, 3]
        # Only the ball's first robot contact pays. In a same-substep tie the racket wins, then the arm.
        first = live & ~self._touched
        self._events["hit_racket"] |= first & racket
        self._events["hit_arm"] |= first & arm & ~racket
        self._events["hit_base"] |= first & base & ~racket & ~arm
        self._touched |= live & (racket | arm | base)
        goal = live & (touching[:, 4] | crosses_goal(self._previous_ball, ball, self.cfg))
        self._events["goal"] |= goal
        # Behind the goal plane or outside the tracking region, the ball can no longer score.
        goal_plane = self.cfg.goal_center[0] + self.cfg.goal_size[0] / 2 + self.cfg.ball_radius_m
        escaped = (ball[:, 0].abs() > 3.0) | (ball[:, 1].abs() > 3.0) | (ball[:, 2] > 4.5)
        self._ball_done |= live & (goal | (ball[:, 0] <= goal_plane) | escaped)
        self._previous_ball.copy_(ball)

        threshold = self.cfg.robot_crash_force_threshold_n
        base_forces = torch.linalg.vector_norm(self.base_contact.data.force_matrix_w[:, 0], dim=-1)
        paddle_forces = torch.linalg.vector_norm(self.paddle_contact.data.force_matrix_w[:, 0], dim=-1)
        self._events["base_crash"] |= (base_forces[:, :2] > threshold).any(-1)  # floor or goal
        self._events["self_collision"] |= (base_forces[:, 2:] > threshold).any(-1) | (paddle_forces > threshold).any(-1)
        root = self.robot.data.root_pos_w - self.scene.env_origins
        self._events["leave_area"] |= ((root < self._area_min) | (root > self._area_max)).any(-1)
        gravity_b = quat_apply_inverse(self.robot.data.root_quat_w, self._gravity)
        self._events["tilt"] |= gravity_b[:, 2] > -math.cos(math.radians(self.cfg.robot_tilt_threshold_degs))

    def _get_dones(self):
        self._sample_physics_events()  # final physics substep of this policy transition
        timeout = self._shot_steps * self.step_dt >= self.cfg.shot_timeout_s
        finished = self._ball_active & (self._ball_done | timeout)
        self._outcome = finished * torch.where(self._events["goal"], OUTCOME_GOAL, OUTCOME_SAVE)
        events, cfg = self._events, self.cfg
        terminated = ((events["base_crash"] & cfg.reset_robot_on_robot_crash)
                      | (events["self_collision"] & cfg.self_collision_triggers_reset)
                      | (events["leave_area"] & cfg.reset_robot_on_leave_area)
                      | (events["tilt"] & cfg.reset_robot_on_tilt_threshold)
                      | (events["goal"] & cfg.reset_on_goal))
        time_limit = self._episode_steps >= self.max_episode_length
        return terminated | (time_limit & cfg.is_finite_horizon), time_limit & ~terminated & (not cfg.is_finite_horizon)

    def _get_rewards(self):
        self._sample_power()  # final completed substep, before automatic resets
        groups = {"task": self._task_rewards(), "integrity": self._integrity_rewards(),
                  "desirability": self._desirability_rewards()}
        terms = {f"{group}/{name}": value for group, values in groups.items() for name, value in values.items()}
        reward = sum(terms.values())
        self._episode_rewards += reward
        for name, value in terms.items():
            self._episode_terms[name] += value
        for name in ("hit_racket", "hit_arm", "hit_base"):
            self._episode_counts[name] += self._events[name]
        # Copy before auto-reset so validation/evaluation sees the transition outcome.
        self.extras["goalkeeper_outcome"] = self._outcome.clone()
        self.extras["goalkeeper_events"] = {name: event.clone() for name, event in self._events.items()}
        self.extras["goalkeeper_reward_terms"] = {name: value.clone() for name, value in terms.items()}
        mean_power_w = self._power_sum_w / self.cfg.decimation
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
        finished = self._outcome != 0
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

    def _task_rewards(self):
        """Stop the ball: racket-ball progress, then its first robot contact and its goal, each paid once."""
        events = self._events
        ball = self.ball.data.root_pos_w - self.scene.env_origins
        progress = plane_progress(self._previous_paddle, self._paddle_pose()[0], ball, self.cfg.rwd_scale_progress)
        return {"progress": progress * self._progress_active,
                "hit_racket": self.cfg.rwd_scale_hit_racket * events["hit_racket"],
                "hit_arm": self.cfg.rwd_scale_hit_arm * events["hit_arm"],
                "hit_base": self.cfg.rwd_scale_hit_base * events["hit_base"],
                "goal": self.cfg.rwd_scale_goal * events["goal"]}

    def _integrity_rewards(self):
        """Keep the robot safe. Paid on every action interval with the event, whatever the reset switches."""
        events = self._events
        return {"base_crash": self.cfg.rwd_scale_base_crash * events["base_crash"],
                "self_collision": self.cfg.rwd_scale_self_collision * events["self_collision"],
                "leave_area": self.cfg.rwd_scale_leave_goal_area * events["leave_area"],
                "tilt": self.cfg.rwd_scale_tilt * events["tilt"]}

    def _desirability_rewards(self):
        """Prefer smooth actuator commands and low energy use."""
        # Group-mean squared changes of normalized post-clipping commands: thrust / F_max, effort / (2*tau_max).
        # Implicit-actuator effort is a clipped estimate, not measured PhysX torque.
        self._actuator_commands[:, 6:] = (
            self.robot.data.applied_torque[:, self.joint_ids] / (2 * self.cfg.effort_lim_sim))
        delta = self._actuator_commands - self._previous_actuator_commands
        action_rate = delta[:, :6].square().mean(-1) + delta[:, 6:].square().mean(-1)
        # Mechanical plus dissipative energy over this action's completed physics steps.
        step_energy_j = self._power_sum_w.sum(-1) * self.physics_dt
        self._episode_energy_j += step_energy_j
        # Default: pay each action's energy now. Option: pay the energy used since the previous payment
        # when a shot ends or the episode terminates. A time-limit cut pays nothing: the learner bootstraps
        # the unfinished shot, its energy included. Dividing by step_dt keeps the per-watt scale.
        energy_j = step_energy_j
        if self.cfg.minimize_energy_not_power:
            self._unpaid_energy_j += step_energy_j
            pay = (self._outcome != 0) | self.reset_terminated
            energy_j = self._unpaid_energy_j * pay
            self._unpaid_energy_j.masked_fill_(pay, 0.0)
        return {"action_rate": self.cfg.rwd_scale_action_rate * action_rate * self.step_dt,
                "energy": self.cfg.rwd_scale_energy * energy_j / self.step_dt,
                "thrust_limits": self.cfg.rwd_scale_thrust_limits * self._action_limit_costs[:, 0],
                "torque_limits": self.cfg.rwd_scale_torque_limits * self._action_limit_costs[:, 1],
                "action_limits": self.cfg.rwd_scale_action_limits * self._action_limit_costs[:, 2]}

    def _get_observations(self):
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        quaternion = self.robot.data.root_quat_w
        paddle_pos, _, normal = self._paddle_pose()
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        target, time_s, valid = predict_intercept(ball_pos, self.ball.data.root_lin_vel_w, self.cfg.keeper_plane_x)
        valid &= self._ball_active
        common = [quat_apply_inverse(quaternion, self._home - root_pos),  # home relative to the base, base frame
                  matrix_from_quat(quaternion).flatten(start_dim=1),  # base-to-world rotation matrix, row-major (9)
                  self.robot.data.root_lin_vel_b,
                  self.robot.data.root_ang_vel_b,
                  self.robot.data.joint_pos[:, self.joint_ids] / self._joint_scale,
                  self.robot.data.joint_vel[:, self.joint_ids] / self.cfg.velocity_lim_sim,
                  quat_apply_inverse(quaternion, paddle_pos - root_pos),  # base to racket, base frame
                  quat_apply_inverse(quaternion, normal)]  # racket face normal, base frame
        ball_parts = [quat_apply_inverse(quaternion, ball_pos - root_pos),  # base to ball, base frame
                      quat_apply_inverse(quaternion, self.ball.data.root_lin_vel_w - self.robot.data.root_lin_vel_w),
                      quat_apply_inverse(quaternion, target - paddle_pos),  # racket to predicted plane crossing
                      time_s[:, None], valid[:, None].float()]  # crossing time (clipped to 0-1 s) and validity
        ball_parts = [value * self._ball_active[:, None] for value in ball_parts]
        phase = torch.stack((self._touched.float(),  # first robot contact already happened
                             (self._shot_steps * self.step_dt / self.cfg.shot_timeout_s).clamp_max(1),
                             self._ball_active.float(),
                             self._next_launch_steps * self.step_dt / max(self.cfg.shot_resampling_time_range_s[1], self.step_dt),
                             self._episode_steps * self.step_dt / self.cfg.episode_length_s), dim=-1)
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
        return {"policy": torch.cat(policy, dim=-1),
                "critic": torch.cat(common + ball_parts + [phase] + tail + [self._ball_parameters], dim=-1)}

    def _update_curriculum(self, env_ids):
        """Log full episodes and gate stage-pure episode-return checkpoints."""
        completed = env_ids[self._episode_steps[env_ids] > 0]
        if not len(completed):
            return
        log = {"Episode/return": self._episode_rewards[completed].mean(),
               "Episode/duration_s": (self._episode_steps[completed] * self.step_dt).mean(),
               "Episode/energy_j": self._episode_energy_j[completed].mean(),
               "Episode/actuator_delay_s": self._actuator_delay.seconds[completed].mean(),
               "Goalkeeper/cumulative_save_fraction": (
                   self.outcome_counts[OUTCOME_SAVE] / self.outcome_counts.sum().clamp_min(1)),
               "Curriculum/collection_stage": self._episode_stage[completed].float().mean()}
        log.update({f"Episode/{name}": count[completed].float().mean() for name, count in self._episode_counts.items()})
        # Episode sum of every reward term, plus one total per category.
        for group, names in REWARD_TERMS.items():
            log[f"Episode_Reward/{group}"] = sum(self._episode_terms[f"{group}/{name}"][completed] for name in names).mean()
        log.update({f"Episode_Reward/{name}": value[completed].mean() for name, value in self._episode_terms.items()})
        # RSL selects log keys from the first packet: preserve episode-only packets.
        # Per-environment interval means remain available on every step in extras.
        log.update({f"Power/{name}_mean_w": value.mean() for name, value in self.extras["power_mean_w"].items()})
        current = self._episode_stage[completed] == self.shot_level
        stale = (~current).nonzero()
        if len(stale):
            self.curriculum_stage_pure_episodes = len(completed) - 1 - stale[-1].item()
        else:
            self.curriculum_stage_pure_episodes += len(completed)
        retained = min(self._stage_shots, len(self._curriculum_window))
        log.update({"Curriculum/selected_stage": self.shot_level,
                    "Curriculum/stage_pure_save_fraction": self._curriculum_window.sum() / max(1, retained),
                    "Curriculum/window_completed_shots": retained,
                    "Curriculum/age_policy_steps": self.common_step_counter - self._stage_start_step})
        self.extras["log"] = log

    def _record_shots(self, completed):
        """Count each shot once. A shot without a goal is a save. In-flight old-stage shots cannot promote."""
        codes = self._outcome[completed]
        self.outcome_counts += torch.bincount(codes, minlength=10)
        self._episode_counts["shots"][completed] += 1
        self._episode_counts["saves"][completed] += (codes == OUTCOME_SAVE).long()
        self._episode_counts["goals"][completed] += (codes == OUTCOME_GOAL).long()
        eligible = completed[self._shot_stage[completed] == self.shot_level]
        if not len(eligible):
            return
        outcomes = (self._outcome[eligible] == OUTCOME_SAVE).float()
        self._stage_shots += len(outcomes)
        outcomes = outcomes[-len(self._curriculum_window):]
        indices = (torch.arange(len(outcomes), device=self.device) + self._window_cursor) % len(self._curriculum_window)
        self._curriculum_window[indices] = outcomes
        self._window_cursor = (self._window_cursor + len(outcomes)) % len(self._curriculum_window)
        rate = self._curriculum_window.sum() / min(self._stage_shots, len(self._curriculum_window))
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
        for buffer in (self._touched, self._ball_done, self._shot_steps):
            buffer[env_ids] = 0
        self._shot_stage[env_ids] = self.shot_level
        self._next_launch_steps[env_ids] = sample_shot_delay_steps(n, self.device, self.cfg, self.step_dt)
        self._ball_active[env_ids] = True

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._update_curriculum(env_ids)
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
        for buffer in (self.actions, self._action_limit_costs, self._rate_integral, self._touched, self._ball_done, self._outcome,
                       self._episode_rewards, self._episode_steps, self._shot_steps, self._episode_energy_j,
                       self._unpaid_energy_j,
                       *self._events.values(), *self._episode_terms.values(), *self._episode_counts.values()):
            buffer[env_ids] = 0
        self._actuator_delay.reset(env_ids)  # after _update_curriculum logged the finished episodes' delays
        self._rotor_thrust[env_ids] = self._hover_per_rotor
        if self.cfg.control_mode == "indi":
            self._indi.reset(env_ids)
        for buffer in (self._actuator_commands, self._previous_actuator_commands):
            buffer[env_ids, :6] = 1.0 / self.cfg.thrust_to_weight
            buffer[env_ids, 6:] = 0.0
        self._power_sum_w[env_ids] = 0.0
        self._episode_stage[env_ids] = self.shot_level
        self._launch_ball(env_ids)
