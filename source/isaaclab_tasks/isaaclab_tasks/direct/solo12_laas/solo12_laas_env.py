# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import schemas as sim_schemas
from isaaclab.utils.buffers import DelayBuffer

from .solo12_laas_env_cfg import Solo12LaasEnvCfg


class Solo12LaasEnv(DirectRLEnv):
    cfg: Solo12LaasEnvCfg

    def __init__(self, cfg: Solo12LaasEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._delayed_processed_actions = torch.zeros_like(self._actions)

        max_action_delay = self.cfg.actuation_delay_range[1]
        self._action_delay_buffer = DelayBuffer(max_action_delay, self.num_envs, device=self.device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_resample_interval = max(1, int(round(self.cfg.command_resampling_time_s / self.step_dt)))
        if not 0.0 <= self.cfg.opposite_direction_cmd_prob <= 1.0:
            raise ValueError(
                "opposite_direction_cmd_prob must be between 0 and 1, "
                f"got {self.cfg.opposite_direction_cmd_prob}"
            )

        self._joint_ids, _ = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        self._base_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.base_body_name)
        self._feet_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.feet_body_names)
        self._thigh_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.undesired_contact_body_names)
        self._base_wrench_body_ids, _ = self._robot.find_bodies(self.cfg.base_body_name)
        self._joint_wrench_body_ids, _ = self._robot.find_bodies(self.cfg.joint_wrench_body_names)
        self._reset_joint_pos = self._build_reset_joint_pos()
        self._q_offset_action_and_obs = self._build_q_offset_action_and_obs()
        self._base_push_interval_step_range = self._seconds_range_to_steps(self.cfg.base_push_interval_range_s)
        self._base_push_duration_step_range = self._seconds_range_to_steps(self.cfg.base_push_duration_range_s)
        self._base_push_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._base_push_steps_until_next = self._sample_int_steps(
            self._base_push_interval_step_range, self.num_envs
        )
        self._base_push_forces_b = torch.zeros(
            self.num_envs, len(self._base_wrench_body_ids), 3, device=self.device
        )
        self._base_push_torques_b = torch.zeros_like(self._base_push_forces_b)
        self._base_push_force_curriculum_values = self._parse_base_push_force_curriculum()
        self._base_push_force_curriculum_idx = 0
        self._base_push_mean_reward_smooth: float | None = None
        self._base_push_last_curriculum_step = 0
        if self._base_push_force_curriculum_values:
            self._set_base_push_force_curriculum_level(0)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "force_transmited_through_joints",
                "foot_contact",
            ]
        }
        self._episode_reward_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._apply_configured_base_mass()
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_configured_base_mass(self):
        if self.cfg.base_mass is None:
            return

        source_robot_path = self.cfg.robot.prim_path.replace(self.scene.env_regex_ns, self.scene.env_prim_paths[0])
        source_base_path = f"{source_robot_path}/{self.cfg.base_body_name}"
        mass_cfg = sim_utils.MassPropertiesCfg(mass=self.cfg.base_mass)
        sim_schemas.define_mass_properties(source_base_path, mass_cfg)

    def _build_reset_joint_pos(self) -> torch.Tensor:
        initial_position = self.cfg.initial_position.lower()
        joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        if initial_position == "rigid":
            return joint_pos
        if initial_position == "flexed":
            initial_joint_pos_cfg = self.cfg.flexed_initial_joint_pos
        else:
            initial_joint_pos_cfg = self.cfg.initial_joint_pos_by_name.get(initial_position)
        if initial_joint_pos_cfg is None:
            supported_positions = sorted(["rigid", *self.cfg.initial_joint_pos_by_name.keys()])
            raise ValueError(
                f"Unsupported initial_position '{self.cfg.initial_position}'. "
                f"Use one of: {', '.join(supported_positions)}."
            )

        initial_joint_pos = torch.tensor(
            [initial_joint_pos_cfg[joint_name] for joint_name in self.cfg.joint_names],
            device=self.device,
            dtype=joint_pos.dtype,
        )
        joint_pos[:] = initial_joint_pos
        return joint_pos

    def _build_q_offset_action_and_obs(self) -> torch.Tensor:
        offset_cfg = self.cfg.q_offset_action_and_obs
        joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        if offset_cfg is None:
            offset = self._reset_joint_pos[0]
        elif isinstance(offset_cfg, dict):
            offset = torch.tensor(
                [offset_cfg[joint_name] for joint_name in self.cfg.joint_names],
                device=self.device,
                dtype=joint_pos.dtype,
            )
        elif isinstance(offset_cfg, (int, float)):
            offset = torch.full((len(self._joint_ids),), float(offset_cfg), device=self.device, dtype=joint_pos.dtype)
        else:
            offset = torch.tensor(offset_cfg, device=self.device, dtype=joint_pos.dtype)
            if offset.numel() != len(self._joint_ids):
                raise ValueError(
                    "q_offset_action_and_obs must be a scalar, a joint-name dictionary, "
                    f"or have {len(self._joint_ids)} values, got {offset.numel()}."
                )

        joint_pos[:] = offset.reshape(1, -1)
        return joint_pos

    def _seconds_range_to_steps(self, seconds_range: tuple[float, float]) -> tuple[int, int]:
        low_s, high_s = seconds_range
        if low_s < 0.0 or high_s < low_s:
            raise ValueError(f"Invalid seconds range: {seconds_range}")
        low_steps = max(1, int(round(low_s / self.step_dt)))
        high_steps = max(low_steps, int(round(high_s / self.step_dt)))
        return low_steps, high_steps

    def _sample_int_steps(self, step_range: tuple[int, int], count: int) -> torch.Tensor:
        low, high = step_range
        if low == high:
            return torch.full((count,), low, dtype=torch.long, device=self.device)
        return torch.randint(low, high + 1, (count,), dtype=torch.long, device=self.device)

    def _fill_uniform_range(self, tensor: torch.Tensor, value_range: tuple[float, float]):
        low, high = value_range
        if low == high:
            tensor.fill_(low)
        else:
            tensor.uniform_(low, high)

    def _parse_base_push_force_curriculum(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.cfg.forces_applied_to_base_curriculum)
        if any(value < 0.0 for value in values):
            raise ValueError(f"forces_applied_to_base_curriculum must be non-negative, got {values}.")
        if not 0.0 < self.cfg.forces_curriculum_smoothing <= 1.0:
            raise ValueError(
                "forces_curriculum_smoothing must be in (0, 1], "
                f"got {self.cfg.forces_curriculum_smoothing}."
            )
        return values

    def _set_base_push_force_curriculum_level(self, level_idx: int):
        force = self._base_push_force_curriculum_values[level_idx]
        self._base_push_force_curriculum_idx = level_idx
        self.cfg.base_push_force_xy_range = (-force, force)

    def _update_base_push_force_curriculum(self, completed_episode_returns: torch.Tensor):
        if (
            not self._base_push_force_curriculum_values
            or self._base_push_force_curriculum_idx >= len(self._base_push_force_curriculum_values) - 1
            or len(completed_episode_returns) == 0
        ):
            return

        if self.common_step_counter - self._base_push_last_curriculum_step < self.max_episode_length:
            return

        mean_reward = torch.mean(completed_episode_returns).item()
        if self._base_push_mean_reward_smooth is None:
            self._base_push_mean_reward_smooth = mean_reward
        else:
            smoothing = self.cfg.forces_curriculum_smoothing
            self._base_push_mean_reward_smooth = (
                (1.0 - smoothing) * self._base_push_mean_reward_smooth + smoothing * mean_reward
            )

        if self._base_push_mean_reward_smooth < self.cfg.forces_curriculum_threshold_reward:
            return

        self._set_base_push_force_curriculum_level(self._base_push_force_curriculum_idx + 1)
        self._base_push_mean_reward_smooth = None
        self._base_push_last_curriculum_step = self.common_step_counter

    def _reset_base_pushes(self, env_ids: torch.Tensor):
        self._base_push_steps_left[env_ids] = 0
        self._base_push_steps_until_next[env_ids] = self._sample_int_steps(
            self._base_push_interval_step_range, len(env_ids)
        )
        self._base_push_forces_b[env_ids] = 0.0
        self._base_push_torques_b[env_ids] = 0.0
        self._robot.permanent_wrench_composer.reset(env_ids)

    def _start_base_pushes(self, env_ids: torch.Tensor):
        self._base_push_steps_left[env_ids] = self._sample_int_steps(
            self._base_push_duration_step_range, len(env_ids)
        )
        self._base_push_steps_until_next[env_ids] = self._sample_int_steps(
            self._base_push_interval_step_range, len(env_ids)
        )

        forces = torch.zeros((len(env_ids), len(self._base_wrench_body_ids), 3), device=self.device)
        torques = torch.zeros_like(forces)
        self._fill_uniform_range(forces[..., 0], self.cfg.base_push_force_xy_range)
        self._fill_uniform_range(forces[..., 1], self.cfg.base_push_force_xy_range)
        self._fill_uniform_range(forces[..., 2], self.cfg.base_push_force_z_range)
        self._fill_uniform_range(torques[..., 0], self.cfg.base_push_torque_xy_range)
        self._fill_uniform_range(torques[..., 1], self.cfg.base_push_torque_xy_range)
        self._fill_uniform_range(torques[..., 2], self.cfg.base_push_torque_z_range)
        self._base_push_forces_b[env_ids] = forces
        self._base_push_torques_b[env_ids] = torques

    def _update_base_push_wrench(self):
        inactive = self._base_push_steps_left <= 0
        if torch.any(inactive):
            self._base_push_steps_until_next[inactive] -= 1
            start_env_ids = torch.nonzero(
                inactive & (self._base_push_steps_until_next <= 0), as_tuple=False
            ).squeeze(-1)
            if len(start_env_ids) > 0:
                self._start_base_pushes(start_env_ids)

        active = self._base_push_steps_left > 0
        self._base_push_forces_b[~active] = 0.0
        self._base_push_torques_b[~active] = 0.0
        if torch.any(active):
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                forces=self._base_push_forces_b,
                torques=self._base_push_torques_b,
                body_ids=self._base_wrench_body_ids,
            )
            self._base_push_steps_left[active] -= 1
        else:
            self._robot.permanent_wrench_composer.reset()

    def _pre_physics_step(self, actions: torch.Tensor):
        resample_env_ids = torch.nonzero(self._command_steps_left <= 0, as_tuple=False).squeeze(-1)
        if len(resample_env_ids) > 0:
            self._resample_commands(resample_env_ids)
            self._command_steps_left[resample_env_ids] = self._command_resample_interval
        self._command_steps_left -= 1
        self._update_base_push_wrench()

        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._q_offset_action_and_obs

    def _apply_action(self):
        self._delayed_processed_actions = self._action_delay_buffer.compute(self._processed_actions)
        self._robot.set_joint_position_target(self._delayed_processed_actions, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        obs_terms = []
        if not self.cfg.remove_root_lin_vel_b_from_obs:
            obs_terms.append(self._maybe_corrupt(self._robot.data.root_lin_vel_b, self.cfg.base_lin_vel_noise))
        obs_terms.extend(
            [
                self._maybe_corrupt(self._robot.data.root_ang_vel_b, self.cfg.base_ang_vel_noise),
                self._maybe_corrupt(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise),
                self._commands,
                self._maybe_corrupt(joint_pos, self.cfg.joint_pos_noise),
                self._maybe_corrupt(joint_vel, self.cfg.joint_vel_noise),
                self._actions,
            ]
        )
        obs = torch.cat(tuple(obs_terms), dim=-1)

        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._joint_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_body_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_body_ids]
        feet_air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1)
        feet_air_time *= torch.norm(self._commands[:, :2], dim=1) > 0.1

        undesired_contacts = self._compute_contact_count(self._thigh_body_ids, self.cfg.undesired_contact_threshold)
        force_transmited_through_joints = self._compute_force_transmited_through_joints()
        foot_contact = self._compute_foot_contact_penalty()

        rewards = {
            "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / self.cfg.tracking_std**2)
            * self.cfg.track_lin_vel_xy_reward_scale
            * self.step_dt,
            "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / self.cfg.tracking_std**2)
            * self.cfg.track_ang_vel_z_reward_scale
            * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.base_tilt_penalty_reward_scale * self.step_dt,
            "force_transmited_through_joints": force_transmited_through_joints
            * self.cfg.force_transmited_through_joints_reward_scale
            * self.step_dt,
            "foot_contact": foot_contact * self.cfg.foot_contact_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_reward_sums += reward
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contacts = torch.max(torch.norm(net_contact_forces[:, :, self._base_body_ids], dim=-1), dim=1)[0]
        terminated = torch.any(base_contacts > self.cfg.base_contact_threshold, dim=1)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        completed_env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        completed_episode_returns = self._episode_reward_sums[completed_env_ids]
        mean_episode_return = (
            torch.mean(completed_episode_returns).item() if len(completed_episode_returns) > 0 else 0.0
        )
        self._update_base_push_force_curriculum(completed_episode_returns)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._delayed_processed_actions[env_ids] = 0.0

        self._resample_commands(env_ids, allow_opposite=False)
        self._command_steps_left[env_ids] = self._command_resample_interval
        self._reset_base_pushes(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._reset_joint_pos[env_ids]
        if self.cfg.initial_position.lower() != "rigid":
            low, high = self.cfg.flexed_initial_joint_pos_noise_range
            if low != 0.0 or high != 0.0:
                joint_pos[:, self._joint_ids] += torch.empty(
                    (num_resets, len(self._joint_ids)), device=self.device
                ).uniform_(low, high)
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        root_pose = self._robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, 0] += self.cfg.reset_x_pos
        root_pose[:, 1] += self.cfg.reset_y_pos
        yaw = torch.full((num_resets,), self.cfg.reset_yaw, device=self.device)
        zeros = torch.zeros_like(yaw)
        root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

        root_velocity = torch.zeros_like(self._robot.data.default_root_state[env_ids, 7:])
        root_velocity[:, 0:3].uniform_(*self.cfg.reset_base_lin_vel_range)
        root_velocity[:, 3:6].uniform_(*self.cfg.reset_base_ang_vel_range)

        action_delays = torch.randint(
            low=self.cfg.actuation_delay_range[0],
            high=self.cfg.actuation_delay_range[1] + 1,
            size=(num_resets,),
            device=self.device,
            dtype=torch.int,
        )
        self._action_delay_steps[env_ids] = action_delays
        self._action_delay_buffer.set_time_lag(action_delays, env_ids)
        self._action_delay_buffer.reset(env_ids)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos[:, self._joint_ids], joint_ids=self._joint_ids, env_ids=env_ids)

        extras = {}
        for key in self._episode_sums:
            # Included abs to see all positive (reward / penalty) in wandb
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids]).abs() / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras['Episode_Reward/cmd_tracking'] = extras[f"Episode_Reward/track_lin_vel_xy_exp"] + extras[f"Episode_Reward/track_ang_vel_z_exp"]
        extras["Episode_Reward/total"] = mean_episode_return
        self._episode_reward_sums[env_ids] = 0.0

        force_low, force_high = self.cfg.base_push_force_xy_range
        extras["Curriculum/base_push_force_xy_abs"] = max(abs(force_low), abs(force_high))
        extras["Curriculum/base_push_force_idx"] = self._base_push_force_curriculum_idx
        if self._base_push_mean_reward_smooth is not None:
            extras["Curriculum/base_push_mean_reward_smooth"] = self._base_push_mean_reward_smooth

        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # include logs in wandb
        self.extras["log"] = extras

    def _resample_commands(self, env_ids: torch.Tensor, allow_opposite: bool = True):
        commands = torch.empty((len(env_ids), 3), device=self.device)
        commands[:, 0].uniform_(*self.cfg.command_lin_vel_x_range)
        commands[:, 1].uniform_(*self.cfg.command_lin_vel_y_range)
        commands[:, 2].uniform_(*self.cfg.command_ang_vel_z_range)

        opposite_prob = self.cfg.opposite_direction_cmd_prob
        if allow_opposite and opposite_prob > 0.0:
            previous_commands = self._commands[env_ids]
            flip_mask = torch.rand_like(commands) < opposite_prob
            nonzero_previous = torch.abs(previous_commands) > 1e-6
            commands = torch.where(flip_mask & nonzero_previous, -previous_commands, commands)

        standing_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_env_prob
        commands[standing_mask] = 0.0
        self._commands[env_ids] = commands

    def _compute_contact_count(self, body_ids: list[int], threshold: float) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)

    def _compute_force_transmited_through_joints(self) -> torch.Tensor:
        incoming_joint_wrench_b = self._robot.data.body_incoming_joint_wrench_b[:, self._joint_wrench_body_ids]
        incoming_joint_force_b = incoming_joint_wrench_b[..., :3]
        return torch.sum(torch.sum(torch.square(incoming_joint_force_b), dim=-1), dim=1)

    def _compute_foot_contact_penalty(self) -> torch.Tensor:
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_body_ids, :]
        contact_force_norm = torch.norm(contact_forces, dim=-1)
        excess_force = torch.clamp(contact_force_norm - self.cfg.foot_contact_safe_threshold, min=0.0)
        if self.cfg.square_foot_penalty: 
            excess_force = excess_force**2
        return torch.sum(excess_force, dim=1)

    def _maybe_corrupt(self, tensor: torch.Tensor, noise_range: tuple[float, float]) -> torch.Tensor:
        if not self.cfg.enable_observation_corruption or noise_range[0] == noise_range[1] == 0.0:
            return tensor
        return tensor + torch.empty_like(tensor).uniform_(noise_range[0], noise_range[1])
