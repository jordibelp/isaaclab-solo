# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import inspect
import math

import torch

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.direct.borinot.action_limits import clipped_limit_rewards
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env import BorinotCmdVelEnv

from .borinot_goto_env_cfg import BorinotGoToEnvCfg
from .task_math import SuccessCurriculum, goal_facing_quat, goal_success_parts, sample_goals, stage_force_limit, stage_settings


class BorinotGoToEnv(BorinotCmdVelEnv):
    """Reach a fixed 3D base position at a sampled deadline, without a heading target."""
    cfg: BorinotGoToEnvCfg

    def __init__(self, cfg: BorinotGoToEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.goal_curriculum = SuccessCurriculum(cfg, self.device)
        # The parent push scheduler reads _force_limits[_episode_stage]. Here both
        # index goal-curriculum levels; _reset_idx tags each episode with its level.
        self._force_limits = torch.tensor([stage_force_limit(cfg, level) for level in range(
            self.goal_curriculum.max_level + 1)], device=self.device)
        self._goal_position_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_deadline_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_episode_level = torch.full_like(self._episode_steps, self.goal_curriculum.level)
        self._goal_progress_factor = torch.ones(self.num_envs, device=self.device)
        self._initial_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_goal_distance = torch.zeros_like(self._initial_goal_distance)
        self._goal_failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_deadline_reached = torch.zeros_like(self._goal_failed)
        self._goal_finished = torch.zeros_like(self._goal_failed)
        self._goal_success = torch.zeros_like(self._goal_failed)
        self._first_arrival_s = torch.full_like(self._initial_goal_distance, -1.0)
        self._minimum_goal_distance = torch.zeros_like(self._initial_goal_distance)
        self._episode_terms = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "terminal_distance", "progress", "terminal_linear_vel", "terminal_ang_vel", "terminal_success",
            "action_rate", "energy", "base_collision", "leave_area", "self_collision", "tilt",
            "thrust_limits", "torque_limits", "action_limits", "time")}
        self._goal_markers = None

    def _resample_commands(self, env_ids):
        # Fixed goals change only at episode boundaries. Parent observations must
        # not overwrite this task's command or consume command-sampling RNG.
        pass

    def _update_pushes(self):
        # Same pulses as Borinot-cmd-vel. Without any nonzero force, skip the
        # sampling so push-free runs keep their previous random stream.
        if any(self.cfg.forces_applied_to_base_curriculum):
            super()._update_pushes()

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._previous_goal_distance.copy_((self._goal_position_w - self.robot.data.root_pos_w).norm(dim=-1))
        self._goal_finished.zero_()

    def _get_dones(self):
        failed, _ = super()._get_dones()  # completed-substep safety events and power
        self._goal_failed.copy_(failed)
        self._goal_deadline_reached.copy_(self._episode_steps >= self._goal_deadline_steps)
        distance = (self._goal_position_w - self.robot.data.root_pos_w).norm(dim=-1)
        reached, slow, still = goal_success_parts(
            self.cfg, distance, self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w)
        self._goal_success.copy_(reached & slow & still & ~failed
                                 & (self._goal_deadline_reached | self.cfg.reset_at_success))
        self._goal_finished.copy_(failed | self._goal_deadline_reached | self._goal_success)
        first = reached & (self._first_arrival_s < 0)
        self._first_arrival_s[first] = self._episode_steps[first] * self.step_dt
        self._minimum_goal_distance.copy_(torch.minimum(self._minimum_goal_distance, distance))
        # Deadline and early settled success are genuine terminals, NOT artificial timeouts.
        return self._goal_finished.clone(), torch.zeros_like(self._goal_finished)

    def _get_rewards(self):
        distance = (self._goal_position_w - self.robot.data.root_pos_w).norm(dim=-1)
        terminal_distance = torch.where(self._goal_failed, distance.clamp_min(self.cfg.failure_distance_floor_m), distance)
        self._actuator_commands[:, 6:] = self.robot.data.applied_torque[:, self.joint_ids] / (2 * self.cfg.effort_lim_sim)
        delta = self._actuator_commands - self._previous_actuator_commands
        action_rate = delta[:, :6].square().mean(-1) + delta[:, 6:].square().mean(-1)
        power = self._power_sum_w / max(1, self._power_samples)
        clean_terminal = (self._goal_deadline_reached | (self._goal_success & self.cfg.reset_at_success)) & ~self._goal_failed
        terms = {
            "terminal_distance": self.cfg.rwd_scale_terminal_distance * terminal_distance * self._goal_finished,
            "progress": self.cfg.rwd_scale_progress * self._goal_progress_factor * (self._previous_goal_distance - distance),
            "terminal_linear_vel": (self.cfg.rwd_scale_terminal_linear_vel * clean_terminal
                                    * self.robot.data.root_lin_vel_w.square().sum(-1)
                                    if self.cfg.use_terminal_linear_vel_penalty else torch.zeros_like(distance)),
            "terminal_ang_vel": (self.cfg.rwd_scale_terminal_ang_vel * clean_terminal
                                 * self.robot.data.root_ang_vel_w.square().sum(-1)
                                 if self.cfg.use_terminal_ang_vel_penalty else torch.zeros_like(distance)),
            "terminal_success": (self.cfg.rwd_scale_terminal_success * self._goal_success
                                 if self.cfg.use_terminal_success_bonus else torch.zeros_like(distance)),
            "action_rate": self.cfg.rwd_scale_action_rate * action_rate * self.step_dt,
            "energy": self.cfg.rwd_scale_energy * power.sum(-1),
            "base_collision": self.cfg.rwd_scale_base_collision * self._events["base_collision"],
            "leave_area": self.cfg.rwd_scale_leave_area * self._events["leave_area"],
            "self_collision": self.cfg.rwd_scale_self_collision * self._self_contact_bodies.sum(-1) * self.step_dt,
            "tilt": self.cfg.rwd_scale_tilt * self._events["tilt"],
            "time": torch.full_like(distance, self.cfg.per_step_penalty_weight),
            **clipped_limit_rewards(self._action_limit_costs, self.cfg),
        }
        for name, value in terms.items():
            self._episode_terms[name] += value
        self._episode_energy_j += self._power_sum_w.sum(-1) * self.physics_dt
        self.extras["reward_terms"] = terms
        self.extras["power_mean_w"] = {name: power[:, i] for i, name in enumerate(
            ("rotor_mechanical", "rotor_dissipative", "arm_mechanical", "arm_copper_loss"))}
        # Preserve terminal values across automatic reset. Numeric arrays remain
        # available at the native policy rate; they are not image-only telemetry.
        self.extras["goal"] = {"distance_m": distance, "deadline_s": self._goal_deadline_steps * self.step_dt,
                              "remaining_s": (self._goal_deadline_steps - self._episode_steps).clamp_min(0) * self.step_dt,
                              "deadline_reached": self._goal_deadline_reached.clone(),
                              "failed": self._goal_failed.clone(), "success": self._goal_success.clone(),
                              "stage": self._goal_episode_level.clone()}
        return sum(terms.values())

    def _get_observations(self):
        remaining_s = (self._goal_deadline_steps - self._episode_steps).clamp_min(0) * self.step_dt
        self._commands[:, :3] = quat_apply_inverse(
            self.robot.data.root_quat_w, self._goal_position_w - self.robot.data.root_pos_w)
        self._commands[:, 3] = remaining_s / (self._goal_deadline_steps * self.step_dt)
        # Reuse all measured state and controller-memory fields, in the same order.
        observation = torch.cat((super()._get_observations()["policy"], remaining_s[:, None]), dim=-1)
        policy = observation
        if self.cfg.remove_time_from_actor_obs:
            # Normalized time follows v, omega, [acc], q, dq, action, [gravity], R,
            # [height], goal XYZ. Controller memory follows it; seconds is last.
            time_index = (22 + self.cfg.action_space + int(self.cfg.include_base_height)
                          + 3 * int(self.cfg.projected_gravity_in_obs)
                          + 3 * self.cfg.obs_include_base_lin_acc_steps * self.cfg.obs_include_base_lin_acc)
            policy = torch.cat((observation[:, :time_index], observation[:, time_index + 1:-1]), dim=-1)
        if self.cfg.goal_debug_vis:
            self._draw_goal_markers()
        return {"policy": policy, "critic": observation}

    def _update_curriculum(self, env_ids):
        completed = env_ids[self._goal_finished[env_ids] & (self._episode_steps[env_ids] > 0)]
        if not len(completed):
            return  # startup/manual abort is not a completed task
        old_level = self.goal_curriculum.level
        current = self._goal_episode_level[completed] == old_level
        stale = (~current).nonzero().flatten()
        self.curriculum_stage_pure_episodes = (len(completed) - 1 - int(stale[-1]) if len(stale)
                                             else self.curriculum_stage_pure_episodes + len(completed))
        decision = self.goal_curriculum.observe(self._goal_success[completed], self._goal_episode_level[completed],
                                                self.common_step_counter)
        if self.goal_curriculum.level != old_level:
            self.curriculum_stage_pure_episodes = 0
        duration = self._episode_steps[completed] * self.step_dt
        goal_info = self.extras["goal"]
        deadline = goal_info["deadline_reached"][completed] & ~goal_info["failed"][completed]
        arrivals = self._first_arrival_s[completed]
        logs = {
            "Episode/return": sum(v[completed] for v in self._episode_terms.values()).mean(),
            "Episode/duration_s": duration.mean(), "Episode/energy_j": self._episode_energy_j[completed].mean(),
            "Episode/actuator_delay_s": self._actuator_delay.seconds[completed].mean(),
            "Episode/completed_count": len(completed), "Episode/success_count": self._goal_success[completed].sum(),
            "Goal/terminal_distance_m": goal_info["distance_m"][completed].mean(),
            "Goal/initial_distance_m": self._initial_goal_distance[completed].mean(),
            "Goal/minimum_distance_m": self._minimum_goal_distance[completed].mean(),
            "Goal/deadline_s": goal_info["deadline_s"][completed].mean(),
            "Goal/early_failure_count": (goal_info["failed"][completed] & ~goal_info["deadline_reached"][completed]).sum(),
            "Goal/deadline_count": deadline.sum(),
            "Goal/ever_reached_count": (arrivals >= 0).sum(),
            "Curriculum/collection_stage": self._goal_episode_level[completed].float().mean(),
            **{f"Curriculum/{key}": value for key, value in decision.items()},
            "Curriculum/selected_difficulty": stage_settings(self.cfg, self.goal_curriculum.level)[0],
            "Curriculum/selected_progress_factor": stage_settings(self.cfg, self.goal_curriculum.level)[1],
            "Curriculum/selected_force_limit_n": self._force_limits[self.goal_curriculum.level],
            "Goal/collection_progress_factor": self._goal_progress_factor[completed].mean(),
            "Goal/collection_force_limit_n": self._force_limits[self._goal_episode_level[completed]].mean(),
        }
        # Which success conditions hold at clean deadlines; divide by Goal/deadline_count.
        ids, data = completed[deadline], self.robot.data
        parts = goal_success_parts(self.cfg, goal_info["distance_m"][ids],
                                   data.root_lin_vel_w[ids], data.root_ang_vel_w[ids])
        logs.update({f"Goal/deadline_{name}_count": mask.sum()
                     for name, mask in zip(("inside", "lin_vel_ok", "ang_vel_ok"), parts)})
        if deadline.any():
            logs["Goal/deadline_distance_m"] = goal_info["distance_m"][ids].mean()
            logs["Goal/deadline_speed_m_s"] = data.root_lin_vel_w[ids].norm(dim=-1).mean()
            logs["Goal/deadline_ang_speed_rad_s"] = data.root_ang_vel_w[ids].norm(dim=-1).mean()
        if (arrivals >= 0).any():
            logs["Goal/first_arrival_s_when_reached"] = arrivals[arrivals >= 0].mean()
        successes = self._goal_success[completed]
        if successes.any():
            logs["Goal/success_duration_s"] = duration[successes].mean()
        logs["Goal/early_success_count"] = (successes & ~goal_info["deadline_reached"][completed]).sum()
        logs.update({f"Episode_Reward/{name}": value[completed].mean() for name, value in self._episode_terms.items()})
        self.extras["log"] = logs

    def _sample_reset_goals(self, start_position_w, env_ids):
        difficulty, _ = stage_settings(self.cfg, self.goal_curriculum.level)
        return sample_goals(start_position_w, self.cfg, difficulty, self.step_dt)

    def _sample_reset_attitude(self, root, env_ids):
        if not self.cfg.init_base_attitude_points_to_goal:
            return super()._sample_reset_attitude(root, env_ids)
        # The parent has sampled the actual reset position, but has not written
        # it or reset controller histories yet. Choose the episode goal now.
        goals, steps = self._sample_reset_goals(root[:, :3], env_ids)
        self._goal_position_w[env_ids], self._goal_deadline_steps[env_ids] = goals, steps
        root[:, 3:7] = goal_facing_quat(root[:, :3], goals)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)  # logs old outcome, resets the SAME shared plant/control state
        difficulty, factor = stage_settings(self.cfg, self.goal_curriculum.level)
        if self.cfg.init_base_attitude_points_to_goal:
            goals, steps = self._goal_position_w[env_ids], self._goal_deadline_steps[env_ids]
        else:
            goals, steps = sample_goals(self.robot.data.root_pos_w[env_ids], self.cfg, difficulty, self.step_dt)
        self._goal_position_w[env_ids] = goals
        self._goal_deadline_steps[env_ids] = steps
        self._goal_episode_level[env_ids] = self._episode_stage[env_ids] = self.goal_curriculum.level
        if any(self.cfg.forces_applied_to_base_curriculum):
            # Episodes (about 1-3.5 s) are shorter than the 2-5 s push spacing, and the
            # parent waits one full interval. Start the push clock at a random phase:
            # the first push may begin at any tick of that interval.
            interval = self._base_push_steps_until_next[env_ids]
            self._base_push_steps_until_next[env_ids] = (
                torch.rand(len(env_ids), device=self.device) * interval).long()
        self._goal_progress_factor[env_ids] = factor
        distance = (goals - self.robot.data.root_pos_w[env_ids]).norm(dim=-1)
        self._initial_goal_distance[env_ids] = distance
        self._previous_goal_distance[env_ids] = distance
        self._minimum_goal_distance[env_ids] = distance
        self._first_arrival_s[env_ids] = torch.where(distance <= self.cfg.goal_success_tolerance_m, 0.0, -1.0)
        for buffer in (self._goal_failed, self._goal_deadline_reached, self._goal_finished, self._goal_success):
            buffer[env_ids] = False

    def get_curriculum_global_idx(self):
        return self.goal_curriculum.level

    def get_curriculum_max_global_idx(self):
        return self.goal_curriculum.max_level

    def get_curriculum_sampling_metrics(self):
        """Current stage's configured bounds, not averages over completed episodes."""
        difficulty, _ = stage_settings(self.cfg, self.goal_curriculum.level)
        low, high = self.cfg.goal_deadline_range_s[difficulty]
        return {
            "Curriculum/selected_deadline_min_s": low,
            "Curriculum/selected_deadline_max_s": high,
            "Curriculum/selected_goal_distance_max_m": math.sqrt(
                sum(v * v for v in self.cfg.goal_half_extents_m[difficulty])),
        }

    def get_code_snapshot_files(self):
        return super().get_code_snapshot_files() + [inspect.getsourcefile(obj) for obj in (
            BorinotCmdVelEnv, BorinotGoToEnvCfg, SuccessCurriculum)]

    def _draw_goal_markers(self):
        if self._goal_markers is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            self._goal_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/BorinotGoTo/goals",
                markers={"goal": sim_utils.SphereCfg(radius=self.cfg.goal_success_tolerance_m,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.25), opacity=0.35))}))
        self._goal_markers.visualize(translations=self._goal_position_w)
