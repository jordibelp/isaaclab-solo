# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg

from .task_math import deadline_references, validate_goal_config


@configclass
class BorinotGoToEnvCfg(BorinotCmdVelEnvCfg):
    """Same flight plant, reset distribution and controllers; a different command and objective."""
    is_finite_horizon = True  # the deadline is part of the task, never a value bootstrap
    goal_conditioned = True
    # Flat base +X toward the sampled goal in world XY; skip all reset RPY noise.
    init_base_attitude_points_to_goal = False
    goal_half_extents_m = ((0.25, 0.25, 0.10), (0.5, 0.5, 0.20), (1.0, 1.0, 0.35),
                           (2.0, 2.0, 0.60), (3.0, 3.0, 0.80), (6.0, 6.0, 1.0))
    goal_deadline_range_s = ((2.5, 3.5), (2.0, 3.0), (1.5, 2.5), (1.0, 2.0), (0.7, 1.8), (1.0, 2.5))
    # Optional deadline floors: zero disables, one value broadcasts to all stages.
    # Omitted CLI references use only the configured phase time ranges.
    goal_reference_speed_m_s = (0.0,)
    goal_reference_accel_m_s2 = (0.0,)
    # Success at a clean deadline, or first settled arrival when reset_at_success is enabled.
    goal_success_tolerance_m = 0.05  # 3D base-link distance <= this
    goal_success_lin_vel_tolerance_m_s = 0.1  # |base linear velocity| < this
    goal_success_ang_vel_tolerance_rad_s = 0.2  # |base angular velocity|, all three axes, < this
    reset_at_success = False  # otherwise success is evaluated only at the deadline
    per_step_penalty_weight = -0.0  # signed reward per policy step, including the terminal step; no dt
    remove_time_from_actor_obs = False  # remove both clocks from policy inputs; critic keeps both
    goal_curriculum_start_level = 0
    curriculum_success_threshold = 0.8  # >=, not reward-weight-dependent
    curriculum_window_episodes = 2048
    curriculum_min_episodes = 512
    curriculum_min_steps = 1000  # 10 simulated seconds/environment at 100 Hz
    progress_reward_curriculum = True
    progress_reward_factors = (1.0, 0.75, 0.5, 0.25, 0.0)  # after the last geometry is mastered

    rwd_scale_terminal_distance = -10.0  # one-off per metre, no dt
    rwd_scale_progress = 1.0  # scale * (previous_distance - distance), no extra dt
    failure_distance_floor_m = 10.0  # early safety failure pays at least -100, rather than escaping the deadline cost
    use_terminal_linear_vel_penalty = True
    rwd_scale_terminal_linear_vel = -1.0  # one-off per (m/s)^2 at a clean terminal
    use_terminal_ang_vel_penalty = True
    rwd_scale_terminal_ang_vel = -1.0  # one-off per (rad/s)^2 at a clean terminal
    use_terminal_success_bonus = False
    rwd_scale_terminal_success = 10.0  # one-off, on successful termination (inside and at rest)
    rwd_scale_action_rate = 0.0
    rwd_scale_energy = 0.0
    rwd_scale_base_collision = -5.0
    rwd_scale_leave_area = -2.0
    rwd_scale_self_collision = 0.0
    rwd_scale_tilt = 0.0
    rwd_scale_lin_vel = 0.0  # no command-velocity reward in this task
    rwd_scale_ang_vel = 0.0
    goal_debug_vis = False  # marker only; no collider or goal object in the physics scene

    # Vertical base pushes: the inherited cmd-vel force list and push timing.
    # Goal level start_forces_randn_curriculum_idx + k uses force entry k; earlier
    # levels have no pushes, later ones keep the last entry. (0.0,) disables pushes.
    start_forces_randn_curriculum_idx = 5
    curriculum_start_level = 0  # unused parent force level; use goal_curriculum_start_level

    base_push_interval_range_s=(0.7,1.3)
    base_push_duration_range_s=(0.25,0.7)

    def configure_control(self):
        super().configure_control()
        validate_goal_config(self)
        # Parent command has four slots: replace it with body goal XYZ + t_norm,
        # then append remaining seconds after controller memory (+1 observation).
        self.observation_space += 1
        self.state_space = self.observation_space
        if self.remove_time_from_actor_obs:
            self.observation_space -= 2
        self.is_finite_horizon = True
        # A generic horizon must not cut short a sampled deadline. Derive the
        # metadata horizon from all sampling stages after Hydra overrides.
        longest = 0.0
        for difficulty, (extent, bounds) in enumerate(zip(self.goal_half_extents_m, self.goal_deadline_range_s)):
            distance = math.sqrt(sum(v*v for v in extent))
            speed, accel = deadline_references(self, difficulty)
            longest = max(longest, bounds[1], distance / speed if speed > 0 else 0.0,
                          2 * math.sqrt(distance / accel) if accel > 0 else 0.0)
        self.episode_length_s = math.ceil(longest * self.policy_hz) / self.policy_hz


@configclass
class BorinotGoToMinTimeEnvCfg(BorinotGoToEnvCfg):
    """Same go-to task, with an explicit cost of time and first settled-arrival termination."""

    reset_at_success = True
    per_step_penalty_weight = -0.015
