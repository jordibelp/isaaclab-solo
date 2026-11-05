# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg

from .task_math import validate_goal_config


@configclass
class BorinotGoToEnvCfg(BorinotCmdVelEnvCfg):
    """Same flight plant, reset distribution and controllers; a different command and objective."""
    is_finite_horizon = True  # the deadline is part of the task, never a value bootstrap
    goal_conditioned = True
    goal_half_extents_m = ((0.25, 0.25, 0.10), (0.5, 0.5, 0.20), (1.0, 1.0, 0.35),
                           (2.0, 2.0, 0.60), (3.0, 3.0, 0.80), (6.0, 6.0, 1.0))
    goal_deadline_range_s = ((2.5, 3.5), (2.0, 3.0), (1.5, 2.5), (1.0, 2.0), (0.7, 1.8), (1.0, 2.5))
    goal_reference_speed_m_s = (0.4, 0.7, 1.2, 2.0, 3.0, 3.125)
    goal_reference_accel_m_s2 = (0.8, 1.5, 3.0, 5.0, 8.0, 9.0)
    goal_success_tolerance_m = 0.15  # 3D base-link distance, evaluated only at the deadline
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
    # Optional hypotheses, not part of the requested default objective.
    use_terminal_speed_penalty = False
    rwd_scale_terminal_speed = -1.0  # one-off per (m/s)^2, only at a clean deadline
    use_terminal_success_bonus = False
    rwd_scale_terminal_success = 10.0  # one-off, clean deadline inside goal tolerance
    rwd_scale_action_rate = 0.0
    rwd_scale_energy = 0.0
    rwd_scale_base_collision = 0.0
    rwd_scale_leave_area = 0.0
    rwd_scale_self_collision = 0.0
    rwd_scale_tilt = 0.0
    rwd_scale_lin_vel = 0.0  # no command-velocity reward in this task
    rwd_scale_ang_vel = 0.0
    goal_debug_vis = False  # marker only; no collider or goal object in the physics scene

    # The inherited disturbance curriculum is not used. Keep the same plant
    # without adding a confounding push curriculum to controller comparisons.
    forces_applied_to_base_curriculum = (0.0,)
    curriculum_start_level = 0  # unused parent force level; use goal_curriculum_start_level

    def configure_control(self):
        super().configure_control()
        validate_goal_config(self)
        # Parent command has four slots: replace it with body goal XYZ + t_norm,
        # then append remaining seconds after controller memory (+1 observation).
        self.observation_space += 1
        self.state_space = self.observation_space
        self.is_finite_horizon = True
        # A generic horizon must not cut short a sampled deadline. Derive the
        # metadata horizon from all sampling stages after Hydra overrides.
        longest = max(max(bounds[1], math.sqrt(sum(v*v for v in extent)) / speed,
                          2 * math.sqrt(math.sqrt(sum(v*v for v in extent)) / accel))
                      for extent, bounds, speed, accel in zip(self.goal_half_extents_m, self.goal_deadline_range_s,
                                                            self.goal_reference_speed_m_s, self.goal_reference_accel_m_s2))
        self.episode_length_s = math.ceil(longest * self.policy_hz) / self.policy_hz
