# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg


@configclass
class BorinotCmdVelEnvCfg(BorinotGoalkeeperEnvCfg):
    """Share the Borinot plant/control options, not the goalkeeper scene or objective."""
    episode_length_s = 10.0
    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=64.0, replicate_physics=True)
    robot_home = (0.0, 0.0, 1.5)
    robot_home_quat = (1.0, 0.0, 0.0, 0.0)
    reset_xy_noise_m = 0.015
    reset_z_noise_m = 0.0
    reset_rpy_noise_degs = (5.0, 5.0, 5.0)  # independent uniform +/- roll/pitch/yaw about robot_home_quat
    arm_holds_racket = False  # opt-in goalkeeper racket; independent of the calf-tip payload
    # Legacy shaft/point model only; inference leg properties already include a load.
    # Ignored when inertial_properties.use_inference_leg_inertias=True (the default).
    ee_includes_puntual_mass = 0.0  # kg, additional ideal point mass at the distal calf tip in legacy mode

    # [heading vx, heading vy, world vz, world-Z angular velocity], SI units.
    command_lin_vel_x_range = (-1.0, 1.0)
    command_lin_vel_y_range = (-1.0, 1.0)
    command_lin_vel_z_range = (-0.5, 0.5)
    command_ang_vel_z_range = (-1.0, 1.0)
    command_resampling_time_s = 5.0
    standing_env_prob = 0.02  # all four commands zero
    tracking_lin_std = 0.223  # m/s; Solo's tracking_std=sqrt(0.25)
    tracking_ang_std = 0.223  # rad/s
    rwd_scale_lin_vel = 1.5
    rwd_scale_ang_vel = 0.5
    rwd_scale_action_rate = 0.0  # same physical command-rate signal as goalkeeper, times dt
    rwd_scale_energy = 0.0  # same mean-power model as goalkeeper, no extra dt
    rwd_scale_base_collision = -2.0  # external base contact; no dt multiplier
    rwd_scale_leave_area = -2.0  # any XYZ flight-bound violation; no dt multiplier
    rwd_scale_self_collision = -2.25  # per contacting robot body, times policy dt (Solo undesired-contact style)
    rwd_scale_tilt = -1.0  # tilt-threshold violation; no dt multiplier
    enabled_self_collisions = True  # enable physical self contacts for the self-collision cost
    self_collision_triggers_reset = False  # ongoing self-contact cost; keep flying by default
    base_collision_force_threshold_n = 1.0  # Solo base contact threshold
    self_collision_force_threshold_n = 0.6  # Solo undesired-contact threshold
    obs_include_base_lin_acc = False  # body-frame kinematic CoM acceleration; NOT specific force

    base_push_interval_range_s = (2.0, 5.0)  # start-to-start; first push delayed after reset
    base_push_duration_range_s = (0.5, 2.0)
    base_push_application_radius_m = 0.12
    base_push_application_half_height_m = 0.03
    forces_applied_to_base_curriculum = (0.0, 1.0, 3.0)  # max |world-Z force|, N
    curriculum = True
    skip_curriculum = False
    curriculum_start_level = 0  # also selects a fixed level when curriculum=False
    curriculum_tracking_threshold = 0.7
    curriculum_window_episodes = 256
    curriculum_min_episodes = 128
    curriculum_min_steps = 1000  # policy ticks per environment, not aggregate transitions

    # Local-to-clone safety bounds, not a position/altitude tracking objective.
    flight_height_range_m = (0.15, 20.0)
    flight_xy_limit_m = 25.0
    reset_robot_on_tilt_threshold = True
    robot_tilt_threshold_degs = 80.0

    def configure_control(self):
        super().configure_control()
        if not isinstance(self.arm_holds_racket, bool):
            raise ValueError("arm_holds_racket must be a boolean.")
        if not math.isfinite(self.ee_includes_puntual_mass) or self.ee_includes_puntual_mass < 0:
            raise ValueError("ee_includes_puntual_mass must be finite and non-negative (kg).")
        if not math.isfinite(self.shaft_mass_kg) or self.shaft_mass_kg <= 0:
            raise ValueError("shaft_mass_kg must be finite and positive.")
        if (len(self.reset_rpy_noise_degs) != 3
                or any(not math.isfinite(v) or v < 0 for v in self.reset_rpy_noise_degs)):
            raise ValueError("reset_rpy_noise_degs must contain three finite non-negative degree half-ranges.")
        for name in ("base_collision_force_threshold_n", "self_collision_force_threshold_n"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("rwd_scale_base_collision", "rwd_scale_leave_area", "rwd_scale_self_collision", "rwd_scale_tilt"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        # State available to both networks: v_b, omega_b, q, dq, last action,
        # unit gravity_b, row-major R_world_body, command, optional acceleration.
        controller_state = 6 if self.control_mode == "wrench" else (
            8 if self.control_mode == "indi" else (
                9 if self.control_mode == "PI_rotors_PD_joint_pos_control" else 0))
        self.observation_space = 26 + self.action_space + controller_state + 3 * self.obs_include_base_lin_acc
        self.state_space = self.observation_space
        self.robot.init_state.pos = tuple(self.robot_home)
        self.robot.init_state.rot = tuple(self.robot_home_quat)
        for name in ("command_lin_vel_x_range", "command_lin_vel_y_range", "command_lin_vel_z_range",
                     "command_ang_vel_z_range", "flight_height_range_m", "base_push_interval_range_s",
                     "base_push_duration_range_s"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must be two ordered finite values.")
        for name in ("episode_length_s", "command_resampling_time_s", "tracking_lin_std", "tracking_ang_std",
                     "flight_xy_limit_m", "base_push_application_radius_m", "base_push_application_half_height_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.motor_time_constant_s < 0 or not math.isfinite(self.motor_time_constant_s):
            raise ValueError("motor_time_constant_s must be finite and non-negative.")
        if not 0 <= self.standing_env_prob <= 1 or not 0 <= self.curriculum_tracking_threshold <= 1:
            raise ValueError("Standing probability and curriculum tracking threshold must lie in [0,1].")
        if not 0 < self.robot_tilt_threshold_degs < 90:
            raise ValueError("robot_tilt_threshold_degs must lie between 0 and 90.")
        if self.base_push_interval_range_s[0] <= 0 or self.base_push_duration_range_s[0] <= 0:
            raise ValueError("Push intervals and durations must be positive.")
        if self.base_push_duration_range_s[1] > self.base_push_interval_range_s[0]:
            raise ValueError("Maximum push duration must not exceed the minimum start-to-start interval.")
        forces = self.forces_applied_to_base_curriculum
        if not forces or any(not math.isfinite(v) or v < 0 for v in forces):
            raise ValueError("forces_applied_to_base_curriculum must contain non-negative finite force limits.")
        if any(a > b for a, b in zip(forces, forces[1:])):
            raise ValueError("Force curriculum limits must be non-decreasing.")
        if not 0 <= self.curriculum_start_level < len(forces):
            raise ValueError("curriculum_start_level must index forces_applied_to_base_curriculum.")
        if not 1 <= self.curriculum_min_episodes <= self.curriculum_window_episodes or self.curriculum_min_steps < 1:
            raise ValueError("Require 1 <= min episodes <= window episodes and positive curriculum dwell.")
        if self.scene.env_spacing <= 2 * (self.flight_xy_limit_m + 1):
            raise ValueError("env_spacing must separate the flight bounds, with one metre of robot clearance.")
