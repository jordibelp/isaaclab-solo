# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab.utils import configclass

from ..solo12.solo12_env_cfg import Solo12EnvCfg


@configclass
class Solo12BackflipEnvCfg(Solo12EnvCfg):
    """Solo12 task that rewards fast backward rotation about the base lateral axis (repeated backflips).

    The USD, actuators, joint limits, observation layout, sensors, pushes, observation noise and startup
    randomizers come from ``Solo12EnvCfg``. The velocity-command slot of the 48-D observation stays zero.
    """

    # Task reward: scale * (-omega_y) * dt, where omega_y is the base angular velocity about body +y (left).
    # -omega_y > 0 is a nose-up, backward rotation. The term is signed, so rocking back and forth earns
    # nothing and the undiscounted episode sum is scale times the net backward rotation in radians.
    backflip_ang_vel_reward_scale = 5.0
    # Symmetric clip of -omega_y in rad/s before scaling. 0 disables the clip.
    backflip_ang_vel_clip = 0.0

    base_collision_terminal_penalty = -10.0
    # Thigh contacts.
    undesired_contact_reward_scale = -2.25
    # Preferences, off for the first experiments. For example:
    # env.action_rate_reward_scale=-0.05 env.joint_torque_reward_scale=-0.5e-3 env.soft_qlim_penalty_reward_scale=-0.5
    action_rate_reward_scale = 0.0
    joint_torque_reward_scale = 0.0
    foot_contact_reward_scale = 0.0
    soft_qlim_penalty_reward_scale = 0.0
    # Base-task terms that would oppose flipping.
    track_lin_vel_xy_reward_scale = 0.0
    track_ang_vel_z_reward_scale = 0.0
    base_tilt_penalty_reward_scale = 0.0

    # No velocity commands. The inherited velocity/force curricula are disabled.
    command_lin_vel_x_range = (0.0, 0.0)
    command_lin_vel_y_range = (0.0, 0.0)
    command_ang_vel_z_range = (0.0, 0.0)
    opposite_direction_cmd_prob = 0.0
    max_velx_range_curriculum = []
    forces_applied_to_base_curriculum = []

    # A backflip mirrored front-to-back is a frontflip, so symmetry augmentation keeps only left-right.
    front_back_asymetry = True

    episode_length_s = 10.0
    enabled_self_collisions = True
    base_filtered_pairs = ("hip",)
    flat_terrain_grid_enabled = True
    z_forces_applied_both_faces = True
    # A push starts 2-5 s after reset or after the previous push, so pushes happen within an episode.
    base_push_interval_range_s = (2.0, 5.0)
    # Used when backflip_curriculum=False. The curriculum overrides them phase by phase.
    actuation_delay_range = (0, 3)
    base_push_force_xy_range = (-8.0, 8.0)
    base_push_force_z_range = (-8.0, 8.0)

    # Phase k uses entry k-1 of each phase tuple below. The task moves to the next phase when finished
    # episodes that started on the ground average at least the threshold number of completed backflips,
    # measured over a window of num_envs finished episodes.
    backflip_curriculum = True
    backflip_curriculum_start_phase = 1
    backflip_curriculum_advance_thresholds = (3.0, 3.0, 3.0, 3.0)
    include_events_randomization_curriculum = (False, True, True, True, True)
    actuation_delay_range_curriculum = ((0, 0), (0, 3), (0, 3), (0, 3), (0, 3))
    forces_applied_to_base_curriculum_by_phase = (0.0, 0.0, 3.0, 5.0, 8.0)
    base_push_force_z_range_curriculum = ((0.0, 0.0), (0.0, 0.0), (-3.0, 3.0), (-5.0, 5.0), (-8.0, 8.0))

    # Reference-state-style initialization: this fraction of resets starts mid-air and already rotating
    # backwards, so the policy practises landings before it can do a whole flip. 0 disables it.
    backflip_airborne_reset_prob = 0.0
    backflip_airborne_reset_height_range = (0.45, 0.8)  # base height above the terrain, m
    backflip_airborne_reset_rotation_range = (0.0, 2.0 * math.pi)  # backward rotation already done, rad
    backflip_airborne_reset_ang_vel_range = (4.0, 12.0)  # backward rotation rate -omega_y, rad/s

    def prepare_curriculum_event_randomization(self):
        """Keep startup randomizers dormant until a curriculum phase enables them."""
        if not self.backflip_curriculum or self.events is None or self.include_events_randomization_curriculum[0]:
            return
        for term in vars(self.events).values():
            if getattr(term, "mode", None) == "startup":
                term.mode = "curriculum_startup"
