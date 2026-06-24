# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from pathlib import Path
# --------------------------------------------------------------------------------------
# Solo12 asset config
# --------------------------------------------------------------------------------------


SOLO12_FLAT_CFG = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parents[7] / "isaaclab_assets/data/Robots/Solo12/SoloFlat.usd"),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=2)
        ,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),), 
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=2.7,
            stiffness=5.0,
            damping=0.5,
            velocity_limit_sim=30.0,
            friction=0.0
        ),
        # "base_legs": DCMotorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     effort_limit=23.5,
        #     saturation_effort=23.5,
        #     velocity_limit=30.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),

    },
)



@configclass
class Solo12RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = SOLO12_FLAT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # 🔶 actions

        JOINT_NAMES = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = JOINT_NAMES
        self.actions.joint_pos.preserve_order = True



        # scene ground friction tweaking: 
        self.scene.terrain.physics_material.friction_combine_mode = "max"
        self.scene.terrain.physics_material.restitution_combine_mode = "max"
        self.scene.terrain.physics_material.static_friction = 1.0
        self.scene.terrain.physics_material.dynamic_friction = 1.0

        # 🔶 Event; DR 
        # which material is this? 
        self.events.physics_material.params["static_friction_range"] = (2.0, 2.0)
        self.events.physics_material.params["dynamic_friction_range"] = (1.5, 1.5)
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.1, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_base_mass = None # 🔶 removing mass randomization for now
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        # applies force to the base
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # 🔶 remove DR in initial pose for now
        self.events.reset_base.params = {
            "pose_range": {"x": (+0.5, 0.5), "y": (+0.5, 0.5), "yaw": (+3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # 🔶 rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf"
        self.rewards.feet_air_time.weight = 0.00
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -1e-5
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        # 🔶 This term in the reward is significant: ~ 0.1 perhaps should decrease weight more. 
        self.rewards.dof_acc_l2.weight = -2.5e-7


        

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base"]

        # 🔶 Perhaps we can add thigh to undesired contacts rather than terminations (?)

        # commands
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-1., 1.)   # or (0, 0) if you don't want lateral motion
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)





@configclass
class Solo12FlatEnvCfg(Solo12RoughEnvCfg):
    """Flat-terrain manager-based velocity task for Solo12."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------------------------------------------
        # Robot
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Terrain: plane only
        # ------------------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.height_scanner = None
        self.scene.terrain.terrain_generator = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


        # ------------------------------------------------------------------
        # Rewards / terminations
        # ------------------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-1., 1.)   # or (0, 0) if you don't want lateral motion
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # EXP: setting all additional terms to zero
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = -1e-5
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0




@configclass
class Solo12RoughEnvCfg_PLAY(Solo12RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class Solo12FlatEnvCfg_PLAY(Solo12FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # fixed command = (vx, vy, wz) = (1, 0, 0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # good to pin this too if the command cfg has heading enabled / available
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None