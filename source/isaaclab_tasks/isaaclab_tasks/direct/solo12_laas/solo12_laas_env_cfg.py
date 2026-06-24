# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


SOLO12_LAAS_USD_PATH = Path(__file__).parents[4] / "isaaclab_assets/data/Robots/Solo12Laas/solo12_description.usd"

ROOT_LIN_VEL_OBS_DIM = 3
BASE_OBSERVATION_SPACE = 48
BASE_BODY_NAME = "base_link"
FEET_BODY_NAMES = ".*_FOOT"
UNDESIRED_CONTACT_BODY_NAMES = ".*_UPPER_LEG"
JOINT_WRENCH_BODY_NAMES = [".*_UPPER_LEG", ".*_LOWER_LEG"]

JOINT_NAMES = [
    "FL_HAA",
    "FL_HFE",
    "FL_KFE",
    "FR_HAA",
    "FR_HFE",
    "FR_KFE",
    "HL_HAA",
    "HL_HFE",
    "HL_KFE",
    "HR_HAA",
    "HR_HFE",
    "HR_KFE",
]

ACTUATOR_JOINT_NAMES = [
    "FL_HAA",
    "FL_HFE",
    "FL_KFE",
    "FR_HAA",
    "FR_HFE",
    "FR_KFE",
    "HR_HAA",
    "HR_HFE",
    "HR_KFE",
    "HL_HAA",
    "HL_HFE",
    "HL_KFE",
]

FLEXED_INITIAL_JOINT_POS = {
    "FL_HAA": 0.0,
    "FL_HFE": 0.8,
    "FL_KFE": -1.5,
    "FR_HAA": 0.0,
    "FR_HFE": 0.8,
    "FR_KFE": -1.5,
    "HL_HAA": 0.0,
    "HL_HFE": 0.8,
    "HL_KFE": -1.5,
    "HR_HAA": 0.0,
    "HR_HFE": 0.8,
    "HR_KFE": -1.5,
}

CRAB_INITIAL_JOINT_POS = {
    "FL_HAA": 0.0,
    "FL_HFE": 0.8,
    "FL_KFE": -1.5,
    "FR_HAA": 0.0,
    "FR_HFE": 0.8,
    "FR_KFE": -1.5,
    "HL_HAA": 0.0,
    "HL_HFE": -0.8,
    "HL_KFE": 1.5,
    "HR_HAA": 0.0,
    "HR_HFE": -0.8,
    "HR_KFE": 1.5,
}

SAFE_INITIAL_JOINT_POS = {
    "FL_HAA": 0.05,
    "FL_HFE": 0.4,
    "FL_KFE": -0.8,
    "FR_HAA": -0.05,
    "FR_HFE": 0.4,
    "FR_KFE": -0.8,
    "HL_HAA": 0.05,
    "HL_HFE": 0.4,
    "HL_KFE": -0.8,
    "HR_HAA": -0.05,
    "HR_HFE": 0.4,
    "HR_KFE": -0.8,
}

INITIAL_JOINT_POS_BY_NAME = {
    "flexed": FLEXED_INITIAL_JOINT_POS,
    "crab": CRAB_INITIAL_JOINT_POS,
    "safe": SAFE_INITIAL_JOINT_POS,
}

KP = 4.0
KD = 0.2
effort_limit=2.7

SOLO12_LAAS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SOLO12_LAAS_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0, # it was 100, changing to 1.
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos=SAFE_INITIAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=ACTUATOR_JOINT_NAMES,
            armature=0.00036207,
            effort_limit=effort_limit, # they had 10 Nm
            velocity_limit=100.0,
            stiffness={".*": KP},
            damping={".*": KD},
        )
    },
)


def randomize_rigid_body_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: str = "scale",
):
    """Randomize rigid-body inertia tensors for an articulation."""

    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    inertias = asset.root_physx_view.get_inertias()
    inertias[env_ids[:, None], body_ids] = asset.data.default_inertia[env_ids[:, None], body_ids].clone()

    low, high = inertia_distribution_params
    scales = math_utils.sample_uniform(low, high, (len(env_ids), len(body_ids)), device="cpu")

    if operation == "scale":
        inertias[env_ids[:, None], body_ids] *= scales[..., None]
    elif operation == "add":
        inertias[env_ids[:, None], body_ids] += scales[..., None]
    elif operation == "abs":
        inertias[env_ids[:, None], body_ids] = scales[..., None]
    else:
        raise ValueError(f"Unsupported inertia randomization operation: {operation}")

    asset.root_physx_view.set_inertias(inertias, env_ids)


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 2.0),
            "dynamic_friction_range": (0.7, 1.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (0.95, 1.2),
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 1.0),
            "operation": "abs",
        },
    )

    inertia_scale = EventTerm(
        func=randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )


@configclass
class Solo12LaasEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = len(JOINT_NAMES)
    observation_space = BASE_OBSERVATION_SPACE
    state_space = 0 # why is this zero? 
    kp = KP; kd = KD; effort_limit=effort_limit
    remove_root_lin_vel_b_from_obs = False

    joint_names = JOINT_NAMES
    # None uses initial_position as the action/observation offset.
    q_offset_action_and_obs = None
    base_body_name = BASE_BODY_NAME
    feet_body_names = FEET_BODY_NAMES
    undesired_contact_body_names = UNDESIRED_CONTACT_BODY_NAMES
    joint_wrench_body_names = JOINT_WRENCH_BODY_NAMES

    command_resampling_time_s = 10.0
    standing_env_prob = 0.02
    opposite_direction_cmd_prob = 0.05
    command_lin_vel_x_range = (-1.5, 1.5)
    command_lin_vel_y_range = (-1.0, 1.0)
    command_ang_vel_z_range = (-1.0, 1.0)

    reset_x_pos = 0.5
    reset_y_pos = 0.5
    reset_yaw = math.pi
    base_mass: float | None = 1.72
    initial_position = "safe"  # Options: "rigid", "flexed", "crab", "safe".
    initial_joint_pos_by_name = INITIAL_JOINT_POS_BY_NAME
    flexed_initial_joint_pos = FLEXED_INITIAL_JOINT_POS
    flexed_initial_joint_pos_noise_range = (-0.07, 0.07)
    reset_base_lin_vel_range = (-0.3, 0.3)
    reset_base_ang_vel_range = (-0.1, 0.1)
    # should we include observation delay?
    actuation_delay_range = (0, 3)
    base_push_interval_range_s = (10.0, 15.0)
    base_push_duration_range_s = (0.20, 1.0)
    base_push_force_xy_range = (-4.5, 4.5)
    base_push_force_z_range = (0.0, 0.0)
    base_push_torque_xy_range = (0.0, 0.0)
    # 2.5 N at a ~0.3 m lateral lever arm is about 0.75 N*m.
    base_push_torque_z_range = (-0.75, 0.75)

    forces_applied_to_base_curriculum = [5, 15, 25]
    forces_curriculum_threshold_reward = 32.0
    forces_curriculum_smoothing = 0.05

    tracking_std = math.sqrt(0.25)
    feet_air_time_threshold = 0.5
    base_contact_threshold = 1.0
    undesired_contact_threshold = 1.0

    # observations noise. 
    enable_observation_corruption = True
    base_lin_vel_noise = (-0.1, 0.1)
    base_ang_vel_noise = (-0.2, 0.2)
    projected_gravity_noise = (-0.05, 0.05)
    joint_pos_noise = (-0.01, 0.01)
    joint_vel_noise = (-1.5, 1.5)

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot: ArticulationCfg = SOLO12_LAAS_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=1 / 200, # physics dt
        track_air_time=True,
    )
    track_lin_vel_xy_reward_scale = 1.5
    track_ang_vel_z_reward_scale = 0.75
    lin_vel_z_reward_scale = 0.0
    ang_vel_xy_reward_scale = 0.00
    joint_accel_reward_scale = 0.0
    feet_air_time_reward_scale = 1.0
    undesired_contact_reward_scale = -2.25
    base_tilt_penalty_reward_scale = -0.33
    # Penalize force transmitted through thigh/calf joints.
    force_transmited_through_joints_reward_scale = 0.0
    # Penalize foot contact
    # Baseline 
    action_rate_reward_scale = -0.05; joint_torque_reward_scale = -0.5e-3;foot_contact_reward_scale = -1.0e-3
    foot_contact_safe_threshold = 10.0
    square_foot_penalty = True

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = BASE_OBSERVATION_SPACE - (
            ROOT_LIN_VEL_OBS_DIM if self.remove_root_lin_vel_b_from_obs else 0
        )
