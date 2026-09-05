# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.solo12.solo12_env_cfg import JOINT_NAMES, randomize_rigid_body_inertia


KP = 15.0
KD = 0.5
# KP = 15
# KD = 0.5
SOLO12_USD_PATH = Path(__file__).parents[4] / "isaaclab_assets/data/Robots/Solo12/SoloFlat.usd"

SOLO12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SOLO12_USD_PATH),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=5,
            solver_velocity_iteration_count=2, # (was 2)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=2.65,
            stiffness=KP,
            damping=KD,
            velocity_limit_sim=100.0,
            friction=0.0,
            armature=0.00036207,
            # armature=0.036207,

        )
    },
)


SOLO12_RACE_SCENE_USD_DIR = Path(__file__).parents[4] / "isaaclab_assets/data/Robots/Solo12"
SOLO12_RACE_SCENE_USD_FILES = {
    "full": "solo_IMU_race_waypoints.usd",
    "simple": "solo_IMU_race_waypoints_simple.usd",
    "simple_zigzag": "solo_IMU_race_waypoints_simple_zigzag.usd",
    "simple_zigzag_01": "solo_IMU_race_waypoints_simple_zigzag_01.usd",
    "old": "solo_IMU_race_waypoints_old.usd",
    "straightSimple": "solo_IMU_race_straightLine.usd"
}
SOLO12_RACE_DEFAULT_SCENE = "straightSimple"


def resolve_solo12_race_scene_usd_path(race_scene: str | Path) -> Path:
    """Resolve a named Solo12 race scene selector or explicit USD path."""

    race_scene = str(race_scene)
    if race_scene in SOLO12_RACE_SCENE_USD_FILES:
        return SOLO12_RACE_SCENE_USD_DIR / SOLO12_RACE_SCENE_USD_FILES[race_scene]

    for scene_file in SOLO12_RACE_SCENE_USD_FILES.values():
        if race_scene in (scene_file, Path(scene_file).stem):
            return SOLO12_RACE_SCENE_USD_DIR / scene_file

    scene_path = Path(race_scene).expanduser()
    if scene_path.suffix == ".usd":
        return scene_path

    valid_scenes = ", ".join(sorted(SOLO12_RACE_SCENE_USD_FILES))
    raise ValueError(f"Unknown Solo12 race_scene '{race_scene}'. Valid selectors: {valid_scenes}.")


SOLO12_RACE_SCENE_USD_PATH = resolve_solo12_race_scene_usd_path(SOLO12_RACE_DEFAULT_SCENE)

SOLO12_RACE_WAYPOINT_NAMES = (
    "waypoint_start",
    "waypoint_00",
    "waypoint_01",
    "waypoint_02",
    "waypoint_03",
    "waypoint_04",
    "waypoint_05",
    "waypoint_end",
)


@configclass
class EventCfg:

    # Set  robot friction coefs to one for now, will randomize the patches coefs (combine mean). 
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.8, 1.2),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    # Deactivating for now -- we might add it later.
    push_robot = None


@configclass
class Solo12RaceEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 10#10#4
    physics_dt = 1/500
    action_scale = 0.25
    action_space = len(JOINT_NAMES)
    observation_space = 63
    state_space = 0

    joint_names = JOINT_NAMES

    # authored race scene
    scene_usd: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(usd_path=str(SOLO12_RACE_SCENE_USD_PATH))
    scene_prim_path = "/World/envs/env_.*/Scene"
    scene_source_prim_path = "/World/envs/env_0/Scene"
    race_scene: str = SOLO12_RACE_DEFAULT_SCENE
    # Constant world-frame force [N] applied at the base COM opposite the straightSimple
    # start-to-end direction. Zero disables it.
    backward_force: float = 0.0
    waypoint_names = SOLO12_RACE_WAYPOINT_NAMES
    patch_name_pattern = "patch.*"

    # observations, proprio + current/following gate pillar vectors + cClose/cClose1 pillar vectors
    # policy_model is bookkeeping for experiment configs:
    # - "simple_mlp": current flat observation only
    # - "tcn_joint_states_encoder": current flat observation + joint error/velocity-error history
    # - "tcn_foot_imu_encoder": current flat observation + foot IMU history
    # - "tcn_joint_states_foot_imu_encoder": current flat observation + [joint history, foot IMU history]
    # - "env_params_conditioned_actor": current flat observation + privileged GT env params
    # - "env_params_conditioned_encoder_actor": current flat observation + encoded privileged GT env params
    policy_model = "simple_mlp"
    include_foot_imu_obs = False
    include_joint_state_history_obs = False
    include_root_lin_vel_b_obs = True

    # Set True to recover the legacy 57D race observation used by checkpoints/exported policies
    # trained before cClose/cClose1 were added. Default False keeps the new 63D observation.
    remove_c_close_vectors_from_observation = False

    # Privileged GT env-parameter observations for the params-conditioned actor.
    # Layout, when enabled: [foot contact forces body-frame (4 * 3), patch static friction below feet (4)].
    include_forces_to_gt_obs = False
    include_mu_coefs_to_gt_obs = False
    gt_foot_contact_force_obs_dim = 12
    gt_patch_mu_obs_dim = 4
    gt_obs_default_mu = 1.0
    # If False, privileged patch-friction observations are latched only after each foot contacts the floor.
    # If True, the teacher sees the static friction coefficient under each foot before touchdown.
    teachers_sees_future_friction_coef = False
    # Deprecated: divide GT contact forces by 100 before they enter the observation. This was removed
    # because the actor's EmpiricalNormalization already normalizes the forces. Set True only to play
    # old checkpoints that were trained with the prescale; leave False for new training.
    include_deprecated_force_normalization = False

    # foot IMU history for the TCN policy: D = 4 feet * (3 gyro + 3 accel) = 24.
    foot_imu_obs_dim = 24
    foot_imu_history_policy_steps = 5
    foot_imu_history_length = decimation * foot_imu_history_policy_steps
    foot_imu_tcn_channels = 32
    foot_imu_tcn_latent_dim = 64
    foot_imu_tcn_kernel_size = 5
    foot_imu_tcn_activation = "relu"

    # Joint-state history baseline for the TCN policy: D = 12 q error + 12 qd error = 24.
    # It intentionally uses the same TCN history length and architecture as the foot-IMU policy.
    joint_state_history_obs_dim = 24
    joint_state_history_policy_steps = foot_imu_history_policy_steps
    joint_state_history_length = decimation * joint_state_history_policy_steps
    joint_state_tcn_channels = foot_imu_tcn_channels
    joint_state_tcn_latent_dim = foot_imu_tcn_latent_dim
    joint_state_tcn_kernel_size = foot_imu_tcn_kernel_size
    joint_state_tcn_activation = foot_imu_tcn_activation

    # Combined TCN baseline: D = 24 joint-state history + 24 foot-IMU history = 48.
    joint_imu_history_obs_dim = joint_state_history_obs_dim + foot_imu_obs_dim
    joint_imu_history_policy_steps = foot_imu_history_policy_steps
    joint_imu_history_length = decimation * joint_imu_history_policy_steps
    joint_imu_tcn_channels = foot_imu_tcn_channels
    joint_imu_tcn_latent_dim = foot_imu_tcn_latent_dim
    joint_imu_tcn_kernel_size = foot_imu_tcn_kernel_size
    joint_imu_tcn_activation = foot_imu_tcn_activation

    enable_observation_corruption = True
    enable_actuation_delay = False
    enable_events_randomization = False
    enable_reset_pose_randomization = False
    randomize_fric_coefs = True
    # Use a deterministic stratified bucket grid by default. This keeps the friction distribution fixed across
    # seeds, so seed changes test different patch assignments rather than a different sampled material table.
    # Set True only when intentionally testing/training on randomly sampled bucket values.
    randomize_friction_bucket_values = False
    group_all_patches_single_bucket = True
    within_episode_fric_resample = True
    within_episode_fric_resample_time_range = (0.7, 3.0) # seconds
    # Keep False for training throughput. Interactive play can flip this on so timed friction resamples also update
    # patch colors/inspectable USD bindings, not only PhysX contact tensors.
    within_episode_fric_resample_update_usd = False
    # Optional inference/testing override: when set, per-patch bucket assignments use this seed instead of the
    # environment-wide RNG. If randomize_friction_bucket_values=True, bucket values use it too.
    friction_seed: int | None = None

    # race objective
    gate_radius = 0.30
    finish_radius = 0.30
    gate_progress_bodyrate_coeff = 0.00
    floor_collision_penalty = -10.0
    pillar_collision_penalty = -5.0
    reward_reach_waypoint = 5.0
    finish_reward = 50.0

    forward_progress_reward_scale = 100.0

    # solo-style regularizers, excluding the command-tracking terms
    lin_vel_z_reward_scale = 0.0
    ang_vel_xy_reward_scale = 0.0
    joint_accel_reward_scale = 0.0
    feet_air_time_reward_scale = 0.0
    undesired_contact_reward_scale = -2.0
    base_tilt_penalty_reward_scale = 0.0
    action_rate_reward_scale = 0.00
    joint_torque_reward_scale = 0.00
    foot_contact_reward_scale = 0.00
    # Sum over feet of forward reaction-force alignment times the normalized friction-angle magnitude.
    # This is a per-policy-step reward (it is not multiplied by step_dt). Zero preserves legacy behavior.
    scale_dense_reaction_force_reward = 0.0

    # randomization / reset
    reset_x_pos = 0.0
    reset_y_pos = 0.0
    reset_yaw_noise = 0.18
    reset_base_lin_vel_range = (-0.15, 0.15)
    reset_base_ang_vel_range = (-0.10, 0.10)
    actuation_delay_range = (0, 2)


    # friction patches
    friction_num_buckets = 1000
    friction_static_range = (3.0, 3.0)
    # Dynamic friction is always derived as mu_dynamic_static_ratio * static_friction, for every patch/bucket,
    # whether or not the static coefficient is randomized, and identically at train and inference time.
    # Override per-run with `env.mu_dynamic_static_ratio=<value>`.
    mu_dynamic_static_ratio = 0.5
    if randomize_fric_coefs:
        friction_static_range = (0.03, 1.5)

    friction_color_low = (0.10, 0.35, 0.95)
    friction_color_high = (0.95, 0.20, 0.10)

    # cameras are for inference-time visual evaluation only, never policy inputs
    enable_inference_cameras = False
    active_camera = "overhead"
    camera_width = 160
    camera_height = 120
    camera_focal_length = 24.0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt, #1/500, #1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # These friction coefs are the defaults for rigid bodies that don't have a physics material assigned

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=16.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot: ArticulationCfg = SOLO12_CFG.replace(
        spawn=None,
        prim_path="/World/envs/env_.*/Scene/SoloFlat",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.36)),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/.*",
        history_length=3,
        update_period=1 / 200,
        track_air_time=True,
    )
    # Update period for the auxiliary per-foot reaction sensors used by slip diagnostics
    # (normal + friction/contact-point data). A value <= 0 follows physics_dt; set 0.005 for 200 Hz.
    foot_reaction_sensor_update_period_s: float = 0.0
    # Deprecated longer alias. Kept so older commands keep working.
    foot_reaction_contact_sensor_update_period_s: float = 0.0
    base_pillar_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/base",
        history_length=3,
        update_period=1 / 200,
        filter_prim_paths_expr=[],
    )
    base_floor_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/base",
        history_length=3,
        update_period=1 / 200,
        filter_prim_paths_expr=[],
    )

    # Foot-mounted IMUs. The authored foot xForms are fixed-joint siblings, not rigid descendants, so the sensors
    # attach to the calf rigid bodies with offsets that place the IMU frames at FL/FR/RL/RR_foot.
    imu_fl: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/FL_calf",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.009000003337860107, -0.1599999964237213)),
        gravity_bias=(0.0, 0.0, 9.81),
    )
    imu_fr: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/FR_calf",
        offset=ImuCfg.OffsetCfg(pos=(0.0, -0.009000003337860107, -0.1599999964237213)),
        gravity_bias=(0.0, 0.0, 9.81),
    )
    imu_rl: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/RL_calf",
        offset=ImuCfg.OffsetCfg(pos=(0.0, 0.009000003337860107, -0.1599999964237213)),
        gravity_bias=(0.0, 0.0, 9.81),
    )
    imu_rr: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/RR_calf",
        offset=ImuCfg.OffsetCfg(pos=(0.0, -0.009000003337860107, -0.1599999964237213)),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # cameras, created only if enable_inference_cameras=True
    overhead_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Scene/OverheadCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.7, 0.0, 7.5), rot=(0.5, -0.5, 0.5, -0.5), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=camera_width,
        height=camera_height,
        update_latest_camera_pose=False,
    )
    side_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Scene/SoloFlat/base/SideCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.55, 0.0, 0.18), rot=(0.7071, 0.0, 0.7071, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=camera_focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=camera_width,
        height=camera_height,
        update_latest_camera_pose=True,
    )

    feet_air_time_threshold = 0.35
    base_contact_threshold = 1.0
    undesired_contact_threshold = 1.0
    foot_contact_safe_threshold = 10.0
    square_foot_penalty = True

    base_ang_vel_noise = (-0.15, 0.15)
    base_lin_vel_noise = (-0.10, 0.10)
    projected_gravity_noise = (-0.05, 0.05)
    joint_pos_noise = (-0.01, 0.01)
    joint_vel_noise = (-1.25, 1.25)

    # foot_imu_ang_vel_noise = (-0.15, 0.15)
    # foot_imu_lin_acc_noise = (-0.5, 0.5)

    # IMU noise recommendation: https://chatgpt.com/share/69ee36d5-6740-832f-8bad-cd1fb7c66d48
    # per-step white noise
    foot_imu_gyro_noise_range = (-0.05, 0.05)      # rad/s
    foot_imu_acc_noise_range  = (-0.3, 0.3)        # m/s²

    # per-episode constant bias
    foot_imu_gyro_bias_range = (-0.05, 0.05)       # rad/s
    foot_imu_acc_bias_range  = (-0.3, 0.3)         # m/s²


    def __post_init__(self):
        super().__post_init__()
        obs_dim = 3 + 3 + 12 + 12 + 12  # root angular velocity, gravity, joints, joint velocities, actions
        obs_dim += 4 * 3  # current gate c1/c2 + following gate c1/c2
        if not self.remove_c_close_vectors_from_observation:
            obs_dim += 2 * 3  # cClose/cClose1 vectors: closest and second-closest pillars
        if self.include_root_lin_vel_b_obs:
            obs_dim += 3
        self.base_observation_dim = obs_dim
        self.gt_env_params_obs_dim = 0
        if self.include_forces_to_gt_obs:
            self.gt_env_params_obs_dim += self.gt_foot_contact_force_obs_dim
        if self.include_mu_coefs_to_gt_obs:
            self.gt_env_params_obs_dim += self.gt_patch_mu_obs_dim
        obs_dim += self.gt_env_params_obs_dim
        self.proprio_observation_dim = obs_dim
        self.foot_imu_history_length = self.decimation * self.foot_imu_history_policy_steps
        self.joint_state_history_length = self.decimation * self.joint_state_history_policy_steps
        self.joint_imu_history_length = self.decimation * self.joint_imu_history_policy_steps
        if self.include_foot_imu_obs and self.include_joint_state_history_obs:
            if self.foot_imu_history_length != self.joint_state_history_length:
                raise ValueError("Combined joint-state/IMU TCN requires matching history lengths.")
            self.joint_imu_history_length = self.foot_imu_history_length
            obs_dim += self.joint_imu_history_length * self.joint_imu_history_obs_dim
        elif self.include_foot_imu_obs:
            obs_dim += self.foot_imu_history_length * self.foot_imu_obs_dim
        elif self.include_joint_state_history_obs:
            obs_dim += self.joint_state_history_length * self.joint_state_history_obs_dim
        self.observation_space = obs_dim


@configclass
class Solo12RaceEvalCameraEnvCfg(Solo12RaceEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=16.0, replicate_physics=True)
    enable_inference_cameras = True
    active_camera = "overhead"


@configclass
class Solo12RaceIMUEnvCfg(Solo12RaceEnvCfg):
    policy_model = "tcn_foot_imu_encoder"
    include_foot_imu_obs = True


@configclass
class Solo12RaceJointStateTcnEnvCfg(Solo12RaceEnvCfg):
    policy_model = "tcn_joint_states_encoder"
    include_joint_state_history_obs = True


@configclass
class Solo12RaceJointStateImuTcnEnvCfg(Solo12RaceEnvCfg):
    policy_model = "tcn_joint_states_foot_imu_encoder"
    include_foot_imu_obs = True
    include_joint_state_history_obs = True


@configclass
class Solo12RaceParamsConditionedEnvCfg(Solo12RaceEnvCfg):
    policy_model = "env_params_conditioned_actor"
    include_forces_to_gt_obs = True
    include_mu_coefs_to_gt_obs = True


@configclass
class Solo12RaceParamsConditionedEncEnvCfg(Solo12RaceParamsConditionedEnvCfg):
    policy_model = "env_params_conditioned_encoder_actor"


@configclass
class Solo12RaceParamsDaggerJointStateTcnEnvCfg(Solo12RaceParamsConditionedEncEnvCfg):
    """Phase-2 DAgger env: privileged GT labels plus joint-error history for a latent TCN adapter."""

    policy_model = "env_params_dagger_joint_state_tcn"
    include_joint_state_history_obs = True


@configclass
class Solo12RaceParamsDaggerJointStateImuTcnEnvCfg(Solo12RaceParamsConditionedEncEnvCfg):
    """Phase-2 DAgger env: privileged GT labels plus joint-error and foot-IMU history."""

    policy_model = "env_params_dagger_joint_state_imu_tcn"
    include_foot_imu_obs = True
    include_joint_state_history_obs = True


@configclass
class Solo12RaceIMUEvalCameraEnvCfg(Solo12RaceEvalCameraEnvCfg):
    policy_model = "tcn_foot_imu_encoder"
    include_foot_imu_obs = True


@configclass
class Solo12RaceJointStateTcnEvalCameraEnvCfg(Solo12RaceEvalCameraEnvCfg):
    policy_model = "tcn_joint_states_encoder"
    include_joint_state_history_obs = True


@configclass
class Solo12RaceJointStateImuTcnEvalCameraEnvCfg(Solo12RaceEvalCameraEnvCfg):
    policy_model = "tcn_joint_states_foot_imu_encoder"
    include_foot_imu_obs = True
    include_joint_state_history_obs = True
