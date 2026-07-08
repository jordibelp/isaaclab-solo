# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass


SOLO12_USD_PATH = Path(__file__).parents[4] / "isaaclab_assets/data/Robots/Solo12/SoloFlat.usd"

ROOT_LIN_VEL_OBS_DIM = 3
BASE_OBSERVATION_SPACE = 48
COMMAND_OBS_DIM = 3
BASE_IMU_RAW_OBS_DIM = 6
BASE_IMU_PROJECTED_GRAVITY_OBS_DIM = 9
BASE_IMU_ROTATION_MATRIX_OBS_DIM = 15

JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

FLEXED_INITIAL_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
}

CRAB_INITIAL_JOINT_POS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": -0.8,
    "RL_calf_joint": 1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": -0.8,
    "RR_calf_joint": 1.5,
}

SAFE_INITIAL_JOINT_POS = dict(
    zip(
        JOINT_NAMES,
        [
            0.0,
            0.4,
            -0.8,
            0.0,
            0.4,
            -0.8,
            0.0,
            -0.4,
            0.8,
            0.0,
            -0.4,
            0.8,
        ],
    )
)

INITIAL_JOINT_POS_BY_NAME = {
    "flexed": FLEXED_INITIAL_JOINT_POS,
    "crab": CRAB_INITIAL_JOINT_POS,
    "safe": SAFE_INITIAL_JOINT_POS,
}

KP = 9.0
KD = 0.2
# KP = 15
# KD = 0.5

SOLO12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SOLO12_USD_PATH),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1, # (was 2)
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
        )
    },
)

# Sub-terrain proportions. In curriculum mode Isaac Lab assigns each terrain column a single
# sub-terrain type by walking the cumulative (normalized) proportions, so a type with proportion p
# occupies ~p * num_cols contiguous columns. We reuse the same proportions to weight where robots
# spawn, so a higher-proportion terrain also receives proportionally more spawns.
proportionFlatTerrain = 0.25
proportionHfRandomUniformTerrain = 0.20
proportionHfDiscreteObstaclesTerrain = 0.55

SOLO12_TRICKY_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=None,
    curriculum=True,
    # size=(20.0, 20.0),
    size = (13., 13.),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.10,
    vertical_scale=0.002,
    slope_threshold=0.75,
    color_scheme="none",
    use_cache=False,
    sub_terrains={
        # Columns 0-2: flat terrain. We spawn on the inner flat column to avoid edge falls.
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=proportionFlatTerrain),
        # Columns 3-5: dense low rugosity up to 30 mm.
        "low_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=proportionHfRandomUniformTerrain,
            noise_range=(0.0, 0.030),
            noise_step=0.002,
            downsampled_scale=0.20,
            border_width=0.50,
        ),
        # Columns 6-11: dense 2-4 cm-ish raised patches to force real foot clearance.
        "small_raised_patches": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=proportionHfDiscreteObstaclesTerrain,
            obstacle_height_mode="fixed",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.010, 0.025),
            num_obstacles=400,
            platform_width=1.00,
            border_width=0.10,
        ),
    },
)


def _curriculum_subterrain_per_column(terrain_cfg) -> list[str]:
    """Return the sub-terrain name occupying each terrain column under curriculum generation.

    Mirrors ``TerrainGenerator._generate_curriculum_terrains``: columns are partitioned by the
    cumulative (normalized) proportions, so a sub-terrain with proportion ``p`` fills ~``p * num_cols``
    contiguous columns. This describes the terrain layout only, not where robots are spawned.
    """
    names = list(terrain_cfg.sub_terrains.keys())
    proportions = [terrain_cfg.sub_terrains[name].proportion for name in names]
    total = sum(proportions)
    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)
    per_column = []
    for index in range(terrain_cfg.num_cols):
        q = index / terrain_cfg.num_cols + 0.001
        sub_index = next((i for i, c in enumerate(cumulative) if q < c), len(names) - 1)
        per_column.append(names[sub_index])
    return per_column


def _proportional_tricky_spawn_columns(
    terrain_cfg, flat_name: str = "flat", exclude_edge_cols: int = 2
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Build ``(columns, weights)`` for spawning while the tricky curriculum is active.

    - Drops the outermost ``exclude_edge_cols`` columns on each side so robots can't fall off the world.
    - Keeps only the non-flat ("tricky") columns.
    - Weights columns so the probability of spawning on a sub-terrain type matches its proportion:
      a type's proportion is shared equally across its kept columns. Weights are unnormalized;
      ``torch.multinomial`` normalizes them at sampling time.
    """
    per_column = _curriculum_subterrain_per_column(terrain_cfg)
    allowed = range(exclude_edge_cols, terrain_cfg.num_cols - exclude_edge_cols)
    tricky_cols = [col for col in allowed if per_column[col] != flat_name]
    columns_per_type: dict[str, int] = {}
    for col in tricky_cols:
        columns_per_type[per_column[col]] = columns_per_type.get(per_column[col], 0) + 1
    columns = tuple(tricky_cols)
    weights = tuple(
        terrain_cfg.sub_terrains[per_column[col]].proportion / columns_per_type[per_column[col]]
        for col in tricky_cols
    )
    return columns, weights


# Central curriculum rows keep robots away from the easiest/hardest difficulty extremes.
SOLO12_TRICKY_TERRAIN_SPAWN_ROWS = (4, 5)
# Inactive-curriculum fallback: spawn on the inner flat column (index 2), away from the world edge.
SOLO12_TRICKY_TERRAIN_FLAT_COLS = (2,)
# Active curriculum: spawn on the tricky (non-flat) columns, dropping the outer two columns on each
# side (fall-off-world risk). Per-column weights make the spawn share of each terrain type follow its
# proportion, so the higher-proportion terrain receives proportionally more spawns.
SOLO12_TRICKY_TERRAIN_COLS, SOLO12_TRICKY_TERRAIN_COL_WEIGHTS = _proportional_tricky_spawn_columns(
    SOLO12_TRICKY_TERRAINS_CFG
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
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.90, 1.2),
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
            "com_range": {"x": (-0.015, 0.015), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        },
    )


@configclass
class Solo12EnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = len(JOINT_NAMES)
    observation_space = BASE_OBSERVATION_SPACE
    state_space = 0 # why is this zero? 
    kp = KP; kd = KD
    proportion_steps = proportionHfDiscreteObstaclesTerrain; proportion_low_random_rough_terrain = proportionHfRandomUniformTerrain
    remove_root_lin_vel_b_from_obs = False
    # Selects the policy/observation layout. Supported values:
    # - "simple_mlp": legacy flat observation.
    # - "simple_dreamer_v3": Dreamer observation layout controlled by the Dreamer runner config.
    # - "base_imu_teacher": privileged encoder input -> latent -> actor.
    # - "base_imu_student_rl": base-IMU history TCN -> latent -> actor, privileged critic.
    # - "base_imu_student_dagger": exposes both teacher labels and student history for supervised DAgger.
    policy_model = "simple_mlp"
    # ``imu_raw_inputs`` enables the physics-rate history buffer for the student tasks.
    # ``imu_ekf_processed_inputs`` selects the signal contract inside that history.
    imu_raw_inputs = False
    imu_processed_inputs = False  # Deprecated compatibility field; use imu_ekf_processed_inputs.
    imu_ekf_processed_inputs = True
    use_rotMat_on_imu_encoder = False
    include_foot_height_obs = False
    teacher_latent_dim = 32
    teacher_encoder_hidden_dims = [256, 128, 64]
    feed_history_encoding_to_critic = False

    # Base IMU history for student TCN: one sample per physics step.
    # With decimation=4 and policy_steps=3, this gives 13 samples covering t-3 ... t.
    base_imu_history_policy_steps = 3
    base_imu_obs_dim = BASE_IMU_PROJECTED_GRAVITY_OBS_DIM
    base_imu_history_length = decimation * base_imu_history_policy_steps + 1
    base_imu_history_joint_state_dim = 24
    base_imu_history_action_dim = len(JOINT_NAMES)
    base_imu_history_sample_dim = base_imu_history_joint_state_dim + base_imu_obs_dim + base_imu_history_action_dim
    base_imu_tcn_channels = 64
    base_imu_tcn_latent_dim = teacher_latent_dim
    base_imu_tcn_kernel_size = 5
    base_imu_tcn_activation = "elu"
    # Do we want to use last executed action? 
    base_imu_history_action_is_last_executed = False

    # Base-mounted IMU stochastic model. The deterministic signal comes from PhysX link accelerations.
    # Defaults are robustness-oriented but still modest compared with the old foot-IMU corruption.
    # Claude recommendation: https://claude.ai/share/30c6895d-7493-46b6-86ba-05d93dcd3add
    noisy_imu = True
    imu_noise_scale = 1.0
    imu_acc_noise_scale = 1.0
    imu_gyro_noise_scale = 1.0
    imu_orientation_noise_scale = 1.0

    # Legacy raw-specific-force model used when imu_ekf_processed_inputs=False.
    base_imu_acc_noise_std = 0.01 #0.02          # m/s^2, Gaussian white noise per physics sample
    base_imu_gyro_noise_std = 0.003 #0.005        # rad/s, Gaussian white noise per physics sample (0.003 it is also OK)
    base_imu_acc_bias_init_std = 0.05 #0.1      # m/s^2, per-episode Gaussian initial bias
    base_imu_gyro_bias_init_std = 0.003 #0.01     # rad/s, per-episode Gaussian initial bias
    base_imu_acc_bias_rw_std_per_step = 5.0e-5   # m/s^2 per physics step
    base_imu_gyro_bias_rw_std_per_step = 2.0e-6  # rad/s per physics step
    base_imu_clip = True
    base_imu_acc_clip = 20.0 * 9.80665 # 20 g
    base_imu_gyro_clip = math.radians(900.0)  # 900 deg/s

    # Approximation of the signals currently published by /odri/imu:
    # gravity-free EKF linear acceleration, direct scaled gyro, and EKF attitude.
    # The CX5 filter dynamics/latency are not modeled until measured hardware logs are available.
    base_imu_ekf_acc_noise_std = 0.02  # m/s^2, residual high-frequency output noise
    base_imu_ekf_acc_bias_init_std = 0.05  # m/s^2, correlated gravity-removal/model residual
    base_imu_ekf_acc_bias_rw_std_per_step = 2.0e-5  # m/s^2 per 200 Hz physics step
    base_imu_ekf_acc_clip = 32767.0 / 2048.0  # master-board signed Q11 transport limit
    # Attitude error is applied as an axis-angle perturbation before deriving projected gravity/rotation matrix.
    # Robustness bias scales follow the CX5 attitude accuracy: 0.25 deg roll/pitch and 0.8 deg heading.
    base_imu_orientation_noise_std_rpy = tuple(math.radians(v) for v in (0.03, 0.03, 0.10))
    base_imu_orientation_bias_init_std_rpy = tuple(math.radians(v) for v in (0.25, 0.25, 0.80))
    base_imu_orientation_bias_rw_std_per_step_rpy = (2.0e-6, 2.0e-6, 5.0e-6)
    # True reports the delayed/executed raw action; False reports the policy-commanded raw action.
    action_obs_is_last_executed = False

    joint_names = JOINT_NAMES
    # None uses initial_position as the action/observation offset.
    q_offset_action_and_obs = None

    command_resampling_time_s = 10.0
    standing_env_prob = 0.02
    opposite_direction_cmd_prob = 0.05
    command_lin_vel_x_range = (-2.0, 2.0)
    command_lin_vel_y_range = (-1.0, 1.0)
    command_ang_vel_z_range = (-1.0, 1.0)

    reset_x_pos = 0.5
    reset_y_pos = 0.5
    reset_yaw = math.pi
    base_mass: float | None = 1.75124
    initial_position = "safe"  # Options: "rigid", "flexed", "crab", "safe".
    initial_joint_pos_by_name = INITIAL_JOINT_POS_BY_NAME
    flexed_initial_joint_pos = FLEXED_INITIAL_JOINT_POS
    flexed_initial_joint_pos_noise_range = (-0.07, 0.07)
    reset_base_lin_vel_range = (-0.3, 0.3)
    reset_base_ang_vel_range = (-0.1, 0.1)
    # Delays the joint-position target by physics steps.
    actuation_delay_range = (0, 3)
    base_push_interval_range_s = (10.0, 15.0)
    base_push_duration_range_s = (0.5, 2.0)
    base_push_force_xy_range = (-4.5, 4.5)
    base_push_force_z_range = (-10.0, 10.0)
    # Half-extents of base/collision_main. Push points are sampled by area over its top and four side faces.
    base_push_application_half_extents = (0.22475, 0.09866, 0.01853)

    forces_applied_to_base_curriculum = [5.0]
    max_velx_range_curriculum = [1.0, 1.5]
    forces_curriculum_threshold_reward = 28.0
    forces_curriculum_smoothing = 0.05

    # Values larger than the available curriculum range are clipped to the final curriculum index.
    curriculum_tricky_terrain_idx: None | int = 2
    tricky_terrain = False
    tricky_terrain_spawn_rows = SOLO12_TRICKY_TERRAIN_SPAWN_ROWS
    tricky_terrain_flat_cols = SOLO12_TRICKY_TERRAIN_FLAT_COLS
    tricky_terrain_cols = SOLO12_TRICKY_TERRAIN_COLS
    # Per-column spawn weights aligned with tricky_terrain_cols; proportion-matched (see
    # _proportional_tricky_spawn_columns). None falls back to uniform sampling over the columns.
    tricky_terrain_col_weights = SOLO12_TRICKY_TERRAIN_COL_WEIGHTS
    # terrain damping
    compliant_contact_stiffness = 0.0 #3_005.0
    compliant_contact_damping = 0.0 #60.0


    tracking_std = math.sqrt(0.25)
    feet_air_time_threshold = 0.5
    base_contact_threshold = 1.0
    undesired_contact_threshold = 0.6
    feet_ground_contact_threshold = 1.0
    two_feet_above_height_threshold = 0.6
    two_feet_above_height_alpha = 10.0

    base_z_desired = 0.16
    track_base_height_reward_scale = 0.0
    base_height_exp_scale = 80.0

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
            compliant_contact_stiffness=compliant_contact_stiffness,
            compliant_contact_damping=compliant_contact_damping,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot: ArticulationCfg = SOLO12_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=1 / 200, # physics dt
        track_air_time=True,
    )
    base_imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        # On the real robot the IMU is mounted 1 cm in front of the CoM. The base link frame
        # origin nominally coincides with the CoM, so offset the sensor +1 cm along base +x
        # (forward). The sensor accounts for the lever arm, adding the centripetal/tangential
        # acceleration this mounting picks up under angular motion.
        offset=ImuCfg.OffsetCfg(pos=(0.01, 0.0, 0.0)),
        gravity_bias=(0.0, 0.0, 9.81),
    )
    track_lin_vel_xy_reward_scale = 1.5
    track_ang_vel_z_reward_scale = 0.75
    lin_vel_z_reward_scale = 0.0
    ang_vel_xy_reward_scale = 0.00
    joint_accel_reward_scale = 0.0
    feet_air_time_reward_scale = 0.0
    two_feet_above_height_reward_scale = 0.0
    three_or_more_feet_contact_penalty_reward_scale = 0.0
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
        self.refresh_runtime_dependent_config()

    def refresh_runtime_dependent_config(self):
        self.refresh_observation_dimensions()
        self._apply_runtime_overrides_to_nested_cfg()
        if self.tricky_terrain:
            self.terrain.terrain_type = "generator"
            self.terrain.terrain_generator = SOLO12_TRICKY_TERRAINS_CFG.copy()
            self.terrain.use_terrain_origins = True

    def _apply_runtime_overrides_to_nested_cfg(self):
        self.robot.actuators["legs"].stiffness = self.kp
        self.robot.actuators["legs"].damping = self.kd
        self.terrain.physics_material.compliant_contact_stiffness = self.compliant_contact_stiffness
        self.terrain.physics_material.compliant_contact_damping = self.compliant_contact_damping

    def refresh_observation_dimensions(self):
        """Recompute observation dimensions after Hydra overrides."""
        self.refresh_base_imu_dimensions()
        legacy_obs_dim = BASE_OBSERVATION_SPACE - (
            ROOT_LIN_VEL_OBS_DIM if self.remove_root_lin_vel_b_from_obs else 0
        )

        if self.policy_model == "simple_dreamer_v3":
            command_outside_observation = bool(getattr(self, "_dreamer_command_outside_observation", False))
            command_dim = 0 if command_outside_observation else COMMAND_OBS_DIM
            self.observation_space = legacy_obs_dim - COMMAND_OBS_DIM - len(JOINT_NAMES) + command_dim
            self.state_space = 0
        elif self.policy_model not in {"base_imu_teacher", "base_imu_student_rl", "base_imu_student_dagger"}:
            self.observation_space = legacy_obs_dim

    def refresh_base_imu_dimensions(self):
        """Recompute student observation dimensions after Hydra overrides."""
        for name in (
            "imu_noise_scale",
            "imu_acc_noise_scale",
            "imu_gyro_noise_scale",
            "imu_orientation_noise_scale",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        teacher_encoder_dim = 3 + 3 + 3 + 24 + len(JOINT_NAMES)
        if self.include_foot_height_obs:
            teacher_encoder_dim += 4
        self.teacher_encoder_obs_dim = teacher_encoder_dim
        self.teacher_critic_obs_dim = teacher_encoder_dim + COMMAND_OBS_DIM

        if self.imu_ekf_processed_inputs:
            self.base_imu_obs_dim = (
                BASE_IMU_ROTATION_MATRIX_OBS_DIM
                if self.use_rotMat_on_imu_encoder
                else BASE_IMU_PROJECTED_GRAVITY_OBS_DIM
            )
        else:
            self.base_imu_obs_dim = BASE_IMU_RAW_OBS_DIM
        self.base_imu_history_length = self.decimation * self.base_imu_history_policy_steps + 1
        self.base_imu_history_sample_dim = (
            self.base_imu_history_joint_state_dim + self.base_imu_obs_dim + self.base_imu_history_action_dim
        )
        self.base_imu_history_flat_dim = self.base_imu_history_length * self.base_imu_history_sample_dim

        if self.policy_model == "base_imu_teacher":
            self.observation_space = self.teacher_critic_obs_dim
            self.state_space = self.teacher_critic_obs_dim
        elif self.policy_model == "base_imu_student_rl":
            self.imu_raw_inputs = True
            self.observation_space = self.base_imu_history_flat_dim + COMMAND_OBS_DIM
            self.state_space = self.teacher_critic_obs_dim
            if self.feed_history_encoding_to_critic:
                self.state_space += self.base_imu_history_flat_dim
        elif self.policy_model == "base_imu_student_dagger":
            self.imu_raw_inputs = True
            self.observation_space = self.teacher_critic_obs_dim + self.base_imu_history_flat_dim + COMMAND_OBS_DIM
            self.state_space = self.teacher_critic_obs_dim


@configclass
class Solo12BaseImuTeacherEnvCfg(Solo12EnvCfg):
    policy_model = "base_imu_teacher"
    include_foot_height_obs = False


@configclass
class Solo12BaseImuStudentRlEnvCfg(Solo12EnvCfg):
    policy_model = "base_imu_student_rl"
    imu_raw_inputs = True


@configclass
class Solo12BaseImuStudentDaggerEnvCfg(Solo12EnvCfg):
    policy_model = "base_imu_student_dagger"
    imu_raw_inputs = True


@configclass
class Solo12SimpleDreamerV3EnvCfg(Solo12EnvCfg):
    """Solo12 task variant for DreamerV3-style latent dynamics training."""

    policy_model = "simple_dreamer_v3"


@configclass
class Solo12TwoFeetEnvCfg(Solo12EnvCfg):
    """Solo12 task variant that rewards walking with either front or rear feet airborne."""

    track_lin_vel_xy_reward_scale = 1.0
    track_ang_vel_z_reward_scale = 0.5
    two_feet_above_height_reward_scale = 1.5
    three_or_more_feet_contact_penalty_reward_scale = -0.1

    command_ang_vel_z_range = (-0.5, 0.5)
    command_lin_vel_y_range = (-0.3, 0.3)
    command_lin_vel_x_range = (-0.6, 0.6)
    max_velx_range_curriculum = [0.6]

    initial_position = "safe"
    track_base_height_reward_scale = 0.0
    enable_observation_corruption = False
    tricky_terrain = False
    base_push_force_z_range = (0.0, 0.0)
    forces_applied_to_base_curriculum = [0.0]
    actuation_delay_range = (0, 0)
    opposite_direction_cmd_prob = 0.0
    kp = 15.0
    kd = 0.5
    episode_length_s = 10.0
