# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class BorinotGoalkeeperEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 3.0
    is_finite_horizon = True
    action_space = 6  # collective thrust, body rates xyz, shoulder/elbow position targets
    observation_space = 55
    state_space = 55
    sim = sim_utils.SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(),
    )
    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=7.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(4.0, 4.0, 2.7), lookat=(0.3, 0.0, 1.2))

    robot_home = (0.0, 0.0, 1.5)
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot_home,
            joint_pos={"flying_arm_2__j_bl_link1": 0.0, "flying_arm_2__j_link1_link2": 0.0},
        ),
        actuators={"arm": ImplicitActuatorCfg(
            joint_names_expr=["flying_arm_2__j_bl_link1", "flying_arm_2__j_link1_link2"],
            stiffness=35.0, damping=0.6, effort_limit_sim=2.7, velocity_limit_sim=15.0,
        )},
    )
    joint_names = ("flying_arm_2__j_bl_link1", "flying_arm_2__j_link1_link2")
    joint_target_scale = (1.1, 1.4)
    hold_arm = False  # neutral PD targets; this is NOT a mechanically welded arm

    # Replace the original nominal 100 g tip ballast, rather than adding 100 g on top.
    shaft_mass_kg = 0.015
    paddle_mass_kg = 0.100
    paddle_size = (0.012, 0.24, 0.24)  # thickness along +x; broad face is yz
    paddle_offset_ee = (-0.05, -0.16, -0.021)  # face sits 5 cm ahead of the old tip geometry
    paddle_quat_ee = (0.0, 0.0, -0.7071067811865476, -0.7071067811865476)
    # Neutral FK from the USD joint frames, not the file's authored preview pose.
    keeper_plane_x = 0.1147006
    target_center_yz = (-0.0803284, 1.1292502)

    ball_radius_m = 0.03
    ball_mass_range_kg = (0.020, 0.050)
    ball_restitution_range = (0.35, 0.65)
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=ball_radius_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.0, angular_damping=0.0,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.25, dynamic_friction=0.25, restitution=0.5,
                restitution_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.04)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.55)),
    )
    goal_center = (-0.45, 0.0, 1.2)
    goal_size = (0.04, 2.4, 2.4)
    launch_x_range = (1.9, 2.1)
    launch_z = 1.55
    launch_y_noise = 0.15
    shot_half_extents_yz = ((0.045, 0.045), (0.20, 0.12), (0.40, 0.22), (0.65, 0.32))
    shot_flight_time_ranges_s = ((0.42, 0.50), (0.35, 0.50), (0.30, 0.46), (0.25, 0.42))
    shot_level = 0
    curriculum = True
    curriculum_window_episodes = 2048
    curriculum_min_episodes = 1024
    curriculum_min_steps = 3000  # 50 simulated seconds per environment, NOT 3000 aggregate env transitions
    curriculum_success_threshold = 0.7

    thrust_to_weight = 2.0  # assumed thrust reserve; needs hardware identification
    collective_action_scale = 0.7
    body_rate_scale = (2.0, 2.0, 1.5)
    rate_kp = (0.35, 0.35, 0.15)
    rate_ki = (0.10, 0.10, 0.04)
    motor_time_constant_s = 0.04
    rotor_pos_b = ((0.20, 0.00, 0.00), (0.10, 0.17, 0.00), (-0.10, 0.17, 0.00),
                   (-0.20, 0.00, 0.00), (-0.10, -0.17, 0.00), (0.10, -0.17, 0.00))
    rotor_spin = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    km_over_kf = 6.991478005829954e-08 / 4.138394792004922e-06

    contact_force_threshold_n = 0.05
    clear_distance_m = 0.25
    clear_speed_m_s = 0.3
    recovery_dwell_s = 0.25
    recovery_position_tolerance_m = 0.35
    recovery_speed_m_s = 0.5
    recovery_angular_speed_rad_s = 0.8
    recovery_tilt_rad = 0.35
    recovery_joint_tolerance_rad = 0.5
    require_recovery_for_save = False
    goal_area_x_m = (-0.08, 0.9)
    goal_area_y_m = (-1.2, 1.2)
    goal_area_z_m = (0.55, 2.6)
    reset_xy_noise_m = 0.015
    reset_z_noise_m = 0.015

    # Rate rewards/costs are multiplied by policy dt. Event rewards are paid once.
    # Main reward: Jordi's goal-parallel signed distance progress and shot outcomes.
    progress_scale = 0.5  # reward per metre of reduced yz distance
    progress_use_prediction = False  # optional: use predicted intercept rather than current ball yz
    save_bonus = 1.0
    goal_penalty = 1.0  # also used for ball/base collision, by design
    arm_hit_penalty = 0.25
    leave_goal_area_penalty = 1.0
    ball_floor_penalty = 0.25
    miss_penalty = 1.0
    deadline_penalty = 1.0
    robot_crash_penalty = 1.0

    # Optional additions: disabled by default. Enable explicitly for an ablation.
    intercept_reward_rate = 0.0
    align_reward_rate = 0.0
    recovery_reward_rate = 0.0
    intercept_sigma_m = 0.30
    hit_bonus = 0.0
    centered_hit_bonus = 0.0
    action_rate_cost = 0.0
    body_angular_velocity_cost = 0.0
    body_linear_velocity_cost = 0.0
    thrust_cost = 0.0
