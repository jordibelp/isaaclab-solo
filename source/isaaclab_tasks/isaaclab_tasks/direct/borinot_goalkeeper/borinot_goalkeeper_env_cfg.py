# SPDX-License-Identifier: BSD-3-Clause
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .events import BorinotEventCfg
from isaaclab_tasks.controllers.borinot_indi import IndiControllerCfg


@configclass
class BorinotGoalkeeperEnvCfg(DirectRLEnvCfg):
    control_mode = "thrust_and_torque_control"
    # INDI PPO actions: world acceleration [m/s^2], body rates [rad/s], heading offset [rad],
    # joint angles [rad]. No action clipping/scaling; heading adds robot_home_quat yaw.
    # INDI SAC keeps normalized [-1, 1] references with scales in indi/body_rate_scale.
    indi = IndiControllerCfg()  # used only by control_mode="indi"; see INDI_CONTROL.md
    # geom = method 3; alternatives: method_1, method_2. Definitions:
    # /home/jordibelp/Desktop/emmaDocs/2026-10-23-tau-log-exp-comparison.ipynb
    indi_tilt_method = "geom"  # used only by control_mode="indi"; inherited by Borinot-cmd-vel
    enabled_self_collisions = False  # env.enabled_self_collisions=True enables eligible nonadjacent body pairs
    physics_hz = 200
    policy_hz = 100
    decimation = 2  # derived from physics_hz / policy_hz after Hydra overrides
    episode_length_s = 20.0
    is_finite_horizon = False  # artificial collection limit; bootstrap the pre-reset state
    shot_timeout_s = 3.0  # unresolved ball ends this shot, not the robot episode
    shot_resampling_time_range_s = (2.0, 3.0)
    shot_resampling_mode = "launch"  # "launch": launch-to-launch; "resolution": delay after each result
    reset_on_goal = False  # True resets the robot too; goal scoring/penalty stay unchanged
    ball_park_position = (0.0, 0.0, -2.0)  # inactive ball is removed below the local floor
    action_space = 8  # derived: 8 direct, 6 PI, 9 INDI
    observation_space = 44  # derived: actor inputs selected by the obs_actor_includes_* flags, plus the current actions
    state_space = 56  # derived: 48 + actions + controller state (9 PI, 8 INDI) + optional rotor speeds (6)
    sim = sim_utils.SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(),
    )
    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=7.0, replicate_physics=True)
    viewer = ViewerCfg(eye=(4.0, 4.0, 2.7), lookat=(0.3, 0.0, 1.2))

    # Sample once per environment at startup, as in Solo12. Hydra accepts dotted dict keys.
    include_events_randomization = True
    event_randomization = {
        "joint_friction": True, "add_base_mass": True, "racket_mass": True, "base_com": True,
    }
    events: BorinotEventCfg | None = BorinotEventCfg()

    robot_home = (0.0, 0.0, 1.5)  # centre of the reset grid; also the observation/recovery reference
    robot_home_quat = (0.7071067811865476, 0.0, 0.0, -0.7071067811865476)  # wxyz; -90 deg yaw
    effort_lim_sim = 2.7  # N*m soft limit; scales SAC torque actions only. PPO outputs N*m directly.
    velocity_lim_sim = 100.0  # rad/s
    kp = 9.0  # N*m/rad; implicit PD in position modes, explicit PID in INDI
    kd = 0.2  # N*m*s/rad; disabled in direct torque mode unless hold_arm=True
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=enabled_self_collisions,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot_home,
            rot=robot_home_quat,  # arm sweep lies in the goal's Y-Z plane
            joint_pos={"flying_arm_2__j_bl_link1": 0.0, "flying_arm_2__j_link1_link2": 0.0},
        ),
        actuators={"arm": ImplicitActuatorCfg(
            joint_names_expr=["flying_arm_2__j_bl_link1", "flying_arm_2__j_link1_link2"],
            stiffness=0.0, damping=0.0, effort_limit_sim=effort_lim_sim,
            velocity_limit_sim=velocity_lim_sim, friction=0.0, armature=0.00036207,
        )},
    )
    joint_names = ("flying_arm_2__j_bl_link1", "flying_arm_2__j_link1_link2")
    joint_target_scale = (1.1, 1.4) # why this target scales? 
    hold_arm = False  # neutral PD/PID targets; this is NOT a mechanically welded arm

    # Enabled by the SAC runner before env creation. INDI PPO uses raw physical units.
    sac_action_mapping = False
    sac_q_offset_init_actions = False  # Solo semantics: initialize mean at q_offset, not the soft midpoint
    sac_joint_pos_offset = (0.0, 0.0)  # rad, in joint_names order (thigh, calf)
    joint_physical_limit_thigh = (-90.0, 90.0)  # deg; actual PhysX stops
    joint_physical_limit_calf = (-160.0, 160.0)
    joint_soft_limit_thigh = (-75.0, 75.0)  # deg; SAC target endpoints, independent of physical stops
    joint_soft_limit_calf = (-200.0, 200.0)  # intentionally outside the physical calf stops

    # Replace the original nominal 100 g tip ballast, rather than adding 100 g on top.
    shaft_mass_kg = 0.015
    paddle_mass_kg = 0.100
    paddle_diameter_m = 0.16  # round ping-pong head; broad face is local Y-Z
    paddle_thickness_m = 0.012  # along local X (face normal)
    paddle_handle_length_m = 0.06  # short grip toward the wrist along local +Y; clears the upper arm at neutral
    paddle_handle_width_m = 0.026  # along local Z; same thickness as the head
    paddle_offset_ee = (0.0, -0.16, 0.029)  # 5 cm standoff along the arm-plane normal (+EE z)
    paddle_mount_delta_m = 0.07  # positive extends the whole racket along second-link -Y (toward the tip)
    racket_parallel_to_arm_plane = True  # False twists the whole racket 90 deg about its handle; head centre stays fixed
    paddle_quat_ee = (0.7071067811865476, 0.0, -0.7071067811865476, 0.0)  # local +x normal -> EE +z
    # Neutral FK from the USD joint frames, not the file's authored preview pose.
    keeper_plane_x = -0.0303288
    target_center_yz = (-0.0647006, 1.1292502)  # neutral paddle at the reference home height of 1.5 m

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
    launch_y_noise = 0.15 # y_launch sampled around y_des with a uniform noise pm this.
    shot_half_extents_yz = ((0.045, 0.045), (0.20, 0.12), (0.40, 0.22), (0.65, 0.32), (0.8,0.6), (1.0,1.0))
    shot_flight_time_ranges_s = ((0.42, 0.50), (0.35, 0.50), (0.30, 0.46), (0.25, 0.42), (0.25, 0.50), (0.25, 0.6))
    shot_level = 0
    curriculum = True
    curriculum_window_shots = 2048
    curriculum_min_shots = 1024
    curriculum_min_steps = 3000  # 30 s at 100 Hz (50 s at 60 Hz), per environment, not aggregate transitions
    curriculum_success_threshold = 0.7

    rotor_thrust_limit_n = 10.0  # hardware-profile ceiling per rotor; used when thrust clipping is enabled
    thrust_to_weight = 3.5  # further limits thrust; resolved against the 10 N ceiling at env creation
    rotor_action_scale = 1.0  # direct modes: hover_per_rotor * (1 + scale * action)
    action_clamp=10.0  # PPO rotor/reference clamp when remove_thrust_clipping=False; never direct joint torques
    remove_action_clipping=True  # legacy unused field; remove_thrust_clipping controls clipping
    remove_thrust_clipping=False # False: cap thrust at min(TWR * nominal hover, rotor_thrust_limit_n)
    collective_action_scale = 0.7
    body_rate_scale = (2.0, 2.0, 1.5)  # PI mode and INDI SAC only; INDI PPO is already rad/s
    rate_kp = (0.35, 0.35, 0.15)
    rate_ki = (0.10, 0.10, 0.04)
    motor_time_constant_s = 0.04  # motor lag; 0 disables filtering (instantaneous thrust commands)
    # Policy-command latency [s], in whole physics steps (5 ms at 200 Hz). The old command reaches
    # INDI/PI/PD/PID and the motors; those loops still use current measurements. See README.
    actuator_delay = 0.0  # fixed for every episode, e.g. env.actuator_delay=0.02 at evaluation
    actuator_delay_range = None  # (min, max) [s]: a new uniform delay per environment at every robot reset
    critic_obs_includes_rotor_omega = False  # append six current post-lag rotor speed magnitudes (rad/s), critic only
    # Eagle/PX4 motor order, body-frame metres, 0.185 m radius; spin is the body-yaw torque sign.
    # Hardware profile: hidro-iri/eagle_mpc_lib@96c65ab, yaml/borinot_flying_arm_2/platform/borinot.yaml.
    rotor_pos_b = ((0.0, -0.185, 0.0), (0.0, 0.185, 0.0),
                   (0.1602147, 0.0925, 0.0), (-0.1602147, -0.0925, 0.0),
                   (0.1602147, -0.0925, 0.0), (-0.1602147, 0.0925, 0.0))
    rotor_spin = (1.0, -1.0, 1.0, -1.0, -1.0, 1.0)
    kf = 3.847e-6  # N / (rad/s)^2; source identifies the ratio, not independent absolute fits
    km = 5.361e-8  # N*m / (rad/s)^2
    p_el_motor_efficiency = 0.95
    km_over_kf = km / kf

    contact_force_threshold_n = 0.05
    clear_distance_m = 0.25
    clear_speed_m_s = 0.2
    recovery_dwell_s = 0.25
    recovery_position_tolerance_m = 0.35
    recovery_speed_m_s = 0.5
    recovery_angular_speed_rad_s = 0.8
    recovery_tilt_rad = 0.35
    recovery_joint_tolerance_rad = 0.5
    require_recovery_for_save = False
    goal_area_x_m = (-0.08, 0.9)
    goal_area_y_m = (-1.2, 1.2)
    goal_area_z_m = (0.07, 2.6)
    reset_xy_noise_m = 0.015
    reset_z_noise_m = 0.015
    # Before this zero-based shot level, spawn at robot_home. Afterwards sample
    # uniformly from the Cartesian product of these world Y/Z offsets from home.
    reset_grid_enabled = True
    reset_grid_start_level = 2
    reset_grid_y_offsets_m = (-0.8, 0.0, 0.8)
    reset_grid_z_offsets_m = (-0.65, 0.0, 0.65)

    # Every reward adds rwd_scale_* * its signal: negative scales penalize, positive scales reward.
    # Rate terms use policy dt; events are paid once; mean-power energy uses no extra dt.
    # Main reward: Jordi's goal-parallel signed distance progress and shot outcomes.
    rwd_scale_progress = 0.5  # reward per metre of reduced yz distance
    progress_use_prediction = False  # optional: use predicted intercept rather than current ball yz
    rwd_scale_save = 1.0
    rwd_scale_goal = -1.0  # ball crosses/hits the goal
    rwd_scale_ball_base_collision = -1.0  # ball hits the base; independent of the goal
    rwd_scale_ball_arm_collision = -0.25
    rwd_scale_leave_goal_area = -1.0
    rwd_scale_ball_floor = -0.25  # live floor event or a deflected ball landing in front of goal
    rwd_scale_miss = -1.0 # Ball is not cleared and goes outside the region, might handle weird racket contacts
    rwd_scale_deadline = -1.0
    rwd_scale_robot_crash = -1.0  # non-ball contact on robot_crash_body_names or excessive tilt; independent of reset flags
    reset_robot_on_robot_crash = True
    self_collision_triggers_reset = True  # False: keep self-contact physics/penalties, but reset only for scene contact
    reset_robot_on_tilt_threshold = True
    robot_crash_body_names = ("base",)  # monitored crash bodies; reset behavior is controlled by the switches above
    robot_crash_force_threshold_n = 0.1
    robot_tilt_threshold_degs = 70.0  # degrees from world vertical; converted internally for the cosine test

    # Optional additions: disabled by default. Enable explicitly for an ablation.
    rwd_scale_intercept = 0.0
    rwd_scale_alignment = 0.0
    rwd_scale_recovery = 0.0
    intercept_sigma_m = 0.30
    rwd_scale_hit = 0.0
    rwd_scale_centered_hit = 0.0
    # Group-mean squared command changes, normalized by full thrust/torque spans; times policy dt.
    rwd_scale_action_rate = -0.0
    rwd_scale_body_angular_velocity = -0.0
    rwd_scale_body_linear_velocity = -0.0
    rwd_scale_thrust = -0.0
    # Add rwd_scale_energy * mean(mechanical + dissipative power) [W].
    # Negative scales penalize consumption; no additional dt multiplier.
    rwd_scale_energy = -0.0001
    # Physical soft limits: signed scale * sum((excess / upper limit)**2), no extra dt.
    # Controller costs average the physics substeps in each policy step.
    rwd_scale_thrust_limits = -1.0  # relative rotor-demand excess; raw policy or unconstrained controller
    rwd_scale_torque_limits = -1.0  # relative joint-demand excess, both signs, before torque saturation
    rwd_scale_action_limits = -1.0  # otherwise-clipped PD/PI/SAC-INDI references; not PPO INDI
    # Nominal Solo joint copper loss: c * joint_torque**2 [W], c in W/(N*m)^2.
    # Rounded from (3/2)*R_phase/(G*Kt)^2, R_phase=0.2324751 ohm, G=9, Kt=0.025 N*m/A.
    arm_copper_loss_coefficient = 6.888

    # Actor-only input switches. The critic always receives these values.
    obs_actor_includes_time_ball_collides_plane=False
    obs_actor_includes_ball_plane_intersection_point=False
    obs_actor_includes_phase=False
    obs_actor_includes_ball_params=True  # sampled mass (kg) and restitution

    def __post_init__(self):
        self.configure_control()

    def configure_control(self):
        """Refresh derived settings before simulator/space creation, including after Hydra."""
        self.robot.spawn.articulation_props.enabled_self_collisions = self.enabled_self_collisions
        modes = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control", "PI_rotors_PD_joint_pos_control", "indi")
        if self.control_mode not in modes:
            raise ValueError(f"Unknown control_mode {self.control_mode!r}; choose one of {modes}.")
        if self.physics_hz <= 0 or self.policy_hz <= 0:
            raise ValueError("physics_hz and policy_hz must be positive.")
        ratio = self.physics_hz / self.policy_hz
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("physics_hz must be an integer multiple of policy_hz.")
        if self.effort_lim_sim <= 0 or self.velocity_lim_sim <= 0:
            raise ValueError("effort_lim_sim and velocity_lim_sim must be positive.")
        if not math.isfinite(self.thrust_to_weight) or self.thrust_to_weight <= 0:
            raise ValueError("thrust_to_weight must be finite and positive.")
        if not math.isfinite(self.rotor_thrust_limit_n) or self.rotor_thrust_limit_n <= 0:
            raise ValueError("rotor_thrust_limit_n must be finite and positive.")
        if self.control_mode == "indi":
            self.indi.validate(self)
        if self.sac_action_mapping:
            if self.control_mode not in (*modes[:2], "indi"):
                raise ValueError("sac_action_mapping requires direct thrust control (torque/PD position) or indi.")
            if not math.isfinite(self.thrust_to_weight) or self.thrust_to_weight <= 1:
                raise ValueError("SAC thrust_to_weight must be finite and greater than 1 (hover).")
            if len(self.joint_target_scale) != 2 or any(not math.isfinite(s) or s <= 0 for s in self.joint_target_scale):
                raise ValueError("SAC joint_target_scale must contain two finite positive values.")
            if len(self.sac_joint_pos_offset) != 2:
                raise ValueError("sac_joint_pos_offset must contain two values in joint_names order.")
            for i, name in enumerate(("thigh", "calf")):
                for kind in ("physical", "soft"):
                    bounds = getattr(self, f"joint_{kind}_limit_{name}")
                    if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
                        raise ValueError(f"joint_{kind}_limit_{name} must be two ordered finite degrees.")
                    if not math.radians(bounds[0]) < self.sac_joint_pos_offset[i] < math.radians(bounds[1]):
                        raise ValueError(f"sac_joint_pos_offset must lie inside the {kind} {name} limits.")
        self.decimation = round(ratio)
        self.sim.dt = 1 / self.physics_hz
        self.sim.render_interval = self.decimation
        self.action_space = 9 if self.control_mode == "indi" else (6 if self.control_mode == modes[2] else 8)
        controller_state = 8 if self.control_mode == "indi" else (9 if self.control_mode == modes[2] else 0)
        self.state_space = 48 + self.action_space + controller_state + 6 * self.critic_obs_includes_rotor_omega
        point, time = self.obs_actor_includes_ball_plane_intersection_point, self.obs_actor_includes_time_ball_collides_plane
        # Optional actor inputs: plane point (3), time (1), validity (1), phase (7), ball parameters (2).
        self.observation_space = (34 + self.action_space + controller_state + 3 * point + time + (point or time)
                                 + 7 * self.obs_actor_includes_phase + 2 * self.obs_actor_includes_ball_params)
        actuator = self.robot.actuators["arm"]
        use_pd = self.control_mode != "indi" and (self.control_mode != modes[0] or self.hold_arm)
        actuator.stiffness = self.kp if use_pd else 0.0
        actuator.damping = self.kd if use_pd else 0.0
        actuator.effort_limit_sim = self.effort_lim_sim
        actuator.velocity_limit_sim = self.velocity_lim_sim
        # Keep only the non-deprecated limits, including when reusing a config instance.
        actuator.effort_limit = actuator.velocity_limit = None
