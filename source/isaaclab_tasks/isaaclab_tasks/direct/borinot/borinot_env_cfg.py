# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuator, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from pathlib import Path
import numpy as np 

@configclass
class BorinotEnvCfg(DirectRLEnvCfg):

    # control every N physics steps
    decimation = 2 # -> dt_control: 60Hz
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)

    episode_len: int = 256
    dt_control: float = sim.dt * decimation
    episode_length_s: float = float(episode_len) * float(dt_control)


    des_joint_out = True
    enabled_self_collisions=True

    action_space = 8 if not des_joint_out else 10  
    # 6 thrusts; 4 angles; 4 kp,kd 
    observation_space = 32 if not des_joint_out else 34 # incl. prev action
    state_space = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=True,
    )


    usd_path = str(Path(__file__).resolve().parents[5] / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd")
    ground_usd_path = str(
        Path(__file__).resolve().parents[5]
        / "borinotIsaacLab/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd"
    )
    static_friction = 1.2
    dynamic_friction = 1.0
    torque_lim = 2.7
    joint_load_limit = 2.8
    # contact force computed assuming an angle of 75 degs between leg and the ground, max torque of 2.7Nm and feet to joint len == 160mm
    max_contact_force: float = float(2.7 / (16e-2 * np.cos(np.deg2rad(75))))
    contact_force_reset_threshold: float = 150
    min_contatact_force_penalty:float = 2.5
    max_contact_force_penalty: float = 10.0
    rwd_w_contact_excess_torque = 0.0
    reset_above_force_threshold: bool = True
    reset_on_base_contact: bool = True
    terminate_on_leg_contact: bool = True
    include_force_info_in_obs: bool = True
    only_feet_contact_penalized: bool = False
    lineal_cf_penalty: bool = True
    lineal_cf_slope: float = 1/50
    mult_energy_cost = 11.0
    rwd_w_energy_base: float = 0.045
    rwd_w_energy = rwd_w_energy_base
    reset_on_overload: bool = False

    rwd_w_action_delta: float = 0.001
    rwd_w_delta_contact_force: float = 1e-8
    rwd_w_ddq: float = 1e-6
    rwd_w_dq: float = 0.0 # Not limited for now. 
    
    contact_force_threshold: float = 1.0
    e_m_cf = 4.0e-6
    e_m_t = 3.0e-4
    e_t_sigma = 15.0

    init_pos = (0.0, 0.0, 0.50)

    joint_damping=0.0
    joint_stiffness=0.0
    velocity_limit=None
    effort_limit=None
    if des_joint_out:
        joint_stiffness = 35.0
        joint_damping = 0.6
        effort_limit = torque_lim
        velocity_limit = 100.0

    # angle control 


    robot_cfg: ArticulationCfg = ArticulationCfg(
        # prim_path="/World/envs/env_.*/Robot",
        prim_path="/World/envs/env_.*/Robot",
        # articulation_root_prim_path="/borinot_base/borinot_base"
        articulation_root_prim_path=None,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
             articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=enabled_self_collisions), # Defined in .usd
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            # rot=(0.49621, -0.23004, 0.78314, -0.29589),   
            rot=(1, 0, 0, 0),   
            joint_pos={
            "flying_arm_2__j_link1_link2": 0.0,
            # optionally also set the other joint
            "flying_arm_2__j_bl_link1": 0.0,
        },
        ),
        actuators={
            "leg_torque": ImplicitActuatorCfg(
                joint_names_expr=[
                    "flying_arm_2__j_bl_link1",
                    "flying_arm_2__j_link1_link2",
                ],
                stiffness=joint_stiffness,
                damping=joint_damping,
                effort_limit=effort_limit,
                velocity_limit=velocity_limit,
            ),
        },
    )

    base_body_name: str = "base"
    ee_body_name: str = "flying_arm_2__link2"
    leg_joint_names: Tuple[str, str] = (
        "flying_arm_2__j_bl_link1",
        "flying_arm_2__j_link1_link2",
    )

    # Bodies considered "undesired" if they contact the floor (used for reward gating / reset logic)
    undesired_contact_body_names: Tuple[str, ...] = (
        "base",
        "flying_arm_2__link1"
    )

    reset_z_noise: float = 0.0
    reset_joint_noise: float = 0.0
    reset_xy_noise: float = 0.00

    terminate_z_min: float = 0.08

    kf: float = 4.138394792004922e-06
    km: float = 6.991478005829954e-08
    p_el_motor_efficiency: float = 0.95
    rotor_spin: Tuple[float, ...] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

    thrust_mode: str = "delta_hover"  

    thrust_min: float = 0.0
    thrust_max: float = -1.0                 
    delta_thrust_scale: float = 1.5          
    hover_thrust_per_rotor: float = 2.91 * 9.81 / 6.
    thrust_max_scale = .99

    bias_torque_j2: float = 0.0

    rotor_pos_b: Tuple[Tuple[float, float, float], ...] = (
        (0.20, 0.00, 0.00),
        (0.10, 0.17, 0.00),
        (-0.10, 0.17, 0.00),
        (-0.20, 0.00, 0.00),
        (-0.10, -0.17, 0.00),
        (0.10, -0.17, 0.00),
    )

    torque_min: Tuple[float, float] = (-torque_lim, -torque_lim)
    torque_max: Tuple[float, float] = (torque_lim, torque_lim)


    teacher_learning: bool = True
    obs_include_prev_action: bool = True
    obs_include_ee_vel: bool = False

    train_fixed_cmd: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    is_fixed_cmd: bool = True
    # eval_cmds = [[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[-1,0,0],[0,-1,0],[0,0,2],[.5,0,-1]]
    cmd_period_s: float = 10.0


    cmd_train_period_s: float = 10.0
    cmd_eval_period_s: float = 10.0

    cmd_vx_min: float = -1.0
    cmd_vx_max: float = 1.5
    cmd_vy_min: float = -1.0
    cmd_vy_max: float = 1.0
    cmd_wz_min: float = -2.0
    cmd_wz_max: float = 2.0

    cmd_v_cutoff: float = 0.10
    cmd_wz_cutoff: float = 0.26


    rwd_w_task: float = 1.0
    rwd_vel_sigma: float = 1.0
    rwd_wz_sigma: float = 1.0
    rwd_vel_abs: int = 0
    rwd_airbone_only: int = 0

    rwd_w_collide_penalty: float = 10.0
    rwd_joint_torque_above_lim: float = 1.5
    rwd_contact_force_above_lim: float = 2.
    rwd_w_penalize_omega: float = 0.005

    rwd_w_energy_thrust: float = 2e-2
    rwd_w_energy_torque: float = 2e-2
    rwd_incl_true_energy: int = 1
    rwd_incl_small_actions: int = 0
    p_el_motor_efficiency: float = 0.95

    y_deviation_lim: float = 999.

    

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        # prim_path="/World/envs/env_.*/Robot/borinot_base/.*",
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=1,
        track_air_time=False,
    )
