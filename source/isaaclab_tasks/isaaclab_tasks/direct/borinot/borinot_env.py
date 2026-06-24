# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .borinot_env_cfg import BorinotEnvCfg


# -----------------------
# Quaternion helpers
# -----------------------
def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    # q = [w,x,y,z]
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Hamilton product, q = q1 ⊗ q2, both [w,x,y,z]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Rotate v by q: v' = q ⊗ [0,v] ⊗ q*
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., 1:]


def _quat_inv_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Rotate v by q^{-1} (i.e., into local/body frame if q is body->world)
    return _quat_apply(_quat_conj(q), v)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    # q: (...,4) in [w,x,y,z]
    w, x, y, z = q.unbind(-1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    return torch.stack(
        (
            torch.stack((r00, r01, r02), dim=-1),
            torch.stack((r10, r11, r12), dim=-1),
            torch.stack((r20, r21, r22), dim=-1),
        ),
        dim=-2,
    )


class BorinotEnv(DirectRLEnv):
    """Command-conditioned Borinot task (DirectRLEnv).

    Key features implemented:
      - Rotor wrench application (6 rotors) + 2 leg joint torques
      - Command schedule (vx, vy, wz) updated every cmd_period_s
      - MuJoCo-like observation (z, R, body-frame v/omega, q/dq, prev_action, cmd)
      - Reward: velocity/yaw tracking + action/energy/omega penalties + collision penalty
      - Termination: timeout, undesired contacts, z crash, optional y deviation
    """

    cfg: BorinotEnvCfg
    
    def __init__(self, cfg: BorinotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_dof_ids, _ = self.robot.find_joints(list(self.cfg.leg_joint_names))
        self._base_body_id, _ = self.robot.find_bodies(self.cfg.base_body_name)
        self._ee_body_id, _ = self.robot.find_bodies(self.cfg.ee_body_name)

        self.cfg.rwd_w_energy = self.cfg.rwd_w_energy_base * self.cfg.mult_energy_cost

        self._dt_physics = float(self.cfg.sim.dt)
        self._dt_control = self._dt_physics * self.cfg.decimation

        self._rotor_pos_b = torch.tensor(self.cfg.rotor_pos_b, dtype=torch.float32, device=self.device)  # (6,3)
        self._rotor_spin = torch.tensor(self.cfg.rotor_spin, dtype=torch.float32, device=self.device)  # (6,)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        self._kf = self.cfg.kf
        self._km = self.cfg.km
        self._km_over_kf = float(self._km / self._kf) if self._kf > 0 else 0.0


        act_dim = self.cfg.action_space
        self.actions = torch.zeros((self.num_envs, act_dim), dtype=torch.float32, device=self.device)
        self._last_actions = torch.zeros_like(self.actions)
        self._last_contact_f = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self._thrusts = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._leg_torques = torch.zeros((self.num_envs, len(self._leg_dof_ids)), dtype=torch.float32, device=self.device)

        # For ddq finite-difference 
        self._joint_vel_prev = torch.zeros((self.num_envs, len(self._leg_dof_ids)), dtype=torch.float32, device=self.device)
        self._joint_acc = torch.zeros_like(self._joint_vel_prev)


        cmd_period_s = self.cfg.cmd_period_s
        self._cmd_period_steps = max(1, int(round(cmd_period_s / self._dt_control)))

        self._cmd = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)       # current cmd (for obs next)
        self._cmd_step = torch.zeros_like(self._cmd)  # cmd used for the current transition reward
        self._cmd_steps_left = torch.full((self.num_envs,), self._cmd_period_steps, dtype=torch.int64, device=self.device)

        # Resolve contact-sensor body indices (indices are in sensor-body order)
        self._resolve_contact_body_ids()

        # Initialize commands
        self._reset_commands(torch.arange(self.num_envs, device=self.device))

        # in __init__
        self._step_max_leg_joint_load = torch.zeros((self.num_envs, len(self._leg_dof_ids)), device=self.device)
        self._step_max_contact_force = torch.zeros((self.num_envs,), device=self.device)
        self._joint_torque_overload = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._contact_force_overload = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)


    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane - raises error
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # local_usd = Path(self.cfg.ground_usd_path).resolve()

        # spawn_ground_plane(
        #     prim_path="/World/ground",
        #     cfg=GroundPlaneCfg(usd_path=local_usd.as_uri()),  # file:///...
        # )


        # Spawn a grid ground plane (high contrast)
        spawn_ground_plane(
            prim_path="/World/defaultGroundPlane",
            # prim_path="/World/ground"
            cfg=GroundPlaneCfg(
                usd_path=self.cfg.ground_usd_path,
                color=(.2, .2, .2), 
                size=(200.0, 200.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=self.cfg.static_friction,          # grip when nearly not sliding
                dynamic_friction=self.cfg.dynamic_friction,         # grip while sliding
                restitution=0.0,
                friction_combine_mode="max",   # or "multiply", "min", "max"
        ),
            ),
        )


        # Clone envs
        self.scene.clone_environments(copy_from_source=False)

        # CPU collision filtering (per IsaacLab templates)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register robot
        self.scene.articulations["robot"] = self.robot

        # Contact sensor view (net forces on robot links)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact"] = self._contact_sensor

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # import isaaclab.sim as sim_utils

        # # Make the environment/background black
        # dome_cfg = sim_utils.DomeLightCfg(intensity=0.0, color=(0.0, 0.0, 0.0))
        # dome_cfg.func("/World/Light", dome_cfg)

        # # Optional: add a key light so the robot is still visible
        # key_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        # key_cfg.func("/World/KeyLight", key_cfg)

    def _resolve_contact_body_ids(self):
        """Resolve contact sensor indices from body-name patterns.

        Important: ContactSensor data indices follow the sensor's internal body order,
        so use contact_sensor.find_bodies(...) rather than robot.find_bodies(...).
        """
        undesired_patterns = self.cfg.undesired_contact_body_names
        foot_patterns = [self.cfg.ee_body_name]
        base_patterns = [self.cfg.base_body_name]
    
        self._undesired_contact_ids = None
        self._foot_contact_ids = None
        self._base_contact_ids = None

        if len(undesired_patterns) > 0:
            ids, _ = self._contact_sensor.find_bodies(undesired_patterns)
            self._undesired_contact_ids = ids

        if len(foot_patterns) > 0:
            ids, _ = self._contact_sensor.find_bodies(foot_patterns)
            self._foot_contact_ids = ids

        if len(base_patterns) > 0:
            ids, _ = self._contact_sensor.find_bodies(base_patterns)
            self._base_contact_ids = ids


    # -----------------------
    # Commands
    # -----------------------
    def _sample_commands(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample commands in BODY frame: (vx, vy, wz)."""
        vx_min = self.cfg.cmd_vx_min
        vx_max = self.cfg.cmd_vx_max
        vy_min = self.cfg.cmd_vy_min
        vy_max = self.cfg.cmd_vy_max
        wz_min = self.cfg.cmd_wz_min
        wz_max = self.cfg.cmd_wz_max
        v_dead = self.cfg.cmd_v_cutoff
        w_dead = self.cfg.cmd_wz_cutoff

        n = env_ids.numel()
        vx = torch.empty((n,), device=self.device).uniform_(vx_min, vx_max)
        vy = torch.empty((n,), device=self.device).uniform_(vy_min, vy_max)
        wz = torch.empty((n,), device=self.device).uniform_(wz_min, wz_max)

        if v_dead > 0:
            vx = torch.where(vx.abs() < v_dead, torch.zeros_like(vx), vx)
            vy = torch.where(vy.abs() < v_dead, torch.zeros_like(vy), vy)
        if w_dead > 0:
            wz = torch.where(wz.abs() < w_dead, torch.zeros_like(wz), wz)

        return torch.stack([vx, vy, wz], dim=-1)

    def _reset_commands(self, env_ids: torch.Tensor):
        is_fixed = bool(self.cfg.is_fixed_cmd)

        # Optional per-env fixed commands (play/debug)
        eval_cmds = getattr(self.cfg, "eval_cmds", ())
        if is_fixed and eval_cmds is not None and len(eval_cmds) > 0:
            cmds = torch.tensor(eval_cmds, device=self.device, dtype=torch.float32)  # (M, 3)
            m = cmds.shape[0]
            idx = env_ids % m
            self._cmd[env_ids] = cmds[idx]
        elif is_fixed:
            fx, fy, fw = map(float, self.cfg.train_fixed_cmd)
            self._cmd[env_ids, 0] = fx
            self._cmd[env_ids, 1] = fy
            self._cmd[env_ids, 2] = fw
        else:
            self._cmd[env_ids] = self._sample_commands(env_ids)

        self._cmd_steps_left[env_ids] = self._cmd_period_steps



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # _pre_physics_step is called ONCE per env step before stepping physics, then _map_actions is called each demination step, finally _get_rewards/_get_dones is called once at the end. 
        self.actions = actions.clamp(-1.0, 1.0)
        self._cmd_step.copy_(self._cmd)

        # reset buffers for joint contact overload
        self._step_max_leg_joint_load.zero_()
        self._joint_torque_overload.zero_()
        self._step_max_contact_force.zero_()
        self._contact_force_overload.zero_()


    def _map_actions_angle_des(self) -> torch.Tensor:
        a_th = self.actions[:, :6].clip(-1.0, 1.0)
        theta_des = self.actions[:, 6:10]
        r1 = theta_des[:, :2]
        r1 = r1 / torch.linalg.norm(r1, dim=-1, keepdim=True).clamp_min(1e-8) 
        qdes1 = torch.atan2(r1[:, 1], r1[:, 0])
        r2 = theta_des[:, 2:4]
        r2 = r2 / torch.linalg.norm(r2, dim=-1, keepdim=True).clamp_min(1e-8)
        qdes2 = torch.atan2(r2[:, 1], r2[:, 0])
        # deltaKp = self.actions[:, 10:12] 
        # deltaKd = self.actions[:, 12:14] 
        # # deltaKp = torch.exp(logKp).clamp(min=5.0, max=35.0)   # or max=60.0
        # # deltaKd = torch.exp(logKd).clamp(min=0.2, max=2.0)    # or max=5.0
        # Kp0 = 25.0
        # Kd0 = 0.6
        # sp = 0.5
        # sd = 0.5
        # Kp = (Kp0 * torch.exp(sp * deltaKp)).clamp(min=5.0, max=35.0)
        # Kd = (Kd0 * torch.exp(sd * deltaKd)).clamp(min=0.2, max=2.0)
        q_des = torch.stack((qdes1, qdes2), dim=-1)

        self.robot.set_joint_position_target(q_des.float(), joint_ids=self._leg_dof_ids)
        dq_des = torch.zeros_like(q_des)
        self.robot.set_joint_velocity_target(dq_des.float(), joint_ids=self._leg_dof_ids)

        thrusts = self._map_thrust(a_th) 
        

        return thrusts

    def _map_thrust(self, a_th): 
        thrust_mode = self.cfg.thrust_mode
        hover = self.cfg.hover_thrust_per_rotor

        thrust_min = self.cfg.thrust_min
        thrust_max = self.cfg.thrust_max

        # Define clamps
        thrust_max_scale = self.cfg.thrust_max_scale
        # If cfg.thrust_max not set, derive from hover + scale
        thrust_max = thrust_max_scale * hover if thrust_max <= 0.0 else thrust_max

        if thrust_mode == "absolute":
            # T = Tmin + (a+1)/2 * (Tmax - Tmin)
            thrusts = thrust_min + (a_th + 1.0) * 0.5 * (thrust_max - thrust_min)
        else:
            # delta-hover: T = clamp(Thover + a * (scale*Thover), Tmin, Tmax)
            delta_thrust_scale = self.cfg.delta_thrust_scale
            delta_mag = delta_thrust_scale * hover
            thrusts = (hover + a_th * delta_mag).clamp(thrust_min, thrust_max)

        return thrusts

    def _map_actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps actions in [-1,1] to:
          thrusts: (N,6) in Newtons
          leg_torques: (N,2) in Nm
        """
        # Ensure actions in -1,1
        self.actions = self.actions.clamp(-1.0, 1.0)
        a_th = self.actions[:, :6]
        a_tau = self.actions[:, 6:]

        thrusts = self._map_thrust(a_th)

        # Leg torques
        tmin = torch.tensor(self.cfg.torque_min, device=self.device, dtype=torch.float32)
        tmax = torch.tensor(self.cfg.torque_max, device=self.device, dtype=torch.float32)
        leg_torques = tmin + (a_tau + 1.0) * 0.5 * (tmax - tmin)
        leg_torques[:,-1] += self.cfg.bias_torque_j2
        leg_torques.clamp_(-self.cfg.torque_lim, +self.cfg.torque_lim)
        return thrusts, leg_torques

    def _apply_action(self) -> None:
        thrusts, leg_torques = self._map_actions() if not self.cfg.des_joint_out else (self._map_actions_angle_des(), None)


        # this clips the joint torque by effort_limit before applying
        if not self.cfg.des_joint_out:
            self.robot.set_joint_effort_target(leg_torques, joint_ids=self._leg_dof_ids)

        # Rotor wrench in BASE (body local) frame
        F_i_b = thrusts.unsqueeze(-1) * self._z_axis.view(1, 1, 3)  # (N,6,3)
        r_b = self._rotor_pos_b.view(1, 6, 3).expand(self.num_envs, -1, -1)
        tau_lever_b = torch.cross(r_b, F_i_b, dim=-1).sum(dim=1)  # sum over thrust dim (N,3)

        # Reaction/drag torque about +Z (base frame)
        tau_drag_z = self._km_over_kf * (thrusts * self._rotor_spin.view(1, 6)).sum(dim=1)  # (N,)
        tau_drag_b = tau_drag_z.unsqueeze(-1) * self._z_axis.view(1, 3)  # (N,3)

        F_sum_b = F_i_b.sum(dim=1)              # (N,3)
        tau_sum_b = tau_lever_b + tau_drag_b    # (N,3)

        # Apply wrench on base link.
        # Use LOCAL frame (is_global=False) since F_sum_b/tau_sum_b are in base frame.
        # self.robot.set_external_force_and_torque(
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=F_sum_b.unsqueeze(1),          # (N,1,3)
            torques=tau_sum_b.unsqueeze(1),        # (N,1,3)
            body_ids=[int(self._base_body_id[0])],
            is_global=False,
        )

        self._update_j_force_overload()
        self._each_decimation_calls()
        # sends stuff to sim, inclduing the PD angle targets 
        self.robot.write_data_to_sim()

        # applied_torque: approximation of the torque applied in the physics sim (pd controller)
        self.leg_torques = self.robot.data.applied_torque[:, self._leg_dof_ids] if self.cfg.des_joint_out else leg_torques
        self._thrusts = thrusts


    def _update_j_force_overload(self): 
        dof_load = self.robot.root_physx_view.get_dof_projected_joint_forces() 
        contact_forces_w = self._contact_sensor.data.net_forces_w  # (N,B,3)
        contact_forces_w_norm = torch.linalg.norm(contact_forces_w, dim=-1) 
        feet_contact_forces = contact_forces_w_norm[:, self._foot_contact_ids].squeeze() if self.cfg.only_feet_contact_penalized else contact_forces_w_norm.max(dim=1).values
        leg_load = dof_load[:, self._leg_dof_ids].abs() 
        self._step_max_leg_joint_load = torch.maximum(self._step_max_leg_joint_load, leg_load)
        self._step_max_contact_force = torch.maximum(self._step_max_contact_force, feet_contact_forces)

        self._joint_torque_overload = (self._step_max_leg_joint_load > self.cfg.joint_load_limit).any(dim=-1)
        self._contact_force_overload = self._step_max_contact_force > self.cfg.max_contact_force
        self.above_force_reset_threshold = self._step_max_contact_force > self.cfg.contact_force_reset_threshold
    
    def _each_decimation_calls(self):
        """Used for ddq estimate - triggers inside _apply_action"""
        # ddq estimate for legs
        dq = self.robot.data.joint_vel[:, self._leg_dof_ids]
        self._joint_acc = (dq - self._joint_vel_prev) / self._dt_physics
        self._joint_vel_prev = dq

    def _scale_linear(self, a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Scale a tensor from [-1,1] to [lo,hi]."""
        return lo + .5 * (a + 1) * (hi - lo)
        


    def _get_observations(self) -> dict:
        # Positions and orientation
        base_pos_w = self.robot.data.body_pos_w[:, self._base_body_id[0], :]
        base_quat_w = self.robot.data.body_quat_w[:, self._base_body_id[0], :]  # body->world

        # Use env-origin local position for invariance across clones
        base_pos_l = base_pos_w - self.scene.env_origins
        z = base_pos_l[:, 2:3]

        # Rotation matrix (body->world) flattened
        Rb = _quat_to_rotmat(base_quat_w).reshape(self.num_envs, 9)

        # Body-frame velocities (MuJoCo local twist analogue)
        v_w = self.robot.data.body_lin_vel_w[:, self._base_body_id[0], :]
        w_w = self.robot.data.body_ang_vel_w[:, self._base_body_id[0], :]
        v_b = _quat_inv_apply(base_quat_w, v_w)
        w_b = _quat_inv_apply(base_quat_w, w_w)

        # Joint states (legs)
        q = self.robot.data.joint_pos[:, self._leg_dof_ids]
        dq = self.robot.data.joint_vel[:, self._leg_dof_ids]

        # Compose obs
        include_prev_action = bool(self.cfg.obs_include_prev_action)
        parts = [z, Rb, v_b, w_b, q, dq]

        if self.cfg.include_force_info_in_obs:
            parts.append(self._step_max_contact_force.unsqueeze(-1))

        if include_prev_action:
            parts.append(self._last_actions)

        # Always include command (so you don't need trainer-side appending)
        parts.append(self._cmd)

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    def _contact_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (undesired_collision, foot_contact) boolean tensors of shape (N,)."""
        thr = self.cfg.contact_force_threshold
        forces_w = self._contact_sensor.data.net_forces_w  # (N,B,3)
        norms = torch.linalg.norm(forces_w, dim=-1)        # (N,B)

        undesired = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        base_contact = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        feet = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if self._undesired_contact_ids is not None and len(self._undesired_contact_ids) > 0:
            undesired = torch.any(norms[:, self._undesired_contact_ids] > thr, dim=1)

        if self._base_contact_ids is not None and len(self._base_contact_ids) > 0:
            base_contact = torch.any(norms[:, self._base_contact_ids] > thr, dim=1)

        if self._foot_contact_ids is not None and len(self._foot_contact_ids) > 0:
            feet = torch.any(norms[:, self._foot_contact_ids] > thr, dim=1)

        other_contact = feet if self.cfg.rwd_airbone_only else undesired
        
        return base_contact, other_contact

    def _get_rewards(self) -> torch.Tensor:

        def func_vel_err(err: torch.Tensor, sigma):
            absn = torch.linalg.norm(err, dim=-1)  # shape: (...)
            delta = torch.where(absn <= 1.0, absn.square(), absn)  # shape: (...)
            return torch.exp(-.5 * delta / sigma**2)

        # Body-frame velocities (for tracking)
        base_quat_w = self.robot.data.body_quat_w[:, self._base_body_id[0], :]
        v_w = self.robot.data.body_lin_vel_w[:, self._base_body_id[0], :]
        w_w = self.robot.data.body_ang_vel_w[:, self._base_body_id[0], :]
        v_b = _quat_inv_apply(base_quat_w, v_w)
        w_b = _quat_inv_apply(base_quat_w, w_w)

        # Command used for THIS transition
        cmd = self._cmd_step  # (N,3) = (vx, vy, wz) in body frame

        # Tracking rewards (Gaussian)
        v_err = v_b[:, 0:2] - cmd[:, 0:2]
        # v_err2 = torch.sum((v_b[:, 0:2] - cmd[:, 0:2]) ** 2, dim=1)
        w_err2 = (w_b[:, 2] - cmd[:, 2]) ** 2

        # r_v = torch.exp(-.5 * v_err2 / (self.cfg.rwd_vel_sigma**2 + 1e-8))
        r_v = func_vel_err(v_err, self.cfg.rwd_vel_sigma)
        r_w = torch.exp(-.5 * w_err2 / (self.cfg.rwd_wz_sigma**2 + 1e-8))
        r_task = r_v + r_w

        # Contact flags
        base_collision, other_contact = self.is_base_collision, self.is_leg_or_base_contact

        r_task = r_task * (~other_contact).float() # zero out reward if undesired contact

        # Penalties

        action_delta = self.actions - self._last_actions
        # I think current weigiht 5e-3 might be too small
        r_delta_action = self.cfg.rwd_w_action_delta * torch.sum(action_delta**2, dim=1)

        contact_force_delta = self._step_max_contact_force - self._last_contact_f
        r_contact_force_delta = self.cfg.rwd_w_delta_contact_force * contact_force_delta.abs()
        self._last_contact_f = self._step_max_contact_force.clone()
    
        excess = (self._step_max_contact_force - self.cfg.max_contact_force).clamp_min(0.0)
        # r_excess_force = torch.maximum(self.cfg.rwd_w_contact_excess_contact_force * excess,
        #                             torch.tensor(self.cfg.rwd_contact_force_above_lim, device=excess.device, dtype=excess.dtype))
        # rwd_contact_force_excess = r_excess_force * (excess > 0).float()
        r_contact_force_excess =  torch.exp(self.cfg.e_m_cf * excess**2) if not self.cfg.lineal_cf_penalty else excess * self.cfg.lineal_cf_slope
        r_contact_force_excess = r_contact_force_excess.clamp(min=self.cfg.min_contatact_force_penalty, 
                                                                              max=self.cfg.max_contact_force_penalty) * (excess > 0).float()

        excess_torque = torch.max(self._step_max_leg_joint_load - self.cfg.joint_load_limit, dim=-1).values 
        excess_torque = excess_torque.clamp_min(0.0)

        r_excess_torque_1 = torch.exp(self.cfg.e_m_t * excess_torque**2) * (excess_torque > 0).float()
        r_excess_torque_2 = torch.clip(1/self.cfg.e_t_sigma * excess_torque, min=0.0, max=1.0) 
        r_excess_torque = self.cfg.rwd_w_contact_excess_torque * torch.where(excess_torque < 20, r_excess_torque_2, r_excess_torque_1).clamp_max(7.)
        # r_excess_torque = self.cfg.rwd_w_contact_excess_torque * excess_torque * (excess_torque > 0).float()

        # Omega XY penalty (body frame roll/pitch rates)
        omega_xy_pen = self.cfg.rwd_w_penalize_omega * torch.sum(w_b[:, 0:2] ** 2, dim=1)

        omega = torch.sqrt(torch.clamp(self._thrusts, min=0.0) / self._kf)
        rotor_power = 1/self.cfg.p_el_motor_efficiency * torch.sum(self._km  * omega**(3/2), dim=1)

        # Leg power ~ sum(|tau * dq|)
        dq = self.robot.data.joint_vel[:, self._leg_dof_ids]
        leg_power = torch.sum(torch.abs(self._leg_torques * dq), dim=1)

        energy = (rotor_power + leg_power) * self._dt_physics
        energy_pen = self.cfg.rwd_w_energy * energy

        # ddq penalty (finite difference)
        ddq_pen = self.cfg.rwd_w_ddq * torch.sum(self._joint_acc**2, dim=1)

        dq_penalty = self.cfg.rwd_w_dq * torch.sum(dq**2, dim=1)

        # Collision penalty
        neg_rwd_indicator = base_collision.float() if not self.cfg.terminate_on_leg_contact else self.is_leg_or_base_contact.float()
        coll_pen = self.cfg.rwd_w_collide_penalty * neg_rwd_indicator.float()
        # reaction_torque_pen = self.cfg.rwd_joint_torque_above_lim * self._joint_torque_overload.float()

        reward = self.cfg.rwd_w_task * r_task - r_delta_action - r_contact_force_delta - energy_pen - omega_xy_pen - dq_penalty - ddq_pen - coll_pen - r_excess_torque  - r_contact_force_excess
        
        self._last_actions = self.actions.clone()
        log = {}
        log["rewards/r_task"] = r_task.mean()
        log['rewards/r_energy'] = energy_pen.mean()
        log["rewards/r_delta_action"] = r_delta_action.mean()
        log["rewards/r_contact_force_excess"] = r_contact_force_excess.mean()
        log["overload/j2_torque_abs_mean"] = self._step_max_leg_joint_load[:, 1].mean()
        log['overload/contact_force_mean'] = self._step_max_contact_force.mean()   
        log['overload/joint_torque_overload_ratio'] = self._joint_torque_overload.float().mean()
        log['overload/contact_force_overload_ratio'] = self._contact_force_overload.float().mean()
        self.extras['log'] = log

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_pos_w = self.robot.data.body_pos_w[:, self._base_body_id[0], :]
        base_pos_l = base_pos_w - self.scene.env_origins

        # z crash termination
        z_bad = base_pos_l[:, 2] < self.cfg.terminate_z_min

        # optional y deviation termination
        y_bad = self.cfg.y_deviation_lim < 5. and torch.abs(base_pos_l[:, 1]) > self.cfg.y_deviation_lim 

        # undesired contact termination
        self.is_base_collision, self.is_leg_or_base_contact = self._contact_flags()

        terminated = z_bad | y_bad | (self.is_base_collision if self.cfg.reset_on_base_contact else False) 
        if self.cfg.terminate_on_leg_contact: 
            terminated = terminated | self.is_leg_or_base_contact
        # print(f'terminated 2 is: {terminated[0]}')
        if self.cfg.reset_on_overload: 
            terminated = terminated | self._joint_torque_overload | self._contact_force_overload
        # print(f'terminated 3 is: {terminated[0]}')
        if self.cfg.reset_above_force_threshold:
            terminated = terminated | self.above_force_reset_threshold
        # print(f'terminated 4 is: {terminated[0]}; time_out is: {time_out[0]}')
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)

        super()._reset_idx(env_ids)

        # Reset root state near default
        root_state = self.robot.data.default_root_state[env_ids_t].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids_t]

        # Optional spawn noise
        pos_noise_xy = float(self.cfg.reset_xy_noise)
        pos_noise_z = float(self.cfg.reset_z_noise)
        if pos_noise_xy > 0:
            noise_xy = pos_noise_xy * (2.0 * torch.rand((len(env_ids_t), 2), device=self.device) - 1.0)
            root_state[:, 0:2] += noise_xy
        if pos_noise_z > 0:
            noise_z = pos_noise_z * (2.0 * torch.rand((len(env_ids_t), 1), device=self.device) - 1.0)
            root_state[:, 2:3] += noise_z

        # Reset joint states with optional noise
        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()

        j_noise_mag = float(self.cfg.reset_joint_noise)
        if j_noise_mag > 0:
            j_noise = j_noise_mag * (2.0 * torch.rand((len(env_ids_t), len(self._leg_dof_ids)), device=self.device) - 1.0)
            joint_pos[:, self._leg_dof_ids] += j_noise
        joint_vel[:, self._leg_dof_ids] *= 0.0

        # Write to sim
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids_t)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids_t)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_t)

        # Clear external wrench and buffers
        zero = torch.zeros((len(env_ids_t), 1, 3), device=self.device, dtype=torch.float32)
        # replaced: self.robot.set_external_force_and_torque

        # self.robot.set_external_force_and_torque(
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zero,
            torques=zero,
            body_ids=[int(self._base_body_id[0])],
            env_ids=env_ids_t,
            is_global=False,
        )

        self.actions[env_ids_t] = 0.0
        self._last_actions[env_ids_t] = 0.0
        self._thrusts[env_ids_t] = 0.0
        self._leg_torques[env_ids_t] = 0.0
        self._joint_vel_prev[env_ids_t] = 0.0
        self._joint_acc[env_ids_t] = 0.0

        # Reset commands for these envs
        self._reset_commands(env_ids_t)


