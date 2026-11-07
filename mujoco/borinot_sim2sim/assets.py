"""Articulation, RigidObject and ContactSensor stand-ins on the MuJoCo backend.

Data follows the Isaac Lab conventions: positions/orientations are link frames, ``root_*_vel_w``
are centre-of-mass velocities, quaternions are (w, x, y, z), tensors are float32 with a
leading environment dimension of one. Forces from the PhysX tensor API keep their PhysX
meaning (see backend.py for the per-step averaging).
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from . import backend as mj_backend
from . import usd_scene

GRAVITY_DIRECTION = np.array((0.0, 0.0, -1.0))


def _resolve_names(keys, names, preserve_order=False):
    from isaaclab.utils.string import resolve_matching_names

    return resolve_matching_names(keys, names, preserve_order)


def _ids(ids, count) -> np.ndarray:
    """Isaac index argument (None, slice, sequence, tensor) -> integer array."""
    if ids is None or (isinstance(ids, slice) and ids == slice(None)):
        return np.arange(count)
    if isinstance(ids, torch.Tensor):
        return ids.detach().cpu().numpy().astype(int).reshape(-1)
    return np.asarray(ids, dtype=int).reshape(-1)


def _env_rows(env_ids) -> bool:
    """True when the call addresses environment 0, the only one."""
    ids = _ids(env_ids, 1)
    if len(ids) and (ids != 0).any():
        raise IndexError("The MuJoCo backend runs exactly one environment (num_envs=1).")
    return len(ids) > 0


def _np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=float)


def _tensor(array, device) -> torch.Tensor:
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=device)


def _quat_matrix(quat) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _xyzw(quat) -> np.ndarray:
    return np.asarray(quat)[..., [1, 2, 3, 0]]


class WrenchComposer:
    """Link-frame force (at the centre of mass) and torque per body, like Isaac Lab without positions."""

    def __init__(self, asset):
        self._asset = asset
        self.force_b = np.zeros((asset.num_bodies, 3))
        self.torque_b = np.zeros((asset.num_bodies, 3))
        self.active = False

    def _write(self, forces, torques, positions, body_ids, env_ids, is_global, add):
        if positions is not None or is_global:
            raise NotImplementedError("The MuJoCo backend composes local wrenches without positions only.")
        if not _env_rows(env_ids) or (forces is None and torques is None):
            return
        ids = _ids(body_ids, self._asset.num_bodies)
        for target, value in ((self.force_b, forces), (self.torque_b, torques)):
            if value is not None:
                value = _np(value)[0]
                target[ids] = target[ids] + value if add else value
        self.active = True

    def set_forces_and_torques(self, forces=None, torques=None, positions=None, body_ids=None, env_ids=None,
                               is_global=False):
        self._write(forces, torques, positions, body_ids, env_ids, is_global, add=False)

    def add_forces_and_torques(self, forces=None, torques=None, positions=None, body_ids=None, env_ids=None,
                               is_global=False):
        self._write(forces, torques, positions, body_ids, env_ids, is_global, add=True)

    def reset(self, env_ids=None):
        if _env_rows(env_ids):
            self.force_b.fill(0.0)
            self.torque_b.fill(0.0)
            self.active = False


class _PhysxView:
    """Subset of the PhysX tensor view used by the Borinot tasks and Isaac Lab events (CPU tensors)."""

    def __init__(self, asset):
        self._asset = asset

    def _model(self):
        return self._asset._backend.model

    def get_masses(self):
        return torch.tensor(self._model().body_mass[self._asset._body_ids], dtype=torch.float32)[None]

    def set_masses(self, masses, indices=None):
        if _env_rows(indices):
            self._model().body_mass[self._asset._body_ids] = _np(masses)[0]
            self._asset._backend.update_constants()

    def get_inertias(self):
        return torch.tensor(self._asset._inertia_tensors(), dtype=torch.float32).reshape(1, -1, 9)

    def set_inertias(self, inertias, indices=None):
        if not _env_rows(indices):
            return
        model = self._model()
        for body, tensor in zip(self._asset._body_ids, _np(inertias)[0].reshape(-1, 3, 3)):
            moments, axes = np.linalg.eigh(0.5 * (tensor + tensor.T))
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            model.body_inertia[body] = moments
            model.body_iquat[body] = usd_scene.matrix_to_quat(axes)
        self._asset._backend.update_constants()

    def get_coms(self):
        model = self._model()
        ids = self._asset._body_ids
        return torch.tensor(np.concatenate((model.body_ipos[ids], _xyzw(model.body_iquat[ids])), -1),
                            dtype=torch.float32)[None]

    def set_coms(self, coms, indices=None):
        if _env_rows(indices):
            self._model().body_ipos[self._asset._body_ids] = _np(coms)[0, :, :3]
            self._asset._backend.update_constants()

    def get_material_properties(self):
        return torch.tensor(self._asset._materials, dtype=torch.float32)[None]

    def set_material_properties(self, materials, indices=None):
        if _env_rows(indices):
            self._asset._set_materials(_np(materials)[0])

    def get_link_incoming_joint_force(self):
        return self._asset.data.body_incoming_joint_wrench_b


class _Cache:
    """Values recomputed once per MuJoCo state version, like Isaac Lab's timestamped buffers."""

    def __init__(self, asset):
        self._asset = asset
        self._version = -1
        self._values = {}

    def get(self, name, compute):
        version = self._asset._backend.version
        if version != self._version:
            self._values.clear()
            self._version = version
        if name not in self._values:
            self._values[name] = compute()
        return self._values[name]


class _RigidBodies:
    """Bodies of one asset with a free root joint: shared by Articulation and RigidObject."""

    kind = None

    def __init__(self, cfg):
        self.cfg = cfg.copy()
        self._backend = mj_backend.current()
        self.asset_path = cfg.prim_path.replace("env_.*", "env_0")
        if cfg.spawn is not None:
            cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos, orientation=cfg.init_state.rot)
        self._backend.register(self)
        self._velocity_before = None
        self._acceleration = None

    @property
    def device(self):
        from isaaclab.sim import SimulationContext

        return SimulationContext.instance().device

    @property
    def num_instances(self):
        return 1

    @property
    def num_bodies(self):
        return len(self.body_names)

    def _initialize_bodies(self, backend, paths):
        model = backend.model
        self._body_paths = list(paths)
        self.body_names = [path.rsplit("/", 1)[-1] for path in paths]
        self._body_ids = np.array([backend.index.body_ids[path] for path in paths])
        root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, paths[0] + ":root")
        if root_joint < 0:
            raise RuntimeError(f"{paths[0]} has no free joint in the MuJoCo model.")
        self._root_qpos = model.jnt_qposadr[root_joint]
        self._root_dof = model.jnt_dofadr[root_joint]
        self._ALL_INDICES = torch.arange(1, dtype=torch.long, device=self.device)
        self._cache = _Cache(self)
        self._acceleration = np.zeros((len(paths), 6))
        self.root_physx_view = _PhysxView(self)
        self.permanent_wrench_composer = WrenchComposer(self)
        self.instantaneous_wrench_composer = WrenchComposer(self)
        state = self.cfg.init_state
        self._default_root_state = np.concatenate((state.pos, state.rot, state.lin_vel, state.ang_vel))[None]
        self._default_mass = model.body_mass[self._body_ids].copy()[None]
        self._default_inertia = self._inertia_tensors().reshape(1, -1, 9)

    def _inertia_tensors(self):
        model = self._backend.model
        tensors = []
        for body in self._body_ids:
            axes = _quat_matrix(model.body_iquat[body])
            tensors.append(axes @ np.diag(model.body_inertia[body]) @ axes.T)
        return np.array(tensors)

    # -- kinematics read by the data classes -----------------------------------------

    def _kinematics(self):
        def compute():
            d, model, ids = self._backend.data, self._backend.model, self._body_ids
            pos, quat = d.xpos[ids].copy(), d.xquat[ids].copy()
            mat, com = d.xmat[ids].reshape(-1, 3, 3).copy(), d.xipos[ids].copy()
            ang = d.cvel[ids, :3].copy()
            # cvel is a spatial velocity at the kinematic tree's subtree CoM, in world axes.
            lin = d.cvel[ids, 3:] + np.cross(ang, com - d.subtree_com[model.body_rootid[ids]])
            return SimpleNamespace(pos=pos, quat=quat, mat=mat, com=com, lin=lin, ang=ang)

        return self._cache.get("kinematics", compute)

    def _before_step(self):
        kin = self._kinematics()
        self._velocity_before = np.concatenate((kin.lin, kin.ang), -1)

    def _after_step(self):
        kin = self._kinematics()
        self._acceleration = (np.concatenate((kin.lin, kin.ang), -1) - self._velocity_before) / self._backend.physics_dt

    # -- state writes ------------------------------------------------------------------

    def write_root_state_to_sim(self, root_state, env_ids=None):
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

    def write_root_link_state_to_sim(self, root_state, env_ids=None):
        self.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_ids)

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        self.write_root_link_pose_to_sim(root_pose, env_ids)

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        self.write_root_com_velocity_to_sim(root_velocity, env_ids)

    def write_root_link_pose_to_sim(self, root_pose, env_ids=None):
        if not _env_rows(env_ids):
            return
        pose = _np(root_pose)[0]
        d = self._backend.data
        d.qpos[self._root_qpos:self._root_qpos + 3] = pose[:3]
        d.qpos[self._root_qpos + 3:self._root_qpos + 7] = pose[3:7] / np.linalg.norm(pose[3:7])
        self._backend.forward()

    def write_root_com_velocity_to_sim(self, root_velocity, env_ids=None):
        if not _env_rows(env_ids):
            return
        velocity = _np(root_velocity)[0]
        d, model = self._backend.data, self._backend.model
        rotation = _quat_matrix(d.qpos[self._root_qpos + 3:self._root_qpos + 7])
        offset = rotation @ model.body_ipos[self._body_ids[0]]
        # The free joint moves the link origin: v_origin = v_com - omega x (com - origin).
        d.qvel[self._root_dof:self._root_dof + 3] = velocity[:3] - np.cross(velocity[3:], offset)
        d.qvel[self._root_dof + 3:self._root_dof + 6] = rotation.T @ velocity[3:]
        self._backend.forward()

    def write_root_link_velocity_to_sim(self, root_velocity, env_ids=None):
        if not _env_rows(env_ids):
            return
        velocity = _np(root_velocity)[0]
        d = self._backend.data
        rotation = _quat_matrix(d.qpos[self._root_qpos + 3:self._root_qpos + 7])
        d.qvel[self._root_dof:self._root_dof + 3] = velocity[:3]
        d.qvel[self._root_dof + 3:self._root_dof + 6] = rotation.T @ velocity[3:]
        self._backend.forward()

    def find_bodies(self, name_keys, preserve_order=False):
        return _resolve_names(name_keys, self.body_names, preserve_order)

    def _write_external_wrench(self):
        self._backend.data.xfrc_applied[self._body_ids] = 0.0
        composers = [c for c in (self.permanent_wrench_composer, self.instantaneous_wrench_composer) if c.active]
        if not composers:
            return
        force = sum(c.force_b for c in composers)
        torque = sum(c.torque_b for c in composers)
        d = self._backend.data
        rotation = d.xmat[self._body_ids].reshape(-1, 3, 3)
        # Isaac Lab applies the composed link-frame force at the centre of mass, as MuJoCo does.
        d.xfrc_applied[self._body_ids, :3] = np.einsum("bij,bj->bi", rotation, force)
        d.xfrc_applied[self._body_ids, 3:] = np.einsum("bij,bj->bi", rotation, torque)
        self.instantaneous_wrench_composer.reset()

    def reset(self, env_ids=None):
        self.permanent_wrench_composer.reset(env_ids)
        self.instantaneous_wrench_composer.reset(env_ids)

    def update(self, dt):
        pass

    def _accumulate_joint_wrench(self, weight):
        pass


class _BodyData:
    """Root and body quantities (float32, environment dimension first)."""

    def __init__(self, asset):
        self._asset = asset

    def _get(self, name, compute):
        return self._asset._cache.get(name, lambda: _tensor(compute(), self._asset.device))

    @property
    def body_names(self):
        return self._asset.body_names

    # -- links -------------------------------------------------------------------------
    @property
    def body_link_pose_w(self):
        return self._get("body_link_pose_w", lambda: np.concatenate(
            (self._asset._kinematics().pos, self._asset._kinematics().quat), -1)[None])

    @property
    def body_link_pos_w(self):
        return self.body_link_pose_w[..., :3]

    @property
    def body_link_quat_w(self):
        return self.body_link_pose_w[..., 3:7]

    body_pose_w = body_link_pose_w
    body_pos_w = body_link_pos_w
    body_quat_w = body_link_quat_w

    @property
    def body_com_vel_w(self):
        kin = self._asset._kinematics()
        return self._get("body_com_vel_w", lambda: np.concatenate((kin.lin, kin.ang), -1)[None])

    @property
    def body_com_lin_vel_w(self):
        return self.body_com_vel_w[..., :3]

    @property
    def body_com_ang_vel_w(self):
        return self.body_com_vel_w[..., 3:]

    body_vel_w = body_com_vel_w
    body_lin_vel_w = body_com_lin_vel_w
    body_ang_vel_w = body_com_ang_vel_w

    @property
    def body_link_vel_w(self):
        kin = self._asset._kinematics()
        return self._get("body_link_vel_w", lambda: np.concatenate(
            (kin.lin + np.cross(kin.ang, kin.pos - kin.com), kin.ang), -1)[None])

    @property
    def body_link_lin_vel_w(self):
        return self.body_link_vel_w[..., :3]

    @property
    def body_link_ang_vel_w(self):
        return self.body_link_vel_w[..., 3:]

    @property
    def body_com_pos_w(self):
        return self._get("body_com_pos_w", lambda: self._asset._kinematics().com[None])

    @property
    def body_com_quat_w(self):
        def compute():
            d = self._asset._backend.data
            return np.array([usd_scene.matrix_to_quat(d.ximat[b].reshape(3, 3)) for b in self._asset._body_ids])[None]

        return self._get("body_com_quat_w", compute)

    @property
    def body_com_pose_b(self):
        coms = self._asset.root_physx_view.get_coms()[..., [0, 1, 2, 6, 3, 4, 5]]
        return coms.to(self._asset.device)

    @property
    def body_com_pos_b(self):
        return self.body_com_pose_b[..., :3]

    @property
    def body_com_quat_b(self):
        return self.body_com_pose_b[..., 3:7]

    com_pos_b = body_com_pos_b

    @property
    def body_com_acc_w(self):
        return self._get("body_com_acc_w", lambda: self._asset._acceleration[None])

    @property
    def body_com_lin_acc_w(self):
        return self.body_com_acc_w[..., :3]

    @property
    def body_com_ang_acc_w(self):
        return self.body_com_acc_w[..., 3:]

    body_acc_w = body_com_acc_w
    body_lin_acc_w = body_com_lin_acc_w
    body_ang_acc_w = body_com_ang_acc_w

    @property
    def body_state_w(self):
        return torch.cat((self.body_link_pose_w, self.body_com_vel_w), -1)

    # -- root (first body) -----------------------------------------------------------
    @property
    def root_link_pose_w(self):
        return self.body_link_pose_w[:, 0]

    @property
    def root_link_pos_w(self):
        return self.root_link_pose_w[:, :3]

    @property
    def root_link_quat_w(self):
        return self.root_link_pose_w[:, 3:7]

    root_pose_w = root_link_pose_w
    root_pos_w = root_link_pos_w
    root_quat_w = root_link_quat_w

    @property
    def root_com_vel_w(self):
        return self.body_com_vel_w[:, 0]

    @property
    def root_com_lin_vel_w(self):
        return self.root_com_vel_w[:, :3]

    @property
    def root_com_ang_vel_w(self):
        return self.root_com_vel_w[:, 3:]

    root_vel_w = root_com_vel_w
    root_lin_vel_w = root_com_lin_vel_w
    root_ang_vel_w = root_com_ang_vel_w

    @property
    def root_link_vel_w(self):
        return self.body_link_vel_w[:, 0]

    @property
    def root_link_lin_vel_w(self):
        return self.root_link_vel_w[:, :3]

    @property
    def root_link_ang_vel_w(self):
        return self.root_link_vel_w[:, 3:]

    def _to_root_frame(self, name, vector_w):
        def compute():
            rotation = self._asset._kinematics().mat[0]
            return (rotation.T @ _np(vector_w)[0])[None]

        return self._get(name, compute)

    @property
    def root_com_lin_vel_b(self):
        return self._to_root_frame("root_com_lin_vel_b", self.root_com_lin_vel_w)

    @property
    def root_com_ang_vel_b(self):
        return self._to_root_frame("root_com_ang_vel_b", self.root_com_ang_vel_w)

    root_lin_vel_b = root_com_lin_vel_b
    root_ang_vel_b = root_com_ang_vel_b

    @property
    def root_link_lin_vel_b(self):
        return self._to_root_frame("root_link_lin_vel_b", self.root_link_lin_vel_w)

    @property
    def root_link_ang_vel_b(self):
        return self.root_com_ang_vel_b

    @property
    def projected_gravity_b(self):
        return self._to_root_frame("projected_gravity_b", GRAVITY_DIRECTION[None])

    @property
    def heading_w(self):
        rotation = self._asset._kinematics().mat[0]
        return self._get("heading_w", lambda: np.array([np.arctan2(rotation[1, 0], rotation[0, 0])]))

    @property
    def root_com_pos_w(self):
        return self.body_com_pos_w[:, 0]

    @property
    def root_state_w(self):
        return torch.cat((self.root_link_pose_w, self.root_com_vel_w), -1)

    @property
    def root_link_state_w(self):
        return torch.cat((self.root_link_pose_w, self.root_link_vel_w), -1)

    # -- defaults ----------------------------------------------------------------------
    @property
    def default_root_state(self):
        return _tensor(self._asset._default_root_state, self._asset.device)

    @property
    def default_mass(self):
        return torch.tensor(self._asset._default_mass, dtype=torch.float32)

    @property
    def default_inertia(self):
        return torch.tensor(self._asset._default_inertia, dtype=torch.float32)


class ArticulationData(_BodyData):
    @property
    def joint_names(self):
        return self._asset.joint_names

    @property
    def joint_pos(self):
        return self._get("joint_pos", lambda: self._asset._backend.data.qpos[self._asset._qpos_adr][None])

    @property
    def joint_vel(self):
        return self._get("joint_vel", lambda: self._asset._backend.data.qvel[self._asset._dof_adr][None])

    @property
    def joint_acc(self):
        return self._get("joint_acc", lambda: self._asset._joint_acceleration[None])

    @property
    def body_incoming_joint_wrench_b(self):
        backend = self._asset._backend
        backend.track_joint_wrench = True  # measured from the next physics step on
        return self._get("body_incoming_joint_wrench_b", lambda: backend.joint_wrench[self._asset._body_ids][None])

    @property
    def default_joint_pos(self):
        return _tensor(self._asset._default_joint_pos, self._asset.device)

    @property
    def default_joint_vel(self):
        return _tensor(self._asset._default_joint_vel, self._asset.device)

    @property
    def joint_pos_target(self):
        return _tensor(self._asset._targets["position"][None], self._asset.device)

    @property
    def joint_vel_target(self):
        return _tensor(self._asset._targets["velocity"][None], self._asset.device)

    @property
    def joint_effort_target(self):
        return _tensor(self._asset._targets["effort"][None], self._asset.device)

    @property
    def computed_torque(self):
        return _tensor(self._asset._computed_torque[None], self._asset.device)

    @property
    def applied_torque(self):
        return _tensor(self._asset._applied_torque[None], self._asset.device)

    @property
    def joint_pos_limits(self):
        model = self._asset._backend.model
        return _tensor(model.jnt_range[self._asset._joint_mj_ids][None], self._asset.device)

    soft_joint_pos_limits = joint_pos_limits

    @property
    def joint_stiffness(self):
        return _tensor(self._asset._kp[None], self._asset.device)

    @property
    def joint_damping(self):
        return _tensor(self._asset._kd[None], self._asset.device)

    @property
    def joint_armature(self):
        model = self._asset._backend.model
        return _tensor(model.dof_armature[self._asset._dof_adr][None], self._asset.device)

    @property
    def joint_friction_coeff(self):
        model = self._asset._backend.model
        return _tensor(model.dof_frictionloss[self._asset._dof_adr][None], self._asset.device)

    @property
    def default_joint_friction_coeff(self):
        return _tensor(self._asset._default_friction[None], self._asset.device)


class Articulation(_RigidBodies):
    """Floating-base articulation: free root, revolute joints, fixed tools, implicit PD actuators."""

    kind = "articulation"

    def __init__(self, cfg):
        super().__init__(cfg)
        props = getattr(cfg.spawn, "articulation_props", None)
        self.self_collisions = bool(getattr(props, "enabled_self_collisions", False))
        self.data = ArticulationData(self)

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def is_fixed_base(self):
        return False

    def _initialize(self, backend):
        index, model = backend.index, backend.model
        self._initialize_bodies(backend, index.articulations[self.asset_path])
        infos = [index.bodies[path] for path in self._body_paths]
        self.joint_names = [info.joint_name for info in infos if info.joint_name is not None]
        self._joint_mj_ids = np.array([index.joint_ids[name] for name in self.joint_names], dtype=int)
        self._qpos_adr = model.jnt_qposadr[self._joint_mj_ids]
        self._dof_adr = model.jnt_dofadr[self._joint_mj_ids]
        self._pd_actuators = np.array([index.actuator_ids[(name, "pd")] for name in self.joint_names], dtype=int)
        self._ff_actuators = np.array([index.actuator_ids[(name, "ff")] for name in self.joint_names], dtype=int)
        # Child joint frames of every non-root body, for incoming joint wrenches.
        self._joint_frames = [(k, info.joint_pos, _quat_matrix(info.joint_frame_quat))
                              for k, info in enumerate(infos) if info.parent is not None]
        count = len(self.joint_names)
        self._targets = {key: np.zeros(count) for key in ("position", "velocity", "effort")}
        self._computed_torque, self._applied_torque = np.zeros(count), np.zeros(count)
        self._joint_acceleration, self._joint_velocity_before = np.zeros(count), np.zeros(count)
        self._default_joint_pos = self._joint_values(self.cfg.init_state.joint_pos)[None]
        self._default_joint_vel = self._joint_values(self.cfg.init_state.joint_vel)[None]
        self._setup_actuators(model)
        backend.data.qpos[self._qpos_adr] = self._default_joint_pos[0]
        backend.data.qvel[self._dof_adr] = self._default_joint_vel[0]

    def _joint_values(self, values, default=0.0):
        """Isaac Lab {regex: value} or scalar -> one value per joint."""
        result = np.full(len(self.joint_names), float(default))
        if values is None:
            return result
        if not isinstance(values, dict):
            result[:] = float(values)
            return result
        ids, _, matched = _resolve_names_values(values, self.joint_names)
        result[ids] = matched
        return result

    def _setup_actuators(self, model):
        count = len(self.joint_names)
        self._kp, self._kd = np.zeros(count), np.zeros(count)
        self._effort_limit = np.full(count, np.inf)
        armature, friction = np.zeros(count), np.zeros(count)
        for actuator in self.cfg.actuators.values():
            ids, _ = self.find_joints(actuator.joint_names_expr)
            for name, target in (("stiffness", self._kp), ("damping", self._kd),
                                 ("effort_limit_sim", self._effort_limit), ("armature", armature),
                                 ("friction", friction)):
                value = getattr(actuator, name, None)
                if value is not None:
                    target[ids] = self._joint_values(value)[ids] if isinstance(value, dict) else float(value)
        self._default_friction = friction.copy()
        model.dof_armature[self._dof_adr] = armature
        model.dof_frictionloss[self._dof_adr] = friction
        print(f"[MuJoCo] Joint armature [kg*m^2]: {dict(zip(self.joint_names, armature.tolist()))}", flush=True)
        for j, actuator in enumerate(self._pd_actuators):
            # PhysX implicit drive: force = kp*(q* - q) + kd*(v* - v), limited by the effort limit.
            # gain 1 on ctrl = kp*q* + kd*v*; implicitfast integrates the velocity term implicitly.
            model.actuator_gainprm[actuator, 0] = 1.0
            model.actuator_biasprm[actuator, :3] = (0.0, -self._kp[j], -self._kd[j])
            limit = self._effort_limit[j]
            model.actuator_forcelimited[actuator] = bool(np.isfinite(limit))
            model.actuator_forcerange[actuator] = (-limit, limit) if np.isfinite(limit) else (0.0, 0.0)

    def find_joints(self, name_keys, joint_subset=None, preserve_order=False):
        names = self.joint_names if joint_subset is None else [self.joint_names[i] for i in joint_subset]
        return _resolve_names(name_keys, names, preserve_order)

    # -- commands ----------------------------------------------------------------------

    def _set_target(self, key, values, joint_ids, env_ids):
        if _env_rows(env_ids):
            self._targets[key][_ids(joint_ids, self.num_joints)] = _np(values)[0]

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self._set_target("position", target, joint_ids, env_ids)

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self._set_target("velocity", target, joint_ids, env_ids)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        self._set_target("effort", target, joint_ids, env_ids)

    def write_data_to_sim(self):
        self._write_external_wrench()
        d = self._backend.data
        q, dq = d.qpos[self._qpos_adr], d.qvel[self._dof_adr]
        position, velocity, effort = (self._targets[k] for k in ("position", "velocity", "effort"))
        # Isaac Lab's implicit-actuator estimates; the drive itself runs inside the physics step.
        self._computed_torque = self._kp * (position - q) + self._kd * (velocity - dq) + effort
        self._applied_torque = np.clip(self._computed_torque, -self._effort_limit, self._effort_limit)
        d.ctrl[self._pd_actuators] = self._kp * position + self._kd * velocity
        d.ctrl[self._ff_actuators] = effort  # PhysX joint actuation forces are not limited by the drive

    # -- state writes ------------------------------------------------------------------

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        if not _env_rows(env_ids):
            return
        ids = _ids(joint_ids, self.num_joints)
        d = self._backend.data
        d.qpos[self._qpos_adr[ids]] = _np(position)[0]
        d.qvel[self._dof_adr[ids]] = _np(velocity)[0]
        self._backend.forward()

    def write_joint_position_to_sim(self, position, joint_ids=None, env_ids=None):
        self.write_joint_state_to_sim(position, self.data.joint_vel[:, _ids(joint_ids, self.num_joints)],
                                      joint_ids, env_ids)

    def write_joint_velocity_to_sim(self, velocity, joint_ids=None, env_ids=None):
        self.write_joint_state_to_sim(self.data.joint_pos[:, _ids(joint_ids, self.num_joints)], velocity,
                                      joint_ids, env_ids)

    def write_joint_position_limit_to_sim(self, limits, joint_ids=None, env_ids=None, warn_limit_violation=True):
        if not _env_rows(env_ids):
            return
        ids = _ids(joint_ids, self.num_joints)
        limits = _np(limits)[0].reshape(len(ids), 2)
        self._backend.model.jnt_range[self._joint_mj_ids[ids]] = limits
        self._backend._viewer_needs_model = True
        self._default_joint_pos[0, ids] = np.clip(self._default_joint_pos[0, ids], limits[:, 0], limits[:, 1])

    def write_joint_friction_coefficient_to_sim(self, joint_friction_coeff, joint_dynamic_friction_coeff=None,
                                                joint_viscous_friction_coeff=None, joint_ids=None, env_ids=None):
        """PhysX 5 static friction effort -> MuJoCo Coulomb frictionloss [N*m] (an approximation)."""
        if not _env_rows(env_ids):
            return
        ids = _ids(joint_ids, self.num_joints)
        value = np.broadcast_to(_np(joint_friction_coeff), (1, len(ids)))[0]
        self._backend.model.dof_frictionloss[self._dof_adr[ids]] = value
        self._backend._viewer_needs_model = True
        if joint_viscous_friction_coeff is not None:
            viscous = np.broadcast_to(_np(joint_viscous_friction_coeff), (1, len(ids)))[0]
            self._backend.model.dof_damping[self._dof_adr[ids]] = viscous

    # -- physics-step hooks ------------------------------------------------------------

    def _before_step(self):
        super()._before_step()
        self._joint_velocity_before = self._backend.data.qvel[self._dof_adr].copy()

    def _after_step(self):
        super()._after_step()
        velocity = self._backend.data.qvel[self._dof_adr]
        self._joint_acceleration = (velocity - self._joint_velocity_before) / self._backend.physics_dt

    def _accumulate_joint_wrench(self, weight):
        """Parent-on-child [F, M] at the child joint, in child joint axes (PhysX incoming joint force)."""
        d, model, out = self._backend.data, self._backend.model, self._backend.joint_wrench
        for k, joint_pos, joint_rotation in self._joint_frames:
            body = self._body_ids[k]
            # cfrc_int: [torque, force] from the parent at the tree's subtree CoM, world axes.
            torque, force = d.cfrc_int[body, :3], d.cfrc_int[body, 3:]
            centre = d.subtree_com[model.body_rootid[body]]
            child_rotation = d.xmat[body].reshape(3, 3)
            anchor = d.xpos[body] + child_rotation @ joint_pos
            frame = child_rotation @ joint_rotation
            out[body, :3] += weight * (frame.T @ force)
            out[body, 3:] += weight * (frame.T @ (torque + np.cross(centre - anchor, force)))


def _resolve_names_values(values, names):
    from isaaclab.utils.string import resolve_matching_names_values

    return resolve_matching_names_values(values, names)


class RigidObjectData(_BodyData):
    pass


class RigidObject(_RigidBodies):
    """Single free rigid body (the goalkeeper balls)."""

    kind = "rigid_object"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.data = RigidObjectData(self)

    @property
    def num_objects(self):
        return 1

    def _initialize(self, backend):
        self._initialize_bodies(backend, [self.asset_path])
        model, body = backend.model, self._body_ids[0]
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        self._geoms = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
        shapes = [shape for shape in backend.index.bodies[self.asset_path].shapes if shape.collision]
        self._materials = np.array([[s.friction, s.friction, s.restitution] for s in shapes]).reshape(-1, 3)

    def _set_materials(self, materials):
        model = self._backend.model
        self._backend._viewer_needs_model = True
        self._materials = materials.reshape(-1, 3).copy()
        for geom, (_, dynamic, restitution) in zip(self._geoms, self._materials):
            model.geom_friction[geom, 0] = dynamic
            if model.geom_priority[geom] > 0:  # bouncy free bodies own their contact parameters
                model.geom_solref[geom] = usd_scene.restitution_solref(restitution)

    def write_data_to_sim(self):
        self._write_external_wrench()


class ContactSensorData:
    def __init__(self):
        self.net_forces_w = None
        self.force_matrix_w = None


class ContactSensor:
    """Per-body contact forces from MuJoCo contacts, with PhysX-style filtered force matrices."""

    kind = "sensor"

    def __init__(self, cfg):
        self.cfg = cfg
        self._backend = mj_backend.current()
        self._backend.register(self)
        self.data = ContactSensorData()

    @property
    def num_instances(self):
        return 1

    @property
    def num_bodies(self):
        return len(self.body_names)

    @property
    def device(self):
        from isaaclab.sim import SimulationContext

        return SimulationContext.instance().device

    def _initialize(self, backend):
        paths = list(backend.index.bodies)
        sources = usd_scene.match_paths(self.cfg.prim_path, paths)
        if not sources:
            raise RuntimeError(f"Contact sensor path {self.cfg.prim_path!r} matched no rigid body.")
        self.body_names = [path.rsplit("/", 1)[-1] for path in sources]
        self._body_ids = np.array([backend.index.body_ids[path] for path in sources])
        filters = [path for expression in (self.cfg.filter_prim_paths_expr or [])
                   for path in usd_scene.match_paths(expression, paths)]
        self._filter_ids = np.array([backend.index.body_ids[path] for path in filters], dtype=int)
        self.contact_physx_view = SimpleNamespace(filter_count=len(self._filter_ids))
        self.reset()

    def find_bodies(self, name_keys, preserve_order=False):
        return _resolve_names(name_keys, self.body_names, preserve_order)

    def reset(self, env_ids=None):
        if not _env_rows(env_ids):
            return
        self.data.net_forces_w = torch.zeros(1, self.num_bodies, 3, device=self.device)
        if len(self._filter_ids):
            self.data.force_matrix_w = torch.zeros(1, self.num_bodies, len(self._filter_ids), 3, device=self.device)

    def _before_step(self):
        pass

    def _after_step(self):
        pairs = self._backend.pair_force[self._body_ids]
        self.data.net_forces_w = _tensor(pairs.sum(1)[None], self.device)
        if len(self._filter_ids):
            self.data.force_matrix_w = _tensor(pairs[:, self._filter_ids][None], self.device)

    def update(self, dt, force_recompute=False):
        pass
