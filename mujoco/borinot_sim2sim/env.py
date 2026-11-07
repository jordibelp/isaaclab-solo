"""SimulationContext, InteractiveScene and DirectRLEnv on the MuJoCo backend.

``DirectRLEnv`` keeps Isaac Lab's call order exactly: ``_pre_physics_step`` once per policy
step, then per physics step ``_apply_action`` -> ``scene.write_data_to_sim`` -> physics ->
``scene.update``; then dones, rewards, automatic resets and observations.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

from . import backend as mj_backend


class SimulationContext:
    """Physics/render front-end with the methods the Borinot tasks and play scripts call."""

    class RenderMode(enum.IntEnum):
        NO_GUI_OR_RENDERING = -1
        NO_RENDERING = 0
        PARTIAL_RENDERING = 1
        FULL_RENDERING = 2

    _instance = None

    def __init__(self, cfg=None):
        if SimulationContext._instance is not None:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        self.cfg = cfg
        if str(getattr(cfg, "device", "cpu")) != "cpu":
            print(f"[MuJoCo] sim.device={cfg.device!r} ignored: the MuJoCo backend keeps tensors on the CPU.")
        self.device = "cpu"
        self.backend = mj_backend.Backend(cfg.dt, cfg.gravity)
        self.render_mode = (self.RenderMode.NO_GUI_OR_RENDERING if mj_backend.OPTIONS["headless"]
                            else self.RenderMode.FULL_RENDERING)
        SimulationContext._instance = self

    @classmethod
    def instance(cls):
        return cls._instance

    @classmethod
    def clear_instance(cls):
        cls._instance = None

    def get_physics_dt(self):
        return self.cfg.dt

    def get_initial_stage(self):
        return self.backend.stage

    @property
    def stage(self):
        return self.backend.stage

    def has_gui(self):
        return not mj_backend.OPTIONS["headless"]

    def has_rtx_sensors(self):
        return False

    def is_playing(self):
        return True

    def is_stopped(self):
        return False

    def reset(self, soft=False):
        if self.backend.model is None:
            self.backend.build()

    def step(self, render=True):
        self.backend.step()
        if render:
            self.render()

    def forward(self):
        self.backend.forward()

    def render(self, mode=None):
        self.backend.render()

    def set_camera_view(self, eye, target, camera_prim_path=None):
        self.backend.set_camera(eye, target)

    def stop(self):
        pass

    def clear(self):
        pass

    def clear_all_callbacks(self):
        pass


class InteractiveScene:
    """One environment at the world origin; assets register here as in Isaac Lab."""

    def __init__(self, cfg):
        if cfg.num_envs != 1:
            raise ValueError(f"The MuJoCo backend runs one environment; got num_envs={cfg.num_envs}. "
                             "Use --num_envs=1.")
        self.cfg = cfg
        self.articulations, self.rigid_objects, self.sensors, self.extras = {}, {}, {}, {}
        self.env_origins = torch.zeros(1, 3, device=SimulationContext.instance().device)
        self.env_prim_paths = ["/World/envs/env_0"]

    @property
    def num_envs(self):
        return 1

    @property
    def device(self):
        return SimulationContext.instance().device

    @property
    def stage(self):
        return SimulationContext.instance().stage

    def clone_environments(self, copy_from_source=False):
        pass

    def filter_collisions(self, global_prim_paths=None):
        pass

    def _entities(self):
        return [*self.articulations.values(), *self.rigid_objects.values()]

    def reset(self, env_ids=None):
        for entity in [*self._entities(), *self.sensors.values()]:
            entity.reset(env_ids)

    def write_data_to_sim(self):
        for entity in self._entities():
            entity.write_data_to_sim()

    def update(self, dt):
        for entity in [*self._entities(), *self.sensors.values()]:
            entity.update(dt)

    def __getitem__(self, key):
        for group in (self.articulations, self.rigid_objects, self.sensors, self.extras):
            if key in group:
                return group[key]
        raise KeyError(f"Scene entity {key!r} not found.")

    def keys(self):
        return [*self.articulations, *self.rigid_objects, *self.sensors, *self.extras]


class ViewportCameraController:
    """Isaac Lab viewer camera semantics (origin types) on the MuJoCo free camera."""

    def __init__(self, env, cfg):
        self._env, self.cfg = env, cfg
        self.default_cam_eye = np.asarray(cfg.eye, dtype=float)
        self.default_cam_lookat = np.asarray(cfg.lookat, dtype=float)
        self.update_view_location(cfg.eye, cfg.lookat)

    def _origin(self):
        cfg, scene = self.cfg, self._env.scene
        if cfg.origin_type == "env":
            return scene.env_origins[cfg.env_index].cpu().numpy()
        if cfg.origin_type in ("asset_root", "asset_body"):
            data = scene[cfg.asset_name].data
            if cfg.origin_type == "asset_root":
                return data.root_pos_w[cfg.env_index].cpu().numpy()
            body = scene[cfg.asset_name].find_bodies(cfg.body_name)[0][0]
            return data.body_pos_w[cfg.env_index, body].cpu().numpy()
        return np.zeros(3)

    def update_view_location(self, eye=None, lookat=None):
        if eye is not None:
            self.default_cam_eye = np.asarray(eye, dtype=float)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat, dtype=float)
        origin = self._origin()
        self._env.sim.set_camera_view(self.default_cam_eye + origin, self.default_cam_lookat + origin)

    def set_view_env_index(self, env_index):
        self.cfg.env_index = env_index
        self.update_view_location()

    def update_view_to_world(self):
        self.cfg.origin_type = "world"
        self.update_view_location()

    def update_view_to_env(self):
        self.cfg.origin_type = "env"
        self.update_view_location()


# --------------------------------------------------------------------------------------
# Startup events used by the Borinot tasks (opt-in with env.include_events_randomization=True)
# --------------------------------------------------------------------------------------

def _uniform(low, high, shape):
    return torch.empty(shape).uniform_(float(low), float(high))


def _sample(values, params, operation, distribution="uniform"):
    if distribution != "uniform":
        raise NotImplementedError(f"Only uniform startup randomization is implemented, got {distribution!r}.")
    sample = _uniform(*params, values.shape)
    if operation == "scale":
        return values * sample
    if operation == "add":
        return values + sample
    if operation == "abs":
        return sample
    raise ValueError(f"Unknown randomization operation {operation!r}.")


def randomize_rigid_body_mass(env, env_ids, asset_cfg, mass_distribution_params, operation,
                              distribution="uniform", recompute_inertia=True):
    asset = env.scene[asset_cfg.name]
    bodies = _body_ids(asset, asset_cfg)
    masses = asset.root_physx_view.get_masses()
    masses[0, bodies] = _sample(asset.data.default_mass[0, bodies], mass_distribution_params, operation, distribution)
    asset.root_physx_view.set_masses(masses, torch.tensor([0]))
    if recompute_inertia:
        inertias = asset.root_physx_view.get_inertias()
        ratios = masses[0, bodies] / asset.data.default_mass[0, bodies]
        inertias[0, bodies] = asset.data.default_inertia[0, bodies] * ratios[:, None]
        asset.root_physx_view.set_inertias(inertias, torch.tensor([0]))


def randomize_rigid_body_com(env, env_ids, com_range, asset_cfg):
    asset = env.scene[asset_cfg.name]
    bodies = _body_ids(asset, asset_cfg)
    coms = asset.root_physx_view.get_coms()
    ranges = [com_range.get(axis, (0.0, 0.0)) for axis in "xyz"]
    shift = torch.stack([_uniform(low, high, (len(bodies),)) for low, high in ranges], -1)
    coms[0, bodies, :3] += shift
    asset.root_physx_view.set_coms(coms, torch.tensor([0]))


def randomize_joint_parameters(env, env_ids, asset_cfg, friction_distribution_params=None,
                               armature_distribution_params=None, lower_limit_distribution_params=None,
                               upper_limit_distribution_params=None, operation="abs", distribution="uniform"):
    asset = env.scene[asset_cfg.name]
    joints = np.arange(asset.num_joints) if asset_cfg.joint_ids is None else np.asarray(asset_cfg.joint_ids)
    if friction_distribution_params is not None:
        values = _sample(asset.data.default_joint_friction_coeff[0, joints].cpu(), friction_distribution_params,
                         operation, distribution)
        asset.write_joint_friction_coefficient_to_sim(values[None], joint_ids=joints, env_ids=torch.tensor([0]))
        print(f"[MuJoCo] Joint friction {values.tolist()} applied as MuJoCo Coulomb frictionloss [N*m]; "
              "PhysX 5 static friction is only approximated.", flush=True)
    if armature_distribution_params is not None:
        model = asset._backend.model
        model.dof_armature[asset._dof_adr[joints]] = _sample(
            torch.tensor(model.dof_armature[asset._dof_adr[joints]]), armature_distribution_params, operation,
            distribution).numpy()
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        raise NotImplementedError("Joint-limit randomization has no MuJoCo stand-in.")


def _body_ids(asset, asset_cfg):
    if asset_cfg.body_names is None:
        return list(range(asset.num_bodies))
    return asset.find_bodies(asset_cfg.body_names, preserve_order=asset_cfg.preserve_order)[0]


def _event_terms(events):
    names = getattr(events, "__dataclass_fields__", None) or vars(events)
    for name in names:
        term = getattr(events, name, None)
        if term is not None and hasattr(term, "func") and hasattr(term, "mode"):
            yield name, term


def apply_events(env, mode):
    """Minimal EventManager: resolve SceneEntityCfg ids, then call each term of ``mode``."""
    for name, term in _event_terms(env.cfg.events):
        if term.mode != mode:
            continue
        params = dict(term.params)
        for key, value in params.items():
            if type(value).__name__ == "SceneEntityCfg":
                asset = env.scene[value.name]
                if value.joint_names is not None:
                    value.joint_ids = asset.find_joints(value.joint_names, preserve_order=value.preserve_order)[0]
                else:
                    value.joint_ids = None
                if value.body_names is not None:
                    value.body_ids = asset.find_bodies(value.body_names, preserve_order=value.preserve_order)[0]
        term.func(env, env.scene.articulations["robot"]._ALL_INDICES, **params)
        print(f"[MuJoCo] Applied {mode} event {name!r}.", flush=True)


# --------------------------------------------------------------------------------------
# DirectRLEnv
# --------------------------------------------------------------------------------------

class DirectRLEnv(gym.Env):
    is_vector_env = True
    metadata = {"render_modes": [None, "human", "rgb_array"]}

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        from isaaclab.utils.seed import configure_seed

        cfg.validate()
        self.cfg = cfg
        self.render_mode = render_mode
        self._is_closed = False
        if self.cfg.seed is not None:
            self.cfg.seed = configure_seed(self.cfg.seed)
        self.sim = SimulationContext(self.cfg.sim)
        print(f"[MuJoCo] Physics {1 / self.physics_dt:g} Hz, policy {1 / self.step_dt:g} Hz, seed {self.cfg.seed}.",
              flush=True)
        self.scene = InteractiveScene(self.cfg.scene)
        self._setup_scene()
        self.sim.reset()  # compile the authored stage and initialise the assets
        self.scene.update(dt=self.physics_dt)
        need_camera = self.sim.has_gui() or render_mode == "rgb_array"
        self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer) if need_camera else None
        self.has_debug_vis_implementation = False
        self._window = None
        self.extras = {}
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._configure_gym_env_spaces()
        if self.cfg.action_noise_model or self.cfg.observation_noise_model:
            raise NotImplementedError("Action/observation noise models are not implemented on the MuJoCo backend.")
        if self.cfg.events:
            apply_events(self, "startup")
            for _, term in _event_terms(self.cfg.events):
                if term.mode not in ("startup", "prestartup"):
                    raise NotImplementedError(f"Event mode {term.mode!r} is not implemented on the MuJoCo backend.")
        self.metadata["render_fps"] = 1 / self.step_dt

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def num_envs(self):
        return self.scene.num_envs

    @property
    def physics_dt(self):
        return self.cfg.sim.dt

    @property
    def step_dt(self):
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        return self.sim.device

    @property
    def max_episode_length_s(self):
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.int64, device=self.device))
        self.scene.write_data_to_sim()
        self.sim.forward()
        return self._get_observations(), self.extras

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            time_outs_obs = self._get_observations()
            self.extras["time_outs_obs"] = {key: value.detach().clone() for key, value in time_outs_obs.items()}
            self._reset_idx(reset_env_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        from isaaclab.utils.seed import configure_seed

        return configure_seed(seed)

    def render(self, recompute: bool = False):
        self.sim.render()
        if self.render_mode in (None, "human"):
            return None
        if self.render_mode == "rgb_array":
            width, height = self.cfg.viewer.resolution
            return self.sim.backend.render_rgb(int(width), int(height))
        raise NotImplementedError(f"Render mode {self.render_mode!r} is not supported.")

    def close(self):
        if not self._is_closed:
            self.sim.backend.close()
            SimulationContext.clear_instance()
            self._is_closed = True

    def set_debug_vis(self, debug_vis: bool) -> bool:
        return False

    def _configure_gym_env_spaces(self):
        from isaaclab.envs.utils.spaces import sample_space, spec_to_gym_space

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.state_space = None
        if self.cfg.state_space:
            self.single_observation_space["critic"] = spec_to_gym_space(self.cfg.state_space)
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)
        self.actions = sample_space(self.single_action_space, self.device, batch_size=self.num_envs, fill_value=0)

    def _reset_idx(self, env_ids: Sequence[int]):
        self.scene.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

    def _setup_scene(self):
        pass

    def _pre_physics_step(self, actions):
        raise NotImplementedError

    def _apply_action(self):
        raise NotImplementedError

    def _get_observations(self):
        raise NotImplementedError

    def _get_states(self):
        return None

    def _get_rewards(self):
        raise NotImplementedError

    def _get_dones(self):
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError
