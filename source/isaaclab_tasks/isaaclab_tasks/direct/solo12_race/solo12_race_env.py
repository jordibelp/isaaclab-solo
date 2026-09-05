# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import re

import gymnasium as gym
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaacsim.core.simulation_manager import SimulationManager

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, TiledCamera
from isaaclab.utils.buffers import DelayBuffer

from .reward_utils import dense_reaction_force_reward
from .solo12_race_env_cfg import Solo12RaceEnvCfg, resolve_solo12_race_scene_usd_path


def _straight_track_start_to_end_distance_m(race_scene: str, waypoints_w: torch.Tensor) -> float | None:
    """Return planar start-to-end distance for the straight track, or ``None`` for other scenes."""
    if race_scene != "straightSimple":
        return None
    if waypoints_w.ndim != 2 or waypoints_w.shape[0] < 2 or waypoints_w.shape[1] < 2:
        raise ValueError(f"Expected at least two 2D/3D waypoints for straightSimple, got {tuple(waypoints_w.shape)}.")
    distance_m = float(torch.linalg.vector_norm(waypoints_w[-1, :2] - waypoints_w[0, :2]).item())
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError(f"straightSimple start-to-end distance must be finite and positive, got {distance_m} m.")
    return distance_m


class Solo12RaceEnv(DirectRLEnv):
    cfg: Solo12RaceEnvCfg

    def __init__(self, cfg: Solo12RaceEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self._configure_backward_force_curriculum()

        cfg.scene_usd = sim_utils.UsdFileCfg(usd_path=str(resolve_solo12_race_scene_usd_path(cfg.race_scene)))
        if not getattr(cfg, "enable_events_randomization", False):
            cfg.events = None

        self._patch_names: list[str] = []
        self._patch_rel_paths: list[str] = []
        self._patch_material_paths: list[list[str]] = []
        self._patch_visual_material_paths: list[list[str]] = []
        self._patch_physx_view = None
        self._patch_physx_paths: list[str] = []
        self._patch_physx_view_failed = False
        self._patch_material_bucket_values = torch.empty(0, 3)
        self._patch_bucket_ids = torch.empty(0, 0, dtype=torch.long)
        self._patch_friction_static = torch.empty(0, 0)
        self._patch_friction_dynamic = torch.empty(0, 0)
        self._gt_patch_mu_latched = torch.empty(0, 0)
        self._friction_generator: torch.Generator | None = None
        self._patch_xy_min = torch.empty(0, 2)
        self._patch_xy_max = torch.empty(0, 2)
        self._track_waypoint_names = tuple(cfg.waypoint_names)
        self._track_waypoints_w = torch.empty(0, 3)
        self._gate_pillars_w = torch.empty(0, 2, 3)
        self._segment_lengths = torch.empty(0)
        self._track_total_length = torch.tensor(0.0)
        self.straight_track_start_to_end_distance_m: float | None = None
        self._segment_cumulative = torch.empty(0)
        self._track_targets_w = torch.empty(0, 3)
        self._track_start_yaw = torch.tensor(0.0)
        self._gate_count = 0
        self._target_count = 0
        self._previous_base_pos_w = torch.empty(0, 3)
        self._next_friction_resample_time_s = torch.empty(0)
        self._foot_reaction_contact_sensors: dict[str, ContactSensor] = {}
        self._foot_reaction_contact_sensor_labels: tuple[str, ...] = ()

        super().__init__(cfg, render_mode, **kwargs)

        self._segment_lengths = self._segment_lengths.to(self.device)
        self._track_total_length = self._track_total_length.to(self.device)
        self._segment_cumulative = self._segment_cumulative.to(self.device)
        self._track_targets_w = self._track_targets_w.to(self.device)
        self._track_start_yaw = self._track_start_yaw.to(self.device)
        self._track_waypoints_w = self._track_waypoints_w.to(self.device)
        self._gate_pillars_w = self._gate_pillars_w.to(self.device)
        self._previous_base_pos_w = self._previous_base_pos_w.to(self.device)
        self._next_friction_resample_time_s = torch.full(
            (self.num_envs,), float("inf"), dtype=torch.float, device=self.device
        )
        self._patch_material_bucket_values = self._patch_material_bucket_values.to(self.device)
        self._patch_bucket_ids = self._patch_bucket_ids.to(self.device)
        self._patch_friction_static = self._patch_friction_static.to(self.device)
        self._patch_friction_dynamic = self._patch_friction_dynamic.to(self.device)
        self._patch_xy_min = self._patch_xy_min.to(self.device)
        self._patch_xy_max = self._patch_xy_max.to(self.device)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._delayed_processed_actions = torch.zeros_like(self._actions)

        self._foot_imu_history_len = self.cfg.foot_imu_history_length if self.cfg.include_foot_imu_obs else 0
        self._foot_imu_sample_dim = self.cfg.foot_imu_obs_dim
        self._foot_imu_history = torch.zeros(
            self.num_envs,
            self._foot_imu_history_len,
            self._foot_imu_sample_dim,
            device=self.device,
        )
        self._foot_imu_bias = torch.zeros(self.num_envs, self._foot_imu_sample_dim, device=self.device)

        self._joint_state_history_len = (
            self.cfg.joint_state_history_length if self.cfg.include_joint_state_history_obs else 0
        )
        self._joint_state_sample_dim = self.cfg.joint_state_history_obs_dim
        self._joint_state_history = torch.zeros(
            self.num_envs,
            self._joint_state_history_len,
            self._joint_state_sample_dim,
            device=self.device,
        )

        self._enable_actuation_delay = bool(getattr(self.cfg, "enable_actuation_delay", False))
        max_action_delay = self.cfg.actuation_delay_range[1] if self._enable_actuation_delay else 0
        self._action_delay_buffer = DelayBuffer(max_action_delay, self.num_envs, device=self.device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self._joint_ids, _ = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        self._configure_joint_position_limits()
        self._feet_body_ids, self._feet_body_names = self._contact_sensor.find_bodies(".*_calf")
        self._feet_robot_body_ids, self._feet_robot_body_names = self._robot.find_bodies(".*_calf")
        self._gt_patch_mu_latched = torch.full(
            (self.num_envs, len(self._feet_body_ids)),
            float(self.cfg.gt_obs_default_mu),
            dtype=torch.float,
            device=self.device,
        )
        self._feet_robot_body_to_foot_offsets_b = self._build_feet_robot_body_to_foot_offsets_b(
            self._feet_robot_body_names
        )
        self._thigh_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        self._joint_wrench_body_ids, _ = self._robot.find_bodies([".*_thigh", ".*_calf"])

        self._current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._active_camera_name = self.cfg.active_camera

        if self.cfg.enable_inference_cameras:
            self._camera_views = {
                "overhead": self._overhead_camera,
                "side": self._side_camera,
            }
        else:
            self._camera_views = {}

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "gate_progress",
                "bodyrate_penalty",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "foot_contact",
                "dense_reaction_force",
                "floor_collision",
                "pillar_collision",
                "reach_waypoint",
                "finish_reward",
            ]
        }

    def _configure_backward_force_curriculum(self) -> None:
        """Validate and initialize the global backward-force curriculum."""
        initial_force = float(self.cfg.backward_force)
        force_stages = tuple(float(force) for force in self.cfg.backward_force_curriculum)
        all_forces = (initial_force, *force_stages)
        if any(not math.isfinite(force) or force < 0.0 for force in all_forces):
            raise ValueError(
                "backward_force and backward_force_curriculum must contain finite non-negative forces in newtons, "
                f"got {all_forces}."
            )

        threshold = float(self.cfg.backward_force_curriculum_sr_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "backward_force_curriculum_sr_threshold must be finite and in [0, 1], "
                f"got {self.cfg.backward_force_curriculum_sr_threshold}."
            )
        if any(force > 0.0 for force in all_forces) and str(self.cfg.race_scene) != "straightSimple":
            raise ValueError("backward_force is only supported when race_scene='straightSimple'.")

        self._backward_force_curriculum = force_stages
        self._backward_force_curriculum_stage = 0
        self._current_backward_force = initial_force

    def update_backward_force_curriculum(self, success_rate: float) -> bool:
        """Advance one force stage when the logged episode success rate clears the configured threshold."""
        success_rate = float(success_rate)
        if not math.isfinite(success_rate) or not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"Episode/successRate must be finite and in [0, 1], got {success_rate}.")
        if self._backward_force_curriculum_stage >= len(self._backward_force_curriculum):
            return False
        if success_rate <= float(self.cfg.backward_force_curriculum_sr_threshold):
            return False

        previous_force = self._current_backward_force
        self._current_backward_force = self._backward_force_curriculum[self._backward_force_curriculum_stage]
        self._backward_force_curriculum_stage += 1
        print(
            "[INFO] Backward-force curriculum advanced: "
            f"{previous_force:g} N -> {self._current_backward_force:g} N "
            f"(Episode/successRate={success_rate:.4f} > "
            f"{float(self.cfg.backward_force_curriculum_sr_threshold):.4f}; "
            f"stage {self._backward_force_curriculum_stage}/{len(self._backward_force_curriculum)}).",
            flush=True,
        )
        return True

    @property
    def current_backward_force(self) -> float:
        """Backward force currently applied by the environment, in newtons."""
        return self._current_backward_force

    def _configure_joint_position_limits(self) -> None:
        """Override the race USD's physical joint limits from the task configuration."""
        joint_types = []
        for joint_name in self.cfg.joint_names:
            joint_type = next(
                (name for name in ("hip", "thigh", "calf") if joint_name.endswith(f"_{name}_joint")), None
            )
            if joint_type is None:
                raise ValueError(f"Cannot select configured joint limits for unknown Solo12 joint '{joint_name}'.")
            joint_types.append(joint_type)

        physical_limits_degrees = torch.tensor(
            [getattr(self.cfg, f"joint_physical_limit_{joint_type}") for joint_type in joint_types],
            device=self._robot.data.joint_pos_limits.device,
            dtype=self._robot.data.joint_pos_limits.dtype,
        )
        if not torch.isfinite(physical_limits_degrees).all() or torch.any(
            physical_limits_degrees[:, 0] >= physical_limits_degrees[:, 1]
        ):
            raise ValueError(
                "Solo12 race physical joint limits must be finite (lower, upper) pairs with lower < upper; "
                f"got {physical_limits_degrees}."
            )

        physical_limits = (
            torch.deg2rad(physical_limits_degrees)
            .unsqueeze(0)
            .expand(self._robot.data.joint_pos_limits.shape[0], -1, -1)
        )
        self._robot.write_joint_position_limit_to_sim(physical_limits, joint_ids=self._joint_ids)
        print(
            "[INFO] Applied Solo12 race physical joint limits: "
            f"hip={self.cfg.joint_physical_limit_hip} deg, "
            f"thigh={self.cfg.joint_physical_limit_thigh} deg, "
            f"calf={self.cfg.joint_physical_limit_calf} deg.",
            flush=True,
        )

    def _apply_backward_force(self):
        """Refresh the straight-track opposing force at the floating-base COM."""
        force_magnitude = self._current_backward_force
        self._robot.permanent_wrench_composer.reset()
        if force_magnitude == 0.0:
            return

        straight_xy = self._track_waypoints_w[-1, :2] - self._track_waypoints_w[0, :2]
        straight_length = torch.linalg.vector_norm(straight_xy)
        if float(straight_length.item()) <= 1e-6:
            raise ValueError(
                "Cannot apply backward_force: waypoint_start and waypoint_end have identical XY positions."
            )

        forces_w = torch.zeros((self.num_envs, 1, 3), dtype=torch.float, device=self.device)
        forces_w[:, 0, :2] = -force_magnitude * straight_xy / straight_length
        # WrenchComposer stores forces in the link frame even when they are supplied globally. Refreshing it each
        # physics step keeps the force fixed in the world frame as the robot rotates.
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces_w,
            body_ids=[0],
            is_global=True,
        )

    def _setup_scene(self):
        self.cfg.scene_usd.func(self.cfg.scene_prim_path, self.cfg.scene_usd)
        stage = sim_utils.get_current_stage()

        source_scene_root = self.cfg.scene_source_prim_path
        self._track_waypoint_names = self._discover_waypoint_names_from_scene(stage)
        patch_items = sorted(
            (
                str(prim.GetPath())[len(source_scene_root) :],
                prim.GetName(),
            )
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(f"{source_scene_root}/")
            and prim.GetName().startswith("patch")
            and prim.IsA(UsdGeom.Cube)
        )
        self._patch_rel_paths = [rel_path for rel_path, _ in patch_items]
        # Keep the real USD prim names.  The race USDs currently number patches from patch_01, while tensor/material
        # arrays are zero-indexed.  Generating synthetic names (patch_00, patch_01, ...) makes UI selection of
        # patch_01 map to tensor index 1, so clicking the first patch edits the second patch.
        self._patch_names = [patch_name for _, patch_name in patch_items]
        self._init_patch_material_cache()

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        sim_utils.activate_contact_sensors(f"{source_scene_root}/SoloFlat", stage=stage)

        contact_cfg = copy.deepcopy(self.cfg.contact_sensor)
        track_foot_contact_points = bool(getattr(contact_cfg, "track_contact_points", False))
        track_foot_friction_forces = bool(getattr(contact_cfg, "track_friction_forces", False))
        use_dense_reaction_force_reward = float(getattr(self.cfg, "scale_dense_reaction_force_reward", 0.0)) != 0.0
        # IsaacLab filtered contact data is one sensor body to many filtered bodies. The main SoloFlat sensor tracks
        # many robot bodies, so keep it unfiltered and create one filtered reaction sensor per foot below when needed.
        contact_cfg.track_contact_points = False
        contact_cfg.track_friction_forces = False
        contact_cfg.filter_prim_paths_expr = []
        self._contact_sensor = ContactSensor(contact_cfg)
        self.cfg.contact_sensor = contact_cfg
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if track_foot_contact_points or track_foot_friction_forces or use_dense_reaction_force_reward:
            floor_filter_paths = self._build_base_floor_filter_paths(stage)
            max_contact_count = max(int(getattr(self.cfg.contact_sensor, "max_contact_data_count_per_prim", 4)), 8)
            foot_reaction_update_period = float(
                getattr(
                    self.cfg,
                    "foot_reaction_sensor_update_period_s",
                    getattr(self.cfg, "foot_reaction_contact_sensor_update_period_s", 0.0),
                )
            )
            if foot_reaction_update_period <= 0.0:
                foot_reaction_update_period = float(
                    getattr(self.cfg, "foot_reaction_contact_sensor_update_period_s", 0.0)
                )
            if foot_reaction_update_period <= 0.0:
                foot_reaction_update_period = float(self.physics_dt)
            self._foot_reaction_contact_sensors = {}
            self._foot_reaction_contact_sensor_labels = ("FL", "FR", "RL", "RR")
            for label in self._foot_reaction_contact_sensor_labels:
                foot_cfg = copy.deepcopy(self.cfg.contact_sensor)
                foot_cfg.prim_path = f"/World/envs/env_.*/Scene/SoloFlat/{label}_calf"
                foot_cfg.history_length = 0
                foot_cfg.update_period = foot_reaction_update_period
                foot_cfg.track_air_time = False
                foot_cfg.track_contact_points = track_foot_contact_points
                foot_cfg.track_friction_forces = track_foot_friction_forces
                foot_cfg.max_contact_data_count_per_prim = max_contact_count
                foot_cfg.filter_prim_paths_expr = floor_filter_paths
                sensor_name = f"{label.lower()}_foot_reaction_contact_sensor"
                sensor = ContactSensor(foot_cfg)
                self._foot_reaction_contact_sensors[label] = sensor
                self.scene.sensors[sensor_name] = sensor

        pillar_contact_cfg = copy.deepcopy(self.cfg.base_pillar_contact_sensor)
        pillar_contact_cfg.filter_prim_paths_expr = self._build_base_pillar_filter_paths()
        self._base_pillar_contact_sensor = ContactSensor(pillar_contact_cfg)
        self.scene.sensors["base_pillar_contact_sensor"] = self._base_pillar_contact_sensor

        floor_contact_cfg = copy.deepcopy(self.cfg.base_floor_contact_sensor)
        floor_contact_cfg.filter_prim_paths_expr = self._build_base_floor_filter_paths(stage)
        self._base_floor_contact_sensor = ContactSensor(floor_contact_cfg)
        self.scene.sensors["base_floor_contact_sensor"] = self._base_floor_contact_sensor

        if self.cfg.include_foot_imu_obs:
            self._imu_fl = Imu(self.cfg.imu_fl)
            self._imu_fr = Imu(self.cfg.imu_fr)
            self._imu_rl = Imu(self.cfg.imu_rl)
            self._imu_rr = Imu(self.cfg.imu_rr)
            self.scene.sensors["imu_fl"] = self._imu_fl
            self.scene.sensors["imu_fr"] = self._imu_fr
            self.scene.sensors["imu_rl"] = self._imu_rl
            self.scene.sensors["imu_rr"] = self._imu_rr

        if self.cfg.enable_inference_cameras:
            self._overhead_camera = TiledCamera(self.cfg.overhead_camera)
            self._side_camera = TiledCamera(self.cfg.side_camera)
            self.scene.sensors["overhead_camera"] = self._overhead_camera
            self.scene.sensors["side_camera"] = self._side_camera

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self._enable_patch_tensor_material_updates(stage)

        self._previous_base_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        num_patches = len(self._patch_rel_paths)
        self._patch_bucket_ids = torch.zeros((self.num_envs, num_patches), dtype=torch.long, device=self.device)
        self._patch_friction_static = torch.ones((self.num_envs, num_patches), dtype=torch.float, device=self.device)
        self._patch_friction_dynamic = torch.ones((self.num_envs, num_patches), dtype=torch.float, device=self.device)
        self._cache_patch_xy_bounds(stage)
        self._cache_waypoints_from_scene()
        self._sample_and_apply_track_layout(torch.arange(self.num_envs, device=self.device))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _discover_waypoint_names_from_scene(self, stage) -> tuple[str, ...]:
        scene_root = stage.GetPrimAtPath(self.cfg.scene_source_prim_path)
        if not scene_root.IsValid():
            return tuple(self._track_waypoint_names)

        numbered_waypoints: list[tuple[int, str]] = []
        has_start = False
        has_end = False
        for child in scene_root.GetChildren():
            name = child.GetName()
            if name == "waypoint_start":
                has_start = True
                continue
            if name == "waypoint_end":
                has_end = True
                continue
            match = re.fullmatch(r"waypoint_(\d+)", name)
            if match is not None:
                numbered_waypoints.append((int(match.group(1)), name))

        if not numbered_waypoints:
            endpoint_waypoints = []
            if has_start:
                endpoint_waypoints.append("waypoint_start")
            if has_end:
                endpoint_waypoints.append("waypoint_end")
            if len(endpoint_waypoints) >= 2:
                return tuple(endpoint_waypoints)
            return tuple(self._track_waypoint_names)

        numbered_waypoints.sort(key=lambda item: item[0])
        waypoint_names: list[str] = []
        if has_start:
            waypoint_names.append("waypoint_start")
        waypoint_names.extend(name for _, name in numbered_waypoints)
        if has_end:
            waypoint_names.append("waypoint_end")
        return tuple(waypoint_names)

    def _build_base_pillar_filter_paths(self) -> list[str]:
        filter_paths: list[str] = []
        for waypoint_name in self._track_waypoint_names:
            if waypoint_name == "waypoint_start":
                continue
            filter_paths.extend(
                [
                    f"/World/envs/env_.*/Scene/{waypoint_name}/c1",
                    f"/World/envs/env_.*/Scene/{waypoint_name}/c2",
                ]
            )
        return filter_paths

    def _build_base_floor_filter_paths(self, stage) -> list[str]:
        filter_paths = [f"/World/envs/env_.*/Scene{patch_rel_path}" for patch_rel_path in self._patch_rel_paths]

        ground_root = stage.GetPrimAtPath(f"{self.cfg.scene_source_prim_path}/ground")
        if ground_root.IsValid():
            for prim in ground_root.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    rel_path = str(prim.GetPath())[len(self.cfg.scene_source_prim_path) :]
                    filter_paths.append(f"/World/envs/env_.*/Scene{rel_path}")

        filter_paths = sorted(dict.fromkeys(filter_paths))
        if not filter_paths:
            raise ValueError("Could not resolve any floor contact filter paths for the race scene.")
        return filter_paths

    def _cache_waypoints_from_scene(self):
        stage = sim_utils.get_current_stage()
        xform_cache = UsdGeom.XformCache()
        waypoint_positions_w = []
        gate_pillars_w = []
        for waypoint_name in self._track_waypoint_names:
            prim = stage.GetPrimAtPath(f"{self.cfg.scene_source_prim_path}/{waypoint_name}")
            if not prim.IsValid():
                raise ValueError(f"Waypoint prim not found: {prim.GetPath()}")
            prim_tf = xform_cache.GetLocalToWorldTransform(prim)
            waypoint_positions_w.append(prim_tf.ExtractTranslation())

            if waypoint_name != "waypoint_start":
                pillars = self._try_find_gate_pillars(prim, xform_cache)
                if pillars is None:
                    raise ValueError(f"Could not find two gate pillars under {prim.GetPath()}")
                gate_pillars_w.append(pillars)

        waypoint_positions_w = torch.tensor(waypoint_positions_w, dtype=torch.float, device=self.device)
        self._track_waypoints_w = waypoint_positions_w - self.scene.env_origins[0]
        self._track_targets_w = self._track_waypoints_w[1:]
        self._gate_pillars_w = torch.stack(gate_pillars_w, dim=0) - self.scene.env_origins[0]

        track_xy = self._track_waypoints_w[:, :2]
        segment_vecs = track_xy[1:] - track_xy[:-1]
        self._segment_lengths = torch.norm(segment_vecs, dim=1)
        self._track_total_length = torch.sum(self._segment_lengths)
        self._segment_cumulative = torch.cat(
            [torch.zeros(1, device=self.device, dtype=self._segment_lengths.dtype), torch.cumsum(self._segment_lengths, dim=0)]
        )
        self._track_start_yaw = torch.atan2(segment_vecs[0, 1], segment_vecs[0, 0])
        self._gate_count = len(gate_pillars_w)
        self._target_count = len(self._track_targets_w)
        self.straight_track_start_to_end_distance_m = _straight_track_start_to_end_distance_m(
            str(self.cfg.race_scene), self._track_waypoints_w
        )
        if self.straight_track_start_to_end_distance_m is not None:
            print(
                "[INFO] Solo12 straightSimple track start-to-end distance: "
                f"{self.straight_track_start_to_end_distance_m:.6f} m.",
                flush=True,
            )

    def _try_find_gate_pillars(self, waypoint_prim, xform_cache):
        pillar_prims = [child for child in waypoint_prim.GetChildren() if child.GetName() in ("c1", "c2")]
        if len(pillar_prims) < 2:
            pillar_prims = [
                child
                for child in waypoint_prim.GetChildren()
                if child.GetName().startswith("c") and (child.IsA(UsdGeom.Gprim) or child.IsA(UsdGeom.Xform))
            ]
        if len(pillar_prims) < 2:
            pillar_prims = [
                child
                for child in waypoint_prim.GetChildren()
                if child.GetName() != "Looks" and (child.IsA(UsdGeom.Gprim) or child.IsA(UsdGeom.Xform))
            ]
        pillar_prims = sorted(pillar_prims, key=lambda prim: prim.GetName())[:2]
        if len(pillar_prims) < 2:
            return None
        pillar_positions_w = [xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation() for prim in pillar_prims]
        return torch.tensor(pillar_positions_w, dtype=torch.float, device=self.device)

    def _find_gate_pillars(self, waypoint_prim, xform_cache):
        pillars = self._try_find_gate_pillars(waypoint_prim, xform_cache)
        if pillars is None:
            raise ValueError(f"Could not find two gate pillars under {waypoint_prim.GetPath()}")
        return pillars

    def _cache_patch_xy_bounds(self, stage):
        if not self._patch_rel_paths:
            self._patch_xy_min = torch.empty(0, 2, device=self.device)
            self._patch_xy_max = torch.empty(0, 2, device=self.device)
            return

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        mins = []
        maxs = []
        env0_origin_xy = self.scene.env_origins[0, :2].detach().cpu()
        for patch_rel_path in self._patch_rel_paths:
            patch_prim = stage.GetPrimAtPath(f"{self.cfg.scene_source_prim_path}{patch_rel_path}")
            if not patch_prim.IsValid():
                continue
            aligned_range = bbox_cache.ComputeWorldBound(patch_prim).ComputeAlignedBox()
            patch_min = aligned_range.GetMin()
            patch_max = aligned_range.GetMax()
            mins.append(
                [float(patch_min[0]) - float(env0_origin_xy[0]), float(patch_min[1]) - float(env0_origin_xy[1])]
            )
            maxs.append(
                [float(patch_max[0]) - float(env0_origin_xy[0]), float(patch_max[1]) - float(env0_origin_xy[1])]
            )

        if len(mins) != len(self._patch_rel_paths):
            raise ValueError("Could not cache XY bounds for all Solo12 race friction patches.")
        self._patch_xy_min = torch.tensor(mins, dtype=torch.float, device=self.device)
        self._patch_xy_max = torch.tensor(maxs, dtype=torch.float, device=self.device)

    def _friction_rand(self, shape: tuple[int, ...]) -> torch.Tensor:
        generator = self._get_friction_generator()
        if generator is None:
            return torch.rand(*shape, device=self.device)
        return torch.rand(*shape, device=self.device, generator=generator)

    def _get_friction_generator(self) -> torch.Generator | None:
        seed = getattr(self.cfg, "friction_seed", None)
        if seed is None:
            return None
        if self._friction_generator is None:
            self._friction_generator = torch.Generator(device=self.device)
            self._friction_generator.manual_seed(int(seed))
        return self._friction_generator

    def _init_patch_material_cache(self):
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            self._patch_material_paths = []
            self._patch_visual_material_paths = []
            self._patch_material_bucket_values = torch.empty(0, 3, device=self.device)
            return

        num_buckets = self.cfg.friction_num_buckets if self.cfg.randomize_fric_coefs else 1
        self._patch_material_bucket_values = torch.empty((num_buckets, 3), device=self.device)
        # Dynamic friction is always a fixed ratio of the static coefficient (train and inference alike),
        # regardless of whether the static coefficient itself is randomized.
        dynamic_ratio = float(self.cfg.mu_dynamic_static_ratio)
        if self.cfg.randomize_fric_coefs:
            static_range = torch.tensor(self.cfg.friction_static_range, device=self.device)
            if getattr(self.cfg, "randomize_friction_bucket_values", False):
                self._patch_material_bucket_values[:, 0] = (
                    self._friction_rand((num_buckets,)) * (static_range[1] - static_range[0]) + static_range[0]
                )
            else:
                # Stratified deterministic support: every seed sees the same coefficient grid.
                bucket_q = (torch.arange(num_buckets, device=self.device, dtype=torch.float) + 0.5) / num_buckets
                self._patch_material_bucket_values[:, 0] = bucket_q * (static_range[1] - static_range[0]) + static_range[0]
        else:
            self._patch_material_bucket_values[:, 0] = float(self.cfg.friction_static_range[0])
        self._patch_material_bucket_values[:, 1] = dynamic_ratio * self._patch_material_bucket_values[:, 0]
        self._patch_material_bucket_values[:, 2] = 0.0

        stage = sim_utils.get_current_stage()
        self._patch_material_paths = []
        self._patch_visual_material_paths = []
        for patch_idx in range(num_patches):
            bucket_paths: list[str] = []
            visual_bucket_paths: list[str] = []
            for bucket_idx in range(num_buckets):
                bucket_values = self._patch_material_bucket_values[bucket_idx]
                material_path = f"/World/Materials/Solo12RacePatches/patch_{patch_idx:02d}/bucket_{bucket_idx:03d}"
                material_prim = stage.GetPrimAtPath(material_path)
                if not material_prim.IsValid():
                    phys_cfg = sim_utils.RigidBodyMaterialCfg(
                        static_friction=float(bucket_values[0].item()),
                        dynamic_friction=float(bucket_values[1].item()),
                        restitution=0.0,
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                    )
                    phys_cfg.func(material_path, phys_cfg)
                else:
                    usd_physics_material_api = UsdPhysics.MaterialAPI(material_prim)
                    if not usd_physics_material_api:
                        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
                    physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
                    if not physx_material_api:
                        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
                    sim_utils.safe_set_attribute_on_usd_schema(
                        usd_physics_material_api, "static_friction", float(bucket_values[0].item()), camel_case=True
                    )
                    sim_utils.safe_set_attribute_on_usd_schema(
                        usd_physics_material_api, "dynamic_friction", float(bucket_values[1].item()), camel_case=True
                    )
                    sim_utils.safe_set_attribute_on_usd_schema(usd_physics_material_api, "restitution", 0.0, camel_case=True)
                    sim_utils.safe_set_attribute_on_usd_schema(
                        physx_material_api, "friction_combine_mode", "multiply", camel_case=True
                    )
                    sim_utils.safe_set_attribute_on_usd_schema(
                        physx_material_api, "restitution_combine_mode", "multiply", camel_case=True
                    )
                bucket_paths.append(material_path)
                visual_material_path = f"/World/Looks/Solo12RacePatches/patch_{patch_idx:02d}/bucket_{bucket_idx:03d}"
                visual_material = UsdShade.Material.Define(stage, visual_material_path)
                shader = UsdShade.Shader.Define(stage, f"{visual_material_path}/PreviewSurface")
                shader.CreateIdAttr("UsdPreviewSurface")
                color = self._friction_to_color(float(bucket_values[0].item()))
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                visual_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                visual_bucket_paths.append(visual_material_path)
            self._patch_material_paths.append(bucket_paths)
            self._patch_visual_material_paths.append(visual_bucket_paths)

    def _apply_patch_materials(
        self,
        bucket_ids: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        *,
        update_usd: bool = True,
    ):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = self._env_ids_tensor(env_ids)

        bucket_ids = torch.as_tensor(bucket_ids, device=self.device, dtype=torch.long)

        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return
        if bucket_ids.ndim == 1:
            bucket_ids = bucket_ids.reshape(1, -1)
        if bucket_ids.shape == (len(env_ids), 1):
            bucket_ids = bucket_ids.expand(-1, num_patches).clone()
        if bucket_ids.shape != (len(env_ids), num_patches):
            raise ValueError(
                f"Expected bucket_ids shape {(len(env_ids), num_patches)} or {(len(env_ids), 1)}, got {tuple(bucket_ids.shape)}"
            )
        bucket_ids = torch.clamp(bucket_ids, 0, max(0, int(self.cfg.friction_num_buckets) - 1))

        self._patch_bucket_ids[env_ids] = bucket_ids
        static_values = self._patch_material_bucket_values[bucket_ids, 0].to(self.device)
        dynamic_values = self._patch_material_bucket_values[bucket_ids, 1].to(self.device)
        self._patch_friction_static[env_ids] = static_values
        self._patch_friction_dynamic[env_ids] = dynamic_values

        if update_usd:
            stage = sim_utils.get_current_stage()
            for patch_idx, patch_name in enumerate(self._patch_names):
                for local_idx, env_id in enumerate(env_ids.tolist()):
                    bucket_idx = int(bucket_ids[local_idx, patch_idx].item())
                    patch_rel_path = self._patch_rel_paths[patch_idx]
                    patch_path = f"/World/envs/env_{env_id}/Scene{patch_rel_path}"
                    patch_prim = stage.GetPrimAtPath(patch_path)
                    material_path = self._patch_material_paths[patch_idx][bucket_idx]
                    physics_material = UsdShade.Material(stage.GetPrimAtPath(material_path))
                    physics_binding_api = UsdShade.MaterialBindingAPI.Apply(patch_prim)
                    physics_binding_api.Bind(
                        physics_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                        materialPurpose="physics",
                    )

                    visual_material_path = self._patch_visual_material_paths[patch_idx][bucket_idx]
                    visual_material = UsdShade.Material(stage.GetPrimAtPath(visual_material_path))
                    physics_binding_api.Bind(
                        visual_material,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    )

                    # The visual material already carries the friction color. Avoid writing displayColor/displayOpacity here:
                    # Fabric may warn about missing `primvars:displayColor:indices` on authored patch cubes.

        self._apply_patch_materials_to_physx(bucket_ids, env_ids)

    def _env_ids_tensor(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], device=self.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long).reshape(-1)

    def get_patch_index_from_name(self, patch_name: str) -> int | None:
        patch_name = str(patch_name)
        for patch_idx, rel_path in enumerate(self._patch_rel_paths):
            if rel_path.rstrip("/").split("/")[-1] == patch_name:
                return patch_idx
        for patch_idx, name in enumerate(self._patch_names):
            if str(name) == patch_name:
                return patch_idx
        return None

    def set_patch_friction_grouping(self, group_all_patches_single_bucket: bool) -> None:
        self.cfg.group_all_patches_single_bucket = bool(group_all_patches_single_bucket)

    def apply_uniform_patch_friction_bucket(
        self, bucket_idx: int, env_ids=None, *, update_usd: bool = True
    ) -> None:
        env_ids = self._env_ids_tensor(env_ids)
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return
        bucket_idx = int(max(0, min(int(bucket_idx), int(self.cfg.friction_num_buckets) - 1)))
        bucket_ids = torch.full((len(env_ids), num_patches), bucket_idx, device=self.device, dtype=torch.long)
        self._apply_patch_materials(bucket_ids, env_ids=env_ids, update_usd=update_usd)

    def apply_single_patch_friction_bucket(
        self, env_index: int, patch_idx: int, bucket_idx: int, *, update_usd: bool = True
    ) -> None:
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return
        patch_idx = int(patch_idx)
        if patch_idx < 0 or patch_idx >= num_patches:
            raise IndexError(f"patch_idx {patch_idx} is outside [0, {num_patches})")
        env_ids = torch.tensor([int(env_index)], device=self.device, dtype=torch.long)
        bucket_ids = self._patch_bucket_ids[env_ids].clone()
        bucket_ids[:, patch_idx] = int(max(0, min(int(bucket_idx), int(self.cfg.friction_num_buckets) - 1)))
        self._apply_patch_materials(bucket_ids, env_ids=env_ids, update_usd=update_usd)

    def resample_patch_friction(self, env_ids=None, *, update_usd: bool = True) -> None:
        env_ids = self._env_ids_tensor(env_ids)
        bucket_ids = self._sample_patch_friction_bucket_ids(len(env_ids))
        self._apply_patch_materials(bucket_ids, env_ids=env_ids, update_usd=update_usd)

    def set_within_episode_patch_friction_resampling(
        self,
        enabled: bool,
        env_ids=None,
        *,
        update_usd_on_resample: bool | None = None,
        resample_now: bool = False,
    ) -> None:
        self.cfg.within_episode_fric_resample = bool(enabled)
        if update_usd_on_resample is not None:
            self.cfg.within_episode_fric_resample_update_usd = bool(update_usd_on_resample)
        self._ensure_friction_resample_buffer()
        env_ids = self._env_ids_tensor(env_ids)
        if bool(enabled) and self._within_episode_friction_resample_enabled():
            if bool(resample_now):
                self.resample_patch_friction(
                    env_ids,
                    update_usd=bool(getattr(self.cfg, "within_episode_fric_resample_update_usd", False)),
                )
            self._schedule_next_patch_friction_resample(env_ids)
        else:
            self._next_friction_resample_time_s[env_ids] = float("inf")

    def _enable_patch_tensor_material_updates(self, stage: Usd.Stage):
        """Make patch colliders kinematic rigid bodies so PhysX material tensors can update them at reset time."""

        self._patch_physx_paths = []
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0 or not self.cfg.randomize_fric_coefs:
            return

        for env_id in range(self.num_envs):
            for patch_rel_path in self._patch_rel_paths:
                patch_path = f"/World/envs/env_{env_id}/Scene{patch_rel_path}"
                patch_prim = stage.GetPrimAtPath(patch_path)
                if not patch_prim.IsValid():
                    continue

                rigid_body_api = UsdPhysics.RigidBodyAPI(patch_prim)
                if not rigid_body_api:
                    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(patch_prim)
                if not PhysxSchema.PhysxRigidBodyAPI(patch_prim):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(patch_prim)
                mass_api = UsdPhysics.MassAPI(patch_prim)
                if not mass_api:
                    mass_api = UsdPhysics.MassAPI.Apply(patch_prim)

                # Static colliders are baked by PhysX and cannot have their material changed reliably after play.
                # Kinematic rigid bodies stay fixed but are visible to the Tensor API, so resets can update the
                # actual contact material instead of only USD bookkeeping/visuals.  The large authored mass is a
                # safety guard if a patch is ever made dynamic by accident; kinematic bodies themselves are immovable
                # by contacts regardless of mass.
                sim_utils.safe_set_attribute_on_usd_schema(
                    rigid_body_api, "rigid_body_enabled", True, camel_case=True
                )
                sim_utils.safe_set_attribute_on_usd_schema(
                    rigid_body_api, "kinematic_enabled", True, camel_case=True
                )
                sim_utils.safe_set_attribute_on_usd_schema(mass_api, "mass", 1.0e9, camel_case=True)
                self._patch_physx_paths.append(patch_path)

    def _get_patch_physx_view(self):
        if self._patch_physx_view is not None or self._patch_physx_view_failed:
            return self._patch_physx_view
        if len(self._patch_physx_paths) == 0:
            self._patch_physx_view_failed = True
            return None

        try:
            physics_sim_view = SimulationManager.get_physics_sim_view()
            if physics_sim_view is None:
                # Scene creation calls _apply_patch_materials before the simulator is playing. Retry on the first
                # real reset, after DirectRLEnv has started PhysX and tensor views are available.
                return None
            self._patch_physx_view = physics_sim_view.create_rigid_body_view(self._patch_physx_paths)
        except Exception as exc:
            print(f"[WARN] Solo12 race patch PhysX material view unavailable: {exc}")
            self._patch_physx_view = None
            self._patch_physx_view_failed = True
            return None

        if getattr(self._patch_physx_view, "_backend", None) is None:
            print("[WARN] Solo12 race patch PhysX material view did not match any rigid bodies.")
            self._patch_physx_view = None
            self._patch_physx_view_failed = True
        return self._patch_physx_view

    def _apply_patch_materials_to_physx(self, bucket_ids: torch.Tensor, env_ids: torch.Tensor):
        view = self._get_patch_physx_view()
        if view is None:
            return

        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return

        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        patch_offsets = torch.arange(num_patches, dtype=torch.long)
        view_indices = (env_ids_cpu[:, None] * num_patches + patch_offsets[None, :]).reshape(-1)

        material_values = self._patch_material_bucket_values[bucket_ids.reshape(-1)].to(
            device="cpu", dtype=torch.float32
        )
        materials = view.get_material_properties()
        if hasattr(materials, "detach"):
            materials = materials.detach().to(device="cpu")
        materials[view_indices, :, :] = material_values[:, None, :]
        view.set_material_properties(materials, view_indices)

    def get_patch_friction_summary(self, env_index: int = 0) -> list[dict[str, float | int | str]]:
        env_index = int(env_index)
        if self._patch_bucket_ids.numel() == 0:
            return []

        summary: list[dict[str, float | int | str]] = []
        for patch_idx, patch_name in enumerate(self._patch_names):
            summary.append(
                {
                    "patch": patch_name,
                    "patch_index": patch_idx,
                    "path": self._patch_rel_paths[patch_idx],
                    "bucket": int(self._patch_bucket_ids[env_index, patch_idx].item()),
                    "static": float(self._patch_friction_static[env_index, patch_idx].item()),
                    "dynamic": float(self._patch_friction_dynamic[env_index, patch_idx].item()),
                    "physics_material": self._patch_material_paths[patch_idx][int(self._patch_bucket_ids[env_index, patch_idx].item())],
                    "visual_material": self._patch_visual_material_paths[patch_idx][int(self._patch_bucket_ids[env_index, patch_idx].item())],
                }
            )
        return summary

    def _sample_patch_friction_bucket_ids(self, num: int) -> torch.Tensor:
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return torch.empty((int(num), 0), device=self.device, dtype=torch.long)

        if self.cfg.randomize_fric_coefs:
            bucket_shape = (num, 1) if getattr(self.cfg, "group_all_patches_single_bucket", False) else (num, num_patches)
            bucket_ids = torch.randint(
                0,
                self.cfg.friction_num_buckets,
                bucket_shape,
                device=self.device,
                dtype=torch.long,
                generator=self._get_friction_generator(),
            )
            if bucket_ids.shape[1] == 1:
                bucket_ids = bucket_ids.expand(-1, num_patches).clone()
        else:
            bucket_ids = torch.zeros((num, num_patches), device=self.device, dtype=torch.long)
        return bucket_ids

    def _sample_and_apply_track_layout(self, env_ids: torch.Tensor, *, update_usd: bool = True):
        env_ids = self._env_ids_tensor(env_ids)
        num_patches = len(self._patch_rel_paths)
        if num_patches == 0:
            return

        bucket_ids = self._sample_patch_friction_bucket_ids(len(env_ids))
        self._apply_patch_materials(bucket_ids, env_ids=env_ids, update_usd=update_usd)

    def _within_episode_friction_resample_enabled(self) -> bool:
        return (
            bool(getattr(self.cfg, "within_episode_fric_resample", False))
            and bool(getattr(self.cfg, "randomize_fric_coefs", False))
            and len(self._patch_rel_paths) > 0
            and int(getattr(self.cfg, "friction_num_buckets", 0)) > 0
        )

    def _friction_resample_time_range(self) -> tuple[float, float]:
        try:
            low, high = getattr(self.cfg, "within_episode_fric_resample_time_range")
            low = float(low)
            high = float(high)
        except Exception:
            low, high = 0.0, 0.0
        low = max(0.0, low)
        high = max(0.0, high)
        if high < low:
            low, high = high, low
        return low, high

    def _ensure_friction_resample_buffer(self):
        target_device = torch.device(self.device)
        if (
            self._next_friction_resample_time_s.shape != (self.num_envs,)
            or self._next_friction_resample_time_s.device != target_device
        ):
            self._next_friction_resample_time_s = torch.full(
                (self.num_envs,), float("inf"), dtype=torch.float, device=self.device
            )

    def _schedule_next_patch_friction_resample(self, env_ids: torch.Tensor):
        self._ensure_friction_resample_buffer()
        env_ids = self._env_ids_tensor(env_ids)
        if len(env_ids) == 0:
            return
        if not self._within_episode_friction_resample_enabled():
            self._next_friction_resample_time_s[env_ids] = float("inf")
            return

        low, high = self._friction_resample_time_range()
        if high <= 0.0:
            self._next_friction_resample_time_s[env_ids] = float("inf")
            return
        delay = self._friction_rand((len(env_ids),)) * (high - low) + low
        now = self.episode_length_buf[env_ids].to(dtype=torch.float) * float(self.step_dt)
        self._next_friction_resample_time_s[env_ids] = now + delay

    def _maybe_resample_patch_friction_within_episode(self):
        if not self._within_episode_friction_resample_enabled():
            return
        self._ensure_friction_resample_buffer()
        now = self.episode_length_buf.to(dtype=torch.float) * float(self.step_dt)
        env_ids = torch.nonzero(now >= self._next_friction_resample_time_s, as_tuple=False).squeeze(-1)
        if len(env_ids) == 0:
            return
        # During training this path may touch thousands of envs repeatedly.  The PhysX tensor material write is the
        # contact-critical update; skipping USD visual rebinding keeps within-episode randomization cheap and stable.
        # Interactive play can opt into USD updates so the patch colors/popups change at each timed resample too.
        # Use the env helper directly rather than _sample_and_apply_track_layout so play-mode UI monkey-patching of
        # reset-time behavior cannot accidentally disable true within-episode resampling.
        self.resample_patch_friction(
            env_ids,
            update_usd=bool(getattr(self.cfg, "within_episode_fric_resample_update_usd", False)),
        )
        self._schedule_next_patch_friction_resample(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids]
        self._processed_actions = self.cfg.action_scale * self._actions + default_joint_pos

    def _apply_action(self):
        self._apply_backward_force()
        if self._enable_actuation_delay:
            self._delayed_processed_actions = self._action_delay_buffer.compute(self._processed_actions)
        else:
            self._delayed_processed_actions = self._processed_actions
        self._robot.set_joint_position_target(self._delayed_processed_actions, joint_ids=self._joint_ids)

    def step(self, action: torch.Tensor):
        """Step the env while recording foot IMUs at every physics step."""
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self._record_foot_imu_sample()
            self._record_joint_state_sample()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._maybe_resample_patch_friction_within_episode()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _record_foot_imu_sample(self):
        if not self.cfg.include_foot_imu_obs or self._foot_imu_history_len == 0:
            return

        sample = self._read_foot_imu_sample()
        sample = self._maybe_corrupt_foot_imu_sample(sample)
        self._foot_imu_history = torch.roll(self._foot_imu_history, shifts=-1, dims=1)
        self._foot_imu_history[:, -1, :] = sample

    def _read_foot_imu_sample(self) -> torch.Tensor:
        imu_terms = []
        for imu in (self._imu_fl, self._imu_fr, self._imu_rl, self._imu_rr):
            imu_terms.append(imu.data.ang_vel_b)
            imu_terms.append(imu.data.lin_acc_b)
        return torch.cat(tuple(imu_terms), dim=-1)

    def _maybe_corrupt_foot_imu_sample(self, sample: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_observation_corruption:
            return sample

        noisy_sample = sample + self._foot_imu_bias
        noisy_terms = []
        for foot_idx in range(4):
            start = foot_idx * 6
            noisy_terms.append(self._maybe_corrupt(noisy_sample[:, start : start + 3], self.cfg.foot_imu_gyro_noise_range))
            noisy_terms.append(self._maybe_corrupt(noisy_sample[:, start + 3 : start + 6], self.cfg.foot_imu_acc_noise_range))
        return torch.cat(tuple(noisy_terms), dim=-1)

    def _sample_foot_imu_bias(self, env_ids: torch.Tensor):
        self._foot_imu_bias[env_ids] = 0.0
        if not self.cfg.enable_observation_corruption:
            return

        for foot_idx in range(4):
            start = foot_idx * 6
            self._foot_imu_bias[env_ids, start : start + 3].uniform_(*self.cfg.foot_imu_gyro_bias_range)
            self._foot_imu_bias[env_ids, start + 3 : start + 6].uniform_(*self.cfg.foot_imu_acc_bias_range)

    def _record_joint_state_sample(self):
        if not self.cfg.include_joint_state_history_obs or self._joint_state_history_len == 0:
            return

        self._joint_state_history = torch.roll(self._joint_state_history, shifts=-1, dims=1)
        self._joint_state_history[:, -1, :] = self._read_joint_state_history_sample()

    def _read_joint_state_history_sample(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        joint_pos_error = self._delayed_processed_actions - joint_pos
        joint_vel_error = -joint_vel
        joint_pos_error = self._maybe_corrupt(joint_pos_error, self.cfg.joint_pos_noise)
        joint_vel_error = self._maybe_corrupt(joint_vel_error, self.cfg.joint_vel_noise)
        return torch.cat((joint_pos_error, joint_vel_error), dim=-1)

    def _get_current_gate_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_idx = torch.clamp(self._current_gate_idx, max=max(self._target_count - 1, 0))
        gate_center_w = self._track_targets_w[target_idx].clone()
        gate_center_w[:, :2] += self.scene.env_origins[:, :2]

        gate_pillars_w = self._gate_pillars_w[target_idx].clone()
        gate_pillars_w[:, :, :2] += self.scene.env_origins[:, None, :2]
        return gate_center_w, gate_pillars_w, target_idx

    def _get_following_gate_pillars(self, target_idx: torch.Tensor) -> torch.Tensor:
        following_idx = torch.clamp(target_idx + 1, max=max(self._gate_count - 1, 0))
        following_gate_pillars_w = self._gate_pillars_w[following_idx].clone()
        following_gate_pillars_w[:, :, :2] += self.scene.env_origins[:, None, :2]
        return following_gate_pillars_w

    def _get_closest_pillar_vectors_b(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        flat_pillars = self._gate_pillars_w.reshape(-1, 3)
        base_xy_local = self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        dist2 = torch.sum(torch.square(flat_pillars[None, :, :2] - base_xy_local[:, None, :]), dim=-1)
        closest_ids = torch.topk(dist2, k=2, largest=False, dim=1).indices

        closest_pillars_w = flat_pillars[closest_ids].clone()
        closest_pillars_w[:, :, :2] += self.scene.env_origins[:, None, :2]
        closest_vectors_w = closest_pillars_w - self._robot.data.root_pos_w[:, None, :]

        root_quat_pillars_w = root_quat_w[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        closest_vectors_b = math_utils.quat_apply_inverse(root_quat_pillars_w, closest_vectors_w.reshape(-1, 3))
        closest_vectors_b = closest_vectors_b.reshape(self.num_envs, 2, 3)

        return closest_vectors_b

    def _project_to_track(self, base_xy_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num = base_xy_w.shape[0]
        best_dist2 = torch.full((num,), float("inf"), device=self.device)
        best_progress = torch.zeros(num, device=self.device)
        best_center = torch.zeros(num, 2, device=self.device)
        best_tangent = torch.zeros(num, 2, device=self.device)

        for seg_idx in range(len(self._segment_lengths)):
            start = self._track_waypoints_w[seg_idx, :2]
            seg_vec = self._track_waypoints_w[seg_idx + 1, :2] - start
            seg_len = torch.clamp(self._segment_lengths[seg_idx], min=1e-6)
            rel = base_xy_w - start
            t = torch.sum(rel * seg_vec, dim=1) / torch.square(seg_len)
            t = torch.clamp(t, 0.0, 1.0)
            proj = start + t.unsqueeze(-1) * seg_vec
            dist2 = torch.sum(torch.square(base_xy_w - proj), dim=1)
            progress = self._segment_cumulative[seg_idx] + t * seg_len
            mask = dist2 < best_dist2
            best_dist2 = torch.where(mask, dist2, best_dist2)
            best_progress = torch.where(mask, progress, best_progress)
            best_center = torch.where(mask.unsqueeze(-1), proj, best_center)
            tangent = seg_vec / seg_len
            best_tangent = torch.where(mask.unsqueeze(-1), tangent.expand_as(best_tangent), best_tangent)

        lateral_dist = torch.sqrt(best_dist2)
        center_w = torch.zeros(num, 3, device=self.device)
        center_w[:, :2] = best_center + self.scene.env_origins[:, :2]
        tangent_w = torch.zeros(num, 3, device=self.device)
        tangent_w[:, :2] = best_tangent
        return best_progress, center_w, tangent_w, lateral_dist

    def _track_point_at_progress(self, progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_idx = torch.bucketize(progress, self._segment_cumulative[1:], right=False)
        seg_idx = torch.clamp(seg_idx, max=len(self._segment_lengths) - 1)
        seg_start = self._track_waypoints_w[seg_idx, :2]
        seg_vec = self._track_waypoints_w[seg_idx + 1, :2] - seg_start
        seg_len = torch.clamp(self._segment_lengths[seg_idx], min=1e-6)
        local = torch.clamp(progress - self._segment_cumulative[seg_idx], min=0.0)
        t = torch.clamp(local / seg_len, 0.0, 1.0)
        pos_xy = seg_start + t.unsqueeze(-1) * seg_vec
        tangent_xy = seg_vec / seg_len.unsqueeze(-1)
        return pos_xy, tangent_xy

    def _friction_to_color(self, friction: float) -> tuple[float, float, float]:
        low = torch.tensor(self.cfg.friction_static_range[0], device=self.device)
        high = torch.tensor(self.cfg.friction_static_range[1], device=self.device)
        t = float(torch.clamp((torch.tensor(friction, device=self.device) - low) / torch.clamp(high - low, min=1e-6), 0.0, 1.0).item())
        low_rgb = torch.tensor(self.cfg.friction_color_low, device=self.device)
        high_rgb = torch.tensor(self.cfg.friction_color_high, device=self.device)
        rgb = (1.0 - t) * low_rgb + t * high_rgb
        return float(rgb[0].item()), float(rgb[1].item()), float(rgb[2].item())

    def _compute_contact_count(self, body_ids: list[int], threshold: float) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)

    def _compute_filtered_base_contact(self, sensor: ContactSensor, threshold: float) -> torch.Tensor:
        force_matrix_history = sensor.data.force_matrix_w_history
        if force_matrix_history is not None:
            contact_force_norm = torch.norm(force_matrix_history, dim=-1)
            return torch.amax(contact_force_norm, dim=(1, 2, 3)) > threshold

        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        contact_force_norm = torch.norm(force_matrix, dim=-1)
        return torch.amax(contact_force_norm, dim=(1, 2)) > threshold

    def _compute_force_transmited_through_joints(self) -> torch.Tensor:
        incoming_joint_wrench_b = self._robot.data.body_incoming_joint_wrench_b[:, self._joint_wrench_body_ids]
        incoming_joint_force_b = incoming_joint_wrench_b[..., :3]
        return torch.sum(torch.sum(torch.square(incoming_joint_force_b), dim=-1), dim=1)

    def _compute_foot_contact_penalty(self) -> torch.Tensor:
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_body_ids, :]
        contact_force_norm = torch.norm(contact_forces, dim=-1)
        excess_force = torch.clamp(contact_force_norm - self.cfg.foot_contact_safe_threshold, min=0.0)
        if self.cfg.square_foot_penalty:
            excess_force = excess_force**2
        return torch.sum(excess_force, dim=1)

    def _get_foot_friction_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the static and dynamic floor-friction coefficients below each foot."""
        num_feet = len(self._feet_robot_body_ids)
        default_static = float(self.cfg.sim.physics_material.static_friction)
        default_dynamic = float(self.cfg.sim.physics_material.dynamic_friction)
        if self._patch_friction_static.numel() == 0 or self._patch_xy_min.numel() == 0:
            shape = (self.num_envs, num_feet)
            return (
                torch.full(shape, default_static, device=self.device),
                torch.full(shape, default_dynamic, device=self.device),
            )

        foot_xy = self._get_foot_positions_w()[:, :, :2] - self.scene.env_origins[:, None, :2]
        inside_patch = torch.logical_and(
            foot_xy[:, :, None, :] >= self._patch_xy_min[None, None, :, :],
            foot_xy[:, :, None, :] <= self._patch_xy_max[None, None, :, :],
        ).all(dim=-1)
        has_patch = inside_patch.any(dim=-1)
        patch_ids = torch.argmax(inside_patch.to(torch.long), dim=-1)
        static_friction = torch.gather(self._patch_friction_static, dim=1, index=patch_ids)
        dynamic_friction = torch.gather(self._patch_friction_dynamic, dim=1, index=patch_ids)
        return (
            torch.where(has_patch, static_friction, torch.full_like(static_friction, default_static)),
            torch.where(has_patch, dynamic_friction, torch.full_like(dynamic_friction, default_dynamic)),
        )

    def _compute_dense_reaction_force_reward(self) -> torch.Tensor:
        reaction_forces = []
        for label in self._foot_reaction_contact_sensor_labels:
            force_matrix_w = self._foot_reaction_contact_sensors[label].data.force_matrix_w
            if force_matrix_w is None:
                reaction_forces.append(torch.zeros(self.num_envs, 3, device=self.device))
            else:
                reaction_forces.append(torch.sum(force_matrix_w, dim=(1, 2)))
        reaction_forces_w = torch.stack(reaction_forces, dim=1)
        foot_quat_w = self._robot.data.body_quat_w[:, self._feet_robot_body_ids, :]
        foot_forward_b = torch.zeros_like(reaction_forces_w)
        foot_forward_b[..., 0] = 1.0
        foot_forward_axes_w = math_utils.quat_apply(
            foot_quat_w.reshape(-1, 4), foot_forward_b.reshape(-1, 3)
        ).reshape_as(reaction_forces_w)
        mu_static, mu_dynamic = self._get_foot_friction_coefficients()
        return dense_reaction_force_reward(
            reaction_forces_w,
            foot_forward_axes_w,
            mu_static,
            mu_dynamic,
            self.cfg.base_contact_threshold,
        )

    def _get_gt_foot_contact_forces_obs(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        forces_w = self._contact_sensor.data.net_forces_w[:, self._feet_body_ids, :]
        num_feet = forces_w.shape[1]
        root_quat_feet_w = root_quat_w[:, None, :].expand(-1, num_feet, -1).reshape(-1, 4)
        forces_b = math_utils.quat_apply_inverse(root_quat_feet_w, forces_w.reshape(-1, 3)).reshape(
            self.num_envs, num_feet, 3
        )
        # Raw body-frame forces: the actor's EmpiricalNormalization already brings them to ~unit
        # variance per dimension, so a constant prescale was redundant (and, via the normalizer's
        # eps floor, mildly attenuated the small tangential/friction components).
        if self.cfg.include_deprecated_force_normalization:
            # Compatibility path for checkpoints trained with the old /100 prescale.
            forces_b = forces_b / 100.0
        return forces_b.reshape(self.num_envs, -1)

    def _build_feet_robot_body_to_foot_offsets_b(self, foot_body_names: list[str]) -> torch.Tensor:
        imu_offsets_by_body_name = {
            "FL_calf": self.cfg.imu_fl.offset.pos,
            "FR_calf": self.cfg.imu_fr.offset.pos,
            "RL_calf": self.cfg.imu_rl.offset.pos,
            "RR_calf": self.cfg.imu_rr.offset.pos,
        }
        offsets = []
        for body_name in foot_body_names:
            short_name = str(body_name).split("/")[-1]
            if short_name not in imu_offsets_by_body_name:
                raise ValueError(f"No foot offset is configured for robot body '{body_name}'.")
            offsets.append(imu_offsets_by_body_name[short_name])
        return torch.tensor(offsets, dtype=torch.float32, device=self.device)

    def _get_foot_positions_w(self) -> torch.Tensor:
        calf_pos_w = self._robot.data.body_pos_w[:, self._feet_robot_body_ids, :]
        calf_quat_w = self._robot.data.body_quat_w[:, self._feet_robot_body_ids, :]
        foot_offsets_b = self._feet_robot_body_to_foot_offsets_b.to(dtype=calf_pos_w.dtype)
        foot_offsets_w = math_utils.quat_apply(calf_quat_w, foot_offsets_b.unsqueeze(0).expand(self.num_envs, -1, -1))
        return calf_pos_w + foot_offsets_w

    def _get_feet_contact_mask(self) -> torch.Tensor:
        contact_forces_history = self._contact_sensor.data.net_forces_w_history
        if contact_forces_history is not None:
            contact_force_norm = torch.norm(contact_forces_history[:, :, self._feet_body_ids], dim=-1)
            return torch.amax(contact_force_norm, dim=1) > self.cfg.base_contact_threshold

        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_body_ids, :]
        return torch.norm(contact_forces, dim=-1) > self.cfg.base_contact_threshold

    def _get_gt_patch_mu_obs(self) -> torch.Tensor:
        if self._patch_friction_static.numel() == 0 or self._patch_xy_min.numel() == 0:
            return torch.full((self.num_envs, 4), self.cfg.gt_obs_default_mu, device=self.device)

        if self._gt_patch_mu_latched.shape != (self.num_envs, len(self._feet_body_ids)):
            self._gt_patch_mu_latched = torch.full(
                (self.num_envs, len(self._feet_body_ids)),
                float(self.cfg.gt_obs_default_mu),
                dtype=torch.float,
                device=self.device,
            )

        foot_xy = self._get_foot_positions_w()[:, :, :2] - self.scene.env_origins[:, None, :2]
        inside_patch = torch.logical_and(
            foot_xy[:, :, None, :] >= self._patch_xy_min[None, None, :, :],
            foot_xy[:, :, None, :] <= self._patch_xy_max[None, None, :, :],
        ).all(dim=-1)

        has_patch = inside_patch.any(dim=-1)
        patch_ids = torch.argmax(inside_patch.to(torch.long), dim=-1)
        mu = torch.gather(self._patch_friction_static, dim=1, index=patch_ids)
        default_mu = torch.full_like(mu, self.cfg.gt_obs_default_mu)
        measured_mu = torch.where(has_patch, mu, default_mu)

        if self.cfg.teachers_sees_future_friction_coef:
            return measured_mu

        contact_mask = self._get_feet_contact_mask()
        self._gt_patch_mu_latched = torch.where(contact_mask, measured_mu, self._gt_patch_mu_latched)
        return self._gt_patch_mu_latched

    def _get_gt_env_params_obs(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        obs_terms = []
        if self.cfg.include_forces_to_gt_obs:
            obs_terms.append(self._get_gt_foot_contact_forces_obs(root_quat_w))
        if self.cfg.include_mu_coefs_to_gt_obs:
            obs_terms.append(self._get_gt_patch_mu_obs())
        if not obs_terms:
            return torch.empty(self.num_envs, 0, device=self.device)
        return torch.cat(tuple(obs_terms), dim=-1)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        root_quat_w = self._robot.data.root_quat_w

        _, gate_pillars_w, target_idx = self._get_current_gate_data()
        following_gate_pillars_w = self._get_following_gate_pillars(target_idx)
        r_c1_b = math_utils.quat_apply_inverse(root_quat_w, gate_pillars_w[:, 0] - self._robot.data.root_pos_w)
        r_c2_b = math_utils.quat_apply_inverse(root_quat_w, gate_pillars_w[:, 1] - self._robot.data.root_pos_w)
        r_c3_b = math_utils.quat_apply_inverse(
            root_quat_w, following_gate_pillars_w[:, 0] - self._robot.data.root_pos_w
        )
        r_c4_b = math_utils.quat_apply_inverse(
            root_quat_w, following_gate_pillars_w[:, 1] - self._robot.data.root_pos_w
        )

        obs_terms = []
        if self.cfg.include_root_lin_vel_b_obs:
            obs_terms.append(self._maybe_corrupt(self._robot.data.root_lin_vel_b, self.cfg.base_lin_vel_noise))
        obs_terms.extend(
            [
                self._maybe_corrupt(self._robot.data.root_ang_vel_b, self.cfg.base_ang_vel_noise),
                self._maybe_corrupt(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise),
                self._maybe_corrupt(joint_pos, self.cfg.joint_pos_noise),
                self._maybe_corrupt(joint_vel, self.cfg.joint_vel_noise),
                self._actions,
                r_c1_b,
                r_c2_b,
                r_c3_b,
                r_c4_b,
            ]
        )

        if not getattr(self.cfg, "remove_c_close_vectors_from_observation", False):
            c_close_vectors_b = self._get_closest_pillar_vectors_b(root_quat_w)
            obs_terms.extend([c_close_vectors_b[:, 0], c_close_vectors_b[:, 1]])

        if self.cfg.include_forces_to_gt_obs or self.cfg.include_mu_coefs_to_gt_obs:
            obs_terms.append(self._get_gt_env_params_obs(root_quat_w))

        if self.cfg.include_foot_imu_obs and self.cfg.include_joint_state_history_obs:
            joint_imu_history = torch.cat((self._joint_state_history, self._foot_imu_history), dim=-1)
            obs_terms.append(joint_imu_history.reshape(self.num_envs, -1))
        elif self.cfg.include_foot_imu_obs:
            obs_terms.append(self._foot_imu_history.reshape(self.num_envs, -1))
        elif self.cfg.include_joint_state_history_obs:
            obs_terms.append(self._joint_state_history.reshape(self.num_envs, -1))

        obs = torch.cat(tuple(obs_terms), dim=-1)
        self._previous_actions = self._actions.clone()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        base_pos_w = self._robot.data.root_pos_w
        base_lin_vel_b = self._robot.data.root_lin_vel_b
        base_ang_vel_b = self._robot.data.root_ang_vel_b

        gate_center_w, gate_pillars_w, target_idx = self._get_current_gate_data()
        prev_dist = torch.norm(self._previous_base_pos_w[:, :2] - gate_center_w[:, :2], dim=1)
        curr_dist = torch.norm(base_pos_w[:, :2] - gate_center_w[:, :2], dim=1)
        gate_progress = prev_dist - curr_dist
        bodyrate_penalty = self.cfg.gate_progress_bodyrate_coeff * torch.norm(base_ang_vel_b, dim=1)

        z_vel_error = torch.square(base_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(base_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._joint_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_body_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_body_ids]
        feet_air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1)
        feet_air_time *= torch.norm(base_lin_vel_b[:, :2], dim=1) > 0.1

        undesired_contacts = self._compute_contact_count(self._thigh_body_ids, self.cfg.undesired_contact_threshold)
        foot_contact = self._compute_foot_contact_penalty()
        if self.cfg.scale_dense_reaction_force_reward != 0.0:
            dense_reaction_force = self._compute_dense_reaction_force_reward()
        else:
            dense_reaction_force = torch.zeros(self.num_envs, device=self.device)

        pillar_collision = self._compute_filtered_base_contact(self._base_pillar_contact_sensor, self.cfg.base_contact_threshold)
        floor_collision = self._compute_filtered_base_contact(self._base_floor_contact_sensor, self.cfg.base_contact_threshold)

        pass_radius = torch.full_like(curr_dist, self.cfg.finish_radius)
        pass_radius = torch.where(target_idx < self._gate_count, torch.full_like(curr_dist, self.cfg.gate_radius), pass_radius)
        gate_passed = (curr_dist < pass_radius) & (self._current_gate_idx < self._target_count)
        self._current_gate_idx[gate_passed] += 1

        finished = self._current_gate_idx >= self._target_count

        rewards = {
            "gate_progress": self.cfg.forward_progress_reward_scale * gate_progress,
            "bodyrate_penalty": -bodyrate_penalty,
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale,
            "flat_orientation_l2": flat_orientation * self.cfg.base_tilt_penalty_reward_scale * self.step_dt,
            "foot_contact": foot_contact * self.cfg.foot_contact_reward_scale * self.step_dt,
            "dense_reaction_force": dense_reaction_force * self.cfg.scale_dense_reaction_force_reward,
            "floor_collision": floor_collision.float() * self.cfg.floor_collision_penalty,
            "pillar_collision": pillar_collision.float() * self.cfg.pillar_collision_penalty,
            "reach_waypoint": gate_passed.float() * self.cfg.reward_reach_waypoint,
            "finish_reward": finished.float() * self.cfg.finish_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        step_log = {f"RewardsPerStep/{key}": torch.mean(value).item() for key, value in rewards.items()}
        step_log["RewardsPerStep/total"] = torch.mean(reward).item()
        self.extras["log"] = step_log

        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._previous_base_pos_w = base_pos_w.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_pos_w = self._robot.data.root_pos_w
        pillar_collision = self._compute_filtered_base_contact(self._base_pillar_contact_sensor, self.cfg.base_contact_threshold)
        floor_collision = self._compute_filtered_base_contact(self._base_floor_contact_sensor, self.cfg.base_contact_threshold)
        finished = self._current_gate_idx >= self._target_count
        terminated = floor_collision | finished
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        episode_finished = self._current_gate_idx[env_ids] >= self._target_count
        episode_floor_collision = self._compute_filtered_base_contact(
            self._base_floor_contact_sensor, self.cfg.base_contact_threshold
        )[env_ids]
        episode_terminated = self.reset_terminated[env_ids]
        episode_timed_out = self.reset_time_outs[env_ids]
        episode_completion = self._compute_episode_completion(env_ids)
        finish_ratio = torch.mean(episode_finished.float()).item()
        if torch.any(episode_finished):
            finish_time_steps = torch.mean(self.episode_length_buf[env_ids][episode_finished].float()).item()
        else:
            finish_time_steps = float(self.max_episode_length)
        finish_time_seconds = finish_time_steps * self.step_dt

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._delayed_processed_actions[env_ids] = 0.0
        self._current_gate_idx[env_ids] = 0
        if self.cfg.include_foot_imu_obs:
            self._foot_imu_history[env_ids] = 0.0
            self._sample_foot_imu_bias(env_ids)
        if self.cfg.include_joint_state_history_obs:
            self._joint_state_history[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        root_pose = self._robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 0] += self.cfg.reset_x_pos + self._track_waypoints_w[0, 0]
        root_pose[:, 1] += self.cfg.reset_y_pos + self._track_waypoints_w[0, 1]
        if getattr(self.cfg, "enable_reset_pose_randomization", False):
            yaw = self._track_start_yaw + torch.empty(num_resets, device=self.device).uniform_(
                -self.cfg.reset_yaw_noise, self.cfg.reset_yaw_noise
            )
        else:
            yaw = torch.full((num_resets,), float(self._track_start_yaw.item()), device=self.device)
        root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)

        root_velocity = torch.zeros_like(self._robot.data.default_root_state[env_ids, 7:])
        if getattr(self.cfg, "enable_reset_pose_randomization", False):
            root_velocity[:, 0:3].uniform_(*self.cfg.reset_base_lin_vel_range)
            root_velocity[:, 3:6].uniform_(*self.cfg.reset_base_ang_vel_range)

        if self._enable_actuation_delay:
            action_delays = torch.randint(
                low=self.cfg.actuation_delay_range[0],
                high=self.cfg.actuation_delay_range[1] + 1,
                size=(num_resets,),
                device=self.device,
                dtype=torch.int,
            )
        else:
            action_delays = torch.zeros(num_resets, device=self.device, dtype=torch.int)
        self._action_delay_steps[env_ids] = action_delays
        self._action_delay_buffer.set_time_lag(action_delays, env_ids)
        self._action_delay_buffer.reset(env_ids)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self._previous_base_pos_w[env_ids] = root_pose[:, :3]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos[:, self._joint_ids], joint_ids=self._joint_ids, env_ids=env_ids)

        self._sample_and_apply_track_layout(env_ids)
        self._schedule_next_patch_friction_resample(env_ids)
        if self._gt_patch_mu_latched.numel() > 0:
            self._gt_patch_mu_latched[env_ids] = float(self.cfg.gt_obs_default_mu)

        extras = dict(self.extras.get("log", {}))
        for key in self._episode_sums:
            extras[f"Episode_RewardPerSecond/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]).abs() / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        extras["Episode/gateProgressRatio"] = torch.mean(episode_completion).item()
        extras["Episode/finishRatio"] = finish_ratio
        extras["Episode/successRate"] = finish_ratio
        extras["Episode/finishTimeSteps"] = finish_time_steps
        extras["Episode/finishTimeSeconds"] = finish_time_seconds
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(episode_floor_collision).item()
        extras["Episode_Termination/floor_collision"] = torch.count_nonzero(episode_floor_collision).item()
        extras["Episode_Termination/finish"] = torch.count_nonzero(episode_finished).item()
        extras["Episode_Termination/terminated"] = torch.count_nonzero(episode_terminated).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(episode_timed_out).item()
        self.extras["log"] = extras

    def _compute_episode_completion(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._target_count <= 0:
            return torch.zeros(len(env_ids), device=self.device)
        return torch.clamp(self._current_gate_idx[env_ids].float() / self._target_count, max=1.0)

    def _maybe_corrupt(self, tensor: torch.Tensor, noise_range: tuple[float, float]) -> torch.Tensor:
        if not self.cfg.enable_observation_corruption or noise_range[0] == noise_range[1] == 0.0:
            return tensor
        return tensor + torch.empty_like(tensor).uniform_(noise_range[0], noise_range[1])
