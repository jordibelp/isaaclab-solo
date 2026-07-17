# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim import schemas as sim_schemas
from isaaclab.utils.buffers import DelayBuffer

from .solo12_env_cfg import Solo12EnvCfg


_EPISODE_REWARD_KEYS = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "action_rate_l2",
    "feet_air_time",
    "two_feet_above_height",
    "three_or_more_feet_contact",
    "undesired_contacts",
    "base_collision_terminal",
    "flat_orientation_l2",
    "track_base_height_exp",
    "force_transmited_through_joints",
    "foot_contact",
)


def _per_step_reward_ratios(
    rewards: dict[str, torch.Tensor], reward_scales: dict[str, float], step_dt: float
) -> dict[str, float]:
    """Remove reward scale and step duration from per-step reward metrics."""
    return {
        key: torch.mean(rewards[key]).item() / (scale * step_dt)
        for key, scale in reward_scales.items()
        if scale != 0.0
    }


def _world_velocity_in_heading_frame_xy(
    root_lin_vel_w: torch.Tensor, root_quat_w: torch.Tensor
) -> torch.Tensor:
    """Express horizontal world velocity in the base heading frame.

    The projected base lateral axis defines heading without extracting Euler yaw, which is
    singular when the two-feet robot approaches 90 degrees of pitch. Both resulting axes are
    parallel to the world XY plane, so vertical motion cannot satisfy a planar command.
    """
    base_lateral_axis_b = torch.zeros_like(root_lin_vel_w)
    base_lateral_axis_b[:, 1] = 1.0
    lateral_axis_xy_w = math_utils.quat_apply(root_quat_w, base_lateral_axis_b)[:, :2]
    lateral_axis_xy_w = lateral_axis_xy_w / torch.linalg.vector_norm(
        lateral_axis_xy_w, dim=1, keepdim=True
    ).clamp_min(torch.finfo(root_lin_vel_w.dtype).eps)
    forward_axis_xy_w = torch.stack((lateral_axis_xy_w[:, 1], -lateral_axis_xy_w[:, 0]), dim=1)
    root_lin_vel_xy_w = root_lin_vel_w[:, :2]
    return torch.stack(
        (
            torch.sum(root_lin_vel_xy_w * forward_axis_xy_w, dim=1),
            torch.sum(root_lin_vel_xy_w * lateral_axis_xy_w, dim=1),
        ),
        dim=1,
    )


def _sample_reset_root_rpy(
    num_resets: int,
    nominal_rpy: tuple[float, float, float],
    noise_half_ranges: tuple[float, float, float],
    device: str,
) -> torch.Tensor:
    """Sample independent symmetric uniform reset noise around a nominal root orientation."""
    rpy = torch.tensor(nominal_rpy, dtype=torch.float32, device=device).expand(num_resets, -1).clone()
    noise = torch.tensor(noise_half_ranges, dtype=torch.float32, device=device)
    if any(value != 0.0 for value in noise_half_ranges):
        rpy += torch.empty_like(rpy).uniform_(-1.0, 1.0) * noise
    return rpy


def _external_contact_forces(
    net_forces_w_history: torch.Tensor, self_forces_w_history: torch.Tensor
) -> torch.Tensor:
    """Remove forces from self-collisions from a body's net contact-force history."""
    return net_forces_w_history - torch.sum(self_forces_w_history, dim=-2)


def _combine_mass_properties_with_point_mass(
    original_mass: float,
    original_com: np.ndarray,
    original_diagonal_inertia: np.ndarray,
    original_principal_axes,
    point_mass: float,
    point_position: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Combine authored rigid-body mass properties with a point mass in the body frame."""
    total_mass = original_mass + point_mass
    combined_com = (original_mass * original_com + point_mass * point_position) / total_mass

    principal_axes_wxyz = torch.tensor(
        (original_principal_axes.GetReal(), *original_principal_axes.GetImaginary()), dtype=torch.float64
    )
    principal_rotation = math_utils.matrix_from_quat(principal_axes_wxyz).numpy()
    original_inertia = principal_rotation @ np.diag(original_diagonal_inertia) @ principal_rotation.T

    def parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
        return mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))

    combined_inertia = original_inertia
    combined_inertia += parallel_axis(original_mass, original_com - combined_com)
    combined_inertia += parallel_axis(point_mass, point_position - combined_com)
    diagonal_inertia, principal_rotation = np.linalg.eigh(combined_inertia)
    if np.linalg.det(principal_rotation) < 0.0:
        principal_rotation[:, 0] *= -1.0
    principal_axes = tuple(
        float(value) for value in math_utils.quat_from_matrix(torch.from_numpy(principal_rotation)).tolist()
    )
    return total_mass, combined_com, diagonal_inertia, principal_axes


class Solo12Env(DirectRLEnv):
    cfg: Solo12EnvCfg

    def __init__(self, cfg: Solo12EnvCfg, render_mode: str | None = None, **kwargs):
        cfg.refresh_runtime_dependent_config()
        cfg.apply_events_randomization_setting()
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = torch.zeros_like(self._actions)
        self._delayed_processed_actions = torch.zeros_like(self._actions)
        self._applied_actions = torch.zeros_like(self._actions)

        curriculum_two_feet = self._curriculum_two_feet_enabled()
        self._validate_actuation_delay_range(self.cfg.actuation_delay_range, "actuation_delay_range")
        if curriculum_two_feet:
            self._validate_two_feet_curriculum_config()

        max_action_delay = self.cfg.actuation_delay_range[1]
        if curriculum_two_feet:
            max_action_delay = max(max_action_delay, self._max_two_feet_curriculum_action_delay())
        self._action_delay_buffer = DelayBuffer(max_action_delay, self.num_envs, device=self.device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_resample_interval = max(1, int(round(self.cfg.command_resampling_time_s / self.step_dt)))
        if not 0.0 <= self.cfg.opposite_direction_cmd_prob <= 1.0:
            raise ValueError(
                "opposite_direction_cmd_prob must be between 0 and 1, "
                f"got {self.cfg.opposite_direction_cmd_prob}"
            )

        self._joint_ids, _ = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        self._base_body_ids, _ = self._contact_sensor.find_bodies("base")
        self._feet_body_ids, self._feet_body_names = self._contact_sensor.find_bodies(".*_calf")
        self._feet_robot_body_ids, self._feet_robot_body_names = self._robot.find_bodies(".*_calf")
        self._front_feet_contact_indices = self._find_feet_indices(self._feet_body_names, ("FL", "FR"))
        self._rear_feet_contact_indices = self._find_feet_indices(self._feet_body_names, ("RL", "RR"))
        self._front_feet_robot_indices = self._find_feet_indices(self._feet_robot_body_names, ("FL", "FR"))
        self._rear_feet_robot_indices = self._find_feet_indices(self._feet_robot_body_names, ("RL", "RR"))
        self._feet_robot_body_to_foot_offsets_b = self._build_feet_robot_body_to_foot_offsets_b(
            self._feet_robot_body_names
        )
        self._thigh_body_ids, self._thigh_body_names = self._contact_sensor.find_bodies(".*_thigh")
        self._front_thigh_contact_indices = self._find_feet_indices(self._thigh_body_names, ("FL", "FR"))
        self._base_wrench_body_ids, _ = self._robot.find_bodies("base")
        self._joint_wrench_body_ids, _ = self._robot.find_bodies([".*_thigh", ".*_calf"])
        self._reset_joint_pos = self._build_reset_joint_pos()
        self._q_offset_action_and_obs = self._build_q_offset_action_and_obs()
        self._base_push_interval_step_range = self._seconds_range_to_steps(self.cfg.base_push_interval_range_s)
        self._base_push_duration_step_range = self._seconds_range_to_steps(self.cfg.base_push_duration_range_s)
        self._base_push_steps_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._base_push_steps_until_next = self._sample_int_steps(
            self._base_push_interval_step_range, self.num_envs
        )
        self._base_push_forces_b = torch.zeros(
            self.num_envs, len(self._base_wrench_body_ids), 3, device=self.device
        )
        self._base_push_application_points_b = torch.zeros_like(self._base_push_forces_b)
        self._base_push_application_half_extents = torch.tensor(
            self.cfg.base_push_application_half_extents, dtype=torch.float, device=self.device
        )
        if self._base_push_application_half_extents.shape != (3,) or torch.any(
            self._base_push_application_half_extents <= 0.0
        ):
            raise ValueError(
                "base_push_application_half_extents must contain three positive values, "
                f"got {self.cfg.base_push_application_half_extents}."
            )
        half_x, half_y, half_z = self._base_push_application_half_extents
        self._base_push_face_areas = torch.stack(
            (
                4.0 * half_x * half_y,
                4.0 * half_y * half_z,
                4.0 * half_y * half_z,
                4.0 * half_x * half_z,
                4.0 * half_x * half_z,
            )
        )
        self._max_velx_range_curriculum_values = self._parse_max_velx_range_curriculum()
        self._max_velx_range_curriculum_idx = 0
        self._base_push_force_curriculum_values = self._parse_base_push_force_curriculum()
        self._base_push_force_curriculum_idx = 0
        self._base_push_mean_reward_smooth: float | None = None
        self._base_push_last_curriculum_step = 0
        self._two_feet_curriculum_phase = 1 if curriculum_two_feet else 0
        self._tricky_terrain_active = False
        if curriculum_two_feet:
            self._set_two_feet_curriculum_phase(1)
        if self._max_velx_range_curriculum_values:
            self._set_max_velx_range_curriculum_level(0)
        if self._base_push_force_curriculum_values:
            self._set_base_push_force_curriculum_level(0)
        self._refresh_tricky_terrain_origins(force=True)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in _EPISODE_REWARD_KEYS
        }
        self._episode_reward_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._base_collision_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._forbidden_feet_contact_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._base_imu_history_len = self.cfg.base_imu_history_length if self.cfg.imu_raw_inputs else 0
        self._base_imu_history_sample_dim = self.cfg.base_imu_history_sample_dim
        self._base_imu_history = torch.zeros(
            self.num_envs,
            self._base_imu_history_len,
            self._base_imu_history_sample_dim,
            device=self.device,
        )
        self._base_imu_bias = torch.zeros(self.num_envs, 6, device=self.device)
        self._base_imu_orientation_bias_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_imu_heading_reference = torch.zeros(self.num_envs, 4, device=self.device)
        self._base_imu_heading_reference[:, 0] = 1.0
        self._imu_noise_scale = float(self.cfg.imu_noise_scale)
        self._base_imu_gravity_bias_w = torch.tensor(
            self.cfg.base_imu.gravity_bias, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self._gravity_direction_w = torch.tensor((0.0, 0.0, -1.0), dtype=torch.float, device=self.device).unsqueeze(0)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._configure_base_collision_filters()
        self._apply_configured_base_mass()
        self._apply_extra_mass_on_front_feet()
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._base_external_contact_sensor = ContactSensor(self.cfg.base_external_contact_sensor)
        self.scene.sensors["base_external_contact_sensor"] = self._base_external_contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._terrain_height_wp_mesh = self._build_terrain_height_query_mesh()
        self._spawn_flat_terrain_grid_overlay()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_base_collision_filters(self):
        """Apply ``base_filtered_pairs`` symmetrically to the source Solo12 USD prim.

        The asset filters the base against all hips and thighs by default. A PhysX filtered pair is
        reciprocal in this USD, so both the base-to-link and link-to-base relationship targets are
        updated. Directly connected hip--thigh and thigh--calf filters are left untouched.
        """
        if not bool(self.cfg.enabled_self_collisions):
            return

        allowed_groups = {"hip", "thigh"}
        requested_groups = set(self.cfg.base_filtered_pairs)
        unknown_groups = requested_groups - allowed_groups
        if unknown_groups:
            raise ValueError(
                f"base_filtered_pairs contains unsupported groups {sorted(unknown_groups)}; "
                f"expected a subset of {sorted(allowed_groups)}."
            )

        stage = sim_utils.get_current_stage()
        robot_paths = sim_utils.find_matching_prim_paths(self.cfg.robot.prim_path, stage=stage)
        if len(robot_paths) != 1:
            raise RuntimeError(
                "Expected exactly one source Solo12 prim before environment cloning, "
                f"found {robot_paths} for {self.cfg.robot.prim_path}."
            )
        robot_path = robot_paths[0]

        base_path = f"{robot_path}/base"
        grouped_link_paths = {
            group: {f"{robot_path}/{leg}_{group}" for leg in ("FL", "FR", "RL", "RR")}
            for group in allowed_groups
        }
        all_link_paths = set().union(*grouped_link_paths.values())
        desired_link_paths = set().union(*(grouped_link_paths[group] for group in requested_groups))

        base_prim = stage.GetPrimAtPath(base_path)
        if not base_prim.IsValid():
            raise RuntimeError(f"Could not find Solo12 base prim at {base_path}.")
        base_relationship = base_prim.GetRelationship("physics:filteredPairs")
        if not base_relationship:
            raise RuntimeError(f"Solo12 base prim at {base_path} has no physics:filteredPairs relationship.")
        base_targets = [target for target in base_relationship.GetTargets() if str(target) not in all_link_paths]
        base_relationship.SetTargets(base_targets + [Sdf.Path(path) for path in sorted(desired_link_paths)])

        for link_path in sorted(all_link_paths):
            link_prim = stage.GetPrimAtPath(link_path)
            if not link_prim.IsValid():
                raise RuntimeError(f"Could not find Solo12 link prim at {link_path}.")
            relationship = link_prim.GetRelationship("physics:filteredPairs")
            if not relationship:
                raise RuntimeError(f"Solo12 link prim at {link_path} has no physics:filteredPairs relationship.")
            targets = [target for target in relationship.GetTargets() if str(target) != base_path]
            group = link_path.rsplit("_", maxsplit=1)[-1]
            if group in requested_groups:
                targets.append(Sdf.Path(base_path))
            relationship.SetTargets(targets)

    def _spawn_flat_terrain_grid_overlay(self):
        if not bool(getattr(self.cfg, "flat_terrain_grid_enabled", False)):
            return

        terrain_origins = getattr(self._terrain, "terrain_origins", None)
        terrain_generator_cfg = getattr(self.cfg.terrain, "terrain_generator", None)

        if terrain_origins is not None and terrain_generator_cfg is not None:
            rows = tuple(int(row) for row in self.cfg.flat_terrain_grid_rows)
            cols = tuple(int(col) for col in self.cfg.flat_terrain_grid_cols)
            if not rows or not cols:
                return

            num_rows, num_cols = terrain_origins.shape[:2]
            invalid_rows = [row for row in rows if row < 0 or row >= num_rows]
            invalid_cols = [col for col in cols if col < 0 or col >= num_cols]
            if invalid_rows or invalid_cols:
                raise ValueError(
                    "flat_terrain_grid_rows/cols must index generated terrain origins; "
                    f"invalid rows={invalid_rows}, invalid cols={invalid_cols}, origins_shape={terrain_origins.shape}."
                )
            tile_size_x, tile_size_y = (float(value) for value in terrain_generator_cfg.size)
        else:
            # Fallback for plane terrain: spawn a large grid centered at the origin.
            rows = (0,)
            cols = (0,)
            terrain_origins = torch.zeros((1, 1, 3))
            tile_size_x, tile_size_y = (100.0, 100.0)

        tile_size_x, tile_size_y = (float(value) for value in (tile_size_x, tile_size_y))
        spacing = float(self.cfg.flat_terrain_grid_spacing)
        line_width = float(self.cfg.flat_terrain_grid_line_width)
        z_offset = float(self.cfg.flat_terrain_grid_z_offset)
        if spacing <= 0.0 or line_width <= 0.0:
            raise ValueError(
                "flat_terrain_grid_spacing and flat_terrain_grid_line_width must be positive, "
                f"got spacing={spacing}, line_width={line_width}."
            )

        grid_color = tuple(float(channel) for channel in self.cfg.flat_terrain_grid_color)
        if len(grid_color) != 3 or any(channel < 0.0 or channel > 1.0 for channel in grid_color):
            raise ValueError(f"flat_terrain_grid_color must contain three values in [0, 1], got {grid_color}.")

        grid_prim_path = f"{self.cfg.terrain.prim_path}/flat_grid_overlay"
        stage = sim_utils.get_current_stage()
        if stage.GetPrimAtPath(grid_prim_path).IsValid():
            return

        import numpy as np
        import trimesh
        from isaaclab.terrains import create_prim_from_mesh

        def line_offsets(length: float) -> list[float]:
            offsets = [min(i * spacing, length) for i in range(int(math.floor(length / spacing)) + 1)]
            if not math.isclose(offsets[-1], length):
                offsets.append(length)
            return offsets

        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        rectangle_keys: set[tuple[float, float, float, float, float]] = set()

        def add_rectangle(x_min: float, x_max: float, y_min: float, y_max: float, z: float):
            key = tuple(round(value, 6) for value in (x_min, x_max, y_min, y_max, z))
            if key in rectangle_keys:
                return
            rectangle_keys.add(key)

            vertex_start = len(vertices)
            vertices.extend(
                (
                    (x_min, y_min, z),
                    (x_max, y_min, z),
                    (x_max, y_max, z),
                    (x_min, y_max, z),
                )
            )
            faces.extend(
                (
                    (vertex_start, vertex_start + 1, vertex_start + 2),
                    (vertex_start, vertex_start + 2, vertex_start + 3),
                )
            )

        x_offsets = line_offsets(tile_size_x)
        y_offsets = line_offsets(tile_size_y)
        half_width = 0.5 * line_width

        for row in rows:
            for col in cols:
                origin = terrain_origins[row, col]
                if isinstance(origin, torch.Tensor):
                    origin = origin.detach().cpu().tolist()
                center_x, center_y, center_z = (float(value) for value in origin)
                x_min = center_x - 0.5 * tile_size_x
                y_min = center_y - 0.5 * tile_size_y
                z = center_z + z_offset

                for x_offset in x_offsets:
                    x = x_min + x_offset
                    add_rectangle(x - half_width, x + half_width, y_min, y_min + tile_size_y, z)
                for y_offset in y_offsets:
                    y = y_min + y_offset
                    add_rectangle(x_min, x_min + tile_size_x, y - half_width, y + half_width, z)

        if not vertices:
            return

        grid_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        grid_material = sim_utils.PreviewSurfaceCfg(diffuse_color=grid_color, roughness=0.8, metallic=0.0)

        create_prim_from_mesh(grid_prim_path, grid_mesh, visual_material=grid_material)
        sim_utils.define_collision_properties(
            f"{grid_prim_path}/mesh", sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        )

    def _apply_configured_base_mass(self):
        if self.cfg.base_mass is None:
            return

        source_robot_path = self.cfg.robot.prim_path.replace(self.scene.env_regex_ns, self.scene.env_prim_paths[0])
        source_base_path = f"{source_robot_path}/base"
        mass_cfg = sim_utils.MassPropertiesCfg(mass=self.cfg.base_mass)
        sim_schemas.define_mass_properties(source_base_path, mass_cfg)

    def _apply_extra_mass_on_front_feet(self):
        """Add a point mass at each front foot and update the calf mass properties consistently."""
        extra_mass = float(self.cfg.extra_mass_on_front_feet)
        if not math.isfinite(extra_mass) or extra_mass < 0.0:
            raise ValueError(f"extra_mass_on_front_feet must be finite and non-negative, got {extra_mass}.")
        if extra_mass == 0.0:
            return

        stage = sim_utils.get_current_stage()
        source_robot_path = self.cfg.robot.prim_path.replace(self.scene.env_regex_ns, self.scene.env_prim_paths[0])
        foot_positions = {
            "FL_calf": np.array((0.0, 0.009000003337860107, -0.1599999964237213)),
            "FR_calf": np.array((0.0, -0.009000003337860107, -0.1599999964237213)),
        }
        for body_name, foot_position in foot_positions.items():
            body_path = f"{source_robot_path}/{body_name}"
            body_prim = stage.GetPrimAtPath(body_path)
            if not body_prim.IsValid():
                raise RuntimeError(f"Could not find Solo12 front calf prim at {body_path}.")
            mass_api = UsdPhysics.MassAPI(body_prim)
            original_mass = mass_api.GetMassAttr().Get()
            original_com = mass_api.GetCenterOfMassAttr().Get()
            original_diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
            original_principal_axes = mass_api.GetPrincipalAxesAttr().Get()
            if any(value is None for value in (original_mass, original_com, original_diagonal_inertia, original_principal_axes)):
                raise RuntimeError(f"Solo12 front calf at {body_path} has incomplete authored mass properties.")

            total_mass, combined_com, diagonal_inertia, principal_axes = _combine_mass_properties_with_point_mass(
                float(original_mass),
                np.asarray(original_com, dtype=np.float64),
                np.asarray(original_diagonal_inertia, dtype=np.float64),
                original_principal_axes,
                extra_mass,
                foot_position,
            )
            mass_api.GetMassAttr().Set(total_mass)
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*combined_com))
            mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia))
            mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(principal_axes[0], *principal_axes[1:]))

    @staticmethod
    def _find_feet_indices(body_names: list[str], labels: tuple[str, ...]) -> tuple[int, ...]:
        indices_by_label = {}
        for index, body_name in enumerate(body_names):
            short_name = str(body_name).split("/")[-1]
            for label in labels:
                if short_name.startswith(f"{label}_"):
                    indices_by_label[label] = index

        missing_labels = [label for label in labels if label not in indices_by_label]
        if missing_labels:
            raise ValueError(
                f"Could not find Solo12 feet {missing_labels} in body names: "
                f"{[str(name).split('/')[-1] for name in body_names]}"
            )
        return tuple(indices_by_label[label] for label in labels)

    def _build_feet_robot_body_to_foot_offsets_b(self, foot_body_names: list[str]) -> torch.Tensor:
        foot_offsets_by_body_name = {
            "FL_calf": (0.0, 0.009000003337860107, -0.1599999964237213),
            "FR_calf": (0.0, -0.009000003337860107, -0.1599999964237213),
            "RL_calf": (0.0, 0.009000003337860107, -0.1599999964237213),
            "RR_calf": (0.0, -0.009000003337860107, -0.1599999964237213),
        }
        offsets = []
        for body_name in foot_body_names:
            short_name = str(body_name).split("/")[-1]
            if short_name not in foot_offsets_by_body_name:
                raise ValueError(f"No foot offset is configured for robot body '{body_name}'.")
            offsets.append(foot_offsets_by_body_name[short_name])
        return torch.tensor(offsets, dtype=torch.float32, device=self.device)

    def _build_terrain_height_query_mesh(self):
        needs_foot_height_query = (
            self.cfg.include_foot_height_obs or self.cfg.two_feet_above_height_reward_scale != 0.0
        )
        if not needs_foot_height_query:
            return None

        terrain_prim_paths = getattr(self._terrain, "terrain_prim_paths", None)
        if not terrain_prim_paths:
            return None

        import numpy as np
        import omni
        from pxr import UsdGeom

        from isaaclab.terrains.trimesh.utils import make_plane
        from isaaclab.utils.warp import convert_to_warp_mesh

        terrain_prim_path = terrain_prim_paths[0]
        plane_prim = sim_utils.get_first_matching_child_prim(
            terrain_prim_path, lambda prim: prim.GetTypeName() == "Plane"
        )
        if plane_prim is not None:
            plane_mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            return convert_to_warp_mesh(plane_mesh.vertices, plane_mesh.faces, device=self.device)

        mesh_prim = sim_utils.get_first_matching_child_prim(
            terrain_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            return None

        usd_mesh = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(usd_mesh)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
        return convert_to_warp_mesh(points, indices, device=self.device)

    def _build_reset_joint_pos(self) -> torch.Tensor:
        initial_position = self.cfg.initial_position.lower()
        joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        if initial_position == "rigid":
            return joint_pos
        if initial_position == "flexed":
            initial_joint_pos_cfg = self.cfg.flexed_initial_joint_pos
        else:
            initial_joint_pos_cfg = self.cfg.initial_joint_pos_by_name.get(initial_position)
        if initial_joint_pos_cfg is None:
            supported_positions = sorted(["rigid", *self.cfg.initial_joint_pos_by_name.keys()])
            raise ValueError(
                f"Unsupported initial_position '{self.cfg.initial_position}'. "
                f"Use one of: {', '.join(supported_positions)}."
            )

        initial_joint_pos = torch.tensor(
            [initial_joint_pos_cfg[joint_name] for joint_name in self.cfg.joint_names],
            device=self.device,
            dtype=joint_pos.dtype,
        )
        joint_pos[:] = initial_joint_pos
        return joint_pos

    def _build_q_offset_action_and_obs(self) -> torch.Tensor:
        offset_cfg = self.cfg.q_offset_action_and_obs
        joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()

        if offset_cfg is None:
            offset = self._reset_joint_pos[0]
        elif isinstance(offset_cfg, dict):
            offset = torch.tensor(
                [offset_cfg[joint_name] for joint_name in self.cfg.joint_names],
                device=self.device,
                dtype=joint_pos.dtype,
            )
        elif isinstance(offset_cfg, (int, float)):
            offset = torch.full((len(self._joint_ids),), float(offset_cfg), device=self.device, dtype=joint_pos.dtype)
        else:
            offset = torch.tensor(offset_cfg, device=self.device, dtype=joint_pos.dtype)
            if offset.numel() != len(self._joint_ids):
                raise ValueError(
                    "q_offset_action_and_obs must be a scalar, a joint-name dictionary, "
                    f"or have {len(self._joint_ids)} values, got {offset.numel()}."
                )

        joint_pos[:] = offset.reshape(1, -1)
        return joint_pos

    def _seconds_range_to_steps(self, seconds_range: tuple[float, float]) -> tuple[int, int]:
        low_s, high_s = seconds_range
        if low_s < 0.0 or high_s < low_s:
            raise ValueError(f"Invalid seconds range: {seconds_range}")
        low_steps = max(1, int(round(low_s / self.step_dt)))
        high_steps = max(low_steps, int(round(high_s / self.step_dt)))
        return low_steps, high_steps

    def _sample_int_steps(self, step_range: tuple[int, int], count: int) -> torch.Tensor:
        low, high = step_range
        if low == high:
            return torch.full((count,), low, dtype=torch.long, device=self.device)
        return torch.randint(low, high + 1, (count,), dtype=torch.long, device=self.device)

    @staticmethod
    def _validate_actuation_delay_range(delay_range: tuple[int, int], name: str):
        low, high = delay_range
        if low < 0 or high < low:
            raise ValueError(f"{name} must be a non-negative ordered range, got {delay_range}.")

    def _curriculum_two_feet_enabled(self) -> bool:
        return bool(getattr(self.cfg, "curriculum_two_feet", False))

    @staticmethod
    def _curriculum_values(values) -> tuple:
        return tuple(values)

    def _two_feet_curriculum_phase_count(self) -> int:
        return len(self._curriculum_values(self.cfg.two_feet_above_height_reward_scale_curriculum))

    def _two_feet_curriculum_phase_index(self, phase: int | None = None) -> int:
        phase = self._two_feet_curriculum_phase if phase is None else phase
        return int(phase) - 1

    def _two_feet_curriculum_value(self, name: str, phase: int | None = None):
        values = self._curriculum_values(getattr(self.cfg, f"{name}_curriculum"))
        return values[self._two_feet_curriculum_phase_index(phase)]

    def _max_two_feet_curriculum_action_delay(self) -> int:
        return max(int(delay_range[1]) for delay_range in self.cfg.actuation_delay_range_curriculum)

    def _two_feet_curriculum_advance_metric_key(self, transition_idx: int) -> str:
        use_vx = self._curriculum_values(self.cfg.two_feet_curriculum_advance_thresholds_vx_indicator)[transition_idx]
        return "track_lin_vel_xy_exp" if use_vx else "two_feet_above_height"

    def _validate_two_feet_curriculum_config(self):
        phase_fields = {
            "two_feet_above_height_reward_scale_curriculum": self.cfg.two_feet_above_height_reward_scale_curriculum,
            "track_lin_vel_xy_reward_scale_curriculum": self.cfg.track_lin_vel_xy_reward_scale_curriculum,
            "three_or_more_feet_contact_penalty_reward_scale_curriculum": (
                self.cfg.three_or_more_feet_contact_penalty_reward_scale_curriculum
            ),
            "two_feet_above_height_alpha_curriculum": self.cfg.two_feet_above_height_alpha_curriculum,
            "two_feet_above_height_threshold_curriculum": self.cfg.two_feet_above_height_threshold_curriculum,
            "actuation_delay_range_curriculum": self.cfg.actuation_delay_range_curriculum,
            "tricky_terrain_curriculum": self.cfg.tricky_terrain_curriculum,
            "opposite_direction_cmd_prob_curriculum": self.cfg.opposite_direction_cmd_prob_curriculum,
            "front_back_asymetry_curriculum": self.cfg.front_back_asymetry_curriculum,
        }
        phase_count = self._two_feet_curriculum_phase_count()
        if phase_count < 1:
            raise ValueError("two_feet_above_height_reward_scale_curriculum must contain at least one phase.")
        for name, values in phase_fields.items():
            if len(values) != phase_count:
                raise ValueError(f"{name} must have {phase_count} values, got {len(values)}.")

        advance_thresholds = self._curriculum_values(self.cfg.two_feet_curriculum_advance_thresholds)
        advance_vx_indicators = self._curriculum_values(self.cfg.two_feet_curriculum_advance_thresholds_vx_indicator)
        if len(advance_thresholds) != phase_count - 1:
            raise ValueError(
                "two_feet_curriculum_advance_thresholds must have one value per phase transition, "
                f"got {len(advance_thresholds)} for {phase_count} phases."
            )
        if len(advance_vx_indicators) != phase_count - 1:
            raise ValueError(
                "two_feet_curriculum_advance_thresholds_vx_indicator must have one value per phase transition, "
                f"got {len(advance_vx_indicators)} for {phase_count} phases."
            )
        for i, threshold in enumerate(advance_thresholds):
            if threshold < 0.0:
                raise ValueError(f"two_feet_curriculum_advance_thresholds[{i}] must be non-negative, got {threshold}.")

        for i, threshold in enumerate(self.cfg.two_feet_above_height_threshold_curriculum):
            if threshold < 0.0:
                raise ValueError(f"two_feet_above_height_threshold_curriculum[{i}] must be non-negative, got {threshold}.")
        for i, alpha in enumerate(self.cfg.two_feet_above_height_alpha_curriculum):
            if alpha < 0.0:
                raise ValueError(f"two_feet_above_height_alpha_curriculum[{i}] must be non-negative, got {alpha}.")
        for i, delay_range in enumerate(self.cfg.actuation_delay_range_curriculum):
            self._validate_actuation_delay_range(tuple(delay_range), f"actuation_delay_range_curriculum[{i}]")
        for i, opposite_prob in enumerate(self.cfg.opposite_direction_cmd_prob_curriculum):
            if not 0.0 <= opposite_prob <= 1.0:
                raise ValueError(
                    f"opposite_direction_cmd_prob_curriculum[{i}] must be between 0 and 1, got {opposite_prob}."
                )

    def _fill_uniform_range(self, tensor: torch.Tensor, value_range: tuple[float, float]):
        low, high = value_range
        if low == high:
            tensor.fill_(low)
        else:
            tensor.uniform_(low, high)

    def _parse_base_push_force_curriculum(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.cfg.forces_applied_to_base_curriculum)
        if any(value < 0.0 for value in values):
            raise ValueError(f"forces_applied_to_base_curriculum must be non-negative, got {values}.")
        if not 0.0 < self.cfg.forces_curriculum_smoothing <= 1.0:
            raise ValueError(
                "forces_curriculum_smoothing must be in (0, 1], "
                f"got {self.cfg.forces_curriculum_smoothing}."
            )
        return values

    def _parse_max_velx_range_curriculum(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.cfg.max_velx_range_curriculum)
        if any(value < 0.0 for value in values):
            raise ValueError(f"max_velx_range_curriculum must be non-negative, got {values}.")
        return values

    def _set_max_velx_range_curriculum_level(self, level_idx: int):
        max_vel_x = self._max_velx_range_curriculum_values[level_idx]
        self._max_velx_range_curriculum_idx = level_idx
        self.cfg.command_lin_vel_x_range = (-max_vel_x, max_vel_x)

    def _set_base_push_force_curriculum_level(self, level_idx: int):
        force = self._base_push_force_curriculum_values[level_idx]
        self._base_push_force_curriculum_idx = level_idx
        self.cfg.base_push_force_xy_range = (-force, force)

    def get_curriculum_global_idx(self) -> int | None:
        has_velx_curriculum = bool(self._max_velx_range_curriculum_values)
        has_force_curriculum = bool(self._base_push_force_curriculum_values)
        if not has_velx_curriculum and not has_force_curriculum:
            return None
        if not has_velx_curriculum:
            return self._base_push_force_curriculum_idx

        velx_idx = self._max_velx_range_curriculum_idx
        if velx_idx < len(self._max_velx_range_curriculum_values) - 1:
            return velx_idx
        return velx_idx + self._base_push_force_curriculum_idx

    def get_curriculum_max_global_idx(self) -> int | None:
        has_velx_curriculum = bool(self._max_velx_range_curriculum_values)
        has_force_curriculum = bool(self._base_push_force_curriculum_values)
        if not has_velx_curriculum and not has_force_curriculum:
            return None
        if not has_velx_curriculum:
            return len(self._base_push_force_curriculum_values) - 1
        if not has_force_curriculum:
            return len(self._max_velx_range_curriculum_values) - 1
        return len(self._max_velx_range_curriculum_values) + len(self._base_push_force_curriculum_values) - 2

    def _clip_curriculum_activation_idx(self, start_idx: int) -> int:
        max_idx = self.get_curriculum_max_global_idx()
        if max_idx is None:
            return int(start_idx)
        return min(int(start_idx), max_idx)

    def _two_feet_curriculum_uses_tricky_terrain(self) -> bool:
        return bool(self._curriculum_two_feet_enabled() and any(self.cfg.tricky_terrain_curriculum))

    def _tricky_terrain_is_available(self) -> bool:
        return bool(getattr(self.cfg, "tricky_terrain", False) or self._two_feet_curriculum_uses_tricky_terrain())

    def _tricky_terrain_should_be_active(self) -> bool:
        if self._two_feet_curriculum_uses_tricky_terrain():
            return bool(self._two_feet_curriculum_value("tricky_terrain"))
        if not getattr(self.cfg, "tricky_terrain", False):
            return False
        start_idx = getattr(self.cfg, "curriculum_tricky_terrain_idx", None)
        if start_idx is None:
            return True
        start_idx = self._clip_curriculum_activation_idx(start_idx)
        curriculum_idx = self.get_curriculum_global_idx()
        if curriculum_idx is None:
            return start_idx <= 0
        return curriculum_idx is not None and curriculum_idx >= start_idx

    def _sample_terrain_origins_from_tiles(
        self,
        rows: tuple[int, ...],
        columns: tuple[int, ...],
        env_ids: torch.Tensor,
        column_weights: tuple[float, ...] | None = None,
    ):
        terrain_origins = getattr(self._terrain, "terrain_origins", None)
        if terrain_origins is None:
            return
        if len(rows) == 0:
            raise ValueError("At least one terrain row must be provided.")
        if len(columns) == 0:
            raise ValueError("At least one terrain column must be provided.")
        if column_weights is not None and len(column_weights) != len(columns):
            raise ValueError(
                f"column_weights length {len(column_weights)} must match columns length {len(columns)}."
            )

        num_rows, num_cols = terrain_origins.shape[:2]
        invalid_rows = [row for row in rows if row < 0 or row >= num_rows]
        invalid_cols = [col for col in columns if col < 0 or col >= num_cols]
        if invalid_rows:
            raise ValueError(f"Terrain rows {invalid_rows} are outside available range [0, {num_rows - 1}].")
        if invalid_cols:
            raise ValueError(f"Terrain columns {invalid_cols} are outside available range [0, {num_cols - 1}].")

        row_ids = torch.tensor(rows, dtype=torch.long, device=self.device)
        column_ids = torch.tensor(columns, dtype=torch.long, device=self.device)
        origin_pool = terrain_origins[row_ids[:, None], column_ids[None, :], :].reshape(
            len(rows) * len(columns), 3
        )
        if column_weights is None:
            selected_ids = torch.randint(origin_pool.shape[0], (len(env_ids),), device=self.device)
        else:
            # origin_pool is row-major over (rows, columns): pool index k -> column (k % len(columns)).
            # Rows are weighted uniformly, columns by their proportion-matched weight. torch.multinomial
            # normalizes the (unnormalized) weights internally.
            col_weights = torch.tensor(column_weights, dtype=torch.float, device=self.device)
            pool_weights = col_weights.repeat(len(rows))
            selected_ids = torch.multinomial(pool_weights, len(env_ids), replacement=True)
        self._terrain.env_origins[env_ids] = origin_pool[selected_ids]

    def _refresh_tricky_terrain_origins(self, env_ids: torch.Tensor | None = None, force: bool = False):
        if not self._tricky_terrain_is_available():
            return
        if getattr(self._terrain, "terrain_origins", None) is None:
            return

        should_be_active = self._tricky_terrain_should_be_active()
        changed = should_be_active != self._tricky_terrain_active
        if env_ids is None:
            if not force and not changed:
                return
            env_ids = self._robot._ALL_INDICES
        elif changed:
            env_ids = self._robot._ALL_INDICES

        if should_be_active:
            columns = self.cfg.tricky_terrain_cols
            column_weights = getattr(self.cfg, "tricky_terrain_col_weights", None)
        else:
            columns = self.cfg.tricky_terrain_flat_cols
            column_weights = None
        self._sample_terrain_origins_from_tiles(
            tuple(self.cfg.tricky_terrain_spawn_rows),
            tuple(columns),
            env_ids,
            column_weights=tuple(column_weights) if column_weights is not None else None,
        )
        self._tricky_terrain_active = should_be_active

    def _update_base_push_force_curriculum(self, completed_episode_returns: torch.Tensor):
        can_increase_velx = (
            bool(self._max_velx_range_curriculum_values)
            and self._max_velx_range_curriculum_idx < len(self._max_velx_range_curriculum_values) - 1
        )
        can_increase_force = (
            bool(self._base_push_force_curriculum_values)
            and self._base_push_force_curriculum_idx < len(self._base_push_force_curriculum_values) - 1
        )
        if (
            (not can_increase_velx and not can_increase_force)
            or len(completed_episode_returns) == 0
        ):
            return

        if self.common_step_counter - self._base_push_last_curriculum_step < self.max_episode_length:
            return

        mean_reward = torch.mean(completed_episode_returns).item()
        if self._base_push_mean_reward_smooth is None:
            self._base_push_mean_reward_smooth = mean_reward
        else:
            smoothing = self.cfg.forces_curriculum_smoothing
            self._base_push_mean_reward_smooth = (
                (1.0 - smoothing) * self._base_push_mean_reward_smooth + smoothing * mean_reward
            )

        if self._base_push_mean_reward_smooth < self.cfg.forces_curriculum_threshold_reward:
            return

        if can_increase_velx:
            self._set_max_velx_range_curriculum_level(self._max_velx_range_curriculum_idx + 1)
        else:
            self._set_base_push_force_curriculum_level(self._base_push_force_curriculum_idx + 1)
        self._base_push_mean_reward_smooth = None
        self._base_push_last_curriculum_step = self.common_step_counter
        self._refresh_tricky_terrain_origins()

    def _episode_reward_metric(self, key: str, env_ids: torch.Tensor) -> float:
        return torch.mean(self._episode_sums[key][env_ids]).abs().item() / self.max_episode_length_s

    def _set_two_feet_curriculum_phase(self, phase: int):
        if phase < 1 or phase > self._two_feet_curriculum_phase_count():
            raise ValueError(
                f"Two-feet curriculum phase must be in [1, {self._two_feet_curriculum_phase_count()}], got {phase}."
            )
        self._two_feet_curriculum_phase = phase
        self.cfg.two_feet_above_height_reward_scale = self._two_feet_curriculum_value(
            "two_feet_above_height_reward_scale", phase
        )
        self.cfg.track_lin_vel_xy_reward_scale = self._two_feet_curriculum_value(
            "track_lin_vel_xy_reward_scale", phase
        )
        self.cfg.three_or_more_feet_contact_penalty_reward_scale = self._two_feet_curriculum_value(
            "three_or_more_feet_contact_penalty_reward_scale", phase
        )
        self.cfg.two_feet_above_height_alpha = self._two_feet_curriculum_value(
            "two_feet_above_height_alpha", phase
        )
        self.cfg.two_feet_above_height_threshold = self._two_feet_curriculum_value(
            "two_feet_above_height_threshold", phase
        )
        self.cfg.actuation_delay_range = tuple(self._two_feet_curriculum_value("actuation_delay_range", phase))
        self.cfg.opposite_direction_cmd_prob = self._two_feet_curriculum_value(
            "opposite_direction_cmd_prob", phase
        )
        self.cfg.front_back_asymetry = bool(self._two_feet_curriculum_value("front_back_asymetry", phase))
        self._refresh_tricky_terrain_origins(force=True)

    def _update_two_feet_curriculum(self, completed_env_ids: torch.Tensor):
        if (
            not self._curriculum_two_feet_enabled()
            or len(completed_env_ids) == 0
            or self._two_feet_curriculum_phase >= self._two_feet_curriculum_phase_count()
        ):
            return

        transition_idx = self._two_feet_curriculum_phase_index()
        metric_key = self._two_feet_curriculum_advance_metric_key(transition_idx)
        threshold = self.cfg.two_feet_curriculum_advance_thresholds[transition_idx]
        reward_metric = self._episode_reward_metric(metric_key, completed_env_ids)
        if reward_metric > threshold:
            self._set_two_feet_curriculum_phase(self._two_feet_curriculum_phase + 1)

    def _reset_base_pushes(self, env_ids: torch.Tensor):
        self._base_push_steps_left[env_ids] = 0
        self._base_push_steps_until_next[env_ids] = self._sample_int_steps(
            self._base_push_interval_step_range, len(env_ids)
        )
        self._base_push_forces_b[env_ids] = 0.0
        self._base_push_application_points_b[env_ids] = 0.0
        self._robot.permanent_wrench_composer.reset(env_ids)

    def _sample_base_push_surface_points(self, count: int) -> torch.Tensor:
        half_extents = self._base_push_application_half_extents
        unit_samples = torch.rand((count, 3), device=self.device).clamp_min(torch.finfo(torch.float).eps)
        points = (2.0 * unit_samples - 1.0) * half_extents
        face_ids = torch.multinomial(self._base_push_face_areas, count, replacement=True)

        points[face_ids == 0, 2] = half_extents[2]  # top
        points[face_ids == 1, 0] = half_extents[0]  # front
        points[face_ids == 2, 0] = -half_extents[0]  # rear
        points[face_ids == 3, 1] = half_extents[1]  # left
        points[face_ids == 4, 1] = -half_extents[1]  # right
        return points[:, None, :].expand(-1, len(self._base_wrench_body_ids), -1)

    def _start_base_pushes(self, env_ids: torch.Tensor):
        self._base_push_steps_left[env_ids] = self._sample_int_steps(
            self._base_push_duration_step_range, len(env_ids)
        )
        self._base_push_steps_until_next[env_ids] = self._sample_int_steps(
            self._base_push_interval_step_range, len(env_ids)
        )

        forces = torch.zeros((len(env_ids), len(self._base_wrench_body_ids), 3), device=self.device)
        self._fill_uniform_range(forces[..., 0], self.cfg.base_push_force_xy_range)
        self._fill_uniform_range(forces[..., 1], self.cfg.base_push_force_xy_range)
        self._fill_uniform_range(forces[..., 2], self.cfg.base_push_force_z_range)
        self._base_push_forces_b[env_ids] = forces
        self._base_push_application_points_b[env_ids] = self._sample_base_push_surface_points(len(env_ids))

    def _update_base_push_wrench(self):
        inactive = self._base_push_steps_left <= 0
        if torch.any(inactive):
            self._base_push_steps_until_next[inactive] -= 1
            start_env_ids = torch.nonzero(
                inactive & (self._base_push_steps_until_next <= 0), as_tuple=False
            ).squeeze(-1)
            if len(start_env_ids) > 0:
                self._start_base_pushes(start_env_ids)

        active = self._base_push_steps_left > 0
        self._base_push_forces_b[~active] = 0.0
        self._base_push_application_points_b[~active] = 0.0
        if torch.any(active):
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                forces=self._base_push_forces_b,
                positions=self._base_push_application_points_b,
                body_ids=self._base_wrench_body_ids,
            )
            self._base_push_steps_left[active] -= 1
        else:
            self._robot.permanent_wrench_composer.reset()

    def _pre_physics_step(self, actions: torch.Tensor):
        resample_env_ids = torch.nonzero(self._command_steps_left <= 0, as_tuple=False).squeeze(-1)
        if len(resample_env_ids) > 0:
            self._resample_commands(resample_env_ids)
            self._command_steps_left[resample_env_ids] = self._command_resample_interval
        self._command_steps_left -= 1
        self._update_base_push_wrench()

        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._q_offset_action_and_obs

    def _apply_action(self):
        self._delayed_processed_actions = self._action_delay_buffer.compute(self._processed_actions)
        self._applied_actions = (
            self._delayed_processed_actions - self._q_offset_action_and_obs
        ) / self.cfg.action_scale
        self._robot.set_joint_position_target(self._delayed_processed_actions, joint_ids=self._joint_ids)

    def step(self, action: torch.Tensor):
        """Step the env while recording base IMU history at physics rate."""
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
            self._record_base_imu_history_sample()

        self.episode_length_buf += 1
        self.common_step_counter += 1

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

    def _record_base_imu_history_sample(self):
        if not self.cfg.imu_raw_inputs or self._base_imu_history_len == 0:
            return

        self._update_base_imu_bias_random_walk()
        sample = torch.cat(
            (
                self._get_joint_state_obs(corrupt=True),
                self._read_base_imu_sample(corrupt=True),
                self._get_base_imu_history_action_obs(),
            ),
            dim=-1,
        )
        self._base_imu_history = torch.roll(self._base_imu_history, shifts=-1, dims=1)
        self._base_imu_history[:, -1, :] = sample

    def _get_base_imu_history_action_obs(self) -> torch.Tensor:
        if self.cfg.base_imu_history_action_is_last_executed:
            return self._applied_actions
        return self._actions

    def _read_base_imu_sample(self, corrupt: bool = True) -> torch.Tensor:
        base_body_id = self._base_wrench_body_ids[0]
        base_quat_w = self._robot.data.body_link_quat_w[:, base_body_id]
        base_ang_vel_w = self._robot.data.body_com_ang_vel_w[:, base_body_id]
        base_ang_acc_w = self._robot.data.body_com_ang_acc_w[:, base_body_id]
        base_com_acc_w = self._robot.data.body_com_lin_acc_w[:, base_body_id]

        # PhysX exposes COM acceleration. Convert it to the base link origin, where the IMU is mounted.
        com_to_link_pos_b = -self._robot.data.body_com_pos_b[:, base_body_id]
        com_to_link_pos_w = math_utils.quat_apply(base_quat_w, com_to_link_pos_b)
        base_link_linear_acc_w = (
            base_com_acc_w
            + torch.linalg.cross(base_ang_acc_w, com_to_link_pos_w, dim=-1)
            + torch.linalg.cross(
                base_ang_vel_w, torch.linalg.cross(base_ang_vel_w, com_to_link_pos_w, dim=-1), dim=-1
            )
        )

        gyro_b = math_utils.quat_apply_inverse(base_quat_w, base_ang_vel_w)
        if self.cfg.imu_ekf_processed_inputs:
            acc_b = math_utils.quat_apply_inverse(base_quat_w, base_link_linear_acc_w)
            ekf_quat_w = math_utils.quat_mul(self._base_imu_heading_reference, base_quat_w)
            if corrupt:
                gyro_b, acc_b = self._maybe_corrupt_base_imu_vectors(gyro_b, acc_b, ekf_processed=True)
                ekf_quat_w = self._maybe_corrupt_base_imu_orientation(ekf_quat_w)
            orientation = self._get_base_imu_orientation_obs(ekf_quat_w)
            if self.cfg.base_imu_clip:
                gyro_b = gyro_b.clamp(-self.cfg.base_imu_gyro_clip, self.cfg.base_imu_gyro_clip)
                acc_b = acc_b.clamp(-self.cfg.base_imu_ekf_acc_clip, self.cfg.base_imu_ekf_acc_clip)
            return torch.cat((gyro_b, acc_b, orientation), dim=-1)

        specific_force_w = base_link_linear_acc_w + self._base_imu_gravity_bias_w
        acc_b = math_utils.quat_apply_inverse(base_quat_w, specific_force_w)
        if corrupt:
            gyro_b, acc_b = self._maybe_corrupt_base_imu_vectors(gyro_b, acc_b, ekf_processed=False)
        sample = torch.cat((gyro_b, acc_b), dim=-1)
        if self.cfg.base_imu_clip:
            gyro = sample[:, :3].clamp(-self.cfg.base_imu_gyro_clip, self.cfg.base_imu_gyro_clip)
            acc = sample[:, 3:6].clamp(-self.cfg.base_imu_acc_clip, self.cfg.base_imu_acc_clip)
            sample = torch.cat((gyro, acc), dim=-1)
        return sample

    def _get_base_imu_orientation_obs(self, quat_w: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_rotMat_on_imu_encoder:
            return math_utils.matrix_from_quat(quat_w).reshape(quat_w.shape[0], 9)
        gravity_w = self._gravity_direction_w.expand(quat_w.shape[0], -1)
        return math_utils.quat_apply_inverse(quat_w, gravity_w)

    def _maybe_corrupt_base_imu_vectors(
        self, gyro_b: torch.Tensor, acc_b: torch.Tensor, *, ekf_processed: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enable_observation_corruption or not self.cfg.noisy_imu or self._imu_noise_scale == 0.0:
            return gyro_b, acc_b

        gyro_b = gyro_b + self._base_imu_bias[:, :3]
        acc_b = acc_b + self._base_imu_bias[:, 3:6]
        gyro_noise_std = (
            self._imu_noise_scale * self.cfg.imu_gyro_noise_scale * self.cfg.base_imu_gyro_noise_std
        )
        acc_noise_std = self._imu_noise_scale * self.cfg.imu_acc_noise_scale * (
            self.cfg.base_imu_ekf_acc_noise_std if ekf_processed else self.cfg.base_imu_acc_noise_std
        )
        if gyro_noise_std > 0.0:
            gyro_b = gyro_b + torch.randn_like(gyro_b) * gyro_noise_std
        if acc_noise_std > 0.0:
            acc_b = acc_b + torch.randn_like(acc_b) * acc_noise_std
        return gyro_b, acc_b

    def _maybe_corrupt_base_imu_orientation(self, quat_w: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_observation_corruption or not self.cfg.noisy_imu or self._imu_noise_scale == 0.0:
            return quat_w
        noise_std = torch.tensor(
            self.cfg.base_imu_orientation_noise_std_rpy, dtype=quat_w.dtype, device=quat_w.device
        )
        noise_std *= self._imu_noise_scale * self.cfg.imu_orientation_noise_scale
        orientation_error = self._base_imu_orientation_bias_rpy + torch.randn_like(
            self._base_imu_orientation_bias_rpy
        ) * noise_std
        return math_utils.quat_box_plus(quat_w, orientation_error)

    def _sample_base_imu_bias(self, env_ids: torch.Tensor):
        self._base_imu_bias[env_ids] = 0.0
        self._base_imu_orientation_bias_rpy[env_ids] = 0.0
        if not self.cfg.enable_observation_corruption or not self.cfg.noisy_imu or self._imu_noise_scale == 0.0:
            return
        gyro_bias_init_std = (
            self._imu_noise_scale * self.cfg.imu_gyro_noise_scale * self.cfg.base_imu_gyro_bias_init_std
        )
        acc_bias_init_std = self._imu_noise_scale * self.cfg.imu_acc_noise_scale * (
            self.cfg.base_imu_ekf_acc_bias_init_std
            if self.cfg.imu_ekf_processed_inputs
            else self.cfg.base_imu_acc_bias_init_std
        )
        if gyro_bias_init_std > 0.0:
            self._base_imu_bias[env_ids, :3] = torch.randn(
                len(env_ids), 3, dtype=self._base_imu_bias.dtype, device=self.device
            ) * gyro_bias_init_std
        if acc_bias_init_std > 0.0:
            self._base_imu_bias[env_ids, 3:6] = torch.randn(
                len(env_ids), 3, dtype=self._base_imu_bias.dtype, device=self.device
            ) * acc_bias_init_std
        if self.cfg.imu_ekf_processed_inputs and self.cfg.imu_orientation_noise_scale > 0.0:
            orientation_bias_std = torch.tensor(
                self.cfg.base_imu_orientation_bias_init_std_rpy,
                dtype=self._base_imu_orientation_bias_rpy.dtype,
                device=self.device,
            )
            orientation_bias_std *= self._imu_noise_scale * self.cfg.imu_orientation_noise_scale
            self._base_imu_orientation_bias_rpy[env_ids] = torch.randn(
                len(env_ids), 3, device=self.device
            ) * orientation_bias_std

    def _update_base_imu_bias_random_walk(self):
        if not self.cfg.enable_observation_corruption or not self.cfg.noisy_imu or self._imu_noise_scale == 0.0:
            return
        gyro_bias_rw_std = (
            self._imu_noise_scale * self.cfg.imu_gyro_noise_scale * self.cfg.base_imu_gyro_bias_rw_std_per_step
        )
        acc_bias_rw_std = self._imu_noise_scale * self.cfg.imu_acc_noise_scale * (
            self.cfg.base_imu_ekf_acc_bias_rw_std_per_step
            if self.cfg.imu_ekf_processed_inputs
            else self.cfg.base_imu_acc_bias_rw_std_per_step
        )
        if gyro_bias_rw_std > 0.0:
            self._base_imu_bias[:, :3] += (
                torch.randn_like(self._base_imu_bias[:, :3]) * gyro_bias_rw_std
            )
        if acc_bias_rw_std > 0.0:
            self._base_imu_bias[:, 3:6] += (
                torch.randn_like(self._base_imu_bias[:, 3:6]) * acc_bias_rw_std
            )
        if self.cfg.imu_ekf_processed_inputs and self.cfg.imu_orientation_noise_scale > 0.0:
            orientation_rw_std = torch.tensor(
                self.cfg.base_imu_orientation_bias_rw_std_per_step_rpy,
                dtype=self._base_imu_orientation_bias_rpy.dtype,
                device=self.device,
            )
            orientation_rw_std *= self._imu_noise_scale * self.cfg.imu_orientation_noise_scale
            self._base_imu_orientation_bias_rpy += torch.randn_like(
                self._base_imu_orientation_bias_rpy
            ) * orientation_rw_std

    def _get_observations(self) -> dict:
        if self.cfg.policy_model == "base_imu_teacher":
            teacher_obs = self._get_teacher_critic_obs(corrupt=True)
            self._previous_actions = self._actions.clone()
            return self._with_optional_rnd_state({"policy": teacher_obs, "critic": teacher_obs})
        if self.cfg.policy_model == "base_imu_student_rl":
            policy_obs = torch.cat((self._base_imu_history.reshape(self.num_envs, -1), self._commands), dim=-1)
            critic_obs = self._get_teacher_critic_obs(corrupt=False)
            if self.cfg.feed_history_encoding_to_critic:
                critic_obs = torch.cat((critic_obs, self._base_imu_history.reshape(self.num_envs, -1)), dim=-1)
            self._previous_actions = self._actions.clone()
            return self._with_optional_rnd_state({"policy": policy_obs, "critic": critic_obs})
        if self.cfg.policy_model == "base_imu_student_dagger":
            teacher_obs = self._get_teacher_critic_obs(corrupt=False)
            policy_obs = torch.cat(
                (teacher_obs, self._base_imu_history.reshape(self.num_envs, -1), self._commands), dim=-1
            )
            self._previous_actions = self._actions.clone()
            return self._with_optional_rnd_state({"policy": policy_obs, "critic": teacher_obs})
        if self.cfg.policy_model == "simple_dreamer_v3":
            policy_obs = self._get_simple_proprioceptive_obs(corrupt=True)
            self._previous_actions = self._actions.clone()
            if getattr(self.cfg, "_dreamer_command_outside_observation", False):
                return self._with_optional_rnd_state({"policy": policy_obs, "command": self._commands})
            return self._with_optional_rnd_state(
                {"policy": torch.cat((policy_obs, self._commands), dim=-1)}
            )

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        action_obs = self._applied_actions if self.cfg.action_obs_is_last_executed else self._actions

        obs_terms = []
        if not self.cfg.remove_root_lin_vel_b_from_obs:
            obs_terms.append(self._maybe_corrupt(self._robot.data.root_lin_vel_b, self.cfg.base_lin_vel_noise))
        obs_terms.extend(
            [
                self._maybe_corrupt(self._robot.data.root_ang_vel_b, self.cfg.base_ang_vel_noise),
                self._maybe_corrupt(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise),
                self._commands,
                self._maybe_corrupt(joint_pos, self.cfg.joint_pos_noise),
                self._maybe_corrupt(joint_vel, self.cfg.joint_vel_noise),
                action_obs,
            ]
        )
        obs = torch.cat(tuple(obs_terms), dim=-1)

        self._previous_actions = self._actions.clone()
        return self._with_optional_rnd_state({"policy": obs})

    def _with_optional_rnd_state(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.cfg.rnd_network:
            observations["rnd_state"] = self._get_rnd_curiosity_state()
        return observations

    def _get_rnd_curiosity_state(self) -> torch.Tensor:
        """Return a clean, task-focused posture/contact state for RND (19 dimensions)."""

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs
        feet_contact = self._get_feet_contact_mask(self.cfg.feet_ground_contact_threshold).to(joint_pos.dtype)
        return torch.cat((self._robot.data.projected_gravity_b, joint_pos, feet_contact), dim=-1)

    def _get_simple_proprioceptive_obs(self, corrupt: bool) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]

        obs_terms = []
        if not self.cfg.remove_root_lin_vel_b_from_obs:
            obs_terms.append(
                self._maybe_corrupt(self._robot.data.root_lin_vel_b, self.cfg.base_lin_vel_noise)
                if corrupt
                else self._robot.data.root_lin_vel_b
            )
        obs_terms.extend(
            [
                self._maybe_corrupt(self._robot.data.root_ang_vel_b, self.cfg.base_ang_vel_noise)
                if corrupt
                else self._robot.data.root_ang_vel_b,
                self._maybe_corrupt(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise)
                if corrupt
                else self._robot.data.projected_gravity_b,
                self._maybe_corrupt(joint_pos, self.cfg.joint_pos_noise) if corrupt else joint_pos,
                self._maybe_corrupt(joint_vel, self.cfg.joint_vel_noise) if corrupt else joint_vel,
            ]
        )
        return torch.cat(tuple(obs_terms), dim=-1)

    def _get_joint_state_obs(self, corrupt: bool) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids] - self._q_offset_action_and_obs
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        if corrupt:
            joint_pos = self._maybe_corrupt(joint_pos, self.cfg.joint_pos_noise)
            joint_vel = self._maybe_corrupt(joint_vel, self.cfg.joint_vel_noise)
        return torch.cat((joint_pos, joint_vel), dim=-1)

    def _get_foot_height_obs(self) -> torch.Tensor:
        foot_pos_w = self._robot.data.body_pos_w[:, self._feet_robot_body_ids, :]
        terrain_z = self._get_terrain_height_below_feet(foot_pos_w)
        return foot_pos_w[..., 2] - terrain_z

    def _get_foot_positions_w(self) -> torch.Tensor:
        calf_pos_w = self._robot.data.body_pos_w[:, self._feet_robot_body_ids, :]
        calf_quat_w = self._robot.data.body_quat_w[:, self._feet_robot_body_ids, :]
        foot_offsets_b = self._feet_robot_body_to_foot_offsets_b.to(dtype=calf_pos_w.dtype)
        foot_offsets_w = math_utils.quat_apply(
            calf_quat_w, foot_offsets_b.unsqueeze(0).expand(self.num_envs, -1, -1)
        )
        return calf_pos_w + foot_offsets_w

    def _get_reward_foot_heights(self) -> torch.Tensor:
        foot_pos_w = self._get_foot_positions_w()
        terrain_z = self._get_terrain_height_below_feet(foot_pos_w)
        return foot_pos_w[..., 2] - terrain_z

    def _get_terrain_height_below_feet(self, foot_pos_w: torch.Tensor) -> torch.Tensor:
        fallback_z = self._terrain.env_origins[:, 2].reshape(-1, 1).expand(-1, foot_pos_w.shape[1])
        if self._terrain_height_wp_mesh is None:
            return fallback_z

        from isaaclab.utils.warp import raycast_mesh

        ray_starts = foot_pos_w.reshape(-1, 3).clone()
        ray_starts[:, 2] += 1.0
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[:, 2] = -1.0
        ray_hits = raycast_mesh(ray_starts, ray_directions, self._terrain_height_wp_mesh, max_dist=5.0)[0]
        terrain_z = ray_hits[:, 2].reshape(foot_pos_w.shape[0], foot_pos_w.shape[1])
        return torch.where(torch.isfinite(terrain_z), terrain_z, fallback_z)

    def _get_teacher_encoder_obs(self, corrupt: bool) -> torch.Tensor:
        terms = [
            self._maybe_corrupt(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise)
            if corrupt
            else self._robot.data.projected_gravity_b,
            self._maybe_corrupt(self._robot.data.root_lin_vel_b, self.cfg.base_lin_vel_noise)
            if corrupt
            else self._robot.data.root_lin_vel_b,
            self._maybe_corrupt(self._robot.data.root_ang_vel_b, self.cfg.base_ang_vel_noise)
            if corrupt
            else self._robot.data.root_ang_vel_b,
        ]
        if self.cfg.include_foot_height_obs:
            terms.append(self._get_foot_height_obs())
        terms.extend(
            [
                self._applied_actions if self.cfg.action_obs_is_last_executed else self._actions,
                self._get_joint_state_obs(corrupt=corrupt),
            ]
        )
        return torch.cat(tuple(terms), dim=-1)

    def _get_teacher_critic_obs(self, corrupt: bool) -> torch.Tensor:
        return torch.cat((self._get_teacher_encoder_obs(corrupt=corrupt), self._commands), dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.track_commands_in_world_heading_frame:
            tracked_lin_vel_xy = _world_velocity_in_heading_frame_xy(
                self._robot.data.root_lin_vel_w, self._robot.data.root_quat_w
            )
            tracked_yaw_rate = self._robot.data.root_ang_vel_w[:, 2]
        else:
            tracked_lin_vel_xy = self._robot.data.root_lin_vel_b[:, :2]
            tracked_yaw_rate = self._robot.data.root_ang_vel_b[:, 2]
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - tracked_lin_vel_xy), dim=1)
        yaw_rate_error = torch.square(self._commands[:, 2] - tracked_yaw_rate)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self._joint_ids]), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._joint_ids]), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        base_height_error = torch.square(base_height - self.cfg.base_z_desired)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_body_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_body_ids]
        feet_air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1)
        feet_air_time *= torch.norm(self._commands[:, :2], dim=1) > 0.1

        feet_contact_mask = self._get_feet_contact_mask(self.cfg.feet_ground_contact_threshold)
        two_feet_above_height = self._compute_two_feet_above_height_reward(feet_contact_mask)
        front_thigh_contact_mask = None
        if self.cfg.front_back_asymetry:
            front_thigh_contact_mask = self._get_body_contact_mask(
                self._thigh_body_ids, self.cfg.feet_ground_contact_threshold
            )[:, self._front_thigh_contact_indices]
        three_or_more_feet_contact = self._compute_three_or_more_feet_contact_penalty(
            feet_contact_mask, front_thigh_contact_mask
        )
        undesired_contacts = self._compute_contact_count(self._thigh_body_ids, self.cfg.undesired_contact_threshold)
        force_transmited_through_joints = self._compute_force_transmited_through_joints()
        foot_contact = self._compute_foot_contact_penalty()

        track_lin_vel_xy = torch.exp(-lin_vel_error / self.cfg.tracking_std**2)
        track_ang_vel_z = torch.exp(-yaw_rate_error / self.cfg.tracking_std**2)
        track_base_height = torch.exp(-self.cfg.base_height_exp_scale * base_height_error)

        rewards = {
            "track_lin_vel_xy_exp": self._scale_bounded_positive_reward(
                track_lin_vel_xy, self.cfg.track_lin_vel_xy_reward_scale
            ),
            "track_ang_vel_z_exp": self._scale_bounded_positive_reward(
                track_ang_vel_z, self.cfg.track_ang_vel_z_reward_scale
            ),
            "lin_vel_z_l2": z_vel_error * self.cfg.lin_vel_z_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_xy_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "two_feet_above_height": self._scale_bounded_positive_reward(
                two_feet_above_height, self.cfg.two_feet_above_height_reward_scale
            ),
            "three_or_more_feet_contact": three_or_more_feet_contact
            * self.cfg.three_or_more_feet_contact_penalty_reward_scale
            * self.step_dt,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "base_collision_terminal": self._base_collision_terminated.float()
            * self.cfg.base_collision_terminal_penalty,
            "flat_orientation_l2": flat_orientation * self.cfg.base_tilt_penalty_reward_scale * self.step_dt,
            "track_base_height_exp": self._scale_bounded_positive_reward(
                track_base_height, self.cfg.track_base_height_reward_scale
            ),
            "force_transmited_through_joints": force_transmited_through_joints
            * self.cfg.force_transmited_through_joints_reward_scale
            * self.step_dt,
            "foot_contact": foot_contact * self.cfg.foot_contact_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        step_log = {f"RewardsPerStep/{key}": torch.mean(value).item() for key, value in rewards.items()}
        reward_scales = {
            "track_lin_vel_xy_exp": self.cfg.track_lin_vel_xy_reward_scale,
            "track_ang_vel_z_exp": self.cfg.track_ang_vel_z_reward_scale,
            "lin_vel_z_l2": self.cfg.lin_vel_z_reward_scale,
            "ang_vel_xy_l2": self.cfg.ang_vel_xy_reward_scale,
            "dof_torques_l2": self.cfg.joint_torque_reward_scale,
            "dof_acc_l2": self.cfg.joint_accel_reward_scale,
            "action_rate_l2": self.cfg.action_rate_reward_scale,
            "feet_air_time": self.cfg.feet_air_time_reward_scale,
            "two_feet_above_height": self.cfg.two_feet_above_height_reward_scale,
            "three_or_more_feet_contact": self.cfg.three_or_more_feet_contact_penalty_reward_scale,
            "undesired_contacts": self.cfg.undesired_contact_reward_scale,
            "flat_orientation_l2": self.cfg.base_tilt_penalty_reward_scale,
            "track_base_height_exp": self.cfg.track_base_height_reward_scale,
            "force_transmited_through_joints": self.cfg.force_transmited_through_joints_reward_scale,
            "foot_contact": self.cfg.foot_contact_reward_scale,
        }
        step_log.update(
            {
                f"PerStepRewardRatio/{key}": ratio
                for key, ratio in _per_step_reward_ratios(rewards, reward_scales, self.step_dt).items()
            }
        )
        step_log["RewardsPerStep/cmd_tracking"] = (
            step_log["RewardsPerStep/track_lin_vel_xy_exp"] + step_log["RewardsPerStep/track_ang_vel_z_exp"]
        )
        step_log["RewardsPerStep/total"] = torch.mean(reward).item()
        self.extras["log"] = step_log
        for key, value in rewards.items():
            self._episode_sums[key] += value
        self._episode_reward_sums += reward
        return reward

    def _scale_bounded_positive_reward(self, reward: torch.Tensor, scale: float) -> torch.Tensor:
        if self.cfg.negate_positive_rewards and scale > 0.0:
            reward = reward - 1.0
        return reward * scale * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_contact_data = self._base_external_contact_sensor.data
        external_forces = _external_contact_forces(
            base_contact_data.net_forces_w_history,
            base_contact_data.force_matrix_w_history,
        )
        max_external_force = torch.amax(torch.norm(external_forces, dim=-1), dim=(1, 2))
        self._base_collision_terminated = max_external_force > self.cfg.base_contact_threshold
        if self.cfg.finish_on_front_feet_contact:
            grace_period_finished = (
                self.episode_length_buf * self.step_dt >= self.cfg.finish_on_front_feet_contact_after
            )
            self._forbidden_feet_contact_terminated = (
                self._get_forbidden_feet_contact_indicator().bool() & grace_period_finished
            )
        else:
            self._forbidden_feet_contact_terminated.zero_()
        terminated = self._base_collision_terminated | self._forbidden_feet_contact_terminated
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        completed_env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        completed_episode_returns = self._episode_reward_sums[completed_env_ids]
        mean_episode_return = (
            torch.mean(completed_episode_returns).item() if len(completed_episode_returns) > 0 else 0.0
        )
        mean_episode_length_steps = (
            torch.mean(self.episode_length_buf[completed_env_ids].float()).item()
            if len(completed_env_ids) > 0
            else 0.0
        )
        self._update_two_feet_curriculum(completed_env_ids)
        self._update_base_push_force_curriculum(completed_episode_returns)
        self._refresh_tricky_terrain_origins(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._delayed_processed_actions[env_ids] = 0.0
        self._applied_actions[env_ids] = 0.0
        if self.cfg.imu_raw_inputs:
            self._base_imu_history[env_ids] = 0.0
            self._sample_base_imu_bias(env_ids)

        self._resample_commands(env_ids, allow_opposite=False)
        self._command_steps_left[env_ids] = self._command_resample_interval
        self._reset_base_pushes(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._reset_joint_pos[env_ids]
        if self.cfg.initial_position.lower() != "rigid":
            low, high = self.cfg.flexed_initial_joint_pos_noise_range
            if low != 0.0 or high != 0.0:
                joint_pos[:, self._joint_ids] += torch.empty(
                    (num_resets, len(self._joint_ids)), device=self.device
                ).uniform_(low, high)
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])

        root_pose = self._robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, 0] += self.cfg.reset_x_pos
        root_pose[:, 1] += self.cfg.reset_y_pos
        if self.cfg.reset_root_height is not None:
            root_pose[:, 2] = self._terrain.env_origins[env_ids, 2] + self.cfg.reset_root_height
        root_rpy = _sample_reset_root_rpy(
            num_resets,
            (self.cfg.reset_root_roll, self.cfg.reset_root_pitch, self.cfg.reset_yaw),
            self.cfg.reset_root_rpy_noise,
            self.device,
        )
        roll, pitch, yaw = root_rpy.unbind(dim=1)
        root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        zeros = torch.zeros_like(yaw)
        self._base_imu_heading_reference[env_ids] = math_utils.quat_from_euler_xyz(zeros, zeros, -yaw)

        root_velocity = torch.zeros_like(self._robot.data.default_root_state[env_ids, 7:])
        root_velocity[:, 0:3].uniform_(*self.cfg.reset_base_lin_vel_range)
        root_velocity[:, 3:6].uniform_(*self.cfg.reset_base_ang_vel_range)

        action_delays = torch.randint(
            low=self.cfg.actuation_delay_range[0],
            high=self.cfg.actuation_delay_range[1] + 1,
            size=(num_resets,),
            device=self.device,
            dtype=torch.int,
        )
        self._action_delay_steps[env_ids] = action_delays
        self._action_delay_buffer.set_time_lag(action_delays, env_ids)
        self._action_delay_buffer.reset(env_ids)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos[:, self._joint_ids], joint_ids=self._joint_ids, env_ids=env_ids)

        extras = {}
        for key in self._episode_sums:
            # Included abs to see all positive (reward / penalty) in wandb
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids]).abs() / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        extras['Episode_Reward/cmd_tracking'] = extras[f"Episode_Reward/track_lin_vel_xy_exp"] + extras[f"Episode_Reward/track_ang_vel_z_exp"]
        extras["Episode_Reward/total"] = mean_episode_return
        self._episode_reward_sums[env_ids] = 0.0
        extras["Episode/length_steps"] = mean_episode_length_steps
        extras["Episode/length_seconds"] = mean_episode_length_steps * self.step_dt

        force_low, force_high = self.cfg.base_push_force_xy_range
        velx_low, velx_high = self.cfg.command_lin_vel_x_range
        extras["Curriculum/command_lin_vel_x_abs"] = max(abs(velx_low), abs(velx_high))
        extras["Curriculum/max_velx_range_idx"] = self._max_velx_range_curriculum_idx
        extras["Curriculum/base_push_force_xy_abs"] = max(abs(force_low), abs(force_high))
        extras["Curriculum/base_push_force_idx"] = self._base_push_force_curriculum_idx
        extras["Curriculum/tricky_terrain_active"] = float(self._tricky_terrain_active)
        extras["Curriculum/two_feet_phase"] = self._two_feet_curriculum_phase
        extras["Curriculum/two_feet_above_height_reward_scale"] = self.cfg.two_feet_above_height_reward_scale
        extras["Curriculum/track_lin_vel_xy_reward_scale"] = self.cfg.track_lin_vel_xy_reward_scale
        extras["Curriculum/three_or_more_feet_contact_penalty_reward_scale"] = (
            self.cfg.three_or_more_feet_contact_penalty_reward_scale
        )
        extras["Curriculum/two_feet_above_height_alpha"] = self.cfg.two_feet_above_height_alpha
        extras["Curriculum/two_feet_above_height_threshold"] = self.cfg.two_feet_above_height_threshold
        extras["Curriculum/actuation_delay_max"] = self.cfg.actuation_delay_range[1]
        extras["Curriculum/opposite_direction_cmd_prob"] = self.cfg.opposite_direction_cmd_prob
        extras["Curriculum/front_back_asymetry"] = float(self.cfg.front_back_asymetry)
        curriculum_idx = self.get_curriculum_global_idx()
        if curriculum_idx is not None:
            extras["Curriculum/global_idx"] = curriculum_idx
        if self._base_push_mean_reward_smooth is not None:
            extras["Curriculum/base_push_mean_reward_smooth"] = self._base_push_mean_reward_smooth

        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self._base_collision_terminated[env_ids]
        ).item()
        extras["Episode_Termination/forbidden_feet_contact"] = torch.count_nonzero(
            self._forbidden_feet_contact_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # include logs in wandb
        self.extras["log"] = extras

    def _resample_commands(self, env_ids: torch.Tensor, allow_opposite: bool = True):
        commands = torch.empty((len(env_ids), 3), device=self.device)
        commands[:, 0].uniform_(*self.cfg.command_lin_vel_x_range)
        commands[:, 1].uniform_(*self.cfg.command_lin_vel_y_range)
        commands[:, 2].uniform_(*self.cfg.command_ang_vel_z_range)

        opposite_prob = self.cfg.opposite_direction_cmd_prob
        if allow_opposite and opposite_prob > 0.0:
            previous_commands = self._commands[env_ids]
            flip_mask = torch.rand_like(commands) < opposite_prob
            nonzero_previous = torch.abs(previous_commands) > 1e-6
            commands = torch.where(flip_mask & nonzero_previous, -previous_commands, commands)

        standing_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.standing_env_prob
        commands[standing_mask] = 0.0
        self._commands[env_ids] = commands

    def _compute_contact_count(self, body_ids: list[int], threshold: float) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)

    def _get_feet_contact_mask(self, threshold: float) -> torch.Tensor:
        return self._get_body_contact_mask(self._feet_body_ids, threshold)

    def _get_body_contact_mask(self, body_ids, threshold: float) -> torch.Tensor:
        contact_forces_history = self._contact_sensor.data.net_forces_w_history
        if contact_forces_history is not None:
            contact_force_norm = torch.norm(contact_forces_history[:, :, body_ids], dim=-1)
            return torch.amax(contact_force_norm, dim=1) > threshold

        contact_forces = self._contact_sensor.data.net_forces_w[:, body_ids, :]
        return torch.norm(contact_forces, dim=-1) > threshold

    def _compute_two_feet_above_height_reward(self, feet_contact_mask: torch.Tensor) -> torch.Tensor:
        foot_heights = self._get_reward_foot_heights()
        front_airborne = torch.all(~feet_contact_mask[:, self._front_feet_contact_indices], dim=1)
        front_avg_height = torch.mean(foot_heights[:, self._front_feet_robot_indices], dim=1)
        front_reward = self._two_feet_height_kernel(front_avg_height) * front_airborne.float()
        if self.cfg.front_back_asymetry:
            if not self.cfg.rear_feet_in_contact_for_twofeet:
                return front_reward
            rear_grounded = torch.all(feet_contact_mask[:, self._rear_feet_contact_indices], dim=1)
            return front_reward * rear_grounded.float()

        rear_airborne = torch.all(~feet_contact_mask[:, self._rear_feet_contact_indices], dim=1)
        rear_avg_height = torch.mean(foot_heights[:, self._rear_feet_robot_indices], dim=1)
        rear_reward = self._two_feet_height_kernel(rear_avg_height) * rear_airborne.float()
        return torch.maximum(front_reward, rear_reward)

    def _compute_three_or_more_feet_contact_penalty(
        self, feet_contact_mask: torch.Tensor, front_thigh_contact_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the contact-penalty indicator selected by the task symmetry mode."""
        if self.cfg.front_back_asymetry:
            front_foot_contact = torch.any(feet_contact_mask[:, self._front_feet_contact_indices], dim=1)
            if front_thigh_contact_mask is not None:
                front_foot_contact |= torch.any(front_thigh_contact_mask, dim=1)
            return front_foot_contact.float()
        return (torch.sum(feet_contact_mask, dim=1) >= 3).float()

    def _get_forbidden_feet_contact_indicator(self) -> torch.Tensor:
        """Return the same contact indicator used by the mode-dependent reward penalty."""
        feet_contact_mask = self._get_feet_contact_mask(self.cfg.feet_ground_contact_threshold)
        front_thigh_contact_mask = None
        if self.cfg.front_back_asymetry:
            front_thigh_contact_mask = self._get_body_contact_mask(
                self._thigh_body_ids, self.cfg.feet_ground_contact_threshold
            )[:, self._front_thigh_contact_indices]
        return self._compute_three_or_more_feet_contact_penalty(feet_contact_mask, front_thigh_contact_mask)

    def _two_feet_height_kernel(self, avg_height: torch.Tensor) -> torch.Tensor:
        threshold = self.cfg.two_feet_above_height_threshold
        below_threshold = torch.exp(-self.cfg.two_feet_above_height_alpha * torch.square(threshold - avg_height))
        return torch.where(avg_height >= threshold, torch.ones_like(avg_height), below_threshold)

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

    def _maybe_corrupt(self, tensor: torch.Tensor, noise_range: tuple[float, float]) -> torch.Tensor:
        if not self.cfg.enable_observation_corruption or noise_range[0] == noise_range[1] == 0.0:
            return tensor
        return tensor + torch.empty_like(tensor).uniform_(noise_range[0], noise_range[1])
