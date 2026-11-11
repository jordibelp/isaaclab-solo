# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded, startup-only compliant materials for the generated Solo12 terrain tiles."""

from __future__ import annotations

import math

import numpy as np


def validate_delta_x_range(values) -> tuple[float, float] | None:
    """Zero/zero disables sampling; an enabled interval must have positive finite endpoints."""
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("delta_x_floor_stiffness_range_mm must contain exactly two numbers.")
    low, high = (float(value) for value in values)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("delta_x_floor_stiffness_range_mm endpoints must be finite.")
    if low == high == 0.0:
        return None
    if not 0.0 < low <= high:
        raise ValueError("delta_x_floor_stiffness_range_mm requires 0 < min <= max, or [0.0, 0.0].")
    return low, high


def sample_floor_stiffness(values, mass_kg: float, gravity_m_s2: float, shape, seed):
    """Sample reference indentation uniformly, then convert using k = m*g/(2*delta_x)."""
    bounds = validate_delta_x_range(values)
    if bounds is None:
        raise ValueError("Cannot sample a disabled floor-stiffness range.")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0 or not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("Floor-stiffness calibration requires positive finite mass and gravity.")
    # An independent stream keeps reset, terrain, and policy random draws unchanged.
    rng = np.random.default_rng(None if seed is None else np.random.SeedSequence([seed, 0xF100]))
    delta_x_mm = rng.uniform(*bounds, size=shape)
    return delta_x_mm, mass_kg * gravity_m_s2 / (2.0 * delta_x_mm * 1.0e-3)


def terrain_face_tiles(points, faces, tile_size, grid_shape):
    """Map triangle centroids to the centered generator grid; -1 denotes its outer border."""
    centers_xy = points[faces, :2].mean(axis=1)
    size = np.asarray(tile_size)
    shape = np.asarray(grid_shape)
    coordinates = np.floor((centers_xy + size * shape / 2.0) / size).astype(np.int64)
    inside = ((coordinates >= 0) & (coordinates < shape)).all(axis=1)
    return np.where(inside, coordinates[:, 0] * shape[1] + coordinates[:, 1], -1)


def randomize_floor_stiffness(env):
    """Build static tile colliders before PhysX starts. Never mutate them at reset."""
    if validate_delta_x_range(env.cfg.delta_x_floor_stiffness_range_mm) is None:
        return None

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

    import isaaclab.sim as sim_utils

    terrain_cfg = env.cfg.terrain
    if terrain_cfg.terrain_type != "generator":
        raise ValueError("delta_x_floor_stiffness_range_mm currently requires generated terrain tiles.")
    stage = sim_utils.get_current_stage()
    source_path = env.cfg.robot.prim_path.replace(env.scene.env_regex_ns, env.scene.env_prim_paths[0])
    # Source masses already include base_mass and optional foot payloads, but not startup DR.
    mass_kg = 0.0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(source_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if mass is None or not math.isfinite(mass) or mass <= 0.0:
                raise ValueError(f"Floor-stiffness calibration needs an explicit positive mass on {prim.GetPath()}.")
            mass_kg += mass
    gravity = float(np.linalg.norm(env.cfg.sim.gravity))
    generator = terrain_cfg.terrain_generator
    shape = (generator.num_rows, generator.num_cols)
    # One additional material covers border faces, so the range overrides the entire floor.
    delta_x_mm, stiffness = sample_floor_stiffness(
        env.cfg.delta_x_floor_stiffness_range_mm, mass_kg, gravity, shape[0] * shape[1] + 1, env.cfg.seed
    )
    mesh_path = f"{terrain_cfg.prim_path}/terrain/mesh"
    mesh = UsdGeom.Mesh.Get(stage, mesh_path)
    points = np.asarray(mesh.GetPointsAttr().Get())
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    tile_ids = terrain_face_tiles(points, faces, generator.size, shape)
    # Sort once rather than scan the whole rough-terrain mesh once per tile.
    order = np.argsort(tile_ids, kind="stable")
    sorted_ids = tile_ids[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(order)]
    colliders_path = f"{terrain_cfg.prim_path}/terrain/floor_stiffness_colliders"
    UsdGeom.Scope.Define(stage, colliders_path)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        tile_id = int(sorted_ids[begin])
        label = "border" if tile_id == -1 else f"r{tile_id // shape[1]}_c{tile_id % shape[1]}"
        tile_path = f"{colliders_path}/{label}"
        UsdGeom.Xform.Define(stage, tile_path)
        material_path = f"{tile_path}/physicsMaterial"
        material_cfg = terrain_cfg.physics_material.copy()
        material_cfg.compliant_contact_stiffness = float(stiffness[tile_id])
        material_cfg.func(material_path, material_cfg)
        tile_faces = faces[order[begin:end]]
        vertices, inverse = np.unique(tile_faces, return_inverse=True)
        collider = UsdGeom.Mesh.Define(stage, f"{tile_path}/mesh")
        collider.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[vertices].astype(np.float32)))
        collider.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(inverse.astype(np.int32)))
        collider.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(end - begin, 3, dtype=np.int32)))
        collider.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        # Purpose=guide hides the duplicate collision surface without hiding the original visual mesh.
        collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("none")
        sim_utils.bind_physics_material(str(collider.GetPath()), material_path)
        collider.GetPrim().CreateAttribute("floor:delta_x_mm", Sdf.ValueTypeNames.Double).Set(
            float(delta_x_mm[tile_id])
        )
    # Keep the original mesh for rendering and height raycasts, but never double up its contacts.
    UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    result = {
        "reference_mass_kg": mass_kg,
        "reference_gravity_m_s2": gravity,
        "reference_support_feet": 2,
        "seed": env.cfg.seed,
        "tile_shape": list(shape),
        "tile_size_m": list(generator.size),
        "delta_x_mm": delta_x_mm[:-1].reshape(shape).tolist(),
        "stiffness_n_per_m": stiffness[:-1].reshape(shape).tolist(),
        "border_delta_x_mm": float(delta_x_mm[-1]),
        "border_stiffness_n_per_m": float(stiffness[-1]),
        "damping_n_s_per_m": terrain_cfg.physics_material.compliant_contact_damping or 0.0,
        "lifetime": "fixed_per_physical_tile",
    }
    print(
        f"[INFO]: Floor stiffness: {np.prod(shape)} fixed terrain tiles; "
        f"uniform reference indentation {env.cfg.delta_x_floor_stiffness_range_mm} mm; "
        f"nominal mass {mass_kg:.6g} kg, two support feet; "
        f"sampled stiffness {stiffness.min():.6g}..{stiffness.max():.6g} N/m.",
        flush=True,
    )
    return result
