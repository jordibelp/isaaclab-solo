"""USD authoring for the Isaac Lab spawner stand-ins, and USD -> MuJoCo model conversion.

The environments author their scene exactly as in Isaac Lab: the robot USD is referenced,
then the task code edits it (racket, payload and inertial overrides) and spawns shapes.
:func:`build_model` converts that stage into one MuJoCo model:

* every ``UsdPhysics.RigidBodyAPI`` prim becomes a MuJoCo body whose frame is the USD link
  frame; mass, centre of mass and full inertia come from ``UsdPhysics.MassAPI``, or from the
  collision shapes with uniform density when the inertia is not authored (PhysX does the same);
* articulation children are placed from the joint frames at zero joint angle, like PhysX
  reduced coordinates: ``T_parent_child = T_joint0 * inverse(T_joint1)``;
* kinematic rigid bodies become static bodies; free rigid bodies get a free joint;
* colliders keep their primitive type; convex-hull mesh colliders use MuJoCo's convex hull.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

_STAGE = None
GROUND_ATTR = "mujoco:groundPlane"
# PhysX default material when a collider has no bound physics material.
DEFAULT_FRICTION = 0.5
# Bouncy contacts are stiff spring-dampers (rad/s); the damping ratio sets the restitution.
BOUNCE_NATURAL_FREQUENCY = 1000.0
BOUNCE_SOLIMP = (0.99, 0.99, 0.001, 0.5, 2.0)
# Display colours for the robot meshes: the USD binds a white placeholder material.
ROBOT_COLORS = {"base": (0.85, 0.61, 0.34, 1.0), "flying_arm_2__link1": (0.72, 0.74, 0.78, 1.0),
                "flying_arm_2__link2": (0.72, 0.74, 0.78, 1.0)}


# --------------------------------------------------------------------------------------
# Stage and spawners (stand-ins for isaaclab.sim spawner functions)
# --------------------------------------------------------------------------------------

def new_stage():
    from pxr import Usd, UsdGeom

    global _STAGE
    _STAGE = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(_STAGE, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(_STAGE, 1.0)
    UsdGeom.Xform.Define(_STAGE, "/World")
    UsdGeom.Xform.Define(_STAGE, "/World/envs")
    UsdGeom.Xform.Define(_STAGE, "/World/envs/env_0")
    return _STAGE


def get_current_stage():
    return _STAGE if _STAGE is not None else new_stage()


def _concrete_paths(prim_path: str) -> list[str]:
    """Isaac Lab spawns regex paths ("env_.*") in the single source environment."""
    return [prim_path.replace("env_.*", "env_0")]


def _set_xform(prim, translation=None, orientation=None, scale=None):
    from pxr import Gf, UsdGeom

    def precision(name):
        # Referenced assets may already author float ops; reuse their precision.
        attribute = prim.GetAttribute(f"xformOp:{name}")
        if attribute and str(attribute.GetTypeName()) in ("quatf", "float3"):
            return UsdGeom.XformOp.PrecisionFloat
        return UsdGeom.XformOp.PrecisionDouble

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    if translation is not None:
        vector = Gf.Vec3f if precision("translate") == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d
        xformable.AddTranslateOp(precision("translate")).Set(vector(*map(float, translation)))
    if orientation is not None:
        w, x, y, z = map(float, orientation)
        if precision("orient") == UsdGeom.XformOp.PrecisionFloat:
            value = Gf.Quatf(w, Gf.Vec3f(x, y, z))
        else:
            value = Gf.Quatd(w, Gf.Vec3d(x, y, z))
        xformable.AddOrientOp(precision("orient")).Set(value)
    if scale is not None:
        vector = Gf.Vec3f if precision("scale") == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d
        xformable.AddScaleOp(precision("scale")).Set(vector(*map(float, scale)))


def spawn_from_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import UsdGeom

    stage = get_current_stage()
    for path in _concrete_paths(prim_path):
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        prim.GetReferences().AddReference(cfg.usd_path)
        _set_xform(prim, translation, orientation, cfg.scale)
    return stage.GetPrimAtPath(_concrete_paths(prim_path)[0])


def spawn_preview_surface(prim_path, cfg, **kwargs):
    from pxr import Gf, Sdf, UsdShade

    stage = get_current_stage()
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, prim_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cfg.diffuse_color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cfg.emissive_color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(cfg.roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(cfg.metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(cfg.opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material.GetPrim()


def spawn_physics_material(prim_path, cfg, **kwargs):
    from pxr import UsdPhysics, UsdShade

    stage = get_current_stage()
    material = UsdShade.Material.Define(stage, prim_path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(cfg.static_friction))
    api.CreateDynamicFrictionAttr().Set(float(cfg.dynamic_friction))
    api.CreateRestitutionAttr().Set(float(cfg.restitution))
    return material.GetPrim()


def _bind(prim_path, material_path, purpose):
    from pxr import UsdShade

    stage = get_current_stage()
    for path in _concrete_paths(prim_path):
        prim = stage.GetPrimAtPath(path)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, purpose)


def bind_visual_material(prim_path, material_path, stage=None, stronger_than_descendants=True):
    from pxr import UsdShade

    _bind(prim_path, material_path, UsdShade.Tokens.allPurpose)


def bind_physics_material(prim_path, material_path, stage=None, stronger_than_descendants=True):
    _bind(prim_path, material_path, "physics")


def _spawn_shape(prim_path, cfg, prim_type, attributes, translation=None, orientation=None, scale=None):
    """Isaac Lab layout: ``{path}`` Xform (rigid/mass APIs) + ``{path}/geometry/mesh`` shape."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = get_current_stage()
    for path in _concrete_paths(prim_path):
        if stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"A prim already exists at path: '{path}'.")
        root = UsdGeom.Xform.Define(stage, path).GetPrim()
        _set_xform(root, translation, orientation)
        UsdGeom.Xform.Define(stage, path + "/geometry")
        mesh = stage.DefinePrim(path + "/geometry/mesh", prim_type)
        for name, value in attributes.items():
            mesh.GetAttribute(name).Set(value)
        if scale is not None:
            _set_xform(mesh, scale=scale)
        if cfg.collision_props is not None:
            UsdPhysics.CollisionAPI.Apply(mesh)
            enabled = getattr(cfg.collision_props, "collision_enabled", None)
            if enabled is not None:
                UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr().Set(bool(enabled))
        if cfg.visual_material is not None:
            material_path = (cfg.visual_material_path if cfg.visual_material_path.startswith("/")
                             else f"{path}/geometry/{cfg.visual_material_path}")
            cfg.visual_material.func(material_path, cfg.visual_material)
            bind_visual_material(path + "/geometry/mesh", material_path)
        if cfg.physics_material is not None:
            material_path = (cfg.physics_material_path if cfg.physics_material_path.startswith("/")
                             else f"{path}/geometry/{cfg.physics_material_path}")
            cfg.physics_material.func(material_path, cfg.physics_material)
            bind_physics_material(path + "/geometry/mesh", material_path)
        if cfg.mass_props is not None:
            api = UsdPhysics.MassAPI.Apply(root)
            if cfg.mass_props.mass is not None:
                api.CreateMassAttr().Set(float(cfg.mass_props.mass))
            if cfg.mass_props.density is not None:
                api.CreateDensityAttr().Set(float(cfg.mass_props.density))
        if cfg.rigid_props is not None:
            api = UsdPhysics.RigidBodyAPI.Apply(root)
            props = cfg.rigid_props
            if getattr(props, "kinematic_enabled", None) is not None:
                api.CreateKinematicEnabledAttr().Set(bool(props.kinematic_enabled))
            if getattr(props, "rigid_body_enabled", None) is not None:
                api.CreateRigidBodyEnabledAttr().Set(bool(props.rigid_body_enabled))
            if getattr(props, "disable_gravity", None) is not None:
                PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateDisableGravityAttr().Set(bool(props.disable_gravity))
        if not getattr(cfg, "visible", True):
            UsdGeom.Imageable(root).MakeInvisible()
    return stage.GetPrimAtPath(_concrete_paths(prim_path)[0])


def spawn_sphere(prim_path, cfg, translation=None, orientation=None, **kwargs):
    return _spawn_shape(prim_path, cfg, "Sphere", {"radius": float(cfg.radius)}, translation, orientation)


def spawn_cylinder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    attributes = {"radius": float(cfg.radius), "height": float(cfg.height), "axis": cfg.axis.upper()}
    return _spawn_shape(prim_path, cfg, "Cylinder", attributes, translation, orientation)


def spawn_capsule(prim_path, cfg, translation=None, orientation=None, **kwargs):
    attributes = {"radius": float(cfg.radius), "height": float(cfg.height), "axis": cfg.axis.upper()}
    return _spawn_shape(prim_path, cfg, "Capsule", attributes, translation, orientation)


def spawn_cone(prim_path, cfg, translation=None, orientation=None, **kwargs):
    attributes = {"radius": float(cfg.radius), "height": float(cfg.height), "axis": cfg.axis.upper()}
    return _spawn_shape(prim_path, cfg, "Cone", attributes, translation, orientation)


def spawn_cuboid(prim_path, cfg, translation=None, orientation=None, **kwargs):
    size = min(cfg.size)
    return _spawn_shape(prim_path, cfg, "Cube", {"size": float(size)}, translation, orientation,
                        scale=[value / size for value in cfg.size])


def spawn_ground_plane(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Sdf, UsdGeom

    stage = get_current_stage()
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    _set_xform(prim, translation, orientation)
    prim.CreateAttribute(GROUND_ATTR, Sdf.ValueTypeNames.Bool).Set(True)
    if cfg.physics_material is not None:
        spawn_physics_material(prim_path + "/physicsMaterial", cfg.physics_material)
        bind_physics_material(prim_path, prim_path + "/physicsMaterial")
    return prim


def spawn_light(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Sdf, UsdLux

    stage = get_current_stage()
    light = UsdLux.DomeLight.Define(stage, prim_path)
    light.CreateIntensityAttr().Set(float(cfg.intensity))
    light.CreateColorAttr().Set(Gf.Vec3f(*cfg.color))
    light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
    return light.GetPrim()


# --------------------------------------------------------------------------------------
# Small geometry helpers (wxyz quaternions, column vectors)
# --------------------------------------------------------------------------------------

def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                     (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                     (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))))


def _push_only_restitution(zeta: float) -> float:
    """Speed ratio of a linear spring-damper contact that can push but not pull.

    The contact ends when the force reaches zero, before the spring is unloaded
    (Schwager and Poeschel, Phys. Rev. E 2007); valid for 0 < zeta < 1.
    """
    s = math.sqrt(1.0 - zeta * zeta)
    return math.exp(-zeta / s * (math.pi - math.atan2(2.0 * zeta * s, 1.0 - 2.0 * zeta * zeta)))


def restitution_solref(restitution: float) -> tuple[float, float]:
    """Direct (-stiffness, -damping) MuJoCo solref whose bounce has the PhysX restitution.

    PhysX applies restitution as a velocity rule; MuJoCo needs a soft contact. The damping
    ratio comes from inverting the push-only spring-damper restitution (monotonic in zeta).
    mujoco/test_borinot_sim2sim.py measures the resulting bounces.
    """
    target = min(max(float(restitution), 0.02), 0.98)
    low, high = 1e-6, 0.999
    for _ in range(60):
        zeta = 0.5 * (low + high)
        low, high = (zeta, high) if _push_only_restitution(zeta) > target else (low, zeta)
    w = BOUNCE_NATURAL_FREQUENCY
    return -w * w, -2.0 * zeta * w


def matrix_to_quat(rotation) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=float).reshape(9))
    return quat


def quat_mul(a, b) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _gf_quat(q) -> np.ndarray:
    """Gf quaternion -> normalised wxyz; a zero quaternion (unset) is identity."""
    value = np.array((q.GetReal(), *q.GetImaginary()), dtype=float)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else np.array((1.0, 0.0, 0.0, 0.0))


def _decompose(matrix):
    """Row-vector Gf matrix -> (translation, wxyz rotation, scale) for column vectors."""
    from pxr import Gf

    transform = Gf.Transform(matrix)
    rotation = transform.GetRotation().GetQuat()
    return (np.array(transform.GetTranslation(), dtype=float), _gf_quat(rotation),
            np.array(transform.GetScale(), dtype=float))


def _transform_points(matrix, points) -> np.ndarray:
    m = np.array(matrix, dtype=float)  # row-vector convention: p' = [p, 1] @ M
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return (homogeneous @ m)[:, :3]


# --------------------------------------------------------------------------------------
# USD -> MuJoCo
# --------------------------------------------------------------------------------------

@dataclass
class ShapeInfo:
    prim_path: str
    kind: str  # box, sphere, cylinder, capsule, cone, mesh
    pos: np.ndarray
    quat: np.ndarray
    size: np.ndarray  # MuJoCo size semantics; mesh: unused
    collision: bool
    visual: bool
    rgba: tuple
    friction: float
    restitution: float
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


@dataclass
class BodyInfo:
    prim_path: str
    name: str
    kind: str  # articulation_root, articulation_link, free, static
    shapes: list = field(default_factory=list)
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    parent: str | None = None
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in parent frame (or world)
    quat: np.ndarray = field(default_factory=lambda: np.array((1.0, 0.0, 0.0, 0.0)))
    joint_name: str | None = None
    joint_pos: np.ndarray | None = None  # in this body's frame
    joint_axis: np.ndarray | None = None
    joint_range_deg: tuple | None = None
    joint_frame_quat: np.ndarray | None = None  # USD localRot1, child joint axes in this body's frame


@dataclass
class SceneIndex:
    """Maps between USD prim paths and compiled MuJoCo ids."""

    bodies: dict  # prim path -> BodyInfo
    body_ids: dict = field(default_factory=dict)  # prim path -> MuJoCo body id
    joint_ids: dict = field(default_factory=dict)  # joint name -> MuJoCo joint id
    actuator_ids: dict = field(default_factory=dict)  # (joint name, "pd" | "ff") -> actuator id
    articulations: dict = field(default_factory=dict)  # asset prim path -> [body prim paths, root first]
    ground_geom: int | None = None


def _material_color(prim) -> tuple | None:
    from pxr import Usd, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    color, opacity = None, 1.0
    for shader_prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            continue
        for name in ("diffuseColor", "diffuse_color_constant"):
            value = shader.GetInput(name)
            if value and value.Get() is not None:
                color = tuple(float(v) for v in value.Get())
        for name in ("opacity", "opacity_constant"):
            value = shader.GetInput(name)
            if value and value.Get() is not None:
                opacity = float(value.Get())
    return None if color is None else (*color, opacity)


def _physics_material(prim) -> tuple[float, float]:
    from pxr import UsdPhysics, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        friction = api.GetDynamicFrictionAttr().Get()
        restitution = api.GetRestitutionAttr().Get()
        return (DEFAULT_FRICTION if friction is None else float(friction),
                0.0 if restitution is None else float(restitution))
    return DEFAULT_FRICTION, 0.0


def _shape_from_prim(prim, matrix, robot_body_name=None) -> ShapeInfo | None:
    from pxr import UsdGeom, UsdPhysics

    kind_map = {"Cube": "box", "Sphere": "sphere", "Cylinder": "cylinder", "Capsule": "capsule",
                "Cone": "cone", "Mesh": "mesh"}
    kind = kind_map.get(prim.GetTypeName())
    if kind is None:
        return None
    imageable = UsdGeom.Imageable(prim)
    visual = (imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
              and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render))
    collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    if collision:
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        collision = enabled is None or bool(enabled)
    if not (visual or collision):
        return None
    translation, rotation, scale = _decompose(matrix)
    rgba = ROBOT_COLORS.get(robot_body_name) or _material_color(prim) or (0.6, 0.6, 0.6, 1.0)
    friction, restitution = _physics_material(prim)
    shape = ShapeInfo(prim.GetPath().pathString, kind, translation, rotation, np.zeros(3), collision, visual,
                      rgba, friction, restitution)
    if kind == "box":
        shape.size = 0.5 * float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * scale
    elif kind == "sphere":
        shape.size = np.array((float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scale.max(), 0.0, 0.0))
    elif kind in ("cylinder", "capsule", "cone"):
        geom = {"cylinder": UsdGeom.Cylinder, "capsule": UsdGeom.Capsule, "cone": UsdGeom.Cone}[kind](prim)
        axis = "XYZ".index(geom.GetAxisAttr().Get())
        radial = [scale[i] for i in range(3) if i != axis]
        shape.size = np.array((float(geom.GetRadiusAttr().Get()) * max(radial),
                               0.5 * float(geom.GetHeightAttr().Get()) * scale[axis], 0.0))
        # MuJoCo cylinders/capsules run along local z.
        align = {0: (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
                 1: (math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0), 2: (1.0, 0.0, 0.0, 0.0)}[axis]
        shape.quat = quat_mul(rotation, align)
    else:
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if len(points) < 4 or len(counts) == 0:
            return None
        faces = []
        offset = 0
        for count in counts:  # fan triangulation (the Borinot meshes are triangles already)
            polygon = indices[offset:offset + count]
            faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1))
            offset += count
        shape.vertices = _transform_points(matrix, points)
        shape.faces = np.asarray(faces, dtype=np.int32)
        shape.pos, shape.quat = np.zeros(3), np.array((1.0, 0.0, 0.0, 0.0))
    return shape


def _shape_volume_inertia(shape: ShapeInfo) -> tuple[float, np.ndarray]:
    """Volume and unit-density inertia about the shape centre, in the shape's MuJoCo frame."""
    if shape.kind == "box":
        a, b, c = 2 * shape.size
        volume = a * b * c
        inertia = volume / 12 * np.diag((b * b + c * c, a * a + c * c, a * a + b * b))
    elif shape.kind == "sphere":
        r = shape.size[0]
        volume = 4 / 3 * math.pi * r ** 3
        inertia = np.eye(3) * 0.4 * volume * r * r
    elif shape.kind == "cylinder":
        r, h = shape.size[0], 2 * shape.size[1]
        volume = math.pi * r * r * h
        transverse = volume * (3 * r * r + h * h) / 12
        inertia = np.diag((transverse, transverse, 0.5 * volume * r * r))
    elif shape.kind == "capsule":
        r, h = shape.size[0], 2 * shape.size[1]
        volume = math.pi * r * r * h + 4 / 3 * math.pi * r ** 3
        transverse = volume * (3 * r * r + h * h) / 12
        inertia = np.diag((transverse, transverse, 0.5 * volume * r * r))
    else:
        raise ValueError(f"Shape-based inertia is not implemented for {shape.kind} ({shape.prim_path}).")
    return volume, inertia


def _mass_properties(prim, shapes) -> tuple[float, np.ndarray, np.ndarray]:
    """Authored MassAPI values; missing inertia uses the colliders with uniform density (PhysX rule)."""
    from pxr import UsdPhysics

    api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
    mass = float(api.GetMassAttr().Get() or 0.0) if api else 0.0
    density = float(api.GetDensityAttr().Get() or 0.0) if api else 0.0
    diagonal = np.array(api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0), dtype=float) if api else np.zeros(3)
    com_attr = api.GetCenterOfMassAttr().Get() if api else None
    if np.any(diagonal > 0) and mass > 0:
        axes = quat_to_matrix(_gf_quat(api.GetPrincipalAxesAttr().Get()))
        com = np.array(com_attr if com_attr is not None and all(map(math.isfinite, com_attr)) else (0, 0, 0), float)
        return mass, com, axes @ np.diag(diagonal) @ axes.T
    colliders = [shape for shape in shapes if shape.collision]
    if not colliders:
        raise ValueError(f"{prim.GetPath()} has no authored inertia and no collider to infer it from.")
    volumes, centres, inertias = [], [], []
    for shape in colliders:
        volume, inertia = _shape_volume_inertia(shape)
        rotation = quat_to_matrix(shape.quat)
        volumes.append(volume)
        centres.append(shape.pos)
        inertias.append(rotation @ inertia @ rotation.T)
    total_volume = float(sum(volumes))
    if mass <= 0:
        mass = (density if density > 0 else 1000.0) * total_volume
    rho = mass / total_volume
    com = sum(v * c for v, c in zip(volumes, centres)) / total_volume
    if com_attr is not None and all(map(math.isfinite, com_attr)):
        com = np.array(com_attr, dtype=float)
    inertia = np.zeros((3, 3))
    for volume, centre, own in zip(volumes, centres, inertias):
        d = centre - com
        inertia += rho * own + rho * volume * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, 0.5 * (inertia + inertia.T)


def _joint_frames(joint):
    p0 = np.array(joint.GetLocalPos0Attr().Get(), dtype=float)
    p1 = np.array(joint.GetLocalPos1Attr().Get(), dtype=float)
    r0 = _gf_quat(joint.GetLocalRot0Attr().Get())
    r1 = _gf_quat(joint.GetLocalRot1Attr().Get())
    return p0, r0, p1, r1


def collect_scene(stage, articulation_paths, rigid_object_paths):
    """Walk the stage once and describe every body, shape and articulation joint."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    rigid = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            if enabled is None or enabled:
                rigid[prim.GetPath().pathString] = prim
    bodies = {}
    world_shapes = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        owner = next((a for a in _ancestors(prim) if a.GetPath().pathString in rigid), None)
        if owner is None:
            shape = _shape_from_prim(prim, cache.GetLocalToWorldTransform(prim))
            if shape is not None:
                world_shapes.append(shape)
            continue
        name = owner.GetName()
        matrix, _ = cache.ComputeRelativeTransform(prim, owner)
        robot = any(owner.GetPath().pathString.startswith(path + "/") for path in articulation_paths)
        shape = _shape_from_prim(prim, matrix, name if robot else None)
        if shape is not None:
            bodies.setdefault(owner.GetPath().pathString, []).append(shape)

    infos = {}
    for path, prim in rigid.items():
        info = BodyInfo(path, prim.GetName(), "static", bodies.get(path, []))
        info.mass, info.com, info.inertia = _mass_properties(prim, info.shapes) if info.shapes or prim.HasAPI(
            UsdPhysics.MassAPI) else (0.0, np.zeros(3), np.zeros((3, 3)))
        kinematic = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        info.kind = "static" if kinematic else "free"
        translation, rotation, _ = _decompose(cache.GetLocalToWorldTransform(prim))
        info.pos, info.quat = translation, rotation
        infos[path] = info

    joints = [UsdPhysics.Joint(prim) for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
              if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False]
    articulations = {}
    for asset_path in articulation_paths:
        root = next((p for p in rigid if p.startswith(asset_path + "/")
                     and rigid[p].HasAPI(UsdPhysics.ArticulationRootAPI)), None)
        if root is None:
            raise ValueError(f"No articulation root body under {asset_path}.")
        infos[root].kind = "articulation_root"
        order, queue = [root], [root]
        while queue:
            parent = queue.pop(0)
            for joint in joints:
                body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
                body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
                if body0 != [parent] or len(body1) != 1 or body1[0] in order:
                    continue
                child = infos[body1[0]]
                p0, r0, p1, r1 = _joint_frames(joint)
                # T_parent_child = T_j0 * inv(T_j1)
                rel_quat = quat_mul(r0, np.array((r1[0], -r1[1], -r1[2], -r1[3])))
                child.pos = p0 - quat_to_matrix(rel_quat) @ p1
                child.quat = rel_quat
                child.parent, child.kind = parent, "articulation_link"
                child.joint_pos, child.joint_frame_quat = p1, r1  # child joint frame, also for fixed joints
                if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
                    revolute = UsdPhysics.RevoluteJoint(joint.GetPrim())
                    axis = np.zeros(3)
                    axis["XYZ".index(revolute.GetAxisAttr().Get())] = 1.0
                    child.joint_name = joint.GetPrim().GetName()
                    child.joint_axis = quat_to_matrix(r1) @ axis
                    child.joint_range_deg = (float(revolute.GetLowerLimitAttr().Get()),
                                             float(revolute.GetUpperLimitAttr().Get()))
                elif not joint.GetPrim().IsA(UsdPhysics.FixedJoint):
                    raise ValueError(f"Unsupported articulation joint type: {joint.GetPrim().GetTypeName()}.")
                order.append(body1[0])
                queue.append(body1[0])
        articulations[asset_path] = order
    for path in rigid_object_paths:
        if path in infos:
            infos[path].kind = "free"
    ground = next((prim for prim in stage.Traverse() if prim.HasAttribute(GROUND_ATTR)), None)
    return infos, world_shapes, articulations, ground


def _ancestors(prim):
    while prim and prim.GetPath().pathString != "/":
        yield prim
        prim = prim.GetParent()


def adjacent_pairs(order, infos):
    return {frozenset((path, infos[path].parent)) for path in order if infos[path].parent is not None}


def build_model(stage, articulations, rigid_objects, physics_dt, substeps=1, gravity=(0.0, 0.0, -9.81),
                self_collisions=None, ground_size=60.0):
    """Compile the authored stage into a MuJoCo model.

    ``articulations``/``rigid_objects`` are asset prim paths. ``self_collisions`` maps an
    articulation path to the PhysX ``enabled_self_collisions`` flag.
    """
    from pxr import UsdPhysics

    infos, world_shapes, chains, ground = collect_scene(stage, list(articulations), list(rigid_objects))
    spec = mujoco.MjSpec()
    spec.modelname = "borinot_sim2sim"
    spec.option.timestep = physics_dt / substeps
    spec.option.gravity = np.asarray(gravity, dtype=float)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.degree = False
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
    spec.visual.headlight.diffuse = (0.55, 0.55, 0.55)
    spec.visual.map.znear = 0.01
    spec.stat.extent = 4.0
    contact_timeconst = max(2.0 * spec.option.timestep, 0.002)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=(0.55, 0.68, 0.82), rgb2=(0.06, 0.08, 0.12), width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=(0.24, 0.27, 0.31), rgb2=(0.30, 0.33, 0.37), mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=(0.55, 0.58, 0.62), width=512, height=512)
    grid = spec.add_material(name="grid", texrepeat=(ground_size / 2, ground_size / 2), reflectance=0.05)
    grid.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    # Shadowless fill light; the shadow-casting spotlight follows the robot (added below).
    # A shadow-casting directional light gave shadow acne on the large ground plane.
    fill = spec.worldbody.add_light(name="fill", pos=(0.0, 0.0, 10.0), dir=(0.25, 0.35, -1.0))
    fill.type, fill.castshadow = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, False
    fill.diffuse, fill.specular = (0.35, 0.35, 0.35), (0.1, 0.1, 0.1)
    index = SceneIndex(bodies=infos)
    mesh_counter = [0]

    def add_shapes(body, info, robot):
        for shape in info.shapes:
            add_shape(body, shape, robot, info.kind == "free")

    def add_shape(body, shape, robot, free_body):
        geom = body.add_geom(name=f"{shape.prim_path}")
        geom.pos, geom.quat, geom.rgba = shape.pos, shape.quat, shape.rgba
        geom.friction = (shape.friction, 0.005, 0.0001)
        geom.solref = (contact_timeconst, 1.0)
        geom.mass = 0.0
        geom.density = 0.0
        if shape.kind == "mesh":
            mesh_counter[0] += 1
            mesh = spec.add_mesh(name=f"mesh_{mesh_counter[0]}")
            mesh.uservert = shape.vertices.astype(np.float32).reshape(-1)
            mesh.userface = shape.faces.reshape(-1)
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX if shape.collision else (
                mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            geom.type, geom.meshname = mujoco.mjtGeom.mjGEOM_MESH, mesh.name
        elif shape.kind == "cone":  # visual-only stand-in; MuJoCo has no cone primitive
            geom.type, geom.size = mujoco.mjtGeom.mjGEOM_CYLINDER, (0.5 * shape.size[0], shape.size[1], 0.0)
        else:
            geom.type = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                         "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                         "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}[shape.kind]
            geom.size = shape.size
        if not shape.collision:
            geom.contype = geom.conaffinity = 0
            geom.group = 1
        elif robot:  # robot-world and robot-ball contacts only; self contacts use explicit pairs
            geom.contype, geom.conaffinity = 1, 0
            geom.group = 0 if shape.visual else 3
        elif free_body:
            geom.contype, geom.conaffinity = 1, 1
            geom.group = 0 if shape.visual else 3
        else:  # static scenery
            geom.contype, geom.conaffinity = 0, 1
            geom.group = 0 if shape.visual else 3
        if shape.collision and shape.restitution > 0 and free_body:
            # Bouncy free objects own their contact parameters (the Borinot balls use PhysX "max").
            geom.priority = 1
            geom.solref = restitution_solref(shape.restitution)
            geom.solimp = BOUNCE_SOLIMP

    def add_inertial(body, info):
        if info.mass <= 0:
            return
        body.explicitinertial = True
        body.mass, body.ipos = info.mass, info.com
        inertia = info.inertia
        body.fullinertia = (inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2])

    # Articulations: free root, joint-frame children.
    for asset_path, order in chains.items():
        mj_bodies = {}
        for path in order:
            info = infos[path]
            parent = spec.worldbody if info.parent is None else mj_bodies[info.parent]
            body = parent.add_body(name=path, pos=info.pos, quat=info.quat)
            add_inertial(body, info)
            if info.parent is None:
                body.add_freejoint(name=path + ":root")
            elif info.joint_name is not None:
                low, high = info.joint_range_deg
                joint = body.add_joint(name=info.joint_name, type=mujoco.mjtJoint.mjJNT_HINGE,
                                       pos=info.joint_pos, axis=info.joint_axis)
                joint.range = (math.radians(low), math.radians(high))
                joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                joint.solref_limit = (contact_timeconst, 1.0)
            add_shapes(body, info, robot=True)
            mj_bodies[path] = body
            if info.parent is None and spec.light("sun") is None:
                # Spotlight 6 m above the robot's centre of mass, fixed world direction: its
                # shadow on the floor shows the flight height wherever the robot goes.
                world = quat_to_matrix(info.quat).T
                sun = body.add_light(name="sun", pos=world @ np.array((0.0, 0.0, 6.0)),
                                     dir=world @ np.array((0.0, 0.0, -1.0)))
                sun.type, sun.mode = mujoco.mjtLightType.mjLIGHT_SPOT, mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
                sun.castshadow, sun.cutoff, sun.exponent = True, 70.0, 1.0
                sun.diffuse, sun.specular = (0.55, 0.55, 0.55), (0.15, 0.15, 0.15)
        for path in order:
            info = infos[path]
            if info.joint_name is None:
                continue
            pd = spec.add_actuator(name=f"{info.joint_name}:pd", target=info.joint_name,
                                   trntype=mujoco.mjtTrn.mjTRN_JOINT)
            pd.gaintype, pd.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
            pd.gainprm[0], pd.biasprm[1], pd.biasprm[2] = 0.0, 0.0, 0.0
            pd.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            pd.forcerange = (-1e6, 1e6)
            ff = spec.add_actuator(name=f"{info.joint_name}:ff", target=info.joint_name,
                                   trntype=mujoco.mjtTrn.mjTRN_JOINT)
            ff.gaintype, ff.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
            ff.gainprm[0] = 1.0
        # PhysX self-collisions: every collider pair across non-adjacent links.
        if (self_collisions or {}).get(asset_path, False):
            adjacent = adjacent_pairs(order, infos)
            for i, a in enumerate(order):
                for b in order[i + 1:]:
                    if frozenset((a, b)) in adjacent:
                        continue
                    for shape_a in (s for s in infos[a].shapes if s.collision):
                        for shape_b in (s for s in infos[b].shapes if s.collision):
                            pair = spec.add_pair(geomname1=shape_a.prim_path, geomname2=shape_b.prim_path)
                            pair.condim = 3
                            pair.friction = (min(shape_a.friction, shape_b.friction),) * 2 + (0.005, 0.0001, 0.0001)
                            pair.solref = (contact_timeconst, 1.0)
        index.articulations[asset_path] = order

    in_articulation = {path for order in chains.values() for path in order}
    for path, info in infos.items():
        if path in in_articulation:
            continue
        body = spec.worldbody.add_body(name=path, pos=info.pos, quat=info.quat)
        if info.kind == "free":
            add_inertial(body, info)
            body.add_freejoint(name=path + ":root")
        add_shapes(body, info, robot=False)
    for shape in world_shapes:
        add_shape(spec.worldbody, shape, robot=False, free_body=False)
    if ground is not None:
        friction, _ = _physics_material(ground)
        geom = spec.worldbody.add_geom(name=ground.GetPath().pathString, type=mujoco.mjtGeom.mjGEOM_PLANE,
                                       size=(ground_size, ground_size, 0.1), material="grid")
        geom.contype, geom.conaffinity = 0, 1
        geom.friction = (friction, 0.005, 0.0001)
        geom.solref = (contact_timeconst, 1.0)
    # USD FilteredPairsAPI -> MuJoCo body exclusions.
    for path in infos:
        prim = stage.GetPrimAtPath(path)
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            target = target.pathString
            if target in infos and target != path:
                spec.add_exclude(bodyname1=path, bodyname2=target)

    model = spec.compile()
    for path in infos:
        index.body_ids[path] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, path)
    for path in in_articulation:
        info = infos[path]
        if info.joint_name is not None:
            index.joint_ids[info.joint_name] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info.joint_name)
            for kind in ("pd", "ff"):
                index.actuator_ids[(info.joint_name, kind)] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{info.joint_name}:{kind}")
    if ground is not None:
        index.ground_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground.GetPath().pathString)
    return spec, model, index


def match_paths(expression: str, paths) -> list[str]:
    """Isaac Lab prim-path regex (``env_.*``) against concrete env-0 paths, in stage order."""
    pattern = re.compile("^" + expression.replace("env_.*", "env_0") + "$")
    return [path for path in paths if pattern.match(path)]
