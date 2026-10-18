# SPDX-License-Identifier: BSD-3-Clause
"""Task-local USD overrides; the source robot file is never modified."""
import math

import isaaclab.sim as sim_utils
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


def attach_paddle(stage, robot_path, cfg):
    dimensions = (cfg.paddle_diameter_m, cfg.paddle_thickness_m,
                  cfg.paddle_handle_length_m, cfg.paddle_handle_width_m)
    if any(not math.isfinite(v) or v <= 0 for v in dimensions):
        raise ValueError("Paddle diameter, thickness, handle length and width must be finite and positive.")
    if not math.isfinite(cfg.paddle_mount_delta_m):
        raise ValueError("paddle_mount_delta_m must be finite.")
    # The distal shaft points along local -Y. Keep the normal standoff separate.
    offset = (cfg.paddle_offset_ee[0], cfg.paddle_offset_ee[1] - cfg.paddle_mount_delta_m,
              cfg.paddle_offset_ee[2])
    ee_path = f"{robot_path}/flying_arm_2__link2"
    ee = stage.GetPrimAtPath(ee_path)
    mass_api = UsdPhysics.MassAPI(ee)
    mass_api.GetMassAttr().Set(cfg.shaft_mass_kg)
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, -0.08, -0.021))
    transverse = cfg.shaft_mass_kg * 0.16**2 / 12
    mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(transverse, 1.0e-6, transverse))
    mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(1.0))

    # Spawn in the file's authored pose, then weld using matching local frames.
    cache = UsdGeom.XformCache()
    ee_in_robot = cache.ComputeRelativeTransform(ee, stage.GetPrimAtPath(robot_path))[0]
    position = ee_in_robot.Transform(Gf.Vec3d(*offset))
    ee_quat = ee_in_robot.ExtractRotationQuat()
    local_quat = Gf.Quatd(cfg.paddle_quat_ee[0], Gf.Vec3d(*cfg.paddle_quat_ee[1:]))
    if not cfg.racket_parallel_to_arm_plane:
        # Twist about racket-local +Y (the handle axis), not a world axis.
        local_quat = local_quat * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), 90.0).GetQuat()
    rotation = ee_quat * local_quat
    paddle_cfg = sim_utils.CylinderCfg(
        radius=cfg.paddle_diameter_m / 2, height=cfg.paddle_thickness_m, axis="X",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=cfg.paddle_mass_kg),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.25, dynamic_friction=0.25, restitution=0.0,
            restitution_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.04, 0.08)),
        activate_contact_sensors=True,
    )
    paddle_path = f"{robot_path}/paddle"
    paddle_cfg.func(paddle_path, paddle_cfg, translation=tuple(position),
                    orientation=(rotation.GetReal(), *rotation.GetImaginary()))
    # A compound collider on the SAME rigid body: one mass/contact identity for
    # head and handle. The body origin stays at the head centre for task rewards.
    radius = cfg.paddle_diameter_m / 2
    handle_y = radius + cfg.paddle_handle_length_m / 2 - min(0.01, radius / 4)
    handle_cfg = sim_utils.CuboidCfg(
        size=(cfg.paddle_thickness_m, cfg.paddle_handle_length_m, cfg.paddle_handle_width_m),
        collision_props=paddle_cfg.collision_props,
        physics_material=paddle_cfg.physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.30, 0.10)),
    )
    handle_cfg.func(paddle_path + "/handle", handle_cfg, translation=(0.0, handle_y, 0.0))
    joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/paddle_fixed_joint")
    joint.CreateBody0Rel().SetTargets([ee_path])
    joint.CreateBody1Rel().SetTargets([paddle_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(local_quat))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(paddle_path)).CreateEnableSpeculativeCCDAttr().Set(True)
