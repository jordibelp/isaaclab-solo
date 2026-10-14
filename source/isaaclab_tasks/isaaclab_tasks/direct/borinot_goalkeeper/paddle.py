# SPDX-License-Identifier: BSD-3-Clause
"""Task-local USD overrides; the source robot file is never modified."""
import isaaclab.sim as sim_utils
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics


def attach_paddle(stage, robot_path, cfg):
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
    position = ee_in_robot.Transform(Gf.Vec3d(*cfg.paddle_offset_ee))
    ee_quat = ee_in_robot.ExtractRotationQuat()
    local_quat = Gf.Quatd(cfg.paddle_quat_ee[0], Gf.Vec3d(*cfg.paddle_quat_ee[1:]))
    rotation = ee_quat * local_quat
    paddle_cfg = sim_utils.CuboidCfg(
        size=cfg.paddle_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(enable_gyroscopic_forces=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=cfg.paddle_mass_kg),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.25, dynamic_friction=0.25, restitution=0.0,
            restitution_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.65, 0.9)),
        activate_contact_sensors=True,
    )
    paddle_path = f"{robot_path}/paddle"
    paddle_cfg.func(paddle_path, paddle_cfg, translation=tuple(position),
                    orientation=(rotation.GetReal(), *rotation.GetImaginary()))
    joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_path}/paddle_fixed_joint")
    joint.CreateBody0Rel().SetTargets([ee_path])
    joint.CreateBody1Rel().SetTargets([paddle_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*cfg.paddle_offset_ee))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(cfg.paddle_quat_ee[0], Gf.Vec3f(*cfg.paddle_quat_ee[1:])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(paddle_path)).CreateEnableSpeculativeCCDAttr().Set(True)
