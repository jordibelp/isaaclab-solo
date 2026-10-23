# SPDX-License-Identifier: BSD-3-Clause
"""Velocity-task-only shaft/payload mass overrides and optional goalkeeper racket."""
import torch
from pxr import Gf, UsdPhysics

from isaaclab.utils.math import quat_from_matrix
from isaaclab_tasks.direct.borinot_goalkeeper.paddle import attach_paddle

from .task_math import calf_payload_properties


def configure_end_effector(stage, robot_path, cfg):
    if cfg.arm_holds_racket:
        attach_paddle(stage, robot_path, cfg)
    # Always replace the source USD's baked-in distal mass, even for zero payload.
    # The optional racket is a separate body and is not counted in this merge.
    mass, com, inertia = calf_payload_properties(cfg.shaft_mass_kg, cfg.ee_includes_puntual_mass)
    if cfg.ee_includes_puntual_mass == 0:
        diagonal, quaternion = inertia.diag(), torch.tensor((1.0, 0.0, 0.0, 0.0))
    else:
        diagonal, axes = torch.linalg.eigh(inertia)
        # Eigenvectors are principal axes expressed in the calf frame. Enforce
        # a proper rotation before converting its columns to a wxyz quaternion.
        if torch.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        quaternion = quat_from_matrix(axes[None])[0]
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{robot_path}/flying_arm_2__link2"))
    api.GetMassAttr().Set(mass)
    api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com.tolist()))
    api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal.tolist()))
    api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*quaternion[1:].tolist())))
