# SPDX-License-Identifier: BSD-3-Clause
"""Shared spawn-layer mass overrides. Never write to the source USD asset."""
import math

import torch

LEG_BODY_NAMES = ("flying_arm_2__link1", "flying_arm_2__link2")


def merge_point_masses(mass, com, inertia, point_mass, positions):
    """Merge equal ideal point masses; return mass, CoM and full own-CoM tensor."""
    com, inertia, positions = (torch.as_tensor(x, dtype=torch.float64) for x in (com, inertia, positions))
    if not math.isfinite(mass) or mass <= 0 or not math.isfinite(point_mass) or point_mass < 0:
        raise ValueError("Base mass must be positive and motor point mass non-negative, both finite.")
    if com.shape != (3,) or inertia.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected CoM (3,), inertia (3,3), and point positions (N,3).")
    if not all(torch.isfinite(x).all() for x in (com, inertia, positions)):
        raise ValueError("Mass properties and point positions must be finite.")
    total = mass + len(positions) * point_mass
    combined_com = (mass * com + point_mass * positions.sum(0)) / total
    offsets = torch.cat(((com - combined_com)[None], positions - combined_com))
    masses = torch.tensor([mass] + [point_mass] * len(positions), dtype=torch.float64)
    shifts = offsets.square().sum(-1)[:, None, None] * torch.eye(3, dtype=torch.float64)
    shifts -= offsets[:, :, None] * offsets[:, None, :]
    combined_inertia = inertia + (masses[:, None, None] * shifts).sum(0)
    return total, combined_com, (combined_inertia + combined_inertia.T) / 2


def validate_inertial_properties(cfg):
    for name in ("include_motor_masses", "use_inference_leg_inertias"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"inertial_properties.{name} must be a boolean.")
    if cfg.include_motor_masses:
        positions = torch.as_tensor(cfg.motor_mass_positions_b, dtype=torch.float64)
        if positions.shape != (6, 3) or not torch.isfinite(positions).all():
            raise ValueError("motor_mass_positions_b must contain six finite XYZ positions in metres.")
        if not math.isfinite(cfg.motor_mass_kg) or cfg.motor_mass_kg < 0:
            raise ValueError("motor_mass_kg must be finite and non-negative.")
    if cfg.use_inference_leg_inertias:
        mass, com, inertia = (torch.as_tensor(getattr(cfg, key), dtype=torch.float64) for key in
                              ("leg_masses_kg", "leg_com_positions_l", "leg_inertias_l_kg_m2"))
        if mass.shape != (2,) or com.shape != (2, 3) or inertia.shape != (2, 3, 3):
            raise ValueError("Leg properties must describe two masses, two XYZ CoMs and two 3x3 tensors.")
        if not all(torch.isfinite(x).all() for x in (mass, com, inertia)) or not (mass > 0).all():
            raise ValueError("Leg masses must be positive and all leg properties finite.")
        if not torch.allclose(inertia, inertia.transpose(-1, -2), atol=1e-12, rtol=0):
            raise ValueError("Leg inertia tensors must be symmetric.")
        moments = torch.linalg.eigvalsh(inertia)
        if not (moments > 0).all() or (2 * moments[:, -1] > moments.sum(-1) + 1e-12).any():
            raise ValueError("Leg inertia tensors must be positive and satisfy the triangle inequality.")


def apply_inertial_properties(stage, robot_path, cfg):
    """Apply once after racket/payload construction, before cloning and physics.

    Inference leg settings replace the COMPLETE legacy shaft/payload description;
    they do not add another tip mass. A separately attached racket stays separate.
    Motor masses are lumped into the base with no extra body, collider or joint.
    """
    from pxr import Gf, UsdPhysics
    from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

    validate_inertial_properties(cfg)

    def write(name, mass, com, inertia):
        moments, axes = torch.linalg.eigh(torch.as_tensor(inertia, dtype=torch.float64))
        if torch.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        q = quat_from_matrix(axes[None])[0]
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{robot_path}/{name}"))
        api.GetMassAttr().Set(float(mass))
        api.GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com)))
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*moments.tolist()))
        api.GetPrincipalAxesAttr().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(*q[1:].tolist())))

    if cfg.use_inference_leg_inertias:
        for name, mass, com, inertia in zip(LEG_BODY_NAMES, cfg.leg_masses_kg,
                                           cfg.leg_com_positions_l, cfg.leg_inertias_l_kg_m2):
            write(name, mass, com, inertia)
    if cfg.include_motor_masses and cfg.motor_mass_kg:
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{robot_path}/base"))
        axes = api.GetPrincipalAxesAttr().Get()
        q = torch.tensor([axes.GetReal(), *axes.GetImaginary()], dtype=torch.float64)
        rotation = matrix_from_quat((q / q.norm())[None])[0] if q.norm() else torch.eye(3, dtype=torch.float64)
        inertia = rotation @ torch.diag(torch.tensor(tuple(api.GetDiagonalInertiaAttr().Get()), dtype=torch.float64)) @ rotation.T
        mass, com, inertia = merge_point_masses(float(api.GetMassAttr().Get()),
            tuple(api.GetCenterOfMassAttr().Get()), inertia, cfg.motor_mass_kg, cfg.motor_mass_positions_b)
        write("base", mass, com, inertia)
