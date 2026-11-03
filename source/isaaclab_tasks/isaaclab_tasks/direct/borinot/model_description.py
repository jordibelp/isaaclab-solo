# SPDX-License-Identifier: BSD-3-Clause
"""Export nominal spawned USD calibration to the simulator-independent arm model.

USD authoring contains configured payload/inertial overrides, but not the hidden
per-clone mass/CoM/friction randomization written through PhysX tensor setters.
No articulation dynamics, joint force, or state readbacks are used here.
"""
import math


def nominal_model_description(stage, robot_path, joint_names, armature, automatic_properties=None):
    from pxr import Gf, Usd, UsdPhysics

    root = stage.GetPrimAtPath(robot_path)

    def rotation(quaternion):
        # Gf transforms row vectors; transpose into our column-vector convention.
        matrix = Gf.Matrix3d(Gf.Quatd(quaternion).GetNormalized())
        return [[float(matrix[j][i]) for j in range(3)] for i in range(3)]

    rigid = {prim.GetPath(): prim for prim in Usd.PrimRange(root)
             if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [UsdPhysics.Joint(prim) for prim in Usd.PrimRange(root)
              if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get()]
    base_path = root.GetPath().AppendChild("base")
    ordered = [base_path]
    specs = []
    pending = dict(rigid)
    while pending:
        path = next((p for p in ordered if p in pending), None)
        if path is None:
            raise ValueError("model_wrench requires a rigid tree rooted at base with no world constraints.")
        prim = pending.pop(path)
        api = UsdPhysics.MassAPI(prim)
        mass = float(api.GetMassAttr().Get())
        moments = tuple(api.GetDiagonalInertiaAttr().Get())
        axes = rotation(api.GetPrincipalAxesAttr().Get())
        inertia = [[sum(axes[i][k]*moments[k]*axes[j][k] for k in range(3))
                    for j in range(3)] for i in range(3)]
        spec = dict(name=prim.GetName(), mass=mass, com=list(api.GetCenterOfMassAttr().Get()), inertia=inertia)
        if not any(moments):
            # Compound tools leave USD diagonalInertia=0 for shape-based inference.
            # Resolve ONCE from nominal asset calibration, never randomized masses.
            if automatic_properties is None or prim.GetName() not in automatic_properties:
                raise ValueError(f"Export nominal automatic mass properties for {prim.GetName()!r} first.")
            spec.update(automatic_properties[prim.GetName()])
        if path != base_path:
            incoming = [j for j in joints if j.GetBody1Rel().GetTargets() == [path]]
            if len(incoming) != 1:
                raise ValueError("Every model body needs exactly one incoming joint.")
            joint = incoming[0]
            parent = joint.GetBody0Rel().GetTargets()
            if len(parent) != 1 or parent[0] not in ordered[:len(specs)]:
                raise ValueError("Invalid nominal model parent order.")
            name = joint.GetPrim().GetName()
            revolute = joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
            if not revolute and not joint.GetPrim().IsA(UsdPhysics.FixedJoint):
                raise ValueError("model_wrench supports revolute joints and fixed tools only.")
            if revolute and name not in joint_names:
                raise ValueError(f"Unexpected revolute joint {name!r} in nominal model.")
            axis = [0., 0., 0.]
            if revolute:
                axis["XYZ".index(UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get())] = 1.
            spec.update(parent=ordered.index(parent[0]), dof=list(joint_names).index(name) if revolute else -1,
                        p0=list(joint.GetLocalPos0Attr().Get()), r0=rotation(joint.GetLocalRot0Attr().Get()),
                        p1=list(joint.GetLocalPos1Attr().Get()), r1=rotation(joint.GetLocalRot1Attr().Get()), axis=axis)
        specs.append(spec)
        children = [j.GetBody1Rel().GetTargets()[0] for j in joints
                    if j.GetBody0Rel().GetTargets() == [path] and len(j.GetBody1Rel().GetTargets()) == 1]
        ordered.extend(p for p in children if p not in ordered)
    if len(specs) < 3 or len(joints) != len(specs)-1:
        raise ValueError("model_wrench requires an unconstrained tree: base, two arm links, optional fixed tools.")
    if not isinstance(armature, (int, float)) or not math.isfinite(armature) or armature < 0:
        raise ValueError("model_wrench requires a finite nonnegative scalar actuator armature.")
    return {"format_version": 1, "source": "nominal USD plus cached automatic tool calibration; no randomized masses/inertias",
            "joint_names": list(joint_names), "armature": [float(armature)]*2, "bodies": specs}
