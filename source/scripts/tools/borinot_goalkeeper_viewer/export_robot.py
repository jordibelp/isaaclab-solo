#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Refresh cached visual geometry from USD. Requires usd-core, not Isaac Sim."""

import array
import base64
import hashlib
import json
from pathlib import Path
import sys

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from borinot_goalkeeper_viewer import task_defaults


def encoded(values, code):
    data = array.array(code, values)
    if sys.byteorder != "little":
        data.byteswap()
    return base64.b64encode(data.tobytes()).decode()


def matrix(value):
    # Gf uses row vectors; its row-major array is Three.js's transposed column matrix.
    return [float(x) for row in value for x in row]


def quaternion(value):
    return [*map(float, value.GetImaginary()), float(value.GetReal())]


def export_model():
    asset = Path(task_defaults()["usd_path"])
    stage = Usd.Stage.Open(str(asset))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != "Z":
        raise ValueError("Exporter expects this task's metre-scale, Z-up asset.")
    cache = UsdGeom.XformCache()
    bodies = {str(p.GetPath()): {"name": p.GetName(), "meshes": [], "colliders": []}
              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            rev = UsdPhysics.RevoluteJoint(prim)
            if not rev:
                raise ValueError(f"Unsupported joint: {prim.GetPath()}")
            joints.append({"name": prim.GetName(), "parent": stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0]).GetName(),
                           "child": stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0]).GetName(),
                           "pos0": list(joint.GetLocalPos0Attr().Get()), "rot0": quaternion(joint.GetLocalRot0Attr().Get()),
                           "pos1": list(joint.GetLocalPos1Attr().Get()), "rot1": quaternion(joint.GetLocalRot1Attr().Get()),
                           "axis": str(rev.GetAxisAttr().Get()),
                           "limits_deg": [rev.GetLowerLimitAttr().Get(), rev.GetUpperLimitAttr().Get()]})
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder)):
            continue
        ancestor = prim.GetParent()
        while ancestor and str(ancestor.GetPath()) not in bodies:
            ancestor = ancestor.GetParent()
        if not ancestor:
            continue
        body = bodies[str(ancestor.GetPath())]
        local = cache.ComputeRelativeTransform(prim, ancestor)[0]
        item = {"name": str(prim.GetPath()), "matrix": matrix(local)}
        if not prim.IsA(UsdGeom.Mesh):
            item["type"] = prim.GetTypeName()
            for name in ("size", "height", "radius", "axis"):
                attr = prim.GetAttribute(name)
                if attr:
                    item[name] = attr.Get()
            body["colliders"].append(item)
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == "invisible":
            continue
        points = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexIndicesAttr().Get()
        indices, start = [], 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            for i in range(1, count - 1):
                indices.extend((faces[start], faces[start + i], faces[start + i + 1]))
            start += count
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            for i in range(0, len(indices), 3):
                indices[i + 1], indices[i + 2] = indices[i + 2], indices[i + 1]
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        color = [0.56, 0.63, 0.75]
        if material:
            for shader in material.GetPrim().GetChildren():
                for attr_name in ("inputs:diffuseColor", "inputs:diffuse_color_constant"):
                    attr = shader.GetAttribute(attr_name)
                    if attr and attr.Get() is not None:
                        color = list(attr.Get())
        item.update(vertices=encoded((x for p in points for x in p), "f"), indices=encoded(indices, "I"),
                    color=color, vertex_count=len(points), triangle_count=len(indices) // 3)
        body["meshes"].append(item)
    return {"usd_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(), "bodies": list(bodies.values()), "joints": joints}


if __name__ == "__main__":
    result = export_model()
    output = Path(__file__).with_name("robot_model.json")
    output.write_text(json.dumps(result, separators=(",", ":")))
    print(f"Exported {len(result['bodies'])} bodies to {output} ({output.stat().st_size:,} bytes)")
