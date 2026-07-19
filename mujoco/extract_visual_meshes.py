"""Extract the Solo12 visual meshes from ``SoloFlat.usd`` into ``mujoco/meshes/*.obj``.

The MJCF collision model is authored from the USD's *primitive* colliders (boxes, knee
spheres, foot cylinders) and is what the sim-to-sim experiment actually measures. This
script only touches the ``visuals`` prototypes, so regenerating the meshes can never
change the dynamics.

``pxr`` ships inside the Isaac Sim extension cache rather than as a normal package, and
its shared libraries must be on ``LD_LIBRARY_PATH`` before the interpreter starts, so the
script re-executes itself once with the environment prepared.

    python mujoco/extract_visual_meshes.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

SITE = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
EXTSCACHE = SITE / "isaacsim" / "extscache"


def bootstrap_pxr() -> None:
    """Re-exec with the Isaac Sim USD libraries on the loader path (idempotent)."""
    if "pxr" in sys.modules or os.environ.get("_SOLO12_PXR_READY"):
        return
    matches = sorted(glob.glob(str(EXTSCACHE / "omni.usd.libs-*")))
    if not matches:
        raise SystemExit(f"could not find omni.usd.libs in {EXTSCACHE}; is isaacsim installed?")
    libs = Path(matches[0])
    env = dict(os.environ)
    env["_SOLO12_PXR_READY"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(libs), env.get("PYTHONPATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(libs / "bin"), str(Path(sys.prefix) / "lib"), env.get("LD_LIBRARY_PATH", "")])
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


bootstrap_pxr()

from pxr import Gf, Usd, UsdGeom, UsdShade  # noqa: E402

DEFAULT_USD = (
    Path(__file__).parents[1]
    / "source/isaaclab_assets/data/Robots/Solo12/SoloFlat.usd"
)
DEFAULT_OUT = Path(__file__).with_name("meshes")


def diffuse_rgb(material: Usd.Prim | None) -> tuple[float, float, float]:
    """Diffuse colour of a bound MDL material, defaulting to mid grey."""
    if material:
        for child in material.GetChildren():
            shader = UsdShade.Shader(child)
            if not shader:
                continue
            colour = shader.GetInput("diffuse_color_constant")
            if colour and colour.Get() is not None:
                c = colour.Get()
                return (float(c[0]), float(c[1]), float(c[2]))
    return (0.5, 0.5, 0.5)


def extract(usd_path: Path, out_dir: Path) -> dict:
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    root = stage.GetPrimAtPath("/Solo_description")
    if not root:
        raise SystemExit(f"/Solo_description not found in {usd_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list] = {}
    for link in root.GetChildren():
        if link.GetTypeName() != "Xform" or link.GetName() == "Looks":
            continue
        visuals = link.GetChild("visuals")
        if not visuals:
            continue

        # The meshes live in instanced prototypes, so a plain Traverse() would skip them.
        groups: dict[str, dict] = collections.defaultdict(
            lambda: {"v": [], "f": [], "rgb": (0.5, 0.5, 0.5)}
        )
        link_inv = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        for prim in Usd.PrimRange(visuals, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if prim.GetTypeName() != "Mesh" or "collision" in prim.GetName().lower():
                continue  # *_collision_* are convex-decomposition leftovers, not visuals
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                continue

            to_link = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * link_inv
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            key = material.GetPath().name if material else "default"
            group = groups[key]
            group["rgb"] = diffuse_rgb(material.GetPrim() if material else None)

            base = len(group["v"])
            for point in points:
                p = to_link.Transform(Gf.Vec3d(point[0], point[1], point[2]))
                group["v"].append((p[0], p[1], p[2]))
            offset = 0
            for count in counts:
                face = [base + indices[offset + k] + 1 for k in range(count)]
                for k in range(1, count - 1):  # fan triangulation
                    group["f"].append((face[0], face[k], face[k + 1]))
                offset += count

        entries = []
        for key, group in sorted(groups.items()):
            if not group["f"]:
                continue
            name = f"{link.GetName()}__{key}"
            with open(out_dir / f"{name}.obj", "w") as fh:
                fh.write(f"# generated by extract_visual_meshes.py from {usd_path.name}\n")
                fh.write(f"# link={link.GetName()} material={key}\n")
                for v in group["v"]:
                    fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                for f in group["f"]:
                    fh.write(f"f {f[0]} {f[1]} {f[2]}\n")
            entries.append(
                {"name": name, "file": f"{name}.obj", "rgb": group["rgb"],
                 "nv": len(group["v"]), "nf": len(group["f"])}
            )

        manifest[link.GetName()] = entries
        print(f"{link.GetName():10s} groups={len(entries):2d} tris={sum(e['nf'] for e in entries)}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    extract(args.usd, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
