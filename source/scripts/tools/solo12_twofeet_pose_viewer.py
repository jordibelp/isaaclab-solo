#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Build and serve an offline Solo12 posed-reset viewer. No Isaac Sim is needed.

python3 source/scripts/tools/solo12_twofeet_pose_viewer.py
python3 source/scripts/tools/solo12_twofeet_pose_viewer.py --export /tmp/solo12-poses.html

The exported HTML includes the robot meshes and Three.js (MIT). It also works via file://.
The preview uses the task's safe reset pose, flat ground, and no physics or policy.
"""

from __future__ import annotations

import argparse
import ast
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ASSETS = HERE / "twofeet_pose_viewer"


def task_defaults():
    """Read selected source literals without importing Isaac Sim or executing the config."""
    task = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12"
    tree = ast.parse((task / "solo12_env_cfg.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "JOINT_NAMES":
                values["joint_names"] = ast.literal_eval(node.value)
            elif name == "SAFE_INITIAL_JOINT_POS":
                values["joint_pose"] = ast.literal_eval(node.value.args[0].args[1])
        if isinstance(node, ast.ClassDef) and node.name == "Solo12EnvCfg":
            for field in node.body:
                if not isinstance(field, ast.Assign):
                    continue
                name = field.targets[0].id
                if name.startswith(("twofeet_airborne_reset_", "joint_physical_limit_", "joint_soft_limit_")) or name in (
                    "flexed_initial_joint_pos_noise_range", "reset_base_lin_vel_range", "reset_base_ang_vel_range"
                ):
                    values[name] = ast.literal_eval(field.value)
    env_tree = ast.parse((task / "solo12_env.py").read_text())
    offset = next(n.value for n in env_tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_REAR_THIGH_JOINT_OFFSET_X")
    values["rear_offset"] = ast.literal_eval(offset)
    return values


def robot_model():
    """Export visual meshes and the exact link/joint tree from our USD-derived MuJoCo model."""
    model = ET.parse(REPO / "mujoco/solo12.xml").getroot()
    meshes = {}
    for mesh in model.findall("asset/mesh"):
        vertices, indices = [], []
        for line in (REPO / "mujoco/meshes" / mesh.attrib["file"]).read_text().splitlines():
            words = line.split()
            if words and words[0] == "v":
                vertices.extend(float(x) for x in words[1:4])
            elif words and words[0] == "f":
                face = [int(x.split("/")[0]) - 1 for x in words[1:]]
                for i in range(1, len(face) - 1):
                    indices.extend((face[0], face[i], face[i + 1]))
        meshes[mesh.attrib["name"]] = {"vertices": vertices, "indices": indices}
    materials = {m.attrib["name"]: [float(x) for x in m.attrib["rgba"].split()]
                 for m in model.findall("asset/material") if "rgba" in m.attrib}

    def body(node):
        joint = node.find("joint")
        return {
            "name": node.attrib["name"],
            "pos": [float(x) for x in node.attrib.get("pos", "0 0 0").split()],
            "joint": joint.attrib["name"] if joint is not None else None,
            "axis": [float(x) for x in joint.attrib["axis"].split()] if joint is not None else None,
            "visuals": [{"mesh": g.attrib["mesh"], "material": g.attrib["material"]}
                        for g in node.findall("geom") if g.attrib.get("class") == "visual"],
            "children": [body(b) for b in node.findall("body")],
        }

    return {"meshes": meshes, "materials": materials, "body": body(model.find("worldbody/body"))}


def build_html():
    payload = {"defaults": task_defaults(), "robot": robot_model()}
    template = (ASSETS / "viewer.html").read_text()
    library = (ASSETS / "vendor/three-r160.min.js").read_text()
    license_text = (ASSETS / "vendor/THREE-LICENSE").read_text()
    return template.replace("/* THREE_LIBRARY */", library).replace("/* THREE_LICENSE */", license_text).replace(
        "/* MODEL_DATA */null", json.dumps(payload, separators=(",", ":"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--export", type=Path, help="Write a standalone HTML file and exit.")
    args = parser.parse_args()
    html = build_html().encode()
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_bytes(html)
        print(args.export)
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Solo12 pose viewer: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
