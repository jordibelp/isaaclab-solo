#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Serve the offline Solo12 joint viewer without Isaac Sim or extra Python packages.

python3 source/scripts/tools/solo12_joint_web_viewer.py
python3 source/scripts/tools/solo12_joint_web_viewer.py --export /tmp/solo12-joints.html
"""

from __future__ import annotations

import argparse
import ast
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

from solo12_twofeet_pose_viewer import ASSETS, REPO, robot_model, task_defaults


TEMPLATE = Path(__file__).with_name("joint_web_viewer") / "viewer.html"
COMMAND_LIMITS = {
    "joint_physical_limit_hip": [-70.0, 70.0],
    "joint_physical_limit_calf": [-179.0, 179.0],
    "use_asymmetric_thigh_limits": True,
    "joint_physical_limit_front_thigh": [-145.0, 55.0],
    "joint_physical_limit_rear_thigh": [-55.0, 145.0],
    "joint_soft_limit_hip_delta": 20.0,
    "joint_soft_limit_thigh_delta": 20.0,
    "joint_soft_limit_calf_delta": 24.0,
}


def build_html():
    defaults = task_defaults()
    tree = ast.parse((REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env_cfg.py").read_text())
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Solo12EnvCfg")
    for node in config.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in ("use_asymmetric_thigh_limits", "use_asymmetric_calf_limits"):
                defaults[key] = ast.literal_eval(node.value)
    task_limits = {key: value for key, value in defaults.items()
                   if key.startswith(("joint_physical_limit_", "joint_soft_limit_", "use_asymmetric_"))}
    payload = {"defaults": defaults, "task_limits": task_limits,
               "command_limits": task_limits | COMMAND_LIMITS, "robot": robot_model()}
    return TEMPLATE.read_text().replace("/* THREE_LIBRARY */", (ASSETS / "vendor/three-r160.min.js").read_text()).replace(
        "/* THREE_LICENSE */", (ASSETS / "vendor/THREE-LICENSE").read_text()
    ).replace("/* MODEL_DATA */null", json.dumps(payload, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export", type=Path, help="Write a self-contained offline HTML file and exit.")
    args = parser.parse_args()
    html = build_html().encode()
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_bytes(html)
        print(args.export)
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
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
    print(f"Solo12 joint viewer: {url}  (Ctrl+C to stop)", flush=True)
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
