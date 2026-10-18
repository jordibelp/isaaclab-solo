#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Serve or export an offline 3D scene inspector. No Isaac Sim or policy needed."""

import argparse
import ast
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ASSETS = HERE / "borinot_goalkeeper_viewer"
CONFIG = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/borinot_goalkeeper_env_cfg.py"
FIELDS = (
    "robot_home", "robot_home_quat", "joint_names", "joint_target_scale", "paddle_diameter_m", "paddle_thickness_m",
    "paddle_handle_length_m", "paddle_handle_width_m", "paddle_mass_kg", "paddle_offset_ee", "paddle_mount_delta_m", "paddle_quat_ee",
    "keeper_plane_x", "target_center_yz", "ball_radius_m", "goal_center", "goal_size", "launch_x_range",
    "launch_z", "launch_y_noise", "shot_half_extents_yz", "shot_flight_time_ranges_s", "shot_level",
    "goal_area_x_m", "goal_area_y_m", "goal_area_z_m", "reset_xy_noise_m", "reset_z_noise_m",
    "reset_grid_enabled", "reset_grid_start_level", "reset_grid_y_offsets_m", "reset_grid_z_offsets_m",
    "enabled_self_collisions", "racket_parallel_to_arm_plane",
)


def keyword(call, name):
    return next(k.value for k in call.keywords if k.arg == name)


def task_defaults():
    """Read literals without importing or executing the task config."""
    tree = ast.parse(CONFIG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BorinotGoalkeeperEnvCfg")
    fields = {n.targets[0].id: n.value for n in cls.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    values = {name: ast.literal_eval(fields[name]) for name in FIELDS}
    values["env_spacing"] = ast.literal_eval(keyword(fields["scene"], "env_spacing"))
    values["joint_pose"] = ast.literal_eval(keyword(keyword(fields["robot"], "init_state"), "joint_pos"))
    asset_expr = keyword(keyword(fields["robot"], "spawn"), "usd_path")
    # Resolve only the supported Path expression; never eval config source.
    expected = 'str(Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd")'
    if ast.dump(asset_expr) != ast.dump(ast.parse(expected, mode="eval").body):
        raise ValueError("Robot asset path changed. Update the viewer exporter before previewing this config.")
    values["usd_path"] = str(CONFIG.resolve().parents[4] / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd")
    values["config_path"] = str(CONFIG)
    values["config_sha256"] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    return values


def build_html():
    defaults = task_defaults()
    robot = json.loads((ASSETS / "robot_model.json").read_text())
    asset_hash = hashlib.sha256(Path(defaults["usd_path"]).read_bytes()).hexdigest()
    if robot["usd_sha256"] != asset_hash:
        raise ValueError("USD changed. Refresh geometry with borinot_goalkeeper_viewer/export_robot.py.")
    vendor = HERE / "twofeet_pose_viewer/vendor"
    return ((ASSETS / "viewer.html").read_text()
            .replace("/* THREE_LIBRARY */", (vendor / "three-r160.min.js").read_text())
            .replace("/* THREE_LICENSE */", (vendor / "THREE-LICENSE").read_text())
            .replace("/* COLLISION_CODE */", (ASSETS / "collision.js").read_text())
            .replace("/* CURRICULUM_CODE */", (ASSETS / "curriculum.js").read_text())
            .replace("/* SPAWN_CODE */", (ASSETS / "spawn.js").read_text())
            .replace("/* MODEL_DATA */null", json.dumps({"defaults": defaults, "robot": robot}, separators=(",", ":"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--export", type=Path, help="Save a standalone HTML file and exit.")
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
    print(f"Borinot scene inspector: {url} (Ctrl+C to stop)", flush=True)
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
