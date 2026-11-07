"""Run an Isaac Lab Borinot play script unchanged on the MuJoCo backend.

:func:`run_play_script` prepares the process (bootstrap.py), replaces ``isaaclab.app.AppLauncher``
and ``isaaclab.utils.assets.retrieve_file_path``, registers the Borinot tasks and executes
``source/scripts/rsl_rl/<script>`` as ``__main__``. Its flags, Hydra overrides, live controls
and printed reports therefore stay identical to the Isaac Sim player.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import runpy
import sys
import types

from . import backend, bootstrap

PLAY_DIR = bootstrap.REPO_ROOT / "source" / "scripts" / "rsl_rl"
TASK_PACKAGES = ("isaaclab_tasks.direct.borinot_goto", "isaaclab_tasks.direct.borinot_cmd_vel",
                 "isaaclab_tasks.direct.borinot_goalkeeper", "isaaclab_tasks.direct.borinot_goalkeeper_simple")
# Sim-to-sim playback uses the nominal robot. An explicit Hydra value always wins.
DEFAULT_OVERRIDES = {"env.include_events_randomization": "False"}
IGNORED_ISAAC_FLAGS = ("--kit_args", "--experience", "--livestream", "--rendering_mode", "--xr")


class SimulationApp:
    def is_running(self):
        return backend.is_running()

    def update(self):
        if backend._current is not None:
            backend._current.render()

    def close(self):
        backend.shutdown()


class AppLauncher:
    """Accepts the Isaac Lab launcher flags the MuJoCo backend can honour, plus its own options."""

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("MuJoCo backend")
        group.add_argument("--headless", action="store_true", help="No viewer window and no control panel.")
        group.add_argument("--device", default="cpu", help="Ignored: the MuJoCo backend runs on the CPU.")
        group.add_argument("--enable_cameras", action="store_true", help="Offscreen rendering (used by --video).")
        group.add_argument("--mj_substeps", "--mj-substeps", type=int, default=None,
                           help="MuJoCo steps per Isaac physics step (default: 1 for flight, 10 for goalkeeper).")
        group.add_argument("--mj_viewer_ui", "--mj-viewer-ui", action="store_true",
                           help="Show the MuJoCo viewer's own side panels.")
        group.add_argument("--mj_fps", type=float, default=30.0,
                           help="Maximum viewer/control-panel refresh rate; physics/policy rates stay unchanged.")
        for flag in IGNORED_ISAAC_FLAGS:
            group.add_argument(flag, default=None, help=argparse.SUPPRESS)
        group.add_argument("--verbose", "--info", action="store_true", help=argparse.SUPPRESS)

    def __init__(self, launcher_args=None, **kwargs):
        args = launcher_args if isinstance(launcher_args, argparse.Namespace) else argparse.Namespace(
            **dict(launcher_args or {}))
        for key, value in kwargs.items():
            setattr(args, key, value)
        ignored = [flag for flag in IGNORED_ISAAC_FLAGS if getattr(args, flag.lstrip("-"), None) is not None]
        if ignored:
            print(f"[MuJoCo] Ignoring Isaac Sim launcher flags: {', '.join(ignored)}.", flush=True)
        if getattr(args, "device", "cpu") not in (None, "cpu"):
            print(f"[MuJoCo] --device={args.device} ignored: environment and policy run on the CPU.", flush=True)
        args.device = "cpu"  # the play script copies this into env and agent configs
        max_fps = getattr(args, "mj_fps", 30.0)
        if not math.isfinite(max_fps) or max_fps <= 0:
            raise ValueError("--mj_fps must be positive and finite.")
        backend.OPTIONS.update(headless=bool(getattr(args, "headless", False)),
                               substeps=getattr(args, "mj_substeps", None),
                               max_fps=max_fps,
                               show_viewer_ui=bool(getattr(args, "mj_viewer_ui", False)))
        self.app = SimulationApp()


def retrieve_file_path(path: str, download_dir: str | None = None, force_download: bool = True) -> str:
    """Local files only (the Isaac version also fetches Nucleus/HTTP assets)."""
    resolved = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Unable to find the file: {path}")
    return resolved


def _install_modules():
    import isaaclab
    import isaaclab.utils

    app = types.ModuleType("isaaclab.app")
    app.AppLauncher = AppLauncher
    assets = types.ModuleType("isaaclab.utils.assets")
    assets.retrieve_file_path = retrieve_file_path
    sys.modules["isaaclab.app"], sys.modules["isaaclab.utils.assets"] = app, assets
    isaaclab.app, isaaclab.utils.assets = app, assets


def run_play_script(script: str, default_substeps: int = 1):
    """Execute ``source/scripts/rsl_rl/<script>`` with the current command line."""
    bootstrap.bootstrap()
    _install_modules()
    for package in TASK_PACKAGES:  # gym registration; isaaclab_tasks itself stays init-less
        importlib.import_module(package)
    backend.OPTIONS["default_substeps"] = default_substeps
    path = PLAY_DIR / script
    argv = sys.argv[1:]
    keys = {token.split("=", 1)[0].lstrip("+") for token in argv if "=" in token and not token.startswith("-")}
    extra = [f"{key}={value}" for key, value in DEFAULT_OVERRIDES.items() if key not in keys]
    if extra:
        print(f"[MuJoCo] Default override: {' '.join(extra)} (nominal robot; pass the key to change it).", flush=True)
    sys.argv = [str(path), *argv, *extra]
    runpy.run_path(str(path), run_name="__main__")
