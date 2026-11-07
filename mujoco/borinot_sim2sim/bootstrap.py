"""Import the real Isaac Lab Borinot code without starting Isaac Sim.

Isaac Lab needs ``pxr`` (USD) even for its config classes, and Isaac Sim ships ``pxr``
only inside its extension cache. Its shared libraries must be on ``LD_LIBRARY_PATH``
before the interpreter starts, so :func:`bootstrap` re-executes the process once with
the paths prepared (the same trick as ``mujoco/extract_visual_meshes.py``).

After that it:

1. registers the PhysX USD schema with ``pxr``;
2. installs small stand-ins for the simulator-only ``isaaclab`` modules
   (``isaaclab.sim``, ``assets``, ``envs``, ``scene``, ``sensors``, ``managers``,
   ``markers``) and for ``omni.ui``;
3. makes ``isaaclab_tasks`` packages "init-less": importing one task module no longer
   runs the package ``__init__`` that imports every task in the repository.

Configs, controllers, task math and environment classes then import unchanged.
"""

from __future__ import annotations

import glob
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_READY_FLAG = "BORINOT_SIM2SIM_USD_READY"
# Packages whose __init__ is light (gym ids, Hydra's config module); every other
# isaaclab_tasks package is imported without running its __init__.
RUN_INIT_PACKAGES = {
    "isaaclab_tasks.utils",
    "isaaclab_tasks.controllers",
    "isaaclab_tasks.direct.borinot_goto",
    "isaaclab_tasks.direct.borinot_cmd_vel",
    "isaaclab_tasks.direct.borinot_goalkeeper",
    "isaaclab_tasks.direct.borinot_goalkeeper_simple",
}
_done = False


def _extension_dirs() -> tuple[Path, Path]:
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    cache = site / "isaacsim" / "extscache"
    libs = sorted(glob.glob(str(cache / "omni.usd.libs-*")))
    physx = sorted(glob.glob(str(cache / "omni.usd.schema.physx-*")))
    if not libs or not physx:
        raise SystemExit(
            f"Could not find omni.usd.libs and omni.usd.schema.physx in {cache}. "
            "Run this inside the Isaac Lab conda environment (env_isaaclab)."
        )
    return Path(libs[-1]), Path(physx[-1])


def ensure_usd_runtime() -> None:
    """Re-execute once with Isaac Sim's USD libraries on the loader path (idempotent)."""
    if os.environ.get(_READY_FLAG) == "1":
        return
    libs, physx = _extension_dirs()
    env = dict(os.environ)
    env[_READY_FLAG] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(libs), env.get("PYTHONPATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [
        str(libs / "bin"), str(physx / "bin"), str(Path(sys.prefix) / "lib"), env.get("LD_LIBRARY_PATH", "")]))
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def register_physx_schema() -> None:
    """Make ``from pxr import PhysxSchema`` work and register its USD schema plugin."""
    import pxr
    from pxr import Plug

    _, physx = _extension_dirs()
    if str(physx / "pxr") not in pxr.__path__:
        pxr.__path__.append(str(physx / "pxr"))
    Plug.Registry().RegisterPlugins(str(physx / "plugins" / "PhysxSchema" / "resources"))
    from pxr import PhysxSchema  # noqa: F401


class _EmptyPackageLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        return None


class InitlessPackageFinder(importlib.abc.MetaPathFinder):
    """Import packages under the managed roots without executing their ``__init__.py``."""

    def __init__(self, roots: dict[str, Path], run_init: set[str]):
        self.roots = roots
        self.run_init = run_init

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.run_init:
            return None
        for root, directory in self.roots.items():
            if fullname != root and not fullname.startswith(root + "."):
                continue
            parts = fullname[len(root):].lstrip(".").split(".") if fullname != root else []
            package_dir = directory.joinpath(*parts)
            if not (package_dir / "__init__.py").is_file():
                return None
            spec = importlib.machinery.ModuleSpec(fullname, _EmptyPackageLoader(), is_package=True)
            spec.submodule_search_locations = [str(package_dir)]
            return spec
        return None


def bootstrap() -> None:
    """Prepare the process. Call before importing ``isaaclab``/``isaaclab_tasks``."""
    global _done
    if _done:
        return
    ensure_usd_runtime()
    register_physx_schema()
    # The Isaac play scripts add the vendored SAC library the same way.
    vendor = str(REPO_ROOT / "source" / "rsl_rl_sac_vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    # Small tensors: one thread avoids thread-pool overhead on every controller op.
    import torch

    torch.set_num_threads(1)
    tasks_spec = importlib.util.find_spec("isaaclab_tasks")
    if tasks_spec is None or not tasks_spec.submodule_search_locations:
        raise SystemExit("isaaclab_tasks is not installed in this Python environment.")
    import isaaclab

    roots = {
        "isaaclab_tasks": Path(list(tasks_spec.submodule_search_locations)[0]),
        "isaaclab.envs": Path(isaaclab.__file__).parent / "envs",
    }
    sys.meta_path.insert(0, InitlessPackageFinder(roots, RUN_INIT_PACKAGES))
    from . import isaac_stubs

    isaac_stubs.install()
    _done = True
