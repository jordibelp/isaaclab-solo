# SPDX-License-Identifier: BSD-3-Clause
"""Install configurable adaptive LR limits into the active environment's RSL-RL.

Run with that environment's Python, first without arguments to preview, then with
``--apply``. No pip upgrade, simulator, GPU, Slurm job, or service restart is used.
The repo's RslRlOnPolicyRunnerCfg exposes agent.min_learning_rate/max_learning_rate;
agent.algorithm.learning_rate still selects the initial LR. Defaults stay 1e-5/1e-2.

Only recognized source fragments are changed. All files are checked before any
write; applying twice is a no-op. Original files are saved beside each source as
``<filename>.before-configurable-lr.bak``. To undo, restore both backed-up sources.
After upgrading RSL-RL, move the prior backups aside before reapplying; unknown
versions fail rather than guess.
Already running Python processes retain their loaded code.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil


PATCHES = {
    "algorithms/ppo.py": (
        ("import torch\n", "import math\nimport torch\n"),
        (
            '        multi_gpu_cfg: dict | None = None,\n    ) -> None:',
            '        multi_gpu_cfg: dict | None = None,\n'
            '        min_learning_rate: float = 1e-5,\n'
            '        max_learning_rate: float = 1e-2,\n    ) -> None:',
        ),
        (
            '        # Device-related parameters\n',
            '        # Configurable adaptive LR limits (legacy defaults are unchanged).\n'
            '        for name, value in (("learning_rate", learning_rate),\n'
            '                            ("min_learning_rate", min_learning_rate),\n'
            '                            ("max_learning_rate", max_learning_rate)):\n'
            '            if (isinstance(value, bool) or not isinstance(value, (int, float))\n'
            '                    or not math.isfinite(value) or value <= 0):\n'
            '                raise ValueError(f"{name} must be finite and positive, got {value!r}.")\n'
            '        if min_learning_rate > max_learning_rate:\n'
            '            raise ValueError("min_learning_rate must not exceed max_learning_rate.")\n'
            '        if schedule == "adaptive" and not min_learning_rate <= learning_rate <= max_learning_rate:\n'
            '            raise ValueError("Adaptive PPO requires min_learning_rate <= learning_rate <= max_learning_rate.")\n'
            '        self.min_learning_rate = min_learning_rate\n'
            '        self.max_learning_rate = max_learning_rate\n\n'
            '        # Device-related parameters\n',
        ),
        (
            'self.learning_rate = max(1e-5, self.learning_rate / 1.5)',
            'self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)',
        ),
        (
            'self.learning_rate = min(1e-2, self.learning_rate * 1.5)',
            'self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)',
        ),
    ),
    "runners/on_policy_runner.py": (
        (
            '        # Initialize the algorithm\n',
            '        # Root agent fields control the adaptive PPO limits.\n'
            '        self.alg_cfg["min_learning_rate"] = self.cfg.get("min_learning_rate", 1e-5)\n'
            '        self.alg_cfg["max_learning_rate"] = self.cfg.get("max_learning_rate", 1e-2)\n\n'
            '        # Initialize the algorithm\n',
        ),
    ),
}


def patched_source(source: str, relative_path: str) -> str:
    """Return a checked patch, refusing changed/ambiguous upstream fragments."""
    for old, new in PATCHES[relative_path]:
        if source.count(new) == 1:
            continue
        if source.count(old) != 1:
            raise RuntimeError(f"Unsupported RSL-RL source in {relative_path}: {old!r}")
        source = source.replace(old, new, 1)
    compile(source, relative_path, "exec")
    return source


def install(package_dir: Path, *, apply: bool = False) -> list[Path]:
    """Validate both modules, then optionally back up and update changed files."""
    changes = []
    for relative_path in PATCHES:
        path = package_dir / relative_path
        source = path.read_text()
        updated = patched_source(source, relative_path)
        if updated != source:
            backup = path.with_name(path.name + ".before-configurable-lr.bak")
            if backup.exists():
                raise RuntimeError(f"Refusing to overwrite an existing backup: {backup}")
            changes.append((path, backup, updated))
    if apply:
        for path, backup, _ in changes:
            shutil.copy2(path, backup)
        for path, _, updated in changes:
            path.write_text(updated)
    return [path for path, _, _ in changes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Back up and patch the active environment's RSL-RL.")
    args = parser.parse_args()
    spec = importlib.util.find_spec("rsl_rl")
    if spec is None or spec.origin is None:
        raise RuntimeError("RSL-RL is not installed in this Python environment.")
    paths = install(Path(spec.origin).parent, apply=args.apply)
    for path in paths:
        print(f"{'Updated' if args.apply else 'Would update'}: {path}")
    if not paths:
        print("Configurable PPO LR limits already installed; no changes.")


if __name__ == "__main__":
    main()
