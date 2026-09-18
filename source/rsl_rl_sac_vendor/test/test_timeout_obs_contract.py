# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard the env-side half of SAC's timeout-aware critic target.

``DirectRLEnv.step`` publishes ``extras["time_outs_obs"]``, the observation reached at a
timeout before the automatic reset overwrites it. ``SAC.process_env_step`` needs it to store
the real next state and to mark the transition as bootstrappable.

An env that overrides ``step`` opts out of the base-class block. Every consumer guards with
``if "time_outs_obs" in extras`` and degrades silently, so the omission shows up only as a
critic that quietly treats every truncation as a terminal state worth zero. That is exactly
how it went unnoticed in ``Solo12Env`` between 2026-06-24 and 2026-09-18.

This test fails on the *next* env that overrides ``step`` without republishing the key.
"""

import ast
from pathlib import Path

import pytest

DIRECT_TASKS = Path(__file__).resolve().parents[3] / "source/isaaclab_tasks/isaaclab_tasks/direct"


def is_env_class(node: ast.ClassDef) -> bool:
    """True for RL env classes, which are the only ones the extras contract applies to."""
    return any(
        isinstance(base, ast.Name) and base.id.endswith("Env")
        or isinstance(base, ast.Attribute) and base.attr.endswith("Env")
        for base in node.bases
    )


def step_overrides() -> list[tuple[Path, ast.FunctionDef]]:
    """Every ``step`` method defined on an RL env class under the direct-task tree."""
    found = []
    for path in sorted(DIRECT_TASKS.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not is_env_class(node):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "step":
                    found.append((path, item))
    return found


@pytest.mark.skipif(not DIRECT_TASKS.is_dir(), reason="isaaclab_tasks is not in this checkout")
def test_every_step_override_publishes_time_outs_obs():
    overrides = step_overrides()
    assert overrides, "Found no step() overrides to check; the search path is probably wrong."

    missing = [
        f"{path.relative_to(DIRECT_TASKS.parents[2])}:{node.lineno}"
        for path, node in overrides
        if "time_outs_obs" not in ast.get_source_segment(path.read_text(), node)
    ]
    assert not missing, (
        "These step() overrides never set extras['time_outs_obs'], so SAC will silently "
        "treat every episode timeout as a terminal state worth zero and store the post-reset "
        "observation as the transition's next state. Copy the reset block from "
        f"DirectRLEnv.step into each one:\n  " + "\n  ".join(missing)
    )
