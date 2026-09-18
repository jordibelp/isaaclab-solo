# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint retention and SAC best-model tracking from ``scripts/rsl_rl/train.py``.

``train.py`` launches Isaac Sim at import time, so the helpers under test are extracted from
its source and executed in an isolated namespace.
"""

import ast
import os
import re
import statistics
from collections import deque
from pathlib import Path

import pytest

TRAIN_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rsl_rl" / "train.py"
HELPERS = (
    "_get_curriculum_state_from_runner",
    "_curriculum_checkpoint_stage",
    "_save_best_model_checkpoints",
    "_patch_runner_save_with_checkpoint_retention",
    "_install_sac_best_model_hook",
)


def _load_helpers() -> dict:
    tree = ast.parse(TRAIN_SCRIPT.read_text())
    wanted = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Assign))
        and (
            getattr(node, "name", None) in HELPERS
            or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_PERIODIC_CHECKPOINT_PATTERN")
        )
    ]
    names = {getattr(node, "name", None) for node in wanted}
    assert set(HELPERS) <= names, f"missing helpers in train.py: {set(HELPERS) - names}"
    namespace = {"os": os, "re": re, "statistics": statistics, "print": lambda *a, **k: None}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace


HELPER_NS = _load_helpers()


class _Runner:
    """Minimal stand-in that writes real files through ``save``."""

    def __init__(self, log_dir, curriculum=None):
        self.log_dir = str(log_dir)
        self.tot_timesteps = 1234
        self.tot_time = 5.0
        self.saved = []
        self.logger = _Logger(log_dir)
        self.env = _Env(curriculum)

    def save(self, path, infos=None):
        Path(path).write_text("checkpoint")
        self.saved.append((os.path.basename(path), infos))


class _Logger:
    def __init__(self, log_dir):
        self.log_dir = str(log_dir)
        self.writer = object()
        self.rewbuffer = deque()
        self.calls = []

    def log(self, **kwargs):
        self.calls.append(kwargs)


class _Env:
    def __init__(self, curriculum):
        self.unwrapped = None if curriculum is None else _RawEnv(curriculum)


class _RawEnv:
    def __init__(self, global_idx):
        self.cfg = type("Cfg", (), {"curriculum_two_feet": False, "command_lin_vel_x_range": (0.0, 1.0),
                                    "base_push_force_xy_range": (0.0, 2.0)})()
        self._global_idx = global_idx
        self._max_velx_range_curriculum_idx = 0
        self._base_push_force_curriculum_idx = 0

    def get_curriculum_global_idx(self):
        return self._global_idx


def _write(directory, *names):
    for name in names:
        (directory / name).write_text("x")


def test_retention_keeps_newest_periodic_checkpoint_and_spares_best_models(tmp_path):
    runner = _Runner(tmp_path)
    _write(tmp_path, "model_100.pt", "model_200.pt", "best_model.pt", "best_model_curriculum_idx_2.pt")
    HELPER_NS["_patch_runner_save_with_checkpoint_retention"](runner, keep_last=1)

    runner.save(str(tmp_path / "model_300.pt"))

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best_model.pt", "best_model_curriculum_idx_2.pt", "model_300.pt"]


def test_retention_keeps_requested_number_and_orders_numerically(tmp_path):
    runner = _Runner(tmp_path)
    # Lexicographic ordering would wrongly rank model_900 above model_1000.
    _write(tmp_path, "model_800.pt", "model_900.pt", "model_1000.pt")
    HELPER_NS["_patch_runner_save_with_checkpoint_retention"](runner, keep_last=2)

    runner.save(str(tmp_path / "model_1100.pt"))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["model_1000.pt", "model_1100.pt"]


def test_retention_does_not_prune_when_saving_a_best_model(tmp_path):
    runner = _Runner(tmp_path)
    _write(tmp_path, "model_100.pt", "model_200.pt")
    HELPER_NS["_patch_runner_save_with_checkpoint_retention"](runner, keep_last=1)

    runner.save(str(tmp_path / "best_model_curriculum_idx_3.pt"))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "best_model_curriculum_idx_3.pt", "model_100.pt", "model_200.pt",
    ]


def test_sac_hook_writes_best_model_per_curriculum_stage(tmp_path):
    runner = _Runner(tmp_path, curriculum=1)
    HELPER_NS["_install_sac_best_model_hook"](runner, str(tmp_path), None, float("-inf"))

    runner.logger.rewbuffer.extend([10.0])
    runner.logger.log(it=100)
    runner.logger.rewbuffer.clear()
    runner.logger.rewbuffer.extend([20.0])
    runner.logger.log(it=200)

    # A worse score after the peak must not overwrite the stage best.
    runner.logger.rewbuffer.clear()
    runner.logger.rewbuffer.extend([5.0])
    runner.logger.log(it=300)

    assert (tmp_path / "best_model_curriculum_idx_1.pt").exists()
    assert (tmp_path / "best_model.pt").exists()
    saved = [infos["best_model_iteration"] for name, infos in runner.saved if name == "best_model.pt"]
    assert saved == [100, 200]
    assert runner.saved[-1][1]["best_model_value"] == 20.0
    assert runner.saved[-1][1]["best_model_curriculum_idx"] == 1
    assert len(runner.logger.calls) == 3, "the original logger.log must still run every iteration"


def test_sac_hook_resets_tracking_when_the_curriculum_advances(tmp_path):
    runner = _Runner(tmp_path, curriculum=1)
    HELPER_NS["_install_sac_best_model_hook"](runner, str(tmp_path), None, float("-inf"))

    runner.logger.rewbuffer.extend([30.0])
    runner.logger.log(it=100)

    # A lower score in the next stage is still that stage's best.
    runner.env.unwrapped._global_idx = 2
    runner.logger.rewbuffer.clear()
    runner.logger.rewbuffer.extend([12.0])
    runner.logger.log(it=200)

    assert (tmp_path / "best_model_curriculum_idx_1.pt").exists()
    assert (tmp_path / "best_model_curriculum_idx_2.pt").exists()
    assert runner.saved[-1][1]["best_model_value"] == 12.0


def test_sac_hook_is_quiet_without_rewards_or_a_writer(tmp_path):
    runner = _Runner(tmp_path, curriculum=1)
    HELPER_NS["_install_sac_best_model_hook"](runner, str(tmp_path), None, float("-inf"))

    runner.logger.log(it=10)  # empty rewbuffer
    runner.logger.rewbuffer.extend([50.0])
    runner.logger.writer = None
    runner.logger.log(it=20)

    assert runner.saved == []


def test_retention_and_sac_hook_compose_without_losing_stage_bests(tmp_path):
    """The end-to-end contract: periodic checkpoints are pruned, stage bests survive."""
    runner = _Runner(tmp_path, curriculum=1)
    HELPER_NS["_install_sac_best_model_hook"](runner, str(tmp_path), None, float("-inf"))
    HELPER_NS["_patch_runner_save_with_checkpoint_retention"](runner, keep_last=1)

    for iteration, reward, stage in [(100, 10.0, 1), (200, 25.0, 1), (300, 8.0, 2), (400, 9.0, 2)]:
        runner.env.unwrapped._global_idx = stage
        runner.logger.rewbuffer.clear()
        runner.logger.rewbuffer.extend([reward])
        runner.logger.log(it=iteration)
        runner.save(str(tmp_path / f"model_{iteration}.pt"))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "best_model.pt",
        "best_model_curriculum_idx_1.pt",
        "best_model_curriculum_idx_2.pt",
        "model_400.pt",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
