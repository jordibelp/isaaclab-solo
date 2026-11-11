# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-free checks of floor calibration, sampling, and terrain-face assignment."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE = Path(__file__).parents[3] / "isaaclab_tasks/isaaclab_tasks/direct/solo12/floor_stiffness.py"
SPEC = importlib.util.spec_from_file_location("solo12_floor_stiffness_test", MODULE)
floor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(floor)


@pytest.mark.parametrize(
    "values",
    [[], [1.0], [1.0, 2.0, 3.0], None, [0.0, 5.0], [-1.0, 2.0], [5.0, 0.2], [float("nan"), 1.0], [0.2, float("inf")]],
)
def test_invalid_indentation_ranges(values):
    with pytest.raises(ValueError, match="delta_x_floor_stiffness_range_mm"):
        floor.validate_delta_x_range(values)


def test_disabled_and_constant_ranges():
    assert floor.validate_delta_x_range([0.0, 0.0]) is None
    delta, stiffness = floor.sample_floor_stiffness([5.0, 5.0], 3.0, 9.81, 7, 230902)
    np.testing.assert_array_equal(delta, np.full(7, 5.0))
    np.testing.assert_allclose(stiffness, 2943.0, rtol=0, atol=1e-10)


def test_uniform_indentation_not_uniform_stiffness():
    delta, stiffness = floor.sample_floor_stiffness([0.2, 5.0], 3.0, 9.81, 100000, 230902)
    assert delta.min() >= 0.2 and delta.max() <= 5.0
    np.testing.assert_allclose(np.quantile(delta, [0.1, 0.5, 0.9]), [0.68, 2.6, 4.52], atol=0.025, rtol=0)
    np.testing.assert_allclose(stiffness * delta * 0.001, 14.715, atol=1e-12, rtol=0)
    # Uniform stiffness has median ~38259, whereas this distribution has median ~5659.6.
    assert 5500 < np.median(stiffness) < 5800


def test_seed_repeatability_and_no_global_rng_consumption():
    np.random.seed(37)
    before = np.random.get_state()
    first = floor.sample_floor_stiffness([0.2, 5.0], 3.0, 9.81, 121, 23)
    second = floor.sample_floor_stiffness([0.2, 5.0], 3.0, 9.81, 121, 23)
    different = floor.sample_floor_stiffness([0.2, 5.0], 3.0, 9.81, 121, 24)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("mass, gravity", [(0.0, 9.81), (-3.0, 9.81), (3.0, 0.0), (3.0, float("nan"))])
def test_invalid_reference_load(mass, gravity):
    with pytest.raises(ValueError, match="positive finite"):
        floor.sample_floor_stiffness([0.2, 5.0], mass, gravity, 1, 0)


def test_face_assignment_keeps_rows_columns_heights_and_border_separate():
    # Centered 2x3 generator with 4x6 m tiles. Each triangle lies around a known cell center.
    centers = [
        (-2.0, -6.0),
        (-2.0, 0.0),
        (-2.0, 6.0),
        (2.0, -6.0),
        (2.0, 0.0),
        (2.0, 6.0),
        (-5.0, 0.0),
        (5.0, 0.0),
        (0.0, -10.0),
        (0.0, 10.0),
    ]
    points = np.array(
        [
            [x + dx, y + dy, 10 * i]
            for i, (x, y) in enumerate(centers)
            for dx, dy in [(-0.1, -0.1), (0.1, -0.1), (0.0, 0.2)]
        ]
    )
    faces = np.arange(len(points)).reshape(-1, 3)
    np.testing.assert_array_equal(
        floor.terrain_face_tiles(points, faces, (4.0, 6.0), (2, 3)), [0, 1, 2, 3, 4, 5, -1, -1, -1, -1]
    )


def test_source_snapshot_retains_sampler_and_actual_floor_draws(tmp_path):
    tree = ast.parse(MODULE.with_name("solo12_env.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12Env")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_code_snapshot_files"
    )
    ns = {"Path": Path, "floor_stiffness": SimpleNamespace(__file__=str(MODULE))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODULE), "exec"), ns)
    env = SimpleNamespace(cfg=SimpleNamespace(log_dir=str(tmp_path)), _floor_stiffness_randomization={})
    assert ns["get_code_snapshot_files"](env) == [str(MODULE), str(tmp_path / "params/floor_stiffness.json")]
    env._floor_stiffness_randomization = None
    assert ns["get_code_snapshot_files"](env) == [str(MODULE)]
