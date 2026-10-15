# SPDX-License-Identifier: BSD-3-Clause
"""CPU checks for staged reset sampling; no Isaac Sim import."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keeper_spawn_math", FOLDER / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


@pytest.fixture
def cfg():
    return SimpleNamespace(robot_home=(0.0, 0.0, 1.5), reset_grid_enabled=True,
        reset_grid_start_level=2, reset_grid_y_offsets_m=(-.8, 0., .8),
        reset_grid_z_offsets_m=(-.65, 0., .65), reset_xy_noise_m=.015, reset_z_noise_m=.015,
        goal_area_x_m=(-.08, .9), goal_area_y_m=(-1.2, 1.2), goal_area_z_m=(.55, 2.6))


@pytest.mark.parametrize("level", [0, 1])
def test_early_phases_keep_home_and_existing_noise(cfg, level):
    task_math.validate_reset_grid(cfg)
    torch.manual_seed(19)
    position, ids = task_math.sample_robot_positions(10000, "cpu", cfg, level)
    torch.manual_seed(19)
    expected = torch.tensor(cfg.robot_home).expand(10000, -1).clone()
    expected[:, :2] += (2*torch.rand(10000, 2)-1)*cfg.reset_xy_noise_m
    expected[:, 2] += (2*torch.rand(10000)-1)*cfg.reset_z_noise_m
    assert torch.equal(position, expected)
    assert (ids == -1).all()


@pytest.mark.parametrize("level", [2, 3, 7])
def test_later_phases_choose_all_nine_uniformly_with_bounded_noise(cfg, level):
    torch.manual_seed(24)
    position, ids = task_math.sample_robot_positions(90000, "cpu", cfg, level)
    # Expected locations are independent explicit points, ordered bottom to top.
    expected = torch.tensor([[0., y, z] for z in (.85, 1.5, 2.15) for y in (-.8, 0., .8)])
    delta = (position-expected[ids]).abs()
    assert (delta <= .015001).all()
    counts = torch.bincount(ids, minlength=9)
    assert ((counts > 9400) & (counts < 10600)).all(), counts.tolist()
    for axis, bounds in enumerate((cfg.goal_area_x_m, cfg.goal_area_y_m, cfg.goal_area_z_m)):
        assert ((position[:, axis] >= bounds[0]) & (position[:, axis] <= bounds[1])).all()


def test_disable_restores_centre_even_at_late_level(cfg):
    cfg.reset_grid_enabled = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
    position, ids = task_math.sample_robot_positions(256, "cpu", cfg, 99)
    assert torch.equal(position, torch.tensor(cfg.robot_home).expand(256, -1))
    assert (ids == -1).all()


def test_custom_grid_sizes_home_and_activation(cfg):
    cfg.robot_home = (.3, .2, 1.6)
    cfg.reset_grid_y_offsets_m = (-.3, .1)
    cfg.reset_grid_z_offsets_m = (-.2, .2, .4, .6)
    cfg.reset_grid_start_level = 0
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
    task_math.validate_reset_grid(cfg)
    position, ids = task_math.sample_robot_positions(2000, "cpu", cfg, 0)
    expected = torch.tensor([[.3, y, z] for z in (1.4, 1.8, 2., 2.2) for y in (-.1, .3)])
    torch.testing.assert_close(position, expected[ids])
    assert ids.unique().numel() == 8


@pytest.mark.parametrize("field,value", [
    ("reset_grid_start_level", -1), ("reset_grid_start_level", 1.5),
    ("reset_grid_y_offsets_m", ()), ("reset_grid_z_offsets_m", (float("nan"),)),
    ("reset_grid_y_offsets_m", (float("inf"),)), ("reset_xy_noise_m", -.01),
    ("reset_z_noise_m", float("nan")), ("robot_home", (0., 0., .5)),
    ("reset_grid_y_offsets_m", (-1.2, 0., 1.2)), ("reset_grid_z_offsets_m", (-1., 0., 1.)),
    ("robot_home", (-.07, 0., 1.5)),
])
def test_bad_settings_fail_before_simulation(cfg, field, value):
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        task_math.validate_reset_grid(cfg)


def test_point_exactly_at_noise_expanded_boundary_is_allowed(cfg):
    cfg.robot_home = (.0, 0., 1.5)
    cfg.goal_area_y_m = (-.815, .815)
    task_math.validate_reset_grid(cfg)


def test_source_reset_uses_selected_stage_before_recording_episode():
    tree = ast.parse((FOLDER / "borinot_goalkeeper_env.py").read_text())
    reset = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx")
    calls = {n.func.id:n for n in ast.walk(reset) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "sample_robot_positions" in calls
    assert ast.unparse(calls["sample_robot_positions"].args[-1]) == "self.shot_level"
    text = ast.unparse(reset)
    assert text.index("self._update_curriculum") < text.index("sample_robot_positions") < text.index("self._episode_stage[env_ids]")


def test_zero_environments(cfg):
    position, ids = task_math.sample_robot_positions(0, "cpu", cfg, 2)
    assert position.shape == (0, 3) and ids.shape == (0,)
