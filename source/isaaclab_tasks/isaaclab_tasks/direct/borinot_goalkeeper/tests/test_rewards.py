# SPDX-License-Identifier: BSD-3-Clause
"""Run the real reward method on controlled tensor inputs, without Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("goalkeeper_reward_math", folder / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)
tree = ast.parse((folder / "borinot_goalkeeper_env.py").read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_rewards")
namespace = {"torch": torch, "plane_progress": task_math.plane_progress}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(folder), "exec"), namespace)
get_rewards = namespace["_get_rewards"]


def config_scales():
    result = {}
    cfg_tree = ast.parse((folder / "borinot_goalkeeper_env_cfg.py").read_text())
    for n in ast.walk(cfg_tree):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id.startswith("rwd_scale_"):
                    result[target.id] = ast.literal_eval(n.value)
    return result


def make_env():
    n = 10
    zero = torch.zeros(n)
    xyz = torch.zeros(n, 3)
    target = xyz.clone()
    target[:, 1] = 1.0
    normal = xyz.clone()
    normal[:, 0] = 1.0
    cfg = SimpleNamespace(**{name: 0.0 for name in config_scales()}, decimation=4,
                          progress_use_prediction=False)
    return SimpleNamespace(
        cfg=cfg, device="cpu", step_dt=1/60, _sample_power=lambda: None,
        _power_sum_w=torch.tensor([4., 8., 12., 16.]).repeat(n, 1),
        _previous_paddle=xyz.clone(), _paddle_pose=lambda: (xyz.clone(), None, normal),
        ball=SimpleNamespace(data=SimpleNamespace(root_pos_w=target, root_lin_vel_w=-normal)),
        scene=SimpleNamespace(env_origins=xyz),
        robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=torch.tensor([1., 2., 2.]).repeat(n, 1),
                                                   root_lin_vel_w=torch.tensor([2., 2., 1.]).repeat(n, 1))),
        _interception=lambda: (target, zero, zero.bool(), torch.full((n,), .4)),
        _recovery_geometry=lambda: (torch.ones(n, dtype=torch.bool), torch.full((n,), .6)),
        _old_progress_active=torch.ones(n, dtype=torch.bool),
        _clear=torch.ones(n, dtype=torch.bool), _hit=torch.ones(n, dtype=torch.bool),
        actions=torch.full((n, 6), .1), _previous_actions=torch.zeros(n, 6),
        _rotor_thrust=torch.full((n, 6), 2.), _hover_per_rotor=1.,
        _outcome=torch.arange(n), _center_quality=torch.full((n,), .5),
        _new_hit=torch.zeros(n, dtype=torch.bool), _flags={name: torch.zeros(n, dtype=torch.bool) for name in ("floor", "crash", "tilt")},
        _episode_rewards=zero.clone(), _episode_terms={
            name: zero.clone() for name in ("plane_progress", "outcome", "floor_after_save", "optional_rates", "optional_hit", "energy")
        }, extras={},
    )


def test_all_reward_fields_use_signed_scale_names():
    scales = config_scales()
    assert len(scales) == 20
    for name in scales:
        assert not any(tag in name for tag in ("penalty", "bonus", "cost"))
    cfg_tree = ast.parse((folder / "borinot_goalkeeper_env_cfg.py").read_text())
    for n in ast.walk(cfg_tree):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name):
                    assert not target.id.endswith(("_penalty", "_bonus", "_cost", "_reward_scale", "_reward_rate"))
    for name in ("goal", "ball_base_collision", "ball_arm_collision", "ball_floor", "robot_crash", "leave_goal_area", "miss", "deadline"):
        assert scales[f"rwd_scale_{name}"] < 0
    assert scales["rwd_scale_save"] > 0


def test_original_event_rewards_are_preserved():
    env = make_env()
    for name, value in config_scales().items():
        setattr(env.cfg, name, value)
    env.cfg.rwd_scale_energy = 0.0  # independent of locally tuned energy settings
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.tensor([0., -1., -.25, -1., -.25, -1., -1., 1., -1., -1.]))


def test_goal_base_and_crash_scales_are_independent():
    env = make_env()
    env.cfg.rwd_scale_goal = -3.0
    env.cfg.rwd_scale_ball_base_collision = -2.0
    env.cfg.rwd_scale_robot_crash = -4.0
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.tensor([0., -2., 0., -3., 0., 0., 0., 0., -4., 0.]))


@pytest.mark.parametrize("scale", [-.2, .2])
@pytest.mark.parametrize("field,raw", [
    ("rwd_scale_intercept", .4), ("rwd_scale_alignment", .4),
    ("rwd_scale_recovery", .6), ("rwd_scale_action_rate", .06),
    ("rwd_scale_body_angular_velocity", 9.), ("rwd_scale_body_linear_velocity", 9.),
    ("rwd_scale_thrust", 1.),
])
def test_optional_rates_add_signed_scale_times_signal(field, raw, scale):
    env = make_env()
    setattr(env.cfg, field, scale)
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, scale * raw * env.step_dt))


@pytest.mark.parametrize("scale", [-.2, .2])
def test_energy_scale_uses_watts_without_extra_dt(scale):
    env = make_env()
    env.cfg.rwd_scale_energy = scale
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, scale * 10.0))


@pytest.mark.parametrize("scale", [-.2, .2])
@pytest.mark.parametrize("field,raw", [("rwd_scale_hit", 1.), ("rwd_scale_centered_hit", .5)])
def test_hit_scales_are_signed_and_failures_receive_no_hit_reward(field, raw, scale):
    env = make_env()
    setattr(env.cfg, field, scale)
    env._new_hit.fill_(True)
    reward = get_rewards(env)
    expected = torch.zeros_like(reward)
    expected[[0, 7]] = scale * raw
    torch.testing.assert_close(reward, expected)


@pytest.mark.parametrize("scale", [-.2, .2])
def test_progress_scale_is_signed_without_dt(scale):
    env = make_env()
    env.cfg.rwd_scale_progress = scale
    after = torch.zeros(10, 3)
    after[:, 1] = .25
    env._paddle_pose = lambda: (after, None, after)
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, scale * .25))


def test_save_on_floor_still_adds_negative_floor_scale_once():
    env = make_env()
    env._outcome.fill_(7)
    env._flags["floor"].fill_(True)
    env.cfg.rwd_scale_save = 1.0
    env.cfg.rwd_scale_ball_floor = -.25
    reward = get_rewards(env)
    torch.testing.assert_close(reward, torch.full_like(reward, .75))
