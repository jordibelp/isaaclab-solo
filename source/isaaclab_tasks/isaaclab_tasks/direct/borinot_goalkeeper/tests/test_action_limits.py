# SPDX-License-Identifier: BSD-3-Clause
"""Raw policy excursions must not disappear at the environment's input clamp."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_control import config, method, raw_action_limit_costs


def test_raw_excursion_is_scored_before_the_existing_input_clamp():
    cfg = config()
    n = 2
    zeros = lambda *shape: torch.zeros(*shape)
    env = SimpleNamespace(
        cfg=cfg, extras={}, _episode_steps=zeros(n), _next_launch_steps=torch.ones(n),
        _ball_active=torch.ones(n, dtype=torch.bool), _shot_steps=zeros(n), _outcome=zeros(n),
        _previous_actuator_commands=zeros(n, 8), _actuator_commands=zeros(n, 8),
        actions=zeros(n, 8), _action_limit_costs=zeros(n, 3), _hover_per_rotor=5.,
        _previous_paddle=zeros(n, 3), _paddle_pose=lambda: (zeros(n, 3),),
        ball=SimpleNamespace(data=SimpleNamespace(root_pos_w=zeros(n, 3), root_lin_vel_w=zeros(n, 3))),
        scene=SimpleNamespace(env_origins=zeros(n, 3)), _old_progress_active=zeros(n),
        _hit=torch.zeros(n, dtype=torch.bool), _new_hit=zeros(n), _flags={},
        _power_sum_w=zeros(n), _power_samples=0, _physics_samples=0)
    raw = torch.zeros(n, 8)
    raw[:, 0] = torch.tensor([20., 40.])
    method('borinot_goalkeeper_env.py', '_pre_physics_step')(env, raw)
    assert env._action_limit_costs[1, 0] > env._action_limit_costs[0, 0] > 0
    torch.testing.assert_close(env.actions[0], env.actions[1])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_torque_threshold_both_signs_and_channel_sum(dtype, sign):
    cfg = config()
    raw = torch.zeros(4, 8, dtype=dtype)
    raw[:, 6] = sign * torch.tensor([.9, 1., 1.1, 2.], dtype=dtype)
    raw[3, 7] = -sign * 2
    result = raw_action_limit_costs(raw, cfg, 5.)
    expected = torch.tensor([0., 0., .27**2, 2*2.7**2], dtype=dtype)
    torch.testing.assert_close(result[:, 1], expected, atol=1e-7, rtol=1e-5)
    assert not result[:, 2].any()


@pytest.mark.parametrize('sac', [False, True])
@pytest.mark.parametrize('scale', [.5, 1., 2.])
def test_thrust_threshold_uses_effective_physical_cap_and_original_mapping(sac, scale):
    cfg = config()
    cfg.sac_action_mapping, cfg.thrust_to_weight, cfg.rotor_action_scale = sac, 2., scale
    raw = torch.zeros(5, 8, dtype=torch.float64)
    lo, hi = (-1., 1.) if sac else (-1/scale, 1/scale)
    raw[:, 0] = torch.tensor([lo, hi, lo-.2, hi+.2, hi+.4])
    before = raw.clone()
    result = raw_action_limit_costs(raw, cfg, 4.)[:, 0]
    span = 4. if sac else 4.*scale
    torch.testing.assert_close(result, torch.tensor([0., 0., (span*.2)**2, (span*.2)**2, (span*.4)**2],
                                                   dtype=raw.dtype), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_thrust_sac_asymmetric_positive_scaling():
    cfg = config()
    cfg.sac_action_mapping, cfg.thrust_to_weight = True, 3.
    raw = torch.zeros(2, 8, dtype=torch.float64)
    raw[:, 0] = torch.tensor([-1.5, 1.5])
    torch.testing.assert_close(raw_action_limit_costs(raw, cfg, 4.)[:, 0], torch.tensor([4., 16.], dtype=raw.dtype))


def test_disabled_thrust_cap_keeps_lower_bound_and_nominal_torque_soft_limit():
    cfg = config()
    cfg.remove_thrust_clipping = True
    raw = torch.zeros(2, 8)
    raw[:, 0] = torch.tensor([100., -2.])
    raw[:, 6] = 2.
    result = raw_action_limit_costs(raw, cfg, 5.)
    torch.testing.assert_close(result[:, 0], torch.tensor([0., 25.]))
    torch.testing.assert_close(result[:, 1], torch.full((2,), 2.7**2))


@pytest.mark.parametrize('mode', ['thrust_and_PD_joint_pos_control', 'PI_rotors_PD_joint_pos_control', 'indi'])
@pytest.mark.parametrize('sac', [False, True])
def test_reference_modes_do_not_mistake_reference_units_for_torques(mode, sac):
    cfg = config()
    cfg.control_mode, cfg.sac_action_mapping = mode, sac
    raw = torch.full((2, 9 if mode == 'indi' else 6 if mode.startswith('PI') else 8), 20.)
    result = raw_action_limit_costs(raw, cfg, 5.)
    assert not result[:, 1].any()
    if mode == 'indi':
        assert not result[:, :2].any()
        torch.testing.assert_close(result[:, 2], torch.full((2,), 9*19.**2 if sac else 0.))
    elif not sac:
        count = 6 if mode.startswith('PI') else 2
        torch.testing.assert_close(result[:, 2], torch.full((2,), count*10.**2))


@pytest.mark.parametrize('mode,sac,size', [('thrust_and_torque_control', False, 8),
    ('thrust_and_PD_joint_pos_control', False, 8), ('PI_rotors_PD_joint_pos_control', False, 6), ('indi', True, 9)])
def test_held_arm_ignores_unused_policy_joint_outputs(mode, sac, size):
    cfg = config()
    cfg.hold_arm, cfg.control_mode, cfg.sac_action_mapping = True, mode, sac
    raw = torch.zeros(2, size)
    raw[:, -2:] = 100.
    assert not raw_action_limit_costs(raw, cfg, 5.).any()


def test_zero_weights_skip_signals_even_for_extreme_outputs():
    cfg = config()
    for name in ('rwd_scale_thrust_limits', 'rwd_scale_torque_limits', 'rwd_scale_action_limits'):
        setattr(cfg, name, 0.)
    assert not raw_action_limit_costs(torch.full((2, 8), 1e30), cfg, 5.).any()


def test_tiny_excess_is_not_hidden_by_an_incorrect_absolute_tolerance():
    cfg = config()
    raw = torch.zeros(3, 8, dtype=torch.float64)
    raw[:, 6] = torch.tensor([1.-1e-8, 1., 1.+1e-8], dtype=raw.dtype)
    actual = raw_action_limit_costs(raw, cfg, 5.)[:, 1]
    assert not actual[:2].any()
    torch.testing.assert_close(actual[2], raw.new_tensor((2.7e-8)**2), atol=1e-23, rtol=1e-7)


@pytest.mark.parametrize('angle', [False, True])
@pytest.mark.parametrize('mapping', ['absolute', 'delta_hover'])
def test_legacy_mapping_and_angle_vectors(angle, mapping):
    cfg = SimpleNamespace(des_joint_out=angle, rwd_scale_thrust_limits=-1., rwd_scale_torque_limits=-1.,
        rwd_scale_action_limits=-1., thrust_mode=mapping, thrust_min=0., thrust_max=10., thrust_max_scale=.99,
        delta_thrust_scale=1., torque_min=(-2.7, -2.7), torque_max=(2.7, 2.7), torque_lim=2.7, bias_torque_j2=0.)
    raw = torch.zeros(2, 10 if angle else 8, dtype=torch.float64)
    raw[:, 0] = torch.tensor([-2., 2.])
    raw[:, 6:] = 2.
    result = raw_action_limit_costs(raw, cfg, 5., legacy=True)
    torch.testing.assert_close(result[:, 0], torch.tensor([25., 25.], dtype=raw.dtype))
    torch.testing.assert_close(result[:, 1], torch.full((2,), 0. if angle else 2*2.7**2, dtype=raw.dtype))
    torch.testing.assert_close(result[:, 2], torch.full((2,), 4. if angle else 0., dtype=raw.dtype))


@pytest.mark.parametrize('policy_hz', [60, 100, 200])
@pytest.mark.parametrize('scale', [-1., -.25, 0.])
def test_actual_reward_adds_unscaled_hinge_cost_once_per_policy_step(policy_hz, scale):
    from test_rewards import make_env, get_rewards
    env = make_env()
    env.step_dt = 1/policy_hz
    env._action_limit_costs[:] = torch.tensor([1., 4., 9.])
    env.cfg.rwd_scale_thrust_limits = scale
    env.cfg.rwd_scale_torque_limits = 2*scale
    env.cfg.rwd_scale_action_limits = 3*scale
    for _ in range(2):
        torch.testing.assert_close(get_rewards(env), torch.full((10,), 36*scale), atol=1e-7, rtol=0)


def test_every_prephysics_scores_before_mutating_and_every_reset_clears_selected_costs():
    root = Path(__file__).resolve().parents[2]
    for task in ('borinot', 'borinot_goalkeeper', 'borinot_goalkeeper_simple', 'borinot_cmd_vel'):
        tree = ast.parse((root / task / f'{task}_env.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        pre = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_pre_physics_step')
        assert 'raw_action_limit_costs' in ast.unparse(pre.body[0])
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_reset_idx')
        assert '_action_limit_costs' in ast.unparse(reset)
