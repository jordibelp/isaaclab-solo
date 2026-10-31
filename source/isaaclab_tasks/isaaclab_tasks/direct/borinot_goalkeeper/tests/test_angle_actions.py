# SPDX-License-Identifier: BSD-3-Clause
"""Independent angle/layout oracles and actual PD/PID/controller integration."""
import copy
import itertools
import math

import pytest
import torch

from test_control import action_env, angles, config, method, raw_action_limit_costs, actuator_delay
from test_indi import fixture as indi_fixture, home_quat
from test_wrench import fixture as wrench_fixture
from test_cbrt import fixture as cbrt_fixture
from test_sac import algorithm_class, sac_env

MODES = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control",
         "PI_rotors_PD_joint_pos_control", "indi", "wrench", "cbrt", "cbrt_indi")
BITS = tuple(itertools.product((False, True), repeat=3))


def set_flags(cfg, bits):
    cfg.predict_yaw_as_sincos, cfg.predict_thigh_joint_pos_as_sincos, cfg.predict_calf_joint_pos_as_sincos = bits


def pair_slots(mode, bits):
    yaw, thigh, calf = bits
    if mode == MODES[0]:
        return []
    yaw_slot, joint = {MODES[1]: (None, 6), MODES[2]: (None, 4),
                      "indi": (6, 7), "wrench": (3, 7), "cbrt": (None, 4), "cbrt_indi": (None, 4)}[mode]
    return ([yaw_slot] if yaw and yaw_slot is not None else []) + ([joint] if thigh else []) + ([joint+1] if calf else [])


def encode(canonical, slots, radius=3.7):
    # Independent encoder: replace selected columns in reverse order.
    result = canonical.clone()
    for index in reversed(slots):
        theta = canonical[:, index:index+1]
        result = torch.cat((result[:, :index], radius*theta.cos(), radius*theta.sin(), result[:, index+1:]), -1)
    return result


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bits", BITS)
def test_every_flag_combination_resolves_spaces_and_preserves_other_channels(mode, bits):
    cfg = config()
    cfg.control_mode = mode
    # Controller validation is exercised by the controller integration tests below.
    for name in ("indi", "wrench", "cbrt"):
        setattr(cfg, name, type("Cfg", (), {"validate": lambda self, task: None})())
    set_flags(cfg, bits)
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    baseline = config()
    baseline.control_mode = mode
    for name in ("indi", "wrench", "cbrt"):
        setattr(baseline, name, getattr(cfg, name))
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(baseline)
    slots = pair_slots(mode, bits)
    assert cfg.action_space == baseline.action_space + len(slots)
    assert cfg.observation_space == baseline.observation_space + len(slots)
    assert cfg.state_space == baseline.state_space + len(slots)
    canonical = torch.linspace(-2., 2., baseline.action_space, dtype=torch.float64)[None].repeat(3, 1)
    raw = encode(canonical, slots)
    torch.testing.assert_close(angles.decode_angle_actions(raw, cfg), canonical, atol=1e-15, rtol=0)
    clipped = angles.clamp_scalar_actions(raw, cfg, 1.)
    decoded = angles.decode_angle_actions(clipped, cfg)
    expected = canonical.clamp(-1, 1)
    expected[:, slots] = canonical[:, slots]
    torch.testing.assert_close(decoded, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quadrants_branch_cut_zero_and_arbitrary_magnitude(dtype):
    theta = torch.tensor([-math.pi+1e-4, -2.3, -math.pi/2, -.2, 0, .4, math.pi/2, 2.7, math.pi-1e-4], dtype=dtype)
    for radius in (1e-30, .1, 1., 1e30):
        actual = angles.angle_from_pair(radius*torch.stack((theta.cos(), theta.sin()), -1))
        torch.testing.assert_close(actual, theta, atol=3e-7 if dtype==torch.float32 else 1e-15, rtol=0)
    zero = torch.tensor([[0., 0.], [-0., -0.]], dtype=dtype)
    assert torch.equal(angles.angle_from_pair(zero), torch.zeros(2, dtype=dtype))


@pytest.mark.parametrize("mode", [MODES[1], MODES[2]])
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("sac", [False, True])
def test_actual_implicit_pd_targets_mix_scalar_and_pair_units(mode, bits, sac):
    if sac and mode == MODES[2]:
        pytest.skip("PI+SAC is not a supported task interface")
    env, calls = sac_env() if sac else action_env(mode)
    env.cfg.control_mode = mode
    set_flags(env.cfg, bits)
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(env.cfg)
    size = 8 if mode == MODES[1] else 6
    canonical = torch.zeros(3, size, dtype=torch.float64)
    canonical[:, -2:] = torch.tensor([[.3, -.7], [1.4, -2.8], [-1.4, 2.8]])
    env.actions = encode(canonical, pair_slots(mode, bits))
    env._actuator_delay = actuator_delay.ActuatorDelay(env.cfg, 3, env.cfg.action_space, "cpu", torch.float64)
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    expected = canonical[:, -2:] * env._joint_scale
    if sac:
        expected += env._sac_joint_offset
    for index, enabled in enumerate(bits[1:]):
        if enabled:
            expected[:, index] = canonical[:, -2+index]
    torch.testing.assert_close(calls["position"], expected)
    assert not calls["effort"].any() and not calls["velocity"].any()


@pytest.mark.parametrize("mode", ["indi", "wrench", "cbrt", "cbrt_indi"])
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("sac", [False, True])
def test_actual_controllers_match_equivalent_scalar_reference(mode, bits, sac):
    ctrl = indi_fixture() if mode=="indi" else (wrench_fixture() if mode=="wrench" else cbrt_fixture(mode))
    ctrl.task.sac_action_mapping = sac
    baseline = copy.deepcopy(ctrl)
    set_flags(ctrl.task, bits)
    count = 9 if mode in ("indi", "wrench") else 6
    physical = torch.zeros(4, count, dtype=torch.float64)
    physical[:, -2:] = torch.tensor([.42, -.65])
    if mode in ("indi", "wrench"):
        physical[:, 6 if mode=="indi" else 3] = .9
    elif not sac:
        physical[:, 3] = ctrl.hover_total
    raw = encode(physical, pair_slots(mode, bits), radius=.3 if sac else 5.)
    scalar = physical.clone()
    if sac:
        if bits[0] and mode in ("indi", "wrench"):
            scalar[:, 6 if mode=="indi" else 3] /= math.radians(ctrl.cfg.heading_range_degs)
        for index, enabled in enumerate(bits[1:]):
            if enabled:
                scalar[:, -2+index] /= ctrl.joint_scale[index]
    rates = torch.zeros(4, 3, dtype=torch.float64)
    position, velocity = torch.full((4, 2), .1, dtype=torch.float64), torch.full((4, 2), -.2, dtype=torch.float64)
    if mode=="indi":
        args = (home_quat(ctrl), rates, position, velocity)
    elif mode=="wrench":
        args = (physical.new_tensor(ctrl.task.robot_home_quat).expand(4, -1), rates, position, velocity, rates, rates)
    else:
        args = (rates, position, velocity)
    for _ in range(3):
        actual, expected = ctrl.step(raw, *args), baseline.step(scalar, *args)
        for result, oracle in zip(actual, expected):
            torch.testing.assert_close(result, oracle, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(ctrl.requested_joint_effort, baseline.requested_joint_effort, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mode", [MODES[1], "indi", "wrench", "cbrt", "cbrt_indi"])
@pytest.mark.parametrize("bits", BITS)
def test_sac_bounds_expand_only_pairs_and_preserve_scalar_joint_bounds(mode, bits):
    env, _ = sac_env()
    env.cfg.control_mode = mode
    set_flags(env.cfg, bits)
    env.cfg.action_space = angles.angle_action_size(env.cfg)
    upper, lower = algorithm_class()._compute_action_scaling(env, "cpu")
    assert upper.shape == lower.shape == (env.cfg.action_space,)
    if mode != MODES[1]:
        assert (upper==1).all() and (lower==1).all()
    else:
        start = 6
        for i, enabled in enumerate(bits[1:]):
            if enabled:
                assert (upper[start:start+2]==1).all() and (lower[start:start+2]==1).all()
            else:
                torch.testing.assert_close(upper[start], (env._sac_joint_soft_limits[i, 1]-env._sac_joint_offset[i])/env._joint_scale[i])
                torch.testing.assert_close(lower[start], (env._sac_joint_offset[i]-env._sac_joint_soft_limits[i, 0])/env._joint_scale[i])
            start += 2 if enabled else 1


@pytest.mark.parametrize("hold", [False, True])
@pytest.mark.parametrize("mode", MODES)
def test_action_limit_costs_ignore_pair_radius_and_all_held_joint_channels(mode, hold):
    cfg = config()
    cfg.control_mode, cfg.hold_arm, cfg.sac_action_mapping = mode, hold, mode in ("indi", "wrench", "cbrt", "cbrt_indi")
    cfg.remove_thrust_clipping = False
    cfg.action_clamp = 1.0
    cfg.rwd_scale_action_limits = -1.
    set_flags(cfg, (True, True, True))
    base = 9 if mode in ("indi", "wrench") else (8 if mode.startswith("thrust_and_") else 6)
    canonical = torch.zeros(2, base)
    raw = encode(canonical, pair_slots(mode, (True, True, True)), radius=100.)
    assert not raw_action_limit_costs(raw, cfg, 5.)[:, 2].any()
    # A genuine clipped scalar reference still costs (3-1)^2, even with expanded pairs.
    if mode in ("indi", "wrench", "cbrt", "cbrt_indi", MODES[2]):
        raw[:, 0] = 3.
        torch.testing.assert_close(raw_action_limit_costs(raw, cfg, 5.)[:, 2], torch.full((2,), 4.))
