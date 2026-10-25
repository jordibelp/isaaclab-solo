# SPDX-License-Identifier: BSD-3-Clause
"""PPO commands joint N*m directly; SAC retains its normalized torque contract."""
import pytest
import torch

from test_control import action_env, config, method, raw_action_limit_costs


@pytest.mark.parametrize('sac', [False, True])
@pytest.mark.parametrize('limit', [1.8, 2.7, 5.])
def test_direct_joint_effort_units_and_independent_channels(sac, limit):
    env, calls = action_env('thrust_and_torque_control')
    env.cfg.sac_action_mapping = sac
    env.cfg.effort_lim_sim = limit
    values = [[-1., 1.], [.5, -.25], [0., 0.]] if sac else [[-12.5, 15.], [.5, -.25], [0., 0.]]
    env.actions[:, 6:] = env.actions.new_tensor(values)
    before = env.actions.clone()
    method('borinot_goalkeeper_env.py', '_apply_action')(env)
    expected = env.actions.new_tensor(values) * (limit if sac else 1.)
    torch.testing.assert_close(calls['effort'], expected, rtol=0, atol=0)
    torch.testing.assert_close(env.actions, before, rtol=0, atol=0)
    assert 'position' not in calls and 'velocity' not in calls


@pytest.mark.parametrize('sac', [False, True])
@pytest.mark.parametrize('limit', [1.8, 2.7, 5.])
@pytest.mark.parametrize('action_clamp', [.5, 10.])
def test_excess_cost_uses_each_algorithms_torque_units(sac, limit, action_clamp):
    cfg = config()
    cfg.sac_action_mapping, cfg.effort_lim_sim, cfg.action_clamp = sac, limit, action_clamp
    raw = torch.zeros(5, 8, dtype=torch.float64)
    requested = raw.new_tensor([[-limit, limit], [0., 0.], [-.9*limit, .9*limit],
                                [-1.1*limit, 1.1*limit], [-2*limit, 3*limit]])
    raw[:, 6:] = requested / limit if sac else requested
    costs = raw_action_limit_costs(raw, cfg, 5.)
    torch.testing.assert_close(costs[:, 1], raw.new_tensor([0., 0., 0., .02, 5.]), atol=1e-14, rtol=0)
    assert not costs[:, 2].any()
