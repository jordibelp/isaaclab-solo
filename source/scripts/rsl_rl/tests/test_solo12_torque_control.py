"""CPU checks of the production control methods, without starting Isaac Sim."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOLO = Path(__file__).resolve().parents[3] / 'isaaclab_tasks/isaaclab_tasks/direct/solo12'


def method(filename, cls, name):
    tree = ast.parse((SOLO / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {'math': math, 'torch': torch, 'EFFORT_LIMIT_SIM': 2.65}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, 'exec'), ns)
    return ns[name]


def config(**kwargs):
    values = dict(physics_hz=200, policy_hz=50, control_mode='position', action_scale=0.25,
                  effort_limit_sim=2.65, bringup_clip_rad=0., action_rate_reference_hz=50.,
                  sim=NS(dt=.005, render_interval=4), contact_sensor=NS(update_period=.005),
                  base_external_contact_sensor=NS(update_period=.005))
    values.update(kwargs)
    return NS(**values)


@pytest.mark.parametrize('policy,physics', [(50,200), (100,200), (200,400), (500,1000), (1000,2000)])
def test_rates_refresh_after_overrides(policy, physics):
    cfg = config(policy_hz=policy, physics_hz=physics)
    refresh = method('solo12_env_cfg.py', 'Solo12EnvCfg', '_refresh_control_config')
    refresh(cfg)
    assert cfg.decimation == physics // policy
    assert cfg.sim.dt == 1 / physics
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / policy)
    assert cfg.contact_sensor.update_period == cfg.sim.dt
    assert cfg.base_external_contact_sensor.update_period == cfg.sim.dt
    assert cfg.sim.render_interval == cfg.decimation
    refresh(cfg)  # idempotent
    assert cfg.decimation == physics // policy


@pytest.mark.parametrize('kwargs', [dict(physics_hz=0), dict(policy_hz=-1), dict(physics_hz=float('nan')),
    dict(policy_hz=float('inf')), dict(physics_hz=True), dict(policy_hz=400), dict(physics_hz=333),
    dict(control_mode='bad'), dict(control_mode='torque', effort_limit_sim=0),
    dict(control_mode='torque', effort_limit_sim=float('inf')),
    dict(control_mode='torque', bringup_clip_rad=.1), dict(action_rate_reference_hz=0)])
def test_invalid_control_settings(kwargs):
    with pytest.raises(ValueError):
        method('solo12_env_cfg.py', 'Solo12EnvCfg', '_refresh_control_config')(config(**kwargs))


@pytest.mark.parametrize('mode', ['position', 'torque'])
@pytest.mark.parametrize('scale', [0.25, 1., 2.65])
@pytest.mark.parametrize('cap', [1.25, 2.65])
def test_action_mapping_clipping_and_feedback(mode, scale, cap):
    requested = torch.tensor([[-3., -cap, -.5, 0., .5, cap, 3.]])
    cfg = config(control_mode=mode, action_scale=scale, effort_limit_sim=cap)
    env = NS(cfg=cfg, _command_steps_left=torch.ones(1), _update_base_push_wrench=lambda: None,
             _q_offset_action_and_obs=torch.full_like(requested, .4),
             _action_delay_buffer=NS(compute=lambda x: x.clone()))
    output = {}
    env._robot = NS(set_joint_effort_target=lambda x, **kw: output.update(effort=x.clone()),
                    set_joint_position_target=lambda x, **kw: output.update(position=x.clone()))
    env._joint_ids = list(range(requested.shape[1]))
    method('solo12_env.py', 'Solo12Env', '_pre_physics_step')(env, requested)
    method('solo12_env.py', 'Solo12Env', '_apply_action')(env)
    if mode == 'torque':
        expected = requested.clamp(-cap, cap)
        torch.testing.assert_close(output['effort'], expected)
        torch.testing.assert_close(env._applied_actions, expected)
        assert 'position' not in output
    else:
        torch.testing.assert_close(output['position'], requested * scale + .4)
        torch.testing.assert_close(env._applied_actions, requested)
        assert 'effort' not in output
    torch.testing.assert_close(env._actions, requested)


@pytest.mark.parametrize('hz', [50,100,200,500,1000])
@pytest.mark.parametrize('scale', [0.25, 1., 2.65])
def test_torque_smoothness_matches_same_physical_ramp(hz, scale):
    env = NS(cfg=config(control_mode='torque', action_scale=scale), step_dt=1/hz,
             _actions=torch.full((3,12), .5 + 5/hz), _previous_actions=torch.full((3,12), .5))
    value = method('solo12_env.py', 'Solo12Env', '_action_rate_l2')(env)
    torch.testing.assert_close(value, torch.full((3,), 12 * (5 / (50 * 2.65))**2))


def test_saturated_commands_and_legacy_penalty():
    env = NS(cfg=config(control_mode='torque', action_scale=2.65), step_dt=.001,
             _actions=torch.full((2,12), 4.), _previous_actions=torch.full((2,12), 3.))
    rate = method('solo12_env.py', 'Solo12Env', '_action_rate_l2')
    assert torch.equal(rate(env), torch.zeros(2))
    env.cfg.control_mode = 'position'
    assert torch.equal(rate(env), torch.full((2,), 12.))


@pytest.mark.parametrize('scale', [0.25, 1., 2.65])
@pytest.mark.parametrize('cap', [1.25, 2.65])
def test_sac_torque_bounds_ignore_joint_position_limits(scale, cap):
    cfg = config(control_mode='torque', action_scale=scale, effort_limit_sim=cap)
    env = NS(cfg=cfg, _joint_ids=list(range(12)))
    upper, lower = method('agents/rsl_rl_sac_cfg.py', 'Solo12SAC', '_compute_action_scaling')(env, 'cpu')
    torch.testing.assert_close(upper, torch.full((12,), cap))
    torch.testing.assert_close(lower, upper)


def test_telemetry_never_labels_torque_as_position_target():
    player = Path(__file__).resolve().parents[1] / 'play_direct_0325.py'
    node = next(n for n in ast.parse(player.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_position_target_for_telemetry')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(player), 'exec'), ns)
    env = NS(cfg=config(control_mode='torque'), _robot=NS(data=NS(joint_pos=torch.zeros(2,12))),
             _joint_ids=list(range(12)), _delayed_processed_actions=torch.ones(2,12))
    read = ns['_position_target_for_telemetry']
    assert torch.isnan(read(env)).all()
    env.cfg.control_mode = 'position'
    torch.testing.assert_close(read(env), env._delayed_processed_actions)


def test_delayed_torque_is_held_and_not_replaced_by_new_request():
    old = torch.full((2,12), -.75)
    env = NS(cfg=config(control_mode='torque', action_scale=2.), _joint_ids=list(range(12)),
             _processed_actions=torch.full((2,12), .5), _action_delay_buffer=NS(compute=lambda x: old))
    sent = []
    env._robot = NS(set_joint_effort_target=lambda x, **kw: sent.append(x.clone()))
    apply = method('solo12_env.py', 'Solo12Env', '_apply_action')
    apply(env)
    apply(env)
    assert len(sent) == 2
    for value in sent:
        torch.testing.assert_close(value, old)
    torch.testing.assert_close(env._applied_actions, old)


@pytest.mark.parametrize('cap', [1.25, 2.65])
def test_torque_smoothness_preserves_previous_physical_cost(cap):
    previous_nm = torch.tensor([[-4., -1., 0., .5, 4.]])
    current_nm = torch.tensor([[-3., .5, 0., 1., 3.]])
    env = NS(cfg=config(control_mode='torque', action_scale=1., effort_limit_sim=cap),
             step_dt=.01, _actions=current_nm, _previous_actions=previous_nm)
    # Independent reconstruction of the old normalized-action formula at scale=2.65.
    previous_action = (previous_nm / 2.65).clamp(-cap / 2.65, cap / 2.65)
    current_action = (current_nm / 2.65).clamp(-cap / 2.65, cap / 2.65)
    expected = ((current_action - previous_action) / (50 * env.step_dt)).square().sum(1)
    actual = method('solo12_env.py', 'Solo12Env', '_action_rate_l2')(env)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('mode', ['position', 'torque'])
def test_nested_gain_override_cannot_reintroduce_pd_in_torque_mode(mode):
    cfg = config(control_mode=mode)
    cfg.kp, cfg.kd = 9., .2
    cfg.enabled_self_collisions = True
    cfg.robot = NS(spawn=NS(articulation_props=NS()), actuators={'legs': NS()})
    cfg.terrain = NS(physics_material=NS())
    cfg.compliant_contact_stiffness = cfg.compliant_contact_damping = 0.
    path = SOLO / 'solo12_env_cfg.py'
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_optional_compliant_contact_value')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Solo12EnvCfg')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_apply_runtime_overrides_to_nested_cfg')
    ns = {}
    exec(compile(ast.Module(body=[helper,fn],type_ignores=[]),str(path),'exec'),ns)
    ns[fn.name](cfg)
    legs = cfg.robot.actuators['legs']
    assert legs.stiffness == (0. if mode == 'torque' else 9.)
    assert legs.damping == (0. if mode == 'torque' else .2)
    assert legs.effort_limit_sim == cfg.effort_limit_sim


def test_legacy_imu_contract_restores_authoritative_clocks():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'solo12_imu_pipeline.py'
    spec = importlib.util.spec_from_file_location('torque_imu_contract_test', path)
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    cfg = config()
    cfg.refresh_runtime_dependent_config = lambda: method(
        'solo12_env_cfg.py', 'Solo12EnvCfg', '_refresh_control_config')(cfg)
    pipeline.apply_observation_contract(cfg, {'physics_dt': .001, 'decimation': 2})
    assert cfg.physics_hz == 1000
    assert cfg.policy_hz == 500
    assert cfg.sim.dt == .001
    assert cfg.decimation == 2
