# SPDX-License-Identifier: BSD-3-Clause
"""Exercise actual shot scheduling, outcome and reward methods without Isaac Sim."""
import ast
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

FOLDER=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('continuous_math',FOLDER/'task_math.py')
math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
spec=importlib.util.spec_from_file_location('control_helpers',FOLDER/'tests'/'test_control.py')
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
tree=ast.parse((FOLDER/'borinot_goalkeeper_env.py').read_text())
names=('_pre_physics_step','_interception','_get_dones','_get_rewards','_get_observations','_record_shots','_park_ball','_launch_ball')
scope={name:getattr(math,name) for name in ('terminal_outcome','sample_shots','sample_shot_delay_steps','promotion_ready','plane_progress','predict_intercept')}
scope.update(torch=torch,quat_apply_inverse=lambda q,v:v,matrix_from_quat=lambda q:torch.eye(3).expand(len(q),-1,-1))
scope['raw_action_limit_costs'] = control.raw_action_limit_costs
for node in ast.walk(tree):
    if isinstance(node,ast.FunctionDef) and node.name in names:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(FOLDER),'exec'),scope)


@pytest.fixture
def env():
    n=9;zero=lambda:torch.zeros(n);xyz=torch.zeros(n,3)
    cfg_tree=ast.parse((FOLDER/'borinot_goalkeeper_env_cfg.py').read_text())
    values={}
    for node in cfg_tree.body:
        if isinstance(node,ast.ClassDef):
            for item in node.body:
                if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name):
                    try:values[item.targets[0].id]=ast.literal_eval(item.value)
                    except (ValueError,TypeError):pass
    cfg=SimpleNamespace(**values);cfg.curriculum=False;cfg.rwd_scale_energy=0.
    root=torch.zeros(n,13);root[:,3]=1.;root[:,2]=1.5
    ball_data=SimpleNamespace(default_root_state=root.clone(),root_pos_w=root[:,:3].clone(),root_lin_vel_w=xyz.clone())
    def write_ball(state,ids):
        ball_data.root_pos_w[ids]=state[:,:3];ball_data.root_lin_vel_w[ids]=state[:,7:10]
    noop=lambda *a,**kw:None
    e=SimpleNamespace(cfg=cfg,device='cpu',num_envs=n,step_dt=.01,max_episode_length=2000,
        actions=torch.zeros(n,8),_actuator_commands=torch.zeros(n,8),_previous_actuator_commands=torch.zeros(n,8),_rate_integral=torch.ones(n,3),
        _action_limit_costs=torch.zeros(n,3),_desired_actuation=None,
        _previous_paddle=xyz.clone(),_old_progress_active=torch.zeros(n,dtype=torch.bool),
        _hit=torch.zeros(n,dtype=torch.bool),_clear=torch.zeros(n,dtype=torch.bool),_new_hit=torch.zeros(n,dtype=torch.bool),
        _center_quality=zero(),_recovery_time=zero(),_outcome=torch.zeros(n,dtype=torch.long),
        _shot_steps=torch.zeros(n,dtype=torch.long),_episode_steps=torch.zeros(n,dtype=torch.long),
        _next_launch_steps=torch.full((n,),200,dtype=torch.long),_ball_active=torch.ones(n,dtype=torch.bool),
        _shot_stage=torch.zeros(n,dtype=torch.long),_episode_shots=torch.zeros(n,dtype=torch.long),
        _episode_saves=torch.zeros(n,dtype=torch.long),_episode_paddle_hits=torch.zeros(n,dtype=torch.long),
        _flags={k:torch.zeros(n,dtype=torch.bool) for k in ('base','arm','goal','floor','miss','crash','tilt','area')},
        _power_sum_w=torch.zeros(n,4),_power_samples=0,_physics_samples=0,
        _rotor_thrust=torch.full((n,6),5.),_hover_per_rotor=5.,_home=torch.tensor([0.,0.,1.5]),
        _joint_scale=torch.ones(2),joint_ids=[0,1],_ball_parameters=torch.zeros(n,2),
        _previous_ball=xyz.clone(),_ball_masses=torch.ones(n,1),_ball_inertias=torch.ones(n,9),_ball_materials=torch.ones(n,1,3),
        _episode_rewards=zero(),_episode_terms={k:zero() for k in ('plane_progress','outcome','floor_after_save','optional_rates','optional_hit','energy',
                                                               'thrust_limits','torque_limits','action_limits')},
        outcome_counts=torch.zeros(10,dtype=torch.long),shot_level=0,_stage_shots=0,_stage_start_step=0,common_step_counter=0,
        _curriculum_window=torch.zeros(8),_window_cursor=0,curriculum_stage_pure_episodes=0,
        extras={},scene=SimpleNamespace(env_origins=xyz),ball_contact=SimpleNamespace(reset=noop),
        ball=SimpleNamespace(data=ball_data,write_root_state_to_sim=write_ball,root_physx_view=SimpleNamespace(set_masses=noop,set_inertias=noop,set_material_properties=noop)),
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=root[:,:3].clone(),root_quat_w=root[:,3:7].clone(),
            root_lin_vel_w=xyz.clone(),root_lin_vel_b=xyz.clone(),root_ang_vel_b=xyz.clone(),joint_pos=torch.zeros(n,2),joint_vel=torch.zeros(n,2))),
        _sample_physics_events=noop,_sample_power=noop,
        _paddle_pose=lambda:(xyz.clone(),None,torch.tensor([1.,0.,0.]).expand(n,-1)),
        _recovery_geometry=lambda:(torch.ones(n,dtype=torch.bool),torch.ones(n)))
    for name in names:setattr(e,name,MethodType(scope[name],e))
    return e


@pytest.mark.parametrize('reset_on_goal', [False, True])
def test_only_goals_can_reset_ball_results_and_safety_is_terminal(env, reset_on_goal):
    env.cfg.reset_on_goal = reset_on_goal
    for k,i in [('base',0),('arm',1),('goal',2),('floor',3),('miss',4),('crash',6),('tilt',7),('area',8)]:env._flags[k][i]=True
    env._hit[5]=env._clear[5]=True
    term,trunc=env._get_dones()
    assert term.tolist()==[False,False,reset_on_goal,False,False,False]+[True]*3
    assert not trunc.any()
    assert env._outcome.tolist()==[1,2,3,4,9,7,8,8,5]


def test_save_reward_once_and_no_robot_or_controller_reset(env):
    env._hit[:]=env._clear[:]=True
    env._shot_steps[:]=42
    original=env._rate_integral.clone();root=env.robot.data.root_pos_w.clone()
    assert not env._get_dones()[0].any()
    assert torch.equal(env._get_rewards(),torch.ones(9))
    assert not env._ball_active.any() and (env.ball.data.root_pos_w[:,2]<0).all()
    assert torch.equal(env._rate_integral,original) and torch.equal(env.robot.data.root_pos_w,root)
    env._pre_physics_step(torch.zeros(9,8));env._get_dones()
    assert not env._get_rewards().any()
    assert env.outcome_counts[7]==9 and (env._episode_shots==1).all()


@pytest.mark.parametrize('mode,remaining',[('launch',73),('resolution',250)])
def test_scheduler_origin_and_no_overlap(env,mode,remaining):
    env.cfg.shot_resampling_mode=mode;env.cfg.shot_resampling_time_range_s=(2.5,2.5)
    env._next_launch_steps[:]=73;env._flags['goal'][:]=True
    env._get_dones();env._get_rewards()
    assert (env._next_launch_steps==remaining).all()
    env._next_launch_steps[:]=1
    env._pre_physics_step(torch.zeros(9,8))
    assert env._ball_active.all() and (env._shot_steps==1).all()
    assert not env._hit.any() and not env._clear.any()
    # An overdue timer cannot relaunch an unresolved ball.
    before=env.ball.data.root_pos_w.clone();env._next_launch_steps[:]=0
    env._pre_physics_step(torch.zeros(9,8))
    assert torch.equal(before,env.ball.data.root_pos_w)


def test_timeout_ends_only_shot_and_episode_clock_is_independent(env):
    env._shot_steps[:]=300
    term,trunc=env._get_dones()
    assert not term.any() and not trunc.any() and (env._outcome==6).all()
    env._get_rewards();env._episode_steps[:]=2000
    term,trunc=env._get_dones()
    assert not term.any() and trunc.all()
    env.cfg.is_finite_horizon=True
    term,trunc=env._get_dones()
    assert term.all() and not trunc.any()


def test_partial_shot_relaunch_preserves_other_ball_and_robot_state(env):
    before=env.ball.data.root_pos_w.clone();integral=env._rate_integral.clone()
    env._launch_ball(torch.tensor([0,3,7]))
    assert torch.equal(env.ball.data.root_pos_w[[1,2,4,5,6,8]],before[[1,2,4,5,6,8]])
    assert torch.equal(integral,env._rate_integral)


def test_waiting_observations_hide_parked_ball_and_expose_schedule(env):
    env._ball_active[:]=False;env._park_ball(torch.arange(9))
    obs=env._get_observations()
    assert obs['critic'].shape==(9,56) and obs['policy'].shape==(9,44)
    assert (obs['critic'][:,28:39]==0).all() and (obs['policy'][:,28:34]==0).all()
    assert (obs['critic'][:,43]==0).all()  # ball-active flag after four previous phase values
    assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all()


@pytest.mark.parametrize('mode', (*control.MODES, 'indi'))
def test_rotor_omega_is_default_off_and_updates_only_critic_size(mode):
    from test_indi import indi

    cfg = control.config()
    assert cfg.critic_obs_includes_rotor_omega is False
    cfg.control_mode, cfg.indi = mode, indi.IndiControllerCfg()
    configure = control.method('borinot_goalkeeper_env_cfg.py', 'configure_control')
    configure(cfg)
    original = cfg.action_space, cfg.observation_space, cfg.state_space
    for enabled in (True, True, False, True, False):
        cfg.critic_obs_includes_rotor_omega = enabled
        configure(cfg)
        assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (
            original[0], original[1], original[2] + 6 * enabled)


@pytest.mark.parametrize('mode', (*control.MODES, 'indi'))
def test_optional_rotor_omega_appends_current_speeds_without_changing_actor(env, mode):
    env.cfg.control_mode = mode
    env.actions = torch.full((env.num_envs, 9 if mode == 'indi' else 6 if mode == control.MODES[2] else 8), .25)
    env._indi = SimpleNamespace(observation=lambda: torch.zeros(env.num_envs, 8))
    # Deliberately unrelated commands and realized rotor speeds; preserve motor order.
    omega = torch.tensor([0., 50., 400., 800., 1200., 1600.]).expand(env.num_envs, -1)
    env._rotor_thrust = env.cfg.kf * omega.square()
    before = env._get_observations()
    env.cfg.critic_obs_includes_rotor_omega = True
    after = env._get_observations()
    assert torch.equal(after['policy'], before['policy'])
    assert torch.equal(after['critic'][:, :-6], before['critic'])
    torch.testing.assert_close(after['critic'][:, -6:], omega)
    assert torch.equal(env._get_observations()['critic'], after['critic'])
    assert torch.equal(env._rotor_thrust, env.cfg.kf * omega.square())
    env.cfg.critic_obs_includes_rotor_omega = False
    assert torch.equal(env._get_observations()['critic'], before['critic'])


@pytest.mark.parametrize('tau', [0., .04])
@pytest.mark.parametrize('sac', [False, True])
def test_observed_rotor_speed_follows_actual_motor_response(env, tau, sac):
    import math as python_math

    motor, _ = control.action_env(control.MODES[0], motor_time_constant_s=tau)
    motor.cfg.sac_action_mapping = sac
    motor.actions[:, :6] = .5
    requested = motor._hover_per_rotor * (1 + motor.cfg.rotor_action_scale * .5)
    if sac:
        requested = motor._hover_per_rotor + .5 * motor._hover_per_rotor * (motor.cfg.thrust_to_weight - 1)
    env.cfg.critic_obs_includes_rotor_omega = True
    for substep in range(1, 5):
        control.method('borinot_goalkeeper_env.py', '_apply_action')(motor)
        env._rotor_thrust = motor._rotor_thrust.repeat(3, 1)
        expected_thrust = requested if tau == 0 else (
            requested + (motor._hover_per_rotor - requested) * python_math.exp(-substep * motor.physics_dt / tau))
        expected_speed = python_math.sqrt(expected_thrust / motor.cfg.kf)
        torch.testing.assert_close(env._get_observations()['critic'][:, -6:],
                                   torch.full((env.num_envs, 6), expected_speed, dtype=torch.float64))


@pytest.mark.parametrize('mode',control.MODES)
@pytest.mark.parametrize('point,time,phase,params',[
    (p,t,h,b) for p in (False,True) for t in (False,True) for h in (False,True) for b in (False,True)])
def test_actor_flags_select_columns_and_critic_keeps_all(env,mode,point,time,phase,params):
    env.cfg.control_mode=mode
    configure=control.method('borinot_goalkeeper_env_cfg.py','configure_control')
    cfg=control.config();cfg.control_mode=mode;configure(cfg)
    env.actions=torch.zeros(9,cfg.action_space)
    # Nonzero, mostly distinct values in every optional column: an incoming ball and a live shot clock.
    env.ball.data.root_pos_w[:]=torch.tensor([2.,.3,1.6]);env.ball.data.root_lin_vel_w[:]=torch.tensor([-4.,.1,.5])
    env._hit[:]=True;env._clear[:]=True;env._shot_steps[:]=7;env._episode_steps[:]=11;env._recovery_time[:]=.2
    env._ball_parameters[:]=torch.tensor([.035,.5])
    critic_before=env._get_observations()['critic']
    for config in (env.cfg,cfg):
        config.obs_actor_includes_ball_plane_intersection_point=point
        config.obs_actor_includes_time_ball_collides_plane=time
        config.obs_actor_includes_phase=phase
        config.obs_actor_includes_ball_params=params
    obs=env._get_observations()
    assert torch.equal(obs['critic'],critic_before) and (obs['critic'][:,37:39]>0).all()
    end=cfg.state_space
    keep=(list(range(34))+[34,35,36]*point+[37]*time+[38]*(point or time)
          +list(range(39,46))*phase+list(range(46,end-2))+list(range(end-2,end))*params)
    assert torch.equal(obs['policy'],obs['critic'][:,keep])
    configure(cfg)
    assert (cfg.observation_space,cfg.state_space)==(obs['policy'].shape[1],obs['critic'].shape[1])


@pytest.mark.parametrize('mode',control.MODES)
def test_rotor_thrust_and_rate_integral_are_observed_only_in_pi_mode(env,mode):
    env.cfg.control_mode=mode;pi=mode==control.MODES[2]
    env.actions=torch.full((9,6 if pi else 8),.25)  # fixture thrust/hover and integral are ones
    obs=env._get_observations()
    tail=[env.actions]+[torch.ones(9,9)]*pi
    assert torch.equal(obs['critic'][:,46:-2],torch.cat(tail,-1))
    assert torch.equal(obs['policy'][:,34:-2],torch.cat(tail,-1))


def test_disabled_ball_params_do_not_leak_to_actor(env):
    env.cfg.obs_actor_includes_ball_params=False
    before=env._get_observations()
    env._ball_parameters[:]=torch.tensor([.049,.64])
    after=env._get_observations()
    assert torch.equal(before['policy'],after['policy'])
    assert torch.equal(before['critic'][:,:-2],after['critic'][:,:-2])
    assert torch.equal(after['critic'][:,-2:],env._ball_parameters)
    assert not torch.equal(before['critic'][:,-2:],after['critic'][:,-2:])


@pytest.mark.parametrize('mode,limits',[('bad',(2.,3.)),('launch',(-1.,3.)),('launch',(3.,2.)),('launch',(0.,float('nan')))])
def test_invalid_schedule_is_rejected(env,mode,limits):
    env.cfg.shot_resampling_mode=mode;env.cfg.shot_resampling_time_range_s=limits
    with pytest.raises(ValueError):math.validate_shot_schedule(env.cfg)


def test_random_intervals_and_fixed_zero_delay(env):
    values=math.sample_shot_delay_steps(10000,'cpu',env.cfg,.01)
    assert values.min()>=200 and values.max()<=300 and values.unique().numel()>90
    env.cfg.shot_resampling_time_range_s=(0.,0.)
    assert not math.sample_shot_delay_steps(100,'cpu',env.cfg,.01).any()


@pytest.mark.parametrize('columns,expected',[([1],[False,True,False,False]),([0,1,2,3],[True]*4)])
def test_only_configured_bodies_trigger_non_ball_crash(columns,expected):
    detector=next(n for n in ast.walk(tree) if isinstance(n,ast.AugAssign)
                  and isinstance(n.target,ast.Subscript) and isinstance(n.target.slice,ast.Constant)
                  and n.target.slice.value=='crash')
    forces=torch.zeros(4,4,3)
    for i in range(4):forces[i,i,0]=.2
    e=SimpleNamespace(_crash_contact_columns=columns,cfg=SimpleNamespace(robot_crash_force_threshold_n=.1),
                      _flags={'crash':torch.zeros(4,dtype=torch.bool)})
    namespace={'self':e,'other_contact':forces,'torch':torch}
    exec(compile(ast.Module(body=[detector],type_ignores=[]),'crash_detector','exec'),namespace)
    assert e._flags['crash'].tolist()==expected
