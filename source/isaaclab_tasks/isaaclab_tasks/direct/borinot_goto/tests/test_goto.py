# SPDX-License-Identifier: BSD-3-Clause
"""Independent CPU oracles for the real sampler, config and task methods."""
import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FOLDER.parent / "borinot_cmd_vel/tests"), str(FOLDER.parent / "borinot_goalkeeper/tests")]
from test_observation_flags import config as flight_config, observation_env, inverse, matrix, MODES
from test_task import method as flight_method, limits

spec = importlib.util.spec_from_file_location("goto_math_test", FOLDER / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


def config(mode=MODES[0], **overrides):
    source = ast.parse((FOLDER / "borinot_goto_env_cfg.py").read_text())
    node = next(n for n in source.body if isinstance(n, ast.ClassDef))
    base = flight_config(mode)
    parent = type(base)
    values = vars(base).copy()
    for n in node.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    node.bases, node.decorator_list = [ast.Name(id="Parent", ctx=ast.Load())], []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    scope = {"Parent": parent, "math": math, **vars(task_math)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "goto_config", "exec"), scope)
    cfg = scope[node.name]()
    cfg.__dict__.update(values)
    cfg.__dict__.update(overrides)
    cfg.configure_control()
    return cfg


def environment_class():
    node = next(n for n in ast.parse((FOLDER / "borinot_goto_env.py").read_text()).body if isinstance(n, ast.ClassDef))
    node.decorator_list, node.bases = [], [ast.Name(id="Parent", ctx=ast.Load())]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name != "__init__"]
    parent = type("FlightMethods", (), {
        "_get_observations": flight_method("_get_observations", quat_apply_inverse=inverse, matrix_from_quat=matrix),
        "_get_dones": lambda self: (self.safety_failures.clone(), self._episode_steps >= self.max_episode_length),
        "_update_pushes": flight_method("_update_pushes")})
    scope = {"Parent": parent, "torch": torch, "quat_apply_inverse": inverse,
             "clipped_limit_rewards": limits.clipped_limit_rewards, **vars(task_math)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "goto_env", "exec"), scope)
    return scope[node.name]


def fake_env(cfg=None):
    cfg = cfg or config()
    env = environment_class()()
    env.__dict__.update(vars(observation_env(cfg)))
    env.device, env.num_envs, env.step_dt, env.physics_dt = "cpu", 2, .01, .005
    env.max_episode_length = 350
    env._goal_position_w = env.robot.data.root_pos_w + torch.tensor([[.8, -.3, .4], [-.5, .7, .2]])
    env._goal_deadline_steps = torch.tensor([100, 500])
    env._episode_steps = torch.tensor([0, 100])
    env.safety_failures = torch.zeros(2, dtype=torch.bool)
    env._goal_failed = env.safety_failures.clone()
    env._goal_deadline_reached = env.safety_failures.clone()
    env._goal_finished = env.safety_failures.clone()
    env._goal_success = env.safety_failures.clone()
    env._previous_goal_distance = (env._goal_position_w - env.robot.data.root_pos_w).norm(dim=-1) + torch.tensor([.1, -.2])
    env._initial_goal_distance = env._previous_goal_distance.clone()
    env._minimum_goal_distance = env._previous_goal_distance.clone()
    env._first_arrival_s = torch.full((2,), -1.)
    env._goal_progress_factor = torch.tensor([1., .25])
    env._goal_episode_level = torch.zeros(2, dtype=torch.long)
    env._commands = torch.zeros(2, 4, dtype=torch.float64)
    env._actuator_commands = torch.zeros(2, 8)
    env._previous_actuator_commands = torch.zeros(2, 8)
    env.robot.data.applied_torque = torch.zeros(2, 2)
    env.robot.data.root_lin_vel_w = torch.tensor([[3., 4., 0.], [0., 0., 2.]])
    env.robot.data.root_ang_vel_w = torch.zeros(2, 3)
    env._power_sum_w, env._power_samples = torch.zeros(2, 4), 2
    env._action_limit_costs = torch.zeros(2, 3)
    env._events = {name: torch.zeros(2, dtype=torch.bool) for name in ("base_collision", "leave_area", "tilt")}
    env._self_contact_bodies = torch.zeros(2, 3, dtype=torch.bool)
    names = ("terminal_distance", "progress", "terminal_linear_vel", "terminal_ang_vel", "terminal_success",
             "action_rate", "energy", "base_collision", "leave_area", "self_collision", "tilt",
             "thrust_limits", "torque_limits", "action_limits")
    env._episode_terms = {name: torch.zeros(2, dtype=torch.float64) for name in names}
    env._episode_energy_j = torch.zeros(2)
    env._actuator_delay = SimpleNamespace(seconds=torch.zeros(2))
    env.extras = {}
    env.common_step_counter = 1000
    env.curriculum_stage_pure_episodes = 0
    env.goal_curriculum = task_math.SuccessCurriculum(cfg, "cpu")
    # Parent push state after a parent reset: a full first interval, no push active.
    env._episode_stage = torch.zeros(2, dtype=torch.long)
    env._force_limits = torch.tensor([task_math.stage_force_limit(cfg, level)
                                      for level in range(env.goal_curriculum.max_level + 1)])
    env._base_push_steps_until_next = torch.tensor([300, 500])
    env._base_push_steps_left = torch.zeros(2, dtype=torch.long)
    env._base_push_force_z_w = torch.zeros(2)
    env._base_push_points_b = torch.zeros(2, 3)
    return env


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("acc,gravity,height", [(False, False, True), (True, True, False)])
def test_observation_frames_time_information_and_idempotent_reads(mode, acc, gravity, height):
    cfg = config(mode, obs_include_base_lin_acc=acc, projected_gravity_in_obs=gravity, include_base_height=height)
    env = fake_env(cfg)
    out = env._get_observations()
    obs = out["policy"]
    assert obs.shape == (2, cfg.observation_space) and cfg.state_space == cfg.observation_space
    assert out["critic"] is obs
    torch.testing.assert_close(env._commands[:, :3], inverse(env.robot.data.root_quat_w,
                                                          env._goal_position_w-env.robot.data.root_pos_w))
    torch.testing.assert_close(env._commands[:, 3], torch.tensor([1., .8], dtype=torch.float64))
    torch.testing.assert_close(obs[:, -1], torch.tensor([1., 4.], dtype=obs.dtype))
    for _ in range(3):
        assert torch.equal(env._get_observations()["policy"], obs)
    # Equal t_norm but different absolute budgets MUST remain distinguishable.
    env._episode_steps[:] = 0
    assert torch.equal(env._get_observations()["policy"][:, -1], torch.tensor([1., 5.], dtype=obs.dtype))
    env._episode_steps[:] = env._goal_deadline_steps + 1
    obs = env._get_observations()["policy"]
    assert not env._commands[:, 3].any() and not obs[:, -1].any()


def test_config_is_idempotent_derived_horizon_follows_clock_and_not_runner_counter():
    cfg = config()
    assert cfg.observation_space == cfg.state_space == 33
    assert cfg.is_finite_horizon and cfg.episode_length_s == 3.5
    cfg.configure_control()
    assert cfg.observation_space == 33
    cfg.physics_hz, cfg.policy_hz = 300, 150
    cfg.goal_deadline_range_s = ((4.123, 4.123),) * len(cfg.goal_half_extents_m)
    cfg.configure_control()
    assert cfg.episode_length_s == pytest.approx(math.ceil(4.123*150)/150)


@pytest.mark.parametrize("difficulty", range(len(config().goal_half_extents_m)))
def test_sampling_uniform_box_no_hidden_direction_bias_and_deadline_floors(difficulty):
    cfg = config()
    torch.manual_seed(13)
    start = torch.tensor([[100., -200., 1.5]], dtype=torch.float64).expand(30000, -1)
    goals, steps = task_math.sample_goals(start, cfg, difficulty, .01)
    delta = goals - start
    extent = torch.tensor(cfg.goal_half_extents_m[difficulty], dtype=delta.dtype)
    assert (delta.abs() <= extent + 1e-12).all()
    assert (delta.mean(0).abs() < extent*.02).all()
    torch.testing.assert_close(delta.square().mean(0), extent.square()/3, rtol=.02, atol=0)
    distance, duration = delta.norm(dim=-1), steps*.01
    assert (duration >= cfg.goal_deadline_range_s[difficulty][0]-1e-6).all()
    assert (duration >= distance/cfg.goal_reference_speed_m_s[difficulty]-1e-6).all()
    assert (duration >= 2*torch.sqrt(distance/cfg.goal_reference_accel_m_s2[difficulty])-1e-6).all()
    assert duration.max() <= cfg.episode_length_s + 1e-6


def test_progress_is_signed_no_dt_and_dense_factor_is_frozen_per_episode():
    env = fake_env()
    reward = env._get_rewards()
    torch.testing.assert_close(reward, torch.tensor([.1, -.05], dtype=reward.dtype), atol=1e-7, rtol=0)
    env.step_dt = .1
    assert torch.equal(env._get_rewards(), reward)
    env._goal_progress_factor.zero_()
    assert not env._get_rewards().any()


def test_terminal_distance_floor_velocity_costs_on_by_default_and_success_bonus_off():
    env = fake_env()
    env._goal_finished[:] = True
    env._goal_deadline_reached[:] = True
    env._goal_failed[1] = True
    env._goal_success[0] = True
    env.robot.data.root_ang_vel_w = torch.tensor([[1., -2., 2.], [3., 0., 0.]])
    distance = (env._goal_position_w-env.robot.data.root_pos_w).norm(dim=-1)
    env._get_rewards()
    terms = env.extras["reward_terms"]
    torch.testing.assert_close(terms["terminal_distance"], -10*torch.tensor([distance[0], 10.], dtype=distance.dtype))
    # Clean deadline only: the failed robot pays the distance floor, not velocity costs.
    torch.testing.assert_close(terms["terminal_linear_vel"], torch.tensor([-25., 0.]))
    torch.testing.assert_close(terms["terminal_ang_vel"], torch.tensor([-9., 0.]))
    assert not terms["terminal_success"].any()
    env.cfg.rwd_scale_terminal_linear_vel, env.cfg.rwd_scale_terminal_ang_vel = -.5, -.25
    env.cfg.use_terminal_success_bonus = True
    env._get_rewards()
    torch.testing.assert_close(env.extras["reward_terms"]["terminal_linear_vel"], torch.tensor([-12.5, 0.]))
    torch.testing.assert_close(env.extras["reward_terms"]["terminal_ang_vel"], torch.tensor([-2.25, 0.]))
    assert env.extras["reward_terms"]["terminal_success"].tolist() == [10., 0.]
    env._goal_deadline_reached[0] = False  # failure-free but before the deadline: no terminal costs
    env._get_rewards()
    assert not env.extras["reward_terms"]["terminal_linear_vel"].any()
    assert not env.extras["reward_terms"]["terminal_ang_vel"].any()
    env._goal_deadline_reached[0] = True
    env.cfg.use_terminal_linear_vel_penalty = env.cfg.use_terminal_ang_vel_penalty = False
    env._get_rewards()
    assert not env.extras["reward_terms"]["terminal_linear_vel"].any()
    assert not env.extras["reward_terms"]["terminal_ang_vel"].any()
    saved_failed = env.extras["goal"]["failed"].clone()
    env._goal_failed.zero_()
    assert torch.equal(env.extras["goal"]["failed"], saved_failed)


def test_deadline_is_terminal_not_timeout_early_arrival_not_success_and_failure_wins_tie():
    env = fake_env()
    env._goal_position_w.copy_(env.robot.data.root_pos_w)
    env.robot.data.root_lin_vel_w = torch.zeros(2, 3)
    env._episode_steps[:] = torch.tensor([99, 499])
    # Generic runner counter / horizon must not shorten the independent goal clock.
    env.episode_length_buf = torch.tensor([10000, 10000])
    env.max_episode_length = 1
    done, timeout = env._get_dones()
    assert not done.any() and not timeout.any() and not env._goal_success.any()
    assert (env._first_arrival_s >= 0).all()
    env._episode_steps += 1
    env.safety_failures[1] = True
    done, timeout = env._get_dones()
    assert done.all() and not timeout.any() and env._goal_success.tolist() == [True, False]
    assert not env._get_observations()["policy"][:, -1].any()


def test_success_needs_the_goal_ball_and_both_speeds_strictly_below_their_tolerances():
    cfg = config(goal_success_tolerance_m=.25, goal_success_lin_vel_tolerance_m_s=.5,
                 goal_success_ang_vel_tolerance_rad_s=.25)
    env = fake_env(cfg)
    data = env.robot.data
    env._goal_position_w = data.root_pos_w + torch.tensor([[0., 0., .25], [0., 0., .125]], dtype=torch.float64)
    env._episode_steps.copy_(env._goal_deadline_steps)  # clean deadline for both robots

    def success(lin, ang):
        data.root_lin_vel_w = torch.tensor(lin, dtype=torch.float64)
        data.root_ang_vel_w = torch.tensor(ang, dtype=torch.float64)
        done, _ = env._get_dones()
        assert torch.equal(done, env._goal_deadline_reached)
        return env._goal_success.tolist()

    slow, still = [[0., 0., .25]] * 2, [[0., 0., .125]] * 2
    assert success(slow, still) == [True, True]  # the ball edge counts as inside
    assert success([[0., 0., .5], [0., -.25, 0.]], still) == [False, True]  # |v| == tolerance fails
    assert success(slow, [[.25, 0., 0.], [0., .125, 0.]]) == [False, True]  # |omega| on any axis
    assert success([[.3, .3, .3], [0., 0., 0.]], [[0., 0., 0.], [.15, .15, .15]]) == [False, False]  # norms, not axes
    env._goal_position_w[1, 2] += .25  # robot 1 is now 0.375 m from its goal
    assert success(slow, still) == [True, False]
    # Before the deadline nothing succeeds; first arrival still uses position only.
    env._episode_steps -= 1
    env._first_arrival_s[:] = -1.
    assert success([[3., 0., 0.]] * 2, [[2., 0., 0.]] * 2) == [False, False]
    assert env._first_arrival_s[0] >= 0 and env._first_arrival_s[1] < 0


def test_force_entries_start_at_their_goal_level_and_extend_a_short_schedule():
    cfg = config()  # 6 geometry + 4 fade levels, the inherited cmd-vel force list from level 5
    assert cfg.forces_applied_to_base_curriculum == (0., 1., 3., 6., 9.)
    assert cfg.start_forces_randn_curriculum_idx == 5 and task_math.max_curriculum_level(cfg) == 9
    assert [task_math.stage_force_limit(cfg, level) for level in range(10)] == [0.] * 6 + [1., 3., 6., 9.]
    # Forces running past the fade add force-only final levels at the final geometry and factor.
    cfg = config(start_forces_randn_curriculum_idx=7)
    assert task_math.max_curriculum_level(cfg) == 11
    assert [task_math.stage_force_limit(cfg, level) for level in range(12)] == [0.] * 8 + [1., 3., 6., 9.]
    assert [task_math.stage_settings(cfg, level) for level in (9, 10, 11)] == [(5, 0.)] * 3
    assert task_math.SuccessCurriculum(config(start_forces_randn_curriculum_idx=7, skip_curriculum=True),
                                       "cpu").level == 11
    config(start_forces_randn_curriculum_idx=7, goal_curriculum_start_level=11)  # a valid stage now
    # Early start: the last force holds through the fade.
    cfg = config(start_forces_randn_curriculum_idx=1)
    assert task_math.max_curriculum_level(cfg) == 9
    assert [task_math.stage_force_limit(cfg, level) for level in range(10)] == [0., 0., 1., 3., 6.] + [9.] * 5
    # A push-free list never lengthens the schedule.
    cfg = config(forces_applied_to_base_curriculum=(0.,), start_forces_randn_curriculum_idx=50)
    assert task_math.max_curriculum_level(cfg) == 9 and task_math.stage_force_limit(cfg, 60) == 0.


def test_pushes_follow_the_episode_goal_level_and_push_free_lists_keep_the_random_stream():
    cfg = config(base_push_duration_range_s=(.05, .05), base_push_interval_range_s=(.1, .1))
    env = fake_env(cfg)
    env._episode_stage = torch.tensor([5, 9])  # level 5 is the zero entry; level 9 allows 9 N
    env._base_push_steps_until_next.zero_()
    torch.manual_seed(3)
    forces = []
    for _ in range(200):
        env._update_pushes()
        forces.append(env._base_push_force_z_w.clone())
    forces = torch.stack(forces)
    assert not forces[:, 0].any()
    assert (forces[:, 1].abs() <= 9.).all() and forces[:, 1].abs().max() > 6.
    assert (forces[:, 1] > 0).any() and (forces[:, 1] < 0).any()
    env = fake_env(config(forces_applied_to_base_curriculum=(0.,)))
    env._base_push_steps_until_next.zero_()
    state = torch.get_rng_state()
    env._update_pushes()
    assert torch.equal(torch.get_rng_state(), state) and not env._base_push_steps_left.any()


def test_reset_tags_the_push_level_and_starts_the_push_clock_at_a_random_phase():
    goto = environment_class()

    def flight_reset(self, env_ids):
        self._episode_stage[env_ids] = 0  # the parent's unused force level
        self._base_push_steps_until_next[env_ids] = 400  # the parent's full first interval

    goto.__mro__[1]._reset_idx = flight_reset
    env = goto()
    env.__dict__.update(vars(fake_env()))
    env.goal_curriculum.level = 7
    env.robot._ALL_INDICES = torch.arange(2)
    phases = []
    for seed in range(300):
        torch.manual_seed(seed)
        env._reset_idx(None)
        phases.append(env._base_push_steps_until_next.clone())
    phases = torch.stack(phases)
    assert env._episode_stage.tolist() == env._goal_episode_level.tolist() == [7, 7]
    assert (phases >= 0).all() and (phases < 400).all()
    assert phases.float().mean() == pytest.approx(199.5, abs=12) and phases.min() < 10 and phases.max() > 390
    env.cfg.forces_applied_to_base_curriculum = (0.,)
    env._reset_idx(None)
    assert env._base_push_steps_until_next.tolist() == [400, 400]  # push-free: parent clock unchanged


def test_curriculum_episode_weighting_unequal_batches_not_mean_of_rates():
    cfg = config(curriculum=False)
    c = task_math.SuccessCurriculum(cfg, "cpu")
    c.observe(torch.ones(1), torch.zeros(1), 1)
    result = c.observe(torch.cat((torch.ones(60), torch.zeros(40))), torch.zeros(100), 2)
    assert result["window_success_rate"] == pytest.approx(61/101)
    assert result["window_completed_count"] == 101


def test_curriculum_dwell_evidence_stage_purity_threshold_and_full_fade():
    cfg = config(curriculum_min_episodes=2, curriculum_window_episodes=4, curriculum_min_steps=10)
    c = task_math.SuccessCurriculum(cfg, "cpu")
    c.observe(torch.ones(2), torch.zeros(2), 9)
    assert c.level == 0
    c.observe(torch.ones(0), torch.zeros(0), 10)
    assert c.level == 0  # old evidence cannot promote during an empty batch
    decision = c.observe(torch.ones(1), torch.zeros(1), 10)
    assert c.level == 1 and decision["evaluated_stage"] == 0 and decision["window_completed_count"] == 3
    assert not c.window.any() and c.count == 0
    c.observe(torch.ones(50), torch.zeros(50), 100)
    assert c.level == 1 and c.count == 0  # old episodes are not eligible
    geometry_levels = len(cfg.goal_half_extents_m)
    final_level = geometry_levels + len(cfg.progress_reward_factors) - 2
    for level in range(1, final_level + 1):
        assert c.level == level
        c.observe(torch.ones(10), torch.full((10,), level), 100 + level*10)
    assert c.level == final_level
    expected_stages = [(level, 1.) for level in range(geometry_levels)]
    expected_stages += [(geometry_levels - 1, factor) for factor in cfg.progress_reward_factors[1:]]
    for level, expected in enumerate(expected_stages):
        assert task_math.stage_settings(cfg, level) == expected
    c.observe(torch.ones(10), torch.full((10,), final_level), 10000)
    assert c.level == final_level


def test_curriculum_exact_threshold_rolling_window_and_manual_reset_exclusion():
    cfg = config(curriculum_min_episodes=5, curriculum_window_episodes=5, curriculum_min_steps=1)
    c = task_math.SuccessCurriculum(cfg, "cpu")
    decision = c.observe(torch.tensor([1.,1.,1.,1.,0.]), torch.zeros(5), 1)
    assert c.level == 1 and decision["window_success_rate"] == pytest.approx(.8)
    cfg.curriculum = False
    c.observe(torch.zeros(12), torch.ones(12), 2)
    assert c.count == 12 and not c.window.any()
    c.observe(torch.ones(3), torch.ones(3), 3)
    assert c.observe(torch.zeros(0), torch.zeros(0), 4)["window_success_rate"] == pytest.approx(.6)
    env = fake_env()
    env._episode_steps[:] = 100
    env._update_curriculum(torch.tensor([0,1]))
    assert not env.extras and env.goal_curriculum.count == 0


@pytest.mark.parametrize("name,value", [
    ("goal_half_extents_m", ()), ("goal_half_extents_m", ((0.,0.,0.),)),
    ("goal_reference_speed_m_s", (1.,)), ("goal_reference_accel_m_s2", (0.,)*5),
    ("goal_deadline_range_s", ((0.,1.),)*5), ("goal_curriculum_start_level", -1),
    ("goal_curriculum_start_level", True), ("progress_reward_factors", (1.,.5,.6,0.)),
    ("progress_reward_factors", (1.,.5)), ("curriculum_success_threshold", float('nan')),
    ("goal_success_tolerance_m", 0.), ("failure_distance_floor_m", -1.),
    ("rwd_scale_terminal_distance", 10.), ("rwd_scale_progress", -1.),
    ("curriculum_min_episodes", 2.5), ("curriculum_min_steps", 0),
    ("use_terminal_linear_vel_penalty", "False"), ("use_terminal_ang_vel_penalty", 1), ("robot_home", (0.,0.,.5)),
    ("goal_success_lin_vel_tolerance_m_s", 0.), ("goal_success_ang_vel_tolerance_rad_s", float('nan')),
    ("rwd_scale_terminal_linear_vel", float('inf')), ("rwd_scale_terminal_ang_vel", .5),
    ("start_forces_randn_curriculum_idx", -1), ("start_forces_randn_curriculum_idx", True),
    ("start_forces_randn_curriculum_idx", 2.),
])
def test_invalid_config_is_rejected_before_simulation(name, value):
    with pytest.raises(ValueError):
        config(**{name:value})


def test_curriculum_start_beyond_the_current_schedule_is_rejected():
    cfg = config()
    first_invalid = len(cfg.goal_half_extents_m) + len(cfg.progress_reward_factors) - 1
    with pytest.raises(ValueError):
        config(goal_curriculum_start_level=first_invalid)


def test_dense_telescoping_means_no_intrinsic_rush_for_same_terminal_state():
    early = torch.tensor([2.,1.,0.,0.,0.], dtype=torch.float64)
    late = torch.tensor([2.,2.,2.,1.,0.], dtype=torch.float64)
    assert (early[:-1]-early[1:]).sum() == (late[:-1]-late[1:]).sum() == 2


@pytest.mark.parametrize("mode", ["indi", "wrench", "cbrt", "cbrt_indi", "thrust_and_PD_joint_pos_control"])
def test_existing_sincos_action_expansion_also_expands_goal_observations(mode):
    flags = {"predict_thigh_joint_pos_as_sincos":True, "predict_calf_joint_pos_as_sincos":True}
    if mode in ("indi", "wrench"):
        flags["predict_yaw_as_sincos"] = True
    base, expanded = config(mode), config(mode, **flags)
    added = 3 if mode in ("indi", "wrench") else 2
    assert expanded.action_space == base.action_space + added
    assert expanded.observation_space == base.observation_space + added
    env = fake_env(expanded)
    assert env._get_observations()["policy"].shape == (2, expanded.observation_space)


def test_skip_fixed_and_disabled_fade_curriculum_semantics():
    cfg = config(skip_curriculum=True)
    final_geometry = len(cfg.goal_half_extents_m) - 1
    assert task_math.SuccessCurriculum(cfg,"cpu").level == final_geometry + len(cfg.progress_reward_factors) - 1
    cfg = config(progress_reward_curriculum=False, skip_curriculum=True, forces_applied_to_base_curriculum=(0.,))
    assert task_math.SuccessCurriculum(cfg,"cpu").level == final_geometry
    assert task_math.stage_settings(cfg,final_geometry) == (final_geometry,1.)
    # Without the fade, the default forces (levels 5-9) keep force-only levels after the geometry.
    cfg = config(progress_reward_curriculum=False, skip_curriculum=True)
    assert task_math.SuccessCurriculum(cfg,"cpu").level == 9
    assert task_math.stage_settings(cfg,9) == (final_geometry,1.) and task_math.stage_force_limit(cfg,9) == 9.
    cfg = config(curriculum=False,goal_curriculum_start_level=6)
    c = task_math.SuccessCurriculum(cfg,"cpu")
    c.observe(torch.ones(10000),torch.full((10000,),6),100000)
    assert c.level == 6


def test_episode_log_includes_failures_and_preserves_pre_promotion_decision():
    cfg = config(curriculum_min_episodes=1,curriculum_min_steps=1)
    env = fake_env(cfg)
    env._goal_finished[:] = True
    env._goal_deadline_reached[:] = True
    env._goal_failed[1] = True
    env._goal_success[0] = True
    env._get_rewards()
    env._episode_steps[:] = 100
    env._update_curriculum(torch.tensor([0,1]))
    logs = env.extras["log"]
    assert logs["Episode/completed_count"] == 2 and logs["Episode/success_count"] == 1
    assert logs["Curriculum/window_success_rate"] == .5 and env.goal_curriculum.level == 0
    # Only robot 0 reached a clean deadline: about 0.94 m away, 5 m/s, not rotating.
    assert logs["Goal/deadline_count"] == 1 and logs["Goal/deadline_inside_count"] == 0
    assert logs["Goal/deadline_lin_vel_ok_count"] == 0 and logs["Goal/deadline_ang_vel_ok_count"] == 1
    assert logs["Goal/deadline_speed_m_s"] == pytest.approx(5.) and logs["Goal/deadline_ang_speed_rad_s"] == 0
    assert logs["Curriculum/selected_force_limit_n"] == 0 and logs["Goal/collection_force_limit_n"] == 0
    env._goal_episode_level[:] = torch.tensor([6, 9])
    env._update_curriculum(torch.tensor([0,1]))
    assert env.extras["log"]["Goal/collection_force_limit_n"] == pytest.approx(5.)  # mean of 1 N and 9 N
    env._goal_episode_level.zero_()
    cfg.curriculum_success_threshold = .5
    env._update_curriculum(torch.tensor([0,1]))
    assert env.goal_curriculum.level == 1
    assert env.extras["log"]["Curriculum/evaluated_stage"] == 0
    assert env.extras["log"]["Curriculum/selected_stage"] == 1
    assert not env.curriculum_stage_pure_episodes


def test_goal_task_is_not_mistaken_for_velocity_symmetry_layout():
    from test_symmetry import symmetry
    with pytest.raises(ValueError, match="Go-to"):
        symmetry.BorinotSymmetry(config())
