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
        "_get_dones": lambda self: (self.safety_failures.clone(), self._episode_steps >= self.max_episode_length)})
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
    env._power_sum_w, env._power_samples = torch.zeros(2, 4), 2
    env._action_limit_costs = torch.zeros(2, 3)
    env._events = {name: torch.zeros(2, dtype=torch.bool) for name in ("base_collision", "leave_area", "tilt")}
    env._self_contact_bodies = torch.zeros(2, 3, dtype=torch.bool)
    names = ("terminal_distance", "progress", "terminal_speed", "terminal_success", "action_rate", "energy",
             "base_collision", "leave_area", "self_collision", "tilt", "thrust_limits", "torque_limits", "action_limits")
    env._episode_terms = {name: torch.zeros(2, dtype=torch.float64) for name in names}
    env._episode_energy_j = torch.zeros(2)
    env._actuator_delay = SimpleNamespace(seconds=torch.zeros(2))
    env.extras = {}
    env.common_step_counter = 1000
    env.curriculum_stage_pure_episodes = 0
    env.goal_curriculum = task_math.SuccessCurriculum(cfg, "cpu")
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
    cfg.goal_deadline_range_s = ((4.123, 4.123),) * 5
    cfg.configure_control()
    assert cfg.episode_length_s == pytest.approx(math.ceil(4.123*150)/150)


@pytest.mark.parametrize("difficulty", range(5))
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


def test_terminal_distance_is_one_off_failure_floor_and_optional_rewards_are_off():
    env = fake_env()
    env._goal_finished[:] = True
    env._goal_deadline_reached[:] = True
    env._goal_failed[1] = True
    env._goal_success[0] = True
    distance = (env._goal_position_w-env.robot.data.root_pos_w).norm(dim=-1)
    env._get_rewards()
    terms = env.extras["reward_terms"]
    torch.testing.assert_close(terms["terminal_distance"], -10*torch.tensor([distance[0], 10.], dtype=distance.dtype))
    assert not terms["terminal_speed"].any() and not terms["terminal_success"].any()
    env.cfg.use_terminal_speed_penalty = env.cfg.use_terminal_success_bonus = True
    env._get_rewards()
    torch.testing.assert_close(env.extras["reward_terms"]["terminal_speed"], torch.tensor([-25., 0.]))
    assert env.extras["reward_terms"]["terminal_success"].tolist() == [10., 0.]
    saved_failed = env.extras["goal"]["failed"].clone()
    env._goal_failed.zero_()
    assert torch.equal(env.extras["goal"]["failed"], saved_failed)


def test_deadline_is_terminal_not_timeout_early_arrival_not_success_and_failure_wins_tie():
    env = fake_env()
    env._goal_position_w.copy_(env.robot.data.root_pos_w)
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
    for level in range(1, 9):
        assert c.level == level
        c.observe(torch.ones(10), torch.full((10,), level), 100 + level*10)
    assert c.level == 8
    for level, expected in enumerate([(0,1.),(1,1.),(2,1.),(3,1.),(4,1.),(4,.75),(4,.5),(4,.25),(4,0.)]):
        assert task_math.stage_settings(cfg, level) == expected
    c.observe(torch.ones(10), torch.full((10,), 8), 10000)
    assert c.level == 8


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
    ("goal_deadline_range_s", ((0.,1.),)*5), ("goal_curriculum_start_level", 9),
    ("goal_curriculum_start_level", True), ("progress_reward_factors", (1.,.5,.6,0.)),
    ("progress_reward_factors", (1.,.5)), ("curriculum_success_threshold", float('nan')),
    ("goal_success_tolerance_m", 0.), ("failure_distance_floor_m", -1.),
    ("rwd_scale_terminal_distance", 10.), ("rwd_scale_progress", -1.),
    ("curriculum_min_episodes", 2.5), ("curriculum_min_steps", 0),
    ("use_terminal_speed_penalty", "False"), ("robot_home", (0.,0.,.5)),
])
def test_invalid_config_is_rejected_before_simulation(name, value):
    with pytest.raises(ValueError):
        config(**{name:value})


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
    assert task_math.SuccessCurriculum(cfg,"cpu").level == 8
    cfg = config(progress_reward_curriculum=False, skip_curriculum=True)
    assert task_math.SuccessCurriculum(cfg,"cpu").level == 4
    assert task_math.stage_settings(cfg,4) == (4,1.)
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
