# SPDX-License-Identifier: BSD-3-Clause
"""Run the simplified task's real event, reward, reset and observation methods on CPU tensors.

Nothing imports Isaac Sim: methods are extracted from the source files and bound to a small fake env.
"""
import ast
import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
ORIGINAL = FOLDER.parent / "borinot_goalkeeper"
SOURCE = FOLDER / "borinot_goalkeeper_simple_env.py"
spec = importlib.util.spec_from_file_location("simple_task_math", ORIGINAL / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)
N = 8
RACKET, ARM1, ARM2, BASE, GOAL = range(5)  # ball sensor columns
FLOOR, GOAL_BODY, LINK2, PADDLE = range(4)  # base sensor columns


def quat_apply(q, v):
    w, axis = q[:, :1], q[:, 1:]
    t = 2 * torch.linalg.cross(axis, v, dim=-1)
    return v + w * t + torch.linalg.cross(axis, t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(torch.cat((q[:, :1], -q[:, 1:]), dim=-1), v)


def class_node(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))


def methods(path):
    return {n.name: n for n in class_node(path).body if isinstance(n, ast.FunctionDef)}


SCOPE = {"torch": torch, "math": math, "quat_apply": quat_apply, "quat_apply_inverse": quat_apply_inverse,
         "matrix_from_quat": lambda q: torch.eye(3).expand(len(q), -1, -1)}
SCOPE.update({name: getattr(task_math, name) for name in (
    "crosses_goal", "plane_progress", "predict_intercept", "promotion_ready", "sample_shots", "sample_shot_delay_steps")})
limits_spec = importlib.util.spec_from_file_location("simple_action_limits", FOLDER.parent / "borinot/action_limits.py")
limits = importlib.util.module_from_spec(limits_spec)
limits_spec.loader.exec_module(limits)
SCOPE["raw_action_limit_costs"] = limits.raw_action_limit_costs
SCOPE["relative_limit_cost"] = limits.relative_limit_cost
SCOPE["clipped_limit_rewards"] = limits.clipped_limit_rewards
for node in ast.parse(SOURCE.read_text()).body:  # OUTCOME_*, EVENTS, REWARD_TERMS
    if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), SCOPE)
METHODS = methods(SOURCE)
BOUND = ("_paddle_pose", "_pre_physics_step", "_sample_physics_events", "_get_dones", "_get_rewards", "_task_rewards",
         "_integrity_rewards", "_desirability_rewards", "_get_observations", "_update_curriculum", "_record_shots",
         "_park_ball", "_launch_ball")
for node in [*(METHODS[name] for name in BOUND), methods(ORIGINAL / "borinot_goalkeeper_env.py")["_sample_power"]]:
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), SCOPE)
BOUND += ("_sample_power",)


def literal_fields(path):
    values = {}
    for n in class_node(path).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return values


def config(**overrides):
    values = literal_fields(FOLDER / "borinot_goalkeeper_simple_env_cfg.py")
    values["km_over_kf"] = values["km"] / values["kf"]
    values["sim"] = SimpleNamespace(dt=None, render_interval=None)
    values["robot"] = SimpleNamespace(
        spawn=SimpleNamespace(articulation_props=SimpleNamespace(enabled_self_collisions=None)),
        actuators={"arm": SimpleNamespace(stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=0.,
                                          effort_limit=None, velocity_limit=None)})
    cfg = SimpleNamespace(**{**values, **overrides})
    configure = methods(FOLDER / "borinot_goalkeeper_simple_env_cfg.py")["configure_control"]
    scope = {"math": math}
    exec(compile(ast.Module(body=[configure], type_ignores=[]), "cfg", "exec"), scope)
    scope["configure_control"](cfg)
    return cfg


class Sensor:
    def __init__(self, filters):
        self.data = SimpleNamespace(force_matrix_w=torch.zeros(N, 1, filters, 3))

    def reset(self, ids=None):
        self.data.force_matrix_w[ids] = 0.0

    def push(self, column, newtons):
        """Normal force on the given column for the listed environments (tensor of length N)."""
        self.data.force_matrix_w[:, 0, column, 2] = newtons


def make_env(**overrides):
    """Every scale except the hit/goal/integrity events is zero unless a test enables it."""
    defaults = {"rwd_scale_progress": 0.0, "rwd_scale_action_rate": 0.0, "rwd_scale_energy": 0.0, "curriculum": False}
    cfg = config(**{**defaults, **overrides})
    zeros3 = torch.zeros(N, 3)
    root_state = torch.zeros(N, 13)
    root_state[:, 3] = 1.0
    ball_data = SimpleNamespace(root_pos_w=torch.tensor([1.0, 0.0, 1.2]).repeat(N, 1),
                                root_lin_vel_w=torch.tensor([-5.0, 0.0, 0.0]).repeat(N, 1),
                                default_root_state=root_state.clone())

    def write_ball(state, ids):
        ball_data.root_pos_w[ids] = state[:, :3]
        ball_data.root_lin_vel_w[ids] = state[:, 7:10]

    body_pos = torch.zeros(N, 2, 3)
    body_pos[:, 0] = torch.tensor([0.0, 0.0, 1.5])  # base
    body_pos[:, 1] = torch.tensor([-0.03, 0.0, 1.13])  # paddle
    body_quat = torch.zeros(N, 2, 4)
    body_quat[..., 0] = 1.0
    noop = lambda *args, **kwargs: None
    env = SimpleNamespace(
        cfg=cfg, device="cpu", num_envs=N, physics_dt=1 / cfg.physics_hz, step_dt=cfg.decimation / cfg.physics_hz,
        max_episode_length=math.ceil(cfg.episode_length_s * cfg.policy_hz), extras={}, common_step_counter=0,
        scene=SimpleNamespace(env_origins=zeros3.clone()),
        ball=SimpleNamespace(data=ball_data, write_root_state_to_sim=write_ball, root_physx_view=SimpleNamespace(
            set_masses=noop, set_inertias=noop, set_material_properties=noop)),
        robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([0.0, 0.0, 1.5]).repeat(N, 1), root_quat_w=root_state[:, 3:7].clone(),
            root_lin_vel_w=zeros3.clone(), root_lin_vel_b=zeros3.clone(), root_ang_vel_b=zeros3.clone(),
            joint_pos=torch.zeros(N, 2), joint_vel=torch.zeros(N, 2), applied_torque=torch.zeros(N, 2),
            computed_torque=torch.zeros(N, 2),
            body_pos_w=body_pos, body_quat_w=body_quat)),
        ball_contact=Sensor(5), base_contact=Sensor(4), paddle_contact=Sensor(1),
        base_id=0, paddle_id=1, joint_ids=[0, 1],
        _home=torch.tensor(cfg.robot_home), _x_axis=torch.tensor([1.0, 0.0, 0.0]).expand(N, -1),
        _gravity=torch.tensor([0.0, 0.0, -1.0]).expand(N, -1),
        _area_min=torch.tensor([cfg.goal_area_x_m[0], cfg.goal_area_y_m[0], cfg.goal_area_z_m[0]]),
        _area_max=torch.tensor([cfg.goal_area_x_m[1], cfg.goal_area_y_m[1], cfg.goal_area_z_m[1]]),
        actions=torch.zeros(N, cfg.action_space), _rate_integral=torch.zeros(N, 3), _joint_scale=torch.ones(2),
        _action_limit_costs=torch.zeros(N, 3),
        _hover_per_rotor=5.0, _rotor_thrust=torch.full((N, 6), 5.0),
        _actuator_commands=torch.full((N, 8), 0.0), _previous_actuator_commands=torch.zeros(N, 8),
        _power_sum_w=torch.zeros(N, 4), _power_samples=0, _physics_samples=0, _episode_energy_j=torch.zeros(N),
        _unpaid_energy_j=torch.zeros(N),
        _ball_active=torch.ones(N, dtype=torch.bool), _touched=torch.zeros(N, dtype=torch.bool),
        _ball_done=torch.zeros(N, dtype=torch.bool), _progress_active=torch.zeros(N, dtype=torch.bool),
        _events={name: torch.zeros(N, dtype=torch.bool) for name in SCOPE["EVENTS"]},
        _previous_paddle=zeros3.clone(), _previous_ball=ball_data.root_pos_w.clone(),
        _outcome=torch.zeros(N, dtype=torch.long), _shot_steps=torch.zeros(N, dtype=torch.long),
        _episode_steps=torch.zeros(N, dtype=torch.long), _next_launch_steps=torch.full((N,), 10_000),
        _shot_stage=torch.zeros(N, dtype=torch.long), _ball_parameters=torch.zeros(N, 2),
        _ball_masses=torch.ones(N, 1), _ball_inertias=torch.ones(N, 9), _ball_materials=torch.ones(N, 1, 3),
        _episode_rewards=torch.zeros(N),
        _episode_terms={f"{g}/{t}": torch.zeros(N) for g, terms in SCOPE["REWARD_TERMS"].items() for t in terms},
        _episode_counts={name: torch.zeros(N, dtype=torch.long)
                         for name in ("shots", "saves", "goals", "hit_racket", "hit_arm", "hit_base")},
        outcome_counts=torch.zeros(10, dtype=torch.long), _episode_stage=torch.zeros(N, dtype=torch.long),
        shot_level=0, _stage_start_step=0, _curriculum_window=torch.zeros(cfg.curriculum_window_shots),
        _window_cursor=0, _stage_shots=0, curriculum_stage_pure_episodes=0,
        reset_buf=torch.zeros(N, dtype=torch.bool), reset_terminated=torch.zeros(N, dtype=torch.bool),
        _actuator_delay=SimpleNamespace(seconds=torch.zeros(N)),
    )
    for name in BOUND:
        setattr(env, name, MethodType(SCOPE[name], env))
    return env


def step(env, *substeps):
    """One policy transition. Each substep callback writes that physics step's result."""
    env._pre_physics_step(torch.zeros(N, env.cfg.action_space))
    for index, write_result in enumerate(substeps or [lambda e: None] * env.cfg.decimation):
        if index:  # the shared _apply_action() inspects the previous result first
            env._sample_power()
            env._sample_physics_events()
        env._physics_samples += 1
        write_result(env)
    terminated, truncated = env._get_dones()
    env.reset_terminated, env.reset_buf = terminated, terminated | truncated
    reward = env._get_rewards()
    return reward, terminated, truncated


def ball_force(column, envs, newtons=1.0):
    def write(env):
        forces = torch.zeros(N)
        forces[envs] = newtons
        env.ball_contact.push(column, forces)
    return write


def ball_at(position, envs=slice(None)):
    def write(env):
        env.ball.data.root_pos_w[envs] = torch.tensor(position)
    return write


def nothing(env):
    pass


def term(env, name):
    return env.extras["goalkeeper_reward_terms"][name]


# --- Configuration -------------------------------------------------------------------------------------------


def test_reward_scales_follow_the_requested_ratios():
    cfg = config()
    assert cfg.rwd_scale_hit_racket == 5.0 and cfg.rwd_scale_base_crash == -5.0
    assert cfg.rwd_scale_hit_arm == cfg.rwd_scale_hit_racket / 2
    assert cfg.rwd_scale_hit_base == cfg.rwd_scale_hit_racket / 4
    assert cfg.rwd_scale_goal == -cfg.rwd_scale_hit_racket
    assert cfg.rwd_scale_self_collision == cfg.rwd_scale_base_crash / 2
    assert cfg.rwd_scale_leave_goal_area == cfg.rwd_scale_tilt == cfg.rwd_scale_base_crash
    assert cfg.rwd_scale_progress > 0 and cfg.rwd_scale_action_rate < 0 and cfg.rwd_scale_energy < 0


def test_requested_reset_and_energy_defaults():
    cfg = config()
    assert (cfg.reset_robot_on_tilt_threshold, cfg.reset_robot_on_leave_area, cfg.reset_robot_on_robot_crash,
            cfg.self_collision_triggers_reset, cfg.reset_on_goal) == (False, True, True, False, False)
    assert cfg.minimize_energy_not_power is False
    assert cfg.enabled_self_collisions is True and cfg.robot.spawn.articulation_props.enabled_self_collisions


def test_shared_settings_match_the_original_task():
    """Only the reward/reset section and self-collisions differ; everything else switches 1:1."""
    simple = literal_fields(FOLDER / "borinot_goalkeeper_simple_env_cfg.py")
    original = literal_fields(ORIGINAL / "borinot_goalkeeper_env_cfg.py")
    changed = {name for name in simple.keys() & original.keys() if simple[name] != original[name]}
    assert changed == {"enabled_self_collisions", "state_space", "rwd_scale_progress", "rwd_scale_goal",
                       "rwd_scale_leave_goal_area", "rwd_scale_action_rate", "rwd_scale_energy",
                       "reset_robot_on_tilt_threshold", "self_collision_triggers_reset"}
    assert simple.keys() - original.keys() == {
        "rwd_scale_hit_racket", "rwd_scale_hit_arm", "rwd_scale_hit_base", "rwd_scale_base_crash",
        "rwd_scale_self_collision", "rwd_scale_tilt", "minimize_energy_not_power", "reset_robot_on_leave_area"}
    assert original.keys() - simple.keys() == {
        "critic_obs_includes_rotor_omega", "remove_action_clipping", "clear_distance_m", "clear_speed_m_s", "recovery_dwell_s",
        "recovery_position_tolerance_m", "recovery_speed_m_s", "recovery_angular_speed_rad_s", "recovery_tilt_rad",
        "recovery_joint_tolerance_rad", "require_recovery_for_save", "progress_use_prediction", "rwd_scale_save",
        "rwd_scale_ball_base_collision", "rwd_scale_ball_arm_collision", "rwd_scale_ball_floor", "rwd_scale_miss",
        "rwd_scale_deadline", "rwd_scale_robot_crash", "robot_crash_body_names", "rwd_scale_intercept",
        "rwd_scale_alignment", "rwd_scale_recovery", "intercept_sigma_m", "rwd_scale_hit", "rwd_scale_centered_hit",
        "rwd_scale_body_angular_velocity", "rwd_scale_body_linear_velocity", "rwd_scale_thrust"}


def test_control_and_power_code_is_the_original():
    tree = ast.parse(SOURCE.read_text())
    shared = {t.id: ast.unparse(n.value) for n in class_node(SOURCE).body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name)}
    assert shared["_apply_action"] == "BorinotGoalkeeperEnv._apply_action"
    assert shared["_sample_power"] == "BorinotGoalkeeperEnv._sample_power"
    assert "_apply_action" not in METHODS and "_sample_power" not in METHODS
    assert any(isinstance(n, ast.ImportFrom) and n.module.endswith("borinot_goalkeeper.borinot_goalkeeper_env")
               for n in tree.body)


@pytest.mark.parametrize("mode", ["thrust_and_torque_control", "thrust_and_PD_joint_pos_control",
                                  "PI_rotors_PD_joint_pos_control"])
@pytest.mark.parametrize("flags", [(False, False, False, True), (True, True, True, True), (False, False, False, False),
                                   (True, False, True, False)])
def test_observation_sizes_match_derived_spaces(mode, flags):
    point, time, phase, params = flags
    env = make_env(control_mode=mode, obs_actor_includes_ball_plane_intersection_point=point,
                   obs_actor_includes_time_ball_collides_plane=time, obs_actor_includes_phase=phase,
                   obs_actor_includes_ball_params=params)
    obs = env._get_observations()
    assert obs["policy"].shape == (N, env.cfg.observation_space)
    assert obs["critic"].shape == (N, env.cfg.state_space)
    pi = mode == "PI_rotors_PD_joint_pos_control"
    assert env.cfg.state_space == (61 if pi else 54)
    if all(flags):
        assert env.cfg.observation_space == env.cfg.state_space
    if flags == (False, False, False, True):
        assert env.cfg.observation_space == (51 if pi else 44)
    # Rotor thrust/hover (6) and the rate-PI integral (3) follow the actions only in PI mode.
    end = 34 + 3 * flags[0] + flags[1] + (flags[0] or flags[1]) + 5 * flags[2] + env.cfg.action_space
    assert obs["policy"].shape[1] == end + 9 * pi + 2 * flags[3]
    if pi:
        pi_values = torch.cat([env._rotor_thrust / env._hover_per_rotor, env._rate_integral], -1)
        assert torch.equal(obs["policy"][:, end:end + 9], pi_values)


# --- Task: first contact and goal, each once per ball ---------------------------------------------------------


def test_racket_contact_pays_once_while_it_lasts():
    env = make_env()
    rewards = [step(env, ball_force(RACKET, [0]), ball_force(RACKET, [0]))[0][0].item() for _ in range(4)]
    assert rewards == [5.0, 0.0, 0.0, 0.0]
    assert env._touched.tolist() == [True] + [False] * (N - 1)
    assert env._episode_counts["hit_racket"][0] == 1


@pytest.mark.parametrize("order,expected", [
    ((RACKET, ARM2, BASE), 5.0), ((ARM1, RACKET, BASE), 2.5), ((BASE, RACKET, ARM2), 1.25)])
def test_only_the_first_robot_contact_pays(order, expected):
    env = make_env()
    total = sum(step(env, ball_force(column, [0]), nothing)[0][0].item() for column in order)
    assert total == expected


@pytest.mark.parametrize("columns,expected", [
    ((RACKET, ARM1, BASE), "hit_racket"), ((ARM2, BASE), "hit_arm"), ((BASE,), "hit_base")])
def test_same_substep_tie_prefers_racket_then_arm(columns, expected):
    env = make_env()
    for column in columns:
        ball_force(column, [0])(env)
    step(env, lambda e: None, lambda e: None)
    assert [name for name in ("hit_racket", "hit_arm", "hit_base") if env.extras["goalkeeper_events"][name][0]] == [expected]


def test_hit_then_goal_sum_to_zero_and_score_once():
    env = make_env()
    reward, terminated, _ = step(env, ball_force(RACKET, [0]), ball_at([-0.5, 0.0, 1.2], [0]))
    assert reward[0] == 0.0 and term(env, "task/hit_racket")[0] == 5.0 and term(env, "task/goal")[0] == -5.0
    assert env.extras["goalkeeper_outcome"][0] == SCOPE["OUTCOME_GOAL"] and env.outcome_counts[3] == 1
    assert not env._ball_active[0] and env._ball_active[1:].all()  # the scored ball is parked
    assert not terminated.any()  # reset_on_goal=False
    reward, *_ = step(env, ball_at([-0.6, 0.0, 1.2], [0]), ball_force(GOAL, [0]))
    assert reward[0] == 0.0 and env.outcome_counts[3] == 1


def test_contacts_after_a_goal_are_ignored():
    env = make_env()
    reward, *_ = step(env, ball_force(GOAL, [0]), ball_force(RACKET, [0]))
    assert reward[0] == -5.0 and not env.extras["goalkeeper_events"]["hit_racket"][0]


def test_goal_reset_is_optional_and_keeps_the_reward():
    env = make_env(reset_on_goal=True)
    reward, terminated, _ = step(env, ball_force(GOAL, [2]), nothing)
    assert terminated.tolist() == [False, False, True] + [False] * (N - 3) and reward[2] == -5.0


@pytest.mark.parametrize("position", [[-0.5, 1.4, 1.2], [3.2, 0.0, 1.2], [1.0, 3.1, 1.2], [1.0, 0.0, 4.6]])
def test_wide_or_escaped_ball_ends_the_shot_without_reward(position):
    env = make_env()
    reward, *_ = step(env, ball_at(position, [0]), nothing)
    assert reward[0] == 0.0 and env.extras["goalkeeper_outcome"][0] == SCOPE["OUTCOME_SAVE"]
    assert env.extras["goalkeeper_shot_finished"].tolist() == [True] + [False] * (N - 1)
    assert env._episode_counts["saves"][0] == 1 and env._episode_counts["goals"][0] == 0


def test_a_ball_that_went_wide_cannot_score_later():
    env = make_env()
    reward, *_ = step(env, ball_at([-0.5, 1.4, 1.2], [0]), ball_force(GOAL, [0]))
    assert reward[0] == 0.0 and env.extras["goalkeeper_outcome"][0] == SCOPE["OUTCOME_SAVE"]


def test_shot_timeout_ends_without_reward_and_unresolved_balls_stay_live():
    env = make_env()
    env._shot_steps[0] = round(env.cfg.shot_timeout_s / env.step_dt) - 1
    reward, *_ = step(env)
    assert reward[0] == 0.0 and env.extras["goalkeeper_outcome"].tolist() == [7] + [0] * (N - 1)
    assert not env._ball_active[0] and env._ball_active[1:].all()


def test_inactive_ball_contacts_do_not_count():
    env = make_env()
    env._ball_active[0] = False
    reward, *_ = step(env, ball_force(RACKET, [0]), ball_force(GOAL, [0]))
    assert reward[0] == 0.0 and not env._touched[0]


def test_ball_on_base_is_a_hit_not_a_crash():
    env = make_env()
    reward, terminated, _ = step(env, ball_force(BASE, [0]), nothing)
    assert reward[0] == 1.25 and not env.extras["goalkeeper_events"]["base_crash"][0] and not terminated[0]


def test_progress_is_paid_until_the_first_contact():
    env = make_env(rwd_scale_progress=2.5)
    ball_at([1.0, 0.5, 1.13])(env)

    def move_racket(y):
        def write(e):
            e.robot.data.body_pos_w[:, 1, 1] = y
        return write

    reward, *_ = step(env, move_racket(0.10), move_racket(0.25))
    torch.testing.assert_close(reward, torch.full((N,), 2.5 * 0.25))  # yz distance 0.50 -> 0.25 m
    reward, *_ = step(env, nothing, nothing)
    torch.testing.assert_close(reward, torch.zeros(N))  # a still racket earns nothing, even with a moving ball
    reward, *_ = step(env, move_racket(0.35), ball_force(RACKET, [0]))
    assert reward[0] == pytest.approx(2.5 * 0.10 + 5.0)  # the contact interval keeps its progress
    reward, *_ = step(env, move_racket(0.45), nothing)
    assert reward[0] == 0.0 and reward[1] == pytest.approx(2.5 * 0.10)  # touched ball: no more progress


# --- Integrity ------------------------------------------------------------------------------------------------


def integrity_fixture(env):
    """Env 0 base-floor, 1 base-goal, 2 base-calf, 3 base-racket, 4 racket-thigh, 5 outside area, 6 tilt, 7 none."""
    for env_id, column in ((0, FLOOR), (1, GOAL_BODY), (2, LINK2), (3, PADDLE)):
        forces = torch.zeros(N)
        forces[env_id] = 2.0
        env.base_contact.data.force_matrix_w[:, 0, column, 2] += forces
    env.paddle_contact.data.force_matrix_w[4, 0, 0, 2] = 2.0
    env.robot.data.root_pos_w[5, 0] = 0.95
    env.robot.data.root_quat_w[6] = torch.tensor([math.cos(math.radians(40)), math.sin(math.radians(40)), 0.0, 0.0])


def test_integrity_events_penalties_and_default_resets():
    env = make_env()
    reward, terminated, truncated = step(env, integrity_fixture, nothing)
    events = env.extras["goalkeeper_events"]
    assert events["base_crash"].tolist() == [True, True] + [False] * 6
    assert events["self_collision"].tolist() == [False, False, True, True, True, False, False, False]
    assert events["leave_area"].tolist() == [False] * 5 + [True, False, False]
    assert events["tilt"].tolist() == [False] * 6 + [True, False]
    assert reward.tolist() == [-5.0, -5.0, -2.5, -2.5, -2.5, -5.0, -5.0, 0.0]
    assert term(env, "integrity/tilt").tolist() == [0.0] * 6 + [-5.0, 0.0]
    assert terminated.tolist() == [True, True, False, False, False, True, False, False]
    assert not truncated.any()


def test_reset_switches_change_only_resets():
    env = make_env(reset_robot_on_robot_crash=False, self_collision_triggers_reset=True,
                   reset_robot_on_leave_area=False, reset_robot_on_tilt_threshold=True)
    reward, terminated, _ = step(env, integrity_fixture, nothing)
    assert reward.tolist() == [-5.0, -5.0, -2.5, -2.5, -2.5, -5.0, -5.0, 0.0]
    assert terminated.tolist() == [False, False, True, True, True, False, True, False]


def test_integrity_penalties_repeat_while_the_event_lasts():
    env = make_env()
    rewards = [step(env, integrity_fixture, integrity_fixture)[0][[3, 6]].tolist() for _ in range(3)]
    assert rewards == [[-2.5, -5.0]] * 3  # self-contact and tilt, neither resets by default


def test_simultaneous_events_add_up_without_priority():
    env = make_env()

    def everything(e):
        ball_force(RACKET, [0])(e)
        integrity_fixture(e)
        e.robot.data.root_pos_w[0, 0] = 0.95
        e.paddle_contact.data.force_matrix_w[0, 0, 0, 2] = 2.0
    reward, *_ = step(env, everything, ball_at([-0.5, 0.0, 1.2], [0]))
    assert reward[0] == pytest.approx(5.0 - 5.0 - 5.0 - 2.5 - 5.0)
    total = sum(value for value in env.extras["goalkeeper_reward_terms"].values())
    torch.testing.assert_close(total, reward)
    assert {name.split("/")[0] for name in env.extras["goalkeeper_reward_terms"]} == {"task", "integrity", "desirability"}


# --- Desirability ---------------------------------------------------------------------------------------------


def test_action_rate_uses_group_means_of_normalized_commands_and_policy_dt():
    env = make_env(rwd_scale_action_rate=-5.0)
    env._actuator_commands[:, :6] = 0.1

    def command(e):
        e._actuator_commands[:, :6] = torch.tensor([0.4, 0.1, 0.1, 0.1, 0.1, 0.1])
        e.robot.data.applied_torque[:] = torch.tensor([2.7, 0.0])
    reward, *_ = step(env, command, nothing)
    signal = 0.3**2 / 6 + (2.7 / 5.4)**2 / 2
    torch.testing.assert_close(reward, torch.full((N,), -5.0 * signal * env.step_dt))


def test_each_limit_term_is_clipped_to_the_max_penalty():
    env = make_env()
    env._action_limit_costs[:] = torch.linspace(0., 30., N)[:, None] * torch.tensor([1., .5, 2.])
    terms = env._desirability_rewards()
    for i, name in enumerate(("thrust_limits", "torque_limits", "action_limits")):
        expected = (getattr(env.cfg, f"rwd_scale_{name}") * env._action_limit_costs[:, i]).clamp(-10., 10.)
        torch.testing.assert_close(terms[name], expected)
    assert terms["action_limits"].min() == -10.


def write_power(index):
    """Different rotor thrusts and joint motion in every physics step."""
    def write(env):
        env._rotor_thrust[:] = 4.0 + index + torch.arange(N)[:, None] * 0.1
        env.robot.data.applied_torque[:] = torch.tensor([0.5 * index, -1.0])
        env.robot.data.joint_vel[:] = torch.tensor([2.0, 1.0 + index])
    return write


def expected_power(index):
    thrust = (4.0 + index + torch.arange(N)[:, None] * 0.1).expand(N, 6)
    cfg = config()
    rotor = (cfg.km * (thrust / cfg.kf).sqrt()**3).sum(-1) / cfg.p_el_motor_efficiency
    torque, velocity = torch.tensor([0.5 * index, -1.0]), torch.tensor([2.0, 1.0 + index])
    return rotor + (torque * velocity).sum() + cfg.arm_copper_loss_coefficient * torque.square().sum()


@pytest.mark.parametrize("physics_hz", [200, 400])
def test_power_mode_keeps_the_original_per_watt_scale(physics_hz):
    env = make_env(rwd_scale_energy=-2.5e-5, physics_hz=physics_hz)
    substeps = [write_power(i) for i in range(env.cfg.decimation)]
    reward, *_ = step(env, *substeps)
    mean_power = sum(expected_power(i) for i in range(env.cfg.decimation)) / env.cfg.decimation
    torch.testing.assert_close(reward, -2.5e-5 * mean_power)


@pytest.mark.parametrize("physics_hz", [200, 400])
def test_energy_mode_pays_each_shot_window_when_the_shot_ends(physics_hz):
    power_env = make_env(rwd_scale_energy=-2.5e-5, physics_hz=physics_hz)
    energy_env = make_env(rwd_scale_energy=-2.5e-5, physics_hz=physics_hz, minimize_energy_not_power=True)
    ends = {2: [0], 4: [1], 5: [0]}  # env 0 ends two shots (relaunched at step 3), env 1 one shot
    per_step, paid = [], []
    for policy_step in range(7):
        substeps = [write_power(policy_step * power_env.cfg.decimation + i) for i in range(power_env.cfg.decimation)]
        step(power_env, *substeps)
        per_step.append(term(power_env, "desirability/energy").clone())
        if policy_step in ends:  # the ball passes the goal plane in the last substep
            substeps[-1] = lambda e, write=substeps[-1], envs=ends[policy_step]: (
                write(e), ball_at([-0.5, 0.0, 1.2], envs)(e))
        step(energy_env, *substeps)
        assert energy_env.extras["goalkeeper_shot_finished"].nonzero().flatten().tolist() == ends.get(policy_step, [])
        paid.append(term(energy_env, "desirability/energy").clone())
        if policy_step == 2:
            energy_env._next_launch_steps[0] = 1
    per_step, paid = torch.stack(per_step), torch.stack(paid)
    assert (paid[:, 0] != 0).nonzero().flatten().tolist() == [2, 5]
    assert (paid[:, 1] != 0).nonzero().flatten().tolist() == [4]
    assert not paid[:, 2:].any()
    # Each payment is the sum of the per-step terms since the previous payment (or the episode start).
    torch.testing.assert_close(paid[2, 0], per_step[:3, 0].sum())
    torch.testing.assert_close(paid[5, 0], per_step[3:6, 0].sum())
    torch.testing.assert_close(paid[4, 1], per_step[:5, 1].sum())
    unpaid = -2.5e-5 * energy_env._unpaid_energy_j / energy_env.step_dt
    torch.testing.assert_close(paid.sum(0) + unpaid, per_step.sum(0))


@pytest.mark.parametrize("physics_hz", [200, 400])
@pytest.mark.parametrize("ending", ["time_limit", "finite_horizon", "crash"])
def test_energy_mode_pays_the_unfinished_shot_only_on_termination(physics_hz, ending):
    power_env = make_env(rwd_scale_energy=-2.5e-5, physics_hz=physics_hz, episode_length_s=0.05)
    energy_env = make_env(rwd_scale_energy=-2.5e-5, physics_hz=physics_hz, episode_length_s=0.05,
                          minimize_energy_not_power=True, is_finite_horizon=ending == "finite_horizon")
    steps = power_env.max_episode_length
    totals, energies = torch.zeros(N), []
    for policy_step in range(steps):
        substeps = [write_power(policy_step * power_env.cfg.decimation + i) for i in range(power_env.cfg.decimation)]
        if ending == "crash" and policy_step == steps - 2:
            substeps[-1] = lambda e, write=substeps[-1]: (write(e), e.base_contact.push(FLOOR, torch.full((N,), 2.0)))
        step(power_env, *substeps)
        totals += term(power_env, "desirability/energy")
        _, terminated, truncated = step(energy_env, *substeps)
        energies.append(term(energy_env, "desirability/energy").clone())
        if terminated.all() or truncated.all():
            break
    last = len(energies) - 1
    assert last == (steps - 2 if ending == "crash" else steps - 1)
    assert all(not energy.any() for energy in energies[:last])
    physics_steps = (last + 1) * power_env.cfg.decimation
    joules = sum(expected_power(i) for i in range(physics_steps)) / physics_hz
    torch.testing.assert_close(energy_env._episode_energy_j, joules)
    if ending == "time_limit":  # a cut, not an end: the learner's bootstrap covers the unfinished shot
        assert truncated.all() and not energies[last].any()
        torch.testing.assert_close(energy_env._unpaid_energy_j, joules)
    else:
        assert terminated.all() and not energy_env._unpaid_energy_j.any()
        torch.testing.assert_close(energies[last], totals)
        torch.testing.assert_close(energies[last], -2.5e-5 * joules / energy_env.step_dt)


# --- Shots, logging and curriculum ---------------------------------------------------------------------------


def test_curriculum_counts_shots_without_goal_as_saves():
    env = make_env(curriculum=True, curriculum_min_shots=4, curriculum_window_shots=8, curriculum_min_steps=1,
                   curriculum_success_threshold=0.7)
    env._curriculum_window = torch.zeros(8)
    env.common_step_counter = 10
    env._outcome[:] = torch.tensor([7, 7, 7, 7, 7, 3, 0, 0])
    env._record_shots(torch.arange(6))
    assert env._episode_counts["saves"].tolist() == [1] * 5 + [0] * 3
    assert env._episode_counts["goals"].tolist() == [0] * 5 + [1, 0, 0]
    assert env.outcome_counts[7] == 5 and env.outcome_counts[3] == 1
    assert env.shot_level == 1  # 5/6 = 0.83 > 0.7 after at least four shots


def test_episode_log_groups_terms_by_category():
    env = make_env()
    step(env, ball_force(RACKET, [0]), integrity_fixture)
    env._update_curriculum(torch.arange(N))
    log = env.extras["log"]
    assert log["Episode_Reward/task"] == pytest.approx(5.0 / N)
    assert log["Episode_Reward/integrity"] == pytest.approx(-27.5 / N)
    assert log["Episode_Reward/integrity/tilt"] == pytest.approx(-5.0 / N)
    assert log["Episode_Reward/desirability"] == 0.0
    assert log["Episode/hit_racket"] == pytest.approx(1 / N)
    assert {"Episode/return", "Episode/energy_j", "Goalkeeper/cumulative_save_fraction",
            "Episode_Reward/task/hit_racket", "Episode_Reward/integrity/self_collision",
            "Episode_Reward/desirability/energy", "Power/total_estimate_mean_w"} <= log.keys()


def test_episode_log_reports_the_mean_actuator_delay_of_completed_episodes():
    env = make_env()
    step(env, ball_force(RACKET, [0]), integrity_fixture)
    env._actuator_delay.seconds = torch.linspace(0., .04, N)
    env._episode_steps[0] = 0  # an initial reset is not a completed episode
    env._update_curriculum(torch.arange(N))
    torch.testing.assert_close(env.extras["log"]["Episode/actuator_delay_s"], env._actuator_delay.seconds[1:].mean())


def test_launch_resets_only_per_shot_state():
    env = make_env()
    step(env, ball_force(RACKET, [0, 1]), ball_at([-0.5, 0.0, 1.2], [0]))
    assert env._touched[:2].all() and not env._ball_active[0]
    env._launch_ball(torch.tensor([0]))
    assert env._ball_active[0] and not env._touched[0] and not env._ball_done[0] and env._shot_steps[0] == 0
    assert env._touched[1]  # another environment's live, touched ball is untouched by this launch
