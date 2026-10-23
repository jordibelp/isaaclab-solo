# SPDX-License-Identifier: BSD-3-Clause
"""CPU oracles for the real task math, scheduler, rewards and curriculum."""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cmd_vel_math", FOLDER / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)


def method(name, **extra):
    node = next(n for n in ast.parse((FOLDER / "borinot_cmd_vel_env.py").read_text()).body if isinstance(n, ast.ClassDef))
    function = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"torch": torch, "math": math, **vars(task_math), **extra}
    exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), scope)
    return scope[name]


def quaternion(roll, pitch, yaw):
    cr, cp, cy = [math.cos(value / 2) for value in (roll, pitch, yaw)]
    sr, sp, sy = [math.sin(value / 2) for value in (roll, pitch, yaw)]
    return torch.tensor([[cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                          cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]], dtype=torch.float64)


@pytest.mark.parametrize("yaw", [0.0, 0.5, math.pi/2, -math.pi/2, math.pi])
@pytest.mark.parametrize("roll,pitch", [(0., 0.), (.6, -.8), (-.9, .7)])
def test_heading_matches_yaw_not_tilt_or_rotation_log(roll, pitch, yaw):
    q = quaternion(roll, pitch, yaw)
    r = task_math.heading_rotation(q)
    c, s = math.cos(yaw), math.sin(yaw)
    expected = torch.tensor([[[c, -s, 0], [s, c, 0], [0, 0, 1]]], dtype=q.dtype)
    torch.testing.assert_close(r, expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(r.transpose(1, 2) @ r, torch.eye(3, dtype=q.dtype)[None])
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(1, dtype=q.dtype))
    torch.testing.assert_close(task_math.heading_rotation(-q), r)
    command = torch.tensor([[.3, -.7, 1.1, -.4]], dtype=q.dtype)
    velocity_w = torch.tensor([[c*.3+s*.7, s*.3-c*.7, 1.1]], dtype=q.dtype)
    angular_w = torch.tensor([[10., -20., -.4]], dtype=q.dtype)
    for kernel in task_math.tracking_kernels(command, velocity_w, angular_w, q, .5, .5):
        torch.testing.assert_close(kernel, torch.ones(1, dtype=q.dtype))


def test_vertical_velocity_cannot_satisfy_planar_command_and_yaw_is_world_z():
    command = torch.tensor([[1., 0., 2., .5]], dtype=torch.float64)
    xyz, yaw, xy = task_math.tracking_kernels(command, torch.tensor([[0., 0., 2.]], dtype=command.dtype),
        torch.tensor([[4., -3., 1.]], dtype=command.dtype), quaternion(.7, -.7, .8), .5, .5)
    torch.testing.assert_close(xyz, torch.tensor([math.exp(-4)], dtype=command.dtype))
    torch.testing.assert_close(xy, xyz)
    torch.testing.assert_close(yaw, torch.tensor([math.exp(-1)], dtype=command.dtype))


@pytest.mark.parametrize("pitch", [math.pi/2, -math.pi/2])
def test_heading_singularity_is_finite_and_deterministic(pitch):
    torch.testing.assert_close(task_math.heading_rotation(quaternion(0, pitch, 0)), torch.eye(3, dtype=torch.float64)[None])


def test_pushes_are_area_uniform_independent_signed_and_bounded():
    torch.manual_seed(7)
    limits = torch.full((50000,), 8.0)
    force, point = task_math.sample_vertical_pushes(limits, .12, .03)
    assert force.abs().max() <= 8 and .49 < (force > 0).float().mean() < .51
    assert abs(float(force.abs().mean()) - 4) < .04
    r2 = point[:, :2].square().sum(-1)
    assert r2.max() <= .12**2 + 1e-8
    assert abs(float(r2.mean()) - .12**2/2) < .0001
    assert point[:, :2].mean(0).abs().max() < .001
    torch.testing.assert_close(point[:, 2], torch.where(force >= 0, -.03, .03))
    force, _ = task_math.sample_vertical_pushes(torch.zeros(9), .12, .03)
    assert not force.any()


def scheduler_env():
    return SimpleNamespace(device="cpu", step_dt=.1,
        cfg=SimpleNamespace(base_push_application_radius_m=.12, base_push_application_half_height_m=.03,
            base_push_duration_range_s=(.2, .2), base_push_interval_range_s=(.5, .5)),
        _force_limits=torch.tensor([0., 5., 8.]), _episode_stage=torch.tensor([0, 1, 2]),
        _base_push_steps_until_next=torch.tensor([0, 0, 0]), _base_push_steps_left=torch.zeros(3, dtype=torch.long),
        _base_push_force_z_w=torch.zeros(3), _base_push_points_b=torch.zeros(3, 3))


def test_push_pulses_have_exact_duration_no_stale_force_and_start_to_start_clock():
    env = scheduler_env()
    forces = []
    for _ in range(11):
        method("_update_pushes")(env)
        forces.append(env._base_push_force_z_w.clone())
    assert all(value[0] == 0 for value in forces)
    for start in (0, 5):
        assert forces[start][1:].abs().min() > 0
        torch.testing.assert_close(forces[start], forces[start+1])
        assert all(not forces[t].any() for t in range(start+2, start+5))
    assert forces[10][1:].abs().min() > 0
    assert all(value[1].abs() <= 5 and value[2].abs() <= 8 for value in forces)


def curriculum_env():
    return SimpleNamespace(device="cpu", extras={}, step_dt=.1, common_step_counter=100,
        cfg=SimpleNamespace(episode_length_s=10., curriculum=True, curriculum_min_episodes=2,
            curriculum_min_steps=100, curriculum_tracking_threshold=.7),
        force_level=0, _force_limits=torch.tensor([0., 5., 8.]),
        _episode_steps=torch.tensor([100, 100, 100, 0]), _episode_stage=torch.tensor([0, 0, 0, 0]),
        _episode_xy_integral=torch.tensor([8., 9., 10., 0.]), _episode_energy_j=torch.ones(4),
        _episode_terms={"track": torch.ones(4)}, _curriculum_window=torch.zeros(3),
        _window_cursor=0, _stage_episodes=0, _stage_start_step=0, curriculum_stage_pure_episodes=0)


def test_curriculum_promotes_once_then_ignores_old_stage_episodes():
    env = curriculum_env()
    update = method("_update_curriculum")
    update(env, torch.tensor([0, 1, 3]))
    assert env.force_level == 1 and env._stage_episodes == 0 and not env._curriculum_window.any()
    assert env.extras["log"]["Curriculum/stage_pure_xy_tracking"] == pytest.approx(.85)
    env.common_step_counter = 200
    update(env, torch.tensor([2]))
    assert env.force_level == 1 and env._stage_episodes == 0
    assert env.curriculum_stage_pure_episodes == 0


@pytest.mark.parametrize("case", ["short_flight", "dwell", "count", "disabled", "threshold"])
def test_curriculum_does_not_promote_on_insufficient_evidence(case):
    env = curriculum_env()
    if case == "short_flight":
        env._episode_steps[:] = 5
        env._episode_xy_integral[:] = .5  # perfect for 0.5 s is NOT a successful 10 s flight
    elif case == "dwell":
        env.common_step_counter = 99
    elif case == "count":
        env.cfg.curriculum_min_episodes = 3
    elif case == "disabled":
        env.cfg.curriculum = False
    else:
        env._episode_xy_integral[:] = 7.0  # strictly ABOVE 0.7
    method("_update_curriculum")(env, torch.tensor([0, 1]))
    assert env.force_level == 0


def test_curriculum_ring_retains_newest_episodes_when_batch_exceeds_capacity():
    env = curriculum_env()
    env.cfg.curriculum = False
    env._episode_steps[:] = 100
    env._episode_xy_integral[:] = torch.tensor([1., 2., 3., 4.])
    method("_update_curriculum")(env, torch.arange(4))
    torch.testing.assert_close(env._curriculum_window, torch.tensor([.2, .3, .4]))
    assert env.extras["log"]["Curriculum/window_completed_episodes"] == 3


def reward_env():
    return SimpleNamespace(step_dt=.01, physics_dt=.005, joint_ids=[0, 1], extras={},
        cfg=SimpleNamespace(tracking_lin_std=.5, tracking_ang_std=.5, effort_lim_sim=2.,
            rwd_scale_lin_vel=1.5, rwd_scale_ang_vel=.5, rwd_scale_action_rate=0., rwd_scale_energy=0.),
        robot=SimpleNamespace(data=SimpleNamespace(root_lin_vel_w=torch.zeros(2, 3),
            root_ang_vel_w=torch.zeros(2, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]).expand(2, -1),
            applied_torque=torch.zeros(2, 2))), _commands=torch.zeros(2, 4),
        _actuator_commands=torch.zeros(2, 8), _previous_actuator_commands=torch.zeros(2, 8),
        _power_sum_w=torch.tensor([[20., 2., -4., 6.], [30., 4., 8., 10.]]), _power_samples=2,
        _episode_terms={name: torch.zeros(2) for name in ("track_lin_vel_xyz_exp", "track_ang_vel_z_exp", "action_rate", "energy")},
        _episode_xy_integral=torch.zeros(2), _episode_energy_j=torch.zeros(2))


def test_exact_reward_ceiling_and_disabled_costs():
    env = reward_env()
    reward = method("_get_rewards")(env)
    torch.testing.assert_close(reward, torch.full((2,), .02))
    torch.testing.assert_close(env._episode_energy_j, torch.tensor([.12, .26]))
    assert not env._episode_terms["action_rate"].any() and not env._episode_terms["energy"].any()


def test_power_is_mean_watts_and_action_rate_is_group_means_times_dt():
    env = reward_env()
    env.cfg.rwd_scale_energy, env.cfg.rwd_scale_action_rate = -.1, -2.
    env._actuator_commands[:, :6] = 1.
    env.robot.data.applied_torque[:] = 4.  # one full normalized joint span
    reward = method("_get_rewards")(env)
    expected_cost = -.1 * torch.tensor([12., 26.]) - 2 * 2 * .01
    torch.testing.assert_close(reward, .02 + expected_cost, rtol=0, atol=3e-7)


def test_time_limit_uses_task_clock_not_randomized_runner_clock():
    env = SimpleNamespace(cfg=SimpleNamespace(flight_height_range_m=(.15, 20.), flight_xy_limit_m=25.,
        reset_robot_on_tilt_threshold=False), _sample_power=lambda: None,
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0, 1.5], [0., 0, .1]]))),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)), _episode_steps=torch.tensor([1000, 3]),
        max_episode_length=1000, episode_length_buf=torch.tensor([5, 999]))
    failed, timeout = method("_get_dones")(env)
    assert failed.tolist() == [False, True] and timeout.tolist() == [True, False]


def test_command_changes_are_visible_before_the_policy_acts_and_observation_reads_are_stable():
    data = SimpleNamespace(root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
        root_quat_w=torch.tensor([[1., 0, 0, 0]]).expand(2, -1), joint_pos=torch.zeros(2, 2), joint_vel=torch.zeros(2, 2))
    env = SimpleNamespace(cfg=SimpleNamespace(obs_include_base_lin_acc=False, control_mode="thrust_and_torque_control"),
        robot=SimpleNamespace(data=data), joint_ids=[0, 1], actions=torch.zeros(2, 8),
        _gravity=torch.tensor([[0., 0, -1.]]).expand(2, -1), _commands=torch.zeros(2, 4),
        _command_steps_left=torch.tensor([0, 2]))
    calls = []
    def resample(ids):
        calls.append(ids.clone())
        env._commands[ids] = 1.
        env._command_steps_left[ids] = 500
    env._resample_commands = resample
    get_obs = method("_get_observations", quat_apply_inverse=lambda q, v: v,
                     matrix_from_quat=lambda q: torch.eye(3).expand(len(q), -1, -1))
    first = get_obs(env)["policy"]
    second = get_obs(env)["policy"]
    assert len(calls) == 1 and calls[0].tolist() == [0]
    assert first.shape == (2, 34) and first[0, -4:].tolist() == [1.] * 4
    assert not first[1, -4:].any()
    torch.testing.assert_close(first, second)
