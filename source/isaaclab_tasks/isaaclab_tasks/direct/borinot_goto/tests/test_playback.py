# SPDX-License-Identifier: BSD-3-Clause
"""CPU checks for the live go-to player: goal ownership, deadlines, holds, resets and views."""
import ast
import math
import time
from pathlib import Path

import pytest
import torch

from test_goto import FOLDER, config, environment_class, fake_env, inverse, task_math


def load_definitions(path, scope, constants=False):
    kinds = (ast.ClassDef, ast.FunctionDef) + ((ast.Assign,) if constants else ())
    nodes = [n for n in ast.parse(Path(path).read_text()).body if isinstance(n, kinds)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


CMD_UI = load_definitions(FOLDER.parent / "borinot_cmd_vel/playback_ui.py", {"math": math})
UI = load_definitions(FOLDER / "playback_ui.py", {
    "math": math, "time": time, "quat_apply_inverse": inverse, "camera_pose": CMD_UI["camera_pose"],
    "training_deadline_interval": task_math.training_deadline_interval}, constants=True)
GOAL = (1.0, -0.5, 2.0)


def play_env(goal=GOAL, deadline_s=None, reset_at_deadline=False, goal_relative=False, **overrides):
    """Real go-to and play methods over a fake flight plant whose reset puts robots at home."""
    goto = environment_class()
    flight = goto.__mro__[1]

    def flight_init(self, cfg, **kwargs):
        self.__dict__.update(vars(fake_env(cfg)))
        self.robot._ALL_INDICES = torch.arange(self.num_envs)
        self.reset_calls = []

    def flight_reset(self, env_ids):
        self.reset_calls.append(env_ids.tolist())
        data = self.robot.data
        data.root_pos_w[env_ids] = data.root_pos_w.new_tensor(self.cfg.robot_home) + self.scene.env_origins[env_ids]
        if self.cfg.init_base_attitude_points_to_goal:
            root = data.root_pos_w.new_zeros((len(env_ids), 13))
            root[:, :3] = data.root_pos_w[env_ids]
            self._sample_reset_attitude(root, env_ids)
            data.root_quat_w[env_ids] = root[:, 3:7]
        data.root_lin_vel_w[env_ids] = 0.0
        self._episode_steps[env_ids] = 0

    flight.__init__, flight._reset_idx = flight_init, flight_reset
    scope = load_definitions(FOLDER / "playback.py", {
        "torch": torch, "math": math, "BorinotGoToEnv": goto, "stage_settings": task_math.stage_settings,
        "training_deadline_interval": task_math.training_deadline_interval})
    cfg = config(goal_curriculum_start_level=9, **overrides)
    env = scope["BorinotGoToPlayEnv"](cfg, goal=goal, deadline_s=deadline_s,
                                    reset_at_deadline=reset_at_deadline, goal_relative=goal_relative)
    env._reset_idx(None)  # the vector wrapper resets once before the first observation
    return env


def auto_steps(cfg, distance, difficulty=5, hold=False):
    """Independent oracle: middle (goal) or start (hold) of [max(low, floor), max(high, floor)], in 0.01 s ticks."""
    speed, accel = (v[0] if len(v) == 1 else v[difficulty]
                    for v in (cfg.goal_reference_speed_m_s, cfg.goal_reference_accel_m_s2))
    floor = max(distance / speed if speed else 0., 2 * math.sqrt(distance / accel) if accel else 0.)
    low, high = cfg.goal_deadline_range_s[difficulty]
    seconds = max(low, floor) if hold else (max(low, floor) + max(high, floor)) / 2
    return math.ceil(round(seconds / 0.01, 6))


def test_live_goal_replaces_sampled_goal_at_every_reset_and_restarts_the_goal_clock():
    env = play_env()
    origin, home = env.scene.env_origins, torch.tensor(env.cfg.robot_home, dtype=torch.float64)
    torch.testing.assert_close(env._goal_position_w, origin + torch.tensor(GOAL, dtype=torch.float64))
    torch.testing.assert_close(env._segment_start_w, origin + home)
    distance = (torch.tensor(GOAL, dtype=torch.float64) - home).norm()
    assert env._goal_deadline_steps.tolist() == [auto_steps(env.cfg, float(distance))] * 2
    assert not env._episode_steps.any() and not env._segment_hold.any()
    for buffer in (env._initial_goal_distance, env._previous_goal_distance, env._minimum_goal_distance):
        torch.testing.assert_close(buffer, distance.expand(2).to(buffer.dtype))
    assert env._first_arrival_s.tolist() == [-1.0, -1.0]
    assert env.reset_at_deadline is False and env.play_difficulty == 5


@pytest.mark.parametrize("relative", [False, True])
def test_goal_facing_reset_uses_the_live_goal_but_new_inflight_goals_do_not_teleport(relative):
    env = play_env(goal=(1., -.5, 2.), goal_relative=relative,
                   init_base_attitude_points_to_goal=True, reset_rpy_noise_degs=(40., 50., 170.))

    def check_facing():
        from test_goto import matrix
        rotation = matrix(env.robot.data.root_quat_w)
        delta = env._goal_position_w - env.robot.data.root_pos_w
        delta[:, 2] = 0
        torch.testing.assert_close(rotation[:, :, 0], delta / delta.norm(dim=-1, keepdim=True))
        torch.testing.assert_close(rotation[:, :, 2], torch.tensor([[0., 0., 1.]], dtype=delta.dtype).repeat(2, 1))

    check_facing()
    q = env.robot.data.root_quat_w.clone()
    position = env.robot.data.root_pos_w.clone()
    env.set_play_goal((-2., .8, 1.), 2.0, relative=relative)
    assert torch.equal(env.robot.data.root_quat_w, q) and torch.equal(env.robot.data.root_pos_w, position)
    target = env._goal_position_w.clone()
    env._reset_idx(torch.tensor([1]))
    assert torch.equal(env._goal_position_w, target)
    assert torch.equal(env.robot.data.root_quat_w[0], q[0])
    # The untouched clone keeps its in-flight attitude; only clone 1 reorients.
    expected = task_math.goal_facing_quat(env.robot.data.root_pos_w[1:], target[1:])
    torch.testing.assert_close(env.robot.data.root_quat_w[1:], expected)


@pytest.mark.parametrize("distance", [0.0, 0.5, 2.0, 2.25, 3.0, 6.0, 7.8, 8.544])
def test_auto_goal_deadline_is_the_middle_and_hold_deadline_the_start_of_the_training_interval(distance):
    env = play_env()
    distance_t = torch.tensor([distance], dtype=torch.float64)
    assert env._play_deadline_steps(distance_t, hold=False).tolist() == [auto_steps(env.cfg, distance)]
    assert env._play_deadline_steps(distance_t, hold=True).tolist() == [auto_steps(env.cfg, distance, hold=True)]
    env.play_deadline_s = 4.0  # manual inference time applies to goal and hold segments
    assert env._play_deadline_steps(distance_t, hold=False).tolist() == [400]
    assert env._play_deadline_steps(distance_t, hold=True).tolist() == [400]


def test_training_interval_contains_every_sampled_deadline_for_its_distance():
    cfg = config()
    torch.manual_seed(5)
    for difficulty in range(len(cfg.goal_half_extents_m)):
        start = torch.zeros(20000, 3, dtype=torch.float64)
        goals, steps = task_math.sample_goals(start, cfg, difficulty, 0.01)
        low, high = task_math.training_deadline_interval(cfg, difficulty, goals.norm(dim=-1))
        seconds = steps.double() * 0.01
        assert (seconds >= low - 1e-9).all() and (seconds <= high + 0.01 + 1e-9).all()
        # Both ends are reached: the interval is not wider than the sampler.
        assert ((seconds - low) < 0.011).any() and ((high - seconds) < 0.011).any()


def test_sampler_refactor_is_bit_exact_with_the_previous_inline_floor():
    cfg = config(goal_reference_speed_m_s=(.4,.7,1.2,2.,3.,3.125),
                 goal_reference_accel_m_s2=(.8,1.5,3.,5.,8.,9.))
    start = torch.randn(5000, 3)
    for difficulty in range(len(cfg.goal_half_extents_m)):
        torch.manual_seed(difficulty)
        goals, steps = task_math.sample_goals(start, cfg, difficulty, 0.01)
        torch.manual_seed(difficulty)
        extent = start.new_tensor(cfg.goal_half_extents_m[difficulty])
        offset = (2 * torch.rand(5000, 3) - 1) * extent
        distance = offset.norm(dim=-1)
        low, high = cfg.goal_deadline_range_s[difficulty]
        requested = low + (high - low) * torch.rand(5000)
        floor = torch.maximum(distance / cfg.goal_reference_speed_m_s[difficulty],
                              2 * torch.sqrt(distance / cfg.goal_reference_accel_m_s2[difficulty]))
        assert torch.equal(goals, start + offset)
        assert torch.equal(steps, torch.ceil(torch.maximum(requested, floor) / 0.01).long().clamp_min(1))


@pytest.mark.parametrize("deadline,expected", [(1.5, 150), (1.75, 175), (2.0, 200), (0.004, 1), (1.231, 124)])
def test_new_goal_starts_from_the_current_state_with_a_fixed_deadline(deadline, expected):
    env = play_env()
    env.robot.data.root_pos_w += torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)  # moved during flight
    moved = env.robot.data.root_pos_w.clone()
    env._episode_steps[:] = 37
    env._goal_failed[1] = env._goal_success[0] = True
    env.set_play_goal((-2.0, 1.0, 1.0), deadline)
    origin = env.scene.env_origins
    torch.testing.assert_close(env._goal_position_w, origin + torch.tensor([-2.0, 1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(env._segment_start_w, moved)
    assert env._goal_deadline_steps.tolist() == [expected] * 2 and not env._episode_steps.any()
    assert not (env._goal_failed | env._goal_success | env._goal_finished | env._goal_deadline_reached).any()
    assert env.reset_calls == [[0, 1]]  # no teleport: only the initial wrapper reset
    obs = env._get_observations()["policy"]
    torch.testing.assert_close(env._commands[:, :3], inverse(env.robot.data.root_quat_w,
                                                          env._goal_position_w - env.robot.data.root_pos_w))
    assert env._commands[:, 3].tolist() == [1.0, 1.0]
    torch.testing.assert_close(obs[:, -1], torch.full((2,), expected * 0.01, dtype=obs.dtype))


@pytest.mark.parametrize("goal,deadline", [
    ((1.0, 2.0), None), ((0.0, math.nan, 1.5), None), ((0.0, 0.0, 0.15), None), ((0.0, 0.0, 20.0), None),
    ((25.0, 0.0, 1.5), None), ((0.0, -25.0, 1.5), None), ((0.0, 0.0, 1.5), 0.0), ((0.0, 0.0, 1.5), math.inf)])
def test_invalid_goal_or_deadline_keeps_the_previous_goal_and_segment(goal, deadline):
    env = play_env()
    before = [b.clone() for b in (env.play_goal, env._goal_position_w, env._goal_deadline_steps, env._segment_start_w)]
    with pytest.raises(ValueError):
        env.set_play_goal(goal, deadline)
    after = (env.play_goal, env._goal_position_w, env._goal_deadline_steps, env._segment_start_w)
    assert all(torch.equal(a, b) for a, b in zip(after, before)) and env.play_deadline_s is None


def test_clean_deadline_holds_the_same_goal_without_reset_and_records_the_outcome():
    env = play_env()
    goal_w = env._goal_position_w.clone()
    start = env._segment_start_w.clone()
    data = env.robot.data
    data.root_pos_w.copy_(goal_w + torch.tensor([[0.05, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=torch.float64))
    data.root_lin_vel_w = torch.tensor([[0.0, 0.05, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    env._episode_steps.copy_(env._goal_deadline_steps)
    done, timeout = env._get_dones()
    assert not done.any() and not timeout.any() and env._goal_finished.all()
    env._get_rewards()
    assert env.play_totals == {"goals": 2, "goal_successes": 1, "goal_failures": 0,
                               "holds": 0, "hold_successes": 0, "hold_failures": 0}
    [result] = env.play_results
    assert result["kind"] == "goal" and result["success"] and not result["failed"]
    assert result["goal_m"] == pytest.approx(list(GOAL)) and result["distance_m"] == pytest.approx(0.05)
    assert result["speed_m_s"] == pytest.approx(0.05) and result["elapsed_s"] == pytest.approx(result["deadline_s"])
    # Hold: same goal and start view, fresh shortest training clock from the current distance, no reset.
    assert env._segment_hold.all() and not env._episode_steps.any() and env.reset_calls == [[0, 1]]
    torch.testing.assert_close(env._goal_position_w, goal_w)
    torch.testing.assert_close(env._segment_start_w, start)
    assert env._goal_deadline_steps.tolist() == [auto_steps(env.cfg, 0.05, hold=True),
                                                 auto_steps(env.cfg, 0.6, hold=True)] == [100, 100]
    assert not env._goal_finished.any() and env._first_arrival_s.tolist() == [0.0, -1.0]
    # A successful hold only updates totals; a crash during a hold is reported.
    env._episode_steps.copy_(env._goal_deadline_steps)
    env.safety_failures[0] = True
    done, _ = env._get_dones()
    assert done.tolist() == [True, False]
    env._get_rewards()
    assert env.play_totals["holds"] == 2 and env.play_totals["hold_failures"] == 1
    assert env.play_totals["hold_successes"] == 0 and env.play_hold_worst_distance_m == pytest.approx(0.6)
    assert env.play_results[-1]["kind"] == "hold" and env.play_results[-1]["failed"]
    assert env._segment_hold.tolist() == [True, True]
    assert env._episode_steps.tolist() == [int(env._goal_deadline_steps[0]), 0]  # env 0 waits for the real reset


def test_safety_failure_resets_then_flies_the_goal_again_from_home():
    env = play_env()
    env._episode_steps[:] = 40
    env.safety_failures[1] = True
    done, _ = env._get_dones()
    assert done.tolist() == [False, True]
    env._get_rewards()
    assert env.play_totals["goals"] == 1 and env.play_totals["goal_failures"] == 1 and not env.play_results
    assert not env._segment_hold.any()  # a failed segment never becomes a hold
    env.robot.data.root_pos_w[1] += 3.0
    env._reset_idx(torch.tensor([1]))
    home = env.scene.env_origins[1] + torch.tensor(env.cfg.robot_home, dtype=torch.float64)
    torch.testing.assert_close(env.robot.data.root_pos_w[1], home)
    torch.testing.assert_close(env._goal_position_w[1], env.scene.env_origins[1] + torch.tensor(GOAL, dtype=torch.float64))
    torch.testing.assert_close(env._segment_start_w[1], home)
    assert env._episode_steps.tolist() == [40, 0] and not env._segment_hold.any()


def test_reset_at_deadline_mode_terminates_and_repeats_the_goal_like_training():
    env = play_env(reset_at_deadline=True, deadline_s=1.2)
    env._episode_steps.copy_(env._goal_deadline_steps)
    done, timeout = env._get_dones()
    assert done.all() and not timeout.any()
    env._get_rewards()
    assert env.play_totals["goals"] == 2 and not env._segment_hold.any() and len(env.play_results) == 1
    env._reset_idx(torch.tensor([0, 1]))
    torch.testing.assert_close(env._goal_position_w, env.scene.env_origins + torch.tensor(GOAL, dtype=torch.float64))
    assert env._goal_deadline_steps.tolist() == [120, 120] and not env._episode_steps.any()
    env.set_play_goal((0.0, 0.0, 2.5), 2.0)  # applied by the caller's reset, not mid-episode
    assert env._goal_deadline_steps.tolist() == [120, 120]
    env._reset_idx(None)
    assert env._goal_deadline_steps.tolist() == [200, 200]


def test_random_mode_keeps_the_training_sampler_and_reports_sampled_outcomes():
    torch.manual_seed(3)
    env = play_env(goal=None)
    assert env.reset_at_deadline and env.play_goal is None
    start = env.robot.data.root_pos_w.clone()
    torch.manual_seed(11)
    env._reset_idx(None)
    torch.manual_seed(11)
    goals, steps = task_math.sample_goals(start, env.cfg, 5, env.step_dt)
    torch.testing.assert_close(env._goal_position_w, goals)
    assert torch.equal(env._goal_deadline_steps, steps)
    torch.testing.assert_close(env._segment_start_w, start)
    env._episode_steps.copy_(env._goal_deadline_steps)
    assert env._get_dones()[0].all()
    env._get_rewards()
    assert env.play_results[0]["kind"] == "sampled" and env.play_totals["goals"] == 2


@pytest.mark.parametrize("start,goal", [
    ((0, 0, 1.5), (3, 0, 1.5)), ((0, 0, 1.5), (-6, 0, 1.0)), ((0, 0, 1.5), (0, 6, 2.5)),
    ((0, 0, 1.5), (6, -6, 0.5)), ((1, 1, 1.5), (1, 1, 2.5)), ((0, 0, 1.5), (0, 0, 1.5)), ((-6, -6, 0.5), (6, 6, 2.5))])
def test_goal_camera_frames_the_start_and_the_goal(start, goal):
    eye, lookat = UI["goal_camera_pose"](start, goal)
    view = [l - e for l, e in zip(lookat, eye)]
    forward = [v / math.sqrt(sum(c * c for c in view)) for v in view]
    right = [forward[1], -forward[0], 0.0]
    right = [c / math.hypot(right[0], right[1]) for c in right]
    up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2],
          right[0] * forward[1] - right[1] * forward[0]]
    tan_h, tan_v = math.tan(math.radians(30)), math.tan(math.radians(30)) * 9 / 16
    for point in (start, goal):
        p = [a - e for a, e in zip(point, eye)]
        depth = sum(a * b for a, b in zip(p, forward))
        assert depth > 0
        assert abs(sum(a * b for a, b in zip(p, right))) / depth < tan_h * 0.9
        assert abs(sum(a * b for a, b in zip(p, up))) / depth < tan_v * 0.9
    if abs(goal[0] - start[0]) > 0.05 and abs(goal[1] - start[1]) < 0.05:
        assert eye[1] < lookat[1]  # x flights are seen from the -Y side
    if abs(goal[0] - start[0]) < 0.05 and abs(goal[1] - start[1]) < 0.05:
        assert eye[0] == lookat[0] and eye[1] < lookat[1]


def test_training_region_notes_name_each_out_of_distribution_reason():
    cfg = config(goal_curriculum_start_level=9, include_base_height=True)
    notes = UI["training_region_notes"]
    assert notes(cfg, 5, (5.9, -5.9, 0.99), 2.49, 0.1, 1.75, (1.0, 2.5), 0.01) == []
    assert notes(cfg, 5, (0, 0, 0), 1.5, 0.0, 2.509, (1.0, 2.5), 0.01) == []  # within one tick above
    found = notes(cfg, 5, (6.5, 0.0, -1.2), 3.0, 1.0, 0.5, (1.2, 2.5), 0.01)
    assert len(found) == 5 and found[0].startswith("body x") and found[1].startswith("body z")
    assert "training heights 0.50-2.50 m" in found[2] and "start from rest" in found[3] and "deadline" in found[4]
    cfg.include_base_height = False
    assert notes(cfg, 5, (0, 0, 0), 3.0, 0.0, 1.75, (1.0, 2.5), 0.01) == []


def test_results_read_as_plain_sentences():
    base = {"kind": "goal", "goal_m": [3.0, 0.0, 1.5], "deadline_s": 1.83, "elapsed_s": 1.83, "distance_m": 0.0091,
            "speed_m_s": 0.07, "ang_speed_rad_s": 0.12, "min_distance_m": 0.004, "first_arrival_s": 1.62,
            "success": True, "failed": False}
    text = UI["describe_result"](base, 3)
    assert text.startswith("Goal #3 (+3.00, +0.00, +1.50) m, deadline 1.83 s: SUCCESS.") and "first inside 1.62 s" in text
    assert "speed 0.07 m/s, angular speed 0.12 rad/s" in text
    missed = UI["describe_result"]({**base, "success": False, "first_arrival_s": None}, 4)
    assert "MISSED" in missed and "never inside" in missed
    failed = UI["describe_result"]({**base, "kind": "hold", "success": False, "failed": True, "elapsed_s": 0.4}, 5)
    assert failed.startswith("Hold #5") and "FAILED after 0.40 s" in failed


def test_relative_start_uses_each_reset_base_position_not_absolute_height():
    env = play_env(goal=(1.0, -0.5, 0.0), goal_relative=True)
    torch.testing.assert_close(env._goal_position_w,
                               env.robot.data.root_pos_w + torch.tensor([1., -.5, 0.], dtype=torch.float64))
    assert env.play_goal_relative and env.play_goal.tolist() == [1., -.5, 0.]
    assert env._commands.shape == (2, 4)


def test_repeating_a_relative_goal_snapshots_each_current_base_along_world_axes():
    env = play_env()
    data = env.robot.data
    data.root_pos_w += torch.tensor([[2., -1., .2], [-2., 1., -.2]], dtype=torch.float64)
    data.root_quat_w[:] = torch.tensor([0., 0., 0., 1.], dtype=torch.float64)  # yaw 180 degrees
    displacement = torch.tensor([1., .5, .1], dtype=torch.float64)
    state = [v.clone() for v in (data.root_pos_w, data.root_quat_w, data.root_lin_vel_w)]
    env.set_play_goal(displacement.tolist(), 1.5, relative=True)
    target = data.root_pos_w + displacement
    torch.testing.assert_close(env._goal_position_w, target)
    for before, after in zip(state, (data.root_pos_w, data.root_quat_w, data.root_lin_vel_w)):
        assert torch.equal(before, after)  # no teleport or yaw-dependent rotation
    obs = env._get_observations()["policy"]
    torch.testing.assert_close(env._commands[:, :3], inverse(data.root_quat_w, displacement.expand(2, -1)))
    assert obs[:, -1].tolist() == [1.5, 1.5]
    data.root_pos_w.copy_(target)  # arrive, then press Go again with unchanged input
    env.set_play_goal(displacement.tolist(), 1.5, relative=True)
    torch.testing.assert_close(env._goal_position_w, target + displacement)


@pytest.mark.parametrize("reset_at_deadline", [False, True])
def test_relative_goal_stays_fixed_across_holds_and_safety_or_episode_resets(reset_at_deadline):
    env = play_env(goal=(1., 0., 0.), goal_relative=True, reset_at_deadline=reset_at_deadline)
    target = env._goal_position_w.clone()
    env.robot.data.root_pos_w.copy_(target)
    env._episode_steps.copy_(env._goal_deadline_steps)
    env._get_dones()
    env._get_rewards()
    torch.testing.assert_close(env._goal_position_w, target)  # not a moving target
    env._reset_idx(torch.tensor([1]))
    torch.testing.assert_close(env._goal_position_w, target)  # safety reset cannot add the offset again
    assert env.play_goal_relative


def test_relative_goal_rejects_all_robots_atomically_when_one_target_is_out_of_bounds():
    env = play_env()
    env.robot.data.root_pos_w[1, 0] = env.scene.env_origins[1, 0] + env.cfg.flight_xy_limit_m - 0.5
    before = [v.clone() for v in (env.play_goal, env._play_goal_targets, env._goal_position_w,
                                 env._goal_deadline_steps, env._episode_steps)]
    with pytest.raises(ValueError, match="x and y"):
        env.set_play_goal((1., 0., 0.), 2.0, relative=True)
    for a, b in zip(before, (env.play_goal, env._play_goal_targets, env._goal_position_w,
                            env._goal_deadline_steps, env._episode_steps)):
        assert torch.equal(a, b)
    assert not env.play_goal_relative and env.play_deadline_s is None
    with pytest.raises(ValueError, match="Goal z"):
        env.set_play_goal((0., 0., -2.), relative=True)


def controls(env, goal, relative=False):
    return UI["GoalControls"](env, goal, None, [(-6., 6.)] * 4, show_ui=False, relative=relative)


@pytest.mark.parametrize("index,value", [(0, 7.5), (1, -7.5), (2, 4.0), (3, 12.345)])
def test_slider_edits_only_change_the_draft_until_go(index, value):
    env = play_env()
    panel = controls(env, GOAL)
    before = env._goal_position_w.clone(), env._episode_steps.clone()
    panel.set_component(index, value)
    assert not panel.go_requested
    assert panel.apply_requests(None, "unchanged") == ("unchanged", False)
    assert torch.equal(env._goal_position_w, before[0]) and torch.equal(env._episode_steps, before[1])
    if index == 3:
        assert not panel.auto_deadline and panel.requested_deadline() == value


def test_playback_speed_changes_only_wall_pacing_and_measures_unpaused_progress(monkeypatch):
    from types import SimpleNamespace
    env = play_env()
    assert controls(env, GOAL).speed is None  # unpaced unless the player passes a speed
    panel = UI["GoalControls"](env, GOAL, None, [(-6., 6.)] * 4, show_ui=False, speed=1.0)
    assert UI["PLAYBACK_SPEEDS"] == (0.25, 0.5, 0.75, 1.0)
    before = env._goal_position_w.clone(), env._episode_steps.clone()
    panel.set_speed(0.25)
    assert panel.speed == 0.25 and panel.speed_description() == "Target x0.25 of real time."
    assert panel.apply_requests(None, "unchanged") == ("unchanged", False)
    assert torch.equal(env._goal_position_w, before[0]) and torch.equal(env._episode_steps, before[1])
    clock = iter((10.0, 15.0, 15.4))
    monkeypatch.setitem(UI, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    panel._measure_speed(1.0)
    panel._measure_speed(1.0)  # five paused wall seconds: no simulated progress, no measurement
    assert panel.measured_speed is None
    panel._measure_speed(1.1)  # 0.1 simulated s in 0.4 wall s
    assert panel.measured_speed == pytest.approx(0.25)
    assert panel.speed_description() == "Target x0.25 of real time; measured x0.25."


@pytest.mark.parametrize("episodes", [False, True])
def test_reset_keeps_active_goal_and_does_not_send_draft_even_while_paused(episodes):
    from types import SimpleNamespace
    env = play_env(reset_at_deadline=episodes)
    panel = controls(env, GOAL)
    panel.paused = True
    panel.set_component(0, 8.0)
    panel.set_component(3, 12.345)
    panel.request_go()  # Reset wins even if Go was also queued.
    active_goal = env._goal_position_w.clone()
    panel.request_reset()
    resets = []

    def reset():
        env._reset_idx(None)
        resets.append(True)
        return env._get_observations(), {}

    wrapper = SimpleNamespace(reset=reset, get_observations=env._get_observations)
    observation, did_reset = panel.apply_requests(wrapper, None)
    assert did_reset and len(resets) == 1 and panel.manual_resets == 1
    assert not panel.go_requested and not panel.reset_requested and panel.paused
    assert env.play_deadline_s is None
    assert not env._episode_steps.any()
    torch.testing.assert_close(env._goal_position_w, active_goal)
    assert panel.goal[0] == 8.0 and panel.deadline_s == 12.345  # draft is not discarded
    assert panel.apply_requests(wrapper, observation) == (observation, False)


def test_manual_deadline_survives_holds_outside_training_range_and_generic_horizon():
    env = play_env(deadline_s=12.345)
    assert env._goal_deadline_steps.tolist() == [1235, 1235]
    env._episode_steps[:] = env.max_episode_length + 1
    done, timeout = env._get_dones()
    assert not done.any() and not timeout.any()  # generic training horizon cannot cut the command short
    env._episode_steps.copy_(env._goal_deadline_steps)
    assert not env._get_dones()[0].any()
    env._get_rewards()
    assert env._segment_hold.all() and env._goal_deadline_steps.tolist() == [1235, 1235]


def test_auto_checkbox_only_edits_draft_and_curriculum_metadata_is_only_a_bounded_reference():
    env = play_env()
    panel = controls(env, GOAL)
    panel.set_auto_deadline(False)
    assert not panel.go_requested and panel.requested_deadline() == 2.0
    scope = load_definitions(FOLDER / "playback.py", {
        "BorinotGoToEnv": environment_class(), "max_curriculum_level": task_math.max_curriculum_level})
    level = scope["checkpoint_goal_level"]
    assert level({"best_model_curriculum_idx": 12}, 0, set(), max_level=9) == 9
    assert level({"best_model_curriculum_idx": 12}, 4, {"env.goal_curriculum_start_level"}, max_level=9) == 4


def wrapper_for(env):
    from types import SimpleNamespace

    def reset():
        env._reset_idx(None)
        return env._get_observations(), {}

    return SimpleNamespace(reset=reset, get_observations=env._get_observations)


def finish_segment(env, success=True):
    env.robot.data.root_pos_w.copy_(env._goal_position_w)
    env.robot.data.root_lin_vel_w[:] = 0.0 if success else 2.0
    env._episode_steps.copy_(env._goal_deadline_steps)
    env._get_dones()
    env._get_rewards()


def sequence_panel(env=None):
    env = env or play_env()
    panel = controls(env, (3., 0., 1.5))
    panel.add_waypoint()
    panel.waypoints[0]["values"][4] = 2.0
    panel.set_component(0, -3.)
    panel.add_waypoint()
    return panel, env, wrapper_for(env)


def test_go_snapshots_all_fields_and_later_edits_do_not_leak_into_the_sent_command():
    env = play_env()
    panel = controls(env, GOAL)
    panel.set_component(0, 3.)
    panel.set_component(1, 2.)
    panel.set_component(2, 1.5)
    panel.set_component(3, 1.75)
    panel.request_go()
    panel.set_component(0, -3.)
    panel.set_auto_deadline(True)
    panel.set_relative(True)
    observation, reset = panel.apply_requests(wrapper_for(env), None)
    assert not reset and env.play_deadline_s == 1.75 and not env.play_goal_relative
    torch.testing.assert_close(env._goal_position_w - env.scene.env_origins,
                               torch.tensor([[3., 2., 1.5]] * 2, dtype=torch.float64))
    assert observation["policy"][:, -1].tolist() == [1.75, 1.75]


def test_sequence_snapshots_rows_and_waits_two_sim_seconds_after_all_waypoint_deadlines():
    panel, env, wrapper = sequence_panel()
    original = env._goal_position_w.clone()
    assert not panel.go_requested and torch.equal(original, env._goal_position_w)
    panel.request_sequence_start()
    panel.waypoints[0]["values"][0] = 7.0  # editing the list cannot change the queued snapshot
    panel.remove_waypoint(1)
    observation, reset = panel.apply_requests(wrapper, None, 10.)
    assert not reset and panel.sequence_phase == "flying"
    assert env.play_goal.tolist() == [3., 0., 1.5]
    assert observation["policy"][:, -1].min() > 0
    panel.apply_requests(wrapper, observation, 11.)  # time alone cannot complete a flight
    assert panel.sequence_phase == "flying"
    finish_segment(env)
    panel.apply_requests(wrapper, observation, 12.)
    assert panel.sequence_phase == "waiting"
    panel.paused = True
    panel.apply_requests(wrapper, observation, 12.)  # official player freezes simulated time while paused
    panel.paused = False
    panel.apply_requests(wrapper, observation, 13.99)
    assert env.play_goal.tolist() == [3., 0., 1.5]
    observation, _ = panel.apply_requests(wrapper, observation, 14.)
    assert panel.sequence_index == 1 and panel.sequence_phase == "flying"
    assert env.play_goal.tolist() == [-3., 0., 1.5]
    assert not env._episode_steps.any() and not env._segment_hold.any()
    finish_segment(env)
    panel.apply_requests(wrapper, observation, 18.)
    assert panel.sequence_phase == "completed" and panel._sequence is None
    assert env.play_goal.tolist() == [-3., 0., 1.5] and env._segment_hold.all()


@pytest.mark.parametrize("bad_component,bad_value", [(0, math.nan), (2, 50.), (3, -1.), (4, -1.), (4, math.inf)])
def test_invalid_sequence_is_rejected_in_full_before_any_goal_is_sent(bad_component, bad_value):
    panel, env, wrapper = sequence_panel()
    panel.waypoints[1]["values"][bad_component] = bad_value
    before = env._goal_position_w.clone(), env._episode_steps.clone()
    panel.request_sequence_start()
    panel.apply_requests(wrapper, None, 0.)
    assert panel._sequence is None and panel._pending_sequence is None
    assert "not started" in panel.sequence_message
    assert torch.equal(before[0], env._goal_position_w) and torch.equal(before[1], env._episode_steps)


@pytest.mark.parametrize("failure", ["goal safety", "hold safety"])
def test_sequence_stops_on_safety_reset_without_advancing(failure):
    panel, env, wrapper = sequence_panel()
    panel.request_sequence_start()
    observation, _ = panel.apply_requests(wrapper, None, 0.)
    if failure == "hold safety":
        finish_segment(env)
        panel.apply_requests(wrapper, observation, 5.)
        assert panel.sequence_phase == "waiting"
        env.safety_failures[0] = True
        finish_segment(env)
    else:
        env.safety_failures[0] = failure == "goal safety"
        finish_segment(env, success=False)
    panel.apply_requests(wrapper, observation, 6.)
    assert panel.sequence_phase == "stopped" and env.play_goal.tolist() == [3., 0., 1.5]
    assert "safety" in panel.sequence_message


def test_a_training_score_miss_does_not_block_the_timed_sequence():
    panel, env, wrapper = sequence_panel()
    panel.request_sequence_start()
    panel.apply_requests(wrapper, None, 0.)
    finish_segment(env, success=False)
    assert env.play_totals["goal_successes"] == 0
    panel.apply_requests(wrapper, None, 5.)
    assert panel.sequence_phase == "waiting"
    panel.apply_requests(wrapper, None, 7.)
    assert panel.sequence_phase == "flying" and env.play_goal.tolist() == [-3., 0., 1.5]


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("action", ["stop", "go", "home", "reset"])
def test_explicit_actions_cancel_queued_or_running_sequence(action, running):
    panel, env, wrapper = sequence_panel()
    panel.request_sequence_start()
    observation = None
    if running:
        observation, _ = panel.apply_requests(wrapper, None, 0.)
    target = env._goal_position_w.clone()
    {"stop": panel.stop_sequence, "go": panel.request_go,
     "home": panel.fly_home, "reset": panel.request_reset}[action]()
    assert panel._sequence is None and panel._pending_sequence is None
    panel.apply_requests(wrapper, observation, 10.)
    if action in ("stop", "reset"):
        assert torch.equal(env._goal_position_w, target)
    elif action == "go":
        assert env.play_goal.tolist() == [-3., 0., 1.5]
    else:
        assert env.play_goal.tolist() == list(env.cfg.robot_home)


def test_relative_waypoints_resolve_when_each_row_starts_not_when_the_list_is_built():
    env = play_env()
    panel = controls(env, (1., 0., 0.), relative=True)
    panel.add_waypoint()
    panel.add_waypoint()
    start = env.robot.data.root_pos_w.clone()
    panel.request_sequence_start()
    wrapper = wrapper_for(env)
    panel.apply_requests(wrapper, None, 0.)
    finish_segment(env)
    observation, _ = panel.apply_requests(wrapper, None, 5.)
    torch.testing.assert_close(env._goal_position_w, start + start.new_tensor([2., 0., 0.]))
    assert env.play_goal_relative and observation["policy"][:, -1].min() > 0


def test_sequence_waits_for_the_last_clone_and_rejects_a_later_invalid_relative_target():
    panel, env, wrapper = sequence_panel()
    panel.request_sequence_start()
    panel.apply_requests(wrapper, None, 0.)
    env.robot.data.root_pos_w.copy_(env._goal_position_w)
    env.robot.data.root_lin_vel_w[:] = 0.
    env._episode_steps[0] = env._goal_deadline_steps[0]
    env._get_dones()
    env._get_rewards()
    panel.apply_requests(wrapper, None, 5.)
    assert panel.sequence_phase == "flying"  # one clone has not finished
    env._episode_steps[0] = 0  # clone 0 is now in a hold, not another goal
    env._episode_steps[1] = env._goal_deadline_steps[1]
    env._get_dones()
    env._get_rewards()
    panel.apply_requests(wrapper, None, 6.)
    assert panel.sequence_phase == "waiting" and panel._wait_until_s == 8.

    # Relative rows cannot be bounds-checked against their future start positions.
    panel.stop_sequence()
    panel.waypoints.clear()
    panel.goal = [1., 0., 0.]
    panel.set_relative(True)
    panel.add_waypoint()
    panel.add_waypoint()
    env.robot.data.root_pos_w[:, 0] = env.scene.env_origins[:, 0] + env.cfg.flight_xy_limit_m - 1.5
    panel.request_sequence_start()
    panel.apply_requests(wrapper, None, 10.)
    finish_segment(env)
    active = env._goal_position_w.clone()
    observation = env._get_observations()
    returned, _ = panel.apply_requests(wrapper, observation, 12.)
    assert returned is observation and panel.sequence_phase == "stopped"
    assert "row 2" in panel.sequence_message
    assert torch.equal(active, env._goal_position_w)


def test_sequences_reject_empty_lists_and_reset_at_deadline_modes():
    env = play_env()
    panel = controls(env, GOAL)
    panel.request_sequence_start()
    assert "at least one" in panel.sequence_message
    panel.add_waypoint()
    env.reset_at_deadline = True
    panel.request_sequence_start()
    assert panel._pending_sequence is None and "continuous" in panel.sequence_message


def test_ui_mode_switch_preserves_input_without_sending_and_relative_preview_tracks_only_the_next_go():
    env = play_env()
    panel = controls(env, (1., -.5, 0.))
    assert panel.preview_goal() == [1., -.5, 0.]
    panel.set_relative(True)
    assert panel.goal == [1., -.5, 0.] and not panel.go_requested
    target = env._goal_position_w.clone()
    panel.request_go()
    assert panel.go_requested
    env.set_play_goal(panel.goal, panel.requested_deadline(), relative=panel.relative)
    expected = env.robot.data.root_pos_w[0] - env.scene.env_origins[0] + torch.tensor(panel.goal)
    assert panel.preview_goal() == pytest.approx(expected.tolist())
    sent = env._goal_position_w.clone()
    env.robot.data.root_pos_w[:, 0] += 0.2
    assert panel.preview_goal()[0] == pytest.approx(expected[0] + .2)
    assert torch.equal(sent, env._goal_position_w) and not torch.equal(target, sent)
    panel.go_requested = False
    panel.fly_home()
    assert not panel.relative and panel.goal == list(env.cfg.robot_home) and panel.go_requested
    assert "world position" in panel.frame_description()


def test_live_state_logs_and_labels_read_actual_position_velocity_error_and_time_without_mutation():
    from types import SimpleNamespace
    env = play_env()
    env.robot.data.root_pos_w[0] = env.scene.env_origins[0] + torch.tensor([.2, -.1, 1.6])
    env.robot.data.root_lin_vel_w[0] = torch.tensor([.3, -.4, 0.])
    env.robot.data.root_ang_vel_w[0] = torch.tensor([0., .6, -.8])
    env._episode_steps[:] = 25
    before = [v.clone() for v in (env._goal_position_w, env._episode_steps, env._commands,
                                 env.robot.data.root_pos_w, env.robot.data.root_lin_vel_w)]
    state = UI["flight_state"](env, 0.25)
    assert state["position_m"] == pytest.approx([.2, -.1, 1.6])
    assert state["velocity_m_s"] == pytest.approx([.3, -.4, 0.]) and state["speed_m_s"] == pytest.approx(.5)
    assert state["ang_speed_rad_s"] == pytest.approx(1.)
    assert state["goal_m"] == pytest.approx(list(GOAL))
    assert state["error_m"] == pytest.approx([.8, -.4, .4])
    assert state["distance_m"] == pytest.approx(math.sqrt(.8**2 + .4**2 + .4**2))
    assert state["remaining_s"] == pytest.approx((int(env._goal_deadline_steps[0]) - 25) * .01)
    text = UI["describe_state"](state)
    assert "position XYZ=(+0.200, -0.100, +1.600) m" in text and "velocity XYZ=" in text and "error XYZ=" in text
    assert "angular speed=1.000 rad/s" in text
    panel = controls(env, (1., 0., 0.), relative=True)
    for name in ("state_label", "goal_label", "preview_label", "result_label", "status_label"):
        setattr(panel, name, SimpleNamespace(text=""))
    panel._refresh_labels(.25, state["position_m"], state["goal_m"])
    assert "CURRENT POSITION XYZ (+0.200, -0.100, +1.600) m" in panel.state_label.text
    assert "RELATIVE displacement" in panel.goal_label.text
    assert "world XYZ (+1.200, -0.100, +1.600) m" in panel.preview_label.text
    for a, b in zip(before, (env._goal_position_w, env._episode_steps, env._commands,
                            env.robot.data.root_pos_w, env.robot.data.root_lin_vel_w)):
        assert torch.equal(a, b)
