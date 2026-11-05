# SPDX-License-Identifier: BSD-3-Clause
"""CPU checks for the live go-to player: goal ownership, deadlines, holds, resets and views."""
import ast
import math
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
    "math": math, "quat_apply_inverse": inverse, "camera_pose": CMD_UI["camera_pose"],
    "training_deadline_interval": task_math.training_deadline_interval}, constants=True)
GOAL = (1.0, -0.5, 2.0)


def play_env(goal=GOAL, deadline_s=None, reset_at_deadline=False, **overrides):
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
        data.root_lin_vel_w[env_ids] = 0.0
        self._episode_steps[env_ids] = 0

    flight.__init__, flight._reset_idx = flight_init, flight_reset
    scope = load_definitions(FOLDER / "playback.py", {
        "torch": torch, "math": math, "BorinotGoToEnv": goto, "stage_settings": task_math.stage_settings,
        "training_deadline_interval": task_math.training_deadline_interval})
    cfg = config(goal_curriculum_start_level=9, **overrides)
    env = scope["BorinotGoToPlayEnv"](cfg, goal=goal, deadline_s=deadline_s, reset_at_deadline=reset_at_deadline)
    env._reset_idx(None)  # the vector wrapper resets once before the first observation
    return env


def auto_steps(cfg, distance, difficulty=5, hold=False):
    """Independent oracle: middle (goal) or start (hold) of [max(low, floor), max(high, floor)], in 0.01 s ticks."""
    floor = max(distance / cfg.goal_reference_speed_m_s[difficulty],
                2 * math.sqrt(distance / cfg.goal_reference_accel_m_s2[difficulty]))
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


@pytest.mark.parametrize("distance", [0.0, 0.5, 2.0, 2.25, 3.0, 6.0, 7.8, 8.544])
def test_auto_goal_deadline_is_the_middle_and_hold_deadline_the_start_of_the_training_interval(distance):
    env = play_env()
    distance_t = torch.tensor([distance], dtype=torch.float64)
    assert env._play_deadline_steps(distance_t, hold=False).tolist() == [auto_steps(env.cfg, distance)]
    assert env._play_deadline_steps(distance_t, hold=True).tolist() == [auto_steps(env.cfg, distance, hold=True)]
    env.play_deadline_s = 4.0  # a fixed user deadline never changes the hold rule
    assert env._play_deadline_steps(distance_t, hold=False).tolist() == [400]
    assert env._play_deadline_steps(distance_t, hold=True).tolist() == [auto_steps(env.cfg, distance, hold=True)]


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
    cfg = config()
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
    data.root_lin_vel_w = torch.tensor([[0.0, 0.1, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    env._episode_steps.copy_(env._goal_deadline_steps)
    done, timeout = env._get_dones()
    assert not done.any() and not timeout.any() and env._goal_finished.all()
    env._get_rewards()
    assert env.play_totals == {"goals": 2, "goal_successes": 1, "goal_failures": 0,
                               "holds": 0, "hold_successes": 0, "hold_failures": 0}
    [result] = env.play_results
    assert result["kind"] == "goal" and result["success"] and not result["failed"]
    assert result["goal_m"] == pytest.approx(list(GOAL)) and result["distance_m"] == pytest.approx(0.05)
    assert result["speed_m_s"] == pytest.approx(0.1) and result["elapsed_s"] == pytest.approx(result["deadline_s"])
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
            "speed_m_s": 0.07, "min_distance_m": 0.004, "first_arrival_s": 1.62, "success": True, "failed": False}
    text = UI["describe_result"](base, 3)
    assert text.startswith("Goal #3 (+3.00, +0.00, +1.50) m, deadline 1.83 s: SUCCESS.") and "first inside 1.62 s" in text
    missed = UI["describe_result"]({**base, "success": False, "first_arrival_s": None}, 4)
    assert "MISSED" in missed and "never inside" in missed
    failed = UI["describe_result"]({**base, "kind": "hold", "success": False, "failed": True, "elapsed_s": 0.4}, 5)
    assert failed.startswith("Hold #5") and "FAILED after 0.40 s" in failed
