# SPDX-License-Identifier: BSD-3-Clause
"""CPU checks for play-only command ownership, metadata precedence and flight views."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]


class TrainingEnv:
    def __init__(self, cfg, **kwargs):
        self.cfg, self.device, self.step_dt = cfg, "cpu", 0.01
        self.robot = SimpleNamespace(_ALL_INDICES=torch.arange(3))
        self._commands = torch.zeros(3, 4)
        self._command_steps_left = torch.zeros(3, dtype=torch.long)

    def _resample_commands(self, env_ids):
        self._commands[env_ids] = 99  # distinguish delegation from live-command ownership


def load_definitions(filename, scope):
    nodes = [n for n in ast.parse((FOLDER / filename).read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), scope)
    return scope


PLAY = load_definitions("playback.py", {"torch": torch, "math": math, "BorinotCmdVelEnv": TrainingEnv})
UI = load_definitions("playback_ui.py", {"math": math})


def test_live_command_survives_partial_and_expired_resampling_without_standing_probability():
    env = PLAY["BorinotCmdVelPlayEnv"](SimpleNamespace(command_resampling_time_s=5.0, standing_env_prob=1.0),
                                        command=(0.1, -0.2, 0.3, -0.4))
    expected = torch.tensor([[0.1, -0.2, 0.3, -0.4]]).expand(3, -1)
    torch.testing.assert_close(env._commands, expected)
    env._commands[1] = 0
    env._command_steps_left[1] = 0
    env._resample_commands(torch.tensor([1]))
    torch.testing.assert_close(env._commands, expected)
    assert env._command_steps_left.tolist() == [500, 500, 500]
    env.set_play_command((0.0, 0.0, -0.25, 0.0))
    torch.testing.assert_close(env._commands[:, 2], torch.full((3,), -0.25))
    assert not env._commands[:, [0, 1, 3]].any()


def test_random_mode_delegates_to_training_sampler():
    env = PLAY["BorinotCmdVelPlayEnv"](SimpleNamespace(command_resampling_time_s=5.0))
    env._resample_commands(torch.tensor([1]))
    assert env._commands[1].tolist() == [99] * 4
    assert not env._commands[[0, 2]].any()


@pytest.mark.parametrize("command", [(1, 2, 3), (0, 0, math.nan, 0), (0, 0, 0, math.inf)])
def test_bad_command_does_not_replace_the_last_valid_command(command):
    env = PLAY["BorinotCmdVelPlayEnv"](SimpleNamespace(command_resampling_time_s=5.0), command=(0, 0, 0, 0))
    with pytest.raises(ValueError):
        env.set_play_command(command)
    assert not env._commands.any()


@pytest.mark.parametrize("infos,configured,overrides,expected", [
    ({"best_model_curriculum_idx": 2, "curriculum_global_idx": 1}, 0, set(), 2),
    ({"best_model_curriculum_idx": None, "curriculum_global_idx": 1}, 0, set(), 1),
    ({}, 1, set(), 1),
    ({"best_model_curriculum_idx": 2}, 0, {"env.curriculum_start_level"}, 0),
    ({"best_model_curriculum_idx": 2}, 1, {"env.skip_curriculum"}, 1),
])
def test_saved_stage_never_overwrites_explicit_environment_selection(infos, configured, overrides, expected):
    assert PLAY["checkpoint_force_level"](infos, configured, overrides) == expected


@pytest.mark.parametrize("yaw", [0, math.pi / 2, -math.pi / 2, math.pi])
def test_chase_camera_is_yaw_aligned_but_follow_camera_is_not(yaw):
    position = (3, -4, 5)
    eye, target = UI["camera_pose"]("chase", position, yaw)
    assert target == position
    assert eye == pytest.approx((3 - 2.8*math.cos(yaw), -4 - 2.8*math.sin(yaw), 6))
    assert UI["camera_pose"]("follow", position, yaw) == ((5.4, -6.8, 6.1), position)
    assert UI["camera_pose"]("free", position, yaw) is None


def test_zero_command_and_reset_intents_do_not_touch_physics():
    controls = UI["FlightControls"](None, [1, 2, 3, 4], None, show_ui=False)
    controls.zero_commands()
    assert controls.command == [0] * 4
    controls.set_component(2, 0.3)
    controls.set_component(2, math.nan)
    assert controls.command[2] == 0.3
    controls.request_reset()
    assert controls.reset_requested
    controls.toggle_pause()
    assert controls.paused
    controls.toggle_pause()
    assert not controls.paused
