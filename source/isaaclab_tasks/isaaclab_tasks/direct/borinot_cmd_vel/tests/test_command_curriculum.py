# SPDX-License-Identifier: BSD-3-Clause
"""Command/force ordering and episode-boundary sampling, without simulator imports."""
import ast
from types import SimpleNamespace

import pytest
import torch

from test_task import curriculum_env, method, task_math


def command_env(limits=(1., 3., 5.), forces=(0., 1., 3.), start=0, skip=False):
    env = curriculum_env()
    env.cfg.command_lin_vel_lim_curriculum = limits
    env.cfg.forces_applied_to_base_curriculum = forces
    env.cfg.command_lin_vel_x_range = (-.2, .4)
    env.cfg.command_lin_vel_y_range = (-.6, .8)
    env.cfg.command_lin_vel_z_range = (-.5, .7)
    env.cfg.command_ang_vel_z_range = (-2., 1.)
    env.cfg.command_resampling_time_s = 5.
    env.cfg.standing_env_prob = 0.
    env.cfg.curriculum_start_level, env.cfg.skip_curriculum = start, skip
    env._command_steps_left = torch.zeros(4, dtype=torch.long)
    env._commands = torch.zeros(4, 4)
    method("_init_command_curriculum")(env)
    return env


@pytest.mark.parametrize("forces,expected", [
    ((0., 1., 3.), [(1., 0.), (3., 0.), (5., 0.), (5., 1.), (5., 3.)]),
    ((2., 4.), [(1., 0.), (3., 0.), (5., 0.), (5., 2.), (5., 4.)]),
    ((0.,), [(1., 0.), (3., 0.), (5., 0.)]),
])
def test_all_velocity_stages_precede_forces(forces, expected):
    assert task_math.command_force_curriculum_stages((1., 3., 5.), forces) == expected


@pytest.mark.parametrize("limits", [(-1.,), (1., .5), (float("nan"),), (float("inf"),)])
def test_invalid_velocity_curriculum_fails_before_simulation(limits):
    with pytest.raises(ValueError, match="command_lin_vel_lim_curriculum"):
        task_math.command_force_curriculum_stages(limits, (0., 1.))


def test_empty_curriculum_preserves_legacy_ranges_and_random_samples_exactly():
    env = command_env(limits=())
    assert env._force_limits.tolist() == [0., 1., 3.]
    torch.manual_seed(17)
    samples = torch.rand(4, 4)
    ranges = torch.tensor([[-.2, .4], [-.6, .8], [-.5, .7], [-2., 1.]])
    expected = ranges[:, 0] + samples * (ranges[:, 1] - ranges[:, 0])
    torch.manual_seed(17)
    method("_resample_commands")(env, torch.arange(4))
    assert torch.equal(env._commands, expected)


@pytest.mark.parametrize("start,skip,expected", [(0, False, 0), (2, False, 2), (0, True, 4)])
def test_fixed_and_skip_start_select_combined_stage(start, skip, expected):
    env = command_env(start=start, skip=skip)
    assert env.force_level == expected
    assert env._episode_stage.tolist() == [expected] * 4
    assert method("get_curriculum_global_idx")(env) == expected
    assert method("get_curriculum_max_global_idx")(env) == 4
    limit = [1., 3., 5., 5., 5.][expected]
    assert env.cfg.command_lin_vel_x_range == env.cfg.command_lin_vel_y_range == (-limit, limit)


def test_velocity_promotions_wait_for_evidence_and_old_episodes_keep_their_ranges():
    env = command_env()
    update = method("_update_curriculum")
    resample = method("_resample_commands")
    for expected in range(1, 5):
        env._episode_stage[:2] = env.force_level
        env.common_step_counter = expected * 100 - 1
        update(env, torch.tensor([0, 1]))
        assert env.force_level == expected - 1  # stage dwell has not completed
        env.common_step_counter += 1
        update(env, torch.tensor([0, 1]))
        assert env.force_level == expected
        assert not env._curriculum_window.any() and env._stage_episodes == 0
        env._episode_stage[0] = expected  # selected-environment reset
        torch.manual_seed(31)
        resample(env, torch.arange(4))
        limits = [1., 3., 5., 5., 5.]
        torch.manual_seed(31)
        samples = torch.rand(4, 4)
        for i, stage in enumerate(env._episode_stage):
            torch.testing.assert_close(env._commands[i, :2], -limits[stage] + samples[i, :2] * 2 * limits[stage])
        torch.testing.assert_close(env._commands[:, 2], -.5 + samples[:, 2] * 1.2)
        torch.testing.assert_close(env._commands[:, 3], -2. + samples[:, 3] * 3.)
        assert env._force_limits[expected] == [0., 0., 0., 1., 3.][expected]
        assert env.extras["log"]["Curriculum/command_lin_vel_x_abs_m_s"] == limits[expected]
        env.common_step_counter += 99
        update(env, torch.tensor([2]))  # old-stage outcome cannot promote
        assert env.force_level == expected and env._stage_episodes == 0
    env._episode_stage[:2] = env.force_level
    update(env, torch.tensor([0, 1]))
    assert env.force_level == 4  # final stage never overflows


def test_standing_and_partial_resampling_are_preserved():
    env = command_env()
    env._commands.fill_(99.)
    env.cfg.standing_env_prob = 1.
    method("_resample_commands")(env, torch.tensor([1, 3]))
    assert not env._commands[[1, 3]].any()
    assert (env._commands[[0, 2]] == 99.).all()
    assert env._command_steps_left.tolist() == [0, 50, 0, 50]


@pytest.mark.parametrize("limits", [(-1.,), (3., 1.), (float("nan"),), (float("inf"),)])
def test_actual_config_rejects_invalid_command_limits(limits):
    from test_observation_flags import config
    with pytest.raises(ValueError, match="command_lin_vel_lim_curriculum"):
        config(command_lin_vel_lim_curriculum=limits)


def test_actual_config_accepts_list_overrides_and_combined_start_levels():
    from test_observation_flags import config
    config(command_lin_vel_lim_curriculum=[1., 3., 5.],
           forces_applied_to_base_curriculum=(0., 1., 3.), curriculum_start_level=4)
    with pytest.raises(ValueError, match="combined command/force"):
        config(command_lin_vel_lim_curriculum=[1., 3., 5.],
               forces_applied_to_base_curriculum=(0., 1., 3.), curriculum_start_level=5)


def test_training_checkpoint_hook_tracks_combined_stage_and_selected_command_limit():
    from test_task import FOLDER
    tree = ast.parse((FOLDER.parents[4] / "source/scripts/rsl_rl/train.py").read_text())
    names = ("_get_curriculum_state_from_runner", "_curriculum_checkpoint_stage")
    scope = {}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), "runner_hooks", "exec"), scope)
    env = command_env(start=4)
    env.get_curriculum_global_idx = lambda: method("get_curriculum_global_idx")(env)
    runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env))
    state = scope[names[0]](runner)
    assert state["global_idx"] == 4 and state["max_velx_range_idx"] == 2 and state["base_push_force_idx"] == 2
    assert state["command_lin_vel_x_abs"] == 5.
    assert scope[names[1]](state) == ((4, None), "best_model_curriculum_idx_4.pt")
