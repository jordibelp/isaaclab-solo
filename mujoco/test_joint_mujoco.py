"""CPU regressions for shared-command inference; GUI integration is checked separately."""
import csv
import json
from pathlib import Path
import os

import mujoco
import numpy as np
import pytest

import play_direct_mujoco as single
import play_joint_mujoco as joint

CHECKPOINTS = (
    Path('/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1113_jd6b4err_best_model_curriculum_idx_2_walk.pt'),
    Path('/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0716_nf3m49ip_best_model.pt'),
)


def arguments(*extra):
    return joint.parse_args(['--checkpoints', *map(str, CHECKPOINTS), *extra])


def test_duplicate_range_uses_last_value_and_matches_original():
    args, overrides, ignored, ranges = arguments('env.command_lin_vel_x_range=[-0.6,0.6]',
                                                'env.command_lin_vel_x_range=[-1.5,1.5]', 'env.kp=9',
                                                'env.kd=0.2', '--disable_training_gain_sync')
    assert ranges[0] == (-1.5, 1.5)
    assert (args.kp, args.kd) == (9, 0.2)
    assert ignored == ['--disable_training_gain_sync']


@pytest.mark.parametrize('extra', [('--duration_s', 'nan'), ('--episode_length_s', '0'),
                                  ('--command-period-s', '-1'), ('--cmd_init', 'inf', '0', '0'),
                                  ('--track-cmds', 'nan 0 0'), ('env.kp=nan',),
                                  ('env.command_lin_vel_x_range=[1,-1]',), ('--headless',)])
def test_invalid_inputs_fail_before_loading(extra):
    with pytest.raises(SystemExit):
        arguments(*extra)


class SpySim:
    def __init__(self, contact=False):
        self.contact = contact
        self.reset_reasons = []
        self.seen_commands = []
        self.steps = 0

    def observation(self, command):
        self.seen_commands.append(command)
        return np.array(command)

    def step(self, action):
        self.steps += 1
        np.testing.assert_array_equal(action, self.seen_commands[-1])

    def base_hit_ground(self):
        return self.contact

    def base_height(self):
        return 0.3

    def tracked_velocity(self):
        return (0, 0, 0)

    def reset(self, reason):
        self.reset_reasons.append(reason)
        self.contact = False


def spy_worlds():
    return [joint.PolicyWorld(f'p{i}', Path(f'p{i}.pt'), lambda x: x, SpySim()) for i in range(2)]


def test_command_jump_reaches_both_actor_inputs_on_same_tick():
    worlds = spy_worlds()
    for tick, command in enumerate(((0.1, 0, 0), (-1.5, 1, -1), (1.5, -1, 1))):
        rows, _ = joint.step_worlds(worlds, command, (tick + 1) * single.POLICY_DT, 80)
        for w, row in zip(worlds, rows):
            assert w.sim.seen_commands[-1] == command
            assert row[6:9] == command
    assert worlds[0].sim.seen_commands == worlds[1].sim.seen_commands


def test_base_collision_counts_once_and_only_resets_failed_robot():
    worlds = spy_worlds()
    worlds[0].sim.contact = True
    rows, events = joint.step_worlds(worlds, (1, 0, 0), 0.02, 80)
    assert [w.collisions for w in worlds] == [1, 0]
    assert [w.episode for w in worlds] == [1, 0]
    assert worlds[1].sim.reset_reasons == []
    assert rows[0][-1] == 'base_contact'
    assert rows[0][4] == 0  # collision belongs to pre-reset episode
    assert len(events) == 1
    joint.step_worlds(worlds, (1, 0, 0), 0.04, 80)
    assert worlds[0].collisions == 1
    worlds[0].sim.contact = True
    joint.step_worlds(worlds, (1, 0, 0), 0.06, 80)
    assert worlds[0].collisions == 2  # new contact after reset counts again


def test_timeout_and_manual_reset_are_not_collisions_and_contact_takes_priority():
    worlds = spy_worlds()
    worlds[0].sim.contact = True
    _, events = joint.step_worlds(worlds, (0, 0, 0), 0.02, 0.02)
    assert [e[4] for e in events] == ['base_contact', 'timeout']
    assert [(w.collisions, w.timeouts) for w in worlds] == [(1, 0), (0, 1)]
    for world in worlds:
        world.reset('manual')
    assert [w.collisions for w in worlds] == [1, 0]
    assert [w.manual_resets for w in worlds] == [1, 1]


def test_real_base_contact_latches_even_if_only_one_physics_substep(monkeypatch):
    sim = single.Solo12Mujoco(Path(single.__file__).with_name('solo12.xml'))
    contacts = iter((False, True, False, False))
    monkeypatch.setattr(sim, '_base_hit_ground_now', lambda: next(contacts))
    sim.step(np.zeros(12))
    assert sim.base_hit_ground()  # short circuit uses latched contact, not last substep


def test_real_feet_on_floor_are_not_a_base_collision():
    sim = single.Solo12Mujoco(Path(single.__file__).with_name('solo12.xml'))
    had_foot_contact = False
    for _ in range(80):
        sim.step(np.zeros(12))
        had_foot_contact |= sim.data.ncon > 0
        assert not sim.base_hit_ground()
    assert had_foot_contact


@pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS), reason='Local supplied policies not installed')
def test_identical_policies_remain_bit_exact_with_independent_worlds():
    args, overrides, _, _ = arguments('env.kp=9', 'env.kd=0.2')
    args.checkpoints = [CHECKPOINTS[0], CHECKPOINTS[0]]
    worlds = joint.load_worlds(args, overrides)
    for i in range(80):
        command = (0.1, 0, 0) if i < 40 else (-0.4, 0.3, 0.2)
        joint.step_worlds(worlds, command, (i + 1) * single.POLICY_DT, 80)
        np.testing.assert_array_equal(worlds[0].sim.data.qpos, worlds[1].sim.data.qpos)
        np.testing.assert_array_equal(worlds[0].sim.action, worlds[1].sim.action)


@pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS), reason='Local supplied policies not installed')
def test_both_supplied_policies_headless_log_same_commands_and_collision_counts(tmp_path):
    args, overrides, ignored, ranges = arguments('--headless', '--track-cmds', '2 0 0;-2 0 0',
                                               '--command-period-s', '0.1', '--duration_s', '0.4',
                                               '--spawn-z', '0.01', '--output-dir', str(tmp_path))
    output = joint.run(args, overrides, ignored, ranges)
    rows = list(csv.DictReader((output / 'tracking.csv').open()))
    summary = json.loads((output / 'summary.json').read_text())
    assert len(rows) == 40
    assert {float(r['cmd_vx_mps']) for r in rows} == {-0.5, 0.5}  # clipped defaults
    for left, right in zip(rows[::2], rows[1::2]):
        assert left['time_s'] == right['time_s']
        assert left['cmd_vx_mps'] == right['cmd_vx_mps']
        assert left['cmd_vy_mps'] == right['cmd_vy_mps']
        assert left['cmd_wz_radps'] == right['cmd_wz_radps']
    assert summary['stopped'] == 'duration'
    assert all(w['base_collisions'] > 0 for w in summary['policies'])
    events = list(csv.DictReader((output / 'events.csv').open()))
    for i, stats in enumerate(summary['policies']):
        assert stats['base_collisions'] == sum(e['reason'] == 'base_contact' and int(e['policy_index']) == i for e in events)
    config = json.loads((output / 'run_config.json').read_text())
    assert [p['checkpoint_name'] for p in config['policies']] == [p.name for p in CHECKPOINTS]


@pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS), reason="Local supplied policies not installed")
def test_error_and_interrupt_save_partial_summary(tmp_path, monkeypatch):
    for exception, status in ((KeyboardInterrupt, 'interrupted'), (RuntimeError, 'error')):
        args, overrides, ignored, ranges = arguments('--headless', '--track-cmds', '0 0 0',
                                                   '--duration_s', '0.1', '--output-dir', str(tmp_path / status))
        # Use real worlds to exercise startup metadata, but interrupt before advancing.
        monkeypatch.setattr(joint, 'step_worlds', lambda *a: (_ for _ in ()).throw(exception()))
        if exception is RuntimeError:
            with pytest.raises(RuntimeError):
                joint.run(args, overrides, ignored, ranges)
        else:
            joint.run(args, overrides, ignored, ranges)
        summary = next((tmp_path / status).rglob('summary.json'))
        assert json.loads(summary.read_text())['stopped'] == status


@pytest.mark.parametrize('checkpoint', CHECKPOINTS)
@pytest.mark.skipif(not all(p.exists() for p in CHECKPOINTS), reason='Local supplied policies not installed')
def test_constant_command_rollout_bit_exact_with_existing_player(checkpoint):
    args, overrides, _, _ = arguments('env.kp=9', 'env.kd=0.2', 'env.joint_physical_limit_hip=[-70,70]',
                                     'env.joint_physical_limit_calf=[-179,179]')
    args.checkpoints = [checkpoint, checkpoint]
    worlds = joint.load_worlds(args, overrides)
    reference = single.Solo12Mujoco(Path(single.__file__).with_name('solo12.xml'), kp=9, kd=.2,
                                    env_overrides=overrides)
    policy = single.Policy(checkpoint)
    command = (.1, 0, 0)
    obs = reference.observation(command)
    for i in range(100):
        reference.step(policy(obs))
        if reference.base_hit_ground():
            reference.reset('base_contact')
        obs = reference.observation(command)
        joint.step_worlds(worlds, command, (i + 1) * single.POLICY_DT, 80)
        for world in worlds:
            np.testing.assert_array_equal(world.sim.data.qpos, reference.data.qpos)
            np.testing.assert_array_equal(world.sim.data.qvel, reference.data.qvel)
            np.testing.assert_array_equal(world.sim.action, reference.action)


@pytest.mark.skipif(os.environ.get('SOLO12_JOINT_GUI_TEST') != '1', reason='Opt-in desktop UI test')
def test_real_tk_and_glfw_controls_and_cleanup():
    args, overrides, _, ranges = arguments('env.command_lin_vel_x_range=[-1.5,1.5]', 'env.kp=9', 'env.kd=.2')
    worlds = joint.load_worlds(args, overrides)
    commands = joint.interactive.LiveCommandState(.1234, 0, 0)
    viewer = joint.JointViewer(worlds, args, commands, ranges)
    panel = None
    try:
        panel = joint.ControlPanel(commands, ranges, viewer, False)
        panel.poll()
        assert commands.get() == (.1234, 0, 0)
        commands.set_command(.3333, -.2222, .1111)
        panel.poll()
        panel.poll()
        assert commands.get() == (.3333, -.2222, .1111)
        commands.set_command(.1, 0, 0)
        panel.poll()
        panel.scales[0].set(1.25)
        panel.root.update()
        assert commands.get() == (1.25, 0, 0)
        panel.poll()
        assert commands.get() == (1.25, 0, 0)
        panel.buttons['Reverse command'].invoke()
        panel.poll()
        assert commands.get() == (-1.25, 0, 0)
        rows, _ = joint.step_worlds(worlds, commands.get(), .02, 80)
        assert rows[0][6:9] == rows[1][6:9] == (-1.25, 0, 0)
        panel.buttons['Zero command'].invoke()
        panel.poll()
        assert commands.get() == (0, 0, 0)
        panel.buttons['Reset both'].invoke()
        assert commands.consume_reset_request()
        panel.buttons['Pause / resume'].invoke()
        assert viewer.paused
        viewer.draw(commands.get(), .02, 1.0, advance=False)
        g = viewer.glfw
        viewer.on_key(viewer.window, g.KEY_UP, 0, g.PRESS, 0)
        panel.poll()
        assert commands.get() == (.05, 0, 0)
        panel.stop()
        assert not panel.poll()
    finally:
        if panel:
            panel.close()
        viewer.close()
