#!/usr/bin/env python3
"""Compare two Solo12 policies in independent MuJoCo worlds with shared commands."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

import interactive
import play_direct_mujoco as single


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs=2, type=Path, required=True, metavar=('LEFT', 'RIGHT'))
    parser.add_argument('--labels', nargs=2, default=('Policy A', 'Policy B'))
    parser.add_argument('--task', choices=('solo12-two-feet', 'solo12-v0'), default='solo12-two-feet')
    parser.add_argument('--num_envs', type=int, choices=(1,), default=1, help='One independent robot per policy.')
    parser.add_argument('--duration_s', type=float, default=2000.0, help='Total simulated seconds, including scripted mode.')
    parser.add_argument('--episode_length_s', type=float, default=80.0)
    parser.add_argument('--cmd_init', nargs=3, type=float, default=(0.1, 0.0, 0.0))
    parser.add_argument('--spawn-z', type=float, default=0.35)
    parser.add_argument('--kp', type=float)
    parser.add_argument('--kd', type=float)
    parser.add_argument('--bringup_clip_rad', '--bringup-clip-rad', type=float, default=0.0)
    parser.add_argument('--camera', choices=interactive.CAMERA_MODES, default='side')
    parser.add_argument('--output-dir', type=Path, default=Path('logs/mujoco/joint_inference'))
    parser.add_argument('--track-cmds', type=single.parse_commands, help='Repeat this command sequence; e.g. "1 0 0;-1 0 0".')
    parser.add_argument('--command-period-s', type=float, default=0.5, help='Scripted command dwell in simulation seconds.')
    parser.add_argument('--headless', action='store_true', help='No windows; requires --track-cmds.')
    parser.add_argument('--realtime', action='store_true', help='Also pace headless runs; GUI is always paced.')
    parser.add_argument('--width', type=int, default=1500)
    parser.add_argument('--height', type=int, default=700)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides, ignored = single.consume_env_overrides(unknown)
    for name in ('duration_s', 'episode_length_s', 'command_period_s', 'spawn_z'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be finite and positive')
    if not math.isfinite(args.bringup_clip_rad) or args.bringup_clip_rad < 0:
        parser.error('--bringup_clip_rad must be finite and nonnegative')
    if not all(math.isfinite(v) for v in args.cmd_init):
        parser.error('--cmd_init must be finite')
    if args.track_cmds and not np.isfinite(args.track_cmds).all():
        parser.error('--track-cmds must be finite')
    if args.headless and not args.track_cmds:
        parser.error('--headless requires --track-cmds')
    if args.width < 640 or args.height < 360:
        parser.error('Window must be at least 640 x 360')
    ranges = tuple(overrides.get(key.removeprefix('env.'), single.DEFAULT_COMMAND_RANGES[i])
                   for key, i in single._ENV_RANGE_KEYS.items())
    if any(len(r) != 2 or not np.isfinite(r).all() or r[0] >= r[1] for r in ranges):
        parser.error('Command ranges must have finite lower < upper limits')
    defaults = (9.0, 0.2) if args.task == 'solo12-v0' else (single.DEFAULT_KP, single.DEFAULT_KD)
    args.kp = args.kp if args.kp is not None else overrides.get('kp', defaults[0])
    args.kd = args.kd if args.kd is not None else overrides.get('kd', defaults[1])
    if not all(math.isfinite(v) and v >= 0 for v in (args.kp, args.kd)):
        parser.error('kp and kd must be finite and nonnegative')
    return args, overrides, ignored, ranges


@dataclass
class PolicyWorld:
    label: str
    checkpoint: Path
    policy: single.Policy
    sim: single.Solo12Mujoco
    elapsed: float = 0.0
    collisions: int = 0
    timeouts: int = 0
    manual_resets: int = 0
    episode: int = 0
    last_collision_s: float | None = None

    def reset(self, reason):
        self.sim.reset(reason=reason)
        self.elapsed = 0.0
        self.episode += 1
        if reason == 'manual':
            self.manual_resets += 1


def load_worlds(args, overrides):
    worlds = []
    for label, checkpoint in zip(args.labels, args.checkpoints):
        checkpoint = checkpoint.expanduser().resolve()
        policy = single.Policy(checkpoint)
        sim = single.Solo12Mujoco(Path(single.__file__).with_name('solo12.xml'), kp=args.kp, kd=args.kd,
                                  spawn_z=args.spawn_z, env_overrides=overrides,
                                  bringup_clip_rad=args.bringup_clip_rad)
        size = sim.observation(tuple(args.cmd_init)).size
        if size != policy.input_dim:
            raise ValueError(f'{label}: checkpoint expects {policy.input_dim} observations, player supplies {size}. '
                             'Match the observation settings used in training.')
        worlds.append(PolicyWorld(label, checkpoint, policy, sim))
        print(f'[INFO] {label}: {checkpoint.name}, {policy.checkpoint_format}, {size} inputs; '
              f'kp={sim.kp:g}, kd={sim.kd:g}, effort={sim.effort_limit:g} Nm', flush=True)
    return worlds


def step_worlds(worlds, command, end_time_s, episode_length_s):
    """Sample one command for both actors, log pre-reset outcomes, then reset only failures.

    The existing simulator latches base/ground contact across all four physics ticks.
    An immediate reset ends that contact episode, so sustained contact or several base
    geoms in the same tick increment the collision counter only once.
    """
    actions = [world.policy(world.sim.observation(command)) for world in worlds]
    samples, events = [], []
    for index, (world, action) in enumerate(zip(worlds, actions)):
        sim = world.sim
        sim.step(action)
        world.elapsed += single.POLICY_DT
        hit = sim.base_hit_ground()
        timeout = world.elapsed + 1e-10 >= episode_length_s
        reason = 'base_contact' if hit else ('timeout' if timeout else '')
        if hit:
            world.collisions += 1
            world.last_collision_s = end_time_s
        elif timeout:
            world.timeouts += 1
        samples.append((end_time_s, index, world.label, world.checkpoint.name, world.episode,
                        world.elapsed, *command, *sim.tracked_velocity(), sim.base_height(),
                        int(hit), world.collisions, reason))
        if reason:
            events.append((end_time_s, index, world.label, world.episode, reason, world.collisions, *command))
            world.reset(reason)
    return samples, events


class JointViewer:
    """One GLFW window with two native MuJoCo viewports and independent cameras."""
    def __init__(self, worlds, args, commands, ranges):
        single.prepare_viewer_environment()
        import glfw
        self.glfw = glfw
        self.window = None
        self.contexts = []
        if not glfw.init():
            raise RuntimeError('Cannot initialize GLFW. Run from a desktop session or use --headless.')
        try:
            self.window = glfw.create_window(args.width, args.height, 'Solo12 | joint policy comparison', None, None)
            if not self.window:
                raise RuntimeError('Cannot open the MuJoCo comparison window')
            glfw.make_context_current(self.window)
            glfw.swap_interval(0)
            self.worlds, self.commands, self.ranges = worlds, commands, ranges
            self.paused = False
            self.cameras = [mujoco.MjvCamera() for _ in worlds]
            self.follow = [interactive.FollowCamera(single.POLICY_DT, args.camera) for _ in worlds]
            self.scenes = [mujoco.MjvScene(w.sim.model, maxgeom=2000) for w in worlds]
            self.option = mujoco.MjvOption()
            self.option.geomgroup[3] = False  # invisible collision proxies; retain visual meshes
            for world in worlds:
                self.contexts.append(mujoco.MjrContext(world.sim.model, mujoco.mjtFontScale.mjFONTSCALE_150))
            self.last_mouse = (0.0, 0.0)
            glfw.set_key_callback(self.window, self.on_key)
            glfw.set_cursor_pos_callback(self.window, self.on_mouse)
            glfw.set_scroll_callback(self.window, self.on_scroll)
        except BaseException:
            self.close()
            raise

    def set_camera(self, mode):
        for follow in self.follow:
            follow.mode = mode

    def on_key(self, window, key, scancode, action, mods):
        g = self.glfw
        if action not in (g.PRESS, g.REPEAT):
            return
        if key == g.KEY_ESCAPE:
            g.set_window_should_close(window, True)
        elif key == g.KEY_P and action == g.PRESS:
            self.paused = not self.paused
        elif key == g.KEY_C and action == g.PRESS:
            mode = self.follow[0].cycle_mode()
            self.set_camera(mode)
        else:
            # Clip the stored state too, so reversing after a limit responds immediately.
            interactive.make_key_callback(self.commands, None, self.follow[0])(key)
            self.commands.set_command(*self.commands.get_clipped(self.ranges))

    def mouse_index(self):
        x, _ = self.glfw.get_cursor_pos(self.window)
        width, _ = self.glfw.get_window_size(self.window)
        return int(x >= width / 2)

    def on_mouse(self, window, x, y):
        g = self.glfw
        dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
        self.last_mouse = x, y
        index = self.mouse_index()
        if self.follow[index].mode != 'free':
            return
        left = g.get_mouse_button(window, g.MOUSE_BUTTON_LEFT) == g.PRESS
        right = g.get_mouse_button(window, g.MOUSE_BUTTON_RIGHT) == g.PRESS
        if not (left or right):
            return
        move = mujoco.mjtMouse.mjMOUSE_MOVE_V if right else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        _, height = g.get_window_size(window)
        mujoco.mjv_moveCamera(self.worlds[index].sim.model, move, dx / max(height, 1), dy / max(height, 1),
                             self.scenes[index], self.cameras[index])

    def on_scroll(self, window, xoffset, yoffset):
        index = self.mouse_index()
        if self.follow[index].mode == 'free':
            mujoco.mjv_moveCamera(self.worlds[index].sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                 0, -0.05 * yoffset, self.scenes[index], self.cameras[index])

    def poll(self):
        self.glfw.poll_events()
        return not self.glfw.window_should_close(self.window)

    def draw(self, command, sim_time_s, realtime_factor, advance=True):
        g = self.glfw
        width, height = g.get_framebuffer_size(self.window)
        if width < 2 or height < 1:
            return
        for i, world in enumerate(self.worlds):
            sim, camera, follow = world.sim, self.cameras[i], self.follow[i]
            if advance or follow._anchor_w is None:
                follow.update(sim.data.qpos[:3], sim.data.qpos[3:7])
            follow.apply(camera)
            if camera.distance == 0:
                mujoco.mjv_defaultFreeCamera(sim.model, camera)
            viewport = mujoco.MjrRect(i * (width // 2), 0, width // 2 if i == 0 else width - width // 2, height)
            mujoco.mjv_updateScene(sim.model, sim.data, self.option, None, camera,
                                  mujoco.mjtCatBit.mjCAT_ALL, self.scenes[i])
            context = self.contexts[i]
            mujoco.mjr_render(viewport, self.scenes[i], context)
            recent = world.last_collision_s is not None and sim_time_s - world.last_collision_s < 1.0
            title = (f'{world.label}\n{world.checkpoint.name}\n'
                     f'BASE COLLISIONS: {world.collisions}' + ('  [CONTACT]' if recent else '') +
                     f'\nTimeouts: {world.timeouts} | Manual resets: {world.manual_resets}\n'
                     f'cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f}')
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, title, '', context)
            status = f't={sim_time_s:.2f}s | {realtime_factor:.2f}x real time' + (' | PAUSED' if self.paused else '')
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               viewport, status, 'P pause | R reset both | SPACE zero | C camera | ESC quit', context)
        g.swap_buffers(self.window)

    def close(self):
        for context in self.contexts:
            context.free()
        if self.window:
            self.glfw.destroy_window(self.window)
        self.glfw.terminate()


class ControlPanel:
    """Tk and GLFW are both pumped on the main thread; no shared model or UI threads."""
    def __init__(self, commands, ranges, viewer, scripted):
        import tkinter as tk
        self.tk, self.commands, self.viewer = tk, commands, viewer
        self.root = tk.Tk()
        self.root.title('Shared commands | both policies')
        self.closed = False
        self.root.protocol('WM_DELETE_WINDOW', self.stop)
        self.root.geometry('+35+65')
        tk.Label(self.root, text='ONE COMMAND → BOTH POLICIES', font=('TkDefaultFont', 12, 'bold')).pack(pady=8)
        tk.Label(self.root, text='Applied together at the next 50 Hz tick. No smoothing.').pack()
        self.variables, self.scales = [], []
        for i, (name, unit, limits) in enumerate(zip(('vx', 'vy', 'wz'), ('m/s', 'm/s', 'rad/s'), ranges)):
            row = tk.Frame(self.root); row.pack(fill='x', padx=8)
            tk.Label(row, text=f'{name} [{unit}]', width=10).pack(side='left')
            var = tk.DoubleVar(value=commands.get()[i])
            scale = tk.Scale(row, from_=limits[0], to=limits[1], resolution=0.01, orient='horizontal',
                             length=290, variable=var)
            scale.pack(side='left')
            if scripted:
                scale.configure(state='disabled')
            self.variables.append(var); self.scales.append(scale)
        # Scale rounds its display to 0.01. Flush initialization before attaching
        # callbacks so an exact initial/scripted command is never rounded in the model.
        self.root.update_idletasks()
        for name, scale in zip(('vx', 'vy', 'wz'), self.scales):
            scale.configure(command=lambda value, axis=name: commands.set_command(**{axis: float(value)}))
        self.shown_command = commands.get()
        row = tk.Frame(self.root); row.pack(pady=8)
        self.buttons = {}
        for text, callback in (('Zero command', lambda: commands.set_command(0, 0, 0)),
                               ('Reverse command', lambda: commands.set_command(*(-v for v in commands.get()))),
                               ('Reset both', commands.request_reset), ('Pause / resume', self.toggle_pause)):
            button = tk.Button(row, text=text, command=callback)
            button.pack(side='left', padx=3)
            self.buttons[text] = button
            if scripted and text in ('Zero command', 'Reverse command'):
                button.configure(state='disabled')
        row = tk.Frame(self.root); row.pack()
        for mode in interactive.CAMERA_MODES:
            tk.Button(row, text=f'{mode.title()} camera', command=lambda m=mode: viewer.set_camera(m)).pack(side='left', padx=3)
        self.stats = tk.Label(self.root, justify='left', font=('TkFixedFont', 10), padx=12, pady=12)
        self.stats.pack(fill='x')
        tk.Label(self.root, text=('Scripted sequence (sliders disabled)' if scripted else
                 'Viewer keys: ↑/↓ vx · ←/→ wz · A/D vy · SPACE zero') +
                 '\nP pause · R reset BOTH (counts stay) · C camera · ESC quit\n'
                 'Base touching ground = +1 and reset only that robot.\n'
                 'Feet/legs/self-contacts and timeouts do not count.\nClosing either window stops and saves the session.', fg='#555').pack(padx=10, pady=6)

    def toggle_pause(self):
        self.viewer.paused = not self.viewer.paused

    def stop(self):
        self.closed = True

    def poll(self):
        # Synchronize keyboard/script changes without Scale callbacks writing stale values.
        command = self.commands.get()
        if command != self.shown_command:
            callbacks = [scale.cget('command') for scale in self.scales]
            for var, scale, value in zip(self.variables, self.scales, command):
                scale.configure(command='')
                var.set(value)
            self.root.update_idletasks()  # drain deferred Scale callbacks while disabled
            for scale, callback in zip(self.scales, callbacks):
                scale.configure(command=callback)
            self.shown_command = command
        self.root.update_idletasks()
        self.root.update()
        return not self.closed

    def update_stats(self, worlds, sim_time_s):
        self.stats.configure(text='\n'.join(
            f'{w.label}: {w.collisions} base collisions | {w.timeouts} timeouts' for w in worlds) +
            f'\nSimulation time: {sim_time_s:.2f} s' + ('  PAUSED' if self.viewer.paused else ''))

    def close(self):
        self.root.destroy()


SAMPLE_FIELDS = ('time_s', 'policy_index', 'label', 'checkpoint_name', 'episode', 'episode_time_s',
                 'cmd_vx_mps', 'cmd_vy_mps', 'cmd_wz_radps', 'actual_vx_mps', 'actual_vy_mps',
                 'actual_wz_radps', 'base_height_m', 'base_contact', 'base_collision_count', 'reset_reason')
EVENT_FIELDS = ('time_s', 'policy_index', 'label', 'episode', 'reason', 'base_collision_count',
                'cmd_vx_mps', 'cmd_vy_mps', 'cmd_wz_radps')


def run(args, overrides, ignored, ranges):
    torch.set_num_threads(1)  # batch-one CPU actors; avoid thread-pool overhead
    worlds = load_worlds(args, overrides)
    commands = interactive.LiveCommandState(*args.cmd_init)
    commands.set_command(*commands.get_clipped(ranges))
    output = single.create_run_directory(args.output_dir, 'comparison', datetime.now().strftime('%Y%m%d_%H%M%S'))
    config = dict(argv=sys.argv, task=args.task, physics_dt_s=single.PHYSICS_DT, policy_dt_s=single.POLICY_DT,
                  command_timing='Same current command sampled before both actor calls; no one-step command lag.',
                  collision_definition='Any base collision geom touching ground during a physics substep; one count then immediate independent reset.',
                  episode_length_s=args.episode_length_s, duration_s=args.duration_s, command_ranges=ranges,
                  cmd_init=commands.get(), track_cmds=args.track_cmds, command_period_s=args.command_period_s,
                  env_overrides=overrides, ignored_isaaclab_overrides=ignored, mujoco_version=mujoco.__version__,
                  git_revision=single.git_revision(), model_sha256=single.checkpoint_sha256(Path(single.__file__).with_name('solo12.xml')),
                  policies=[dict(label=w.label, checkpoint=str(w.checkpoint), checkpoint_name=w.checkpoint.name,
                                 sha256=single.checkpoint_sha256(w.checkpoint), format=w.policy.checkpoint_format,
                                 input_dim=w.policy.input_dim, kp=w.sim.kp, kd=w.sim.kd, spawn_z=w.sim.spawn_z,
                                 effort_limit_nm=w.sim.effort_limit, bringup_clip_rad=w.sim.bringup_clip_rad,
                                 joint_limits_rad=w.sim.model.jnt_range.tolist(),
                                 observation=dict(base_obs_from_imu=w.sim.base_obs_from_imu,
                                                  include_root_lin_vel=w.sim.include_root_lin_vel,
                                                  imu_acc_steps=w.sim.imu_acc_steps),
                                 action_scale=single.ACTION_SCALE, joint_names=single.JOINT_NAMES) for w in worlds])
    (output / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'[INFO] All samples and collision/reset events saved locally: {output}', flush=True)
    if ignored:
        print('[INFO] Ignoring IsaacLab-only overrides:', ' '.join(ignored), flush=True)
    viewer = panel = None
    steps, stopped = 0, 'duration'
    try:
        with (output / 'tracking.csv').open('w', newline='') as sample_file, (output / 'events.csv').open('w', newline='') as event_file:
            samples, events = csv.writer(sample_file), csv.writer(event_file)
            samples.writerow(SAMPLE_FIELDS); events.writerow(EVENT_FIELDS)
            if not args.headless:
                viewer = JointViewer(worlds, args, commands, ranges)
                panel = ControlPanel(commands, ranges, viewer, args.track_cmds is not None)
            start_wall = time.perf_counter()
            while steps < int(math.ceil(args.duration_s / single.POLICY_DT)):
                tick_start = time.perf_counter()
                if viewer and (not viewer.poll() or not panel.poll()):
                    stopped = 'window_closed'; break
                if args.track_cmds:
                    index = int((steps * single.POLICY_DT + 1e-10) / args.command_period_s) % len(args.track_cmds)
                    commands.set_command(*args.track_cmds[index])
                command = commands.get_clipped(ranges)
                commands.set_command(*command)
                advance = viewer is None or not viewer.paused
                if advance:
                    if commands.consume_reset_request():
                        for i, w in enumerate(worlds):
                            events.writerow((steps * single.POLICY_DT, i, w.label, w.episode, 'manual', w.collisions, *command))
                            w.reset('manual')
                    rows, reset_events = step_worlds(worlds, command, (steps + 1) * single.POLICY_DT, args.episode_length_s)
                    samples.writerows(rows); events.writerows(reset_events)
                    steps += 1
                    if reset_events or steps % 50 == 0:
                        sample_file.flush(); event_file.flush()
                    for event in reset_events:
                        print(f'[EVENT] t={event[0]:.2f}s {event[2]}: {event[4]}, base collisions={event[5]}', flush=True)
                sim_time = steps * single.POLICY_DT
                if viewer:
                    factor = sim_time / max(time.perf_counter() - start_wall, 1e-9)
                    viewer.draw(command, sim_time, factor, advance)
                    panel.update_stats(worlds, sim_time)
                if viewer or args.realtime:
                    time.sleep(max(0, single.POLICY_DT - (time.perf_counter() - tick_start)))
    except KeyboardInterrupt:
        stopped = 'interrupted'
    except BaseException:
        stopped = 'error'
        raise
    finally:
        if panel:
            panel.close()
        if viewer:
            viewer.close()
        summary = dict(simulated_s=steps * single.POLICY_DT, policy_steps=steps, stopped=stopped,
                       policies=[dict(label=w.label, checkpoint_name=w.checkpoint.name, base_collisions=w.collisions,
                                      timeouts=w.timeouts, manual_resets=w.manual_resets) for w in worlds])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('[RESULT]', json.dumps(summary), flush=True)
    return output


def main():
    run(*parse_args())


if __name__ == '__main__':
    main()
