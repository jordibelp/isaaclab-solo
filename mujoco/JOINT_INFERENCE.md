# Two policies, one command

## Start the supplied comparison

Run this from any terminal. The launcher activates `env_isaaclab` and sets the usual certificates.

```bash
bash /home/jordibelp/IsaacLab/mujoco/compare_jd6b4err_nf3m49ip.sh
```

- **Left:** `1113_jd6b4err_best_model_curriculum_idx_2_walk.pt`.
- **Right:** `0716_nf3m49ip_best_model.pt`.
- Both run in separate MuJoCo worlds, in one window with two side-by-side views.
- The small **Shared commands** window controls both robots. Move it beside the viewer if needed.
- The three sliders change forward velocity `vx` [m/s], sideways velocity `vy` [m/s], and yaw rate `wz` [rad/s].
- One command is sampled before both actor calls, every 20 ms of simulation time. There is no command smoothing or extra one-step command delay. This deliberately differs from the old player's command-update timing at command changes.
- Each world uses the existing player's physics, actor loader, normalizer, action bounds and PD control. Physics runs at 200 Hz; policies run at 50 Hz.
- GUI playback is paced to wall time. If rendering is too slow, simulation slows down together for both policies. The overlay shows simulated time and the average simulation/wall-time ratio, including pauses.

The launcher keeps the effective supported settings from your commands: gains 9/0.2,
effort cap 2.65 Nm, no bring-up clipping, clean non-IMU observations, hip stops ±70 degrees,
calf stops ±179 degrees, and command ranges ±1.5 / ±1 / ±1.

**Existing-player limitation preserved:** `use_asymmetric_thigh_limits=False` means your
front/rear thigh-stop overrides are inactive. The XML thigh limits remain in use, just
as in your original MuJoCo commands. Soft-limit deltas and network-width overrides do
not modify checkpoint-saved actor scaling or network sizes. The launcher omits ignored
Isaac-only curriculum, terrain, gain-sync, extra-mass and collision-filter flags.
The existing XML defines flat ground and collisions; this tool does not make those
ignored flags effective. Inspect each run's resolved configuration before comparing
against a different training or hardware setup.

## Controls

| Control | Effect |
| --- | --- |
| Three sliders | Apply the selected command to both policies on the next policy tick |
| Zero command | Set all three axes to zero; this is **not** a physics stop |
| Reverse command | Flip all three signs at once, without a ramp |
| Reset both | Reset both robot poses and episode timers; keep collision counters |
| Pause / resume | Freeze both simulations; commands can still be changed |
| Side / front / free camera | Choose the camera mode for both views |
| Close either window | Stop both simulations and save the partial session |

With the **MuJoCo viewer focused**: Up/Down changes `vx` by 0.05 m/s; Left/Right changes
`wz` by ±0.1 rad/s; A/D changes `vy` by ±0.05 m/s. Space zeros all commands, R resets
both, P pauses, C cycles cameras, and Escape exits. In free-camera mode, drag with the
left mouse button to rotate, right button to move, or scroll to zoom in either view.

## Base collision counters

Each policy has its own counter, shown above its robot and in the control window.
A collision means **a collision geom on the base touches the ground** during any
of the four physics substeps. Several contact points or a sustained contact within
that policy step count as one event. A brief contact between rendered frames still counts.

On contact we increment the counter and immediately reset **only that robot**.
The other robot continues. This matches the old player's base-contact reset rule.
These are base-contact/reset counts, not all possible forms of failure or contacts.
Feet/legs touching the floor and self-contacts do not count. A later contact after a
reset counts again. The counters are cumulative for the session and reset when you
start a new session.

Episode timeouts (80 seconds per robot by default) and manual resets are separate.
If a base contact coincides with a timeout, it counts as a base contact only.
Both policies always receive the same command, but their episode ages differ after
independent falls. Use **Reset both** to begin another matched trial.

## Local recording

Every run creates a new directory below:

```text
/home/jordibelp/IsaacLab/logs/mujoco/joint_inference/comparison/<timestamp>/
```

- `tracking.csv`: every policy tick for both policies, with the actual applied command,
  measured velocity, base height, episode number, base-contact flag and cumulative count.
  Velocities follow the existing player's gravity-aligned base-footprint tracking frame;
  yaw rate is world-Z angular velocity.
- `events.csv`: each base-contact reset, timeout and manual reset, with simulation time,
  policy identity, episode and command. Collision events identify the completed 20 ms
  policy interval; timestamps are not exact 5 ms contact-onset times.
- `run_config.json`: full checkpoint paths, filename-only names, SHA256 hashes, model hash,
  gains, joint limits, observation settings, command ranges, overrides and code revision.
- `summary.json`: final counters, simulated duration and stop reason.

Samples are recorded before automatic resets, so they describe the state that caused
the event. Files are flushed every simulated second and on automatic resets; normal
window closure and Ctrl+C save a partial summary. No W&B upload, video recording,
cluster work, hardware motion, or change to the old player is involved.

## Repeatable fast command changes

This headless example repeats seven commands with a 0.25-second dwell for 20 seconds:

```bash
bash /home/jordibelp/IsaacLab/mujoco/compare_jd6b4err_nf3m49ip.sh --headless --track-cmds='1.2 0 0;-1.2 0 0;0 0.8 0;0 -0.8 0;0 0 1;0 0 -1;0 0 0' --command-period-s=0.25 --duration_s=20
```

Omit `--headless` to watch it. Scripted mode disables the sliders and command buttons.
The viewer keyboard cannot override the scheduled command. Pause and resets still work.
The duration is simulation time in both modes; headless mode runs as fast as possible
unless you add `--realtime`. Commands outside the configured ranges are clipped.

For other plain-MLP PPO/SAC policies, use `play_joint_mujoco.py --checkpoints LEFT RIGHT`
with the same supported observation/actuator overrides as `play_direct_mujoco.py`.
Both checkpoints must use the same task and observation settings. Encoder/TCN tasks
and torque policies are not supported by this comparison player.

## Verification on 2026-11-14

- Both supplied checkpoints load as 48-input PPO actors.
- 140 CPU regression checks passed; the optional native GUI check also passed (20 joint-player checks including GUI).
- A 12-second scripted GUI run and headless run produced byte-identical tracking and event CSVs (1,200 tracking rows).
- Constant-command rollouts are bit-exact against the existing player for each checkpoint.
- Same-checkpoint pairs remain bit-exact across a command jump.
- Tests cover current-command delivery, clipping, collision counting, timeout priority,
  independent resets, partial output on interrupt/error, and physics-substep contacts.
- Actual Tk slider callbacks and buttons, GLFW keyboard controls, pause, cameras and
  window-close saving were exercised on the laptop. A forced left-base collision
  incremented only the left counter and left the right robot running.
- The 20-second headless sequence above produced 0 base contacts for jd6b4err and 3
  for nf3m49ip. This is one bounded smoke test, **not a general robustness ranking**.
  The forced GUI collision is a separate wiring test and must not be read as policy performance.

CPU checks, after activating `env_isaaclab`:

```bash
cd /home/jordibelp/IsaacLab && python -m pytest -q mujoco/test_joint_mujoco.py --confcutdir=mujoco --rootdir=mujoco
```

The optional real-window test requires a valid desktop display:

```bash
cd /home/jordibelp/IsaacLab && SOLO12_JOINT_GUI_TEST=1 python -m pytest -q mujoco/test_joint_mujoco.py -k real_tk --confcutdir=mujoco --rootdir=mujoco
```
