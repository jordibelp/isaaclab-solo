# Solo12 H1a: impact accelerations and a wrapping test

## Use with the existing command

Keep your existing `play_direct_mujoco.py` command and append **one** flag:

- `--log-imu-impacts`: measure without changing the policy inputs.
- `--ill-wrapping-test`: apply the suspected wrapping to acceleration inputs. This also enables logging.

Both work in the live viewer and in scripted headless playback. Neither is enabled
by default. Live runs save local diagnostics but do not upload to W&B. Scripted
runs include the diagnostics in the existing W&B upload unless `--no-wandb` is set.

For a fixed-command local A/B check, add this to each command as well:

```bash
--headless --track-cmds='0.1 0 0;0.1 0 0;0.1 0 0;0.1 0 0;0.1 0 0;0.1 0 0' --no-wandb
```

This runs **30 simulation seconds**: each scripted command lasts 5 seconds.
`--duration_s` controls live mode, not the length of a scripted sequence.
Keep the checkpoint, gains, limits, initial height and command sequence identical
between baseline and wrapping. Do not apply extra pushes to only one condition.

## What the measurements mean

The sensor is 1 cm ahead of the base-link origin. We sample its gravity-free
linear acceleration after every 5 ms physics step, including the tangential and
centripetal terms from its lever arm. Rest reads zero; free fall reads gravity.
This is **not** the raw accelerometer's specific force.

World Z shows vertical impacts. The transport limit, however, applies separately
to **sensor/body X, Y and Z**, not world Z or vector magnitude. When the base rears
up, vertical motion mainly appears on body X. Rotation can also make world Z
exceed 16 while every body component stays below 16. Read the axis overflow flags.

The probe uses the existing 200 Hz simulation clock. It does not resolve faster
hardware impacts, chassis vibration or the real IMU filter. A small simulated peak
does not rule out a larger real peak.

## Wrapping model and source evidence

The inspected local master-board sender uses `IMU_QN_ACC=11` and
`((int16_t)(value * (1 << 11)))`, with no clamp, for processed linear acceleration.
The packet uses signed 16-bit fields. The SDK decodes them by dividing by 2048,
without a gravity conversion for this processed signal. The raw accelerometer is
a different field and has different units. The local ODRI axis map is `[1,2,3]`.

Sources inspected:

- `firmware/components/uart_imu/uart_imu.c` in the local master-board repository.
- `firmware/main/defines.h` and SDK `protocol.h` / `master_board_interface.cpp`.
- ODRI `imu.cpp` and the bring-up `solo12.yaml` axis map.

**An out-of-range C float-to-integer cast does not guarantee modular wrapping.**
The deployed firmware/compiler behavior has not been verified. This flag tests
the proposed failure mechanism; it is not a calibrated hardware model.

The test truncates each body component toward zero after multiplying by 2048,
retains its low 16 bits, interprets them as signed, and divides by 2048. Thus +17
becomes -15 m/s² and -17 becomes +15 m/s². Multiple wraps are supported. The signed
transport range is -16 to +15.99951171875 m/s²; its step is 1/2048 m/s².

Wrapping happens **before** the existing observation clip and acceleration-history
insertion. Subsequent software clipping cannot recover the original acceleration.
The normal clip setting remains active, including its symmetric negative limit.
Setting `env.base_imu_clip=False` does not disable the wrapping test.

The flag changes only acceleration observations, including student TCN history.
It leaves gyro, gravity direction, dynamics, gains, actions and privileged teacher
acceleration unchanged. It includes Q11 quantization even inside the transport range.
Policies without an IMU-acceleration input reject the wrapping flag.

## Outputs

The terminal prints the output directory, first overflow and a summary every 5
simulation seconds. By default, diagnostics go under `logs/mujoco/imu_impacts`,
then the checkpoint name and a timestamp ending in `baseline` or `wrap`.
`--output-dir` changes the root. Scripted tracking files share that run directory.

- `imu_acceleration.csv`: every physics sample, with continuous run time, episode
  ID/time, clean world/body XYZ, normal-path and wrapped alternatives, delivered
  acceleration, overflow flags, loaded calf/foot–ground contacts and summed normal
  force, base–ground contact, base height and gravity X.
- `imu_summary.json`: signed world-Z extrema, body-axis peaks/counts, first overflow,
  overflow counts with foot contact or without base contact, and reset reasons.
- `imu_impacts.png`: world Z, body XYZ with transport boundaries, and contact force
  with overflow markers. Dotted vertical lines mark resets.
- `imu_config.json`: checkpoint/model hashes, revision, CLI, effective sensor/gain
  settings, and assumptions. Scripted runs also retain `run_config.json`.

CSV names include units and axes. All numeric CSV samples are logged as W&B
`imu/*` signals on `imu/time_s` during scripted upload; the PNG is only an extra.
CSV writes are streamed and flushed every 5 simulation seconds. Normal exit or
Ctrl+C closes the log and creates the summary/plot; a hard kill may lose buffered rows.
Reset clears histories and rebases velocity differences, so teleporting does not
create a false impact. Diagnostics alone leave observations and physics bit-identical.
Contact forces refer to the completed MuJoCo solve; acceleration uses the integrated
endpoint velocity difference over that same physics step. Contact counts are contact
points, not the number of distinct feet.

The clipped/wrapped alternatives on a plot use **that run's trajectory**. They do
not predict the trajectory of a separate closed-loop run. World versions of those
vectors are reconstructed from body axes with the simulator's true orientation.

## First bounded check, 14 November 2026

We ran the supplied `1111_cokdbj3y` checkpoint and supplied overrides in MuJoCo
3.3.7, headless, at command `(0.1, 0, 0)` for 30 seconds per condition. No W&B upload.
Both runs used 6,000 physics samples and had zero base-contact resets.

| Measurement | Normal path | Wrapping |
|---|---:|---:|
| Clean world-Z minimum / maximum, m/s² | -20.92 / +22.65 | -19.67 / +26.58 |
| Any-body-axis overflow samples | 24 | 28 |
| Overflow samples with loaded feet–ground contact | 18 | 21 |
| Overflow samples without base–ground contact | 24 | 28 |
| First overflow, s | 0.130 | 0.130 |
| Overflow samples during first second | 21 | 13 |
| Mean planar tracking error, m/s | 0.101 | 0.130 |

This establishes that the simulated signal crosses the range, including during
loaded foot contact, and that the test changes the rollout. **It does not reproduce
a fall in this short check or establish H1a as the hardware cause.** Most baseline
overflows happened during the initial landing/take-off, not steady hopping.

The player prints ignored Isaac-only arguments. In particular, this MuJoCo test
does not restore all training collision filters, terrain, noise, delays or domain
randomization from the pasted command. Soft-limit flags do not change saved actor
bounds. This is a one-factor test in this player, not full training/hardware parity.

## What would give stronger evidence?

1. Compare these two conditions over the same longer fixed command sequence. Check
   overflow timing **before** loss of balance, not only after a fall.
2. On hardware, record the processed linear-acceleration floats **before** the
   firmware conversion and decoded values after transport. This directly tests
   wrapping. The raw-accelerometer topic is useful supporting evidence, but is a
   different signal, not the pre-cast processed float.
3. Compare raw and processed signals only after accounting for axes, gravity,
   units, sample timing and filter behavior. Wrapped software values cannot be
   repaired by an extra clamp in the ROS driver.

A failed wrapped simulation is evidence of policy sensitivity, not proof of
hardware wrapping. A stable wrapped simulation does not rule out H1a: real impacts,
IMU timing and estimator errors may differ. Do not remove acceleration channels
from this 57-input checkpoint; a no-acceleration policy needs matching training.

## Verification

The new tests cover signed boundaries, multi-wrap behavior, an independent integer
bit oracle, upright frame mapping, world-Z versus body-axis limits, insertion before
clip/history, teacher isolation, reset/timestamp handling, unchanged logged rollouts,
all-sample W&B logging, and live-dispatch finalization on Ctrl+C.

A 30-second unflagged rollout of the actual checkpoint matched the unchanged HEAD
player bit-for-bit in observations, actions, positions and velocities.

The relevant regression run passed 122 tests. Two existing Isaac bring-up mock tests
were excluded after reproducing their same missing `control_mode` fixture failure
with the unchanged HEAD player. No Isaac task or hardware source was changed.
