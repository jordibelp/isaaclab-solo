# Solo12 IsaacLab → MuJoCo sim-to-sim

This folder runs the `solo12-two-feet` 48-observation RSL-RL policy in MuJoCo without importing Isaac Sim. It preserves the inference contract used by `play_direct_0325.py`:

- 200 Hz physics and 50 Hz policy (`dt=0.005`, decimation 4)
- joint order `FL, FR, RL, RR` with hip/thigh/calf per leg (`cfg.joint_names`)
- safe-pose action and observation offset, action scale `0.25`
- PhysX-style implicit PD via MuJoCo `<position>` actuators (`implicitfast` handles the kv term implicitly), force-limited to `±2.65 Nm`, armature `0.00036207 kg m²`
- observation velocities follow the IsaacLab contract exactly: `root_com_{lin,ang}_vel_b = R_link^T @ world COM velocity` (**not** `mj_objectVelocity(mjOBJ_BODY, local=1)`, which returns the inertial/ximat frame — that bug made the robot freeze in the crouch)
- RSL-RL observation normalization `(x - mean) / (std + 0.01)` and the checkpoint's deterministic actor
- planar tracking in the two-feet gravity-aligned base-footprint frame
- 5 s per requested command and matching CSV/error-plot format

## PD gains: play vs training

`play_direct_0325.py --disable_training_gain_sync` plays the **env cfg** gains, which for
`solo12-two-feet` are `kp=15, kd=0.5`. The checkpoint's training run (`q3a68133`, verified on W&B)
actually used `kp=9, kd=0.2`. The MuJoCo script defaults to `--kp 15 --kd 0.5` to reproduce the
Isaac play command byte-for-byte; pass `--kp 9 --kd 0.2` to play at training gains.

Measured on `0717_q3a68133_model_15008` (steady-state ‖Δv_xy‖ mean over the 8-command sequence):

| run | overall error | falls |
|---|---|---|
| Isaac play, kp15/kd0.5 | 0.149 m/s | 0 |
| MuJoCo, kp15/kd0.5 | 0.175 m/s | 1 |
| MuJoCo, kp9/kd0.2 (training) | 0.139 m/s | 0 |

## Model matching

`solo12.xml` is a lightweight, self-contained MJCF cross-checked field-by-field against the active
`SoloFlat.usd` (joints dumped with `pxr`): joint axes (+x hips, +y thighs/calves on both sides),
anchor positions, the modified asymmetric limits (front thigh `[-250°, 90°]`, rear `[-90°, 250°]`,
hips `±179°`, calves `[-179°, 180°]`), link masses/inertias, and the authored base COM `(-0.01, 0, 0)`.
Total mass is `3.028572 kg` (base `1.75124` per `cfg.base_mass`).

Collision topology mirrors the USD exactly — hips and leg shafts have **no** colliders:

- base: `collision_main` box + the two `borde` side-rail boxes
- thigh: `collider_top` box + `collider_bot` knee sphere (r=0.0371)
- calf: the foot contact **cylinder** (r=0.0186, half-width 0.0063, axis y) — not a sphere
- capsules/spheres in the visual group are display-only (`contype=0`)

MuJoCo and PhysX remain different contact/solver implementations. Matching parameters does not make
them the same simulator; the remaining difference is what this sim-to-sim experiment measures.

## Run

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Then run the same experiment (Isaac-only Hydra overrides are accepted and reported as ignored):

```bash
./isaaclab.sh -p mujoco/play_direct_mujoco.py \
  --task="solo12-two-feet" --num_envs 1 --duration_s 2000 \
  --cmd_init 0.1 0.0 0.0 --episode_length_s 80 \
  --checkpoint="/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt" \
  --track-cmds="0.5 0. 0.;-0.5 0. 0.;0. 0.3 0.;0. -0.3 0.;0.5 0.3 0.;-0.5 -0.3 0.;+0.5 -0.3 0.;-0.5 +0.3 0." \
  --headless
```

Remove `--headless` for the interactive MuJoCo viewer (on Wayland sessions the script switches GLFW
to XWayland automatically to avoid the GLFW/GLib warning spam). Add `--realtime` to pace it at real
time. MuJoCo's left and right helper panels start hidden; pass `--show-viewer-ui` to restore them.
Plots are rendered with the Agg backend, so no GUI toolkit is touched in headless runs.

`env.kp=…`, `env.kd=…`, and `env.command_*_range=[lo,hi]` from the Isaac command line are honored
(explicit `--kp/--kd` win if both forms are present); all other `env.*`/Isaac-only flags are reported
and ignored. Duplicate equal gains are harmless and reported; conflicting duplicates produce a warning.

## Live mode (sliders, forces, cameras)

Omit `--track-cmds` (and `--headless`) to drive the robot interactively, mirroring the Isaac play UI:

```bash
./isaaclab.sh -p mujoco/play_direct_mujoco.py \
  --checkpoint="/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt" \
  --cmd_init 0.1 0.0 0.0 env.kp=9.0 env.kd=0.2
```

- **Command sliders** (tkinter panel): vx/vy/wz clipped to the command ranges
  (defaults ±0.5/±0.3/±0.5, widened by `env.command_*_range`), plus Zero cmd / Reset robot.
- **Base force UI**: body-frame force from magnitude/azimuth/elevation sliders, applied at a
  selectable point (front/rear/left/right/top/center of the base) with Isaac-style
  **Pulse** (for the chosen duration) / **Hold** / **Release**; a red arrow in the viewer shows the
  active force (`--force-ui-max` caps the magnitude slider, default 10 N).
- **Cameras**: `side` and `front` follow cameras use the same EMA-smoothed, heading-stable anchor
  as the IsaacLab chase cameras (safe at ~90° pitch), plus the `free` mouse camera. Start mode via
  `--camera {side,front,free}` (default side); cycle with `C`. The follow cameras also work in
  `--track-cmds` viewer runs.
- **Keyboard** (works without the panel): arrows = vx/wz, `A/D` = vy, `SPACE` = zero command,
  `R` = reset robot, `C` = cycle camera, `F` = release force.

Live mode always paces at wall-clock speed and records no artifacts; close the viewer to stop.
Every execution gets an immutable timestamped folder under
`logs/mujoco/cmd_tracking/<checkpoint_stem>/<YYYYMMDD_HHMMSS>/` by default:

- `run_config.json` — checkpoint hash, commands, ignored Isaac overrides, timing, gains, model mass,
  MuJoCo version, and Git revision
- `summary.json` — aggregate tracking/fall/stance metrics
- `command_tracking.csv` — Isaac columns plus `reset`, `base_height_m`, `gravity_x_b`
  (`gravity_x_b` ≈ 0 on four feet, ≈ −0.96 in the upright two-feet stance)
- `vxy_tracking_error.png`
- `wz_tracking_error.png` when any requested `wz` is nonzero

Purple dashed lines denote automatic resets caused by base-ground contact or episode timeout.

Completed runs are uploaded by default to the W&B project `solo12-two-feet-exp`, including the
configuration, summary, CSV, plots, and exact MJCF model as one versioned artifact. Useful options:

```bash
--wandb-project solo12-two-feet-exp  # default
--wandb-entity <entity>
--wandb-name <custom-run-name>
--no-wandb                           # local artifacts only
--output-dir <custom-output-root>    # model/timestamp folders are still appended
```

## Tests

```bash
cd mujoco && python -m pytest test_mujoco_sim2sim.py -q
```

Covers the model contract, PD/actuator limits, a finite-difference verification of the body-frame
velocity observation, the obs slot layout, normalizer parity with rsl_rl, and a 10 s real-checkpoint
rollout asserting the robot reaches the upright stance and walks.

## Dependency

Tested in `env_isaaclab` with MuJoCo 3.3.7 and Python 3.11. Install with:

```bash
python -m pip install 'mujoco>=3.3,<3.4'
```
