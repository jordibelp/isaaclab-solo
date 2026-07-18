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
time. Plots are rendered with the Agg backend, so no GUI toolkit is touched in headless runs.
Outputs land in `logs/mujoco/cmd_tracking/<checkpoint_stem>/` by default:

- `command_tracking.csv` — Isaac columns plus `reset`, `base_height_m`, `gravity_x_b`
  (`gravity_x_b` ≈ 0 on four feet, ≈ −0.96 in the upright two-feet stance)
- `vxy_tracking_error.png`
- `wz_tracking_error.png` when any requested `wz` is nonzero

Purple dashed lines denote automatic resets caused by base-ground contact or episode timeout.

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
