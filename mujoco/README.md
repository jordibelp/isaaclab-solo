# Solo12 IsaacLab → MuJoCo sim-to-sim

This folder runs the `solo12-two-feet` 48-observation RSL-RL policy in MuJoCo without importing Isaac Sim. It preserves the inference contract used by `play_direct_0325.py`:

- 200 Hz physics and 50 Hz policy (`dt=0.005`, decimation 4)
- joint order `FL, FR, RL, RR`, with hip/thigh/calf per leg
- safe-pose action and observation offset, action scale `0.25`
- implicit-PD equivalent control (`kp=9`, `kd=0.2`, `±2.65 Nm`)
- actuator armature `0.00036207 kg m²`
- normalized 48-D observation and the checkpoint's deterministic actor
- planar tracking in the two-feet gravity-aligned base-footprint frame
- 5 s per requested command and matching CSV/error-plot format

## Model matching

`solo12.xml` is a lightweight, self-contained MJCF derived from the downloaded Solo12 URDF and corrected using values queried from the active IsaacLab USD at runtime. The model mass is `3.028572 kg`: base `1.75124 kg`, each hip `0.14196 kg`, thigh `0.147373 kg`, calf `0.023 kg`, and foot `0.007 kg`. It also uses the USD limits rather than the URDF's placeholder `[-10, 10]` limits. Primitive visuals/collisions avoid copying an 11 MB generated mesh tree and keep the experiment reproducible with ordinary Git. The repository already tracks `*.usd` with Git LFS if a future USD asset is added.

MuJoCo and PhysX remain different contact/solver implementations. Matching parameters does not make them the same simulator; the remaining difference is exactly what this sim-to-sim experiment measures.

## Run

Activate the existing environment:

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Then run the same experiment (Isaac-only Hydra overrides are accepted and reported as ignored):

```bash
./isaaclab.sh -p mujoco/play_direct_mujoco.py \
  --task="solo12-two-feet" --num_envs 1 --duration_s 2000 \
  --cmd_init 0.1 0.0 0.0 --episode_length_s 80 \
  env.tricky_terrain=True \
  --checkpoint="/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt" \
  --disable_training_gain_sync env.enabled_self_collisions=True \
  'env.base_filtered_pairs=["hip"]' 'env.command_lin_vel_x_range=[-0.6,0.6]' \
  env.extra_mass_on_front_feet=0.0 env.curriculum_tricky_terrain_idx=0 \
  env.curriculum_two_feet=False \
  --track-cmds="0.5 0. 0.;-0.5 0. 0.;0. 0.3 0.;0. -0.3 0.;0.5 0.3 0.;-0.5 -0.3 0.;+0.5 -0.3 0.;-0.5 +0.3 0." \
  --headless
```

Remove `--headless` for the interactive MuJoCo viewer. Add `--realtime` to pace it at real time. Outputs are written to `logs/mujoco/cmd_tracking/<checkpoint_stem>/` by default:

- `command_tracking.csv`
- `vxy_tracking_error.png`
- `wz_tracking_error.png` when any requested `wz` is nonzero

Purple dashed lines denote automatic resets caused by base-ground contact or episode timeout.

## Dependency

Tested in `env_isaaclab` with MuJoCo 3.3.7 and Python 3.11. Install it with:

```bash
python -m pip install 'mujoco>=3.3,<3.4'
```
