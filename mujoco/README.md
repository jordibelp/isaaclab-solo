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

`solo12.xml` is a self-contained MJCF cross-checked field-by-field against the active
`SoloFlat.usd` (dumped with `pxr`): joint axes (+x hips, +y thighs/calves on both sides),
anchor positions, the modified asymmetric limits (front thigh `[-250°, 90°]`, rear `[-90°, 250°]`,
hips `±179°`, calves `[-179°, 180°]`), link masses/inertias, and the authored base COM `(-0.01, 0, 0)`.
Total mass is `3.028572 kg` (base `1.75124` per `cfg.base_mass`).

Collision topology mirrors the USD exactly — hips and leg shafts have **no** colliders:

- base: `collision_main` box + the two `borde` side-rail boxes
- thigh: `collider_top` box + `collider_bot` knee sphere (r=0.0371)
- calf: the foot contact **cylinder** (r=0.0186, half-width 0.0063, axis y) — a disk, not a sphere
- everything in the visual group is display-only (`contype=0`)

Note that `SoloFlat.usd` itself has **no mesh colliders**: the URDF→USD conversion authored PhysX
colliders as primitives, so these boxes/spheres/cylinders *are* the shape Isaac simulates. Matching
them is therefore the whole of the collision contract — the meshes below never enter physics.

MuJoCo and PhysX remain different contact/solver implementations. Matching parameters does not make
them the same simulator; the remaining difference is what this sim-to-sim experiment measures.

### Visual meshes

`meshes/*.obj` are the real Solo12 visual meshes extracted from the same `SoloFlat.usd`, so the
viewer shows the actual robot (ring feet, hip/thigh castings, PCB deck) instead of placeholder
primitives. They are regenerated with:

```bash
python mujoco/extract_visual_meshes.py
```

Details worth knowing:

- The meshes live in USD *instanced prototypes*, so a plain `Stage.Traverse()` finds zero meshes —
  `Usd.TraverseInstanceProxies` is required. That is why they were easy to miss.
- The prototypes also contain `*_collision_*` convex-decomposition leftovers; the extractor skips
  them, since they are neither the visual shape nor the shape PhysX uses.
- One OBJ per (link, material), coloured from the USD MDL `diffuse_color_constant`.
- The `*_foot` links carry **no** visual mesh in the USD — the foot disk is part of the calf mesh.
  The old black sphere at the foot was a placeholder with no counterpart in the real robot.
- Feet are visually ~3 mm shorter than the contact cylinder, exactly as in the USD. This is
  reproduced rather than corrected, since Isaac trains against the cylinder.

Swapping the primitives for meshes is dynamically inert, and this is enforced two ways: the geoms
are `class="visual"` (`contype=0`), guarded by `test_visual_meshes_never_collide`, and the 8-command
tracking run reproduces the previous summary bit-for-bit (every metric delta exactly `0.0`).

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
  MuJoCo version, Git revision, and `visual_meshes_sha256`. Since `solo12.xml` now references
  `meshes/*.obj`, the archived MJCF is no longer standalone; the meshes are versioned in-repo
  instead of re-uploaded per run, and the digest pins which ones were used. Physics is unaffected
  either way — the meshes are `contype=0`.
- `summary.json` — aggregate tracking/fall/stance metrics
- `command_tracking.csv` — Isaac columns plus `reset`, `base_height_m`, `gravity_x_b`
  (`gravity_x_b` ≈ 0 on four feet, ≈ −0.96 in the upright two-feet stance)
- `vxy_tracking_error.png`
- `wz_tracking_error.png` when any requested `wz` is nonzero

Purple dashed lines denote automatic resets caused by base-ground contact or episode timeout.

Both the MuJoCo and Isaac Sim tracking runners report the same error-distribution fields for
`vxy_error` and `wz_error`: mean, median, p05, p95, population standard deviation, and maximum.
They also report `resets` and the equivalent `failures` count for rollout resets.
Isaac Sim writes these files beside its tracking CSV/plots and uploads them to the same W&B project
only when `--track-cmds` is active (`--no-tracking-wandb` disables that Isaac upload).

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

## LoRA-PPO fine-tuning with MJX

`train_lora.py` implements the adapter-only part of SLowRL (arXiv:2603.17092): frozen dense
actor/critic weights, zero-output LoRA residuals, PPO updates to **both** actor and critic, and a
trainable exploration standard deviation. It intentionally has no recovery policy or safety filter.
The paper's sim-to-sim validation runs MuJoCo in real time to emulate the real robot and does not
report massively parallel target-domain rollouts. This implementation additionally uses MJX so the
same experiment can run with either `--num_envs=1` or hundreds of GPU-parallel environments.

`--rank=<r>` sets the rank of the `A`/`B` factors (default 1); any `r >= 1` works, and the applied
residual is scaled by `--lora-alpha / rank`. Because `B` starts at zero, iteration 0 reproduces the
pretrained policy exactly at every rank. Adapter placement is selected with
`--trainable_layers={all,input_and_output,input,output}` and is always mirrored in actor and
critic. With the asymmetric two-feet task, `--symmetry-mode=augmentation` applies the valid
left/right reflection (front/back augmentation is deliberately excluded).

### Stability

The paper quotes a LoRA learning rate of `1e-2`. Applied here as a *fixed* rate it diverges: PPO
takes `ppo_epochs * num_minibatches` Adam steps per iteration, so the policy leaves the trust
region within the first iteration (clip fraction > 0.7), the ratio `exp(logp - old_logp)` overflows,
and the parameters go NaN permanently. The defaults therefore follow the Isaac solo12 PPO config
that produced the frozen checkpoint: `--learning-rate=1e-3` with `--lr-schedule=adaptive`
(KL-targeting on `--desired-kl=0.01`, as in RSL-RL), `--entropy-coef=0.002`, `--max-grad-norm=0.5`.
Three further guards are always on:

- `--log-std-range` (default `-4 0`) bounds the exploration std. The entropy bonus applies a
  constant gradient that Adam turns into a fixed-size upward step every update, so an unbounded
  `log_std` drifts up monotonically until the policy is noise.
- The PPO log-ratio is clamped before `exp`, keeping a diverged update finite.
- Adam rejects any non-finite gradient and leaves the parameters untouched. A sustained rejection
  (5 consecutive iterations) aborts the run with the last good checkpoint rather than logging NaN
  for the remaining iterations.

Watch `policy/kl`, `policy/clip_fraction` and `policy/action_std` in W&B: KL should sit near
`desired_kl`, clip fraction well under 0.5, and action std roughly flat. Batch size matters more
than anything else here — `--num_envs=1 --rollout-steps=24` is 24 samples (0.48 s of experience)
per iteration and the run warns about it. Use a large `--num_envs` to train, then evaluate the
merged checkpoint at `--num_envs=1`.

```bash
./isaaclab.sh -p mujoco/train_lora.py \
  --task=solo12-two-feet --trainable_layers=all --rank=1 --num_envs=512 \
  --checkpoint=/absolute/path/to/model_22043.pt \
  --run-name="mujoco LoRA|all layers" --symmetry-mode=augmentation --headless \
  env.curriculum_two_feet=False env.initial_position=safe \
  env.front_back_asymetry=True env.three_or_more_feet_contact_triggers_reset=False \
  env.tricky_terrain=False env.include_events_randomization=False \
  'env.forces_applied_to_base_curriculum=[0.0]' \
  'env.base_push_force_z_range=[0.0,0.0]' 'env.actuation_delay_range=[0,3]' \
  env.kp=9.0 env.kd=0.2 env.enable_observation_corruption=True \
  env.three_or_more_feet_contact_penalty_reward_scale=-100.0 \
  env.track_lin_vel_xy_reward_scale=1.6 \
  env.two_feet_above_height_reward_scale=1.3 \
  env.two_feet_above_height_threshold=0.45 env.two_feet_above_height_alpha=25.0 \
  env.rear_feet_in_contact_for_twofeet=False
```

Runs are stored under `logs/mujoco/lora_ppo/<timestamp>_<run-name>/`. Each save contains
`adapter_<iteration>.pt` (small LoRA-only state) and `model_<iteration>.pt` (LoRA merged into an
ordinary RSL-RL checkpoint). The merged checkpoint works directly with `play_direct_mujoco.py`.
W&B defaults to project `solo12-two-feet-lora`; use `--no-wandb` for local-only training.

MJX 3.3 does not implement cylinder-box collision. The training backend therefore preserves the
cylindrical foot/ground contacts but filters only foot-cylinder versus self-box candidate pairs;
all other authored collisions remain active. Ordinary MuJoCo evaluation keeps the full model.

## Interaction-budgeted SAC fine-tuning with MJX

`train_sac.py` defaults to the single-robot collection schedule discussed in
[What Matters for Simulation-to-Online Reinforcement Learning on Real Robots](https://arxiv.org/html/2602.20220#S3.SS1).
Collect a fixed number of environment interactions, pause collection, and run the gradient
updates before collecting again. MJX still runs as fast as it can; this emulates the data
budget of one robot, not real-time pacing. Isaac training and `train_lora.py` keep their
existing schedules.

Both the total budget and the update period are counted in **environment interactions**, not
episodes. An episode that ends early costs the run one short episode instead of a whole
iteration, and collection continues straight into the next episode.

| Setting | Default | Configuration |
|---|---|---|
| Environments | 1 | `--num-envs` |
| Total interaction budget | 50000 | `--max-env-interactions` |
| Interactions between updates | 1000 per environment | `--rollout-steps` |
| Maximum episode length | 1000 control steps (20 s at 50 Hz) | `env.episode_length_s=20` |
| Critic update-to-data ratio (UTD) | 1.25 | `--utd` |
| Replay samples per mini-batch | 512 | `--batch-size` |
| New transitions before weight updates | 5000 | `--num-transitions-before-weight-updates` |
| Actor update interval | Every 20 critic updates | `--actor-update-every` |
| Online replay capacity | 500000 transitions | `--replay-buffer-size` |

The authors' [Go1 online config](https://github.com/yardenas/safe-learning/blob/ffee61c6bc95555591c2ccfe67676e668784d540/ss2r/configs/experiment/go1_online.yaml)
uses 1000-step episodes and 1250 updates. It inherits batch size 512 from
[go1_joystick.yaml](https://github.com/yardenas/safe-learning/blob/ffee61c6bc95555591c2ccfe67676e668784d540/ss2r/configs/experiment/go1_joystick.yaml).
Its retained-replay example sets `min_replay_size=1000`; our 5000-transition warm-up is Jordi's
requested default, not an exact copy of that setting. Other existing SAC hyperparameters remain unchanged.

### Collection and warm-up semantics

- At the defaults, iterations 1–4 collect only. After iteration 5 reaches 5000 new
  transitions, run 1250 critic updates, then another 1250 after each later iteration. Do not
  run a catch-up burst for the earlier warm-up iterations.
- A fall ends the episode but not the iteration: the environment resets and collection
  continues to the full `--rollout-steps`. Each update phase therefore always gets the same
  number of fresh transitions, whatever the policy's episode length.
- Warm-up counts actual new transitions, even when replay wraps. Offline transitions do not
  count.
- `--max-env-interactions=50000` is the total interaction budget, including warm-up. It is
  divided by `--num-envs × --rollout-steps` to get the iteration count, rounding up so a
  partial final period still runs.
- `--num_transitions_before_weight_updates` and `--transitions-before-updates` are aliases.
  Replace the old `--start-training=1` with `--num-transitions-before-weight-updates=5000`.
  The old flag is still accepted, but explicitly converts full rollout lengths to a transition
  threshold: `--start-training=1` gives 1000, not 5000, with the new default episode length.
- An optimizer resume still starts with empty online replay, so it repeats warm-up using
  fresh transitions. The saved iteration number does not bypass warm-up.
- Batch size counts samples drawn from replay. Left/right symmetry augmentation adds mirrored
  samples afterward; it does not change UTD or the number of collected transitions.

### Retained replay and usage

```bash
./isaaclab.sh -p mujoco/train_sac.py --task=solo12-two-feet --checkpoint=/absolute/path/to/model.pt --offline-replay-buffer=/absolute/path/to/replay_buffer.pt --offline-fraction=0.5 --offline-fraction-final=0.0 --num-envs=1 --max-env-interactions=50000 --rollout-steps=1000 --utd=1.25 --batch-size=512 --num-transitions-before-weight-updates=5000 --actor-update-every=20 --symmetry-mode=augmentation --headless env.episode_length_s=20 env.curriculum_two_feet=False env.initial_position=safe env.tricky_terrain=False env.include_events_randomization=False 'env.forces_applied_to_base_curriculum=[0.0]' 'env.base_push_force_z_range=[0.0,0.0]'
```

The offline share starts at 0.5 at the **first gradient phase**, not at the first warm-up
iteration. `--offline-anneal-iterations` counts iterations that actually perform updates.
Its default is half of the resolved iteration count; set it explicitly for a faster anneal.
The online replay and retained offline replay stay separate.

To use a fixed update count instead of UTD, pass `--updates-per-iteration=1250` and omit
`--utd`. For parallel collection, scale `--rollout-steps` down instead of collecting longer:
e.g. `--num-envs=256 --rollout-steps=24 --updates-per-iteration=200` still updates every
6144 interactions.

The resolved schedule is printed at startup and saved in `run_config.json` under
`agent.update_schedule`. TensorBoard/W&B schedule metrics report actual collected transitions,
cumulative online transitions, updates per iteration, completed update phases, and warm-up state.
Full, LoRA, and frozen actor/critic choices remain available independently.

## Dependency

Tested in `env_isaaclab` with MuJoCo 3.3.7 and Python 3.11. Install with:

```bash
python -m pip install -r mujoco/requirements.txt
```
