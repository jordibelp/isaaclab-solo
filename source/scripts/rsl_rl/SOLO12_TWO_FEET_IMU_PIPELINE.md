# Two-feet policies with an IMU history encoder

## What is ready

We reuse the existing base-IMU PPO and DAgger implementations. Three new task IDs select the right observations and networks. All three inherit `Solo12TwoFeetEnvCfg`. They do not change its rewards, resets, actuator gains, contact rules, or five-phase curriculum.

| Stage | Task | Trainable networks |
| --- | --- | --- |
| A. Privileged PPO teacher | `solo12-two-feet-IMU-teacher` | State encoder, actor head, direct-state critic, action noise |
| B. Supervised DAgger | `solo12-two-feet-IMU-student-dagger` | History TCN only |
| C. Student PPO | `solo12-two-feet-IMU-student-rl` | History TCN, copied actor head, copied privileged critic, action noise |

The original `solo12-two-feet` task and old IMU task defaults remain available. This is a PPO pipeline. Do not select the SAC agent entry point for these encoded tasks.

## Observation contract

### Clean teacher actor and clean critic

The teacher encoder receives 64 inputs by default, in this exact order:

| Signal | Values | Meaning |
| --- | --- | --- |
| Projected gravity | 3 | Unit gravity direction in base axes |
| Base linear velocity | 3 | Simulator velocity in base axes, m/s |
| Base angular velocity | 3 | Simulator angular velocity in base axes, rad/s |
| Previous action | 12 | Previous issued, environment-clipped action; before simulated actuator delay |
| Joint position and velocity | 24 | Joint position relative to the task's observation offset, rad; joint velocity, rad/s |
| Base acceleration history | 12 | Four gravity-free acceleration samples at the **base link origin**, m/s² |
| Applied base push | 3 | Held simulator push vector in base axes, N; zero when no push is applied |
| Calf contact-force norms | 4 | Current net contact force on FL, FR, RL, RR calf bodies, N |

Commands add three more inputs. They are **outside the teacher encoder**. The actor sees the 32-value latent and three commands. The critic sees all 67 clean inputs directly. It has no encoder or TCN.

The teacher's encoder, actor head, and critic each use hidden widths 256, 128, 64 by default. The new teacher bounds its final latent with `tanh`, matching the student's output range. Old teachers keep their original unbounded latent by default.

Both teacher observation-noise switches default to false. Domain randomization and task disturbances are separate: clean observations do not mean clean physics. DAgger labels and the student PPO critic are always clean even when the student actor is noisy.

Useful teacher/critic ablations:

- `env.include_lin_acc_steps_in_imu_obs=0`: omit clean acceleration. A positive integer selects that many physics samples. This does not stack extra windows inside each student sample.
- `env.teacher_include_base_force=False`: omit the applied-push vector.
- `env.teacher_foot_force_obs=none`, `norm`, or `vector`: omit contact forces, use four norms, or use four full vectors in base axes.
- `env.include_foot_height_obs=True`: also include the old four foot-height features. Off by default here.

Norms are a small starting point. They identify contact and load but discard force direction. `vector` is the directional-force ablation. These are net **calf-body contacts**, not isolated foot-tip force sensors or incoming joint wrenches. The applied-push field describes this task's configured base push, not every contact on the robot. Pushes can be off-centre; the three force values do not encode their application point or resulting moment.

### Student actor: no privileged inputs

Each history sample has 45 values: joint position and velocity (24), gyro (3), gravity-free EKF-style IMU acceleration (3), projected gravity (3), and the last issued action (12).

The history contains 13 samples at 200 Hz. It includes both endpoints and spans 60 ms. Policy inference runs at 50 Hz. Samples are oldest first. Histories are zero-padded after reset and advance once per completed physics step, never on observation reads. Each sample uses the base axes at acquisition time.

The TCN uses 64 channels, kernel size 5, ELU activations, dilations 1/1/2/4, and a 32-value `tanh` output. The actor gets 585 history values plus three current commands: **588 inputs**. It never receives simulator linear velocity, applied force, contact force, or the teacher's acceleration window. The clean acceleration window is privileged; the student already has its own noisy acceleration sequence inside the TCN history.

The student's acceleration comes from the simulated IMU mounting point. The privileged acceleration comes from the base link origin. Both are gravity-free and expressed in base axes. They differ by the rigid-body lever-arm terms. The privileged value has no sensor noise or transport clipping. The student keeps the existing IMU noise, bias, orientation, and clipping model.

We use issued-action history for the new tasks. The robot can know its issued commands. It cannot directly observe which command an unknown simulated delay line executed. Existing tasks still allow their old executed-action setting.

## Commands

Commands below run from the canonical repository. Every command is one shell line. They are launch recipes, not jobs already submitted. Cluster code synchronization and Slurm submission are separate actions.

### Setup

Local machine:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

On IRICluster, use the corresponding checkout under `/home/jbeltran/IsaacLab` and its `isaaclab` conda environment. Update that checkout before using the new task IDs; this work did not push or synchronize code there. The recipes use 9,000 environments as a conservative starting point for the existing L40S/PhysX setup. This is not a throughput benchmark. For laptop smokes use 8–32 environments.

### A. Train a new privileged teacher

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task solo12-two-feet-IMU-teacher --num_envs 9000 --headless --seed 230914 --symmetry-mode augmentation --max_iterations 30000 --run_name two-feet-clean-teacher '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The inherited curriculum starts at phase 1 and can advance through phase 5. Use the same explicit reward, actuator, and curriculum overrides as your chosen recent run if it differed from current defaults. Inheritance preserves current source defaults; it does not reconstruct an unspecified historical run.

Choose a checkpoint that performs well at the difficulty you want to transfer. Use a numbered checkpoint that will not be overwritten; the adapter records its teacher path. Do not reuse a plain-MLP two-feet checkpoint as this teacher. It has no privileged latent encoder.

### B. Train the history encoder with DAgger

Set the full teacher checkpoint path:

```bash
SOLO_IMU_TEACHER=/absolute/path/to/the/teacher_run/checkpoint.pt
```

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train_solo12_base_imu_dagger.py --task solo12-two-feet-IMU-student-dagger --teacher-checkpoint "$SOLO_IMU_TEACHER" --num_envs 9000 --headless --seed 230914 --dagger_iterations 3000 --num_steps_per_iter 24 --dataset_capacity 1500000 --num_mini_batches 8 --train_epochs 4 --learning_rate 0.0003 --supervise-action --action-loss-coef 0.25 --dagger_experiment_name solo12_two_feet_imu_dagger '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The student controls collection with deterministic mean actions, as in the old pipeline. The teacher labels states that the student actually visits. We aggregate the histories in a replay ring. Only the TCN learns. The teacher encoder, actor head, critic, action noise, and teacher normalizers stay frozen.

The loss is latent mean-squared error plus 0.25 times action mean-squared error. The action loss backpropagates through the frozen actor into the TCN. It does **not** train a separate actor head. Omit `--supervise-action` for latent-only supervision. `--stochastic-actions` optionally adds the frozen teacher's action noise during collection.

The encoder starts untrained. Early student-controlled rollouts can fail. Keep the easy curriculum start unless a tested adapter is already available. A low latent loss alone is not proof of good control: inspect rollout return, terminations, curriculum progress, action error, and held-out commands. Replay capacity 1.5 million stores about 3.5 GB of float32 histories on CPU, plus labels and commands.

The teacher's observation contract and network sizes load from its checkpoint and `params/solo12_imu_contract.json`. Keep that file beside the run. It does not load rewards or curriculum state. Repeat intentional non-default task overrides across stages.

### C. Fine-tune the student with PPO

```bash
SOLO_IMU_ADAPTER=/absolute/path/to/the/dagger_run/checkpoints/adapter_latest.pt
```

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task solo12-two-feet-IMU-student-rl --init-from-dagger-adapter "$SOLO_IMU_ADAPTER" --num_envs 9000 --headless --seed 230914 --symmetry-mode augmentation --max_iterations 15000 --run_name two-feet-imu-student-ppo '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

This initializes the actor TCN from DAgger and copies the teacher actor, direct privileged critic, and action-noise parameter. It composes actor normalization from DAgger history statistics and teacher command statistics. Critic normalization comes from the teacher. Transfer is strict: missing or mismatched tensors fail instead of leaving random parts of the policy.

PPO gets a **fresh optimizer and learning clock**, not the teacher's optimizer. The initial learning rate is 0.0002 with the inherited adaptive schedule. All student policy networks then train. There is no imitation loss in this phase. Do not also pass `--checkpoint` or `--resume` for this initialization. Ordinary PPO checkpoint resume is a different operation.

Curriculum progress and promotion statistics do not transfer. Each stage starts from its configured phase. `env.two_feet_curriculum_start_phase=K` selects phases 1–5. An adapter filename does not select a phase. For a fixed-difficulty comparison, disable advancement explicitly and match physical settings, commands, reset distributions, and seeds.

### Playback

For the default architecture, use the existing public player and a teacher or final PPO checkpoint:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_direct_0325.py --task solo12-two-feet-IMU-student-rl --checkpoint /absolute/path/to/student_ppo_checkpoint.pt --num_envs 1 --duration_s 20 --cmd_init 0.3 0.0 0.0 --disable_training_gain_sync
```

Select the teacher task for a teacher checkpoint. For custom observations or architecture, use the matching saved settings. Keep the full run's `params` folder with checkpoints. The player can also evaluate an adapter with the DAgger task and its teacher checkpoint; a final PPO checkpoint is self-contained for network inference.

## What changed beyond task aliases

The learning algorithms already existed. Configuration alone was not sufficient:

1. Added the requested clean acceleration and force features without leaking them to the student actor.
2. Allowed a clean encoded teacher/critic without the old simple-MLP-only noise guard.
3. Saved signal semantics and TCN settings in DAgger metadata. PPO now understands base-IMU checkpoints as well as race checkpoints, including endpoint-inclusive histories.
4. Fixed exact asymmetric transfer: no attempt to copy a nonexistent critic TCN, and correct normalization when one branch is disabled.
5. Added symmetry transforms for clean acceleration, push vectors, and per-leg scalar/vector contact forces. The inherited two-feet front/back restriction remains in force.
6. Saved the actual policy implementation and contract helper with offline W&B source snapshots.

## Validation boundary

Verified locally: 30 CPU checks (including the existing race DAgger regressions), 22 task/live-physics checks, and 34 legacy-observation checks passed. One teacher PPO iteration, one action-supervised DAgger iteration, and one student PPO iteration completed with 8 environments. Both PPO stages used symmetry augmentation and offline W&B snapshots. The public headless player loaded teacher, adapter, and final-student checkpoints. Independent physics checks covered acceleration sampling, contact-force values, partial resets, and terminal histories.

These small runs check integration and exact initialization. They do not establish learned two-feet performance, better generalization, high-environment-count throughput, or hardware transfer. The encoder's benefit still needs a matched comparison against the direct-IMU policy, with held-out commands, pushes, delays, friction, and IMU noise.

ROS deployment and MuJoCo TCN playback are not part of this change. The existing plain-MLP IMU export/deployment path must not be assumed to support this TCN automatically.
