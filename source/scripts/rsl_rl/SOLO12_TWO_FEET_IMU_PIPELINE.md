# Two-feet policies with an IMU history encoder

## What is ready

We reuse the existing base-IMU PPO and DAgger implementations. Three new task IDs select the right observations and networks. All three inherit `Solo12TwoFeetEnvCfg`. They do not change its rewards, resets, actuator gains, contact rules, or five-phase curriculum.

| Stage | Task | Trainable networks |
| --- | --- | --- |
| A. Privileged PPO teacher | `solo12-two-feet-IMU-teacher` | State encoder, actor head, direct-state critic, action noise |
| B. Supervised DAgger | `solo12-two-feet-IMU-student-dagger` | History TCN; optional student actor-head copy |
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

The teacher's encoder, actor head, and critic each use hidden widths 256, 128, 64 by default.

### Optional latent bounding

For the three two-feet IMU tasks, one Hydra setting controls **both** the teacher encoder output and the student TCN output:

```text
env.apply_tanh_to_latent=True
```

This is the default. Each latent coordinate is bounded between -1 and 1. To leave both outputs unbounded, append this override to the teacher-training command:

```text
env.apply_tanh_to_latent=False
```

With `False`, the final teacher encoder layer and the final TCN projection are linear. The hidden-layer ELU activations, actor action outputs, and direct privileged critic do not change. This is a design choice, not a requirement of TCNs.

The teacher's saved contract carries this setting into DAgger. The adapter then carries it into student PPO. Resume and playback also restore it when the saved metadata is available. You do not need to repeat the flag during handoff. The saved setting takes precedence: changing it after training would change the learned policy even though all tensor shapes still match. New PPO checkpoints embed the flag inside the `.pt` file, so copying that file alone preserves it. Older PPO checkpoints fall back to their `params` metadata, or to the explicit flag when that metadata is absent.

The older base-IMU task defaults remain unchanged: legacy teachers use the separate `teacher_latent_activation` setting, and legacy student TCNs default to `tanh`. For the new two-feet tasks, `teacher_latent_activation` is derived from `apply_tanh_to_latent`; use the boolean switch, not the derived field.

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

The TCN uses 64 channels, kernel size 5, ELU activations, dilations 1/1/2/4, and a 32-value output with the optional final `tanh` described above. The actor gets 585 history values plus three current commands: **588 inputs**. It never receives simulator linear velocity, applied force, contact force, or the teacher's acceleration window. The clean acceleration window is privileged; the student already has its own noisy acceleration sequence inside the TCN history.

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
./isaaclab.sh -p source/scripts/rsl_rl/train_solo12_base_imu_dagger.py --task solo12-two-feet-IMU-student-dagger --teacher-checkpoint "$SOLO_IMU_TEACHER" --num_envs 9000 --headless --seed 230914 --dagger_iterations 3000 --num_steps_per_iter 24 --dataset_capacity 1500000 --num_mini_batches 8 --train_epochs 4 --learning_rate 0.0003 --finetune-student-actor --action-loss-coef 0.25 --dagger_experiment_name solo12_two_feet_imu_dagger '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The student controls collection with deterministic mean actions. The teacher labels states that the student actually visits. We aggregate the histories in a replay ring. The labeling teacher always stays frozen: its encoder, original actor head, critic, action noise, and normalizers do not learn during DAgger.

The loss is latent mean-squared error plus `--action-loss-coef` times action mean-squared error. The coefficient defaults to **0.25** and alone controls action supervision. Set it to **0** for latent-only supervision. The old `--supervise-action` flag is accepted with a deprecation notice, but has no effect. In particular, omitting that flag no longer disables the action loss.

| Options | Networks updated by DAgger |
| --- | --- |
| `--action-loss-coef 0` | TCN, using latent loss only |
| Positive coefficient, no fine-tuning flag | TCN, using latent loss plus action loss through the frozen teacher head |
| `--finetune-student-actor --action-loss-coef 0.25` | TCN and a separate student actor-head copy |

With `--finetune-student-actor`, we initialize the student head from the teacher and use that copy for both collection and supervised predictions. Action loss updates the TCN and student head; latent loss updates only the TCN. Teacher targets still come from the unchanged original teacher. Fine-tuning with a zero action coefficient is rejected because it gives the head no learning signal. Coefficients must be finite and non-negative. `--stochastic-actions` optionally adds the frozen teacher's action noise during collection.

The encoder starts untrained. Early student-controlled rollouts can fail. Keep the easy curriculum start unless a tested adapter is already available. A low latent loss alone is not proof of good control: inspect rollout return, terminations, curriculum progress, action error, and held-out commands. Replay capacity 1.5 million stores about 3.5 GB of float32 histories on CPU, plus labels and commands.

The teacher's observation contract and network sizes load from its checkpoint and `params/solo12_imu_contract.json`. Keep that file beside the run. It does not load rewards or curriculum state. Repeat intentional non-default task overrides across stages.

#### DAgger checkpoints

Curriculum-stage best saving is automatic. Your existing command needs no new flag.
The run's `checkpoints/` directory contains:

- `adapter_best.pt`: the lowest supervised training loss across the run, as before.
- `adapter_latest.pt`: the latest supervised update, every `--save_interval` iterations and at the end.
- `adapter_best_curriculum_idx_0_two_feet_phase_1.pt`, and so on: an independent best for each visited curriculum stage. These use the highest mean return over the last 100 eligible completed episodes in that stage, not imitation loss. A later phase can save a best even when its return is lower than an earlier phase's best. These saves do not wait for `--save_interval`.

Stage-best adapters are saved after collection and before the next supervised update. This keeps the adapter weights that collected the scored rollouts. Episodes can span several updates, so the rolling score is training evidence, not a separate fixed-policy evaluation. Episodes that cross a curriculum change or start in a special airborne pose do not count. A stage needs at least one eligible completed episode before it can save a best. If one rollout visits several stages, each stage keeps its own file.

Each stage-best file includes the stage indices, selection metric, score, episode count, iteration, and the usual adapter, normalizer, optimizer and teacher metadata. It supports the same playback and PPO initialization commands as `adapter_latest.pt`. The phase in the filename does not set the next run's curriculum start: choose that explicitly. `--adapter-checkpoint` starts fresh per-stage score tracking in the new run. Tasks without a curriculum keep the existing best/latest behavior.

Fine-tuned adapters also save `student_actor_state_dict` and declare the mode in `objective.finetune_student_actor`. Latest, loss-best, and stage-best files all carry the same network set. Isaac and MuJoCo playback use the saved student head when present; older adapters still use the original teacher head. Contradictory head metadata is rejected rather than silently falling back.

To continue a fine-tuned adapter, pass `--adapter-checkpoint` together with `--finetune-student-actor` and a positive action coefficient. This restores the TCN, modified head and history statistics. As before, it starts a fresh optimizer, replay dataset and iteration counter; it is not an exact optimizer-state resume. You can also initialize fine-tuning from an older TCN-only adapter: its student head starts from the teacher.

### C. Fine-tune the student with PPO

```bash
SOLO_IMU_ADAPTER=/absolute/path/to/the/dagger_run/checkpoints/adapter_latest.pt
```

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task solo12-two-feet-IMU-student-rl --init-from-dagger-adapter "$SOLO_IMU_ADAPTER" --num_envs 9000 --headless --seed 230914 --symmetry-mode augmentation --max_iterations 15000 --run_name two-feet-imu-student-ppo '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

This initializes the actor TCN from DAgger. If DAgger fine-tuned a student head, PPO starts from that **modified head**, not the original teacher actor. Older or frozen-head adapters use the original teacher head. By default, the **direct privileged critic and action-noise parameter start from the original teacher**. Actor normalization combines DAgger history statistics with original teacher command statistics. Critic normalization also comes from the original teacher. Transfer is strict: missing or mismatched tensors fail instead of leaving random parts of the policy. The optional critic-input expansion below is explicit and initializes only the new columns to zero.

PPO gets a **fresh optimizer and learning clock**, not the teacher's optimizer. The initial learning rate is 0.0002 with the inherited adaptive schedule. All student policy networks then train. There is no imitation loss in this phase. Do not also pass `--checkpoint` or `--resume` for this initialization. Ordinary PPO checkpoint resume is a different operation.

### Optional: give the privileged critic the actor's perception encoding

Both the teacher and student PPO tasks accept these environment overrides:

```bash
env.priv_critic_includes_perception_encoding=False env.priv_critic_detached_perception_encoding=True
```

These are the defaults, so existing runs keep their original architecture and gradient paths.
Set the first flag to `True` to give the critic **[privileged features, actor encoding, command]**.
For the usual two-feet settings this is 64 + 32 + 3 = 99 inputs, rather than 67.
The teacher uses its privileged MLP encoding. The student uses its actor TCN encoding.
The critic recomputes that encoding from the **same actor observations, normalization and encoder weights**.
It does not encode the clean critic observations or create a separate perception network.
The actor's own observations and action computation do not change.

With detachment enabled, only the critic's connection to the encoding is detached. The actor loss
still trains the perception encoder. With detachment disabled, the value loss also trains that
same encoder. This can affect the policy even when the actor head is unchanged.

The older `env.feed_history_encoding_to_critic` flag creates a separate critic TCN. It is not the
same experiment and cannot be enabled together with the new shared-perception option.

The existing teacher task name remains valid. `solo12-two-feet-IMU-teacher-rl` is an alias for
`solo12-two-feet-IMU-teacher`; both use the same environment and PPO configuration.

For a DAgger handoff from a direct privileged critic, enabling the new input copies every old
critic weight and inserts **zero-weight encoding columns** in its first layer. This preserves
the inherited value function up to floating-point roundoff, instead of starting a random critic.
If the teacher already used the encoding, enable it for the student too; dropping those learned
inputs is rejected. DAgger reconstructs either teacher architecture automatically.

New PPO checkpoints embed both flags. Full PPO resume and public playback restore the saved
topology, including when the checkpoint is copied away from its sidecar files. A full optimizer
resume does not switch architectures; use the new option when starting a teacher or initializing
a student from DAgger. A finite learned encoding is not a guarantee of a fully Markov state.

### Optional: critic-only warm-up at the student handoff

The student runner adds `agent.critic_warmup_iterations=0`. Zero disables warm-up.
For example, append the following to the student PPO handoff command above:

```bash
env.priv_critic_includes_perception_encoding=True agent.critic_warmup_iterations=50
```

Here 50 is an example, not a tuned recommendation. Each warm-up iteration collects the usual
sampled-policy rollout and runs the configured number of epochs and minibatches. These iterations
**count toward `max_iterations`**. During them:

- The actor head, perception encoder, action noise and actor-normalization statistics stay fixed.
- The critic and its observation-normalization statistics can adapt.
- The critic fits unclipped mean squared error against discounted rollout returns. Lambda is 1,
  so intermediate old value predictions do not enter the targets. Rollout ends and timeouts still
  bootstrap from the critic; these are not full-episode Monte Carlo targets.
- No actor, entropy, symmetry-augmentation or mirror-loss update runs. The adaptive KL learning-rate
  schedule does not run either. The critic uses the configured optimizer and learning rate.
- The encoder stays frozen **even if detachment is disabled** for subsequent normal PPO training.

Normal PPO starts on the next rollout after the last warm-up update. Its usual normalization,
losses and gradient settings resume. Warm-up is a separate ablation and also works without the
new critic encoding input.

The runner logs `Loss/critic_warmup_active`, `Loss/critic_warmup_remaining_iterations` and
`Loss/critic_warmup_completed_iterations`, as well as the critic loss and gradient norm.
Checkpoints save the remaining/completed counts. Resuming an unfinished warm-up continues it;
resuming after completion does not start it again. A fresh warm-up requires a DAgger handoff,
not an ordinary PPO checkpoint with no saved warm-up progress.

Warm-up currently supports single-process base-IMU student PPO. Do not combine it with CBP,
plasticity interventions or actor weight projection, which can modify the policy outside its loss.
It is not enabled for teacher training. Keeping actor weights fixed does not guarantee good
behavior: verify the sampled frozen policy on the intended simulator before interpreting results.

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

Shared-perception and critic-warm-up follow-up (2026-11-13): 123 CPU checks passed, covering
both policy types, normalization, gradient detachment, critic expansion, saved topology,
warm-up isolation with existing AdamW momentum, and PPO/BatchNorm/compile regressions.
A local 16-environment chain trained an encoding-aware teacher, fine-tuned a DAgger head,
performed two critic-only student updates, then performed a normal PPO update, with symmetry
configured. All 23 actor-side state tensors were bit-identical across the warm-up checkpoints;
the critic changed during warm-up and the actor changed after it. A partial warm-up checkpoint
resumed its one remaining critic-only update before PPO. Public headless playback loaded a copied
standalone student checkpoint with neither flag repeated. Four CPU comparisons against the
pre-change code found bit-identical default parameters, actions, values and one full PPO update,
for teacher/student with normalization on/off. These checks do not demonstrate better learning
or resolve the original cluster collapse.

Actor-fine-tuning follow-up (2026-11-11): 149 focused CPU checks passed, including both latent activations, coefficient-driven supervision, selected-network gradients, modified-head playback and exact PPO handoff. Eleven local integration checks passed with four training environments: a fresh teacher, joint DAgger, continuation, rejection of a discarded saved head, latent-only and frozen-head DAgger, one PPO iteration, and public Isaac/MuJoCo adapter and PPO playback. The live trainer changed the TCN and student head while preserving every original teacher tensor exactly. The saved adapter also passed exact TCN/head transfer, original-teacher critic/noise transfer and five batch-one action comparisons. Live checks used the default bounded latent. This proves integration, not learning performance, large-scale throughput, cluster readiness or hardware transfer.

Original pipeline validation: 30 CPU checks (including the existing race DAgger regressions), 22 task/live-physics checks, and 34 legacy-observation checks passed. One teacher PPO iteration, one action-supervised DAgger iteration, and one student PPO iteration completed with 8 environments. Both PPO stages used symmetry augmentation and offline W&B snapshots. The public headless player loaded teacher, adapter, and final-student checkpoints. Independent physics checks covered acceleration sampling, contact-force values, partial resets, and terminal histories.

Latent-switch follow-up: 54 CPU checks and 28 task/live-physics checks passed. Both `True` and `False` completed real four-environment teacher → action-supervised DAgger → PPO chains. Unbounded teacher, adapter, and PPO checkpoints passed public headless playback. A standalone copied PPO `.pt` restored `False` for playback and resumed PPO without a sidecar or repeated flag.

Curriculum-checkpoint follow-up: 71 focused CPU checks passed. A four-environment, two-iteration DAgger smoke used short episodes and zero promotion thresholds to visit all five phases. It saved all five stage-best files despite `--save_interval 50`. Their scores matched an independent calculation from recorded rewards and resets. A stage-best adapter passed public headless playback and initialized one PPO iteration. The tracker preserves the training random-number state and input tensors. Separate CUDA/PhysX baseline runs were not bit-exact; this check does not prove deterministic trajectory parity or learned performance.

These small runs check integration and exact initialization. They do not establish learned two-feet performance, better generalization, high-environment-count throughput, or hardware transfer. The encoder's benefit still needs a matched comparison against the direct-IMU policy, with held-out commands, pushes, delays, friction, and IMU noise.

MuJoCo playback now supports all three task IDs through `mujoco/play_direct_mujoco.py`, including teacher PPO, DAgger adapters with their frozen teacher, and student PPO. See `mujoco/README.md` for commands, checkpoint contracts and simulator limits. ROS deployment is still separate. The existing plain-MLP IMU export/deployment path must not be assumed to support this TCN automatically.
