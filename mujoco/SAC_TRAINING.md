# Solo12 SAC: Isaac pretraining and MJX fine-tuning

The SAC implementation follows Sabatini, Li, and Hutter,
“Bridging the Gap: Enabling Soft Actor-Critic for High Performance Legged
Locomotion” (arXiv:2605.24975).

It includes:

- asymmetric per-joint action bounds derived from Solo12 joint limits, the
  actual action center, and action scale;
- actor mean initialized near zero and initial standard deviation `0.15`;
- pre-reset observations for timeout bootstrapping;
- replay-buffer n-step returns (default `n=5`);
- twin Q critics, automatic entropy tuning, RND, symmetry augmentation/loss,
  W&B logging, checkpointing, and parallel environments.

## Isaac/PhysX pretraining

Use the existing training script and select the SAC agent entry point:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task="solo12-two-feet" \
  --agent=rsl_rl_sac_cfg_entry_point \
  --run-name="Solo12 two-feet SAC" \
  --symmetry-mode=augmentation \
  --headless \
  --num_envs=10000 \
  --max_iterations=1500 \
  env.curriculum_two_feet=False \
  env.initial_position=safe
```

All existing `env.*` overrides can be appended as with PPO. Paper defaults are
in `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/rsl_rl_sac_cfg.py`.
Hydra can override them, for example:

```bash
agent.algorithm.mini_batch_size=8192 \
agent.algorithm.num_mini_batches=200 \
agent.algorithm.n_steps=5
```

### Keeping the replay data for later fine-tuning

Add these two overrides to record the pretraining transitions:

```bash
agent.save_replay_buffer=True agent.save_replay_buffer_every=500
```

Saving is off by default. When it is on, the run writes
`<log_dir>/replay_buffer.pt` every `save_replay_buffer_every` iterations and once
more on the last iteration. The same file is overwritten each time, so a run that
is stopped early still leaves usable data behind and the job never accumulates one
copy per snapshot. The write goes to a temporary file and is then renamed, so a
job killed mid-save keeps the previous snapshot.

The file holds the whole buffer, so it is large: roughly
`replay_buffer_size * (2 * obs_dim + action_dim + 3) * 4` bytes. With
`replay_buffer_size=5e6`, a 48-dimensional observation and 12 actions that is about
2.2 GB. Each save prints its exact size.

Snapshots store raw observations, before the model normalizers run, so the data
stays valid while the normalizer keeps adapting during fine-tuning.

## Parallel MJX fine-tuning

Start from an Isaac SAC checkpoint:

```bash
./isaaclab.sh -p mujoco/train_sac.py \
  --task="solo12-two-feet" \
  --checkpoint="/absolute/path/to/isaac_sac/model_1500.pt" \
  --run-name="[cluster] MuJoCo SAC fine-tune" \
  --symmetry-mode=augmentation \
  --headless \
  --num_envs=256 \
  --max-iterations=2000 \
  env.curriculum_two_feet=False \
  env.initial_position=safe \
  env.tricky_terrain=False \
  env.include_events_randomization=False \
  "env.forces_applied_to_base_curriculum=[0.0]" \
  "env.base_push_force_z_range=[0.0,0.0]"
```

By default, `--checkpoint` transfers actor and twin critics but starts fresh
optimizers, entropy state, and iteration count. Add `--resume` for an exact
optimizer/iteration resume.

The first MJX iteration compiles the vectorized physics graph and can take a
few minutes. Later iterations reuse the compiled graph.

## Sim-to-online recipe

Fine-tuning defaults follow Yarden As et al., “What Matters for Simulation to
Online Reinforcement Learning on Real Robots” (arXiv:2602.20220). The paper runs
100+ real-robot trials and finds that naive fine-tuning collapses: the critic has
not yet adapted to the new dynamics, so its error distorts the actor update, which
shifts the data distribution further. Three choices fix it, and all three are
available here.

### Retained replay

The transitions recorded during pretraining are kept in a second, read-only
buffer. Every mini-batch takes a share of its samples from that buffer and the
rest from MJX data. The old data anchors the critic while the policy meets the new
dynamics. The share is then annealed to zero, so the final policy is fitted on MJX
data only.

```bash
--offline-replay-buffer=/path/to/isaac_run/replay_buffer.pt \
--offline-fraction=0.5 \
--offline-fraction-final=0.0 \
--offline-anneal-iterations=750
```

`--offline-fraction` defaults to `0.5` and `--offline-fraction-final` to `0.0`,
matching the paper. `--offline-anneal-iterations` defaults to half of
`--max-iterations`. The current share is logged as `Replay/offline_fraction`.

The paper writes this mixture with `alpha` as the *online* share, annealed up to
1. The flags here name the offline share instead, so `--offline-fraction=0.5`
annealed to `0.0` is the same schedule as the paper's `alpha=0.5 -> 1`.

Loading fails loudly if the snapshot's observation or action layout differs from
the current environment, which is the case that would otherwise train on
meaningless data.

### Delayed actor updates

`--actor-update-every` (default `20`) is `M` from the paper: one actor update per
`M` critic updates. Together with the conservative `--actor-learning-rate`
(default `1e-5`, against `2e-4` for the critic) this is what keeps a
not-yet-adapted critic from wrecking the policy. The paper's ablation shows that
`M=1` with a shared learning rate fails on every platform it tested. Pass
`--actor-update-every=1` to reproduce the synchronous baseline.

### Warm start

`--start-training` (default `1`) is the number of iterations collected with the
loaded policy before the first gradient update. One iteration already collects
`num_envs * rollout_steps` transitions, so the default gives 6144 transitions at
`--num_envs=256`. The paper prefills with about 5000. Use this when no pretraining
snapshot is available; it approximates retained replay but works less well.

### Low-rank fine-tuning (optional)

`--rank` freezes the pretrained actor weights and trains only a low-rank
correction on top, the same idea as the PPO path in `mujoco/train_lora.py`:

```bash
--rank=64 --lora-alpha=64 --lora-layers=all
```

`--rank=0` (the default) fine-tunes every actor weight. `--lora-alpha` defaults
to `--rank`, which makes the applied scale `alpha/rank` equal to 1.
`--lora-layers` takes the same modes as the PPO script: `all`, `input`,
`output`, `input_and_output`. The adapter starts at exactly zero, so training
begins from the pretrained policy rather than near it.

Only the actor is adapted. In SAC the critic is the learning signal, and
arXiv:2602.20220 traces the transfer failure to a critic that has *not* yet
adapted to the new dynamics. Constraining the critic would work against the
fine-tuning instead of protecting it.

Checkpoints from a LoRA run store the actor with the adapter folded back into
the dense weights, plus a `mujoco_lora` entry recording the settings. The files
are therefore ordinary SAC checkpoints: the play and evaluation scripts load
them unchanged, and a later run can fine-tune from them again. Because the saved
actor has no adapters, `--resume` is rejected together with `--rank`; start a new
LoRA run from the merged checkpoint with `--checkpoint` instead.

### Chaining runs

Add `--save-replay-buffer` to record the MJX transitions too. A later run can then
pass that file as its `--offline-replay-buffer`, which is the paper's
data-recycling setup. Saving from a fine-tuning run records the MJX data only, not
the pretraining data it was mixed with, so trials chain without duplicating.

Putting it together:

```bash
./isaaclab.sh -p mujoco/train_sac.py \
  --task="solo12-two-feet" \
  --checkpoint="/absolute/path/to/isaac_sac/model_1500.pt" \
  --offline-replay-buffer="/absolute/path/to/isaac_sac/replay_buffer.pt" \
  --run-name="[cluster] MuJoCo SAC fine-tune | retained replay" \
  --symmetry-mode=augmentation --headless \
  --num_envs=256 --max-iterations=1500 \
  env.curriculum_two_feet=False env.initial_position=safe \
  env.tricky_terrain=False env.include_events_randomization=False \
  "env.forces_applied_to_base_curriculum=[0.0]" "env.base_push_force_z_range=[0.0,0.0]"
```

## Checkpoint compatibility

SAC and PPO checkpoints are intentionally different. `train_sac.py` accepts
RSL-RL-SAC checkpoints, not PPO/LoRA PPO checkpoints. PPO training remains on
the existing RSL-RL 3.1.2 path and is not migrated or altered by SAC.
