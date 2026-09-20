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

## Episodic MJX fine-tuning

Start from an Isaac SAC checkpoint:

```bash
./isaaclab.sh -p mujoco/train_sac.py \
  --task="solo12-two-feet" \
  --checkpoint="/absolute/path/to/isaac_sac/model_1500.pt" \
  --run-name="[cluster] MuJoCo SAC fine-tune" \
  --symmetry-mode=augmentation \
  --headless \
  --num_envs=1 \
  --max-env-interactions=50000 \
  env.curriculum_two_feet=False \
  env.initial_position=safe \
  env.tricky_terrain=False \
  env.include_events_randomization=False \
  "env.forces_applied_to_base_curriculum=[0.0]" \
  "env.base_push_force_z_range=[0.0,0.0]"
```

By default, `--checkpoint` transfers actor and twin critics but starts fresh
optimizers, entropy state, and training progress. Add `--resume` for an exact
optimizer/progress resume.

The first MJX episode compiles the physics graph and can take a few minutes.
Later episodes reuse the compiled graph.

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
matching the paper. `--offline-anneal-iterations` defaults to half of the
resolved iteration count. The current share is logged as `Replay/offline_fraction`.

Both halves are drawn at exactly their requested size, with replacement, so the
batch always has the composition you asked for. Early in a run the online buffer
may hold fewer distinct transitions than its half of the batch, in which case
some of them repeat. Watch `Replay/online_valid_transitions` for that: it counts
the distinct online transitions the last batch could draw from. If it stays far
below `--batch-size`, the online half carries much less information than its size
suggests, and you want either more environments or a smaller batch.

This matters most with the default single environment. The default warm start
collects 5000 new transitions before any gradient update. With full 1000-step
episodes, the first update phase therefore runs after episode 5. Early falls
require more episodes to reach the same transition threshold. Use
`--num-transitions-before-weight-updates` to change it.

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

`--num-transitions-before-weight-updates` defaults to `5000`. It counts new MJX
transitions, not retained replay data or the saved checkpoint iteration. Learning
starts at the first update boundary where the count reaches or exceeds the
threshold. The warm-up iterations do not create a backlog of gradient updates.

### Actor/critic fine-tuning ablations

Both networks use **full fine-tuning by default**. Choose each network independently:

| Experiment | Flags to append |
| --- | --- |
| Full actor + full critic (default) | `--rank=0` (or omit) |
| Rank-1 LoRA on both | `--rank=1` |
| Rank-4 actor + rank-16 critic | `--actor-rank=4 --critic-rank=16` |
| Rank-1 actor + full critic | `--actor-rank=1 --critic-rank=0` |
| Full actor + rank-1 critic | `--actor-rank=0 --critic-rank=1` |
| Full actor + frozen critic | `--freeze-critic` |
| Rank-1 actor + frozen critic | `--rank=1 --freeze-critic` |
| Frozen actor + full critic | `--freeze-actor` |
| Frozen actor + rank-1 critic | `--rank=1 --freeze-actor` |
| Frozen actor + frozen critic | `--freeze-actor --freeze-critic` |

`--rank` is a shared default. `--actor-rank` and `--critic-rank` override it
independently. A rank of zero means **full training**, not freezing. A freeze flag
wins over the shared rank; combining it with an explicit positive rank for that
same network is rejected as contradictory. Options do not depend on their order.
**Changed from the original SAC CLI:** `--rank=1` now adapts both actor and critic.
Use `--rank=1 --critic-rank=0` to recover the previous actor-only LoRA behavior.

For the critic, LoRA applies to **both online Q-networks**. Each gets its own
adapters. Their dense target networks still follow normal Polyak averaging, using
the effective adapted weights. We do not average the two low-rank factors
separately, which would give different target weights.

A fully frozen network keeps its weights **and observation normalizer** fixed.
A frozen critic also keeps its target networks fixed. It still passes Q-value
gradients through actions to train the actor. No critic optimizer step runs.
With a frozen critic, `--actor-update-every` still counts sampled update batches;
use `--actor-update-every=1` if you want one actor step per batch. The default
remains 20. Critic losses may still be logged as diagnostics even when frozen.
Automatic entropy-temperature tuning stays active when the actor is trainable;
it is also frozen when `--freeze-actor` is used. Freezing both networks collects
rollouts without changing either model or the entropy temperature.

LoRA freezes the existing parameters, including biases and LayerNorm parameters,
and trains only the selected adapters. Observation normalizers continue adapting
in LoRA and full modes. Adapters start at zero, preserving initial network outputs.

#### Adapter gain and layer selection

`--lora-alpha` and `--lora-layers` are shared defaults. Per-network overrides are:

```bash
--actor-lora-alpha=4 --critic-lora-alpha=16 --actor-lora-layers=all --critic-lora-layers=output
```

If alpha is omitted, each network uses its **own resolved rank**, making its
scale `alpha/rank` equal to 1 even when actor and critic ranks differ. Supported
layer selections are `all`, `input`, `output`, and `input_and_output`. Critic
layer selection is applied separately to each online Q-network.

The resolved settings are printed at startup, recorded in `run_config.json` and
the W&B agent configuration, and saved as `mujoco_finetuning` in checkpoints.

#### Saving and starting another run

LoRA checkpoints merge adapters into ordinary actor and critic weights, so play
and evaluation need no adapters. Both online and target critics are saved.
Start another fine-tuning run with `--checkpoint` and whichever modes you want.

LoRA optimizer resume is not supported because merged files do not retain the
original adapter factors. `--resume` is rejected if either current network uses
LoRA, or if the source file was saved from a LoRA run. Dense/frozen runs can
restore optimizers with `--resume` only when actor/critic freeze modes match.
To change modes, use `--checkpoint` without `--resume`.

### Loading different pretrained architectures

The MJX trainer now reads hidden sizes, LayerNorm presence, observation
normalization, actor standard-deviation layout, and categorical support from the
checkpoint. It no longer assumes a 512-256-128 network with scalar Q heads. This
supports the 1024-512-256, 199-bin `tacjuisg/model_3700.pt` checkpoint too.

Keep the checkpoint beside its original `params/agent.yaml` (Isaac) or
`run_config.json` (MJX). These files preserve settings that tensors cannot tell
us, notably activations and whether categorical labels use two-hot or HL-Gauss.
If only the `.pt` file is available, the trainer reports its assumptions: Swish
activations, and two-hot labels for categorical heads. Override them if needed:

```bash
--actor-activation=swish --critic-activation=swish --critic-loss=hl_gauss --hl-gauss-sigma-ratio=0.75
```

CLI values override sidecar values. Scalar versus categorical head type must
match the checkpoint; two-hot versus HL-Gauss can be selected as an ablation.

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
  --num_envs=1 --max-env-interactions=50000 \
  env.curriculum_two_feet=False env.initial_position=safe \
  env.tricky_terrain=False env.include_events_randomization=False \
  "env.forces_applied_to_base_curriculum=[0.0]" "env.base_push_force_z_range=[0.0,0.0]"
```

## Checkpoint compatibility

SAC and PPO checkpoints are intentionally different. `train_sac.py` accepts
RSL-RL-SAC checkpoints, not PPO/LoRA PPO checkpoints. PPO training remains on
the existing RSL-RL 3.1.2 path and is not migrated or altered by SAC.

### The action map belongs to the policy, not to the simulator

A SAC actor emits `action = action_range * tanh(latent) + action_bias`, and both
`action_range` and `action_bias` are stored in the checkpoint. They are part of the
trained policy. A loaded checkpoint keeps them, and `train_sac.py` only prints them
at startup.

Do not reset them from the target simulator's joint ranges. Both simulators apply
`target = SAFE_Q + ACTION_SCALE * action`, so the raw `solo12.xml` ranges are much
wider than the span any trained policy uses. Replacing the checkpoint values with
those ranges multiplied Solo12 hip targets by 3.6x and thigh targets by 2.1x, and
moved the neutral calf pose by 45 degrees. A checkpoint that walked for the whole
20 s episode then fell on its base after 0.28 s, and the resulting 13-step episodes
looked like a policy that had learned nothing during pretraining. Runs
`jordibelp/solo12-two-feet-lora/8bjzi9ap` and `4mtmdake` are examples.

`MjxSolo12VecEnv._action_bounds` still derives bounds from the XML for a run that
starts without a checkpoint. It measures them from the `SAFE_Q` action centre,
because `q=0` is not the centre of this action space.
