# Solo12 DreamerV3 Tuning Guide

This file documents the tunable parameters in `dreamer_v3_cfg.py` for the
`Solo12-simple-dreamerV3` task and the local trainer at
`source/scripts/dreamer/train.py`.

The Dreamer task exposes:

- `obs["policy"]`: 33D proprioception: base linear velocity, base angular
  velocity, projected gravity, joint positions, and joint velocities.
- `obs["command"]`: 3D command: `vx_cmd`, `vy_cmd`, `wz_cmd`.
- Actions: 12D joint-position action targets, squashed to `[-1, 1]` by the actor
  and then handled by the Solo12 environment action path.

The Dreamer feature used by reward prediction, continue prediction, actor, and
critic is:

```text
[deterministic RSSM state, discrete stochastic RSSM state, command]
```

## Where To Tune

Persistent defaults live in:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/dreamer_v3_cfg.py
```

Common run-level overrides are available through script flags:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --num_envs 1024 \
  --max_iterations 10000 \
  --logger wandb \
  --run-name "[Local]-Solo12 simple DreamerV3"
```

All fields in `dreamer_v3_cfg.py` can also be overridden without editing the
file by using IsaacLab/Hydra-style `agent.<field>=<value>` command-line
overrides. This is the preferred way to launch cluster sweeps without creating
repo diffs:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  agent.train_steps_per_iteration=32 \
  agent.batch_size=1024 \
  agent.hidden_vector_deter_dims=128 \
  agent.stoch_dim=32 \
  agent.num_bins_encoding=64 \
  agent.model_lr=5e-5 \
  'agent.actor_hidden_dims=[512,256,128]' \
  'agent.run_name="[Cluster]-dreamer-bins64"'
```

Use `agent.<field>` for Dreamer runner parameters and `env.<field>` for
environment parameters, following the normal IsaacLab convention. Quote list
values and names containing shell-special characters. The explicit argparse
flags `--num_envs`, `--seed`, `--max_iterations`, `--run-name`, and `--logger`
are convenience aliases; if both forms are provided for the same value, the
explicit flag wins.

Confirm the resolved config in:

```text
logs/dreamer/<experiment_name>/<timestamp>_<run_name>/params/agent.yaml
```

## Runtime And Data Collection

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `seed` | `42` | Random seed for Python, Torch, and the IsaacLab env. `--seed -1` samples a random seed. | Change for repeated runs or robustness checks. |
| `device` | `"cuda:0"` | Simulation and model device unless overridden by IsaacLab app args. | Use another GPU or CPU only for debugging. |
| `num_envs` | `1024` | Number of parallel IsaacLab environments. | Increase for faster collection if GPU memory allows; decrease when PhysX or model training runs out of memory. |
| `max_iterations` | `10000` | Number of collect/train loop iterations. | Increase for full runs; lower for smoke tests. |
| `steps_per_env` | `24` | Environment steps collected per env before each training block. Total new transitions per iteration is `num_envs * steps_per_env`. | Increase to collect more fresh data per update; decrease if learning lags behind collection. |
| `train_steps_per_iteration` | `16` | Number of replay batches trained after each collection block. | Increase when the replay buffer grows faster than the model learns; decrease if training dominates wall time or overfits stale data. |
| `prefill_steps` | `8192` | Minimum replay transitions before training starts. Before this, actions are random uniform actions. | Increase for more diverse initial data; decrease for faster first updates. Must be at least enough to sample `batch_length`. |
| `replay_size` | `2_000_000` | Maximum stored transitions across all envs. Internally this is converted to `replay_size // num_envs` time steps per env. | Increase for more diverse replay and less forgetting; decrease to save host memory. |
| `batch_size` | `2048` | Number of sequence snippets per training batch. | Increase for smoother gradients if memory allows; decrease if CUDA memory is tight. |
| `batch_length` | `24` | Length of each replay sequence used to train the world model. | Increase for longer temporal credit/dynamics; decrease for faster updates and lower memory. |

Useful ratios:

- Collection per iteration: `num_envs * steps_per_env`.
- Replay time depth per env: `replay_size // num_envs`.
- Batch transitions per world-model update: `batch_size * batch_length`.
- Updates per environment step: `train_steps_per_iteration / (num_envs * steps_per_env)`.
- Replay training transitions per collected transition:
  `(train_steps_per_iteration * batch_size * batch_length) / (num_envs * steps_per_env)`.
  This is the main ratio to watch when scaling to large cluster GPUs: increasing
  `batch_size` can silently turn a mostly data-collection-limited run into a
  replay-heavy run.

For a 45 GB cluster GPU, a practical first sweep is:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --num_envs=10000 \
  agent.steps_per_env=24 \
  agent.train_steps_per_iteration=8 \
  agent.batch_size=1024 \
  agent.batch_length=24 \
  'agent.run_name="[Cluster]-dreamer-bs1024-train8"'
```

Then compare against `agent.train_steps_per_iteration=16` and
`agent.batch_size=2048` one at a time. A larger batch is useful if it lowers
model/critic noise without slowing policy improvement, but using it together
with many replay updates greatly increases replay reuse. If return improves
slowly while `loss/model` looks good and `policy/entropy` stays high, reduce
`train_steps_per_iteration` or `batch_size` before increasing model size.

## World Model Size

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `hidden_vector_deter_dims` | `128` | Size of the deterministic GRU/RSSM state `h_t`. | Increase if the model cannot predict observations/rewards; decrease for speed. |
| `stoch_dim` | `32` | Number of categorical stochastic variables in the RSSM state. The sampled latent has this many independent one-hot variables. | Increase for more latent capacity; decrease for speed/stability. |
| `num_bins_encoding` | `32` | Number of classes/bins per stochastic variable. Total stochastic feature size is `stoch_dim * num_bins_encoding`. | Increase for richer discrete latents; decrease if KL/losses are unstable or memory is high. |
| `encoder_hidden_dims` | `[128, 128]` | MLP hidden sizes for encoding symlog observations before posterior inference. | Increase if observation reconstruction is poor; decrease for faster training. |
| `model_hidden_dim` | `256` | Shared hidden size for RSSM prior/posterior, decoder, reward head, and continue head. | Main world-model capacity knob. Increase before making all subnetworks custom. |
| `actor_hidden_dims` | `[256, 128, 64]` | MLP hidden sizes for the tanh-normal actor. | Increase if policy seems underpowered after the model is learning; decrease for faster actor updates. |
| `critic_hidden_dims` | `[256, 128, 64]` | MLP hidden sizes for critic and slow target critic. | Increase if critic loss/returns are noisy; decrease if critic dominates compute. |

The model predicts symlog-transformed observations and rewards. Rewards used for
imagined actor-critic are transformed back with `symexp`.

## Imagination And Value Learning

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `imag_horizon` | `15` | Number of imagined RSSM steps for actor/critic training. | Increase for longer-horizon behavior; decrease if imagined rollouts become unreliable or slow. |
| `discount` | `0.99` | Base discount used with predicted continuation probability. | Lower for more short-sighted policies; raise for longer-horizon tracking if critic remains stable. |
| `lambda_` | `0.95` | Lambda-return mixing factor. | Lower for lower-variance targets; raise for more long-horizon bootstrapping. |
| `slow_critic_tau` | `0.01` | Soft-update rate from critic to target critic. | Lower for a slower, steadier target; higher for faster tracking of critic changes. |

In this implementation, imagined discounts are:

```text
discount * sigmoid(continue_head(feature))
```

So a low `imag/continue` metric shortens the effective imagination horizon even
if `imag_horizon` is large.

## World-Model Loss Weights

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `free_nats` | `1.0` | Minimum KL contribution per step for both KL terms. Prevents tiny KL values from over-regularizing. | Increase if the posterior collapses or KL is too aggressively optimized; decrease if latent capacity is not being used enough. |
| `kl_dyn_scale` | `0.5` | Weight on `KL(stop_grad(posterior) || prior)`, pushing the prior dynamics toward posterior states. | Increase when imagined dynamics drift; decrease if KL dominates model learning. |
| `kl_rep_scale` | `0.1` | Weight on `KL(posterior || stop_grad(prior))`, regularizing posterior representation. | Increase for more compact/stable representations; decrease if reconstruction/reward learning suffers. |
| `obs_loss_scale` | `1.0` | Weight on symlog observation reconstruction MSE. | Increase if observations are poorly modeled; decrease if observation loss overwhelms reward/continue learning. |
| `reward_loss_scale` | `1.0` | Weight on symlog reward prediction MSE. | Increase if reward prediction is poor and actor training is noisy; decrease if reward loss dominates. |
| `continue_loss_scale` | `1.0` | Weight on continuation prediction BCE. Continue target is `1 - done`. | Increase if done/timeout prediction is wrong; decrease if continue loss dominates early training. |

The logged world-model loss is:

```text
loss/model =
  obs_loss_scale * obs_loss
+ reward_loss_scale * reward_loss
+ continue_loss_scale * continue_loss
+ kl_dyn_scale * kl_dyn
+ kl_rep_scale * kl_rep
```

## Actor And Exploration

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `actor_entropy_scale` | `3.0e-4` | Entropy bonus in imagined actor loss. | Increase if policy collapses too early; decrease if actions stay noisy and return does not improve. |

The actor is a tanh-normal policy. Its mean and log standard deviation are
predicted from Dreamer features, with `log_std` clamped to `[-5, 2]` in code.
During prefill, the trainer ignores the actor and samples uniform random actions.

## Optimizers And Stability

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `model_lr` | `1.0e-4` | Adam learning rate for RSSM, encoder, decoder, reward head, and continue head. | Lower if world-model losses explode or oscillate; raise cautiously if model learning is too slow. |
| `actor_lr` | `3.0e-5` | Adam learning rate for the actor. | Lower if actions/returns are unstable; raise if actor loss changes very slowly after the world model is usable. |
| `critic_lr` | `1.0e-4` | Adam learning rate for critic and target-critic source network. | Lower if critic loss explodes; raise if critic cannot track imagined returns. |
| `grad_clip` | `100.0` | Global gradient norm clip for world model, actor, and critic updates. | Lower if any optimizer has spikes/NaNs; raise only if clipping is clearly limiting learning. |

Start by changing learning rates by factors of 2 to 3, not by orders of
magnitude, unless the run is clearly diverging.

## Continual Backpropagation

Dreamer training supports the same Continual Backpropagation (CBP) defaults used
in the RSL-RL Solo12 runs. Enable it with:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --use-cbp
```

CBP is attached to hidden units in three separate optimizer groups:

- `world`: encoder, prior, posterior, decoder, reward head, and continue head
  MLPs.
- `actor`: tanh-normal actor MLP.
- `critic`: critic MLP. The slow target critic is not directly CBP-reset; it
  follows the critic through the usual soft update.

| Parameter / flag | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `use_cbp` / `--use-cbp` | `False` | Enables CBP for Dreamer MLP hidden units. Can also be enabled with `agent.use_cbp=true`. | Use for plasticity experiments or long runs where stale hidden units may accumulate. |
| `cbp_replacement_rate` / `--cbp-replacement-rate` | `1.0e-4` | Expected fraction of mature hidden units replaced per optimizer step. | Increase for stronger plasticity; decrease if training becomes too disruptive. |
| `cbp_maturity_threshold` / `--cbp-maturity-threshold` | `10000` | Number of optimizer steps before a hidden unit can be replaced. | Lower for short smoke experiments; keep high for real runs to avoid early churn. |
| `cbp_decay_rate` / `--cbp-decay-rate` | `0.99` | EMA decay for activation and utility estimates. | Lower for faster adaptation; raise for smoother utility estimates. |
| `cbp_util_type` / `--cbp-util-type` | `"contribution"` | Utility rule used to choose low-utility mature units. | Keep `contribution` unless comparing CBP variants. |
| `cbp_init` / `--cbp-init` | `"kaiming"` | Initialization bound for reset incoming weights. | Change only for ablations against the RSL-RL CBP setup. |
| `cbp_accumulate` / `--cbp-accumulate` / `--no-cbp-accumulate` | `True` | Accumulates fractional expected replacements across optimizer steps. | Disable only for direct comparison with non-accumulating replacement schedules. |

CBP settings can be provided either as parser flags or Hydra-style agent
overrides. Explicit `--cbp-*` flags win over `agent.cbp_*` values:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --use-cbp \
  --cbp-replacement-rate 1e-4 \
  agent.train_steps_per_iteration=32 \
  'agent.run_name="[Cluster]-dreamer-cbp"'
```

Dreamer checkpoints save exact per-neuron CBP state under
`continual_backprop_state_dict`, plus a summary under `continual_backprop`.
Resume with `--use-cbp --checkpoint <path>` to restore utilities, activations,
ages, replacement counters, and optimizer-step counts. The replay buffer is
still not checkpointed.

## Logging And Checkpoints

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `log_interval` | `10` | Iteration interval for stdout and scalar logging. Iteration 1 is always logged. | Lower for debugging; raise to reduce logging overhead. |
| `save_interval` | `100` | Iteration interval for numbered checkpoints. A final `last.pt` is saved at the end. | Lower for risky runs; raise to reduce disk usage. |
| `experiment_name` | `"solo12_dreamer_v3"` | Top-level folder under `logs/dreamer/`. | Change to separate experiment families. |
| `run_name` | `"[Local]-Solo12 simple DreamerV3"` | Human-readable run suffix. The trainer prepends a timestamp and sanitizes slashes. | Change for each hypothesis so W&B/log folders are searchable. |
| `logger` | `"wandb"` | Logging backend: `"wandb"`, `"tensorboard"`, or `"none"`. | Use `"none"` for fast smoke tests; `"tensorboard"` for local-only logging. |
| `wandb_project` | `"borinotIsaacLab"` | W&B project name. | Change only if moving the run to another project. |
| `wandb_entity` | `None` | Optional W&B entity/team. | Set if W&B needs a non-default entity. |

When W&B is enabled, the run config includes the resolved `agent_cfg`, the
resolved `env_cfg`, the parsed `cli` arguments, and the old top-level Dreamer
agent fields for dashboard compatibility.

Checkpoints are written to:

```text
logs/dreamer/<experiment_name>/<timestamp>_<run_name>/checkpoints/
```

Resume with:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --checkpoint /absolute/path/to/checkpoints/last.pt
```

The replay buffer is not checkpointed, so a resumed run reloads the model and
optimizers but starts with a fresh replay buffer and must prefill again.

## Metrics To Watch

The trainer logs these scalar groups to W&B/TensorBoard when enabled:

| Metric | Meaning |
| --- | --- |
| `loss/model` | Combined world-model objective after loss weighting. |
| `loss/obs` | Mean symlog observation reconstruction MSE on non-terminal valid steps. |
| `loss/reward` | Mean symlog reward prediction MSE. |
| `loss/continue` | Binary cross-entropy for predicting `1 - done`. |
| `loss/kl_dyn` | Prior dynamics KL term after free-nats clamp. |
| `loss/kl_rep` | Posterior representation KL term after free-nats clamp. |
| `loss/actor` | Imagined policy-gradient objective with entropy bonus. |
| `loss/critic` | MSE between critic values and lambda returns. |
| `imag/reward` | Mean imagined reward after `symexp`. |
| `imag/continue` | Mean predicted continuation probability. |
| `imag/return` | Mean lambda return in imagination. |
| `policy/entropy` | Mean actor entropy in imagined rollouts. |
| `CBP/world/optimizer_steps` | Number of world-model optimizer steps tracked by CBP. |
| `CBP/world/replacements_total` | Total world-model hidden units replaced by CBP. |
| `CBP/actor/optimizer_steps` | Number of actor optimizer steps tracked by CBP. |
| `CBP/actor/replacements_total` | Total actor hidden units replaced by CBP. |
| `CBP/critic/optimizer_steps` | Number of critic optimizer steps tracked by CBP. |
| `CBP/critic/replacements_total` | Total critic hidden units replaced by CBP. |
| `episode/return_mean_100` | Mean completed-episode return over the latest 100 completed episodes. |
| `replay/steps` | Number of transitions inserted into replay. |
| `train/fps` | Environment steps per second since run start. |
| `env/RewardsPerStep/*` | Reward-term diagnostics emitted by the Solo12 env. |

The first logged iterations usually show zero losses because the replay buffer
has not reached `prefill_steps` yet. Real training metrics begin once:

```text
replay/steps >= prefill_steps
```

## Practical Tuning Order

1. Keep architecture fixed and verify data flow.
   Watch `replay/steps`, `loss/model`, `loss/obs`, `loss/reward`, and
   `loss/continue`.
2. If model losses are unstable, reduce `model_lr`, then consider lowering
   `grad_clip`, `train_steps_per_iteration`, or model size.
3. If the model learns but returns stay flat, tune `actor_entropy_scale`,
   `actor_lr`, `critic_lr`, `imag_horizon`, and `discount`.
4. If imagined metrics look unrealistic, prioritize `kl_dyn_scale`,
   `kl_rep_scale`, `free_nats`, `batch_length`, and world-model capacity before
   tuning the actor.
5. If training is slow but stable, increase `num_envs`, `train_steps_per_iteration`,
   or `batch_size` one at a time, checking GPU memory and `train/fps`.

Prefer changing one group of parameters per run and put the hypothesis in
`run_name`; the resolved `agent.yaml` already captures the exact values.
