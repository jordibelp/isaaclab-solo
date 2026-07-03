# Solo12 DreamerV3 Tuning Guide

> **High-performance rewrite (branch `high-performance-dreamerV3`).** The learning
> core was reimplemented in `source/scripts/dreamer/dreamer_core/` following the
> efficient PyTorch DreamerV3 reproduction from the R2-Dreamer codebase (using the
> plain `rep_loss=dreamer` objective — no decoder-free / augmentation variants).
> `train.py` keeps all the Isaac Lab glue (env stepping, command handling,
> W&B/TensorBoard, checkpoint/best, optional Continual-Backprop).
>
> **Efficiency techniques** (all in `dreamer_core`):
> block-diagonal GRU + RMSNorm RSSM; a single **fused backward** over the combined
> world-model+actor+critic loss with imagination rolled out through **frozen
> network views** (up-to-date weights, no autograd graph through the rollout);
> **AMP** fp16 + `torch.compile(reduce-overhead)` with CUDA graphs; **LaProp**
> optimizer with **adaptive gradient clipping**; and a **GPU-resident replay
> buffer with latent caching** (compact int16/fp16 replay-context, warm-starting
> the RSSM instead of a cold zero state each batch).
>
> **DreamerV3 pieces added that the previous single-file version omitted:**
> the slow-critic EMA regularizer (`slowreg`), replay-based critic learning
> (`repval`), replay-context latent carry, and the faithful bounded-normal actor.
>
> **Microbenchmark** (RTX 5070 Laptop, 8 GB; synthetic Solo12-shaped update, no
> Isaac Sim — see `source/scripts/dreamer/benchmark_update.py`):
>
> | batch | legacy ms/update | new (AMP+compile) | speedup | legacy mem | new mem |
> |------:|-----------------:|------------------:|--------:|-----------:|--------:|
> | 256   | 269              | 163               | 1.66x   | 3.0 GB     | 1.5 GB  |
> | 512   | 562              | 323               | 1.74x   | 5.7 GB     | 3.0 GB  |
> | 1024  | OOM              | 666               | —       | OOM        | 5.8 GB  |
>
> The new core is both faster and ~2x more memory-efficient, so it fits batches
> (B=1024) that the previous implementation cannot on the same GPU. The table
> speedups need `agent.use_compile=true`; compile is **disabled by default**
> (`use_compile=False`) because on the IRICluster Torch Inductor can stall while
> querying `nvidia-smi` during the first compiled update. Enable it locally for
> the full speedup — the eager block-GRU path is intentionally slower. On small
> GPUs set `agent.replay_storage_device=cpu` or lower `agent.replay_size` if the
> on-GPU replay does not fit alongside Isaac Sim.
>
> **Stability fixes (2026-07-02)** after the first cluster runs showed
> reach-10-then-collapse reward cycles (actor std saturating at `max_std`,
> entropy pinned at ~17.03 = 12·ln(√(2πe))):
> 1. **Arrival-aligned replay + deferred reset** — reward/terminal flags now
>    describe the transition *into* each stored obs and true terminal
>    observations are preserved, matching danijar/r2dreamer (see
>    "Reset/terminal handling" below). Previously both were shifted one step.
> 2. **Two-hot mean numerics** — `TwoHotSymexp.mean()` now pairs the symmetric
>    ±bins before reducing (as r2dreamer does). The naive fp32 dot product over
>    bins reaching ±4.9e8 left ~0.3 absolute noise in reward/value predictions
>    at high head entropy — 30x a typical Solo12 per-step reward (~0.01).
> 3. The trainer now **warns when the replay depth per env is under two
>    episodes** (e.g. `replay_size=2M` at 2048 envs is only 976 steps/env ≈ 1
>    episode): a too-recent buffer lets the world model forget failure modes,
>    which matches the observed collapse periodicity. Prefer scaling
>    `agent.replay_size` with `num_envs`.

This file documents the tunable parameters in `dreamer_v3_cfg.py` for the
`Solo12-simple-dreamerV3` task and the local trainer at
`source/scripts/dreamer/train.py`. The learning core is in
`source/scripts/dreamer/dreamer_core/` (`agent.py`, `rssm.py`, `networks.py`,
`optim.py`, `distributions.py`, `buffer.py`).

The Dreamer task exposes:

- `obs["policy"]`: by default, 36D = 33D proprioception plus 3D command
  (`vx_cmd`, `vy_cmd`, `wz_cmd`). This full observation is symlog-transformed by
  the encoder and reconstructed by the decoder.
- `obs["command"]`: absent by default. Set `agent.command_outside_observation=true`
  to restore the old layout with 33D proprioception in `obs["policy"]` and 3D
  command returned separately here.
- Actions: 12D joint-position action targets, soft-clipped to roughly `[-1, 1]`
  by the actor and then handled by the Solo12 environment action path.

With the default layout, the Dreamer feature used by reward prediction, continue
prediction, actor, and critic is:

```text
[discrete stochastic RSSM state, deterministic RSSM state]
```

(the flattened `stoch_dim * num_bins_encoding` one-hot latent followed by the
`hidden_vector_deter_dims` GRU state). With
`agent.command_outside_observation=true`, the 3D command is concatenated after
the feature: `[discrete stochastic RSSM state, deterministic RSSM state, command]`.

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
  --run-name "[Local]-Solo12 high-performance DreamerV3"
```

All fields in `dreamer_v3_cfg.py` can also be overridden without editing the
file by using IsaacLab/Hydra-style `agent.<field>=<value>` command-line
overrides. This is the preferred way to launch cluster sweeps without creating
repo diffs:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  agent.num_batches_trained_per_iteration=32 \
  agent.batch_size=1024 \
  agent.hidden_vector_deter_dims=256 \
  agent.stoch_dim=32 \
  agent.num_bins_encoding=32 \
  agent.model_lr=5e-5 \
  agent.use_compile=true \
  'agent.actor_hidden_dims=[512,256,128]' \
  'agent.run_name="[Cluster]-dreamer-deter256"'
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
| `num_batches_trained_per_iteration` | `16` | Number of replay batches trained (one fused world+actor+critic update each) after each collection block. | Increase when the replay buffer grows faster than the model learns; decrease if training dominates wall time or overfits stale data. |
| `command_outside_observation` | `False` | Selects the Dreamer observation contract. `False` puts the command inside `obs["policy"]`; `True` restores the old separate `obs["command"]` conditioning appended to the Dreamer feature. | Keep `False` for new runs; use `True` only for old-layout ablations or checkpoint compatibility. |
| `prefill_steps` | `8192` | Minimum replay transitions before training starts. Before this, actions are uniform random in `[-1, 1]`. | Increase for more diverse initial data; decrease for faster first updates. Must be at least enough to sample one sequence. |
| `replay_size` | `2_000_000` | Maximum stored transitions across all envs. Internally this is converted to `replay_size // num_envs` time steps per env. | Increase for more diverse replay and less forgetting; decrease to save memory. |
| `replay_storage_device` | `"cuda:0"` | Device the replay ring buffer lives on. On GPU, collection is a plain in-place tensor write with no host↔device traffic. | Set to `"cpu"` on small GPUs where the on-GPU replay does not fit alongside Isaac Sim. |
| `replay_recent_fraction` | `0.1` | Fraction of each sampled batch biased toward the most-recent steps (replaces the old online replay queue). `0.0` means pure uniform sampling. | Raise to weight training toward fresh data; set `0.0` for a pure-uniform-replay ablation. |
| `use_uniform_replay_buffer_with_online_queue` | `True` | Legacy compatibility flag. When `True` and `replay_recent_fraction == 0`, the trainer falls back to `replay_recent_fraction = 0.1` so old configs keep mixing in recent data. Otherwise `replay_recent_fraction` is used directly. | Prefer setting `replay_recent_fraction` explicitly; set this `False` only when you also want `replay_recent_fraction=0` for pure uniform replay. |
| `batch_size` | `2048` | Number of sequence snippets per training batch. | Increase for smoother gradients if memory allows; decrease if CUDA memory is tight. |
| `batch_length` | `24` | Number of steps per replay sequence used to train the world model. The buffer actually reads `batch_length + 1` steps; the extra first step is a **context step** that warm-starts the RSSM from the cached posterior latent. | Increase for longer temporal credit/dynamics; decrease for faster updates and lower memory. |

**Replay buffer and latent caching.** The buffer is a single GPU-resident ring
buffer shared across all vectorized envs. Every stored step also caches the RSSM
posterior latent computed when the action was chosen, compactly: the discrete
stochastic state as an int16 argmax index and the deterministic state in fp16.
Each sampled slice therefore starts from an up-to-date warm state (`initial_stoch`
/ `initial_deter`) instead of a cold zero state, and after each update the freshly
recomputed posteriors are written back into the cache (`update_latents`).

Useful ratios:

- Collection per iteration: `num_envs * steps_per_env`.
- Replay time depth per env: `replay_size // num_envs`.
- Batch transitions per world-model update: `batch_size * batch_length`.
- Gradient updates per policy/control step (`num_gradients_per_policy_step`):
  `num_batches_trained_per_iteration / (num_envs * steps_per_env)`.
- Replay training transitions per collected transition (`replay_ratio`):
  `(num_batches_trained_per_iteration * batch_size * batch_length) / (num_envs * steps_per_env)`.
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
  agent.num_batches_trained_per_iteration=8 \
  agent.batch_size=1024 \
  agent.batch_length=24 \
  'agent.run_name="[Cluster]-dreamer-bs1024-train8"'
```

Then compare against `agent.num_batches_trained_per_iteration=16` and
`agent.batch_size=2048` one at a time. A larger batch is useful if it lowers
model/critic noise without slowing policy improvement, but using it together
with many replay updates greatly increases replay reuse. If return improves
slowly while `total_loss` looks good and `action_entropy` stays high, reduce
`num_batches_trained_per_iteration` or `batch_size` before increasing model size.

## World Model Size

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `hidden_vector_deter_dims` | `128` | Size of the deterministic block-GRU/RSSM state `h_t`. Must be divisible by `blocks` (the trainer auto-lowers `blocks` to the largest divisor ≤ the configured value if not). | Increase if the model cannot predict observations/rewards; decrease for speed. |
| `stoch_dim` | `32` | Number of categorical stochastic variables in the RSSM state. The sampled latent has this many independent one-hot variables. | Increase for more latent capacity; decrease for speed/stability. |
| `num_bins_encoding` | `32` | Number of classes per stochastic variable. Total stochastic feature size is `stoch_dim * num_bins_encoding`. | Increase for richer discrete latents; decrease if KL/losses are unstable or memory is high. |
| `encoder_hidden_dims` | `[128, 128]` | MLP hidden sizes for encoding symlog observations. A final linear projects to `model_hidden_dim` (the embedding fed to the posterior). | Increase if observation reconstruction is poor; decrease for faster training. |
| `model_hidden_dim` | `256` | RSSM width: encoder embedding size, posterior/prior MLP hidden width, and the block-GRU input-projection width. | Main RSSM capacity knob. |
| `head_hidden_dims` | `[256, 256]` | Hidden trunk sizes for the decoder, reward head, and continue head. | Increase if reconstruction/reward/continue prediction is poor; decrease for faster/lighter heads. |
| `actor_hidden_dims` | `[256, 128, 64]` | MLP hidden sizes for the bounded-normal actor. | Increase if policy seems underpowered after the model is learning; decrease for faster actor updates. |
| `critic_hidden_dims` | `[256, 128, 64]` | MLP hidden sizes for the critic and the slow target critic. | Increase if critic loss/returns are noisy; decrease if critic dominates compute. |
| `reward_value_num_bins` | `255` | Number of bins for the DreamerV3-style symexp-two-hot reward and value distributions. | Keep `255` unless explicitly ablating output distribution resolution. |
| `reward_value_symlog_range` | `20.0` | Range used to create symlog-spaced raw-value bins: `symexp(linspace(-range, range))`. | Keep `20.0` to match DreamerV3; reduce only if a narrower value support is deliberately desired. |

The model predicts symlog-transformed observations with a Gaussian-in-symlog-space
head (sum-of-squared error against the symlog-warped target). Reward and value
heads use DreamerV3-style symexp-two-hot categorical distributions: the heads
output logits over symlog-spaced raw-value bins, prediction is the expected raw
reward/value, and training uses interpolated two-hot cross entropy against raw
targets.

## Block-GRU / RSSM Structure

These control the DreamerV3 blocked recurrent core (`dreamer_core/rssm.py`). The
recurrence projects and RMSNorms its inputs, splits the deterministic state into
`blocks` independent groups, and combines them with a block-diagonal linear layer
(fewer FLOPs/params than a dense GRU at the same width — a large part of the
speedup).

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `blocks` | `8` | Number of block-diagonal groups in the recurrent core. `hidden_vector_deter_dims` must be divisible by this. | Lower for a denser (slower, higher-capacity) recurrence; keep the divisibility constraint in mind. |
| `obs_layers` | `1` | Depth of the posterior MLP `q(stoch \| deter, embed)`. | Increase if the posterior underfits observations. |
| `img_layers` | `2` | Depth of the prior MLP `p(stoch \| deter)` used in imagination. | Increase if imagined dynamics are inaccurate. |
| `dyn_layers` | `1` | Number of block-GRU hidden layers before the gating projection. | Increase for a deeper recurrent transition; decrease for speed. |
| `unimix` | `0.01` | Uniform mixture blended into every discrete latent categorical, preventing degenerate zero-probability classes. | Keep at `0.01` to match DreamerV3; change only for ablations. |

## Imagination And Value Learning

Actor and critic are trained on imagined rollouts that start from **every**
posterior state in each sampled replay sequence (`batch_size * batch_length`
starts). The rollout is produced by frozen network views under `no_grad`, so no
autograd graph runs through the sequence; termination is handled by continuation
weighting in the lambda return rather than by dropping starts.

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `imag_horizon` | `15` | Number of imagined RSSM steps used for actor/critic training. | Increase for longer-horizon behavior; decrease if imagined rollouts become unreliable or slow. |
| `discount` | `0.99` | Base discount used with predicted continuation probability (≈2 s effective horizon). | Lower for more short-sighted policies; raise for longer-horizon tracking if the critic remains stable. |
| `lambda_` | `0.95` | Lambda-return mixing factor. | Lower for lower-variance targets; raise for more long-horizon bootstrapping. |
| `slow_critic_tau` | `0.01` | Soft-update rate from the critic to the slow target critic (applied once per fused update). | Lower for a slower, steadier target; higher for faster tracking of critic changes. |
| `normalize_actor_returns` | `True` | Enables DreamerV3-style percentile return scaling for the actor update. The actor advantage is divided by a running return scale, while the critic still predicts raw lambda-return targets. | Keep enabled unless comparing against the previous local implementation. |
| `return_norm_rate` | `0.01` | EMA rate for the actor return percentile range. | Increase for faster adaptation to changing reward scale; decrease for smoother scaling. |
| `return_norm_limit` | `1.0` | Minimum denominator for return normalization. This prevents tiny return ranges from amplifying noise. | Keep at `1.0` for DreamerV3-style behavior. |
| `return_norm_percentile_low` | `5.0` | Lower percentile used for the running return range. | Change only for ablations. |
| `return_norm_percentile_high` | `95.0` | Upper percentile used for the running return range. | Change only for ablations. |

In this implementation, imagined per-step discounts are:

```text
discount * sigmoid(continue_head(feature))
```

So a low `imag_cont` metric shortens the effective imagination horizon even if
`imag_horizon` is large.

**Reset/terminal handling (arrival-aligned, fixed 2026-07-02).** The trainer
patches the Direct env with a **deferred reset**
(`source/scripts/dreamer/deferred_reset.py`, adapted from r2dreamer): when an
env terminates or times out, its true terminal observation is returned and
stored, and the reset is applied one step later (that env sims one discarded
zero-action "junk" step). Each replay slot therefore holds `obs_t` together
with the reward and `is_terminal`/`is_last` flags **that arrived with**
`obs_t` (produced by the transition into it), plus the action chosen **from**
`obs_t` (zeroed on terminal observations). `is_first` marks fresh reset
observations, whose reward is zeroed and where the RSSM zeroes its carried
state. This matches danijar's DreamerV3 and r2dreamer exactly: the reward and
continue heads are readouts of the arrival state (the continue target is
`1 - is_terminal`, so time-limit resets do not teach the model that the task
ended), imagination correctly credits each action's immediate reward, and the
transition into death states is part of world-model training. The previous
convention (auto-reset, reward/flags describing the transition *out of* the
stored state) made the reward head predict the outcome of a not-yet-taken
action and hid action-conditioned terminations from imagination — a likely
driver of the reach-10-then-collapse reward instability.

## Loss Weights

The world model, actor, and critic are optimized together by a **single fused
backward** over one weighted total loss. The logged `total_loss` is:

```text
total_loss =
  obs_loss_scale      * loss/decoder
+ reward_loss_scale   * loss/reward
+ continue_loss_scale * loss/cont
+ kl_dyn_scale        * loss/dyn
+ kl_rep_scale        * loss/rep
+ policy_loss_scale   * loss/policy
+ value_loss_scale    * loss/value
+ repval_loss_scale   * loss/repval
```

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `free_nats` | `1.0` | Minimum KL contribution per step for both KL terms. Prevents tiny KL values from over-regularizing. | Increase if the posterior collapses or KL is too aggressively optimized; decrease if latent capacity is not being used enough. |
| `kl_dyn_scale` | `0.5` | Weight on `KL(stop_grad(posterior) \|\| prior)`, pushing the prior dynamics toward posterior states. | Increase when imagined dynamics drift; decrease if KL dominates model learning. |
| `kl_rep_scale` | `0.1` | Weight on `KL(posterior \|\| stop_grad(prior))`, regularizing the posterior representation. | Increase for more compact/stable representations; decrease if reconstruction/reward learning suffers. |
| `obs_loss_scale` | `1.0` | Weight on symlog observation reconstruction (`loss/decoder`). | Increase if observations are poorly modeled; decrease if it overwhelms reward/continue learning. |
| `reward_loss_scale` | `1.0` | Weight on reward two-hot cross entropy (`loss/reward`). | Increase if reward prediction is poor and actor training is noisy; decrease if reward loss dominates. |
| `continue_loss_scale` | `1.0` | Weight on continuation BCE (`loss/cont`, target `1 - is_terminal`). | Increase if true terminal prediction is wrong; decrease if continue loss dominates early training. |
| `policy_loss_scale` | `1.0` | Weight on the imagined actor loss (`loss/policy`). | Rarely changed; lower only to soften the actor's pull on shared compute. |
| `value_loss_scale` | `1.0` | Weight on the imagined critic loss (`loss/value`). | Rarely changed. |
| `repval_loss_scale` | `0.3` | Weight on the replay-based critic loss (`loss/repval`). `0` disables replay-value learning. | Lower/disable to fall back to imagination-only critic learning; keep >0 for DreamerV3-style replay value. |
| `slowreg` | `1.0` | Strength of the slow-critic EMA regularizer added inside both `loss/value` and `loss/repval` (regresses the critic toward the slow target's own predictions for stability). | Lower if the critic tracks too slowly; keep at `1.0` for DreamerV3 behavior. |
| `actor_entropy_scale` | `3.0e-4` | Entropy bonus folded into `loss/policy`. | Increase if the policy collapses too early; decrease if actions stay noisy and return does not improve. |

The actor is trained only from imagination (`loss/policy`). The critic (`self.value`)
is trained from both imagination (`loss/value`) and real replay sequences
(`loss/repval`); the replay-value term additionally lets a world-model gradient
flow through the critic's posterior features. Resets are handled by the RSSM
`is_first` state zeroing rather than by masking individual reconstruction/reward
losses.

## Actor And Exploration

The actor is a DreamerV3 **bounded-normal** continuous policy. Its network
outputs a mean and a std: the mean is `tanh`-squashed, and the std is mapped to
`[actor_min_std, actor_max_std]` via `min_std + (max_std - min_std) * sigmoid(x + 2)`.
Samples are **raw** diagonal-Gaussian draws; bounding to roughly `[-1, 1]` happens
at the consumers (the env-facing `act()` soft-clips with `x / max(1, |x|)`, and the
RSSM applies the same clip to actions inside its dynamics). The actor's `log_prob`
is always evaluated at the raw sample — evaluating it at the *clipped* sample
biases the REINFORCE score function (`E[∇ log π] ≠ 0`) and couples the mean
advantage level to a systematic drift of std/mean toward saturation, which showed
up as `action_entropy` pinning at its max (17.03 for 12 dims at std 1.0). During
prefill, the trainer ignores the actor and samples uniform random actions in
`[-1, 1]`.

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `actor_min_std` | `0.1` | Lower bound on the actor action std. | Raise for a noisier exploration floor; lower for a more deterministic policy. |
| `actor_max_std` | `1.0` | Upper bound on the actor action std. | Raise to allow more exploration; lower to cap action noise. |
| `actor_entropy_scale` | `3.0e-4` | Entropy bonus (also listed under Loss Weights). | Increase if the policy collapses too early; decrease if actions stay noisy. |

## Optimizers And Stability

A **single LaProp optimizer** holds three parameter groups with independent
learning rates: `world` (encoder, RSSM, decoder, reward head, continue head),
`actor`, and `critic` (the value head). LaProp normalizes the gradient by its
running second moment before applying momentum. Gradients are clipped with
**adaptive gradient clipping (AGC)**, not a global norm clip.

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `model_lr` | `1.0e-4` | LaProp LR for the world-model group (encoder, RSSM, decoder, reward head, continue head). | Lower if world-model losses explode or oscillate; raise cautiously if model learning is too slow. |
| `actor_lr` | `3.0e-5` | LaProp LR for the actor. | Lower if actions/returns are unstable; raise if the actor loss changes very slowly after the world model is usable. |
| `critic_lr` | `1.0e-4` | LaProp LR for the critic (value head). | Lower if critic loss explodes; raise if the critic cannot track imagined returns. |
| `laprop_beta1` | `0.9` | LaProp momentum coefficient. | Rarely changed. |
| `laprop_beta2` | `0.999` | LaProp second-moment coefficient. | Rarely changed. |
| `laprop_eps` | `1.0e-20` | LaProp epsilon (small by design because the gradient is second-moment-normalized before momentum). | Keep as-is unless debugging numerical issues. |
| `agc_clip` | `0.3` | AGC ratio: each tensor's gradient is scaled so `\|\|g\|\| <= agc_clip * max(\|\|p\|\|, agc_pmin)`. | Lower if an optimizer group has spikes/NaNs; raise only if clipping is clearly limiting learning. |
| `agc_pmin` | `1.0e-3` | Floor on the parameter norm used by AGC (protects near-zero parameters). | Rarely changed. |
| `warmup_steps` | `1000` | Linear LR warmup over the first N optimizer steps. | Increase for a gentler start; set `0` to disable warmup. |
| `grad_clip` | `100.0` | Legacy global-norm clip value, kept for reference only. AGC is used instead. | No effect on training; ignore. |

Start by changing learning rates by factors of 2 to 3, not by orders of
magnitude, unless the run is clearly diverging.

### Runtime efficiency

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `use_amp` | `True` | fp16 autocast + `GradScaler` around the fused update. | Disable only if AMP causes numerical problems. |
| `use_compile` | `False` | `torch.compile(reduce-overhead)` with CUDA graphs over the gradient computation. Disabled by default because Inductor can stall querying `nvidia-smi` during the first compiled update on the IRICluster. | Set `true` locally for the microbenchmark speedups; the eager block-GRU path is slower. |

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

- `world`: encoder, RSSM posterior (`_obs_net`) and prior (`_img_net`) MLPs,
  decoder, reward head, and continue head.
- `actor`: bounded-normal actor MLP.
- `critic`: value MLP. The slow target critic is not directly CBP-reset; it
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
  agent.num_batches_trained_per_iteration=32 \
  'agent.run_name="[Cluster]-dreamer-cbp"'
```

Dreamer checkpoints save per-group CBP manager state under the `cbp` key.
Resume with `--use-cbp --checkpoint <path>` to restore utilities, activations,
ages, replacement counters, and optimizer-step counts. The replay buffer is
still not checkpointed.

## Logging And Checkpoints

| Parameter | Default | What it controls | When to change it |
| --- | ---: | --- | --- |
| `log_interval` | `10` | Iteration interval for stdout and scalar logging. Iteration 1 is always logged. | Lower for debugging; raise to reduce logging overhead. |
| `save_interval` | `25` | Iteration interval for numbered checkpoints. A final `last.pt` is saved at the end. | Lower for risky runs; raise to reduce disk usage. |
| `experiment_name` | `"solo12_dreamer_v3"` | Top-level folder under `logs/dreamer/`. | Change to separate experiment families. |
| `run_name` | `"[Local]-Solo12 high-performance DreamerV3"` | Human-readable run suffix. The trainer prepends a timestamp and sanitizes slashes. | Change for each hypothesis so W&B/log folders are searchable. |
| `logger` | `"wandb"` | Logging backend: `"wandb"`, `"tensorboard"`, or `"none"`. | Use `"none"` for fast smoke tests; `"tensorboard"` for local-only logging. |
| `save_best_checkpoint` | `True` | Saves `checkpoints/best_model.pt` whenever `episode/episodic_reward` improves after at least one completed episode. | Disable only for storage-constrained ablations with `agent.save_best_checkpoint=false`. |
| `wandb_project` | `"solo12-dreamer"` | W&B project name. | Change only if moving the run to another project. |
| `wandb_entity` | `None` | Optional W&B entity/team. | Set if W&B needs a non-default entity. |

When W&B is enabled, the run config includes the resolved `agent_cfg`, the
resolved `env_cfg`, the parsed `cli` arguments, and the old top-level Dreamer
agent fields for dashboard compatibility. It also includes derived
`replay_ratio` and `num_gradients_per_policy_step` fields, and logs the same
values as `train/replay_ratio` and `train/num_gradients_per_policy_step`.
The local run folder also appends the W&B run id, making it easy to match a
folder with its W&B run.

The logger calls W&B/TensorBoard with the global environment-interaction count as
the scalar step, and W&B is configured so the default step axis is
`num_env_interactions` (env transitions collected across all parallel envs). The
trainer also logs explicit `num_env_interactions` and `num_optimization_steps`
scalars so plots can be read without relying on the UI label.
`num_optimization_steps` counts fused optimizer steps: **one per trained batch**
(the world model, actor, and critic are updated together in a single step).

Checkpoints are written to:

```text
logs/dreamer/<experiment_name>/<timestamp>_<run_name>/checkpoints/
```

`save_interval` is measured in training-loop iterations. For example,
`agent.save_interval=25` writes `model_25.pt`, `model_50.pt`, `model_75.pt`, and
so on. Since each iteration collects `num_envs * steps_per_env` environment
interactions, a run with `num_envs=4096` and `steps_per_env=24` saves numbered
checkpoints every `2,457,600` env interactions when `save_interval=25`.

Each checkpoint payload stores the full `model` state, the LaProp `optimizer`
state, the AMP `scaler` state, `num_optimization_steps`, and (when CBP is
enabled) per-group CBP state under `cbp`, plus `iteration`, `total_steps`, and
the resolved `cfg`. When `save_best_checkpoint=True`, the trainer also writes
`checkpoints/best_model.pt` whenever `episode/episodic_reward` improves; the best
payload additionally stores `best_model_metric` and `best_model_value`.

Checkpoints from the previous single-file Dreamer implementation are **not**
compatible with this high-performance core: the RSSM (block-GRU), heads
(symexp two-hot), optimizer (LaProp), and frozen-view setup all differ. Only
checkpoints produced by this core can be resumed. When resuming within this core,
match the observation layout: if a checkpoint was trained with the old
command-outside layout, resume it with `agent.command_outside_observation=true`;
otherwise start a fresh run so the encoder, decoder, actor, critic, and optimizer
shapes all match.

Resume with:

```bash
./isaaclab.sh -p source/scripts/dreamer/train.py \
  --task Solo12-simple-dreamerV3 \
  --headless \
  --checkpoint /absolute/path/to/checkpoints/last.pt
```

The replay buffer is not checkpointed, so a resumed run reloads the model and
optimizer but starts with a fresh replay buffer and must prefill again.

## Metrics To Watch

The trainer logs these scalar groups to W&B/TensorBoard when enabled:

| Metric | Meaning |
| --- | --- |
| `total_loss` | Combined fused world-model + actor + critic objective after loss weighting. |
| `loss/decoder` | Mean symlog observation reconstruction (negative log-prob = sum-of-squared error in symlog space). |
| `loss/reward` | Mean reward symexp-two-hot cross entropy. |
| `loss/cont` | Continuation BCE for predicting `1 - is_terminal`. |
| `loss/dyn` | Dynamics KL `KL(stop_grad(posterior) \|\| prior)` after the free-nats clamp. |
| `loss/rep` | Representation KL `KL(posterior \|\| stop_grad(prior))` after the free-nats clamp. |
| `loss/policy` | Imagined policy-gradient objective (with the entropy bonus folded in). |
| `loss/value` | Imagined value symexp-two-hot cross entropy vs lambda returns (plus the slow-critic regularizer). |
| `loss/repval` | Replay-based value cross entropy vs lambda returns from real replay rewards (only logged when `repval_loss_scale > 0`). |
| `prior_entropy` | Mean entropy of the prior categorical over discrete latents. |
| `post_entropy` | Mean entropy of the posterior categorical over discrete latents. |
| `imag_reward` | Mean imagined reward (from the reward head). |
| `imag_value` | Mean imagined value. |
| `imag_return` | Mean lambda return in imagination. |
| `imag_cont` | Mean predicted continuation probability; low values shorten the effective imagination horizon. |
| `return_scale` | Current actor return-normalization denominator (percentile-range EMA, floored at `return_norm_limit`). |
| `action_entropy` | Mean actor entropy in imagined rollouts. |
| `num_env_interactions` | Total environment transitions collected across all parallel envs. This is also the W&B/TensorBoard scalar step. |
| `num_optimization_steps` | Total fused optimizer steps (one per trained batch). |
| `train/iteration` | Current collect/train loop iteration. |
| `train/env_steps` | Alias for `num_env_interactions`. |
| `train/fps` | Environment steps per second since run start. |
| `train/replay_ratio` | Replay training transitions per collected transition. |
| `train/num_gradients_per_policy_step` | Fused updates per collected control step. |
| `replay/steps` | Number of transitions inserted into replay. |
| `replay/recent_fraction` | Fraction of each batch biased toward recent replay steps. |
| `episode/episodic_reward` | Mean undiscounted completed-episode reward over the latest (up to `num_envs`) completed episodes. |
| `checkpoint/best_episodic_reward` | Best `episode/episodic_reward` value that has triggered `best_model.pt` so far. |
| `checkpoint/best_iteration` | Training iteration where the current `best_model.pt` was saved. |
| `CBP/<group>/optimizer_steps` | Optimizer steps tracked by CBP for `world`/`actor`/`critic` (only with `--use-cbp`). |
| `CBP/<group>/replacements_last_update` | Hidden units replaced in the last CBP update for that group. |
| `CBP/<group>/replacements_total` | Total hidden units replaced by CBP for that group. |
| `env/RewardsPerStep/*` | Reward-term diagnostics emitted by the Solo12 env. |

The first logged iterations usually show zero losses because the replay buffer
has not reached `prefill_steps` yet. Real training metrics begin once:

```text
replay/steps >= prefill_steps
```

## Practical Tuning Order

1. Keep architecture fixed and verify data flow.
   Watch `replay/steps`, `total_loss`, `loss/decoder`, `loss/reward`, and
   `loss/cont`.
2. If model losses are unstable, reduce `model_lr`, then consider lowering
   `num_batches_trained_per_iteration` or model size (AGC already bounds gradient
   spikes, so tune `agc_clip` only if you see NaNs).
3. If the model learns but returns stay flat, tune `actor_entropy_scale`,
   `actor_lr`, `critic_lr`, `imag_horizon`, and `discount`.
4. If imagined metrics look unrealistic, prioritize `kl_dyn_scale`,
   `kl_rep_scale`, `free_nats`, `batch_length`, and world-model capacity before
   tuning the actor.
5. If training is slow but stable, increase `num_envs`,
   `num_batches_trained_per_iteration`, or `batch_size` one at a time (and enable
   `agent.use_compile=true` locally), checking GPU memory and `train/fps`.

Prefer changing one group of parameters per run and put the hypothesis in
`run_name`; the resolved `agent.yaml` already captures the exact values.
