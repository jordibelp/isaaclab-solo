# Local redundancy during PPO / SAC training

Enabled by default in `source/scripts/rsl_rl/train.py` for both
`rsl_rl_cfg_entry_point` (PPO) and `rsl_rl_sac_cfg_entry_point` (SAC).
No environment-specific changes or extra CLI flags are needed. Existing
`Plasticity/*` metrics remain unchanged.

## What is measured

The implementation uses the regression gradient proxy from
[Local Redundancy, §3.5–3.6](https://arxiv.org/html/2607.13432v1).
It does **not** train the live network (or its copy) to memorize random labels.
For each input, define a synthetic Gaussian regression task:

```text
z_i ~ Normal(0, I)
y_i = stop_gradient(f_theta(x_i)) + sigma * z_i
ell_i = sum_j (f_theta(x_i)_j - y_ij)^2 / (2 * sigma^2)
local_redundancy = mean_i ||gradient_theta ell_i||_2^2
```

This is a Monte Carlo estimate of the per-input first-order gradient coefficient,
not an exact redundancy, finite-step information gain, or a score in bits.
With independent centered targets, the expectation of the **summed-dataset**
squared gradient is `N` times this per-input score: cross-example score terms
have zero expectation. We normalize by input count so changing the number of
samples does not systematically rescale the curve.

Each example's gradient is squared **before** averaging. Squaring a batch-mean
gradient instead allows opposing examples to cancel and changes the scale with
batch size. The implementation computes exact per-example vector–Jacobian
products in memory-bounded microbatches. There is no auxiliary optimizer,
learning rate, fitting budget, or probe checkpoint to manage.

The loss convention is a fixed-variance Gaussian negative log likelihood, omitting
parameter-independent constants. This fixes ambiguities in MSE reduction:
for `sigma=1`, it is **half the sum** of squared output errors per input, not
PyTorch's default mean MSE. Changing `sigma` scales the expected score by
`1 / sigma^2`. Keep it fixed when comparing runs.

## Targets and prediction paths

- **PPO actor:** deterministic action mean via `act_inference`, including any
  privileged/history encoders and shared trunk on that path.
- **SAC actor:** latent Gaussian action mean **before tanh and action scaling**.
  This provides the same mean-function regression diagnostic as PPO. It is not
  the Fisher trace of the learned stochastic policy: exploration standard
  deviations and the log-std-only output rows do not contribute. Shrinking policy
  entropy or changing physical action bounds should not alone inflate this score.
- **PPO critic:** current value prediction, including its learned encoders.
- **SAC critic:** current **online** Q1 and Q2, measured separately on the same
  observation–action inputs and Gaussian noise; `critic/local_redundancy` is
  their arithmetic mean. Never use `min(Q1,Q2)` for the gradient probe, which
  would hide the unselected critic. Frozen Polyak target networks are excluded.

For critics, random targets should be **current prediction + independent noise**,
not unrelated zero-centered targets and not targets centered on SAC's lagged
critic. The former would mix output/value magnitude into the result; the latter
would mix online–target tracking error into it. "Frozen" here means detaching
the current prediction for this measurement, not keeping an old critic across
iterations. Sampling only the detached prediction with no noise gives zero
gradient and measures nothing.

Parameters intentionally frozen by the training setup stay frozen in the probe.
Shared parameters are counted once per prediction path. Independent actor and
critic scores may both contain their shared encoder/trunk contribution.

## Inputs and sample-size choices

Default: **4,096 Gaussian inputs**, generated at the normalized observation
boundary. The disposable copy bypasses only non-learned observation normalizers;
learned encoders, LayerNorm, activations, and all prediction layers remain active.
Thus the synthetic input distribution does not drift with observation-normalizer
statistics or the policy's changing state visitation. Observation-group shapes
are taken from the current rollout, including asymmetric actor/critic groups.
This is an off-distribution synthetic capacity diagnostic, not a claim that the
inputs represent physically realizable robot states.

`input_mode=observations` instead selects up to 4,096 **distinct rows** from the
latest rollout's final vectorized observation, using the real frozen observation
normalizers. This is an on-distribution variant; changes in state visitation can
change its curve. It does not sample the full PPO rollout or SAC replay buffer.
If there are fewer environments, the actual sample count is logged; at least two
rows are required in this mode. Gaussian mode is not capped by environment count.

SAC additionally defaults to uniformly sampled actions inside its configured
action bounds, covering the Q-functions' action domain independently of actor
entropy. `sac_action_source=policy_mean` uses detached, squashed/scaled current
mean actions instead. That option makes critic inputs depend on actor changes.

4,096 is a practical initial accuracy/cost choice, not a convergence guarantee.
Inspect the logged standard error relative to the score; increase samples if it
is too noisy. Default cadence is every 100 learning iterations, including index
0, at the existing post-update logging point (during SAC warm-up there may not
yet have been an update). Microbatch size 16 limits per-example gradient memory;
larger values can improve throughput without changing the estimator or samples.

## Hydra settings

All settings live under `agent.local_redundancy`, are saved in `params/agent.yaml`,
and work identically for both agent entry points. Normal overrides need no `+`.

| Setting | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Enable this diagnostic independently of the older plasticity CLI flag |
| `actor` / `critic` | `true` / `true` | Select branches |
| `interval` | `100` | Learning-iteration cadence |
| `num_samples` | `4096` | Probe input count, at least 2 |
| `batch_size` | `16` | Per-example gradient microbatch size |
| `input_mode` | `gaussian` | `gaussian` or `observations` |
| `input_std` | `1.0` | Gaussian normalized-input standard deviation; ignored for observations |
| `actor_target_std` | `1.0` | Fixed actor regression noise/likelihood standard deviation |
| `critic_target_std` | `1.0` | Fixed value/Q regression noise/likelihood standard deviation |
| `sac_action_source` | `uniform` | `uniform` or `policy_mean`; ignored for PPO |
| `seed` | `1729` | Independent diagnostic seed, deliberately separate from training seed |
| `resample` | `false` | Reuse base inputs/noise across measurements; `true` adds absolute iteration to seed |

For your existing command, **no additions are required**. Example overrides:

```bash
agent.local_redundancy.num_samples=8192 agent.local_redundancy.batch_size=32 agent.local_redundancy.interval=50
```

On-distribution alternative:

```bash
agent.local_redundancy.input_mode=observations agent.local_redundancy.sac_action_source=policy_mean
```

Disable only this probe with `agent.local_redundancy.enabled=false`.
`--no-plasticity-metrics` controls the pre-existing diagnostics, not this probe.
For a no-plasticity-diagnostics control, set both.

## W&B / TensorBoard scalars

Under `Plasticity/actor/`, `Plasticity/critic/`, and SAC's
`Plasticity/critic1/` and `Plasticity/critic2/`:

- `local_redundancy`: mean squared per-example gradient norm; higher means more
  local gradient responsiveness under this fixed probe, not guaranteed better RL performance.
- `local_redundancy_per_output`: divide the raw score by output dimension, useful
  when contrasting a multi-action mean with a scalar value. Still architecture-
  and parameterization-dependent; it does not make arbitrary architectures comparable.
- `local_redundancy_stderr`: sample standard deviation of squared norms / sqrt(N).
- `local_redundancy_num_samples`, `local_redundancy_output_dim`,
  `local_redundancy_parameter_count`: measurement metadata. Parameter count counts
  trainable parameter tensors reached by the path; a state-dependent actor's
  combined mean/std tensor includes the unused std rows in this metadata only.

SAC's aggregate `critic/` reports the first two scalars; detailed statistics are
on `critic1/` and `critic2/`. Standard errors are sampling diagnostics, not
confidence intervals accounting for correlated environment observations.

Global `Plasticity/local_redundancy_seconds` measures probe wall time.
`Plasticity/local_redundancy_valid` is 1 on success, 0 on a failed measurement.
Failed measurements print a warning and retry at the next interval; no artificial
zero score is logged. Only the logging rank probes during distributed training.

## Isolation and reproducibility

Each measurement uses temporary FP32 evaluation-mode network copies, without
optimizer state or training hooks. No live parameters, gradients, buffers,
normalizers, cached distributions, CBP captures, or RNG streams are modified.
Noise is produced by private CPU generators, transferred to the model device,
and shared consistently across microbatch partitions. Fixed-seed probes resume
without extra checkpoint state. The inputs/noise are fixed for a given input
layout; their target **centers** are always recomputed at the current weights.

Supports feed-forward PPO/SAC and feed-forward TCN history encoders. Stateful
recurrent policies are rejected explicitly when enabled. Distillation does not
enable the diagnostic by default. Run the numerical/isolation tests with:

```bash
PYTHONPATH=source/scripts/rsl_rl:source/rsl_rl_sac_vendor python -m pytest source/scripts/rsl_rl/test/test_local_redundancy.py -q
```
