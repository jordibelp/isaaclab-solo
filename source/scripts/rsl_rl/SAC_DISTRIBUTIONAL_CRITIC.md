# SAC with a symexp two-hot cross-entropy critic

## Use

Append this to an existing **SAC** training command:

```text
agent.distributional_critic_ce=True
```

Default is `False`: the original scalar heads, MSE, initialization, state-dict
keys, and optimizer behavior are retained. PPO is unchanged. Start a **fresh
experiment** when switching modes; scalar and categorical critic checkpoints
have incompatible head shapes. Resume CE checkpoints with CE enabled and the
same number of bins; the support is stored in the checkpoint. Actor-only
inference/export is unchanged.

Optional settings:

```text
agent.critic.distributional_num_bins=255 agent.critic.distributional_symlog_limit=8.0
```

The number of bins must be odd and at least 3. The symmetric support includes
zero and is `symexp(linspace(-8, 8, 255))`, approximately **[-2979.96, 2979.96]**
in reward units. Neither rewards nor environment penalties are modified.

## What problem does this address?

The two-feet analysis found large negative reward events, large critic weights,
deep swish saturation, and poor adaptation to shifted inputs. Large TD errors
are a plausible contributor, **not an experimentally isolated cause**.

For one sample, scalar MSE has output gradient `2 * (Q - y)`. Categorical
cross-entropy has logit gradient `p - t`, where `p` is the predicted probability
vector and `t` the normalized two-hot target. Each component is in [-1, 1], and
its L1 norm is at most 2, independently of target magnitude. Batch averaging and
the existing twin-loss factor scale this down further.

This removes the explicit large-TD-error multiplier. It does **not** bound the
network Jacobian, hidden-layer gradients, optimizer history, or actor gradients;
nor does it make every reward scaling produce identical updates. The CE loss
itself is also unbounded as target-bin probabilities approach zero.

MSE does not require Gaussian returns to estimate their mean. The issue is
optimization under large/noisy errors, not that a non-Gaussian return makes MSE
statistically invalid.

## Implemented equations

Let `B_i = sign(z_i) * (exp(abs(z_i)) - 1)` for evenly spaced `z_i`. Each of the
two critics predicts logits `l_j(s,a)` and exposes the scalar value

```text
p_j = softmax(l_j)
Q_j = sum_i p_ji * B_i
```

The **existing SAC scalar target** is unchanged. For one step:

```text
y = r + gamma * bootstrap_mask * (min(Q1_target(s',a'), Q2_target(s',a')) - alpha * log_pi(a'|s'))
```

The current replay implementation's accumulated n-step reward, effective n-step
discount, and terminal/timeout masks are retained. This change does not alter
its n-step-return convention.

Clip `y` to the support, find `B_k <= y <= B_(k+1)`, and set:

```text
t_k     = (B_(k+1) - y) / (B_(k+1) - B_k)
t_(k+1) = (y - B_k)     / (B_(k+1) - B_k)
L_j     = -sum_i t_i * log_softmax(l_j)_i
L       = (L_1 + L_2) / 2
```

Interpolation is in **raw reward units**, not in symlog space. Thus `sum(t * B)`
equals the clipped target. At an ideal conditional CE optimum, the decoded mean
is the conditional mean of the scalar targets. Rare penalties retain their
contribution: 99% targets of +1 and 1% of -100 have mean -0.01. This property
would not hold if we decoded `symexp(E[symlog(y)])` instead.

The actor still minimizes `mean(alpha * log_pi - min(Q1,Q2))`, differentiating
through decoded raw-unit Q. Automatic temperature tuning and Polyak target
updates are unchanged. We take the minimum of the two **means**, not a
componentwise minimum of categorical probabilities.

This is a categorical representation trained on scalar Bellman targets. It is
**not C51's full distributional Bellman projection**, so its spread should not
be interpreted as a calibrated full-return uncertainty estimate. See
[C51](https://proceedings.mlr.press/v70/bellemare17a.html) for that distinction.

## Why the default support is narrower than Dreamer's

[Dreamer, Nature 2025](https://arxiv.org/html/2301.04104v2), equations 10–11,
uses a weighted raw-unit mean over exponentially spaced atoms and two-hot CE.
It also zero-initializes output weights. Its actor uses normalized returns and
a REINFORCE estimator, unlike SAC's pathwise derivative through Q.

Copying its log-space endpoints +/-20 creates raw endpoints near +/-4.85e8.
An explicit float32-vs-float64 check on a small SAC head after five updates
found cancellation errors in the input/action derivative: absolute error
about 0.00256 against a reference gradient maximum about 0.000232. Pairing
positive/negative probabilities fixes the initial *forward* mean but does not
eliminate cancellation in the final linear layer's backward reduction.

Consequently, this SAC implementation defaults to +/-8 in log-space. The
default action/input gradients pass a float64-reference regression check on
CPU and CUDA. The endpoints cover the discussed -2 contact and -10 collision
events with substantial room for accumulation, but they are **not a bound on
every possible force penalty, soft return, or extrapolation error**.

The output layers start at zero, giving an exactly symmetric initial
distribution and Q=0. Opposite atoms are paired when computing the mean.

Monitor `Loss/critic_target_clipped_fraction` (0–1). Any nonzero value means
some targets were outside the chosen support and their expectation is no
longer preserved. Widen the support deliberately if justified by the target
distribution; arbitrarily huge supports reintroduce action-gradient numerical
problems. CE is magnitude-robust **within a finite representation**, not a
support-free solution.

`Loss/critic1` and `Loss/critic2` now report CE in nats, so do not compare their
numerical magnitudes to MSE losses from baseline runs. Existing hidden-feature
plasticity metrics still operate on the hidden layers; local-redundancy probes
decode scalar Q, rather than silently measuring 255 logits. Even so, head
parameterization changes those probes, so compare behavioral adaptation too.

## Has this been explored before?

- **Direct SAC + categorical critics in locomotion:**
  [FastSAC / Learning Sim-to-Real Humanoid Locomotion in 15 Minutes](https://younggyo.me/fastsac-humanoid/static/paper/fast_sac_paper.pdf)
  uses C51 distributional critics. This is a close application precedent, but
  its full distributional target and other algorithm choices differ from this
  deliberately small SAC modification.
- **Further actor-critic + classification precedent:**
  [Coachable agents for interactive gameplay](https://arxiv.org/html/2607.00642v1)
  uses Cat-RAC with HL-Gauss CE on TD(0) targets for Horizon Forbidden West,
  with 255 bins and range [-600, 1000]. Its separate humanoid experiment uses
  SAC; these should not be conflated as evidence of SAC + HL-Gauss.
- **Classification rather than regression:**
  [Farebrother et al., Stop Regressing (2024)](https://arxiv.org/html/2403.03950v1)
  studies categorical value losses. HL-Gauss, which smooths labels over nearby
  bins, outperforms simple two-hot in their evaluated settings. It is a good
  next loss variant, not proof of superiority on Solo12.
- **Related continuous-control method:**
  [TD-MPC2](https://arxiv.org/html/2310.16828v2) uses soft CE reward/value
  prediction in log-transformed space and separately balances the policy's
  Q-versus-entropy terms. It is model-based, not a drop-in SAC equivalence.
- **Distributional SAC is not automatically bounded-gradient SAC:**
  [DSAC-T](https://arxiv.org/html/2310.05858v4) uses Gaussian value distributions
  with variance-related refinements. Its paper explicitly discusses reward
  scaling sensitivity and instability in earlier DSAC. This differs from a
  categorical logit-gradient bound.

## Alternatives and recommended comparison

| Intervention | Benefit | Important qualification |
|---|---|---|
| Critic LayerNorm | Targets hidden activation/feature conditioning and Q extrapolation | Complementary to CE; does not guarantee no dormant units |
| Huber TD loss | Minimal scalar-head change; caps output-error influence | Generally changes the conditional mean estimator under asymmetric tails |
| Scalar symlog MSE | Compresses large targets | Inverse-transformed mean in log-space is not the raw expected return |
| Fixed positive reward scaling | Cheap magnitude reduction | Leaves relative tail severity unchanged; preserving SAC's objective requires corresponding entropy-temperature scaling |
| PopArt | Adaptive target normalization while preserving unnormalized outputs | More moving parts; does not by itself fix rare-event sampling or policy-gradient scale |
| HL-Gauss | Smooth categorical targets; promising empirical results | Adds a smoothing bandwidth/support choice; not implemented here |

Relevant primary references: [RLPD / Ball et al.](https://proceedings.mlr.press/v202/ball23a.html)
for critic LayerNorm and [PopArt / van Hasselt et al.](https://arxiv.org/abs/1602.07714)
for output-preserving target normalization. The tradeoffs above are mathematical
reasoning and recommendations for this codebase, not measured Solo12 results.

Start with a **2 x 2 comparison: MSE vs CE, each with/without critic LayerNorm**.
The latter is already available as `agent.critic.layer_norm=True`. Keep rewards,
network widths, optimizer settings, env count, seeds, and training duration
matched; leave actor LayerNorm unchanged. Use multiple paired seeds and run
long enough to reach the previously observed late-training failure regime.
Measure return, collisions/forbidden contacts, dormancy, feature rank, weight
norm, shifted-input refitting, and target clipping. CE alone is the clean test
of the large-error-loss hypothesis; CE plus LayerNorm is a useful combined arm.

## Verification (2026-09-16)

- 65 tests passed across the SAC and plasticity suites, including CPU/CUDA
  projection boundaries, mean preservation, bounded logit gradients, action
  gradient precision, default MSE output/gradient equality, scalar probe shape,
  terminal/timeout/n-step bootstrapping, Polyak updates, construction, and
  model/optimizer checkpoint round trips.
- Real `solo12-two-feet` headless smoke: 64 envs, five rollout iterations,
  two minibatches of 128 per training iteration; TensorBoard and both
  plasticity diagnostics enabled. Final CE losses approximately 5.5227/5.5225;
  target-clipping fraction zero throughout training.
- Independent comparison against the pre-change Git version: three complete
  default-MSE SAC updates produced bit-identical losses and actor/critic state
  dictionaries on both CPU and CUDA.
- These checks establish implementation/numerical behavior, **not a demonstrated
  cure for long-run plasticity loss or an improvement in task performance**.

Run the numerical suite from the repository root:

```bash
PYTHONPATH=source/rsl_rl_sac_vendor:source/scripts/rsl_rl /home/jordibelp/miniconda3/envs/env_isaaclab/bin/python -m pytest -q source/rsl_rl_sac_vendor/test source/scripts/rsl_rl/test/test_local_redundancy.py source/scripts/rsl_rl/test/test_plasticity_metrics.py
```
