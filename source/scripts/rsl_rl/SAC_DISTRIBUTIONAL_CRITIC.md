# SAC with a categorical cross-entropy critic on a symexp support

## Use

One flag on an existing **SAC** training command picks the critic loss:

```text
agent.critic.distributional_loss=mse       # default: the original scalar heads
agent.critic.distributional_loss=two_hot   # two-hot labels, Dreamer style
agent.critic.distributional_loss=hl_gauss  # Gaussian labels, Farebrother et al. 2024
agent.critic.distributional_loss=mse_target_norm_popart  # scalar MSE on PopArt-normalized targets
```

`mse` keeps the original scalar heads, MSE, initialization, state-dict keys, and
optimizer behavior. PPO is unchanged in all four modes. The PopArt mode is not
categorical; see [its own section](#popart-mse-on-normalized-targets).

Start a **fresh experiment** when you move between `mse` and either categorical
mode, because the head shapes differ. `two_hot` and `hl_gauss` share the same
head and the same state dict, so you can resume one from the other's checkpoint
and change only the labels. Keep the number of bins the same when you resume;
the support is stored in the checkpoint. Actor-only inference and export are
unchanged.

The old flag `agent.distributional_critic_ce=True` still works and now selects
`two_hot`. It prints a deprecation line. Setting both flags at once is an error.

Optional settings:

```text
agent.critic.distributional_num_bins=255 agent.critic.distributional_symlog_limit=8.0
agent.critic.hl_gauss_sigma_ratio=0.75
agent.critic.popart_beta=3e-4
```

The number of bins must be odd and at least 3. The symmetric support includes
zero and is `symexp(linspace(-8, 8, 255))`, approximately **[-2979.96, 2979.96]**
in reward units. `hl_gauss_sigma_ratio` only applies to `hl_gauss`, and
`popart_beta` only to `mse_target_norm_popart`. The bin settings are ignored by
the two scalar modes. Neither rewards nor environment penalties are modified.

To average the two critics instead of taking their minimum, add
`agent.algorithm.q_reduction_method=mean` or `=mean_pi_q_none`. See
[Twin-critic reduction](#twin-critic-reduction-min-or-mean).

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

`min` is the default. See [Twin-critic reduction](#twin-critic-reduction-min-or-mean)
for the `mean` option.

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
updates are unchanged. We take the minimum of the two **means** (or their
average, with `mean`), not a componentwise minimum of categorical probabilities.

This is a categorical representation trained on scalar Bellman targets. It is
**not C51's full distributional Bellman projection**, so its spread should not
be interpreted as a calibrated full-return uncertainty estimate. See
[C51](https://proceedings.mlr.press/v70/bellemare17a.html) for that distinction.

## HL-Gauss labels

`hl_gauss` changes the label `t` and nothing else. The support, the scalar
target `y`, the decode `Q = sum(p * B)`, the actor, and the temperature are all
exactly as above.

Two-hot puts every bit of mass on the two atoms next to the target. HL-Gauss
spreads it with a Gaussian instead:

```text
sigma = hl_gauss_sigma_ratio * atom_spacing
t_i   = Phi((e_(i+1) - z) / sigma) - Phi((e_i - z) / sigma),  rescaled to sum to 1
```

Here `e_i` are the bin edges and `z` is the target. The atoms are treated as bin
centers, so the outer edges sit half a spacing past the limit. At the default
ratio 0.75 the mass lands on about three atoms. That is the point: the label
says "the return is somewhere near here" instead of "the return is between these
two exact atoms".

[Farebrother et al. (2024)](https://arxiv.org/abs/2403.03950) report that this
beats two-hot across their domains. They also find that the best ratio does not
depend on the number of bins, which is why one default is reasonable. They use
0.75, so we use 0.75.

### Why the Gaussian lives in symlog space

The paper's support is evenly spaced in reward units. Ours is not. Ours is
evenly spaced in **symlog** units, so the gap between neighbouring atoms grows
from about 0.04 near zero to about 5.8 near the edge at `limit=5`. A single
`sigma` in reward units would therefore cover hundreds of atoms near zero and
less than one atom near the edge.

So we put the Gaussian in symlog space, where the atoms really are evenly
spaced. `hl_gauss_sigma_ratio` then keeps the meaning it has in the paper: sigma
as a fraction of the bin width. The label covers about three atoms everywhere on
the support.

We kept the symexp support instead of switching to an evenly spaced one for two
reasons. It resolves small Q values far more finely, which matters when rewards
are order 1 and returns are order 10. And it keeps `two_hot` and `hl_gauss`
differing **only** in the label, which is the comparison the paper actually
makes.

### The skew correction

A Gaussian that is symmetric in symlog units is **not** symmetric after
`symexp`. Its upper tail stretches more than its lower tail. So the decoded mean
comes out too large, by a factor of about `exp(sigma^2 / 2)`.

Per decode that is tiny. But the critic decodes once per Bellman backup, so it
compounds. A constant relative bias `b` moves the fixed point by roughly
`(1 - gamma) / (1 - gamma * (1 + b))`. At `gamma = 0.97`, with no correction:

| bins | limit | bias per decode | resulting Q inflation |
|---|---|---|---|
| 255 | 5.0 | 4.4e-4 | 1.4% |
| 255 | 8.0 | 1.1e-3 | 3.7% |
| 101 | 8.0 | 7.2e-3 | **31%** |
| 51 | 5.0 | 1.1e-2 | **58%** |

The paper's own bin count is 101, so a perfectly reasonable-looking setting
lands in the bad rows. We therefore center the Gaussian at
`z - sign(z) * sigma^2 / 2` rather than at `z`. That cancels the skew to first
order. The shift is less than a twentieth of one bin width, so the smoothing
behavior is unchanged.

What survives is a much smaller outward skew of about `0.1 * sigma^2`: roughly
9e-5 at `limit=5` and 2e-4 at `limit=8`, both with 255 bins. Those cost about
0.3% and 0.65% of Q. This part is still systematic, so it still compounds. It is
simply five times smaller than what the correction removed.

`SACCriticModel.label_decode_bias()` measures the error, and every categorical
run prints it at startup:

```text
SAC critic: hl_gauss labels on 255 symexp atoms (symlog limit 5.0); worst decoded-mean bias 3.33e-04
```

That printed number is a **worst-case envelope** over the whole support, not the
systematic part on its own. It is relative for targets past 1.0 and absolute
below, and near zero it is dominated by a fixed offset of the order of
`sigma^2 / 2` rather than by anything that compounds. Two-hot prints about 1e-7,
because its label preserves the mean exactly by construction. If you sweep the
bin count, watch this line.

## PopArt: MSE on normalized targets

`mse_target_norm_popart` keeps the scalar MSE critic and adds PopArt
([van Hasselt et al. 2016](https://arxiv.org/abs/1602.07714)). It is meant as an
ablation for the CE result. It fixes the *global* scale and offset of the
regression problem, but it keeps the per-sample weighting of MSE.

### What it does

Each critic head now predicts a normalized value `g`. Everything outside the
critic loss sees the raw value:

```text
Q = sigma * g + mu
```

`mu` and `sigma` are running estimates of the mean and standard deviation of
the Bellman targets `y`. They describe the targets, not the rewards. The target
`y` is computed exactly as before, in raw reward units. Each minibatch update
then does three things, in the order of the paper's Algorithm 1:

1. **ART: update the statistics** with the moments of this minibatch's `y`:

   ```text
   mu'    = (1 - beta) * mu + beta * mean(y)
   nu'    = (1 - beta) * nu + beta * mean(y^2)        where nu = sigma^2 + mu^2
   sigma' = max(sqrt(nu' - mu'^2), 1e-4)
   ```

2. **POP: rescale the last linear layer** so that no Q value changes:

   ```text
   w' = w * sigma / sigma'
   b' = (sigma * b + mu - mu') / sigma'
   ```

3. **Fit** with plain MSE on the normalized target:

   ```text
   L_j = mean((g_j - (y - mu') / sigma')^2)
   L   = (L_1 + L_2) / 2
   ```

Step 1 comes before step 3. So the update that sees a new target already uses
the new normalization. The paper stresses that this order matters.

### Design choices

- **The actor, the Bellman target, and the temperature all see raw Q.** SAC's
  objective and the balance between Q and the entropy bonus are unchanged. Only
  the critic's regression problem is rescaled. The multi-task PopArt paper also
  normalizes its policy loss. We do not, so this stays a pure critic-loss ablation.
- **One pair of statistics for both critics and their targets.** Both critics
  regress the same `y`. The target networks share the statistics, so step 2
  rescales their heads too, and the target Q values stay exactly the same.
  Polyak averaging stays exact as well, because averaging commutes with this
  rescaling.
- **It starts at `mu = 0`, `sigma = 1`.** At step 0 the PopArt critic is identical
  to the MSE critic with the same seed. The statistics then move toward the data
  with a time constant of `1 / beta` updates. At `beta = 3e-4` and 400
  minibatches per iteration, that is about 8 iterations.
- **The defaults `beta = 3e-4` and the `1e-4` floor on sigma** come from the
  multi-task PopArt paper ([Hessel et al. 2018](https://arxiv.org/abs/1809.04474),
  Table 3). There, beta "didn't require any tuning". Change it with
  `agent.critic.popart_beta`.
- **Adam's state is not rescaled** when step 2 changes the head. Each update
  changes the head by a factor of about `1 ± beta`, and Adam's moving averages
  absorb that.
- The code updates the variance in a rearranged form. It is equal to step 1, but
  float32 never has to subtract two large squares. A test checks it against the
  formula above.

### What to expect with Adam

The paper's main argument is about plain SGD. There, doubling the target scale
makes the hidden-layer updates four times larger, and PopArt removes this. Our
critic uses Adam, and Adam divides each gradient by its own running size. So a
fixed rescaling of all targets already cancels out.

With constant statistics, PopArt + Adam behaves almost like plain MSE + Adam
with the head's step size multiplied by sigma. What still differs:

- how the critic reacts when the target scale *changes* during training, for
  example at curriculum phases or as the policy improves;
- the head's effective step size, which is multiplied by sigma;
- the norm-1 gradient clip. It acts on the whole critic gradient, and PopArt
  divides the head gradient by sigma and the hidden-layer gradient by sigma².
  So PopArt changes how often the clip triggers.

If `PopArt/std` stays near 1, expect this arm to look almost like plain MSE.
That result would still be useful: it would say the CE gain does not come from
the global scale of the targets.

### What PopArt does not change

- **Per-sample influence.** Every residual is divided by the same sigma. A
  transition with a 10× larger TD error still gets a 10× larger output gradient.
  CE's `p - t` has no such multiplier. This is the main contrast with the CE arms.
- **Rare large targets.** The paper bounds each normalized target by
  `sqrt((1 - beta) / beta)`. That bound holds for one-sample updates. Here the
  statistics move once per minibatch of about 16k targets. One rare -10
  collision target barely moves them, so the bound does not protect against it.
- **Rewards and penalties.** Nothing in the environment changes.

### Logs, checkpoints, and other tools

- `PopArt/mean` and `PopArt/std` are logged every iteration.
- `Loss/critic1` and `Loss/critic2` are in *normalized* units. Multiply them by
  `PopArt/std²` to compare with a plain MSE run.
- The checkpoint stores the statistics as `popart_mean` and `popart_std`, so a
  resume restores them.
- Start a fresh experiment when you switch between PopArt and any other mode.
  The two extra buffers make strict checkpoint loading fail loudly, on purpose.
- `play_direct_0325.py --q_value_log` needs the same
  `agent.critic.distributional_loss=mse_target_norm_popart` override. Actor-only
  playback and export are unchanged.
- Local-redundancy probes decode raw Q in every mode, so PopArt and MSE runs are
  measured on the same quantity.
- `mujoco/train_sac.py` does not support PopArt checkpoints yet. Loading one
  fails loudly on the two extra buffers.

## Twin-critic reduction: `min` or `mean`

SAC has two critics. One flag picks how their two values are combined:

```text
agent.algorithm.q_reduction_method=min             # default: clipped double Q-learning
agent.algorithm.q_reduction_method=mean            # the average, in the target and the actor loss
agent.algorithm.q_reduction_method=mean_pi_q_none  # FastSAC reference code: average in the actor loss only
```

The flag works with every critic loss above. `min` and `mean` apply the same
reduction in two places:

```text
y          = r + gamma * bootstrap_mask * (reduce(Q1_target(s',a'), Q2_target(s',a')) - alpha * log_pi(a'|s'))
actor loss = mean(alpha * log_pi(a|s) - reduce(Q1(s,a), Q2(s,a)))

min:   reduce(Q1, Q2) = min(Q1, Q2)
mean:  reduce(Q1, Q2) = (Q1 + Q2) / 2
```

With these two, both critics regress onto the same target `y`.

`mean_pi_q_none` does not reduce the target at all. Each critic gets its own
target, from its own target network. The actor loss uses the average:

```text
y_1        = r + gamma * bootstrap_mask * (Q1_target(s',a') - alpha * log_pi(a'|s'))    # critic 1 learns y_1
y_2        = r + gamma * bootstrap_mask * (Q2_target(s',a') - alpha * log_pi(a'|s'))    # critic 2 learns y_2
actor loss = mean(alpha * log_pi(a|s) - (Q1(s,a) + Q2(s,a)) / 2)
```

Both critics use the same sampled next action `a'`. Note that `(y_1 + y_2) / 2`
is exactly the `mean` target. The two modes differ only in what each critic
learns, not in the average target.

With a categorical critic, `Q1` and `Q2` are the decoded means. So `mean`
averages two means. It does not mix the two probability vectors. Under
`mean_pi_q_none`, each critic gets two-hot (or HL-Gauss) labels of its own `y_i`.

### Why try `mean`

- Clipped double Q-learning (CDQ, [Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477))
  takes the minimum to fight overestimation. The price is a low bias.
- For two critics, `min(Q1, Q2) = (Q1 + Q2) / 2 - |Q1 - Q2| / 2`. So `min` is
  `mean` minus half of the critic disagreement, at every backup.
- FastSAC ([Seo et al. 2025](https://arxiv.org/abs/2512.01996), Figure 2a)
  reports that the average beats the minimum for humanoid locomotion. The paper
  links this to BRO ([Nauman et al. 2024](https://arxiv.org/abs/2405.16158)),
  which found CDQ harmful together with layer normalization. Our Solo12 critic
  runs with `agent.critic.layer_norm=True`, so this ablation is relevant for us.

Expect higher Q values with both `mean` modes. Most of that is the pessimism
that `min` added. Some of it can be real overestimation, which is exactly what
CDQ was designed to prevent. The `Q - G` error of `q_spread_plots.py` shows
which one you get.

### `mean` or `mean_pi_q_none`?

The paper only says "the average of Q-values". Its reference code (Holosoma
FastSAC, and FastTD3 with `use_cdq=False`) does this:

- **Actor:** the average of the critics.
- **Critic target:** each critic bootstraps from **its own** target network.
  There is no minimum and no average in the target.

`mean_pi_q_none` copies exactly this. `mean` also averages in the target. Both
remove the CDQ pessimism. The difference is how independent the two critics
stay:

- With `mean`, both critics learn the same target. So they are pulled toward
  each other, and only their initialization and their own fit errors keep them
  apart. Averaging also reduces the target noise, unless both critics make
  exactly the same errors.
- With `mean_pi_q_none`, each critic keeps its own bootstrap chain. An error in
  one critic feeds only its own next targets, so the two stay more different.
  The actor then averages two more independent estimates.

`mean_pi_q_none` copies how the reference code combines the critics. It does
not copy the rest of FastSAC. For example, FastSAC uses a C51 distributional
projection, and our categorical critic regresses scalar targets.

### Logs, checkpoints, and other tools

- The value appears in `params/agent.yaml` and in the W&B config, under
  `train_cfg.algorithm.q_reduction_method`.
- Checkpoints store `q_reduction_method`. Older checkpoints have no entry. They
  were all trained with `min`, and every tool reads them that way.
- Resuming uses the value from the command, like `gamma` or `tau`. Pass the
  same flag again to continue a `mean` or `mean_pi_q_none` run.
- Under `mean_pi_q_none`, `critic_target_clipped_fraction` counts both critics'
  targets. PopArt keeps one pair of statistics, fitted to both critics' targets.
- `play_direct_0325.py --q_value_log` reads the value from the checkpoint. It
  builds the bootstrap value the same way and saves the value in the `.npz`.
  Under `mean_pi_q_none` it uses the average of the two targets, which is the
  target of the averaged Q that the log compares. You do not need the flag at
  play time.
- `q_spread_plots.py` measures the Q that the actor used: `min(Q1,Q2)`, or
  `mean(Q1,Q2)` for both `mean` modes. It prints a note for those. Older logs
  count as `min`.
- `mujoco/train_sac.py` keeps the checkpoint's value when fine-tuning. Change it
  with `--q-reduction-method=min`, `=mean`, or `=mean_pi_q_none`.
- Multi-GPU needs nothing new. The reduction works per sample, before the
  gradient all-reduce.

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

## How much of the support is actually used?

With CE enabled, every SAC update also logs `CriticDist/*` from the mini-batch it
just trained on. The twin critics are mixed as `(p1 + p2) / 2` first, so critic
disagreement widens the reported spread exactly as it widens the range of values
the pair can encode.

Everything is reported in **symlog units** — the `x` in `Q = sign(x)(exp|x| - 1)` —
so the numbers are directly comparable to `agent.critic.distributional_symlog_limit`.

| Key | Meaning |
| --- | --- |
| `symlog_mean` | Batch-mean distribution center. |
| `symlog_std_within_state` | Typical width of *one* state-action's distribution. |
| `symlog_std_across_states` | How much the center moves *between* states in the batch. |
| `symlog_q05` / `q50` / `q95` | Atom-resolution quantiles; robust when the mass is near two-hot. |
| `symlog_q05_q95_width` | `q95 - q05`, the occupied range. |
| `active_atoms_p10` | Atoms above probability 0.1 — "how many categories are used". |
| `effective_atoms` | `exp(entropy)`, the same question without a threshold. |
| `edge_mass` | Mass on the two outermost atoms. |

The two std keys answer different questions: plotting `symlog_mean` shaded with
`symlog_std_across_states` shows the range of Q across the state distribution,
while `symlog_std_within_state` shows how sharp each individual prediction is. A
well-fit critic on a deterministic-return task should drive the second toward the
atom spacing while the first stays wide.

Read `edge_mass` together with `critic_target_clipped_fraction`: clipping says
targets fell outside the support, `edge_mass` says the *prediction* is piling up
against the boundary. Either one means the limit is too small.

These describe the categorical **representation**. The critics are trained on
scalar Bellman targets, so a wide distribution is fit error plus target spread —
not a calibrated estimate of return uncertainty.

### Same statistics at inference

`play_direct_0325.py --q_value_log <file>.npz` records per-step twin-critic Q,
the categorical probabilities, and the support. Two optional flags bound the
collection:

```text
--q_value_log_episodes 3     # stop after 3 episode endings instead of the full --duration_s
--q_value_log_stochastic     # act with the learned std instead of the deterministic mean
```

`q_spread_plots.py` turns one or more of those files into the support-occupancy
histogram, a per-step center with std and q05–q95 bands, and a histogram of
per-state widths. It reuses `symlog_distribution_stats`, the same function behind
the `CriticDist/*` scalars, so training curves and eval plots are comparable.
Runs with different limits can be overlaid because the axis is symlog:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/q_spread_plots.py a.npz b.npz --labels "limit 5.0" "limit 4.0" --out spread.png --no-show
```

### Estimation error `Q - G`

The same figure ends with `min(Q1, Q2)` against the return it predicts, and the
error between them, in raw reward units. `gamma` and `alpha` come from the log,
so the comparison uses training values rather than assumed ones.

Three corrections matter, and skipping any of them produces a plausible-looking
but wrong error:

- **Entropy.** Q predicts the *soft* return, which adds `alpha * -log_pi` for
  every step *after* the evaluated action — never for the action itself, which Q
  already conditions on. The summary prints the error with and without this term.
- **Truncation.** A timed-out episode has not really ended, so closing it at
  `V = 0` would drag `G` down toward every timeout. It is instead bootstrapped
  with `V(s')` exactly as training does: the pre-reset `time_outs_obs`, a sampled
  next action, and the frozen target critics. A *terminated* episode is worth
  zero afterwards and ignores the bootstrap. `--no-bootstrap` restores the
  zero-tail behavior if you want to see that bias.

  Bootstrapping makes `G` complete everywhere, but the steps nearest a timeout
  draw most of their value from the critic itself, so their "error" approaches a
  one-step TD residual rather than a comparison against real data. Each step
  therefore reports the fraction of `G` that came from actual rewards;
  `--min-observed` (default 0.9) restricts the statistics to steps that are
  mostly real, and the rest are drawn faintly rather than hidden.
- **Determinism.** Q is defined under the stochastic policy. A deterministic
  rollout is a different trajectory distribution; `--q_value_log_stochastic`
  removes that mismatch.

With SAC's `gamma=0.97` at 50 Hz control, the effective horizon is about 33
steps, so a 5 s episode leaves roughly the last 1.5 s below the default
threshold and keeps about 70–80% of steps. Longer `--episode_length_s` keeps
proportionally more. A run where nothing passes the threshold means the episodes
are short relative to `1/(1-gamma)`, not that the critic is bad.

Sign convention: positive mean error is the critic **overestimating** the return
it will actually collect.

A useful read of the `G` curve: with bootstrapping on, a flat `G` across a
timeout means `V(s')` agrees with the return that was actually being collected.
A visible step down at every timeout means it does not, and the size of that
step is itself the diagnostic.

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
  bins, outperforms simple two-hot in their evaluated settings. It is now
  implemented here as `distributional_loss=hl_gauss`, but their result is not
  proof of superiority on Solo12; that still has to be measured.
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
| PopArt | Adaptive target normalization while preserving unnormalized outputs | Now `distributional_loss=mse_target_norm_popart`; keeps MSE's per-sample weighting, and under Adam mostly changes the head step size and gradient clipping |
| HL-Gauss | Smooth categorical targets; promising empirical results | Adds a smoothing bandwidth; on this log support it needs the skew correction above |

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

Treat `two_hot` versus `hl_gauss` as a **separate, later** question. They differ
only in the label, so comparing them answers "does label smoothing help here?",
which is a much narrower question than "does classification beat regression
here?". Settle the MSE-versus-CE arm first, then run the label comparison from
the winning CE configuration. Both share a state dict, so the second comparison
can branch off a common checkpoint if you want to control for early training.

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

## Verification of HL-Gauss (2026-09-18)

- 123 tests pass with the command below, up from 98. The new ones cover the
  HL-Gauss label and the flag rename.
- The label is checked to be non-negative, to sum to one, to carry no gradient,
  and to clamp out-of-support targets onto the edge exactly like two-hot.
- Mass spread is measured, not assumed: at ratio 0.75 the label occupies about
  3.3 effective atoms against two-hot's at-most-2, and ratios 0.375 / 0.75 / 1.5
  give about 2.0 / 3.3 / 6.3 atoms.
- The decoded mean is checked against the target it encodes. The systematic part
  is measured **on one side of zero only**, because a probe symmetric about zero
  cancels an outward skew and makes any correction look perfect.
- The skew correction is checked to do real work: on a deliberately coarse
  51-bin support, the uncorrected kernel biases the decode by more than 1e-2
  relative, and the correction cuts that by more than five times. Near zero the
  leftover is verified to be a bounded offset rather than an error that grows
  with the target.
- Logit gradients are confirmed to stay `p - t` with L1 norm at most 2, so
  HL-Gauss keeps the bounded-gradient property that motivated CE in the first
  place.
- `float32` and `float64` agree on the decode bias to three digits, so the
  reported number is real discretization, not float32 noise.
- Checkpoints are confirmed interchangeable: the bin edges are a non-persistent
  buffer, so a `two_hot` state dict loads into an `hl_gauss` critic with
  `strict=True`.
- The local-redundancy probe is re-checked against both categorical modes. That
  test had silently stopped constructing a categorical critic during the rename,
  and it now asserts the head really is wider than one output before probing.
- Real `solo12-two-feet` headless smoke with `distributional_loss=hl_gauss`:
  64 envs, five iterations, two minibatches of 128, `symlog_limit=5.0`,
  critic LayerNorm on. Startup reported a worst decoded-mean bias of 3.33e-04
  and target clipping stayed at zero for every iteration.
- **Not verified:** no long run and no converged policy. Whether HL-Gauss helps
  Solo12 is unmeasured. The `CriticDist/symlog_std_within_state` value near 2.89
  in the smoke log is just the zero-initialized head being uniform over
  `[-5, 5]`, not a trained prediction.

## Verification of the spread diagnostics (2026-09-18)

- 98 tests pass across the SAC, local-redundancy, plasticity, and spread suites,
  using the command below.
- `CriticDist/*` is checked against closed-form values on CPU and CUDA: a
  one-hot distribution reports zero width and one active atom; a uniform
  distribution reports `effective_atoms == num_bins` and `edge_mass == 2/bins`;
  and a case where the twin critics peak on different atoms confirms that
  within-state and across-state spread are separated correctly.
- The keys are confirmed to reach `alg.update()`'s output with CE on, to be
  absent with CE off, to carry no gradient, and to bypass the `Loss/` prefix in
  the logger.
- `q_spread_plots.py` was run end-to-end on `.npz` files written through the real
  critic and the real save call, for two different `distributional_symlog_limit`
  values overlaid in one figure.
- The discounted-return reconstruction is checked against hand-computed values:
  terminated versus timed-out episodes, a log that ends mid-episode, no leakage
  across an episode boundary, per-env independence, the geometric observed
  fraction, and the entropy bonus starting one step *after* the evaluated action.
- Bootstrapping is checked to close timeouts on `V(s')`, to be ignored at true
  terminations, to apply the soft value only to the soft return, and — on a
  stationary problem where the critic is exactly right — to hold `G` flat
  through a timeout where the zero-tail version visibly sags.
- **Not covered by an automated test:** `_bootstrap_value` in the play script
  needs a running env, so it is verified only by mirroring training's
  `process_env_step` masking of `time_outs_obs` line for line.
- `executed_action_logp` is checked to reproduce `sample_action_logp`'s value for
  the same action to 2e-4, and to score the distribution mode above a sample.
- **Not yet verified:** no long training run or Isaac Sim rollout has been logged
  with these metrics, so their behavior on a converged policy is unmeasured. The
  `alpha` used for the entropy correction is the checkpoint's final value, which
  is only the value in force during training if `auto_alpha` had settled.

## Verification of PopArt (2026-09-24)

- 182 tests pass with the command below, and the 126 MuJoCo SAC tests in
  `mujoco/test_train_sac.py` still pass.
- At step 0, a PopArt critic built with the same seed gives bit-identical Q to
  the MSE critic. Its state dict is the MSE one plus `popart_mean` and
  `popart_std`.
- Proposition 1 is checked in float64 over large jumps of the statistics, up to
  a mean of about 190 and a std of about 420. The online and target Q values stay
  equal to 1e-9. The statistics match the paper's formula to 1e-9 relative, and
  the 1e-4 floor holds for targets with no spread.
- The loss is checked to be plain MSE on `(y - mu) / sigma`. Each normalized
  residual is checked to be the raw residual divided by sigma.
- A complete SAC update with PopArt, including terminal, timeout, and mixed
  n-step samples, builds the same Bellman target as a hand computation. The
  loss already uses the statistics from that same batch. `PopArt/mean` and
  `PopArt/std` reach the loss dict, and a checkpoint round trip restores Q
  exactly.
- The config flag builds a scalar head with the requested `popart_beta`. Values
  of beta outside (0, 1] are rejected.
- The local-redundancy probe is checked to measure raw Q: setting `sigma = 3`
  multiplies its score by 9, and the mean has no effect.
- The old modes are unchanged. Six complete SAC updates (swish, LayerNorm
  critic, mixed terminals and timeouts) give bit-identical losses, actor and
  critic state dicts, and `log_alpha` against git HEAD for `mse`, `two_hot`,
  and `hl_gauss`, on CPU and CUDA.
- Real `solo12-two-feet` headless smoke: 64 envs, five iterations, two
  minibatches of 128, 1024-512-256 critic with LayerNorm, symmetry augmentation,
  and both plasticity diagnostics every iteration. It exited cleanly.
  `PopArt/std` went 1.0125 → 1.0528 and `PopArt/mean` -0.0035 → -0.0148 over
  eight updates, and `Loss/critic1` fell from 32.9 to 11.9. The checkpoint holds
  the same two values.
- From how fast the statistics moved, the first targets had a mean near -6 and a
  std near 3. This is a rough estimate from eight small batches. It suggests
  sigma will leave 1 quickly in a real run.
- **Not verified:** no long run, and no learning result yet. The multi-GPU
  all-reduce of the target moments is untested, because no multi-GPU machine
  was used.

## Verification of the twin-critic reduction (2026-09-24)

- The default is unchanged. We ran one complete SAC update of six minibatches
  (swish, LayerNorm critic, mixed terminals, timeouts, and n-step lengths) in 16
  settings: `mse`, `two_hot`, `hl_gauss`, and PopArt; actor update every 1 or 2
  steps; CPU and CUDA. Losses, actor and critic state dicts, `log_alpha`, and
  the critic Adam state are bit-identical to the code before this flag, both
  without the flag and with `min`. After `mean_pi_q_none` was added, `mean` is
  still bit-identical to its first version (commit `4704da4`) in all 16. Each
  `mean` mode differs from `min` in all 16, and from each other, so every value
  has an effect.
- `test_q_reduction.py` (48 cases) rebuilds the Bellman target by hand for all
  three modes and all four critic losses, and the actor loss for `mse` and
  `two_hot`, on CPU and CUDA. It checks that under `mean_pi_q_none` each critic
  is trained on its own target column, and that the two targets average to the
  `mean` target. It also checks the config path and the checkpoint entry.
- We broke the code on purpose to check that these tests notice. Each of these
  small breaks fails at least 4 tests: forcing the target or the actor loss
  to one fixed reduction, reducing the `mean_pi_q_none` target, swapping its two
  target columns (in the target or in the loss), giving both critics the first
  column, and giving the `mean_pi_q_none` actor the minimum.
- 234 tests pass with the command below. `mujoco/test_train_sac.py` passes all
  134 tests, including 8 new ones, and the other 68 MuJoCo tests still pass.
- Real Isaac Sim smoke, once for `mean` and once for `mean_pi_q_none`: the full
  `solo12-two-feet` SAC command with a 199-bin categorical critic and symmetry
  augmentation, scaled down to 64 envs, five iterations, and 20 minibatches of
  1024. Both exited cleanly with finite losses. The value reached
  `params/agent.yaml`, the checkpoints, and the W&B config.
- `play_direct_0325.py --q_value_log` on each checkpoint, without any reduction
  flag, read the mode from the checkpoint and stored it in the `.npz`.
  `q_spread_plots.py` then measured `mean(Q1,Q2)` and printed its note.
- `mujoco/train_sac.py` fine-tuned each checkpoint for 2000 interactions
  without the flag. It kept the checkpoint's mode and recorded it in
  `run_config.json` and in its own checkpoints.
- **Not verified:** no long run, and no learning comparison yet. Multi-GPU is
  untested, but the change adds no communication.

Run the numerical suite from the repository root:

```bash
PYTHONPATH=source/rsl_rl_sac_vendor:source/scripts/rsl_rl /home/jordibelp/miniconda3/envs/env_isaaclab/bin/python -m pytest -q source/rsl_rl_sac_vendor/test source/scripts/rsl_rl/test/test_local_redundancy.py source/scripts/rsl_rl/test/test_plasticity_metrics.py source/scripts/rsl_rl/test/test_q_spread_plots.py
```

## The env-side contract SAC depends on

`DirectRLEnv.step` publishes `extras["time_outs_obs"]`, the observation reached
at a timeout before the automatic reset overwrites it. `SAC.process_env_step`
needs it to store the true next state and to mark the transition as
bootstrappable, which is the "timeout-aware critic targets" contribution of
[Bridging the Gap](https://arxiv.org/abs/2605.24975).

**An env that overrides `step` opts out of that block.** Every consumer guards
with `if "time_outs_obs" in extras` and degrades silently, so the only symptom
is a critic that treats every truncation as a terminal state worth zero while
storing the post-reset observation as the transition's next state. `Solo12Env`
carried exactly that defect from 2026-06-24 until 2026-09-18, because its
override predates the SAC integration that patched the base class.

`test_timeout_obs_contract.py` fails on any direct-task env class that overrides
`step` without republishing the key, so the next one is caught at test time
rather than rediscovered from a suspicious value plot. If it fires, copy the
reset block from `DirectRLEnv.step`:

```python
time_outs_obs = self._get_observations()
if self.cfg.observation_noise_model:
    time_outs_obs["policy"] = self._observation_noise_model(time_outs_obs["policy"])
self.extras["time_outs_obs"] = {key: value.detach().clone() for key, value in time_outs_obs.items()}
self._reset_idx(reset_env_ids)
```
