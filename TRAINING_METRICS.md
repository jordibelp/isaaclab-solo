# Solo12 training metrics: a complete reading guide

Written: 2026-10-12. Training source checked at local commit `e7e9893`.

This file describes the quantities logged by our four Solo12 two-feet training paths:

- IsaacLab PPO: `source/scripts/rsl_rl/train.py`, with the PPO agent entry point.
- IsaacLab SAC: the same script, with `rsl_rl_sac_cfg_entry_point`.
- MuJoCo/MJX PPO, with full weights or LoRA: `mujoco/train_lora.py`.
- MuJoCo/MJX SAC, with full weights or LoRA: `mujoco/train_sac.py`.

It covers existing metrics, the new return metrics, optional training diagnostics, older names, and W&B's automatic system metrics. It is not a catalogue of every other IsaacLab task or the separate ROS inference debugger.

We checked the logged names in these two example runs as well as the current source:

- [IsaacLab SAC tacjuisg](https://wandb.ai/jordibelp/borinotIsaacLab/runs/tacjuisg), recorded commit `6eee81f`.
- [MuJoCo SAC 2cijzf7n](https://wandb.ai/jordibelp/solo12-two-feet-lora/runs/2cijzf7n), recorded commit `6b08c30`.

Together their W&B summaries contain 222 distinct fields. These older runs do not contain every quantity added later. Optional metrics also depend on the launch flags.

## Start here: what does `policy_mean_over_theoretical_pos` mean?

The full key is `CriticQ/policy_mean_over_theoretical_pos`.

It means:

```text
Q predicted for newly sampled current-policy actions at replay-buffer states
-------------------------------------------------------------------------
        the current positive discounted-return ceiling
```

We first calculate `CriticQ/policy_mean`. For each replay state, we sample an action from the current policy. We evaluate both critics. We apply the configured min/mean rule to that pair. We average over the batch, then over the actor evaluations in the training iteration.

We divide that result by `Returns/theoretical_return_lim_pos`.

For example, if the mean predicted Q is 1.4 and the ceiling is 2.3333, the ratio is approximately `1.4 / 2.3333 = 0.60`.

**Here, `mean` means an average of Q values. It does not mean that we use the actor's deterministic mean action. The actions are sampled.**

This is a critic prediction, not a measured episode return. The states come from replay, not necessarily the latest rollout. With retained replay, some states come from the source simulator. SAC Q also includes future entropy, while the positive ceiling is reward-only. Therefore a ratio above 1 is not, by itself, proof of critic overestimation.

## Contents

1. [Basic words, units, and averaging](#1-basic-words-units-and-averaging)
2. [Discounted returns and theoretical limits](#2-discounted-returns-and-theoretical-limits)
3. [SAC Q and PPO V predictions](#3-sac-q-and-ppo-v-predictions)
4. [Categorical critic distributions](#4-categorical-critic-distributions)
5. [Every reward term and its logging families](#5-every-reward-term-and-its-logging-families)
6. [Episode totals, lengths, endings, and smoothing](#6-episode-totals-lengths-endings-and-smoothing)
7. [Losses, gradients, learning rates, and policy noise](#7-losses-gradients-learning-rates-and-policy-noise)
8. [Curriculum](#8-curriculum)
9. [Replay and update schedule](#9-replay-and-update-schedule)
10. [Performance and progress](#10-performance-and-progress)
11. [Network plasticity metrics](#11-network-plasticity-metrics)
12. [Local redundancy](#12-local-redundancy)
13. [Optional interventions and permutation experiments](#13-optional-interventions-and-permutation-experiments)
14. [RND, media, and neighbouring race-only fields](#14-rnd-media-and-neighbouring-race-only-fields)
15. [W&B system metrics and bookkeeping](#15-wb-system-metrics-and-bookkeeping)
16. [Common mistakes and useful plots](#16-common-mistakes-and-useful-plots)
17. [Source map and verification](#17-source-map-and-verification)

## 1. Basic words, units, and averaging

- **Reward `r`**: the number produced by one policy transition. It already includes its coefficients, penalties, and any timestep scaling.
- **Undiscounted episode reward `S`**: the sum of all rewards before the episode ends. `S = r_0 + r_1 + ... + r_(T-1)`.
- **Discounted return `G`**: earlier rewards receive more weight. `G = r_0 + gamma*r_1 + gamma^2*r_2 + ...`.
- **Gamma**: the discount factor. Read it from the selected algorithm's effective configuration. Our SAC example uses 0.97. PPO often uses a different value.
- **Policy timestep `dt`**: physics timestep times action decimation. Here, `0.005 s * 4 = 0.02 s`, or 50 policy steps per second.
- **Termination**: a real episode-ending event, such as a base collision. We use no future-value tail for the new return diagnostic.
- **Truncation / timeout**: the episode stops because the recording horizon ends. The new return diagnostic estimates the remaining value from the final state, before reset.
- **Bootstrap**: that estimated future-value tail. It is a prediction, not an observed reward.
- **Q**: predicted discounted value of a state and action. SAC uses two Q networks.
- **V**: predicted discounted value of a state. PPO uses V.
- **Replay**: stored transitions used for off-policy SAC updates. These need not come from the current policy or simulator.
- **LoRA**: low-rank adapters added to network weights. The metric definitions do not change just because adapters, rather than all weights, are trained.

### The averaging rule is part of the quantity

| Data group | What one plotted value averages |
|---|---|
| `Returns/actual_return*` | Completed episodes within the logging interval. Each episode has equal weight. Partial episodes continue into later intervals. |
| `CriticQ/q1_mean`, `q2_mean`, `mean`, `reduced_mean` | Replay minibatch means, then critic updates within the current update phase. |
| `CriticQ/policy_mean` | Replay minibatch means for freshly sampled policy actions, then actual actor evaluations. |
| `CriticV/mean` | V values already collected during rollout. Each environment-transition value has equal weight. |
| `RewardsPerStep/*` | Environment means on each policy step, then step means within the logging interval. MJX PPO directly averages its rollout array. |
| Isaac `Episode_Reward/*`, ratios, and lengths | The environment emits a value for each reset batch. The writer usually averages those batch values; batches do not necessarily have the same episode count. |
| MJX PPO episode metrics | Completed episodes pooled by count over the iteration. Smoothed episode metrics pool counts over the selected iteration window. |
| `Train/mean_reward` | A rolling buffer of completed episodes, not just the current iteration. |
| Network diagnostics | A separate probe with its own inputs and cadence. See the specific diagnostic below. |

An absent field means it was not emitted. It does not mean zero. For example, there is no `Returns/actual_return` point when no episode finishes. There is no `CriticQ/policy_mean` point when no actor evaluation occurs. A reward term with a zero coefficient may still have a zero `RewardsPerStep/*` field, but its scale-independent ratio is omitted.

## 2. Discounted returns and theoretical limits

These are available in all four updated paths. They use raw environment rewards, before any learning-time timeout correction. They are not GAE or lambda returns.

Let `T` be the number of transitions observed from an episode's start. Define:

```text
reward_prefix = sum over t=0,...,T-1 of gamma^t * r_t
bootstrap_contribution = gamma^T * B       on truncation
bootstrap_contribution = 0                on true termination
actual_return = reward_prefix + bootstrap_contribution
```

The first reward has weight 1. The value after the last reward has weight `gamma^T`, not `gamma^(T-1)`.

### Every `Returns/*` field

All return-valued fields below use discounted reward units. Counts use episodes. A mean describes the completed episodes in that logging interval, not a rolling 100-episode average.

| Exact key | Calculation and interpretation |
|---|---|
| `Returns/theoretical_return_lim_pos` | Current bounded-positive reward rate times `dt / (1-gamma)`. A positive-only ceiling, assuming an infinite discounted continuation. It is not a guarantee that all rewards can be maximized together with zero costs. |
| `Returns/theoretical_return_lim_neg` | The smaller of a collision-terminal estimate and a continuing-contact estimate. This is an approximation, not a strict lower bound on the full reward. |
| `Returns/actual_return` | Mean of `reward_prefix + bootstrap_contribution` over all completed episodes. |
| `Returns/actual_return_min` | Smallest completed `actual_return` in this logging interval. Not the smallest value ever seen during the run. |
| `Returns/actual_return_max` | Largest completed `actual_return` in this logging interval. Not a theoretical maximum. |
| `Returns/actual_return_terminal` | Mean `actual_return` among true terminations only. Their bootstrap is zero. |
| `Returns/actual_return_truncated` | Mean `actual_return` among timeouts only. Includes the estimated final-state tail. |
| `Returns/reward_only_return` | Mean discounted sum of the rewards actually observed before the end. No value tail and no explicit entropy term. |
| `Returns/bootstrap_contribution` | Mean of the discounted final Q/V tail across all completed episodes. True terminations contribute zero to this mean. |
| `Returns/actual_soft_return` | SAC only. Adds the sampled entropy contributions described below, for comparison with initial soft Q. |
| `Returns/completed_episodes` | Number of completed episodes in the logging interval. |
| `Returns/terminal_episodes` | Number of true terminations in that interval. |
| `Returns/truncated_episodes` | Number of timeouts without a simultaneous true termination. |

Useful checks:

```text
actual_return = reward_only_return + bootstrap_contribution
completed_episodes = terminal_episodes + truncated_episodes
```

These identities hold for the emitted means and counts, apart from floating-point rounding.

### The positive line

The helper includes the positive parts of these active scalar coefficients:

- `track_lin_vel_xy_reward_scale`
- `track_ang_vel_z_reward_scale`
- `two_feet_above_height_reward_scale`
- `track_base_height_reward_scale`, if present

It reads the current scalar values after curriculum updates. It does not sum every entry in the curriculum arrays. It excludes costs and uses `0 <= gamma < 1`.

For your command, with `dt=0.02` and `gamma=0.97`:

| Phase | Lift coefficient | XY tracking coefficient | Yaw coefficient | Calculation of positive ceiling |
|---|---:|---:|---:|---|
| 1 | 2.0 | 1.2 | 0.5 | `(2.0+1.2+0.5)*0.02/0.03 = 2.4667` |
| 2 | 1.2 | 1.8 | 0.5 | `(1.2+1.8+0.5)*0.02/0.03 = 2.3333` |
| 3–5 | 1.5 | 1.5 | 0.5 | `(1.5+1.5+0.5)*0.02/0.03 = 2.3333` |

If `negate_positive_rewards=True`, these positive terms are shifted downward. The positive ceiling becomes zero. Ratios that divide by the positive ceiling are then omitted.

For an episode with a true finite end after `T` steps, the constant-positive sum would instead be `r_max*(1-gamma^T)/(1-gamma)`. We log the infinite-continuation ceiling because the timeout return includes a continuation estimate.

### The negative line

For the usual positive-reward mode, define:

```text
base = min(base_collision_terminal_penalty, 0)
forbidden = min(three_or_more_feet_contact_penalty_reward_scale, 0) * dt
thighs = 4 * min(undesired_contact_reward_scale, 0) * dt

terminal_candidate = base + forbidden + thighs
continuing_candidate = thighs / (1-gamma)       when forbidden contact resets
continuing_candidate = (thighs+forbidden)/(1-gamma) otherwise

negative_line = min(terminal_candidate, continuing_candidate)
```

The implementation also adds the negative range of the bounded tracking/height terms when those coefficients are negative or positive rewards are shifted. If the configured contact grace period exceeds the episode duration, it excludes the forbidden-contact cost.

For your normal 10-second command:

```text
terminal_candidate = -10 + (-100*0.02) + 4*(-2.25*0.02) = -12.18
continuing_candidate = -0.18/(1-0.97) = -6
negative_line = min(-12.18, -6) = -12.18
```

This line deliberately excludes action-rate, torque, soft joint-limit, and foot-force costs. It also does not prove that every maximum contact cost can occur together. A real return can fall below the line. The W&B configuration records these assumptions under `return_metrics`.

### What do we bootstrap from?

| Algorithm | Boundary prediction `B` |
|---|---|
| PPO | Current V evaluated at the final **pre-reset** observation. |
| SAC | Current online twin Q evaluated at one newly sampled policy action at the final pre-reset observation. Use the actor's min/mean reduction. Do not use a frozen target critic here. |

SAC boundary sampling preserves the training random state. Diagnostic forwards also preserve model modes, normalizer state, actor distribution caches, and recurrent memory. If failure and timeout coincide, failure wins: there is no diagnostic bootstrap.

The networks may change while an episode is running. A timeout value is therefore a current estimate of the unfinished tail, not an exact observation under one frozen policy. The new diagnostic does not change the learner's existing target construction.

### Reward-only versus soft SAC return

SAC's Q includes future entropy. A truncated `actual_return` is therefore a measured reward prefix plus an estimated **soft-Q** tail. It is not an entirely observed reward-only G.

Let `h_t = -alpha_t * log pi(a_t | s_t)`, using the sampled scaled action's log probability. The extra soft-return diagnostic is:

```text
true termination:
  sum(gamma^t*r_t) + sum over t=1,...,T-1 of gamma^t*h_t

truncation:
  the same prefix + gamma^T * (Q_reduced(s_T,a_T) + h_T)
```

There is no entropy contribution at `t=0`: Q already conditions on the initial action. Continuous-action entropy can be negative. This extra return is closer in definition to SAC Q, but its episode-start states still differ from the replay states used by the Q mean plots.

### Small arithmetic example

Suppose rewards are 1 and 2, `gamma=0.5`, and the boundary value is 6.

- True termination: `G = 1 + 0.5*2 = 2`.
- Truncation: `G = 1 + 0.5*2 + 0.5^2*6 = 3.5`.
- The reward prefix is 2 and the discounted bootstrap is 1.5.

We never add the learning-time timeout correction to the reward prefix and then add the same tail again.

## 3. SAC Q and PPO V predictions

### SAC `CriticQ/*`

These use **raw reward-unit Q**, not the log coordinates of categorical bins. For categorical outputs, each Q is `sum_i probability_i * raw_atom_i`. Scalar and PopArt critics use their own raw-value decoder.

Let `q1` and `q2` be the online twin critics. Apply reduction to each state/action pair before averaging:

```text
min:              q_reduced = min(q1, q2)
mean:             q_reduced = (q1+q2)/2
mean_pi_q_none:   q_reduced = (q1+q2)/2 for the actor and these diagnostic logs
```

The last mode still uses separate per-critic Bellman targets in training. The diagnostic does not change those targets.

| Exact key | Actions scored | Formula and averaging |
|---|---|---|
| `CriticQ/q1_mean` | Stored replay actions | Batch mean of q1, then mean over critic updates. |
| `CriticQ/q2_mean` | Stored replay actions | Same for q2. |
| `CriticQ/mean` | Stored replay actions | Batch mean of `(q1+q2)/2`, then mean over critic updates. Always the arithmetic twin mean, regardless of the actor rule. |
| `CriticQ/reduced_mean` | Stored replay actions | Batch mean of the configured per-sample reduction, then mean over critic updates. |
| `CriticQ/policy_mean` | Fresh stochastic current-policy actions, at replay states | Batch mean of the actor's reduced Q, then mean over actual actor evaluations. |
| `CriticQ/policy_mean_over_theoretical_pos` | Same as `policy_mean` | `CriticQ/policy_mean / Returns/theoretical_return_lim_pos`. Unitless. |
| `CriticQ/reduced_mean_over_theoretical_pos` | Same as `reduced_mean` | `CriticQ/reduced_mean / Returns/theoretical_return_lim_pos`. Unitless. |

Replay Q is read before the critic optimizer step. Policy Q is read after that critic step and before the actor step. The two means have different evaluation counts when the actor updates less often. If an actor evaluation does not occur, we omit its Q field rather than fabricate zero.

The Q ratios do not need extra network forwards. With sparse SAC logging, the raw Q loss dictionary describes the most recent update phase; the completed-return accumulator covers the whole logging interval. Do not assume these are the same time window.

`mean(min(q1,q2))` is not `min(mean(q1),mean(q2))`. Also, `policy_mean` in this metric is not the deterministic action option called `sac_action_source=policy_mean` in the separate local-redundancy probe.

### PPO `CriticV/*`

| Exact key | Definition |
|---|---|
| `CriticV/mean` | Mean of the state values already evaluated while collecting the rollout. Includes all collected states, not only episode starts or ends. No new V forward is needed for this mean. |
| `CriticV/over_theoretical_pos` | `CriticV/mean / Returns/theoretical_return_lim_pos`. Unitless; omitted when that ceiling is zero. |

V uses discounted reward units. The averages describe visited rollout states. They are not averages of completed episode returns. Isaac PPO values are collected before the associated optimizer update. MJX PPO uses `traj.value` from its collector.

## 4. Categorical critic distributions

These SAC-only fields are emitted when the critic uses categorical outputs. They describe the learned representation, not a calibrated confidence interval for the real episode return.

For each replay state/action pair, take the probability mixture `p = (p1+p2)/2`. Let `z_i` be the saved raw atoms. Define their symlog coordinates:

```text
x_i = sign(z_i) * log(1+abs(z_i))
mu = sum_i p_i*x_i
sigma = sqrt(sum_i p_i*(x_i-mu)^2)
```

Quantiles are atom-resolution locations where the cumulative probability reaches 0.05, 0.50, or 0.95. Compute each sample's statistic first. Average those statistics over the replay batch, then over critic updates.

| Exact key | Definition | Units |
|---|---|---|
| `CriticDist/symlog_mean` | Mean of the probability-weighted centers `mu`. | Symlog coordinates |
| `CriticDist/symlog_std_within_state` | Mean of each sample's distribution width `sigma`. Twin disagreement can widen it because we mix probabilities first. | Symlog coordinates |
| `CriticDist/symlog_std_across_states` | Population standard deviation of the centers `mu` across a replay batch, then averaged over updates. The samples also differ in action. | Symlog coordinates |
| `CriticDist/symlog_q05` | Mean of the per-sample 5% atom quantiles. Not the 5% quantile of all observed episodes. | Symlog coordinates |
| `CriticDist/symlog_q50` | Mean of the per-sample medians. | Symlog coordinates |
| `CriticDist/symlog_q95` | Mean of the per-sample 95% atom quantiles. | Symlog coordinates |
| `CriticDist/symlog_q05_q95_width` | Mean difference between the per-sample 95% and 5% quantiles. | Symlog coordinates |
| `CriticDist/active_atoms_p10` | Mean number of atoms with probability strictly greater than 0.10. The `p10` suffix means a probability threshold, not a percentile. | Atoms |
| `CriticDist/effective_atoms` | Mean of `exp(H)`, where `H = -sum_i p_i*log(p_i)`. A smooth count of occupied categories. | Effective atoms |
| `CriticDist/edge_mass` | Mean of `p_first + p_last`. Measures probability on the outermost support atoms. | Fraction, 0–1 |
| `Loss/critic_target_clipped_fraction` | Two-hot / HL-Gauss: fraction of scalar Bellman targets outside the support. C51: expected probability mass of transformed target atoms outside support, averaged over both targets. Despite `Loss/`, this is a diagnostic fraction, not a loss. | Fraction, 0–1 |

Do not convert `CriticDist/symlog_mean` with symexp and call it mean Q. A nonlinear transform does not commute with averaging. Use `CriticQ/*` for raw Q.

For symexp support with limit X, the raw endpoints are `+/- (exp(X)-1)`. Thus X=5 means about +/-147.413 reward units, not +/-5. Linear support instead uses its saved raw endpoints. The support bounds decoded Q and target projection, not the environment's possible rewards or G.

For two-hot and HL-Gauss, these distributions are trained to represent scalar targets. Width can reflect critic disagreement and fitting behaviour. It is not automatically uncertainty about the robot's future.

### PopArt

These fields are present when the running-target-normalization critic is enabled:

| Exact key | Definition |
|---|---|
| `PopArt/mean` | Current running mean of Bellman targets. Discounted reward units. |
| `PopArt/std` | Current running target standard deviation, with a minimum of `1e-4`. Discounted reward units. |

The head is trained in normalized units, but raw Q is decoded as `std*normalized_output + mean`. The PopArt update rescales online and target heads to preserve their raw outputs when the statistics move. These statistics are not measured episode-return moments. Check supported checkpoint modes before moving a PopArt model to MJX.

## 5. Every reward term and its logging families

### Inputs and notation

For the table below, `w` is that term's effective coefficient. `dt` is the policy timestep. `sum` means the sum over the listed joints, feet, or vector coordinates within one environment. Position angles use radians. Velocity coordinates use metres/second or radians/second. Forces use newtons. Torques use N m.

The expression in the table is the already-scaled **one-step contribution**. Negative coefficients turn a nonnegative raw cost into a negative reward.

Isaac has all 17 terms below. MJX currently has the ten rows marked “Both”. A zero coefficient produces a zero term; it does not remove the name from every log family.

| Reward suffix `<term>` | Backend | One-step calculation and meaning |
|---|---|---|
| `track_lin_vel_xy_exp` | Both | `w*dt*exp(-sum((command_xy-tracked_velocity_xy)^2)/tracking_std^2)`. Reward for matching planar velocity. Two-feet uses the configured world-heading tracking frame. |
| `track_ang_vel_z_exp` | Both | `w*dt*exp(-(command_yaw_rate-tracked_yaw_rate)^2/tracking_std^2)`. Reward for yaw-rate tracking. Two-feet tracks world z yaw rate. |
| `lin_vel_z_l2` | Isaac | `w*dt*vertical_body_velocity^2`. Usually a cost for vertical motion. Raw units `(m/s)^2`. |
| `ang_vel_xy_l2` | Isaac | `w*dt*sum(body_roll_pitch_angular_velocity^2)`. Usually a roll/pitch-motion cost. Raw units `(rad/s)^2`. |
| `dof_torques_l2` | Both | `w*dt*sum(joint_torque^2)`. Raw units `(N m)^2`. Isaac uses `robot.data.applied_torque`, a clipped implicit-PD approximation, not a measured PhysX drive torque. MJX uses actuator generalized forces. |
| `dof_acc_l2` | Isaac | `w*dt*sum(joint_acceleration^2)`. Raw units `(rad/s^2)^2`. |
| `soft_qlim_penalty` | Both | `w*sum(max(soft_lower-q,0)+max(q-soft_upper,0))`. Sum of actual joint-position violations in radians. **No dt factor.** |
| `action_rate_l2` | Both | `w*dt*sum((action-current_previous_action)^2)`. Penalizes changes in the raw policy command, before actuator mapping/delay. It is not directly squared joint velocity. |
| `feet_air_time` | Isaac | `w*dt*sum((last_air_time-air_time_threshold)*first_contact)`, enabled only when planar command norm exceeds 0.1. First-contact indicators select landing feet. Raw units seconds; short flight times can give negative raw values. |
| `two_feet_above_height` | Both | `w*dt*lift_kernel`. Kernel is 1 above the selected feet's height threshold, otherwise `exp(-alpha*(threshold-average_height)^2)`. Contact gates require the selected feet airborne; rear contact can also be required by config. Isaac supports front/back symmetry selection. |
| `three_or_more_feet_contact` | Both | `w*dt*forbidden_contact_indicator`. In asymmetric mode this is **any front foot or front thigh contact**, not literally three feet. In symmetric mode it is at least three feet. Inactive during the start grace period. |
| `undesired_contacts` | Both | `w*dt*number_of_contacting_thighs`. Up to four thighs, detected above the separate undesired-contact threshold. It has no start-grace exemption. |
| `base_collision_terminal` | Both | `w*base_collision_indicator`. **No dt factor.** External base contact causes termination independently of the forbidden-contact option. Isaac filters out own-link contributions from the external base sensor. |
| `flat_orientation_l2` | Isaac | `w*dt*sum(projected_gravity_xy^2)`. Penalizes base tilt away from a horizontal posture. Raw units are dimensionless. Its two-feet coefficient is normally zero. |
| `track_base_height_exp` | Isaac | `w*dt*exp(-base_height_exp_scale*(base_height-desired_height)^2)`. Height is relative to the environment's terrain origin. |
| `force_transmited_through_joints` | Isaac | `w*dt*sum(incoming_joint_force_xyz^2)` over selected thigh/calf links. Raw units N^2. The spelling `transmited` is the actual logged spelling. |
| `foot_contact` | Both | Normally `w*dt*sum(max(foot_force_norm-safe_threshold,0)^2)`. Penalizes excessive foot force. Isaac can use the unsquared excess when `square_foot_penalty=False`; MJX currently uses the square. No configured reward clipping. |

The positive exponential/kernel terms are shifted to `kernel-1` when Isaac's `negate_positive_rewards=True` and the coefficient is positive. Interpret their sign and ratios using that active mode.

For your example, thigh contact above 0.6 N costs `-2.25*0.02 = -0.045` per thigh per step. Front foot/thigh contact above 1 N, after the 1-second grace, also costs `-100*0.02 = -2` and resets because your reset flag is true. Rear-thigh contact alone does not trigger that reset. These numbers are configuration examples, not universal defaults for every run.

### Four reward logging families

Each family below expands to every supported reward suffix in the table above.

| Key pattern | Formula | Units and interpretation |
|---|---|---|
| `RewardsPerStep/<term>` | Mean of the signed, scaled one-step contribution. | Reward units per policy transition. Shows what the environment actually adds. |
| `PerStepRewardRatio/<term>` | `RewardsPerStep/<term> / (w*step_factor)`, where `step_factor=dt`, except it is 1 for base terminal and soft joint limits. Omitted when w=0. | The average raw term. Its units are those of the input cost/kernel: radians for soft-limit violation, counts for contacting thighs, etc. **Not every ratio is a dimensionless fraction between 0 and 1.** |
| `Episode_Reward/<term>` | Absolute value of the mean undiscounted episode contribution, divided by configured maximum episode seconds. | Reward units per configured episode second. Penalties are displayed positive. This is neither signed G nor the mean reward divided by the actual episode length. |
| `Episode_Reward_ratio/<term>` | `abs(mean episode contribution) / (abs(w)*H*step_factor/dt)`, where H is configured maximum episode seconds. | Raw term averaged over the **nominal full horizon**. Early termination reduces it. Its units still depend on the raw term. Isaac only; current MJX trainers do not emit this family. |

Example: a -10 base penalty in a 10-second task appears as `Episode_Reward/base_collision_terminal = 10/10 = 1`, not -10. With 500 nominal policy steps, one base collision gives `Episode_Reward_ratio/base_collision_terminal = 10/(10*500) = 0.002`.

The `abs` is applied to the mean episode contribution. It is not a mean of per-transition absolute values. If a term can change sign, cancellation can happen before the absolute value.

In current Isaac two-feet, `Episode_Reward/total`, episode lengths, reward ratios, and all-time return extrema use regular completed episodes; posed airborne starts are excluded. The per-term `Episode_Reward/<term>` loop instead averages the reset batch, including posed starts. The new `Returns/*` tracker and runner rolling reward buffers include all completed trajectories. These cohorts can differ when posed starts are enabled.

### Combined reward fields

| Exact key | Definition |
|---|---|
| `RewardsPerStep/cmd_tracking` | Sum of the signed XY and yaw tracking contributions per policy step. |
| `RewardsPerStep/total` | Mean signed sum of all reward terms per policy step. |
| `Episode_Reward/cmd_tracking` | Sum of the two positive-magnitude, duration-normalized tracking episode fields. |
| `Episode_Reward/total` | Mean **signed, undiscounted** completed episode reward. Unlike the individual term fields, it has no absolute value and no duration division. |
| `Episode_Reward/alltime_min` | Smallest regular completed undiscounted episode reward seen since environment creation. Isaac only. |
| `Episode_Reward/alltime_max` | Largest regular completed undiscounted episode reward seen since environment creation. Isaac only. |

The all-time values reset when a new environment is created. They are not restored as a lifetime history by simply loading policy weights. SAC's writer reduces `_min` and `_max` fields appropriately over reset batches. The standard PPO writer averages emitted reset-batch values, so its plotted all-time field can be an average of intermediate running extrema rather than the final endpoint.

## 6. Episode totals, lengths, endings, and smoothing

### Completed episodes and runner rolling buffers

| Exact key | Available in | Definition |
|---|---|---|
| `Train/mean_reward` | Isaac PPO/SAC, MJX SAC | Rolling mean of signed, undiscounted episode reward. Default window is 100 completed episodes; MJX SAC uses `--episode-log-window`. With RND it also includes intrinsic rewards. |
| `Train/mean_episode_length` | Same | Rolling mean number of policy steps per completed episode. Steps, not seconds. |
| `Train/mean_reward/time` | SAC W&B; local TensorBoard variants | Same rolling reward, with training walltime supplied for a time-axis plot. Not a new reward definition. |
| `Train/mean_episode_length/time` | Local TensorBoard variants | Same rolling length against training time. The current W&B paths do not always emit it. |
| `Train/mean_reward_smooth` | Isaac PPO callback | Median of the last 50 emitted rolling-mean reward values. It is not a median of individual episodes. |
| `Train/mean_reward_smooth/time` | Isaac PPO, non-W&B local writer | Same median against training time. |
| `Episode/length_steps` | Isaac environment, MJX PPO | Mean completed episode length in policy steps, with the averaging/cohort rules described above. |
| `Episode/length_seconds` | Same | `Episode/length_steps * dt`. Seconds. |
| `Episodes/return` | MJX SAC | Signed, undiscounted rewards of completed episodes; the writer pools the emitted per-episode values. The plural namespace is intentional. |
| `Episodes/length_steps` | MJX SAC | Completed episode lengths; the writer pools the emitted values. |
| `episode/length_s` | MJX PPO | Alias of its completed mean episode length in seconds. Lowercase name is intentional. |

`Train/mean_reward` and `Returns/actual_return` are not interchangeable: one is an undiscounted rolling average, the other is a discounted current-interval average with a possible value tail.

### Ending reasons

Isaac's `Episode_Termination/*` values are counts produced on reset steps. The standard writer averages these count-valued reset payloads; they are not necessarily summed counts per training iteration. Reasons can overlap: base collision and forbidden contact can coincide, and the old raw timeout flag can also coincide with failure. Use the new disjoint `Returns/*_episodes` counts when you need that terminal/timeout distinction.

| Exact key | Meaning |
|---|---|
| `Episode_Termination/base_contact` | Base external collision count. |
| `Episode_Termination/forbidden_feet_contact` | Count ending because the configured forbidden-contact reset is active. |
| `Episode_Termination/time_out` | Count reaching the environment's horizon. |
| `Episode_Termination/diverged` | MJX PPO only: episodes with non-finite simulator positions. Their reward terms are zeroed to keep those failures from introducing non-finite rewards. |
| `Terminations/base_contact_per_step` | MJX SAC: fraction of environments showing base termination on each step, averaged over steps. |
| `Terminations/front_contact_per_step` | MJX SAC: fraction showing forbidden-contact termination. The name `front` comes from the usual asymmetric task; the predicate follows the configured symmetry mode. |
| `Terminations/diverged_per_step` | MJX SAC: fraction with a divergent simulator state. |
| `Terminations/timeout_per_step` | MJX SAC: fraction ending by timeout without a true termination. |

MJX PPO counts are pooled over the whole rollout:

| Exact key | Formula |
|---|---|
| `episodes` | Completed episodes in the iteration. Alias of `Episode/count`. |
| `terminations` | True terminations in the iteration. Alias of `Episode/termination_count`. |
| `timeouts` | Timeout-only episodes in the iteration. Alias of `Episode/timeout_count`. |
| `Episode/count` | Number of completed episodes. |
| `Episode/termination_count` | Number with a true termination. |
| `Episode/timeout_count` | Completed count minus true-termination count. |
| `Episode/completion_rate_per_env_step` | Completed count divided by `num_envs*rollout_steps`. |
| `Episode/termination_rate_per_env_step` | True-termination count divided by that same transition count. |
| `Episode/timeout_rate_per_env_step` | Timeout-only count divided by that same transition count. |

These per-step rates are event frequencies, not fractions of episodes that succeed. A timeout is not a successful task completion.

### MJX PPO smoothing

These use the rolling `--metrics-smoothing-window` iteration window. They do not smooth the new `Returns/*` metrics, losses, or Q/V predictions.

| Key pattern or exact key | Definition |
|---|---|
| `Smoothed/window_iterations` | Number of iterations currently present in the smoothing buffer. |
| `SmoothedReward/mean_step` | Arithmetic mean of `reward/mean_step` over buffered iterations. This underlying reward already includes PPO's learning-time timeout V correction. |
| `SmoothedRewardsPerStep/<term>` | Mean of each signed raw rollout reward-term mean over those iterations. All ten MJX terms, plus `cmd_tracking` and `total`. |
| `SmoothedPerStepRewardRatio/<term>` | Mean of each nonzero-coefficient per-step ratio over those iterations. |
| `SmoothedEpisode/completed_count` | Sum of completed episode counts across the buffer. |
| `SmoothedEpisode/length_steps` | Pooled sum of completed episode steps divided by pooled completed count. |
| `SmoothedEpisode/length_seconds` | That pooled mean times dt. |
| `SmoothedEpisode_Reward/total` | Pooled signed undiscounted episode rewards divided by pooled completed count. |
| `SmoothedEpisode_Reward/<term>` | Absolute pooled mean term contribution divided by configured maximum episode seconds. All ten MJX terms, plus `cmd_tracking`. |

Pooled episode means are not unweighted averages of each iteration's episode mean when those iterations complete different numbers of episodes.

## 7. Losses, gradients, learning rates, and policy noise

Loss units depend on the objective. They are not episode return units. Compare a loss across runs only when its definition, label type, scale, and reduction match.

### SAC losses

Let y be the Bellman target. It includes n-step discounted rewards and, where bootstrapping is allowed, the target Q and next-state entropy contribution. The target critic is not the same network used for the new diagnostic timeout tail.

| Exact key | Definition and averaging |
|---|---|
| `Loss/critic1` | First critic's objective, averaged over update minibatches. Scalar MSE is mean `(Q1-y)^2`. PopArt uses normalized target units. Categorical modes use cross entropy `-mean(sum(label_i*log(prob_i)))`. C51 uses its full projected labels. |
| `Loss/critic2` | Same for the second critic. |
| `Loss/actor` | Mean `alpha*log_pi - Q_reduced`, plus the weighted symmetry term when enabled. The current logger divides summed actor losses by `max(floor(number_of_critic_updates/policy_frequency),1)`. This nominal denominator can differ from the actual actor count when the phase boundary is not aligned. The Q mean uses the actual count. |
| `Loss/alpha` | Auto-temperature objective: `-mean(log_alpha * stop_gradient(log_pi+target_entropy))`. Averaged over critic-update slots. A fixed or frozen alpha can leave this field at zero; it is not a temperature value. |
| `Loss/symmetry` | Mean squared mismatch between mirrored policy outputs and mirrored original outputs. A weighted copy may enter actor loss, or the field may be diagnostic-only. Check `use_mirror_loss`. Its active averaging uses the logger's nominal actor-update denominator; diagnostic-only cadence emits the probe value. |
| `Loss/rnd` | Optional RND predictor MSE against a frozen target embedding. |
| `Loss/learning_rate` | SAC actor learning rate used by the runner's logger. It does not separately report critic or alpha LR. Those are in the effective configuration. |

For mean-per-critic targets, y can differ between Q1 and Q2. Cross-entropy loss and MSE loss have different numerical scales. A lower CE number than an MSE number does not establish a better policy.

### Isaac PPO losses

The PPO ratio is `exp(new_log_probability-old_log_probability)`. The surrogate loss uses normalized or unnormalized advantages according to configuration. The value target is the learner's GAE/lambda-return target, not the new completed-episode return diagnostic.

| Exact key | Definition |
|---|---|
| `Loss/surrogate` | Mean of the clipped PPO policy objective, with the sign chosen for minimization. |
| `Loss/value_function` | Mean squared V error against the PPO target. With value clipping, use the larger of clipped and unclipped squared errors. No universal “good value” across reward scales. |
| `Loss/entropy` | Mean Gaussian policy entropy across minibatches. Despite the namespace, this is the entropy value, not just the signed entropy penalty in the total objective. Differential entropy can be negative. |
| `Loss/symmetry` | Mirroring MSE, if symmetry is configured. It can be diagnostic-only even when augmentation is used. |
| `Loss/rnd` | Optional RND predictor MSE. |
| `Loss/learning_rate` | Current PPO optimizer learning rate after its adaptive/fixed schedule. |

### MJX PPO losses and policy diagnostics

These names intentionally use lowercase namespaces and match that script's own logger. One iteration averages its update diagnostics over epochs and minibatches.

| Exact key | Definition |
|---|---|
| `loss/policy` | Clipped PPO surrogate objective. |
| `loss/value` | `0.5 * mean((PPO_target-V)^2)`. This script does not use Isaac's optional clipped value loss. |
| `loss/total` | `loss/policy + value_loss_coef*loss/value - entropy_coef*policy/entropy`. |
| `policy/entropy` | Sum over action coordinates of `log_std + 0.5*log(2*pi*e)`, averaged over update calls. Nats. |
| `policy/clip_fraction` | Fraction of likelihood ratios with `abs(ratio-1)>clip_param`, averaged over updates. Fraction, 0–1. |
| `policy/kl` | Analytical old-Gaussian to new-Gaussian KL divergence, summed over action coordinates and averaged over samples/updates. Nats. Drives the adaptive LR schedule when enabled. |
| `grad_norm` | Global pre-clipping gradient L2 norm across trainable parameter arrays, averaged over updates. |
| `learning_rate` | Mean effective LR returned by the update calls in that iteration, rather than necessarily the last LR. |
| `updates_applied_frac` | Fraction of optimizer updates accepted by the finite-gradient guard. If a gradient norm is non-finite, that update is rejected. Fraction, 0–1. |
| `policy/action_std` | Mean `exp(log_std)` across action coordinates after the iteration's updates. Raw PPO action units. |
| `reward/mean_step` | Mean collector reward after the script adds `gamma*V(final_state)` to timeout transitions for learning. It is therefore not always equal to `RewardsPerStep/total`, which uses the raw environment terms. |

### Gradient fields

Norms are square roots of sums of squared gradient entries. They measure the gradient passed to the optimizer, not the actual parameter change made by Adam/AdamW. Their scale depends on losses and parameter count.

| Exact key | Definition |
|---|---|
| `Gradients/grad_norm_actor` | Pre-clipping norm for actor parameters, averaged over actual updates of that group. |
| `Gradients/grad_norm_critic` | Pre-clipping norm for critic parameters. SAC pools trainable parameters from both critics; it is not a separate Q1 or Q2 norm. |
| `Gradients/clipped_grad_norm_actor` | Actor norm after clipping. SAC logs the pre-norm clamped to the threshold; Isaac PPO measures its group after the shared policy clip. |
| `Gradients/clipped_grad_norm_critic` | Same convention for critic parameters. |
| `Gradients/grad_norm_alpha` | SAC absolute gradient of the scalar **log-alpha** parameter. It is not the gradient with respect to alpha itself. |

Isaac PPO clips actor and critic together with one global policy norm. SAC clips its actor and twin-critic parameter groups separately. The meaning of “clipped actor norm” therefore differs slightly across algorithms.

Older or local TensorBoard PPO paths can use `Loss/grad_norm_actor`, `Loss/grad_norm_critic`, `Loss/clipped_grad_norm_actor`, and `Loss/clipped_grad_norm_critic`. The current Isaac W&B writer remaps these four to `Gradients/*`. These aliases have the same calculations.

### Policy noise and temperature

| Exact key | Definition |
|---|---|
| `Policy/mean_noise_std` | Mean standard deviation from the actor's most recent cached Gaussian distribution, across batch/action dimensions. Usually the last update forward, not a mean over every rollout step. PPO uses raw action units; SAC uses pre-tanh latent Gaussian units. |
| `Policy/alpha` | Current SAC entropy temperature `exp(log_alpha)`, after updates. Not entropy and not actor standard deviation. |

SAC standard deviation is **not** the standard deviation of joint angles after tanh and action scaling. For PPO's standard direct mapping, multiplying raw action displacement by 0.25 converts it to a position-target displacement in radians. Clipping and exploration constraints can change the executed distribution.

## 8. Curriculum

These fields report configuration/state, not rewards achieved. Most environment fields are emitted on reset batches and the writer averages those values. If a phase changes inside an iteration, a plotted curriculum value can be fractional. The new theoretical return limits instead read the current scalar configuration at logging time.

| Exact key | Definition and units |
|---|---|
| `Curriculum/two_feet_phase` | Current coordinated two-feet phase. Phases are one-based. |
| `Curriculum/global_idx` | Composite curriculum-stage index used for checkpoint grouping. The source task defines the mapping; it is not automatically the same as phase minus one. |
| `Curriculum/command_lin_vel_x_abs` | Maximum absolute endpoint of the active x-command range, m/s. |
| `Curriculum/max_velx_range_idx` | Zero-based index into the velocity-range curriculum. |
| `Curriculum/base_push_force_xy_abs` | Maximum absolute endpoint of the configured horizontal push range, N. Not the measured force on a particular step. |
| `Curriculum/base_push_force_idx` | Zero-based index into the base-push curriculum. |
| `Curriculum/base_push_mean_reward_smooth` | Smoothed mean undiscounted episode reward used by the legacy force-curriculum gate. Not G. |
| `Curriculum/tricky_terrain_active` | 0/1 state indicating that the tricky-terrain stage is active. |
| `Curriculum/two_feet_above_height_reward_scale` | Active lift coefficient. |
| `Curriculum/track_lin_vel_xy_reward_scale` | Active planar tracking coefficient. |
| `Curriculum/three_or_more_feet_contact_penalty_reward_scale` | Active forbidden-contact coefficient. Multiply by dt for its one-step contribution. |
| `Curriculum/two_feet_above_height_alpha` | Active lift-kernel sharpness. In `exp(-alpha*height_error^2)`, larger alpha penalizes height error more strongly. Units 1/m^2. Not SAC entropy alpha. |
| `Curriculum/two_feet_above_height_threshold` | Active desired airborne foot-height threshold, m. |
| `Curriculum/actuation_delay_max` | Maximum configured actuation delay in **physics steps**, not policy steps. At 5 ms physics dt, a value 3 means up to 15 ms. |
| `Curriculum/opposite_direction_cmd_prob` | Probability of selecting the opposite command under the command-resampling rule. Fraction. |
| `Curriculum/front_back_asymetry` | 0/1 asymmetric-mode state. The spelling `asymetry` is intentional in the code. Changes reward and forbidden-contact predicates. |
| `Curriculum/events_randomization_active` | 0/1 state of phase-controlled startup/property randomization. |
| `Curriculum/twofeet_airborne_reset_prob` | Probability of a posed airborne/two-feet reset. Fraction. |
| `Curriculum/advance_reward_ratio` | Latest scale-independent completed-episode reward ratio used for the current phase transition. The selected raw tracking/lift signal depends on the transition. It is not the total return. |
| `Curriculum/terrain_levels` | Shared runner support for the environment's terrain difficulty level. Present only when the selected task emits it. Not a two-feet performance score. |

An episode can span a curriculum change. Its return includes the rewards actually produced during that episode. A current-phase theoretical line and a just-completed episode return need not describe one identical reward configuration throughout the episode.

## 9. Replay and update schedule

These fields are used by MJX SAC's interaction-budgeted schedule. The replay mixing fields can also appear with the shared mixed buffer.

| Exact key | Definition and units |
|---|---|
| `Replay/offline_fraction` | Current requested probability/fraction of minibatch samples drawn from the retained offline buffer. Follows the configured annealing schedule. It is not the fraction of current robot interactions that are offline. |
| `Replay/online_valid_transitions` | Number of valid online replay entries available for sampling, respecting the buffer's n-step-window requirements. Samples are drawn with replacement, so a batch can contain many more online draws than distinct valid entries. |
| `Schedule/collected_transitions` | Environment transitions collected in this rollout phase, including all parallel environments. |
| `Schedule/online_transitions` | Cumulative interactions collected since this `learn()` invocation. Not retained offline-buffer size. |
| `Schedule/gradient_updates` | Number of critic update iterations allocated to the current phase. Actor and alpha updates can have different clocks. |
| `Schedule/update_phases` | Cumulative number of phases that actually ran `alg.update()`. Warm-up phases do not increment it. |
| `Schedule/warmup` | 1 until the interaction warm-up threshold is reached, otherwise 0. A phase can still have zero updates for another reason, such as an incomplete n-step window. |

The configured UTD is critic update iterations divided by collected environment transitions. It is not minibatch samples divided by transitions. Multiplying UTD by batch size estimates sample reuse; augmentation can increase the actual processed batch further. These ratios are configuration/calculation concepts, not additional time-series keys unless a specific run logs them.

## 10. Performance and progress

| Exact key | Definition and units |
|---|---|
| `env_steps` | Cumulative environment interactions, counting each parallel environment once per policy step. Isaac counts all distributed ranks when applicable. Not physics substeps. |
| `env_interactions` | MJX PPO alias of cumulative environment transitions. |
| `iteration` | MJX PPO one-based iteration number. Isaac and the SAC runner generally use zero-based logged iteration steps. |
| `Perf/total_fps` | Total collected environment transitions divided by collection plus learning seconds for the logged interval. FPS means transitions/second here, not rendered video frames. |
| `Perf/collection_time` | Collection time, seconds. SAC can sum this over its sparse logging window. |
| `Perf/collection time` | Current upstream Isaac PPO spelling of the same collection-time quantity, with a space. Keep this alias when selecting charts. |
| `Perf/learning_time` | Time spent in the update phase, seconds. |
| `throughput_steps_s` | MJX PPO: `num_envs*rollout_steps / elapsed_iteration_seconds`. Environment transitions/second. |
| `wall_time_s` | MJX PPO elapsed time for this iteration, not total run runtime. |
| `walltime` | SAC writer's supplied cumulative timed training seconds, used for `Train/mean_reward/time`. It excludes some startup/logging/probe/save overhead. |

These timings are not always end-to-end program runtime. Writer work, optional diagnostics, checkpoint writes, and startup can sit outside the timed collection/update region. Initial JIT compilation inside a timed region can make the first iteration much slower. Compare steady-state throughput separately from first-iteration startup.

## 11. Network plasticity metrics

“Plasticity” here means the network's ability to keep adapting. These metrics are useful clues. None alone proves that adaptability is lost or restored.

The Isaac flag `--plasticity-metrics` enables the cheaper weight/activation/gradient diagnostics. Their cadence is `--plasticity-metrics-interval`; the activation sample cap defaults to 4096. They use the latest vectorized observation batch at the logging hook, not all SAC replay or the full PPO rollout. Network weights are measured at the post-update hook.

The dynamic network name `<net>` is `actor` and `critic` for PPO, or `actor`, `critic1`, and `critic2` for SAC. Per-layer indices `layer_00`, `layer_01`, etc. follow hidden activation execution order. Encoders can add activation layers; do not always assume the index is one of three dense layers.

### Per-layer family

Every name below is available as:

```text
Plasticity/per_layer/<net>/<metric>/layer_NN
```

| `<metric>` | Exact calculation | Units |
|---|---|---|
| `dormant_pct` | 100 times fraction of hidden units whose mean absolute activation is strictly below `1e-5`. | Percent |
| `dormant_tau_pct` | 100 times fraction with `unit_mean_abs_activation / layer_mean_abs_activation <= 0.025`. If the layer mean is zero, all units count as dormant. | Percent |
| `feature_rank` | Smallest k whose top-k squared singular values contain at least 99% of the activation matrix's squared-singular-value sum. The matrix is not mean-centered. | Features |
| `feature_rank_frac` | `feature_rank / feature_num` for that same layer. | Fraction |
| `feature_num` | Number of feature columns in that layer's activation matrix. | Features |

The rank can also be limited by the number of probe rows. Rank fraction is a fraction, not a percentage: multiply by 100 to express percent. It is not algebraic full rank and not a centered PCA dimension.

### Whole-network family

Every name below appears under `Plasticity/summary/<net>/<metric>`.

| `<metric>` | Calculation |
|---|---|
| `weight_norm` | Global parameter L2 norm: `sqrt(sum(all parameter entries squared))`. Includes biases and normalization parameters. Not divided by parameter count and not a weight-only mean. |
| `grad_kurtosis` | With `L=log(abs(gradient)+1e-8)`, calculate `mean((L-mean(L))^4) / mean((L-mean(L))^2)^2`. This is ordinary kurtosis, not excess kurtosis, and it is not kurtosis of the raw signed gradients. |
| `dormant_pct` | Dormant-unit count pooled over all hidden units divided by total unit count, times 100. Not an unweighted mean of layer percentages. |
| `dormant_tau_pct` | Same pooled rule for the layer-relative dormant criterion. |
| `feature_rank` | Median of the raw per-layer ranks. |
| `feature_rank_frac_median` | Median of per-layer `rank/width` fractions. |
| `feature_rank_frac_mean` | Arithmetic mean of those fractions, giving each layer equal weight. |
| `feature_rank_frac_min` | Smallest of those fractions. |
| `feature_num` | Width of the **last collected hidden activation layer**, not the sum of all hidden widths or a denominator for the network median rank. |
| `feature_rank_frac` | Legacy summary name in `tacjuisg`: median of per-layer fractions. In current source it is replaced by the explicit median/mean/min names above. |

Gradient kurtosis is captured at the first armed optimizer step for that measurement, after the learner's clipping, not averaged over every step in the iteration. Missing/degenerate gradients are skipped. Weight norm grows with architecture size; compare like-for-like networks or explicitly normalize before interpreting size differences.

Do not calculate a network fraction as `summary/feature_rank / summary/feature_num`. The numerator is a median across layers; the denominator is the last hidden width. Use the explicit fraction metric or the per-layer rank/width pair.

## 12. Local redundancy

This separate Isaac probe is enabled by default under `agent.local_redundancy`, independently of `--plasticity-metrics`. It uses disposable model copies. It does not train the live model or fit an auxiliary optimizer to completion.

For a prediction f and independent Gaussian noise z, the probe uses:

```text
synthetic_target = stop_gradient(f(input)) + sigma*z
probe_loss = sum_outputs((f(input)-synthetic_target)^2) / (2*sigma^2)
local_redundancy = mean_inputs(sum_parameter_entries(probe_gradient^2))
```

We square each example's gradient before averaging. This differs from squaring a mean batch gradient, where examples can cancel. The result is a first-order regression-gradient proxy, not bits, a task return, or a demonstrated learning rate.

Default inputs are Gaussian at the normalized-observation boundary. In `input_mode=observations`, use distinct rows from the latest vectorized observation instead. Check `agent.local_redundancy` in the saved config: input distribution, sigma, action-source choice, and sample count can change the scale and interpretation.

The PPO actor probe predicts deterministic action means. The SAC actor probe predicts the **latent pre-tanh mean**, excluding log-std-only output parameters from the prediction. Critic probes predict online V, Q1, and Q2 in raw value units. SAC critic actions are uniform inside the action bounds by default, or deterministic current mean actions with `sac_action_source=policy_mean`.

The following keys appear under `Plasticity/summary/<net>/<metric>`. Networks are `actor` and `critic` for PPO. SAC has `actor`, `critic1`, and `critic2`; its extra `critic` average emits only the two score fields, not a combined standard error.

| `<metric>` | Definition |
|---|---|
| `local_redundancy` | Mean squared per-example gradient L2 norm over parameters that participate in this prediction. Its units depend on the model parameterization and sigma. |
| `local_redundancy_per_output` | Same score divided by the prediction's output dimension. Helps compare a 12-output actor with a scalar value head, but does not remove all architecture/scale effects. |
| `local_redundancy_stderr` | Sample standard deviation of the per-input scores divided by sqrt(sample count). An estimate of sampling error, not confidence in robot performance. |
| `local_redundancy_num_samples` | Actual number of probe inputs. Gaussian mode can exceed environment count; observation mode cannot. |
| `local_redundancy_output_dim` | Number of predicted output coordinates. |
| `local_redundancy_parameter_count` | Number of parameter entries with a gradient connection to this prediction. Frozen or unused parameters are excluded. |

SAC's `Plasticity/summary/critic/local_redundancy` and `.../local_redundancy_per_output` are arithmetic means of the corresponding Q1/Q2 scores. They are not a probe of `min(Q1,Q2)`.

| Exact key | Definition |
|---|---|
| `Plasticity/local_redundancy_valid` | 1 when the scheduled probe succeeds, 0 when it fails. Missing score fields on failure are not zeros. |
| `Plasticity/local_redundancy_seconds` | Wall-clock seconds spent in the probe and its logging. |

Default cadence is every 100 iterations including 0; your command selects 50. Probe seeds are independent of training seeds. Larger scores can indicate more local gradient response, but comparisons must hold input distribution, sigma, parameterization, and action-source choice fixed.

More implementation detail: [LOCAL_REDUNDANCY.md](source/scripts/rsl_rl/LOCAL_REDUNDANCY.md).

## 13. Optional interventions and permutation experiments

These fields appear only when the matching intervention is enabled. Counts can continue from loaded intervention state; they are not always counts since the current run started.

### Continual backpropagation, `--use-cbp`

| Exact key or pattern | Definition |
|---|---|
| `CBP/optimizer_steps` | Optimizer steps processed by the CBP manager. Its maturity clock is optimizer steps, not environment interactions. |
| `CBP/replacements_last_update` | Sum of units replaced in the manager's latest update. Not necessarily the total replaced throughout the whole PPO iteration. |
| `CBP/replacements_total` | Cumulative units replaced across groups. |
| `CBP/replacements_total/<group>` | Cumulative units replaced in that named network/layer group. The group names come from the collected CBP topology. |

### Observation-permutation experiment

All flags use 0/1. Phase/permutation counters are integers. These markers explain intentional input-distribution shifts and network resets; they are not performance scores.

| Exact key | Definition |
|---|---|
| `PlasticityExperiment/permutation_index` | Zero-based current observation-permutation round. |
| `PlasticityExperiment/permutations_seen` | Current index plus 1. |
| `PlasticityExperiment/round_step` | Zero-based iteration position within the round. |
| `PlasticityExperiment/round_progress` | `(round_step+1)/round_duration`, a fraction ending at 1 at the boundary. |
| `PlasticityExperiment/reset_all_enabled` | Whether reset-all intervention is configured. |
| `PlasticityExperiment/network_reset_event` | Boundary marker when a network reset is scheduled after the current iteration. |
| `PlasticityExperiment/network_resets_total` | Cumulative reset count. |
| `PlasticityExperiment/first_layer_only_enabled` | Whether the input-layer-only intervention is configured. |
| `PlasticityExperiment/first_layer_only_active` | Whether the rest of the continued network is currently frozen. |
| `PlasticityExperiment/first_layer_only_freeze_event` | Boundary marker when that freeze is activated. |
| `PlasticityExperiment/first_layer_trainable_parameter_fraction` | Number of trainable input-layer parameter entries divided by total policy parameter entries. Fraction. |

### Plasticity mitigation

These use the prefix `PlasticityMitigation/`. Each suffix in this table is an emitted quantity.

| Suffix | Definition |
|---|---|
| `enabled` | 1 when a mitigation controller is present. |
| `layernorm_enabled` | 0/1: LayerNorm intervention selected. |
| `regenerative_l2_enabled` | 0/1: attraction toward stored initial parameters selected. |
| `soft_shrink_perturb_enabled` | 0/1: small continuous shrink/perturb steps selected. |
| `boundary_shrink_perturb_enabled` | 0/1: distribution-boundary shrink/perturb selected. |
| `target_parameter_fraction` | Entries selected by the intervention divided by all policy parameter entries. |
| `target_parameter_l2` | Global L2 norm of the selected current parameters. |
| `regenerative_distance_from_init` | L2 distance of selected current parameters from their stored initial values. Zero when regeneration is disabled. |
| `regenerative_penalty` | Latest computed coefficient times that L2 distance. **This implementation logs coefficient times the norm, not coefficient times the squared norm.** It is separate from the PPO loss fields. |
| `optimizer_steps` | Steps processed by the mitigation controller. |
| `soft_perturbations_total` | Cumulative continuous perturbation events. |
| `boundary_perturbations_total` | Cumulative boundary perturbation events. |
| `last_perturbation_l2` | L2 norm of the parameter change made by the latest perturbation. Not the optimizer's learning update. |
| `layernorm_layers` | Number of LayerNorm layers inserted by the intervention. |
| `regenerative_l2_coef` | Active regeneration coefficient, or zero when disabled. |
| `soft_sp_beta` | Active continuous shrink/perturb coefficient, or zero when disabled. |
| `sp_beta` | Active boundary shrink/perturb coefficient, or zero when disabled. |
| `boundary_event` | 0/1 marker for a boundary shrink/perturb event. |

The full event key is `PlasticityMitigation/boundary_event`; it marks the intervention boundary, not an episode ending.

LayerNorm, parameter projection, and intervention configuration fields are not additional time-series quantities just because they are present in W&B config. The weight-normalization intervention is visible through configuration and the parameter/activation diagnostics; it does not add a separate family of WN scalar logs.

More detail: [PLASTICITY_MITIGATION.md](source/scripts/rsl_rl/PLASTICITY_MITIGATION.md).

## 14. RND, media, and neighbouring race-only fields

### Random Network Distillation (RND)

RND gives an intrinsic novelty reward from a learned predictor's mismatch with a fixed target network. It is optional. Its normalization and weight are part of the effective config.

| Exact key | Definition |
|---|---|
| `Rnd/mean_extrinsic_reward` | Rolling mean completed episode sum of the environment's original rewards. Undiscounted. |
| `Rnd/mean_intrinsic_reward` | Rolling mean completed episode sum of the intrinsic RND rewards. Undiscounted. |
| `Rnd/weight` | Current RND intrinsic-reward weight. |

`Loss/rnd` is defined in the loss section. With RND, `Train/mean_reward` includes extrinsic plus intrinsic reward. The new return tracker records raw environment rewards before the algorithm adds intrinsic rewards, while its Q/V boundary can reflect the objective used to train the critic. Do not call that an entirely extrinsic-return estimate without considering the trained value tail.

### Media and saved files

- `EvalVideo/periodic`: optional PPO evaluation-video upload. It is a video, not a scalar. The video uses a saved checkpoint and a separate evaluation process.
- `video`: shared SAC writer's MP4 media upload. Again, not a training quantity.
- Checkpoint/model files and code-diff uploads are artifacts/files. `checkpoint_name` and `checkpoint_path` are provenance configuration fields, not reward samples.

### Shared runner fields that belong to race tasks

The training entry point also supports Solo12 race environments. These extra fields do not appear merely because we train two-feet. They are listed here so shared-runner keys are not mistaken for missing two-feet metrics. Race reward terms and gate telemetry belong to the race task's own environment implementation, not the two-feet reward table above.

| Exact key | Definition |
|---|---|
| `Episode/success_count` | Completed episodes that finished the race successfully, pooled by the PPO runner. |
| `Episode/completed_count` | Completed race episodes, pooled over the rollout. |
| `Episode/successRate` | Successful completed episodes divided by completed episodes. Omitted when the denominator is zero. |
| `Episode/finishRatio` | Alias of that completion/success fraction. |
| `Curriculum/backward_force_N` | Current backward perturbation force after a possible promotion, N. |
| `Curriculum/backward_force_rollout_N` | Backward force that produced the just-collected rollout, N. |
| `Curriculum/backward_force_stage` | Current force stage after a possible promotion. |
| `Curriculum/backward_force_stage_iterations` | Iterations completed in the current force stage. |
| `Curriculum/backward_force_eligible_success_count` | Successful endings from episodes that started and ended in the current force stage. |
| `Curriculum/backward_force_eligible_completed_count` | All endings eligible for that stage-pure success gate. |
| `Curriculum/backward_force_window_episodes` | Eligible completed episodes in the promotion window, which can extend backward to satisfy its sample floor. |
| `Curriculum/backward_force_window_successes` | Eligible successes in that window. |
| `Curriculum/backward_force_window_iterations` | Actual number of included rollout iterations. |
| `Curriculum/backward_force_success_rate` | Window successes divided by window episodes. |
| `Curriculum/backward_force_rollout_stage` | Stage used for the promotion decision's rollout. |
| `Curriculum/backward_force_rollout_stage_iterations` | Stage-local iteration count used for that decision. |
| `Curriculum/backward_force_promoted` | 0/1: the decision promoted the stage. |

## 15. W&B system metrics and bookkeeping

W&B collects these automatically, on a separate system-time sampling clock. They are not averaged over our RL minibatches. The exact available fields depend on the machine, SDK version, and driver. GPU-indexed fields can include **all GPUs on the node**, not only the one allocated to this run. Use allocation/config provenance before blaming a particular GPU curve on the policy.

The definitions and displayed units below follow the [W&B system metrics reference](https://docs.wandb.ai/models/ref/python/experiments/system-metrics) and the names observed in your runs. The SDK may show names without the leading `system.` in the System tab.

### NVIDIA GPU fields

Each suffix below expands to `system.gpu.<gpu_index>.<suffix>`. Some SDK versions also provide `system.gpu.process.<gpu_index>.<suffix>` for process-related monitoring.

| Suffix | Definition and units |
|---|---|
| `gpu` | GPU compute utilization, percent. |
| `memory` | GPU memory utilization/activity, percent. Not the same as the amount of VRAM allocated. |
| `memoryAllocated` | GPU memory allocated as a percent of total memory. |
| `memoryAllocatedBytes` | Allocated GPU memory, bytes. |
| `temp` | GPU temperature, degrees Celsius. |
| `powerWatts` | GPU power consumption, watts. |
| `powerPercent` | Power consumption as a percent of the GPU's power capacity/limit. |
| `enforcedPowerLimitWatts` | Enforced GPU power limit reported by the driver, watts. |
| `smClock` | Streaming-multiprocessor clock, MHz. These cores do the compute work. |
| `memoryClock` | GPU memory clock, MHz. |
| `graphicsClock` | Graphics clock, MHz; available on some SDK/driver combinations. |
| `correctedMemoryErrors` | Driver-reported corrected memory-error count. Hardware telemetry, not critic loss. |
| `uncorrectedMemoryErrors` | Driver-reported uncorrected memory-error count. Some documentation/versions use `unCorrectedMemoryErrors`. |
| `encoderUtilization` | Video encoder utilization, percent, when emitted. |

### CPU, RAM, disk, and network

| Exact key or pattern | Definition and displayed units |
|---|---|
| `system.cpu` | CPU utilization, percent. |
| `system.memory_percent` | Total host memory use, percent. |
| `system.proc.cpu.threads` | Number of threads in the training process. Not the Slurm CPU allocation. |
| `system.proc.memory.rssMB` | Process resident RAM, reported MB. Includes CPU-side replay/storage but is not identical to total cgroup memory usage. |
| `system.proc.memory.percent` | Process memory use as a percentage of host memory. |
| `system.proc.memory.availableMB` | Available **system** memory, reported MB, despite the `proc` namespace. |
| `system.disk.<path>.usageGB` | Used space on the monitored filesystem, reported GB. `<path>` can be `/`. |
| `system.disk.<path>.usagePercent` | Used filesystem space, percent. |
| `system.disk.<device>.in` | Device disk-read byte counter reported by the SDK. Not RL transitions or a reward term. |
| `system.disk.<device>.out` | Device disk-write byte counter reported by the SDK. |
| `system.network.recv` | Network bytes received as reported by the SDK. |
| `system.network.sent` | Network bytes sent as reported by the SDK. |

Check the SDK sampling/counter window before treating byte counters as bytes/second. Reported “MB” and “GB” labels come from the SDK; use the implementation's conversion when precise storage arithmetic matters.

### W&B internal fields

| Exact field | Meaning |
|---|---|
| `_step` | W&B history step/index. With our explicit logging it usually follows training iteration. It is not environment-interaction count. |
| `_timestamp` | Event timestamp in Unix seconds. |
| `_runtime` | Runtime seconds supplied by W&B for the event stream. Different from the trainer's timed `walltime`. |
| `_wandb.runtime` | W&B run runtime bookkeeping in the summary. Not an RL statistic. |
| `_wandb` | Internal W&B metadata object. Not a scalar performance quantity. |

W&B summaries usually retain the latest value of a history field unless a different summary reduction is configured. A summary's `CriticQ/mean`, for example, is not necessarily the mean Q over the entire run. W&B's optional visual chart smoothing is also separate from our explicitly logged `Smoothed*` fields.

## 16. Common mistakes and useful plots

### Mistakes to avoid

1. Treating `Train/mean_reward` or `Episode_Reward/total` as discounted G.
2. Treating positive `Episode_Reward/foot_contact` as a positive reward rather than a cost magnitude.
3. Assuming all `PerStepRewardRatio/*` values are success percentages.
4. Assuming `CriticQ/policy_mean` is measured on the most recent online trajectory or uses deterministic mean actions.
5. Using inverse symlog of `CriticDist/symlog_mean` as raw mean Q.
6. Calling categorical width a calibrated uncertainty band.
7. Treating the collision negative line or categorical support as a hard environment-return floor.
8. Comparing mean Q with completed mean G as if they use the same state/action cohort.
9. Dividing a network's median feature rank by its last hidden width.
10. Interpreting missing metrics as zero, or assuming updating source changes an already-running process.

### Useful groups of plots

- **Observed task performance:** `Train/mean_reward` or `Episode_Reward/total`, episode length, and relevant reward-term ratios. Keep the raw signed step costs visible.
- **Discounted return:** `Returns/actual_return` with its min/max, terminal/truncated split, and bootstrap contribution. A large tail means much of the plotted value is a critic estimate.
- **Critic scale:** `CriticQ/policy_mean`, `CriticQ/reduced_mean`, their positive-limit ratios, and `Returns/theoretical_return_lim_pos`. For PPO use `CriticV/mean` and its ratio.
- **Categorical support:** `CriticDist/edge_mass` with `Loss/critic_target_clipped_fraction`. A broad support does not make Q accurate, and low clipping does not prove calibration.
- **Optimization:** actor/critic losses, gradient norms before/after clipping, LR, alpha/noise, and MJX PPO's update acceptance.
- **Continued adaptability:** per-layer rank fractions and dormancy, whole-network norms, gradient kurtosis, and the separate local-redundancy probe. Keep architecture and probe settings matched.
- **Fine-tuning dynamics:** offline replay fraction, valid online entries, update count, performance, and the Q plots. The replay distribution changes as offline mixing anneals.

For real Q calibration, collect matched state/action predictions and returns under a clearly specified policy. These training means alone do not provide that matched test.

## 17. Source map and verification

The guide follows the actual emitted names and code, rather than assuming a name has conventional meaning. The relative source paths below are inside the IsaacLab repository.

| Topic | Main source |
|---|---|
| Reward formulas, ratios, reset cohorts, and curriculum | `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py` |
| Scalar defaults and phase schedules | `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env_cfg.py` |
| Shared return limits, episode accumulator, and Q/V tail | `source/rsl_rl_sac_vendor/rsl_rl_sac/utils/return_metrics.py` |
| SAC losses, update clocks, raw Q, clipping, and gradients | `source/rsl_rl_sac_vendor/rsl_rl_sac/algorithms/sac.py` |
| Categorical statistics, raw decoder, and PopArt | `source/rsl_rl_sac_vendor/rsl_rl_sac/models/sac_mlp_model.py` |
| SAC rolling means and writer reductions | `source/rsl_rl_sac_vendor/rsl_rl_sac/utils/logger.py` |
| Replay and interaction-budget schedule | `source/rsl_rl_sac_vendor/rsl_rl_sac/runners/off_policy_runner.py` |
| Isaac PPO callbacks and metric-name remapping | `source/scripts/rsl_rl/train.py` |
| Isaac PPO gradient-group measurements | `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/solo12_ppo.py` |
| MJX PPO rewards, smoothing, losses, and new G | `mujoco/train_lora.py` |
| MJX SAC environment bridge and episode fields | `mujoco/train_sac.py` |
| Plasticity activation and gradient formulas | `source/scripts/rsl_rl/plasticity_metrics.py` |
| Local-redundancy probe and measurement metadata | `source/scripts/rsl_rl/local_redundancy.py` |
| Intervention and distribution-shift counters | `source/scripts/rsl_rl/plasticity_mitigation.py`, `observation_permutation.py`, and `continual_backprop.py` |

Isaac PPO's standard `Loss/*`, `Policy/*`, `Perf/*`, and rolling `Train/*` fields also depend on the installed RSL-RL `algorithms/ppo.py` and `runners/on_policy_runner.py`. Recheck those when the package version changes.

Documentation checks compare the named fields/families against the source inventory, both example-run summaries, and saved smoke-run event tags. Worked arithmetic uses the effective examples above. The underlying return implementation was separately verified with 269 tests and bounded simulator smoke runs. No historical W&B metrics were rewritten to create this guide.

The guide must evolve when a logger, formula, namespace, filter, or reduction changes. Editing a document does not make older runs use the current definition; always keep the launch revision and effective configuration with a comparison.
