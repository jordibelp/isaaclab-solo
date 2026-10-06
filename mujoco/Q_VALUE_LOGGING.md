# Raw Q logs during SAC fine-tuning

The shared SAC trainer now logs expected Q values in raw reward units. This works
in MuJoCo fine-tuning and Isaac pretraining. No extra command-line flag is needed.
Both full-weight and LoRA fine-tuning use the same logs.

## What to plot

| W&B / TensorBoard key | Meaning |
| --- | --- |
| `CriticQ/q1_mean` | First critic's mean Q for the sampled replay actions. |
| `CriticQ/q2_mean` | Second critic's mean Q for the sampled replay actions. |
| `CriticQ/mean` | Average of the two critics, then averaged across the replay batch. |
| `CriticQ/reduced_mean` | Apply the configured twin-critic rule to each replay sample, then average. |
| `CriticQ/policy_mean` | Q for actions sampled from the current policy at replay states, using the actor's twin-critic rule. |

Start with `CriticQ/policy_mean` to see how valuable the critic thinks the current
policy's actions are. Compare it with `CriticQ/mean` to see how it scores actions
stored in replay. If replay mixes old offline data with new online data, these
logs describe that mixed state distribution, not only the latest rollout.

The `min` rule takes the smaller Q **for each sample**. It does not take the
smaller of the two batch means. Both `mean` and `mean_pi_q_none` use the average
for these logs. In `mean_pi_q_none`, each critic still has its own Bellman target;
the new logging does not combine those targets or change training.

## Actual bin values, not symlog coordinates

For a categorical critic, each critic's Q is `sum(probability_i * bin_value_i)`.
We use the model's existing Q decoder and its saved `value_support` tensor.

- Symexp support: bin values are `sign(x) * (exp(abs(x)) - 1)`.
- Linear support: bin values are already in raw reward units, such as -20 to +20.

We do **not** apply symexp to `CriticDist/symlog_mean`. That generally gives a
different answer. Existing `CriticDist/*` metrics keep their original meanings.
Scalar and PopArt critics also log raw Q through their existing decoders.

MuJoCo reconstructs linear checkpoint support from the saved atoms, even without
a config sidecar. A linear-trained policy therefore keeps its original bins.
The strict support check during checkpoint loading remains enabled.

## Timing and interpretation

Replay-action Q uses the critic outputs already computed before each critic
optimizer step. Each logged point averages the batch means over that iteration's
critic updates. Current-policy Q uses the value already computed after the critic
step, before the actor step. It averages over actual actor updates and is absent
when an iteration has no actor update. Logging adds no extra policy samples or
network forward passes.

These are predicted **discounted SAC values**, including the future entropy term
used in the backup. They are not the undiscounted episode reward and do not prove
that the predictions are accurate. Keep plotting measured episode rewards too.

The new logs appear in runs launched with the updated code. Editing local files
does not update an existing training process or backfill earlier W&B history.

## Verification

- 348 regression tests passed. They cover raw-unit decoding on CPU and CUDA,
  exact support reconstruction, logger delivery, and compiled/eager updates.
- In 24 matched old/new update cases, existing metrics, model weights, target
  weights, optimizer state, and RNG state were bit-identical. These cases cover
  CPU/CUDA, both support types, all three twin-critic rules, and full/LoRA tuning.
