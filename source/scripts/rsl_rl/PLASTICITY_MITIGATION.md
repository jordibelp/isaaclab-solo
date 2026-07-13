# Plasticity mitigation strategies

`train.py` implements the interventions from Juliani and Ash, [*A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning*](https://arxiv.org/abs/2405.19153), behind one composable argument. The formulas and defaults were cross-checked against the authors' [reference implementation at commit `75f0d5a`](https://github.com/awjuliani/deep-rl-plasticity/tree/75f0d5af374e3b429316765928351234384ef261).

```bash
--plasticity-mitigation-strategy=<strategy>
```

## Strategies

| Strategy | Timing | Paper-default update |
|---|---|---|
| `none` | — | Baseline warm start |
| `layernorm` | Architecture, from initialization | LayerNorm before every actor/critic hidden activation |
| `regenerative-l2` | Every PPO gradient update | Add `0.01 * ||theta - theta_init||_2` |
| `regenerative-l2-layernorm` | Both above | Regenerative L2 + LayerNorm |
| `shrink-perturb` | Every observation-permutation boundary | `theta <- 0.5 theta + 0.5 theta_fresh`; clear Adam state |
| `soft-shrink-perturb` | After every PPO optimizer step | `theta <- 0.999999 theta + 0.000001 theta_fresh` |
| `soft-shrink-perturb-layernorm` | Both above | Soft shrink+perturb + LayerNorm |

`theta_fresh` is independently sampled from the default initialization distribution for every Linear layer. LayerNorm affine parameters use their fresh identity initialization (gain one, bias zero).

The intervention target is all actor/critic layer weights and biases. The separate continuous-action `std`/`log_std` parameter and empirical observation-normalizer buffers are deliberately excluded so the method does not silently become an exploration or input-statistics intervention. Boundary shrink+perturb still clears the complete PPO Adam state, matching the paper's fresh-optimizer behavior at a shift.

## Commands

Append exactly one strategy to the existing permutation experiment command:

```bash
--plasticity-loss-exp \
--plasticity-mitigation-strategy=soft-shrink-perturb-layernorm
```

Run the requested conditions separately:

```bash
# LayerNorm only
--plasticity-mitigation-strategy=layernorm

# Regenerative regularization
--plasticity-mitigation-strategy=regenerative-l2

# Regenerative regularization + LayerNorm
--plasticity-mitigation-strategy=regenerative-l2-layernorm

# Boundary shrink+perturb
--plasticity-mitigation-strategy=shrink-perturb

# Continuous/soft shrink+perturb
--plasticity-mitigation-strategy=soft-shrink-perturb

# Continuous/soft shrink+perturb + LayerNorm
--plasticity-mitigation-strategy=soft-shrink-perturb-layernorm
```

LayerNorm and the continuous strategies can also be used in ordinary Solo PPO runs without `--plasticity-loss-exp`. Boundary `shrink-perturb` requires the permutation experiment because its intervention point is the distribution-shift boundary.

Paper defaults can be swept explicitly:

```bash
--plasticity-regen-l2-coef=0.01
--plasticity-soft-sp-beta=1e-6
--plasticity-sp-beta=0.5
--plasticity-mitigation-seed=1618075
```

For both shrink+perturb variants, `alpha = 1 - beta`.

## Controlled-condition rules

The strategy flag is mutually exclusive with:

- `--use-cbp`
- `--plasticity-loss-exp-reset-all`
- `--plasticity-exp-first-layer-only`
- `--shared_networks`
- distributed training

These are separate experimental conditions. Combining them would make attribution ambiguous or require explicit synchronization/topology rules.

Set `agent.weight_decay=0.0` for paper-style comparisons. A nonzero value is allowed but logged as a warning because it adds ordinary AdamW/L2 decay as another intervention.

The current Solo policy retains its configured ELU activation. LayerNorm is inserted immediately before each ELU, matching the paper's placement before the activation without changing the activation family as an additional confound.

## Logging and checkpoints

TensorBoard/W&B metrics are under `PlasticityMitigation/*`, including:

- intervention enable flags and hyperparameters;
- target parameter count/fraction and L2 norm;
- regenerative distance and penalty;
- optimizer-step and soft-perturbation counts;
- boundary events/counts and perturbation magnitude;
- number of inserted LayerNorm layers.

Each run also writes `params/plasticity_mitigation.yaml`.

Mitigation checkpoints store `plasticity_mitigation_state_dict`, containing the regenerative initialization reference, dedicated shrink+perturb RNG state, and event counters. Resume with the same strategy and hyperparameters. LayerNorm preserves the existing Linear keys but adds LayerNorm keys, so a LayerNorm checkpoint must also be resumed with its LayerNorm-containing strategy flag. To initialize a new LayerNorm run from a non-LayerNorm checkpoint, use `--reuse-mlp --checkpoint=<path>`; this imports compatible Linear weights while deliberately starting a new optimizer and iteration schedule.

## Implementation files

- `plasticity_mitigation.py`: strategy composition, LayerNorm insertion, exact losses/updates, RNG, checkpoint state.
- `train.py`: CLI, lifecycle integration, boundary events, logging, save/resume.
- `tests/test_plasticity_mitigation.py`: formula, timing, optimizer-reset, LayerNorm, target-scope, and exact-resume tests.
