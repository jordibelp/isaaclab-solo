# Solo12 Random Network Distillation

Solo12 RSL-RL PPO runs can add an RND exploration bonus with two environment overrides:

```bash
env.rnd_network=True env.beta_curiosity=1.0
```

Both defaults are disabled (`rnd_network=False`, `beta_curiosity=0.0`), so existing runs are unchanged.

## Curiosity state

RND receives a clean 12-dimensional simulator state rather than the noisy policy observation:

- FL foot-tip position in the base frame (3)
- FR foot-tip position in the base frame (3)
- RL foot-tip position in the base frame (3)
- RR foot-tip position in the base frame (3)

The positions use the physical foot tips, including the fixed calf-to-foot offsets, rather than the calf link origins. Commands, observation corruption, velocities, actions, contacts, base orientation, and absolute world position are excluded. This focuses novelty directly on front-hand/rear-foot configurations and avoids rewarding command resampling, sensor noise, or simply traveling farther through the world. It follows the curiosity-state idea in Katayama et al., [Learning Bipedal Locomotion on Gear-Driven Humanoid Robot Using Foot-Mounted IMUs](https://arxiv.org/abs/2504.00614), generalized from two feet to Solo12's four end effectors.

The target has one 5-unit hidden layer; the predictor has two 5-unit hidden layers; both produce one output. The curiosity state is empirically normalized, while the intrinsic reward is not normalized so it can decay as states become familiar. This follows the robotics-focused design in Schwarke et al., [Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks](https://proceedings.mlr.press/v229/schwarke23a.html).

## Beta

Start with:

```bash
env.beta_curiosity=1.0
```

For an ablation, use `0.5`, `1.0`, and `2.0`. RSL-RL multiplies this value by the environment step time; at Solo12's `step_dt=0.02`, beta `1.0` appears as `Rnd/weight=0.02`.

Watch these metrics:

- `Rnd/mean_intrinsic_reward`
- `Rnd/mean_extrinsic_reward`
- `Rnd/weight`
- `Loss/rnd`
- `Train/mean_reward` (extrinsic + intrinsic)

`Episode_Reward/total` remains the environment/extrinsic return. RND is added inside PPO afterward.

RND promotes novelty, not the desired gait directly. It can initially reward falling or new front-foot contact modes too, so keep the base-collision penalty and the task rewards/penalties active. Reduce beta if intrinsic return dominates the positive task-reward components or remains large after the policy has found a useful gait.

RND model, optimizer, and state-normalizer parameters are saved in RSL-RL checkpoints and restored on resume.
Checkpoints containing the previous 19-dimensional RND model are not shape-compatible with the new 12-dimensional
curiosity state; start a new run (or resume a pre-RND policy checkpoint, which initializes RND from scratch).
