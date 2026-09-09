# Robust race policy with an optional privileged critic

Keep `--task=Isaac-Solo12-Race-Direct-v0` and add this **single Hydra override** to the usual PPO training command:

```bash
agent.policy.asymmetric_actor_critic=True
```

Omit it, or set it to `False`, for the existing symmetric baseline. No new task, argparse flag, actor history, or teacher checkpoint is required. Keep the rest of the experiment command unchanged for the ablation.

## Architecture

| Branch | Input | Network |
|---|---|---|
| Actor | Current race observations only (normally 63D) | Existing actor MLP → 12 actions |
| Critic | Same current observation + 12 foot-force components + 4 static-friction coefficients (normally 79D) | Normalize full input; encode the 16 privileged features with `16 → 64 → 32 → 8` ELU MLP; concatenate the latent with the 63D current observation; critic MLP → value |

The critic uses the **same RSL-RL `MLP` encoder implementation, architecture, GT feature ordering, and normalization-before-encoding convention** as `Solo12-Race-ParamsConditionedEnc-Direct-v0`. Only the critic has an encoder. Its encoder and value head are trained jointly from scratch by PPO, not initialized from or frozen to a teacher checkpoint.

The policy switch automatically:

- Enables `env.asymmetric_actor_critic`.
- Refreshes environment observation/state dimensions after Hydra overrides.
- Routes actor observations to `policy` and critic observations to `critic`.
- Adds `_asymmetric-critic` to the training run directory/name suffix.
- Saves the resolved settings in the run's `params/agent.yaml` and `params/env.yaml`.

Do **not** enable `env.include_forces_to_gt_obs` or `env.include_mu_coefs_to_gt_obs`: those flags expose privileged inputs to the actor. The robust ablation rejects those combinations, actor history, and `agent.policy.shared_networks=True`.

Existing teacher and TCN task behavior is unchanged. The critic's friction visibility still obeys `env.teachers_sees_future_friction_coef`: `False` means per-foot contact-latched coefficients; `True` means the coefficient below each foot is available before touchdown. The current proprioceptive part retains the actor's configured observation corruption.

## Encoder and head overrides

The teacher-matching defaults are:

```bash
'agent.policy.env_params_encoder_hidden_dims=[64,32]' agent.policy.env_params_latent_dim=8 agent.policy.env_params_encoder_activation=elu
```

Keep those defaults to match the teacher's default encoder, or match any encoder overrides used for your particular teacher experiment. The actor/critic head widths remain independently configurable as usual:

```bash
'agent.policy.actor_hidden_dims=[512,256,128,64]' 'agent.policy.critic_hidden_dims=[512,256,128,64]'
```

`env_params_dim` must remain 16 for this task's privileged force/friction layout. No friction-only variant is introduced.

## Checkpoints and inference

- Resume training with the same task and `agent.policy.asymmetric_actor_critic=True`, the same architecture settings, and the normal `--checkpoint` option. Matching checkpoints restore actor, critic, encoder, normalization and optimizer state.
- `play_direct_race_0423.py` and `solo_race_eval.py` detect a robust critic encoder in a checkpoint and restore its widths and observation routing before creating the environment. The new switch need not be repeated for ordinary playback/evaluation of this ablation. As with other policy activations, repeat any **nondefault activation** overrides; activation functions cannot be inferred from weight tensors.
- Deployment still needs only the actor and its 63D observation normalizer; `act_inference()` does not require privileged observations or the critic. There is no new inference-time encoder.

## Verification

Focused tests cover actor independence from privileged inputs, no actor gradients from the value loss, bit-identical teacher/new-critic outputs after copying critic state, unchanged actor initialization, exact checkpoint/optimizer continuation, Hydra configuration routing and default/teacher preservation.

For a quick local CPU policy regression check in `env_isaaclab`:

```bash
python -m pytest source/isaaclab_tasks/test/direct/solo12/test_race_robust_privileged_critic.py source/isaaclab_tasks/test/direct/solo12/test_env_params_conditioned_encoder_actor.py source/isaaclab_tasks/test/direct/solo12/test_race_asymmetric_tcn_actor_critic.py -q
```

The environment/configuration checks in `test_race_asymmetric_observations.py` require Isaac Sim startup. Activate `env_isaaclab` and the usual CA-certificate exports before using `./isaaclab.sh`.
