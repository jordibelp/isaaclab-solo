# Solo12 SAC: Isaac pretraining and MJX fine-tuning

The SAC implementation follows Sabatini, Li, and Hutter,
“Bridging the Gap: Enabling Soft Actor-Critic for High Performance Legged
Locomotion” (arXiv:2605.24975).

It includes:

- asymmetric per-joint action bounds derived from Solo12 joint limits, the
  actual action center, and action scale;
- actor mean initialized near zero and initial standard deviation `0.15`;
- pre-reset observations for timeout bootstrapping;
- replay-buffer n-step returns (default `n=5`);
- twin Q critics, automatic entropy tuning, RND, symmetry augmentation/loss,
  W&B logging, checkpointing, and parallel environments.

## Isaac/PhysX pretraining

Use the existing training script and select the SAC agent entry point:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task="solo12-two-feet" \
  --agent=rsl_rl_sac_cfg_entry_point \
  --run-name="Solo12 two-feet SAC" \
  --symmetry-mode=augmentation \
  --headless \
  --num_envs=10000 \
  --max_iterations=1500 \
  env.curriculum_two_feet=False \
  env.initial_position=safe
```

All existing `env.*` overrides can be appended as with PPO. Paper defaults are
in `source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/rsl_rl_sac_cfg.py`.
Hydra can override them, for example:

```bash
agent.algorithm.mini_batch_size=8192 \
agent.algorithm.num_mini_batches=200 \
agent.algorithm.n_steps=5
```

## Parallel MJX fine-tuning

Start from an Isaac SAC checkpoint:

```bash
./isaaclab.sh -p mujoco/train_sac.py \
  --task="solo12-two-feet" \
  --checkpoint="/absolute/path/to/isaac_sac/model_1500.pt" \
  --run-name="[cluster] MuJoCo SAC fine-tune" \
  --symmetry-mode=augmentation \
  --headless \
  --num_envs=256 \
  --max-iterations=2000 \
  env.curriculum_two_feet=False \
  env.initial_position=safe \
  env.tricky_terrain=False \
  env.include_events_randomization=False \
  "env.forces_applied_to_base_curriculum=[0.0]" \
  "env.base_push_force_z_range=[0.0,0.0]"
```

By default, `--checkpoint` transfers actor and twin critics but starts fresh
optimizers, entropy state, iteration count, and replay buffer. Add `--resume`
for an exact optimizer/iteration resume. Replay data is not serialized by the
released RSL-RL-SAC implementation, so it always starts empty.

The first MJX iteration compiles the vectorized physics graph and can take a
few minutes. Later iterations reuse the compiled graph.

## Checkpoint compatibility

SAC and PPO checkpoints are intentionally different. `train_sac.py` accepts
RSL-RL-SAC checkpoints, not PPO/LoRA PPO checkpoints. PPO training remains on
the existing RSL-RL 3.1.2 path and is not migrated or altered by SAC.
