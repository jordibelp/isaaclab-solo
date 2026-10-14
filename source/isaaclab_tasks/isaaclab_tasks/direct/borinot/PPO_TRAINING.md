# Borinot RSL-RL PPO: original task and goalkeeper

This setup uses the same shared training entry point as Solo12:

```text
source/scripts/rsl_rl/train.py
```

Both tasks use `--agent=rsl_rl_cfg_entry_point`. The original task keeps its existing SKRL registration too. SAC is not added in this change.

## Recipe matched to the reference run

The reference is [Solo12 run 8958tyzt](https://wandb.ai/jordibelp/borinotIsaacLab/runs/8958tyzt). Values below come from its saved configuration, not just its command or run name. Despite the word SAC in that run's name, it used PPO.

| Setting | Borinot default |
|---|---|
| Parallel environments in the cluster command | 10,000 |
| `agent.num_steps_per_env` | 24 |
| `agent.algorithm.num_mini_batches` | 4 |
| `agent.algorithm.num_learning_epochs` | 5 |
| Actor and critic hidden layers | `[512, 256, 128]`, ELU |
| Actor / critic observation normalization | Both enabled |
| Initial action standard deviation | 1.0, with learned log standard deviation |
| Initial learning rate / schedule | 0.001 / adaptive |
| PPO clip / value-loss coefficient | 0.2 / 1.0 |
| Entropy coefficient / GAE lambda | 0.005 / 0.95 |
| Gamma | **0.99**, increased from reference 0.97 |
| KL target / maximum gradient norm | 0.01 / 1.0 |
| Optimizer / Adam betas / weight decay | AdamW / `(0.9, 0.95)` / 0.001 |
| Forward compilation | Enabled |
| Local redundancy | Enabled, interval 50, diagnostic microbatch 64, 4096 probe samples |
| Maximum iterations / periodic save interval | 10,000 / 469 |
| W&B project | `borinotIsaacLab` |

The environment count is selected by `--num_envs`, not by the agent. The task's default scene count remains unchanged, so ordinary previews do not start with 10,000 robots.

### What batch size means

Each update collects `10,000 environments * 24 steps = 240,000 transitions`. Four minibatches give `240,000 / 4 = 60,000 transitions per minibatch`. Five epochs give `5 * 4 = 20 optimizer steps per update`.

These are the reference run's **unaugmented** sizes. Solo's symmetry augmentation adds transformed examples. Borinot does not yet have a verified symmetry map, so the commands use `--symmetry-mode=none`. We do not copy the quadruped's transformation. The effective augmented Solo batch is therefore larger than Borinot's batch.

`agent.local_redundancy.batch_size=64` is unrelated to the PPO minibatch. It is the microbatch used by a read-only network diagnostic.

The original Borinot task runs its policy at 60 Hz. The goalkeeper defaults to 100 Hz. Matching transitions per update does not match simulated seconds per update: 24 steps span 0.4 s and 0.24 s, respectively. Gamma is also per policy step. The new 0.99 value is a starting choice, not a claim of identical real-time discounting.

## Cluster launch

Pull the local code commit into the cluster checkout before using these entry points. This setup does not copy code or submit a job on IRICluster. Activate its existing conda environment:

```bash
source /home/jbeltran/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && cd /home/jbeltran/IsaacLab
```

### Original Borinot

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Direct-v0 --agent=rsl_rl_cfg_entry_point --headless --num_envs=10000 --max_iterations=10000 --seed=230901 --symmetry-mode=none '--run-name=[Cluster]-Borinot PPO - run 1/2' agent.num_steps_per_env=24 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 agent.algorithm.value_loss_coef=1.0 agent.algorithm.entropy_coef=0.005 agent.algorithm.learning_rate=1.0e-3 agent.algorithm.gamma=0.99 agent.algorithm.max_grad_norm=1.0 'agent.policy.actor_hidden_dims=[512,256,128]' 'agent.policy.critic_hidden_dims=[512,256,128]' agent.policy.actor_obs_normalization=True agent.policy.critic_obs_normalization=True agent.local_redundancy.interval=50 agent.local_redundancy.batch_size=64 agent.adam_beta2=0.95 agent.weight_decay=0.001 agent.optimizer=adamW agent.algorithm.torch_compile=True
```

This uses the original environment and its current action semantics: six individual thrusts and four numbers encoding two PD angle targets. It does not substitute the goalkeeper environment. Its asset paths now resolve under the canonical `source/` layout. Rewards, timing, actuator settings, and the original SKRL recipe are unchanged.

### Goalkeeper, default thrust-and-torque mode

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Goalkeeper-Direct-v0 --agent=rsl_rl_cfg_entry_point --headless --num_envs=10000 --max_iterations=10000 --seed=230901 --symmetry-mode=none '--run-name=[Cluster]-Borinot goalkeeper PPO - torque - run 1/2' env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 env.effort_lim_sim=2.7 'env.robot_home=[0.0,0.0,1.5]' agent.num_steps_per_env=24 agent.algorithm.num_learning_epochs=5 agent.algorithm.num_mini_batches=4 agent.algorithm.value_loss_coef=1.0 agent.algorithm.entropy_coef=0.005 agent.algorithm.learning_rate=1.0e-3 agent.algorithm.gamma=0.99 agent.algorithm.max_grad_norm=1.0 'agent.policy.actor_hidden_dims=[512,256,128]' 'agent.policy.critic_hidden_dims=[512,256,128]' agent.policy.actor_obs_normalization=True agent.policy.critic_obs_normalization=True agent.local_redundancy.interval=50 agent.local_redundancy.batch_size=64 agent.adam_beta2=0.95 agent.weight_decay=0.001 agent.optimizer=adamW agent.algorithm.torch_compile=True
```

The example explicitly selects home z=1.5 m, as in the integration checks. The source still preserves Jordi's tuned z=0.5 m, which is below the current allowed minimum of 0.55 m. For a lower-home experiment, align the allowed area and ball target deliberately. Do not treat an immediate area reset as a PPO failure.

For the other control variants, replace only the mode value with `thrust_and_PD_joint_pos_control` or `PI_rotors_PD_joint_pos_control`. Match the mode, rates, and actuator settings when loading checkpoints. Equal action shapes do not make torque and position checkpoints interchangeable.

## Requested flags and training features

All these overrides are available for **both** tasks:

```bash
agent.policy.actor_obs_normalization=True agent.policy.critic_obs_normalization=True agent.local_redundancy.interval=50 agent.local_redundancy.batch_size=64 agent.adam_beta2=0.95 agent.weight_decay=0.001 agent.optimizer=adamW agent.algorithm.torch_compile=True
```

The shared training script applies optimizer selection, betas, and weight decay to the live optimizer. It keeps action-noise parameters in a separate decay-free group, as in Solo. The PPO wrapper is still named `Solo12PPO`, but its code is task-independent: it adds compiled actor/critic forwards and gradient-norm diagnostics. It does not change Borinot into a Solo environment. Compilation does not include the simulator or the full optimizer/update loop.

Use **`source/scripts/rsl_rl/train.py`** for the complete recipe. The upstream `scripts/reinforcement_learning/rsl_rl/train.py` does not apply this custom optimizer/diagnostic setup merely because the fields exist in the config.

The shared entry point also supplies:

- W&B logging and local TensorBoard records, with serialized effective environment and agent configs.
- Environment, agent-config inheritance, PPO implementation, and training-helper source snapshots.
- Separate model/environment RNG streams and recorded seeds.
- Actor/critic gradient norms, plasticity metrics, and local-redundancy probes.
- Periodic checkpoints, best-return checkpoints, and resume handling.

The existing periodic inference-video and symmetry features are task-specific to Solo tasks. They are not advertised as Borinot features. Checkpoint retention follows the shared script's `--keep-last-checkpoints` option; use `0` to keep every periodic checkpoint.

For a local integration check, use `--num_envs=32 --max_iterations=2 --logger=tensorboard`, and reduce only the diagnostic probe cost with `agent.local_redundancy.num_samples=4 agent.local_redundancy.batch_size=2 agent.local_redundancy.interval=1`. Keep the 24-step rollout, four minibatches, and five PPO epochs. Add the standard local headless cache flags when launching Isaac Sim.

## Evidence boundary

Verification passed eight CPU checks, four compiled training cases (original Borinot plus all three goalkeeper modes), and four 30-step checkpoint inference checks. Each training case used 32 environments, 24 rollout steps, and two iterations: `32 * 24 * 2 = 1536 transitions`. Each checkpoint contained 40 optimizer steps. Numeric TensorBoard output contained 18 finite gradient/local-redundancy signals per case. AdamW resume continued from 40 to 60 optimizer steps.

The offline W&B check retained the primary goalkeeper agent config, its inherited Borinot config, and the shared PPO implementation as separate source files. A regression check covers the duplicate-basename collision that this inheritance exposed. No online W&B run was uploaded.

Local verification checks the actual shared entry point, live optimizer settings, compiled updates, numeric diagnostic outputs, saved configs, offline W&B snapshots, and checkpoint inference/resume. An offline W&B test does not upload a run. Small local runs are integration proof, not learned performance or a 10,000-environment cluster benchmark. The cluster's older simulator and PyTorch installation still need their own launch check after you pull the code.
