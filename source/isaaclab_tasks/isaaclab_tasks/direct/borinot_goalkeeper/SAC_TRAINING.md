# Borinot goalkeeper SAC

Both SAC agents use the same recipe as Jordi's supplied Solo12 SAC command.
Only the critic loss changes between the two agents.
This integration is for `Borinot-Goalkeeper-Direct-v0`.
PPO keeps its action mapping. Goalkeeper PD defaults are now `kp=9.0` and `kd=0.2` for every agent.
The original `Borinot-Direct-v0` task is unchanged.

## Train

Use the canonical checkout at `/home/jordibelp/IsaacLab`.
On piuMachine, activate `env_isaaclab` and export the normal Isaac Sim CA certificate variables first.

**CE agent: 101 evenly spaced atoms from -20 to +20, in raw reward units.**
The loss is two-hot cross-entropy on a scalar Bellman target, not a C51 distributional backup.
There is no new symexp implementation. The existing shared linear support is reused.

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Goalkeeper-Direct-v0 --agent=rsl_rl_sac_cfg_entry_point --headless --num_envs=10000 --max_iterations=4500 --seed=230999 --symmetry-mode=none '--run-name=[Cluster]-SAC CE: heavier balls (200g max)' env.control_mode=thrust_and_PD_joint_pos_control env.physics_hz=200 env.policy_hz=100 'env.robot_home=[0.0,0.0,1.5]' env.rwd_scale_energy=0.0 env.enabled_self_collisions=True 'env.ball_mass_range_kg=[0.050,0.200]' env.thrust_to_weight=3.5
```

**MSE agent: ordinary scalar twin critics.**

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Goalkeeper-Direct-v0 --agent=rsl_rl_sac_mse_cfg_entry_point --headless --num_envs=10000 --max_iterations=4500 --seed=230999 --symmetry-mode=none '--run-name=[Cluster]-SAC MSE: heavier balls (200g max)' env.control_mode=thrust_and_PD_joint_pos_control env.physics_hz=200 env.policy_hz=100 'env.robot_home=[0.0,0.0,1.5]' env.rwd_scale_energy=0.0 env.enabled_self_collisions=True 'env.ball_mass_range_kg=[0.050,0.200]' env.thrust_to_weight=3.5
```

Both SAC agents support `thrust_and_PD_joint_pos_control` and `thrust_and_torque_control`.
The commands above select position targets. To train with joint torques instead, replace the mode with:

```bash
env.control_mode=thrust_and_torque_control env.effort_lim_sim=2.5
```

This example gives both joints a torque range of -2.5 to +2.5 N·m.
Without the effort override, the default range remains -2.7 to +2.7 N·m.
The SAC config still rejects the PI rotor mode.
It enables the SAC action mapping before the environment and its spaces are created.
The effective environment YAML records this choice.

These are fresh-training commands. Do not use a PPO checkpoint as an SAC resume checkpoint.
Use the same SAC entry point and support settings when resuming an SAC checkpoint.
The cluster must have this local source revision before using these commands.
No cluster copy, push, or training submission is part of this change.

## Action meanings

The action order is six independent rotors, then thigh, then calf.
Here `u` means the normalized tanh output, not the scaled action stored in replay for a joint.

### Position control

| Output | u = -1 | u = 0 | u = +1 | Physical stops |
| --- | ---: | ---: | ---: | --- |
| Thigh target | -75 degrees | 0 degrees | +75 degrees | [-90, +90] degrees |
| Calf target | -200 degrees | 0 degrees | +200 degrees | [-160, +160] degrees |
| Each rotor thrust command | 0 | hover / 6 | 3.5 times hover / 6 | nonnegative thrust |

The joint targets interpolate linearly between the soft endpoints.
The actor's affine action transform compensates for `joint_target_scale` and the configured offset.
Thus changing that legacy scale does not change the target endpoints.
The target midpoint remains the midpoint of the soft range, even for a nonzero offset.

The physical stops are written to the live PhysX articulation.
The calf target range intentionally extends outside them, as requested.
An out-of-range target can command PD effort against a stop; it does not move the physical stop.
The actuator effort and velocity limits remain in force.
The source USD is unchanged.
The task keeps a separate SAC target-range tensor; IsaacLab's generic `soft_joint_pos_limits`
still describes a range derived from the physical stops and is not the SAC target support.

### Torque control

For `thrust_and_torque_control`, all eight actor outputs stay in [-1, +1].
The last two outputs independently request thigh and calf torque:

| Output | u = -1 | u = 0 | u = +1 |
| --- | ---: | ---: | ---: |
| Thigh torque | -effort_lim_sim | 0 | +effort_lim_sim |
| Calf torque | -effort_lim_sim | 0 | +effort_lim_sim |

Torque is in N·m. Each requested torque is `u * effort_lim_sim`.
The scalar effort setting applies to both joints and also sets their PhysX effort limit.
PD stiffness and damping are zero in torque mode; angle-target scales and offsets do not scale the torque.
As before, the explicit `hold_arm=True` override replaces joint actions with neutral-position PD.
Leave it False for policy torque control.
The configured physical joint stops still apply.

### Rotor control in both modes

Thrust uses two linear segments around hover.
Below zero: `F / F_hover_per_rotor = 1 + u`.
Above zero: `F / F_hover_per_rotor = 1 + (thrust_to_weight - 1) * u`.
A single affine line cannot hit all three requested points when maximum thrust is 3.5 times hover.
The hover value uses the spawned robot's total mass, including the paddle, times 9.81, divided by six.
The existing motor lag remains active, so the table describes the command, not immediate applied thrust.
The SAC rotor bounds do not depend on PPO's `rotor_action_scale` or action clamp.

### Joint settings

Set position-mode PD gains with the same names as Solo12:

```bash
env.kp=9.0 env.kd=0.2
```

These are the new defaults: Kp is in N·m/rad and Kd is in N·m·s/rad.
The old `joint_stiffness` and `joint_damping` config names have been removed.
To replay a position policy trained with the previous defaults, explicitly set `env.kp=35.0 env.kd=0.6`.

The endpoint and physical-limit config values are in degrees:

```text
env.joint_physical_limit_thigh=[-90.0,90.0]
env.joint_physical_limit_calf=[-160.0,160.0]
env.joint_soft_limit_thigh=[-75.0,75.0]
env.joint_soft_limit_calf=[-200.0,200.0]
```

The optional offset follows Solo12's initialization semantics:

```text
env.sac_joint_pos_offset=[0.0,0.0]
env.sac_q_offset_init_actions=False
```

The offset is in radians, in thigh/calf order.
With the flag False, the initial mean-head bias is zero: zero latent output gives the soft midpoint.
With True, an inverse-tanh bias makes the initial deterministic target close to the offset.
Small random mean-head weights remain, so this is an initial center, not an exact constant target.
The flag does not permanently remap `u=0` to the offset and does not change the endpoints.
The offset must lie strictly inside both physical and target ranges.
With the default symmetric ranges and zero offset, both initializations are equivalent.

## Shared defaults

| Setting | Value |
| --- | --- |
| Actor and both critics | 1024, 512, 256 hidden units; Swish |
| Observation normalization | Actor and critic enabled; separate policy and privileged critic inputs |
| LayerNorm | Critic on; actor off |
| BatchNorm / weight normalization | Off |
| Actor noise | State-dependent; initial std 0.15; log_std bounded to [-20, 2] |
| Actor / critic optimizer | AdamW, betas (0.9, 0.95), weight decay 0.001 |
| Temperature optimizer | AdamW, betas (0.9, 0.999), no weight decay |
| Actor / critic / temperature LR | 0.0002 / 0.0002 / 0.00002 |
| Gamma / tau / n-step | 0.99 / 0.003 / 5 (Solo uses gamma 0.97) |
| Initial alpha / automatic alpha / entropy scale | 0.001 / on / 0.166 |
| Twin reduction | `mean_pi_q_none`: actor uses mean Q; each critic bootstraps from its own target |
| Collection per iteration | 24 steps per environment |
| Replay capacity | 5,000,000 transitions |
| Updates | 1 epoch, 400 mini-batches of 8192 |
| Warm-up | 1 collection iteration |
| Gradient norm limit | 1.0 |
| SAC update compilation | On |
| Local redundancy | Every 50 iterations; microbatch 64 |
| Replay snapshots | Off; interval 250 if enabled |

At 10,000 environments, one collection iteration adds 240,000 transitions.
The mini-batch-count UTD is `400 / 240000 = 1/600`.
Each collected transition is sampled about `400 * 8192 / 240000 = 13.65` times per update iteration on average.
These are two different ratios; neither includes the initial warm-up iteration.

The CE support can be changed with `agent.critic.distributional_num_bins`
and `agent.critic.distributional_linear_limit`.
The bin count must be odd. A positive linear limit takes precedence over the shared symlog field.
Keep the MSE entry's linear limit at zero: its scalar head has no bin support.
Use the canonical `critic.distributional_loss` setting; do not add the deprecated
`agent.distributional_critic_ce=True` flag to an already categorical config.

## Playback

Use the existing match player with the matching agent:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --agent=rsl_rl_sac_cfg_entry_point --checkpoint=/absolute/path/model_4499.pt --num_envs=1 env.control_mode=thrust_and_PD_joint_pos_control env.enabled_self_collisions=True 'env.ball_mass_range_kg=[0.050,0.200]' env.thrust_to_weight=3.5
```

For MSE, replace the agent with `rsl_rl_sac_mse_cfg_entry_point`.
For a torque-trained policy, select `env.control_mode=thrust_and_torque_control` and match `env.effort_lim_sim`.
Position and torque policies have the same action count but different meanings; do not interchange them.
Match any training changes to network sizes, observations, joint bounds, robot geometry, control rates, and support.
Playback restores networks and normalizers, but not optimizers.
It allocates only a minimal unused replay buffer.
The match panel, recent balls, cameras, and deterministic policy path are shared with PPO.
Evaluation holds the curriculum fixed by default, as before.

## Verification

Evidence is in the workspace's `artifacts/2026-10-16-borinot-sac/` folder.
Tests use small environment counts and smaller networks, not a 10,000-environment performance benchmark.
They establish integration and numerical behavior, not learned goalkeeper performance.

- 242 task CPU checks, including 14 SAC checks: endpoints, nonzero offsets, initialization, invalid config,
  real constructor dispatch, finite updates, and same-mode model/optimizer checkpoint round trips.
- 53 shared SAC actor and twin-reduction checks; 91 categorical-critic checks.
- 15 real PhysX checks: live hard limits, PD gains, armature, separate observations, linear support,
  target/thrust endpoints with motor lag, finite inference, and partial-reset isolation.
- Four compiled training iterations for each loss through the public training entry point.
  Each saved checkpoint contains six actor, critic, and temperature optimizer steps.
  Both runs use offline W&B and include numeric diagnostics and inherited source snapshots.
- The public match player reloads each smoke checkpoint for finite headless inference.

Source changes are local only. No new long training run or cluster sync has been launched.

The torque-mode and PD-gain follow-up is checked separately in
`artifacts/2026-10-16-goalkeeper-torque/`. Its checks cover both action modes,
both critic losses, gain overrides, torque endpoints, live PhysX properties, and checkpoint playback.
