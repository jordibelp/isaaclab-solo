# Solo12 direct torque control and rate experiments

## What is implemented

Two new tasks use the existing Solo12 environment:

- `solo12-v0-torque`: velocity-command locomotion with direct joint effort.
- `solo12-two-feet-torque`: the existing two-feet rewards, resets and curriculum, with direct joint effort.

The original `solo12-v0` and `solo12-two-feet` remain position-reference tasks.
All four default to **50 Hz policy / 200 Hz physics**. The torque tasks are new
training conditions. Do not load a position-policy checkpoint into them and expect
its actions to mean the same thing.

### Action contract

Torque tasks compute `torque_nm = clip(action_scale * action, -effort_limit_sim, +effort_limit_sim)`.
The default `action_scale` is **2.65 N*m per unit action**. The default cap is
**2.65 N*m**. These are separate settings: reducing the cap does not silently change
the action scale. Zero action means zero torque, with no nominal-pose torque offset.
The nominal pose still centres the joint-position observations.

Both actuator gains are zero. The environment sends effort targets, not angle
targets. There is no hidden damping controller, gravity compensation, torque filter,
or slew limiter. Commands are held between policy updates and pass through the
existing physics-rate delay buffer. Commands are clipped before entering that
buffer. Reset clears the held effort for the reset robots, as well as action and
delay histories. Existing armature, friction, velocity limits, joint limits and
randomization remain part of the plant; removing PD does not remove them.

The actuator class remains `ImplicitActuator`, but with zero stiffness and damping;
its name does not mean a nonzero PD controller remains active. Joint efforts in
this mode are the feed-forward efforts sent to PhysX. Contact/limit constraint
forces are not an actuator torque measurement.

RSL PPO and SAC registrations are included, along with the existing symmetry
configuration. SAC bounds are symmetric `effort_limit_sim / action_scale`, not
bounds calculated from joint angles. The new SKRL entry points reference the
existing `skrl_ppo_cfg.yaml`; only the RSL training paths were exercised here.

The regular player can run these tasks. Position-target telemetry is unavailable
(NaN) in torque mode: a torque value must never appear on a `q_desired` angle plot.
Actual joint angles and actuator-effort traces remain meaningful.

### Clock contract

Use `env.physics_hz` and `env.policy_hz`. Physics must be an integer multiple of
policy rate and at least as fast. Invalid, zero and nonfinite rates fail early.
The environment derives `sim.dt`, `decimation`, rendering cadence and both contact
sensor periods after command-line overrides. IMU history dimensions also refresh
from the effective decimation. These Hz fields are authoritative: migrate old
`env.sim.dt` / `env.decimation` overrides instead of combining both interfaces.

| Policy Hz | Physics Hz | Decimation | Policy interval |
|---:|---:|---:|---:|
| 50 | 200 | 4 | 20 ms |
| 100 | 200 | 2 | 10 ms |
| 200 | 400 | 2 | 5 ms |
| 500 | 1000 | 2 | 2 ms |
| 1000 | 2000 | 2 | 1 ms |

Two physics steps per action are supported, not a guarantee of contact accuracy.
Your proposed sweep changes both the controller and integration rates. Also test
all policy rates at fixed 2000 Hz physics before attributing a change to policy
frequency alone. Refine physics again for selected checkpoints to check numerical
sensitivity. PD uses an implicit solver, so its effective discrete behaviour can
also change when the physics timestep changes.

### Action-rate reward

Position tasks keep their original raw-action difference penalty and weight
`action_rate_reward_scale=-0.05`.

Torque tasks default to **`action_rate_reward_scale=-0.01`**. Their signal is the
sum over joints of squared changes in the **clipped, issued torque command divided
by action_scale**, scaled by `(policy_hz / action_rate_reference_hz)^2`.
The reference rate defaults to 50 Hz. As before, the weighted signal is multiplied
by the policy timestep when added to reward.

This approximates an integrated squared torque-slope cost for a smooth trajectory.
The same normalized ramp has the same reward density at every tested policy rate.
Raw actions that differ but saturate to the same command do not count as a torque
change. The penalty is on issued commands, not on delayed commands or measured
joint acceleration. A reset starts the previous command at zero.

The weight is a starting point, not a tuned optimum. Rapid alternating exploration
at 1000 Hz is deliberately much more expensive than at 50 Hz. A strong smoothness
cost can also prevent learning quick contact corrections. Tune it using tracking,
falls and torque traces together, not return alone. It does not impose a hard slew
limit or guarantee smoothness on hardware.

## Commands

Run from the canonical IsaacLab checkout after activating `env_isaaclab` and the
usual CA certificate variables. This example starts **fresh** PPO training:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task solo12-v0-torque --headless --num_envs 10000 \
  env.policy_hz=200 env.physics_hz=400 \
  env.effort_limit_sim=2.65 env.action_rate_reward_scale=-0.01
```

For two feet, replace the task with `solo12-two-feet-torque`. For the other rates,
use the table above. Omit `--checkpoint` for fresh training. Add
`--agent rsl_rl_sac_cfg_entry_point` to select the existing SAC recipe instead of
PPO. This changes the algorithm as well as the action interface; compare within
one algorithm first.

Playback must use the **training task, rates, action scale and effort cap**:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_direct_0325.py \
  --task solo12-v0-torque --checkpoint /absolute/path/model.pt \
  --num_envs 1 --duration_s 30 --disable_training_gain_sync \
  env.policy_hz=200 env.physics_hz=400 env.effort_limit_sim=2.65
```

`--disable_training_gain_sync` avoids the player's optional W&B gain lookup;
torque mode forces zero gains anyway. Keep the correct task and rates explicit.
Changing a trained policy's deployment rate is a separate robustness test, not a
replacement for retraining at each rate.

These changes are **simulation/training only**. The current position-policy ROS
inference contract is not a torque deployment interface. Do not send torque-policy
outputs through a position-policy converter or setpoint publisher. A hardware path
needs explicit torque/current units, order/sign checks, torque caps, watchdogs and
safe fallback behaviour. A simulator cap does not configure a motor-driver limit.

## What the papers do and do not establish

1. [Aractingi et al., *Controlling the Solo12 quadruped robot with deep reinforcement learning*, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10366154/).
   The authors argue that impedance control helps transfer, especially for the
   low-inertia swing legs. Their deployed policy runs at 100 Hz, with PD at 1 kHz
   in simulation and a 10 kHz low-level communication loop on hardware. This is
   useful Solo12 evidence, but not a torque-policy frequency sweep.
2. [Chen et al., *Learning Torque Control for Quadrupedal Locomotion*, Humanoids 2023](https://hybrid-robotics.berkeley.edu/publications/Humanoids2023_RL_TorqueControl.pdf).
   Direct-torque RL transfers to a real A1 at 200 Hz. The authors report no extra
   transfer difficulty in their setup and acknowledge some jerky torque commands.
   Their real-world comparison uses a 50 Hz position policy with a 200 Hz PD loop;
   it is not proof of equal-rate superiority or a Solo12 stability threshold.
3. [Peng and van de Panne, *Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?*, 2017](https://xbpeng.github.io/projects/ActionSpace/index.html).
   This compares action representations, learning, robustness and policy query
   rates in simulated planar characters. It motivates the experiment, but does
   not establish a sim-to-real result for a physical Solo12.

Your hypothesis is plausible, but **direct torque does not inherently imply a
larger transfer gap in every system**. PD supplies fast local restoring feedback
and damping. It is not simply an interpolator: a held angle reference still causes
torque to change with the measured joint state between policy updates. Without
PD, that stabilizing response must come from the learned feedback and the plant.
A higher policy rate reduces hold time, but does not fix sensor delay, torque
calibration, friction errors, motor bandwidth or network scheduling jitter.

I would treat **100 Hz as a challenging low-rate condition**, **200 Hz as a useful
first baseline**, and **500/1000 Hz as tests of additional benefit and deployment
cost**. This is an experiment recommendation, not a measured stability ranking.
The two-feet balance task may differ substantially from four-foot locomotion.
No reviewed source establishes stability for this exact task at all four rates.

## How to make the comparison informative

- Train a separate policy per rate, with several training seeds. Start on a fixed
  terrain/curriculum phase and common command/disturbance scenarios. Match physics
  rate for the causal policy-rate comparison. Match PD inner-loop rate too.
- Report fall/reset rate, uninterrupted balance duration, command tracking error,
  base roll/pitch motion, joint-speed/acceleration RMS, torque RMS, torque-slope
  RMS, saturation fraction and torque spectrum. Separate useful gait motion from
  high-frequency vibration. Save numeric samples, not just videos or plot images.
- Use physics-rate traces for vibration analysis. Policy-rate samples can hide
  between-action oscillation; a spectrum only resolves frequencies below half its
  sampling rate. Compare equal-duration windows and common spectral bands.
- Measure end-to-end sensor-to-torque delay and deadline misses on the deployment
  computer. A 1000 Hz loop has a 1 ms budget. Fast isolated network inference is
  not proof that the complete hardware loop meets that budget.
- Keep perturbations in physical units. Existing `actuation_delay_range` and its
  curriculum values remain **physics ticks**. `(0,3)` means 0–15 ms at 200 Hz but
  only 0–1.5 ms at 2000 Hz. To preserve the 15 ms maximum, use maxima 3, 6, 15 and
  30 at physics 200, 400, 1000 and 2000 Hz, respectively; update every relevant
  curriculum phase too. The discrete delay distributions are still not identical.
- Keep training horizons in seconds when comparing rates. For PPO's current
  50 Hz defaults, use `gamma = 0.99^(50/policy_hz)` and
  `lambda = 0.95^(50/policy_hz)` as starting continuous-time matches. Scale
  `num_steps_per_env` from 24 to 48/96/240/480 at 100/200/500/1000 Hz to keep a
  0.48 s rollout. This increases memory and update work. Match total simulated
  time and inspect update counts; equal iterations or equal transitions do not
  give equal physical interaction time. SAC has different defaults and replay/
  update clocks; do not copy the PPO numbers into it.
- The code does **not** silently rescale algorithm hyperparameters, delay ranges,
  curriculum windows counted in steps/episodes, or IMU noise calibrated per
  physics sample. With IMU observation options, check history duration, bias
  diffusion and noise bandwidth separately. For a complete recipe sweep, report
  these differences rather than calling it an isolated rate experiment.

## Verification boundary

Verified locally on 13 November 2026, using Isaac Sim 5.1. The parent revision
was `a5a9ad1`. **182 CPU tests passed**, including 32 new control checks. A
50/200 Hz PD rollout matched the parent bit-for-bit across 24 steps and four robots,
with observation noise and startup randomization disabled in both. All pre-existing
default config fields also matched (normalizing the test module’s function path).
Both torque tasks passed all five rate pairs. Additional 1000/2000 Hz checks used
a reduced 1.25 N*m cap and verified selected-robot resets preserve the other robots’
held efforts. Both tasks completed two PPO iterations with symmetry augmentation;
the locomotion task also completed two SAC iterations with the reduced cap. Their
public players loaded the resulting checkpoints and ran short headless rollouts.

The local checks cover the implementation, not the learned behaviour. CPU tests
exercise invalid rates, torque scaling/caps, feedback, delay routing, nested gains,
SAC bounds, reward normalization and telemetry labels. Real simulator checks cover
both torque tasks at all five rate pairs above, including partial resets. Separate
checks compare the default PD configuration and trajectories against the parent
revision. Short public RSL training/playback smokes check integration only.

Run the focused CPU checks from the repository root:

```bash
/home/jordibelp/miniconda3/envs/env_isaaclab/bin/python -m pytest -q \
  source/scripts/rsl_rl/tests/test_solo12_torque_control.py \
  --confcutdir=source/scripts/rsl_rl/tests --rootdir=source/scripts/rsl_rl/tests
```

No full torque-policy training campaign, stability sweep, real-robot test, cluster
submission, cluster synchronization or push is included in this change.
