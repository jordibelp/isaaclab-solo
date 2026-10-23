# Borinot-cmd-vel

A free-flight velocity task for Borinot. There is no ball or goal.

Gym IDs: `Borinot-cmd-vel` and the equivalent `Borinot-Cmd-Vel-Direct-v0`.

## What the task tracks

The command has four entries: `[vx, vy, vz, yaw_rate]`.

We build a heading frame from the base yaw. Its x axis is
`[cos(yaw), sin(yaw), 0]`. Its z axis is world z. Its y axis is `z cross x`.
The columns of `R_world_heading` are those three axes. We compare the command
with `R_world_heading.T @ base_velocity_world`.

This keeps vertical velocity separate from the horizontal command when the robot
tilts. We use yaw from the quaternion, **not** the z component of `Log(R)`.
Those values differ for a tilted robot. At the exact vertical-body-x singularity,
the helper selects zero heading. The default 70-degree tilt reset avoids it.

The angular command tracks the **world-z component of angular velocity**. It is
not body-z angular velocity, an Euler-yaw derivative, or a three-axis angular
command. This follows the supplied sketch. Observed angular velocity remains in
the body frame.

The measured linear velocity is the base rigid body's center-of-mass velocity,
as in IsaacLab's `root_lin_vel_w`. It is not the whole articulated robot's CoM
velocity. Quaternions are body-to-world, in wxyz order.

### Defaults

| Setting | Default |
| --- | --- |
| Physics / policy frequency | 200 / 100 Hz |
| Episode collection limit | 10 s; timeout bootstraps |
| Base reset position | (0, 0, 1.5) m; x/y noise ±0.015 m, no height noise |
| Reset attitude | Nominal upright/zero yaw, with independent uniform ±5° roll/pitch/yaw offsets |
| Heading vx, vy | Each uniform in [-1, 1] m/s |
| World vz | Uniform in [-0.5, 0.5] m/s |
| World-z angular velocity | Uniform in [-1, 1] rad/s |
| Command resampling | Every 5 s and at episode reset |
| Zero-command probability | 0.02; all four command entries become zero |
| Parallel environments | 2048; override with `--num_envs` |

The next command is exposed in the observation **before** the policy takes the
action that is scored against it. Repeated observation reads do not resample it.

The task has its own episode clock. RSL's initial randomization of
`episode_length_buf` does not shorten a new flight.

`env.reset_rpy_noise_degs=[5.0,5.0,5.0]` sets the roll, pitch and yaw half-ranges
in degrees. New independent samples are applied at every episode reset, including
partial and automatic resets, from the first curriculum stage. The small local
rotation is composed after `robot_home_quat`; the nominal pose is not replaced.
For example, `'env.reset_rpy_noise_degs=[3.0,3.0,10.0]'` changes the three ranges.
Use `'env.reset_rpy_noise_degs=[0.0,0.0,0.0]'` for the original fixed attitude.
These offsets change the physical starting pose. Observations still report the
actual attitude; this is not sensor noise. Base linear/angular reset velocities
remain zero. The nominal USD spawn attitude stays fixed until the episode reset.
Observation and checkpoint dimensions do not change. For an exact fixed-attitude
evaluation of an older run, explicitly set all three noise half-ranges to zero.

## Rewards

Only these eight terms are used. Other goalkeeper reward fields inherited for
configuration reuse are not consumed by this task.

| Config weight | Default | Signal |
| --- | --- | --- |
| `rwd_scale_lin_vel` | 1.5 | `exp(-sum((command_xyz - measured_heading_xyz)^2) / tracking_lin_std^2)`, times policy dt |
| `rwd_scale_ang_vel` | 0.5 | `exp(-(command_yaw_rate - measured_world_omega_z)^2 / tracking_ang_std^2)`, times policy dt |
| `rwd_scale_action_rate` | 0.0 | Physical command-change cost from goalkeeper, times policy dt |
| `rwd_scale_energy` | 0.0 | Mean modeled total power in W; no extra dt multiplier, as in goalkeeper |
| `rwd_scale_base_collision` | -2.0 | External base contact above 1 N; once per policy interval, **no dt** |
| `rwd_scale_leave_area` | -2.0 | Any XYZ flight-bound violation; once per policy interval, **no dt** |
| `rwd_scale_self_collision` | -2.25 | Number of robot bodies with self contact above 0.6 N, times policy dt |
| `rwd_scale_tilt` | -1.0 | Tilt-threshold violation; once per policy interval, **no dt** |

Both tracking widths default to 0.5: m/s for linear velocity and rad/s for angular
velocity. The exponent denominator is 0.25. The linear reward uses one kernel
over **all three components**, not a sum of three independent kernels.

At exact tracking, the default reward is 2 per simulated second, or 0.02 per
policy step at 100 Hz. A full perfect 10-second episode returns 20 before
discounting and penalties. There are no position, altitude-hold, joint-pose or
survival rewards. A clean 10-second timeout has no additional penalty.

The four new costs are independent and additive. Leaving the flight area while
over the tilt threshold costs -3.0, before tracking or other costs. Several
violated area bounds still count as one area event. At 100 Hz, self contact on
two bodies costs `-2.25 * 2 * 0.01 = -0.045` for that policy interval. This is a
**body count**, like Solo's undesired-contact term, not a count of contact points
or unique pairs. Both ends of a self-contact pair can count. Each body counts
at most once per policy interval, even if it contacts several bodies or across
several physics substeps.

Contacts, bounds and tilt are latched across physics substeps so a brief event
does not disappear before scoring. The final reward includes these costs before
an automatic reset. Signed per-step contributions remain available in
`info["reward_terms"]`, and episode totals are logged under
`Episode_Reward/base_collision`, `leave_area`, `self_collision` and `tilt`.

Set a cost weight negative to penalize its signal. The action-rate signal sums
two group means: squared rotor-command changes divided by the full thrust span,
and squared joint-effort changes divided by the full signed torque span.
It uses clipped commands, not raw policy differences. Reset history starts at
hover thrust and zero joint effort.

The inherited power model includes rotor drag power, motor dissipation, signed
joint mechanical power, and joint copper loss. It samples every completed
physics substep, including the final substep before automatic reset. Joint
effort in implicit PD mode is IsaacLab's clipped estimate, not measured PhysX
drive torque. Episode energy is also logged in joules. The name
`rwd_scale_energy` retains the goalkeeper API, but its reward signal is **mean
power**, not episode energy.

## Observations and control modes

Policy and critic receive the same simple vector. No absolute base position,
force, force point, curriculum index, or randomized physics parameters are added.

Order:

1. Base linear velocity in body axes: 3, m/s.
2. Base angular velocity in body axes: 3, rad/s.
3. Optional body-frame base CoM linear acceleration: 3, m/s².
4. Two leg joint positions: 2, rad, in `joint_names` order.
5. Two leg joint velocities: 2, rad/s.
6. Previous executed policy action: action count below.
7. Projected unit gravity in body axes: 3, dimensionless.
8. Body-to-world rotation matrix, row-major: 9, dimensionless.
9. Current velocity command: 4, in the mixed heading/world frames above.
10. Controller state, only for PI or INDI, as below.

Acceleration is off by default. Enable `env.obs_include_base_lin_acc=True` to
add it. This is simulated **kinematic acceleration**, not an accelerometer's
specific force or a noisy IMU model. Its reset observation is zero. Changing this
flag changes policy size and requires a compatible checkpoint.

| `env.control_mode` | Actions | Observations | Agents |
| --- | --- | --- | --- |
| `thrust_and_torque_control` (default) | 6 rotor thrusts + 2 joint torques = 8 | 34 | PPO, SAC CE, SAC MSE |
| `thrust_and_PD_joint_pos_control` | 6 rotor thrusts + 2 joint targets = 8 | 34 | PPO, SAC CE, SAC MSE |
| `PI_rotors_PD_joint_pos_control` | Collective + body rates + 2 joint targets = 6 | 41 | PPO |
| `indi` | World acceleration + body rates + heading + 2 joint targets = 9 | 43 | PPO, SAC CE, SAC MSE |

Optional acceleration adds 3 observations in every mode. PI adds six actual
rotor thrust states and three rate integrals. INDI adds its six thrust states and
two joint integrals. As in goalkeeper, not all low-level filter memory is exposed.
The common direct mode has only the requested features plus the command.

Controls and actuator dynamics reuse the current goalkeeper implementation:
40 ms motor lag, nominal hover calibration, the 10 N per-rotor ceiling,
2.7 N·m joint effort limits, and zero implicit PD gains in direct torque mode.
The physical joint armature and friction are preserved. INDI uses the existing
speed-lag model and explicit joint PID. No low-level flight controller is added
to the direct modes.

PPO and SAC retain their existing action transforms. SAC maps rotor -1/0/+1
to zero/hover/maximum thrust. Direct SAC joint torques remain normalized until
the environment multiplies by `effort_lim_sim`. PPO uses its existing affine
hover-centered rotor mapping. Torque and position checkpoints are not
interchangeable even though both have eight actions.

The plant keeps the same two-joint leg and racket, including its mass and shape.
Only the ball and goal are removed. The source robot USD and both goalkeeper
tasks are unchanged.

## Randomization and vertical disturbances

Startup randomization is the same as goalkeeper:

- Joint friction sampled in [0.01, 1.0].
- Base mass and inertia scaled by a sample in [0.9, 1.2].
- Racket mass and inertia scaled by a sample in [0.9, 1.2].
- Base CoM shifted independently by ±1 cm per axis.

Each clone keeps its sampled values across resets. The common controller/action
calibration stays nominal. Use `env.include_events_randomization=False` to
disable all four terms, or, for example,
`env.event_randomization.racket_mass=False` to disable one term.
The master startup switch does not disable the separate push curriculum.
Friction semantics differ between Isaac Sim 4.5 and 5.x, as documented in the
shared randomizer; matching numerical ranges alone does not match both plants.

Push defaults:

| Setting | Default |
| --- | --- |
| Start-to-start interval | Uniform 2–5 s; first push delayed after reset |
| Pulse duration | Uniform 0.5–2 s |
| Direction | World +z or -z, equal probability |
| Magnitude | Uniform 0 to the episode's current force limit |
| Application point | Area-uniform body-frame disk, radius 0.12 m |
| Top/bottom offset | ±0.03 m from base-link origin; force enters the corresponding face |

We sample radius as `R * sqrt(U)` and angle uniformly. A uniform radius would
overweight the center. The sampled point stays attached to the base during a
pulse. Force direction stays vertical in **world** axes during the pulse.
We rotate the force into body axes at every physics tick and add the moment
`(point_body - actual_base_CoM_body) cross force_body` to the rotor wrench.
This respects the randomized CoM and does not overwrite thrust or yaw torque.

Intervals and durations round up to policy ticks. Duration cannot exceed the
minimum start-to-start interval. Pulses clear when they end, at partial reset,
and at automatic reset. Disturbance magnitude, duration and point dimensions are
configurable assumptions, not a calibrated disturbance model.

## Force curriculum

Defaults: `forces_applied_to_base_curriculum=(0.0, 5.0, 8.0)` N.
Each entry is the maximum magnitude, not a fixed force.

Promotion uses the unweighted XY tracking kernel, with the same linear width as
the XYZ reward. It excludes vertical and angular tracking. For each completed
episode, we integrate the XY kernel and divide by the **full 10-second horizon**.
Thus an early crash cannot promote through a brief good tracking interval.

Advance one level when all are true:

- The rolling mean is strictly above `curriculum_tracking_threshold=0.7`.
- At least `curriculum_min_episodes=128` eligible episodes completed.
- At least `curriculum_min_steps=1000` policy ticks elapsed at this level.

The ring window retains at most `curriculum_window_episodes=256` episodes.
Only episodes that started at the current level can contribute. A promoted level
applies at each environment's next episode reset, not halfway through its flight.
Old-stage episodes cannot promote again. Dwell and evidence reset on promotion.
These settings do not depend on reward weights or on the number of parallel
environments.

Useful overrides:

- No pushes: `env.curriculum=False env.curriculum_start_level=0`.
- Fixed 5 N limit: `env.curriculum=False env.curriculum_start_level=1`.
- Start at the last level: `--skip_curriculum` for training, or
  `env.skip_curriculum=True` for playback.
- Use just two levels: `'env.forces_applied_to_base_curriculum=[0.0,8.0]'`.

The runner writes per-stage best checkpoints through `get_curriculum_global_idx`.
Its policy/optimizer checkpoint does not restore this environment's rolling
evidence, clocks, or live physics state. For a resumed curriculum, select the
intended `env.curriculum_start_level` explicitly; promotion evidence starts fresh.

## Safety boundaries

Reset below 0.15 m or above 20 m base height, beyond ±25 m in either local world
horizontal axis, above 70 degrees tilt, or on external base contact above 1 N.
The height range and XY limit define the area penalty as well as the area reset.
They are not command limits or position targets.

Four single-body contact sensors report the base, two leg links and paddle.
Each sensor explicitly filters the other three robot bodies. For the base crash
test we subtract those self-contact forces from its total contact force. Thus
self contact alone does not falsely trigger an external base crash. Paddle or
leg contact with the ground is not a self-collision or base-collision penalty.

Physical self-collisions are **enabled by default in this task**. Existing
articulation pair filters, including parent-child exclusions, are unchanged.
`env.enabled_self_collisions=False` disables physical self contact and its cost.
The force thresholds are configurable with `base_collision_force_threshold_n`
and `self_collision_force_threshold_n`; both are strict greater-than tests.

Self contact costs reward but does not reset by default
(`self_collision_triggers_reset=False`). Set it to `True` for an early reset.
Disable the external base reset with `env.reset_robot_on_robot_crash=False`, or
the tilt reset with `env.reset_robot_on_tilt_threshold=False`. Those two reset
switches do **not** disable their costs. Set the corresponding reward weight to
zero if the cost should also be disabled. The area reset remains enabled.

A sustained downward or upward velocity command can eventually reach a height
boundary. We do not replace the
requested vertical-velocity objective with an altitude-hold objective.
In particular, zero vertical velocity does not prefer 1.5 m over another height;
a stationary robot supported by its leg can also match a zero-velocity command.

Clone spacing is 64 m. Validation requires enough spacing for both flight boxes
plus robot clearance, so their robots cannot reach each other's collisions.

## Train and play

Run from the canonical `/home/jordibelp/IsaacLab` checkout. Before any launch:

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

PPO, with a modest local environment count:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-cmd-vel --num_envs 128 --headless --logger tensorboard '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

SAC CE, with laptop-sized replay/update batches:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-cmd-vel --agent rsl_rl_sac_cfg_entry_point --num_envs 32 --headless --logger tensorboard agent.algorithm.replay_buffer_size=4096 agent.algorithm.num_mini_batches=2 agent.algorithm.mini_batch_size=32 agent.local_redundancy.batch_size=8 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Use `--agent rsl_rl_sac_mse_cfg_entry_point` for scalar MSE. CE keeps the existing
101 linear atoms on [-20,20]; optional large costs can require a wider support.
These are the same recipes as goalkeeper, with separate experiment names.
SAC PI is intentionally unsupported, as in the source task.

Example fixed command: add `'env.command_lin_vel_x_range=[0.5,0.5]'`, and set the
other ranges as needed. Set `env.standing_env_prob=0.0` to prevent zero-command
resampling in a fixed-command test.

Interactive public player (PPO by default):

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_cmd_vel.py --checkpoint /absolute/path/model.pt --duration_s 2000 --cmd_init 0.1 0.0 0.0 0.0
```

Match the training `--agent`, control mode, architecture, observation flags and
physics settings. For example, add `--agent rsl_rl_sac_cfg_entry_point` for CE.
The player restores observation normalizers, skips optimizer restoration, uses
minimal replay allocation, and checks finite outputs. It uses deterministic
policy inference. Evaluation freezes the curriculum by default. The force level
comes from checkpoint metadata (`best_model_curriculum_idx`, then
`curriculum_global_idx`), with the environment default as a fallback. An explicit
`--curriculum-level N` takes priority. Explicit Hydra stage/skip overrides are
also preserved instead of being overwritten by checkpoint metadata.

`--cmd_init` takes **four** numbers: `vx vy vz wz`. The first two are heading-frame
speeds, `vz` is world-up speed in m/s, and `wz` is angular velocity about world Z
in rad/s. The default is all zeros. The GUI provides four sliders with exact
numeric entry, Zero commands, Reset robot, and Pause/Resume. A command applies
to all clones and persists through timed resampling and all episode resets.
The command is visible to the policy before the action that uses it. These
play-only controls do not change the registered training environment.

The camera follows robot 0. Switch live between world-aligned Follow, yaw-aligned
Chase, Side, Top and mouse-controlled Free. Start with `--camera chase` or
`--camera free` if preferred. Optional `--vx_range MIN MAX`, `--vy_range`,
`--vz_range` and `--wz_range` change the slider ranges, not the training task.
An explicit slider range must include the initial value and zero.

`--episode_length_s 80` changes only the play timeout. Omitted keeps the task's
10-second timeout. Crash, tilt and flight-bound resets still apply. Zero `vz`
requests zero vertical speed; it does not hold a chosen altitude. A sustained
nonzero `vz` can eventually reach a height boundary.

Use `--no-ui` to hide the panel, `--headless` for no window,
`--no-real-time` to disable wall-clock pacing, or `--random-commands` to retain
the original training-style command sampler (no manual command panel).

For the trained `aunmwlq7` PPO checkpoint:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_cmd_vel.py --task Borinot-cmd-vel --agent rsl_rl_cfg_entry_point --checkpoint /home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1023_aunmwlq7_best_model_curriculum_idx_2.pt --num_envs 1 --duration_s 2000 --cmd_init 0.1 0.0 0.0 0.0 --episode_length_s 80 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100
```

The saved stage is 2. This run's recorded force levels are `[0.0,1.0,3.0]` N,
matching the current local config. Stage metadata selects an index; it does not
restore the force magnitudes or other environment settings. For this run,
`--curriculum-level 0` disables pushes. Match the recorded force list when
playing a run trained with different magnitudes.

Validated locally on 2026-10-23: 65 CPU checks (50 task and 15 playback tests),
1,200 public-player steps with this exact checkpoint and nonzero `vz`/`wz`,
and a 400-step, two-clone GUI callback smoke. The latter checks all four
fields, zero commands, manual/partial/automatic resets, expired command
resampling, pause/resume and all camera modes. Rendered viewport and panel
were inspected. This is playback integration evidence, not a long-flight,
cluster-simulator parity or hardware-transfer claim.

Training logs go under `logs/rsl_rl/borinot_cmd_vel` or
`logs/rsl_rl/borinot_cmd_vel_sac`. Existing research logs are not changed.
W&B source snapshots include the inherited environment configs, shared plant
helpers, math and controller sources, plus the existing agent/algorithm chain.

## Verification

CPU checks do not need Isaac Sim:

```bash
python -m pytest -q --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_cmd_vel/tests -p no:cacheprovider source/isaaclab_tasks/isaaclab_tasks/direct/borinot_cmd_vel/tests/test_task.py
```

Keep `--confcutdir`: otherwise pytest imports the parent IsaacLab package before
the simulator starts. The general live fixture is `tests/runtime_smoke.py`; launch it
through `isaaclab.sh` with `--headless --output /absolute/path/result.json` and
the desired Hydra overrides. Run one simulator process at a time.

`tests/penalty_smoke.py` checks real self-only, floor-only and mixed contacts,
reset switches, exact reward values, automatic-reset reward delivery and clean
timeouts. Run it the same way; also use `env.enabled_self_collisions=False` and
an alternate policy rate to check the disabled path and dt scaling. Its folded
test pose deliberately makes contact above the configured force threshold.

Saved evidence is in the OpenClaw workspace under
`artifacts/2026-10-23-borinot-cmd-vel`. It separates CPU tests, live property and
wrench checks, short public training runs and public checkpoint playback.
These checks establish implementation/integration, not a learned flight policy,
10,000-environment throughput, cluster readiness or hardware transfer.

Original task verification on 2026-10-23:

- 31 new CPU tests and all 438 existing goalkeeper tests passed.
- Six live control/randomization fixtures passed 703 checks in total.
- A separate fixture passed 13 checks, including vertical-force momentum,
  off-center pitching, live curriculum transitions, invalid-config rejection and
  an inspected rendered frame.
- All 10 supported control/agent pairs trained for four small iterations and
  reloaded through the public player for 150 finite steps each.
- An additional SAC-CE smoke used the default enabled update compilation.
- An offline W&B run archived the inherited configuration and shared source
  files; their saved bytes matched the actual source files.

Collision-penalty update, also verified locally on 2026-10-23:

- 50 velocity-task CPU tests and all 438 goalkeeper regression tests passed.
- 250 live checks passed: default contacts (42), self-collisions disabled (38),
  INDI with 300/150 Hz physics/policy rates (42), and general runtime (128).
- Real fixtures separated self-only, paddle-floor, base-floor and mixed
  self/floor contacts. They checked exact costs, additive events, reset switches,
  partial-reset isolation and clean timeouts. Automatic-reset checks confirmed
  the final costs reach the agent and the episode logger.
- PPO, SAC-CE and SAC-MSE each completed four small direct-torque training
  iterations and 150 finite public-player steps, including 24 completed episodes
  per playback. Checkpoints contain nonzero optimizer steps. TensorBoard records
  all four new episode reward terms.
- Evidence: `artifacts/2026-10-23-borinot-penalties` in the OpenClaw workspace.
  Saved configs retain the working-tree tuning used during these checks.

Training smokes used 16 environments, 64/32 networks, eight steps per iteration,
and 0.12-second episodes to exercise repeated automatic resets. They are not
performance results. No cluster job, cluster sync, online W&B upload or hardware
action was performed.
