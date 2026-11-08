# Borinot go-to: position plus deadline

Gym IDs: **Borinot-goto** and **Borinot-GoTo-Direct-v0**.

## Minimum-time variant

Use **Borinot-goto-minTime** (alias **Borinot-GoTo-MinTime-Direct-v0**) to train
arrival-and-stop behavior with an explicit cost of time. The original task keeps
its existing deadline-only behavior.

| Setting | Original task | Minimum-time task |
| --- | --- | --- |
| `env.reset_at_success` | `False` | `True` |
| `env.per_step_penalty_weight` | `-0.0` | `-0.015` |
| `env.remove_time_from_actor_obs` | `False` | `False` |

The minimum-time task ends on the first policy step that meets **all three**
existing success conditions: distance at or below `goal_success_tolerance_m`,
linear speed below `goal_success_lin_vel_tolerance_m_s`, and angular speed below
`goal_success_ang_vel_tolerance_rad_s`. It does not count a fast fly-through.
There is no extra dwell requirement. A safety failure still wins a simultaneous
success, and the sampled deadline still ends an unsuccessful episode. Both early
success and the deadline are true terminal states, not bootstrapped timeouts.

The time term adds **-0.015 on every policy step**, including the terminal step.
It is not multiplied by the timestep. At the default 100 Hz policy rate this is
-1.5 reward per simulated second. Changing the policy rate changes that rate;
for example, 50 Hz gives -0.75 per second with the same setting. Set the weight
to zero to disable it. The signed weight must be finite and nonpositive.

All other rewards keep their existing defaults. Terminal distance and enabled
terminal velocity costs apply on an early-success step as well as at a clean
deadline. The optional terminal success bonus also applies on that step when
enabled; it remains **off by default**, as in the original task. Existing
`env.use_terminal_success_bonus=True env.rwd_scale_terminal_success=5.0`
overrides continue to work. Failure distance costs remain unchanged. Both PPO
and SAC keep gamma 1.0; the explicit time cost supplies the speed incentive.

Train with the same options as the original task, changing only the task ID:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto-minTime --num_envs=10000 --headless env.control_mode=wrench
```

To hide time from the actor, append `env.remove_time_from_actor_obs=True`.
This removes **both** normalized remaining time and remaining seconds. It keeps
every other actor input in the same order, including controller memory. The
critic still receives both clocks. The default keeps the original observation
layout. Hiding time reduces the actor input by two, so a visible-time checkpoint
cannot be loaded unchanged into a hidden-time actor. Use the same setting during
training and playback.

The registered PPO, SAC-CE, and SAC-MSE recipes use separate experiment folders:
`borinot_goto_min_time` and `borinot_goto_min_time_sac`. Controllers, goal sampling,
success thresholds, disturbance schedules, and stage-pure curriculum rules are
inherited. Early successes count toward curriculum promotion. Monitor
`Episode/success_count` alongside `Episode/completed_count`,
`Goal/success_duration_s` (mean duration of successful completed episodes),
`Goal/early_success_count`, and `Episode_Reward/time`. Existing
`Goal/deadline_*` metrics still describe only episodes that reach the deadline.

The public player accepts both minimum-time IDs. It preserves episode resets on
early success even for a fixed live goal, instead of switching to the original
task's continuous hold mode:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goto.py --task=Borinot-goto-minTime --checkpoint=/absolute/path/checkpoint.pt --random-goals env.control_mode=wrench
```

Pass the same controller, observation, architecture, and agent overrides used in
training. The MuJoCo wrapper forwards the same task ID to this player too.

The following sections describe the original deadline task unless stated otherwise.

**Arm-aware INDI modes:** `env.control_mode=model_wrench_indi` and `env.control_mode=model_wrench_yawless_indi` retain the corresponding explicit-arm model and add a causal, physics-rate torque-residual correction. Both support PPO and SAC. See the shared [hybrid method, sensing, timing and configuration guide](../borinot/MODEL_WRENCH_INDI_CONTROL.md). Collective thrust remains model-based; the first post-reset tick is nominal.

This is a separate task beside `borinot_cmd_vel`. We ask Borinot to put its base
at a sampled 3D position when a sampled time budget ends. There is no ball,
heading target, velocity-tracking reward, obstacle course, or physical goal object.

The scene, robot inertias, motor lag, joint limits, startup randomization, reset
noise, controller implementations, and policy/physics clocks come from the
command-tracking task. The go-to task does not modify that task's settings.
It also reuses that task's vertical push disturbances. Here they grow with the
goal curriculum, from `start_forces_randn_curriculum_idx` on (see Curriculum).

## Start training

Run these setup commands in your terminal, once:

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jordibelp/IsaacLab
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

PPO, using your usual 10,000-environment training scale:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto --num_envs=10000 --headless env.control_mode=cbrt
```

This is an example command, **not a claim that 10,000 environments were tested on
the laptop**. Use your training machine and usual memory/compile settings.

Choose an existing or hybrid model-wrench control mode:

| `env.control_mode` | Scalar-angle actions | Default observations |
| --- | ---: | ---: |
| `thrust_and_torque_control` | 8 | 33 |
| `thrust_and_PD_joint_pos_control` | 8 | 33 |
| `PI_rotors_PD_joint_pos_control` | 6 | 40 |
| `dsam_indi` | 9 | 42 |
| `wrench` | 9 | 40 |
| `model_wrench` / `model_wrench_indi` | 9 | 40 |
| `model_wrench_yawless` / `model_wrench_yawless_indi` | 8 | 39 |
| `cbrt` | 6 | 37 |
| `cbrt_indi` | 6 | 37 |

The action meanings and PPO/SAC mappings are unchanged. The existing sine/cosine
angle flags, arm hold, actuator delay, motor lag, inertial model, and observation
options remain available. An added sine/cosine component adds one action and one
previous-action observation. Optional acceleration adds three observations per
physics sample: six with the default `env.obs_include_base_lin_acc_steps=2`.
Changing these settings can make checkpoints incompatible.

SAC MSE:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto --agent=rsl_rl_sac_mse_cfg_entry_point --num_envs=10000 --headless env.control_mode=cbrt
```

Use `--agent=rsl_rl_sac_cfg_entry_point` for SAC CE. SAC has the same inherited
restriction: PI is not supported; the other six modes are supported. CE uses
401 linear atoms over [-200, 200], rather than the inherited [-20, 20], because
the terminal cost can reach -100 on a safety failure. This is a practical starting
support, not a guaranteed bound. Larger workspaces, weights, or extra costs may
need wider support. MSE has no categorical support clipping.

Do not load a command-velocity checkpoint directly. The command and input size
have changed. Start a fresh policy. Go-to symmetry augmentation is not implemented;
keep `--symmetry-mode=none`. The launcher rejects that incompatible layout explicitly.

## What the policy sees

Actor and critic receive the **same** vector, in this order:

1. Base-body-frame linear velocity, 3 values, m/s. As in command tracking, this
   is the base rigid body's CoM velocity, not the whole robot's CoM velocity.
2. Base-body-frame angular velocity, 3 values, rad/s.
3. Optional body-frame kinematic acceleration history, m/s². Set
   `env.obs_include_base_lin_acc=True` (default False). The most recent
   `env.obs_include_base_lin_acc_steps` completed physics samples are flattened
   oldest to newest, XYZ per sample. The default length 2 adds six values.
   Each sample keeps its acquisition-time body axes; no averaging or filtering.
   Physical resets clear the selected histories and missing samples are zero.
   Live goal/deadline changes preserve the history. This is not accelerometer
   specific force. Use length 1 for older single-sample acceleration checkpoints.
4. Thigh and calf positions, then velocities, 2 + 2 values, rad and rad/s.
5. Previous policy action, with the selected controller's action dimension.
6. Optional projected gravity, 3 values. Off by default:
   `env.projected_gravity_in_obs=False`.
7. Full row-major body-to-world rotation matrix, 9 values.
8. Base-link world height, 1 value, m. On by default:
   `env.include_base_height=True`.
9. Goal minus current **base-link origin**, rotated into the full body frame,
   3 values, m. This is a displacement, not an absolute position or unit vector.
10. Normalized remaining time, 1 value: remaining seconds / assigned seconds.
    It starts at 1 and reaches 0 at the terminal state.
11. The existing controller-state observations, if any: realized thrust and PI
    integrator, or the existing INDI/wrench/CBRT memory. Their order is unchanged.
12. Remaining time in **seconds**, 1 value.

Why add seconds as well as normalized time? At reset, normalized time is 1 for
both a 0.8 s task and a 3 s task. With only that fraction and the same position,
a feed-forward policy cannot tell which command requires the faster response.
The additional scalar removes that ambiguity. There is no hidden change of goal
after the policy has already chosen its action.

Full attitude is retained because some action interfaces use world-frame
acceleration or heading. Height tells the policy where the ground boundary is.
The default attitude is upright with independent ±5° reset offsets on each axis;
the default base height is 1.5 m. All state inputs report the actual reset pose.

## Deadline and reward

Let `d` be the Euclidean 3D distance in metres between the base-link origin and
the fixed world-frame goal. On each policy transition:

```text
progress = rwd_scale_progress * episode_progress_factor * (previous_d - d)
terminal_distance = rwd_scale_terminal_distance * d               # only at the deadline
terminal_linear_vel = rwd_scale_terminal_linear_vel * |v|²        # only at a clean deadline
terminal_ang_vel = rwd_scale_terminal_ang_vel * |ω|²              # only at a clean deadline
```

Defaults are `rwd_scale_terminal_distance=-10.0` and `rwd_scale_progress=1.0`.
Neither term has an extra `dt` multiplier. A measured 0.1 m reduction contributes
+0.1 with the full progress factor. A clean deadline 0.3 m from the goal
contributes -3, in addition to that transition's progress. Moving away gives
negative progress; we do not clip it to zero.

The two terminal velocity costs are **on by default**
(`env.use_terminal_linear_vel_penalty=True`, `env.use_terminal_ang_vel_penalty=True`).
Each scale is -1.0, so arriving at 1 m/s costs -1 and spinning at 1 rad/s costs -1.
`v` is the base linear velocity and `ω` the base angular velocity, all three
axes, both in world axes. They are paid once, at a deadline without a safety
failure. These switches replace the old `env.use_terminal_speed_penalty`; the
old name and `rwd_scale_terminal_speed` no longer exist, so old commands fail loudly.

We keep the target fixed until reset. We measure progress against the same
target on both sides of a physics interval. An observation read never resamples
the target. The final transition is scored **before** automatic reset.

An episode ends when its own sampled deadline is reached. This is a real terminal
state, not a collection timeout: PPO and SAC do **not** bootstrap through it.
The task uses its own policy-step counter, independent of the runner's initial
randomization of `episode_length_buf`. Deadlines round **up** to a whole policy
step, so the extra time is less than one policy interval. The metadata horizon
is derived from the sampler after Hydra overrides; `env.episode_length_s` is not
an independent way to change deadlines. Change the goal-time settings instead.

### Early failures must not escape the cost

The shared safety checks still detect flight-bound violations, base crashes,
and excessive tilt. Their existing reset switches still apply. Self-contact
does not trigger a reset by default, as in command tracking.

If a safety failure ends a flight early, we also charge terminal distance, with
`max(d, failure_distance_floor_m)` instead of `d`. The default floor is 10 m,
so this reward term is at most -100. A crash cannot simply skip the terminal
cost or receive a near-zero distance cost by crashing next to the goal.
This is an explicit safety extension of the requested deadline reward, not an
extra dense reward. It is a starting penalty, not a proof that every possible
failure strategy is dominated. Revisit it if you enlarge the workspace.
Failure takes precedence over success if both happen on the last step.

### Discount and the question of rushing

The new PPO and both SAC recipes use **gamma = 1.0**. They inherit the other
algorithm settings. With finite terminal episodes, there is no continuing-task
return beyond the deadline. For a fixed episode progress factor, the sum of raw
progress increments is exactly `initial_d - final_d`. It does not independently
prefer an early approach over a late one with the same final distance.

This matters at 100 Hz. With gamma 0.99, a reward 3.5 s away has about 0.03 of
its initial weight. Discounted progress can also favor moving early. You can
override gamma, but it changes this timing behavior and the shaping argument.
SAC entropy regularization and other optional costs are separate objectives;
the telescoping statement is about the raw progress reward.

The default task means **be at the goal, and at rest, at the deadline**.
Reaching early and holding there is valid. Merely touching the goal early and
leaving is not success. A fast pass through the goal at the deadline is not a
success either: success also needs low linear and angular speed. There is no
guarantee of an aggressive response just because the normalized time is small;
the policy must learn what is feasible from the distance, velocity, and time inputs.

If you want **first arrival exactly at the requested time**, that is a different
task. A future version should measure first-arrival time and score a time window.
A dense penalty for early occupancy alone can lead to waiting just outside the
goal boundary. I have not added that ambiguous objective to the default task.

## Curriculum

We sample each goal as an independent uniform XYZ offset from the actual reset
position. XY forms a square; Z forms a smaller vertical interval. We do not clip
goals onto the safety boundary. Invalid goal boxes fail validation before startup.

| Difficulty | XY half-width, m | Z half-width, m | Sampled time, s |
| ---: | ---: | ---: | --- |
| 0 | 0.25 | 0.10 | 2.5–3.5 |
| 1 | 0.50 | 0.20 | 2.0–3.0 |
| 2 | 1.00 | 0.35 | 1.5–2.5 |
| 3 | 2.00 | 0.60 | 1.0–2.0 |
| 4 | 3.00 | 0.80 | 0.7–1.8 |
| 5 | 6.00 | 1.00 | 1.0–2.5 |

By default, deadlines use **only the phase's time range**, rounded up to a
policy tick. Both `goal_reference_speed_m_s` and `goal_reference_accel_m_s2`
default to `(0.0,)`. Omit these CLI overrides or set either to `[0.0]` to
disable its floor. Use `[0.0]`, not `[0.,]`: Hydra rejects the trailing comma.
A single value applies to
every difficulty, so adding goal stages does not require extending disabled
reference arrays. Alternatively, provide one value per goal difficulty; zero
disables that reference only for that difficulty. Negative or non-finite
references, empty arrays, and other array lengths are invalid.

Positive references optionally increase the final budget to at least
`distance / reference_speed` and/or
`2 * sqrt(distance / reference_acceleration)`. Each floor is independent: a
positive speed with zero acceleration enables only the speed floor, and vice
versa. The acceleration expression is the time for an
ideal symmetric accelerate/brake profile starting and ending at rest. These
are **sampling heuristics**, not physical controller limits or a reachability
certificate. Real Borinot must tilt, handle its arm, and respect thrust limits.
The floor can extend a budget beyond the table's nominal upper endpoint.
The metadata episode horizon accounts for every stage and enabled floor.

To reproduce the previous six-stage reference recipe, explicitly pass
`'env.goal_reference_speed_m_s=[0.4,0.7,1.2,2.0,3.0,3.125]'` and
`'env.goal_reference_accel_m_s2=[0.8,1.5,3.0,5.0,8.0,9.0]'`. Positive-reference
sampling is unchanged. When replaying an older checkpoint, pass its original
references if you need its training-time deadline distribution.

The configurable arrays are `goal_half_extents_m`, `goal_deadline_range_s`,
`goal_reference_speed_m_s`, and `goal_reference_accel_m_s2`. Defaults are a
starting recipe, not a tuned learning result. For a difficult last stage,
increase its times or reduce the reference speed/acceleration before concluding
that a controller cannot perform the task.

A success means: no safety failure, and at the deadline **all three** of:

- 3D base-link distance ≤ `goal_success_tolerance_m` (the green sphere radius);
- base linear speed `|v|` < `goal_success_lin_vel_tolerance_m_s` (default **0.1 m/s**);
- base angular speed `|ω|` < `goal_success_ang_vel_tolerance_rad_s` (default **0.2 rad/s**).

Speeds are vector norms over all three axes; yaw rate counts too. Success does
not depend on reward weights, minimum distance earlier in the flight, or the
time of first entry. The speed defaults are a first guess, not tuned values.
Tighter values make the 80% promotion gate below harder to pass.

Promotion requires all of the following:

- At least 512 completed episodes in the current stage's evidence window.
- At least 80% success in the rolling window of up to 2048 completed episodes.
- At least 1000 policy ticks at that stage: 10 simulated seconds per environment
  at the default 100 Hz. More environments do not shorten this time.
- At least one newly completed, eligible episode in the current reset batch.

Each episode keeps its original stage, goal, deadline, and progress factor.
Episodes still flying at a promotion finish normally, but their old-stage
outcomes cannot promote the new stage. Early failures count as failures.
Startup resets and manual aborted flights do not count as completed tasks.
Only one promotion is possible per reset batch, and evidence clears on promotion.
The window pools episode outcomes; it does not average unequal batch rates.

| Global stage | Goal difficulty | Progress factor | Push limit, N |
| ---: | ---: | ---: | ---: |
| 0–4 | 0–4 | 1.0 | 0 |
| 5 | 5 | 1.0 | 0 |
| 6 | 5 | 0.75 | 1 |
| 7 | 5 | 0.50 | 3 |
| 8 | 5 | 0.25 | 6 |
| 9 | 5 | 0.00 | 9 |

We first learn the full goal/time distribution. We then require fresh success
at each reduced progress weight. There is no automatic fade based only on
training iteration, and no immediate removal at the first easy success.

### Push disturbances

The pushes are the command-velocity task's pulses: a vertical world-Z force on
the base, with a uniform random size up to the stage limit and a random sign.
Each push lasts 0.5–2 s, and pushes start 2–5 s apart. The force list is the
inherited `forces_applied_to_base_curriculum = (0, 1, 3, 6, 9)` N.

`start_forces_randn_curriculum_idx` (default 5) says where that list starts.
Stage `start + k` uses list entry `k`. Earlier stages have no pushes; later
stages keep the last entry. So with the default list, stage 5 still has 0 N and
the first real push (1 N) appears at stage 6. Use `start - 1`, or drop the
leading 0 from the list, if you want pushes to begin exactly at a chosen stage.
Each episode keeps the push limit of the stage that started it.

Geometry, time and pushes grow together. For example, with a 9-stage goal
schedule (difficulties 0–8) and four fade stages, stages 6, 7 and 8 raise the
goal box and the push limit (1, 3, 6 N) in parallel. Stages 9–12 fade the
progress reward at 9 N. If the list runs past the last fade stage, the schedule
gets extra final stages that only raise the force.

Goal episodes last only about 1–3.5 s, shorter than the 2–5 s push spacing.
The command-velocity task waits a full spacing interval after each reset, so
almost no goal episode would see a push. Here, each reset instead starts the
push clock at a random point of that interval: the first push may begin at
any tick. Later pushes keep the normal spacing.

To train or replay without pushes, set `env.forces_applied_to_base_curriculum=[0.0]`.
A list with no positive force skips push sampling entirely, so the random
stream matches a push-free run.

Useful overrides:

```text
env.curriculum=False env.goal_curriculum_start_level=5
env.skip_curriculum=True
env.progress_reward_curriculum=False
env.rwd_scale_progress=0.0
env.start_forces_randn_curriculum_idx=3
env.forces_applied_to_base_curriculum=[0.0]
```

The first fixes the hardest geometry with full progress and no pushes. The
second starts at the last stage, including zero progress and the largest push.
The third keeps full progress and removes the four fade stages; with the
default forces, stages 6–9 then only raise the push limit. The fourth is a
sparse-reward ablation from the chosen stage. The fifth starts pushes earlier,
in parallel with the geometry. The sixth removes pushes.
`curriculum_start_level` belongs to the unused inherited command/push
curriculum; use **goal_curriculum_start_level** for this task.

Policy/optimizer checkpoints do **not** restore the evidence window, stage age,
or unfinished flights. To resume training at a chosen difficulty, set
`env.goal_curriculum_start_level=N` explicitly. The training launcher keeps its
existing stage-best checkpoint metadata. The player reads that metadata when
present; ordinary periodic checkpoints may need `--curriculum-level=N`.

## Optional additions: off by default

| Switch / scale | Behavior |
| --- | --- |
| `env.use_terminal_success_bonus=True` | Adds `rwd_scale_terminal_success` at a successful deadline; default scale +10.0. This rewards the full success test: inside the goal ball and nearly at rest. |
| `env.rwd_scale_action_rate` | Existing normalized physical command-change cost, default 0.0, multiplied by policy dt. |
| `env.rwd_scale_energy` | Existing modeled mean-power cost, default 0.0; no extra dt, matching the shared implementation. |

Inherited collision/tilt and raw-action-limit reward scales are also zero by
default. Their safety reset conditions are separate. Do not turn on several
regularizers at once for the first controller comparison. To compare with older
position-only runs, set `env.use_terminal_linear_vel_penalty=False
env.use_terminal_ang_vel_penalty=False`. Success still needs low speeds then.

## Playback

The player flies to a goal position that you choose, live. It opens a window
with X/Y/Z goal fields and sliders, a deadline field, and buttons. Give the
first goal with `--goal_init X Y Z` (alias `--goal`). The goal is in world
metres, in the same frame as `env.robot_home`: z is the height above ground.
The default goal is the robot home, so the robot first hovers in place.
Select **Relative to base position** in the panel to enter displacements
instead. The switch keeps the same numbers and does not send a goal.
Relative XYZ uses world-aligned axes, not axes rotated with the robot yaw.
Only **Go to goal** adds the displacement to each robot's current base position.
For example, `(1, 0, 0)` flies one metre along world X at the current height.
Press Go again with unchanged inputs to repeat that displacement from the new
position. Once sent, the target stays fixed, including across safety resets;
the yellow sphere previews where the *next* Go would send it. Fly home
switches back to world mode and sends the absolute home position.

Start directly in relative mode with `--goal-frame=relative --goal_init 1 0 0`.
In this mode the default input is `(0, 0, 0)`, keeping the initial position.

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goto.py --checkpoint=/absolute/path/model.pt --num_envs=1 --duration_s=2000 --goal_init 3.0 0.0 1.5 env.control_mode=model_wrench
```

How a goal works in the player:

- **Deadline.** By default the deadline is automatic: the middle of the
  deadlines in the current training reference for this distance.
  `--deadline_s` or the deadline slider sets a fixed time instead. Moving the
  slider switches Auto off. Any positive finite manual time is accepted, rounded
  up to policy ticks. It need not match a training range or the metadata horizon.
- **Edits and Go to goal.** XYZ, time, Auto and frame edits change only the draft.
  **Go to goal** snapshots all inputs together and sends them before the next
  policy action, including while paused. Later edits do not change that command.
  The green sphere shows the active goal; yellow previews the next target.
  **Fly home** sends the robot home.
- **Reset robot.** Reset starts a new rollout from home with the active goal.
  It never sends draft fields, even if Go is also queued. An active relative
  target stays fixed; Reset does not add the displacement again.
- **Chain mode (default).** A new goal starts from the current state. The
  robot is not teleported. The goal clock restarts, so the policy sees the new
  body-frame offset, a full `t_norm` and the new remaining seconds.
- **Hold.** At a clean deadline the robot keeps the same goal. A hold segment
  starts by itself. Manual time stays unchanged; Auto uses the shortest deadline
  in the reference interval for the current distance. Stage rewards pay only the
  state at the deadline: position and, by default, linear and angular speed.
  So the policy may loop away and come back between deadlines, and the loop
  grows with the time left.
- **Pushes.** The player applies the vertical pushes of its reference stage.
  The startup line and the final JSON report that push limit. Checkpoints
  trained before pushes existed need `env.forces_applied_to_base_curriculum=[0.0]`
  to replay their training conditions.
- **Failures.** Crash, tilt or leaving the area still resets the robot to home.
  Then it flies the same goal again.
- **`--reset-at-deadline`.** Training episodes: every deadline resets the robot
  to home and repeats the goal. A new goal also starts a new episode from home.
- **`--random-goals`.** Training goal and deadline sampling at the checkpoint
  stage, with a reset at every deadline. No goal window.

### Optional command sequence

Open **Command sequence (optional)** in the same panel. It is collapsed by
default. No new launch flags are needed.

1. Enter `(3, 0, 1.5)` in the main fields. Click **Add waypoint from draft**.
2. Set the first row's **Wait [s]** to `2`.
3. Enter `(-3, 0, 1.5)` in the main fields and add another waypoint.
4. Click **Start sequence**.

Each row has editable X/Y/Z, **Flight [s]** and **Wait [s]** fields, plus a
remove button. Flight `0` means Auto; a positive value sets that waypoint's
deadline. The WORLD/REL label records the main frame checkbox when you add
the row. Relative displacements resolve from each robot's position when
that row starts, along world axes.

The player flies the first waypoint. At its flight deadline, it holds
the same target for two extra simulated seconds, then sends the second
waypoint. The wait does not start at first arrival. With multiple robots,
all flight deadlines must finish before the wait starts. A training-score
miss does not stop the sequence; the result still appears in the goal log.
A safety reset stops the sequence.

Pause freezes sequence waits. Start snapshots the whole list; editing or
removing rows during a run affects only the next run. **Stop sequence**
cancels future waypoints but keeps the current target. It is not a brake.
**Go to goal**, **Fly home** and **Reset robot** also cancel the sequence.
After the last row's wait, the robot keeps the last target. Holds use the
existing policy and are not a promise of perfectly stationary hover.
Sequences require the default continuous live-goal mode, not
`--reset-at-deadline`, `--random-goals` or the minTime task.

Each finished goal prints one line, for example `[GOAL] ... SUCCESS. Final
distance 0.004 m, speed 0.05 m/s, angular speed 0.08 rad/s ...`. The run ends with one
`BORINOT_GOTO_PLAY {...}` JSON line with totals for all robots, the final
state of robot 0, and explicit input frame labels. Goal results also include
the final XYZ position and velocity before any reset.

The panel shows the current XYZ position, XYZ velocity, XYZ error to the
active goal, distance, speed, angular speed, yaw and time left. Position and error use
metres; velocity uses metres/second. All XYZ readouts use world axes. With
multiple robots, the readouts describe robot 0, relative to its environment
origin (the same frame as the world input fields).
The terminal prints these values in a `[STATE]` line every simulated second.
Set `--log_interval_s=0.5` for twice per second, or `--log_interval_s=0` to
disable periodic terminal lines. The panel remains live in either case.

The window
also says when an edited goal leaves what the checkpoint stage sampled: a
body-frame offset outside the stage box, a goal height outside the training
heights (only when the policy observes height), a moving start (training goals
start from rest), or a deadline outside the training range for the distance.
Such goals are allowed, but the result is then out of the training
distribution. The yaw is free in this task, and the rotation matrix is an
observation. In long chained sessions the yaw can drift tens of degrees from
the training start (±5°); watch the yaw value in the window.

Cameras: **Goal** (default) is a fixed side view that frames the start and
the goal of the latest flight; Follow, Chase, Side, Top and Free work as in
the command-velocity player. `--x_range`, `--y_range` and `--z_range` change
the slider ranges; defaults cover the flight area, not a curriculum box.
The exact-value fields are not clamped to slider ranges. Targets still have
to lie inside the configured flight boundaries.

Playback freezes curriculum. The stage comes from the checkpoint metadata
(`best_model_curriculum_idx`), unless `--curriculum-level` or
`env.goal_curriculum_start_level` selects one. A saved stage outside the current
schedule is bounded to that schedule. It is only a reference for Auto time,
training-region labels and random-goal sampling; it does not restrict live
goals or manual deadlines. You do not need a curriculum-level argument for
inference. `--duration_s` is how long the
player runs, not a deadline. Match the trained controller, observation flags,
physics settings, and algorithm; add the SAC `--agent` key when playing SAC.
Inference restores normalizers and weights, not optimizer state or a
training-sized replay buffer. `--headless --no-real-time` is useful for a
smoke check.

## Logs and interpretation

`info["reward_terms"]` contains signed per-step reward contributions.
`info["goal"]` keeps pre-reset distance, assigned time, remaining seconds,
deadline/failure/success masks, and collection stage for every environment.
The terminal snapshots remain valid after automatic reset.

Episode logs include terminal distance, clean-deadline distance, speed and
angular speed, initial/minimum distance, first-arrival time when reached,
energy, and counts of completed tasks, successes, clean deadlines, and early
failures. To see which success condition fails, divide
`Goal/deadline_inside_count`, `Goal/deadline_lin_vel_ok_count` and
`Goal/deadline_ang_vel_ok_count` by `Goal/deadline_count`. "First arrival" and
`Goal/ever_reached_count` use position only. `Curriculum/selected_force_limit_n`
is the push limit of the current stage; `Goal/collection_force_limit_n` is the
mean limit of the episodes that just finished. The public
PPO trainer pools `Episode/success_count` and `Episode/completed_count` before
reporting success rate. The existing SAC logger averages those two count fields
over reset packets; their ratio still gives the pooled success fraction for that
rollout. It does not publish PPO's pooled `Episode/successRate` field. Do not
replace this ratio with an average of reset-batch success fractions.
Curriculum logs distinguish the stage that collected
an episode, the stage whose evidence was evaluated, and the newly selected
stage. A selected-stage log is not a claim that the triggering episode used it.
Distance means are reset-batch means; use the numeric episode counts for exact
aggregation across unequal batches. RSL averages the curriculum decision fields
over reset packets within a rollout, so displayed window counts can be fractional.
The internal promotion uses the actual integer window and the exact rate.
`Curriculum/global_idx` is the trainer's current stage at logging time; the
per-reset `info["log"]` packet preserves each exact promotion decision.
The public PPO trainer also logs `Curriculum/selected_deadline_min_s`,
`Curriculum/selected_deadline_max_s`, and `Curriculum/selected_goal_distance_max_m`
once per iteration from that current stage, without reset-packet averaging.
The times are the configured `goal_deadline_range_s` bounds, before optional
distance-based floors and policy-tick rounding. The distance is the norm of the
current goal box's half-extents: its maximum sampled displacement from the
episode's starting base position, not the robot's remaining distance to its goal.
Reward-fade and force-only stages keep the final geometry/time row.

For a fair controller comparison, evaluate the same fixed goals and budgets,
seeds, model/randomization settings, push limits, and safety limits. Record success at the
deadline, deadline distance, speeds, and failures. First touch alone is not a
deadline success. Training to different curriculum levels is not a matched
comparison. No short smoke test establishes learned performance or hardware
transfer.

## Verification

CPU tests run without Isaac Sim:

```bash
PYTHONPATH=source/rsl_rl_sac_vendor python -m pytest --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goto/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goto/tests/test_goto.py -q
```

`tests/runtime_smoke.py` checks the real registered task, spaces, body-frame goal,
clocks, pre-reset terminal reward, failure handling, partial resets, shared
controller stepping, the speed conditions of success, stage-dependent pushes,
and controlled curriculum promotion. It writes a numeric
JSON receipt. Those deliberately controlled successes prove integration only.

Initial local verification: 53 new CPU tests, 288 command-task regression tests,
and 513 simulator checks across all seven controllers plus a 300/150 Hz SAC
case with expanded angle actions and optional observations. Four PPO runs and
both SAC losses completed two small eager training iterations and reloaded
through the public player. A further 112 saved-evidence checks covered source
snapshots, actual optimizer updates, gamma, and the saved CE support and gradients.
W&B stayed offline. These runs used small networks and short, deliberately
overridden deadlines. They do not establish convergence, large-scale throughput,
or hardware performance. Compilation and cluster deployment were not tested.

## Reference

Inspired by the position/time command and terminal-distance formulation in
[ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots](https://arxiv.org/abs/2306.14874),
especially sections IV-B.2 and IV-B.3. This task is not a reproduction of its
hierarchical navigation stack. The 3D flight sampler, safety-failure cost,
observation choices, and success-gated shaping fade are Borinot-specific design
choices. The defaults still need actual training and matched-controller trials.
