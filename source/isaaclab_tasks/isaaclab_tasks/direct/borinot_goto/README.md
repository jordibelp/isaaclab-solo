# Borinot go-to: position plus deadline

Gym IDs: **Borinot-goto** and **Borinot-GoTo-Direct-v0**.

This is a separate task beside `borinot_cmd_vel`. We ask Borinot to put its base
at a sampled 3D position when a sampled time budget ends. There is no ball,
heading target, velocity-tracking reward, obstacle course, or physical goal object.

The scene, robot inertias, motor lag, joint limits, startup randomization, reset
noise, controller implementations, and policy/physics clocks come from the
command-tracking task. The go-to task does not modify that task's settings.
External push disturbances are disabled here, at every curriculum level.

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

Choose any of the seven existing PPO control modes:

| `env.control_mode` | Scalar-angle actions | Default observations |
| --- | ---: | ---: |
| `thrust_and_torque_control` | 8 | 33 |
| `thrust_and_PD_joint_pos_control` | 8 | 33 |
| `PI_rotors_PD_joint_pos_control` | 6 | 40 |
| `indi` | 9 | 42 |
| `wrench` | 9 | 40 |
| `cbrt` | 6 | 37 |
| `cbrt_indi` | 6 | 37 |

The action meanings and PPO/SAC mappings are unchanged. The existing sine/cosine
angle flags, arm hold, actuator delay, motor lag, inertial model, and observation
options remain available. An added sine/cosine component adds one action and one
previous-action observation. Optional acceleration adds three observations.
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
3. Optional body-frame kinematic acceleration, 3 values, m/s². Set
   `env.obs_include_base_lin_acc=True`. This is not accelerometer specific force.
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
terminal_distance = rwd_scale_terminal_distance * d   # only at the deadline
```

Defaults are `rwd_scale_terminal_distance=-10.0` and `rwd_scale_progress=1.0`.
Neither term has an extra `dt` multiplier. A measured 0.1 m reduction contributes
+0.1 with the full progress factor. A clean deadline 0.3 m from the goal
contributes -3, in addition to that transition's progress. Moving away gives
negative progress; we do not clip it to zero.

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

The default task means **be at the goal at the deadline**. Reaching early and
holding there is valid. Merely touching the goal early and leaving is not
success. Conversely, a fast pass through the goal exactly at the deadline can
succeed: the default objective does not require stopping. There is no guarantee
of an aggressive response just because the normalized time is small; the
policy must learn what is feasible from the distance, velocity, and time inputs.

If you want **first arrival exactly at the requested time**, that is a different
task. A future version should measure first-arrival time and score a time window.
A dense penalty for early occupancy alone can lead to waiting just outside the
goal boundary. I have not added that ambiguous objective to the default task.

## Curriculum

We sample each goal as an independent uniform XYZ offset from the actual reset
position. XY forms a square; Z forms a smaller vertical interval. We do not clip
goals onto the safety boundary. Invalid goal boxes fail validation before startup.

| Difficulty | XY half-width, m | Z half-width, m | Nominal sampled time, s | Reference speed, m/s | Reference acceleration, m/s² |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 0.25 | 0.10 | 2.5–3.5 | 0.4 | 0.8 |
| 1 | 0.50 | 0.20 | 2.0–3.0 | 0.7 | 1.5 |
| 2 | 1.00 | 0.35 | 1.5–2.5 | 1.2 | 3.0 |
| 3 | 2.00 | 0.60 | 1.0–2.0 | 2.0 | 5.0 |
| 4 | 3.00 | 0.80 | 0.7–1.8 | 3.0 | 8.0 |

To avoid combining the furthest corner with an arbitrarily short time, the final
budget is at least `distance / reference_speed` and
`2 * sqrt(distance / reference_acceleration)`. The latter is the time for an
ideal symmetric accelerate/brake profile starting and ending at rest. These
are **sampling heuristics**, not physical controller limits or a reachability
certificate. Real Borinot must tilt, handle its arm, and respect thrust limits.
The floor can extend a budget beyond the table's nominal upper endpoint.

The configurable arrays are `goal_half_extents_m`, `goal_deadline_range_s`,
`goal_reference_speed_m_s`, and `goal_reference_accel_m_s2`. Defaults are a
starting recipe, not a tuned learning result. For a difficult last stage,
increase its times or reduce the reference speed/acceleration before concluding
that a controller cannot perform the task.

A success means: no safety failure, and distance ≤ `goal_success_tolerance_m`
(default **0.15 m**) at the deadline. It does not depend on reward weights,
minimum distance earlier in the flight, or the time of first entry.

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

| Global stage | Goal difficulty | Progress factor |
| ---: | ---: | ---: |
| 0–4 | 0–4 | 1.0 |
| 5 | 4 | 0.75 |
| 6 | 4 | 0.50 |
| 7 | 4 | 0.25 |
| 8 | 4 | 0.00 |

We first learn the full goal/time distribution. We then require fresh success
at each reduced progress weight. There is no automatic fade based only on
training iteration, and no immediate removal at the first easy success.

Useful overrides:

```text
env.curriculum=False env.goal_curriculum_start_level=4
env.skip_curriculum=True
env.progress_reward_curriculum=False
env.rwd_scale_progress=0.0
```

The first fixes the hardest geometry with full progress. The second starts at
the last stage, including zero progress. The third keeps full progress and
removes the four fade stages. The fourth is a sparse-reward ablation from the
chosen stage. `curriculum_start_level` belongs to the unused inherited push
curriculum; use **goal_curriculum_start_level** for this task.

Policy/optimizer checkpoints do **not** restore the evidence window, stage age,
or unfinished flights. To resume training at a chosen difficulty, set
`env.goal_curriculum_start_level=N` explicitly. The training launcher keeps its
existing stage-best checkpoint metadata. The player reads that metadata when
present; ordinary periodic checkpoints may need `--curriculum-level=N`.

## Optional additions: off by default

| Switch / scale | Behavior |
| --- | --- |
| `env.use_terminal_speed_penalty=True` | Adds `rwd_scale_terminal_speed * speed²` at a clean deadline; default scale -1.0. This favors arriving with low base-CoM velocity. Success still uses position only. |
| `env.use_terminal_success_bonus=True` | Adds `rwd_scale_terminal_success` at a successful deadline; default scale +10.0. This sharpens the importance of entering the 0.15 m tolerance. |
| `env.rwd_scale_action_rate` | Existing normalized physical command-change cost, default 0.0, multiplied by policy dt. |
| `env.rwd_scale_energy` | Existing modeled mean-power cost, default 0.0; no extra dt, matching the shared implementation. |

Inherited collision/tilt and raw-action-limit reward scales are also zero by
default. Their safety reset conditions are separate. Do not turn on several
regularizers at once for the first controller comparison. The terminal-speed
switch is the first useful ablation if a position-only policy learns fly-throughs.

## Playback

The player flies to a goal position that you choose, live. It opens a window
with X/Y/Z goal fields and sliders, a deadline field, and buttons. Give the
first goal with `--goal_init X Y Z` (alias `--goal`). The goal is in world
metres, in the same frame as `env.robot_home`: z is the height above ground.
The default goal is the robot home, so the robot first hovers in place.

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goto.py --checkpoint=/absolute/path/model.pt --num_envs=1 --duration_s=2000 --goal_init 3.0 0.0 1.5 env.control_mode=model_wrench env.use_terminal_speed_penalty=True
```

How a goal works in the player:

- **Deadline.** By default the deadline is automatic: the middle of the
  deadlines that training assigns to this distance at the checkpoint stage.
  `--deadline_s` or the deadline slider sets a fixed time instead. Moving the
  slider switches Auto off.
- **Go to goal.** Editing a field shows a yellow sphere: the goal is not sent
  yet. **Go to goal** sends it, and the green sphere (success radius) moves
  there. **Fly home** sends the robot home.
- **Chain mode (default).** A new goal starts from the current state. The
  robot is not teleported. The goal clock restarts, so the policy sees the new
  body-frame offset, a full `t_norm` and the new remaining seconds.
- **Hold.** At a clean deadline the robot keeps the same goal. A hold segment
  starts by itself, with the shortest deadline that training assigns to the
  current distance (1.0 s at the final stage). Stage rewards pay only the
  position (and the optional speed) at the deadline. So the policy may loop
  away and come back between deadlines, and the loop grows with the time left.
- **Failures.** Crash, tilt or leaving the area still resets the robot to home.
  Then it flies the same goal again.
- **`--reset-at-deadline`.** Training episodes: every deadline resets the robot
  to home and repeats the goal. A new goal also starts a new episode from home.
- **`--random-goals`.** Training goal and deadline sampling at the checkpoint
  stage, with a reset at every deadline. No goal window.

Each finished goal prints one line, for example `[GOAL] ... SUCCESS. Final
distance 0.004 m, speed 0.05 m/s ...`. The run ends with one
`BORINOT_GOTO_PLAY {...}` JSON line with totals for all robots. The window
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
the slider ranges; the default is home ± the stage goal box.

Playback freezes curriculum. The stage comes from the checkpoint metadata
(`best_model_curriculum_idx`), unless `--curriculum-level` or
`env.goal_curriculum_start_level` selects one. `--duration_s` is how long the
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

Episode logs include terminal distance, clean-deadline distance and speed,
initial/minimum distance, first-arrival time when reached, energy, and counts
of completed tasks, successes, clean deadlines, and early failures. The public
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

For a fair controller comparison, evaluate the same fixed goals and budgets,
seeds, model/randomization settings, and safety limits. Record success at the
deadline, deadline distance, speed, and failures. First touch alone is not a
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
controller stepping, and controlled curriculum promotion. It writes a numeric
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
