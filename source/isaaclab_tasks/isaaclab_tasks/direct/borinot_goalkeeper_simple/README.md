# Borinot goalkeeper — simplified reward

Date: 2026-10-18. Task ID: `Borinot-Goalkeeper-Simple-Direct-v0`.

## Start here

**Shared physical-model defaults:** motor point masses and inference leg
inertial properties are enabled here too. Read
[shared inertial properties](../borinot/INERTIAL_PROPERTIES.md) for the independent
switches, payload precedence and older-checkpoint opt-outs.

In `thrust_and_torque_control`, PPO's last two actions are joint torques in N·m, passed through without scaling or action clipping. `rwd_scale_torque_limits` penalizes squared relative excess beyond ±`effort_lim_sim` (default ±2.7 N·m); it does not hard-clamp the request. SAC keeps normalized [-1, 1] joint actions multiplied by `effort_lim_sim`. Rotor mappings, PD/PI/INDI modes and neutral arm hold are unchanged. Older direct-torque PPO checkpoints need their original source revision because their joint actions used different units. SAC checkpoints remain compatible.

**New controller:** `env.control_mode=indi` selects the shared nine-action INDI + joint PID interface. Both low-level controllers run at physics rate during training and playback. PPO and both SAC critics are supported. Read the original task's [step-by-step INDI guide](../borinot_goalkeeper/INDI_CONTROL.md) for the full design and commands.

This is a new task next to the original `Borinot-Goalkeeper-Direct-v0`. The original task's training code, config and rewards are unchanged. Only its playback file was split into a reusable part, so that one player can show both tasks. Playback of the original task behaves as before.

The robot, racket, ball, goal, shots, curriculum, control modes and actor observations are the same. Only the reward and the reset rules change. The new reward is a plain **sum** of a few terms in three groups:

```text
reward = task + integrity + desirability
```

- **Task:** stop the ball. Pay for the ball's first robot contact and penalize a goal.
- **Integrity:** keep the robot safe. Penalize base crashes, self-collisions, leaving the allowed area and tilting too far.
- **Desirability:** prefer smooth, low-energy motion.

There is no priority list and no "selected outcome". If several events happen in the same step, all of them add up. Every contact or goal is paid **at most once per ball**.

### Train

Activate the environment as in the original README, then run, for example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Simple-Direct-v0 --num_envs 512 --max_iterations 10000 --headless --logger tensorboard '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Your usual cluster options work unchanged, because the non-reward settings keep their names. For example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Simple-Direct-v0 --headless --num_envs 10000 --logger wandb --log_project_name borinotIsaacLab env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 'env.robot_home=[0.0,0.0,1.5]' env.paddle_mount_delta_m=0.07 'env.ball_mass_range_kg=[0.050,0.200]' env.thrust_to_weight=3.5
```

SAC uses the same entry points as the original task: `--agent rsl_rl_sac_cfg_entry_point` or `--agent rsl_rl_sac_mse_cfg_entry_point`. Logs go to `logs/rsl_rl/borinot_goalkeeper_simple` (PPO) and `logs/rsl_rl/borinot_goalkeeper_simple_sac` (SAC).

**SAC CE value range.** The CE critic predicts values on a fixed grid from −20 to +20 (`distributional_linear_limit=20`). Targets outside the grid are clipped to its edge. Some situations in this task are worse than −20:

- a self-contact that lasts more than about 8 policy steps (−2.5 each);
- a crash, area exit or tilt with its reset switched off (−5 each step). The tilt reset is off by default, so a robot that stays tilted for more than 4 policy steps is already below −20.

The energy option pays about −3 to −4 per shot (section 3), which fits inside the grid. The critic sees −20 for all the cases above. It still learns to avoid them, but it cannot tell how bad they are. If this matters, widen the grid, for example `agent.critic.distributional_linear_limit=50.0`, or use the MSE entry point. PPO has no such limit.

### Play

The goalkeeper player now accepts both tasks:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --task Borinot-Goalkeeper-Simple-Direct-v0 --num_envs 1 --checkpoint /absolute/path/model_N.pt env.shot_level=5
```

The match panel counts **goals** and **blocked** shots. In this task, "blocked" means the shot ended without a goal.

Checkpoints trained in the original task do not load here. The actor inputs are the same, but the critic input is two values shorter (see "Observations").

## 1. Task rewards

| Config field | Default | When it is paid |
|---|---:|---|
| `rwd_scale_hit_racket` | +5.0 | The ball's first robot contact is the racket. |
| `rwd_scale_hit_arm` | +2.5 | The first contact is the thigh or calf link (half the racket value). |
| `rwd_scale_hit_base` | +1.25 | The first contact is the base (a quarter of the racket value). |
| `rwd_scale_goal` | −5.0 | The ball touches the goal or crosses its front plane inside the goal. |
| `rwd_scale_progress` | +2.5 | Per metre that the racket gets closer to the ball in the y–z plane. |

**First contact only.** Each ball pays one contact reward: the first robot body it touches. A long contact pays once. Later contacts of the same ball pay nothing, whatever body they touch. If two bodies are touched in the same physics step (5 ms at 200 Hz), the racket wins, then the arm.

**Goal once.** A goal pays −5 once and ends the shot. It is added to any earlier contact reward of the same ball. For example, a racket touch followed by a goal gives +5 − 5 = 0. A base touch followed by a goal gives +1.25 − 5 = −3.75. Contacts after a goal are ignored.

**When a shot ends.** There is no "clearance" rule anymore. A shot ends at the first of:

1. a goal (−5);
2. the ball passes the goal's front plane outside the goal (it can no longer score);
3. the ball leaves the tracking region (|x| > 3 m, |y| > 3 m or z > 4.5 m);
4. `shot_timeout_s` (3 s) after the launch.

Cases 2–4 pay nothing extra. A shot that ends without a goal counts as a **save** for the curriculum, the logs and the match panel. A ball that rolls into the goal after a weak touch still counts as a goal, because it really scored.

**Progress.** We compare the racket's y–z distance to the current ball position at the start and at the end of each policy step:

```text
progress = rwd_scale_progress * (d_before - d_after)
```

Both distances use the same ball position, so a racket that does not move earns zero. Progress is paid only while the ball is live and has not touched the robot yet. The step that contains the first contact still pays its progress.

## 2. Integrity rewards

| Config field | Default | When it is paid |
|---|---:|---|
| `rwd_scale_base_crash` | −5.0 | The base touches the floor or the goal. |
| `rwd_scale_self_collision` | −2.5 | Two robot bodies touch each other (half the crash value). |
| `rwd_scale_leave_goal_area` | −5.0 | The base origin is outside the goal-area box. |
| `rwd_scale_tilt` | −5.0 | The base is tilted more than `robot_tilt_threshold_degs` (70°) from vertical. |

These penalties are paid on **every policy step** that has the event. With the default resets, a base crash or leaving the area ends the episode, so it is paid once. The tilt reset is off by default, so a tilted robot pays −5 on every step until it is upright again. Self-collisions do not reset by default, so a racket pressed against the base keeps costing −2.5 per step. At 100 Hz that is −250 per second of contact. This is on purpose: it discourages the "racket resting on the base" pose seen in recent policies. The original task also charged every step, as a robot crash of −1, but only for contacts that involve the base. In a 20-iteration PPO check with a random starting policy, self-contacts cost only about −0.1 to −0.3 per episode, so they do not swamp early learning.

If you switch a reset off, its penalty is still paid on every step. For example, with `reset_robot_on_robot_crash=False` a base lying on the floor costs −5 per step.

**Base crash** means the base against the scene only. A ball hitting the base is a task event (+1.25), not a crash. A self-collision is not a crash either. Leg and racket contacts with the floor are allowed and cost nothing.

**Self-collision** covers every robot pair that can touch. PhysX never reports pairs joined by a joint (base–thigh, thigh–calf, calf–racket). The remaining pairs are base–calf, base–racket and racket–thigh. We measured all three in the simulator; see section 7. This needs physical self-collisions, so this task sets `enabled_self_collisions=True` by default (the original task defaults to `False`).

**Leaving the area** keeps the original box: x in [−0.08, 0.90] m, y in [−1.2, 1.2] m, z in [0.07, 2.6] m, measured at the base origin. The original sketch had this as a reset only. We added the penalty for one reason: an episode that ends without a penalty is an escape. The robot could fly 8 cm backwards out of the box just before a goal and avoid the −5. With −5 for leaving, an escape costs as much as the goal it avoids. It only pays off if the robot expects the rest of its episode to be negative.

**Tilt** has a penalty for the same escape reason. Run `inwdqoc1` (2026-10-19) trained with `reset_robot_on_tilt_threshold=True` and no tilt penalty. Its policy leaned into the ball, hit it with the racket (+5) and tipped past 70° right after. The free reset ended every episode after about 0.34 s. With `gamma=0.99` at 100 Hz, the next shot (2–3 s later) is worth almost nothing, and the reset also avoids a later goal (−5). With −5 for the tilt, this trick gains about nothing: the hit and the penalty cancel. The tilt event is the same with or without its reset.

Contact thresholds: ball contacts use `contact_force_threshold_n=0.05` N. Base-crash and self-collision contacts use `robot_crash_force_threshold_n=0.1` N.

## 3. Desirability rewards

| Config field | Default | Signal |
|---|---:|---|
| `rwd_scale_action_rate` | −5.0 | Same definition as the original task, times the policy step `dt`. |
| `rwd_scale_energy` | −2.5e-5 | Mean mechanical plus dissipative power of each step, in watts. |
| `minimize_energy_not_power` | `False` | `True`: pay the energy of each shot once, when the shot ends. |

**Action rate.** The original definition is kept exactly:

```text
action_rate = mean over 6 rotors of (delta thrust command / F_max)^2
            + mean over 2 joints of (delta joint effort / (2 * effort_lim_sim))^2
reward      = rwd_scale_action_rate * action_rate * step_dt
```

Commands are compared after clipping, at successive policy steps.

**Power (default).** Each policy step pays `rwd_scale_energy * P_mean`. `P_mean` is the average over the step's physics substeps of rotor mechanical power, rotor losses, signed joint mechanical power and joint copper losses. This is the same model and the same per-watt scale as `rwd_scale_energy` in the original task.

**Energy (option).** With `env.minimize_energy_not_power=True`, energy is paid once per shot instead of on every step. We add up the energy of every physics substep since the last payment:

```text
E = sum over the physics substeps k since the last payment of P(t_k) * physics_dt      [J]
```

A payment window starts at the episode start or at the previous payment. It ends on the step where the shot ends (goal, ball past the goal plane, ball out of the tracking region, or shot timeout). So one window covers the wait before a launch plus the shot itself. That step pays:

```text
r_shot = rwd_scale_energy * E / step_dt
```

Dividing by `step_dt` keeps the same per-watt weight as the power mode. Each step's energy is paid once, so the **undiscounted** sum is the same in both modes, except for the time-limit case below.

With the default `shot_resampling_mode="launch"`, a ball is launched every 2–3 s, so a window lasts 2–3 s. At about 550 W while flying, one payment is about −2.75 to −4.1. In `"resolution"` mode the wait comes after each shot, so a window can last about 6 s (about −8).

**If the episode ends in the middle of a window:**

- A **termination** pays the unpaid energy on its last step. This covers a crash, an area exit, any other enabled reset, and the time limit when `is_finite_horizon=True`. Nothing comes after a termination, so this is the only chance to charge that energy. It also means that a crash cannot erase the energy the robot already used.
- The **time limit** (default `is_finite_horizon=False`) pays nothing for the unfinished window. Here the time limit is a cut, not a real end. PPO and SAC treat the episode as if it would continue, and add the critic's estimate of the future reward. That estimate already includes the payment that the unfinished shot would make later. Paying it at the cut would charge it twice.

Because of the time-limit rule, `Episode_Reward/desirability/energy` misses the last unfinished window of a time-limited episode. On average this is half a window: about −1.7 of about −27 in a 20 s episode. To compare the physical energy of runs, use `Episode/energy_j`. It always counts every step.

**Why one payment per shot.** A payment `T` steps in the future is weighted by `gamma^T`. PPO and SAC both use `gamma=0.99` by default. The table gives this weight for the first step of a window, and its mean over all steps of the window, at 100 Hz:

| Window length | First step | Mean |
|---|---:|---:|
| 2 s | 0.14 | 0.43 |
| 3 s | 0.05 | 0.32 |
| 20 s (one payment per episode, for comparison) | 2e-9 | 0.05 |

In the power mode the weight is always 1, because each step pays its own energy. With one payment per shot, no step is more than about 3 s from its payment, and the mean weight is 6–9 times higher than with one payment per episode. With Solo's SAC value `agent.algorithm.gamma=0.97`, the horizon is only about 1 / (1 − 0.97) ≈ 33 steps, or 0.33 s. The mean weight then drops to 0.11–0.17, and the energy of the first second of a shot is almost invisible.

**Inside a window, timing does not matter.** All energy of one window is paid at the same moment. So the robot gains nothing by moving its energy use to later in the shot. It pays for how much energy the shot used, not for when it used it. This is the "energy, not power" idea. In the power mode, energy used later is discounted more, so the cost also depends on when it is used.

**Hidden state.** The size of a payment depends on the energy used since the last payment. The critic does not observe this number. It does see the shot time and the time to the next launch, so it can predict the average payment, but not its exact size. Windows are short, so this noise is small. If it matters, the unpaid energy could become one more critic input.

## 4. Resets

| Reset source | Config field | Default |
|---|---|---|
| Excessive tilt | `reset_robot_on_tilt_threshold` (`robot_tilt_threshold_degs=70`) | `False` |
| Leaving the area | `reset_robot_on_leave_area` | `True` |
| Base–scene contact | `reset_robot_on_robot_crash` | `True` |
| Robot–robot contact | `self_collision_triggers_reset` | `False` |
| Goal | `reset_on_goal` | `False` |

The reset switches never change the rewards. The episode time limit (`episode_length_s=20`) is a truncation, as in the original task.

The design sketch listed "Base – body collision" with a different default in the typed notes (True) and the drawing (False). We followed the drawing and the voice note: self-collisions do not reset by default.

## 5. Where the default weights come from

The main events are now ±5 instead of ±1. To keep the same balance as your run `qu68pf74`, we multiplied its other weights by 5:

| Weight | qu68pf74 (original task) | This task |
|---|---:|---:|
| progress | 0.5 | 2.5 |
| action rate | −1 | −5 |
| power | −5e-6 | −2.5e-5 |

Outside the reward and reset sections, all defaults equal the original config, except `enabled_self_collisions=True`. Your usual launch overrides still apply.

## 6. Switching from the original task

Same names: control, timing, racket, ball, shots, curriculum, reset grid, area box, observation flags, `rwd_scale_progress`, `rwd_scale_goal`, `rwd_scale_leave_goal_area`, `rwd_scale_action_rate`, `rwd_scale_energy`, `reset_robot_on_robot_crash`, `reset_robot_on_tilt_threshold`, `self_collision_triggers_reset`, `reset_on_goal`.

New names:

| New field | Replaces in the original task |
|---|---|
| `rwd_scale_hit_racket`, `rwd_scale_hit_arm`, `rwd_scale_hit_base` | `rwd_scale_save`, `rwd_scale_hit`, `rwd_scale_ball_arm_collision`, `rwd_scale_ball_base_collision` |
| `rwd_scale_base_crash` | `rwd_scale_robot_crash` (base contacts only now, no tilt) |
| `rwd_scale_tilt` | the tilt part of `rwd_scale_robot_crash` |
| `rwd_scale_self_collision` | self-contact used to count as a robot crash |
| `minimize_energy_not_power` | new |
| `reset_robot_on_leave_area` | leaving the area always reset |

Removed: floor, miss and deadline penalties, the clearance rule (`clear_*`), recovery (`recovery_*`, `require_recovery_for_save`), intercept/alignment/centered-hit shaping, velocity/thrust regularizers, `progress_use_prediction`, `robot_crash_body_names`, and the unused `remove_action_clipping`. An old override for a removed field fails loudly in Hydra instead of being silently ignored.

## 7. Observations, logs and code

**Observations.** The actor inputs are identical to the original task: 44 values with the default flags (51 in PI mode). The critic's phase block has 5 values instead of 7: first contact happened, shot time, ball active, time to the next launch, and episode time. The old "cleared" and "recovery" values no longer exist. The critic therefore has 54 inputs (61 in PI mode).

Since 2026-10-19, direct modes no longer observe the 6 rotor thrusts and the 3 PI-integral values; only PI mode does. Direct-mode checkpoints trained before that (53 / 63 inputs) need the earlier commit for playback.

**Step extras.** `goalkeeper_reward_terms` holds every term, already scaled, named `task/...`, `integrity/...` and `desirability/...`. Their sum is the reward. `goalkeeper_events` holds the raw events of the step, including `tilt`. `goalkeeper_outcome` is 3 (goal) or 7 (save) on the step a shot ends, else 0; `goalkeeper_shot_finished` marks that step. `power_mean_w` is unchanged.

**Episode logs.** `Episode_Reward/task`, `Episode_Reward/integrity` and `Episode_Reward/desirability` are the episode totals per group. `Episode_Reward/<group>/<term>` gives each term. `Episode/shots`, `saves`, `goals`, `hit_racket`, `hit_arm`, `hit_base` are counts per episode, and `Episode/energy_j` is the episode energy in joules. Curriculum and power logs are as before.

**Code layout.** `borinot_goalkeeper_simple_env.py` has one method per group: `_task_rewards()`, `_integrity_rewards()` and `_desirability_rewards()`. `_sample_physics_events()` records the events after every physics substep. `_get_dones()` ends shots and applies the reset switches. Rotor/arm control and the power model are the original task's methods, reused unchanged (`_apply_action`, `_sample_power`). Shot geometry comes from the original `task_math.py`, and the racket from `paddle.py`.

Contacts use three filtered sensors: ball against racket/thigh/calf/base/goal, base against floor/goal/calf/racket, and racket against thigh.

**Verification (2026-10-18, laptop, Isaac Sim 5.1, GPU PhysX).**

- 53 CPU tests run the real reward, event, reset and observation methods on controlled tensors. Command:

  ```bash
  python -m pytest -q --import-mode=importlib --rootdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests/test_simple_reward.py
  ```

- `tests/simple_smoke.py` checks real contacts, goals, resets and the energy logic through Hydra. It passed all 28 checks in power mode and all 29 in energy mode. In energy mode it checks every step against a reference: a payment exactly when a shot ends or the environment terminates, and nothing at the time-limit cut. Command (add `env.minimize_energy_not_power=True` for the second run):

  ```bash
  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests/simple_smoke.py --headless --output /tmp/simple.json
  ```

- Self-collision pairs: we imposed 325 arm poses (a 15° grid) and stepped PhysX once. The first step reported base–racket contact in 127 poses, base–calf contact in 41 and racket–thigh contact in 42. Racket–thigh contact appeared only with the elbow at 165° or more.
- The training and playback checks below ran before the per-shot energy payment. That change touches only the energy term of the energy option.
- Training: PPO ran 20 iterations with 256 environments (about 11 500 steps/s). SAC CE and SAC MSE each ran 4 iterations with small laptop batches. The default SAC batch sizes ran out of memory on the 8 GB laptop GPU; they are sized for the cluster.
- Playback: the shared player ran 300 steps with the PPO and SAC CE smoke checkpoints of this task. It also ran 300 steps of the original task with checkpoint `1018_qu68pf74_best_model_curriculum_idx_5.pt`.
- The original task's 298 CPU tests and its 84 playback physics checks still pass.

- Tilt penalty (2026-10-19): the 53 CPU tests check −5 per tilted step, with and without the tilt reset, and the `Episode_Reward/integrity/tilt` log. The power-mode simulator smoke passed all 29 checks, including a real 80° tilted base that pays only −5 and an upright robot that pays nothing.

This shows that the reward is computed as designed and that training and playback run. It does not show that a policy learns the task well; no full training run has been done yet.

**Files.**

- `borinot_goalkeeper_simple_env_cfg.py`: all settings; the reward and reset section is at the end.
- `borinot_goalkeeper_simple_env.py`: events, shots, rewards, resets, observations and logs.
- `playback.py`: playback class for the shared goalkeeper player.
- `agents/`: the original PPO and SAC recipes with their own log names.
- `tests/test_simple_reward.py`, `tests/simple_smoke.py`: CPU and simulator checks.


## Soft action and controller-output limits

All four Borinot tasks score raw direct policy outputs before environment clipping.
Goalkeeper, goalkeeper simple and command velocity also score pre-saturation
controller demands in PD, PI and INDI modes, for both PPO and supported SAC modes.
The new signed scales default to **-1.0**. Set all three to `0.0` to disable them.
Each term is a sum over channels, with no extra policy-dt multiplier.
Thrust and torque use `(excess / upper limit)^2`, so these costs are dimensionless.
Both lower- and upper-bound excesses use the same positive upper-limit scale.
Equal relative violations now have equal per-actuator costs; six rotors still
contribute three times as much as two joints when every channel violates equally.

- `env.rwd_scale_thrust_limits`: squared relative excess of requested per-rotor thrusts.
  Requests below zero also pay a cost. The upper threshold uses the effective
  thrust-to-weight ratio after the per-rotor hardware cap is resolved.
- `env.rwd_scale_torque_limits`: squared relative excess of requested joint torques
  beyond either signed `effort_lim_sim` limit. At the default 2.7 N·m limit,
  a 3.0 N·m request contributes `(0.3 / 2.7)^2`, about 0.01235, before its signed weight.
- `env.rwd_scale_action_limits`: squared excess of other policy references that
  are actually clipped. This covers PD/PI input clamps, normalized SAC INDI
  inputs, and the legacy task's clipped angle-vector components.

The helper is `../borinot/action_limits.py`. Direct thrust/torque costs use the
original mapping and raw sample, including excursions hidden by `action_clamp`.
The legacy task uses its own thrust and torque mapping. Unused held-arm policy
outputs are ignored; the neutral hold controller can still incur a physical
torque-demand cost when it saturates. An explicitly removed upper thrust cap has no upper-thrust cost
for PPO, but negative thrust requests still pay a cost, normalized by the nominal
upper thrust scale. SAC's direct rotor
mapping still clips its normalized inputs to ±1.

Each scaled limit term is then clipped to ±`env.rwd_limits_max_penalty`
(default 10.0) in every policy step. So one huge request costs at most 10 per
term and step. The clip acts on the reward term, not on the raw cost. Use
`env.rwd_limits_max_penalty=inf` to turn it off.

PPO INDI references are unbounded physical quantities; they are **not** given
artificial ±1 limits. Their physical controller demands now pay thrust/torque
excess costs. Bounded SAC references can also make a controller saturate.

| Control mode | Thrust-demand source | Joint-torque-demand source |
| --- | --- | --- |
| Direct thrust/torque | Raw policy mapping, before input clipping | Raw direct torque mapping |
| Direct thrust + PD position | Raw policy thrust mapping | Unsaturated implicit PD demand estimate (`computed_torque`) |
| PI flight + PD position | Unclipped allocation of collective thrust and PI body torque | Unsaturated implicit PD demand estimate |
| INDI | Unweighted G1 pseudoinverse of the original requested wrench, before feasibility projection | Explicit PID demand after anti-windup, before torque saturation |

Each controller contribution is the mean of the **per-substep costs**, not the
cost of the mean demand. It is added once per policy step, without another dt
multiplier. Direct policy costs keep their existing once-per-policy-step timing.
The code selects one source per actuator group, so it does not count the same
thrust or torque demand twice. `rwd_scale_action_limits` stays independent and
covers only actual input clamps; it remains zero for physical-unit PPO INDI.

Borinot's G1 has four rows and six rotor columns. Its pseudoinverse gives one
minimum-norm rotor distribution. A violation in that distribution is the chosen
soft cost, not proof that no other bounded rotor distribution can produce the
same wrench. A bounded-allocation wrench residual would be a different objective.
The constrained allocator still supplies all actual INDI motor commands.
Implicit PD demand is a model estimate, not measured PhysX drive torque.

The physical controls, clipping, motor dynamics and observation sizes are
unchanged. In particular, this is a **soft reward cost**, not a new hard torque
clamp. Direct PPO feed-forward requests can exceed the configured implicit
actuator limit even when the reported torque estimate is clipped.

The reward breakdown and episode logger expose `thrust_limits`, `torque_limits`
and `action_limits` (under `desirability/` in the simple task). The legacy task
uses `rewards/<term>` and `info["action_limit_reward_terms"]`. Physical resets
clear only the selected environments' cost buffers; ball-only resets keep the
current robot transition. Tests: `borinot_goalkeeper/tests/test_action_limits.py`
`borinot_goalkeeper/tests/test_controller_limits.py`, and
`borinot_goalkeeper/tests/action_limits_smoke.py`.


## Command delays and evaluation

Use `env.actuator_delay=0.02` for a fixed 20 ms policy-command delay, or
`env.actuator_delay=0.0 'env.actuator_delay_range=[0.0,0.04]'` to sample a new
delay per robot episode. Defaults add no delay. Inner controllers still use
current measurements. See [timing, training and evaluation commands](../borinot_goalkeeper/ACTUATOR_DELAY.md)
for the shared conventions and the fixed-stage goalkeeper checkpoint evaluator.
