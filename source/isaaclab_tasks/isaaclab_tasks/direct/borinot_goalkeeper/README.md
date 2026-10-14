# Borinot goalkeeper — first Isaac Lab prototype

Date: 2026-10-14. Task ID: `Borinot-Goalkeeper-Direct-v0`.

## Start here

This is a separate task. It does not change the existing Borinot locomotion task or the source USD.
We start with privileged simulator information. Both the actor and critic receive true robot state, ball position and velocity, and the sampled ball mass and restitution. There is no camera or state estimator yet.

The default reward follows Jordi's requested approach. Goal and base hits cost −1. Arm hits cost −0.25. Leaving the goal area costs −1. Moving the racket closer to the ball in the goal-parallel plane earns signed progress. A verified racket save earns +1. Additional shaping and the recovery requirement are off by default.

Use **PPO** first. We can change the algorithm later without changing what counts as a goal or save.

### Activate the local environment

Run this once in your terminal:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Train on the laptop

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Borinot-Goalkeeper-Direct-v0 --num_envs 512 --max_iterations 3000 --headless --logger tensorboard '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The task's default environment count is 2048. The command starts smaller on the laptop's 8 GB GPU. Increase it after checking memory use.
For an initial visible simulation, omit `--headless` and use fewer environments. GUI rendering has not been verified in this task's headless validation.

### Train at your usual scale

On IRICluster, activate its `isaaclab` conda environment first. After you have pulled the local code commit, use the same command with `--num_envs 10000`. We did not sync code or submit a cluster job. Ten thousand environments have not been benchmarked for this task.

### Optional W&B logging

Replace `--logger tensorboard` with `--logger wandb --log_project_name borinot-goalkeeper`.
The verification runs use local TensorBoard only. No W&B run was created for them.

### Play a checkpoint

Replace `/absolute/path/model_N.pt` with your actual checkpoint:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Borinot-Goalkeeper-Direct-v0 --num_envs 1 --checkpoint /absolute/path/model_N.pt env.curriculum=False env.shot_level=1
```

The play command uses the current source configuration, not an automatic reconstruction of an old run. Match the training options when comparing policies.

## 1. Scene and physical model

The local environment uses these coordinates:

- **x:** shot direction. The ball starts near x = 2 m and moves toward negative x.
- **y:** sideways across the goal.
- **z:** height above the floor.
- The goal is a solid rectangular surface centered at x = −0.45 m. Its width and height are 2.4 m.
- Borinot starts with its base at `(0, 0, 1.5)` m.
- At the reference home height of 1.5 m, the neutral racket center is approximately `(−0.0303, −0.0647, 1.1293)` m.
- The body starts at −90° yaw. The arm swings in the goal's Y–Z plane, and the paddle's broad face is parallel to that plane. Both joints can move the paddle across the goal without turning it edge-on to an incoming ball.
- The progress plane is y–z, parallel to the goal. Forward motion along x alone earns no progress.

```mermaid
flowchart RL
    A[Incoming ball: positive x] --> B[Racket and two-joint arm]
    B --> C[Borinot flying base]
    C --> D[Goal surface: negative x]
```

The robot comes from:

```text
/home/jordibelp/IsaacLab/source/isaaclab_assets/data/Robots/Borinot/borinotFlat.usd
```

### Paddle and mass

The original USD has a 115 g distal link. We model it as a 15 g shaft plus a separate 100 g paddle. This is a prototype interpretation of the existing tip ballast, not a measured separation of the physical parts.
The paddle is a rigid cuboid: 12 mm thick, 240 mm wide, and 240 mm high. A fixed joint connects it to the distal link. Its center has a 50 mm standoff from the old tip center along the arm-plane normal, toward the incoming ball. The local mount rotates the paddle's +X normal onto the distal link's +Z axis. The root yaw then maps this normal onto world +X. The original USD is unchanged.

The task changes mass, center of mass, and inertia in the spawned USD layer only. It never saves over the original file. The shaft inertia uses a simple 160 mm rod approximation. PhysX computes the paddle inertia from its cuboid geometry and mass. Total robot mass remains **2.910538 kg** in the simulator.

The old distal mesh remains visible and collidable. A ball hitting it counts as an arm hit, not a paddle hit. The paddle's new mass distribution is intentionally different from the old ballast's distribution, even though the total mass is unchanged.

### Ball and contacts

The ball has a 30 mm radius. Each reset samples mass uniformly from 20–50 g and restitution from 0.35–0.65. Restitution means how strongly the ball rebounds. We use the `max` material combination rule with a zero-restitution paddle, so the sampled ball value sets the ball–paddle restitution.

When mass changes, the code also changes the sphere's inertia using `I = 0.4 * mass * radius²`. The property setters change the active PhysX model, not just USD metadata. Validation reads those properties back.

Physics runs at 240 Hz. The policy runs at 60 Hz. Contacts are inspected after **every physics substep**, not only after policy steps. A ball-source contact sensor has separate filters for the paddle, base, upper arm, lower arm, goal, and floor. Each environment has its own floor and goal, including at large environment counts.

GPU PhysX does not support ordinary swept CCD in this runtime. The task uses speculative CCD and a swept geometric goal-plane check. These do not establish accurate real impact dynamics. Smaller physics steps and contact-parameter sensitivity tests are still needed before hardware transfer.

There is no ball drag, wind, racket compliance, or camera latency model yet.

## 2. Your default reward

Every reward weight uses `rwd_scale_*`. The code adds `scale * signal`: **negative scales penalize**, and **positive scales reward**. The scale holds the sign; there is no separate minus sign for costs.

| Event or term | Default contribution | Config field |
|---|---:|---|
| Ball scores a goal | −1 | `rwd_scale_goal=-1.0` |
| Ball hits the base | −1, independent of the goal scale | `rwd_scale_ball_base_collision=-1.0` |
| Ball hits the upper or lower arm | −0.25 | `rwd_scale_ball_arm_collision=-0.25` |
| Base leaves the allowed goal area | −1 | `rwd_scale_leave_goal_area=-1.0` |
| Racket prevents a goal | +1, once | `rwd_scale_save=1.0` |
| Ball touches the floor while the shot is live | −0.25, once | `rwd_scale_ball_floor=-0.25` |
| Robot crashes, shot escapes unresolved, or the deadline expires | −1 | `rwd_scale_robot_crash`, `rwd_scale_miss`, `rwd_scale_deadline` |
| Signed racket progress in y–z | `0.5 * (d_before − d_after)` | `rwd_scale_progress=0.5` |

A ball hitting the base uses `rwd_scale_ball_base_collision`. It is not a robot crash. `rwd_scale_robot_crash` applies to robot contact not accounted for by ball contact, or tilt beyond `robot_tilt_threshold_rad` (60 degrees by default). The contact check removes the ball normal-force contribution first. Goal, ball/base collision, and robot crash have independent scales. Existing outcome priority still selects one reward event.

`reset_robot_on_robot_crash=True` ends the episode after non-ball robot contact. `reset_robot_on_tilt_threshold=True` ends it after excessive tilt. These switches work independently. Setting one to `False` disables only that reset, not the crash reward. A persistent violation can therefore incur the crash reward in each action interval. Other endings, including leaving the allowed area and the shot deadline, still reset the environment. Per-step `goalkeeper_robot_contact_crash` and `goalkeeper_tilt_threshold` identify the two detected causes before automatic reset.

Both initial spawn and every reset use `robot_home_quat` in wxyz order. The scene inspector reads the same quaternion. The fixed shot center remains at the reference height; lowering `robot_home` does not lower the ball targets or the allowed base-height range.

The event contributions are not multiplied by dt. Progress is a displacement, so it is not multiplied by dt either. Optional rate rewards are multiplied by the policy duration, 1/60 s.

### What exactly is d?

Within each transition, we project the current ball target and the racket positions onto y–z. We measure the previous and current racket positions against the **same target**. This matches the structure of the Solo12 race gate-progress reward.

```text
d_before = distance(previous racket yz, current ball yz)
d_after  = distance(current racket yz, current ball yz)
progress = rwd_scale_progress * (d_before - d_after)
```

For example, reducing distance from 0.30 m to 0.20 m earns `0.5 * 0.10 = +0.05`. Moving away earns a negative value. Changing only x earns zero. A stationary racket earns zero even when the ball moves.

Progress is active while the ball is incoming and before the first paddle hit. We do not carry a distance difference across resets. This is a practical moving-target shaping reward. It is **not** a claim of guaranteed policy invariance.

### What counts as a save?

A save must include a PhysX paddle contact. Proximity alone is insufficient. No base or arm hit is allowed. The robot must remain inside the allowed area and avoid a crash.
After that contact, one of these outcomes confirms a save:

1. The ball moves back toward the shooter and passes 0.25 m in front of the defense plane, with x velocity above 0.3 m/s.
2. The ball passes the goal plane outside the goal surface, including its radius.
3. The ball lands on the floor **in front of the goal** after the paddle deflects it.

The third outcome earns +1 and still pays the −0.25 floor penalty. Its event subtotal is **+0.75**, before signed progress. A floor contact without a paddle hit earns no save. Once a ball is safely cleared, later out-of-play floor contacts do not incur another penalty.

A paddle touch followed by a goal is a failure. Any failure overrides a save in the same transition. We use one failure reward label, so a simultaneous base hit and goal do not produce two −1 event penalties. Signed progress remains a separate term.

The default ends the episode when the shot outcome is confirmed. It does not require post-save recovery. The three-second limit is a real task deadline, not a trajectory truncation; PPO does not bootstrap across it.

### Allowed goal area

The area is defined by the **base origin**, relative to its environment:

```text
x: [-0.08, 0.90] m
y: [-1.20, 1.20] m
z: [ 0.55, 2.60] m
```

Tune `goal_area_x_m`, `goal_area_y_m`, and `goal_area_z_m`. The constraints do not require every point on the racket to stay inside that box.

### Tune the main reward

For example, append these overrides to the training command:

```bash
env.rwd_scale_progress=1.0 env.rwd_scale_save=1.5 env.rwd_scale_goal=-1.0 env.rwd_scale_ball_base_collision=-1.0 env.rwd_scale_ball_arm_collision=-0.4 env.rwd_scale_leave_goal_area=-1.0 env.rwd_scale_ball_floor=-0.25
```

Use floats such as `1.0` for float config fields. The defaults are starting values, not an optimized reward recipe. If the agent chooses arm hits to avoid a goal, make `rwd_scale_ball_arm_collision` more negative while keeping its magnitude smaller than `rwd_scale_ball_base_collision`.

## 3. Action and observation contract

The six actions are clipped to `[-1, 1]`:

| Index | Meaning |
|---:|---|
| 0 | Collective thrust around hover: `weight * (1 + 0.7 * action)` |
| 1–3 | Body-rate targets, scaled by `(2.0, 2.0, 1.5)` rad/s |
| 4–5 | Shoulder and elbow angle targets, scaled by `(1.1, 1.4)` rad |

A 240 Hz body-rate PI controller converts rate errors into torques. A six-rotor allocation matrix converts collective thrust and torque into rotor thrusts. Rotor thrust is nonnegative and limited to twice the nominal per-rotor hover thrust. A first-order filter uses a 40 ms time constant. These are simulator assumptions, not calibrated ESC or motor limits.

The arm uses implicit joint PD, with stiffness 35, damping 0.6, effort limit 2.7 N·m, and velocity limit 15 rad/s. This policy interface differs from the existing task's ten raw rotor/angle actions. Old locomotion checkpoints are not compatible.

The actor and critic each receive **55 values**. These contain the home-position error, full root quaternion, base linear and angular velocities, joint positions and velocities, relative paddle position and normal, relative ball position and velocity, the predicted intercept and its time/validity, shot phase and elapsed time, current actions, rotor thrust state, controller integral, and the sampled ball mass/restitution.

The predicted intercept is available as privileged information. It does not replace the current-ball target in the default progress reward. All coordinates remove the clone's world offset. The finite shot clock is independent of the generic runner's randomized initial episode-length counter.

## 4. Difficulty and curriculum

Every shot is ballistic and aimed at the defense plane. Tests check that unblocked shots also threaten the goal and do not hit the floor first. This avoids training on harmless throws.

| Level | Sideways target range around the neutral racket | Height range | Flight time to defense plane |
|---:|---:|---:|---:|
| 0 | ±0.045 m | ±0.045 m | 0.42–0.50 s |
| 1 | ±0.20 m | ±0.12 m | 0.35–0.50 s |
| 2 | ±0.40 m | ±0.22 m | 0.30–0.46 s |
| 3 | ±0.65 m | ±0.32 m | 0.25–0.42 s |

These are uniform target offsets around the configured center. Level 0 is intentionally easy: neutral hovering can cover its small spread. It checks geometry and flight stability, not agile goalkeeping.
The sampling checks prove that throws threaten the goal, not that Borinot can reach every throw. Some fast, wide level-3 shots may be dynamically unreachable. Check that limit before interpreting a learning plateau. The two arm joints mainly move the tip in the body's x–z plane; sideways coverage also needs aircraft motion or reorientation.

Automatic promotion uses the last up-to-2048 completed episodes from the current level. It requires at least 1024 completed episodes, a save fraction strictly above 0.7, and 3000 policy steps at that level. The latter is **50 simulated seconds per environment**. More environments increase sample count, not elapsed simulated time.

A shot keeps its starting level until it ends. Older-level endings are excluded from the new level's decision window. Promotion clears that window. An empty batch cannot promote. General outcome counts still include every completed shot.

For a fixed-level experiment, append:

```bash
env.curriculum=False env.shot_level=2
```

Checkpoint resume restores the policy and optimizer, but the standard runner does not save this environment's curriculum window or active level. Set `env.shot_level=N` explicitly when resuming a later level. The dwell and evidence window restart. Do not infer restored curriculum state from the checkpoint iteration number.

## 5. Optional additions — off by default

These are separate experiments, not the main reward:

- `progress_use_prediction=True`: measure progress toward the ball's predicted defense-plane intersection rather than its current projected position.
- `rwd_scale_intercept>0`: reward staying near the predicted intercept. The kernel is `exp(-(distance / intercept_sigma_m)²)`.
- `rwd_scale_alignment>0`: add squared paddle-normal alignment with the incoming ball direction, weighted by the interception kernel.
- `rwd_scale_hit>0`, `rwd_scale_centered_hit>0`: pay a first-hit bonus, optionally larger near the paddle center. A later goal still remains a failure.
- `require_recovery_for_save=True`: delay the save bonus until the robot holds the recovery conditions for 0.25 s.
- `rwd_scale_recovery>0`: give a smooth post-clearance reward for returning near home with low velocity, tilt, and joint displacement.
- `rwd_scale_action_rate`, `rwd_scale_body_angular_velocity`, `rwd_scale_body_linear_velocity`, `rwd_scale_thrust`: regularization terms. Use negative scales to penalize their signals. Their shipped defaults are zero.
- `rwd_scale_energy`: mean mechanical plus dissipative power over the action interval. Physics-rate samples include the final substep before reset. This term has no extra dt multiplier. The arm uses signed torque × joint velocity plus `arm_copper_loss_coefficient * sum(torque²)`. The rotor keeps its existing motor-efficiency model. The shipped scale is zero; use a negative value to penalize estimated electrical input. The copper coefficient is a physical model parameter, not a reward scale.

For example: `env.rwd_scale_energy=-0.0001`. These `rwd_scale_*` names replace the older mixed names; old Hydra/config field names are not aliases.

For a recovery experiment, append:

```bash
env.require_recovery_for_save=True env.rwd_scale_recovery=0.1
```

The recovery gate uses position error below 0.35 m, speed below 0.5 m/s, angular speed below 0.8 rad/s, tilt below 0.35 rad, and joint displacement below 0.5 rad. All thresholds are configurable. The gate resets whenever a condition fails. It does not enforce yaw recovery separately.

Start each ablation from the same main reward and change one addition. With ±1 event rewards, do not copy large dense reward weights from another paper without rescaling them.

## 6. Literature and hypotheses

### What already exists

**JuggleRL** demonstrates real quadrotor ball juggling with PPO, a fixed racket, calibrated dynamics, and a low-level controller under collective-thrust/body-rate actions. It supports our choice of a mid-level interface and a PPO first baseline. It does not establish the best reward for this goalkeeper task. [Paper](https://arxiv.org/html/2509.24892v2)

**Creating a Dynamic Quadrupedal Robotic Goalkeeper** demonstrates reactive goalkeeping with a hierarchical learned planner/controller. It uses end-effector tracking, ball outcomes, and early termination. Importantly, its goalkeeping objective allows body/trunk blocking, while this task rejects it. We reuse the task-level lessons, not its motion library. [Paper](https://xbpeng.github.io/projects/A1Goalkeeping/A1Goalkeeping_2023.pdf)

**Humanoid Goalkeeper** combines position-conditioned task guidance with motion constraints and post-task stabilization. It motivates our optional predicted-intercept and recovery experiments. Its human motion priors are not necessary for this initial Borinot task. [Paper](https://arxiv.org/html/2510.18002v2)

**DSAM whole-body RL control** already coordinates a two-DoF aerial arm and flight in physical experiments. PPO outputs flight commands and joint targets, with low-level attitude and joint controllers. Active-arm aerial RL is therefore not a new contribution by itself. [Paper](https://arxiv.org/html/2512.21085v2)

**Potential-based reward shaping** explains conditions under which shaping preserves optimal policies. Our default moving-target signed-distance reward is not implemented as a matched discounted potential with terminal correction. Treat a formal potential-based variant as future work, not a guarantee of the current code. [Original paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

### A useful first research question

**Hypothesis:** coordinating the arm and aircraft improves paddle-only saves for fast or off-center shots, compared with holding the arm at its neutral targets.

The available neutral-target control is `env.hold_arm=True`. It keeps the same action/observation dimensions and PD gains, but ignores the two arm action outputs. The joints can still deflect under contact. This is **not** a mechanically welded arm. A true rigid-arm comparison and a separately tuned model-based interception controller remain proposed work.

Compare active and held-arm policies on the same recorded, held-out throws at fixed levels 2 and 3. Match paddle geometry, mass, actuator limits, observation information, reward, and interaction budget. Measure save fraction, base/arm collision fractions, goal fraction, and the distribution of robot motion. Use several training seeds. A held-arm policy matching the active arm would weaken the coordination hypothesis.

**Second hypothesis:** predicted-intercept progress helps with short flight times. Compare only `progress_use_prediction=False/True`, with every other reward weight unchanged. Better return alone is not enough: check held-out save fraction and forbidden contacts.

**Third hypothesis:** requiring recovery reduces successful-but-unstable saves. Compare recovery off/on and report both save fraction and a common post-impact stability test. The current default ends at the save, so that common post-impact evaluation needs an extended rollout; it is not supplied by the default episode outcome.

The ball is light relative to the full aircraft, although substantial relative to the paddle. Do not claim this is already a comparable-mass aerial impact benchmark. Goalkeeping and aerial contact RL both have precedents. A publication claim needs a measured advantage and a careful novelty check, including the unresolved aerial-catching leads in yesterday's review.

### If PPO struggles

First inspect physical hit attribution, sampled throws, and the outcome breakdown. Then hold the difficulty fixed. Increase training time or adjust progress/save weights before adding many reward terms.
If flight stability is the main obstacle, pretrain hovering or reuse a flight controller with the same action contract. If sparse discoveries remain the obstacle, a model-based interception teacher could generate demonstrations. SAC or a recurrent policy are later comparisons, not implemented alternatives in this task registration.

## 7. Logging, verification, and limits

`goalkeeper_outcome` in step extras records the selected reward outcome before automatic reset. With either crash-reset switch disabled, outcome 8 can be nonterminal; use `terminated` to identify actual endings. `goalkeeper_had_paddle_hit`, `goalkeeper_robot_contact_crash`, `goalkeeper_tilt_threshold`, and `goalkeeper_reward_terms` support attribution checks. `outcome_counts` pools completed episodes, not means of unequal reset-batch success rates.

TensorBoard logs episode reward components, cumulative save fraction, contact/clearance fractions, collection/decision/selected difficulty, stage age, and the stage-pure window. The standard RSL logger averages these sampled summaries within a rollout. For exact evaluation counts, collect the terminal outcome codes directly. A displayed cumulative save fraction is not a fresh held-out evaluation.

Run tensor-only tests without launching Isaac Sim:

```bash
python -m pytest -q --import-mode=importlib --rootdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_task_math.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_power.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_rewards.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_reset_geometry.py
```

Run the GPU environment check:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/smoke.py --headless --num-envs 16 --steps 180 --output /tmp/borinot_goalkeeper_validation.json '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The checks cover ballistic goal-threatening shots, progress geometry, actual paddle/base/arm/goal/floor contacts, reward scales, full observations, mass/inertia/material readback, partial resets, finite rollouts, and curriculum boundaries. They also check the corrected paddle plane at bent-arm poses, both crash-reset switches independently, and the configurable tilt threshold. The physics check uses a known 1.5 m home-height baseline, not a locally tuned lower home. Earlier short PPO training verified the registered task/agent path. These checks do not show learned agility, convergence, hardware transfer, or ten-thousand-environment throughput.

Local validation uses Isaac Sim 5.1.0, Torch 2.7.0+cu128, RSL-RL 3.1.2, and the RTX 5070 Laptop GPU. The older simulator on IRICluster needs its own launch check after you pull the code.

## Files

- `borinot_goalkeeper_env_cfg.py`: geometry, reward weights, limits, and difficulty settings.
- `borinot_goalkeeper_env.py`: control, observations, contacts, rewards, resets, and curriculum.
- `paddle.py`: task-local paddle attachment and shaft mass/inertia override.
- `task_math.py`: ballistic sampling, goal crossings, plane progress, and outcome priority.
- `agents/rsl_rl_ppo_cfg.py`: PPO networks and optimizer settings.
- `tests/test_task_math.py`: simulator-independent checks.
- `tests/test_reset_geometry.py`: signed crash/reset behavior and USD-derived arm/paddle frames.
- `tests/smoke.py`: real GPU physics and reset checks.
- `tests/checkpoint_smoke.py`: finite deterministic inference after reloading a PPO checkpoint.
