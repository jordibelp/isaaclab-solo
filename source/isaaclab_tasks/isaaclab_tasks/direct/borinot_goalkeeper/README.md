# Borinot goalkeeper — first Isaac Lab prototype

Date: 2026-10-14. Task ID: `Borinot-Goalkeeper-Direct-v0`.

## Start here

**Shared physical-model defaults:** six 5 g motor point masses and the reference
inference leg mass/CoM/inertia are now enabled for all Borinot tasks. These are
spawned-layer overrides; the source USD is unchanged. For switches, payload
precedence, nominal mass and older-checkpoint opt-outs, see
[shared inertial properties](../borinot/INERTIAL_PROPERTIES.md).

**New controller:** `env.control_mode=indi` selects a nine-action acceleration/rate/heading/joint-target policy with physics-rate INDI and joint PID. It works with PPO and both SAC critics. See the shared [step-by-step INDI guide](INDI_CONTROL.md) for equations, design choices, training/playback commands and limits.

This is a separate task. It does not change the existing Borinot locomotion task or the source USD.
We start with privileged simulator information. Both the actor and critic receive true robot state, ball position and velocity, and the sampled ball mass and restitution. There is no camera or state estimator yet. By default the actor does not see the predicted plane crossing or the shot phase; the critic always does (see the observation contract below).

Episodes now contain a continuous sequence of single-ball shots. A save or failed interception does not reset Borinot. Only configured base crashes, excessive tilt, leaving the allowed area, and the episode time limit reset the robot. The default reward follows Jordi's requested approach. Goal and base hits cost −1. Arm hits cost −0.25. Leaving the goal area costs −1. Moving the racket closer to the ball in the goal-parallel plane earns signed progress. A verified racket save earns +1. Additional shaping and the recovery requirement are off by default.

Use **PPO** first. Both Borinot tasks now share the Solo-style PPO recipe: 24 rollout steps, four minibatches, five epochs, `[512, 256, 128]` networks, AdamW, and gamma 0.99. See [PPO training guide](../borinot/PPO_TRAINING.md) for the complete cluster commands and available flags. SAC is also available; see [SAC training](SAC_TRAINING.md).

### Continuous episode and shot timing

```python
episode_length_s = 20.0
is_finite_horizon = False
shot_timeout_s = 3.0
shot_resampling_time_range_s = (2.0, 3.0)
shot_resampling_mode = "launch"
reset_on_goal = False
ball_park_position = (0.0, 0.0, -2.0)
robot_crash_body_names = ("base",)
robot_crash_force_threshold_n = 0.1
reset_robot_on_robot_crash = True
reset_robot_on_tilt_threshold = True
robot_tilt_threshold_degs = 70.0
```

The first ball launches when the robot episode resets. Each following interval is sampled independently for each environment. In **launch** mode, the interval runs from the previous launch: the default requests a new ball every 2–3 seconds. If the previous shot is still active at that time, the environment waits. It never replaces an unresolved ball just because the timer expired.

In **resolution** mode, the sampled delay starts after the previous shot ends. For near-immediate successive shots, use `env.shot_resampling_mode=resolution 'env.shot_resampling_time_range_s=[0.05,0.20]'`. A fixed interval is allowed by setting min=max. A zero delay is also allowed; the next ball still starts no earlier than the next policy action after the old shot ended.

There is exactly **one physical ball per environment**. A completed shot is scored once, marked inactive and moved below the local floor. The same ball body is reused on the next launch, with new mass/restitution. Inactive balls cannot score contacts, goals, floor events or misses. Their ball-related observations are zero. We do not create a second airborne ball. `ball_park_position` must put the whole sphere below the floor.

By default, a save, ball/base hit, ball/arm hit, goal, ball-floor failure, miss or shot timeout ends **only the current shot**. The configured reward or penalty is paid once. The robot keeps its position, orientation, velocities, joint state, actions, motor thrust state and PI integral. It continues balancing and paying any enabled rate/energy terms while waiting. Optional recovery shaping can continue after a save until the next launch.

Set `env.reset_on_goal=True` to reset the whole robot episode after a goal. The default `False` preserves continuous play. The switch works in training and playback. It does not change goal detection, scoring, penalties, or the other reset conditions. A goal reset uses the normal configured spawn grid/noise, clears robot/controller state, and immediately launches a fresh shot, as other episode resets do. With `False`, the next shot follows the configured resampling schedule.

By default, a crash means non-ball normal contact on the **base**, above `robot_crash_force_threshold_n`. Arm/paddle ground contacts do not by themselves trigger that detector. To include them, add their names to `robot_crash_body_names`: `base`, `paddle`, `flying_arm_2__link1`, `flying_arm_2__link2`. Ball-to-base and ball-to-arm impacts remain nonterminal shot failures. The contact and tilt reset switches are independent; leaving the goal area always terminates.

The private episode clock runs for 20 seconds across all shots and waiting periods. It is independent of RSL's randomized warm-start episode counter. This is an artificial collection limit by default: it returns a time-limit truncation and preserves the final observation for PPO bootstrapping. A safety failure returns a true termination. Set `is_finite_horizon=True` only if the time limit should instead be a true finite-horizon terminal state. A time limit alone has no extra shot-deadline penalty.

### Activate the local environment

Run this once in your terminal:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Train on the laptop

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Direct-v0 --num_envs 512 --max_iterations 10000 --headless --logger tensorboard '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The task's default environment count is 2048. The command starts smaller on the laptop's 8 GB GPU. Increase it after checking memory use.
For an initial visible simulation, omit `--headless` and use fewer environments. GUI rendering has not been verified in this task's headless validation.

### Train at your usual scale

On IRICluster, activate its `isaaclab` conda environment first. After you have pulled the local code commit, use the same command with `--num_envs 10000`. We did not sync code or submit a cluster job. Ten thousand environments have not been benchmarked for this task.

### Optional W&B logging

Replace `--logger tensorboard` with `--logger wandb --log_project_name borinotIsaacLab`.
The earlier control checks used local TensorBoard only. The PPO recipe checks also exercise offline W&B, which does not upload a run.

### Play a checkpoint

Use the dedicated goalkeeper player. The Solo12-specific `source/scripts/rsl_rl/play_direct_0325.py` is not compatible with this task. See [playback and camera guide](PLAY.md) for the first trained checkpoint command, camera views, and evaluation settings.

Replace `/absolute/path/model_N.pt` with your actual checkpoint:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --num_envs 1 --checkpoint /absolute/path/model_N.pt 'env.robot_home=[0.0,0.0,1.5]' env.shot_level=0
```

The play command uses the current source configuration, not an automatic reconstruction of an old run. Match the training options and network sizes when comparing policies. Playback and the checkpoint smoke script load policy weights without restoring optimizer state, so the shared AdamW training checkpoints do not conflict with the playback runner's default Adam groups.

## 1. Scene and physical model

The local environment uses these coordinates:

- **x:** shot direction. The ball starts near x = 2 m and moves toward negative x.
- **y:** sideways across the goal.
- **z:** height above the floor.
- The goal is a solid rectangular surface centered at x = −0.45 m. Its width and height are 2.4 m.
- Borinot starts with its base at `(0, 0, 1.5)` m.
- At the reference home height of 1.5 m, the neutral racket center is approximately `(−0.0303, −0.0647, 1.1293)` m.
- The body starts at −90° yaw. The arm swings in the goal's Y–Z plane. By default, the paddle's broad face is parallel to that plane. With `racket_parallel_to_arm_plane=False`, the face is perpendicular to the arm plane instead.
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
The paddle is a round ping-pong-style racket: a **160 mm diameter** head, **12 mm thick**, plus a **60 × 26 mm** handle of the same thickness. Both parts are physical collision shapes on one rigid body named `paddle`. A ball hitting either part counts as a paddle hit. The short handle clears the upper arm at neutral. It extends toward the wrist along paddle-local +Y, overlapping the head by up to 10 mm so the parts join. This is a prototype shape, not a measured commercial racket.

A fixed joint connects it to the distal link. The **head center** keeps the existing 50 mm standoff from the old tip center along the arm-plane normal, toward the incoming ball. With `racket_parallel_to_arm_plane=True`, the local mount rotates the paddle's +X normal onto the distal link's +Z axis. The root yaw then maps this normal onto world +X. Resizing does not move this center, its normal, or the fixed shot target. The original USD is unchanged.

Set `env.racket_parallel_to_arm_plane=False` to twist the **whole racket** 90° around its handle axis (racket-local +Y). Its face becomes perpendicular to the arm's sweep plane. The head center, handle direction, mount offset, total mass and shot target stay fixed. This is a fixed mount choice for an environment, not an extra action. Restart training/playback to change it. The physical face normal already feeds the observations and alignment reward, so their input sizes do not change. Old weights can still load, but the changed contact geometry and inertia orientation can require a different learned strategy; we have not trained or evaluated a policy for it.

Configure the geometry in `borinot_goalkeeper_env_cfg.py`, in metres:

```python
paddle_diameter_m = 0.16
paddle_thickness_m = 0.012
paddle_handle_length_m = 0.06
paddle_handle_width_m = 0.026
paddle_mount_delta_m = 0.0
racket_parallel_to_arm_plane = True
```

`paddle_mount_delta_m` moves the **whole racket** along the second link's length, not along the paddle normal. Positive values extend toward the calf tip along second-link local **−Y**. Negative values retract toward the wrist. Zero keeps the existing mount. The head-center offset from the second-link origin is `(x, y - delta, z)`, starting from `paddle_offset_ee=(0, -0.16, 0.029)` metres. For example, `paddle_mount_delta_m=0.04` changes that position to `(0, -0.20, 0.029)`: **4 cm farther along the calf**, with the same normal standoff and orientation. It follows the link when the joints move. The fixed shot aim does not move with it, so a shifted mount need not start at the aim center.

For launch-only overrides, append `env.paddle_diameter_m=0.12 env.paddle_mount_delta_m=0.04` to the training or player command. Restart the environment after geometry changes. The old square `paddle_size` field is removed. The browser inspector at `http://127.0.0.1:8790` has **Ping-pong racket / Ø (cm)** and **Δ (cm)** sliders and numeric inputs. These update the visible racket, collision hulls and overlap checks together. They only change the preview; copy the displayed settings into the config to use them for training. **Restore config size** and **Restore config mount** restore those values independently. Restart the viewer after editing its config.

The same panel has a **Racket face parallel to arm plane** checkbox. Checked means True/current orientation; unchecked means False/perpendicular. It updates the visible head, handle, collision hulls, overlap checks and bounds immediately. **Restore config orientation** restores only this boolean, preserving your size and mount edits. The preview does not write the environment config.

The task changes mass, center of mass, and inertia in the spawned USD layer only. It never saves over the original file. The shaft inertia uses a simple 160 mm rod approximation. PhysX computes the racket center of mass and inertia from both collision shapes. `paddle_mass_kg=0.100` is the **total** racket mass, including its handle. Resizing keeps that mass fixed unless you also edit it. Total robot mass remains **2.910538 kg**. Mass distribution changes with diameter, so old checkpoints can load with the same observation/action sizes but are not validated for the new dynamics.

The old distal mesh remains visible and collidable. A ball hitting it counts as an arm hit, not a paddle hit. The paddle's new mass distribution is intentionally different from the old ballast's distribution, even though the total mass is unchanged.

### Ball and contacts

The top-level `enabled_self_collisions=False` keeps the previous default. Override it at launch with `env.enabled_self_collisions=True` for training or playback. The environment resolves it into `robot.spawn.articulation_props.enabled_self_collisions` **after Hydra overrides and before spawn**. Do not use a new unregistered `+env.enabled_self_collisions` key or edit only the nested default.

With self-collisions ON, eligible nonadjacent robot links can collide. Parent–child pairs remain filtered by PhysX. The task's robot-contact crash detector can terminate these episodes; the existing `reset_robot_on_robot_crash` option controls that termination separately. Enabling this flag does not change the source USD, masses, actuator settings or observation sizes. The browser self-collision switch remains preview-only, but its initial value reads this top-level config field.

Use `source/scripts/rsl_rl/play_borinot_goalkeeper.py`, not the Solo12 player, for goalkeeper weights. The checkpoint `1014_ii4tbx06_model_4690.pt` has 57-input actor/normalizer tensors; current direct-mode actors have 44 inputs with the default observation flags (56 with every flag on). It cannot load unchanged on current code. Use a checkpoint trained with matching flags, or the older matching code. This collision option does not convert checkpoint inputs.

The ball has a 30 mm radius. Each ball launch samples mass uniformly from 20–50 g and restitution from 0.35–0.65. Restitution means how strongly the ball rebounds. We use the `max` material combination rule with a zero-restitution paddle, so the sampled ball value sets the ball–paddle restitution.

When mass changes, the code also changes the sphere's inertia using `I = 0.4 * mass * radius²`. The property setters change the active PhysX model, not just USD metadata. Validation reads those properties back.

Physics defaults to 200 Hz. The policy defaults to 100 Hz. Both rates are configurable; see section 3. Contacts are inspected after **every physics substep**, not only after policy steps. A ball-source contact sensor has separate filters for the paddle, base, upper arm, lower arm, goal, and floor. Each environment has its own floor and goal, including at large environment counts.

GPU PhysX does not support ordinary swept CCD in this runtime. The task uses speculative CCD and a swept geometric goal-plane check. These do not establish accurate real impact dynamics. Smaller physics steps and contact-parameter sensitivity tests are still needed before hardware transfer.

There is no ball drag, wind, racket compliance, or camera latency model yet.

## 2. Reward and episode endings, term by term

Every reward weight uses `rwd_scale_*`. The code adds `scale * signal`: **negative scales penalize**, and **positive scales reward**. Setting a scale to zero removes that reward contribution. It does **not** disable the event detector or its episode reset.

Here, **racket** and **paddle** mean the same separate rigid body. A **live shot** is a ball that has not yet met the safe-clearance conditions. A paddle touch alone does not make the ball safe.

The tables below give the shipped values in the committed config. Current local reward tuning includes `rwd_scale_energy=-0.0001`, `arm_copper_loss_coefficient=6.888`, and `clear_speed_m_s=0.2`. These values are preserved. The reset-grid update uses `robot_home=(0.0, 0.0, 1.5)` as its centre. The earlier 0.5 m home could not support the default lower reset-grid row. The allowed minimum base height is now 0.07 m. Hydra overrides can change these settings again at launch.

### How one policy step gets its reward

With the default 200 Hz physics and 100 Hz policy, one action lasts two physics steps and `step_dt=0.01 s`. We check contact and safety events after every physics step. An event remains recorded until the end of that action interval, even if the contact has already ended. Paddle-hit and safe-clearance history remain recorded until the next shot launches. The independent episode clock continues across launches.

At the end of the interval, we select an outcome and calculate the reward. A resolved ball is removed and waits for relaunch. Only a safety ending or the episode time limit resets the robot. The total reward is:

```text
reward = plane_progress + outcome + floor_after_save
         + optional_rates + optional_hit + energy
```

`outcome` contains **one** selected event reward, not the sum of all detected failures. The other five components are added separately. Progress, event rewards, first-hit bonuses, and the mean-power term have **no extra dt multiplier**. Only `optional_rates` is multiplied by `step_dt`. Sections 2 and 5 define every signal in this sum.

### Main reward scales

| Config field | scale | Signal and usual episode effect |
|---|---:|---|
| `rwd_scale_goal` | −1.0 | Ball contacts or crosses the goal surface. Ends this shot; episode continues. |
| `rwd_scale_ball_base_collision` | −1.0 | Ball contacts the aircraft base. Ends this shot; episode continues. |
| `rwd_scale_ball_arm_collision` | −0.25 | Ball contacts either arm link, rather than only the paddle. Ends this shot; episode continues. |
| `rwd_scale_leave_goal_area` | −1.0 | Base origin leaves the allowed box. Ends the episode. |
| `rwd_scale_save` | +1.0 | A paddle hit is followed by verified safe clearance. |
| `rwd_scale_ball_floor` | −0.25 | Live-floor shot failure, or a separate floor cost on a valid floor save. Episode continues. |
| `rwd_scale_miss` | −1.0 | An unresolved ball leaves the tracking bounds. Ends this shot; episode continues. |
| `rwd_scale_deadline` | −1.0 | Shot time runs out without a confirmed save. Ends this shot; episode continues. |
| `rwd_scale_robot_crash` | −1.0 | Non-ball base contact or excessive tilt. Reset uses two independent switches. |
| `rwd_scale_progress` | +0.5 | Signed reduction in paddle-to-target y–z distance, in metres. Does not end the episode. |

Each selected event has signal 1, so its contribution is exactly its scale. For example, a selected arm collision contributes −0.25, not −0.25 per second. Ball-result penalties are paid once per resolved shot without ending the episode. Safety penalties are paid on the failure transition. Crash penalties can repeat if you disable their resets.

### `rwd_scale_goal`: the ball scores

A goal is detected in either of two ways:

- The ball's filtered contact force against the goal exceeds `contact_force_threshold_n=0.05 N`.
- The ball crosses the front goal plane toward negative x and overlaps the goal in both y and z. The crossing test checks the path between consecutive physics samples, so it can detect a fast ball that crosses between them.

The crossing plane is `goal_center.x + goal_size.x/2 + ball_radius_m`, currently **x = −0.40 m**. This is the plane reached by the ball's center when its leading edge reaches the goal front. The y–z overlap includes the 0.03 m ball radius; an edge-overlapping ball still counts as a goal. A prior paddle touch does not cancel a later goal. A goal is a failed save and ends the shot, not the episode.

### `rwd_scale_ball_base_collision` and `rwd_scale_ball_arm_collision`: forbidden ball contacts

These use the same 0.05 N filtered ball-contact threshold. A base hit means contact with `base`. An arm hit means contact with `flying_arm_2__link1` or `flying_arm_2__link2`. Both end this shot with their penalty, even if the collision stops the ball from scoring. Neither resets the robot.

The attached paddle is a separate body and is the only legal hitting surface. The original distal-link mesh remains collidable and counts as an **arm**, not as the paddle. A paddle touch and an arm touch in the same interval still constitute an arm-hit failure. These penalties have independent scales; changing the goal scale does not change them.

### `rwd_scale_ball_floor`: ball-to-floor contact, with or without a paddle hit

This term is **not** a penalty for hitting the ball twice. It concerns the **ball touching the floor**. We detect it when the filtered ball/floor force exceeds 0.05 N, or when the ball center has `z <= ball_radius_m + 0.001` (currently **0.031 m**).

There are three important cases:

1. **No paddle hit, then floor:** this is a live-floor failure. It receives −0.25 and ends the shot; the episode continues. It earns no save bonus.
2. **Paddle hit, then floor while still unresolved:** if the ball lands with its center **in front of the goal plane** (`x > −0.40 m`) and no safety/forbidden-contact failure prevents clearance, it qualifies as a save. With default recovery off, it receives +1 for the save **and** −0.25 for the floor: **+0.75** before other terms. An unresolved floor landing that does not qualify receives the floor failure instead.
3. **Ball already safely cleared, then floor in a later interval:** the floor detector ignores that out-of-play contact. There is no new floor penalty. With recovery off, the ball has already been removed; the robot episode continues.

In case 2, the floor cost is logged as `floor_after_save`, rather than replacing the save in `outcome`. If clearance is confirmed in an interval that has already recorded a live-floor contact, that floor cost is still paid in the interval. With recovery required, it can be paid while recovery is pending (`outcome=0`), with the save bonus paid later. It is not paid again after clearance. Higher-priority failures suppress this separate floor-after-save reward component; see the priority list below.

### `rwd_scale_miss`: unresolved ball escapes the tracking region

The exact escape test uses the ball center in environment-local coordinates:

```text
abs(ball.x) > 3.0 m, OR abs(ball.y) > 3.0 m, OR ball.z > 4.5 m
```

If any bound is exceeded **before safe clearance**, the miss flag is set. There is no separate lower-z escape bound; floor detection handles low balls. These escape limits are literal values in `_sample_physics_events()`, not config fields.

For example, a ball at y = 3.1 m without verified clearance receives −1 and ends the shot, unless another failure has higher priority. A ball at that same position after a valid paddle clearance does not incur a miss penalty. A ball passing beside the goal **without touching the paddle** is not automatically a save: it stays unresolved until a floor event, escape, or deadline ends the shot. Thus `miss` means an out-of-bounds unresolved shot, not every unsuccessful interception.

### `rwd_scale_deadline`: the shot clock expires

Each launch resets only the shot clock. It advances while that ball is active. The limit is `shot_timeout_s`, default 3.0 seconds: 300 actions at 100 Hz or 180 at 60 Hz. It is separate from the 20-second robot episode clock.

An unresolved shot at that limit receives the deadline outcome and its signed scale, unless a higher-priority event wins. A confirmed save on the last allowed action wins over timeout. With recovery required, clearance without the requested recovery dwell by the shot deadline is still a timed-out shot.

A shot deadline **does not terminate or truncate the robot episode**. It removes that ball and schedules another. Tune `shot_timeout_s` for this deadline and `episode_length_s` for the full episode. Setting `rwd_scale_deadline=0.0` removes only the penalty, not shot recycling.

### `rwd_scale_leave_goal_area`: the aircraft base leaves its box

The test uses the **base origin**, not the paddle position or the outer edges of the robot. Coordinates remove the environment's world offset. The allowed inclusive bounds are:

```text
x: [-0.08, 0.90] m
y: [-1.20, 1.20] m
z: [ 0.07, 2.60] m
```

Any sampled base coordinate strictly below its minimum or above its maximum sets the area flag, even if the base returns inside before the policy interval ends. The selected area outcome costs −1 and ends the episode. Tune `goal_area_x_m`, `goal_area_y_m`, and `goal_area_z_m` to change the box. The racket may extend outside it without causing this particular penalty.

The 0.07 m minimum allows the base to descend far enough for leg or racket ground contact. These contacts are already allowed by the default base-only crash detector. Base collisions and excessive tilt still reset the robot. This setting changes only the allowed base height; it does not move the floor or change `robot_home`.

Both spawn and reset use `robot_home_quat` in wxyz order. Early phases start at `robot_home` plus small noise. Later phases add a sampled Y/Z grid offset before applying that noise; see section 4. Changing home does **not** move the allowed box or the fixed ball-target centre. With the grid enabled, construction rejects a home/grid/noise combination that can start outside this box. It does not silently clip positions. This checks base coordinates, not full-body collision clearance or dynamic reachability.

### `rwd_scale_robot_crash`: non-ball contact or excessive tilt

Two independent detectors share this reward scale:

- **Base contact:** we remove the ball's equal-and-opposite filtered normal force from the robot's net normal contact force. We inspect only `robot_crash_body_names`, default `("base",)`. A remaining magnitude above `robot_crash_force_threshold_n=0.1` N sets the crash flag. This separates a ball/base impact from a base crash against the scene. Other body names can be added explicitly.
- **Tilt:** the base's up axis tilts from world vertical by more than `robot_tilt_threshold_degs`, **70 degrees** by default. This setting is in degrees; the detector converts it internally with `math.radians()` before taking the cosine. At 69 degrees the tilt detector does not fire; at 71 degrees it does. This does not require a collision. Pure yaw does not trigger this test. For example, append `env.robot_tilt_threshold_degs=70.0` to a launch command. The old `robot_tilt_threshold_rad` field is no longer used.

`reset_robot_on_robot_crash=True` ends the episode for the contact detector. `reset_robot_on_tilt_threshold=True` ends it for the tilt detector. Setting either to `False` disables **only that detector's reset**, not its reward or the other detector. A persistent violation can therefore receive −1 in every action interval. If both detectors fire in one interval, they still select one crash outcome, not two penalties.

`self_collision_triggers_reset=True` is the default and preserves the existing contact-reset behavior. With physical self-collisions enabled, a racket/base impact can therefore reset the robot. Append `env.self_collision_triggers_reset=False` to keep physical self-contact and its crash penalty without resetting for that self-contact. This does not enable or disable physical collisions; `env.enabled_self_collisions` controls those separately. The opt-out path subtracts filtered robot/robot forces on each configured crash body for the reset decision only. Scene contact, including a simultaneous floor impact, still resets when `reset_robot_on_robot_crash=True`. Tilt, area, timeout, reward priority, and save-clearance checks are unchanged. The default path adds no sensors.

Leaving the box still terminates the episode. Ball goals and shot deadlines do not. Contact/tilt violations also prevent new safe-clearance confirmation even when their resets are disabled. Extras `goalkeeper_robot_contact_crash` and `goalkeeper_tilt_threshold` expose the two causes separately.

### `rwd_scale_save`: contact plus safe clearance, not contact alone

A save first requires a real PhysX **paddle contact** above 0.05 N. Proximity alone is insufficient. After that contact, one of these conditions must confirm that the ball is clear:

1. **Returned toward the shooter:** ball center `x > keeper_plane_x + clear_distance_m`, currently about **0.2197 m**, and x velocity `> clear_speed_m_s` (0.3 m/s shipped; 0.2 m/s in your current local tuning). Here `clear_distance_m=0.25 m` is measured from the defense plane, not from the goal.
2. **Deflected outside the goal:** ball center `x < −0.40 m` and outside the radius-expanded goal bounds in **either** y or z. The checks are `abs(y-goal_center.y) > goal_size.y/2 + ball_radius_m` or the equivalent z test. At the shipped dimensions, these are `abs(y) > 1.23 m` or `abs(z-1.2) > 1.23 m`.
3. **Deflected onto the floor in front of the goal:** a live-floor event after the paddle hit, with ball center `x > −0.40 m`. This keeps the separate floor cost described above.

New clearance is rejected if the interval contains a base/arm hit, goal, nonqualifying floor event, robot contact crash, excessive tilt, or leaving the area. Contact and clearance may occur in the same policy interval or in different intervals. Clearance stays recorded once confirmed.

With `require_recovery_for_save=False`, clearance is enough for a one-shot success reward of +1. The episode continues. With it set to `True`, the robot must also complete the recovery dwell described in section 5 before the deadline. A paddle touch followed by a goal **before clearance** is a failure. The resolved ball is removed, but the robot continues with its post-impact motion and later shots. This lets the policy learn to remain stable after saving.

### `rwd_scale_progress`: bring the paddle closer in the goal-parallel plane

We use only y and z, ignoring depth x. At the end of an action interval, we measure the previous and current paddle centers against the **same current ball target**:

```text
d_before = distance(previous paddle yz, current ball yz)
d_after  = distance(current paddle yz, current ball yz)
progress = rwd_scale_progress * (d_before - d_after)
```

Distances are Euclidean distances in metres. Reducing distance from 0.30 m to 0.20 m earns `0.5 * 0.10 = +0.05`. Moving away by 0.10 m earns −0.05. Changing only x earns zero. A stationary paddle earns zero even when the ball moves, because both distances use the same target.

The active mask is taken **at the start** of the interval. It requires an active ball, no previous paddle hit, ball x velocity `< −0.1 m/s`, and ball x `> goal_center.x` (−0.45 m). This means progress can still be paid in the interval containing the first hit or a terminal failure. It stops in following intervals after the first paddle hit. Each launch starts a new distance comparison; there is no progress jump from teleporting the ball or robot.

`progress_use_prediction=True` replaces the current-ball target with its ballistic defense-plane prediction. It does not change the y–z-only distance or this start-of-interval mask. Unlike the optional intercept kernel, progress has no extra prediction-validity mask. This is practical moving-target shaping, not a guarantee of policy invariance.

### If several events happen together: exact priority

The selected `outcome` uses this order, from **highest to lowest** priority:

```text
base hit > goal > robot crash/tilt > leave area > arm hit
         > live-floor failure > miss/escape > deadline > save > ongoing
```

For example, a simultaneous base hit and goal selects only the base penalty. A simultaneous arm hit and goal selects only the goal penalty. A floor failure at the deadline selects only the floor penalty. A detected failure suppresses a save bonus in the same interval.

This priority selects the **reward label**. Episode ending is checked separately against all active ending conditions. With crash resets disabled, a crash can be the selected reward while a simultaneous area violation still ends the episode. A shot deadline alone never ends it. A valid floor save is not a live-floor failure; its floor cost is the separate `floor_after_save` addition. Progress, enabled rate terms, and energy can also contribute on a failure transition.

| Example with shipped scales | Event subtotal, before progress/rates/energy |
|---|---:|
| Paddle hit, then safely returned toward shooter | +1.00 |
| Paddle hit, then valid floor save | +0.75 |
| Floor while unresolved, no paddle hit | −0.25 |
| Paddle touch, then goal before clearance | −1.00 |
| Ball stops scoring because it hits an arm | −0.25 |
| Simultaneous base hit and goal | −1.00, not −2.00 |
| Unresolved escape beyond y = 3 m | −1.00 |
| Unresolved shot still inside tracking bounds at 3 s | −1.00 |

Optional hit rewards paid in an **earlier** interval are not refunded if the ball later scores. Section 5 explains their same-interval failure mask. The examples above have those optional hit scales at zero.

### Tune the main reward

Append float overrides to the training command, for example:

```bash
env.rwd_scale_progress=1.0 env.rwd_scale_save=1.5 env.rwd_scale_goal=-1.0 env.rwd_scale_ball_base_collision=-1.0 env.rwd_scale_ball_arm_collision=-0.4 env.rwd_scale_leave_goal_area=-1.0 env.rwd_scale_ball_floor=-0.25 env.rwd_scale_miss=-1.0 env.rwd_scale_deadline=-1.0
```

Use floats such as `1.0` for float config fields. The defaults are starting values, not an optimized recipe. To discourage arm blocking more strongly, make `rwd_scale_ball_arm_collision` more negative. To recover the shipped reward from the current locally tuned energy setting, also pass `env.rwd_scale_energy=0.0`.

## 3. Select control mode, timing, and joint actuator settings

Use `env.control_mode` as a Hydra override on the training or playback command. Names are case-sensitive. The default is `thrust_and_torque_control`.

| `env.control_mode` | Policy actions | Actor / critic values (default flags) |
|---|---:|---:|
| `thrust_and_torque_control` | 6 individual thrusts + 2 joint torques: **8** | **44** / **56** |
| `thrust_and_PD_joint_pos_control` | 6 individual thrusts + 2 desired joint angles: **8** | **44** / **56** |
| `PI_rotors_PD_joint_pos_control` | Collective thrust + 3 body rates + 2 desired joint angles: **6** | **51** / **63** |

The actor size depends on the `obs_actor_includes_*` flags; see the observation contract below.

Append one of these selectors to the training command:

```bash
env.control_mode=thrust_and_torque_control
```

```bash
env.control_mode=thrust_and_PD_joint_pos_control
```

```bash
env.control_mode=PI_rotors_PD_joint_pos_control
```

### Physics and policy frequency

Defaults are **200 Hz physics / 100 Hz policy**, in every mode. These are independent of the control-mode choice:

```bash
env.physics_hz=200 env.policy_hz=100
```

The physics frequency must be an integer multiple of the policy frequency. At 200/100 Hz, each action spans two physics steps. For the former PI timing, use:

```bash
env.control_mode=PI_rotors_PD_joint_pos_control env.physics_hz=240 env.policy_hz=60
```

The task resolves `sim.dt=1/physics_hz`, `decimation=physics_hz/policy_hz`, render interval, and action/observation dimensions after Hydra overrides and before creating the simulator. Use these frequency selectors rather than overriding the derived `env.sim.dt` or `env.decimation`; the selectors take precedence. Invalid names or frequency ratios fail before environment construction. Rates are integer config fields; write `200`, not `200.0`.

Changing rates also changes the number of actions in a shot. The three-second deadline becomes 300 actions at 100 Hz or 180 at 60 Hz. Contact detection and power sampling still run after every physics substep. Rate rewards use the resolved policy timestep. `curriculum_min_steps` still counts policy steps, so its physical dwell changes with policy frequency. No reward scales or PPO discount settings are automatically retuned.

### Direct rotor modes: a and b

The environment owns action clipping; the RSL-RL wrapper does not clip (`clip_actions=None`). Rotor thrust commands are always clipped below at zero: the rotors only push. With the default `remove_thrust_clipping=False`, PPO rotor inputs are clipped to `[-action_clamp, action_clamp]` and thrust is capped at the smaller of `thrust_to_weight * nominal_hover_per_rotor` and `rotor_thrust_limit_n`. Setting it to `True` removes those upper/input rotor bounds, but never permits negative thrust. Direct PPO joint-torque actions are not clipped by this switch or `action_clamp`. Actions 0–5 independently request thrust from rotors 0–5:

```text
requested rotor thrust = hover_per_rotor * (1 + rotor_action_scale * action)
hover_per_rotor = simulated robot mass * 9.81 / 6
```

With `rotor_action_scale=1.0`, action −1 requests zero thrust, action 0 requests nominal per-rotor hover thrust, and action +1 requests twice that thrust before the physical command cap. Historical policies trained with different clipping or action units require their original configuration and source revision.

There is **no body-rate PI feedback** in either direct mode. The rotor geometry maps the individual forces to the resulting base force and torque. The physical response still uses a first-order thrust filter with `motor_time_constant_s=0.04`. Zero thrust actions request nominal hover thrust, but do not stabilize position or attitude on their own. These are simulator assumptions, not calibrated ESC or motor limits.

In mode **a**, actions 6–7 request shoulder and elbow torques:

```text
PPO joint torque target [N*m] = action
SAC joint torque target [N*m] = effort_lim_sim * action
```

PPO outputs physical N·m directly, with no task-level torque action clamp or scaling. `rwd_scale_torque_limits` penalizes requests outside `[-effort_lim_sim, +effort_lim_sim]` using the sum of squared relative excesses, without a policy-dt multiplier. At the default 2.7 N·m soft limit, action 3 requests 3 N·m and contributes `(0.3/2.7)^2` before the signed reward scale. The penalty does not clip the request. SAC keeps its normalized [-1, 1] mapping and default ±2.7 N·m request range. Joint drive stiffness and damping are zero: no position PD is added to either policy's torque.

**Checkpoint compatibility:** older direct-torque PPO checkpoints used `effort_lim_sim * action` and a shared input clamp. Their tensor shapes still load, but their torque units differ. Use the original source revision for those checkpoints or train a new policy; there is no automatic conversion. SAC torque checkpoints are unchanged. `robot.data.applied_torque` remains a clipped implicit-actuator estimate, not measured PhysX feed-forward torque.

In mode **b**, actions 6–7 request two scalar joint angles:

```text
shoulder angle target = 1.1 * action[6] rad
elbow angle target = 1.4 * action[7] rad
```

These are absolute targets relative to joint zero, not increments. The ranges are ±63.0 degrees and ±80.2 degrees. Implicit joint PD follows them with zero desired joint velocity and zero feedforward effort. Unlike the old locomotion interface, two desired angles use two policy outputs, not four atan2 inputs.

### PI rotor mode: c

This keeps the previous action mapping:

| Index | Meaning |
|---:|---|
| 0 | Collective thrust around hover: `weight * (1 + 0.7 * action)` |
| 1–3 | Body-rate targets, scaled by `(2.0, 2.0, 1.5)` rad/s |
| 4–5 | Shoulder and elbow angle targets, scaled by `(1.1, 1.4)` rad |

At each physics step, the PI controller computes rate error, updates its bounded integral, and requests base torque. A six-rotor allocation matrix converts collective thrust and torque into individual thrust commands. The same thrust clipping and motor-response filter apply. The limb uses the same PD settings as mode b. The PI gains remain `rate_kp` and `rate_ki`; they do not affect modes a or b.

### Solo-style joint actuator

Every mode uses `ImplicitActuatorCfg`, with **armature 0.00036207**, joint friction 0.0, default velocity limit **100 rad/s**, and default effort limit **2.7 N·m**. Armature represents the motor/gear inertia reflected at the joint. The armature matches the Solo actuator configuration, because Borinot uses Solo joint motors.

Configure limits and the position-mode PD gains through Hydra:

```bash
env.effort_lim_sim=2.7 env.velocity_lim_sim=100.0 env.kp=9.0 env.kd=0.2
```

These four fields are floats. The task refreshes limits and gains on the existing actuator before spawn, without replacing its friction or armature settings. Torque mode ignores the PD gains unless `hold_arm=True`. In every mode, `hold_arm=True` means neutral-position PD and zero feedforward torque; it is **not** a mechanically welded limb.

`kp` and `kd` use the same names as Solo12. Their defaults are 9.0 N·m/rad and 0.2 N·m·s/rad.
The former `joint_stiffness` and `joint_damping` names are no longer config fields.
For an older position policy trained with the previous defaults, pass `env.kp=35.0 env.kd=0.6`.

Both SAC agents support the two direct modes. In torque mode, each joint's tanh output
-1 / 0 / +1 maps to -effort_lim_sim / zero / +effort_lim_sim N·m, with no joint PD.
The same scalar limit applies to both independently commanded joints.
In position mode, SAC uses its configured angle endpoints rather than PPO's legacy angle scales.
SAC's rotor mapping is also different from PPO's mapping above; see [SAC_TRAINING.md](SAC_TRAINING.md).

### Observation and checkpoint contract

The critic always receives all 48 values plus the current policy actions: 56 values in direct modes. These values are the home-position error, full root rotation matrix, base linear and angular velocities, joint positions and velocities, relative paddle position and normal, relative ball position and velocity, predicted intercept/time/validity, shot phase/time, ball-active flag, next-launch countdown, episode progress, and sampled ball mass/restitution. PI mode also adds 9 controller values after the actions: the 6 rotor thrusts divided by hover thrust, and the 3 body-rate PI integrals. That gives 63 values. Direct modes leave them out: there, the actions already command each rotor, and the PI integral does not exist. Joint velocity is divided by `velocity_lim_sim`.

The actor (policy) gets a subset. Four config flags choose the optional groups. The first three are `False` by default; ball parameters are `True`:

| Flag | Actor values added | Critic indices |
|---|---|---|
| `obs_actor_includes_ball_plane_intersection_point` | Vector from racket to the predicted point where the ball crosses the keeper plane (3) | `34:37` |
| `obs_actor_includes_time_ball_collides_plane` | Predicted time until the ball crosses that plane, clipped to 0–1 s (1) | `37` |
| either of the two above | Prediction-valid flag: 1 when the ball is incoming and crosses within 1 s (1) | `38` |
| `obs_actor_includes_phase` | Paddle hit, ball cleared, shot age, recovery time, ball active, next-launch countdown, episode progress (7) | `39:46` |
| `obs_actor_includes_ball_params` | Sampled ball mass in kg and restitution (2) | Last two values |

With the defaults, the actor gets 36 values plus the actions: **44** in direct modes. PI mode adds its 9 controller values: **51**. Setting `env.obs_actor_includes_ball_params=false` removes the final two actor values, giving **42** or **49** inputs. With every flag off, the actor gets 34 values plus the actions (and the 9 PI values in PI mode). With every flag on, the actor equals the critic. The actor keeps the critic's order and drops the unused groups. The critic always keeps every group, including mass and restitution. Set the flags with Hydra, for example `env.obs_actor_includes_phase=true`. The old `obs_includes_*` names and the misspelled `obs_actor_inclues_ball_params` are not supported.

With the phase flags off, the actor has no ball-active flag. While no ball is in the air, the ball position and velocity values are zero. That zero block is the actor's only "no ball" signal.

Orientation occupies indices `3:12` (zero-based). It is the base-to-world 3×3 rotation matrix, flattened row by row: `[R00, R01, R02, R10, R11, R12, R20, R21, R22]`. We convert the simulator's `(w, x, y, z)` quaternion with `matrix_from_quat`; internal vector transforms still use that quaternion. The observation is identical for `q` and `-q`. This follows the continuity motivation in [Zhou et al., On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035); we use the requested full 9D matrix, not their compact 6D representation.

**Checkpoint change (2026-10-19):** direct modes no longer observe the rotor thrusts and the PI integral. Those 9 values repeated the rotor actions, apart from clipping and the short motor lag, and the integral was always zero. Direct-mode actors go from 53 to 44 inputs and critics from 65 to 56. PI mode is unchanged (51 / 63). Older direct-mode checkpoints do not load on this code; use the commit before this change to play them back.

**Earlier checkpoint change:** continuous shots add three phase values: ball active, normalized remaining launch delay, and normalized episode time. At that time the critic had 65 inputs in direct modes and 63 in PI mode. With the default flags, the actor has 53 or 51 inputs. Renaming the flags and leaving ball parameters enabled preserves the previous default observation layout. Earlier checkpoints (65/63-input actors, 62/60-input matrix checkpoints and 57/55-input quaternion checkpoints) do not load with the default flags. Train a new policy, and play it back with the same `obs_actor_includes_*` flags. In critic indices, ball-related values at `28:39` are zero while the ball is inactive. The three new phase values are at `43:46`, before the action history. Shot age uses `shot_timeout_s`; episode progress uses `episode_length_s`.

The predicted intercept is privileged information. It does not replace the current-ball target in the default progress reward. All coordinates remove the clone's world offset. The finite shot clock is independent of the generic runner's randomized initial episode-length counter.

Always match the control mode, rates, action scales, and actuator settings when loading a checkpoint. Direct and PI checkpoints have different shapes. Modes a and b have equal shapes but different action meanings; do not interchange their checkpoints merely because loading succeeds. Older PI checkpoints also used a different joint-velocity normalization, velocity limit, and armature. Existing locomotion checkpoints are not compatible with these goalkeeper observations.

## 4. Difficulty and curriculum

Every shot is ballistic and aimed at the defense plane. The original four levels below were checked to threaten the goal without hitting the floor first. Additional, wider or slower custom phases need the same check; increasing a spread does not by itself guarantee a useful shot.

| Level | Sideways target range around the neutral racket | Height range | Flight time to defense plane |
|---:|---:|---:|---:|
| 0 | ±0.045 m | ±0.045 m | 0.42–0.50 s |
| 1 | ±0.20 m | ±0.12 m | 0.35–0.50 s |
| 2 | ±0.40 m | ±0.22 m | 0.30–0.46 s |
| 3 | ±0.65 m | ±0.32 m | 0.25–0.42 s |

These are uniform target offsets around the configured center. Level 0 is intentionally easy: neutral hovering can cover its small spread. It checks geometry and flight stability, not agile goalkeeping.
The sampling checks prove that throws threaten the goal, not that Borinot can reach every throw. Some fast, wide level-3 shots may be dynamically unreachable. Check that limit before interpreting a learning plateau. The two arm joints mainly move the tip in the body's x–z plane; sideways coverage also needs aircraft motion or reorientation.

Automatic promotion uses the last up-to-2048 completed shots from the current level (`curriculum_window_shots`). It requires at least 1024 completed shots (`curriculum_min_shots`), a save fraction strictly above 0.7, and 3000 policy steps at that level. The latter is **30 simulated seconds per environment at 100 Hz**, or 50 seconds at 60 Hz. More environments increase sample count, not elapsed simulated time. The old `curriculum_window_episodes` and `curriculum_min_episodes` fields are replaced by `curriculum_window_shots` and `curriculum_min_shots`; update launch overrides that used those names. Best-checkpoint eligibility still counts full episodes, not shots.

A shot keeps its starting level until it ends. Older-level endings are excluded from the new level's decision window. Promotion clears that window. An empty batch cannot promote. General outcome counts still include every completed shot.

For a fixed-level experiment, append:

```bash
env.curriculum=False env.shot_level=2
```

Checkpoint resume restores the policy and optimizer, but the standard runner does not save this environment's curriculum window or active level. Set `env.shot_level=N` explicitly when resuming a later level. The dwell and evidence window restart. Do not infer restored curriculum state from the checkpoint iteration number.

### Centre-to-grid robot resets

The robot starts near the centre in the first shot phases. From a chosen shot level onward, every reset independently selects one location from a configurable Y/Z grid. The shot target stays fixed in world coordinates. It is **not** recentered around the sampled robot position. This makes the robot move from different starting positions to defend the same goal.

Defaults in the task config:

```python
robot_home = (0.0, 0.0, 1.5)
reset_grid_enabled = True
reset_grid_start_level = 2
reset_grid_y_offsets_m = (-0.8, 0.0, 0.8)
reset_grid_z_offsets_m = (-0.65, 0.0, 0.65)
reset_xy_noise_m = 0.015
reset_z_noise_m = 0.015
```

Phases are zero-based: phases **0–1** use only `robot_home`. Phase **2 and later** samples all nine combinations of the three Y offsets and three Z offsets, with equal probability **1/9**, including the centre. These are **base positions**, not racket positions. Before noise, Y is −0.8, 0 or +0.8 m, and Z is 0.85, 1.5 or 2.15 m. X remains at `robot_home[0]`. The offsets use world axes, not the rotated body frame.

Both modes then add independent uniform noise: X/Y within ±`reset_xy_noise_m`, and Z within ±`reset_z_noise_m`. Set both noise fields to `0.0` for exact grid points. The orientation, joint pose and zero initial velocities are unchanged. `robot_home` stays the centre used by the home-position observation and optional recovery terms; it does not become each episode's randomly selected start. Observation and action sizes are unchanged by this reset feature.

Change either offset tuple to tune spacing or the number of rows/columns. Their Cartesian product defines all locations, so three Y values and four Z values give twelve locations. A location index uses `row * len(reset_grid_y_offsets_m) + column`. `env.spawn_grid_index` stores each episode's selected index; −1 denotes centre-only sampling.

The grid uses the active `shot_level` even when `curriculum=False`. For example, fixed level 3 still uses grid resets with the default threshold. To keep centre-only starts at any level, use `env.reset_grid_enabled=False`. To keep phases 0–3 centred and switch at phase 4, use `env.reset_grid_start_level=4`. You can also override `env.reset_grid_y_offsets_m=[-0.5,0.0,0.5]` and `env.reset_grid_z_offsets_m=[-0.4,0.0,0.4]` at launch. Keep at least one finite offset per axis, non-negative noise, and a non-negative integer switch level.

With the grid enabled, the environment validates the centre, all grid points and the full noise ranges against `goal_area_*_m` before simulator construction. Invalid combinations raise a clear error; they are not clipped into duplicate edge positions. This is a base-position check, not a collision-clearance or reachability guarantee. The centre height is now **1.5 m**, replacing the previous 0.5 m source value, which was below the allowed minimum and left no space for a lower row.

Promotion happens before a reset samples its next start. Episodes already running keep their pose. New balls use the newly selected shot level; an old in-flight ball keeps its own launch stage. The spawn grid changes only at the next full robot reset. Only the selected reset environments change; partial resets leave other robot poses, balls and spawn indices alone. Normal automatic resets use the same sampler. This keeps the existing stage-pure best-checkpoint gate aligned with both shot difficulty and spawn difficulty.

The browser viewer shows the available base-origin crosses for the selected phase. Its **Inspect location** dropdown moves the robot without noise; **Sample robot reset** includes noise. Both restore the configured body orientation and joint pose. They change the preview only. See the [viewer guide](../../../../scripts/tools/borinot_goalkeeper_viewer/README.md).

### Best checkpoint per level

`source/scripts/rsl_rl/train.py` keeps one best checkpoint per shot level in the run folder: `best_model_curriculum_idx_<level>.pt`, for every configured level. "Best" means the highest `Train/mean_reward`. This is the mean return of the last 100 finished episodes. `best_model.pt` is a copy of the most recently improved level best. Each file stores its level, iteration, and score in the checkpoint `infos`.

A promotion does not end the shots already in flight. For a few iterations, the finished episodes still come from the old, easier level. Their return must not count as the new level's score. The environment therefore counts the newest finished episodes that started at the current level (`curriculum_stage_pure_episodes`). The script skips a level's best-checkpoint update until this count covers the whole 100-episode reward buffer. In a local check, this skipped two or three iterations after each promotion. Without the skip, a level-2 score would have become the level-3 best.

With `env.curriculum=False`, the file is named after the fixed `env.shot_level`.

## 5. Optional additions — off by default

All ten optional scales ship as zero. They are separate experiments, not required parts of the main objective. Your current local file enables energy only, as noted in section 2. A **dense** reward means feedback on many policy steps, rather than just at a shot outcome.

### `progress_use_prediction`: change the progress target, not its scale

This switch is `False` by default. Setting it to `True` makes progress aim at the ball's predicted intersection with `keeper_plane_x`, rather than its current position. The prediction assumes gravity of 9.81 m/s² and no drag or future collisions:

```text
t = (keeper_plane_x - ball.x) / ball.vx
target = ball.position + ball.velocity * t
target.z -= 0.5 * 9.81 * t²
```

The helper reports a valid prediction only when `ball.vx < −0.1 m/s`, the ball is in front of the defense plane, and its time to that plane lies in [0, 1] s. It clamps the returned prediction time to [0, 1] s; the denominator is bounded at −0.1 m/s to avoid a near-zero division. Progress uses the returned target even when the validity flag is false. The two optional rewards below do respect that flag.

### `rwd_scale_intercept`: stay near the predicted interception point

This adds a positive rate reward when the paddle center is near the predicted ball center. Unlike progress, it measures **full 3D distance**, including x:

```text
d_3d = distance(current paddle center, predicted ball center)
K = exp(-(d_3d / intercept_sigma_m)²)
intercept reward = rwd_scale_intercept * K * step_dt
```

`K` is set to zero unless the prediction is valid and the ball has **not yet touched the paddle**. These conditions are evaluated at the end of the interval. The shipped width is `intercept_sigma_m=0.30 m`. At d = 0, K = 1; at d = 0.30 m, K is about 0.368. If the scale is 0.1, a perfect match contributes `0.1/100 = +0.001` per action at the default 100 Hz, not +0.1 per action. It rewards being close even without motion, so it is different from signed progress. It does not itself prove contact or a save.

### `rwd_scale_alignment`: orient the paddle face toward the incoming ball

This uses the same valid, pre-hit kernel `K`, then multiplies it by a squared dot product:

```text
direction = -ball.velocity / max(norm(ball.velocity), 0.1 m/s)
A = dot(paddle face normal, direction)²
alignment reward = rwd_scale_alignment * K * A * step_dt
```

The face normal is the paddle's local +x axis rotated into world coordinates. At normal shot speeds, A is 1 for a face perpendicular to the flight direction, and 0 for a face edge-on to it. Squaring means the normal and its opposite get the same value: this term does not choose a front versus back face. Below 0.1 m/s, the bounded denominator reduces the magnitude of A. The kernel prevents a distant but well-oriented paddle from getting the full reward.

### `rwd_scale_hit` and `rwd_scale_centered_hit`: reward the first paddle touch

These are **event bonuses**, not rates. They are paid only on the first detected paddle hit of a shot, and only if the selected outcome in that interval is ongoing (0) or save (7):

```text
sigma = 0.75 * paddle_diameter_m / 2
Q = exp(-(u² + v²) / sigma²)
first-hit reward = rwd_scale_hit + rwd_scale_centered_hit * Q
```

`u` and `v` are the ball center's offsets from the **head center** along the paddle face, in the paddle's own coordinates, at first contact. Paddle thickness/depth is ignored. Q is 1 at the center and decreases toward the edges. The width is 75% of head radius: 0.06 m for a 0.16 m head. It scales with racket size and is separate from `intercept_sigma_m`. Handle contacts still count as hits, but usually have low centered quality. Both hit scales remain zero by default.

For example, with hit scale 0.1 and centered-hit scale 0.2, a centered first contact earns +0.3. A second paddle contact earns no new bonus. A first paddle hit accompanied by a selected arm/base/goal/safety/deadline/miss failure earns neither bonus. **A bonus already paid on an earlier interval is not taken back if the ball later scores.** These terms encourage contact discovery, not verified saves, so keep them separate from `rwd_scale_save`.

### `require_recovery_for_save`: an optional success gate

This is a boolean, not a reward scale. It is `False` by default. Setting it to `True` keeps the shot running after clearance and withholds the save bonus until the robot passes all five checks:

| Quantity | Required value at each policy-step check | Config field |
|---|---:|---|
| 3D base distance from `robot_home` | < 0.35 m | `recovery_position_tolerance_m` |
| Base linear speed | < 0.5 m/s | `recovery_speed_m_s` |
| Base angular speed | < 0.8 rad/s | `recovery_angular_speed_rad_s` |
| Tilt from world vertical | < 0.35 rad, about 20 degrees | `recovery_tilt_rad` |
| Largest absolute arm joint angle from zero | < 0.5 rad | `recovery_joint_tolerance_rad` |

The conditions must hold at consecutive policy-step checks for `recovery_dwell_s=0.25 s`, or 25 actions at the default 100 Hz (15 at 60 Hz). The timer is zero before clearance and resets to zero if any check fails. It measures return toward neutral arm angles, not toward the previous action's targets. There is no separate yaw-angle recovery test. Safety and the three-second deadline remain active during recovery.

This gate works even with `rwd_scale_recovery=0.0`. Conversely, enabling the recovery reward below does not turn on the gate. With the default gate off, clearance resolves and removes the ball. The robot keeps balancing during the wait and the rest of the episode. Recovery shaping, if enabled, can remain active until the next launch.

### `rwd_scale_recovery`: smooth feedback after clearance

This adds a rate reward after the ball is safely cleared. It does not require the hard recovery checks to be satisfied:

```text
S = exp(-(distance_home / 0.35 m)² - (linear_speed / 0.5 m/s)²
        - (angular_speed / 0.8 rad/s)² - (tilt / 0.35 rad)²
        - (largest_absolute_joint_angle / 0.5 rad)²)
recovery reward = rwd_scale_recovery * S * step_dt, only after clearance
```

S is 1 at a stationary, level, neutral-arm home state and decreases as any error grows. **These five widths are fixed in `_recovery_geometry()`**. Changing the configurable gate thresholds above does not change this smooth score. With recovery required, it can guide the robot over several post-clearance steps. Without the gate, it can contribute on the clearance/terminal step only.

For a recovery experiment, append:

```bash
env.require_recovery_for_save=True env.rwd_scale_recovery=0.1
```

### Motion and control regularizers

A regularizer discourages unwanted motion or control changes. Each signal below is nonnegative. Use a **negative** scale to penalize it. These terms are evaluated at the end of every action interval, including terminal intervals, and are not limited to the incoming-ball phase.

| Config field | Raw signal; contribution is `scale * signal * step_dt` | What it discourages |
|---|---|---|
| `rwd_scale_action_rate` | Mean squared normalized thrust-command change over six rotors, **plus** mean squared normalized joint-effort change over two joints. See the definition below. | Changes in post-clipping actuator commands, not raw policy outputs. The two actuator groups have equal weight. Differences are **not divided by dt**. |
| `rwd_scale_body_angular_velocity` | `sum(base_angular_velocity_body_frame²)`, in (rad/s)² | Fast base rotation. All three axes count, including yaw. |
| `rwd_scale_body_linear_velocity` | `sum(base_linear_velocity_world_frame²)`, in (m/s)² | Fast aircraft translation in any direction. |
| `rwd_scale_thrust` | `mean((applied_rotor_thrust / nominal_hover_thrust_per_rotor - 1)²)` over six rotors | Thrust departing from its nominal hover allocation. Applied thrust includes motor lag. |

The action-rate definition is:

```text
F_max = nominal_hover_thrust_per_rotor * thrust_to_weight
rotor_change = (current_clipped_thrust_command - previous_clipped_thrust_command) / F_max
torque_change = (current_clipped_joint_effort - previous_clipped_joint_effort) / (2 * effort_lim_sim)
signal = mean_over_6_rotors(rotor_change²) + mean_over_2_joints(torque_change²)
reward contribution = rwd_scale_action_rate * signal * step_dt
```

The torque denominator is the **full signed span**, from `-effort_lim_sim` to `+effort_lim_sim`. For example, changing all six thrust commands by 10% of their full range gives a signal of 0.01. Changing both joint efforts by 10% of their full signed range also gives 0.01. Doing both gives 0.02. With scale −0.1 at 100 Hz, the reward contribution is `−0.1 * 0.02 * 0.01 = −0.00002`.

We compare commands at the end of successive policy intervals, including terminal intervals. Thrust is sampled **after its command clipping but before motor lag**. Joint effort uses `robot.data.applied_torque`: the implicit actuator's clipped effort estimate, **not measured PhysX torque**. This also covers PD and PI control modes. With `hold_arm=True`, ignored policy joint outputs do not count; changes in the holding controller's effort estimate still count.

Both histories start at hover thrust and zero joint effort after a robot reset. A ball-only reset does not clear them. Repeating raw actions inside the same saturation region adds no action-rate cost. If upper thrust clipping is disabled, `F_max` remains the nominal normalization scale, not an enforced limit; normalized thrust can exceed 1. This reward does not add any new actuator clipping or change the policy observations. With the thrust cap enabled, the total signal lies between 0 and 2. Normalization and group averaging change its numerical scale, so an old action-rate weight does not preserve the old penalty strength.

The thrust signal is zero at nominal hover thrust. It is **not a physical power estimate**: hovering consumes power but pays no thrust regularizer. It also penalizes reducing thrust below nominal hover, not only increasing it above hover.

### `rwd_scale_energy`: mean mechanical plus dissipative power

Despite its name, this term uses **mean power in watts**, not integrated energy in joules. We sample every completed physics step (two per action by default), including the final step before an automatic reset:

```text
energy reward = rwd_scale_energy * mean_over_physics_steps(P_total)
P_total = P_rotor_mechanical + P_rotor_dissipative
          + P_arm_mechanical + P_arm_copper_loss
```

There is **no extra `step_dt` multiplier**. At a mean estimate of 500 W, your local scale −0.0001 contributes −0.05 per policy action. Setting the scale to 0.0 disables this reward without disabling the power measurements.

For each physics sample, the model is:

```text
rotor_speed_i = sqrt(applied_thrust_i / kf)
P_rotor_mechanical = sum(km * rotor_speed_i³)              # six rotors
P_rotor_dissipative = P_rotor_mechanical * (1/efficiency - 1)
P_arm_mechanical = sum(joint_torque_j * joint_velocity_j)  # two joints
P_arm_copper_loss = arm_copper_loss_coefficient * sum(joint_torque_j²)
```

`efficiency` is `p_el_motor_efficiency=0.95`. Thus total rotor power equals rotor mechanical power divided by 0.95. Applied thrust includes the motor's first-order lag, rather than using the raw commanded thrust. Joint torque is `robot.data.applied_torque`: a clipped implicit-actuator PD estimate, **not measured PhysX drive torque**.

Joint mechanical power is signed. Driving with the motion adds power; braking against the motion subtracts it. Copper loss is nonnegative, including when holding a stationary joint under torque. The coefficient has units W/(N·m)²: it ships as 7.0 and is 6.888 in your locally tuned file. `kf`, `km`, efficiency, and the copper coefficient describe the **physical estimate**; they are not reward scales.

The net estimate is not clamped to zero. Strong braking can make it negative, in which case a negative energy scale produces a positive reward. This represents ideal regenerative transfer in the model, not verified battery-energy recovery on Borinot. The combined power model is an estimate, not measured electrical consumption.

### Keep ablations easy to interpret

Start each ablation from the same main reward and change one addition. With ±1 event rewards, do not copy large dense reward weights from another paper without rescaling them.
These `rwd_scale_*` names replace the older mixed names; old Hydra/config field names are not aliases.

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

**Third hypothesis:** requiring recovery reduces successful-but-unstable saves. Compare recovery off/on and report both save fraction and a common post-impact stability test. The continuous default already keeps the robot moving after a save. Report post-impact stability and safety failures separately from shot-save fraction.

The ball is light relative to the full aircraft, although substantial relative to the paddle. Do not claim this is already a comparable-mass aerial impact benchmark. Goalkeeping and aerial contact RL both have precedents. A publication claim needs a measured advantage and a careful novelty check, including the unresolved aerial-catching leads in yesterday's review.

### If PPO struggles

First inspect physical hit attribution, sampled throws, and the outcome breakdown. Then hold the difficulty fixed. Increase training time or adjust progress/save weights before adding many reward terms.
If flight stability is the main obstacle, pretrain hovering or reuse a flight controller with the same action contract. If sparse discoveries remain the obstacle, a model-based interception teacher could generate demonstrations. SAC or a recurrent policy are later comparisons, not implemented alternatives in this task registration.

## 7. Logging, verification, and limits

The staged reset-grid and self-collision override passed **144 CPU tests** and **136 real-simulator assertions** in two launches (64 and 32 environments). These cover centre-only and grid phases, every default grid location, finite noisy starts inside bounds, partial/automatic resets, same-reset promotion, the unchanged observation shape, fixed world shot targets, and actual articulation flags OFF/ON. The ON case used real Hydra overrides for a different switch level, smaller grid offsets and different XY/Z noise. Browser checks passed 64 spawn assertions and 1379 shot-editor regression assertions on the current six-phase preview. These prove configuration/reset integration, not learned performance with random starts or self-collision enabled.

Reproduce the simulator reset checks from the activated local IsaacLab environment:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/spawn_smoke.py --headless --num-envs 64 --output /tmp/goalkeeper-spawns.json env.enabled_self_collisions=True '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

`goalkeeper_outcome` in step extras records the selected reward outcome before automatic reset. With either crash-reset switch disabled, outcome 8 can be nonterminal; use `terminated` to identify actual endings. `goalkeeper_had_paddle_hit`, `goalkeeper_robot_contact_crash`, `goalkeeper_tilt_threshold`, and `goalkeeper_reward_terms` support attribution checks. `outcome_counts` pools resolved shots. `goalkeeper_shot_finished` marks exactly which transition resolved an active ball; safety failures while waiting are not extra shots.

| Outcome code | Meaning | Scale used by the `outcome` component |
|---:|---|---|
| 0 | Ongoing; no selected event | None, contribution 0 |
| 1 | Ball/base collision | `rwd_scale_ball_base_collision` |
| 2 | Ball/arm collision | `rwd_scale_ball_arm_collision` |
| 3 | Goal | `rwd_scale_goal` |
| 4 | Live-floor failure | `rwd_scale_ball_floor` |
| 5 | Base outside goal area | `rwd_scale_leave_goal_area` |
| 6 | Deadline | `rwd_scale_deadline` |
| 7 | Confirmed save | `rwd_scale_save` |
| 8 | Robot contact crash or excessive tilt | `rwd_scale_robot_crash` |
| 9 | Unresolved miss/escape | `rwd_scale_miss` |

These numbers are labels, **not** their priority order. Use the priority list in section 2. A valid floor save has code 7 (or code 0 while recovery is pending), not code 4.

`goalkeeper_reward_terms` provides the six already-scaled components that sum to the actual reward:

| Component | Contents |
|---|---|
| `plane_progress` | Signed y–z distance progress |
| `outcome` | The selected event's scale from the table above |
| `floor_after_save` | Floor scale when a live-floor event accompanies valid clearance and no selected failure |
| `optional_rates` | Intercept, alignment, recovery, and four regularizers, all multiplied by `step_dt` |
| `optional_hit` | First-hit and centered-first-hit bonuses |
| `energy` | Energy scale times mean total power in watts, without dt |

`goalkeeper_shot_stage` records the stage at the current ball launch. `goalkeeper_shot_finished` is true for exactly one transition per resolved ball. `goalkeeper_paddle_hit` is the **first** paddle-hit flag for this interval. `goalkeeper_had_paddle_hit` records whether a hit has occurred anywhere in the current shot. They do not by themselves establish a save. `power_mean_w` contains the named mechanical, dissipative, and total mean-power estimates even when the energy scale is zero.

TensorBoard logs full-episode returns, shots, saves and duration, reward components, cumulative shot-save fraction, contact/clearance fractions, collection/selected difficulty, stage age, and the stage-pure shot window. Episode logs are emitted only when a robot episode ends, not when an individual ball ends. The standard RSL logger averages these sampled summaries within a rollout. For exact shot counts, select outcome codes with `goalkeeper_shot_finished`, not the robot done flag. A displayed cumulative save fraction is not a fresh held-out evaluation.

Run tensor-only tests without launching Isaac Sim:

```bash
python -m pytest -q --import-mode=importlib --rootdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_task_math.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_power.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_rewards.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_reset_geometry.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_control.py source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/test_curriculum_checkpoint.py
```

Run the GPU environment check:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/smoke.py --headless --num-envs 16 --steps 180 --output /tmp/borinot_goalkeeper_validation.json '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

That contact-regression script explicitly selects PI mode at 240/60 Hz to retain its original test conditions. For a control-mode check using real Hydra overrides, run:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/control_smoke.py --headless --output /tmp/borinot_goalkeeper_control.json env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 env.effort_lim_sim=2.7 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Historical contact checks cover the original ballistic goal-threatening shots, progress geometry, actual paddle/base/arm/goal/floor contacts, reward scales, full observations, mass/inertia/material readback, partial resets, finite rollouts, and curriculum boundaries. They also check the corrected paddle plane at bent-arm poses, both crash-reset switches independently, and the configurable tilt threshold. These earlier physics checks used the 1.5 m reference height. The current continuous task uses that height by default.

The round-racket and mount controls passed **121 CPU tests** and **1118 offline browser checks**. Real task physics checks cover 12, 16 and 24 cm heads, actual head/handle hits and outside-rim misses, mass/COM/inertia readback, and the full 71-check contact/reset regression. They also verify −4, 0 and +4 cm mount shifts at neutral and bent joints, plus a real Hydra launch with both diameter and mount overrides. The default 16 cm racket and its +4 cm extension passed neutral clearance with physical self-collisions enabled. A separate CPU bench tested 21 imposed poses and 10 continuous paths; its four intended clear paths had no contacts with either filter setting. These are geometry and integration checks, not a learned-policy comparison or a complete safe-workspace map.

Control-variant validation passed 113 CPU tests and 304 simulator checks. The simulator checks cover all three modes, actuator property readback, 200/100 and 240/60 Hz timing, a held arm at 400/100 Hz, and the original contact regression. Those original control-interface checks completed two PPO updates per mode with 32 environments and 32 rollout steps, then reloaded each checkpoint for finite inference. The new shared PPO recipe uses 24 rollout steps; its separate verification is described in the linked PPO training guide. These checks establish integration, not learned agility, convergence, hardware transfer, or ten-thousand-environment throughput.

Local validation uses Isaac Sim 5.1.0, Torch 2.7.0+cu128, RSL-RL 3.1.2, and the RTX 5070 Laptop GPU. The older simulator on IRICluster needs its own launch check after you pull the code.

## Files

To follow the reward in code, read these methods in order:

1. `_pre_physics_step()`: save previous paddle/actions, set the progress mask, and clear interval flags.
2. `_sample_physics_events()`: detect contacts, safety violations, ball clearance, and escape on completed physics steps.
3. `_get_dones()` and `task_math.terminal_outcome()`: apply recovery/deadline logic, select the event label, and decide whether to reset.
4. `_get_rewards()`: add the six components defined above. `_sample_power()` supplies its physics-rate power average; `_interception()` and `_recovery_geometry()` supply optional shaping signals.
5. `_launch_ball()` / `_park_ball()`: recycle only the ball and per-shot state. `_reset_idx()` resets the robot, controller and full episode only for safety/time limits. `_record_shots()` drives curriculum evidence; `_update_curriculum()` logs full episodes and gates stage-pure episode returns.

- `borinot_goalkeeper_env_cfg.py`: geometry, reward weights, limits, and difficulty settings.
- `borinot_goalkeeper_env.py`: control, observations, contacts, rewards, resets, and curriculum.
- `paddle.py`: task-local paddle attachment and shaft mass/inertia override.
- `task_math.py`: ballistic sampling, goal crossings, plane progress, and outcome priority.
- `agents/rsl_rl_ppo_cfg.py`: PPO networks and optimizer settings.
- `tests/test_task_math.py`: simulator-independent checks.
- `tests/test_rewards.py`: the actual reward method on controlled tensors; signed scales, event amounts, dt handling, and floor-save addition.
- `tests/test_power.py`: physics-rate sampling, power model, and signed power reward.
- `tests/test_reset_geometry.py`: signed crash/reset behavior and USD-derived arm/paddle frames.
- `tests/test_control.py`: mode mapping, PI isolation, motor response, limits, and frequency validation.
- `tests/test_curriculum_checkpoint.py`: the stage-pure episode count that gates each level's best checkpoint.
- `tests/control_smoke.py`: real Hydra selectors, PhysX actuator property readback, power sampling, and reset checks.
- `tests/smoke.py`: original PI/240 Hz/60 Hz contact-regression fixture and reset checks.
- `tests/checkpoint_smoke.py`: finite deterministic inference after reloading a PPO checkpoint.


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

### Desired direct actuation in W&B

For `control_mode=thrust_and_torque_control` with `hold_arm=False`, the existing
training command also logs these scalar charts. No extra flag is needed.

| W&B metric group | Values |
| --- | --- |
| `Desired_Thrust/rotor_1` through `rotor_6` | `mean_n`, `rms_n`, `min_n`, `max_n`, `outside_limits_fraction` |
| `Desired_Torque/<joint name>` | `mean_nm`, `rms_nm`, `min_nm`, `max_nm`, `outside_limits_fraction` |
| `Desired_Actuation` | Actual soft bounds, normalization, and unweighted relative-excess costs |

Rotor labels follow `rotor_pos_b` / action columns 0–5 (Eagle/PX4 motor order).
Joint labels are `flying_arm_2__j_bl_link1` and `flying_arm_2__j_link1_link2`,
in `joint_names` order. Torque means **arm-joint torque**, not body yaw torque.

These are **raw requests before action/thrust clipping, torque saturation and
motor lag**, using the same mapping and bounds as the soft-limit rewards.
Negative thrust requests remain visible even though actual thrust cannot be
negative. PPO torque actions are already N·m; SAC torque actions are multiplied
by `effort_lim_sim`. No force, torque, reward, or observation behavior changes.

Every policy-step sample enters the statistics, including the terminal step.
On a robot reset, each chart averages the per-episode statistic across the same
completed environments as `Episode_Reward/*`. Therefore `min` and `max` are the
**mean of episode minima/maxima**, not global extrema. RMS is the mean of each
episode's RMS and does not hide alternating positive/negative torque. The runner
then averages reset-batch packets within its logging interval, just as it does
for the reward terms. Ongoing episodes enter only when they finish. Ball-only
relaunches do not clear these buffers. These summaries are not individual robot
time traces. Other control modes do not emit the new metric groups.

For an exact reward comparison, multiply
`Desired_Actuation/thrust_relative_excess_episode_sum` by
`rwd_scale_thrust_limits`: this equals `Episode_Reward/thrust_limits`.
The analogous torque keys use `torque_relative_excess_episode_sum` and
`rwd_scale_torque_limits`. The `*_relative_excess_mean_per_step` charts divide
each episode's unweighted cost by its policy-step count. Costs sum actuator
channels and have no policy-dt multiplier. Request/cost logging stays active
when either reward weight is zero. Upper thrust excess is ignored only when
the corresponding reward has no upper bound; the chart reports that switch.

The charts appear once episodes finish on a new launch using this source.
Already-running processes do not load these edits automatically.


## Command delays and evaluation

Use `env.actuator_delay=0.02` for a fixed 20 ms policy-command delay, or
`env.actuator_delay=0.0 'env.actuator_delay_range=[0.0,0.04]'` to sample a new
delay per robot episode. Defaults add no delay. Inner controllers still use
current measurements. See [timing, training and evaluation commands](ACTUATOR_DELAY.md)
for the shared conventions and the fixed-stage goalkeeper checkpoint evaluator.
