# Solo12 backflip task (`solo12-backflip`)

The robot is rewarded for rotating backwards about a horizontal axis perpendicular to its forward axis.
One backflip is enough, but the robot may keep flipping until the episode ends (10 s).

The task reuses the `solo12` environment through subclassing. It shares the USD, actuators, joint limits,
observation layout (48-D), pushes, observation noise and startup randomizers. The velocity command in the
observation is always zero. Only the reward, the curriculum and a few metrics are new.

## Reward

| Term | Default scale | What it does |
|---|---:|---|
| `backflip_ang_vel` | `5.0` | `scale * (-dot(omega, flip_axis)) * dt`. The unit flip axis is horizontal, perpendicular to body `+x`, and signed toward body `+y` (left). |
| `ang_vel_x_penalty` | `-2.0` | `scale * abs(omega_x) * dt` penalizes roll in either direction, measured in the base frame. |
| `ang_vel_z_penalty` | `-2.0` | `scale * abs(omega_z) * dt` penalizes yaw in either direction, measured in the base frame. |
| `base_collision_terminal` | `-10.0` | One-time penalty when the base touches anything. The episode ends. |
| `undesired_contacts` | `-2.25` | Per second, for each thigh in contact. |
| action rate, joint torque, foot contact, soft joint limit | `0.0` | Preferences. They are off for the first experiments. |

The backflip term is **signed**. For planar flips with clipping disabled, rocking back and forth earns
nothing in the undiscounted sum. That sum is `scale` times the net backward rotation in radians, so one
whole planar backflip is worth `2 * pi * scale` (about 31 with the default scale). For general 3-D motion,
the reward integrates speed about a changing axis; it is not a count of completed flips.

### Horizontal flip axis (reward-hacking fix, 2026-09-27)

The old term used `-omega_body_y`. A robot lying on its side could point body Y vertically and earn
reward by spinning around world Z. The new term gives a pure world-Z spin zero flip reward, at any pose
(up to floating-point tolerance).

We construct the axis in three steps:

1. Compute `c = world_up cross body_x`, with both vectors expressed in the same frame.
2. Normalize `c`. Choose its sign so its dot product with body `+y` is nonnegative.
3. If the cross-product norm is at most `1e-6`, use body `+y`. Here body X is vertical within numerical
   tolerance, so body Y is horizontal within the same tolerance.

The sign rule matters: the raw cross product reverses during the inverted half of a planar flip.
Choosing the sign toward body-left keeps a backward flip positive through the full 360 degrees.
When body-left itself is vertical, it cannot resolve the sign. We keep the cross-product sign in that
exact tie; vertical spinning still projects to zero. Close to a vertical body X, the cross product is
inherently sensitive to small changes in pose. The fallback handles the numerical singularity, not a
smooth heading for every possible 3-D motion.

The code computes this in body coordinates, without extra state or quaternion conversions:

```text
g = projected_gravity_b
c_body = (0, -g_z, g_y)       # (-g) cross (1, 0, 0)
backward_speed = -dot(root_ang_vel_b, signed_unit_axis_body)
reward = backflip_ang_vel_reward_scale * backward_speed * step_dt
```

`projected_gravity_b` is the unit world gravity direction expressed in the body frame, not a noisy
accelerometer sample. This matches the [IsaacLab data definitions](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/assets/articulation/articulation_data.html).

The X and Z angular-speed penalties remain in the **body frame**, with their existing absolute-value
form and scales. The flip counter and curriculum are unchanged. This correction removes the incentive
for vertical spinning; it does not require takeoff or a clean landing.

Set `env.ang_vel_x_penalty_scale=-2.0` and `env.ang_vel_z_penalty_scale=-2.0` on the training
command line to change the new penalties. Set either scale to `0.0` to turn that penalty off.

The scale is 5 because a partial flip that ends on the back must still pay off during learning. A half
turn followed by a crash earns `pi * 5 - 10 > 0`. With a scale of 1 this would be `-6.9`, and the policy
would likely learn to avoid rotating at all.

## Curriculum

The task moves to the next phase when finished episodes average at least 3 completed backflips. The average
is taken over a window of `num_envs` finished episodes. Only episodes that started on the ground count.

| Phase | Startup randomization | Actuation delay (physics steps) | Push force XY | Push force Z |
|---:|---|---|---:|---:|
| 1 | off | `[0, 0]` | `0 N` | `0 N` |
| 2 | on | `[0, 3]` | `0 N` | `0 N` |
| 3 | on | `[0, 3]` | `3 N` | `[-3, 3] N` |
| 4 | on | `[0, 3]` | `5 N` | `[-5, 5] N` |
| 5 | on | `[0, 3]` | `8 N` | `[-8, 8] N` |

Observation noise is on in every phase. A push starts 2-5 s after the reset or after the previous push. It
lasts 0.5-2 s.

Useful overrides:

- `env.backflip_curriculum_start_phase=3` starts at a later phase.
- `--skip_curriculum` starts at the final phase.
- `env.backflip_curriculum=False` disables the curriculum. The direct values `env.actuation_delay_range`,
  `env.base_push_force_xy_range` and `env.base_push_force_z_range` are then used, together with
  `env.include_events_randomization`.
- `env.backflip_curriculum_advance_thresholds=[2.0,3.0,3.0,3.0]` changes the gates (one value per transition).

Each phase keeps its own best checkpoint: `best_model_curriculum_idx_<phase - 1>.pt`.

## Optional ideas (off by default)

- `env.backflip_ang_vel_clip=10.0` clips the signed horizontal-axis speed to `[-10, 10]` rad/s. Use it if impact spikes or very
  violent motions appear. The Genesis Go2 backflip clips at 7.2 rad/s.
- `env.backflip_airborne_reset_prob=0.3` starts 30% of the episodes in the air, at 0.45-0.8 m, already
  rotated backwards by 0-360 degrees and spinning backwards at 4-12 rad/s. The policy then practises
  landings before it can do a whole flip. DeepMimic found that its backflip failed to train when episodes
  did not start from states along the motion ("reference state initialization"). This option is a
  reference-free version of that idea. These episodes are excluded from the curriculum gate and the flip
  metrics. The
  ranges are `backflip_airborne_reset_height_range`, `backflip_airborne_reset_rotation_range` and
  `backflip_airborne_reset_ang_vel_range`.

## Metrics

- `Episode/backflips`: completed backward turns per finished episode. A landing up to 45 degrees short of a
  whole turn still counts.
- `Episode/backward_rotation_rad`: net backward rotation per finished episode.
- `Episode_Reward/backflip_ang_vel`: backflip reward per second.
- `Episode_Reward/ang_vel_x_penalty` and `Episode_Reward/ang_vel_z_penalty`: roll and yaw penalties per second.
- `Curriculum/backflip_phase`, `Curriculum/backflip_window_mean_backflips`, `Curriculum/base_push_force_xy_abs`,
  `Curriculum/base_push_force_z_abs`, `Curriculum/actuation_delay_max`, `Curriculum/events_randomization_active`.

The rotation for these metrics comes from the change of the gravity direction in the base frame. It is exact
for a planar flip and is not affected by angular-velocity spikes at impacts.

Backflip PPO, PPO with symmetry, and SAC runs log to the `jordibelp/solo-backflip` W&B project by default.
Other Solo12 tasks keep their existing project. The training script's `--log_project_name` option can override it.

## Symmetry

`front_back_asymetry=True`, so `--symmetry-mode=augmentation` uses only identity and the left-right mirror.
A front-back mirror would turn a backflip into a frontflip.

Do not pass `env.three_or_more_feet_contact_triggers_reset=True` from the two-feet command. With
`front_back_asymetry=True` it ends the episode whenever a front foot touches the ground.


## Things to watch

- Front thigh range: with your asymmetric thigh limits, the front thigh can move only about 12° above the standing pose. A flip may need more, so a run without asymmetric limits is worth trying. 
- Discount: with γ=0.97 the critic looks about 0.7 s ahead, which may be too short to value a clean landing. Try agent.algorithm.gamma=0.99 if the robot flips but keeps crashing.
- Back walkover: the policy may learn to rotate backward while staying on the ground, like a gymnast's back walkover, instead of flipping in the air. Episode/backflips counts both, so check the videos.
