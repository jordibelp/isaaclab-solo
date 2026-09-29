# Solo12 two-feet SAC curriculum

The `solo12-two-feet` task defaults to the coordinated `two_feet_sac` profile. It advances on
scale-independent fixed-horizon reward ratios. These are logged as
`Episode_Reward_ratio/*` and equal `Episode_Reward/* / abs(reward_scale)`.

| Phase | Gate to next phase | Height scale / alpha | XY scale | Terrain | Delay | Startup randomization | Push magnitude |
|---:|---|---|---:|---|---|---|---|
| 1 | `two_feet_above_height >= 0.70` | `1.7 / 15` | `1.2` | flat | `[0, 0]` | off | `0 N` |
| 2 | `track_lin_vel_xy_exp >= 0.70` | `1.2 / 20` | `1.6` | flat | `[0, 0]` | off | `0 N` |
| 3 | `two_feet_above_height >= 0.70` | `1.5 / 25` | `1.5` | tricky | `[0, 3]` | on | `0 N` |
| 4 | `track_lin_vel_xy_exp >= 0.70` | `1.5 / 25` | `1.5` | tricky | `[0, 3]` | on | `5 N`, Z `[-8, 8] N` |
| 5 | final | `1.5 / 25` | `1.5` | tricky | `[0, 3]` | on | `8 N`, Z `[-8, 8] N` |

All phases use a height threshold of `0.45 m`, `kp=9`, `kd=0.2`, observation corruption,
`vx in [-0.5, 0.5]`, `vy in [-0.3, 0.3]`, hip/base collision filtering, and a
three-or-more-feet contact penalty scale of `-100`.

The ratio gate divides the mean completed-episode reward by
`abs(reward_scale) * max_episode_length_s`. A ratio of `0.70` therefore requires 70% of the
maximum fixed-horizon contribution; an episode that terminates early cannot pass merely because it
performed well during its short lifetime.

## Selection and debugging

Normal training needs no extra curriculum arguments:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task=solo12-two-feet \
  --agent=rsl_rl_sac_cfg_entry_point \
  --headless \
  --num_envs=10000
```

Start directly at a phase for a smoke test or continuation experiment:

```bash
env.two_feet_curriculum_start_phase=3
```

Disable the coordinated curriculum and use direct command-line values:

```bash
env.curriculum_two_feet=False
```

Set `env.curriculum_profile=legacy` while leaving `env.curriculum_two_feet=True` to use the legacy
episode-reward gating mechanics and the separate velocity-then-force progression with the configured
phase arrays.

W&B exposes `Curriculum/two_feet_phase`, `Curriculum/global_idx`,
`Curriculum/advance_reward_ratio`, `Episode_Reward_ratio/*`, the active reward scales, terrain/delay settings,
push magnitude, and `Curriculum/events_randomization_active`.

## Posed starts (`twofeet_airborne_reset_prob`)

This is the two-feet version of `backflip_airborne_reset_prob`. A fraction of the resets starts the robot
reared up on its rear feet, so the policy sees standing states before it can reach them by itself.
DeepMimic calls this idea "reference state initialization". Here we need no reference motion.

A posed start is built from the regular reset:

1. Sample a tilt of the base from the world vertical. 0 means nose straight up. Positive values lean
   forward, toward the front feet. Negative values lean backward.
2. Pivot the base nose-up about the rear hip axis to that tilt.
3. Turn both rear thighs by the same angle. The rear legs then keep their regular standing pose in the
   world, and the rear feet stay at their regular reset height.
4. Lift the whole robot by a random drop height.
5. Replace the reset velocities with small random values.

All other joints and the heading keep their regular reset values. It is off by default.

| Parameter | Default | Meaning |
|---|---|---|
| `twofeet_airborne_reset_prob` | `0.0` | Fraction of resets that start posed. |
| `twofeet_airborne_reset_prob_curriculum` | `()` | One fraction per curriculum phase. Empty uses the value above in every phase. |
| `twofeet_airborne_reset_tilt_range` | `(-30.0, 60.0)` | Tilt from the vertical, in degrees. |
| `twofeet_airborne_reset_drop_height_range` | `(0.0, 0.1)` | Extra height of the rear feet above their regular reset height, in m. |
| `twofeet_airborne_reset_lin_vel_range` | `(-0.1, 0.1)` | Linear velocity per world axis, in m/s. |
| `twofeet_airborne_reset_ang_vel_range` | `(-0.1, 0.1)` | Angular velocity per world axis, in rad/s. |

With the defaults, the base starts 0.45-0.65 m above the ground. The tilt sets most of this height.

Examples (add to the training command):

```bash
env.twofeet_airborne_reset_prob=0.3
'env.twofeet_airborne_reset_prob_curriculum=[0.3,0.3,0.2,0.1,0.1]'
'env.twofeet_airborne_reset_tilt_range=[-30.0,60.0]'
```

When the per-phase list is set, it wins over `twofeet_airborne_reset_prob`.

### Curriculum and metrics

Posed episodes do not count for the curriculum gate. They also do not count for `Episode_Reward/total`,
`Episode_Reward_ratio/*`, `Episode/length_*`, and `Curriculum/advance_reward_ratio`. These metrics
therefore stay comparable with runs without posed starts. `Train/mean_reward`, the per-term
`Episode_Reward/<term>` values, and the `Episode_Termination/*` counts still include posed episodes.
`Curriculum/twofeet_airborne_reset_prob` logs the active fraction.

Play and inference build their configuration from the task defaults, so posed starts stay off there.
Add `env.twofeet_airborne_reset_prob=1.0` to the play command to watch them.

### Why the rear thighs turn, and how the defaults were chosen

We probed the reset in Isaac Sim (2026-09-29). The policy was the phase-5 two-feet SAC checkpoint
`0925_d16x3s6n`, which never saw posed starts. The probe used flat ground, a standing command, the
limits and gains of the current two-feet command, and 3 s episodes with 1024 posed starts per run.
"Survived" means the episode reached 3 s: no base hit, and no front-foot contact after the 1 s grace
period.

| Tilt from vertical | Posed start as above | Tilt only (regular legs, base at 0.4-0.8 m) |
|---:|---:|---:|
| -30 to -20 deg | 100% | 0% |
| -20 to -10 deg | 100% | 29% |
| -10 to 0 deg | 100% | 59% |
| 0 to 30 deg | 100% | 87-99% |
| 30 to 40 deg | 86% | 62% |
| 40 to 50 deg | 40% | 39% |
| 50 to 60 deg | 9% | 27% |
| All | 81% | 55% |

- Tilting only the base, like the backflip reset, leaves the legs pointing sideways relative to gravity.
  A robot that just holds its reset pose then hits the ground with its base in 99.8% of the starts,
  usually within 0.3-0.4 s. With the rear thighs turned, the same passive robot lands on its rear feet.
  When it leans forward by 10 degrees or more, it then tips onto its front feet without a base hit
  (95-100% of the starts). Only the policy can stop a backward lean.
- The only place where tilting the base alone did better was a forward lean of 50-60 degrees (27% against
  9%). There the robot is close to its normal four-legged pose anyway.
- A backward lean of more than 40 degrees was never recovered (0 of 752 starts). So the default range
  stops at 30 degrees backward.
- Leaning forward is the hard side for this policy. It lifts its front legs early in every episode, which
  is the likely reason: a forward fall then lands on the base. These starts are the states along the
  rear-up motion, so we keep them up to 60 degrees.
- Between 30 degrees backward and 30 degrees forward, a drop of up to 0.2 m did not matter (98-100%
  survived). With a forward lean, a drop above 5 cm made survival much worse. So the default drop is
  small (up to 0.1 m).
- No posed start touched the ground with its base at the first step. The rear feet started at their
  regular reset height plus the drop.
