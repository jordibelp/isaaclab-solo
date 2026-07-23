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
