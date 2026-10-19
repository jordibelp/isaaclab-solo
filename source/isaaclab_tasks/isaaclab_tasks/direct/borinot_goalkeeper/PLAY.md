# Play Borinot goalkeeper weights

Use the dedicated RSL-RL player. `play_direct_0325.py` is a Solo12 player. It expects velocity-command fields and Solo12 robot attributes that this task does not have.

## Play a current checkpoint

Current goalkeeper policies use a flattened 3×3 rotation matrix. With the default observation flags, the actor has 44 inputs in direct control modes or 51 in PI mode. The critic has 56 or 63. Direct-mode checkpoints trained before 2026-10-19 have 53 / 65 inputs; they still observed rotor thrust and the PI integral and need the older code. Use a checkpoint trained with this observation layout. Match all four `env.obs_actor_includes_*` settings from training: predicted plane point, crossing time and phase default to `false`; `env.obs_actor_includes_ball_params` defaults to `true`. Disabling ball parameters removes mass and restitution from the actor, giving 42 direct-mode inputs or 49 PI inputs when the other flags stay off. The critic is unchanged. The renamed flags preserve the previous default layout, but old `env.obs_includes_*` command-line names must be replaced. Older quaternion-input weights do not load directly.

Activate the local environment once:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Then run:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --checkpoint /absolute/path/new_continuous_checkpoint.pt --num_envs 1 --duration_s 2000 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 'env.robot_home=[0.0,0.0,1.5]' env.rwd_scale_energy=0.0
```

The task and agent default to `Borinot-Goalkeeper-Direct-v0` and `rsl_rl_cfg_entry_point`. There is no Solo12 `--task` flag to keep. The checkpoint path is storage only; the source code comes from the canonical `IsaacLab` checkout.

For historical context, we checked the saved W&B configuration for `jordibelp/borinotIsaacLab/ii4tbx06`. It used these control rates, torque actions, home height, and zero energy-reward scale. Its PPO network has 57 quaternion-based inputs, 8 actions, `[512, 256, 128]` hidden layers, ELU activation, and observation normalization. That older checkpoint needs matching code from before the rotation-matrix observation change; it is not compatible with the current environment. The player restores saved normalizers along with the weights for compatible checkpoints.

## Match panel and recent balls

Rerun the same playback command to use the new visualizer. The GUI opens a **Borinot Match** panel beside the viewport. These features are enabled by default:

- **Goals / Blocked** shows the task's resolved goal and confirmed-save counts for environment 0. A racket touch alone is not a save. **Other** contains base/arm hits, floor events, misses, deadlines, and robot failures. The save percentage uses all resolved shots as its denominator.
- **Recent balls** keep moving after a shot ends. Orange is the live shot. Blue balls retain their post-contact pose, linear/angular velocity, mass, inertia, and restitution. They still bounce on the floor and goal. They cannot push Borinot, hit the live ball, or add another score.
- **Keep last 5 balls** includes the current live shot. The oldest ball leaves when a new launch needs its place. Balls do not expire after a fixed time. Abandoned shots at robot resets may remain visible, but are not counted as saves or goals.
- **Reset Borinot** restores the exact configured home position, home orientation, neutral joints, zero velocities, and fresh controller state. It restarts the current shot. It keeps the score and recent balls. With multiple environments, it resets all robots.
- **Pause / Resume** freezes and resumes physics. Reset and camera controls still work while paused.
- **Clear score** starts a new panel score without resetting the robot. The console's cumulative outcome totals remain available.
- **0.25× / 0.5× / 1× / 2×** changes wall-clock pacing, not physics or policy frequency. Slow motion therefore uses the same control timestep. With `--no-real-time`, playback runs unpaced and ignores the speed setting.

The play scene also has a teal goal frame, a translucent goal surface, and a dark floor for contrast. These are visual changes. The original goal and floor colliders remain unchanged.

Optional flags:

```bash
--keep-balls 8 --playback-speed 0.5 --camera keeper
```

Use `--keep-balls 0` for the original immediate ball removal. The supported range is 0–20. Use `--no-ui` to hide the match panel. Headless playback does not create UI widgets.

The dedicated player creates a play-only environment subclass. Training still uses the original one-ball environment, rewards, observations, contact rules, and reset behaviour. No policy weights are changed.

## Camera views

Add one of these flags to the command:

| Flag | What you see |
|---|---|
| `--camera overview` | Default oblique view of the goal, robot, and incoming ball corridor. |
| `--camera front` | View from the incoming-ball side, facing the goal. Useful for sideways and vertical interception. |
| `--camera side` | View from negative y. Useful for ball depth and timing along x. |
| `--camera top` | Overhead view of the goal and shot corridor. |
| `--camera keeper` | View from behind the translucent goal toward the robot and incoming shots. |
| `--camera follow` | Track the robot position from a stable world-aligned offset. Does not rotate with its yaw. |
| `--camera orbit` | Slowly orbit the field. The orbit follows simulated time and stops while paused. |
| `--camera free` | Keep the configured `env.viewer.eye` and `env.viewer.lookat`. Use the usual viewport mouse controls. |

The panel lets you switch all eight cameras during playback. Views use environment 0. Static presets allow manual orbit, pan, and zoom. Follow and Orbit update the view continuously; select Free to stop tracking and keep the current camera pose. Starting with `--camera free` preserves your Hydra eye/lookat settings.

## Evaluation settings

- The policy uses deterministic inference. Action and thrust clipping follow the env flag `remove_action_and_thrust_clippings` (default `True`: no action clipping and no upper thrust limit; thrust always stays at or above zero). For checkpoints trained with the old `[-1, 1]` / 2× hover limits, add `env.remove_action_and_thrust_clippings=False env.action_clamp=1.0 env.thrust_to_weight=2.0`.
- Playback defaults to `env.curriculum=False`. Shot difficulty therefore stays fixed. The default shot level is 0; add `env.shot_level=1`, `2`, or `3` to test wider/faster shots. Add `env.curriculum=True` only if you want progression during play.
- `--duration_s` measures total playback time. A robot episode lasts 20 seconds by default and contains multiple shots. Saves and ball failures recycle only the ball. Safety failures reset the robot. Tune `env.episode_length_s`, `env.shot_timeout_s`, `env.shot_resampling_time_range_s` and `env.shot_resampling_mode` to change timing. The player counts all resolved shots, not only robot resets.
- `env.reset_on_goal=False` is the default: a goal ends only the shot, and Borinot keeps balancing until the next ball. Set `env.reset_on_goal=True` to end the episode and reset the robot, controller, and ball after a goal. Automatic resets use the usual configured spawn grid/noise, not the panel button's exact home reset. Goal scoring and penalties stay unchanged. Other reset conditions remain active in both settings. Restart playback to apply the flag.
- Playback is paced in real time when possible. Add `--no-real-time` for unpaced headless checks. Rendering may make playback slower than real time; no physics steps are skipped.
- Compilation and local-redundancy diagnostics default off for inference. Explicit Hydra overrides still work.
- The runner does not restore optimizer state, train, export policies, or upload a W&B run.
- Match the training environment and agent settings for each checkpoint. The script does not automatically download or reconstruct old run settings. Torque and joint-PD policies both have eight actions, but their action meanings differ.

For a short recording, add `--video --video_length 400`. This records four simulated seconds at 100 Hz and then exits. The MP4 contains the viewport, not the separate controls/score panel. Use `--video_dir /absolute/path/new-output-folder` to choose its destination. Otherwise the script creates a dated folder under `logs/rsl_rl/borinot_goalkeeper/play/`. You can also record with `--headless`.

The final `GOALKEEPER_PLAY` console line reports the checkpoint, resolved rates, camera, duration, and completed shot outcomes. These are local playback results, not a matched held-out benchmark.

## Verification

The task's `tests/test_playback.py` checks score and camera behaviour without Isaac Sim. `tests/playback_smoke.py` checks the real-physics ball pool, wraparound, exact state handoff, randomized physical properties, collision isolation, floor/goal bounces, partial-environment isolation, and home reset. For example, after activating the environment:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/playback_smoke.py --headless --device cpu --keep-balls 5 --output /absolute/path/physics-checks.json
```

Use separate simulator processes to check `--keep-balls 0`, `1`, and `5`. A real GUI check should also click Pause, Reset, all camera modes, speed controls, and Clear score. Inspect the captured viewport and panel; a callback-only check is not proof that the displayed buttons work.
