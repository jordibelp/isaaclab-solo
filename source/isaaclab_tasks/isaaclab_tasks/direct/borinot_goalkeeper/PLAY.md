# Play Borinot goalkeeper weights

Use the dedicated RSL-RL player. `play_direct_0325.py` is a Solo12 player. It expects velocity-command fields and Solo12 robot attributes that this task does not have.

## Your first trained checkpoint

Activate the local environment once:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Then run:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --checkpoint /home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1014_ii4tbx06_model_4690.pt --num_envs 1 --duration_s 2000 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 'env.robot_home=[0.0,0.0,1.5]' env.rwd_scale_energy=0.0
```

The task and agent default to `Borinot-Goalkeeper-Direct-v0` and `rsl_rl_cfg_entry_point`. There is no Solo12 `--task` flag to keep. The checkpoint path is storage only; the source code comes from the canonical `IsaacLab` checkout.

We checked the saved W&B configuration for `jordibelp/borinotIsaacLab/ii4tbx06`. It used these control rates, torque actions, home height, and zero energy-reward scale. Its PPO network has 57 inputs, 8 actions, `[512, 256, 128]` hidden layers, ELU activation, and observation normalization. The player restores the saved normalizers along with the weights.

## Camera views

Add one of these flags to the command:

| Flag | What you see |
|---|---|
| `--camera overview` | Default oblique view of the goal, robot, and incoming ball corridor. |
| `--camera front` | View from the incoming-ball side, facing the goal. Useful for sideways and vertical interception. |
| `--camera side` | View from negative y. Useful for ball depth and timing along x. |
| `--camera free` | Keep the configured `env.viewer.eye` and `env.viewer.lookat`. Use the usual viewport mouse controls. |

The views are anchored to environment 0, not the robot's yaw. The camera does not chase a falling or rotating robot. All views allow manual orbit, pan, and zoom because the script does not rewrite the camera every step.

## Evaluation settings

- The policy uses deterministic inference, with action clipping from the registered agent config.
- Playback defaults to `env.curriculum=False`. Shot difficulty therefore stays fixed. The default shot level is 0; add `env.shot_level=1`, `2`, or `3` to test wider/faster shots. Add `env.curriculum=True` only if you want progression during play.
- `--duration_s` measures simulated time. It does not change the task's three-second shot deadline or terminal resets. Each completed shot resets automatically.
- Playback is paced in real time when possible. Add `--no-real-time` for unpaced headless checks. Rendering may make playback slower than real time; no physics steps are skipped.
- Compilation and local-redundancy diagnostics default off for inference. Explicit Hydra overrides still work.
- The runner does not restore optimizer state, train, export policies, or upload a W&B run.
- Match the training environment and agent settings for each checkpoint. The script does not automatically download or reconstruct old run settings. Torque and joint-PD policies both have eight actions, but their action meanings differ.

For a short recording, add `--video --video_length 400`. This records four simulated seconds at 100 Hz and then exits. Use `--video_dir /absolute/path/new-output-folder` to choose its destination. Otherwise the script creates a dated folder under `logs/rsl_rl/borinot_goalkeeper/play/`. You can also record with `--headless`.

The final `GOALKEEPER_PLAY` console line reports the checkpoint, resolved rates, camera, duration, and completed shot outcomes. These are local playback results, not a matched held-out benchmark.
