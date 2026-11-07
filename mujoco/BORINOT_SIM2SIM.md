# Borinot: IsaacLab policies in MuJoCo

Run your existing Borinot PPO or SAC checkpoints with MuJoCo physics and rendering.
Isaac Sim does not start. Use the same conda environment, player options and Hydra
`env.*` overrides as the Isaac players.

## Players

| Isaac player under `source/scripts/rsl_rl/` | MuJoCo player under `mujoco/` | Tasks |
| --- | --- | --- |
| `play_borinot_goto.py` | `play_borinot_goto_mujoco.py` | `Borinot-goto` |
| `play_borinot_cmd_vel.py` | `play_borinot_cmd_vel_mujoco.py` | `Borinot-cmd-vel` |
| `play_borinot_goalkeeper.py` | `play_borinot_goalkeeper_mujoco.py` | `Borinot-Goalkeeper-Direct-v0`, `Borinot-Goalkeeper-Simple-Direct-v0` |

The full task aliases accepted by the Isaac players also work here. The goalkeeper
player handles both goalkeeper tasks; there is no separate Simple player to maintain.
The Solo12 player is unchanged.

## Setup

Run these commands in your terminal. The examples below assume the canonical checkout.

```bash
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/jordibelp/IsaacLab
```

MuJoCo 3.3.7 and Tkinter are already installed in this environment. The players also
need the installed IsaacLab Python packages and Isaac Sim's USD libraries. They read
the local USD asset, but they do not launch Kit, PhysX or the Isaac Sim renderer.
This is not a standalone package for a machine without IsaacLab installed.

## GoTo: your supplied checkpoint

This is your original command with only the player path changed:

```bash
./isaaclab.sh -p mujoco/play_borinot_goto_mujoco.py --task=Borinot-goto --checkpoint=/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1107_1ec752ct_best_model_curriculum_idx_9_wrench.pt --num_envs=1 --duration_s=2000 --goal_init 3.0 0.0 1.5 env.control_mode=wrench env.physics_hz=200 env.policy_hz=100 env.curriculum=False env.obs_include_base_lin_acc=True env.obs_include_base_lin_acc_steps=2
```

You get the MuJoCo viewer and a separate native control window. The control window
keeps the existing XYZ goal, world/relative frame, manual/automatic deadline, pause,
reset and camera controls. Press Enter or leave an exact-value field to apply an edit.
Typed values can exceed the slider range. The slider does not round or clamp them.

Use `--deadline_s=2.0` for a fixed two-second goal budget. Without it, the original
player chooses an automatic deadline from the current training schedule. It flies
to the goal and then holds; `--reset-at-deadline` restores training-style episodes.
Use `--random-goals` for the training goal sampler instead of live goal controls.

The checkpoint's acceleration history is part of its input layout. Keep both
`env.obs_include_base_lin_acc=True` and `env.obs_include_base_lin_acc_steps=2` for
this checkpoint. They produce 46 actor inputs with this wrench configuration.

## Command velocity

Set `CMD_CHECKPOINT` to a compatible command-velocity checkpoint. Keep its training
control mode and observation flags. This example assumes the current direct-torque
layout; it does not infer those settings from the filename.

```bash
CMD_CHECKPOINT=/absolute/path/to/your/cmd_vel_checkpoint.pt
./isaaclab.sh -p mujoco/play_borinot_cmd_vel_mujoco.py --task=Borinot-cmd-vel --checkpoint="$CMD_CHECKPOINT" --num_envs=1 --duration_s=2000 --cmd_init 0.1 0.0 0.0 0.0 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 env.curriculum=False
```

The four command values are heading-frame X/Y velocity in m/s, world Z velocity
in m/s, and world-Z angular velocity in rad/s. The native control window keeps
the command fields, zero/reset/pause buttons, camera controls, pushes and overlays.
Use `--random-commands` to restore training command sampling.

Older weights can need explicit legacy observation and inertial settings. For example,
the October 23 `1023_aunmwlq7_best_model_curriculum_idx_2.pt` loads with
`env.include_base_height=False env.projected_gravity_in_obs=True`, giving its 34
inputs. Our loader check also used
`env.inertial_properties.include_motor_masses=False` and
`env.inertial_properties.use_inference_leg_inertias=False` for the older plant.
That short flight was finite but reset four times in five simulated seconds.
It is a compatibility check, not a successful tracking result.

## Goalkeeper and Goalkeeper Simple

Set `KEEPER_CHECKPOINT` to a compatible current goalkeeper checkpoint and use its
training control mode and observation flags. For the standard direct-torque layout:

```bash
KEEPER_CHECKPOINT=/absolute/path/to/your/goalkeeper_checkpoint.pt
./isaaclab.sh -p mujoco/play_borinot_goalkeeper_mujoco.py --task=Borinot-Goalkeeper-Direct-v0 --checkpoint="$KEEPER_CHECKPOINT" --num_envs=1 --duration_s=2000 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 env.curriculum=False
```

For Simple, use a checkpoint trained for Simple and change the task:

```bash
SIMPLE_CHECKPOINT=/absolute/path/to/your/goalkeeper_simple_checkpoint.pt
./isaaclab.sh -p mujoco/play_borinot_goalkeeper_mujoco.py --task=Borinot-Goalkeeper-Simple-Direct-v0 --checkpoint="$SIMPLE_CHECKPOINT" --num_envs=1 --duration_s=2000 env.control_mode=thrust_and_torque_control env.physics_hz=200 env.policy_hz=100 env.curriculum=False
```

The match window keeps the score, recent balls, pause/reset, camera presets and
0.25x/0.5x/1x/2x playback buttons. Close the MuJoCo viewer to stop playback.
Closing only the control window hides it and leaves playback running.

Pre-October-19 quaternion-observation goalkeeper checkpoints, such as
`1014_ii4tbx06_model_4690.pt`, do not match the current Isaac task either. This port
does not silently translate old observation layouts or reinterpret their weights.

## SAC

PPO remains the default. Add the same SAC agent key used in Isaac playback:

| Checkpoint algorithm | Player option |
| --- | --- |
| PPO | `--agent=rsl_rl_cfg_entry_point` |
| SAC categorical critic | `--agent=rsl_rl_sac_cfg_entry_point` |
| SAC MSE critic | `--agent=rsl_rl_sac_mse_cfg_entry_point` |

The original loaders restore the policy and its saved normalizers, but not optimizer
state. SAC retains its own action mapping. Matching tensor sizes alone is not enough:
also match the task, action meanings, angle encoding, controller, gains and rates.

## Playback speed

All players pace playback at real time by default. Add `--no-real-time` to remove
that pacing and run as fast as the CPU/viewer allows. Add `--headless` to remove
both windows. Use `--no-ui` if you want the viewer without the task control window.
The viewer and control window refresh at up to 30 frames/s by default, independently
of physics and policy. Change this with `--mj_fps=60` or `--mj_fps=20`. No physics
or policy steps are skipped. Live numeric labels are drawn only on a UI refresh.

For example, a short unpaced benchmark of the supplied GoTo policy:

```bash
./isaaclab.sh -p mujoco/play_borinot_goto_mujoco.py --checkpoint=/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1107_1ec752ct_best_model_curriculum_idx_9_wrench.pt --num_envs=1 --headless --no-real-time --duration_s=10 --goal_init 3.0 0.0 1.5 --log_interval_s=0 env.control_mode=wrench env.physics_hz=200 env.policy_hz=100 env.curriculum=False env.obs_include_base_lin_acc=True env.obs_include_base_lin_acc_steps=2
```

At shutdown, `MUJOCO_PLAYBACK` prints the simulated time, rollout wall time and
`real_time_factor`. A factor of 1 means real time; 2 means twice real time. The
measurement runs from the first physics-step start to the last physics-step end.
It excludes startup and includes work, pacing and pauses between those steps.
It is a playback measurement, not a physics-kernel-only benchmark.

The supplied GoTo checkpoint ran 10 simulated seconds in 9.44 wall seconds in a
headless, unpaced check on piuMachine: about **1.06x real time**. Other controllers,
the viewer, contact scenes and other running programs change the speed. This port
does not guarantee that every mode will run faster than real time.
Later six-second checks were around **0.9–1.1x headless** and **0.6–0.7x with the
viewer and controls** on the loaded desktop. Visible playback of this wrench policy
has not fully reached real time on this laptop. Physics, policy rates and allocation
iterations have not been reduced to improve that number.

The passive viewer normally syncs integration state rather than copying unchanged
meshes each frame. It performs a full sync after model-property changes. See the
[MuJoCo 3.3.7 passive-viewer API](https://mujoco.readthedocs.io/en/3.3.7/python.html#passive-viewer).
On GNOME Wayland it uses XWayland when available, matching the existing Solo12 player.

## What carries over, and what changes

- The players execute the existing Isaac play scripts on an isolated MuJoCo adapter.
  They reuse task configs, observation order/history, controllers, motor lag, action
  transforms, rewards, reset rules, curriculum playback logic and UI callbacks.
  No training task or Isaac player source is modified.
- MuJoCo runs physics and rendering. The task-edited training USD supplies joint
  frames, full inertias, centres of mass, collisions and visual meshes. The existing
  inference-leg inertia option already includes the URDF-derived leg model. A second
  conversion of the raw URDF would miss task-time payload/racket/inertia changes.
- Actuator **armature is also copied**, separately from the link inertias. The
  current shared arm actuator sets **0.00036207 kg m² for each arm joint** in
  `borinot_goalkeeper_env_cfg.py`. The adapter writes it to MuJoCo's `dof_armature`
  in policy joint order. It adds inertia on the joint coordinates, not an extra link
  mass. The startup log prints both actual values. All four tasks inherit this
  setting; model-based controllers read the same configuration value.
  Your checked URDF/xacro files do not declare armature. Their link inertias and
  joint damping/friction fields are not substitutes for this actuator setting.
  To override both joints through Hydra, add
  `env.robot.actuators.arm.armature=0.00036207` to any player command. The default
  already matches, so no override is needed for the supplied checkpoint.
- Environment and policy tensors run on CPU, with one Torch thread. `--device` does
  not move this backend to CUDA. Only `--num_envs=1` is supported.
- Startup domain randomization defaults to **off** for a nominal, repeatable plant.
  Set `env.include_events_randomization=True` to opt into the supported uniform
  mass, centre-of-mass and joint-property randomizers. Reset pose noise, ball sampling
  and curriculum-selected pushes still follow the task settings.
- Flight uses one MuJoCo step per configured physics tick. Goalkeepers use ten
  substeps per tick: at 200 Hz this gives a 0.5 ms contact timestep. The controller
  and acceleration history still update at the requested physics frequency.
  Override this with `--mj_substeps=N`, where N is a positive integer. Fewer contact
  substeps can change impacts or miss thin obstacles.
- Contact forces and incoming joint wrenches are averaged over a configured physics
  tick. Acceleration uses the velocity change over that same tick. The wrench
  controller therefore reads the MuJoCo joint reaction, not an Isaac placeholder.
- PhysX and MuJoCo contacts are not identical. Restitution is approximated with a
  MuJoCo spring/damper contact. PhysX static joint friction is approximated by
  MuJoCo Coulomb friction. Equal parameter values do not promise identical rollouts.
- Checkpoints and robot assets must be local files. Isaac renderer/Kit launch options
  are reported as ignored. Unsupported runtime APIs fail explicitly.

The goalkeeper's existing `--video` option also works. For headless offscreen rendering,
prefix the command with `MUJOCO_GL=egl`. Videos are slower than ordinary playback.
Use an explicit `--video_dir` to choose where research outputs are written.

## Verification

The completed checks distinguish mechanics, integration and learned performance:

- 63 headless checks cover mass/inertia, state frames, free fall, a Newton–Euler
  joint-wrench oracle, contact force/restitution, markers, reset cleanup, lifecycle,
  invalid substeps, and all four tasks with all 11 controller modes.
- The armature checks cover the default and separate per-joint values in all four
  tasks. They compare the compiled values and the full joint-space mass matrix
  with/without armature, then check persistence after reset and stepping. The matrix
  difference is exactly the requested diagonal joint inertia, within 1e-14 kg m².
- Three native Tk checks verify that exact fields survive deferred slider callbacks,
  including values outside the slider range, while dragging still updates the model.
- GUI/headless checks with the supplied checkpoint match all final rollout report
  values exactly over 600 policy steps, including unchanged typed goal coordinates.
  This checks that rendering does not change the control loop or advance physics.
- A public Hydra launch with `env.robot.actuators.arm.armature=0.0007` reports
  0.0007 kg m² on both compiled joints and completes finite inference.
- Eight public-player checks load and run current PPO and SAC-MSE checkpoints for
  all four tasks. These checkpoints were generated without optimizer updates. They
  establish the loader/playback paths, not trained task performance.
- The supplied trained GoTo checkpoint ran 1,000 finite headless policy steps. Earlier
  saved GUI evidence contains 1,400 finite GoTo steps and a 1,400-step goalkeeper match
  using untrained weights. The short GoTo check missed the strict deadline/rest rule;
  this is not a claim of perfect task transfer.
- Saved one-second, open-loop IsaacSim/MuJoCo traces have identical reported masses
  and centre-of-mass positions, with maximum inertia-entry difference below 1e-8
  kg m². Maximum root-position coordinate difference was 0.0122 m. They use scripted
  wrenches and arm PD targets, not a policy, to separate mechanics from feedback.

Run the headless checks from the repository root:

```bash
./isaaclab.sh -p -m pytest --confcutdir=mujoco mujoco/test_borinot_sim2sim.py -s -q
```

Tk checks skip without `DISPLAY`; run them from a desktop terminal to include them.
Keep `-s`: the bootstrap re-executes Python once to load USD libraries, which otherwise
can inherit pytest's capture file descriptors and hide the result output.

The independent physics comparison helper is `mujoco/borinot_parity_record.py`:

```bash
./isaaclab.sh -p mujoco/borinot_parity_record.py --backend mujoco --out /tmp/borinot-mujoco.npz
./isaaclab.sh -p mujoco/borinot_parity_record.py --backend isaac --out /tmp/borinot-isaac.npz
./isaaclab.sh -p mujoco/borinot_parity_record.py --compare /tmp/borinot-isaac.npz /tmp/borinot-mujoco.npz
```

Only the `--backend isaac` command starts Isaac Sim. Activate `env_isaaclab` and
use the usual certificate environment variables before that command.
