# Borinot: base control using the measured hip wrench

This optional controller is available in `Borinot-Goalkeeper-Direct-v0`,
`Borinot-Goalkeeper-Simple-Direct-v0`, and `Borinot-cmd-vel`.
Select it with `env.control_mode=wrench`.
Existing controller defaults are unchanged. Nothing is deployed to hardware or the cluster.

## Quick start

Activate the local simulator environment first:

```bash
cd /home/jordibelp/IsaacLab
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

PPO example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-cmd-vel --num_envs 128 --headless --logger tensorboard env.control_mode=wrench '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

SAC-CE example with a small replay buffer for the laptop:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-cmd-vel --agent rsl_rl_sac_cfg_entry_point --num_envs 32 --headless --logger tensorboard env.control_mode=wrench agent.algorithm.replay_buffer_size=4096 agent.algorithm.num_mini_batches=2 agent.algorithm.mini_batch_size=32 agent.local_redundancy.batch_size=8 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Use `--agent rsl_rl_sac_mse_cfg_entry_point` for SAC-MSE.
Replace the task name for either goalkeeper task.
These examples do not change the task's reward or curriculum.
On a training machine, choose the environment count and replay size separately.
The examples are local commands, not Slurm submissions.

Playback:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_cmd_vel.py --checkpoint /absolute/path/model.pt env.control_mode=wrench
```

For goalkeeper tasks use `play_borinot_goalkeeper.py --task <task-name>` instead.
For SAC, pass the same `--agent` used during training.
Restore the training overrides, including network sizes, observations, gains and timing.
The players do not automatically restore every setting from `params/env.yaml`.

**Train a fresh policy.** The new action order differs from INDI even though both have nine outputs.
An INDI checkpoint is not compatible with this interface.

## The nine policy outputs

| Indices | Meaning | PPO | SAC default |
| --- | --- | --- | --- |
| `0:3` | Desired base CoM acceleration, world axes | Raw m/s² | `8 * clip(action,-1,1)` per axis |
| `3` | Desired world heading | Absolute radians | Home yaw + `pi * clip(action,-1,1)` |
| `4:7` | Desired angular velocity, current base axes | Raw rad/s | `body_rate_scale * clip(action,-1,1)` |
| `7:9` | Desired thigh and calf angles | Absolute radians | ±90° per joint |

Acceleration excludes gravity: zero means zero kinematic acceleration, not free fall.
PPO references are not clipped to `[-1,1]` or multiplied by the SAC scales.
For PPO, zero heading means world yaw zero. It does **not** mean home yaw.
For SAC, zero heading action means home yaw, as in the existing INDI SAC interface.

The leg uses explicit software PD on every physics step:
`effort = kp * (q_desired - q) - kd * q_dot`, clipped to `±effort_lim_sim`.
There is no integral term. The implicit actuator's stiffness and damping are zero,
so PhysX does not add a second PD controller.
Armature, friction, effort limits and velocity limits remain in the plant.
`hold_arm=True` requests neutral-angle PD rather than zero torque.

The controller appends six realised rotor-thrust states, normalised by nominal hover thrust.
It does not add the privileged hip wrench to the policy observation.
With default observation flags, actor/critic sizes are:

| Task | Actor | Critic | Actions |
| --- | ---: | ---: | ---: |
| Goalkeeper | 51 | 63 | 9 |
| Goalkeeper Simple | 51 | 61 | 9 |
| Command velocity | 41 | 41 | 9 |

The defaults remain 100 Hz policy and 200 Hz physics.
Changing the control mode does not change those clocks.

## What we measure

The runtime reads exactly:

```python
thigh_id = robot.find_bodies("flying_arm_2__link1")[0][0]
raw = robot.data.body_incoming_joint_wrench_b[:, thigh_id]
```

This is the complete proximal-joint wrench, not the thigh motor torque.
The lower leg's inertia, gravity and transmitted environment contacts already affect it.
Do not add calf motor effort, leg gravity or foot contact force a second time.

On the tested Isaac Sim 5.1 / tensor API 107.3 installation:

- Entries are force XYZ followed by moment XYZ, in N and N·m.
- The sign is parent-on-child: base-on-leg.
- The coordinates are **child joint axes**, at the child joint anchor.
- The IsaacLab wrapper's parent-frame wording is misleading for this backend.

We read USD `localRot1` and `localPos1` once to locate that frame in the thigh.
Each physics step uses the current thigh pose to rotate the reading to world axes.
We reverse its sign, shift its moment to the actual base CoM, then express that moment in base axes.
The child joint frame moves with the thigh; a fixed base-frame rotation is not sufficient.

The installed [tensor API documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
states the child-joint convention. The rotated known-load fixture also verifies it.
Re-run that fixture after changing simulator backend or wrench API.

## Base-only control law

Here the subsystem is the **base rigid body**, not the whole robot.
Let `F_leg_world` and `M_leg_base` be the converted leg-on-base wrench.
The force request is:

```text
F_rotor_desired_world = mass_base * (acceleration_desired_world - gravity_world) - F_leg_world
```

We use its magnitude for collective thrust.
Its direction and the desired heading define the desired attitude.
This is a desired thrust direction: the real rotors still push only along the current base +Z axis.
They cannot produce an arbitrary world force immediately.
With near-zero requested force we retain the current thrust axis to avoid an arbitrary attitude jump.

Attitude feedback uses the same tangent-space implementation as INDI.
Defaults are tilt gain 150, yaw gain 3, and rate gains `(20,20,8)`.
Set `env.wrench.tilt_method=geom`, `method_1`, or `method_2` for the existing definitions.
There is no differentiated policy reference or angular-acceleration feedforward.

The rotor moment request, about the base CoM and in base axes, is:

```text
torque_rotor_desired = inertia_base * angular_acceleration_desired
                       + omega cross (inertia_base * omega) - M_leg_base
```

The model uses each clone's **actual base-only** mass, inertia and CoM after startup randomization.
This is an ideal measured-wrench, model-known simulator baseline.
It is not evidence of robustness to unknown inertial parameters.
Those properties are cached at startup; changing them later needs a matching controller refresh.

The hip wrench is used only in this calculation.
We never apply it as an additional simulated force: PhysX already applies the leg reaction.

## Bounded allocation only

The mixer has four rows and six rotor columns: `[collective, tau_x, tau_y, tau_z]`.
It has no ordinary matrix inverse.
Following Jordi's updated request, there is **no inverse-and-clip mode**.
Do not pass the earlier experimental `env.wrench.allocation_mode` option; it has been removed.

We solve weighted least squares with `0 <= rotor_thrust_i <= physical_cap`.
The cap is the smaller of the task's thrust-to-weight cap and `rotor_thrust_limit_n`.
Each clone's mixer is referenced to its actual base CoM and matches the physical rotor geometry.

The four error components are divided by their individual rotor capacities before weighting.
Defaults are `env.wrench.allocation_weights=[1,10,10,1]`: squared-error weights for
collective, roll moment, pitch moment and yaw moment.
The fixed-budget projected solver uses 128 iterations by default.
An infeasible request leaves an allocation residual; the solver does not pretend it was realised exactly.

A pseudoinverse is still used to initialise the bounded solver and to report the original
minimum-norm rotor request for the existing thrust-limit cost.
It is not a selectable actuator path. That diagnostic can exceed limits even when another
rotor distribution could realise the same wrench, so it is not a strict infeasibility test.

Commands then pass through the same first-order **motor-speed** lag as INDI.
No rotor spin-inertia term is introduced; it remains absent from this controller and its plant path.

## Contacts, delay and reset limits

The sample describes the **last completed physics step**, not the next impact.
The controller runs at every physics step, using the newest available sample.
It cannot undo an impulse that has already happened, and motor lag remains present.
After reset, it holds reset hover for one physics tick while waiting for a fresh wrench.
It clears selected controller buffers and joint effort targets without resetting other clones.
Existing policy-command delay options also apply to the new mode.

Leg/environment contact transmitted through the hip is included.
Direct base contact, a base push, and leg/base self-contact create other paths across the base boundary.
The hip sensor alone does not measure all those loads.
Contact penalties do not guarantee those paths never occur.

Strong contact can request an unavailable force direction or a large attitude change.
For example, a large upward leg force with a zero acceleration request can ask the rotors to push downward.
The force-to-attitude construction can then request inversion.
We have not added an unrequested contact-mode switch or attitude restriction.
The bounded rotor solver alone cannot resolve this whole-body feasibility problem.

## Verification and its scope

The implementation has simulator-free mechanics tests, live Hydra/PhysX checks and
short public PPO/SAC training-and-reload checks.
The evidence is in the dated workspace artifact folder, not in production experiment logs.

- Rotated static gravity and known-load fixture: wrench conversion agrees within 0.00011 N/N·m.
- Independent spatial-wrench, Newton/Euler and action-unit tests cover non-diagonal inertia and different clone CoMs.
- Float32 and float64 bounded allocation are checked against SciPy, including feasible and infeasible requests.
- All three tasks exercise measured-wrench input, explicit PD, physical limits, source capture and reset isolation.
- PPO, SAC-CE and SAC-MSE train fresh checkpoints and reload through the public players.

The falling-calf contact probe confirms force transmission through the hip.
At 200 Hz, the base force balance without direct base contact agrees within 0.00019 N,
including a calf contact sample above 500 N.
The same sharp-impact torque comparison has a peak discrepancy of about 4.3 N·m.
The static moment check and continuous equations pass, but that impact comparison does **not**
establish exact instantaneous torque cancellation. A smaller timestep did not remove the issue.
The probe combines finite-step solver averages with endpoint pose/acceleration data;
we have not established the full cause of that discrepancy.

Controller-only eager PyTorch timing on the local RTX 5070, excluding simulator and policy:

| Batch | Bounded controller mean wall time |
| --- | ---: |
| 1 | 4.98 ms |
| 10,000 | 7.18 ms |

These are short local measurements, not a hardware real-time guarantee.
The 10,000-state controller batch is not proof that a 10,000-environment training run fits the laptop.
The integration smokes are not a learned goalkeeper score or a contact-robustness result.

### Reproduce the main checks

CPU tests, from the goalkeeper `tests` directory:

```bash
PYTHONPATH=.:/home/jordibelp/IsaacLab/source/rsl_rl_sac_vendor python -m pytest -q --confcutdir=. --rootdir=. test_wrench.py
```

Live integration, after the activation above:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/wrench_smoke.py --task Borinot-cmd-vel --headless --device cuda:0 --output /absolute/scratch/result.json env.control_mode=wrench
```

The adjacent `wrench_frame_probe.py` checks the installed sensor sign, frame and point.
Run one live simulator script per process.
