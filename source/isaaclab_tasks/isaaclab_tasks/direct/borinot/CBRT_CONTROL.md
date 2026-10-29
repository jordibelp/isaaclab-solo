# Borinot collective-thrust/body-rate controllers

These optional modes are available in:

- `Borinot-Goalkeeper-Direct-v0`
- `Borinot-Goalkeeper-Simple-Direct-v0`
- `Borinot-cmd-vel` (also registered as `Borinot-Cmd-Vel-Direct-v0`)

| Selector | Base controller | Bounded allocations per physics tick |
| --- | --- | ---: |
| `env.control_mode=cbrt` | Proportional body-rate feedback, nominal Euler inversion | 1 |
| `env.control_mode=cbrt_indi` | Same first stage, then measurement-based INDI refinement | 2 |

`cbtr` and `cbtr_indi` are accepted spelling aliases. Saved configs use the canonical names above.
Both modes share `env.cbrt.*` settings.
The default control selector and every older mode's action mapping remain unchanged.

## Relationship to the existing code and paper

The older `PI_rotors_PD_joint_pos_control` mode is not this implementation.
It uses rate PI feedback, a different action order and inverse-then-clip allocation.
It remains unchanged for existing experiments.

Section 6.2 of [Deshmukh et al., arXiv:2512.21085v2](https://arxiv.org/html/2512.21085v2#S6.SS2)
calls the interface **CTBR**: collective thrust plus body rate.
Its baseline retains INDI after proportional rate feedback.
Our `cbrt_indi` follows that architecture. Plain `cbrt` is an additional no-INDI comparison.
This is a Borinot adaptation, not an exact reproduction of the paper's vehicle or training recipe.

## Six policy actions

The order follows Jordi's sketch, not the old PI mode:

| Indices | Meaning | PPO | SAC |
| --- | --- | --- | --- |
| `0:3` | Desired angular velocity in current base axes | Raw rad/s | `body_rate_scale * clip(action,-1,1)` |
| `3` | Desired collective propeller thrust | Raw newtons for all six rotors together | Hover-centred mapping below |
| `4:6` | Desired thigh and calf angles | Absolute radians | `cbrt.joint_target_limit_degs * clip(action,-1,1)`, converted to radians |

For SAC collective action `-1 / 0 / +1` requests `zero / nominal hover / maximum total thrust`.
The two halves are linear, so zero stays at hover even when the thrust-to-weight cap changes.
The effective rotor cap is the smaller of the thrust-to-weight cap and `rotor_thrust_limit_n`.
Default SAC scales are body rates `(2,2,1.5)` rad/s and joint angles `(90,90)` degrees.

PPO references are not clipped to `[-1,1]` or multiplied by the SAC scales.
In particular, PPO action 3 equal to zero requests **zero thrust**, not hover.
The bounded solver constrains the resulting actuators, not the reference vector.
This physical-unit collective can require a different PPO exploration/noise setup than a hover-centred action.

Both modes append six realised rotor-thrust states, normalised by nominal hover thrust.
Neither adds a joint integral state or privileged hip-wrench observation.
With default observation flags:

| Task | Actor inputs | Critic inputs | Actions |
| --- | ---: | ---: | ---: |
| Goalkeeper | 48 | 60 | 6 |
| Goalkeeper Simple | 48 | 58 | 6 |
| Command velocity | 38 | 38 | 6 |

Train a fresh policy. Matching the old PI action count does not make its checkpoint compatible.
The new modes have matching action and observation meanings with each other.
Switching between them still changes the plant/controller response.

## First stage: proportional rate control

The error is desired minus measured **angular velocity**, not angular acceleration.
The requested body angular acceleration is the componentwise product:

```text
alpha_desired = rate_gain * (omega_desired - omega)
torque_desired = I_base * alpha_desired + omega cross (I_base * omega)
```

There is no desired attitude, heading controller, attitude-error gain or differentiated policy reference.
The policy must choose body rates that produce the attitude and motion needed by the task.
The model uses nominal base-only inertia, scaled by `cbrt.inertia_scale`.
It does not silently read each randomized clone's inertia into the model.
Leg reaction and model mismatch remain disturbances.

The bounded solver minimizes weighted collective/moment error while keeping each thrust within `[0, cap]`.
The wrench order is `[collective, tau_x, tau_y, tau_z]`.
The mixer uses Borinot's actual rotor geometry about the nominal base CoM.
The existing plant code still shifts the realised rotor moment to each clone's actual CoM.
There is no ordinary inverse of the 4×6 mixer and no inverse-and-clip mode.
A pseudoinverse supplies the solver's initial guess and existing raw-request limit diagnostic only.

## Second stage: `cbrt_indi`

First, evaluate the wrench produced by the first bounded allocation.
Its achievable angular acceleration is:

```text
alpha_feasible = inverse(I_base) * (first_stage_torque - omega cross (I_base * omega))
```

This matters when the original request cannot be realised.
Using the raw desired acceleration again would discard the first feasibility step.

Next, filter measured body rates and **realised, lagged rotor speeds** with matched second-order Butterworth filters.
Differentiate the filtered body rates at the physics timestep.
Compute filtered rotor torque from the filtered realised speeds and the thrust/moment model.
The new torque request is:

```text
torque_refined = torque_filtered + I_base * (alpha_feasible - alpha_measured_filtered)
```

Do not use the just-computed first-stage thrust commands as the measured torque.
Do not add the gyro term again in this incremental equation.
The first-stage gyro term has already entered `alpha_feasible`.

The second bounded solve targets this refined torque and the first stage's achievable collective.
It may still trade collective against torque when that combined request is infeasible.
Only the final command enters the motor model, once per physics tick.
The allocation residual records the remaining unachieved request.

This mode estimates disturbance effects through measured angular response.
It does **not** read `body_incoming_joint_wrench_b` or add the leg-on-base wrench explicitly.
That is the separate `wrench` mode.
It also does not directly estimate external linear force: collective remains policy-controlled.

## Leg PD, motor model and resets

Both modes use explicit leg PD, with output clipping to `±effort_lim_sim`:

```text
joint_effort = kp * (q_desired - q) - kd * q_dot
```

The implicit actuator gains are zero, so there is no second PD loop.
There is no joint integral term. `hold_arm=True` requests neutral-angle PD.
Armature, friction, joint velocity limits and physical joint stops remain in PhysX.
Motor response is the same first-order **speed** lag as the existing INDI mode.
There is no added rotor spin-inertia term in either new mode or its plant path.

The controller runs at each physics tick; references are held at the policy rate.
Defaults remain 200 Hz physics and 100 Hz policy.
Existing actuator-command delay and limit-cost options also apply.
The original desired rotor distribution is retained for the existing thrust-limit cost;
its pseudoinverse distribution is not a strict wrench-infeasibility certificate.

Selected resets clear effort targets and controller/filter history without clearing other clones.
Rotor states reset to hover, bounded by the physical cap.
The first post-reset INDI tick uses the first allocation without measurement refinement:
no completed acceleration sample exists yet.
The gyro filter starts from the reset's current rate, avoiding an artificial derivative impulse.

## Tunable settings and starting values

| Config | Default | Role |
| --- | --- | --- |
| `env.cbrt.rate_gain` | `[20.0,20.0,8.0]` | Rate-error to angular-acceleration gain, 1/s; zero disables an axis |
| `env.cbrt.inertia_scale` | `1.0` | Scale of the fixed nominal base inertia |
| `env.cbrt.allocation_weights` | `[1.0,10.0,10.0,1.0]` | Squared, capacity-normalised collective/roll/pitch/yaw errors |
| `env.cbrt.allocation_iterations` | `128` | Projected-solver iteration budget per allocation |
| `env.cbrt.filter_cutoff_hz` | `12.0` | Matched gyro/speed cutoff; used only in `cbrt_indi` |
| `env.cbrt.joint_target_limit_degs` | `[90.0,90.0]` | SAC joint-reference half-ranges |
| `env.body_rate_scale` | `[2.0,2.0,1.5]` | SAC body-rate half-ranges, rad/s |
| `env.kp`, `env.kd` | `9.0`, `0.2` | Leg PD gains |
| `env.motor_time_constant_s` | `0.04` | Physical motor-speed lag, seconds |
| `env.rotor_thrust_limit_n` | `10.0` | Physical maximum per rotor, also limited by `thrust_to_weight` |
| `env.physics_hz`, `env.policy_hz` | `200`, `100` | Physics and policy clocks |

The rate gains are engineering starting values, not claimed paper gains or an identified hardware tuning.
Without actuator lag they correspond to roughly 50 ms roll/pitch and 125 ms yaw rate time constants.
The actual motor lag, filtering, allocation and leg dynamics change that response.
Keep the INDI filter cutoff below half the physics frequency.

## Train and play

On the laptop first activate `env_isaaclab` and export the usual CA paths:

```bash
cd /home/jordibelp/IsaacLab
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

PPO example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-cmd-vel --num_envs 128 --headless --logger tensorboard env.control_mode=cbrt_indi 'env.cbrt.rate_gain=[20.0,20.0,8.0]' '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Use `env.control_mode=cbrt` for the one-stage comparison.
For SAC-CE add `--agent rsl_rl_sac_cfg_entry_point`; for SAC-MSE use `rsl_rl_sac_mse_cfg_entry_point`.
On the 8 GB laptop use `--num_envs 32` and small replay/update settings for a smoke:

```bash
agent.algorithm.replay_buffer_size=4096 agent.algorithm.num_mini_batches=2 agent.algorithm.mini_batch_size=32 agent.local_redundancy.batch_size=8
```

Replace the task name for either goalkeeper task.
Use `play_borinot_cmd_vel.py` or `play_borinot_goalkeeper.py --task <task>` with the same `--agent`,
`env.control_mode`, nested controller settings and network/observation overrides as training.
Selecting a mode does not restore those other fields automatically from the checkpoint folder.
These commands do not deploy to hardware, sync code, or submit cluster jobs.

## Validation scope

Simulator-free checks cover the full nominal inertia equation, proportional rate error,
PD and motor-lag units, normalized endpoints, partial resets, matched filters against SciPy,
one versus two allocation calls, and bounded objectives against SciPy in float32 and float64.
Live checks cover Hydra overrides/aliases, all three tasks, both action mappings, command delay,
fresh per-physics feedback, actual PhysX gains/limits and automatic resets.
Short public train/reload runs check PPO, SAC-CE and SAC-MSE separately.
They are integration evidence, not a learned task-performance claim.

INDI measurements still have finite sampling delay and rely on a rotor/motor model.
This is not an instantaneous impact-cancellation guarantee, especially for the compliant leg's ground contacts.
Controller-only timing and the isolated applied-torque test are saved in the dated workspace evidence folder.


### Isolated default-gain check

A 1.6-second simulator check used four identical, non-randomized command-velocity clones,
neutral leg PD, nominal hover collective and zero desired body rates.
We added a constant body torque of `(0.04,-0.025,0.012)` N·m from 0.2 seconds onward.
Both modes kept the default 40 ms motor-speed lag and had zero resets.
The mean rate-vector norm over the final 0.4 seconds was 0.44058 rad/s for plain CBRT
and 0.0001761 rad/s with INDI. This probe includes leg reaction as well as the added torque.
It checks this starting tuning and disturbance mechanism, not general task performance or impact robustness.
The complete samples are retained in the task evidence folder.


### Controller-only timing

Eager PyTorch on the local RTX 5070 Laptop GPU, 128 solver iterations per stage:

| Batch | CBRT | CBRT+INDI |
| --- | ---: | ---: |
| 1 | 3.75 ms | 7.29 ms |
| 10,000 | 5.49 ms | 10.92 ms |

These short timings exclude physics and policy inference. They do not guarantee a 200 Hz wall-clock loop or hardware real-time operation. They also do not prove that 10,000 simulator environments plus a replay buffer fit the laptop.
