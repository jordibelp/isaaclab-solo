# Model-based arm dynamics in free flight

Select `env.control_mode=model_wrench`. This adds a controller; it does not
replace or change the measured `wrench` baseline.

## What is modelled

We include the rigid base, both moving links, and any fixed racket. We use the
actual joint anchor transforms, full own-CoM inertia tensors, gravity, base
rotation, joint motion, and the base–arm coupling. There is no ground, wall,
ball, self-contact, aerodynamic, or unmodelled-load prediction in this model.
It can run in the goalkeeper tasks, but their impacts are outside this model.

The runtime controller only reads base attitude, base angular velocity, joint
position and joint velocity. It first computes the same software arm PD torque
as `wrench`, clips that torque to the actuator limit, and solves for the joint
acceleration. It does not read a hip force sensor, simulator mass matrix,
Jacobian, link acceleration, or joint acceleration. It does not differentiate
noisy encoder velocities or gyro rates.

Mass and geometry are fixed calibration inputs. The exporter reads nominal
spawned USD, including configured motor masses and payload overrides. For the
shape-inferred racket it uses cached pre-randomization mass/inertia and its
non-randomized shape CoM. It never fits the model to each randomized clone.
The existing task randomizers do not vary the racket CoM. A future tool-CoM
randomizer must preserve the nominal CoM separately before this export.
The exported JSON is calibration data, not an onboard dependency on PhysX.

## Controller calculation

Let `z = [a_C_body(3), alpha_body(3), ddq(2)]`. The first term is physical
acceleration of the **base CoM**, expressed in body axes. It is not the time
derivative of body-frame velocity. The model computes `M z + h = [F, tau, tau_j]`.

We eliminate `ddq` from the two joint rows using the **clipped PD torque**.
This gives a 6×6 effective base matrix `E` and bias `eta`. By default, a virtual
force `E_tt a_des + eta_t`, with virtual angular acceleration zero, generates the
attitude reference. Set `env.model_wrench_use_prev_alpha=True` to instead include
the previous physics tick's controller `alpha_desired` in that direction-only
force. Both choices avoid a same-tick force/attitude algebraic loop.
It is a direction-reference design, not an exact solution for all six desired
base accelerations at once.

After attitude feedback chooses `alpha_des`, we impose the real actuator
constraint `F = T e3`. We solve translation so that `e3 dot a_C_body` equals
`e3 dot a_des_body`, then solve the three torque rows. Lateral acceleration
follows the coupled mechanics until the attitude changes. This differs from
the old `wrench` law's force-magnitude shortcut, even though the action layout
is the same. The LaTeX derivation unpacks each step and includes a hand example.

The existing bounded 4×6 rotor allocator and first-order **rotor-speed** lag run
once per physics tick. Allocation saturation, motor lag, model error, passive
friction and joint stops prevent exact desired-acceleration tracking. There is
no INDI correction or extra arm-reaction force applied to the plant. PhysX
already supplies the arm's mechanical reaction.

## Interfaces and settings

Same action order as `wrench`: world acceleration XYZ, world heading, current-body
rate references XYZ, thigh and calf position targets. PPO uses physical units.
SAC uses the existing normalized `wrench` scales. All three optional sin/cos
angle encodings, hold-arm, command delay, reset paths and cmd-vel world-heading
symmetry remain available. There is no new observation state: six realized
rotor-thrust values are appended, as for `wrench`. With current cmd-vel defaults
there are 9 actions and 39 actor/critic inputs. GoTo adds one input (40).

Gains and solver settings are deliberately shared under `env.wrench.*`:
tilt gain 150 s^-2, yaw gain 3 s^-2, rate gains (20,20,8) s^-1,
allocation weights (1,10,10,1), and 128 allocation iterations. Joint gains,
effort limit and motor lag remain `env.kp`, `env.kd`, `env.effort_lim_sim`, and
`env.motor_time_constant_s`. Defaults remain unchanged for every old mode.

`env.model_wrench_use_prev_alpha=False` is the default and preserves the original
zero-virtual-alpha direction reference. It is read only by `model_wrench` and
`model_wrench_yawless` and their `_indi` variants; it has no effect on measured `wrench`, standalone `indi`, CBRT, or
direct control modes.

The scalar simulator armature is included only as two diagonal generalized
joint inertias. It is not a full physical motor-rotor/transmission model. Use
identified hardware transmission dynamics when porting, not an unexplained
copy of the simulator's armature value.

Unlike the measured mode, this controller can use fresh state on the first
tick after reset. There is no incoming-wrench sample to wait for. Reset still
initializes rotor speed to the normal nominal hover speed.

Supported tasks: `Borinot-cmd-vel`, `Borinot-goto`,
`Borinot-Goalkeeper-Direct-v0`, and `Borinot-Goalkeeper-Simple-Direct-v0`.
The older separate `Borinot-Direct-v0` environment does not expose the measured
`wrench` mode and is not changed here.

For an initial contact-free, nominal-model experiment, add:

```text
env.control_mode=model_wrench env.include_events_randomization=False env.forces_applied_to_base_curriculum=[0.0] env.enabled_self_collisions=False
```

These flags disable sampled parameter changes, scheduled base pushes and
self contacts. They do not remove the floor: keep flight clear of it. For a
robustness experiment, enable those disturbances explicitly. Their unknown
parameters/forces remain model error; the controller does not receive their
hidden values. Old checkpoints have compatible tensor shapes but were trained
with another control law. Do not treat switching their mode as a validated
policy transfer. Train or evaluate the new mode deliberately.

## Hardware boundary

Use a calibrated body-frame gyro, a fused base attitude (IMU and/or OptiTrack),
and both joints' encoder positions/rates. IMU and OptiTrack alone do not tell us
the joint state. The dynamics solve does not need base acceleration or absolute
position. The outer learned policy still needs its own configured velocity,
height, command and rotation inputs.

Convert IMU, marker and encoder frames/signs/zero offsets explicitly. Match
timestamps before evaluating the model. Replace simulation calibration with
measured mass, CoM, inertia, torque-response and thrust-map data. The accompanying
report derives sensor lever-arm corrections and gives an ordered porting path.
No ROS controller or flight firmware is changed by this implementation.

## Verification entry points

- `controllers/borinot_model_wrench.py`: pure batched PyTorch model/controller.
- `direct/borinot/model_description.py`: fixed nominal calibration export.
- `borinot_goalkeeper/tests/test_model_wrench.py`: independent Pinocchio mass
  and inverse-dynamics oracles, full eight-row balances, saturation, angle
  encodings, world-frame equivariance and partial-reset checks.
- `borinot_goalkeeper/tests/model_wrench_smoke.py`: real Hydra/PhysX checks for
  spaces, model/plant FK and properties, every-substep execution, motor lag,
  actuator bounds and automatic resets. `--forward-check` adds an independent
  one-step physical-response fixture at the dt selected before simulator startup.

The delivery folder also contains a standalone package, nominal JSON calibration,
an executable numerical example and verification receipts. This is integration
and mechanics evidence, not a demonstration of trained flight quality or hardware
transfer. No cluster or hardware experiment is part of this change.

## Add residual feedback

Select `model_wrench_indi` or `model_wrench_yawless_indi` to retain this nominal
model and correct its requested rotor torque from completed motion. See
[Model-wrench + INDI](MODEL_WRENCH_INDI_CONTROL.md) for sensing, timing, filtering
and launch settings. The original nominal modes remain available.
