# Borinot: physics-rate INDI and joint PID

Date: 2026-10-22. Applies to both goalkeeper tasks.

## 1. What is implemented

Set `env.control_mode=indi` to train or play a policy with **nine outputs**:

1. Three desired linear accelerations in world axes.
2. Three desired body rates, used as feedforward.
3. One desired heading.
4. Two desired joint angles.

The environment turns these references into six rotor-speed commands and two joint torques.
It does this again **on every physics step**, using the newest robot state.
The controller is part of the training environment. It is not a playback-only wrapper.

PPO, SAC with two-hot CE critics, and SAC with MSE critics all support this interface.
The old direct-thrust/torque, direct-thrust/PD and collective/rate-PI modes remain available.
Their default selector, action scaling, motor dynamics and observation sizes are unchanged.
Nothing is deployed to hardware or copied to the cluster by this implementation.

The design follows the architecture in [Global End-Effector Pose Control of an Underactuated Aerial Manipulator via Reinforcement Learning](https://arxiv.org/html/2512.21085v2).
For its low-level details, that paper refers to [A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile Flight](https://arxiv.org/html/2109.01365).
We adapt the architecture to Borinot, rather than claim an exact reproduction of the DSAM vehicle.

### The important design choices

| Question | Choice here | Reason |
| --- | --- | --- |
| Four or six rotors? | Borinot's six actual mixer columns | Its geometry and spin order differ from the source quadrotor. |
| Angular-acceleration feedforward `alpha_r`? | Zero | The policy supplies no jerk or snap trajectory. Differentiating held actions would create impulses. |
| Joint controller? | Explicit software PID; default `ki=0.5` | It runs at a known physics clock. Set `env.indi.joint_ki=0.0` for PD. |
| Motor lag? | First-order **speed** lag, 40 ms by default | The INDI feedback variable is rotor speed. Other modes retain their existing thrust lag. |
| Rotor spin inertia `Ip`? | Zero by default; nonzero supported | No calibrated Borinot value was found. We do not borrow another vehicle's value. |
| Inertia used by INDI? | Nominal USD base-body inertia | This is a fixed controller model, not privileged knowledge of each randomized clone. |
| Mass used for collective thrust? | Nominal mass of the whole robot | The rotors support the base, arm and racket. |
| Allocation priorities? | Capacity-normalized error, weights `(1,10,10,1)` | This makes force and torque errors comparable and gives tilt higher weight. |
| Policy/physics rates? | Keep the task's 100/200 Hz defaults | Selecting a control mode does not silently retime an existing experiment. Set 150/300 Hz explicitly to match the paper's clocks. |

The attitude gains and 12 Hz filters start from the low-level reference paper.
The joint integral gain, action scales and normalized allocation weights are our engineering choices.
They are not reported DSAM gains or a tuned Borinot flight controller.

## 2. Run it

Use the canonical checkout. On the laptop, first activate the simulator environment:

```bash
cd /home/jordibelp/IsaacLab
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### PPO

This local command uses the paper-style clocks and a modest environment count:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Simple-Direct-v0 --num_envs 128 --headless --logger tensorboard env.control_mode=indi env.physics_hz=300 env.policy_hz=150 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Replace the task with `Borinot-Goalkeeper-Direct-v0` for the original reward.
This does not select the paper's reward, observations, robot, or complete training recipe.
All existing goalkeeper reward, randomization, ball and curriculum settings still apply.

For a training machine with enough memory, use your usual environment count and logger.
The controller has also been timed separately over 10,000 states on the local GPU.
That does not prove that a full 10,000-environment simulator plus replay buffer fits the laptop.
On IRICluster, activate its `isaaclab` environment instead of the laptop's `env_isaaclab`.
The cluster needs this source revision before these commands can work there.
These commands do not submit a Slurm job.

### SAC

Use the same INDI switches with either SAC entry point. This is a small local smoke-sized example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task Borinot-Goalkeeper-Simple-Direct-v0 --agent rsl_rl_sac_cfg_entry_point --num_envs 32 --headless --logger tensorboard env.control_mode=indi env.physics_hz=300 env.policy_hz=150 agent.algorithm.replay_buffer_size=4096 agent.algorithm.num_mini_batches=2 agent.algorithm.mini_batch_size=32 agent.local_redundancy.batch_size=8 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

Use `--agent rsl_rl_sac_mse_cfg_entry_point` for MSE critics.
The SAC actor now has nine bounded outputs. All nine action ranges are exactly `[-1,1]`.
Its replay stores these normalized references, not individual rotor thrusts or joint torques.
The old eight-output SAC mapping is unchanged in the direct modes.
The existing PI mode remains unsupported by these SAC agents.

### Playback

```bash
./isaaclab.sh -p source/scripts/rsl_rl/play_borinot_goalkeeper.py --task Borinot-Goalkeeper-Simple-Direct-v0 --checkpoint /absolute/path/to/model.pt env.control_mode=indi env.physics_hz=300 env.policy_hz=150
```

For SAC, also pass the same `--agent` used for training.
Restore the training environment/controller overrides and the network architecture when playing a checkpoint.
The player does not automatically import every setting from the training folder.
In particular, restore `env.indi.*`, `kp`, `kd`, motor lag, physical limits, rates and actor-observation flags.

**Start a fresh policy for this mode.** An eight- or six-action checkpoint cannot be used as a nine-action INDI checkpoint.
Matching an action count alone would not establish matching action meanings either.
Keep each run's saved `params/env.yaml`, agent config and source revision with its checkpoint.

## 3. What the nine actions mean

Write the normalized policy output as `u[0:9]`.
The environment clips every component to `[-1,1]`, for both PPO and SAC.
This bound does not use the old `action_clamp` setting.
The observation stores the clipped action that the controller actually receives.

| Index | Reference | Default conversion | Frame / units |
| --- | --- | --- | --- |
| 0 | Forward world acceleration | `8*u[0]` | World X, m/s² |
| 1 | Sideways world acceleration | `8*u[1]` | World Y, m/s² |
| 2 | Vertical world acceleration | `8*u[2]` | World Z, m/s²; gravity excluded |
| 3 | Roll-rate feedforward | `2*u[3]` | Current body X, rad/s |
| 4 | Pitch-rate feedforward | `2*u[4]` | Current body Y, rad/s |
| 5 | Yaw-rate feedforward | `1.5*u[5]` | Current body Z, rad/s |
| 6 | Heading | `home_yaw + pi*u[6]` | World heading, radians |
| 7 | First joint angle | `(pi/2)*u[7]` | `joint_names[0]`, radians |
| 8 | Second joint angle | `(pi/2)*u[8]` | `joint_names[1]`, radians |

World X does not mean the direction in which the base currently points.
Heading is an absolute heading centered on the configured reset heading, **not a yaw increment**.
For the default reset quaternion, `home_yaw=-90 degrees`.
Zero action therefore requests zero world acceleration, the home heading, zero rate feedforward and neutral joints.
It does not request zero rotor speed.

The DSAM paper uses joint outputs in `[-2,2]`, then scales them by `pi/4`.
Our normalized `[-1,1]` representation gives the same `[-pi/2,pi/2]` joint endpoints.
It is a different raw normalization, not a different physical target range.
The acceleration and rate scales are configurable because the DSAM paper does not specify all action bounds.

INDI joint targets do not use the direct-mode `joint_target_scale` or SAC's direct-mode soft limits/offsets.
That legacy scale still normalizes joint-position observations as before.
INDI writes the configured physical joint stops to PhysX and requires its targets to fit inside them.

## 4. One policy action, several physics steps

At 150 Hz, the policy produces a new reference every 6.67 ms.
At 300 Hz, the controller and physics run every 3.33 ms.
We hold the reference for two physics steps, but we do **not** hold the actuator effort for those two steps.

On each physics step we:

1. Read the current orientation, body rates, joint angles and joint velocities.
2. Convert the held acceleration/heading reference to thrust and desired attitude.
3. Calculate the desired angular acceleration.
4. Project the nominal thrust/torque request into the six rotor bounds.
5. Update the matched body-rate and rotor-speed filters.
6. Estimate angular acceleration and apply the INDI correction.
7. Solve the bounded rotor-speed request.
8. Advance the simulated rotor-speed lag and apply the resulting base wrench.
9. Recalculate the joint PID torque and send that effort to PhysX.

This happens in `_apply_action()`, inside IsaacLab's physics-decimation loop.
`_pre_physics_step()` accepts one policy action; it does not run the low-level controller only once.
Training and the public player call the same implementation.

The task still requires a positive integer ratio `physics_hz/policy_hz`.
Rewards, shot clocks, episode lengths and the existing command-rate penalty retain their policy-clock semantics.
Changing the policy frequency is a real experiment change: for example, a fixed number of curriculum steps then represents a different physical duration.

## 5. From linear acceleration to thrust direction

Define the world gravity vector as `g=(0,0,-9.81)` m/s².
For a requested world acceleration `a_r`, compute:

```text
specific_force = a_r - g
T_d = m * norm(specific_force)
z_desired = specific_force / norm(specific_force)
```

`m` is the nominal total robot mass in kg.
`T_d` is collective thrust in N. `z_desired` is the desired body-up direction in world axes.
`norm` is the Euclidean vector length.
With zero acceleration, the request is `T_d=m*9.81`, directed up.

We combine this direction with the heading to construct a right-handed desired rotation matrix.
Its columns are the desired body X, Y and Z axes expressed in world coordinates.
If thrust direction and heading are exactly parallel, we use a perpendicular heading direction to avoid division by zero.
At exactly zero requested specific force, we request zero thrust and use world up as a finite attitude fallback.

There is **no low-level position controller** and **no translational INDI** here.
The policy must correct position drift, mass errors, contact forces and arm-induced translation using its observations.
Zero action is a hover *request*, not a guarantee that a freely moving arm/base will remain in place.

## 6. Tangent-space attitude control (updated 2026-10-23)

We compare the current and desired orientation in **current body axes**.
For Isaac's active, scalar-first quaternion convention, the corresponding error quaternion is:

```text
q_error = inverse(q_current) * q_desired
```

The multiplication order matters. A world-axis error fed into a body-axis torque law would be wrong at nonzero heading.
The code uses the equivalent rotation-matrix construction and checks it against an independent quaternion calculation.

We now follow Jordi's full-Log z-projection sketch. `Log` returns a three-component
rotation vector whose length is the **full rotation angle in radians**.
It is not the half-angle quaternion logarithm. The calculation is:

```text
tau       = Log(q_error)
tau_z     = (0, 0, tau[2])
q_z       = Exp(tau_z)
q_xy      = q_error * inverse(q_z)
tau_xy    = Log(q_xy)

alpha_d = tilt_gain * tau_xy
        + yaw_gain * tau_z
        + rate_gain * (Omega_r - Omega)
```

Here `q_z` is a unit quaternion about body z. `q_xy` is the remaining quaternion,
with the product order shown above. Both `tau_z` and `tau_xy` are three-component
body-axis vectors in radians. The subscript `xy` is a label from the sketch:
**the z component of `tau_xy` is generally nonzero** and is retained in feedback.
Projecting the full log is not the old geometric thrust-axis tilt/yaw split.
In particular, `tau[2]` is not in general Euler yaw or the old residual yaw angle.

For a unit quaternion `(w,v)`, we use `Log(q) = 2*atan2(norm(v),w)*v/norm(v)`.
We first choose the shortest representative (`w >= 0`), and use the finite
identity limit `Log(q) = 2*v`. For a rotation vector `t` with length `a`,
`Exp(t) = (cos(a/2), sin(a/2)*t/a)`, with vector limit `t/2` at zero.
These conventions match [A micro Lie theory for state estimation in robotics,
Appendix B-A](https://arxiv.org/html/1812.01537v8#A2.SS1).
The implementation uses `atan2` and a stable largest-component matrix-to-quaternion
conversion. It remains finite at identity and at 180 degrees.
At exactly 180 degrees the shortest axis sign is not unique; we choose the sign
that makes its largest-magnitude component positive. A principal-log branch
discontinuity at 180 degrees is unavoidable; this is not a global stability claim.

`alpha_d` is desired body angular acceleration, in rad/s².
`Omega_r` is the policy's body-rate feedforward, and `Omega` is the measured body rate, both in rad/s.
`rate_gain` acts element by element and has units 1/s.
The attitude gains now have units 1/s², multiplying rotation vectors in radians.

Defaults are `tilt_gain=150`, `yaw_gain=3` and `rate_gain=(20,20,8)`.
Their numeric values are unchanged, but **the near-zero attitude feedback is twice
as strong as before**, because the old quaternion-vector magnitude was `sin(angle/2)`.
For pure 120-degree tilt, for example, the old raw request was about 129.90 rad/s²;
the new one is 314.16 rad/s². Physical allocation and motor bounds still apply.
To preserve the previous local attitude gains while increasing large-angle
feedback, explicitly set `env.indi.tilt_gain=75.0 env.indi.yaw_gain=1.5`.
Rate feedback, filters, allocation, motor dynamics, joint PID, action dimensions
and observations are unchanged. Existing INDI checkpoints still load, but now
run against a different controller. Exact historical replay needs the old revision.
The shared controller affects both goalkeeper tasks and `Borinot-cmd-vel` in INDI mode.
Other control modes are unchanged. No training-performance benefit has been established.

### Why `alpha_r=0` is deliberate

The separate `alpha_r` in the local notes is angular-acceleration feedforward.
It is not the policy's linear acceleration and not the controller output `alpha_d`.
The policy supplies body rates, but no smooth trajectory derivatives from which to calculate `alpha_r`.
We therefore omit this term. We do not differentiate discontinuous policy targets.
The proportional body-rate term still uses the policy's full rate feedforward.

## 7. First allocation: make the nominal request feasible

For Borinot, `G1` has four rows and six columns:

```text
row 0: 1, for every rotor
row 1: rotor_y - nominal_base_com_y
row 2: -(rotor_x - nominal_base_com_x)
row 3: rotor_spin * km/kf
```

Multiplying `G1` by six rotor thrusts gives `[collective, roll torque, pitch torque, yaw torque]`.
The first output has units N. The other three have units N·m.
The lever-arm and torque/thrust rows have units m.
We use the configured Eagle/PX4 motor order, positions and yaw-reaction signs; we do not substitute a quadrotor matrix.

The nominal torque request is:

```text
tau_nominal = I * alpha_d + cross(Omega, I * Omega)
```

`I` is the fixed nominal base inertia matrix in body axes, in kg·m².
`cross` is the 3D vector cross product. `tau_nominal` is in N·m.
The arm's inertia changes and reaction torques are not explicitly modeled by this base-only approximation.
INDI handles their rotational effect through feedback, subject to bandwidth and actuator limits.

We solve a bounded least-squares allocation instead of independently clipping an inverse solution.
For each wrench channel, define `capacity = rotor_cap * sum(abs(G1_row))`.
This is a normalization scale, not a claim that all four channel maxima can be reached together.
We minimize:

```text
sum_j weight[j] * ((requested_wrench[j] - (G1 * rotor_thrust)[j]) / capacity[j])^2
subject to: 0 <= each rotor_thrust <= rotor_cap
```

Here `j` runs over collective, roll, pitch and yaw. Weights are dimensionless.
This is a convex quadratic program, or QP: a quadratic cost with simple bounds.
Default weights `(1,10,10,1)` favor roll/pitch in **normalized** error units.
They are not a copy of the reference paper's unnormalized weight matrix and are not a strict lexicographic priority.

The solver uses 96 fixed accelerated projected-gradient iterations, fully batched in PyTorch.
Projection means clipping the *optimization iterate* to its allowed box after a gradient step.
That is different from clipping a final inverse mixer command without optimizing its wrench error.
The solve is approximate and has a fixed cost; it does not promise exact equality under saturation.
Tests compare its objective against SciPy's independent bounded least-squares solver.

The achieved nominal wrench gives `T_feasible` and `alpha_feasible`:

```text
alpha_feasible = inverse(I) * (achieved_nominal_torque - cross(Omega, I * Omega))
```

"Feasible" here means feasible for this nominal static mixer and its limits.
It does not guarantee the same acceleration in the real coupled robot.

The rotor cap is the same effective cap as the task's other bounded modes:
the smaller of `rotor_thrust_limit_n` and `thrust_to_weight * nominal_hover_per_rotor`.
INDI rejects `remove_thrust_clipping=True`; removing all upper bounds contradicts this controller's allocation contract.

## 8. Matched filtering and the actual INDI correction

INDI needs the torque being produced and the angular acceleration being observed at comparable times.
We apply the same second-order, 12 Hz Butterworth filter to body rates and simulated measured rotor speeds.
The digital coefficients are recalculated from the actual physics timestep.

At physics tick `k`:

```text
alpha_measured = (Omega_filtered[k] - Omega_filtered[k-1]) / dt
tau_filtered = G1_torque * (kf * omega_filtered[k]^2)
             + G2_torque * (omega_filtered[k] - omega_filtered[k-1]) / dt
tau_desired = tau_filtered + I * (alpha_feasible - alpha_measured)
```

`dt` is the physics timestep in seconds. Square acts separately on each rotor speed.
`omega_filtered` has six rotor speeds in rad/s; `Omega_filtered` has three body rates in rad/s.
The subscript `torque` selects the final three rows of a matrix.
`tau_filtered` and `tau_desired` are 3-vectors in body axes, in N·m.

The last line is INDI. If measured angular acceleration is too low, it asks for more torque.
If an arm reaction already accelerates the base, that acceleration appears in the feedback, without an analytical arm model.
The nominal gyroscopic term cancels in this incremental correction; we do not add it a second time.

This is not perfect disturbance cancellation.
Fast impacts, filtering delay, incorrect gains and motor saturation can limit it.
The policy also needs to coordinate its acceleration, heading and rate references; inconsistent references can oppose each other.

## 9. Solving the local notes' rotor-speed equation

The local notes call the final inversion equation (16).
For Borinot, it has **four wrench equations and six unknown rotor speeds**:

```text
[T_feasible; tau_desired] = G1 * (kf * omega_command^2)
                         + G2 * (omega_command - previous_omega_command) / dt
```

`G2` is a 4×6 matrix. Only its yaw row is nonzero:
`G2[3,i] = rotor_spin[i] * Ip`.
`Ip` is one rotor's spin-axis inertia, in kg·m².
The signs follow the task's **body reaction torque** convention, not a separately guessed spin convention.
There is no extra factor `kf` on the `G2` term: inertia times angular acceleration is already a torque.

### Default: `Ip=0`

The final inversion becomes another bounded allocation in thrust coordinates.
We then calculate each nonnegative speed with `sqrt(thrust/kf)`.
The six-rotor problem is redundant, so a square inverse from the four-rotor derivation would be inappropriate.
This is still INDI: its defining acceleration feedback and matched torque feedback are active.
Zero `G2` removes rotor-acceleration reaction torque, not the INDI correction itself.

This is the default because the existing Borinot model has no identified rotor spin inertia.
The local thesis and Eagle source inspection did not provide a calibrated `Ip`.
We do not pretend that `km` is rotor inertia: it is an aerodynamic drag-torque coefficient with different units.

### Optional: nonzero `Ip`

Set `env.indi.rotor_spin_inertia_kg_m2` to a measured value when available.
The solver then uses bounded, damped Gauss–Newton in normalized rotor-speed coordinates.
Each of six default iterations solves a **batched 4×4 linear system**, even though there are six rotor outputs.
It uses only the free rotor directions at a bound and tries several bounded step lengths.
It accepts only steps that reduce the weighted residual.
There is no Python loop over environments, SciPy call, CPU optimizer or differentiable-QP package in the training path.

This is a local nonlinear least-squares solve, not a guarantee of a global optimum or zero residual.
A requested wrench can be physically impossible.
The controller retains the full four-channel residual so we can distinguish saturation from successful inversion.
Nonzero-inertia tests also generate known feasible requests and check the recovered wrench in float32 and float64.

For consistency, nonzero `Ip` also adds the matching transient yaw reaction torque to the simulated base wrench.
Leaving it enabled only in the controller would create a model mismatch of our own making.
We do not model rotor precession/gyroscopic coupling (`G3` in the low-level reference model).

**Yes: this numerical solve runs inside IsaacLab during training.**
It is ordinary batched GPU tensor code executed between physics steps.
The feasibility question is about solver cost and model quality, not a restriction that Isaac Sim forbids such calculations.

## 10. Motor dynamics and joint torque

### Rotors

```text
motor_alpha = 1 - exp(-dt / motor_time_constant_s)
omega_next = omega_now + motor_alpha * (omega_command - omega_now)
actual_thrust = kf * omega_next^2
```

`motor_alpha` is dimensionless. `exp` is the exponential function.
If the time constant is zero, speed commands take effect immediately.
The simulated measured speed is this motor-model state, not the unlagged speed command.
INDI filters the state available **before** advancing the next motor step.
There is no future-physics measurement in its feedback.

The configured 40 ms is a model choice. A reported physical motor time constant does not by itself establish whether a first-order *speed* or *thrust* model fits best.
The direct modes continue filtering thrust with their old implementation; INDI deliberately filters speed.

### Joints

Let `error = joint_target - measured_joint_angle`.
The controller uses measured velocity for damping, so a reference step does not cause a derivative kick:

```text
integral_candidate = clip(integral_old + error * dt, -integral_limit, +integral_limit)
joint_effort = kp * error - kd * measured_joint_velocity + ki * integral
```

Angles are in rad, velocities in rad/s, integral in rad·s and effort in N·m.
`kp`, `kd` and `ki` have units N·m/rad, N·m·s/rad and N·m/(rad·s), respectively.
The torque is clipped to `[-effort_lim_sim,+effort_lim_sim]`.
We do not accept an integral update if it would push an already saturated torque farther into saturation.
That is conditional anti-windup. The integral has a separate magnitude limit as well.

The implicit actuator's stiffness and damping are both zero in this mode.
Otherwise PhysX would add another PD controller on top of our explicit PID effort.
Effort limits, velocity limits, armature, friction and startup randomization remain active.
`hold_arm=True` requests neutral joint targets through this PID; it does not weld the arm in place.

## 11. Observations, rewards and resets

The goalkeeper observation is **not** replaced with the paper's pose-regulation observation.
The existing robot and ball inputs remain, followed by nine current actions.
INDI also exposes six actual rotor thrusts divided by nominal hover thrust and two normalized joint integrals.
The latter are divided by `joint_integral_limit`, not by the torque limit.

With the default actor flags:

| Task | Policy actions | Actor inputs | Critic inputs |
| --- | ---: | ---: | ---: |
| Original goalkeeper, INDI | 9 | 53 | 65 |
| Simple goalkeeper, INDI | 9 | 53 | 63 |
| Either task, direct mode | 8 | 44 | Original 56 / simple 54 |
| Either task, existing PI mode | 6 | 51 | Original 63 / simple 61 |

Other actor flags change the actor count as before.
The filter delay states are internal controller memory; they are not all exposed to the actor or critic.
Thus this is not a claim that the observation makes the filtered closed-loop system exactly Markov.
We use true simulator orientation and rates, not an implemented IMU/EKF/noise/latency model.

The current energy and action-rate rewards still measure physical actuator use, not these nine abstract references.
The command-rate reward compares post-bound rotor thrust commands and joint efforts at the policy clock.
It does not add the paper's separate abstract-action smoothing rewards.
This avoids silently changing the reward while adding a control mode.
The power model still uses aerodynamic rotor power and the task's joint work/copper-loss terms.
It does not become a complete battery or rotor kinetic-energy model when `Ip` is nonzero.

A full robot reset restores hover motor speed, the previous hover speed command, zero joint integral and zero previous angular acceleration.
Both filters are initialized at the reset state, not at a fictitious zero rotor speed.
The controller therefore does not interpret reset itself as a huge acceleration impulse.
The task also clears joint efforts and its physical-command histories.
Partial resets affect only the selected environments. Automatic episode resets use the same path.
A ball/shot-only reset preserves robot and controller state.

## 12. Configuration reference

All controller settings below are under `env.indi` and are used only in INDI mode.

| Setting | Default | Meaning |
| --- | --- | --- |
| `acceleration_scale` | `(8,8,8)` | World acceleration per unit action, m/s² |
| `heading_range_degs` | `180` | Heading half-range around home yaw |
| `joint_target_limit_degs` | `(90,90)` | Symmetric joint target half-ranges |
| `tilt_gain` | `150` | Residual full-angle log-vector gain, 1/s² |
| `yaw_gain` | `3` | Full-log z-projection gain, 1/s² |
| `rate_gain` | `(20,20,8)` | Body-rate feedback gains, 1/s |
| `filter_cutoff_hz` | `12` | Common second-order Butterworth cutoff; must be below physics Nyquist |
| `allocation_weights` | `(1,10,10,1)` | Squared normalized wrench-error weights |
| `allocation_iterations` | `96` | Fixed projected-gradient steps per static allocation |
| `nonlinear_iterations` | `6` | Gauss–Newton steps when rotor inertia is nonzero |
| `rotor_spin_inertia_kg_m2` | `0` | Per-rotor spin-axis inertia; zero disables `G2` consistently |
| `inertia_scale` | `1` | Multiply the nominal base inertia used by the controller |
| `joint_ki` | `0.5` | Integral torque gain; set zero for PD |
| `joint_integral_limit` | `0.3` | Integral bound, rad·s |

Existing task fields still control `physics_hz`, `policy_hz`, `kp`, `kd`, `body_rate_scale`, `motor_time_constant_s`, `effort_lim_sim`, `velocity_lim_sim`, rotor positions/signs, `kf`, `km_over_kf`, and rotor limits.
Use, for example, `env.indi.filter_cutoff_hz=10.0` or `'env.indi.acceleration_scale=[6.0,6.0,8.0]'`.
Derived spaces, actuator gains, validation and filter construction happen after Hydra overrides.

## 13. Diagnostics and verification

The environment keeps the following tensors on its controller object, `env.unwrapped._indi`:

| Tensor | Meaning / units |
| --- | --- |
| `alpha_desired` | Raw attitude-law body angular acceleration, rad/s² |
| `alpha_feasible` | Nominal allocation-derived body angular acceleration, rad/s² |
| `angular_acceleration` | Finite difference of filtered measured body rates, rad/s² |
| `filtered_rates` | Three filtered body rates, rad/s |
| `filtered_speeds` | Six filtered rotor speeds, rad/s |
| `nominal_wrench` | First allocation's achieved `[N,N·m,N·m,N·m]` |
| `desired_wrench` | Final INDI target in the same channel order |
| `allocation_residual` | Predicted final command wrench minus final target; not measured tracking error |
| `command_speed` | Six unlagged bounded speed commands, rad/s |
| `rotor_speed` | Six simulated actual motor speeds, rad/s |
| `transient_torque` | Simulated `G2` reaction torque, N·m |
| `joint_integral` | Two PID integrals, rad·s |

These tensors are available for debugging. This change does not silently add a large W&B telemetry stream.

The controller's runtime module is `indi.py` in this directory.
Both tasks use the original task's shared `_apply_action()`.
The SAC action contract lives in `agents/rsl_rl_sac_cfg.py`.
The numerical, live-physics and controller-scale benchmark drivers are in `tests/test_indi.py`, `tests/indi_smoke.py`, and `tests/indi_benchmark.py`.

Run the CPU suites separately to avoid importing simulator task packages before AppLauncher:

```bash
PYTHONPATH=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests:source/rsl_rl_sac_vendor python -m pytest -q --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests
python -m pytest -q --confcutdir=source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper_simple/tests
```

The live smoke accepts either task and real Hydra overrides:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/indi_smoke.py --headless --task Borinot-Goalkeeper-Simple-Direct-v0 --output /absolute/scratch/path/indi-smoke.json env.control_mode=indi env.physics_hz=300 env.policy_hz=150 '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The standalone performance check does not start Isaac Sim:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/borinot_goalkeeper/tests/indi_benchmark.py --num-envs 10000 --device cuda --output /absolute/scratch/path/indi-benchmark.json
```

Local verification evidence is kept under:

```text
/home/jordibelp/.openclaw/workspace/artifacts/2026-10-22-borinot-indi
```

The CPU checks include independent quaternion/frame tests, SciPy filter and bounded-QP comparisons, nonzero-`G2` inversion, speed-lag checks, conditional anti-windup, partial resets and a small disturbance-rejection plant test.
The live checks read actual PhysX gains/limits and exercise both task registrations and automatic reset paths.
Training/playback smokes are **integration checks**, not evidence that a trained INDI goalkeeper outperforms direct control.
The saved verification summary records completed runs, counts and performance measurements.

### Completed local checks

| Check | Observed result |
| --- | --- |
| Original/shared CPU suite | 385 passed, including 35 INDI tests |
| Simple-task CPU suite | 53 passed |
| Live PhysX | Six fixtures × 78 checks passed: both tasks at 200/100 Hz, 300/150 Hz, and 300/150 Hz with nonzero `Ip` |
| Training entry point | PPO, SAC-CE and SAC-MSE on each task: four small iterations, 512 transitions per run |
| Public checkpoint playback | All six checkpoints: 150 finite steps each, including repeated automatic timeouts |
| 10,000-state controller timing | Median 10.02 ms/tick with `Ip=0`; 12.50 ms/tick with test `Ip=1e-6` kg·m² |

The performance checks used eager PyTorch 2.7 and the local RTX 5070 Laptop GPU.
Each timing includes the complete low-level controller for all 10,000 states, with 5 warm-up ticks and 20 measured ticks.
It excludes physics, policy inference, optimizers and logging.
The nonzero `Ip` is a test value, not a Borinot calibration.
The live fixtures use four environments; the training smokes use 16 and smaller `[64,32]` networks with compilation disabled.
They exercise the real launch paths, but are not full-sized training runs or a hardware/GUI demonstration.

The local derivation notes used for this implementation are:

```text
/home/jordibelp/Desktop/emmaDocs/2026-10-21-dsam-controller-indi-notes.tex
/home/jordibelp/Documents/PepTesis.pdf
/home/jordibelp/eagle_ros2
```

The thesis supports the approximate 40 ms motor time-constant scale, but does not establish a calibrated speed-state model or rotor spin inertia.
The implementation reads the actual task's nominal USD base inertia; it does not copy a different robot's inertia from these references.

## 14. What still needs an experiment

1. Train matched direct-control and INDI policies with the same task difficulty and interaction budget.
2. Check the final wrench residuals during difficult contacts and saturation, not just average rewards.
3. Tune attitude/rate gains, joint gains and filter bandwidth for Borinot if learning exposes oscillation or slow response.
4. Identify motor-speed dynamics and rotor spin inertia before interpreting a nonzero-`G2` setting as a hardware model.
5. Add and validate the real estimator, rotor-speed measurement path, actuator interface and delay model before hardware transfer.

The implementation makes this controller available in training now.
It does not turn an untuned policy/controller pair into a validated hardware controller.
