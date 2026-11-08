# Arm-aware model-wrench + INDI

Select `env.control_mode=model_wrench_indi` or
`env.control_mode=model_wrench_yawless_indi`. Both work in command tracking,
GoTo, goalkeeper, and simplified goalkeeper. PPO and SAC (CE or MSE) use the
same action contracts as their matching nominal model-wrench modes.

| Mode | Scalar actions, in order | Heading feedback |
|---|---|---|
| `model_wrench_indi` | world acceleration (3), world heading (1), body rates (3), joint targets (2) | Same as `model_wrench` |
| `model_wrench_yawless_indi` | world acceleration (3), body rates (3), joint targets (2) | No heading target; body-z rate control remains |

Optional angle pairs expand the same slots as in the nominal modes. The yawless
mode rejects `predict_yaw_as_sincos=True`. The six realized-rotor observations
are unchanged. The internal observer state is not appended to policy inputs.
Matching tensor sizes allow loading a nominal-mode policy, but its closed-loop
behavior changes; that is not a performance or checkpoint-equivalence claim.

## Method

Keep the complete nominal arm model, clipped software joint PD, desired thrust
direction, attitude/rate feedback and constrained inverse solve. Correct only
the requested rotor **torque** before bounded allocation. Collective thrust
remains model-based. The arm reaction is never added to reconstructed rotor
torque, and no hip-wrench sensor is used.

At physics tick `k`, use world base-CoM velocity and gyro-derived world angular
velocity increments over the completed interval `[k-1,k]`. Express those
accelerations in the saved interval-start body frame. Apply the **saved**
reduced torque row and bias, which used the **previous bounded joint effort**.
Subtract the saved rotor-only torque reconstructed from **realized speeds after
motor lag**. A changed action at tick `k` cannot enter this old-interval residual.

Rotate the complete residual into world axes. Filter it with one second-order
Butterworth filter. Rotate the result back into current body axes and subtract
it from the nominal torque request. Filtering the full residual retains the
configuration-dependent products; filtering in fixed axes avoids mixing body
axes from different times. The motor state advances exactly once per tick.

This is a causal finite-step implementation, not an exact instantaneous identity.
It freezes the dynamics at the interval start and uses interval-mean
accelerations. Finite-step integration, noise, contacts and changing parameters
can all appear in the residual. It cannot cancel an impact before observing it.

## Configuration and sensing

- `env.model_wrench_indi.filter_cutoff_hz=12.0`: shared residual-filter cutoff.
  It must be positive and below half `env.physics_hz`.
- Gains, SAC reference scales, tilt method and allocation settings stay in
  `env.wrench`. `env.indi` does **not** configure these hybrid modes.
- `env.model_wrench_use_prev_alpha=False` retains the original zero-virtual-alpha
  direction choice. `True` uses the previous requested angular acceleration,
  as in the nominal model-wrench family.
- Joint gains, effort limits, rotor limits and motor lag keep their existing
  task settings. Implicit joint PD is disabled; software PD runs at physics rate.
- Extra feedback input: measured/estimated **physical base-CoM world velocity**.
  Angular velocity is the existing gyro signal. The simulator reads
  `body_com_lin_vel_w[:, base_id]`, not link-origin velocity, hidden acceleration,
  policy acceleration or predicted acceleration. Enabling an acceleration
  observation for the policy is independent of this controller input.
- As in the nominal model-wrench plant, rotor spin inertia is not identified or
  simulated (`G2=0`). Static thrust-lever-arm and aerodynamic-drag torque remain.
  Changing `env.indi.rotor_spin_inertia_kg_m2` affects standalone `dsam_indi`, not these
  modes. A calibrated nonzero spin model needs matching plant and allocation work;
  it is not silently claimed here.

Physical resets clear only selected observer histories and zero the filter.
The first command is exactly nominal, with no stale acceleration difference.
GoTo goal-clock updates that do not reset the robot keep the observer history.

## Use

From the normal IsaacLab environment, for example:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-cmd-vel --num_envs=10000 --headless env.control_mode=model_wrench_indi
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto --num_envs=10000 --headless env.control_mode=model_wrench_yawless_indi env.model_wrench_indi.filter_cutoff_hz=12.0
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Goalkeeper-Direct-v0 --headless env.control_mode=model_wrench_indi
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-Goalkeeper-Simple-Direct-v0 --agent=rsl_rl_sac_mse_cfg_entry_point --headless env.control_mode=model_wrench_yawless_indi
```

For playback, use the same mode and angle-encoding overrides as training with
`play_borinot_cmd_vel.py`, `play_borinot_goto.py`, or
`play_borinot_goalkeeper.py`. Do not interchange the nine- and eight-action layouts.

## Diagnostics and sources

The controller exposes `nominal_wrench`, corrected `requested_wrench`,
`disturbance_b`, `raw_disturbance_w`, `measured_acceleration_b`,
`applied_wrench`, and `allocation_residual`. These are numeric buffers, not new
automatic W&B histories. `measured_acceleration_b` uses the **previous** body
frame; `disturbance_b` uses the **current** one. Requested wrench residuals use
the pre-lag bounded allocation. `applied_wrench` is post-lag. Nominal predicted
base/joint accelerations and diagnostic arm reactions remain **nominal**, not
predictions of the corrected, saturated or lagged response.

- `controllers/borinot_model_wrench_indi.py`: observer and both hybrid classes.
- `controllers/borinot_model_wrench{,_yawless}.py`: unchanged nominal laws plus
  one no-op torque hook for the hybrid subclasses.
- `borinot_goalkeeper/tests/test_model_wrench_indi.py`: full-dynamics forward
  oracle, rotating frames, SciPy filter, action pairs, causality and reset checks.
- `borinot_goalkeeper/tests/model_wrench_indi_smoke.py`: actual registered
  Hydra/PhysX integration and source-snapshot checks.
- [Illustrated derivation](../../../../../docs/borinot/2026-11-07-borinot-model-wrench-indi-illustrated.tex):
  self-contained native LaTeX, input/output diagrams, full equation-to-source map,
  timing, worked example, sensor conversions and glossary.

Local tests establish implementation and launch behavior, not improved learned
tracking, contact performance or hardware transfer.
