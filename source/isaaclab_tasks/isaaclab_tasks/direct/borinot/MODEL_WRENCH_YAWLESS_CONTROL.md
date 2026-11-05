# Yaw-free model-wrench control

Select `env.control_mode=model_wrench_yawless`. This adds a separate controller.
It does not change the `model_wrench`, measured `wrench`, INDI, or CBRT laws.

The original virtual-force direction, predictive arm model, clipped joint PD,
current-thrust-axis inverse dynamics, bounded allocation, and rotor-speed lag
remain in use. Only the attitude reference and policy layout change.

## What “yawless” means

There is **no yaw-angle target** and no yaw-angle restoring gain. We still use
three body-rate references. The policy can request body-z spin, or set that
reference to zero to damp spin. We do not set yaw torque to zero: the inverse
dynamics can require it to realize a rate or cancel arm/gyroscopic coupling.

This is not Euler-yaw locking. A shortest finite tilt can change Euler yaw.
Body-z angular velocity also differs from world-z angular velocity under tilt.
The command task's world-z rate command and reward are unchanged.

## Scalar action order

| Indices | PPO meaning | SAC mapping |
| --- | --- | --- |
| 0–2 | World base-CoM acceleration, m/s² | Normalized sample × `wrench.acceleration_scale` |
| 3–5 | Current-body rate reference, rad/s | Normalized sample × `body_rate_scale` |
| 6–7 | Thigh/calf position targets, rad | Normalized sample × `wrench.joint_target_limit_degs`, converted to radians |

There are eight outputs instead of nine. PPO references are unbounded physical
values; physical actuator bounds remain downstream. SAC scalar inputs are
clipped to [-1, 1]. Optional joint `(cos, sin)` pairs expand slots 6 and 7 in
place and decode directly to radians. Each enabled pair adds one output.
`hold_arm=True` retains those slots but requests the existing zero-angle hold.

`predict_yaw_as_sincos=True` is rejected because there is no yaw output.
Old nine-output checkpoints are not compatible with this mode. Previous-action
observations change size too. Use a new run, and retain the old mode for old
checkpoints.

## Geometry and feedback

`R` maps current body coordinates into world coordinates. Let `e3=(0,0,1)` and
`d` be the unit world direction of the unchanged virtual model force.

1. Compute `d_B = R.T @ d`, `v = e3 × d_B`, `s = ||v||`, and `c = d_B.z`.
2. Set `theta = atan2(s,c)` and `e_t = theta * v / s` away from singular cases.
3. Use `alpha_d = wrench.tilt_gain * e_t + diag(wrench.rate_gain) * (omega_ref - omega)`.

`e_t` is the full-angle rotation vector of the nearest attitude whose third
column is `d`: `R_nearest = R @ Exp(e_t)`. It uses current body axes, and its z
component is exactly zero. It is not the xy projection of a full attitude log.
The minimum uses ordinary SO(3) rotation-angle distance, not minimum torque.

At alignment, `e_t=0`. At zero virtual-force magnitude (≤1e-6 N), the tilt term
is exactly zero and rate feedback remains. At antipodal directions the nearest
rotation is not unique. Within one machine epsilon in `s` with `c<0`, the code
chooses a body-x half-turn. This defines a finite answer, not a globally smooth
controller. No heading history or yaw state is stored.

Gains are shared under `env.wrench`: default tilt gain 150 s⁻² and rate gains
(20,20,8) s⁻¹. `yaw_gain`, `heading_range_degs`, and `tilt_method` do not enter
the new law. The shared configuration validator still checks their ordinary
valid defaults. The new controller always uses the direct shortest-axis error.

With the old `geom` split, setting the yaw gain to zero on a test instance gives
the same tilt/rate law away from the antipode. The new mode also removes the
heading output and heading-completion calculation. No old defaults are changed.

## Tasks and launch

The shared implementation supports command velocity, GoTo, goalkeeper and simple
goalkeeper, with PPO and the existing SAC CE/MSE registrations. Configuration,
observations, delays, resets, actuator gains, action-limit costs, SAC bounds,
source snapshots and command-task symmetry use the eight-output layout.

After the usual conda activation and certificate exports, from the repo root:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto --num_envs=10000 --headless env.control_mode=model_wrench_yawless
```

Use the same task/controller override in that task's public player. For SAC,
select its existing `rsl_rl_sac_mse_cfg_entry_point` or CE entry point using
`--agent`. The default mode, rewards, curriculum and clocks do not change.

## Sources and checks

Implementation: `controllers/borinot_model_wrench_yawless.py`. Initialization,
buffer observations and resets reuse `ModelWrenchController`; its source file
remains unchanged. Focused geometry/dynamics tests and a real-simulator smoke
live under `borinot_goalkeeper/tests/`; symmetry checks are under
`borinot_cmd_vel/tests/test_yawless_symmetry.py`.

The editable derivation is `2026-11-05-borinot-model-wrench-yawless.tex`, with a
native-math PDF and numeric receipts in the corresponding workspace artifact
folder and Desktop/emmaDocs. It defines the nearest-attitude optimization,
proves the minimum, explains frame conventions and singularities, and includes
a worked example and a limited ideal stability argument.

Lie conventions follow [A micro Lie theory for state estimation in robotics,
version 9](https://arxiv.org/abs/1812.01537v9), sections II-D/E and Appendix B.
The yaw-free controller construction here is our derivation, not a controller
attributed to that estimation reference. The contact-free/model/actuator
limitations of [MODEL_WRENCH_CONTROL.md](MODEL_WRENCH_CONTROL.md) still apply.
