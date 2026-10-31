# Borinot angle outputs

All Borinot configurations expose these independent options. All default to `False`:

```text
env.predict_yaw_as_sincos=True
env.predict_calf_joint_pos_as_sincos=True
env.predict_thigh_joint_pos_as_sincos=True
```

Append the options you want to your normal training or playback command. Keep the
same options when loading its checkpoint. Each enabled, supported angle replaces
one scalar output with two outputs **at the same place**: `(cos_component,
sin_component)`. Later channels shift right. Each pair adds one action dimension.
The previous-action observations and critic inputs grow by the same amount.

## Supported modes

The original goalkeeper, simple goalkeeper, and command-velocity task share this layout:

| Control mode | Yaw pair | Thigh/calf pairs | Default actions | Actions with all supported pairs |
| --- | --- | --- | --- | --- |
| `thrust_and_torque_control` | No | No | 8 | 8 |
| `thrust_and_PD_joint_pos_control` | No | Yes | 8 | 10 |
| `PI_rotors_PD_joint_pos_control` | No: it predicts a yaw **rate** | Yes | 6 | 8 |
| `indi` | Yes | Yes | 9 | 12 |
| `wrench` | Yes | Yes | 9 | 12 |
| `cbrt` / `cbrt_indi` | No: they predict a yaw **rate** | Yes | 6 | 8 |

Unsupported flags have no effect. Existing `cbtr` / `cbtr_indi` aliases still work.
Existing algorithm restrictions remain: PI is PPO-only; the other listed modes
support PPO and SAC. SAC CE and MSE use the same action mapping.

The older `Borinot-Direct-v0` task already uses two pairs for both leg joints when
`des_joint_out=True`. Its ten-action interface and existing component clipping
stay unchanged, regardless of the new flags. Its torque mode stays at eight actions.
It has no predicted yaw-angle channel.

## Decoding and units

We normalize each pair and decode `atan2(sin_component, cos_component)` in radians.
The result lies in `[-pi, pi]`. Positive pair rescaling does not change the decoded
angle. `(0, 0)` means zero radians. The decoder handles very small and large finite
components without squaring the original large vector.

For the three shared-control tasks, pairs are not component-clipped or penalized
for their magnitude. Existing scalar clipping and action-limit costs still apply
to the other channels. SAC supplies pair components in `[-1, 1]`, independently
of the bounds on any remaining scalar joint outputs.

An enabled **joint** pair encodes an absolute joint angle in radians. It bypasses
`joint_target_scale`, `sac_joint_pos_offset`, and controller-specific SAC joint
target scales for that joint only. Disabled joints keep their old mapping.

An enabled **yaw** pair follows the existing reference convention:

- INDI PPO and SAC: heading offset from the yaw in `robot_home_quat`.
- Wrench PPO: absolute world heading.
- Wrench SAC: heading offset from the yaw in `robot_home_quat`.

The SAC scalar heading-range scale does not multiply a decoded pair angle.

For example, INDI with all three pairs uses:

```text
[acceleration_world_xyz, angular_rate_body_xyz,
 yaw_cos, yaw_sin, thigh_cos, thigh_sin, calf_cos, calf_sin]
```

## Controller behavior and checkpoints

PD/PID gains, joint position stops, torque saturation, rotor allocation, motor
dynamics, command delay, and `hold_arm` behavior remain unchanged. In particular,
the joint controller still subtracts its joint coordinates. We do not introduce
shortest-path error wrapping for mechanically limited joints. The sine/cosine
output removes the scalar policy-output representation; it does not remove the
branch cut of the decoded scalar angle or make a limited joint continuous.

Leaving every flag `False` preserves the previous action contract. Enabling a
supported flag changes action and observation sizes. Old checkpoints cannot be
loaded directly into that expanded interface. Start a new run, or keep the old
flags for playback. Matching tensor sizes alone is not enough when switching
which angle uses a pair.

Motivation supplied by Jordi: [On the Continuity of Rotation Representations in
Neural Networks](https://arxiv.org/abs/1812.07035). This implementation is the
two-component encoding of individual angles, not a full 6D 3D-attitude output.
