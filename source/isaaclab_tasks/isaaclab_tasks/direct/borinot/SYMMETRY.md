# Borinot command-tracking augmentation

## Result

The current complete robot does **not** have the three requested body symmetries.
The motor positions form a symmetric hexagon, but the motor spin signs and the
off-centre leg break those transformations. Only identity remains from that
requested body-mirror group. We do not expose approximate body mirrors.

We provide a different, two-way augmentation for command tracking:

1. Keep the original sample.
2. Turn the **whole robot and its world-frame references** by 180 degrees about
   world Z. Keep body-frame commands and joint coordinates unchanged.

This is a world-heading augmentation. It does not synthesize a front–back leg
motion or a left–right motor permutation. It works with the current asymmetric
physical robot because all its parts turn together.

The implementation follows the existing RSL PPO/SAC augmentation interface used
by Solo12. The motivating [Symmetry Considerations for Learning Task Symmetric
Robot Policies](https://arxiv.org/abs/2403.04359) requires checking the reward and
transition model, not just whether the input/output tensors can be rearranged.

## Use

Append this to the usual command-tracking training command:

```text
--symmetry-mode=augmentation
```

Existing options also work:

| Option | Behavior |
| --- | --- |
| `--symmetry-mode=none` | Default. No augmentation or mirror loss. |
| `--symmetry-mode=augmentation` | Original + rotated samples in each training minibatch. |
| `--symmetry-mode=loss` | Add the existing policy symmetry loss. |
| `--symmetry-mode=both` | Data augmentation and symmetry loss. |
| `--symmetry-loss-coeff=0.001` | Existing coefficient for `loss` / `both`. |

Supported tasks: `Borinot-cmd-vel`, its `Borinot-Cmd-Vel-Direct-v0` alias, and
the older `Borinot-Direct-v0` command task with a correctly resolved legacy layout.
All seven modern control modes are supported. PPO, SAC CE and SAC MSE use the
same transform. The existing PI+SAC restriction stays in place.

`augmentation-left-right` is rejected for Borinot. It remains available for
the existing Solo12 tasks. Goalkeeper augmentation is not enabled: rotating a
robot while keeping its goal and ball launcher fixed does not preserve that task.

Augmentation does not change action/observation dimensions or inference code.
It is a training option. Existing compatible checkpoints still load, and playback
does not need a symmetry flag. The independent angle-encoding flags must still
match the checkpoint, as explained in `ANGLE_ACTIONS.md`.

## Why the body mirrors fail

### Two properties must match

A rotor has a **position** and a **spin type**. For equal upward thrust, opposite
spin types create opposite yaw reaction torques on the drone. In the code,
`rotor_spin` supplies this yaw-torque sign. A `+` in this guide means positive
body yaw torque, not the blade's rotation direction.

Current motor positions and yaw signs, in configuration order:

| Motor | Body X (m) | Body Y (m) | Yaw sign |
| --- | ---: | ---: | ---: |
| M1 | 0 | -0.185 | + |
| M2 | 0 | +0.185 | - |
| M3 | +0.1602147 | +0.0925 | + |
| M4 | -0.1602147 | -0.0925 | - |
| M5 | +0.1602147 | -0.0925 | - |
| M6 | -0.1602147 | +0.0925 | + |

### A 180-degree body permutation

Imagine M1 alone produces 1 N of upward thrust. It gives:

- Total upward force: 1 N.
- Roll torque: -0.185 N*m.
- Pitch torque: zero.
- Yaw torque: +0.01394 N*m, from the configured `km/kf` ratio.

A 180-degree rotation about Z reverses roll and pitch components, but **does not
reverse yaw**. The transformed command must therefore give +0.185 N*m of roll
and still +0.01394 N*m of yaw.

The position mapping selects M2. Its roll torque is correct. Its yaw torque is
negative, so the transformation fails.

This is not fixed by choosing a clever combination of other positive thrusts.
At total thrust 1 N, a roll torque of +0.185 N*m requires every unit of thrust at
the maximum Y coordinate, +0.185 m. Only M2 has that coordinate. Its yaw sign
cannot be changed by the policy. Thus this example also disproves equality of
the feasible wrench sets, not just one proposed permutation.

### A front–back reflection

Reflect body X: `x -> -x`. M1 has X=0, so its position stays where it is.

A mirror reverses the sense of a circular arrow. Positive yaw must become
negative yaw. But leaving the command on M1 still gives positive yaw. This
reflection therefore fails too. Turning a clock drawing and viewing it in a
mirror are different operations: only the mirror reverses its handedness.

The rotor-only left–right reflection does pass: M1 maps to M2, and the required
yaw sign also reverses. However, the **whole robot** still fails that reflection
because of its off-centre leg.

### The leg is not centred

We evaluated forward kinematics from the actual USD joint frames and compared
the mirrored calf tip with the tip obtained by the proposed joint sign changes.
An exact body symmetry would give zero error. The current asset gives roughly:

| Proposed body transform | Calf-tip mismatch |
| --- | ---: |
| Left–right | 12.86 cm |
| Front–back | 12.94 cm |
| Both / body 180 degrees | 18.24 cm |

The robot-model export's SHA256 matches the current USD. These are geometric
counterexamples, not results from training. Ignoring yaw reaction torque would
not fix the leg geometry. We have not changed the robot, rotor signs or physics.

## What the implemented world-heading transform changes

World axes stay fixed. The whole robot turns, so M1 stays M1 with its original
spin type. Its new world position does not mean that we select M2 instead.

| Signal | Transformed value |
| --- | --- |
| Body-to-world rotation matrix | Negate its first two rows. |
| World acceleration reference | Negate world X and Y; keep Z. |
| Desired world heading | Add 180 degrees, with an equivalent circular representative. |
| Body linear/angular velocity, gravity and acceleration | Unchanged. |
| Heading-frame velocity command and world-Z yaw-rate command | Unchanged. |
| Joint positions, velocities, targets and torques | Unchanged. |
| Rotor thrust commands, measured rotor state, PI/PID integrals | Unchanged. |
| Previous raw actions | Apply the same action transform as above. |

For sin/cos yaw outputs, negate both components. No joint pair is changed.
For scalar yaw, the implementation swaps adjacent pi-wide angle intervals.
This preserves unbounded PPO action values without folding every action into
one interval. SAC scalar headings need `heading_range_degs=180, 360, ...` so
the action range is closed under this rotation. Smaller ranges require the
sin/cos yaw representation or disabling augmentation.

The environment defines a zero yaw pair as zero radians. Its rotated sample
uses `(-1, 0)`, which decodes to pi. Exact scalar SAC endpoints and zero pairs
have duplicate/singular angle representations. We choose a valid representative
there; exact raw-vector involution is not claimed at those singular points.

## Verification boundary

We check the transform with independent rotation/reward calculations, actor and
critic TensorDict batches, actions-only/observations-only calls, mixed angle
encodings, and matched live simulator states in every control mode.

The live counterfactual also rotates the queued delayed references. Delays,
controller feedback and motor response are not removed. Physics and reward
matching are distinct from the reset distribution: we do **not** change the
configured home heading, reset sampler or reset queue fill. Thus we do not claim
that the complete reset process is globally heading-invariant. Nor do these
short integration checks establish faster learning or better trained policies.
