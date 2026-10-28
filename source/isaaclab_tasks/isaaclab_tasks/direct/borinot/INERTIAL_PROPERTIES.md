# Shared Borinot inertial properties

All four Borinot tasks now support the same spawn-time mass overrides:
`Borinot-Direct-v0`, `Borinot-cmd-vel`, `Borinot-Goalkeeper-Direct-v0`, and
`Borinot-Goalkeeper-Simple-Direct-v0`.

We only author overrides on the spawned robot, before cloning and physics startup.
We never edit `borinotFlat.usd`. Body names, joints, collision shapes, observation
sizes, action mappings, and joint limits stay unchanged.

## Two independent switches, both enabled by default

| Setting | Default | Meaning |
| --- | --- | --- |
| `env.inertial_properties.include_motor_masses` | `True` | Merge six ideal motor point masses into the base |
| `env.inertial_properties.use_inference_leg_inertias` | `True` | Replace both leg links' mass, CoM and full inertia with the inference description |
| `env.inertial_properties.motor_mass_kg` | `0.005` | Mass of each motor point, in kg |

The portable numeric reference is in `inertial_properties_cfg.py`. It was
transcribed from `borinot_description/urdf/borinot_flying_arm_2.urdf`, description
revision `76708fb`. Training does not need that external package installed.
CoM means center of mass. Inertia tensors use kg m².

## Motor points

The six points use the reference URDF's base-link positions, in metres:

```python
((0.0, -0.185, 0.05), (0.0, 0.185, 0.05),
 (0.1602146997, 0.0925, 0.05), (-0.1602146997, -0.0925, 0.05),
 (0.1602146997, -0.0925, 0.05), (-0.1602146997, 0.0925, 0.05))
```

You can override these through `env.inertial_properties.motor_mass_positions_b`.
The original base mass is increased by 30 g. We recompute its CoM and full inertia
with the parallel-axis theorem. There are no extra rigid bodies, colliders or DOFs.
The rotor thrust locations and spin signs are not changed.

These are **ideal point masses**, as requested. They do not include the separate
rotor links' intrinsic inertia tensors in the URDF. Thus this matches their mass
and position contribution, not the complete finite-size rotor assembly inertia.
The base's own pre-existing USD inertia is retained before adding the point terms.

## Inference leg properties and payload precedence

The upper link uses mass **0.14853845 kg** and local CoM
**(0.00001377, -0.078707, -0.01935853) m**. Its full tensor includes the nonzero
YZ term **-0.00004671 kg m²**.

The calf uses mass **0.115 kg** and local CoM **(0, -0.1149, -0.021) m**. Its
link-axis tensor is diagonal: **(0.00017554, 0.00001972, 0.00017938) kg m²**.
All six numbers for each full tensor, including off-diagonal terms, are kept in
the config. They are converted into principal moments and axes for USD/PhysX.

The reference 115 g calf **already includes its load**. When the inference leg
switch is on, it replaces the entire old shaft/payload model. Therefore
`env.ee_includes_puntual_mass` and `env.shaft_mass_kg` do not change the final calf
properties in this mode. In particular, an old `ee_includes_puntual_mass=0.1`
command does **not** add a second 100 g. A goalkeeper racket is still a separate
100 g body; it is not removed or merged by these switches.

For custom link properties, edit or override `leg_masses_kg`,
`leg_com_positions_l`, and `leg_inertias_l_kg_m2` inside `env.inertial_properties`.
The order is upper link, then calf. Positions and full own-CoM tensors use each
link's local axes. Positive masses, finite symmetric tensors, and physical
principal moments are validated before simulator construction.

With both switches on, nominal robot mass is **2.940538 kg without a racket**, or
**3.040538 kg with the default 100 g racket**. Hover calibration uses the new
nominal mass. Small final digits depend on PhysX float precision.

## Randomization and old checkpoints

Startup randomization still runs after these overrides. Base mass/inertia scaling
acts on the **combined base**, including motor points. Its CoM offsets are relative
to the new combined CoM. The leg properties are not randomized by those base events.
For an exact nominal comparison in a goalkeeper or cmd-vel task, add
`env.include_events_randomization=False`; disabling the overrides does not itself
disable existing randomization.

To recover previous task behavior for an older checkpoint, add both overrides:

```bash
env.inertial_properties.include_motor_masses=False env.inertial_properties.use_inference_leg_inertias=False
```

Then the cmd-vel task uses its previous 15 g shaft plus selected ideal tip load,
goalkeeper tasks retain their previous shaft-and-racket model, and the original
Borinot task retains its authored leg properties and old thrust calibration.
Use the checkpoint's other original settings as well; equal tensor sizes do not
make changed physical models equivalent.

For example, cmd-vel's old 100 g ideal tip model, without motor masses:

```bash
env.inertial_properties.include_motor_masses=False env.inertial_properties.use_inference_leg_inertias=False env.arm_holds_racket=False env.ee_includes_puntual_mass=0.1
```

## Verification

`borinot_goalkeeper/tests/test_inertial_properties.py` checks first moments,
independent kinetic energy, zero mass, invalid masses, and the affected axes.
`borinot_goalkeeper/tests/inertial_properties_smoke.py` reads the original inference
URDF independently and checks actual cloned PhysX masses, CoMs and full tensors.
It covers all four tasks, both opt-outs, individual switches, randomization,
racket/payload precedence, resets, unchanged body/DOF counts, hover calibration,
and inclusion of both implementation/config files in source snapshots.

Both helper files are archived with future training runs. Existing checkpoints,
training logs, and online W&B records are not rewritten by this change.
