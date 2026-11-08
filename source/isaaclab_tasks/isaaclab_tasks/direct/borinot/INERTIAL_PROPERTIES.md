# Shared Borinot inertial properties

## Default asset: Jan's short arm, revision 6b3e09ce

All Borinot training tasks now select:

`source/isaaclab_assets/data/Robots/Borinot/borinotFlat_6b3e09ce.usd`

This is an independent copy of `borinotFlat.usd`. The original stays unchanged.
The copy replaces only each body's mass, centre of mass (CoM), and full inertia
with `borinot_description` commit `6b3e09ce998f61824045d6ba4c149ba5dc1b06c4`,
`arm:=short`. It retains every existing part name, joint, mesh, transform, material,
collider and collision approximation. Provenance is stored in USD layer metadata.

A 30 mm orange sphere named `tip_mass_visual` marks the weight at the short-arm
end-effector position, `(0, -0.130, 0)` m in the second link's axes. It follows
that link and is visual only: no collider, rigid body or added mass. The existing
127 g lower-link calibration already includes the weight.

The base mass is **2.71746155 kg**, including all motors and propellers. The upper
link is **0.14853845 kg**. The short lower link is **0.127 kg**, including its
104 g nut. Nominal robot mass is **2.993 kg without a racket**, or **3.093 kg with
the separate 100 g goalkeeper racket**, before startup randomization.

The geometry is the short link, not the new long Solo leg. This change does not
add that long variant or rotor-spin acceleration/gyroscopic dynamics. It matches
Jan's inertial model, not newly measured hardware dynamics: the base tensor is
derived from earlier identification and adjusted mass accounting. Existing task
joint stops, drive settings, armature, gains, action mappings and friction
randomization remain unchanged; they are not replaced by URDF actuator limits.

## Startup overrides and double-counting prevention

| Setting | Default | Meaning |
| --- | --- | --- |
| `env.inertial_properties.include_motor_masses` | `False` | Historical point-mass option, only for the old USD |
| `env.inertial_properties.use_inference_leg_inertias` | `True` | Apply the portable short-arm calibration after legacy shaft/payload construction |

Training does not load an external description repository. Portable leg values
are stored in `inertial_properties_cfg.py` and are archived with the source
snapshot. The default values agree with the new USD. They retain all tensor
entries, including the lower link's nonzero YZ term.

A base marker records that the new USD already includes the rotor masses.
Enabling the old six 5 g point masses on this asset raises a clear error instead
of adding a second 30 g. The legacy option remains available for explicit old
asset playback. It does not include finite-size rotor spin inertia.

The default complete leg replacement supersedes `shaft_mass_kg` and
`ee_includes_puntual_mass`. An old `ee_includes_puntual_mass=0.1` flag therefore
does not add another 100 g to the 127 g lower link. The optional goalkeeper racket
stays a separate body. Set `use_inference_leg_inertias=False` to deliberately use
the legacy shaft/point-payload approximation instead.

Base mass/CoM and joint-friction randomization keep their existing settings.
Hover calibration and model-based controllers use the updated nominal properties;
randomized clones remain separate from that nominal model.

## Training and playback

No new flag is required. Existing commands using `source/scripts/rsl_rl/train.py`
select the new asset through their task configuration. Cmd-vel and GoTo inherit
the shared goalkeeper asset; goalkeeper Simple and the original Borinot task
select the same versioned copy.

For example, from the IsaacLab repository after normal environment activation:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py --task=Borinot-goto --num_envs=10000 --headless env.control_mode=model_wrench_indi
```

MuJoCo players reuse the same task configuration and inertial overrides, so they
also select the new asset by default.

## Older checkpoints

Equal observation/action sizes do not make the old and new dynamics equivalent.
To reproduce an older model, restore its **saved configuration and original USD
path**, not only the two flags. Existing logs/checkpoints are not rewritten.

For cmd-vel, GoTo and both goalkeeper tasks the asset field is
`env.robot.spawn.usd_path`; for the original `Borinot-Direct-v0` it is
`env.usd_path`. Point it to the unchanged `borinotFlat.usd`.

- Before the October 28 inference-calibration update, use the old USD, set both
  inertial switches False, and restore the original shaft/payload/racket flags.
- For the October 28--November 8 115 g inference model, use the old USD and the
  saved old inertial config: six 5 g points enabled and the old 115 g leg values.
  Merely selecting the old USD while retaining the new 127 g leg override does
  not restore that old model.

No push, cluster sync, deployment or hardware change is implied by this local
asset/configuration update.

## Verification

`borinot_goalkeeper/tests/test_calibrated_usd.py` checks the frozen URDF's full
inertial tensors, default configuration, stale-flag rejection, and preservation
of all non-inertial USD attributes and relationships.
`inertial_properties_smoke.py --urdf <short-urdf> --output <receipt.json>` checks
actual PhysX masses, CoMs, full tensors, controllers, clones, resets, optional
racket and legacy-payload precedence, including randomized clones. The CPU
point-merge tests continue to validate the explicit old-asset option.
