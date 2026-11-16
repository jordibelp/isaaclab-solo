# Solo12 Joint Lab · choose soft limits

This is the existing October Joint Lab, updated for left/right hip hard limits
and absolute soft ranges. It uses the robot meshes and offline Three.js from
the two-feet Pose Lab. It needs only standard Python to run. No simulator, ROS,
policy, network downloads or robot connection is needed.

## Open

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/solo12_joint_web_viewer.py
```

Open http://127.0.0.1:8767. The server binds only to the local machine.
Use `--no-open` to skip opening a browser, or `--port 8770` for another port.
Stop a terminal-launched server with Ctrl+C.

To save a self-contained HTML file that also works offline:

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/solo12_joint_web_viewer.py --export /home/jordibelp/Desktop/emmaDocs/2026-11-16-solo12-soft-limit-lab.html
```

The page starts with **current source defaults**, not the old October command.
It does not inspect a running job or infer a checkpoint's settings. Paste explicit
`env.*` overrides if you want to inspect another configuration. Both the server
and offline export are snapshots: regenerate/restart after editing source defaults.
A short hash on the page identifies the config source used when it was built.

## Current defaults

These hard-limit defaults match the numeric thresholds in the supplied
`1116_lab_solo12.yaml`. All displayed inputs and outputs use **degrees**.

| Joints | Hard range | Current effective soft range |
| --- | --- | --- |
| Left hips, FL/RL | -69.900851 to 134.645082 | -64.900851 to 129.645082 |
| Right hips, FR/RR | -134.645082 to 69.900851 | -129.645082 to 64.900851 |
| All thighs | -200.535228 to 200.535228 | -190.535228 to 190.535228 |
| All calves | -200.535228 to 200.535228 | -195.535228 to 195.535228 |

Absolute soft fields remain unset in the source. Current soft ranges therefore
use the existing fallback deltas: hip 5°, thigh 10°, calf 5°.
The 20°/20°/24° deltas in the earlier training command are explicit overrides,
not these defaults. The historical October preset is labelled separately.

**Hardware caveat:** the YAML values are measured-position software safety
thresholds, not verified mechanical stops. PhysX uses the configured hard limits
as joint stops. This preview does not simulate the hardware's safety response.

## A useful workflow

1. Start with **Physical limits** so the pose can cross the candidate soft range.
2. Select one joint in **One-joint study**, or click its row/group label.
3. Use **Soft min**, **Soft max**, and **Sweep soft** to inspect its travel.
   Other joint angles stay fixed. Green and blue ghost legs show the lower and
   upper soft endpoints. The foot path is green within soft limits and gray
   over the remaining hard travel. **Fit** includes the path and ghosts.
4. Try different poses for the other joints and change the body pitch. A hip or
   thigh choice can look very different with a bent calf. Side, Front and Top
   views help inspect the direction of motion. Drag to orbit; scroll to zoom.
5. Enter a lower/upper pair in **Choose soft limits**, then **Apply soft edits**.
   These fields affect both matching joints, not only the selected joint.
   The current config supports left/right hips and front/rear thighs/calves;
   it does not support twelve independent soft pairs.
6. Inspect all twelve range bars. Gray is the hard range, green is the soft
   interval, and white is the current angle. Each bar uses its own hard range.
   Amber angles are outside the soft range. The selected-joint readout also
   shows the angle beyond the nearest soft boundary.
7. **Copy soft-only env overrides** when you have a candidate. Replace your old
   soft fields in the launch command with this block. Nothing is saved to task
   config or sent to running training by editing this page.

**Preview earlier posture soft angles** is a comparison, not a recommendation:
hips ±45°, front thighs [-125°,35°], rear thighs [-35°,125°], calves ±155°.
It leaves the current hard limits unchanged. Start there only if you want to
compare with the earlier reward ranges. It does not install them as defaults.

## How soft fields work

Selection is the same as the task's production resolver:

1. A side-specific absolute pair, if set.
2. Otherwise a common absolute pair for that joint type.
3. Otherwise the hard range moved inward by the per-type delta.

Soft selection is independent of the asymmetric **hard** flags. For example,
front/rear soft thigh ranges can differ while all thighs share one hard range.
Only consumed deltas need to be valid, just as in the task. Selected soft pairs
must be finite, ordered and inside each matching joint's hard range.
Invalid edits leave the applied state unchanged and show an error.

Blank pairs inherit. Open **Common soft ranges / fallback deltas** to edit shared
pairs or deltas. **Make current ranges absolute** fills the six side-specific
pairs without changing their effective angles. **Use delta fallback** clears all
absolute pairs and uses the currently applied deltas.

Copied soft-only commands include all nine pair fields, using `null` for unset
pairs, plus the three deltas. This clears stale side-specific fields that could
otherwise override a new common pair. Array tokens are shell-quoted. The full
hard+soft block is under **Edit limits / paste a launch command**.

The import reads explicit joint-limit fields and hard asymmetric flags only.
It does not execute the command, load YAML, resolve ROS profiles or import other
task settings. Missing fields keep the currently applied viewer values. Repeated
fields use the last value. `null` or `None` clears an absolute soft pair.
For an old common hip hard override, also specify
`env.use_asymmetric_hip_limits=False`; the new default uses separate sides.

## Pose controls and persistence

- Twelve sliders and number boxes use absolute joint positions, not actions.
- The down/up buttons jump to the active range's endpoints for that joint.
- **Soft limits** restricts sliders to the resolved soft range. Changing to a
  smaller active range clamps the preview pose; it does not change training.
- **Safe stand**, **All zero**, **All min** and **All max** are pose shortcuts.
  The safe stand is a source-defined reference, not a hardware safety guarantee.
- Body height, nose-up pitch and heading move the root independently.
- Axis, wireframe and ground toggles are under **Body position & display**.
- Save/load JSON preserves joint order, radians/degrees, body pose, all limit
  settings and active slider mode. Older October pose JSON is supported.
- JSON is a pose/settings snapshot. It is not a simulator rollout or recording.

Joint order is FL, FR, RL, RR; each leg is hip, thigh, calf. Hardware HL/HR means
RL/RR. Axes are model hip +X and thigh/calf +Y. The safe front calves are -0.8 rad;
the rear calves are +0.8 rad. No calibration offset or standing-pose subtraction
is applied to the absolute angles.

## What this does not prove

This is frozen **kinematics**: geometry from the local USD-derived MuJoCo model.
Ghosts and paths vary only one joint while holding the others fixed. They are
not a dynamically reachable workspace. Some poses can intersect the body,
other legs or the floor. There is no collision test, contact physics, gravity,
torque/speed check, balance test or hardware command. A floor warning uses only
foot-centre height, not a full collision check.

For PPO two-feet training, soft ranges mark where the joint-position penalty
begins; they do not themselves prevent crossing. SAC also uses the resolved
soft ranges for its position-action scaling. A geometric choice still needs
separate policy/physics evaluation before hardware use. Gains and torque caps
do not affect this preview.

## Regression checks

Exporter tests need only standard Python:

```bash
python3 -m unittest discover -s source/scripts/tools/joint_web_viewer/tests -v
```

Browser checks use an existing `playwright-core` installation and Chrome, with
no package installation. Export a fresh file, then run from the repo root:

```bash
NODE_PATH=/home/jordibelp/openclaw/node_modules node source/scripts/tools/joint_web_viewer/tests/browser.cjs /home/jordibelp/Desktop/emmaDocs/2026-11-16-solo12-soft-limit-lab.html /home/jordibelp/.openclaw/workspace/tmp/solo12-joint-check
```

The browser test writes numeric geometry/config cases and screenshots. Compare
those against MuJoCo forward kinematics and the actual task limit method, using
the existing environment with NumPy, Torch and MuJoCo:

```bash
/home/jordibelp/miniconda3/envs/env_isaaclab/bin/python source/scripts/tools/joint_web_viewer/tests/parity.py /home/jordibelp/.openclaw/workspace/tmp/solo12-joint-check
```

These checks do not launch Isaac Sim or take physics/optimizer steps. They
include each joint's hard endpoints, mixed tilted poses, soft precedence and
fallback, old JSON compatibility, copy/import, invalid-input non-mutation,
endpoint ghosts, colored foot-path vertices, mobile layout and offline requests.
The shared source reader accepts literals, annotated fields and numeric
`math.degrees(...)` expressions; it does not execute the config module.
