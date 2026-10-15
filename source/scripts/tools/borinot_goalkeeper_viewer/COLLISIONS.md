# Borinot arm collisions: findings and inspectors

## Answer

Self-collisions can be enabled without freezing the arm. But setting that flag alone does not prevent every arm–body intersection.

The local Isaac Sim 5.1.0 tests show two different issues:

1. PhysX always filters parent–child articulation pairs. Here these are base–upper arm, upper arm–forearm, and forearm–paddle. Enabling self-collisions does not change that rule.
2. The current paddle really intersects other collision shapes at some joint combinations. These contacts prevent the requested motion when self-collisions are enabled. This is not evidence of a simulator bug.

The goalkeeper config still has `robot.spawn.articulation_props.enabled_self_collisions=False`. This investigation changes neither that setting nor the source USD. No policy or training run was used.

## Start with the browser inspector

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer.py
```

Open <http://127.0.0.1:8790>. Enable **All collision hulls**. Change **Preview filters** to **Self-collisions ON**. Use the shoulder and elbow sliders, then **Focus robot** and the camera views.

- Green outlines have no detected overlap.
- Red outlines overlap a pair eligible for self-contact.
- Orange outlines overlap a pair ignored by the selected filters. A small overlap around the elbow is expected; PhysX ignores that adjacent pair.
- The list names the overlapping shapes and explains each ignored pair.

Try `(shoulder, elbow)` in degrees: `(0, 0)`, `(30, 0)`, `(0, 60)`, `(63, 80)`, and `(0, 180)`. The last two show why checking separate joint limits is not enough.

This is an approximate geometric check, not physics. The arm uses convex hulls of its USD visual meshes. Cylinders use 64-sided polygons in the preview. The checker excludes PhysX contact margins and cooked-hull simplification. It does not prove a pose is reachable from the current pose. Changing the preview filters does not edit the task.

## Actual IsaacLab motion and contacts

Activate the local environment:

```bash
cd /home/jordibelp/IsaacLab && source /home/jordibelp/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab && export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt SSL_CERT_DIR=/etc/ssl/certs CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Open the isolated bench:

```bash
./isaaclab.sh -p source/scripts/tools/borinot_collision_inspector.py '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The **Borinot collision bench** window has shoulder and elbow target sliders. Both robots receive the same PD targets. The negative-Y robot has self-collisions OFF; the positive-Y robot has them ON. The labels show actual angles and pairwise normal contact forces in newtons. Green means no reported contact, not guaranteed clearance. The OFF robot can pass through itself.

The bench uses **CPU physics** for fresh per-step PhysX contact reports, with normal GPU rendering in the GUI. GPU tensor reports can retain old forces after imposed poses, so they are preserved separately in the raw data but do not classify this map. The tool rejects GPU-physics selection rather than silently showing missing reports. This does not change the task’s training device.

The base is level and fixed. Gravity is zero. There is no ball, goal, flight controller, policy, reward or automatic task reset. This isolates the arm geometry and contact solver. The bench uses the task's paddle attachment and arm actuator properties. It is not a floating-flight stability test.

Add `--demo` for a clear motion followed by the obstructed 63°/80° target. Uncheck the demo checkbox to use the sliders. **Target neutral** commands a return; it never teleports through an obstacle. Use `--self-collisions on` or `off` for one robot. Use `--duration 20` for a timed run. Close the application when finished.

Reproduce the grid and continuous-motion tests:

```bash
./isaaclab.sh -p source/scripts/tools/borinot_collision_inspector.py --headless --scan --grid-deg 10 --output /home/jordibelp/.openclaw/workspace/artifacts/2026-10-15-borinot-collisions/repeat-scan.json '--kit_args=--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false'
```

The grid includes zero and the USD limits: shoulder ±90°, elbow ±185°. Each grid pose is deliberately imposed and probed for one 5 ms step. This is a contact probe, not a reachable-pose test; deep overlaps can produce large forces. The separate continuous tests start at neutral, ramp their PD target for 1.5 s, then hold it for 1 s. They do not teleport along the path. All 500 samples per path and both robot results are saved. The report collection is cleared before every step. Normal force is the magnitude of the summed normal impulse vectors divided by the 0.005 s physics step; pairs above 0.001 N are recorded. Lost-contact events do not retain a prior force. Link poses are refreshed after imposed joint writes.

## What the tests found

The four clear paths—shoulder ±30° with elbow zero, and elbow ±60° with shoulder zero—tracked identically with self-collisions ON/OFF. No self-contact was reported along those tested paths.

The 63°/80° target reached its angles with self-collisions OFF. With them ON, the paddle hit the base and the final shoulder angle was about 34°. A 180° elbow target stopped near 99° because the paddle hit the upper arm. These are measured endpoints after the test duration, not certified mechanical joint limits.

The source base has two ski boxes and two **filled cylinders**. `propellerShield` is about 45.0 cm in diameter and 6.20 cm thick. `helixShield` is about 56.5 cm in diameter and 1.60 cm thick. They are not annular rings or detailed rotor/chassis meshes. A visually empty part of the robot can therefore still be solid for physics. The tests prove contact against these proxies; they do not establish whether the proxies match the real robot's clearance.

The delivered browser passed 693 independent checks across 40 poses. Its convex-overlap results were compared against an independent linear-feasibility solver. The IsaacLab GUI was rendered with its slider models driving both robots. These checks are separate from the actual PhysX grid/motion results.

## How to make the intended physical model

1. Keep collision enabled for ball/robot and environment/robot contacts. Self-collision filtering is a separate decision.
2. Check the filled shields against CAD or measured arm clearance. If they cover space the real arm can use, replace them with suitable compound shapes or a suitable cooked decomposition. Do not remove a real obstruction merely to make a target reachable.
3. Keep intended hinge overlap filtered. Do not enable every geometric pair; neighboring pieces overlap around their bearings.
4. If upper-arm contact with the chassis must be physical, the current base–upper-arm topology cannot provide it. One candidate is to move chassis collision geometry onto a separate fixed link, keeping total mass and inertia unchanged and excluding the intended shoulder clearance. That is a model change needing its own validation; it is **not implemented or proven here**. A joint-dependent safe envelope is a different option, but it prevents commands rather than adding physical contact.
5. After fixing and verifying the model, enable self-collisions in a small goalkeeper test. Its current crash detector treats per-body non-ball contact above 0.1 N as a crash. With `reset_robot_on_robot_crash=True`, self-contact can therefore reset an episode. That behavior can look like a frozen arm. The bench intentionally has no such reset.

The independent degree limits and PD target limits do not guarantee that every shoulder/elbow combination is collision-free. Do not infer a global safe range from the four clear paths.

## Files and evidence

The dated artifact folder contains the complete raw PhysX scan, GUI readback, screenshots, browser validation, and an executed notebook with the contact map and motion calculations. Its offline ZIP includes all notebook inputs. Open the notebook in Jupyter, select a Python environment with NumPy and Matplotlib, and use **Run All**. Saved outputs are already visible.

NVIDIA documents the articulation filtering rule in its [articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/guides/articulation_stability_guide.html). This local runtime check used Isaac Sim 5.1.0. IRICluster's different simulator version was not tested.
