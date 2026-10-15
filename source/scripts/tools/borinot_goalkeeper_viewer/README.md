# Borinot goalkeeper scene viewer

This is a lightweight 3D scene inspector. It runs in your browser. You do not need Isaac Sim, a conda environment, training, or a policy checkpoint.

## Collision inspection

The viewer now also shows all USD arm/base collision hulls and the task paddle. Its self-collision panel names geometric overlaps and distinguishes eligible pairs from ignored parent–child pairs. For actual contact forces and joint motion, use the isolated IsaacLab bench. See [collision findings and launch commands](COLLISIONS.md).

## Start

Run from any directory:

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer.py
```

The command opens `http://127.0.0.1:8790`. Keep the terminal open. Press Ctrl+C to stop the server. Use `--no-open` to avoid opening a browser, or `--port 8791` if that port is busy.

A background preview is already running on Jordi's laptop. You can open that URL directly. To stop that background instance before using the terminal launcher:

```bash
systemctl --user stop borinot-goalkeeper-viewer.service
```

## Explore

- Drag Borinot or its paddle to move it across the goal and up/down. This drag keeps X fixed. Use the X slider to move toward or away from the goal.
- Use position, roll/pitch/yaw and arm sliders. Numeric inputs also accept continuous values. Arm sliders use the USD's physical limits, not normalized policy actions.
- Drag empty space to orbit. Right-drag to pan. Scroll to zoom.
- Use Goal front, Side and Top for scale comparisons. These use an orthographic camera: distance does not change the apparent size. The 3D view uses normal perspective.
- Click the scene, then use arrow keys to move Y/Z. Page Up/Down move X. The increment is 2 cm, or 10 cm with Shift.
- Restore config pose returns to the configured home position, `robot_home_quat` orientation, and joint pose. Put paddle at shot aim centre translates the robot so the current paddle centre matches the fixed shot target centre. It does not solve for a flight or arm motion.
- Toggle the allowed base-position box, shot aim region, goal transparency, dimensions, and all collision hulls.
- New shot samples one of the task's four launch levels. Play and the time slider show a gravity-only flight path. The ball passes through the robot. This is not contact simulation or a save prediction.

All controls change the preview only. They never write to the task config.

## Geometry and configuration

The viewer reads selected literal values from the current `borinot_goalkeeper_env_cfg.py` at launch. It never imports the environment. Relaunch after editing the config. It previews one environment at its local origin, with no reset noise.

The visible robot meshes, local transforms, two joint frames and physical limits come from `borinotFlat.usd`. The arm pose uses forward kinematics: it computes each link's position from its joint angles. It does not use the USD's authored display pose. The paddle uses the task's fixed-joint offset and orientation. The corrected task starts at −90° yaw; its arm sweep and paddle face both lie along the goal's Y–Z plane. Changing either arm joint keeps the paddle face in that plane while the body remains level at the home yaw. Colours are simplified. Hidden USD collision shapes are optional overlays; the main mesh spans shown in the header do not include them.

The goal is the task's solid cuboid, not an open net. Its dimensions are 2.4 m wide, 2.4 m tall and 4 cm thick in the current config. The paddle face is 24 × 24 cm. The ball diameter is 6 cm. The floor matches `env_spacing - 0.5` and the grid spacing is 25 cm for the default spacing.

The orange box constrains the base origin only. Warnings flag a base outside that box or visible geometry below the floor. They are geometry checks, not collision or dynamic reachability tests. The shot target remains fixed when you move Borinot, as it does in the environment.

The shot aim centre matches the neutral paddle at the reference home height of 1.5 m. A locally tuned home Z of 0.5 m puts the paddle about 1 m below it and the base below the allowed minimum of 0.55 m. The viewer exposes these differences; it does not fix them.

## Standalone offline HTML

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer.py --export /home/jordibelp/Desktop/borinot-goalkeeper.html
```

Double-click that HTML file. It contains the meshes, config snapshot, pinned local Three.js r160 and its MIT license. It needs no server or network. An exported HTML file does not refresh when you edit the config. Export it again to update it.

The normal launcher uses only Python's standard library. It checks the source USD hash against the cached geometry. If the USD changes, it stops rather than displaying old geometry. To refresh the cache, use a Python environment with `usd-core`, NumPy and SciPy and run:

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer/export_robot.py
```

This refresh uses the [OpenUSD joint frames](https://openusd.org/release/api/class_usd_physics_revolute_joint.html) and mesh data. The browser renderer uses [Three.js BufferGeometry](https://threejs.org/docs/#api/en/core/BufferGeometry). It does not start Isaac Sim.

This viewer does not simulate thrust, gravity on Borinot, collision response, reward, curriculum promotion or policy behaviour. Use Isaac Sim when those dynamics matter. For comparing geometry and moving the robot around, this viewer is enough.

## Verification

The original standalone HTML passed 108 browser checks. These covered real pointer dragging, position and joint inputs, invalid-input rejection, camera presets, orbit/pan/zoom, keyboard movement, reset/alignment, overlay toggles, all four shot levels, playback and mobile layout. The corrected paddle mount and home yaw passed 11 focused browser checks, including five bent-arm poses, orientation restore, target alignment, and offline loading without external requests or JavaScript errors. Those five rendered poses also matched independent raw-USD joint-frame calculations to within 10⁻⁷ m.

All three meshes match the original USD's 169,350 vertices exactly at float32 precision. Fourteen neutral, joint-endpoint, folded, tilted and translated poses were compared against a separate MuJoCo 3.3.7 model built directly from the raw USD joint frames. `mj_forward` computed poses without physics stepping. Maximum link/paddle position error was below 3 × 10⁻¹⁵ m. Maximum visible-span bound error was below 4 × 10⁻⁹ m. This verifies geometry, not flight or impact behaviour.
