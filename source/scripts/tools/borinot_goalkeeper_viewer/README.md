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
- Use **Ping-pong racket / Ø (cm)** to change the round head diameter. The numeric input also accepts values between slider steps. The head, handle placement, collision outlines, overlap checks and robot bounds update together. **Restore config size** restores the configured diameter without changing the pose. The displayed `paddle_diameter_m = ...` line is ready to copy into the config; the slider does not write files.
- Use **Δ (cm)** to shift the entire racket along the second link. Positive values move toward the calf tip along its local −Y axis; negative values retract. Zero retains the current mount. The shift follows bent joints, without changing the face normal or its normal standoff. The displayed link-frame head position gives its distance from the second-link origin. **Restore config mount** resets only this shift; it leaves diameter and pose alone. Copy `paddle_mount_delta_m = ...` into the task config to persist it.
- Toggle **Racket face parallel to arm plane**. Checked keeps the current face orientation (`racket_parallel_to_arm_plane=True`). Unchecked twists the whole racket 90° around its handle axis, making the face perpendicular to the arm's sweep plane (`False`). The head centre, handle direction, size, mount offset and shot target stay fixed. Visible geometry, collision hulls, overlap checks and bounds update together, including with bent joints and a tilted body. **Restore config orientation** restores only this boolean. Copy the displayed setting into the task config, or append `env.racket_parallel_to_arm_plane=False` to a training/player command and restart that environment. Preview edits never change a running training/playback environment.
- Drag empty space to orbit. Right-drag to pan. Scroll to zoom.
- Use Goal front, Side and Top for scale comparisons. These use an orthographic camera: distance does not change the apparent size. The 3D view uses normal perspective.
- Click the scene, then use arrow keys to move Y/Z. Page Up/Down move X. The increment is 2 cm, or 10 cm with Shift.
- Restore config pose returns to the configured home position, `robot_home_quat` orientation, and joint pose. Put paddle at shot aim centre translates the robot so the current paddle centre matches the fixed shot target centre. It does not solve for a flight or arm motion.
- Toggle the allowed base-position box, shot aim region, goal transparency, dimensions, and all collision hulls.
- Use **Shot curriculum / Phase** to select any configured phase or a new preview phase. **New shot** samples its current spread and flight-time range. Play and the time slider show a gravity-only flight path. The ball passes through the robot. This is not contact simulation or a save prediction.

All controls change the preview only. They never write to the task config.

## Design the shot curriculum

The first panel reads all phases from `shot_half_extents_yz` and `shot_flight_time_ranges_s`. The selected phase starts at the configured `shot_level`. Phase indices start at zero. There is no fixed four-phase limit in the editor.

- **Y half-spread ± (m)** controls the horizontal range around `target_center_yz[0]`.
- **Z half-spread ± (m)** controls the vertical range around `target_center_yz[1]`.
- **Flight min/max (s)** controls the time from launch to the keeper plane, not the goal. Shorter times produce faster shots for the same launch and target.
- **Add phase** appends an independent copy of the selected phase and selects it. Edit that copy to design the next difficulty level.
- **Remove** deletes the selected phase and reindexes the remaining phases. At least one phase remains.
- **Restore config phases** restores all source spreads, time ranges and the source starting level. It does not change the robot pose or racket settings.
- Open **Config to copy** and click **Copy config** to copy both phase tuples and the selected `shot_level`. Paste these lines into the task config when ready. No file is saved automatically. Reloading the page discards preview edits.

These are **half-extents**, not full dimensions. For example, `(0.20, 0.12)` means targets span ±20 cm horizontally and ±12 cm vertically: a rectangle 40 cm wide and 24 cm tall. The summary displays those full dimensions. Zero spread is allowed. Flight times must be positive, and min may equal max. Empty, negative, non-finite or inverted inputs leave the last valid phase active and show an error.

The **orange outline** shows the selected target rectangle at `keeper_plane_x`. The **purple filled area** shows the same Y/Z rectangle projected onto the goal's front face. Other purple outlines let you compare all phases. Toggle the projection and comparison separately; the orange outline has its own Scene guides toggle. The overlay stays centred at the configured target when you move the robot. It is not clipped at the goal edges; the summary flags a projected area that extends outside the goal.

The purple projection is **not an impact envelope**. The task samples the aim at the keeper plane, in front of the goal. Gravity and launch noise change where an unblocked ball reaches the goal later. The path guide uses the edited spread and flight times with the original launch ranges, noise and gravity. Very wide or slow custom phases can miss the goal or reach the floor; choosing a phase is not proof that every shot threatens the goal or is reachable by the robot.

## Robot spawn locations

Select a shot phase first. The **Robot spawn locations** panel shows one cyan base-origin cross in centre-only phases, or all available grid locations in later phases. The default is a 3 × 3 grid. The panel follows the configured `reset_grid_enabled` and `reset_grid_start_level`; changing a shot phase does not move the robot automatically.

Use **Inspect location** to put the robot exactly at one location, without small reset noise. **Sample robot reset** picks uniformly from the available locations and adds the configured XYZ reset noise. Both restore the config body orientation and joint pose. Racket size/mount preview edits remain unchanged. The ball aim stays fixed when the robot moves.

Tune `robot_home`, `reset_grid_y_offsets_m`, `reset_grid_z_offsets_m`, `reset_grid_start_level`, `reset_grid_enabled`, `reset_xy_noise_m` and `reset_z_noise_m` in the task config. Restart the viewer after editing it. Grid offsets use world Y/Z, not body coordinates. Locations are numbered by bottom-to-top Z row, then Y column, for the default ascending offsets. The cyan crosses mark base origins, not racket centres. They do not certify collision-free or dynamically reachable poses.

The default home height is now 1.5 m. Y offsets are −0.8, 0 and +0.8 m; Z offsets are −0.65, 0 and +0.65 m. This gives base heights 0.85, 1.5 and 2.15 m, before noise. Phases 0–1 use only the centre; phase 2 onward samples all nine locations, including the centre.

## Geometry and configuration

The viewer reads selected literal values from the current `borinot_goalkeeper_env_cfg.py` at launch. It never imports the environment. Relaunch after editing the config. It previews one environment at its local origin, with no reset noise.

The visible robot meshes, local transforms, two joint frames and physical limits come from `borinotFlat.usd`. The arm pose uses forward kinematics: it computes each link's position from its joint angles. It does not use the USD's authored display pose. The paddle uses the task's fixed-joint offset and orientation. The corrected task starts at −90° yaw; its arm sweep and paddle face both lie along the goal's Y–Z plane. Changing either arm joint keeps the paddle face in that plane while the body remains level at the home yaw. Colours are simplified. Hidden USD collision shapes are optional overlays; the main mesh spans shown in the header do not include them.

The goal is the task's solid cuboid, not an open net. Its dimensions are 2.4 m wide, 2.4 m tall and 4 cm thick in the current config. The default racket head is round, **16 cm in diameter** and **12 mm thick**, with a **6 × 2.6 cm** handle. It replaces the old 24 × 24 cm square. The head center and mount are unchanged. The handle points toward the wrist and keeps its configured dimensions as you resize the head. The ball diameter is 6 cm. The floor matches `env_spacing - 0.5` and the grid spacing is 25 cm for the default spacing.

The task settings are `paddle_diameter_m`, `paddle_thickness_m`, `paddle_handle_length_m`, `paddle_handle_width_m`, and `paddle_mount_delta_m`. The browser reads them at launch. For training or playback, `env.paddle_diameter_m=0.12 env.paddle_mount_delta_m=0.04` selects a 12 cm head shifted 4 cm toward the calf tip. The head's local position becomes `(0, -0.20, 0.029)` m instead of the zero-delta mount `(0, -0.16, 0.029)`. The normal standoff stays fixed. The shot aim stays fixed as you shift the mount. In physics, head and handle share one 100 g racket body and both count as paddle hits. PhysX recalculates its mass distribution when the geometry changes. The browser does not simulate that change in dynamics.

The orange box constrains the base origin only. Warnings flag a base outside that box or visible geometry below the floor. They are geometry checks, not collision or dynamic reachability tests. The shot target remains fixed when you move Borinot, as it does in the environment.

The shot aim centre matches the zero-delta neutral paddle at the reference home height of 1.5 m. A different home or mount changes that alignment without moving the shot aim. A home Z of 0.5 m would put the base below the allowed minimum of 0.55 m. The enabled task reset grid rejects an out-of-bounds home/grid/noise combination; the viewer still shows source values without silently adjusting them.

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

The spawn-grid update passed **144 CPU tests**, **64 spawn-browser assertions**, and **136 real-simulator assertions** across two launches. The simulator checks read actual PhysX root transforms for centre/grid/noisy starts, partial and automatic resets, and a promotion at the switch boundary. They verify fixed shot targets and self-collision flags OFF/ON after real Hydra parsing. The ON launch also changed the grid spacing, switch phase and reset noise. The shot-editor regression passed **1379 assertions** with Jordi's six local phases.

Run the spawn interaction checks on an exported file or live viewer:

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer/check_spawns.py http://127.0.0.1:8790/ --output /tmp/goalkeeper-spawn-checks.json
```

Run the curriculum browser checks on an exported HTML or the live viewer:

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/borinot_goalkeeper_viewer/check_curriculum.py http://127.0.0.1:8790 --output /tmp/goalkeeper-curriculum-checks.json
```

These checks read actual rendered rectangle vertices and compare their bounds with the source or edited half-extents. They independently evaluate sampled ballistic paths. They also exercise arbitrary phase counts, phase-copy independence, removal down to one phase, zero and continuous spreads, fixed flight times, invalid inputs, paired Python config export, clipboard fallback, overlay toggles, camera presets and mobile layout. The standalone page must load and work without external resources or JavaScript errors. This verifies geometry and interaction, not physics or learned performance.

The shot-phase update passed **1045 browser assertions**, plus the existing **1118 geometry/browser regression assertions**. The refreshed local page matched the tested standalone export byte-for-byte. Live desktop/mobile checks passed. The actual CPU task sampler also passed 5120 shots across the four source phases and an added fifth phase with zero Y spread and a fixed flight time. No simulator dynamics or training run was needed for this preview-only change.

The round-racket and mount controls passed **1118 browser checks** over 40 arm poses and five diameter settings. These check the circular radius, thickness, handle location, invalid inputs, slider keyboard input, independent size/mount resets, and convex overlap results against an independent solver. Mount shifts are checked at neutral and two bent poses against the distal link's rotation, including every moved collision hull. Actual task physics separately checked 12, 16 and 24 cm rackets. The 16 cm head with a 6 cm handle clears the upper arm at neutral with self-collisions enabled; this does not guarantee clearance at every pose, size or offset.

The final live page matches the exported HTML byte-for-byte. Its combined diameter/mount controls, independent resets and mobile layout passed a separate live check. Task physics also verified −4, 0 and +4 cm mount shifts at three arm poses and a real Hydra launch with both settings overridden.

The original standalone HTML passed 108 browser checks. These covered real pointer dragging, position and joint inputs, invalid-input rejection, camera presets, orbit/pan/zoom, keyboard movement, reset/alignment, overlay toggles, all four shot levels, playback and mobile layout. The corrected paddle mount and home yaw passed 11 focused browser checks, including five bent-arm poses, orientation restore, target alignment, and offline loading without external requests or JavaScript errors. Those five rendered poses also matched independent raw-USD joint-frame calculations to within 10⁻⁷ m.

All three meshes match the original USD's 169,350 vertices exactly at float32 precision. Fourteen neutral, joint-endpoint, folded, tilted and translated poses were compared against a separate MuJoCo 3.3.7 model built directly from the raw USD joint frames. `mj_forward` computed poses without physics stepping. Maximum link/paddle position error was below 3 × 10⁻¹⁵ m. Maximum visible-span bound error was below 4 × 10⁻⁹ m. This verifies geometry, not flight or impact behaviour.
