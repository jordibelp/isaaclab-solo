# Solo12 Joint Lab

A small browser tool for inspecting Solo12 joint configurations. It reuses the
robot mesh exporter and offline Three.js library from the two-feet Pose Lab.
You do not need Isaac Sim, ROS, Conda, Node.js, or extra Python packages to run it.

## Open the viewer

Run this from any directory. The command opens your browser and serves the viewer
only on your laptop. Press Ctrl+C in the terminal to stop the server.

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/solo12_joint_web_viewer.py
```

The default address is http://127.0.0.1:8767. Use `--port 8768` for a different port,
or `--no-open` to avoid opening a browser window.

You can also export one self-contained HTML file. Open that file in your browser.
It includes the robot meshes and library. It makes no network requests.

```bash
python3 /home/jordibelp/IsaacLab/source/scripts/tools/solo12_joint_web_viewer.py --export /home/jordibelp/Desktop/emmaDocs/2026-10-07-solo12-joint-lab.html
```

## Move the joints

- All 12 sliders use **degrees**. These are absolute joint positions, not policy actions.
- Type an exact value in a joint's number box.
- Use its down/up buttons to jump to the minimum/maximum.
- Choose **Physical** to inspect full travel, or **Soft** to inspect the narrower range.
- Amber number boxes mark joints outside the soft range.
- **Safe stand** restores the task's safe pose. **All zero**, **All min**, and
  **All max** apply to every joint. Individual endpoint buttons are usually more useful.
- Drag the model to orbit. Scroll to zoom. Side, Front, Top and Fit are camera shortcuts.
- Expand **Body position & display** to change height, nose-up pitch or heading.
  Joint values do not change when you move the body.

The initial preset matches the joint-limit overrides in Jordi's 7 October ROS command:

| Joints | Physical range (degrees) | Margin at each end | Soft range (degrees) |
| --- | --- | --- | --- |
| Hips | -70 to 70 | 20 | -50 to 50 |
| Front thighs | -145 to 55 | 20 | -125 to 35 |
| Rear thighs | -55 to 145 | 20 | -35 to 125 |
| Calves | -179 to 179 | 24 | -155 to 155 |

Soft limits use the same rule as the direct task: add the margin to the lower
physical limit and subtract it from the upper limit. Switching to a smaller
range clamps joints outside it. The status bar reports this.

## Use another command or save a pose

Expand **Edit limits / paste a launch command**. You can edit limits directly,
choose current task defaults, or paste a whole `ros2 launch` command. Click
**Import limit overrides** to apply its explicit `env.joint_physical_limit_*`,
`env.joint_soft_limit_*_delta`, and `env.use_asymmetric_*_limits` fields.

The import does not execute the command. It ignores other fields and does not
load a YAML file or resolve a ROS profile. Missing fields keep the current
viewer values. Front/rear thigh or calf values apply only when their asymmetric
flag is enabled. Invalid ranges and empty soft ranges are rejected.

Expand **Save / load joint positions** to save a JSON file or copy an angle array
in radians or degrees. JSON includes joint names, both angle units, body pose,
limit settings and active range. Loading uses its radians array and checks
the joint order. Arrays use the direct task order:

```text
FL hip, thigh, calf; FR hip, thigh, calf;
RL hip, thigh, calf; RR hip, thigh, calf
```

RL/RR correspond to HL/HR in the hardware joint names. Positive rotations follow
the model axes: hip +X, thigh and calf +Y. Safe front calves are -0.8 rad;
safe rear calves are +0.8 rad.

## What the view means

This is frozen geometry. It does not simulate gravity, test self-collisions,
predict contacts, run a policy or send robot commands. PD gains and effort caps
do not affect the view. A ground-plane warning concerns the foot-centre height,
not a full mesh collision check.

The joint tree and meshes come from the local USD-derived MuJoCo model,
`mujoco/solo12.xml`. The safe pose and task-default limits are read from the
current Solo12 task source without importing Isaac Sim. The existing Pose Lab
remains unchanged. Exported HTML is a snapshot; regenerate it after changing
the robot model or task defaults. The MIT license for Three.js r160 is embedded.
