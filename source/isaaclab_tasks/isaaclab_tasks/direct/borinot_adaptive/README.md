# Adaptive direct-actuator Borinot

Two tasks, one observation contract. See **borinot-adaptation.tex** for signal order, frames, timing, observability and experiment design. No geometric controller or allocation inverse is in the action path. Observations are clean by default: body-frame kinematic linear acceleration, the full flattened rotation matrix, and world height (no absolute XY in the sensor history). GoTo still receives the relative goal and deadline. The student still has no direct linear velocity; this is not an identical-input baseline. Hardware would need estimated pose and kinematic acceleration, not just raw IMU and encoder samples.

This is observation contract `borinot-direct-v2`. Earlier adaptive teacher, DAgger and student checkpoints use a different input layout; use the old revision to play them or start a fresh teacher under v2.

From the IsaacLab root, with the usual environment activated:

```bash
task=Borinot-goto                 # or Borinot-cmd-vel
p=source/scripts/rsl_rl
./isaaclab.sh -p "$p/train.py" --task "$task-teacher" --headless
# Select a trained teacher, then run imitation (student executes; teacher labels).
./isaaclab.sh -p "$p/train_solo12_base_imu_dagger.py" --task "$task-student-dagger" \
  --teacher-checkpoint "/absolute/path/to/teacher.pt" --headless \
  --dagger_experiment_name "${task}_dagger" --action-loss-coef 0.25
# Select an adapter and retain its teacher at the recorded path. Fresh PPO optimizer.
./isaaclab.sh -p "$p/train.py" --task "$task-student-rl" --headless \
  --init-from-dagger-adapter "/absolute/path/to/adapter.pt"
# Student playback; use play_borinot_cmd_vel.py for the other family.
./isaaclab.sh -p "$p/play_borinot_goto.py" --task Borinot-goto-student-rl \
  --checkpoint "/absolute/path/to/student_ppo.pt"
```

Carry the same plant, disturbance, reward and curriculum overrides into every stage. Select the curriculum start explicitly: a checkpoint filename is not a reset distribution. The adapter is not a standalone policy. Do not give it to the playback script. W&B reference etiezqty supplies architecture, not weights or Solo12 symmetry transforms. Training and testing here are simulation only.
