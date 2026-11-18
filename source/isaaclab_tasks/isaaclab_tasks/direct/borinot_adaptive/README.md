# Adaptive direct-actuator Borinot

Two tasks, one observation contract. See **borinot-adaptation.tex** for signal order, frames, timing, observability and experiment design. No geometric controller or allocation inverse is in the action path. These tasks require motion-capture-quality pose, an IMU and encoders; they are not IMU-only.

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
