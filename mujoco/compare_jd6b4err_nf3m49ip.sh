#!/usr/bin/env bash
# Local shared-command comparison for Jordi's two supplied position-PD policies.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
cd "$ROOT"
exec ./isaaclab.sh -p mujoco/play_joint_mujoco.py --checkpoints /home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1113_jd6b4err_best_model_curriculum_idx_2_walk.pt /home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0716_nf3m49ip_best_model.pt --labels jd6b4err-walk nf3m49ip --task=solo12-two-feet --duration_s=2000 --cmd_init 0.1 0 0 --episode_length_s=80 env.base_obs_from_imu=False env.kp=9.0 env.kd=0.2 env.effort_limit_sim=2.65 --bringup_clip_rad=0.0 'env.joint_physical_limit_calf=[-179.0,179.0]' 'env.joint_physical_limit_hip=[-70.0,70.0]' env.use_asymmetric_thigh_limits=False 'env.joint_physical_limit_front_thigh=[-135.0,45.0]' 'env.joint_physical_limit_rear_thigh=[-45.0,135.0]' 'env.command_lin_vel_x_range=[-1.5,1.5]' 'env.command_lin_vel_y_range=[-1.0,1.0]' 'env.command_ang_vel_z_range=[-1.0,1.0]' "$@"
