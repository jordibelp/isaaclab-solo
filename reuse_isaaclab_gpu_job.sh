#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Reuse an existing Slurm GPU allocation for IsaacLab by launching a new overlapping
step inside it, with the same environment setup used by train_isaac.sbs.

Usage:
  ./reuse_isaaclab_gpu_job.sh --jobid JOBID -- <command> [args...]

Examples:
  ./reuse_isaaclab_gpu_job.sh --jobid 12345 -- ./isaaclab.sh -p source/scripts/rsl_rl/train.py --task solo12-v0 --num_envs 4096 --headless
  ./reuse_isaaclab_gpu_job.sh --jobid 12345 -- bash -lc './isaaclab.sh -p source/scripts/rsl_rl/train.py --task solo12-v0 --symmetry-mode augmentation --num_envs 10000 --headless'

Notes:
  - This does NOT request a new GPU from Slurm.
  - It reuses the GPU already allocated to the running job JOBID.
  - It activates the IsaacLab conda env and applies the same relevant exports as train_isaac.sbs.
EOF
}

JOBID=""
WORKDIR="${HOME}/IsaacLab"
CONDA_ENV="isaaclab"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobid)
      JOBID="${2:?Missing value for --jobid}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:?Missing value for --workdir}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:?Missing value for --conda-env}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$JOBID" ]]; then
  echo "Error: --jobid is required" >&2
  usage >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Error: missing command after --" >&2
  usage >&2
  exit 1
fi

printf -v CMD_Q '%q ' "$@"
CMD_Q="${CMD_Q% }"

srun --jobid="$JOBID" --overlap bash -lc "
set -eo pipefail
set +u
source ~/miniconda3/etc/profile.d/conda.sh
if [[ \"\${CONDA_DEFAULT_ENV:-}\" != \"${CONDA_ENV}\" ]]; then
  conda activate ${CONDA_ENV}
fi
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

echo '--- GPU visibility ---'
nvidia-smi || true
python -c \"import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())\"

echo \"SLURM_JOB_ID=\$SLURM_JOB_ID\"
if [[ -n \"\${SLURM_JOB_GPUS:-}\" ]]; then
  echo \"SLURM_JOB_GPUS=\$SLURM_JOB_GPUS\"
fi
echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}\"

ACTIVE_GPU=\"\${SLURM_JOB_GPUS:-}\"
ACTIVE_GPU=\"\${ACTIVE_GPU%%,*}\"
if [[ -z \"\$ACTIVE_GPU\" ]]; then
  ACTIVE_GPU=\"\${CUDA_VISIBLE_DEVICES:-}\"
  ACTIVE_GPU=\"\${ACTIVE_GPU%%,*}\"
fi
echo \"Using ACTIVE_GPU=\$ACTIVE_GPU for Omniverse renderer; cudaDevice=0 for PhysX/PyTorch\"

cd ${WORKDIR}

unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE

echo \"Running from: \$PWD\"
echo \"Running: ${CMD_Q}\"
exec ${CMD_Q}
"
