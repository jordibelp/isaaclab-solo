#!/usr/bin/env bash
set -euo pipefail

# Fast-sync the local Solo12 task trees and relevant training scripts to the
# cluster, preserving the same relative paths under ~/IsaacLab on the remote
# machine. Also syncs the local CaT extension needed by solo12_cat_laas.
# Optionally update the local SKRL fast PPO W&B run name before syncing.
# By default this script only syncs files. Pass --train to also submit the cluster job.
#
# The default path is intentionally lightweight:
#   1) one remote mkdir
#   2) one rsync for the Solo12 direct-task tree
#   3) one rsync for the Solo12 LAAS direct-task tree
#   4) one rsync for the manager-based Solo12 CaT LAAS task tree
#   5) one rsync for the Solo12 USD assets used by those tasks
#   6) one rsync for the CaT extension and training scripts
#   7) one batched rsync for extra training/support scripts
# Optional remote file summaries remain available with --verify.
#
# Usage:
#   ./push_solo12_to_cluster.sh
#   ./push_solo12_to_cluster.sh --train
#   ./push_solo12_to_cluster.sh --run-name "my run name"
#   ./push_solo12_to_cluster.sh --host jbeltran@10.4.26.33 --run-name "my run name" --train
#   ./push_solo12_to_cluster.sh --dry-run
#   ./push_solo12_to_cluster.sh --verify
#
# Default target:
#   jbeltran@10.4.26.33

REMOTE_HOST="jbeltran@10.4.26.33"
RUN_NAME=""
DRY_RUN=0
TRAIN=0
VERIFY=0

usage() {
  cat <<'EOF'
Usage: ./push_solo12_to_cluster.sh [--host USER@HOST] [--run-name NAME] [--train] [--dry-run] [--verify]

Options:
  --host USER@HOST  Remote SSH target. Default: jbeltran@10.4.26.33
  --run-name NAME   Update agents/skrl_ppo_cfg_fast.yaml W&B run name before syncing.
  --train           Submit ~/train_isaac.sbs after syncing.
  --dry-run         Do a non-mutating rsync dry-run; remote mkdir/install/train are printed only.
  --verify          Run remote file summaries after sync. Slower; off by default.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      if [[ $# -lt 2 ]]; then
        echo "Error: --run-name requires a value"
        exit 1
      fi
      RUN_NAME="$2"
      shift 2
      ;;
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Error: --host requires a value"
        exit 1
      fi
      REMOTE_HOST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --train)
      TRAIN=1
      shift
      ;;
    --verify)
      VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

LOCAL_ROOT="$HOME/IsaacLab"
REMOTE_ROOT='~/IsaacLab'
LOCAL_SOLO12_DIR="$LOCAL_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12"
REMOTE_SOLO12_DIR="$REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12"
LOCAL_SOLO12_LAAS_DIR="$LOCAL_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12_laas"
REMOTE_SOLO12_LAAS_DIR="$REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12_laas"
LOCAL_SOLO12_CAT_LAAS_DIR="$LOCAL_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/solo12_cat_laas"
REMOTE_SOLO12_CAT_LAAS_DIR="$REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/solo12_cat_laas"
LOCAL_SOLO12_ASSET_FILE="$LOCAL_ROOT/source/isaaclab_assets/data/Robots/Solo12/SoloFlat.usd"
REMOTE_SOLO12_ASSET_DIR="$REMOTE_ROOT/source/isaaclab_assets/data/Robots/Solo12"
LOCAL_SOLO12_LAAS_ASSET_DIR="$LOCAL_ROOT/source/isaaclab_assets/data/Robots/Solo12Laas"
REMOTE_SOLO12_LAAS_ASSET_DIR="$REMOTE_ROOT/source/isaaclab_assets/data/Robots/Solo12Laas"
LOCAL_CAT_ENVS_DIR="$HOME/constraints-as-terminations/exts/cat_envs"
REMOTE_CAT_ENVS_DIR='~/constraints-as-terminations/exts/cat_envs'
LOCAL_CAT_SCRIPTS_DIR="$HOME/constraints-as-terminations/scripts"
REMOTE_CAT_SCRIPTS_DIR='~/constraints-as-terminations/scripts'
LOCAL_AGENT_CFG="$LOCAL_SOLO12_DIR/agents/skrl_ppo_cfg.yaml"

EXTRA_SYNC_FILES=(
  "source/scripts/skrl/train.py"
  "source/scripts/skrl/helpers.py"
  "source/scripts/rsl_rl/train.py"
  "source/scripts/rsl_rl/continual_backprop.py"
  "source/scripts/rsl_rl/train_solo12_base_imu_dagger.py"
  "source/scripts/skrl/solo12_symmetry.py"
)

RSYNC_FILTERS=(
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=*.pyo"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.DS_Store"
)

SSH_CONTROL_PATH="${TMPDIR:-/tmp}/solo12_push_%C"
SSH_RSH="ssh -o ControlMaster=auto -o ControlPersist=120 -o ControlPath=${SSH_CONTROL_PATH}"

echo_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  echo_cmd "$@"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_remote() {
  local cmd="$1"
  echo "+ ssh $REMOTE_HOST $cmd"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    ssh -o ControlMaster=auto -o ControlPersist=120 -o ControlPath="$SSH_CONTROL_PATH" "$REMOTE_HOST" "$cmd"
  fi
}

run_rsync() {
  local args=("$@")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    args=(--dry-run "${args[@]}")
    echo_cmd rsync "${args[@]}"
    rsync "${args[@]}"
  else
    run_cmd rsync "${args[@]}"
  fi
}

cleanup() {
  if [[ -n "${EXTRA_LIST_FILE:-}" && -f "$EXTRA_LIST_FILE" ]]; then
    rm -f "$EXTRA_LIST_FILE"
  fi
}
trap cleanup EXIT

if [[ ! -d "$LOCAL_SOLO12_DIR" ]]; then
  echo "Error: local Solo12 directory not found: $LOCAL_SOLO12_DIR" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_SOLO12_LAAS_DIR" ]]; then
  echo "Error: local Solo12 LAAS directory not found: $LOCAL_SOLO12_LAAS_DIR" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_SOLO12_CAT_LAAS_DIR" ]]; then
  echo "Error: local Solo12 CaT LAAS directory not found: $LOCAL_SOLO12_CAT_LAAS_DIR" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_SOLO12_ASSET_FILE" ]]; then
  echo "Error: local Solo12 USD asset not found: $LOCAL_SOLO12_ASSET_FILE" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_SOLO12_LAAS_ASSET_DIR" ]]; then
  echo "Error: local Solo12 LAAS USD asset directory not found: $LOCAL_SOLO12_LAAS_ASSET_DIR" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_CAT_ENVS_DIR" ]]; then
  echo "Error: local cat_envs extension directory not found: $LOCAL_CAT_ENVS_DIR" >&2
  exit 1
fi

if [[ ! -d "$LOCAL_CAT_SCRIPTS_DIR" ]]; then
  echo "Error: local constraints-as-terminations scripts directory not found: $LOCAL_CAT_SCRIPTS_DIR" >&2
  exit 1
fi

for rel_path in "${EXTRA_SYNC_FILES[@]}"; do
  if [[ ! -f "$LOCAL_ROOT/$rel_path" ]]; then
    echo "Error: required file not found: $LOCAL_ROOT/$rel_path" >&2
    exit 1
  fi
done

EXTRA_LIST_FILE="$(mktemp)"
printf '%s\n' "${EXTRA_SYNC_FILES[@]}" > "$EXTRA_LIST_FILE"

SOLO12_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_SOLO12_DIR/"
  "$REMOTE_HOST:$REMOTE_SOLO12_DIR/"
)

SOLO12_LAAS_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_SOLO12_LAAS_DIR/"
  "$REMOTE_HOST:$REMOTE_SOLO12_LAAS_DIR/"
)

SOLO12_CAT_LAAS_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_SOLO12_CAT_LAAS_DIR/"
  "$REMOTE_HOST:$REMOTE_SOLO12_CAT_LAAS_DIR/"
)

SOLO12_ASSET_RSYNC_ARGS=(
  -az
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "$LOCAL_SOLO12_ASSET_FILE"
  "$REMOTE_HOST:$REMOTE_SOLO12_ASSET_DIR/"
)

SOLO12_LAAS_ASSETS_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_SOLO12_LAAS_ASSET_DIR/"
  "$REMOTE_HOST:$REMOTE_SOLO12_LAAS_ASSET_DIR/"
)

CAT_ENVS_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_CAT_ENVS_DIR/"
  "$REMOTE_HOST:$REMOTE_CAT_ENVS_DIR/"
)

CAT_SCRIPTS_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_CAT_SCRIPTS_DIR/"
  "$REMOTE_HOST:$REMOTE_CAT_SCRIPTS_DIR/"
)

EXTRA_RSYNC_ARGS=(
  -az
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  --files-from="$EXTRA_LIST_FILE"
  "$LOCAL_ROOT/"
  "$REMOTE_HOST:$REMOTE_ROOT/"
)

echo "Target host: $REMOTE_HOST"
echo "Local IsaacLab root: $LOCAL_ROOT"
echo "Remote IsaacLab root: $REMOTE_ROOT"
echo "Local Solo12 dir: $LOCAL_SOLO12_DIR"
echo "Remote Solo12 dir: $REMOTE_SOLO12_DIR"
echo "Local Solo12 LAAS dir: $LOCAL_SOLO12_LAAS_DIR"
echo "Remote Solo12 LAAS dir: $REMOTE_SOLO12_LAAS_DIR"
echo "Local Solo12 CaT LAAS dir: $LOCAL_SOLO12_CAT_LAAS_DIR"
echo "Remote Solo12 CaT LAAS dir: $REMOTE_SOLO12_CAT_LAAS_DIR"
echo "Local Solo12 USD asset: $LOCAL_SOLO12_ASSET_FILE"
echo "Remote Solo12 USD asset dir: $REMOTE_SOLO12_ASSET_DIR"
echo "Local Solo12 LAAS USD assets dir: $LOCAL_SOLO12_LAAS_ASSET_DIR"
echo "Remote Solo12 LAAS USD assets dir: $REMOTE_SOLO12_LAAS_ASSET_DIR"
echo "Local cat_envs dir: $LOCAL_CAT_ENVS_DIR"
echo "Remote cat_envs dir: $REMOTE_CAT_ENVS_DIR"
echo "Local CaT scripts dir: $LOCAL_CAT_SCRIPTS_DIR"
echo "Remote CaT scripts dir: $REMOTE_CAT_SCRIPTS_DIR"
if [[ -n "$RUN_NAME" ]]; then
  echo "Requested run name: $RUN_NAME"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Mode: dry-run"
fi
if [[ "$TRAIN" -eq 1 ]]; then
  echo "Mode: sync + train"
else
  echo "Mode: sync only"
fi
if [[ "$VERIFY" -eq 1 ]]; then
  echo "Verification: enabled"
else
  echo "Verification: skipped (use --verify for remote file summaries)"
fi
echo

if [[ -n "$RUN_NAME" ]]; then
  echo "[0/8] Updating local W&B run name in skrl_ppo_cfg_fast.yaml..."
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run: would set wandb_kwargs.name to \"$RUN_NAME\" in $LOCAL_AGENT_CFG"
  else
    python3 - <<'PY' "$LOCAL_AGENT_CFG" "$RUN_NAME"
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
run_name = sys.argv[2]
text = cfg_path.read_text()
lines = text.splitlines()
changed = False
in_wandb_kwargs = False
wandb_indent = None

for i, line in enumerate(lines):
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(' '))

    if stripped == 'wandb_kwargs:':
        in_wandb_kwargs = True
        wandb_indent = indent
        continue

    if in_wandb_kwargs:
        if stripped and indent <= wandb_indent:
            in_wandb_kwargs = False
        elif stripped.startswith('name:'):
            prefix = line[: len(line) - len(line.lstrip(' '))]
            lines[i] = f'{prefix}name: "{run_name}"'
            changed = True
            in_wandb_kwargs = False
            break

if not changed:
    raise SystemExit('Failed to find wandb_kwargs.name in skrl_ppo_cfg_fast.yaml')

cfg_path.write_text('\n'.join(lines) + '\n')
PY
  fi
  echo
fi

echo "[1/8] Ensuring remote IsaacLab, asset, and CaT directories exist..."
run_remote "mkdir -p $REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct $REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config $REMOTE_SOLO12_ASSET_DIR $REMOTE_SOLO12_LAAS_ASSET_DIR $REMOTE_ROOT/source/scripts/skrl $REMOTE_ROOT/source/scripts/rsl_rl ~/constraints-as-terminations/exts/cat_envs ~/constraints-as-terminations/scripts"
echo

echo "[2/8] Syncing Solo12 direct-task tree..."
run_rsync "${SOLO12_RSYNC_ARGS[@]}"
echo

echo "[3/8] Syncing Solo12 LAAS direct-task tree..."
run_rsync "${SOLO12_LAAS_RSYNC_ARGS[@]}"
echo

echo "[4/8] Syncing Solo12 CaT LAAS manager-task tree..."
run_rsync "${SOLO12_CAT_LAAS_RSYNC_ARGS[@]}"
echo

echo "[5/8] Syncing Solo12 USD assets..."
run_rsync "${SOLO12_ASSET_RSYNC_ARGS[@]}"
run_rsync "${SOLO12_LAAS_ASSETS_RSYNC_ARGS[@]}"
echo

echo "[6/8] Syncing cat_envs extension and CaT training scripts..."
run_rsync "${CAT_ENVS_RSYNC_ARGS[@]}"
run_rsync "${CAT_SCRIPTS_RSYNC_ARGS[@]}"
echo

echo "[7/8] Syncing training/support scripts in one batched rsync..."
run_rsync "${EXTRA_RSYNC_ARGS[@]}"
echo

if [[ "$VERIFY" -eq 1 ]]; then
  echo "Verifying remote file summary..."
  run_remote "find $REMOTE_SOLO12_DIR -maxdepth 3 -type f | sort && echo '--- solo12_laas ---' && find $REMOTE_SOLO12_LAAS_DIR -maxdepth 3 -type f | sort && echo '--- solo12_cat_laas ---' && find $REMOTE_SOLO12_CAT_LAAS_DIR -maxdepth 3 -type f | sort && echo '--- solo12 assets ---' && find $REMOTE_SOLO12_ASSET_DIR -maxdepth 1 -type f -name 'SoloFlat.usd' -print && echo '--- solo12_laas assets ---' && find $REMOTE_SOLO12_LAAS_ASSET_DIR -maxdepth 3 -type f | sort && echo '--- cat_envs import ---' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && python -c 'import cat_envs; print(cat_envs.__file__)' && echo '--- scripts ---' && find $REMOTE_ROOT/source/scripts -maxdepth 3 -type f | grep -E '/(skrl|rsl_rl)/(train\\.py|solo12_symmetry\\.py|helpers\\.py)$' | sort"
  echo
fi

if [[ "$TRAIN" -eq 1 ]]; then
  echo "[8/8] Submitting training job on the cluster..."
  run_remote "cd ~ && sbatch train_isaac.sbs"
  echo
  echo "Done. Solo12 task trees, USD assets, cat_envs, and train scripts synced to the cluster, then sbatch submitted."
else
  echo "[8/8] Skipping training job submission (pass --train to submit)."
  echo
  echo "Done. Solo12 task trees, USD assets, cat_envs, and train scripts synced to the cluster."
fi
