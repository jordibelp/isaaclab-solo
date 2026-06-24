#!/usr/bin/env bash
set -euo pipefail

# Fast-sync the local Solo12 race training files to the cluster, preserving the
# same relative paths under ~/IsaacLab on the remote machine.
#
# The old version opened one SSH/rsync connection per support file and always
# ran remote find summaries. This version keeps the default path lightweight:
#   1) one remote mkdir
#   2) one rsync for the Solo12 race task tree
#   3) one batched rsync for extra scripts/assets
# Optional verification remains available with --verify.
#
# Usage:
#   ./solo_race_push_files.sh
#   ./solo_race_push_files.sh --train
#   ./solo_race_push_files.sh --host jbeltran@10.4.26.33
#   ./solo_race_push_files.sh --dry-run
#   ./solo_race_push_files.sh --verify
#
# Default target:
#   jbeltran@10.4.26.33

usage() {
  cat <<'EOF'
Usage: ./solo_race_push_files.sh [--host USER@HOST] [--train] [--dry-run] [--verify]

Options:
  --host USER@HOST  Remote SSH target. Default: jbeltran@10.4.26.33
  --train           Submit ~/train_isaac.sbs after syncing.
  --dry-run         Do a non-mutating rsync dry-run; remote mkdir/train are printed only.
  --verify          Run remote file summaries after sync. Slower; off by default.
  -h, --help        Show this help.
EOF
}

REMOTE_HOST="jbeltran@10.4.26.33"
DRY_RUN=0
TRAIN=0
VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Error: --host requires a value" >&2
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
LOCAL_RACE_DIR="$LOCAL_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race"
REMOTE_RACE_DIR="$REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct/solo12_race"

EXTRA_SYNC_FILES=(
  "run_clean_env.sh"
  "source/isaaclab_assets/data/Robots/Solo12/solo_IMU_race_waypoints.usd"
  "source/isaaclab_assets/data/Robots/Solo12/solo_IMU_race_waypoints_simple.usd"
  "source/isaaclab_assets/data/Robots/Solo12/solo_IMU_race_waypoints_simple_zigzag.usd"
  "source/isaaclab_assets/data/Robots/Solo12/solo_IMU_race_waypoints_simple_zigzag_01.usd"
  "source/isaaclab_assets/data/Robots/Solo12/solo_IMU_race_waypoints_old.usd"
  "source/scripts/rsl_rl/train.py"
  "source/scripts/rsl_rl/continual_backprop.py"
  "source/scripts/rsl_rl/train_race_env_params_tcn_dagger.py"
  "source/scripts/skrl/helpers.py"
  "source/scripts/skrl/solo12_symmetry.py"
  "source/scripts/skrl/solo12_race_symmetry.py"
  "source/scripts/rsl_rl/export_solo12_race_scene.py"
  "source/scripts/rsl_rl/play_direct_race_0423.py"
  "source/scripts/rsl_rl/race_dagger_adapter_policy.py"
  "source/scripts/rsl_rl/solo_race_eval.py"
)

RSYNC_FILTERS=(
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=*.pyo"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.DS_Store"
)

# Reuse the SSH connection across the mkdir / rsync / optional train calls.
# The short persist window avoids leaving stale sockets around but removes the
# repeated handshake cost that made the previous per-file sync painful.
SSH_CONTROL_PATH="${TMPDIR:-/tmp}/solo_race_push_%C"
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

if [[ ! -d "$LOCAL_RACE_DIR" ]]; then
  echo "Error: local Solo12 race directory not found: $LOCAL_RACE_DIR" >&2
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

RACE_RSYNC_ARGS=(
  -az
  --delete
  --human-readable
  --info=stats1
  -e "$SSH_RSH"
  "${RSYNC_FILTERS[@]}"
  "$LOCAL_RACE_DIR/"
  "$REMOTE_HOST:$REMOTE_RACE_DIR/"
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
echo "Local Solo12 race dir: $LOCAL_RACE_DIR"
echo "Remote Solo12 race dir: $REMOTE_RACE_DIR"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Mode: dry-run"
elif [[ "$TRAIN" -eq 1 ]]; then
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

echo "[1/3] Ensuring remote IsaacLab root exists..."
run_remote "mkdir -p $REMOTE_ROOT/source/isaaclab_tasks/isaaclab_tasks/direct $REMOTE_ROOT/source/scripts $REMOTE_ROOT/source/isaaclab_assets/data/Robots/Solo12"
echo

echo "[2/3] Syncing Solo12 race task tree..."
run_rsync "${RACE_RSYNC_ARGS[@]}"
echo

echo "[3/3] Syncing extra race scripts/assets in one batched rsync..."
run_rsync "${EXTRA_RSYNC_ARGS[@]}"
echo

if [[ "$VERIFY" -eq 1 ]]; then
  echo "Verifying remote file summary..."
  run_remote "find $REMOTE_RACE_DIR -maxdepth 3 -type f | sort && echo '---' && find $REMOTE_ROOT/source/isaaclab_assets/data/Robots/Solo12 -maxdepth 1 -type f | grep -E 'solo_IMU_race_waypoints(_simple|_simple_zigzag|_simple_zigzag_01|_old)?\.usd$' | sort && echo '---' && find $REMOTE_ROOT/source/scripts -maxdepth 3 -type f | grep -E '/(rsl_rl|skrl)/(train\.py|continual_backprop\.py|train_race_env_params_tcn_dagger\.py|helpers\.py|export_solo12_race_scene\.py|play_direct_race_0423\.py|race_dagger_adapter_policy\.py|solo_race_eval\.py|solo12_symmetry\.py|solo12_race_symmetry\.py)$' | sort && echo '---' && ls -l $REMOTE_ROOT/run_clean_env.sh"
  echo
fi

if [[ "$TRAIN" -eq 1 ]]; then
  echo "Submitting training job on the cluster..."
  run_remote "cd ~ && sbatch train_isaac.sbs"
  echo
  echo "Done. Solo12 race files synced to the cluster, then sbatch submitted."
else
  echo "Done. Solo12 race files synced to the cluster."
fi
