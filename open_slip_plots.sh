#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jordibelp/IsaacLab"

source /home/jordibelp/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

cd "$REPO_ROOT"
exec "$REPO_ROOT/isaaclab.sh" -p source/scripts/rsl_rl/slip_plots.py --app "$@"
