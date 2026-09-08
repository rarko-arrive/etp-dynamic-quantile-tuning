#!/usr/bin/env bash
# Populate shadow_eval_history and assess bake-off SLOs.
#
#   DQT_DATA_DIR=data/bakeoff_smoke ./scripts/run_shadow_bakeoff_loop.sh 30

set -euo pipefail
cd "$(dirname "$0")/.."
DAYS="${1:-30}"
LAYER="${DQT_DATA_DIR:-.dev/bakeoff_smoke}"
export DQT_DATA_DIR="$LAYER"
uv run python scripts/run_shadow_bakeoff_loop.py --data-dir "$LAYER" --days "$DAYS" --bootstrap
