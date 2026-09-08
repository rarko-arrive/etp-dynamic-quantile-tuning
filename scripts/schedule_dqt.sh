#!/usr/bin/env bash
# DQT scheduled job entrypoint — daily or weekly.
#
# Usage:
#   ./scripts/schedule_dqt.sh daily
#   ./scripts/schedule_dqt.sh weekly
#
# Requires DQT_DATA_DIR and Snowflake creds in environment (or .env).

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODE="${1:-daily}"
export DQT_DATA_DIR="${DQT_DATA_DIR:-data}"

case "$MODE" in
  daily)
    echo "=== DQT daily schedule (DQT_DATA_DIR=$DQT_DATA_DIR) ==="
    make features
    make hybrid
    make sarima-wf APPEND=1
    make shadow-dqt \
      PUBLISH_MODE="${DQT_PUBLISH_MODE:-cap}" \
      MAX_WEEKLY_MOVE="${DQT_MAX_WEEKLY_MOVE:-0.05}" \
      $(if [[ "${SKIP_SF:-}" == "1" ]]; then echo SKIP_SF=1; fi) \
      $(if [[ "${ALLOW_PROD:-}" == "1" ]]; then echo ALLOW_PROD=1; fi) \
      $(if [[ "${DUAL_AUDIT:-}" == "1" ]]; then echo DUAL_AUDIT=1; fi)
    make shadow-alerts || true
    ;;
  weekly)
    echo "=== DQT weekly schedule ==="
    make sarima-wf APPEND=1
    make guardrails-assess QUICK=1
    make shadow-bakeoff || true
    ;;
  *)
    echo "Usage: $0 {daily|weekly}" >&2
    exit 1
    ;;
esac

echo "=== schedule complete ($MODE) ==="
