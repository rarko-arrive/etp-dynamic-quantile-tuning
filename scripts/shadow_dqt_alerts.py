"""Evaluate and optionally notify on shadow DQT alert conditions."""

from __future__ import annotations

import argparse
import sys

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import resolve_data_dir
from dqt.tuning.alerts import (
    AlertSeverity,
    alerts_to_json,
    evaluate_shadow_alerts,
    send_alerts,
    worst_severity,
)
from dqt.tuning.bakeoff import load_shadow_history


def main() -> int:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--webhook",
        default=None,
        help="Slack/Teams webhook (default: DQT_ALERT_WEBHOOK env)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="POST alerts to webhook when present",
    )
    parser.add_argument("--json", action="store_true", help="print alerts as JSON")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    history = load_shadow_history(data_dir)
    latest = None
    if not history.is_empty():
        latest = history.sort("run_at").tail(1).row(0, named=True)

    alerts = evaluate_shadow_alerts(history, latest=latest)
    if args.json:
        print(alerts_to_json(alerts))
    else:
        for alert in alerts:
            logger.warning("[{}] {}: {}", alert.severity.value, alert.code, alert.message)
        if not alerts:
            logger.info("no shadow alerts")

    if args.notify:
        send_alerts(alerts, latest=latest, webhook_url=args.webhook)

    sev = worst_severity(alerts)
    if sev == AlertSeverity.P0:
        return 1
    if sev == AlertSeverity.P1:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
