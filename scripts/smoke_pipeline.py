"""Pipeline smoke: synthetic fixtures → shadow loop (no Snowflake)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import date
from pathlib import Path

from loguru import logger

from dqt.tuning.alerts import AlertSeverity, evaluate_shadow_alerts, worst_severity
from tests.fixtures_dqt import write_smoke_layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="optional persistent layer (default: temp dir)",
    )
    args = parser.parse_args()

    if args.data_dir:
        layer = Path(args.data_dir)
        layer.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        layer = Path(tempfile.mkdtemp(prefix="dqt_smoke_"))
        cleanup = True

    logger.info("smoke layer → {}", layer)
    shadow = write_smoke_layer(layer)
    report = shadow.run_daily(
        as_of=date(2025, 6, 13),
        scored_date=date(2025, 6, 6),
        write_snowflake=False,
        min_n=30,
        publish_mode="cap",
        max_weekly_move=0.05,
    )
    assert report.n_loads > 0
    assert report.kill_switch_passed is True
    assert shadow.history_path.exists()

    hist = shadow.history_path
    alerts = evaluate_shadow_alerts(
        __import__("polars").read_parquet(hist),
        latest=report.to_frame().row(0, named=True),
    )
    sev = worst_severity(alerts)
    if sev in (AlertSeverity.P0, AlertSeverity.P1):
        logger.error("smoke alerts: {}", alerts)
        return 1

    logger.info("smoke OK: n_loads={} tail_att50={}", report.n_loads, report.tail_att50)
    if cleanup:
        logger.info("ephemeral layer (not removed): {}", layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
