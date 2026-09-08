"""Run consecutive capped shadow days for bake-off history (no Snowflake).

Usage:
  DQT_DATA_DIR=data uv run python scripts/run_shadow_bakeoff_loop.py --days 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from loguru import logger

from dqt import resolve_data_dir
from dqt.sarima_publish import next_weekday
from dqt.shadow_dqt import ShadowDQT
from dqt.tuning.bakeoff import assess_bakeoff_slos, write_bakeoff_report
from tests.fixtures_dqt import write_smoke_layer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--bootstrap", action="store_true", help="build synthetic layer first")
    args = parser.parse_args()

    layer = resolve_data_dir(args.data_dir or ".dev/bakeoff_smoke")
    if args.bootstrap or not (layer / "panel_loads.parquet").exists():
        logger.info("bootstrapping synthetic layer at {}", layer)
        write_smoke_layer(layer)

    shadow = ShadowDQT(layer)
    scored = date(2025, 6, 6)
    while scored.weekday() >= 5:
        scored += timedelta(days=1)
    ok = 0
    for i in range(args.days):
        target = next_weekday(scored + timedelta(days=7))
        try:
            shadow.run_daily(
                as_of=target,
                scored_date=scored,
                write_snowflake=False,
                min_n=30,
                publish_mode="cap",
                max_weekly_move=0.05,
                dual_write_audit=True,
            )
            ok += 1
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("day {} scored={} failed: {}", i + 1, scored, exc)
        scored += timedelta(days=1)
        while scored.weekday() >= 5:
            scored += timedelta(days=1)

    logger.info("completed {}/{} shadow runs", ok, args.days)

    history = shadow.history_path
    hist_df = __import__("polars").read_parquet(history) if history.exists() else None
    if hist_df is None or hist_df.is_empty():
        logger.error("no shadow history written")
        return 1

    report = assess_bakeoff_slos(hist_df, layer, required_weekdays=args.days)
    md_path, _ = write_bakeoff_report(report, layer)
    logger.info("bake-off → {} (pass={})", md_path, report.passed)
    return 0 if report.passed else 2


if __name__ == "__main__":
    sys.exit(main())
