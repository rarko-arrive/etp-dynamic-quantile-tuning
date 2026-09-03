"""Standalone SARIMA dial publish: kill-switch + next-weekday schedule row.

Usage::

    uv run python scripts/publish_sarima_dial.py
    make sarima-publish
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import get_dt_local, repo_root, resolve_data_dir
from dqt.model import GlobalDQT
from dqt.sarima_dial import write_parquet_atomic
from dqt.sarima_publish import (
    evaluate_kill_switch,
    next_weekday,
    publish_schedule_row,
)
from dqt.shadow_dqt import ShadowDQT

REPO = repo_root()


def main() -> int:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="parquet layer (default: DQT_DATA_DIR or data)",
    )
    parser.add_argument(
        "--min-lift-pct",
        type=float,
        default=0.0,
        help="minimum lift vs naive (%%) for kill-switch",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="proposal valid_date (default: next Chicago weekday)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve Snowflake target only; do not append",
    )
    parser.add_argument(
        "--skip-sf",
        action="store_true",
        help="never write Snowflake (staging + schedule parquet only)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    dqt = GlobalDQT(data_dir)
    wf_path = dqt.sarima_wf_path
    if not wf_path.exists():
        logger.error(
            "missing {}; run `make sarima-wf APPEND=1` first",
            wf_path.relative_to(REPO),
        )
        return 1

    today = get_dt_local().date()
    target = (
        date.fromisoformat(args.as_of[:10]) if args.as_of else next_weekday(today)
    )

    import polars as pl

    dial = pl.read_parquet(wf_path)
    kill = evaluate_kill_switch(
        dial,
        target=target,
        min_lift_vs_naive_pct=args.min_lift_pct,
    )
    shadow = ShadowDQT(data_dir, global_dqt=dqt)
    source = "SARIMA_tail" if kill.passed else "shipped_alt_50"
    proposal = shadow.propose_global(target, source=source)

    out_dir = data_dir / "shadow"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = out_dir / "sarima_publish_schedule.parquet"
    row = publish_schedule_row(
        valid_date=target,
        alts=proposal.alts,
        kill=kill,
        proposal_source=source,
    )
    if schedule_path.exists():
        sched = pl.concat(
            [pl.read_parquet(schedule_path), row],
            how="diagonal_relaxed",
        )
    else:
        sched = row
    write_parquet_atomic(sched, schedule_path)
    dqt._write_parquet_audit(proposal)

    logger.info(
        "target={} source={} kill_passed={} gates_passed={} alt_50={:.4f}",
        target,
        source,
        kill.passed,
        proposal.passed,
        proposal.alt_50,
    )
    if kill.reason:
        logger.warning("kill-switch: {}", kill.reason)

    if kill.passed and proposal.passed and not args.skip_sf:
        dest = dqt.publish(
            proposal,
            target="snowflake",
            dry_run=args.dry_run,
            allow_prod=False,
            also_parquet=False,
        )
        logger.info("snowflake target → {}", dest.fqn)

    return 0 if kill.passed else 2


if __name__ == "__main__":
    sys.exit(main())
