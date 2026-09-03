"""Daily shadow: score yesterday, propose tomorrow, audit + optional SF."""

from __future__ import annotations

import argparse
import subprocess
import sys

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import repo_root, resolve_data_dir
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
        help="parquet layer (default: DQT_DATA_DIR or data/current)",
    )
    parser.add_argument(
        "--scored-date",
        default=None,
        help="override yesterday eval date (ISO)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="override proposal valid_date (ISO)",
    )
    parser.add_argument(
        "--min-lift-pct",
        type=float,
        default=0.0,
        help="minimum lift vs naive (%%) for kill-switch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve Snowflake target only",
    )
    parser.add_argument(
        "--skip-sf",
        action="store_true",
        help="parquet audit only; no Snowflake append",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="run `make sarima-wf APPEND=1` before shadow loop",
    )
    parser.add_argument(
        "--settlement-lag",
        type=int,
        default=3,
        help="lag days for last complete scored booking day",
    )
    parser.add_argument(
        "--max-weekly-move",
        type=float,
        default=0.05,
        help="max per-level move (fraction, default 0.05 = 5pp)",
    )
    parser.add_argument(
        "--publish-mode",
        choices=("block", "cap", "full"),
        default="block",
        help="block=fail closed; cap=clip proposal to prior±move; full=no move gate",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    if args.refresh:
        logger.info("refreshing sarima walk-forward (APPEND=1)")
        subprocess.run(
            ["make", "sarima-wf", "APPEND=1"],
            cwd=REPO,
            check=True,
            env={**dict(__import__("os").environ), "DQT_DATA_DIR": str(data_dir)},
        )

    shadow = ShadowDQT(data_dir, settlement_lag_days=args.settlement_lag)
    report = shadow.run_daily(
        as_of=args.as_of,
        scored_date=args.scored_date,
        dry_run=args.dry_run,
        write_snowflake=not args.skip_sf,
        min_lift_vs_naive_pct=args.min_lift_pct,
        allow_prod=False,
        max_weekly_move=args.max_weekly_move,
        publish_mode=args.publish_mode,
    )
    logger.info("shadow report:\n{}", report.to_frame())
    return 0 if report.kill_switch_passed else 2


if __name__ == "__main__":
    sys.exit(main())
