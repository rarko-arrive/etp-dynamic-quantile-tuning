"""Evaluate shadow bake-off SLOs on shadow_eval_history.parquet."""

from __future__ import annotations

import argparse
import sys

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import resolve_data_dir
from dqt.tuning.bakeoff import (
    assess_bakeoff_slos,
    load_shadow_history,
    write_bakeoff_report,
)


def main() -> int:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--required-weekdays",
        type=int,
        default=30,
        help="minimum weekdays for bake-off pass",
    )
    parser.add_argument(
        "--min-weekdays",
        type=int,
        default=None,
        help="alias for --required-weekdays (testing)",
    )
    args = parser.parse_args()
    required = args.min_weekdays if args.min_weekdays is not None else args.required_weekdays

    data_dir = resolve_data_dir(args.data_dir)
    history = load_shadow_history(data_dir)
    report = assess_bakeoff_slos(
        history,
        data_dir,
        required_weekdays=required,
    )
    md_path, pq_path = write_bakeoff_report(report, data_dir)
    logger.info("bake-off assessment → {} {}", md_path, pq_path)
    for slo in report.slos:
        status = "PASS" if slo.passed else "FAIL"
        logger.info("[{}] {} — {}", status, slo.name, slo.detail)

    if report.n_weekdays < required:
        logger.warning(
            "insufficient shadow history ({}/{} weekdays) — run daily "
            "`make shadow-dqt PUBLISH_MODE=cap` until {} weekdays accumulate",
            report.n_weekdays,
            required,
            required,
        )
        return 0

    if not report.passed:
        logger.warning(
            "bake-off SLOs not met ({}/{} weekdays)",
            report.n_weekdays,
            report.required_weekdays,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
