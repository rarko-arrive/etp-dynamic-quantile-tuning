"""Empirical guardrail fire-rates and threshold calibration for DQT."""

from __future__ import annotations

import argparse
import sys

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import resolve_data_dir
from dqt.tuning.assess import assess_guardrails, write_assessment


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
        "--quick",
        action="store_true",
        help="sample last 60 tail proposal days (~2 min)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    logger.info("assessing guardrails under {}", data_dir)
    result = assess_guardrails(data_dir, quick=args.quick)
    pq_path, md_path = write_assessment(result, data_dir=data_dir)
    logger.info("wrote {} and {}", pq_path, md_path)
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
