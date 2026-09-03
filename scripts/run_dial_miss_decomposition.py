"""Run DQT dial miss decomposition pipeline.

Example
-------
    uv run python scripts/run_dial_miss_decomposition.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

load_dotenv(find_dotenv())

from dqt import display_path, repo_root, resolve_data_dir
from dqt.dial_miss.decomposition import run_dial_miss_decomposition


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--phases",
        default="0,1,2,3,4,5,6,7",
        help="comma-separated phase ids (0=gates, 1-7=analysis+readout)",
    )
    args = parser.parse_args(argv)

    repo = repo_root()
    data_dir = resolve_data_dir(args.data_dir, root=repo)
    phases = tuple(p.strip() for p in args.phases.split(",") if p.strip())

    logger.info("Dial miss decomposition data_dir={} phases={}", data_dir, phases)
    result = run_dial_miss_decomposition(data_dir=data_dir, repo_root=repo, phases=phases)
    out = Path(result["out_dir"])
    logger.info("n_loads={:,}", result["n_loads"])
    logger.info("artifacts → {}", display_path(out, root=repo))
    logger.info("hypotheses: {}", result["manifest"].get("hypotheses"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
