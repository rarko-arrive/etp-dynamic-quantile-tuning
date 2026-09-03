"""Build $(DQT_DATA_DIR)/hybrid_quantiles.parquet: dial + per-group conformal offset.

    make hybrid              # skip if the parquet already exists
    make hybrid FORCE=1      # rebuild
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import display_path, repo_root, resolve_data_dir
from dqt.hybrid import HybridConfig, hybrid_quantiles
from dqt.panel import build_panel

REPO = repo_root()


def main() -> int:
    load_dotenv(find_dotenv())
    defaults = HybridConfig()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--group-cols", nargs="+", default=list(defaults.group_cols))
    parser.add_argument("--levels", nargs="+", type=float, default=None)
    parser.add_argument("--cal-window-days", type=int, default=defaults.cal_window_days)
    parser.add_argument("--min-n", type=int, default=defaults.min_n)
    parser.add_argument(
        "--max-weekly-delta",
        type=float,
        default=None,
        help="cap week-over-week |delta| moves per cell; unset = unenforced",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="parquet layer (default: DQT_DATA_DIR or data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output parquet (default: <data-dir>/hybrid_quantiles.parquet)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if output already exists"
    )
    args = parser.parse_args()

    layer = resolve_data_dir(args.data_dir, root=REPO)
    if args.output:
        out = resolve_data_dir(Path(args.output).parent, root=REPO) / Path(args.output).name
    else:
        out = layer / "hybrid_quantiles.parquet"
    if out.exists() and not args.force:
        logger.info(
            "{} exists — skipping (--force to rebuild)",
            display_path(out, root=REPO),
        )
        return 0

    kwargs = {
        "group_cols": tuple(args.group_cols),
        "cal_window_days": args.cal_window_days,
        "min_n": args.min_n,
        "max_weekly_delta": args.max_weekly_delta,
        "output_path": display_path(out, root=REPO),
    }
    if args.levels is not None:
        kwargs["levels"] = tuple(args.levels)
    config = HybridConfig(**kwargs)

    panel = build_panel(REPO, force=args.force, data_dir=layer)
    result = hybrid_quantiles(panel, config, REPO, data_dir=layer)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    result.write_parquet(tmp)
    tmp.replace(out)
    logger.info(
        "wrote {} ({} rows, {:.0f} MB) from {}",
        display_path(out, root=REPO),
        result.height,
        out.stat().st_size / 1e6,
        display_path(layer, root=REPO),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
