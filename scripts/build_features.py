"""Build $(DQT_DATA_DIR)/features.parquet from SQL/dqt/features.sql.

    make features              # skip if the parquet already exists
    make features FORCE=1      # rebuild
    make smoke                 # 14-day slice → data/current

Also writes ``documentation/assets/dqt-features.json`` (catalog) and
``documentation/assets/features.schema.json`` (name→dtype, loadable as
``pl.Schema`` via :func:`dqt.features_schema`).
"""

from __future__ import annotations

import argparse
import json
import os

import polars as pl
from arriveds.snowflake import query_sf
from dotenv import find_dotenv, load_dotenv
from loguru import logger

load_dotenv(find_dotenv())

from dqt import (
    display_path,
    features_schema_path,
    get_dt_local,
    repo_root,
    resolve_data_dir,
    stringify_dates,
)
from dqt.score.constants import DATE_COL, ID_COL, TIME_COL

REPO = repo_root()
FEATURES_SQL = REPO / "SQL" / "dqt" / "features.sql"
CATALOG_PATH = REPO / "documentation" / "assets" / "dqt-features.json"
SCHEMA_PATH = features_schema_path(root=REPO)

logger.add(
    REPO / "Logs" / "build_features.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="5 MB",
    retention=5,
)


def write_features_schema(schema: pl.Schema) -> None:
    """Write the human catalog and the name→dtype object used as ``pl.Schema``."""
    mapping = {name: str(dtype) for name, dtype in schema.items()}
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(mapping, indent=2) + "\n")
    CATALOG_PATH.write_text(
        json.dumps(
            {
                "keys": {"id": ID_COL, "time": TIME_COL, "date": DATE_COL},
                "n_cols": len(schema),
                "columns": [
                    {"name": name, "dtype": dtype} for name, dtype in mapping.items()
                ],
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        "wrote {} and {}",
        display_path(SCHEMA_PATH, root=REPO),
        display_path(CATALOG_PATH, root=REPO),
    )


def fetch_features_from_sql(start_date: str, end_date: str) -> pl.DataFrame:
    return stringify_dates(
        query_sf(
            FEATURES_SQL,
            params={"start_date": start_date, "end_date": end_date},
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if features.parquet exists"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="parquet layer (default: DQT_DATA_DIR or data)",
    )
    parser.add_argument(
        "--start",
        default=os.environ.get("START_DATE", "2024-07-01"),
        help="inclusive booked_on_cst start (default: START_DATE)",
    )
    parser.add_argument(
        "--end",
        default=get_dt_local().strftime("%Y-%m-%d"),
        help="exclusive booked_on_cst end (default: today, Chicago)",
    )
    args = parser.parse_args(argv)

    layer = resolve_data_dir(args.data_dir, root=REPO)
    out = layer / "features.parquet"
    if out.exists() and not args.force:
        logger.info(
            "{} exists — skipping (--force to rebuild)",
            display_path(out, root=REPO),
        )
        write_features_schema(pl.scan_parquet(out).collect_schema())
        return 0

    logger.info(
        "fetching features → {}  [{}, {})",
        display_path(out, root=REPO),
        args.start,
        args.end,
    )
    features = fetch_features_from_sql(start_date=args.start, end_date=args.end)
    logger.info("fetched: {:,} rows x {:,} cols", features.height, features.width)

    required = (ID_COL, TIME_COL, DATE_COL)
    missing = [c for c in required if c not in features.columns]
    if missing:
        raise SystemExit(
            f"features.sql did not emit configured keys {missing} "
            f"(ID_COL={ID_COL} TIME_COL={TIME_COL} DATE_COL={DATE_COL})"
        )
    logger.info(
        "keys ID_COL={}  TIME_COL={}  DATE_COL={}",
        ID_COL,
        TIME_COL,
        DATE_COL,
    )

    if "origin_lat" in features.columns:
        match_rate = 1 - features["origin_lat"].null_count() / max(features.height, 1)
        logger.info("matched coordinates for {:.1%} of loads", match_rate)

    tmp = out.with_suffix(".tmp.parquet")
    out.parent.mkdir(exist_ok=True, parents=True)
    features.write_parquet(tmp)
    tmp.replace(out)
    logger.info(
        "wrote {}: {:,} rows x {:,} cols",
        display_path(out, root=REPO),
        features.height,
        features.width,
    )
    write_features_schema(features.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
