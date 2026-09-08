from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def query_sf(sql: str | Path, **kwargs: Any) -> pl.DataFrame:
    raise RuntimeError(
        "arriveds CI stub: query_sf is unavailable in GitHub Actions. "
        "Install real arrive-ds locally for Snowflake workflows."
    )


def create_table(
    df: pl.DataFrame,
    table_name: str,
    *,
    database: str | None = None,
    schema: str | None = None,
    **kwargs: Any,
) -> int:
    return int(df.height)
