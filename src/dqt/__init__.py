"""DQT quantile-attainment analysis package."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .logging_config import configure_logging

# Quiet by default (loguru ships at DEBUG). Override: DQT_LOG_LEVEL=DEBUG.
configure_logging()

CACHE_HOURS = 24 * 3


def is_fresh(path: Path, *, hours: float = CACHE_HOURS) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < hours * 3600


def get_dt_local() -> datetime:
    """Wall-clock datetime in America/Chicago."""
    return datetime.now(tz=ZoneInfo("America/Chicago"))


def repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: cwd) until pyproject.toml is found."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("pyproject.toml not found above " + str(p))


def resolve_data_dir(
    data_dir: str | Path | None = None,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve the parquet layer.

    Order: explicit ``data_dir`` → ``DQT_DATA_DIR`` → ``data``.
    Empty / whitespace env values are treated as unset. ``~`` and ``$HOME``
    are expanded. Absolute paths are used as-is (off-repo Azure ML layers);
    relative paths are joined to ``root`` (default: :func:`repo_root`).

    Does not create the directory — writers mkdir when they write.
    """
    if data_dir is None:
        raw = (os.environ.get("DQT_DATA_DIR") or "").strip() or "data"
    else:
        raw = os.fspath(data_dir)
    expanded = os.path.expanduser(os.path.expandvars(str(raw).strip().strip("\"'")))
    p = Path(expanded)
    if p.is_absolute():
        return p
    return (root or repo_root()) / p


def display_path(path: Path, *, root: Path | None = None) -> str:
    """Repo-relative string when ``path`` is inside the repo; else as-is."""
    path = Path(path)
    base = (root or repo_root()).resolve()
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path)


def features_schema_path(*, root: Path | None = None) -> Path:
    return (root or repo_root()) / "documentation" / "assets" / "features.schema.json"


def schema_from_dtype_names(mapping: dict[str, str]) -> pl.Schema:
    """``{\"loadnumber\": \"Int32\"}`` → :class:`polars.Schema`."""
    out: dict[str, pl.DataType] = {}
    for name, dtype in mapping.items():
        dt = getattr(pl, dtype, None)
        if dt is None:
            raise ValueError(f"unknown polars dtype {dtype!r} for column {name!r}")
        out[name] = dt
    return pl.Schema(out)


def features_schema(*, root: Path | None = None) -> pl.Schema:
    """Load ``documentation/assets/features.schema.json`` as a Polars schema."""
    path = features_schema_path(root=root)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} is not a non-empty name→dtype object")
    return schema_from_dtype_names(raw)


def parse_date_col(frame: pl.DataFrame, col: str) -> pl.Expr:
    """Restore a Date expr from parquet (string / datetime / date).

    Feature SQL already computed the calendar date — this is dtype restore,
    not a transform.
    """
    if col not in frame.columns:
        raise ValueError(f"missing date column {col!r}")
    dtype = frame.schema[col]
    if dtype == pl.Date:
        return pl.col(col)
    if isinstance(dtype, pl.Datetime):
        return pl.col(col).dt.date()
    return pl.col(col).str.to_date()


def stringify_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Cast native Date/Datetime columns to ``%Y-%m-%d[ %H:%M:%S]`` strings.

    Snowflake's Arrow fetch returns real date/datetime dtypes, but every
    downstream reader (`src/dqt/panel.py`, notebooks) parses these
    columns with `.str.to_date(...)`.
    """
    exprs = []
    for name, dtype in df.schema.items():
        if dtype == pl.Date:
            exprs.append(pl.col(name).dt.strftime("%Y-%m-%d"))
        elif isinstance(dtype, pl.Datetime):
            exprs.append(pl.col(name).dt.strftime("%Y-%m-%d %H:%M:%S"))
    return df.with_columns(exprs) if exprs else df
