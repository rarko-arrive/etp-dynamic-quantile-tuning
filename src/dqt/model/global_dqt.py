"""Scientist-facing facade for the deployed global DQT dial.

Reads production ``alt_*`` schedules from Snowflake, applies them to
Weatherman curves, spot-checks loads against booked ETP, evaluates coverage,
and proposes candidate dials from local research artifacts.

Snowflake **writes** go to a configurable ``DATABASE.SCHEMA.TABLE`` (defaults:
``SNOWFLAKE_DATABASE`` / ``SNOWFLAKE_SCHEMA`` / ``DQT_DIAL_TABLE``) — typically
``DATA_SCIENCE_WORKSPACE.RARKO.…`` for personal-schema testing. Production
``DATA_SCIENCE.ETP_DYNAMIC_QUANTILE_TUNING.HISTORICAL_ETP_DYNAMIC_QUANTILES``
requires an explicit ``allow_prod=True`` gate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from arriveds.snowflake import SnowflakeConfig, create_table, query_sf
from loguru import logger

from dqt import get_dt_local, is_fresh, parse_date_col, repo_root, resolve_data_dir
from dqt.conformal import quantile_at, walk_forward_alts
from dqt.panel import ALT_COLS, ETP_COLS, GRID, WEATHERMAN_COLS, implied_percentile
from dqt.sarima_dial import (
    QUOTE_CLIP_DEFAULT,
    global_alts_from_tail,
    rigid_quote_alts,
    shipped_alt_col,
)
from dqt.score.constants import COST_COL, DATE_COL, ID_COL

ALT_PERCENTILES_SQL = (
    repo_root() / "SQL" / "ds-monitoring-queries" / "dqt_alt_percentiles.sql"
)

ProposeMethod = Literal["shift", "sarima", "sarima_tail", "conformal_global", "identity"]
PublishTarget = Literal["parquet", "staging", "snowflake"]
IfExists = Literal["fail", "replace", "append"]

_DIAL_CACHE_NAME = "dqt_alt_percentiles.parquet"
_FEATURES_NAME = "features.parquet"
_SARIMA_WF_NAME = "sarima_dial_walkforward.parquet"
_DIAL_APPLIED_PREFIX = "dial_"

# Gate defaults (fail closed on oversized week-over-week moves).
_DEFAULT_MAX_WEEKLY_MOVE = 0.05  # 5 pp
_DEFAULT_CLIP = QUOTE_CLIP_DEFAULT  # (0.05, 0.95)

# Prod dial table — blocked unless publish(..., allow_prod=True).
PROD_DIAL_DATABASE = "DATA_SCIENCE"
PROD_DIAL_SCHEMA = "ETP_DYNAMIC_QUANTILE_TUNING"
PROD_DIAL_TABLE = "HISTORICAL_ETP_DYNAMIC_QUANTILES"
# Personal-schema / workspace default table name (same leaf as prod).
DEFAULT_DIAL_TABLE = PROD_DIAL_TABLE
_DIAL_TABLE_ENV = "DQT_DIAL_TABLE"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _parse_day(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _alts_from_row(row: Mapping[str, Any], *, columns: Sequence[str] = ALT_COLS) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in row or row[col] is None:
            raise KeyError(f"missing dial column {col!r}")
        out[col] = float(row[col])
    return out


@dataclass(frozen=True, slots=True)
class DialSchedule:
    """Frozen global dial for one ``valid_date`` (fractions ``alt_5``…``alt_95``)."""

    valid_date: date
    alts: dict[str, float]
    snowflakeupdatedon: date | datetime | None = None
    alpha: float | None = None
    time_window: str | None = None

    def __post_init__(self) -> None:
        missing = [c for c in ALT_COLS if c not in self.alts]
        if missing:
            raise ValueError(f"DialSchedule missing {missing[:4]}{'…' if len(missing) > 4 else ''}")

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> DialSchedule:
        """Build from a history / features row (Polars named dict OK)."""
        vd = row.get("valid_date", row.get("booked_on_date", row.get(DATE_COL)))
        if vd is None:
            raise ValueError("DialSchedule.from_row needs valid_date / booked_on_date / date")
        meta_ts = row.get("snowflakeupdatedon")
        alpha = row.get("alpha")
        tw = row.get("time_window")
        return cls(
            valid_date=_parse_day(vd),
            alts=_alts_from_row(row),
            snowflakeupdatedon=meta_ts,
            alpha=None if alpha is None else float(alpha),
            time_window=None if tw is None else str(tw),
        )

    def level(self, q: float) -> float:
        """Dial fraction at nominal level ``q`` (e.g. ``0.5`` → ``alt_50``)."""
        return float(self.alts[shipped_alt_col(q)])

    @property
    def alt_50(self) -> float:
        return self.level(0.5)

    def as_array(self, levels: Sequence[float] | np.ndarray = GRID) -> np.ndarray:
        """``(n_levels,)`` dial fractions aligned to ``levels``."""
        return np.asarray([self.level(float(a)) for a in levels], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_date": self.valid_date,
            **self.alts,
            "snowflakeupdatedon": self.snowflakeupdatedon,
            "alpha": self.alpha,
            "time_window": self.time_window,
        }

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame([self.to_dict()])


@dataclass(frozen=True, slots=True)
class DialProposal(DialSchedule):
    """Candidate dial for D+1 (or ``as_of``) plus method metadata and gates."""

    method: ProposeMethod = "identity"
    generated_at: datetime = field(default_factory=get_dt_local)
    gates: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.gates.get("passed", False))


@dataclass(frozen=True, slots=True)
class SpotCheck:
    """Per-load QA: shipped dial vs dial⊗Weatherman $ vs booked ETP ``p*``."""

    loadnumber: int
    booked_on_date: date
    alt: dict[str, float]
    weatherman: dict[str, float]
    dial_applied: dict[str, float]
    etp_booked: dict[str, float]
    cost: float | None = None
    wm_r: float | None = None
    etp_r: float | None = None

    @property
    def alt_50(self) -> float:
        return float(self.alt["alt_50"])

    @property
    def dial_applied_50(self) -> float:
        return float(self.dial_applied["dial_50"])

    @property
    def p50(self) -> float:
        return float(self.etp_booked["p50"])

    def delta_dollars(self) -> dict[str, float]:
        """``dial_applied_$ − etp_booked_$`` per grid level (keyed ``d5``…``d95``)."""
        out: dict[str, float] = {}
        for a in GRID:
            lab = int(round(float(a) * 100))
            dial_key = f"dial_{lab}"
            etp_key = f"p{lab:02d}"
            out[f"d{lab}"] = float(self.dial_applied[dial_key] - self.etp_booked[etp_key])
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "loadnumber": self.loadnumber,
            "booked_on_date": self.booked_on_date,
            "alt_50": self.alt_50,
            "dial_applied_50": self.dial_applied_50,
            "p50": self.p50,
            "cost": self.cost,
            "wm_r": self.wm_r,
            "etp_r": self.etp_r,
            "delta_50": self.dial_applied_50 - self.p50,
        }

    def to_frame(self, *, long: bool = False) -> pl.DataFrame:
        """Wide one-row table, or long ``(level, series, value)`` for plotting."""
        if not long:
            row: dict[str, Any] = {
                "loadnumber": self.loadnumber,
                "booked_on_date": self.booked_on_date,
                "cost": self.cost,
                "wm_r": self.wm_r,
                "etp_r": self.etp_r,
                **self.alt,
                **self.weatherman,
                **self.dial_applied,
                **self.etp_booked,
                **{f"delta_{k}": v for k, v in self.delta_dollars().items()},
            }
            return pl.DataFrame([row])

        rows: list[dict[str, Any]] = []
        for a in GRID:
            lab = int(round(float(a) * 100))
            rows.append(
                {
                    "loadnumber": self.loadnumber,
                    "booked_on_date": self.booked_on_date,
                    "level": float(a),
                    "alt": self.alt[f"alt_{lab}"],
                    "weatherman": self.weatherman[f"knn_{lab}"],
                    "dial_applied": self.dial_applied[f"dial_{lab}"],
                    "etp_booked": self.etp_booked[f"p{lab:02d}"],
                    "delta": (
                        self.dial_applied[f"dial_{lab}"]
                        - self.etp_booked[f"p{lab:02d}"]
                    ),
                }
            )
        return pl.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class DialWriteTarget:
    """Resolved ``DATABASE.SCHEMA.TABLE`` for a dial publish."""

    database: str
    schema: str
    table: str

    @property
    def fqn(self) -> str:
        return f"{self.database}.{self.schema}.{self.table}"

    @property
    def is_prod(self) -> bool:
        return (
            self.database == PROD_DIAL_DATABASE
            and self.schema == PROD_DIAL_SCHEMA
            and self.table == PROD_DIAL_TABLE
        )


def resolve_dial_write_target(
    *,
    database: str | None = None,
    schema: str | None = None,
    table: str | None = None,
    config: SnowflakeConfig | None = None,
) -> DialWriteTarget:
    """Resolve write location: explicit args → env / ``SnowflakeConfig``.

    ``SNOWFLAKE_DATABASE`` / ``SNOWFLAKE_SCHEMA`` (via ``SnowflakeConfig.from_env``)
    and ``DQT_DIAL_TABLE`` (default ``HISTORICAL_ETP_DYNAMIC_QUANTILES``).
    Never hardcodes a personal schema — set ``SNOWFLAKE_SCHEMA=RARKO`` in ``.env``.
    """
    cfg = config or SnowflakeConfig.from_env()
    db = (database or cfg.database or "").strip().upper()
    sch = (schema or cfg.schema or "").strip().upper()
    tbl = (
        table
        or (os.environ.get(_DIAL_TABLE_ENV) or "").strip()
        or DEFAULT_DIAL_TABLE
    ).upper()
    if not db or not sch or not tbl:
        raise ValueError(
            "dial write needs database, schema, and table "
            f"(got database={db!r}, schema={sch!r}, table={tbl!r})"
        )
    return DialWriteTarget(database=db, schema=sch, table=tbl)


def dial_row_for_snowflake(proposal: DialSchedule | DialProposal) -> pl.DataFrame:
    """Prod-shaped one-row frame: ``valid_date`` + ``alt_*`` (+ metadata).

    Research fields (``method``, ``gates``, …) are omitted — keep those in the
    local parquet audit trail.
    """
    now = get_dt_local()
    updated = proposal.snowflakeupdatedon or now
    row: dict[str, Any] = {
        "valid_date": proposal.valid_date,
        **proposal.alts,
        "snowflakeupdatedon": updated,
    }
    if proposal.alpha is not None:
        row["alpha"] = proposal.alpha
    if proposal.time_window is not None:
        row["time_window"] = proposal.time_window
    return pl.DataFrame([row])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_date_expr(frame: pl.DataFrame, col: str) -> pl.Expr:
    return parse_date_col(frame, col)


def _dial_applied_cols() -> list[str]:
    return [f"{_DIAL_APPLIED_PREFIX}{int(round(float(a) * 100))}" for a in GRID]


def _matrix(frame: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns {missing[:6]}{'…' if len(missing) > 6 else ''}")
    return frame.select(list(cols)).to_numpy().astype(np.float64)


def _curve_to_matrix(curve: Mapping[str, Any] | Sequence[float] | np.ndarray | pl.DataFrame) -> np.ndarray:
    """Normalize a Weatherman curve to shape ``(1, n_levels)`` or ``(n, n_levels)``."""
    if isinstance(curve, pl.DataFrame):
        cols = [c for c in WEATHERMAN_COLS if c in curve.columns]
        if len(cols) != len(WEATHERMAN_COLS):
            raise ValueError("curve frame needs full WEATHERMAN_COLS (knn_5…knn_95)")
        return _matrix(curve, WEATHERMAN_COLS)
    if isinstance(curve, Mapping):
        return np.asarray([[float(curve[c]) for c in WEATHERMAN_COLS]], dtype=np.float64)
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"curve must be 1-d or 2-d, got shape {arr.shape}")


def _schedule_alts(
    schedule: DialSchedule | Mapping[str, Any] | Sequence[float] | np.ndarray,
    *,
    n_rows: int,
    levels: Sequence[float] | np.ndarray = GRID,
) -> np.ndarray:
    """Broadcast dial fractions to ``(n_rows, n_levels)``."""
    levels_arr = np.asarray(levels, dtype=np.float64)
    if isinstance(schedule, DialSchedule):
        row = schedule.as_array(levels_arr)
    elif isinstance(schedule, Mapping):
        row = np.asarray(
            [float(schedule[shipped_alt_col(float(a))]) for a in levels_arr],
            dtype=np.float64,
        )
    else:
        row = np.asarray(schedule, dtype=np.float64)
        if row.ndim == 2:
            if row.shape != (n_rows, len(levels_arr)):
                raise ValueError(
                    f"alt matrix shape {row.shape} != ({n_rows}, {len(levels_arr)})"
                )
            return row
        if row.shape != (len(levels_arr),):
            raise ValueError(f"alt vector length {row.shape[0]} != {len(levels_arr)}")
    return np.broadcast_to(row.reshape(1, -1), (n_rows, len(levels_arr))).copy()


# ---------------------------------------------------------------------------
# GlobalDQT
# ---------------------------------------------------------------------------


class GlobalDQT:
    """Read / apply / spot-check / evaluate / propose the global dial."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        cache_hours: float = 24.0,
        sql_path: Path | None = None,
        database: str | None = None,
        schema: str | None = None,
        dial_table: str | None = None,
    ) -> None:
        self.data_dir = resolve_data_dir(data_dir)
        self.cache_hours = float(cache_hours)
        self.sql_path = Path(sql_path) if sql_path is not None else ALT_PERCENTILES_SQL
        self.database = database
        self.schema = schema
        self.dial_table = dial_table
        self._history: pl.DataFrame | None = None

    # -- paths ---------------------------------------------------------------

    @property
    def dial_cache_path(self) -> Path:
        return self.data_dir / _DIAL_CACHE_NAME

    @property
    def features_path(self) -> Path:
        return self.data_dir / _FEATURES_NAME

    @property
    def sarima_wf_path(self) -> Path:
        return self.data_dir / _SARIMA_WF_NAME

    def write_target(
        self,
        *,
        database: str | None = None,
        schema: str | None = None,
        table: str | None = None,
    ) -> DialWriteTarget:
        """Resolved dial publish FQN (instance defaults + env + overrides)."""
        return resolve_dial_write_target(
            database=database if database is not None else self.database,
            schema=schema if schema is not None else self.schema,
            table=table if table is not None else self.dial_table,
        )

    # -- read (P1) -----------------------------------------------------------

    def history(
        self,
        start: date | str | None = None,
        end: date | str | None = None,
        *,
        refresh: bool = False,
        use_cache: bool = True,
    ) -> pl.DataFrame:
        """Dial time series from Snowflake (cached under ``data_dir``).

        Returns one row per ``valid_date`` with ``alt_*`` fractions, sorted
        ascending. Optional ``start`` / ``end`` filter on ``valid_date``
        (``end`` exclusive if provided).
        """
        if self._history is None or refresh:
            self._history = self._load_history(refresh=refresh, use_cache=use_cache)

        out = self._history
        if start is not None:
            out = out.filter(pl.col("valid_date") >= _parse_day(start))
        if end is not None:
            out = out.filter(pl.col("valid_date") < _parse_day(end))
        return out

    def read_latest_daily_offsets(self, **kwargs: Any) -> pl.DataFrame:
        """Alias of :meth:`history` (notebook / monitoring back-compat)."""
        return self.history(**kwargs)

    def as_of(self, day: date | str) -> DialSchedule:
        """Latest dial row for calendar day ``day``."""
        d = _parse_day(day)
        hist = self.history()
        day_rows = hist.filter(pl.col("valid_date") == d)
        if day_rows.is_empty():
            raise LookupError(f"no dial schedule for valid_date={d.isoformat()}")
        # SQL already picks latest snowflakeupdatedon per day; keep defensive sort.
        if "snowflakeupdatedon" in day_rows.columns:
            day_rows = day_rows.sort("snowflakeupdatedon", descending=True)
        return DialSchedule.from_row(day_rows.row(0, named=True))

    def latest(self, *, today: date | None = None) -> DialSchedule:
        """``as_of(today)`` in America/Chicago (or explicit ``today``)."""
        today = today or get_dt_local().date()
        hist = self.history()
        if hist.is_empty():
            raise LookupError("dial history is empty")
        # Prefer exact today; else most recent prior day.
        exact = hist.filter(pl.col("valid_date") == today)
        if not exact.is_empty():
            return DialSchedule.from_row(exact.row(0, named=True))
        prior = hist.filter(pl.col("valid_date") <= today)
        if prior.is_empty():
            raise LookupError(f"no dial on or before {today.isoformat()}")
        return DialSchedule.from_row(prior.row(-1, named=True))

    def attach(
        self,
        loads_df: pl.DataFrame,
        date_col: str = "booked_on_date",
        *,
        how: str = "left",
    ) -> pl.DataFrame:
        """Left-join dial ``alt_*`` onto loads (``date_col`` ↔ ``valid_date``)."""
        if date_col not in loads_df.columns:
            raise ValueError(f"loads_df missing date column {date_col!r}")
        hist = self.history()
        dial_cols = ["valid_date", *ALT_COLS]
        extra = [c for c in ("snowflakeupdatedon", "alpha", "time_window") if c in hist.columns]
        dial = hist.select([*dial_cols, *extra])

        left = loads_df.with_columns(_ensure_date_expr(loads_df, date_col).alias("_join_date"))
        # Drop any pre-existing alt_* so join is authoritative.
        drop_alts = [c for c in ALT_COLS if c in left.columns]
        if drop_alts:
            left = left.drop(drop_alts)

        joined = left.join(
            dial.rename({"valid_date": "_join_date"}),
            on="_join_date",
            how=how,  # type: ignore[arg-type]
        ).drop("_join_date")
        return joined

    # -- apply + evaluate + spot-check (P2) ----------------------------------

    def apply(
        self,
        curve: Mapping[str, Any] | Sequence[float] | np.ndarray | pl.DataFrame,
        schedule: DialSchedule | Mapping[str, Any] | Sequence[float] | np.ndarray,
        levels: Sequence[float] | np.ndarray = GRID,
    ) -> np.ndarray:
        """Invert Weatherman at dial percentiles → quoted dollars.

        Returns shape ``(n_rows, n_levels)``.
        """
        qmat = _curve_to_matrix(curve)
        alts = _schedule_alts(schedule, n_rows=qmat.shape[0], levels=levels)
        out = np.empty_like(alts)
        for j in range(alts.shape[1]):
            out[:, j] = quantile_at(qmat, alts[:, j], assume_sorted=False)
        return out

    def apply_frame(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Vectorized dial-applied $ columns ``dial_5``…``dial_95`` on ``panel``."""
        qmat = _matrix(panel, WEATHERMAN_COLS)
        alts = _matrix(panel, ALT_COLS)
        dollars = np.empty_like(alts)
        for j in range(alts.shape[1]):
            dollars[:, j] = quantile_at(qmat, alts[:, j], assume_sorted=False)
        series = [
            pl.Series(name, dollars[:, j])
            for j, name in enumerate(_dial_applied_cols())
        ]
        return panel.with_columns(series)

    def evaluate(
        self,
        panel: pl.DataFrame,
        by: Sequence[str] = (),
        *,
        score_col: str = "wm_r",
    ) -> pl.DataFrame:
        """Attainment of shipped ``alt_*`` vs ``wm_r`` (highlight ``att50`` / ECE).

        Coverage at level ``a`` is ``mean(wm_r <= alt_a)``. When ``wm_r`` is
        missing it is computed from Weatherman + cost if both are present.
        """
        frame = self._ensure_wm_r(panel, score_col=score_col)
        missing_alt = [c for c in ALT_COLS if c not in frame.columns]
        if missing_alt:
            raise ValueError(
                f"evaluate needs shipped dial columns; missing {missing_alt[:4]}…"
            )

        keys = list(by)
        aggs: list[pl.Expr] = [pl.len().alias("n")]
        gap_exprs: list[pl.Expr] = []
        for a in GRID:
            lab = int(round(float(a) * 100))
            alt_col = f"alt_{lab}"
            att_name = f"att{lab}"
            aggs.append((pl.col(score_col) <= pl.col(alt_col)).mean().alias(att_name))
            gap_exprs.append(
                ((pl.col(att_name) - float(a)) * 100).alias(f"gap{lab}_pp")
            )

        grouped = frame.group_by(keys).agg(aggs) if keys else frame.select(aggs)
        out = grouped.with_columns(gap_exprs)
        # ECE = mean |gap_pp| over the grid; att50 called out explicitly.
        gap_cols = [f"gap{int(round(float(a) * 100))}_pp" for a in GRID]
        out = out.with_columns(
            pl.mean_horizontal([pl.col(c).abs() for c in gap_cols]).alias("ece_pp"),
            (pl.col("att50") - 0.5).mul(100).alias("att50_gap_pp"),
        )
        if keys:
            out = out.sort(keys)
        return out

    def diagnostics(self, history_df: pl.DataFrame | None = None) -> pl.DataFrame:
        """Dial path diagnostics: day-over-day Δpp and distance from 0.50."""
        hist = history_df if history_df is not None else self.history()
        if hist.is_empty():
            return hist
        if "valid_date" not in hist.columns or "alt_50" not in hist.columns:
            raise ValueError("diagnostics needs valid_date and alt_50")

        sorted_hist = hist.sort("valid_date")
        return sorted_hist.with_columns(
            (pl.col("alt_50") - 0.5).mul(100).alias("alt50_off_pp"),
            (pl.col("alt_50") - pl.col("alt_50").shift(1)).mul(100).alias("alt50_dod_pp"),
            *[
                (pl.col(c) - pl.col(c).shift(1)).mul(100).alias(f"{c}_dod_pp")
                for c in ("alt_5", "alt_95")
                if c in sorted_hist.columns
            ],
        )

    def compare(
        self,
        a: DialSchedule | DialProposal | Mapping[str, Any],
        b: DialSchedule | DialProposal | Mapping[str, Any],
    ) -> pl.DataFrame:
        """Δpp between two schedules (``a − b``) at each grid level."""
        aa = a if isinstance(a, DialSchedule) else DialSchedule.from_row(dict(a))
        bb = b if isinstance(b, DialSchedule) else DialSchedule.from_row(dict(b))
        rows = []
        for level in GRID:
            col = shipped_alt_col(float(level))
            va, vb = aa.alts[col], bb.alts[col]
            rows.append(
                {
                    "level": float(level),
                    "col": col,
                    "a": va,
                    "b": vb,
                    "delta": va - vb,
                    "delta_pp": (va - vb) * 100.0,
                }
            )
        return pl.DataFrame(rows)

    def spot_check(
        self,
        loadnumber: int,
        *,
        features: pl.DataFrame | None = None,
    ) -> SpotCheck:
        """One load → dial vs dial-applied $ vs booked ``p*``."""
        frame = self._features_for_loads([loadnumber], features=features)
        if frame.is_empty():
            raise LookupError(
                f"load {loadnumber} not found in {self.features_path} "
                "(pass features= or rebuild with make features)"
            )
        return self._spot_check_row(frame.row(0, named=True))

    def spot_check_loads(
        self,
        loadnumbers: Sequence[int],
        *,
        features: pl.DataFrame | None = None,
        long: bool = False,
    ) -> pl.DataFrame:
        """Batch spot-check → Polars frame (wide by default)."""
        frame = self._features_for_loads(list(loadnumbers), features=features)
        if frame.is_empty():
            return pl.DataFrame()
        checks = [self._spot_check_row(row) for row in frame.iter_rows(named=True)]
        if not checks:
            return pl.DataFrame()
        parts = [c.to_frame(long=long) for c in checks]
        return pl.concat(parts, how="diagonal_relaxed")

    def spot_check_date(
        self,
        day: date | str,
        *,
        n: int = 20,
        sample: Literal["random", "first"] = "random",
        features: pl.DataFrame | None = None,
        seed: int = 0,
    ) -> pl.DataFrame:
        """Sample loads booked on ``day`` and spot-check them."""
        d = _parse_day(day)
        feat = features if features is not None else self._read_features()
        date_col = DATE_COL if DATE_COL in feat.columns else "booked_on_date"
        day_loads = feat.filter(_ensure_date_expr(feat, date_col) == d)
        if day_loads.is_empty():
            return pl.DataFrame()
        if sample == "first":
            picked = day_loads.head(n)
        else:
            k = min(int(n), day_loads.height)
            picked = day_loads.sample(n=k, seed=seed, shuffle=True)
        ids = picked.get_column(ID_COL).to_list()
        return self.spot_check_loads(ids, features=picked)

    # -- propose (P3) --------------------------------------------------------

    def propose(
        self,
        method: ProposeMethod = "shift",
        as_of: date | str | None = None,
        *,
        r_hat_50: float | None = None,
        prior: DialSchedule | None = None,
        panel: pl.DataFrame | None = None,
        sarima_col: str = "y_hat_sarima_cal",
        max_weekly_move: float = _DEFAULT_MAX_WEEKLY_MOVE,
        clip: tuple[float, float] = _DEFAULT_CLIP,
        cal_window_days: int = 28,
        min_n: int = 200,
    ) -> DialProposal:
        """Build a ``DialProposal`` for ``as_of`` (default: tomorrow Chicago).

        Methods
        -------
        ``identity``
            Nominal ``GRID`` values.
        ``shift``
            Rigid ``alt_a = clip(a + (r̂_50 − 0.50))``. ``r_hat_50`` required
            (or taken from SARIMA artifact when present for ``as_of``).
        ``sarima``
            ``r̂`` from ``sarima_dial_walkforward.parquet`` + rigid shift.
        ``sarima_tail``
            Blend center + global conformal tails (cell δ = 0) from the tail
            stack — shadow candidate, not rigid ``pp``.
        ``conformal_global``
            Global conformal alt at 0.50 from panel, then rigid shift; other
            levels follow the same δ.
        """
        target = _parse_day(as_of) if as_of is not None else get_dt_local().date() + timedelta(days=1)
        prior = prior or self._prior_schedule(target)

        if method == "identity":
            alts = {shipped_alt_col(float(a)): float(a) for a in GRID}
        elif method == "shift":
            r50 = self._resolve_r_hat(r_hat_50, target, sarima_col=sarima_col)
            alts = self._rigid_alts(r50, clip=clip)
        elif method == "sarima":
            r50 = self._resolve_r_hat(r_hat_50, target, sarima_col=sarima_col, require_artifact=True)
            alts = self._rigid_alts(r50, clip=clip)
        elif method == "sarima_tail":
            alts = self._sarima_tail_alts(
                target,
                prior=prior,
                panel=panel,
                sarima_col=sarima_col,
                cal_window_days=cal_window_days,
                min_n=min_n,
                clip=clip,
            )
        elif method == "conformal_global":
            r50 = self._conformal_global_r50(
                panel,
                as_of=target,
                cal_window_days=cal_window_days,
                min_n=min_n,
            )
            alts = self._rigid_alts(r50, clip=clip)
        else:
            raise ValueError(
                f"unknown method {method!r}; expected "
                "shift|sarima|sarima_tail|conformal_global|identity"
            )

        proposal = DialProposal(
            valid_date=target,
            alts=alts,
            snowflakeupdatedon=None,
            alpha=None,
            time_window=None,
            method=method,
            generated_at=get_dt_local(),
            gates={},
        )
        return self.gates(
            proposal,
            prior=prior,
            max_weekly_move=max_weekly_move,
            clip=clip,
        )

    def gates(
        self,
        proposal: DialProposal,
        prior: DialSchedule | None = None,
        *,
        max_weekly_move: float = _DEFAULT_MAX_WEEKLY_MOVE,
        clip: tuple[float, float] = _DEFAULT_CLIP,
        mae_vs_naive_max: float | None = None,
    ) -> DialProposal:
        """Validate / clip a proposal; fail closed on oversized moves.

        Always clips fractions into ``clip``. If ``prior`` is set, any level
        move exceeding ``max_weekly_move`` fails the gate (proposal alts are
        still clipped, but ``passed=False``).
        """
        lo, hi = clip
        reasons: list[str] = []
        clipped: dict[str, float] = {}
        max_move = 0.0

        for col, raw in proposal.alts.items():
            v = float(np.clip(raw, lo, hi))
            if v != raw:
                reasons.append(f"{col} clipped {raw:.4f}→{v:.4f}")
            clipped[col] = v
            if prior is not None and col in prior.alts:
                move = abs(v - prior.alts[col])
                max_move = max(max_move, move)
                if move > max_weekly_move + 1e-12:
                    reasons.append(
                        f"{col} move {move:.4f} > max_weekly_move {max_weekly_move:.4f}"
                    )

        # Monotonic soft check (warn only — production dials can cross slightly).
        vals = [clipped[c] for c in ALT_COLS]
        if any(vals[i] > vals[i + 1] + 1e-9 for i in range(len(vals) - 1)):
            reasons.append("alts not monotonic non-decreasing")

        naive_mae = None
        if mae_vs_naive_max is not None and prior is not None:
            naive = {c: prior.alts[c] for c in ALT_COLS}
            naive_mae = float(
                np.mean([abs(clipped[c] - naive[c]) for c in ALT_COLS])
            )
            # Gate: proposal should not wander farther from prior than allowed.
            if naive_mae > mae_vs_naive_max:
                reasons.append(
                    f"mae_vs_prior {naive_mae:.4f} > mae_vs_naive_max {mae_vs_naive_max:.4f}"
                )

        hard_fail = any("move" in r or "mae_vs_prior" in r for r in reasons)
        gate_info = {
            "passed": not hard_fail,
            "reasons": reasons,
            "max_abs_move": max_move,
            "max_weekly_move": max_weekly_move,
            "clip": clip,
            "mae_vs_prior": naive_mae,
            "prior_date": prior.valid_date if prior is not None else None,
        }
        return replace(proposal, alts=clipped, gates=gate_info)

    def publish(
        self,
        proposal: DialProposal,
        target: PublishTarget = "parquet",
        *,
        database: str | None = None,
        schema: str | None = None,
        table: str | None = None,
        if_exists: IfExists = "append",
        allow_prod: bool = False,
        dry_run: bool = False,
        also_parquet: bool = True,
    ) -> Path | DialWriteTarget:
        """Publish a gated dial proposal to local parquet and/or Snowflake.

        Targets
        -------
        ``parquet`` / ``staging``
            Audit trail under ``$DQT_DATA_DIR/staging/`` (includes method/gates).
        ``snowflake``
            Prod-shaped row (``valid_date`` + ``alt_*`` + ``snowflakeupdatedon``)
            via ``arriveds.snowflake.create_table``. Location resolves as
            explicit args → instance defaults → ``SNOWFLAKE_DATABASE`` /
            ``SNOWFLAKE_SCHEMA`` / ``DQT_DIAL_TABLE``.

        Production
        ``DATA_SCIENCE.ETP_DYNAMIC_QUANTILE_TUNING.HISTORICAL_ETP_DYNAMIC_QUANTILES``
        is refused unless ``allow_prod=True``. Dev testing: set
        ``SNOWFLAKE_SCHEMA=RARKO`` (workspace DB) and keep the default table name.
        """
        if not proposal.passed:
            raise PermissionError(
                "refusing to publish proposal that failed gates: "
                f"{proposal.gates.get('reasons')}"
            )

        parquet_path: Path | None = None
        if target in ("parquet", "staging") or (target == "snowflake" and also_parquet):
            parquet_path = self._write_parquet_audit(proposal)
            if target in ("parquet", "staging"):
                return parquet_path

        if target != "snowflake":
            raise ValueError(
                f"unknown publish target {target!r}; expected parquet|staging|snowflake"
            )

        dest = self.write_target(database=database, schema=schema, table=table)
        if dest.is_prod and not allow_prod:
            raise PermissionError(
                f"refusing write to production dial table {dest.fqn}; "
                "pass allow_prod=True after eng sign-off, or target a workspace "
                "schema (e.g. SNOWFLAKE_SCHEMA=RARKO)."
            )

        frame = dial_row_for_snowflake(proposal)
        if dry_run:
            logger.info(
                "dry-run: would write {} row(s) x {} cols → {} (if_exists={})",
                frame.height,
                frame.width,
                dest.fqn,
                if_exists,
            )
            return dest

        nrows = create_table(
            frame,
            dest.table,
            database=dest.database,
            schema=dest.schema,
            if_exists=if_exists,
        )
        logger.info("wrote {} dial row(s) → {}", nrows, dest.fqn)
        return dest

    def _write_parquet_audit(self, proposal: DialProposal) -> Path:
        out_dir = self.data_dir / "staging"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            out_dir
            / f"dial_proposal_{proposal.valid_date.isoformat()}_{proposal.method}.parquet"
        )
        row = {
            **proposal.to_dict(),
            "method": proposal.method,
            "generated_at": proposal.generated_at.isoformat(),
            "gates_passed": proposal.passed,
            "gates_reasons": proposal.gates.get("reasons"),
        }
        pl.DataFrame([row]).write_parquet(path)
        logger.info("wrote staging dial proposal → {}", path)
        return path

    # -- private -------------------------------------------------------------

    def _load_history(self, *, refresh: bool, use_cache: bool) -> pl.DataFrame:
        cache = self.dial_cache_path
        if use_cache and not refresh and is_fresh(cache, hours=self.cache_hours):
            logger.debug("dial history cache hit {}", cache)
            return self._normalize_history(pl.read_parquet(cache))

        logger.info("fetching dial history via {}", self.sql_path.name)
        raw = query_sf(self.sql_path)
        hist = self._normalize_history(raw)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            hist.write_parquet(cache)
        except OSError as exc:
            logger.warning("could not cache dial history to {}: {}", cache, exc)
        return hist

    @staticmethod
    def _normalize_history(df: pl.DataFrame) -> pl.DataFrame:
        if "valid_date" not in df.columns:
            raise ValueError("dial history missing valid_date")
        out = df.with_columns(parse_date_col(df, "valid_date").alias("valid_date"))
        # Back-compat column used by ds-monitoring notebook charts.
        if "snowflakeupdatedon" in out.columns:
            out = out.with_columns(
                pl.col("snowflakeupdatedon").cast(pl.Date, strict=False).alias("date")
            )
        elif "date" not in out.columns:
            out = out.with_columns(pl.col("valid_date").alias("date"))

        missing = [c for c in ALT_COLS if c not in out.columns]
        if missing:
            raise ValueError(f"dial history missing {missing[:4]}…")
        return out.sort("valid_date")

    def _read_features(self) -> pl.DataFrame:
        path = self.features_path
        if not path.exists():
            raise FileNotFoundError(
                f"features not found at {path}; run `make features` or pass features="
            )
        return pl.read_parquet(path)

    def _features_for_loads(
        self,
        loadnumbers: Sequence[int],
        *,
        features: pl.DataFrame | None,
    ) -> pl.DataFrame:
        feat = features if features is not None else self._read_features()
        if ID_COL not in feat.columns:
            raise ValueError(f"features missing {ID_COL}")
        wanted = list(dict.fromkeys(loadnumbers))
        return feat.filter(pl.col(ID_COL).is_in(wanted))

    def _spot_check_row(self, row: Mapping[str, Any]) -> SpotCheck:
        missing = [
            c
            for c in (*ALT_COLS, *WEATHERMAN_COLS, *ETP_COLS)
            if c not in row or row[c] is None
        ]
        if missing:
            raise ValueError(
                f"spot_check row missing {missing[:6]}{'…' if len(missing) > 6 else ''}"
            )

        day_raw = row.get("booked_on_date", row.get(DATE_COL))
        if day_raw is None:
            raise ValueError("spot_check needs booked_on_date or date")
        day = _parse_day(day_raw)

        alt = {c: float(row[c]) for c in ALT_COLS}
        wm = {c: float(row[c]) for c in WEATHERMAN_COLS}
        etp = {c: float(row[c]) for c in ETP_COLS}

        qmat = np.asarray([[wm[c] for c in WEATHERMAN_COLS]], dtype=np.float64)
        alt_arr = np.asarray([[alt[c] for c in ALT_COLS]], dtype=np.float64)
        dial_mat = np.empty_like(alt_arr)
        for j in range(alt_arr.shape[1]):
            dial_mat[0, j] = quantile_at(qmat, alt_arr[:, j], assume_sorted=False)[0]
        dial = {
            f"dial_{int(round(float(a) * 100))}": float(dial_mat[0, j])
            for j, a in enumerate(GRID)
        }

        cost = row.get(COST_COL, row.get("cost"))
        cost_f = None if cost is None else float(cost)
        wm_r = etp_r = None
        if cost_f is not None:
            cost_arr = np.asarray([cost_f], dtype=np.float64)
            wm_r = float(implied_percentile(qmat, cost_arr)[0])
            etp_mat = np.asarray([[etp[c] for c in ETP_COLS]], dtype=np.float64)
            etp_r = float(implied_percentile(etp_mat, cost_arr)[0])

        return SpotCheck(
            loadnumber=int(row[ID_COL]),
            booked_on_date=day,
            alt=alt,
            weatherman=wm,
            dial_applied=dial,
            etp_booked=etp,
            cost=cost_f,
            wm_r=wm_r,
            etp_r=etp_r,
        )

    def _ensure_wm_r(self, panel: pl.DataFrame, *, score_col: str) -> pl.DataFrame:
        if score_col in panel.columns:
            return panel
        if score_col != "wm_r":
            raise ValueError(f"panel missing score column {score_col!r}")
        if COST_COL not in panel.columns and "cost" not in panel.columns:
            raise ValueError("evaluate needs wm_r or cost + WEATHERMAN_COLS")
        cost_col = COST_COL if COST_COL in panel.columns else "cost"
        wm_r = implied_percentile(
            _matrix(panel, WEATHERMAN_COLS),
            panel[cost_col].to_numpy().astype(np.float64),
        )
        return panel.with_columns(pl.Series("wm_r", wm_r))

    def _prior_schedule(self, target: date) -> DialSchedule | None:
        hist = self.history()
        prior_rows = hist.filter(pl.col("valid_date") < target)
        if prior_rows.is_empty():
            return None
        return DialSchedule.from_row(prior_rows.row(-1, named=True))

    def _resolve_r_hat(
        self,
        r_hat_50: float | None,
        target: date,
        *,
        sarima_col: str,
        require_artifact: bool = False,
    ) -> float:
        if r_hat_50 is not None:
            return float(r_hat_50)
        path = self.sarima_wf_path
        if not path.exists():
            if require_artifact:
                raise FileNotFoundError(
                    f"sarima propose needs {path}; run `make sarima-wf` or pass r_hat_50="
                )
            raise ValueError("shift propose needs r_hat_50= or a SARIMA walk-forward artifact")
        wf = pl.read_parquet(path)
        date_col = "booked_date" if "booked_date" in wf.columns else "valid_date"
        hit = wf.filter(pl.col(date_col) == target)
        if hit.is_empty():
            # Fall back to latest row on/before target with a finite hat.
            hit = (
                wf.filter(pl.col(date_col) <= target)
                .filter(pl.col(sarima_col).is_not_null() & pl.col(sarima_col).is_finite())
                .sort(date_col)
            )
            if hit.is_empty():
                raise LookupError(f"no SARIMA {sarima_col} on/before {target}")
            logger.warning(
                "no SARIMA row for {}; using latest on/before ({})",
                target,
                hit[date_col][-1],
            )
            return float(hit[sarima_col][-1])
        val = hit[sarima_col][0]
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            raise LookupError(f"SARIMA {sarima_col} is null/non-finite for {target}")
        return float(val)

    @staticmethod
    def _rigid_alts(
        r_hat_50: float,
        *,
        clip: tuple[float, float],
    ) -> dict[str, float]:
        delta = float(r_hat_50) - 0.50
        mat = rigid_quote_alts(tuple(GRID.tolist()), np.asarray([delta]), clip=clip)
        return {
            shipped_alt_col(float(a)): float(mat[0, j]) for j, a in enumerate(GRID)
        }

    def _sarima_tail_alts(
        self,
        target: date,
        *,
        prior: DialSchedule | None,
        panel: pl.DataFrame | None,
        sarima_col: str,
        cal_window_days: int,
        min_n: int,
        clip: tuple[float, float],
    ) -> dict[str, float]:
        path = self.sarima_wf_path
        if not path.exists():
            raise FileNotFoundError(
                f"sarima_tail propose needs {path}; run `make sarima-wf`"
            )
        dial = pl.read_parquet(path)
        if panel is None:
            panel_path = self.data_dir / "panel_loads.parquet"
            if not panel_path.exists():
                raise FileNotFoundError(
                    "sarima_tail needs panel= or panel_loads.parquet "
                    f"under {self.data_dir}"
                )
            panel = pl.read_parquet(panel_path)
        if prior is None:
            raise LookupError(
                f"sarima_tail needs a prior shipped dial before {target.isoformat()}"
            )
        return global_alts_from_tail(
            panel,
            dial,
            prior.alts,
            target,
            r_hat_col=sarima_col,
            cal_window_days=cal_window_days,
            min_n=min_n,
            clip=clip,
        )

    def _conformal_global_r50(
        self,
        panel: pl.DataFrame | None,
        *,
        as_of: date,
        cal_window_days: int,
        min_n: int,
    ) -> float:
        if panel is None:
            panel_path = self.data_dir / "panel_loads.parquet"
            if not panel_path.exists():
                raise FileNotFoundError(
                    "conformal_global needs panel= or panel_loads.parquet "
                    f"under {self.data_dir}"
                )
            panel = pl.read_parquet(panel_path)

        if "week" not in panel.columns:
            date_col = DATE_COL if DATE_COL in panel.columns else "booked_date"
            panel = panel.with_columns(
                parse_date_col(panel, date_col).dt.truncate("1w").alias("week")
            )
        # Restrict to weeks at/before as_of so walk-forward does not peek ahead.
        panel = panel.filter(pl.col("week") <= as_of)
        alts = walk_forward_alts(
            panel,
            group_cols=[],
            levels=(0.5,),
            cal_window_days=cal_window_days,
            min_n=min_n,
        )
        if alts.is_empty() or alts["alt"].drop_nulls().is_empty():
            raise LookupError("conformal_global produced no alt_50")
        # Latest week with a finite global alt.
        latest = alts.filter(pl.col("alt").is_not_null()).sort("week").row(-1, named=True)
        return float(latest["alt"])


__all__ = [
    "ALT_PERCENTILES_SQL",
    "DEFAULT_DIAL_TABLE",
    "PROD_DIAL_DATABASE",
    "PROD_DIAL_SCHEMA",
    "PROD_DIAL_TABLE",
    "DialProposal",
    "DialSchedule",
    "DialWriteTarget",
    "GlobalDQT",
    "SpotCheck",
    "dial_row_for_snowflake",
    "resolve_dial_write_target",
]


if __name__ == "__main__":
    global_dqt = GlobalDQT()
    print(global_dqt.history().tail())

    from dqt.model import GlobalDQT
    dqt = GlobalDQT()  # or GlobalDQT(schema="RARKO", database="DATA_SCIENCE_WORKSPACE")
    print(dqt.write_target().fqn)

    df_dev_dial = query_sf(f"select * from {dqt.write_target().fqn}")
    # → DATA_SCIENCE_WORKSPACE.RARKO.HISTORICAL_ETP_DYNAMIC_QUANTILES

    # proposal = dqt.propose(method="shift", as_of="2026-08-26", r_hat_50=0.52)
    # dqt.publish(proposal, target="snowflake", dry_run=True)   # resolve only
    # dqt.publish(proposal, target="snowflake", if_exists="append")  # real write