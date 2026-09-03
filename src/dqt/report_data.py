"""Pure helpers for the weekly Streamlit report (no Streamlit imports).

Keeps join / week-selection / per-model scoring testable without loading
``app/report.py`` (which has Streamlit side effects at import time).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from dqt import get_dt_local
from dqt.holidays import enabled_specs, resolve_occurrences
from dqt.panel import EQUIPMENT_MAP, haul_band
from dqt.score import QUANTILES, summary_metrics, weekly_att50
from dqt.score.constants import COST_COL

HAUL_SHORT_EDGE = 250
HAUL_LONG_EDGE = 600

EXTRA_MODEL_TEMPLATES: dict[str, str] = {
    "Hybrid": "hybrid_{q:02d}",
    "SARIMA_pp": "sarima_pp_{q:02d}",
    "SARIMA_hybrid": "sarima_hybrid_{q:02d}",
    "SARIMA_blend": "sarima_blend_{q:02d}",
    "SARIMA_tail": "sarima_tail_{q:02d}",
}

# Columns the weekly report needs from features.parquet (drop geo/market baggage).
_REPORT_BASE_COLS: tuple[str, ...] = (
    "loadnumber",
    "booked_on_date",
    "load_type",
    "load_class_bucket",
    "loadmiles",
    "market_lane",
    COST_COL,
    "t1",
    "t2",
    "t3",
    "t4",
)


def report_feature_columns(available: list[str] | None = None) -> list[str]:
    """Slim column set for ``make report`` — ~half the parquet width/RAM."""
    wanted = list(_REPORT_BASE_COLS)
    # Match ``MODEL_QCOL`` templates without importing the full score package surface.
    wanted.extend(f"knn_{q}" for q in QUANTILES)
    wanted.extend(f"p{q:02d}" for q in QUANTILES)
    if available is None:
        return wanted
    avail = set(available)
    return [c for c in wanted if c in avail]


def week_complete(week_start: date, *, today: date | None = None) -> bool:
    """Mon-start week is complete once its Sunday has passed (week_start+7 <= today)."""
    today = today or get_dt_local().date()
    return week_start + timedelta(days=7) <= today


def format_week_range(week_start: date, *, end: date | None = None) -> str:
    """Printable Mon–Sun span: ``August 3–9, 2026`` or ``July 28 – August 3, 2026``."""
    end = end or (week_start + timedelta(days=6))
    if week_start.year == end.year and week_start.month == end.month:
        return f"{week_start.strftime('%B')} {week_start.day}–{end.day}, {end.year}"
    if week_start.year == end.year:
        return (
            f"{week_start.strftime('%B')} {week_start.day} – "
            f"{end.strftime('%B')} {end.day}, {end.year}"
        )
    return (
        f"{week_start.strftime('%B')} {week_start.day}, {week_start.year} – "
        f"{end.strftime('%B')} {end.day}, {end.year}"
    )


def format_week_range_short(week_start: date, *, end: date | None = None) -> str:
    """Compact slider label: ``Aug 3–9`` or ``Jul 28 – Aug 3``."""
    end = end or (week_start + timedelta(days=6))
    if week_start.month == end.month and week_start.year == end.year:
        return f"{week_start.strftime('%b')} {week_start.day}–{end.day}"
    return (
        f"{week_start.strftime('%b')} {week_start.day} – "
        f"{end.strftime('%b')} {end.day}"
    )


def default_report_week(
    weeks: list[date],
    *,
    preferred_covered: list[date] | None = None,
    today: date | None = None,
) -> date:
    """Prefer the latest *complete* week (optionally intersecting coverage)."""
    today = today or get_dt_local().date()
    complete = [w for w in weeks if week_complete(w, today=today)]
    if preferred_covered:
        covered_complete = [w for w in preferred_covered if w in complete]
        if covered_complete:
            return covered_complete[-1]
        if preferred_covered:
            return preferred_covered[-1]
    if complete:
        return complete[-1]
    return weeks[-1]


def available_quantile_cols(
    model_qcol: Mapping[str, str], columns: set[str]
) -> dict[str, str]:
    """Keep models whose full NOMINAL grid columns exist on the frame."""
    kept: dict[str, str] = {}
    for name, tmpl in model_qcol.items():
        cols = [tmpl.format(q=q) for q in QUANTILES]
        if all(c in columns for c in cols):
            kept[name] = tmpl
    return kept


def prepare_features_frame(
    features: pl.DataFrame,
    *,
    haul_short_edge: int = HAUL_SHORT_EDGE,
    haul_long_edge: int = HAUL_LONG_EDGE,
) -> pl.DataFrame:
    """Segment + week columns used by the weekly report."""
    return (
        features.filter(pl.col("load_type").is_in(("DRY", "REEFER")))
        .with_columns(
            pl.col("booked_on_date")
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("load_date"),
            pl.col("load_class_bucket").alias("mode"),
            pl.col("load_type")
            .replace_strict(EQUIPMENT_MAP, default="Other")
            .alias("equipment"),
        )
        .drop_nulls("load_date")
        .with_columns(
            pl.col("load_date").dt.truncate("1w").alias("week"),
            haul_band(pl.col("loadmiles"), (haul_short_edge, haul_long_edge)).alias(
                "haul_band"
            ),
        )
    )


def left_join_quantile_pack(
    frame: pl.DataFrame,
    pack: pl.DataFrame,
    *,
    prefix: str,
) -> pl.DataFrame:
    """Left-join ``loadnumber`` + columns starting with ``prefix`` (no row drop)."""
    cols = [c for c in pack.columns if c.startswith(prefix)]
    if not cols:
        return frame
    return frame.join(pack.select(["loadnumber", *cols]), on="loadnumber", how="left")


def load_optional_packs(
    frame: pl.DataFrame,
    *,
    hybrid_path: Path | None = None,
    join_hybrid: bool = False,
    sarima_path: Path | None = None,
    join_sarima: bool = False,
    sarima_hybrid_path: Path | None = None,
    join_sarima_hybrid: bool = False,
    sarima_blend_path: Path | None = None,
    join_sarima_blend: bool = False,
    sarima_tail_path: Path | None = None,
    join_sarima_tail: bool = False,
) -> pl.DataFrame:
    """Left-join Hybrid and/or SARIMA packs when requested."""
    out = frame
    if join_hybrid and hybrid_path is not None and hybrid_path.exists():
        out = left_join_quantile_pack(
            out, pl.read_parquet(hybrid_path), prefix="hybrid_"
        )
    if join_sarima and sarima_path is not None and sarima_path.exists():
        out = left_join_quantile_pack(
            out, pl.read_parquet(sarima_path), prefix="sarima_pp_"
        )
    if (
        join_sarima_hybrid
        and sarima_hybrid_path is not None
        and sarima_hybrid_path.exists()
    ):
        out = left_join_quantile_pack(
            out, pl.read_parquet(sarima_hybrid_path), prefix="sarima_hybrid_"
        )
    if (
        join_sarima_blend
        and sarima_blend_path is not None
        and sarima_blend_path.exists()
    ):
        out = left_join_quantile_pack(
            out, pl.read_parquet(sarima_blend_path), prefix="sarima_blend_"
        )
    if join_sarima_tail and sarima_tail_path is not None and sarima_tail_path.exists():
        out = left_join_quantile_pack(
            out, pl.read_parquet(sarima_tail_path), prefix="sarima_tail_"
        )
    return out


def load_p50_map(path: Path, col: str) -> pl.DataFrame | None:
    """Slim ``loadnumber`` → quote map for cell-gap scoring."""
    if not path.exists():
        return None
    cols = pl.scan_parquet(path).collect_schema().names()
    if col not in cols:
        return None
    return pl.read_parquet(path, columns=["loadnumber", col])


def covered_weeks(
    frame: pl.DataFrame, p50_map: pl.DataFrame | None
) -> list[date]:
    """Weeks with at least one load that joins the p50 map."""
    if p50_map is None or p50_map.is_empty():
        return []
    return (
        frame.select("loadnumber", "week")
        .join(p50_map.select("loadnumber"), on="loadnumber", how="inner")["week"]
        .unique()
        .sort()
        .to_list()
    )


def p50_col(template: str) -> str:
    return template.format(q=50)


def _finite_quote(col: str) -> pl.Expr:
    """True when a dollar quote is usable (drops null / NaN / ±inf burn-in)."""
    return pl.col(col).is_finite()


def rows_with_model(frame: pl.DataFrame, template: str) -> pl.DataFrame:
    """Loads where the model's p50 quote is finite (weekend decision A)."""
    col = p50_col(template)
    if col not in frame.columns:
        return frame.head(0)
    return frame.filter(_finite_quote(col))


def intersection_for_models(
    frame: pl.DataFrame, model_qcol: Mapping[str, str]
) -> pl.DataFrame:
    """Rows where every listed model has a finite p50 (fair Q–Q / holidays)."""
    out = frame
    for tmpl in model_qcol.values():
        col = p50_col(tmpl)
        if col not in out.columns:
            return frame.head(0)
        out = out.filter(_finite_quote(col))
    return out


def model_coverage_spans(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    time_col: str = "load_date",
) -> dict[str, tuple[date, date]]:
    """First/last ``time_col`` with a finite p50 quote, per model."""
    spans: dict[str, tuple[date, date]] = {}
    if time_col not in frame.columns:
        return spans
    for name, tmpl in model_qcol.items():
        col = p50_col(tmpl)
        if col not in frame.columns:
            continue
        sub = frame.filter(_finite_quote(col))
        if sub.is_empty():
            continue
        spans[name] = (sub[time_col].min(), sub[time_col].max())
    return spans


def coverage_intersection(
    spans: Mapping[str, tuple[date, date]],
) -> tuple[date, date] | None:
    """Latest start → earliest end. ``None`` if empty or disjoint."""
    if not spans:
        return None
    start = max(s for s, _ in spans.values())
    end = min(e for _, e in spans.values())
    if start > end:
        return None
    return start, end


def first_comparable_date(
    daily: pl.DataFrame,
    *,
    time_col: str = "load_date",
    lo: float = 0.02,
    hi: float = 0.98,
    min_n: int = 50,
) -> dict[str, date]:
    """First date per model whose att50 is not stuck at 0/1 (SARIMA burn-in)."""
    if daily.is_empty() or "model" not in daily.columns:
        return {}
    out: dict[str, date] = {}
    for model in daily["model"].unique(maintain_order=True).to_list():
        sub = daily.filter(pl.col("model") == model).sort(time_col)
        if sub.is_empty():
            continue
        hit = sub.filter(
            (pl.col("n_loads") >= min_n)
            & pl.col("att50").is_between(lo, hi, closed="both")
        )
        out[model] = (hit[time_col][0] if not hit.is_empty() else sub[time_col][0])
    return out


def drop_leading_degenerate(
    daily: pl.DataFrame,
    *,
    time_col: str = "load_date",
    lo: float = 0.02,
    hi: float = 0.98,
    min_n: int = 50,
) -> pl.DataFrame:
    """Trim each model's leading 0%/100% att50 run (walk-forward burn-in)."""
    starts = first_comparable_date(
        daily, time_col=time_col, lo=lo, hi=hi, min_n=min_n
    )
    if not starts:
        return daily
    parts = [
        daily.filter((pl.col("model") == model) & (pl.col(time_col) >= start))
        for model, start in starts.items()
    ]
    return pl.concat(parts).sort([time_col, "model"]) if parts else daily.head(0)


def filter_date_range(
    frame: pl.DataFrame,
    start: date,
    end: date,
    *,
    time_col: str = "load_date",
) -> pl.DataFrame:
    """Inclusive ``[start, end]`` slice on ``time_col``."""
    if time_col not in frame.columns:
        return frame.head(0)
    return frame.filter((pl.col(time_col) >= start) & (pl.col(time_col) <= end))


def holiday_event_dates(
    year_start: int,
    year_end: int,
    *,
    include_dot: bool = True,
) -> list[date]:
    """Observed holiday event days (not the ±3 neighborhood) in ``[year_start, year_end]``."""
    occ = resolve_occurrences(
        enabled_specs(include_dot=include_dot),
        year_start=year_start,
        year_end=year_end,
    )
    if occ.is_empty():
        return []
    days: list[date] = []
    for row in occ.iter_rows(named=True):
        d = row["event_start"]
        end = row["event_end"]
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
    return sorted(set(days))


def filter_calendar(
    frame: pl.DataFrame,
    *,
    include_weekends: bool = False,
    include_holidays: bool = True,
    holiday_dates: Sequence[date] | None = None,
    time_col: str = "load_date",
) -> pl.DataFrame:
    """Drop Sat/Sun and/or observed holiday booked dates from scoring frames.

    Polars weekday is ISO (Mon=1 … Sun=7). Weekends off is the report default.
    Holiday days are ``event_start..event_end``, not the ±3 analysis window.
    """
    if time_col not in frame.columns:
        return frame
    out = frame
    if not include_weekends:
        out = out.filter(pl.col(time_col).dt.weekday() < 6)
    if not include_holidays and holiday_dates:
        out = out.filter(~pl.col(time_col).is_in(list(holiday_dates)))
    return out


def calendar_filter_label(
    *,
    include_weekends: bool,
    include_holidays: bool,
) -> str:
    """Short caption fragment for the active calendar policy."""
    parts: list[str] = []
    parts.append("weekends on" if include_weekends else "weekends off")
    parts.append("holidays on" if include_holidays else "holidays off")
    return " · ".join(parts)


def default_compare_window(
    spans: Mapping[str, tuple[date, date]],
    comparable: Mapping[str, date] | None = None,
) -> tuple[date, date] | None:
    """Overlap of finite-quote spans, advanced to each model's first real att50."""
    overlap = coverage_intersection(spans)
    if overlap is None:
        return None
    start, end = overlap
    if comparable:
        starts = [comparable[m] for m in spans if m in comparable]
        if starts:
            start = max(start, *starts)
    if start > end:
        return None
    return start, end


def snapshot_att50(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    actual_col: str = COST_COL,
) -> pl.DataFrame:
    """One row per model: n / att50 / gap. Cheap (no CRPS / ECE / pinball)."""
    rows: list[dict] = []
    for name, tmpl in model_qcol.items():
        col = p50_col(tmpl)
        if col not in frame.columns or actual_col not in frame.columns:
            continue
        sub = frame.filter(_finite_quote(col) & pl.col(actual_col).is_finite())
        n = sub.height
        att = (
            float((sub[actual_col] <= sub[col]).mean()) if n else None
        )
        rows.append(
            {
                "model": name,
                "n_loads": n,
                "att50": round(att, 4) if att is not None else None,
                "att50_gap_pp": (
                    round((att - 0.5) * 100, 2) if att is not None else None
                ),
            }
        )
    return pl.DataFrame(rows)


def cell_att50_by_model(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    group_cols: tuple[str, ...] = ("mode", "equipment", "haul_band"),
    actual_col: str = COST_COL,
    min_n: int = 1,
) -> pl.DataFrame:
    """p50 attainment per 12-cell (mode × equipment × haul) × model.

    One row per ``(cell, model)`` with ``n_loads``, ``att50``, ``att50_gap_pp``.
    """
    missing_dims = [c for c in group_cols if c not in frame.columns]
    if missing_dims or actual_col not in frame.columns:
        return pl.DataFrame()
    cell_expr = pl.concat_str(
        [pl.col(c) for c in group_cols], separator=" \u00b7 "
    ).alias("cell")
    parts: list[pl.DataFrame] = []
    for name, tmpl in model_qcol.items():
        col = p50_col(tmpl)
        if col not in frame.columns:
            continue
        sub = frame.filter(
            _finite_quote(col) & pl.col(actual_col).is_finite()
        ).drop_nulls(list(group_cols))
        if sub.is_empty():
            continue
        scored = (
            sub.with_columns(
                cell_expr,
                (pl.col(actual_col) <= pl.col(col)).alias("_hit"),
            )
            .group_by("cell")
            .agg(
                pl.len().alias("n_loads"),
                pl.col("_hit").mean().alias("att50"),
            )
            .filter(pl.col("n_loads") >= min_n)
            .with_columns(
                pl.lit(name).alias("model"),
                pl.col("att50").round(4),
                ((pl.col("att50") - 0.5) * 100).round(2).alias("att50_gap_pp"),
            )
            .select("cell", "model", "n_loads", "att50", "att50_gap_pp")
        )
        if not scored.is_empty():
            parts.append(scored)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort(["cell", "model"])


def grid_aad_by_model(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    actual_col: str = COST_COL,
) -> pl.DataFrame:
    """Mean |att_q − q| in pp across the 19-level grid (p05–p95). Same as ECE.

    One scan per model; no pinball / CRPS. Requires a finite quote at every
    grid level (same contract as ``available_quantile_cols``).
    """
    rows: list[dict] = []
    n_levels = len(QUANTILES)
    for name, tmpl in model_qcol.items():
        cols = [tmpl.format(q=q) for q in QUANTILES]
        if actual_col not in frame.columns or any(c not in frame.columns for c in cols):
            continue
        sub = frame.filter(
            pl.col(actual_col).is_finite()
            & pl.all_horizontal(pl.col(c).is_finite() for c in cols)
        )
        n = sub.height
        if n == 0:
            continue
        atts = sub.select(
            [
                (pl.col(actual_col) <= pl.col(c)).mean().alias(str(q))
                for q, c in zip(QUANTILES, cols, strict=True)
            ]
        )
        gaps = [abs(float(atts[str(q)][0]) - q / 100) * 100 for q in QUANTILES]
        worst_i = int(max(range(n_levels), key=lambda i: gaps[i]))
        rows.append(
            {
                "model": name,
                "n_loads": n,
                "n_levels": n_levels,
                "aad_pp": round(sum(gaps) / n_levels, 2),
                "worst_gap_pp": round(gaps[worst_i], 2),
                "worst_q": round(QUANTILES[worst_i] / 100, 2),
            }
        )
    return pl.DataFrame(rows)


def summary_metrics_by_model(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    actual_col: str = COST_COL,
) -> pl.DataFrame:
    """Per-model ``summary_metrics`` on rows with that model's quotes (no silent NaNs)."""
    parts: list[pl.DataFrame] = []
    for name, tmpl in model_qcol.items():
        sub = rows_with_model(frame, tmpl)
        if sub.is_empty():
            continue
        parts.append(
            summary_metrics(sub, model_qcol={name: tmpl}, actual_col=actual_col)
        )
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


def report_context_fingerprint(
    *,
    features_mtime: float,
    hybrid_mtime: float,
    sarima_mtime: float,
    sarima_hybrid_mtime: float,
    join_hybrid: bool,
    join_sarima: bool,
    join_sarima_hybrid: bool,
    include_weekends: bool,
    include_holidays: bool,
    include_dot: bool,
    model_names: tuple[str, ...] | list[str],
    sarima_blend_mtime: float = 0.0,
    sarima_tail_mtime: float = 0.0,
    join_sarima_blend: bool = False,
    join_sarima_tail: bool = False,
) -> str:
    """Stable cache key for week-explorer scorecards (paths/mtimes + UI flags)."""
    models = ",".join(model_names)
    return (
        f"f{features_mtime:.0f}|h{hybrid_mtime:.0f}|s{sarima_mtime:.0f}|"
        f"sh{sarima_hybrid_mtime:.0f}|sb{sarima_blend_mtime:.0f}|"
        f"st{sarima_tail_mtime:.0f}|jh{int(join_hybrid)}|js{int(join_sarima)}|"
        f"jsh{int(join_sarima_hybrid)}|jsb{int(join_sarima_blend)}|"
        f"jst{int(join_sarima_tail)}|w{int(include_weekends)}|"
        f"hol{int(include_holidays)}|dot{int(include_dot)}|{models}"
    )


def all_week_summary_metrics(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    actual_col: str = COST_COL,
    week_col: str = "week",
) -> pl.DataFrame:
    """One scorecard row per ``(week, model)`` for the Week explorer preload.

    Loops distinct weeks on an already-slim in-memory frame. Call once per
    fingerprint; week flips should only ``filter`` this result.
    """
    if week_col not in frame.columns or frame.is_empty():
        return pl.DataFrame()
    weeks = frame[week_col].drop_nulls().unique().sort().to_list()
    parts: list[pl.DataFrame] = []
    for week in weeks:
        sub = frame.filter(pl.col(week_col) == week)
        if sub.is_empty():
            continue
        card = summary_metrics_by_model(sub, model_qcol, actual_col=actual_col)
        if card.is_empty():
            continue
        parts.append(card.with_columns(pl.lit(week).alias(week_col)))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort([week_col, "model"])


def weekly_att50_by_model(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
    *,
    time_col: str = "load_date",
    actual_col: str = COST_COL,
) -> pl.DataFrame:
    """Daily/weekly att50 per model, each scored only on rows with its quotes."""
    parts: list[pl.DataFrame] = []
    for name, tmpl in model_qcol.items():
        sub = rows_with_model(frame, tmpl)
        if sub.is_empty():
            continue
        parts.append(
            weekly_att50(
                sub,
                time_col=time_col,
                model_qcol={name: tmpl},
                actual_col=actual_col,
            )
        )
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort([time_col, "model"])


def relabel_cell_gap_policy(
    scorecard: pl.DataFrame, *, policy: str
) -> pl.DataFrame:
    """Rename ``hybrid_gap_pp`` → ``{policy}_gap_pp`` for UI display."""
    if "hybrid_gap_pp" not in scorecard.columns:
        return scorecard
    return scorecard.rename({"hybrid_gap_pp": f"{policy}_gap_pp"})


def annotate_best_ece(
    frame: pl.DataFrame,
    *,
    ece_col: str = "ece_pp",
    label_col: str = "ece_vs_best",
) -> pl.DataFrame:
    """Add a 'best' / '+X.XX vs best' label on the lowest ECE (same rule as Overview).

    Ties at the displayed 2-decimal ECE both get ``best``. The label column
    is inserted immediately after ``ece_col``. Empty / all-null ECE frames
    are returned unchanged (no label column).
    """
    if frame.is_empty() or ece_col not in frame.columns:
        return frame
    values = frame[ece_col].to_list()
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return frame
    best = min(finite)
    labels: list[str | None] = []
    for v in values:
        if v is None:
            labels.append(None)
            continue
        gap = round(float(v) - best, 2)
        labels.append("best" if gap == 0 else f"{gap:+.2f} vs best")
    out = frame.with_columns(pl.Series(label_col, labels, dtype=pl.Utf8))
    ordered: list[str] = []
    for col in out.columns:
        if col == label_col:
            continue
        ordered.append(col)
        if col == ece_col:
            ordered.append(label_col)
    return out.select(ordered)
