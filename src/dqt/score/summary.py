"""Concise multi-model quantile summaries for apps and notebooks.

`summary_metrics` collapses `compare`/`summarize` into one row per
(group ×) model — ECE, CRPS, max gap, p50 attainment, and (when available)
business-slider pinball. Pass `model_qcol` to add a third model (e.g. Hybrid
with ``{"Hybrid": "hybrid_{q:02d}"}``) without touching call sites.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from .api import ModelSpec, compare, summarize
from .business import MODEL_QCOL, business_pinball_for_models
from .constants import COST_COL, NOMINAL


def model_specs(
    model_qcol: Mapping[str, str] | None = None,
    levels: Sequence[float] = NOMINAL,
) -> dict[str, ModelSpec]:
    """Build ``ModelSpec``s from a name → col_template map (defaults to MODELS)."""
    model_qcol = dict(model_qcol or MODEL_QCOL)
    return {
        name: ModelSpec(name, levels, col_template=tmpl)
        for name, tmpl in model_qcol.items()
    }


def summary_metrics(
    frame: pl.DataFrame,
    group_col: str | None = None,
    groups: Sequence | None = None,
    *,
    model_qcol: Mapping[str, str] | None = None,
    actual_col: str = COST_COL,
    levels: Sequence[float] = NOMINAL,
) -> pl.DataFrame:
    """One row per (group ×) model with curve-fit + p50 attainment.

    Parameters
    ----------
    frame
        Loads with quantile columns matching ``model_qcol`` templates.
    group_col, groups
        Optional subgrouping. When ``group_col`` is None the whole frame is
        one slice (no ``group`` column). When ``groups`` is None every distinct
        value of ``group_col`` is scored.
    model_qcol
        Name → ``str.format`` template with ``{q}`` as integer percent
        (e.g. ``"knn_{q}"``, ``"p{q:02d}"``, ``"hybrid_{q:02d}"``).
        Defaults to Weatherman + DQT/ETP.
    """
    model_qcol = dict(model_qcol or MODEL_QCOL)
    models = list(model_qcol)
    specs = model_specs(model_qcol, levels)

    if group_col is None:
        slices: list[tuple[object, pl.DataFrame]] = [(None, frame)]
    else:
        if groups is None:
            groups = frame[group_col].drop_nulls().unique(maintain_order=True).to_list()
        slices = [(grp, frame.filter(pl.col(group_col) == grp)) for grp in groups]

    rows: list[dict] = []
    for grp, sub in slices:
        if sub.is_empty():
            continue
        long = compare(specs, frame=sub, actual_col=actual_col)
        summary = summarize(long)
        tpin = business_pinball_for_models(sub, model_qcol)

        for model in models:
            row = summary.filter(pl.col("model") == model).row(0, named=True)
            att_rows = long.filter(
                (pl.col("model") == model) & (pl.col("quantile") == 0.5)
            )
            att50 = float(att_rows["attainment"][0]) if att_rows.height else None
            t = tpin.get(model)
            rec: dict = {
                "model": model,
                "n_loads": row["n_loads"],
                "att50": round(att50, 4) if att50 is not None else None,
                "att50_gap_pp": (
                    round((att50 - 0.5) * 100, 2) if att50 is not None else None
                ),
                "crps_usd": round(row["crps_usd"], 2),
                "crps_pct_cost": round(row["crps_pct_cost"], 2),
                "ece_pp": round(row["ece_pp"], 2),
                "max_gap_pp": round(row["max_gap_pp"], 2),
                "worst_gap_pp": round(row["worst_gap_pp"], 2),
                "worst_q": (
                    round(float(row["worst_q"]), 2)
                    if row.get("worst_q") is not None
                    else None
                ),
                "pinball_t_usd": round(t, 2) if t is not None else None,
            }
            if grp is not None:
                rec = {"group": grp, **rec}
            rows.append(rec)
    return pl.DataFrame(rows)


def weekly_att50(
    frame: pl.DataFrame,
    *,
    time_col: str = "week",
    model_qcol: Mapping[str, str] | None = None,
    actual_col: str = COST_COL,
    level: float = 0.5,
) -> pl.DataFrame:
    """Lightweight attainment at one level, bucketed by ``time_col``.

    One row per ``time_col`` × model. Use ``time_col="load_date"`` for daily,
    then :func:`smooth_att50` for a trailing weekly smoother. Prefer this over
    ``summary_metrics(..., group_col=time_col)`` when only att50 is needed.

    Implemented as a Polars ``group_by`` (not a Python loop over days) so a
    2M-row daily trend stays memory-cheap for ``make report``.
    """
    model_qcol = dict(model_qcol or MODEL_QCOL)
    q = round(level * 100)
    parts: list[pl.DataFrame] = []
    for name, tmpl in model_qcol.items():
        col = tmpl.format(q=q)
        if col not in frame.columns or actual_col not in frame.columns:
            continue
        sub = frame.filter(pl.col(col).is_finite() & pl.col(actual_col).is_finite())
        if sub.is_empty():
            continue
        parts.append(
            sub.group_by(time_col)
            .agg(
                pl.len().alias("n_loads"),
                (pl.col(actual_col) <= pl.col(col)).mean().alias("att50"),
            )
            .with_columns(
                pl.lit(name).alias("model"),
                (100 * (pl.col("att50") - level)).alias("att50_gap_pp"),
            )
        )
    if not parts:
        return pl.DataFrame(
            schema={
                time_col: frame.schema.get(time_col, pl.Date),
                "model": pl.Utf8,
                "n_loads": pl.UInt32,
                "att50": pl.Float64,
                "att50_gap_pp": pl.Float64,
            }
        )
    return (
        pl.concat(parts, how="diagonal_relaxed")
        .with_columns(
            pl.col("att50").round(4),
            pl.col("att50_gap_pp").round(2),
        )
        .select(time_col, "model", "n_loads", "att50", "att50_gap_pp")
        .sort([time_col, "model"])
    )


def smooth_att50(
    daily: pl.DataFrame,
    *,
    time_col: str = "load_date",
    window: int = 7,
    min_samples: int = 3,
    level: float = 0.5,
) -> pl.DataFrame:
    """Trailing volume-weighted rolling mean of daily att50, per model.

    ``att50_smooth = sum(att50 * n) / sum(n)`` over the last ``window`` observed
    days (index-based, so sparse weekends don't invent empty days). Adds
    ``att50_smooth`` / ``att50_smooth_gap_pp`` / ``n_loads_window`` alongside
    the raw daily columns.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    parts = []
    for model in daily["model"].unique(maintain_order=True).to_list():
        sub = daily.filter(pl.col("model") == model).sort(time_col)
        parts.append(
            sub.with_columns(
                (
                    (pl.col("att50") * pl.col("n_loads")).rolling_sum(
                        window, min_samples=min_samples
                    )
                    / pl.col("n_loads").rolling_sum(window, min_samples=min_samples)
                ).alias("att50_smooth"),
                pl.col("n_loads")
                .rolling_sum(window, min_samples=min_samples)
                .alias("n_loads_window"),
            ).with_columns(
                (100 * (pl.col("att50_smooth") - level)).alias("att50_smooth_gap_pp"),
            )
        )
    return (
        pl.concat(parts)
        .with_columns(
            pl.col("att50_smooth").round(4),
            pl.col("att50_smooth_gap_pp").round(2),
        )
        .sort([time_col, "model"])
    )
