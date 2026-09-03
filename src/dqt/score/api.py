"""Generalized multi-model quantile scoring: array or frame in, one long
analytics DataFrame out.

Core abstraction is `ModelSpec` — either a raw (n_loads, n_levels) prediction
matrix, or a Polars frame + column-name template (e.g. "knn_{q}", "p{q:02d}").
`compare()` runs any number of specs through the same per-level metrics
(`dqt.score.metrics`) and stacks them into one long DataFrame; `summarize()`
collapses that to one row per model (+ group columns).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from . import metrics

_LEVEL_COLS = {
    "quantile",
    "n",
    "attainment",
    "gap_pp",
    "mean_error_usd",
    "median_error_usd",
    "mae_usd",
    "mape_pct",
    "pinball_usd",
    "mean_actual",
}


@dataclass
class ModelSpec:
    """One model's quantile predictions — array-backed or frame-backed.

    Exactly one of `predicted` / `col_template` must be set. `col_template`
    is a `str.format`-style template taking `q` as the quantile level
    expressed as an integer percent, e.g. "knn_{q}" -> "knn_5", "p{q:02d}" ->
    "p05" (matching `round(level * 100)`).
    """

    name: str
    levels: Sequence[float]
    predicted: np.ndarray | None = None
    col_template: str | None = None

    def __post_init__(self) -> None:
        if (self.predicted is None) == (self.col_template is None):
            raise ValueError(
                f"ModelSpec {self.name!r} requires exactly one of "
                "`predicted` or `col_template`"
            )


def _col_for_level(col_template: str, level: float) -> str:
    return col_template.format(q=round(level * 100))


def score_array(
    levels: Sequence[float], predicted: np.ndarray, actual: np.ndarray, *, model: str
) -> pl.DataFrame:
    """One row per quantile level for one model, from raw arrays.

    `predicted` is (n_loads, n_levels); `actual` is (n_loads,).
    """
    return metrics.score_levels(
        np.asarray(levels, dtype=float), predicted, actual, model=model
    )


def score_frame(
    frame: pl.DataFrame,
    levels: Sequence[float],
    col_template: str,
    actual_col: str,
    *,
    model: str,
    group_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Same output as `score_array`, computed from frame columns via `col_template`.

    If `group_cols` is given, produces one block of per-level rows per
    distinct group, with the group columns attached (reuses whatever segment
    columns the caller's frame already has, e.g. `panel.py`'s mode/
    equipment/haul_band/geo/week/month).
    """
    levels = list(levels)
    cols = [_col_for_level(col_template, level) for level in levels]

    if not group_cols:
        predicted = frame.select(cols).to_numpy()
        actual = frame[actual_col].to_numpy()
        return score_array(levels, predicted, actual, model=model)

    group_cols = list(group_cols)
    parts = []
    for _, sub in frame.group_by(group_cols, maintain_order=True):
        group_vals = sub.select(group_cols).row(0)
        predicted = sub.select(cols).to_numpy()
        actual = sub[actual_col].to_numpy()
        scored = score_array(levels, predicted, actual, model=model)
        scored = scored.with_columns(
            [pl.lit(v).alias(c) for v, c in zip(group_vals, group_cols)]
        )
        parts.append(scored)
    return pl.concat(parts, how="diagonal_relaxed")


def compare(
    specs: dict[str, ModelSpec],
    *,
    frame: pl.DataFrame | None = None,
    actual_col: str | None = None,
    actual: np.ndarray | None = None,
    group_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Score every spec and stack the results into one long DataFrame.

    Frame-backed specs (`col_template`) require `frame` + `actual_col`, and
    are the only ones that support `group_cols`. Array-backed specs
    (`predicted`) require `actual` (a raw array) and ignore `group_cols`.
    Mixed usage is allowed — schemas are aligned diagonally, so array-backed
    rows simply get null group columns.
    """
    parts = []
    for name, spec in specs.items():
        if spec.col_template is not None:
            if frame is None or actual_col is None:
                raise ValueError(
                    f"ModelSpec {name!r} uses col_template but no frame/actual_col given"
                )
            parts.append(
                score_frame(
                    frame,
                    spec.levels,
                    spec.col_template,
                    actual_col,
                    model=name,
                    group_cols=group_cols,
                )
            )
        else:
            if actual is None:
                raise ValueError(
                    f"ModelSpec {name!r} uses predicted but no actual given"
                )
            parts.append(score_array(spec.levels, spec.predicted, actual, model=name))
    return pl.concat(parts, how="diagonal_relaxed")


def summarize(long_df: pl.DataFrame) -> pl.DataFrame:
    """Collapse per-quantile-level rows to one row per model (+ group columns).

    ece_pp = mean(|gap_pp|); crps_usd = 2 * mean(pinball_usd) (the repo's
    existing CRPS approximation); crps_pct_cost = crps_usd as a % of mean
    realized cost; max_gap_pp = the signed gap with the largest magnitude;
    worst_gap_pp = max(|gap_pp|) (always ≥ 0 — absolute worst calibration
    miss on the grid); worst_q = nominal quantile where that worst miss
    occurs (e.g. 0.05 / 0.95 for tail failures); bias_usd / mae_usd = the
    curve-average of the per-level dollar bias/MAE.
    """
    group_cols = [c for c in long_df.columns if c not in ({"model"} | _LEVEL_COLS)]
    keys = ["model", *group_cols]
    ranked = long_df.with_columns(pl.col("gap_pp").abs().alias("_abs_gap"))
    return (
        ranked.group_by(keys, maintain_order=True)
        .agg(
            pl.col("n").max().alias("n_loads"),
            pl.col("mean_actual").first().alias("_mean_actual"),
            pl.col("gap_pp").abs().mean().alias("ece_pp"),
            (2 * pl.col("pinball_usd").mean()).alias("crps_usd"),
            pl.col("mean_error_usd").mean().alias("bias_usd"),
            pl.col("mae_usd").mean().alias("mae_usd"),
            pl.col("gap_pp")
            .sort_by("_abs_gap", descending=True)
            .first()
            .alias("max_gap_pp"),
            pl.col("_abs_gap").max().alias("worst_gap_pp"),
            pl.col("quantile")
            .sort_by("_abs_gap", descending=True)
            .first()
            .alias("worst_q"),
        )
        .with_columns(
            (100 * pl.col("crps_usd") / pl.col("_mean_actual")).alias("crps_pct_cost")
        )
        .drop("_mean_actual")
        .sort(keys)
    )
