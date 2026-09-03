"""Hybrid DQT policy: shipped daily dial + slow-moving per-group conformal offset.

The dial supplies daily market reactivity; the per-group conformal offset
(``alt*_cell - alt*_global``, from a trailing walk-forward calibration window)
corrects persistent cross-segment bias the dial can't see on its own.
"""

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .conformal import quantile_at, walk_forward_alts
from .panel import ALT_COLS, GRID, KNN_COLS
from .score.constants import ID_COL


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Settings for :func:`hybrid_quantiles` / :func:`hybrid_policy`."""

    group_cols: tuple[str, ...] = ("mode", "equipment", "haul_band")
    levels: tuple[float, ...] = tuple(GRID.tolist())
    cal_window_days: int = 28
    min_n: int = 200
    clip: tuple[float, float] = (0.01, 0.99)
    max_weekly_delta: float | None = None
    score_col: str = "wm_r"
    output_path: str = "data/hybrid_quantiles.parquet"

    def __post_init__(self) -> None:
        if self.cal_window_days <= 0:
            raise ValueError(
                f"cal_window_days must be positive, got {self.cal_window_days}"
            )
        if self.min_n <= 0:
            raise ValueError(f"min_n must be positive, got {self.min_n}")
        lo, hi = self.clip
        if not (0 < lo < hi < 1):
            raise ValueError(f"clip must satisfy 0 < lo < hi < 1, got {self.clip}")
        if not self.levels:
            raise ValueError("levels must be non-empty")
        if any(not (0 < a < 1) for a in self.levels):
            raise ValueError(f"all levels must be in (0, 1), got {self.levels}")
        if self.max_weekly_delta is not None and self.max_weekly_delta <= 0:
            raise ValueError(
                f"max_weekly_delta must be positive if set, got {self.max_weekly_delta}"
            )


def hybrid_policy(
    panel: pl.DataFrame,
    grp_alts: pl.DataFrame,
    glob_alts: pl.DataFrame,
    group_cols: list[str],
    level: float,
    clip: tuple[float, float] = (0.01, 0.99),
) -> pl.DataFrame:
    """Shipped dial + per-group conformal offset, at a single percentile level.

    ``delta = alt*_cell - alt*_global`` (left-joined, so a cell-week missing
    from the calibration schedule falls back to ``delta=0``, i.e. the shipped
    dial unchanged, never a dropped row). ``alt_used = clip(dial + delta)``,
    with the dial itself falling back to the nominal ``level`` if that week
    has no schedule at all. Adds ``alt_used`` and boolean ``attained``.
    """
    deltas = (
        grp_alts.join(
            glob_alts.select("week", "level", pl.col("alt").alias("alt_glob")),
            on=["week", "level"],
            how="left",
        )
        .with_columns((pl.col("alt") - pl.col("alt_glob")).alias("delta"))
        .filter(pl.col("level") == level)
        .select(["week", *group_cols, "delta"])
    )
    dial = pl.col(f"alt_{int(round(level * 100))}")
    return (
        panel.join(deltas, on=["week", *group_cols], how="left")
        .with_columns(
            (dial.fill_null(level) + pl.col("delta").fill_null(0.0))
            .clip(*clip)
            .alias("alt_used")
        )
        .with_columns((pl.col("wm_r") <= pl.col("alt_used")).alias("attained"))
    )


def cap_weekly_delta(
    deltas: pl.DataFrame, group_cols: list[str], max_delta: float
) -> pl.DataFrame:
    """Sequentially clip week-over-week ``|delta|`` moves per ``(level, *group_cols)``.

    ``capped[t] = capped[t-1] + clip(raw[t] - capped[t-1], -max_delta, +max_delta)``.
    A null raw delta (thin cell-week, below ``min_n``) or a null previous capped
    value resets the chain to the raw value — capping only applies once there is
    a real prior offset to move away from.
    """
    key_cols = ["level", *group_cols]
    sorted_deltas = deltas.sort(["week", *key_cols])
    out = []
    for _, grp in sorted_deltas.group_by(key_cols, maintain_order=True):
        grp = grp.sort("week")
        raw = grp.get_column("delta").to_numpy().astype(np.float64)
        capped = np.empty_like(raw)
        prev = np.nan
        for i, r in enumerate(raw):
            capped[i] = (
                r
                if (np.isnan(prev) or np.isnan(r))
                else prev + np.clip(r - prev, -max_delta, max_delta)
            )
            prev = capped[i]
        # NaN here only ever means "null" (a reset point, never a real value),
        # so map it back to a genuine polars null rather than a float NaN —
        # otherwise downstream .fill_null(0.0) calls would silently miss it.
        capped_col = [None if np.isnan(v) else float(v) for v in capped]
        out.append(grp.with_columns(pl.Series("delta", capped_col, dtype=pl.Float64)))
    return pl.concat(out).sort(["week", *key_cols])


def hybrid_quantiles(
    panel: pl.DataFrame,
    config: HybridConfig,
    root: Path | None = None,
    *,
    data_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Full-grid hybrid dollar quantiles, one row per scored load.

    Generalizes ``WeeklySimulation.ipynb`` cell 19's pure-subgroup full-grid pattern
    to the hybrid (dial + offset) combination: walk-forward calibrate group
    and global schedules, take their delta (optionally capped), add it to the
    shipped dial, clip into percentile space, then invert every level to
    dollars via the load's own full Weatherman curve. Returns ``loadnumber``
    plus zero-padded ``hybrid_05..hybrid_95`` (or whichever levels
    ``config.levels`` requests), directly comparable to ``pXX``/``knn_XX``/
    ``new_dqt_XX``.

    The panel must already carry ``alt_*`` and full ``knn_*`` (see
    :func:`dqt.panel.build_panel`).
    """
    _ = (root, data_dir)
    group_cols = list(config.group_cols)
    levels = tuple(config.levels)
    labels = [f"{int(round(a * 100)):02d}" for a in levels]
    missing = [c for c in (*ALT_COLS, *KNN_COLS) if c not in panel.columns]
    if missing:
        raise ValueError(
            f"hybrid_quantiles: panel missing {missing[:6]}"
            f"{'…' if len(missing) > 6 else ''} — rebuild the panel "
            "(make hybrid FORCE=1)"
        )

    alts_grp = walk_forward_alts(
        panel,
        group_cols,
        levels=levels,
        cal_window_days=config.cal_window_days,
        min_n=config.min_n,
        score_col=config.score_col,
    )
    alts_glob = walk_forward_alts(
        panel,
        [],
        levels=levels,
        cal_window_days=config.cal_window_days,
        min_n=config.min_n,
        score_col=config.score_col,
    )

    deltas = alts_grp.join(
        alts_glob.select("week", "level", pl.col("alt").alias("alt_glob")),
        on=["week", "level"],
        how="left",
    ).with_columns((pl.col("alt") - pl.col("alt_glob")).alias("delta"))
    deltas = deltas.select(["week", *group_cols, "level", "delta"])
    if config.max_weekly_delta is not None:
        deltas = cap_weekly_delta(deltas, group_cols, config.max_weekly_delta)

    lab_expr = (
        (pl.col("level") * 100).round(0).cast(pl.Int32).cast(pl.String).str.zfill(2)
    ).alias("lab")
    deltas_wide = (
        deltas.with_columns(lab_expr)
        .pivot(on="lab", index=["week", *group_cols], values="delta")
        .rename({lab: f"delta_{lab}" for lab in labels})
    )

    # Burn-in (insufficient calibration history) is level-independent, so any
    # single level's non-null global schedule identifies the usable eval weeks.
    ref_level = min(levels, key=lambda a: abs(a - 0.5))
    eval_weeks = (
        alts_glob.filter((pl.col("level") == ref_level) & pl.col("alt").is_not_null())
        .get_column("week")
        .to_list()
    )

    joined = panel.filter(pl.col("week").is_in(eval_weeks)).join(
        deltas_wide, on=["week", *group_cols], how="left"
    )
    alt_used_exprs = [
        (
            pl.col(f"alt_{int(round(a * 100))}").fill_null(a)
            + pl.col(f"delta_{lab}").fill_null(0.0)
        )
        .clip(*config.clip)
        .alias(f"alt_used_{lab}")
        for a, lab in zip(levels, labels)
    ]
    joined = joined.with_columns(alt_used_exprs)

    # Per-level offsets are calibrated independently, so alt_used can cross
    # between adjacent levels — observed in practice at the thin-calibration
    # upper tail (p85-p95). Rearrangement (sort each row's values into level
    # order, same fix `panel.implied_percentile` applies to quantile curves)
    # restores monotonicity before inversion to dollars.
    sorted_pairs = sorted(zip(levels, labels))
    sorted_labels = [lab for _, lab in sorted_pairs]
    alt_used_cols = [f"alt_used_{lab}" for lab in sorted_labels]
    alt_used_mat = np.sort(joined.select(alt_used_cols).to_numpy(), axis=1)
    joined = joined.with_columns(
        [
            pl.Series(f"alt_used_{lab}", alt_used_mat[:, i])
            for i, lab in enumerate(sorted_labels)
        ]
    )

    # A handful of loads (~19 of 2.67M, a raw-Weatherman data anomaly) have a
    # negative knn_5 — a nonsensical p05 shipment-cost prediction. quantile_at's
    # below-grid extrapolation (Q[:, 0] * clip(r / GRID[0], 0, 1)) assumes a
    # non-negative anchor and is not monotonic in r otherwise, so these
    # degenerate curves are excluded rather than allowed to fail the
    # monotonicity check below.
    joined = joined.filter(pl.col("knn_5") >= 0)
    Qmat = np.sort(joined.select(KNN_COLS).to_numpy(), axis=1)

    hybrid_series = [
        pl.Series(
            f"hybrid_{lab}",
            quantile_at(
                Qmat,
                joined.get_column(f"alt_used_{lab}").to_numpy(),
                assume_sorted=True,
            ),
        )
        for a, lab in zip(levels, labels)
    ]
    joined = joined.with_columns(hybrid_series)

    for a, lab in zip(levels, labels):
        d_att = float((joined["cost"] <= joined[f"hybrid_{lab}"]).mean())
        r_att = float((joined["wm_r"] <= joined[f"alt_used_{lab}"]).mean())
        if abs(d_att - r_att) >= 2e-3:
            raise ValueError(
                f"hybrid_quantiles: dollar/percentile attainment mismatch at "
                f"level {a}: {d_att:.4f} (dollar) vs {r_att:.4f} (percentile)"
            )
    for (lo_a, lo_lab), (hi_a, hi_lab) in itertools.pairwise(sorted_pairs):
        ok = joined.select(
            (pl.col(f"hybrid_{hi_lab}") >= pl.col(f"hybrid_{lo_lab}")).all()
        ).item()
        if not ok:
            raise ValueError(
                f"hybrid_quantiles: quantile crossing between levels {lo_a} and {hi_a}"
            )

    out_cols = [ID_COL, *[f"hybrid_{lab}" for _, lab in sorted_pairs]]
    return joined.select(out_cols)


def cell_gap_from_loads(
    frame: pl.DataFrame,
    *,
    level: float = 0.5,
    cost_col: str = "cost",
    actual_col: str | None = None,
    hybrid_col: str | None = None,
    group_cols: tuple[str, ...] = ("mode", "equipment", "haul_band"),
    min_n: int = 1,
) -> pl.DataFrame:
    """Per-cell Actual DQT vs Hybrid gaps from a load-level slice (e.g. one week).

    Dollar attainment: ``cost <= quote`` at ``level`` (default p50 / hybrid_50).
    Improvement is ``|actual_gap| − |hybrid_gap|`` (positive = hybrid closer).
    """
    from .score.constants import COST_COL

    if cost_col == "cost" and cost_col not in frame.columns and COST_COL in frame.columns:
        cost_col = COST_COL
    lab = f"{int(round(level * 100)):02d}"
    actual_col = actual_col or f"p{lab}"
    hybrid_col = hybrid_col or f"hybrid_{lab}"
    required = {cost_col, actual_col, hybrid_col, *group_cols}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cell_gap_from_loads: frame missing columns {sorted(missing)}")

    scored = (
        frame.drop_nulls([cost_col, actual_col, hybrid_col, *group_cols])
        .with_columns(
            pl.concat_str([pl.col(c) for c in group_cols], separator=" \u00b7 ").alias(
                "cell"
            ),
            (pl.col(cost_col) <= pl.col(actual_col)).alias("_actual_hit"),
            (pl.col(cost_col) <= pl.col(hybrid_col)).alias("_hybrid_hit"),
        )
        .group_by("cell")
        .agg(
            pl.len().alias("n"),
            pl.col("_actual_hit").mean().alias("actual_att"),
            pl.col("_hybrid_hit").mean().alias("hybrid_att"),
        )
        .filter(pl.col("n") >= min_n)
    )
    if scored.is_empty():
        return pl.DataFrame(
            schema={
                "cell": pl.String,
                "n": pl.UInt32,
                "actual_gap_pp": pl.Float64,
                "hybrid_gap_pp": pl.Float64,
                "improvement_pp": pl.Float64,
            }
        )
    return (
        scored.with_columns(
            (pl.col("actual_att") - level).alias("actual_gap"),
            (pl.col("hybrid_att") - level).alias("hybrid_gap"),
        )
        .with_columns(
            (pl.col("actual_gap").abs() - pl.col("hybrid_gap").abs()).alias(
                "improvement"
            )
        )
        .select(
            "cell",
            "n",
            (100 * pl.col("actual_gap")).round(1).alias("actual_gap_pp"),
            (100 * pl.col("hybrid_gap")).round(1).alias("hybrid_gap_pp"),
            (100 * pl.col("improvement")).round(1).alias("improvement_pp"),
        )
        .sort("improvement_pp", descending=True)
    )
