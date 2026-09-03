"""Group-conditional split-conformal calibration of DQT alt targets.

The conformity score for a load is its implied percentile on the raw Weatherman curve (``wm_r`` from :mod:`dqt.panel`).
For a group g and nominal level a, the split-conformal choice *which Weatherman percentile to quote* is the ceil((n + 1) * a)-th order statistic of the group's calibration-window
scores. Under exchangeability this attains coverage >= a within the group (Mondrian / group-conditional conformal). Because attainment of any candidate
schedule is just ``wm_r <= alt``, counterfactual evaluation needs no curve re-interpolation.
"""

from datetime import timedelta
from math import isnan

import numpy as np
import polars as pl

from .panel import GRID
from .score.constants import DATE_COL


def conformal_alt(scores: np.ndarray, a: float) -> float:
    """Finite-sample split-conformal quantile of `scores` at level `a`.

    The ceil((n+1)*a)-th order statistic (clipped into the sample) — the
    smallest threshold guaranteeing >= a coverage on an exchangeable future
    score. NaNs are dropped; empty input returns NaN.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = np.sort(s[~np.isnan(s)])
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(int(np.ceil((n + 1) * a)), n)
    return float(s[k - 1])


def walk_forward_alts(
    panel: pl.DataFrame,
    group_cols: list[str],
    levels: tuple[float, ...] = (0.5,),
    cal_window_days: int = 28,
    min_n: int = 200,
    score_col: str = "wm_r",
) -> pl.DataFrame:
    """Per-week, per-group conformal alt schedule from a trailing window.

    For each distinct ``week`` in the panel, calibrate on loads booked in the
    `cal_window_days` days strictly before that week's start (as-of — no
    leakage into the evaluation week) and emit one row per group x level with
    the conformal alt and calibration sample size. Groups with fewer than
    `min_n` calibration loads get a null alt (caller picks the fallback).
    ``group_cols=[]`` yields the global (single-dial) schedule.
    """
    weeks = panel.get_column("week").unique().sort().to_list()
    date_col = DATE_COL if DATE_COL in panel.columns else "booked_date"
    slim = panel.select([date_col, *group_cols, score_col])
    rows: list[dict] = []
    for w in weeks:
        cal = slim.filter(
            (pl.col(date_col) >= w - timedelta(days=cal_window_days))
            & (pl.col(date_col) < w)
        )
        if group_cols:
            recs = (
                cal.group_by(group_cols)
                .agg(pl.col(score_col).alias("scores"), pl.len().alias("n_cal"))
                .iter_rows(named=True)
            )
        else:
            recs = [
                {"scores": cal.get_column(score_col).to_numpy(), "n_cal": cal.height}
            ]
        for rec in recs:
            s = np.asarray(rec["scores"], dtype=np.float64)
            ok = rec["n_cal"] >= min_n
            for a in levels:
                alt = conformal_alt(s, a) if ok else None
                if alt is not None and isnan(alt):
                    alt = None
                rows.append(
                    {
                        "week": w,
                        **{c: rec[c] for c in group_cols},
                        "level": a,
                        "n_cal": rec["n_cal"],
                        "alt": alt,
                    }
                )
    schema = {
        "week": pl.Date,
        **{c: pl.String for c in group_cols},
        "level": pl.Float64,
        "n_cal": pl.Int64,
        "alt": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def quantile_at(
    qmat: np.ndarray, r: np.ndarray, assume_sorted: bool = False
) -> np.ndarray:
    """Dollar value at percentile `r` on each row's quantile curve.

    The inverse of :func:`dqt.panel.implied_percentile`: piecewise-linear
    between grid points, scaled linearly from 0 below the 5th-percentile
    value, and clamped to the 95th-percentile value above the grid (an
    r > 0.95 cannot be realized in dollars, so quotes there are conservative).
    """
    Q = qmat if assume_sorted else np.sort(qmat, axis=1)
    r = np.asarray(r, dtype=np.float64)
    out = np.full(len(r), np.nan)
    ok = ~np.isnan(r)
    lo = ok & (r <= GRID[0])
    hi = ok & (r >= GRID[-1])
    mid = ok & ~lo & ~hi
    out[lo] = Q[lo, 0] * np.clip(r[lo] / GRID[0], 0.0, 1.0)
    out[hi] = Q[hi, -1]
    j = np.searchsorted(GRID, r[mid], side="right") - 1
    g_lo, g_hi = GRID[j], GRID[j + 1]
    out[mid] = Q[mid, j] + (Q[mid, j + 1] - Q[mid, j]) * (r[mid] - g_lo) / (g_hi - g_lo)
    return out
