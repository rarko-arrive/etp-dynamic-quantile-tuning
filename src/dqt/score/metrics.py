"""Per-quantile-level scoring formulas — pure numpy, one model at a time.

Each function operates on a single quantile level's predictions (`predicted_q`,
shape (n,)) against the realized cost (`actual`, shape (n,)). `score_levels`
loops these over a model's full set of levels and returns one row per level.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def attainment(predicted_q: np.ndarray, actual: np.ndarray) -> float:
    """Share of loads whose realized cost is <= the quoted value."""
    return float(np.mean(actual <= predicted_q))


def gap_pp(attain: float, level: float) -> float:
    """Calibration gap in percentage points: (observed - nominal) * 100."""
    return (attain - level) * 100.0


def mean_error_usd(predicted_q: np.ndarray, actual: np.ndarray) -> float:
    """Signed mean dollar bias: mean(predicted - actual)."""
    return float(np.mean(predicted_q - actual))


def median_error_usd(predicted_q: np.ndarray, actual: np.ndarray) -> float:
    """Signed median dollar bias: median(predicted - actual)."""
    return float(np.median(predicted_q - actual))


def mae_usd(predicted_q: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute dollar error: mean(|predicted - actual|)."""
    return float(np.mean(np.abs(predicted_q - actual)))


def mape_pct(predicted_q: np.ndarray, actual: np.ndarray) -> float:
    """Median absolute percentage error, in percent (matches the repo's
    existing median-APE convention, e.g. notebook 04's `agg_bias`)."""
    return float(np.median(np.abs(predicted_q - actual) / actual) * 100.0)


def pinball_usd(predicted_q: np.ndarray, actual: np.ndarray, level: float) -> float:
    """Mean pinball (quantile) loss at `level`, in dollars.

    Same formula as the repo's prior `evals._pinball`: for diff = actual -
    predicted, loss = level * diff when diff >= 0, else (level - 1) * diff.
    """
    diff = actual - predicted_q
    return float(np.mean(np.where(diff >= 0, level * diff, (level - 1) * diff)))


def score_levels(
    levels: np.ndarray, predicted: np.ndarray, actual: np.ndarray, *, model: str
) -> pl.DataFrame:
    """One row per quantile level for one model.

    `predicted` is (n_loads, n_levels); `actual` is (n_loads,); `levels` is
    (n_levels,) ascending fractions (e.g. 0.05..0.95). Rows with a null in
    either `predicted[:, j]` or `actual` are dropped per-level (n can differ
    across levels if null patterns differ).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    mean_actual = float(np.nanmean(actual))

    rows = []
    for j, level in enumerate(levels):
        p = predicted[:, j]
        mask = ~np.isnan(p) & ~np.isnan(actual)
        pm, am = p[mask], actual[mask]
        att = attainment(pm, am)
        rows.append(
            {
                "model": model,
                "quantile": float(level),
                "n": int(mask.sum()),
                "attainment": att,
                "gap_pp": gap_pp(att, level),
                "mean_error_usd": mean_error_usd(pm, am),
                "median_error_usd": median_error_usd(pm, am),
                "mae_usd": mae_usd(pm, am),
                "mape_pct": mape_pct(pm, am),
                "pinball_usd": pinball_usd(pm, am, level),
                "mean_actual": mean_actual,
            }
        )
    return pl.DataFrame(rows)
