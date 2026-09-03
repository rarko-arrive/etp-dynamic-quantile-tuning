"""Walk-forward SARIMA global dial on daily ``median(wm_r)`` + ``sarima_pp`` quotes.

Forecasts tomorrow's Weatherman implied-percentile median (the dial that would
hit ~50% attainment that day), then materializes a rigid identity+δ dollar
surface via :func:`dqt.conformal.quantile_at` for scoring against shipped DQT
and Hybrid.

Production quote policy: dollar alts are clamped to the Weatherman observation
grid ``[0.05, 0.95]`` (see :data:`QUOTE_CLIP_DEFAULT`) so a down-shifted dial
never invents below-``knn_5`` scale-to-zero dollars. Dial forecast ``r̂`` still
uses the wider :data:`CLIP_DEFAULT` for the time-series path only.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
from loguru import logger
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import parse_date_col, resolve_data_dir
from .conformal import quantile_at, walk_forward_alts
from .holidays import resolve_occurrences
from .hybrid import HybridConfig, cap_weekly_delta, cell_gap_from_loads
from .panel import GRID, WEEKDAYS
from .score import summary_metrics
from .score.api import compare
from .score.constants import COST_COL, NOMINAL
from .score.summary import model_specs

# Dial forecast clip (walk-forward r̂). Wider than the Weatherman knn grid.
CLIP_DEFAULT = (0.01, 0.99)
# Dollar-quote clip: stay on the observed knn_5..knn_95 grid. Clipping alts to
# 0.01 and letting quantile_at scale knn_5*(alt/0.05) produces pathological
# left-tail quotes when r̂ ≲ 0.46 (e.g. ~0% p05 attainment).
QUOTE_CLIP_DEFAULT = (float(GRID[0]), float(GRID[-1]))
ORDER_DEFAULT = (1, 1, 0)
SEASONAL_DEFAULT = (2, 0, 0, 5)
MIN_HISTORY_DEFAULT = 120
EVAL_FRAC_DEFAULT = 1.0 / 3.0
ATTAIN_TOL = 2e-3

DIAL_PRED_COLS = ("y_hat_sarima", "y_hat_sarima_cal", "y_hat_naive")


def rigid_quote_alts(
    levels: Sequence[float],
    delta: np.ndarray,
    *,
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
) -> np.ndarray:
    """``alt_a = clip(a + δ, clip)`` for each level; shape ``(n_loads, n_levels)``.

    ``delta`` is per-load (or broadcastable) ``r̂ − 0.50``. Default ``clip`` is
    the Weatherman grid so production quotes never leave ``[knn_5, knn_95]``.
    """
    levels_arr = np.asarray(levels, dtype=np.float64)
    delta_arr = np.asarray(delta, dtype=np.float64).reshape(-1, 1)
    return np.clip(levels_arr.reshape(1, -1) + delta_arr, clip[0], clip[1])


def shipped_alt_col(level: float, columns: set[str] | None = None) -> str:
    """Panel / ``dqt_alt_percentiles`` name: ``alt_5``, ``alt_50`` (not zero-padded)."""
    n = int(round(float(level) * 100))
    candidates = (f"alt_{n}", f"alt_{n:02d}")
    if columns is None:
        return candidates[0]
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"shipped_alt_col: no alt column for level {level} in {sorted(candidates)}"
    )


def blend_quote_alts(
    shipped: np.ndarray,
    r_hat: np.ndarray,
    shipped_50: np.ndarray,
    *,
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
) -> np.ndarray:
    """Keep shipped tail spacing; replace the center with ``r̂``.

    ``alt_a = clip(shipped_alt_a + (r̂ − shipped_alt_50), clip)``.
    ``shipped`` is ``(n_loads, n_levels)``; ``r_hat`` / ``shipped_50`` are
    per-load. p50 equals ``r̂`` before clip.
    """
    shipped_arr = np.asarray(shipped, dtype=np.float64)
    r = np.asarray(r_hat, dtype=np.float64).reshape(-1, 1)
    a50 = np.asarray(shipped_50, dtype=np.float64).reshape(-1, 1)
    return np.clip(shipped_arr + (r - a50), clip[0], clip[1])


def tail_quote_alts(
    base: np.ndarray,
    delta_cell: np.ndarray,
    alt_glob: np.ndarray,
    tail_mask: np.ndarray,
    *,
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
) -> np.ndarray:
    """Blend/rigid base + cell δ + residual global tail offset.

    ``δ_tail_a = alt_glob_a − base_a`` on tail levels (else 0), so a tail
    quote becomes ``clip(alt_glob_a + δ_cell_a)`` — residual, not a second
    dial. Mid-grid stays ``clip(base_a + δ_cell_a)``.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    cell = np.asarray(delta_cell, dtype=np.float64)
    glob = np.asarray(alt_glob, dtype=np.float64)
    mask = np.asarray(tail_mask, dtype=bool).reshape(1, -1)
    d_tail = np.where(mask, glob - base_arr, 0.0)
    d_tail = np.where(np.isfinite(d_tail), d_tail, 0.0)
    return np.clip(base_arr + cell + d_tail, clip[0], clip[1])


def daily_wm_r_median(panel: pl.DataFrame) -> pl.DataFrame:
    """Weekday-only daily ``median(wm_r)`` series from the attainment panel.

    Returns columns ``booked_date``, ``y``, ``n_loads``, and ``alt_50`` when
    present on the panel (mean shipped dial that day).
    """
    if "booked_date" not in panel.columns:
        raise ValueError("daily_wm_r_median: panel missing booked_date")
    if "wm_r" not in panel.columns:
        raise ValueError("daily_wm_r_median: panel missing wm_r")

    dow = panel["booked_date"].dt.strftime("%a")
    weekday = panel.filter(dow.is_in(list(WEEKDAYS)))
    aggs = [
        pl.col("wm_r").median().alias("y"),
        pl.len().alias("n_loads"),
    ]
    if "alt_50" in weekday.columns:
        aggs.append(pl.col("alt_50").drop_nulls().mean().alias("alt_50"))
    out = (
        weekday.group_by("booked_date")
        .agg(aggs)
        .sort("booked_date")
        .drop_nulls("y")
    )
    return out


def daily_wm_r_median_settlement_lag(
    panel: pl.DataFrame,
    settlement_lag_days: int = 3,
) -> pl.DataFrame:
    """Daily ``median(wm_r)`` indexed by *publication* date (booked + lag).

    Proxy for live ops: realized costs for loads booked on day ``d`` are treated
    as known starting ``d + settlement_lag_days``. The returned ``booked_date``
    column is that publication date (what the dial forecaster sees on its
    calendar). Original book date is kept as ``obs_booked_date``.
    """
    if settlement_lag_days < 0:
        raise ValueError(f"settlement_lag_days must be >= 0, got {settlement_lag_days}")
    base = daily_wm_r_median(panel)
    if settlement_lag_days == 0:
        return base.with_columns(pl.col("booked_date").alias("obs_booked_date"))
    out = base.with_columns(
        pl.col("booked_date").alias("obs_booked_date"),
        pl.col("booked_date")
        .dt.offset_by(f"{settlement_lag_days}d")
        .alias("booked_date"),
    ).sort("booked_date")
    # Drop publication dates that land on weekends (no weekday dial).
    dow = out["booked_date"].dt.strftime("%a")
    return out.filter(dow.is_in(list(WEEKDAYS)))


def settlement_coverage_daily(
    panel: pl.DataFrame,
    dial: pl.DataFrame,
    *,
    settlement_lag_days: int = 3,
) -> pl.DataFrame:
    """Per forecast weekday: share of prior weekday loads settled by open.

    Without invoice timestamps we use a lag proxy: loads booked on day ``d``
    count as settled on ``d + settlement_lag_days``. For forecast day ``t``,
    coverage = settled share of loads booked on the prior weekday.
    """
    if settlement_lag_days < 0:
        raise ValueError(f"settlement_lag_days must be >= 0, got {settlement_lag_days}")
    if "booked_date" not in panel.columns:
        raise ValueError("settlement_coverage_daily: panel missing booked_date")

    dow = panel["booked_date"].dt.strftime("%a")
    weekday_loads = panel.filter(dow.is_in(list(WEEKDAYS))).select(
        "booked_date", "loadnumber"
    )
    daily_n = weekday_loads.group_by("booked_date").agg(pl.len().alias("n_loads"))

    dates = dial.sort("booked_date")["booked_date"].to_list()
    rows: list[dict] = []
    for i, t in enumerate(dates):
        if i == 0:
            rows.append(
                {
                    "booked_date": t,
                    "prior_booked_date": None,
                    "n_prior": 0,
                    "n_settled": 0,
                    "coverage_pct": None,
                    "settlement_lag_days": settlement_lag_days,
                }
            )
            continue
        prior = dates[i - 1]
        n_prior_row = daily_n.filter(pl.col("booked_date") == prior)
        n_prior = int(n_prior_row["n_loads"][0]) if n_prior_row.height else 0
        settled_by = prior + timedelta(days=settlement_lag_days)
        n_settled = n_prior if settled_by <= t else 0
        cov = (100.0 * n_settled / n_prior) if n_prior else None
        rows.append(
            {
                "booked_date": t,
                "prior_booked_date": prior,
                "n_prior": n_prior,
                "n_settled": n_settled,
                "coverage_pct": round(cov, 1) if cov is not None else None,
                "settlement_lag_days": settlement_lag_days,
            }
        )
    return pl.DataFrame(rows)


def quantile_level_scorecard(
    frame: pl.DataFrame,
    *,
    model_qcol: Mapping[str, str],
    actual_col: str = COST_COL,
    levels: Sequence[float] = NOMINAL,
) -> pl.DataFrame:
    """Per-level attainment and gap (pp) for each model."""
    specs = model_specs(model_qcol, levels)
    long = compare(specs, frame=frame, actual_col=actual_col)
    return (
        long.select(
            "model",
            "quantile",
            "n",
            "attainment",
            pl.col("gap_pp").alias("gap_pp"),
            (pl.col("attainment") * 100).round(2).alias("att_pct"),
        )
        .with_columns(
            pl.col("quantile").round(2),
            pl.col("gap_pp").round(2),
        )
        .sort("model", "quantile")
    )


def calendar_flags(
    dates: Sequence[date],
    *,
    holiday_dates: set[date] | None = None,
) -> pl.DataFrame:
    """Monday / Friday / holiday flags for each booked date (known at open)."""
    if holiday_dates is None:
        years = {d.year for d in dates}
        if years:
            occ = resolve_occurrences(
                year_start=min(years), year_end=max(years)
            )
            holiday_dates = set(occ["observed_date"].to_list()) if occ.height else set()
        else:
            holiday_dates = set()
    rows = []
    for d in dates:
        rows.append(
            {
                "booked_date": d,
                "is_monday": int(d.weekday() == 0),
                "is_friday": int(d.weekday() == 4),
                "is_holiday": int(d in holiday_dates),
            }
        )
    return pl.DataFrame(rows)


# Optional injectable for unit tests: (y_hist, order, seasonal) -> (y_hat, resid)
SarimaStep = Callable[
    [np.ndarray, tuple[int, int, int], tuple[int, int, int, int]],
    tuple[float, np.ndarray],
]


def _default_sarima_step(
    y_hist: np.ndarray,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int, int],
) -> tuple[float, np.ndarray]:
    """One SARIMAX fit → 1-step forecast and in-sample residuals."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_hist,
            order=order,
            seasonal_order=seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=50)
        pm = res.get_forecast(1).predicted_mean
        pred = float(np.asarray(pm).ravel()[0])
        fitted = np.asarray(res.fittedvalues, dtype=np.float64)
    resid = y_hist.astype(np.float64) - fitted
    return pred, resid


def _calendar_correction(
    resid_hist: np.ndarray,
    cal_hist: np.ndarray,
    cal_t: np.ndarray,
    *,
    min_obs: int = 30,
) -> float:
    """OLS residual ~ calendar on past rows; predict correction for day t.

    ``cal_*`` columns are ``[is_monday, is_friday, is_holiday]``.
    """
    ok = np.isfinite(resid_hist) & np.all(np.isfinite(cal_hist), axis=1)
    if int(ok.sum()) < min_obs:
        return 0.0
    y = resid_hist[ok]
    X = sm.add_constant(cal_hist[ok], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ols = sm.OLS(y, X).fit()
        except (ValueError, np.linalg.LinAlgError):
            return 0.0
    x_t = sm.add_constant(cal_t.reshape(1, -1), has_constant="add")
    return float(np.asarray(ols.predict(x_t)).ravel()[0])


def walk_forward_sarima_dial(
    series: pl.DataFrame,
    *,
    order: tuple[int, int, int] = ORDER_DEFAULT,
    seasonal: tuple[int, int, int, int] = SEASONAL_DEFAULT,
    min_history: int = MIN_HISTORY_DEFAULT,
    eval_frac: float = EVAL_FRAC_DEFAULT,
    clip: tuple[float, float] = CLIP_DEFAULT,
    step_fn: SarimaStep | None = None,
    progress_every: int = 50,
    prior_dial: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Expanding-window 1-step SARIMA + as-of calendar residual correction.

    For each weekday ``t`` with at least ``min_history`` prior observations,
    fits only on ``y[0:t]`` (exclusive of day ``t``), forecasts ``y_hat``,
    then adds an OLS calendar correction fit on past SARIMA residuals only.

    ``in_eval`` marks the final ``eval_frac`` of the series (holdout).

    When ``prior_dial`` is provided, dates with a non-null ``y_hat_sarima_cal``
    reuse those hats (immutable history) and skip the SARIMAX refit — used by
    weekly appends on ``data/current``.
    """
    if min_history < 1:
        raise ValueError(f"min_history must be >= 1, got {min_history}")
    if not (0.0 < eval_frac < 1.0):
        raise ValueError(f"eval_frac must be in (0, 1), got {eval_frac}")
    required = {"booked_date", "y"}
    missing = required - set(series.columns)
    if missing:
        raise ValueError(f"walk_forward_sarima_dial: missing {sorted(missing)}")

    s = series.sort("booked_date")
    dates = s["booked_date"].to_list()
    y = s["y"].to_numpy().astype(np.float64)
    n = len(y)
    if n == 0:
        return pl.DataFrame(
            schema={
                "booked_date": pl.Date,
                "y": pl.Float64,
                "n_loads": pl.Int64,
                "y_hat_sarima": pl.Float64,
                "y_hat_sarima_cal": pl.Float64,
                "y_hat_naive": pl.Float64,
                "in_eval": pl.Boolean,
            }
        )

    cal = calendar_flags(dates)
    cal_mat = cal.select("is_monday", "is_friday", "is_holiday").to_numpy().astype(
        np.float64
    )
    step = step_fn or _default_sarima_step

    eval_start = int(np.floor(n * (1.0 - eval_frac)))
    y_hat = np.full(n, np.nan)
    y_hat_cal = np.full(n, np.nan)
    y_hat_naive = np.full(n, np.nan)
    y_hat_naive[1:] = y[:-1]

    prior_by_date: dict[object, tuple[float, float, float]] = {}
    if prior_dial is not None and not prior_dial.is_empty():
        for row in prior_dial.iter_rows(named=True):
            cal_hat = row.get("y_hat_sarima_cal")
            if cal_hat is None or (
                isinstance(cal_hat, float) and not np.isfinite(cal_hat)
            ):
                continue
            prior_by_date[row["booked_date"]] = (
                float(row["y_hat_sarima"])
                if row.get("y_hat_sarima") is not None
                and np.isfinite(row["y_hat_sarima"])
                else float("nan"),
                float(cal_hat),
                float(row["y_hat_naive"])
                if row.get("y_hat_naive") is not None
                and np.isfinite(row["y_hat_naive"])
                else float("nan"),
            )

    n_fit = 0
    for t in range(min_history, n):
        if dates[t] in prior_by_date:
            hs, hc, hn = prior_by_date[dates[t]]
            y_hat[t] = hs
            y_hat_cal[t] = hc
            if np.isfinite(hn):
                y_hat_naive[t] = hn
            continue

        y_hist = y[:t]
        try:
            pred, resid_hist = step(y_hist, order, seasonal)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            logger.warning("SARIMA fit failed at t={} ({}): {}", t, dates[t], exc)
            continue
        if not np.isfinite(pred):
            continue
        pred = float(np.clip(pred, clip[0], clip[1]))
        y_hat[t] = pred

        if resid_hist is None or len(resid_hist) != t:
            resid_hist = np.full(t, np.nan)
        corr = _calendar_correction(resid_hist, cal_mat[:t], cal_mat[t])
        y_hat_cal[t] = float(np.clip(pred + corr, clip[0], clip[1]))
        n_fit += 1

        if progress_every and n_fit % progress_every == 0:
            logger.info(
                "walk-forward dial fit {} (t={}/{}, {})",
                n_fit,
                t + 1,
                n,
                dates[t],
            )

    if prior_by_date:
        logger.info(
            "walk-forward dial: reused {} prior days, fitted {}",
            len(prior_by_date),
            n_fit,
        )

    out_cols: dict[str, object] = {
        "booked_date": dates,
        "y": y,
        "y_hat_sarima": y_hat,
        "y_hat_sarima_cal": y_hat_cal,
        "y_hat_naive": y_hat_naive,
        "in_eval": [i >= eval_start for i in range(n)],
    }
    if "n_loads" in s.columns:
        out_cols["n_loads"] = s["n_loads"].to_list()
    if "alt_50" in s.columns:
        out_cols["alt_50"] = s["alt_50"].to_list()

    return pl.DataFrame(out_cols).with_columns(
        pl.col("booked_date").cast(pl.Date),
        pl.col("y").cast(pl.Float64),
        pl.col("y_hat_sarima").cast(pl.Float64),
        pl.col("y_hat_sarima_cal").cast(pl.Float64),
        pl.col("y_hat_naive").cast(pl.Float64),
        pl.col("in_eval").cast(pl.Boolean),
    )


def extend_walk_forward_sarima_dial(
    series: pl.DataFrame,
    prior_dial: pl.DataFrame,
    **kwargs,
) -> pl.DataFrame:
    """Append forecasts for new weekdays; keep prior ``y_hat_*`` immutable."""
    return walk_forward_sarima_dial(series, prior_dial=prior_dial, **kwargs)


def dial_errors_pp(
    dial: pl.DataFrame,
    *,
    pred_col: str = "y_hat_sarima_cal",
    actual_col: str = "y",
    eval_only: bool = True,
) -> dict[str, float]:
    """MAE / RMSE in percentage points on the (eval) dial window."""
    df = dial
    if eval_only and "in_eval" in df.columns:
        df = df.filter(pl.col("in_eval"))
    df = df.filter(
        pl.col(pred_col).is_not_null()
        & pl.col(pred_col).is_not_nan()
        & pl.col(actual_col).is_not_null()
        & pl.col(actual_col).is_not_nan()
    )
    if df.is_empty():
        return {"n": 0.0, "mae_pp": float("nan"), "rmse_pp": float("nan")}
    err = (df[pred_col] - df[actual_col]).to_numpy() * 100.0
    return {
        "n": float(len(err)),
        "mae_pp": float(np.mean(np.abs(err))),
        "rmse_pp": float(np.sqrt(np.mean(err**2))),
    }


def dial_scorecard(dial: pl.DataFrame, *, eval_only: bool = True) -> pl.DataFrame:
    """One row per predictor with MAE/RMSE pp and lift vs naive."""
    naive = dial_errors_pp(
        dial, pred_col="y_hat_naive", eval_only=eval_only
    )
    rows = []
    for col, name in [
        ("y_hat_naive", "naive_n1"),
        ("y_hat_sarima", "sarima"),
        ("y_hat_sarima_cal", "sarima_cal"),
    ]:
        if col not in dial.columns:
            continue
        m = dial_errors_pp(dial, pred_col=col, eval_only=eval_only)
        lift = float("nan")
        if (
            np.isfinite(m["mae_pp"])
            and np.isfinite(naive["mae_pp"])
            and naive["mae_pp"] > 0
        ):
            lift = 100.0 * (naive["mae_pp"] - m["mae_pp"]) / naive["mae_pp"]
        rows.append(
            {
                "model": name,
                "n_days": int(m["n"]),
                "mae_pp": round(m["mae_pp"], 4) if np.isfinite(m["mae_pp"]) else None,
                "rmse_pp": round(m["rmse_pp"], 4) if np.isfinite(m["rmse_pp"]) else None,
                "lift_vs_naive_pct": round(lift, 2) if np.isfinite(lift) else None,
            }
        )
    if "alt_50" in dial.columns:
        m = dial_errors_pp(dial, pred_col="alt_50", eval_only=eval_only)
        lift = float("nan")
        if (
            np.isfinite(m["mae_pp"])
            and np.isfinite(naive["mae_pp"])
            and naive["mae_pp"] > 0
        ):
            lift = 100.0 * (naive["mae_pp"] - m["mae_pp"]) / naive["mae_pp"]
        rows.append(
            {
                "model": "shipped_alt_50",
                "n_days": int(m["n"]),
                "mae_pp": round(m["mae_pp"], 4) if np.isfinite(m["mae_pp"]) else None,
                "rmse_pp": round(m["rmse_pp"], 4) if np.isfinite(m["rmse_pp"]) else None,
                "lift_vs_naive_pct": round(lift, 2) if np.isfinite(lift) else None,
            }
        )
    return pl.DataFrame(rows)


def direction_accuracy(
    dial: pl.DataFrame,
    *,
    pred_col: str = "y_hat_sarima_cal",
    actual_col: str = "y",
    eval_only: bool = True,
) -> float:
    """Sign accuracy of day-over-day moves among nonzero actual Δy."""
    df = dial.sort("booked_date")
    if eval_only and "in_eval" in df.columns:
        df = df.filter(pl.col("in_eval"))
    df = df.filter(
        pl.col(pred_col).is_not_null()
        & pl.col(pred_col).is_not_nan()
        & pl.col(actual_col).is_not_null()
        & pl.col(actual_col).is_not_nan()
    )
    if df.height < 2:
        return float("nan")
    y = df[actual_col].to_numpy()
    p = df[pred_col].to_numpy()
    dy = np.diff(y)
    dp = np.diff(p)
    mask = dy != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(dy[mask]) == np.sign(dp[mask])))


def materialize_sarima_pp(
    loads: pl.DataFrame,
    dial: pl.DataFrame,
    *,
    r_hat_col: str = "y_hat_sarima_cal",
    levels: Sequence[float] = tuple(GRID.tolist()),
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
    cost_col: str = "cost",
    validate: bool = True,
) -> pl.DataFrame:
    """Rigid dial ``alt_a = clip(a + (r̂ − 0.50), clip)`` → ``sarima_pp_XX`` dollars.

    Production default ``clip`` is :data:`QUOTE_CLIP_DEFAULT` ``(0.05, 0.95)`` —
    the Weatherman ``knn_*`` grid — so a down-shifted dial floors left-tail
    quotes at ``knn_5`` instead of scaling toward $0 via below-grid
    ``quantile_at``. Dial forecast ``r̂`` is unchanged (still uses
    :data:`CLIP_DEFAULT` on the time-series path).

    ``loads`` must have ``loadnumber``, ``booked_date``, ``knn_*``, and for
    validation also ``wm_r`` and ``cost`` (or ``cost_col``).
    """
    levels = list(levels)
    labels = [f"{int(round(a * 100)):02d}" for a in levels]
    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    required = {"loadnumber", "booked_date", *knn_cols}
    missing = required - set(loads.columns)
    if missing:
        raise ValueError(f"materialize_sarima_pp: loads missing {sorted(missing)}")
    if r_hat_col not in dial.columns:
        raise ValueError(f"materialize_sarima_pp: dial missing {r_hat_col}")

    sched = (
        dial.select("booked_date", pl.col(r_hat_col).alias("r_hat"))
        .drop_nulls("r_hat")
        .unique(subset=["booked_date"])
    )
    joined = loads.join(sched, on="booked_date", how="inner").filter(
        pl.col("knn_5") >= 0
    )
    if joined.is_empty():
        raise ValueError("materialize_sarima_pp: no loads matched dial dates")

    delta = (joined["r_hat"].to_numpy().astype(np.float64) - 0.50)
    Qmat = np.sort(joined.select(knn_cols).to_numpy(), axis=1)
    alts = rigid_quote_alts(levels, delta, clip=clip)
    series = []
    alt_cols = {}
    for j, (a, lab) in enumerate(zip(levels, labels)):
        alt = alts[:, j]
        alt_cols[lab] = alt
        series.append(
            pl.Series(
                f"sarima_pp_{lab}",
                quantile_at(Qmat, alt, assume_sorted=True),
            )
        )
    out = joined.with_columns(series)

    if validate:
        if "wm_r" not in out.columns:
            raise ValueError("materialize_sarima_pp: validate requires wm_r")
        ccol = cost_col
        if ccol not in out.columns and COST_COL in out.columns:
            ccol = COST_COL
        if ccol not in out.columns:
            raise ValueError(
                f"materialize_sarima_pp: validate requires {cost_col} or {COST_COL}"
            )
        wm_r = out["wm_r"].to_numpy()
        cost = out[ccol].to_numpy()
        for a, lab in zip(levels, labels):
            d_att = float(np.mean(cost <= out[f"sarima_pp_{lab}"].to_numpy()))
            r_att = float(np.mean(wm_r <= alt_cols[lab]))
            if abs(d_att - r_att) >= ATTAIN_TOL:
                raise ValueError(
                    f"materialize_sarima_pp: dollar/percentile attainment mismatch "
                    f"at level {a}: {d_att:.4f} vs {r_att:.4f}"
                )
        sorted_pairs = sorted(zip(levels, labels))
        for (lo_a, lo_lab), (hi_a, hi_lab) in itertools.pairwise(sorted_pairs):
            ok = out.select(
                (pl.col(f"sarima_pp_{hi_lab}") >= pl.col(f"sarima_pp_{lo_lab}")).all()
            ).item()
            if not ok:
                raise ValueError(
                    f"materialize_sarima_pp: quantile crossing between "
                    f"{lo_a} and {hi_a}"
                )

    return out.select(["loadnumber", *[f"sarima_pp_{lab}" for lab in labels]])


def materialize_sarima_blend(
    loads: pl.DataFrame,
    dial: pl.DataFrame,
    *,
    r_hat_col: str = "y_hat_sarima_cal",
    levels: Sequence[float] = tuple(GRID.tolist()),
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
    cost_col: str = "cost",
    validate: bool = True,
) -> pl.DataFrame:
    """Shipped tail shape + SARIMA center → ``sarima_blend_XX`` dollars.

    ``alt_a = clip(shipped_alt_a + (r̂ − shipped_alt_50), clip)``. Uses only
    the walk-forward ``r̂`` (no same-day ``y``) and production alts known at
    open. p50 tracks ``r̂`` by construction (pre-clip).
    """
    levels = list(levels)
    labels = [f"{int(round(a * 100)):02d}" for a in levels]
    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    colset = set(loads.columns)
    alt_names = [shipped_alt_col(a, colset) for a in levels]
    alt_50_name = shipped_alt_col(0.5, colset)
    required = {"loadnumber", "booked_date", *knn_cols, *alt_names, alt_50_name}
    missing = required - colset
    if missing:
        raise ValueError(f"materialize_sarima_blend: loads missing {sorted(missing)}")
    if r_hat_col not in dial.columns:
        raise ValueError(f"materialize_sarima_blend: dial missing {r_hat_col}")

    sched = (
        dial.select("booked_date", pl.col(r_hat_col).alias("r_hat"))
        .drop_nulls("r_hat")
        .unique(subset=["booked_date"])
    )
    joined = loads.join(sched, on="booked_date", how="inner").filter(
        pl.col("knn_5") >= 0
    )
    if joined.is_empty():
        raise ValueError("materialize_sarima_blend: no loads matched dial dates")

    shipped = joined.select(alt_names).to_numpy().astype(np.float64)
    r_hat = joined["r_hat"].to_numpy().astype(np.float64)
    shipped_50 = joined[alt_50_name].to_numpy().astype(np.float64)
    alts = blend_quote_alts(shipped, r_hat, shipped_50, clip=clip)
    Qmat = np.sort(joined.select(knn_cols).to_numpy(), axis=1)
    series = []
    alt_cols: dict[str, np.ndarray] = {}
    for j, (a, lab) in enumerate(zip(levels, labels)):
        alt = alts[:, j]
        alt_cols[lab] = alt
        series.append(
            pl.Series(
                f"sarima_blend_{lab}",
                quantile_at(Qmat, alt, assume_sorted=True),
            )
        )
    out = joined.with_columns(series)

    if validate:
        if "wm_r" not in out.columns:
            raise ValueError("materialize_sarima_blend: validate requires wm_r")
        ccol = cost_col
        if ccol not in out.columns and COST_COL in out.columns:
            ccol = COST_COL
        if ccol not in out.columns:
            raise ValueError(
                f"materialize_sarima_blend: validate requires {cost_col} or {COST_COL}"
            )
        wm_r = out["wm_r"].to_numpy()
        cost = out[ccol].to_numpy()
        for a, lab in zip(levels, labels):
            d_att = float(np.mean(cost <= out[f"sarima_blend_{lab}"].to_numpy()))
            r_att = float(np.mean(wm_r <= alt_cols[lab]))
            if abs(d_att - r_att) >= ATTAIN_TOL:
                raise ValueError(
                    f"materialize_sarima_blend: dollar/percentile attainment mismatch "
                    f"at level {a}: {d_att:.4f} vs {r_att:.4f}"
                )
        sorted_pairs = sorted(zip(levels, labels))
        for (lo_a, lo_lab), (hi_a, hi_lab) in itertools.pairwise(sorted_pairs):
            ok = out.select(
                (
                    pl.col(f"sarima_blend_{hi_lab}")
                    >= pl.col(f"sarima_blend_{lo_lab}")
                ).all()
            ).item()
            if not ok:
                raise ValueError(
                    f"materialize_sarima_blend: quantile crossing between "
                    f"{lo_a} and {hi_a}"
                )

    return out.select(["loadnumber", *[f"sarima_blend_{lab}" for lab in labels]])


def materialize_sarima_hybrid(
    panel: pl.DataFrame,
    dial: pl.DataFrame,
    config: HybridConfig,
    root: Path,
    *,
    data_dir: str | Path | None = None,
    r_hat_col: str = "y_hat_sarima_cal",
    validate: bool = True,
) -> pl.DataFrame:
    """SARIMA global dial + Hybrid per-cell conformal offset → ``sarima_hybrid_*`` $.

    Production candidate: ``alt_used_a = clip(a + (r̂ − 0.50) + δ_cell_a)`` on each
    load's Weatherman curve — same δ schedule as :func:`dqt.hybrid.hybrid_quantiles`
    but the dial node comes from the walk-forward SARIMA forecast, not shipped alts.
    """
    group_cols = list(config.group_cols)
    levels = tuple(config.levels)
    labels = [f"{int(round(a * 100)):02d}" for a in levels]

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

    if r_hat_col not in dial.columns:
        raise ValueError(f"materialize_sarima_hybrid: dial missing {r_hat_col}")
    sched = (
        dial.select("booked_date", pl.col(r_hat_col).alias("r_hat"))
        .drop_nulls("r_hat")
        .unique(subset=["booked_date"])
    )

    ref_level = min(levels, key=lambda a: abs(a - 0.5))
    eval_weeks = (
        alts_glob.filter((pl.col("level") == ref_level) & pl.col("alt").is_not_null())
        .get_column("week")
        .to_list()
    )

    joined = (
        panel.filter(pl.col("week").is_in(eval_weeks))
        .join(sched, on="booked_date", how="inner")
        .join(deltas_wide, on=["week", *group_cols], how="left")
    )
    alt_used_exprs = [
        (
            pl.lit(a)
            + (pl.col("r_hat") - 0.50)
            + pl.col(f"delta_{lab}").fill_null(0.0)
        )
        .clip(*config.clip)
        .alias(f"alt_used_{lab}")
        for a, lab in zip(levels, labels)
    ]
    joined = joined.with_columns(alt_used_exprs)

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

    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    if not set(knn_cols) <= set(joined.columns):
        etp_knn = (
            pl.scan_parquet(resolve_data_dir(data_dir, root=root) / "features.parquet")
            .select("loadnumber", *knn_cols)
            .collect()
        )
        joined = joined.join(etp_knn, on="loadnumber", how="inner")
    joined = joined.filter(pl.col("knn_5") >= 0)

    Qmat = np.sort(joined.select(knn_cols).to_numpy(), axis=1)
    out_series = [
        pl.Series(
            f"sarima_hybrid_{lab}",
            quantile_at(
                Qmat,
                joined.get_column(f"alt_used_{lab}").to_numpy(),
                assume_sorted=True,
            ),
        )
        for a, lab in zip(levels, labels)
    ]
    joined = joined.with_columns(out_series)

    if validate:
        if "wm_r" not in joined.columns:
            raise ValueError("materialize_sarima_hybrid: validate requires wm_r")
        ccol = "cost" if "cost" in joined.columns else COST_COL
        wm_r = joined["wm_r"].to_numpy()
        cost = joined[ccol].to_numpy()
        for a, lab in zip(levels, labels):
            d_att = float(np.mean(cost <= joined[f"sarima_hybrid_{lab}"].to_numpy()))
            r_att = float(
                np.mean(wm_r <= joined[f"alt_used_{lab}"].to_numpy())
            )
            if abs(d_att - r_att) >= ATTAIN_TOL:
                raise ValueError(
                    f"materialize_sarima_hybrid: attainment mismatch at {a}: "
                    f"{d_att:.4f} vs {r_att:.4f}"
                )
        for (lo_a, lo_lab), (hi_a, hi_lab) in itertools.pairwise(sorted_pairs):
            ok = joined.select(
                (
                    pl.col(f"sarima_hybrid_{hi_lab}")
                    >= pl.col(f"sarima_hybrid_{lo_lab}")
                ).all()
            ).item()
            if not ok:
                raise ValueError(
                    f"materialize_sarima_hybrid: crossing between {lo_a} and {hi_a}"
                )

    out_cols = ["loadnumber", *[f"sarima_hybrid_{lab}" for _, lab in sorted_pairs]]
    return joined.select(out_cols)


def materialize_sarima_tail(
    panel: pl.DataFrame,
    dial: pl.DataFrame,
    config: HybridConfig,
    root: Path,
    *,
    data_dir: str | Path | None = None,
    r_hat_col: str = "y_hat_sarima_cal",
    tail_levels: Sequence[float] = (0.05, 0.95),
    base: str = "blend",
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
    validate: bool = True,
) -> pl.DataFrame:
    """Blend (or rigid) base + cell δ + residual global tail offset.

    ``δ_tail_a = alt_glob_a − base_a`` at ``tail_levels`` only, so p05/p95
    become ``clip(alt_glob + δ_cell)`` while mid-grid keeps the SARIMA/blend
    center. Quote clip stays :data:`QUOTE_CLIP_DEFAULT` (not Hybrid's 0.01–0.99).
    """
    if base not in {"blend", "rigid"}:
        raise ValueError(f"materialize_sarima_tail: base must be blend|rigid, got {base}")
    group_cols = list(config.group_cols)
    levels = tuple(config.levels)
    labels = [f"{int(round(a * 100)):02d}" for a in levels]
    tail_set = {round(float(a), 2) for a in tail_levels}
    tail_mask = np.array([round(float(a), 2) in tail_set for a in levels])

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
    glob_wide = (
        alts_glob.with_columns(lab_expr)
        .pivot(on="lab", index=["week"], values="alt")
        .rename({lab: f"alt_glob_{lab}" for lab in labels})
    )

    if r_hat_col not in dial.columns:
        raise ValueError(f"materialize_sarima_tail: dial missing {r_hat_col}")
    sched = (
        dial.select("booked_date", pl.col(r_hat_col).alias("r_hat"))
        .drop_nulls("r_hat")
        .unique(subset=["booked_date"])
    )
    ref_level = min(levels, key=lambda a: abs(a - 0.5))
    eval_weeks = (
        alts_glob.filter((pl.col("level") == ref_level) & pl.col("alt").is_not_null())
        .get_column("week")
        .to_list()
    )
    joined = (
        panel.filter(pl.col("week").is_in(eval_weeks))
        .join(sched, on="booked_date", how="inner")
        .join(deltas_wide, on=["week", *group_cols], how="left")
        .join(glob_wide, on="week", how="left")
    )
    if joined.is_empty():
        raise ValueError("materialize_sarima_tail: no loads matched dial + conformal weeks")

    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    if not set(knn_cols) <= set(joined.columns):
        etp_knn = (
            pl.scan_parquet(resolve_data_dir(data_dir, root=root) / "features.parquet")
            .select("loadnumber", *knn_cols)
            .collect()
        )
        joined = joined.join(etp_knn, on="loadnumber", how="inner")
    joined = joined.filter(pl.col("knn_5") >= 0)
    if joined.is_empty():
        raise ValueError("materialize_sarima_tail: no loads matched etp knn")

    r_hat = joined["r_hat"].to_numpy().astype(np.float64)
    if base == "blend":
        colset = set(joined.columns)
        alt_names = [shipped_alt_col(a, colset) for a in levels]
        alt_50_name = shipped_alt_col(0.5, colset)
        shipped = joined.select(alt_names).to_numpy().astype(np.float64)
        shipped_50 = joined[alt_50_name].to_numpy().astype(np.float64)
        base_alts = blend_quote_alts(shipped, r_hat, shipped_50, clip=clip)
    else:
        base_alts = rigid_quote_alts(levels, r_hat - 0.50, clip=clip)

    cell = np.column_stack(
        [
            joined[f"delta_{lab}"].fill_null(0.0).to_numpy().astype(np.float64)
            for lab in labels
        ]
    )
    glob = np.column_stack(
        [
            joined[f"alt_glob_{lab}"].to_numpy().astype(np.float64)
            for lab in labels
        ]
    )
    used = tail_quote_alts(base_alts, cell, glob, tail_mask, clip=clip)
    used = np.sort(used, axis=1)

    Qmat = np.sort(joined.select(knn_cols).to_numpy(), axis=1)
    series = [
        pl.Series(f"sarima_tail_{lab}", quantile_at(Qmat, used[:, j], assume_sorted=True))
        for j, lab in enumerate(labels)
    ]
    out = joined.with_columns(series)

    if validate:
        if "wm_r" not in out.columns:
            raise ValueError("materialize_sarima_tail: validate requires wm_r")
        ccol = "cost" if "cost" in out.columns else COST_COL
        if ccol not in out.columns:
            raise ValueError("materialize_sarima_tail: validate requires cost")
        wm_r = out["wm_r"].to_numpy()
        cost = out[ccol].to_numpy()
        for j, (a, lab) in enumerate(zip(levels, labels)):
            d_att = float(np.mean(cost <= out[f"sarima_tail_{lab}"].to_numpy()))
            r_att = float(np.mean(wm_r <= used[:, j]))
            if abs(d_att - r_att) >= ATTAIN_TOL:
                raise ValueError(
                    f"materialize_sarima_tail: attainment mismatch at {a}: "
                    f"{d_att:.4f} vs {r_att:.4f}"
                )
        sorted_pairs = sorted(zip(levels, labels))
        for (lo_a, lo_lab), (hi_a, hi_lab) in itertools.pairwise(sorted_pairs):
            ok = out.select(
                (pl.col(f"sarima_tail_{hi_lab}") >= pl.col(f"sarima_tail_{lo_lab}")).all()
            ).item()
            if not ok:
                raise ValueError(
                    f"materialize_sarima_tail: crossing between {lo_a} and {hi_a}"
                )

    return out.select(["loadnumber", *[f"sarima_tail_{lab}" for lab in labels]])


# Named market-drift episodes (``notebooks/05_WeeklySimulation.ipynb`` cell 12).
MARKET_EPISODES: tuple[tuple[str, str, str, str, str], ...] = (
    ("Spot, Feb 2025", "mode", "Spot", "2025-02-01", "2025-02-28"),
    ("Van, Fall 2024 (Sep-Nov)", "equipment", "Van", "2024-09-01", "2024-11-30"),
    ("Short-haul, Sep 2024", "haul_band", "Short", "2024-09-01", "2024-09-30"),
)


def score_market_episodes(
    frame: pl.DataFrame,
    *,
    model_qcol: Mapping[str, str],
    episodes: Sequence[tuple[str, str, str, str, str]] = MARKET_EPISODES,
    actual_col: str = COST_COL,
    target: float = 0.5,
) -> pl.DataFrame:
    """p50 attainment and deviation (pp) per named episode × model."""
    if "booked_date" not in frame.columns:
        raise ValueError("score_market_episodes: frame missing booked_date")
    rows: list[dict] = []
    for name, dim, val, start, end in episodes:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        sub = frame.filter(
            (pl.col("booked_date") >= start_d)
            & (pl.col("booked_date") <= end_d)
            & (pl.col(dim) == val)
        )
        if sub.is_empty():
            continue
        for model, tmpl in model_qcol.items():
            col50 = tmpl.format(q=50)
            if col50 not in sub.columns:
                continue
            msub = sub.filter(pl.col(col50).is_not_null())
            if msub.is_empty():
                continue
            card = summary_metrics(
                msub, model_qcol={model: tmpl}, actual_col=actual_col
            )
            for row in card.iter_rows(named=True):
                rows.append(
                    {
                        "episode": name,
                        "dimension": dim,
                        "filter_value": val,
                        "start_date": start,
                        "end_date": end,
                        "model": row["model"],
                        "n_loads": row["n_loads"],
                        "att50": row["att50"],
                        "att50_gap_pp": row["att50_gap_pp"],
                        "ece_pp": row["ece_pp"],
                        "target_att50": target,
                    }
                )
    return pl.DataFrame(rows)


def rolling_origin_dial_scorecard(
    dial: pl.DataFrame,
    *,
    n_blocks: int = 5,
    pred_col: str = "y_hat_sarima_cal",
) -> pl.DataFrame:
    """Blocked MAE (pp) on ``in_eval`` windows for dial stability checks."""
    ev = dial.filter(pl.col("in_eval")).sort("booked_date")
    if ev.is_empty():
        return pl.DataFrame(
            schema={
                "block": pl.Int64,
                "n_days": pl.Int64,
                "start_date": pl.Date,
                "end_date": pl.Date,
                "mae_naive_pp": pl.Float64,
                "mae_sarima_cal_pp": pl.Float64,
                "lift_vs_naive_pct": pl.Float64,
            }
        )
    n = ev.height
    block_size = max(1, n // n_blocks)
    rows: list[dict] = []
    for b in range(n_blocks):
        lo = b * block_size
        hi = n if b == n_blocks - 1 else (b + 1) * block_size
        block = ev.slice(lo, hi - lo)
        if block.is_empty():
            continue
        naive_m = dial_errors_pp(block, pred_col="y_hat_naive", eval_only=False)
        sar_m = dial_errors_pp(block, pred_col=pred_col, eval_only=False)
        lift = float("nan")
        if (
            np.isfinite(sar_m["mae_pp"])
            and np.isfinite(naive_m["mae_pp"])
            and naive_m["mae_pp"] > 0
        ):
            lift = 100.0 * (naive_m["mae_pp"] - sar_m["mae_pp"]) / naive_m["mae_pp"]
        rows.append(
            {
                "block": b + 1,
                "n_days": int(block.height),
                "start_date": block["booked_date"].min(),
                "end_date": block["booked_date"].max(),
                "mae_naive_pp": round(naive_m["mae_pp"], 4),
                "mae_sarima_cal_pp": round(sar_m["mae_pp"], 4),
                "lift_vs_naive_pct": round(lift, 2) if np.isfinite(lift) else None,
            }
        )
    return pl.DataFrame(rows)


def global_alts_from_tail(
    panel: pl.DataFrame,
    dial: pl.DataFrame,
    shipped: Mapping[str, float],
    as_of: date,
    *,
    r_hat_col: str = "y_hat_sarima_cal",
    tail_levels: Sequence[float] = (0.05, 0.95),
    cal_window_days: int = 28,
    min_n: int = 200,
    score_col: str = "wm_r",
    clip: tuple[float, float] = QUOTE_CLIP_DEFAULT,
) -> dict[str, float]:
    """Global ``alt_*`` schedule from the SARIMA_tail stack (cell δ = 0).

    Mid-grid levels follow blend spacing (shipped tails + SARIMA p50 center);
    p05/p95 add the residual global conformal offset on top of blend base.
    """
    if r_hat_col not in dial.columns:
        raise ValueError(f"global_alts_from_tail: dial missing {r_hat_col}")
    date_col = "booked_date" if "booked_date" in dial.columns else "valid_date"
    hit = dial.filter(pl.col(date_col) == as_of)
    if hit.is_empty():
        hit = (
            dial.filter(pl.col(date_col) <= as_of)
            .filter(pl.col(r_hat_col).is_not_null() & pl.col(r_hat_col).is_finite())
            .sort(date_col)
        )
        if hit.is_empty():
            raise LookupError(f"no {r_hat_col} on/before {as_of}")
        r_hat = float(hit[r_hat_col][-1])
    else:
        val = hit[r_hat_col][0]
        if val is None or not np.isfinite(float(val)):
            raise LookupError(f"{r_hat_col} null/non-finite for {as_of}")
        r_hat = float(val)

    levels = tuple(float(a) for a in GRID)
    tail_set = {round(float(a), 2) for a in tail_levels}
    tail_mask = np.array([round(float(a), 2) in tail_set for a in levels])

    colset = set(shipped.keys())
    alt_names = [shipped_alt_col(a, colset) for a in levels]
    shipped_row = np.asarray(
        [[float(shipped[c]) for c in alt_names]], dtype=np.float64
    )
    shipped_50 = float(shipped[shipped_alt_col(0.5, colset)])
    base = blend_quote_alts(
        shipped_row,
        np.asarray([r_hat], dtype=np.float64),
        np.asarray([shipped_50], dtype=np.float64),
        clip=clip,
    )

    cal_panel = panel
    if "week" not in cal_panel.columns:
        bd = "booked_date" if "booked_date" in cal_panel.columns else "date"
        cal_panel = cal_panel.with_columns(
            parse_date_col(cal_panel, bd).dt.truncate("1w").alias("week")
        )
    cal_panel = cal_panel.filter(pl.col("week") <= as_of)
    alts_glob = walk_forward_alts(
        cal_panel,
        [],
        levels=levels,
        cal_window_days=cal_window_days,
        min_n=min_n,
        score_col=score_col,
    )
    target_week = as_of - timedelta(days=as_of.weekday())
    glob_week = alts_glob.filter(
        (pl.col("week") == target_week) & pl.col("alt").is_not_null()
    )
    if glob_week.is_empty():
        ref = min(levels, key=lambda a: abs(a - 0.5))
        glob_week = alts_glob.filter(
            (pl.col("level") == ref) & pl.col("alt").is_not_null()
        ).sort("week")
        if glob_week.is_empty():
            raise LookupError("global_alts_from_tail: no global conformal alts")
        target_week = glob_week["week"][-1]
        glob_week = alts_glob.filter(pl.col("week") == target_week)

    glob_by_level = {
        float(row["level"]): float(row["alt"])
        for row in glob_week.iter_rows(named=True)
    }
    glob_row = np.asarray(
        [[glob_by_level.get(float(a), float("nan")) for a in levels]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(glob_row)):
        missing = [a for a in levels if a not in glob_by_level]
        raise LookupError(
            f"global_alts_from_tail: missing global alts for levels {missing[:4]}"
        )

    cell = np.zeros((1, len(levels)), dtype=np.float64)
    used = tail_quote_alts(base, cell, glob_row, tail_mask, clip=clip)[0]
    used = np.maximum.accumulate(used)
    return {
        shipped_alt_col(float(a)): float(used[j]) for j, a in enumerate(levels)
    }


MODEL_QCOL_SARIMA: dict[str, str] = {
    "Weatherman": "knn_{q}",
    "DQT/ETP": "p{q:02d}",
    "Hybrid": "hybrid_{q:02d}",
    "SARIMA_pp": "sarima_pp_{q:02d}",
    "SARIMA_hybrid": "sarima_hybrid_{q:02d}",
    "SARIMA_blend": "sarima_blend_{q:02d}",
    "SARIMA_tail": "sarima_tail_{q:02d}",
}


def score_sarima_walkforward(
    frame: pl.DataFrame,
    *,
    model_qcol: Mapping[str, str] | None = None,
    group_cols: tuple[str, ...] = ("mode", "equipment", "haul_band"),
    cost_col: str = COST_COL,
) -> dict[str, pl.DataFrame]:
    """Global + optional cell-gap scorecards for the four-model comparison."""
    model_qcol = dict(model_qcol or MODEL_QCOL_SARIMA)
    # Drop models whose columns are absent.
    kept = {}
    for name, tmpl in model_qcol.items():
        cols = [tmpl.format(q=int(round(a * 100))) for a in GRID]
        if all(c in frame.columns for c in cols):
            kept[name] = tmpl
    if not kept:
        raise ValueError("score_sarima_walkforward: no model columns found on frame")

    ccol = cost_col if cost_col in frame.columns else (
        "cost" if "cost" in frame.columns else cost_col
    )
    global_card = summary_metrics(frame, model_qcol=kept, actual_col=ccol)
    level_card = quantile_level_scorecard(frame, model_qcol=kept, actual_col=ccol)

    cell = None
    if (
        "SARIMA_pp" in kept
        and "DQT/ETP" in kept
        and all(c in frame.columns for c in group_cols)
    ):
        cell = cell_gap_from_loads(
            frame,
            level=0.5,
            cost_col=ccol,
            actual_col="p50",
            hybrid_col="sarima_pp_50",
            group_cols=group_cols,
        )

    return {"global": global_card, "quantile_level": level_card, "cell_gap": cell}


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write parquet via ``.tmp`` + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


__all__ = [
    "CLIP_DEFAULT",
    "DIAL_PRED_COLS",
    "EVAL_FRAC_DEFAULT",
    "MARKET_EPISODES",
    "MIN_HISTORY_DEFAULT",
    "MODEL_QCOL_SARIMA",
    "ORDER_DEFAULT",
    "QUOTE_CLIP_DEFAULT",
    "SEASONAL_DEFAULT",
    "blend_quote_alts",
    "calendar_flags",
    "daily_wm_r_median",
    "daily_wm_r_median_settlement_lag",
    "dial_errors_pp",
    "dial_scorecard",
    "direction_accuracy",
    "extend_walk_forward_sarima_dial",
    "global_alts_from_tail",
    "materialize_sarima_blend",
    "materialize_sarima_hybrid",
    "materialize_sarima_pp",
    "materialize_sarima_tail",
    "quantile_level_scorecard",
    "rigid_quote_alts",
    "rolling_origin_dial_scorecard",
    "score_market_episodes",
    "score_sarima_walkforward",
    "settlement_coverage_daily",
    "shipped_alt_col",
    "tail_quote_alts",
    "walk_forward_sarima_dial",
    "write_parquet_atomic",
]
