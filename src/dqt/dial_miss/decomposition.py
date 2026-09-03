"""DQT daily dial miss decomposition — SARIMA r̂ vs shipped alt_50."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from dqt import parse_date_col, resolve_data_dir
from dqt.panel import KNN_COLS
from dqt.sarima_dial import (
    QUOTE_CLIP_DEFAULT,
    dial_scorecard,
    materialize_pp50_batch,
    rigid_quote_alts,
)
from dqt.score.constants import COST_COL, DATE_COL, ID_COL

DEFAULT_EXEC_DIR = "executive-2025-01-01_2026-08-28"
LABELED_COHORT = "labeled-cohort.parquet"
LC_TAIL_NAME = "lc-tail-2025.parquet"
OUT_SUBDIR = "dial-miss"

LEVELING_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LEVELING_POLICY_NAMES = {
    0.0: "status_quo",
    0.25: "partial_25",
    0.5: "partial_50",
    0.75: "partial_75",
    1.0: "full_leveling",
}


def _out_dir(data_dir: Path) -> Path:
    d = data_dir / "etp" / OUT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alt_schedule(data_dir: Path) -> pl.DataFrame:
    """One row per valid_date with alt_50."""
    path = data_dir / "dqt_alt_percentiles.parquet"
    alt = pl.read_parquet(path)
    date_col = "valid_date" if "valid_date" in alt.columns else "date"
    if "snowflakeupdatedon" in alt.columns:
        alt = alt.sort("snowflakeupdatedon", descending=True).unique(
            subset=[date_col], keep="first"
        )
    return alt.select(
        pl.col(date_col).alias("booked_date"),
        pl.col("alt_50"),
    )


def _dial_schedule(data_dir: Path) -> pl.DataFrame:
    dial = pl.read_parquet(data_dir / "sarima_dial_walkforward.parquet")
    return dial.select(
        pl.col("booked_date"),
        pl.col("y_hat_sarima_cal"),
        pl.col("y"),
        pl.col("in_eval"),
        pl.col("n_loads").alias("n_loads_booked"),
    )


def verify_data_gates(*, data_dir: Path | None = None) -> dict[str, Any]:
    """Phase 0 — blocking artifact checks."""
    data_dir = Path(data_dir or resolve_data_dir())
    etp = data_dir / "etp"
    gates: dict[str, Any] = {"data_dir": str(data_dir), "passed": True, "checks": []}

    def _check(name: str, ok: bool, detail: str) -> None:
        gates["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            gates["passed"] = False

    feat_path = data_dir / "features.parquet"
    if feat_path.is_file():
        f = pl.read_parquet(feat_path, n_rows=1)
        knn_ok = all(c in f.columns for c in KNN_COLS)
        _check("features", knn_ok and COST_COL in f.columns, f"cols ok={knn_ok}")
    else:
        _check("features", False, str(feat_path))

    dial_path = data_dir / "sarima_dial_walkforward.parquet"
    if dial_path.is_file():
        d = pl.read_parquet(dial_path, n_rows=1)
        _check(
            "sarima_dial",
            "y_hat_sarima_cal" in d.columns,
            f"rows≈{pl.scan_parquet(dial_path).select(pl.len()).collect().item()}",
        )
    else:
        _check("sarima_dial", False, str(dial_path))

    alt_path = data_dir / "dqt_alt_percentiles.parquet"
    if alt_path.is_file():
        a = pl.read_parquet(alt_path, n_rows=1)
        _check("dqt_alt", "alt_50" in a.columns, "alt_50 present")
    else:
        _check("dqt_alt", False, str(alt_path))

    pp_path = data_dir / "sarima_pp_quantiles.parquet"
    _check("sarima_pp", pp_path.is_file(), "optional cross-check")

    labeled = etp / DEFAULT_EXEC_DIR / LABELED_COHORT
    if labeled.is_file():
        n = pl.scan_parquet(labeled).select(pl.len()).collect().item()
        _check("labeled_cohort", 200_000 <= n <= 300_000, f"n={n:,}")
    else:
        _check("labeled_cohort", False, str(labeled))

    paint_s1 = etp / "paint-experiment-s1.parquet"
    paint_pl = etp / "paint-experiment-s1-per-load.parquet"
    _check("paint_s1", paint_s1.is_file() and paint_pl.is_file(), "S1 lifecycle")

    lc_path = etp / LC_TAIL_NAME
    _check("lc_tail", lc_path.is_file(), "optional regression proxies")

    return gates


def build_daily_dial_miss(*, data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Phase 1 — daily dial_miss_pp registry."""
    dial = _dial_schedule(data_dir)
    alt = _alt_schedule(data_dir)
    daily = (
        dial.join(alt, on="booked_date", how="left")
        .with_columns(
            ((pl.col("y_hat_sarima_cal") - pl.col("alt_50")) * 100.0).alias("dial_miss_pp"),
            ((pl.col("alt_50") - 0.50) * 100.0).alias("shipped_center_pp"),
            ((pl.col("y_hat_sarima_cal") - pl.col("y")) * 100.0).alias("sarima_forecast_err_pp"),
            ((pl.col("y") - pl.col("alt_50")) * 100.0).alias("realized_vs_shipped_pp"),
        )
        .sort("booked_date")
    )

    eval_daily = daily.filter(pl.col("in_eval")) if "in_eval" in daily.columns else daily
    miss = eval_daily.filter(pl.col("dial_miss_pp").is_not_null())["dial_miss_pp"].to_numpy()
    sign_runs = 0
    if len(miss) > 1:
        signs = np.sign(miss)
        sign_runs = int(np.sum(signs[1:] != signs[:-1]))

    dial_full = pl.read_parquet(data_dir / "sarima_dial_walkforward.parquet").join(
        alt, on="booked_date", how="left"
    )
    scorecard = dial_scorecard(dial_full, eval_only=True)
    miss_row = {
        "model": "dial_miss_pp",
        "n_days": int(len(miss)),
        "mae_pp": round(float(np.mean(np.abs(miss))), 4) if len(miss) else None,
        "rmse_pp": round(float(np.sqrt(np.mean(miss**2))), 4) if len(miss) else None,
        "mean_pp": round(float(np.mean(miss)), 4) if len(miss) else None,
        "sign_persistence_runs": sign_runs,
    }
    scorecard_ext = pl.concat([scorecard, pl.DataFrame([miss_row])], how="diagonal")

    summary = {
        "n_days": daily.height,
        "n_eval_days": eval_daily.height,
        "mean_dial_miss_pp": miss_row["mean_pp"],
        "mae_dial_miss_pp": miss_row["mae_pp"],
        "rmse_dial_miss_pp": miss_row["rmse_pp"],
        "scorecard": scorecard_ext.to_dicts(),
    }
    return daily, summary


def _quote_from_alt_batch(knn_mat: np.ndarray, alt: np.ndarray) -> np.ndarray:
    """Rigid pp50 from alt percentile (not r_hat directly)."""
    qmat = np.sort(knn_mat.astype(np.float64), axis=1)
    ok = (qmat[:, 0] >= 0) & np.isfinite(alt)
    out = np.full(len(alt), np.nan, dtype=np.float64)
    if not ok.any():
        return out
    delta = alt[ok] - 0.50
    alts = rigid_quote_alts([0.50], delta, clip=QUOTE_CLIP_DEFAULT)[:, 0]
    from dqt.conformal import quantile_at

    out[ok] = quantile_at(qmat[ok], alts, assume_sorted=True)
    return out


def _labeled_cohort(data_dir: Path) -> pl.DataFrame:
    path = data_dir / "etp" / DEFAULT_EXEC_DIR / LABELED_COHORT
    return pl.read_parquet(path)


def build_load_dial_miss(*, data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Phase 2 — per-load dial_miss_usd on long-lead cohort."""
    cohort = _labeled_cohort(data_dir).select(
        ID_COL,
        "booking_window_hrs",
        "shift_segment",
        "etp50_shift_amt",
        "etp50_shift_pct",
        "etp50_avail",
        "etp50_48hr",
    )
    feat_cols = [ID_COL, DATE_COL, COST_COL, *KNN_COLS]
    features_raw = pl.read_parquet(data_dir / "features.parquet", columns=feat_cols)
    features = features_raw.with_columns(
        parse_date_col(features_raw, DATE_COL).alias(DATE_COL)
    )
    dial_raw = _dial_schedule(data_dir).rename({"booked_date": DATE_COL})
    dial = dial_raw.with_columns(parse_date_col(dial_raw, DATE_COL).alias(DATE_COL))
    alt_raw = _alt_schedule(data_dir).rename({"booked_date": DATE_COL})
    alt = alt_raw.with_columns(parse_date_col(alt_raw, DATE_COL).alias(DATE_COL))

    spine = (
        cohort.join(features, on=ID_COL, how="inner", suffix="_feat")
        .join(dial.drop("n_loads_booked"), on=DATE_COL, how="left")
        .join(alt, on=DATE_COL, how="left")
    )

    knn_mat = spine.select(KNN_COLS).to_numpy()
    r_hat = spine["y_hat_sarima_cal"].to_numpy()
    alt50 = spine["alt_50"].to_numpy()

    quote_sarima = materialize_pp50_batch(knn_mat, r_hat)
    quote_shipped = _quote_from_alt_batch(knn_mat, alt50)
    dial_miss_usd = quote_sarima - quote_shipped
    dial_miss_pp = (r_hat - alt50) * 100.0

    spine = spine.with_columns(
        pl.Series("quote_sarima_50", quote_sarima),
        pl.Series("quote_shipped_50", quote_shipped),
        pl.Series("dial_miss_usd", dial_miss_usd),
        pl.Series("dial_miss_pp", dial_miss_pp),
    )

    pp_path = data_dir / "sarima_pp_quantiles.parquet"
    if pp_path.is_file():
        pp = pl.read_parquet(pp_path, columns=[ID_COL, "sarima_pp_50"])
        spine = spine.join(pp, on=ID_COL, how="left")

    s1_pl = data_dir / "etp" / "paint-experiment-s1-per-load.parquet"
    if s1_pl.is_file():
        s1 = pl.read_parquet(
            s1_pl,
            columns=[ID_COL, "etp50_at_48hr", "etp50_first_paint", "mean_abs_daily_jump"],
        )
        spine = spine.join(s1, on=ID_COL, how="left", suffix="_s1")

    s0_pl = data_dir / "etp" / "paint-experiment-s0-per-load.parquet"
    if s0_pl.is_file():
        s0 = pl.read_parquet(
            s0_pl,
            columns=[ID_COL, "etp50_at_48hr", "mean_abs_daily_jump"],
        ).rename(
            {
                "etp50_at_48hr": "etp50_at_48hr_s0",
                "mean_abs_daily_jump": "mean_abs_daily_jump_s0",
            }
        )
        spine = spine.join(s0, on=ID_COL, how="left")

    valid = spine.filter(
        pl.col("dial_miss_usd").is_not_null() & pl.col("dial_miss_usd").is_finite()
    )
    abs_miss = valid["dial_miss_usd"].abs().to_numpy()
    cross_err = None
    if "sarima_pp_50" in valid.columns:
        diff = (valid["quote_sarima_50"] - valid["sarima_pp_50"]).abs()
        cross_err = float(diff.filter(diff.is_finite()).median())

    summary = {
        "n_cohort": cohort.height,
        "n_with_features": spine.height,
        "n_valid_dial_miss": valid.height,
        "mean_dial_miss_usd": round(float(valid["dial_miss_usd"].mean()), 2),
        "median_abs_dial_miss_usd": round(float(np.median(abs_miss)), 2),
        "p90_abs_dial_miss_usd": round(float(np.percentile(abs_miss, 90)), 2),
        "mean_dial_miss_pp": round(float(valid["dial_miss_pp"].mean()), 4),
        "sarima_pp_crosscheck_median_err": cross_err,
    }
    if "etp50_at_48hr" in valid.columns and valid.height > 10:
        mae48 = (valid["etp50_at_48hr"] - valid[COST_COL]).abs()
        summary["corr_dial_miss_mae48"] = round(
            float(valid.select(pl.corr("dial_miss_usd", pl.col("etp50_at_48hr") - pl.col(COST_COL))).item()),
            4,
        )
        summary["mean_mae48_s1"] = round(float(mae48.mean()), 2)

    return spine, summary


def _clock_index_proxies(data_dir: Path) -> pl.DataFrame:
    """LC-tail endpoint proxies (left-joinable)."""
    lc_path = data_dir / "etp" / LC_TAIL_NAME
    if not lc_path.is_file():
        return pl.DataFrame({ID_COL: pl.Series([], dtype=pl.Int64)})
    lc = pl.read_parquet(
        lc_path,
        columns=[
            ID_COL,
            "book_2_pkup_delta",
            "avail_2_book_delta",
            "lag7_cpm_delta",
            "fuel_cost_delta",
            "dat_rate_delta",
            "index_change_ind",
            "path_archetype",
        ],
    )
    return lc.with_columns(
        (
            pl.col("book_2_pkup_delta").fill_null(0.0)
            + pl.col("avail_2_book_delta").fill_null(0.0)
        ).alias("clock_proxy"),
        (
            pl.col("lag7_cpm_delta").fill_null(0.0)
            + pl.col("fuel_cost_delta").fill_null(0.0)
            + pl.col("dat_rate_delta").fill_null(0.0)
        ).alias("index_proxy"),
    )


def run_endpoint_decomposition(
    load_miss: pl.DataFrame, *, data_dir: Path
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Phase 3 — OLS shares dial vs clock vs index."""
    proxies = _clock_index_proxies(data_dir)
    frame = load_miss.join(proxies, on=ID_COL, how="left").filter(
        pl.col("etp50_shift_amt").is_not_null()
        & pl.col("dial_miss_usd").is_not_null()
        & pl.col("etp50_shift_amt").is_finite()
    )

    # Full-cohort simple correlation
    full_corr = float(
        frame.select(pl.corr("etp50_shift_amt", "dial_miss_usd")).item()
    ) if frame.height > 2 else float("nan")

    reg_frame = frame.filter(
        pl.col("clock_proxy").is_not_null()
        & pl.col("index_proxy").is_not_null()
        & pl.col("dial_miss_usd").is_finite()
        & pl.col("etp50_shift_amt").is_finite()
    )
    result: dict[str, Any] = {
        "n_full": frame.height,
        "n_with_proxies": reg_frame.height,
        "corr_shift_dial_full": round(full_corr, 4) if np.isfinite(full_corr) else None,
        "r2_dial_only": None,
        "r2_full_model": None,
        "coefficients": {},
        "partial_shares_pct": {},
    }

    if reg_frame.height < 50:
        result["note"] = "insufficient proxy coverage for multivariate regression"
        return result, pl.DataFrame()

    y = reg_frame["etp50_shift_amt"].to_numpy()
    x_dial = reg_frame["dial_miss_usd"].to_numpy()
    try:
        r2_dial = _r2(y, _ols_predict(y, x_dial.reshape(-1, 1)))
    except np.linalg.LinAlgError:
        r2_dial = float("nan")
    result["r2_dial_only"] = round(r2_dial, 4) if np.isfinite(r2_dial) else None

    X = np.column_stack(
        [
            reg_frame["clock_proxy"].to_numpy(),
            reg_frame["index_proxy"].to_numpy(),
            x_dial,
            np.ones(len(y)),
        ]
    )
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        result["note"] = "multivariate regression failed (singular design matrix)"
        return result, pl.DataFrame()

    y_hat = X @ coef
    r2_full = _r2(y, y_hat)
    result["r2_full_model"] = round(r2_full, 4)
    names = ["clock_proxy", "index_proxy", "dial_miss_usd", "intercept"]
    result["coefficients"] = {n: round(float(c), 4) for n, c in zip(names, coef, strict=True)}

    contrib = {
        "clock": coef[0] * reg_frame["clock_proxy"].to_numpy(),
        "index": coef[1] * reg_frame["index_proxy"].to_numpy(),
        "dial": coef[2] * x_dial,
    }
    mean_abs = {k: float(np.mean(np.abs(v))) for k, v in contrib.items()}
    total = sum(mean_abs.values()) or 1.0
    result["partial_shares_pct"] = {
        k: round(100.0 * v / total, 2) for k, v in mean_abs.items()
    }
    result["partial_shares_pct"]["residual"] = round(
        100.0 - sum(result["partial_shares_pct"].values()), 2
    )

    seg_rows: list[dict[str, Any]] = []
    for seg_col in ("shift_segment", "path_archetype"):
        if seg_col not in reg_frame.columns:
            continue
        for seg_val in reg_frame[seg_col].drop_nulls().unique().to_list():
            sub = reg_frame.filter(pl.col(seg_col) == seg_val)
            if sub.height < 30:
                continue
            ys = sub["etp50_shift_amt"].to_numpy()
            xs = sub["dial_miss_usd"].to_numpy()
            seg_rows.append(
                {
                    "segment_dim": seg_col,
                    "segment": seg_val,
                    "n": sub.height,
                    "mean_dial_miss_usd": round(float(sub["dial_miss_usd"].mean()), 2),
                    "r2_dial_only": round(_r2(ys, _ols_predict(ys, xs.reshape(-1, 1))), 4),
                }
            )

    return result, pl.DataFrame(seg_rows)


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _ols_predict(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.isfinite(x.reshape(-1))
    yf = y[mask]
    xf = x.reshape(-1)[mask]
    if len(yf) < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    coef, _, _, _ = np.linalg.lstsq(
        np.column_stack([xf, np.ones(len(yf))]), yf, rcond=None
    )
    out = coef[0] * x.reshape(-1) + coef[1]
    out[~mask] = np.nan
    return out


def run_lifecycle_dial_miss(*, data_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Phase 4 — cumulative dial miss on S1 paint calendar."""
    s1_path = data_dir / "etp" / "paint-experiment-s1.parquet"
    if not s1_path.is_file():
        return pl.DataFrame(), {"note": "missing paint S1 timeline"}

    alt_raw = _alt_schedule(data_dir).rename({"booked_date": "paint_date"})
    alt = alt_raw.with_columns(parse_date_col(alt_raw, "paint_date").alias("paint_date"))
    timeline_raw = pl.read_parquet(s1_path)
    timeline = timeline_raw.with_columns(
        parse_date_col(timeline_raw, "paint_date").alias("paint_date")
    ).join(alt, on="paint_date", how="left")

    knn_mat = timeline.select(KNN_COLS).to_numpy()
    r_hat = timeline["r_hat"].to_numpy()
    alt50 = timeline["alt_50"].to_numpy()

    quote_sarima = materialize_pp50_batch(knn_mat, r_hat)
    quote_shipped = _quote_from_alt_batch(knn_mat, alt50)
    daily_miss = quote_sarima - quote_shipped

    timeline = timeline.with_columns(
        pl.Series("daily_dial_miss_usd", daily_miss),
    )

    cum = (
        timeline.sort([ID_COL, "paint_date"])
        .group_by(ID_COL)
        .agg(
            pl.col("daily_dial_miss_usd").sum().alias("cum_dial_miss_usd"),
            pl.col("daily_dial_miss_usd").mean().alias("mean_daily_dial_miss_usd"),
            pl.col("paint_date").max().alias("last_paint_date"),
            pl.len().alias("n_paint_days"),
        )
    )

    mark48 = timeline.filter(pl.col("mark_hrs") == 48)
    if mark48.is_empty() and "hours_before_pickup" in timeline.columns:
        mark48 = timeline.filter(pl.col("hours_before_pickup") <= 48).group_by(ID_COL).tail(1)

    if not mark48.is_empty():
        at48 = mark48.select(ID_COL, pl.col("daily_dial_miss_usd").alias("dial_miss_usd_at_48hr"))
        cum = cum.join(at48, on=ID_COL, how="left")

    summary = {
        "n_loads": cum.height,
        "mean_cum_dial_miss_usd": round(float(cum["cum_dial_miss_usd"].mean()), 2),
        "median_cum_dial_miss_usd": round(float(cum["cum_dial_miss_usd"].median()), 2),
        "mean_daily_dial_miss_usd": round(float(timeline["daily_dial_miss_usd"].mean()), 2),
    }

    lifecycle = timeline.select(
        ID_COL,
        "paint_date",
        "mark_hrs",
        "r_hat",
        "alt_50",
        "daily_dial_miss_usd",
        "etp50_painted",
    )
    return lifecycle.join(cum, on=ID_COL, how="left"), summary


def run_cohort_strategy(
    load_miss: pl.DataFrame, *, decomposition: dict[str, Any]
) -> pl.DataFrame:
    """Phase 5 — segment tables."""
    frame = load_miss.filter(pl.col("dial_miss_usd").is_finite())
    dial_coef = decomposition.get("coefficients", {}).get("dial_miss_usd", 0.0)

    abs_usd = frame["dial_miss_usd"].abs()
    q_edges = list(abs_usd.quantile([0.2, 0.4, 0.6, 0.8], interpolation="linear"))
    frame = frame.with_columns(
        pl.when(pl.col("dial_miss_usd").abs() < 1.0)
        .then(pl.lit("neutral"))
        .when(pl.col("dial_miss_usd") > 0)
        .then(pl.lit("over"))
        .otherwise(pl.lit("under"))
        .alias("dial_miss_sign"),
        pl.when(abs_usd <= q_edges[0])
        .then(pl.lit("Q1"))
        .when(abs_usd <= q_edges[1])
        .then(pl.lit("Q2"))
        .when(abs_usd <= q_edges[2])
        .then(pl.lit("Q3"))
        .when(abs_usd <= q_edges[3])
        .then(pl.lit("Q4"))
        .otherwise(pl.lit("Q5"))
        .alias("dial_miss_quintile"),
    )
    if DATE_COL in frame.columns:
        frame = frame.with_columns(pl.col(DATE_COL).dt.weekday().alias("book_dow"))

    rows: list[dict[str, Any]] = []
    dims = ["shift_segment", "booking_window_hrs", "dial_miss_sign", "dial_miss_quintile", "book_dow"]
    for dim in dims:
        if dim not in frame.columns:
            continue
        for val, grp in frame.group_by(dim):
            v = val[0] if isinstance(val, tuple) else val
            att48 = None
            if "etp50_at_48hr" in grp.columns:
                att48 = float((grp[COST_COL] <= grp["etp50_at_48hr"]).mean())
            mae48 = None
            if "etp50_at_48hr" in grp.columns:
                mae48 = float((grp["etp50_at_48hr"] - grp[COST_COL]).abs().mean())
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(v),
                    "n": grp.height,
                    "mean_dial_miss_pp": round(float(grp["dial_miss_pp"].mean()), 4),
                    "mean_dial_miss_usd": round(float(grp["dial_miss_usd"].mean()), 2),
                    "att50_at_48hr": round(att48, 4) if att48 is not None else None,
                    "mae_at_48hr_s1": round(mae48, 2) if mae48 is not None else None,
                    "est_dial_attrib_mae": round(abs(dial_coef) * abs(float(grp["dial_miss_usd"].mean())), 2),
                }
            )
    return pl.DataFrame(rows)


def run_leveling_counterfactuals(
    load_miss: pl.DataFrame, *, data_dir: Path
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Phase 6 — α-policy att50 / MAE at book."""
    frame = load_miss.filter(
        pl.col("dial_miss_usd").is_finite()
        & pl.col("y_hat_sarima_cal").is_finite()
        & pl.col("alt_50").is_finite()
    )
    if frame.height == 0:
        return {"policies": []}, pl.DataFrame()

    feat_cols = [ID_COL, *KNN_COLS]
    features = pl.read_parquet(data_dir / "features.parquet", columns=feat_cols)
    sub = frame.join(features, on=ID_COL, how="inner", suffix="_knn")
    knn_mat = sub.select(KNN_COLS).to_numpy()
    cost = sub[COST_COL].to_numpy()
    r_hat = sub["y_hat_sarima_cal"].to_numpy()
    alt50 = sub["alt_50"].to_numpy()

    policies: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []

    for alpha in LEVELING_ALPHAS:
        policy_alt = alt50 + alpha * (r_hat - alt50)
        quotes = _quote_from_alt_batch(knn_mat, policy_alt)
        ok = np.isfinite(quotes) & np.isfinite(cost)
        if not ok.any():
            continue
        q = quotes[ok]
        c = cost[ok]
        att50 = float(np.mean(c <= q))
        mae = float(np.mean(np.abs(q - c)))
        name = LEVELING_POLICY_NAMES[alpha]
        policies.append(
            {
                "policy": name,
                "alpha": alpha,
                "n": int(ok.sum()),
                "att50_at_book": round(att50, 4),
                "mae_at_book": round(mae, 2),
            }
        )
        frontier_rows.append(
            {"policy": name, "alpha": alpha, "att50": att50, "mae": mae}
        )

    # S1 @48hr from per-load if available
    if "etp50_at_48hr" in sub.columns:
        ok48 = sub.filter(pl.col("etp50_at_48hr").is_finite())
        if ok48.height > 0:
            c48 = ok48[COST_COL].to_numpy()
            q48 = ok48["etp50_at_48hr"].to_numpy()
            policies.append(
                {
                    "policy": "S1_operational",
                    "alpha": None,
                    "n": ok48.height,
                    "att50_at_48hr": round(float(np.mean(c48 <= q48)), 4),
                    "mae_at_48hr": round(float(np.mean(np.abs(q48 - c48))), 2),
                }
            )

    status = next((p for p in policies if p["policy"] == "status_quo"), None)
    full = next((p for p in policies if p["policy"] == "full_leveling"), None)
    if status and full:
        policies.append(
            {
                "policy": "delta_full_vs_status",
                "delta_att50_at_book": round(full["att50_at_book"] - status["att50_at_book"], 4),
                "delta_mae_at_book": round(full["mae_at_book"] - status["mae_at_book"], 2),
            }
        )

    return {"policies": policies}, pl.DataFrame(frontier_rows)


def _plot_cohort_heatmap(cohort_df: pl.DataFrame, out_path: Path) -> None:
    if cohort_df.is_empty():
        return
    sub = cohort_df.filter(pl.col("dimension") == "dial_miss_quintile")
    if sub.is_empty():
        sub = cohort_df.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = sub["mean_dial_miss_usd"].to_numpy()
    y = sub.get_column("att50_at_48hr").to_numpy() if "att50_at_48hr" in sub.columns else x
    ax.scatter(x, y, s=sub["n"].to_numpy() / 100, alpha=0.7)
    ax.set_xlabel("mean dial_miss_usd")
    ax.set_ylabel("att50 @ 48hr")
    ax.set_title("Dial miss vs att50 by segment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def synthesize_hypotheses(
    *,
    daily_summary: dict[str, Any],
    load_summary: dict[str, Any],
    decomposition: dict[str, Any],
    lifecycle_summary: dict[str, Any],
    leveling: dict[str, Any],
) -> dict[str, str]:
    """H1–H5 verdicts."""
    h: dict[str, str] = {}

    mean_pp = abs(daily_summary.get("mean_dial_miss_pp") or 0)
    mae_pp = daily_summary.get("mae_dial_miss_pp") or 0
    if mae_pp < 3 and mean_pp >= 0.5:
        h["H1"] = "Confirmed — modest MAE but systematically signed mean dial miss"
    elif mae_pp >= 3:
        h["H1"] = "Confirmed — dial miss magnitude non-trivial at book"
    else:
        h["H1"] = "Rejected — mean |dial_miss_pp| < 0.5pp or random sign"

    r2_dial = decomposition.get("r2_dial_only")
    if r2_dial is not None and r2_dial > 0.15:
        h["H2"] = "Rejected — dial miss R² on etp50_shift > 15%"
    elif r2_dial is not None:
        h["H2"] = "Confirmed — dial miss explains <15% of avail→48hr shift"
    else:
        h["H2"] = "Inconclusive — insufficient proxy regression"

    shares = decomposition.get("partial_shares_pct", {})
    dial_share = shares.get("dial", 0)
    if dial_share > 10 and r2_dial and r2_dial > 0.05:
        h["H3"] = "Confirmed — partial dial share meaningful after clock/index controls"
    elif dial_share <= 5:
        h["H3"] = "Rejected — dial partial share negligible vs clock/index"
    else:
        h["H3"] = "Inconclusive"

    policies = leveling.get("policies", [])
    delta = next((p for p in policies if p.get("policy") == "delta_full_vs_status"), {})
    d_att = delta.get("delta_att50_at_book", 0) or 0
    d_mae = delta.get("delta_mae_at_book", 999) or 999
    if d_att > 0.02 and d_mae < 10:
        h["H4"] = "Confirmed — full leveling improves att50 with bounded MAE"
    elif d_att <= 0.02:
        h["H4"] = "Rejected — leveling Δatt50 ≤ 2pp"
    else:
        h["H4"] = "Inconclusive — MAE tradeoff too large"

    s1 = next((p for p in policies if p.get("policy") == "S1_operational"), {})
    full = next((p for p in policies if p.get("policy") == "full_leveling"), {})
    if s1 and full and "mae_at_48hr" in s1 and "mae_at_book" in full:
        gap = abs(s1["mae_at_48hr"] - full["mae_at_book"])
        h["H5"] = (
            "Confirmed — S1 MAE within $5 of full leveling counterfactual"
            if gap <= 5
            else "Rejected — S1 does not match full leveling benefit"
        )
    else:
        h["H5"] = "Inconclusive — missing S1 vs leveling comparison"

    return h


def write_exec_readout(
    *,
    out_dir: Path,
    hypotheses: dict[str, str],
    daily_summary: dict[str, Any],
    load_summary: dict[str, Any],
    decomposition: dict[str, Any],
    lifecycle_summary: dict[str, Any],
    leveling: dict[str, Any],
) -> Path:
    """Executive memo markdown."""
    doc_dir = out_dir.parent.parent.parent / "documentation" / "etp-slider"
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / "dqt-dial-miss-exec-readout.md"

    shares = decomposition.get("partial_shares_pct", {})
    policies = leveling.get("policies", [])

    lines = [
        "# DQT Daily Dial Miss — Executive Readout",
        "",
        f"*Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## 1. Estimated causal shares (clock / index / dial)",
        "",
        f"- Regression sample: **{decomposition.get('n_with_proxies', 'n/a'):,}** loads with LC proxies",
        f"- R² dial-only on etp50_shift: **{decomposition.get('r2_dial_only', 'n/a')}**",
        (
            f"- Partial shares (mean |contrib|): clock **{shares.get('clock', 'n/a')}%**, "
            f"index **{shares.get('index', 'n/a')}%**, dial **{shares.get('dial', 'n/a')}%**"
        ),
        "",
        "## 2. Daily dial miss magnitude",
        "",
        (
            f"- Mean dial_miss_pp @ book: **{daily_summary.get('mean_dial_miss_pp')}** "
            f"(MAE **{daily_summary.get('mae_dial_miss_pp')}** pp)"
        ),
        (
            f"- Per-load median |dial_miss_usd|: **${load_summary.get('median_abs_dial_miss_usd')}** "
            f"(n={load_summary.get('n_valid_dial_miss', 0):,})"
        ),
        "",
        "## 3. Cohort hotspots",
        "",
        "See `cohort-dial-miss.csv` for shift_segment, lead band, dial_miss_sign/quintile cuts.",
        "",
        "## 4. Quantile leveling counterfactuals",
        "",
    ]
    for p in policies:
        if p.get("policy") in LEVELING_POLICY_NAMES.values() or p.get("policy") == "S1_operational":
            lines.append(
                f"- **{p['policy']}**: att50={p.get('att50_at_book', p.get('att50_at_48hr'))}, "
                f"MAE=${p.get('mae_at_book', p.get('mae_at_48hr'))}"
            )
    lines.extend(
        [
            "",
            "## 5. S1 lifecycle cumulative dial miss",
            "",
            f"- Mean cumulative dial_miss_usd to last paint: **${lifecycle_summary.get('mean_cum_dial_miss_usd', 'n/a')}**",
            "",
            "## 6. Hypotheses",
            "",
        ]
    )
    for hid, verdict in sorted(hypotheses.items()):
        lines.append(f"- **{hid}**: {verdict}")
    lines.extend(
        [
            "",
            "## Recommended strategy",
            "",
            (
                "Prefer **daily S1 same-day r̂ paint** over static shipped alt when stability and att50 "
                "alignment matter; global full leveling yields incremental att50 only where dial_miss_usd "
                "is material — see cohort table for segment-specific tradeoffs."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def run_dial_miss_decomposition(
    *,
    data_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    phases: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "7"),
) -> dict[str, Any]:
    """Run dial miss decomposition pipeline."""
    _ = repo_root  # reserved for display_path in CLI
    data_dir = Path(data_dir or resolve_data_dir())
    out = _out_dir(data_dir)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "phases_run": list(phases),
        "out_dir": str(out),
    }

    gates = verify_data_gates(data_dir=data_dir)
    manifest["phase0_gates"] = gates
    if "0" in phases:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if not gates["passed"] and "0" in phases:
        raise RuntimeError(f"Phase 0 gates failed: {gates}")

    daily_summary: dict[str, Any] = {}
    load_summary: dict[str, Any] = {}
    decomposition: dict[str, Any] = {}
    lifecycle_summary: dict[str, Any] = {}
    leveling: dict[str, Any] = {"policies": []}

    daily = pl.DataFrame()
    load_miss = pl.DataFrame()

    if any(p in phases for p in ("1", "2", "3", "4", "5", "6", "7")):
        daily, daily_summary = build_daily_dial_miss(data_dir=data_dir)
        if "1" in phases:
            daily.write_parquet(out / "daily-dial-miss.parquet")
            (out / "daily-dial-miss-summary.json").write_text(
                json.dumps(daily_summary, indent=2, default=str)
            )

    if any(p in phases for p in ("2", "3", "4", "5", "6", "7")):
        load_miss, load_summary = build_load_dial_miss(data_dir=data_dir)
        if "2" in phases:
            load_miss.write_parquet(out / "load-dial-miss.parquet")
            (out / "load-dial-miss-summary.json").write_text(
                json.dumps(load_summary, indent=2, default=str)
            )

    if any(p in phases for p in ("3", "5", "6", "7")) and load_miss.height > 0:
        decomposition, seg_df = run_endpoint_decomposition(load_miss, data_dir=data_dir)
        if "3" in phases:
            (out / "decomposition-regression.json").write_text(
                json.dumps(decomposition, indent=2, default=str)
            )
            if not seg_df.is_empty():
                seg_df.write_csv(out / "decomposition-by-segment.csv")

    lifecycle = pl.DataFrame()
    if any(p in phases for p in ("4", "7")):
        lifecycle, lifecycle_summary = run_lifecycle_dial_miss(data_dir=data_dir)
        if "4" in phases and lifecycle.height > 0:
            lifecycle.write_parquet(out / "lifecycle-dial-miss.parquet")
            (out / "lifecycle-dial-miss-summary.json").write_text(
                json.dumps(lifecycle_summary, indent=2, default=str)
            )

    cohort_df = pl.DataFrame()
    if any(p in phases for p in ("5", "7")) and load_miss.height > 0:
        cohort_df = run_cohort_strategy(load_miss, decomposition=decomposition)
        cohort_df.write_csv(out / "cohort-dial-miss.csv")
        _plot_cohort_heatmap(cohort_df, out / "cohort-strategy-heatmap.png")

    frontier = pl.DataFrame()
    if any(p in phases for p in ("6", "7")) and load_miss.height > 0:
        leveling, frontier = run_leveling_counterfactuals(load_miss, data_dir=data_dir)
        (out / "leveling-counterfactual.json").write_text(
            json.dumps(leveling, indent=2, default=str)
        )
        if not frontier.is_empty():
            frontier.write_csv(out / "leveling-frontier.csv")

    hypotheses = synthesize_hypotheses(
        daily_summary=daily_summary,
        load_summary=load_summary,
        decomposition=decomposition,
        lifecycle_summary=lifecycle_summary,
        leveling=leveling,
    )
    manifest["hypotheses"] = hypotheses
    manifest["daily_summary"] = daily_summary
    manifest["load_summary"] = load_summary

    if "7" in phases:
        write_exec_readout(
            out_dir=out,
            hypotheses=hypotheses,
            daily_summary=daily_summary,
            load_summary=load_summary,
            decomposition=decomposition,
            lifecycle_summary=lifecycle_summary,
            leveling=leveling,
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {
        "out_dir": out,
        "n_loads": load_summary.get("n_valid_dial_miss", 0),
        "manifest": manifest,
    }
