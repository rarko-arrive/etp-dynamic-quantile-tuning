"""Shared synthetic DQT fixtures for unit tests and pipeline smoke."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl

from dqt.model.global_dqt import GlobalDQT
from dqt.panel import ALT_COLS, GRID
from dqt.score.constants import COST_COL, ID_COL
from dqt.shadow_dqt import ShadowDQT


def alts(center: float = 0.50) -> dict[str, float]:
    return {
        f"alt_{int(round(float(a) * 100))}": float(a) + (center - 0.50)
        for a in GRID
    }


def history(days: list[date], center: float = 0.50) -> pl.DataFrame:
    return pl.DataFrame(
        [{"valid_date": d, "snowflakeupdatedon": d, **alts(center)} for d in days]
    )


def panel_row(loadnumber: int, day: date, *, cost: float = 1100.0) -> dict:
    knn = {
        f"knn_{int(round(float(a) * 100))}": 1000.0 + i * 15.0
        for i, a in enumerate(GRID)
    }
    etp = {
        f"p{int(round(float(a) * 100)):02d}": knn[f"knn_{int(round(float(a) * 100))}"] - 2.0
        for a in GRID
    }
    tail = {
        f"sarima_tail_{int(round(float(a) * 100)):02d}": knn[
            f"knn_{int(round(float(a) * 100))}"
        ]
        - 1.0
        for a in GRID
    }
    return {
        ID_COL: loadnumber,
        "booked_date": day,
        "booked_on_date": day,
        "date": day,
        "week": day - timedelta(days=day.weekday()),
        "wm_r": 0.45 + (loadnumber % 10) * 0.005,
        "etp_r": 0.48,
        COST_COL: cost,
        "cost": cost,
        "mode": "Covered",
        "equipment": "Van",
        "haul_band": "Medium",
        "loadmiles": 400.0,
        **alts(0.49),
        **knn,
        **etp,
        **tail,
    }


def dial_series(start: date, n: int = 80) -> pl.DataFrame:
    rows = []
    d = start
    count = 0
    y = 0.48
    prev_y: float | None = None
    while count < n:
        if d.weekday() < 5:
            naive = prev_y if prev_y is not None else y
            sarima_cal = y + (0.002 if count >= n // 2 else 0.008)
            rows.append(
                {
                    "booked_date": d,
                    "y": y,
                    "y_hat_naive": naive,
                    "y_hat_sarima": sarima_cal,
                    "y_hat_sarima_cal": sarima_cal,
                    "in_eval": count >= n // 2,
                }
            )
            prev_y = y
            y += 0.01
            count += 1
        d += timedelta(days=1)
    return pl.DataFrame(rows)


def build_panel_rows(base: date, *, n_weeks: int = 10, per_day: int = 25) -> list[dict]:
    rows: list[dict] = []
    lid = 1000
    for w in range(n_weeks):
        week_start = base + timedelta(weeks=w)
        for day_offset in range(5):
            day = week_start + timedelta(days=day_offset)
            for i in range(per_day):
                rows.append(panel_row(lid, day, cost=1100 + i + w))
                lid += 1
    return rows


def write_smoke_layer(layer: Path, *, base: date | None = None) -> ShadowDQT:
    """Materialize minimal parquet layer for shadow / SARIMA tests."""
    base = base or date(2025, 6, 2)
    panel_rows = build_panel_rows(base, n_weeks=12, per_day=30)
    max_booked = max(r["booked_date"] for r in panel_rows)
    hist_days: list[date] = []
    d = base
    while d <= max_booked + timedelta(days=45):
        if d.weekday() < 5:
            hist_days.append(d)
        d += timedelta(days=1)
    hist = history(hist_days)
    hist.write_parquet(layer / "dqt_alt_percentiles.parquet")

    pl.DataFrame(panel_rows).write_parquet(layer / "panel_loads.parquet")

    tail_cols = ["loadnumber"] + [
        f"sarima_tail_{int(round(float(a) * 100)):02d}" for a in GRID
    ]
    pl.DataFrame(panel_rows).select(tail_cols).write_parquet(
        layer / "sarima_tail_quantiles.parquet"
    )

    min_booked = min(r["booked_date"] for r in panel_rows)
    dial = dial_series(min_booked - timedelta(days=120), n=150)
    dial.write_parquet(layer / "sarima_dial_walkforward.parquet")

    dqt = GlobalDQT(layer)
    dqt._history = dqt._normalize_history(hist)
    return ShadowDQT(layer, global_dqt=dqt)
