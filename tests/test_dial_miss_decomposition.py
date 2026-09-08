"""Tests for DQT dial miss decomposition."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dqt import resolve_data_dir
from dqt.conformal import quantile_at
from dqt.dial_miss.decomposition import (
    _quote_from_alt_batch,
    _r2,
    build_daily_dial_miss,
    synthesize_hypotheses,
    verify_data_gates,
)
from dqt.sarima_dial import materialize_pp50_batch


def _has_repo_data_layer() -> bool:
    data_dir = resolve_data_dir()
    return (data_dir / "sarima_dial_walkforward.parquet").is_file()


@pytest.fixture
def dial_miss_layer(tmp_path: Path) -> Path:
    dates = [date(2025, 6, 2), date(2025, 6, 3)]
    pl.DataFrame(
        {
            "booked_date": dates,
            "y_hat_sarima_cal": [0.52, 0.51],
            "y_hat_sarima": [0.52, 0.51],
            "y_hat_naive": [0.50, 0.50],
            "y": [0.50, 0.49],
            "in_eval": [True, True],
            "n_loads": [100, 110],
        }
    ).write_parquet(tmp_path / "sarima_dial_walkforward.parquet")
    pl.DataFrame(
        {
            "valid_date": dates,
            "snowflakeupdatedon": dates,
            "alt_50": [0.48, 0.47],
        }
    ).write_parquet(tmp_path / "dqt_alt_percentiles.parquet")
    return tmp_path


def test_dial_miss_pp_hand_check():
    r_hat, alt = 0.50, 0.48
    assert (r_hat - alt) * 100 == pytest.approx(2.0)


def test_dial_miss_usd_sign_matches_pp_on_synthetic_knn():
    knn = np.array(
        [
            [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                270,
                280,
            ]
        ],
        dtype=np.float64,
    )
    r_hat = np.array([0.52])
    alt = np.array([0.48])
    q_s = materialize_pp50_batch(knn, r_hat)[0]
    q_a = _quote_from_alt_batch(knn, alt)[0]
    miss = q_s - q_a
    assert miss > 0
    assert (r_hat[0] - alt[0]) > 0


def test_quantile_at_monotonicity():
    knn = np.sort(np.linspace(100, 200, 19).reshape(1, -1), axis=1)
    knn3 = np.repeat(knn, 3, axis=0)
    alts = np.array([0.45, 0.50, 0.55])
    quotes = quantile_at(knn3, alts, assume_sorted=True)
    assert quotes[0] < quotes[1] < quotes[2]


def test_regression_shares_sum_on_synthetic():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y_hat = x * 10.0
    assert _r2(y, y_hat) == pytest.approx(1.0)


@pytest.mark.skipif(not _has_repo_data_layer(), reason="local data layer required")
def test_verify_data_gates_passes_on_repo_data():
    gates = verify_data_gates()
    assert gates["passed"] is True
    assert gates["checks"]


def test_build_daily_dial_miss_has_dial_miss_pp(dial_miss_layer: Path):
    daily, summary = build_daily_dial_miss(data_dir=dial_miss_layer)
    assert "dial_miss_pp" in daily.columns
    assert summary.get("mae_dial_miss_pp") is not None


def test_synthesize_hypotheses_keys():
    h = synthesize_hypotheses(
        daily_summary={"mean_dial_miss_pp": 1.0, "mae_dial_miss_pp": 2.0},
        load_summary={"median_abs_dial_miss_usd": 15.0, "n_valid_dial_miss": 1000},
        decomposition={"r2_dial_only": 0.05, "partial_shares_pct": {"dial": 3}},
        lifecycle_summary={"mean_cum_dial_miss_usd": 20.0},
        leveling={"policies": []},
    )
    assert set(h) >= {"H1", "H2", "H3", "H4", "H5"}


def test_verify_data_gates_fails_on_empty_dir(tmp_path):
    gates = verify_data_gates(data_dir=tmp_path)
    assert gates["passed"] is False
