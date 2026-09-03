"""Unit tests for guardrail assessment helpers (no Snowflake)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from dqt.panel import ALT_COLS, GRID
from dqt.tuning.assess import (
    blocked_day_counterfactual,
    cap_global_alts,
    fire_rates,
    shipped_weekday_moves,
)


def _alts(center: float = 0.50) -> dict[str, float]:
    return {
        f"alt_{int(round(float(a) * 100))}": float(a) + (center - 0.50)
        for a in GRID
    }


def test_cap_global_alts_clips_per_level() -> None:
    prior = _alts(0.50)
    proposal = _alts(0.60)
    capped = cap_global_alts(prior, proposal, max_move=0.05)
    for col in ALT_COLS:
        assert abs(capped[col] - prior[col]) <= 0.05 + 1e-9


def test_shipped_weekday_moves_skips_weekends() -> None:
    mon = date(2025, 1, 6)
    tue = date(2025, 1, 7)
    sat = date(2025, 1, 11)
    hist = pl.DataFrame(
        [
            {"valid_date": mon, **_alts(0.50)},
            {"valid_date": tue, **_alts(0.52)},
            {"valid_date": sat, **_alts(0.55)},
        ]
    )
    moves = shipped_weekday_moves(hist)
    assert moves.height == 1
    assert moves["alt50_move_pp"][0] == pytest.approx(2.0)


def test_blocked_day_counterfactual() -> None:
    tail = pl.DataFrame(
        {
            "target_date": [date(2025, 1, 7)],
            "max_move_pp": [8.0],
            "prior_alt_50": [0.45],
            "prop_alt_50": [0.55],
        }
    )
    cf = blocked_day_counterfactual(tail, block_pp=5.0, cap_pp=5.0)
    assert cf["n_blocked"][0] == 1
    assert cf["mean_full_shift_pp"][0] == pytest.approx(10.0)
    assert cf["mean_capped_shift_pp"][0] == pytest.approx(5.0)


def test_fire_rates_thresholds() -> None:
    moves = pl.Series([1.0, 4.0, 6.0, 9.0])
    rates = fire_rates(moves, (5.0, 8.0))
    assert rates.filter(pl.col("threshold_pp") == 5.0)["n_exceed"][0] == 2
    assert rates.filter(pl.col("threshold_pp") == 8.0)["n_exceed"][0] == 1
