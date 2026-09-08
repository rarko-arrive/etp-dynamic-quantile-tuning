"""Unit tests for SARIMA_tail materialization and global alt schedule."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from dqt.hybrid import HybridConfig
from dqt.panel import ALT_COLS, GRID
from dqt.sarima_dial import global_alts_from_tail, materialize_sarima_tail
from dqt.sarima_publish import next_weekday, prior_weekday
from dqt.tuning.assess import cap_global_alts
from tests.fixtures_dqt import build_panel_rows, dial_series, history, write_smoke_layer


@pytest.fixture
def tail_layer(tmp_path: Path) -> Path:
    write_smoke_layer(tmp_path)
    return tmp_path


class TestGlobalAltsFromTail:
    def test_monotone_and_clip(self, tail_layer: Path) -> None:
        from dqt.shadow_dqt import ShadowDQT

        shadow = ShadowDQT(tail_layer)
        panel = pl.read_parquet(shadow.panel_path)
        dial = pl.read_parquet(shadow.dqt.sarima_wf_path)
        target = next_weekday(date(2025, 6, 12))
        prior = shadow.dqt.as_of(prior_weekday(target))
        alts = global_alts_from_tail(
            panel, dial, prior.alts, target, min_n=30, cal_window_days=28
        )
        vals = [alts[c] for c in ALT_COLS]
        assert vals == sorted(vals)
        assert all(0.05 <= v <= 0.95 for v in vals)


class TestCapGlobalAlts:
    def test_clips_per_level(self) -> None:
        prior = {c: 0.50 for c in ALT_COLS}
        proposal = {c: 0.58 for c in ALT_COLS}
        capped = cap_global_alts(prior, proposal, 0.05)
        assert all(abs(capped[c] - 0.50) <= 0.05 + 1e-9 for c in ALT_COLS)


class TestMaterializeSarimaTail:
    def test_materialize_shape_and_monotonicity(self, tail_layer: Path) -> None:
        panel = pl.read_parquet(tail_layer / "panel_loads.parquet")
        dial = pl.read_parquet(tail_layer / "sarima_dial_walkforward.parquet")
        config = HybridConfig(min_n=30, cal_window_days=28)
        out = materialize_sarima_tail(
            panel,
            dial,
            config,
            tail_layer,
            data_dir=tail_layer,
            validate=False,
        )
        assert "loadnumber" in out.columns
        qcols = [f"sarima_tail_{int(round(float(a) * 100)):02d}" for a in GRID]
        assert set(qcols).issubset(set(out.columns))
        sample = out.head(1)
        dollars = [float(sample[c][0]) for c in qcols]
        assert dollars == sorted(dollars)
