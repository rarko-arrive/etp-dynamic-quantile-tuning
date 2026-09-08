"""Unit tests for shadow DQT daily loop (no Snowflake)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from dqt.model.global_dqt import GlobalDQT
from dqt.panel import ALT_COLS, GRID
from dqt.sarima_dial import global_alts_from_tail
from dqt.sarima_publish import (
    evaluate_kill_switch,
    next_weekday,
    prior_weekday,
    resolve_target_hat,
)
from dqt.score.constants import COST_COL, ID_COL
from dqt.shadow_dqt import ShadowDQT, parse_quantile_attainment_json


def _alts(center: float = 0.50) -> dict[str, float]:
    return {
        f"alt_{int(round(float(a) * 100))}": float(a) + (center - 0.50)
        for a in GRID
    }


def _history(days: list[date], center: float = 0.50) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"valid_date": d, "snowflakeupdatedon": d, **_alts(center)}
            for d in days
        ]
    )


def _panel_row(loadnumber: int, day: date, *, cost: float = 1100.0) -> dict:
    knn = {
        f"knn_{int(round(float(a) * 100))}": 1000.0 + i * 15.0
        for i, a in enumerate(GRID)
    }
    etp = {
        f"p{int(round(float(a) * 100)):02d}": knn[f"knn_{int(round(float(a) * 100))}"] - 2.0
        for a in GRID
    }
    tail = {
        f"sarima_tail_{int(round(float(a) * 100)):02d}": knn[f"knn_{int(round(float(a) * 100))}"] - 1.0
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
        **_alts(0.49),
        **knn,
        **etp,
        **tail,
    }


def _dial_series(start: date, n: int = 80) -> pl.DataFrame:
    rows = []
    d = start
    count = 0
    y = 0.48
    prev_y: float | None = None
    while count < n:
        if d.weekday() < 5:
            naive = prev_y if prev_y is not None else y
            # Slightly better than lag-1 naive on eval (±0.2pp vs ~1pp naive error).
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


def _build_panel_rows(base: date, *, n_weeks: int = 10, per_day: int = 25) -> list[dict]:
    rows: list[dict] = []
    lid = 1000
    for w in range(n_weeks):
        week_start = base + timedelta(weeks=w)
        for day_offset in range(5):
            day = week_start + timedelta(days=day_offset)
            for i in range(per_day):
                rows.append(_panel_row(lid, day, cost=1100 + i + w))
                lid += 1
    return rows


@pytest.fixture
def shadow_layer(tmp_path: Path) -> ShadowDQT:
    base = date(2025, 6, 2)
    days = [base + timedelta(days=i) for i in range(20)]
    hist = _history(days)
    hist.write_parquet(tmp_path / "dqt_alt_percentiles.parquet")

    panel_rows = _build_panel_rows(base)
    pl.DataFrame(panel_rows).write_parquet(tmp_path / "panel_loads.parquet")

    tail_cols = ["loadnumber"] + [
        f"sarima_tail_{int(round(float(a) * 100)):02d}" for a in GRID
    ]
    pl.DataFrame(panel_rows).select(tail_cols).write_parquet(
        tmp_path / "sarima_tail_quantiles.parquet"
    )

    dial = _dial_series(base - timedelta(days=120))
    dial.write_parquet(tmp_path / "sarima_dial_walkforward.parquet")

    dqt = GlobalDQT(tmp_path)
    dqt._history = dqt._normalize_history(hist)
    return ShadowDQT(tmp_path, global_dqt=dqt)


class TestSarimaPublish:
    def test_weekday_helpers(self) -> None:
        fri = date(2025, 6, 6)
        assert next_weekday(fri) == date(2025, 6, 9)
        assert prior_weekday(date(2025, 6, 9)) == fri

    def test_kill_switch_passes_when_sarima_beats_naive(
        self, shadow_layer: ShadowDQT
    ) -> None:
        dial = pl.read_parquet(shadow_layer.dqt.sarima_wf_path)
        target = next_weekday(date(2025, 6, 12))
        kill = evaluate_kill_switch(dial, target=target)
        assert kill.passed is True
        assert kill.target_hat is not None
        assert resolve_target_hat(dial, target) == pytest.approx(kill.target_hat)

    def test_kill_switch_trips_when_sarima_worse(self) -> None:
        dial = _dial_series(date(2025, 1, 6))
        dial = dial.with_columns(pl.col("y_hat_naive").alias("y_hat_sarima_cal"))
        kill = evaluate_kill_switch(dial, target=next_weekday(date(2025, 6, 12)))
        assert kill.passed is False
        assert kill.reason is not None


class TestGlobalAltsFromTail:
    def test_global_row_shape(self, shadow_layer: ShadowDQT) -> None:
        panel = pl.read_parquet(shadow_layer.panel_path)
        dial = pl.read_parquet(shadow_layer.dqt.sarima_wf_path)
        target = next_weekday(date(2025, 6, 12))
        prior = shadow_layer.dqt.as_of(prior_weekday(target))
        alts = global_alts_from_tail(
            panel, dial, prior.alts, target, min_n=30, cal_window_days=28
        )
        assert set(alts.keys()) == set(ALT_COLS)
        vals = [alts[c] for c in ALT_COLS]
        assert all(0.05 <= v <= 0.95 for v in vals)
        assert vals == sorted(vals)


class TestShadowDQT:
    def test_yesterday_booking_day_respects_lag(self) -> None:
        shadow = ShadowDQT(Path("/tmp/unused"), settlement_lag_days=3)
        assert shadow.yesterday_booking_day(date(2025, 6, 11)) == date(2025, 6, 6)

    def test_score_day(self, shadow_layer: ShadowDQT) -> None:
        scored = date(2025, 6, 6)
        out = shadow_layer.score_day(scored)
        assert out["n_loads"] == 25
        assert out["shipped_att50"] is not None
        assert out["tail_att50"] is not None
        assert out["tail_ece_pp"] is not None
        assert out["shipped_crps_usd"] is not None
        assert out["tail_crps_usd"] is not None
        assert out["shipped_ece_pp"] is not None
        assert out["crps_lift_usd"] == pytest.approx(
            out["shipped_crps_usd"] - out["tail_crps_usd"]
        )
        assert out["ece_lift_pp"] == pytest.approx(
            out["shipped_ece_pp"] - out["tail_ece_pp"]
        )
        assert out["hybrid_att50"] is None
        assert out["quantile_attainment_json"] is not None

        level = parse_quantile_attainment_json(out["quantile_attainment_json"])
        assert level.height == len(GRID) * 2
        assert set(level["model"].unique().to_list()) == {"DQT/ETP", "SARIMA_tail"}
        assert level.filter(pl.col("quantile") == 0.5).height == 2

    def test_score_day_includes_hybrid_when_columns_present(
        self, shadow_layer: ShadowDQT
    ) -> None:
        scored = date(2025, 6, 6)
        panel = pl.read_parquet(shadow_layer.panel_path)
        hybrid_exprs = [
            pl.col(f"p{int(round(float(a) * 100)):02d}").alias(
                f"hybrid_{int(round(float(a) * 100)):02d}"
            )
            for a in GRID
        ]
        panel.with_columns(hybrid_exprs).write_parquet(shadow_layer.panel_path)

        out = shadow_layer.score_day(scored)
        assert out["hybrid_att50"] is not None
        assert out["hybrid_crps_usd"] is not None
        assert out["hybrid_ece_pp"] is not None
        level = parse_quantile_attainment_json(out["quantile_attainment_json"])
        assert "Hybrid" in level["model"].unique().to_list()

    def test_run_daily_persists_distribution_metrics(
        self, shadow_layer: ShadowDQT, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dqt.model import global_dqt as mod

        monkeypatch.setattr(mod, "create_table", lambda *a, **k: None)

        shadow_layer.run_daily(
            as_of=date(2025, 6, 13),
            scored_date=date(2025, 6, 6),
            write_snowflake=False,
            min_n=30,
        )
        hist = pl.read_parquet(shadow_layer.history_path)
        row = hist.row(-1, named=True)
        assert row["shipped_crps_usd"] is not None
        assert row["tail_crps_usd"] is not None
        assert row["quantile_attainment_json"] is not None
        level = parse_quantile_attainment_json(row["quantile_attainment_json"])
        assert level.height >= len(GRID) * 2

    def test_propose_sarima_tail(self, shadow_layer: ShadowDQT) -> None:
        target = next_weekday(date(2025, 6, 12))
        prop, full = shadow_layer.propose_global(
            target, source="SARIMA_tail", min_n=30, publish_mode="cap"
        )
        assert prop.method == "sarima_tail"
        assert prop.alt_50 is not None
        assert full is not None

    def test_run_daily_audit_no_snowflake(
        self, shadow_layer: ShadowDQT, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dqt.model import global_dqt as mod

        def fail_sf(*args, **kwargs):
            raise AssertionError("Snowflake should not be called in unit test")

        monkeypatch.setattr(mod, "create_table", fail_sf)

        report = shadow_layer.run_daily(
            as_of=date(2025, 6, 13),
            scored_date=date(2025, 6, 6),
            write_snowflake=True,
            dry_run=False,
            min_n=30,
        )
        assert report.n_loads == 25
        assert report.kill_switch_passed is True
        assert report.proposal_source == "SARIMA_tail"
        assert shadow_layer.history_path.exists()

    def test_run_daily_skips_sf_when_no_loads(
        self, shadow_layer: ShadowDQT, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dqt.model import global_dqt as mod

        def fail_sf(*args, **kwargs):
            raise AssertionError("Snowflake should not be called when n_loads=0")

        monkeypatch.setattr(mod, "create_table", fail_sf)

        report = shadow_layer.run_daily(
            as_of=date(2025, 6, 13),
            scored_date=date(2099, 1, 2),
            write_snowflake=True,
            min_n=30,
        )
        assert report.n_loads == 0
        assert report.published_fqn is None

    def test_run_daily_dual_audit_writes_full_proposal(
        self, shadow_layer: ShadowDQT
    ) -> None:
        report = shadow_layer.run_daily(
            as_of=date(2025, 6, 13),
            scored_date=date(2025, 6, 6),
            write_snowflake=False,
            min_n=30,
            publish_mode="cap",
            dual_write_audit=True,
        )
        assert report.dual_audit_path is not None
        assert report.full_proposal_alt_50 is not None
        assert report.capped_proposal_alt_50 is not None

    def test_run_daily_kill_switch_fallback(self, shadow_layer: ShadowDQT) -> None:
        dial = pl.read_parquet(shadow_layer.dqt.sarima_wf_path)
        dial = dial.with_columns(pl.col("y_hat_naive").alias("y_hat_sarima_cal"))
        dial.write_parquet(shadow_layer.dqt.sarima_wf_path)

        report = shadow_layer.run_daily(
            as_of=date(2025, 6, 13),
            scored_date=date(2025, 6, 6),
            write_snowflake=False,
            min_n=30,
        )
        assert report.kill_switch_passed is False
        assert report.proposal_source == "shipped_alt_50"
        assert report.published_fqn is None

    def test_global_dqt_sarima_tail_propose(self, shadow_layer: ShadowDQT) -> None:
        target = date(2025, 6, 13)
        prop = shadow_layer.dqt.propose(
            method="sarima_tail",
            as_of=target,
            max_weekly_move=0.10,
            min_n=30,
        )
        assert prop.method == "sarima_tail"
        assert 0.05 <= prop.alt_50 <= 0.95
