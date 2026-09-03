"""Unit tests for :mod:`dqt.model.global_dqt` (no Snowflake required)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dqt.model.global_dqt import DialProposal, DialSchedule, GlobalDQT, SpotCheck
from dqt.panel import ALT_COLS, GRID, WEATHERMAN_COLS
from dqt.score.constants import COST_COL, ID_COL


def _alts(center: float = 0.50, spread: float = 0.0) -> dict[str, float]:
    """Monotone dial around ``center`` (identity when spread=0 and center=0.5)."""
    out = {}
    for a in GRID:
        # Rigid shift of nominal grid by (center - 0.5), optional fan.
        v = float(a) + (center - 0.50) + spread * (float(a) - 0.50)
        out[f"alt_{int(round(float(a) * 100))}"] = float(np.clip(v, 0.05, 0.95))
    return out


def _schedule(day: date, center: float = 0.50) -> DialSchedule:
    return DialSchedule(valid_date=day, alts=_alts(center))


def _history_frame(days: list[date], centers: list[float] | None = None) -> pl.DataFrame:
    centers = centers or [0.50] * len(days)
    rows = []
    for d, c in zip(days, centers, strict=True):
        rows.append(
            {
                "valid_date": d,
                "snowflakeupdatedon": d,
                **_alts(c),
            }
        )
    return pl.DataFrame(rows)


def _load_row(
    loadnumber: int,
    day: date,
    *,
    cost: float = 1200.0,
    knn0: float = 1000.0,
    step: float = 20.0,
) -> dict:
    knn = {f"knn_{int(round(float(a) * 100))}": knn0 + i * step for i, a in enumerate(GRID)}
    # Booked ETP slightly below Weatherman (as in real load 6725820).
    etp = {
        f"p{int(round(float(a) * 100)):02d}": knn[f"knn_{int(round(float(a) * 100))}"] - 3.0
        for a in GRID
    }
    return {
        ID_COL: loadnumber,
        "booked_on_date": day.isoformat(),
        "date": day.isoformat(),
        COST_COL: cost,
        **_alts(0.49),
        **knn,
        **etp,
    }


@pytest.fixture
def tmp_dqt(tmp_path: Path) -> GlobalDQT:
    days = [date(2025, 3, 24) + timedelta(days=i) for i in range(5)]
    hist = _history_frame(days, centers=[0.48, 0.49, 0.50, 0.51, 0.52])
    hist.write_parquet(tmp_path / "dqt_alt_percentiles.parquet")

    rows = [_load_row(1000 + i, days[2], cost=1100.0 + 10 * i) for i in range(5)]
    # Extra loads on day 3 for spot_check_date.
    rows += [_load_row(2000 + i, days[3], cost=1150.0) for i in range(8)]
    pl.DataFrame(rows).write_parquet(tmp_path / "features.parquet")

    dqt = GlobalDQT(tmp_path, cache_hours=24.0)
    # Avoid Snowflake: seed in-memory history from cache.
    dqt._history = dqt._normalize_history(hist)
    return dqt


class TestDialSchedule:
    def test_level_and_array(self) -> None:
        s = _schedule(date(2025, 1, 1), center=0.55)
        assert s.alt_50 == pytest.approx(0.55)
        assert s.level(0.05) == pytest.approx(0.10)  # 0.05 + 0.05
        arr = s.as_array()
        assert arr.shape == (len(GRID),)
        assert arr[9] == pytest.approx(0.55)

    def test_from_row(self) -> None:
        row = {"valid_date": "2025-03-26", **_alts(0.4888)}
        s = DialSchedule.from_row(row)
        assert s.valid_date == date(2025, 3, 26)
        assert s.alt_50 == pytest.approx(0.4888)


class TestReadAttach:
    def test_as_of_and_latest(self, tmp_dqt: GlobalDQT) -> None:
        s = tmp_dqt.as_of("2025-03-26")
        assert s.valid_date == date(2025, 3, 26)
        assert s.alt_50 == pytest.approx(0.50)

        latest = tmp_dqt.latest(today=date(2025, 3, 28))
        assert latest.valid_date == date(2025, 3, 28)
        assert latest.alt_50 == pytest.approx(0.52)

        with pytest.raises(LookupError):
            tmp_dqt.as_of("2025-01-01")
        # After last history day → most recent prior schedule.
        prior = tmp_dqt.latest(today=date(2025, 4, 1))
        assert prior.valid_date == date(2025, 3, 28)

    def test_history_filter_and_alias(self, tmp_dqt: GlobalDQT) -> None:
        h = tmp_dqt.history(start="2025-03-26", end="2025-03-28")
        assert h.height == 2
        assert set(h["valid_date"].to_list()) == {date(2025, 3, 26), date(2025, 3, 27)}
        alias = tmp_dqt.read_latest_daily_offsets(start="2025-03-26", end="2025-03-28")
        assert alias.height == h.height

    def test_attach(self, tmp_dqt: GlobalDQT) -> None:
        loads = pl.DataFrame(
            {
                ID_COL: [1, 2],
                "booked_on_date": ["2025-03-26", "2025-03-28"],
            }
        )
        joined = tmp_dqt.attach(loads)
        assert "alt_50" in joined.columns
        assert joined["alt_50"].to_list() == pytest.approx([0.50, 0.52])


class TestApplyEvaluate:
    def test_apply_matches_quantile_at(self, tmp_dqt: GlobalDQT) -> None:
        from dqt.conformal import quantile_at

        feat = pl.read_parquet(tmp_dqt.features_path).head(1)
        sched = tmp_dqt.as_of(feat["booked_on_date"][0])
        dollars = tmp_dqt.apply(feat, sched)
        assert dollars.shape == (1, len(GRID))

        qmat = feat.select(WEATHERMAN_COLS).to_numpy()
        expected = quantile_at(qmat, np.full(1, sched.alt_50))
        mid = list(GRID).index(0.5)
        assert dollars[0, mid] == pytest.approx(expected[0], rel=1e-9)

    def test_apply_frame_and_evaluate(self, tmp_dqt: GlobalDQT) -> None:
        feat = pl.read_parquet(tmp_dqt.features_path)
        # Restrict to loads that have dial day in history join sense — alts already on features.
        scored = tmp_dqt.apply_frame(feat.filter(pl.col(ID_COL) < 2000))
        assert all(c in scored.columns for c in ("dial_5", "dial_50", "dial_95"))

        ev = tmp_dqt.evaluate(scored)
        assert ev.height == 1
        assert "att50" in ev.columns
        assert "ece_pp" in ev.columns
        assert 0.0 <= ev["att50"][0] <= 1.0

    def test_diagnostics_and_compare(self, tmp_dqt: GlobalDQT) -> None:
        diag = tmp_dqt.diagnostics()
        assert "alt50_dod_pp" in diag.columns
        assert diag["alt50_dod_pp"].drop_nulls()[0] == pytest.approx(1.0)  # 0.48→0.49

        a = tmp_dqt.as_of("2025-03-28")
        b = tmp_dqt.as_of("2025-03-24")
        cmp_ = tmp_dqt.compare(a, b)
        row50 = cmp_.filter(pl.col("level") == 0.5).row(0, named=True)
        assert row50["delta_pp"] == pytest.approx(4.0)  # 0.52 - 0.48


class TestSpotCheck:
    def test_spot_check_one(self, tmp_dqt: GlobalDQT) -> None:
        sc = tmp_dqt.spot_check(1000)
        assert isinstance(sc, SpotCheck)
        assert sc.loadnumber == 1000
        assert sc.booked_on_date == date(2025, 3, 26)
        assert "dial_50" in sc.dial_applied
        assert sc.p50 == pytest.approx(sc.etp_booked["p50"])
        assert sc.cost is not None
        assert sc.wm_r is not None
        summary = sc.summary()
        assert "delta_50" in summary
        wide = sc.to_frame()
        assert wide.height == 1
        long = sc.to_frame(long=True)
        assert long.height == len(GRID)

    def test_spot_check_date(self, tmp_dqt: GlobalDQT) -> None:
        batch = tmp_dqt.spot_check_date("2025-03-27", n=3, sample="first")
        assert batch.height == 3
        assert set(batch["booked_on_date"].to_list()) == {date(2025, 3, 27)}


class TestProposeGates:
    def test_identity_and_shift(self, tmp_dqt: GlobalDQT) -> None:
        ident = tmp_dqt.propose(method="identity", as_of="2025-03-29", max_weekly_move=0.10)
        assert ident.method == "identity"
        assert ident.alt_50 == pytest.approx(0.50)
        assert ident.passed

        shift = tmp_dqt.propose(
            method="shift",
            as_of="2025-03-29",
            r_hat_50=0.55,
            max_weekly_move=0.10,
        )
        assert shift.alt_50 == pytest.approx(0.55)
        assert shift.passed

    def test_gates_fail_closed_on_large_move(self, tmp_dqt: GlobalDQT) -> None:
        prior = tmp_dqt.as_of("2025-03-28")
        proposal = DialProposal(
            valid_date=date(2025, 3, 29),
            alts=_alts(0.80),
            method="shift",
            gates={},
        )
        gated = tmp_dqt.gates(proposal, prior=prior, max_weekly_move=0.05)
        assert gated.passed is False
        assert any("move" in r for r in gated.gates["reasons"])
        # Still clipped into bounds.
        assert 0.05 <= gated.alt_50 <= 0.95

    def test_publish_staging(self, tmp_dqt: GlobalDQT) -> None:
        prop = tmp_dqt.propose(
            method="shift",
            as_of="2025-03-29",
            r_hat_50=0.53,
            max_weekly_move=0.10,
        )
        path = tmp_dqt.publish(prop, target="staging")
        assert isinstance(path, Path)
        assert path.exists()
        written = pl.read_parquet(path)
        assert written["alt_50"][0] == pytest.approx(0.53)

    def test_publish_rejects_failed_gates(self, tmp_dqt: GlobalDQT) -> None:
        prior = tmp_dqt.as_of("2025-03-28")
        bad = tmp_dqt.gates(
            DialProposal(valid_date=date(2025, 3, 29), alts=_alts(0.90), method="shift"),
            prior=prior,
            max_weekly_move=0.01,
        )
        with pytest.raises(PermissionError):
            tmp_dqt.publish(bad)

    def test_write_target_from_env(self, tmp_dqt: GlobalDQT, monkeypatch: pytest.MonkeyPatch) -> None:
        from dqt.model.global_dqt import resolve_dial_write_target

        monkeypatch.setenv("SNOWFLAKE_DATABASE", "DATA_SCIENCE_WORKSPACE")
        monkeypatch.setenv("SNOWFLAKE_SCHEMA", "RARKO")
        monkeypatch.delenv("DQT_DIAL_TABLE", raising=False)
        dest = resolve_dial_write_target()
        assert dest.fqn == "DATA_SCIENCE_WORKSPACE.RARKO.HISTORICAL_ETP_DYNAMIC_QUANTILES"
        assert dest.is_prod is False

        override = tmp_dqt.write_target(schema="RARKO", database="DATA_SCIENCE_WORKSPACE")
        assert override.schema == "RARKO"

    def test_publish_snowflake_blocks_prod(
        self, tmp_dqt: GlobalDQT, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dqt.model import global_dqt as mod

        prop = tmp_dqt.propose(
            method="shift",
            as_of="2025-03-29",
            r_hat_50=0.53,
            max_weekly_move=0.10,
        )
        calls: list[dict] = []

        def fake_create_table(df, table_name, **kwargs):
            calls.append({"df": df, "table": table_name, **kwargs})
            return df.height

        monkeypatch.setattr(mod, "create_table", fake_create_table)

        with pytest.raises(PermissionError, match="production dial table"):
            tmp_dqt.publish(
                prop,
                target="snowflake",
                database="DATA_SCIENCE",
                schema="ETP_DYNAMIC_QUANTILE_TUNING",
                table="HISTORICAL_ETP_DYNAMIC_QUANTILES",
                also_parquet=False,
            )
        assert calls == []

        dest = tmp_dqt.publish(
            prop,
            target="snowflake",
            database="DATA_SCIENCE_WORKSPACE",
            schema="RARKO",
            table="HISTORICAL_ETP_DYNAMIC_QUANTILES",
            also_parquet=False,
            dry_run=True,
        )
        assert dest.fqn == "DATA_SCIENCE_WORKSPACE.RARKO.HISTORICAL_ETP_DYNAMIC_QUANTILES"
        assert calls == []

        dest = tmp_dqt.publish(
            prop,
            target="snowflake",
            database="DATA_SCIENCE_WORKSPACE",
            schema="RARKO",
            also_parquet=True,
        )
        assert dest.fqn.endswith(".RARKO.HISTORICAL_ETP_DYNAMIC_QUANTILES")
        assert len(calls) == 1
        assert calls[0]["table"] == "HISTORICAL_ETP_DYNAMIC_QUANTILES"
        assert calls[0]["schema"] == "RARKO"
        assert "alt_50" in calls[0]["df"].columns
        assert "method" not in calls[0]["df"].columns
        assert (tmp_dqt.data_dir / "staging").exists()

    def test_sarima_propose_from_artifact(self, tmp_dqt: GlobalDQT) -> None:
        wf = pl.DataFrame(
            {
                "booked_date": [date(2025, 3, 28), date(2025, 3, 29)],
                "y_hat_sarima": [0.50, 0.54],
                "y_hat_sarima_cal": [0.50, 0.54],
                "y_hat_naive": [0.49, 0.50],
            }
        )
        wf.write_parquet(tmp_dqt.sarima_wf_path)
        prop = tmp_dqt.propose(
            method="sarima",
            as_of="2025-03-29",
            max_weekly_move=0.10,
        )
        assert prop.method == "sarima"
        assert prop.alt_50 == pytest.approx(0.54)
        assert prop.passed


def test_spot_check_real_features_optional() -> None:
    """Integration smoke against repo ``data/features.parquet`` when present."""
    root_data = Path(__file__).resolve().parents[1] / "data"
    feat_path = root_data / "features.parquet"
    if not feat_path.exists():
        pytest.skip("data/features.parquet not present")

    dqt = GlobalDQT(root_data)
    # Don't hit Snowflake — synthesize history from features' alt columns for one day.
    feat = pl.scan_parquet(feat_path).filter(pl.col(ID_COL) == 6725820).collect()
    if feat.is_empty():
        pytest.skip("load 6725820 not in features")

    day = feat["booked_on_date"][0]
    # Seed a minimal history so as_of works if needed; spot_check uses features alts.
    hist = feat.select(
        pl.lit(day).str.to_date().alias("valid_date"),
        *[pl.col(c) for c in ALT_COLS],
    )
    dqt._history = dqt._normalize_history(hist)

    sc = dqt.spot_check(6725820, features=feat)
    assert sc.loadnumber == 6725820
    assert sc.alt_50 == pytest.approx(feat["alt_50"][0])
    assert sc.p50 == pytest.approx(feat["p50"][0])
    assert sc.dial_applied_50 > 0
    # Dial-applied at alt_50 should be near knn interpolated — finite and near p50 band.
    assert abs(sc.dial_applied_50 - sc.p50) < 50.0
