"""Tests for shadow alerts and bake-off SLO assessment."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from dqt.tuning.alerts import AlertSeverity, evaluate_shadow_alerts, worst_severity
from dqt.tuning.bakeoff import assess_bakeoff_slos
from tests.fixtures_dqt import write_smoke_layer


def _history_row(
    day: date,
    *,
    tail_att50: float = 0.50,
    shipped_att50: float = 0.47,
    kill_switch: bool = True,
) -> dict:
    return {
        "run_at": f"{day.isoformat()}T12:00:00",
        "scored_date": day,
        "target_date": day + timedelta(days=1),
        "n_loads": 100,
        "shipped_att50": shipped_att50,
        "tail_att50": tail_att50,
        "tail_ece_pp": 2.5,
        "tail_pinball_t_usd": 1.0,
        "kill_switch_passed": kill_switch,
        "kill_switch_reason": None,
        "proposal_source": "SARIMA_tail",
        "published_fqn": "WORKSPACE.TEST.DIAL",
        "gates_passed": True,
        "gates_reasons": None,
    }


class TestShadowAlerts:
    def test_kill_switch_alert(self) -> None:
        latest = _history_row(date(2025, 6, 6), kill_switch=False)
        alerts = evaluate_shadow_alerts(pl.DataFrame(), latest=latest)
        assert any(a.code == "kill_switch_failed" for a in alerts)
        assert worst_severity(alerts) == AlertSeverity.P1

    def test_no_alerts_on_healthy_history(self) -> None:
        base = date(2025, 6, 2)
        rows = [_history_row(base + timedelta(days=i)) for i in range(10)]
        hist = pl.DataFrame(rows)
        alerts = evaluate_shadow_alerts(hist, latest=rows[-1])
        assert alerts == []


class TestBakeoffSLOs:
    def test_bakeoff_passes_on_synthetic_history(self, tmp_path) -> None:
        write_smoke_layer(tmp_path)
        base = date(2025, 6, 2)
        rows: list[dict] = []
        d = base
        while len(rows) < 40:
            if d.weekday() < 5:
                rows.append(_history_row(d))
            d += timedelta(days=1)
        out = tmp_path / "shadow"
        out.mkdir(parents=True, exist_ok=True)
        hist_df = pl.DataFrame(rows).with_columns(pl.col("scored_date").cast(pl.Date))
        hist_df.write_parquet(out / "shadow_eval_history.parquet")
        report = assess_bakeoff_slos(
            hist_df,
            tmp_path,
            required_weekdays=30,
        )
        assert report.n_weekdays >= 30
        assert report.slos[0].passed
