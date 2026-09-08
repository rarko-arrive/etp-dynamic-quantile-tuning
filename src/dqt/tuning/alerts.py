"""Shadow KPI alerting for daily DQT operations."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any

import polars as pl


class AlertSeverity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


@dataclass(frozen=True, slots=True)
class ShadowAlert:
    severity: AlertSeverity
    code: str
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


def _rolling_mean(series: pl.Series, window: int) -> float | None:
    if series.len() < window:
        return None
    tail = series.tail(window)
    if tail.null_count() == window:
        return None
    return float(tail.drop_nulls().mean())


def _consecutive_outside_band(
    values: pl.Series, lo: float, hi: float, *, min_streak: int
) -> bool:
    streak = 0
    for v in values:
        if v is None:
            streak = 0
            continue
        fv = float(v)
        if fv < lo or fv > hi:
            streak += 1
            if streak >= min_streak:
                return True
        else:
            streak = 0
    return False


def evaluate_shadow_alerts(
    history: pl.DataFrame,
    *,
    latest: dict[str, Any] | None = None,
    tail_att50_band: tuple[float, float] = (0.49, 0.51),
    att50_lift_min_pp: float = 1.0,
    att50_lift_window: int = 5,
    ece_max_pp: float = 4.0,
    ece_window: int = 10,
) -> list[ShadowAlert]:
    """Evaluate alert conditions on shadow eval history + optional latest row."""
    alerts: list[ShadowAlert] = []

    if latest is not None:
        if not latest.get("kill_switch_passed", True):
            alerts.append(
                ShadowAlert(
                    severity=AlertSeverity.P1,
                    code="kill_switch_failed",
                    message="SARIMA kill-switch failed — no Snowflake publish",
                    context={
                        "reason": latest.get("kill_switch_reason"),
                        "target_date": str(latest.get("target_date")),
                    },
                )
            )
        if latest.get("n_loads") == 0:
            alerts.append(
                ShadowAlert(
                    severity=AlertSeverity.P1,
                    code="empty_score_day",
                    message="Scored day has zero loads — check features/panel freshness",
                    context={"scored_date": str(latest.get("scored_date"))},
                )
            )

    if history.is_empty():
        return alerts

    hist = history.sort("run_at") if "run_at" in history.columns else history
    tail_att50 = hist["tail_att50"] if "tail_att50" in hist.columns else pl.Series([])
    shipped_att50 = (
        hist["shipped_att50"] if "shipped_att50" in hist.columns else pl.Series([])
    )
    tail_ece = hist["tail_ece_pp"] if "tail_ece_pp" in hist.columns else pl.Series([])

    if tail_att50.len() >= 5:
        lo, hi = tail_att50_band
        if _consecutive_outside_band(tail_att50, lo, hi, min_streak=2):
            alerts.append(
                ShadowAlert(
                    severity=AlertSeverity.P2,
                    code="tail_att50_band",
                    message=f"tail_att50 outside {lo:.0%}–{hi:.0%} for 2 consecutive scored days",
                    context={
                        "last_tail_att50": tail_att50.tail(2).to_list(),
                        "band": [lo, hi],
                    },
                )
            )

    if tail_att50.len() >= att50_lift_window and shipped_att50.len() >= att50_lift_window:
        lift = (
            tail_att50.tail(att50_lift_window) - shipped_att50.tail(att50_lift_window)
        ).mean()
        if lift is not None and float(lift) < att50_lift_min_pp / 100.0:
            alerts.append(
                ShadowAlert(
                    severity=AlertSeverity.P2,
                    code="att50_lift_low",
                    message=(
                        f"tail_att50 − shipped_att50 < {att50_lift_min_pp}pp "
                        f"over last {att50_lift_window} scored days"
                    ),
                    context={"mean_lift_pp": float(lift) * 100.0},
                )
            )

    ece_roll = _rolling_mean(tail_ece, ece_window)
    if ece_roll is not None and ece_roll > ece_max_pp:
        alerts.append(
            ShadowAlert(
                severity=AlertSeverity.P2,
                code="ece_high",
                message=f"Rolling {ece_window}d ECE {ece_roll:.2f}pp exceeds {ece_max_pp}pp",
                context={"ece_pp": ece_roll},
            )
        )

    if "gates_passed" in hist.columns and "proposal_source" in hist.columns:
        blocked = hist.filter(
            (~pl.col("gates_passed")) & (pl.col("proposal_source") == "SARIMA_tail")
        )
        if blocked.height > 0 and "alt50_move_pp" in hist.columns:
            big = blocked.filter(pl.col("alt50_move_pp").abs() > 8.0)
            if big.height > 0:
                alerts.append(
                    ShadowAlert(
                        severity=AlertSeverity.P2,
                        code="uncapped_move_gt_8pp",
                        message="Uncapped proposal exceeded 8pp on blocked publish days",
                        context={"n_days": big.height},
                    )
                )

    return alerts


def post_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
    """POST JSON alert payload to Slack/Teams incoming webhook."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status >= 400:
            raise RuntimeError(f"webhook returned HTTP {resp.status}")


def format_slack_payload(alerts: list[ShadowAlert], *, latest: dict[str, Any] | None) -> dict:
    lines = [f"*{a.severity.value}* `{a.code}`: {a.message}" for a in alerts]
    if not lines:
        lines = ["All shadow SLO checks passed."]
    if latest:
        lines.insert(
            0,
            (
                f"Shadow run: scored={latest.get('scored_date')} "
                f"target={latest.get('target_date')} "
                f"tail_att50={latest.get('tail_att50')} "
                f"shipped_att50={latest.get('shipped_att50')}"
            ),
        )
    return {"text": "\n".join(lines)}


def send_alerts(
    alerts: list[ShadowAlert],
    *,
    latest: dict[str, Any] | None = None,
    webhook_url: str | None = None,
) -> None:
    url = webhook_url or os.environ.get("DQT_ALERT_WEBHOOK")
    if not url or not alerts:
        return
    post_webhook(url, format_slack_payload(alerts, latest=latest))


def worst_severity(alerts: list[ShadowAlert]) -> AlertSeverity | None:
    if not alerts:
        return None
    order = {AlertSeverity.P0: 0, AlertSeverity.P1: 1, AlertSeverity.P2: 2}
    return min(alerts, key=lambda a: order[a.severity]).severity


def alerts_to_json(alerts: list[ShadowAlert]) -> str:
    return json.dumps([a.to_dict() for a in alerts], indent=2, default=str)


__all__ = [
    "AlertSeverity",
    "ShadowAlert",
    "alerts_to_json",
    "evaluate_shadow_alerts",
    "format_slack_payload",
    "send_alerts",
    "worst_severity",
]
