"""Kill-switch and next-weekday hat resolution for SARIMA dial shadow publish.

Fail closed when the calibrated SARIMA hat does not beat naive on the eval
window, when the target hat is missing, or when lift vs naive is below threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from dqt import get_dt_local
from dqt.sarima_dial import DIAL_PRED_COLS, dial_errors_pp

DEFAULT_PRED_COL = "y_hat_sarima_cal"
DEFAULT_MIN_LIFT_VS_NAIVE_PCT = 0.0


@dataclass(frozen=True, slots=True)
class KillSwitchResult:
    """Outcome of :func:`evaluate_kill_switch`."""

    passed: bool
    reason: str | None
    sarima_mae_pp: float | None
    naive_mae_pp: float | None
    lift_vs_naive_pct: float | None
    target_hat: float | None = None
    target_date: date | None = None


def next_weekday(day: date) -> date:
    """First weekday strictly after ``day`` (Chicago calendar)."""
    d = day + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def prior_weekday(day: date) -> date:
    """Last weekday strictly before ``day``."""
    d = day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def week_start(day: date) -> date:
    """ISO Monday on or before ``day``."""
    return day - timedelta(days=day.weekday())


def resolve_target_hat(
    dial: pl.DataFrame,
    target: date,
    *,
    pred_col: str = DEFAULT_PRED_COL,
) -> float:
    """One-step SARIMA hat for ``target`` from the walk-forward dial artifact."""
    if pred_col not in dial.columns:
        raise ValueError(f"resolve_target_hat: dial missing {pred_col!r}")
    date_col = "booked_date" if "booked_date" in dial.columns else "valid_date"
    wf = dial.sort(date_col)
    hit = wf.filter(pl.col(date_col) == target)
    if hit.is_empty():
        hit = (
            wf.filter(pl.col(date_col) <= target)
            .filter(pl.col(pred_col).is_not_null() & pl.col(pred_col).is_finite())
            .sort(date_col)
        )
        if hit.is_empty():
            raise LookupError(f"no {pred_col} on/before {target.isoformat()}")
        val = float(hit[pred_col][-1])
        if not np.isfinite(val):
            raise LookupError(f"{pred_col} non-finite for latest on/before {target}")
        return val
    val = hit[pred_col][0]
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        raise LookupError(f"{pred_col} null/non-finite for {target.isoformat()}")
    return float(val)


def latest_weekday_hat(
    dial: pl.DataFrame,
    *,
    pred_col: str = DEFAULT_PRED_COL,
    on_or_before: date | None = None,
) -> tuple[date, float]:
    """Most recent weekday row with a finite ``pred_col`` hat."""
    date_col = "booked_date" if "booked_date" in dial.columns else "valid_date"
    wf = dial.sort(date_col)
    if on_or_before is not None:
        wf = wf.filter(pl.col(date_col) <= on_or_before)
    wf = wf.filter(pl.col(pred_col).is_not_null() & pl.col(pred_col).is_finite())
    if wf.is_empty():
        raise LookupError(f"no finite {pred_col} in dial")
    row = wf.row(-1, named=True)
    return row[date_col], float(row[pred_col])


def evaluate_kill_switch(
    dial: pl.DataFrame,
    *,
    target: date | None = None,
    pred_col: str = DEFAULT_PRED_COL,
    eval_only: bool = True,
    min_lift_vs_naive_pct: float = DEFAULT_MIN_LIFT_VS_NAIVE_PCT,
) -> KillSwitchResult:
    """Fail closed when SARIMA dial MAE ≥ naive or target hat is unusable."""
    target = target or next_weekday(get_dt_local().date())
    naive_m = dial_errors_pp(dial, pred_col="y_hat_naive", eval_only=eval_only)
    sarima_m = dial_errors_pp(dial, pred_col=pred_col, eval_only=eval_only)

    naive_mae = naive_m["mae_pp"]
    sarima_mae = sarima_m["mae_pp"]
    lift = None
    if (
        np.isfinite(sarima_mae)
        and np.isfinite(naive_mae)
        and naive_mae > 0
    ):
        lift = 100.0 * (naive_mae - sarima_mae) / naive_mae

    try:
        target_hat = resolve_target_hat(dial, target, pred_col=pred_col)
    except LookupError as exc:
        return KillSwitchResult(
            passed=False,
            reason=str(exc),
            sarima_mae_pp=sarima_mae if np.isfinite(sarima_mae) else None,
            naive_mae_pp=naive_mae if np.isfinite(naive_mae) else None,
            lift_vs_naive_pct=lift,
            target_hat=None,
            target_date=target,
        )

    if not np.isfinite(sarima_mae) or not np.isfinite(naive_mae):
        return KillSwitchResult(
            passed=False,
            reason="eval-window MAE unavailable (empty eval slice)",
            sarima_mae_pp=sarima_mae if np.isfinite(sarima_mae) else None,
            naive_mae_pp=naive_mae if np.isfinite(naive_mae) else None,
            lift_vs_naive_pct=lift,
            target_hat=target_hat,
            target_date=target,
        )

    if sarima_mae >= naive_mae:
        return KillSwitchResult(
            passed=False,
            reason=(
                f"SARIMA MAE {sarima_mae:.4f}pp ≥ naive {naive_mae:.4f}pp"
            ),
            sarima_mae_pp=sarima_mae,
            naive_mae_pp=naive_mae,
            lift_vs_naive_pct=lift,
            target_hat=target_hat,
            target_date=target,
        )

    if lift is not None and lift < min_lift_vs_naive_pct:
        return KillSwitchResult(
            passed=False,
            reason=(
                f"lift_vs_naive {lift:.2f}% < min {min_lift_vs_naive_pct:.2f}%"
            ),
            sarima_mae_pp=sarima_mae,
            naive_mae_pp=naive_mae,
            lift_vs_naive_pct=lift,
            target_hat=target_hat,
            target_date=target,
        )

    return KillSwitchResult(
        passed=True,
        reason=None,
        sarima_mae_pp=sarima_mae,
        naive_mae_pp=naive_mae,
        lift_vs_naive_pct=lift,
        target_hat=target_hat,
        target_date=target,
    )


def publish_schedule_row(
    *,
    valid_date: date,
    alts: dict[str, float],
    kill: KillSwitchResult,
    proposal_source: str,
) -> pl.DataFrame:
    """One audit row for ``sarima_publish_schedule.parquet``."""
    row: dict[str, object] = {
        "valid_date": valid_date,
        "proposal_source": proposal_source,
        "kill_switch_passed": kill.passed,
        "kill_switch_reason": kill.reason,
        "sarima_mae_pp": kill.sarima_mae_pp,
        "naive_mae_pp": kill.naive_mae_pp,
        "lift_vs_naive_pct": kill.lift_vs_naive_pct,
        "target_hat": kill.target_hat,
        **alts,
    }
    return pl.DataFrame([row])


__all__ = [
    "DEFAULT_MIN_LIFT_VS_NAIVE_PCT",
    "DEFAULT_PRED_COL",
    "DIAL_PRED_COLS",
    "KillSwitchResult",
    "evaluate_kill_switch",
    "latest_weekday_hat",
    "next_weekday",
    "prior_weekday",
    "publish_schedule_row",
    "resolve_target_hat",
    "week_start",
]
