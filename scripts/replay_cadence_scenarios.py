#!/usr/bin/env python3
"""Replay shadow DQT scenarios for cadence / holiday backtests (no Snowflake).

Compares counterfactual (scored_date, target_date) pairs — e.g. early pre-holiday
publish vs late post-holiday publish — using the same parquet layer.

Usage:
  DQT_DATA_DIR=data uv run python scripts/replay_cadence_scenarios.py
  DQT_DATA_DIR=data uv run python scripts/replay_cadence_scenarios.py --preset labor_day_2026

Requires: features.parquet, panel_loads.parquet, sarima artifacts on disk.
Run `make hybrid FORCE=1` and `make sarima-wf APPEND=1` first if needed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import get_dt_local, resolve_data_dir
from dqt.sarima_dial import write_parquet_atomic
from dqt.shadow_dqt import ShadowDQT

# Labor Day 2026 cadence matrix (scored_date, target_date).
LABOR_DAY_2026: tuple[tuple[str, str, str], ...] = (
    ("early_wed", "2026-08-29", "2026-09-04"),
    ("early_thu", "2026-09-02", "2026-09-05"),
    ("early_fri", "2026-09-03", "2026-09-08"),
    ("late_sep8", "2026-09-04", "2026-09-09"),
)


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    scored_date: date
    target_date: date


def _parse_scenarios(preset: str) -> tuple[Scenario, ...]:
    if preset != "labor_day_2026":
        raise ValueError(f"unknown preset {preset!r}")
    return tuple(
        Scenario(sid, date.fromisoformat(scored), date.fromisoformat(target))
        for sid, scored, target in LABOR_DAY_2026
    )


def _report_row(scenario_id: str, report) -> dict:
    row = report.to_frame().row(0, named=True)
    row["scenario_id"] = scenario_id
    row["signal_age_days"] = (report.target_date - report.scored_date).days
    shipped = report.shipped_att50
    tail = report.tail_att50
    row["tail_minus_shipped_pp"] = (
        None if shipped is None or tail is None else (float(tail) - float(shipped)) * 100.0
    )
    full = report.full_proposal_alt_50
    capped = report.capped_proposal_alt_50
    row["cap_vs_full_pp"] = (
        None if full is None or capped is None else abs(float(full) - float(capped)) * 100.0
    )
    return row


def _write_report(df: pl.DataFrame, out_dir: Path, *, stamp: date) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"cadence_backtest_{stamp.isoformat()}.md"
    pq_path = out_dir / f"cadence_backtest_{stamp.isoformat()}.parquet"

    lines = [
        f"# Cadence backtest ({stamp.isoformat()})",
        "",
        "Counterfactual shadow replays (`SKIP_SF`, no shadow history append).",
        "",
        (
            "| scenario | scored | target | signal_age_d | n_loads | shipped_att50 | "
            "tail_att50 | tail−ship pp | full alt_50 | capped alt_50 | cap−full pp | "
            "alt50_move pp |"
        ),
        (
            "|----------|--------|--------|--------------|---------|---------------|"
            "-----------|--------------|-------------|---------------|-------------|"
            "--------------|"
        ),
    ]
    for row in df.iter_rows(named=True):
        if row.get("error"):
            lines.append(
                "| {scenario_id} | {scored_date} | {target_date} | "
                "{signal_age_days} | — | — | — | — | — | — | — | — | "
                "*{error}* |".format(
                    scenario_id=row["scenario_id"],
                    scored_date=row["scored_date"],
                    target_date=row["target_date"],
                    signal_age_days=row.get("signal_age_days", ""),
                    error=row["error"],
                )
            )
            continue
        lines.append(
            "| {scenario_id} | {scored_date} | {target_date} | {signal_age_days} | "
            "{n_loads} | {shipped_att50} | {tail_att50} | {tail_minus_shipped_pp} | "
            "{full_proposal_alt_50} | {capped_proposal_alt_50} | {cap_vs_full_pp} | "
            "{alt50_move_pp} |".format(
                **{k: "" if row.get(k) is None else row[k] for k in (
                    "scenario_id", "scored_date", "target_date", "signal_age_days",
                    "n_loads", "shipped_att50", "tail_att50", "tail_minus_shipped_pp",
                    "full_proposal_alt_50", "capped_proposal_alt_50", "cap_vs_full_pp",
                    "alt50_move_pp",
                )}
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_parquet_atomic(df, pq_path)
    return md_path, pq_path


def main() -> int:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="parquet layer (DQT_DATA_DIR)")
    parser.add_argument(
        "--preset",
        default="labor_day_2026",
        help="built-in scenario set (default: labor_day_2026)",
    )
    parser.add_argument(
        "--max-weekly-move",
        type=float,
        default=0.05,
        help="cap move in fraction (default 0.05 = 5pp)",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=200,
        help="conformal min_n passed to propose_global",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    for name in ("features.parquet", "sarima_dial_walkforward.parquet", "sarima_tail_quantiles.parquet"):
        path = data_dir / name
        if not path.exists():
            logger.error("missing {} — run make features/hybrid/sarima-wf first", path)
            return 1

    scenarios = _parse_scenarios(args.preset)
    shadow = ShadowDQT(data_dir)
    rows: list[dict] = []

    for sc in scenarios:
        logger.info(
            "replay {} scored={} target={}",
            sc.scenario_id,
            sc.scored_date,
            sc.target_date,
        )
        try:
            report = shadow.run_daily(
                as_of=sc.target_date,
                scored_date=sc.scored_date,
                write_snowflake=False,
                min_n=args.min_n,
                publish_mode="cap",
                max_weekly_move=args.max_weekly_move,
                dual_write_audit=False,
                append_history=False,
            )
            rows.append(_report_row(sc.scenario_id, report))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("scenario {} failed: {}", sc.scenario_id, exc)
            rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "scored_date": sc.scored_date,
                    "target_date": sc.target_date,
                    "signal_age_days": (sc.target_date - sc.scored_date).days,
                    "error": str(exc),
                }
            )

    if not rows:
        logger.error("no scenarios ran")
        return 1

    df = pl.DataFrame(rows)
    stamp = get_dt_local().date()
    md_path, pq_path = _write_report(df, data_dir / "tuning", stamp=stamp)
    logger.info("cadence backtest → {} {}", md_path, pq_path)
    print(df)
    n_ok = df.filter(pl.col("error").is_null()).height if "error" in df.columns else len(rows)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
