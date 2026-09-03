"""Empirical guardrail fire-rates and global dial cap helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from dqt import get_dt_local, resolve_data_dir
from dqt.model.global_dqt import GlobalDQT
from dqt.panel import ALT_COLS
from dqt.sarima_dial import (
    dial_scorecard,
    global_alts_from_tail,
    write_parquet_atomic,
)
from dqt.sarima_publish import evaluate_kill_switch, prior_weekday


def cap_global_alts(
    prior: dict[str, float],
    proposal: dict[str, float],
    max_move: float,
) -> dict[str, float]:
    """Clip per-level moves: ``prior + clip(proposal - prior, ±max_move)``."""
    out: dict[str, float] = {}
    for col in ALT_COLS:
        if col not in prior or col not in proposal:
            continue
        delta = float(proposal[col]) - float(prior[col])
        out[col] = float(prior[col]) + float(np.clip(delta, -max_move, max_move))
    return out


def shipped_weekday_moves(history: pl.DataFrame) -> pl.DataFrame:
    """Weekday-to-weekday max-level and alt_50 moves (pp) on shipped dial."""
    hist = history.sort("valid_date")
    alt_cols = [c for c in ALT_COLS if c in hist.columns]
    rows: list[dict[str, Any]] = []
    for i in range(1, hist.height):
        prev = hist.row(i - 1, named=True)
        cur = hist.row(i, named=True)
        pd_, cd = prev["valid_date"], cur["valid_date"]
        if pd_.weekday() >= 5 or cd.weekday() >= 5:
            continue
        max_move = max(abs(float(cur[c]) - float(prev[c])) for c in alt_cols)
        p50_move = abs(float(cur["alt_50"]) - float(prev["alt_50"]))
        rows.append(
            {
                "valid_date": cd,
                "max_move_pp": max_move * 100.0,
                "alt50_move_pp": p50_move * 100.0,
            }
        )
    return pl.DataFrame(rows)


def fire_rates(moves_pp: pl.Series, thresholds_pp: tuple[float, ...]) -> pl.DataFrame:
    """Share of rows exceeding each threshold."""
    n = moves_pp.len()
    if n == 0:
        return pl.DataFrame({"threshold_pp": list(thresholds_pp), "pct": [None] * len(thresholds_pp)})
    arr = moves_pp.to_numpy()
    return pl.DataFrame(
        {
            "threshold_pp": list(thresholds_pp),
            "n_exceed": [int((arr > t).sum()) for t in thresholds_pp],
            "pct": [float((arr > t).mean() * 100.0) for t in thresholds_pp],
            "n_total": n,
        }
    )


def simulate_tail_proposal_moves(
    data_dir: Path | str,
    *,
    n_days: int | None = None,
    quick: bool = False,
    min_n: int = 200,
    cal_window_days: int = 28,
    global_dqt: GlobalDQT | None = None,
) -> pl.DataFrame:
    """Per-target max-level move (pp) for ``global_alts_from_tail`` vs prior shipped."""
    layer = resolve_data_dir(data_dir)
    dqt = global_dqt or GlobalDQT(layer)
    dial_path = layer / "sarima_dial_walkforward.parquet"
    panel_path = layer / "panel_loads.parquet"
    if not dial_path.exists() or not panel_path.exists():
        raise FileNotFoundError(
            f"need {dial_path.name} and panel_loads.parquet under {layer}"
        )

    dial = pl.read_parquet(dial_path)
    panel = pl.read_parquet(panel_path)
    wf = dial.filter(pl.col("y_hat_sarima_cal").is_not_null()).sort("booked_date")
    if n_days is not None:
        wf = wf.tail(n_days)
    elif quick:
        wf = wf.tail(60)

    rows: list[dict[str, Any]] = []
    for row in wf.iter_rows(named=True):
        target = row["booked_date"]
        if target.weekday() >= 5:
            continue
        try:
            prior = dqt.as_of(prior_weekday(target))
            alts = global_alts_from_tail(
                panel,
                dial,
                prior.alts,
                target,
                min_n=min_n,
                cal_window_days=cal_window_days,
            )
            max_move = max(abs(alts[c] - prior.alts[c]) for c in ALT_COLS if c in alts)
            p50_move = abs(alts["alt_50"] - prior.alts["alt_50"])
            rows.append(
                {
                    "target_date": target,
                    "max_move_pp": max_move * 100.0,
                    "alt50_move_pp": p50_move * 100.0,
                    "r_hat": float(row["y_hat_sarima_cal"]),
                    "prior_alt_50": prior.alt_50,
                    "prop_alt_50": alts["alt_50"],
                }
            )
        except (LookupError, ValueError, KeyError):
            continue
    return pl.DataFrame(rows)


def blocked_day_counterfactual(
    tail_moves: pl.DataFrame,
    *,
    block_pp: float = 5.0,
    cap_pp: float = 5.0,
) -> pl.DataFrame:
    """Global dial shift on days that would fail ``block_pp`` at full proposal."""
    if tail_moves.is_empty():
        return pl.DataFrame()
    df = tail_moves.with_columns(
        ((pl.col("prop_alt_50") - pl.col("prior_alt_50")) * 100.0).alias("full_shift_pp")
    ).with_columns(
        pl.col("full_shift_pp")
        .clip(-cap_pp, cap_pp)
        .alias("capped_shift_pp")
    )
    blocked = df.filter(pl.col("max_move_pp") > block_pp)
    if blocked.is_empty():
        return pl.DataFrame(
            {
                "block_pp": [block_pp],
                "cap_pp": [cap_pp],
                "n_blocked": [0],
                "mean_full_shift_pp": [None],
                "mean_capped_shift_pp": [None],
                "mean_status_quo_shift_pp": [0.0],
            }
        )
    return pl.DataFrame(
        {
            "block_pp": [block_pp],
            "cap_pp": [cap_pp],
            "n_blocked": [blocked.height],
            "mean_full_shift_pp": [float(blocked["full_shift_pp"].mean())],
            "mean_capped_shift_pp": [float(blocked["capped_shift_pp"].mean())],
            "mean_status_quo_shift_pp": [0.0],
        }
    )


def assess_guardrails(
    data_dir: Path | str | None = None,
    *,
    quick: bool = False,
    thresholds_pp: tuple[float, ...] = (3.0, 5.0, 8.0, 10.0, 12.0),
) -> dict[str, Any]:
    """Full guardrail assessment: shipped moves, tail proposals, kill-switch."""
    layer = resolve_data_dir(data_dir)
    dqt = GlobalDQT(layer)
    hist_path = layer / "dqt_alt_percentiles.parquet"
    dial_path = layer / "sarima_dial_walkforward.parquet"

    if hist_path.exists():
        shipped_hist = pl.read_parquet(hist_path)
    else:
        shipped_hist = dqt.history(refresh=False, use_cache=True)

    shipped_moves = shipped_weekday_moves(shipped_hist)
    shipped_fire = fire_rates(shipped_moves["max_move_pp"], thresholds_pp)

    tail_moves = pl.DataFrame()
    tail_fire = pl.DataFrame()
    counterfactual = pl.DataFrame()
    if dial_path.exists():
        tail_moves = simulate_tail_proposal_moves(
            layer, quick=quick, global_dqt=dqt
        )
        if not tail_moves.is_empty():
            tail_fire = fire_rates(tail_moves["max_move_pp"], thresholds_pp)
            counterfactual = blocked_day_counterfactual(tail_moves)

    kill_pass = kill_fail = 0
    g1_pass = g1_fail = 0
    if dial_path.exists():
        dial = pl.read_parquet(dial_path)
        card = dial_scorecard(dial, eval_only=True)
        cal = card.filter(pl.col("model") == "sarima_cal")
        naive = card.filter(pl.col("model") == "naive_n1")
        if cal.height and naive.height:
            mae_cal = cal["mae_pp"][0]
            mae_naive = naive["mae_pp"][0]
            if mae_cal is not None and mae_naive is not None and mae_cal < mae_naive:
                g1_pass = 1
            else:
                g1_fail = 1

        for row in tail_moves.iter_rows(named=True):
            target = row["target_date"]
            kill = evaluate_kill_switch(dial, target=target)
            if kill.passed:
                kill_pass += 1
            else:
                kill_fail += 1

    return {
        "assessed_at": get_dt_local().isoformat(),
        "data_dir": str(layer),
        "shipped_moves": shipped_moves,
        "shipped_fire": shipped_fire,
        "tail_moves": tail_moves,
        "tail_fire": tail_fire,
        "counterfactual": counterfactual,
        "g1_pass": g1_pass,
        "g1_fail": g1_fail,
        "kill_pass_days": kill_pass,
        "kill_fail_days": kill_fail,
        "thresholds_pp": thresholds_pp,
    }


def write_assessment(
    result: dict[str, Any],
    *,
    data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write parquet bundle + markdown summary under ``$DQT_DATA_DIR/tuning/``."""
    layer = resolve_data_dir(data_dir)
    out_dir = layer / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = get_dt_local().date().isoformat()
    pq_path = out_dir / f"guardrail_assessment_{day}.parquet"
    md_path = out_dir / f"guardrail_assessment_{day}.md"

    summary_rows = []
    for label, fire_df in (
        ("shipped", result.get("shipped_fire")),
        ("sarima_tail", result.get("tail_fire")),
    ):
        if fire_df is None or fire_df.is_empty():
            continue
        for row in fire_df.iter_rows(named=True):
            summary_rows.append({"series": label, **row})

    bundle = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    if not result["shipped_moves"].is_empty():
        sm = result["shipped_moves"].with_columns(pl.lit("shipped").alias("series"))
        bundle = pl.concat([bundle, sm], how="diagonal_relaxed") if not bundle.is_empty() else sm
    write_parquet_atomic(bundle, pq_path)

    md = format_assessment_markdown(result)
    md_path.write_text(md, encoding="utf-8")
    return pq_path, md_path


def format_assessment_markdown(result: dict[str, Any]) -> str:
    """Executive markdown summary for guardrail assessment."""
    lines = [
        "# DQT Guardrail Assessment",
        "",
        f"- **Assessed:** {result['assessed_at']}",
        f"- **Data dir:** `{result['data_dir']}`",
        "",
        "## Shipped dial weekday moves",
        "",
    ]
    sm = result["shipped_moves"]
    if sm.is_empty():
        lines.append("_No shipped history._")
    else:
        lines.extend(
            [
                f"- n pairs: {sm.height}",
                f"- alt_50 median: {sm['alt50_move_pp'].median():.2f}pp",
                f"- max-level p90: {sm['max_move_pp'].quantile(0.9):.2f}pp",
                "",
                "| threshold | exceed % |",
                "|-----------|----------|",
            ]
        )
        for row in result["shipped_fire"].iter_rows(named=True):
            lines.append(f"| >{row['threshold_pp']:.0f}pp | {row['pct']:.1f}% |")

    lines.extend(["", "## SARIMA_tail proposal moves", ""])
    tm = result["tail_moves"]
    if tm.is_empty():
        lines.append("_No tail simulation (missing dial/panel)._")
    else:
        lines.extend(
            [
                f"- n weekdays: {tm.height}",
                f"- alt_50 median: {tm['alt50_move_pp'].median():.2f}pp",
                f"- max-level p90: {tm['max_move_pp'].quantile(0.9):.2f}pp",
                "",
                "| threshold | exceed % |",
                "|-----------|----------|",
            ]
        )
        for row in result["tail_fire"].iter_rows(named=True):
            lines.append(f"| >{row['threshold_pp']:.0f}pp | {row['pct']:.1f}% |")

    cf = result.get("counterfactual")
    if cf is not None and not cf.is_empty() and cf["n_blocked"][0]:
        lines.extend(
            [
                "",
                "## Blocked-day counterfactual (global alt_50 shift)",
                "",
                f"- Days blocked at {cf['block_pp'][0]:.0f}pp: **{cf['n_blocked'][0]}**",
                f"- Mean full proposal shift: **{cf['mean_full_shift_pp'][0]:.2f}pp**",
                f"- Mean capped ({cf['cap_pp'][0]:.0f}pp/day) shift: **{cf['mean_capped_shift_pp'][0]:.2f}pp**",
                "- Status quo (shipped hold): **0pp**",
            ]
        )

    lines.extend(
        [
            "",
            "## Kill-switch alignment",
            "",
            f"- G1 (sarima-wf MAE): pass={result['g1_pass']} fail={result['g1_fail']}",
            f"- G2 per simulated day: pass={result['kill_pass_days']} fail={result['kill_fail_days']}",
            "",
            "## Recommended thresholds (attainment-first)",
            "",
            "| Environment | max_weekly_move | publish_mode |",
            "|-------------|-----------------|--------------|",
            "| Prod SF | 5pp | block |",
            "| Shadow RARKO | 10pp | cap |",
            "| Research audit | none | full (log only) |",
        ]
    )
    return "\n".join(lines) + "\n"
