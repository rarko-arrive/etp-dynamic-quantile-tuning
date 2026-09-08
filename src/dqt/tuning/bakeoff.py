"""30-weekday shadow bake-off SLO assessment for SARIMA_tail."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from dqt import get_dt_local, resolve_data_dir
from dqt.model.global_dqt import GlobalDQT
from dqt.sarima_dial import write_parquet_atomic
from dqt.tuning.assess import cap_global_alts


@dataclass(frozen=True, slots=True)
class BakeoffSLO:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class BakeoffReport:
    assessed_at: date
    n_weekdays: int
    required_weekdays: int
    slos: tuple[BakeoffSLO, ...]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.slos) and self.n_weekdays >= self.required_weekdays

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "assessed_at": self.assessed_at,
                    "n_weekdays": self.n_weekdays,
                    "required_weekdays": self.required_weekdays,
                    "passed": self.passed,
                    **{f"slo_{s.name}": s.passed for s in self.slos},
                }
            ]
        )


def _weekday_rows(history: pl.DataFrame) -> pl.DataFrame:
    if history.is_empty():
        return history
    df = history
    if "scored_date" in df.columns:
        df = df.with_columns(pl.col("scored_date").cast(pl.Date))
        df = df.filter(pl.col("scored_date").dt.weekday() < 5)
    return df.sort("scored_date" if "scored_date" in df.columns else "run_at")


def _cap_vs_full_gap_from_history(history: pl.DataFrame) -> float | None:
    """Mean |full − capped| alt_50 (pp) from shadow audit columns."""
    if not {"full_proposal_alt_50", "capped_proposal_alt_50"}.issubset(history.columns):
        return None
    sub = history.filter(
        pl.col("full_proposal_alt_50").is_not_null()
        & pl.col("capped_proposal_alt_50").is_not_null()
    )
    if sub.is_empty():
        return None
    gaps = (
        (sub["full_proposal_alt_50"] - sub["capped_proposal_alt_50"]).abs() * 100.0
    )
    return float(gaps.mean())


def _cap_vs_full_gap_pp(
    data_dir: Path,
    history: pl.DataFrame,
    *,
    max_move: float = 0.05,
    sample_days: int = 30,
) -> float | None:
    """Mean |full_tail alt_50 − capped alt_50| over recent eval weekdays."""
    from_history = _cap_vs_full_gap_from_history(history)
    if from_history is not None:
        return from_history
    if history.is_empty() or "target_date" not in history.columns:
        return None
    dqt = GlobalDQT(data_dir)
    targets = (
        _weekday_rows(history)
        .select("target_date")
        .unique()
        .sort("target_date")
        .tail(sample_days)["target_date"]
        .to_list()
    )
    gaps: list[float] = []
    for target in targets:
        if target is None:
            continue
        td = target if isinstance(target, date) else date.fromisoformat(str(target)[:10])
        try:
            prior = dqt.as_of(td)
            raw = dqt.propose(
                method="sarima_tail",
                as_of=td,
                prior=prior,
                min_n=30,
                max_weekly_move=1.0,
            )
            capped = cap_global_alts(prior.alts, raw.alts, max_move)
            gaps.append(abs(float(raw.alts["alt_50"]) - float(capped["alt_50"])) * 100.0)
        except (LookupError, KeyError, ValueError):
            continue
    if not gaps:
        return None
    return float(sum(gaps) / len(gaps))


def assess_bakeoff_slos(
    history: pl.DataFrame,
    data_dir: Path | str | None = None,
    *,
    required_weekdays: int = 30,
    cap_gap_max_pp: float = 1.0,
    att50_lift_min_pp: float = 2.0,
    kill_switch_fail_max: int = 0,
) -> BakeoffReport:
    """Check Phase 1 bake-off SLOs on shadow_eval_history."""
    data_dir = resolve_data_dir(data_dir)
    weekdays = _weekday_rows(history)
    n = weekdays.height

    slos: list[BakeoffSLO] = []

    slos.append(
        BakeoffSLO(
            name="min_weekdays",
            passed=n >= required_weekdays,
            detail=f"{n} weekdays recorded (need {required_weekdays})",
            value=float(n),
            threshold=float(required_weekdays),
        )
    )

    if n > 0 and "kill_switch_passed" in weekdays.columns:
        fails = int((~weekdays["kill_switch_passed"].fill_null(False)).sum())
        slos.append(
            BakeoffSLO(
                name="kill_switch",
                passed=fails <= kill_switch_fail_max,
                detail=f"{fails} kill-switch failures",
                value=float(fails),
                threshold=float(kill_switch_fail_max),
            )
        )

    if n >= 10 and {"tail_att50", "shipped_att50"}.issubset(weekdays.columns):
        lift = (
            weekdays["tail_att50"].tail(10) - weekdays["shipped_att50"].tail(10)
        ).mean()
        lift_pp = None if lift is None else float(lift) * 100.0
        slos.append(
            BakeoffSLO(
                name="att50_lift_10d",
                passed=lift_pp is not None and lift_pp >= att50_lift_min_pp,
                detail=f"10d mean tail−shipped att50 lift {lift_pp}",
                value=lift_pp,
                threshold=att50_lift_min_pp,
            )
        )

    if n >= 5 and "tail_att50" in weekdays.columns:
        tail_mean = float(weekdays["tail_att50"].tail(5).mean())
        slos.append(
            BakeoffSLO(
                name="tail_att50_5d_mean",
                passed=0.49 <= tail_mean <= 0.51,
                detail=f"5d mean tail_att50 {tail_mean:.3f}",
                value=tail_mean,
                threshold=0.50,
            )
        )

    gap_pp = _cap_vs_full_gap_pp(data_dir, weekdays)
    if gap_pp is not None:
        slos.append(
            BakeoffSLO(
                name="cap_vs_full_gap",
                passed=gap_pp <= cap_gap_max_pp,
                detail=f"Mean |full−cap| alt_50 {gap_pp:.2f}pp",
                value=gap_pp,
                threshold=cap_gap_max_pp,
            )
        )

    return BakeoffReport(
        assessed_at=get_dt_local().date(),
        n_weekdays=n,
        required_weekdays=required_weekdays,
        slos=tuple(slos),
    )


def write_bakeoff_report(
    report: BakeoffReport,
    data_dir: Path | str,
    *,
    prefix: str = "bakeoff_assessment",
) -> tuple[Path, Path]:
    """Write markdown + parquet assessment under tuning/."""
    root = resolve_data_dir(data_dir)
    out_dir = root / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.assessed_at.isoformat()
    md_path = out_dir / f"{prefix}_{stamp}.md"
    pq_path = out_dir / f"{prefix}_{stamp}.parquet"

    lines = [
        f"# Shadow bake-off assessment ({stamp})",
        "",
        f"- Weekdays recorded: **{report.n_weekdays}** / {report.required_weekdays}",
        f"- Overall: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "| SLO | Pass | Detail |",
        "|-----|------|--------|",
    ]
    for slo in report.slos:
        lines.append(f"| {slo.name} | {slo.passed} | {slo.detail} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_parquet_atomic(report.to_frame(), pq_path)
    return md_path, pq_path


def load_shadow_history(data_dir: Path | str) -> pl.DataFrame:
    path = resolve_data_dir(data_dir) / "shadow" / "shadow_eval_history.parquet"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


__all__ = [
    "BakeoffReport",
    "BakeoffSLO",
    "assess_bakeoff_slos",
    "load_shadow_history",
    "write_bakeoff_report",
]
