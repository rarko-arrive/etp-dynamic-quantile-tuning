"""Daily shadow evaluation and publish loop for SARIMA_tail vs shipped DQT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl
from loguru import logger

from dqt import get_dt_local, repo_root, resolve_data_dir
from dqt.model.global_dqt import (
    _DEFAULT_MAX_WEEKLY_MOVE,
    DialProposal,
    GlobalDQT,
)
from dqt.panel import build_panel
from dqt.sarima_dial import (
    MODEL_QCOL_SARIMA,
    score_sarima_walkforward,
    write_parquet_atomic,
)
from dqt.sarima_publish import (
    KillSwitchResult,
    evaluate_kill_switch,
    next_weekday,
    prior_weekday,
    publish_schedule_row,
)
from dqt.score.business import business_pinball_for_models
from dqt.tuning.assess import cap_global_alts

ProposalSource = Literal["SARIMA_tail", "shipped_alt_50"]
PublishMode = Literal["block", "cap", "full"]
_SHADOW_DIR = "shadow"
_HISTORY_NAME = "shadow_eval_history.parquet"
_SCHEDULE_NAME = "sarima_publish_schedule.parquet"
_TAIL_NAME = "sarima_tail_quantiles.parquet"


@dataclass(frozen=True, slots=True)
class ShadowReport:
    """One row per daily shadow run."""

    run_at: datetime
    scored_date: date
    target_date: date
    n_loads: int
    shipped_att50: float | None
    tail_att50: float | None
    tail_ece_pp: float | None
    tail_pinball_t_usd: float | None
    kill_switch_passed: bool
    kill_switch_reason: str | None
    proposal_source: ProposalSource
    published_fqn: str | None
    gates_passed: bool
    gates_reasons: list[str] = field(default_factory=list)
    sarima_mae_pp: float | None = None
    naive_mae_pp: float | None = None
    lift_vs_naive_pct: float | None = None
    full_proposal_alt_50: float | None = None
    capped_proposal_alt_50: float | None = None
    alt50_move_pp: float | None = None
    dual_audit_path: str | None = None

    def to_frame(self) -> pl.DataFrame:
        row = asdict(self)
        row["run_at"] = self.run_at.isoformat()
        row["scored_date"] = self.scored_date
        row["target_date"] = self.target_date
        row["gates_reasons"] = self.gates_reasons or None
        return pl.DataFrame([row])


class ShadowDQT:
    """Orchestrate score-yesterday + propose-tomorrow shadow loop."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        settlement_lag_days: int = 3,
        global_dqt: GlobalDQT | None = None,
    ) -> None:
        self.data_dir = resolve_data_dir(data_dir)
        self.settlement_lag_days = int(settlement_lag_days)
        self.dqt = global_dqt or GlobalDQT(self.data_dir)

    @property
    def shadow_dir(self) -> Path:
        return self.data_dir / _SHADOW_DIR

    @property
    def history_path(self) -> Path:
        return self.shadow_dir / _HISTORY_NAME

    @property
    def tail_path(self) -> Path:
        return self.data_dir / _TAIL_NAME

    @property
    def panel_path(self) -> Path:
        return self.data_dir / "panel_loads.parquet"

    def yesterday_booking_day(self, today: date | None = None) -> date:
        """Last complete weekday with settlement coverage (lag-3 proxy)."""
        today = today or get_dt_local().date()
        d = today - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        lag = self.settlement_lag_days
        if lag > 0:
            while d + timedelta(days=lag) > today:
                d -= timedelta(days=1)
                while d.weekday() >= 5:
                    d -= timedelta(days=1)
        return d

    def score_day(self, day: date | str) -> dict[str, Any]:
        """Load-level KPIs: shipped DQT/ETP vs ``SARIMA_tail`` on ``day``."""
        scored = _parse_day(day)
        panel = self._load_panel()
        day_panel = panel.filter(pl.col("booked_date") == scored)
        if day_panel.is_empty():
            bd = panel.get_column("booked_date")
            panel_min, panel_max = bd.min(), bd.max()
            if scored > panel_max:
                logger.warning(
                    "scored day {} is after panel max {} (panel {}..{}) — "
                    "run `make features FORCE=1` then `make hybrid FORCE=1`",
                    scored.isoformat(),
                    panel_max,
                    panel_min,
                    panel_max,
                )
            else:
                logger.warning(
                    "scored day {} has n_loads=0 (panel {}..{}) — "
                    "check cohort filters or rebuild with `make hybrid FORCE=1`",
                    scored.isoformat(),
                    panel_min,
                    panel_max,
                )
            return {
                "scored_date": scored,
                "n_loads": 0,
                "shipped_att50": None,
                "tail_att50": None,
                "tail_ece_pp": None,
                "tail_pinball_t_usd": None,
            }

        tail = self._load_tail()
        frame = day_panel.join(tail, on="loadnumber", how="inner")
        if frame.is_empty():
            raise FileNotFoundError(
                f"no sarima_tail quotes for {scored.isoformat()}; "
                f"run `make sarima-wf APPEND=1`"
            )

        models = {
            "DQT/ETP": MODEL_QCOL_SARIMA["DQT/ETP"],
            "SARIMA_tail": MODEL_QCOL_SARIMA["SARIMA_tail"],
        }
        cards = score_sarima_walkforward(frame, model_qcol=models)
        global_card = cards["global"]
        shipped_row = global_card.filter(pl.col("model") == "DQT/ETP")
        tail_row = global_card.filter(pl.col("model") == "SARIMA_tail")
        shipped_att50 = (
            float(shipped_row["att50"][0]) if not shipped_row.is_empty() else None
        )
        tail_att50 = float(tail_row["att50"][0]) if not tail_row.is_empty() else None
        tail_ece = float(tail_row["ece_pp"][0]) if not tail_row.is_empty() else None
        pinball = business_pinball_for_models(
            frame, {"SARIMA_tail": MODEL_QCOL_SARIMA["SARIMA_tail"]}
        ).get("SARIMA_tail")

        return {
            "scored_date": scored,
            "n_loads": frame.height,
            "shipped_att50": shipped_att50,
            "tail_att50": tail_att50,
            "tail_ece_pp": tail_ece,
            "tail_pinball_t_usd": pinball,
        }

    def evaluate_kill_switch(
        self,
        *,
        target: date | None = None,
        min_lift_vs_naive_pct: float = 0.0,
    ) -> KillSwitchResult:
        dial = pl.read_parquet(self.dqt.sarima_wf_path)
        target = target or next_weekday(get_dt_local().date())
        return evaluate_kill_switch(
            dial,
            target=target,
            min_lift_vs_naive_pct=min_lift_vs_naive_pct,
        )

    def propose_global(
        self,
        as_of: date | str,
        *,
        source: ProposalSource = "SARIMA_tail",
        min_n: int = 200,
        cal_window_days: int = 28,
        max_weekly_move: float = _DEFAULT_MAX_WEEKLY_MOVE,
        publish_mode: PublishMode = "block",
    ) -> tuple[DialProposal, DialProposal | None]:
        """Return ``(published_proposal, full_uncapped_or_none)``."""
        target = _parse_day(as_of)
        prior_day = prior_weekday(target)
        prior = self.dqt.as_of(prior_day)
        if source == "shipped_alt_50":
            prop = self.dqt.gates(
                DialProposal(
                    valid_date=target,
                    alts=dict(prior.alts),
                    method="shift",
                    generated_at=get_dt_local(),
                    gates={},
                ),
                prior=prior,
                max_weekly_move=max_weekly_move,
            )
            return prop, None
        if publish_mode == "cap":
            raw = self.dqt.propose(
                method="sarima_tail",
                as_of=target,
                prior=prior,
                min_n=min_n,
                cal_window_days=cal_window_days,
                max_weekly_move=1.0,
            )
            capped = cap_global_alts(prior.alts, raw.alts, max_weekly_move)
            prop = self.dqt.gates(
                replace(raw, alts=capped, gates={}),
                prior=prior,
                max_weekly_move=max_weekly_move,
            )
            return prop, raw
        gate_limit = 1.0 if publish_mode == "full" else max_weekly_move
        prop = self.dqt.propose(
            method="sarima_tail",
            as_of=target,
            prior=prior,
            min_n=min_n,
            cal_window_days=cal_window_days,
            max_weekly_move=gate_limit,
        )
        return prop, None

    def _write_full_proposal_audit(self, proposal: DialProposal) -> Path:
        out_dir = self.data_dir / "staging"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            out_dir
            / f"dial_proposal_{proposal.valid_date.isoformat()}_{proposal.method}_full.parquet"
        )
        row = {
            **proposal.to_dict(),
            "method": proposal.method,
            "generated_at": proposal.generated_at.isoformat(),
            "audit_kind": "full_uncapped",
        }
        write_parquet_atomic(pl.DataFrame([row]), path)
        logger.info("wrote dual-write full proposal audit → {}", path)
        return path

    def append_history(self, report: ShadowReport) -> Path:
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        new_row = report.to_frame()
        if self.history_path.exists():
            hist = pl.read_parquet(self.history_path)
            out = pl.concat([hist, new_row], how="diagonal_relaxed")
        else:
            out = new_row
        write_parquet_atomic(out, self.history_path)
        logger.info("appended shadow eval → {}", self.history_path)
        return self.history_path

    def publish_shadow(
        self,
        proposal: DialProposal,
        *,
        write_snowflake: bool = True,
        dry_run: bool = False,
        allow_prod: bool = False,
        kill_switch_passed: bool = True,
        gates_passed: bool = True,
        score_passed: bool = True,
    ) -> str | None:
        """Staging parquet always; Snowflake only when kill-switch + gates pass."""
        self.dqt._write_parquet_audit(proposal)

        if not score_passed:
            return None
        if not kill_switch_passed:
            return None
        if not gates_passed:
            return None
        if not write_snowflake:
            return None

        dest = self.dqt.publish(
            proposal,
            target="snowflake",
            dry_run=dry_run,
            allow_prod=allow_prod,
            also_parquet=False,
        )
        return dest.fqn

    def run_daily(
        self,
        *,
        as_of: date | str | None = None,
        scored_date: date | str | None = None,
        dry_run: bool = False,
        write_snowflake: bool = True,
        min_lift_vs_naive_pct: float = 0.0,
        allow_prod: bool = False,
        min_n: int = 200,
        cal_window_days: int = 28,
        max_weekly_move: float = _DEFAULT_MAX_WEEKLY_MOVE,
        publish_mode: PublishMode = "block",
        dual_write_audit: bool = False,
        append_history: bool = True,
    ) -> ShadowReport:
        """Score yesterday, propose tomorrow, audit + optional SF append."""
        run_at = get_dt_local()
        today = run_at.date()
        target = _parse_day(as_of) if as_of is not None else next_weekday(today)
        scored = (
            _parse_day(scored_date)
            if scored_date is not None
            else self.yesterday_booking_day(today)
        )

        score = self.score_day(scored)
        score_passed = int(score["n_loads"]) > 0
        kill = self.evaluate_kill_switch(
            target=target,
            min_lift_vs_naive_pct=min_lift_vs_naive_pct,
        )
        source: ProposalSource = (
            "SARIMA_tail" if kill.passed else "shipped_alt_50"
        )
        proposal, full_proposal = self.propose_global(
            target,
            source=source,
            min_n=min_n,
            cal_window_days=cal_window_days,
            max_weekly_move=max_weekly_move,
            publish_mode=publish_mode,
        )
        gates_passed = bool(proposal.passed)
        gates_reasons = list(proposal.gates.get("reasons") or [])

        full_alt_50: float | None = None
        capped_alt_50: float | None = proposal.alt_50
        alt50_move_pp: float | None = None
        dual_audit_path: str | None = None
        if full_proposal is not None:
            full_alt_50 = full_proposal.alt_50
            prior = self.dqt.as_of(prior_weekday(target))
            if full_alt_50 is not None and prior.alt_50 is not None:
                alt50_move_pp = abs(float(full_alt_50) - float(prior.alt_50)) * 100.0
            if dual_write_audit:
                audit_path = self._write_full_proposal_audit(full_proposal)
                dual_audit_path = str(audit_path)

        report = ShadowReport(
            run_at=run_at,
            scored_date=scored,
            target_date=target,
            n_loads=int(score["n_loads"]),
            shipped_att50=score["shipped_att50"],
            tail_att50=score["tail_att50"],
            tail_ece_pp=score["tail_ece_pp"],
            tail_pinball_t_usd=score["tail_pinball_t_usd"],
            kill_switch_passed=kill.passed,
            kill_switch_reason=kill.reason,
            proposal_source=source,
            published_fqn=None,
            gates_passed=gates_passed,
            gates_reasons=gates_reasons,
            sarima_mae_pp=kill.sarima_mae_pp,
            naive_mae_pp=kill.naive_mae_pp,
            lift_vs_naive_pct=kill.lift_vs_naive_pct,
            full_proposal_alt_50=full_alt_50,
            capped_proposal_alt_50=capped_alt_50,
            alt50_move_pp=alt50_move_pp,
            dual_audit_path=dual_audit_path,
        )

        published_fqn = self.publish_shadow(
            proposal,
            write_snowflake=write_snowflake,
            dry_run=dry_run,
            allow_prod=allow_prod,
            kill_switch_passed=kill.passed,
            gates_passed=gates_passed,
            score_passed=score_passed,
        )
        if not score_passed:
            logger.warning(
                "scored day {} has no loads — skipping Snowflake "
                "(run `make hybrid FORCE=1` and `make sarima-wf APPEND=1`)",
                scored,
            )
        elif not kill.passed:
            logger.warning("kill-switch tripped — skipping Snowflake ({})", kill.reason)
        elif not gates_passed:
            logger.warning("gates failed — skipping Snowflake ({})", gates_reasons)

        if published_fqn is not None:
            report = replace(report, published_fqn=published_fqn)

        if append_history:
            self.append_history(report)

            schedule_path = self.shadow_dir / _SCHEDULE_NAME
            sched_row = publish_schedule_row(
                valid_date=target,
                alts=proposal.alts,
                kill=kill,
                proposal_source=source,
            )
            if schedule_path.exists():
                sched = pl.concat(
                    [pl.read_parquet(schedule_path), sched_row],
                    how="diagonal_relaxed",
                )
            else:
                sched = sched_row
            write_parquet_atomic(sched, schedule_path)

        return report

    def _load_panel(self) -> pl.DataFrame:
        feat = self.data_dir / "features.parquet"
        panel_path = self.panel_path
        stale = (
            feat.exists()
            and panel_path.exists()
            and feat.stat().st_mtime > panel_path.stat().st_mtime
        )
        if not panel_path.exists() or stale:
            if stale:
                logger.info(
                    "features.parquet newer than panel_loads — rebuilding panel"
                )
            return build_panel(repo_root(), force=True, data_dir=self.data_dir)
        return pl.read_parquet(panel_path)

    def _load_tail(self) -> pl.DataFrame:
        if not self.tail_path.exists():
            raise FileNotFoundError(
                f"sarima_tail artifact missing at {self.tail_path}; "
                "run `make sarima-wf APPEND=1`"
            )
        return pl.read_parquet(self.tail_path)


def _parse_day(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


__all__ = [
    "PublishMode",
    "ShadowDQT",
    "ShadowReport",
]
