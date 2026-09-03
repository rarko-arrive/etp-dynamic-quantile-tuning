"""Business-slider (t1-t4) pinball scoring — Weatherman vs DQT/ETP.

Migrated from the former `dqt.evals` verbatim: this scores each load's own
configured slider level (`t{k}_setting`), not the fixed quantile grid, so it
doesn't fit the `ModelSpec`/`compare()` per-level abstraction in `api.py` —
kept as its own small, intentionally-2-model helper (Weatherman needs an
exact `knn_XX` lookup by `t{k}_setting`; DQT/ETP already has the dollar quote
in `t{k}`).
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from .constants import COST_COL, QUANTILES

MODEL_QCOL = {"Weatherman": "knn_{q}", "DQT/ETP": "p{q:02d}"}
MODELS = tuple(MODEL_QCOL)
BUSINESS_T = (1, 2, 3, 4)


def _pinball(y: pl.Expr, z: pl.Expr, a: pl.Expr | float) -> pl.Expr:
    return pl.when(y >= z).then((y - z) * a).otherwise((z - y) * (1 - a))


def business_pinball_exprs() -> list[pl.Expr]:
    """Mean-pinball aggregation exprs at the per-load slider levels t1-t4.

    DQT is scored on the quoted t1..t4; Weatherman on its own quantile at the
    same level (exact knn column lookup — tk_setting is always a multiple of 5).
    """
    y = pl.col(COST_COL)
    exprs = []
    for model in MODELS:
        for k in BUSINESS_T:
            a = pl.col(f"t{k}_setting") / 100
            if model == "Weatherman":
                z = pl.coalesce(
                    [
                        pl.when(pl.col(f"t{k}_setting") == q).then(pl.col(f"knn_{q}"))
                        for q in QUANTILES
                    ]
                )
            else:
                z = pl.col(f"t{k}")
            exprs.append(_pinball(y, z, a).mean().alias(f"tpin_{model}_{k}"))
    return exprs


def business_pinball_by_model(frame: pl.DataFrame) -> dict[str, float | None]:
    """Avg pinball loss over the four business slider targets, per model ($/load).

    Returns None for a model when no load in the frame has slider settings.
    """
    return business_pinball_for_models(frame, MODEL_QCOL)


def _quote_at_setting(col_template: str, k: int) -> pl.Expr:
    """Dollar quote at load ``t{k}_setting`` via ``col_template.format(q=…)``."""
    q_int = pl.col(f"t{k}_setting")
    return pl.coalesce(
        [
            pl.when(q_int == q).then(pl.col(col_template.format(q=q)))
            for q in QUANTILES
        ]
    )


def business_pinball_for_models(
    frame: pl.DataFrame,
    model_qcol: Mapping[str, str],
) -> dict[str, float | None]:
    """Mean pinball ($/load) at t1–t4 for each model in ``model_qcol``."""
    slider_cols = [f"t{k}_setting" for k in BUSINESS_T]
    if not all(c in frame.columns for c in slider_cols):
        return {name: None for name in model_qcol}
    if COST_COL not in frame.columns:
        return {name: None for name in model_qcol}

    y = pl.col(COST_COL)
    out: dict[str, float | None] = {}
    for name, tmpl in model_qcol.items():
        sample_col = tmpl.format(q=50)
        if sample_col not in frame.columns:
            out[name] = None
            continue
        exprs = []
        for k in BUSINESS_T:
            a = pl.col(f"t{k}_setting") / 100
            z = _quote_at_setting(tmpl, k)
            exprs.append(_pinball(y, z, a).mean().alias(f"_tpin_{k}"))
        row = frame.select(exprs).row(0)
        vals = [v for v in row if v is not None]
        out[name] = sum(vals) / len(vals) if vals else None
    return out
