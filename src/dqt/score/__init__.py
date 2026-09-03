"""Generalized quantile scoring: attainment, bias, MAE, pinball/CRPS/ECE,
for any number of quantile-producing models — array-driven or frame-driven.

    from dqt.score import ModelSpec, compare, summarize, summary_metrics

    specs = {
        "Weatherman": ModelSpec("Weatherman", NOMINAL, col_template="knn_{q}"),
        "DQT/ETP":    ModelSpec("DQT/ETP", NOMINAL, col_template="p{q:02d}"),
        # optional third model when columns exist:
        # "Hybrid": ModelSpec("Hybrid", NOMINAL, col_template="hybrid_{q:02d}"),
    }
    long = compare(specs, frame=panel, actual_col=COST_COL, group_cols=["haul_band"])
    summary = summarize(long)
"""

from .api import ModelSpec, compare, score_array, score_frame, summarize
from .business import (
    BUSINESS_T,
    MODEL_QCOL,
    MODELS,
    business_pinball_by_model,
    business_pinball_exprs,
)
from .constants import (
    COST_COL,
    DATE_COL,
    ID_COL,
    NOMINAL,
    QUANTILES,
    TIME_COL,
    Cols,
)
from .summary import model_specs, smooth_att50, summary_metrics, weekly_att50

__all__ = [
    "BUSINESS_T",
    "COST_COL",
    "DATE_COL",
    "ID_COL",
    "MODELS",
    "MODEL_QCOL",
    "NOMINAL",
    "QUANTILES",
    "TIME_COL",
    "Cols",
    "ModelSpec",
    "business_pinball_by_model",
    "business_pinball_exprs",
    "business_pinball_for_models",
    "compare",
    "model_specs",
    "score_array",
    "score_frame",
    "smooth_att50",
    "summarize",
    "summary_metrics",
    "weekly_att50",
]
