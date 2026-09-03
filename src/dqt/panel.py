"""Build the load-level attainment panel joining the four DQT sources.

Core idea: for each load, reduce the Weatherman curve (knn_5..knn_95) and the
DQT-adjusted ETP curve (p05..p95) to a single number — the *implied percentile*
of the realized cost on that curve:

    r = smallest quantile level q such that realized_cost <= Q(q)

Then attainment at ANY percentile a is simply ``r <= a``, which makes
reliability curves, subgroup attainment, and counterfactual DQT policies
(evaluate attainment at a shifted percentile) cheap vectorized comparisons —
no re-interpolation per policy.
"""

from pathlib import Path

import numpy as np
import polars as pl

from . import parse_date_col, repo_root, resolve_data_dir
from .dims import MODE_TIER_MAP
from .score.constants import COST_COL, DATE_COL, ID_COL

GRID = np.arange(5, 100, 5) / 100.0  # 0.05 .. 0.95
KNN_COLS = [f"knn_{q}" for q in range(5, 100, 5)]
P_COLS = [f"p{q:02d}" for q in range(5, 100, 5)]
ALT_COLS = [f"alt_{q}" for q in range(5, 100, 5)]  # shipped dial; not zero-padded

# Naming Conveniences
LIGHTNING_COL = "lightning_prediction"
ETP_COLS = P_COLS.copy()
WEATHERMAN_COLS = KNN_COLS.copy()


EQUIPMENT_MAP = {
    "DRY": "Van",
    "REEFER": "Reefer",
    #  , "FLATBED": "Flatbed"
}

# Day-of-week labels, Monday-first. Stored as strings, not ints, because
# `conformal.walk_forward_alts` declares its group columns as `pl.String` — an
# integer weekday would break the conformal/hybrid path that consumes `dow` as
# a grouping dimension. `DOW_ORDER` is the display/sort order.
DOW_ORDER = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WEEKDAYS = DOW_ORDER[:5]


def implied_percentile(
    qmat: np.ndarray, cost: np.ndarray, levels: np.ndarray = GRID
) -> np.ndarray:
    """Implied percentile of `cost` on each row's quantile curve.

    Rows are sorted first (rearrangement fixes any quantile crossing).
    Piecewise-linear between grid points; below the lowest-level value the
    result scales into [0, levels[0]]; above the highest it is 1.0. Satisfies
    ``r <= levels[k]  <=>  cost <= sorted_curve[k]`` exactly at grid points.

    `levels` defaults to the standard 5..95-by-5 `GRID` but can be any
    ascending array of quantile levels matching `qmat`'s column count —
    e.g. a model scored on a different or sparser quantile grid.
    """
    Q = np.sort(qmat, axis=1)
    cost = cost.astype(np.float64)
    d = (Q < cost[:, None]).sum(axis=1)  # count strictly below cost
    n = Q.shape[1]
    r = np.empty(len(cost), dtype=np.float64)

    lo, hi = d == 0, d == n
    mid = ~lo & ~hi
    r[lo] = levels[0] * np.clip(cost[lo] / np.maximum(Q[lo, 0], 1e-9), 0.0, 1.0)
    r[hi] = 1.0
    dm = d[mid]
    q_lo, q_hi = Q[mid, dm - 1], Q[mid, dm]
    g_lo, g_hi = levels[dm - 1], levels[dm]
    r[mid] = g_lo + (g_hi - g_lo) * (cost[mid] - q_lo) / np.maximum(q_hi - q_lo, 1e-9)
    return r


HAUL_EDGES = (250, 600)  # default band edges used by the cached panel


def haul_band(col: pl.Expr, edges: tuple[int, int] = HAUL_EDGES) -> pl.Expr:
    """Label mileage into Short/Mid/Long bands at the given (short, long) edges.

    Single source of truth for haul banding — the panel builds with
    HAUL_EDGES; the Streamlit app derives bands on the fly with a
    user-selected long edge. Note the <250mi boundary is the load-bearing
    one (short-haul attainment barely co-moves with the rest, r~0.6-0.8
    weekly). The 600mi Mid/Long edge is the single-day driveable ceiling
    under FMCSA hours-of-service rules (11hr max driving/day at typical
    highway speed) — beyond it a load structurally requires a second day
    (relay, team, or overnight), a real operational discontinuity rather
    than an arbitrary cut. (600-1000mi correlates 0.93 with 250-600 and
    0.88 with 1000+, so 1000 was statistically defensible too, but 600 is
    the mechanistically grounded edge and is what the panel builds with.)
    """
    short, long_ = edges
    return (
        pl.when(col < short)
        .then(pl.lit(f"Short (<{short}mi)"))
        .when(col < long_)
        .then(pl.lit(f"Mid ({short}-{long_}mi)"))
        .otherwise(pl.lit(f"Long ({long_}mi+)"))
    )


def _panel_feature_cols(available: set[str]) -> list[str]:
    """Columns ``build_panel`` reads from features.parquet — one projection."""
    date_src = DATE_COL if DATE_COL in available else "booked_on_date"
    required = (
        ID_COL,
        date_src,
        COST_COL,
        "load_class_bucket",
        "load_class",
        "load_type",
        "loadmiles",
        "origin_zone_group",
        *KNN_COLS,
        *P_COLS,
        *ALT_COLS,
    )
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"features missing {missing[:8]}{'…' if len(missing) > 8 else ''} "
            "— rebuild with make features FORCE=1"
        )
    return list(required)


def build_panel(
    root: Path | None = None,
    force: bool = False,
    data_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Slim attainment panel from ``features.parquet`` (one column projection).

    Keeps the full Weatherman / ETP / shipped-dial grids so hybrid and SARIMA
    invert to dollars without a second features read. Cached to
    ``<data_dir>/panel_loads.parquet``.
    """
    root = root or repo_root()
    layer = resolve_data_dir(data_dir, root=root)
    out = layer / "panel_loads.parquet"
    if out.exists() and not force:
        return pl.read_parquet(out)

    feat = pl.scan_parquet(layer / "features.parquet")
    df = feat.select(_panel_feature_cols(set(feat.collect_schema().names()))).collect()
    if DATE_COL not in df.columns:
        df = df.with_columns(pl.col("booked_on_date").alias(DATE_COL))

    cost = df[COST_COL].to_numpy()
    wm_r = implied_percentile(df.select(KNN_COLS).to_numpy(), cost)
    etp_r = implied_percentile(df.select(P_COLS).to_numpy(), cost)

    panel = df.with_columns(
        pl.Series("wm_r", wm_r),
        pl.Series("etp_r", etp_r),
        parse_date_col(df, DATE_COL).alias(DATE_COL),
        pl.col("load_class_bucket").alias("mode"),
        pl.col("load_class")
        .replace_strict(MODE_TIER_MAP, default=None)
        .alias("mode_tier"),
        pl.col("load_type")
        .replace_strict(EQUIPMENT_MAP, default="Other")
        .alias("equipment"),
        haul_band(pl.col("loadmiles")).alias("haul_band"),
        pl.col("origin_zone_group").alias("geo"),
        pl.col(COST_COL).alias("cost"),
    ).with_columns(
        pl.col(DATE_COL).dt.truncate("1w").alias("week"),
        pl.col(DATE_COL).dt.strftime("%Y-%m").alias("month"),
        pl.col(DATE_COL).dt.strftime("%a").alias("dow"),
        pl.col(DATE_COL).alias("booked_date"),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out)
    return panel


def attainment_by(
    panel: pl.DataFrame,
    dims: list[str],
    quantiles: tuple[float, ...] = (0.5,),
    min_n: int = 1,
    time_col: str = "week",
) -> pl.DataFrame:
    """Attainment of Weatherman (raw) and ETP (DQT-adjusted) curves over time.

    `time_col` is the time key to bucket on — ``"week"`` (the default, as the
    cached panel builds it), ``"month"``, ``"date"`` for day-level
    series, or ``"dow"`` for a day-of-week profile. Pass ``time_col=""`` to
    aggregate over `dims` only. Returns one row per time bucket x dim with `n`
    and a `wm_att{q}` / `etp_att{q}` pair per requested quantile.
    """
    keys = ([time_col] if time_col else []) + list(dims)
    if not keys:
        raise ValueError("attainment_by needs a time_col or at least one dim")
    aggs = [pl.len().alias("n")]
    for q in quantiles:
        lab = f"{int(round(q * 100)):02d}"
        aggs += [
            (pl.col("wm_r") <= q).mean().alias(f"wm_att{lab}"),
            (pl.col("etp_r") <= q).mean().alias(f"etp_att{lab}"),
        ]
    out = panel.group_by(keys).agg(aggs).sort(keys)
    return out.filter(pl.col("n") >= min_n)
