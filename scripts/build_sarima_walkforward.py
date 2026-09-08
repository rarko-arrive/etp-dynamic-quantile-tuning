"""Build walk-forward SARIMA dial + sarima_pp quantiles + scorecard.

Usage (from the repo root):

    uv run python scripts/build_sarima_walkforward.py --dial-only
    uv run python scripts/build_sarima_walkforward.py --append
    make sarima-wf FORCE=1
    make sarima-wf APPEND=1          # weekly: reuse prior hats, rematerialize $

Honors ``DQT_DATA_DIR``. 
Scorecard needs ``features.parquet`` + ``panel_loads.parquet`` + ``hybrid_quantiles.parquet``.

Weekend policy (A): ``sarima_pp`` only on weekdays present in the dial schedule.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import polars as pl
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dqt import display_path, repo_root, resolve_data_dir
from dqt.hybrid import HybridConfig
from dqt.panel import EQUIPMENT_MAP, GRID, build_panel
from dqt.sarima_dial import (
    MARKET_EPISODES,
    MODEL_QCOL_SARIMA,
    daily_wm_r_median,
    daily_wm_r_median_settlement_lag,
    dial_scorecard,
    materialize_sarima_blend,
    materialize_sarima_hybrid,
    materialize_sarima_pp,
    materialize_sarima_tail,
    rolling_origin_dial_scorecard,
    score_market_episodes,
    score_sarima_walkforward,
    settlement_coverage_daily,
    walk_forward_sarima_dial,
    write_parquet_atomic,
)
from dqt.score.constants import COST_COL, DATE_COL, ID_COL

REPO = Path(__file__).resolve().parents[1]


def _seed_dial_from_frozen(root: Path, dial_path: Path) -> pl.DataFrame | None:
    """Copy frozen ``data/sarima_dial_walkforward.parquet`` if present."""
    frozen = root / "data" / "sarima_dial_walkforward.parquet"
    if not frozen.exists() or dial_path.resolve() == frozen.resolve():
        return None
    dial_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen, dial_path)
    logger.info(
        "seeded {} from frozen {}",
        display_path(dial_path, root=root),
        display_path(frozen, root=root),
    )
    return pl.read_parquet(dial_path)


def _episode_date_bounds() -> tuple:
    from datetime import date

    starts = [date.fromisoformat(ep[3]) for ep in MARKET_EPISODES]
    ends = [date.fromisoformat(ep[4]) for ep in MARKET_EPISODES]
    return min(starts), max(ends)


def _build_features_slice(
    root: Path,
    data_path: Path,
    *,
    date_min,
    date_max,
    hybrid_path: Path,
    pp_path: Path,
    hybrid_stack_path: Path,
    blend_path: Path | None = None,
    tail_path: Path | None = None,
) -> pl.DataFrame:
    """Features + quantile packs for a booked_date window (episodes / extras)."""
    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    p_cols = [f"p{int(round(a * 100)):02d}" for a in GRID]
    feat_cols = [
        ID_COL,
        DATE_COL,
        "booked_on_date",
        "load_type",
        "load_class_bucket",
        COST_COL,
        *knn_cols,
        *p_cols,
        "t1_setting",
        "t2_setting",
        "t3_setting",
        "t4_setting",
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    scan = pl.scan_parquet(data_path / "features.parquet")
    available = set(scan.collect_schema().names())
    extra = [c for c in ("loadmiles", "haul_band", "mode", "equipment") if c in available]
    frame = (
        scan.select([c for c in feat_cols + extra if c in available])
        .collect()
        .filter(pl.col("load_type").is_in(("DRY", "REEFER")))
        .with_columns(
            (
                pl.col(DATE_COL)
                if DATE_COL in available
                else pl.col("booked_on_date")
            )
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("booked_date"),
            pl.col("load_class_bucket").alias("mode"),
            pl.col("load_type")
            .replace_strict(EQUIPMENT_MAP, default="Other")
            .alias("equipment"),
        )
        .drop_nulls("booked_date")
        .filter(
            (pl.col("booked_date") >= date_min) & (pl.col("booked_date") <= date_max)
        )
    )
    if "haul_band" not in frame.columns:
        from dqt.panel import haul_band

        frame = frame.with_columns(haul_band(pl.col("loadmiles")).alias("haul_band"))
    hybrid = pl.read_parquet(hybrid_path)
    hy_cols = [c for c in hybrid.columns if c.startswith("hybrid_")]
    frame = frame.join(
        hybrid.select(["loadnumber", *hy_cols]), on="loadnumber", how="inner"
    )
    pp = pl.read_parquet(pp_path)
    pp_cols = [c for c in pp.columns if c.startswith("sarima_pp_")]
    frame = frame.join(
        pp.select(["loadnumber", *pp_cols]), on="loadnumber", how="left"
    )
    if hybrid_stack_path.exists():
        sh = pl.read_parquet(hybrid_stack_path)
        sh_cols = [c for c in sh.columns if c.startswith("sarima_hybrid_")]
        frame = frame.join(
            sh.select(["loadnumber", *sh_cols]), on="loadnumber", how="left"
        )
    if blend_path is not None and blend_path.exists():
        bl = pl.read_parquet(blend_path)
        bl_cols = [c for c in bl.columns if c.startswith("sarima_blend_")]
        frame = frame.join(
            bl.select(["loadnumber", *bl_cols]), on="loadnumber", how="left"
        )
    if tail_path is not None and tail_path.exists():
        tl = pl.read_parquet(tail_path)
        tl_cols = [c for c in tl.columns if c.startswith("sarima_tail_")]
        frame = frame.join(
            tl.select(["loadnumber", *tl_cols]), on="loadnumber", how="left"
        )
    return frame


def _forecast_dates(dial: pl.DataFrame, *, eval_only: bool) -> list:
    df = dial.filter(
        pl.col("y_hat_sarima_cal").is_not_null() & pl.col("y_hat_sarima_cal").is_not_nan()
    )
    if eval_only and "in_eval" in df.columns:
        df = df.filter(pl.col("in_eval"))
    return df.get_column("booked_date").to_list()


def main() -> int:
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="parquet layer (default: DQT_DATA_DIR or data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild dial/quantiles from scratch (ignore prior hats)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="reuse prior dial hats for known dates; forecast only new weekdays; "
        "always rematerialize sarima_pp",
    )
    parser.add_argument(
        "--seed-from-frozen",
        action="store_true",
        help="if dial missing under data-dir, copy from data/sarima_dial_walkforward.parquet",
    )
    parser.add_argument(
        "--dial-only",
        action="store_true",
        help="stop after sarima_dial_walkforward.parquet",
    )
    parser.add_argument(
        "--skip-scorecard",
        action="store_true",
        help="skip features/hybrid join scorecard",
    )
    parser.add_argument(
        "--eval-only-pp",
        action="store_true",
        help="materialize sarima_pp only on in_eval holdout dates (science repro); "
        "default materializes all dates with a non-null forecast (report/ops)",
    )
    parser.add_argument("--min-history", type=int, default=120)
    parser.add_argument(
        "--allow-mae-fail",
        action="store_true",
        help="continue to dollar scoring even if dial MAE gate fails",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="log every N *new* forecast steps (0 = quiet)",
    )
    parser.add_argument(
        "--settlement-lag",
        type=int,
        default=3,
        metavar="DAYS",
        help="build as-of dial using T+DAYS settlement lag proxy (0 = skip)",
    )
    parser.add_argument(
        "--skip-asof",
        action="store_true",
        help="skip settlement-lag as-of dial rebuild",
    )
    parser.add_argument(
        "--skip-hybrid-stack",
        action="store_true",
        help="skip sarima_hybrid quantile materialization",
    )
    parser.add_argument(
        "--skip-blend",
        action="store_true",
        help="skip sarima_blend (option C: shipped tails + SARIMA center)",
    )
    parser.add_argument(
        "--skip-tail",
        action="store_true",
        help="skip sarima_tail (option D: blend base + global tail conformal)",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    root = repo_root()
    data_path = resolve_data_dir(data_dir, root=root)
    dial_path = data_path / "sarima_dial_walkforward.parquet"
    pp_path = data_path / "sarima_pp_quantiles.parquet"
    results_dir = root / "data" / "results"
    scorecard_path = results_dir / "sarima_walkforward_scorecard.parquet"
    level_card_path = results_dir / "sarima_walkforward_quantile_level.parquet"
    episode_path = results_dir / "sarima_walkforward_episodes.parquet"
    rolling_path = results_dir / "sarima_dial_rolling_origin.parquet"
    coverage_path = results_dir / "sarima_settlement_coverage.parquet"
    dial_card_path = results_dir / "sarima_dial_scorecard.parquet"

    # --- dial ---
    prior: pl.DataFrame | None = None
    seeded = False
    if args.force:
        prior = None
    elif dial_path.exists():
        prior = pl.read_parquet(dial_path)
        logger.info(
            "loaded prior dial {} ({} rows)",
            display_path(dial_path, root=root),
            prior.height,
        )
    elif args.append or args.seed_from_frozen or data_path.name == "current":
        prior = _seed_dial_from_frozen(root, dial_path)
        seeded = prior is not None

    # Rebuild when forced, appending, freshly seeded, or dial still missing.
    rebuild_dial = bool(
        args.force or args.append or seeded or not dial_path.exists()
    )
    if not rebuild_dial:
        logger.info(
            "{} exists — loading (FORCE=1 full rebuild, APPEND=1 extend)",
            display_path(dial_path, root=root),
        )
        dial = pl.read_parquet(dial_path)
    else:
        logger.info("building panel series from {}", data_dir)
        panel = build_panel(root, data_dir=data_dir)
        series = daily_wm_r_median(panel)
        logger.info(
            "weekday series: {} days [{} .. {}]",
            series.height,
            series["booked_date"].min(),
            series["booked_date"].max(),
        )
        dial = walk_forward_sarima_dial(
            series,
            min_history=args.min_history,
            progress_every=args.progress_every,
            prior_dial=None if args.force else prior,
        )
        write_parquet_atomic(dial, dial_path)
        logger.info("wrote {} ({} rows)", display_path(dial_path, root=root), dial.height)

    # Settlement coverage KPI (lag proxy on ex-post panel).
    panel_for_cov = build_panel(root, data_dir=data_dir)
    cov = settlement_coverage_daily(
        panel_for_cov, dial, settlement_lag_days=3
    )
    write_parquet_atomic(cov, coverage_path)
    logger.info("wrote settlement coverage → {}", display_path(coverage_path, root=root))

    rolling = rolling_origin_dial_scorecard(dial)
    write_parquet_atomic(rolling, rolling_path)

    card = dial_scorecard(dial, eval_only=True)
    write_parquet_atomic(card, dial_card_path)
    logger.info("dial scorecard (eval window):\n{}", card)

    cal_row = card.filter(pl.col("model") == "sarima_cal")
    naive_row = card.filter(pl.col("model") == "naive_n1")
    mae_cal = cal_row["mae_pp"][0] if cal_row.height else None
    mae_naive = naive_row["mae_pp"][0] if naive_row.height else None
    gate_ok = (
        mae_cal is not None
        and mae_naive is not None
        and mae_cal < mae_naive
    )
    if gate_ok:
        lift = cal_row["lift_vs_naive_pct"][0]
        logger.info(
            "dial MAE gate PASS: sarima_cal {:.4f} pp < naive {:.4f} pp (lift {:+.2f}%)",
            mae_cal,
            mae_naive,
            lift if lift is not None else float("nan"),
        )
    else:
        logger.warning(
            "dial MAE gate FAIL: sarima_cal={} pp vs naive={} pp",
            mae_cal,
            mae_naive,
        )
        if not args.allow_mae_fail and not args.dial_only:
            logger.error(
                "refusing dollar materialization after MAE gate failure "
                "(pass --allow-mae-fail to override)"
            )
            return 1

    if args.dial_only:
        return 0 if gate_ok or args.allow_mae_fail else 1

    # --- sarima_pp quantiles ---
    rebuild_pp = args.force or args.append or not pp_path.exists()
    if pp_path.exists() and not rebuild_pp:
        logger.info("{} exists — loading (--force/--append to rebuild)", display_path(pp_path, root=root))
        sarima_pp = pl.read_parquet(pp_path)
    else:
        panel = build_panel(root, data_dir=data_dir)
        knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
        etp = (
            pl.scan_parquet(data_path / "features.parquet")
            .select("loadnumber", *knn_cols)
            .collect()
        )
        loads = panel.join(etp, on="loadnumber", how="inner").filter(
            pl.col("knn_5") >= 0
        )
        pp_dates = _forecast_dates(dial, eval_only=args.eval_only_pp)
        loads_pp = loads.filter(pl.col("booked_date").is_in(pp_dates))
        logger.info(
            "materializing sarima_pp on {} loads across {} {} dates",
            loads_pp.height,
            len(pp_dates),
            "eval-only" if args.eval_only_pp else "forecast",
        )
        sarima_pp = materialize_sarima_pp(loads_pp, dial, validate=True)
        write_parquet_atomic(sarima_pp, pp_path)
        logger.info(
            "wrote {} ({} rows, {:.0f} MB)",
            display_path(pp_path, root=root),
            sarima_pp.height,
            pp_path.stat().st_size / 1e6,
        )

    # --- option C: shipped tail shape + SARIMA center ---
    blend_path = data_path / "sarima_blend_quantiles.parquet"
    rebuild_blend = (
        not args.skip_blend
        and (args.force or args.append or not blend_path.exists())
    )
    if args.skip_blend:
        logger.info("skipping sarima_blend (--skip-blend)")
    elif blend_path.exists() and not rebuild_blend:
        logger.info(
            "{} exists — loading (--force/--append to rebuild)",
            display_path(blend_path, root=root),
        )
    else:
        panel_b = build_panel(root, data_dir=data_dir)
        knn_cols_b = [f"knn_{int(round(a * 100))}" for a in GRID]
        etp_b = (
            pl.scan_parquet(data_path / "features.parquet")
            .select("loadnumber", *knn_cols_b)
            .collect()
        )
        loads_b = panel_b.join(etp_b, on="loadnumber", how="inner").filter(
            pl.col("knn_5") >= 0
        )
        blend_dates = _forecast_dates(dial, eval_only=args.eval_only_pp)
        loads_blend = loads_b.filter(pl.col("booked_date").is_in(blend_dates))
        logger.info(
            "materializing sarima_blend on {} loads across {} {} dates",
            loads_blend.height,
            len(blend_dates),
            "eval-only" if args.eval_only_pp else "forecast",
        )
        sarima_blend = materialize_sarima_blend(loads_blend, dial, validate=True)
        write_parquet_atomic(sarima_blend, blend_path)
        logger.info(
            "wrote {} ({} rows, {:.0f} MB)",
            display_path(blend_path, root=root),
            sarima_blend.height,
            blend_path.stat().st_size / 1e6,
        )

    # --- stacked SARIMA dial + Hybrid cell offset ---
    hybrid_stack_path = data_path / "sarima_hybrid_quantiles.parquet"
    if not args.skip_hybrid_stack and not args.dial_only:
        panel_h = build_panel(root, data_dir=data_dir)
        logger.info("materializing sarima_hybrid (SARIMA dial + Hybrid δ)")
        sarima_hybrid = materialize_sarima_hybrid(
            panel_h,
            dial,
            HybridConfig(),
            root,
            data_dir=data_dir,
        )
        write_parquet_atomic(sarima_hybrid, hybrid_stack_path)
        logger.info(
            "wrote {} ({} rows)",
            display_path(hybrid_stack_path, root=root),
            sarima_hybrid.height,
        )

    # --- option D: blend base + cell δ + residual global tail offset ---
    tail_path = data_path / "sarima_tail_quantiles.parquet"
    rebuild_tail = (
        not args.skip_tail
        and (args.force or args.append or not tail_path.exists())
    )
    if args.skip_tail:
        logger.info("skipping sarima_tail (--skip-tail)")
    elif tail_path.exists() and not rebuild_tail:
        logger.info(
            "{} exists — loading (--force/--append to rebuild)",
            display_path(tail_path, root=root),
        )
    else:
        panel_t = build_panel(root, data_dir=data_dir)
        logger.info("materializing sarima_tail (blend base + global tail conformal)")
        sarima_tail = materialize_sarima_tail(
            panel_t,
            dial,
            HybridConfig(),
            root,
            data_dir=data_dir,
            base="blend",
        )
        write_parquet_atomic(sarima_tail, tail_path)
        logger.info(
            "wrote {} ({} rows)",
            display_path(tail_path, root=root),
            sarima_tail.height,
        )

    if not args.skip_asof and args.settlement_lag > 0:
        lag = args.settlement_lag
        asof_dial_path = data_path / f"sarima_dial_walkforward_lag{lag}.parquet"
        asof_card_path = results_dir / f"sarima_dial_scorecard_lag{lag}.parquet"
        logger.info("building as-of dial series (settlement lag {}d)", lag)
        panel_lag = build_panel(root, data_dir=data_dir)
        lag_series = daily_wm_r_median_settlement_lag(panel_lag, settlement_lag_days=lag)
        asof_dial = walk_forward_sarima_dial(
            lag_series.drop("obs_booked_date"),
            min_history=args.min_history,
            progress_every=args.progress_every,
        )
        write_parquet_atomic(asof_dial, asof_dial_path)
        asof_card = dial_scorecard(asof_dial, eval_only=True)
        write_parquet_atomic(asof_card, asof_card_path)
        logger.info("as-of lag {}d dial scorecard:\n{}", lag, asof_card)

    if args.skip_scorecard:
        return 0

    # --- scorecard vs Hybrid / DQT ---
    features_path = data_path / "features.parquet"
    hybrid_path = data_path / "hybrid_quantiles.parquet"
    if not features_path.exists():
        logger.error("missing {} — run make features", features_path)
        return 1
    if not hybrid_path.exists():
        logger.error("missing {} — run make hybrid", hybrid_path)
        return 1

    eval_dates = (
        dial.filter(
            pl.col("in_eval")
            & pl.col("y_hat_sarima_cal").is_not_null()
            & pl.col("y_hat_sarima_cal").is_not_nan()
        )
        .get_column("booked_date")
        .to_list()
    )
    knn_cols = [f"knn_{int(round(a * 100))}" for a in GRID]
    p_cols = [f"p{int(round(a * 100)):02d}" for a in GRID]
    feat_cols = [
        ID_COL,
        DATE_COL,
        "booked_on_date",
        "load_type",
        "load_class_bucket",
        COST_COL,
        *knn_cols,
        *p_cols,
        "t1_setting",
        "t2_setting",
        "t3_setting",
        "t4_setting",
        "t1",
        "t2",
        "t3",
        "t4",
    ]
    # haul_band may live on panel; features often has loadmiles
    scan = pl.scan_parquet(features_path)
    available = set(scan.collect_schema().names())
    extra = [c for c in ("loadmiles", "haul_band", "mode", "equipment") if c in available]
    frame = (
        scan.select([c for c in feat_cols + extra if c in available])
        .collect()
        .filter(pl.col("load_type").is_in(("DRY", "REEFER")))
        .with_columns(
            (
                pl.col(DATE_COL)
                if DATE_COL in available
                else pl.col("booked_on_date")
            )
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("booked_date"),
            pl.col("load_class_bucket").alias("mode"),
            pl.col("load_type")
            .replace_strict(EQUIPMENT_MAP, default="Other")
            .alias("equipment"),
        )
        .drop_nulls("booked_date")
        .filter(pl.col("booked_date").is_in(eval_dates))
    )
    if "haul_band" not in frame.columns:
        from dqt.panel import haul_band

        if "loadmiles" not in frame.columns:
            logger.error("features missing loadmiles/haul_band for cell gaps")
            return 1
        frame = frame.with_columns(haul_band(pl.col("loadmiles")).alias("haul_band"))

    hybrid = pl.read_parquet(hybrid_path)
    hy_cols = [c for c in hybrid.columns if c.startswith("hybrid_")]
    frame = frame.join(
        hybrid.select(["loadnumber", *hy_cols]), on="loadnumber", how="inner"
    )
    pp = pl.read_parquet(pp_path)
    pp_cols = [c for c in pp.columns if c.startswith("sarima_pp_")]
    frame = frame.join(
        pp.select(["loadnumber", *pp_cols]), on="loadnumber", how="inner"
    )
    if hybrid_stack_path.exists():
        sh = pl.read_parquet(hybrid_stack_path)
        sh_cols = [c for c in sh.columns if c.startswith("sarima_hybrid_")]
        frame = frame.join(
            sh.select(["loadnumber", *sh_cols]), on="loadnumber", how="inner"
        )
    if blend_path.exists():
        bl = pl.read_parquet(blend_path)
        bl_cols = [c for c in bl.columns if c.startswith("sarima_blend_")]
        frame = frame.join(
            bl.select(["loadnumber", *bl_cols]), on="loadnumber", how="inner"
        )
    if tail_path.exists():
        tl = pl.read_parquet(tail_path)
        tl_cols = [c for c in tl.columns if c.startswith("sarima_tail_")]
        frame = frame.join(
            tl.select(["loadnumber", *tl_cols]), on="loadnumber", how="inner"
        )
    logger.info("scorecard frame: {} loads", frame.height)

    scored = score_sarima_walkforward(frame)
    global_card = scored["global"].with_columns(pl.lit("global").alias("slice"))
    level_card = scored["quantile_level"].with_columns(pl.lit("global").alias("slice"))
    write_parquet_atomic(level_card, level_card_path)
    logger.info("wrote quantile-level scorecard → {}", display_path(level_card_path, root=root))

    kept_episodes = {
        name: tmpl
        for name, tmpl in MODEL_QCOL_SARIMA.items()
        if all(tmpl.format(q=int(round(a * 100))) in frame.columns for a in GRID)
    }
    ep_min, ep_max = _episode_date_bounds()
    episode_frame = _build_features_slice(
        root,
        data_path,
        date_min=ep_min,
        date_max=ep_max,
        hybrid_path=hybrid_path,
        pp_path=pp_path,
        hybrid_stack_path=hybrid_stack_path,
        blend_path=blend_path,
        tail_path=tail_path,
    )
    logger.info(
        "episode replay frame: {} loads [{} .. {}]",
        episode_frame.height,
        ep_min,
        ep_max,
    )
    episodes = score_market_episodes(episode_frame, model_qcol=kept_episodes)
    write_parquet_atomic(episodes, episode_path)
    logger.info("wrote episode replay → {}\n{}", display_path(episode_path, root=root), episodes)

    cell = scored["cell_gap"]
    if cell is not None and cell.height:
        cell_path = results_dir / "sarima_walkforward_cell_gap.parquet"
        write_parquet_atomic(cell, cell_path)
        logger.info("wrote {} ({} cells)", display_path(cell_path, root=root), cell.height)
        mean_abs = float(cell["hybrid_gap_pp"].abs().mean())
        worst = float(cell["hybrid_gap_pp"].abs().max())
        logger.info(
            "SARIMA_pp cell gaps: mean|gap|={:.2f} pp  worst|gap|={:.2f} pp",
            mean_abs,
            worst,
        )

    write_parquet_atomic(global_card, scorecard_path)
    logger.info("wrote {}\n{}", display_path(scorecard_path, root=root), global_card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
