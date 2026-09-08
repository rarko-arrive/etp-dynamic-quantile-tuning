# etp-dqt

Dynamic Quantile Tuning (DQT) for Elite Truck Pricing — the **quantile pipeline** workstream.

Snowflake features → hybrid quantiles → walk-forward SARIMA global dial → shadow publish loop.

Split from [etp-dqt monorepo](https://github.com/rarko-arrive/etp-dqt). Source of truth for `$DQT_DATA_DIR/features.parquet`, `hybrid_quantiles.parquet`, and `sarima_dial_walkforward.parquet`.

## Setup

```bash
cd ~/Git/Projects/etp/etp-dqt
cp .env.example .env   # Snowflake creds, DQT_DATA_DIR, START_DATE
make install
make test
```

Uses `uv` (Python 3.12). `.venv` may symlink off-repo via `UV_PROJECT_ENVIRONMENT`.

## Commands

```bash
make features          # Snowflake → $DQT_DATA_DIR/features.parquet
make hybrid            # hybrid_quantiles.parquet
make sarima-wf         # walk-forward dial + scorecards  [APPEND=1]
make sarima-publish    # kill-switch + next-day schedule
make shadow-dqt        # daily shadow eval  [SKIP_SF=1]
make guardrails-assess # empirical fire-rates  [QUICK=1]
make smoke             # 14-day pull → data/current
make smoke-pipeline    # synthetic fixtures → shadow (no SF)
make shadow-alerts     # KPI alerts on shadow history  [NOTIFY=1]
make shadow-bakeoff    # 30-weekday SLO assessment  [MIN_WEEKDAYS=30]
make replay-cadence    # holiday cadence counterfactual backtest
make schedule-daily    # daily chain (features → shadow)
make schedule-weekly   # weekly sarima + guardrails + bake-off
make lint && make test
```

All pipeline targets honor `DQT_DATA_DIR` and skip if output exists (`FORCE=1` rebuilds).

## Layout

| Path | Purpose |
|------|---------|
| `src/dqt/` | Panel, hybrid, conformal, SARIMA dial, shadow loop, guardrails |
| `SQL/dqt/features.sql` | Feature store query |
| `SQL/ds-monitoring-queries/` | Tableau / monitoring SQL |
| `scripts/build_*.py` | CLI entrypoints |
| `notebooks/dqt.ipynb` | Pipeline exploration |
| `dqt-tuning.md` | Guardrail + tuning playbook |

## Data outputs

Default root: `$DQT_DATA_DIR` (usually `data/`)

| Artifact | Producer |
|----------|----------|
| `features.parquet` | `make features` |
| `hybrid_quantiles.parquet` | `make hybrid` |
| `sarima_dial_walkforward.parquet` | `make sarima-wf` |
| `sarima_pp_quantiles.parquet` | materialized during sarima-wf |
| `tuning/guardrail_assessment_*.md` | `make guardrails-assess` |

## Sibling repos

| Repo | Relationship |
|------|--------------|
| **[etp-lake](../etp-lake/)** | Consumes DQT parquets for SARIMA-vs-ETP comparisons and paint experiments |
| **[etp-explanations](../etp-explanations/)** | Independent — reads `$DQT_DATA_DIR/etp_lake/` built by etp-lake |

## Gotchas

- **`.env` duplicate keys**: keep one active value per knob (Make uses last, python-dotenv uses first).
- **`START_DATE`**: printed first by `make features`; unquoted zsh dates break argparse.
- **Dial-miss decomposition** may read paint helpers — requires hybrid/sarima artifacts on disk.
