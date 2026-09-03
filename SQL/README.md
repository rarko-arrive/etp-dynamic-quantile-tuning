# SQL catalog

Snowflake queries for ETP-DQT. Execute via `arriveds.snowflake.query_sf` (prefer `Path` + `params=`). Lint with `make lint-sql` (config from arrive-ds `templates/sqlfluff/`, adopted as root `.sqlfluff`).

## Conventions

- **Single statement** per file; trailing `;` OK.
- **Params:** Python `str.format` placeholders `{name}` (same as `query_sf(..., params=)`).
- **Header** on every key file (see template below).
- **Post-fetch:** `query_sf` returns lowercase column names by default (`lowercase=True`).
- **sqlfluff:** ignore generated / incomplete slider fragments (see `.sqlfluffignore`). When bumping `arriveds`, diff `.sqlfluff` against arrive-ds `templates/sqlfluff/.sqlfluff`.

### Header template

```sql
-- Purpose: …
-- Owner: …
-- Params: {name} — meaning (or: none)
-- Consumers: make … | notebooks/… | module.path
```

## Inventory

| Path | Purpose | Params | Consumers |
|------|---------|--------|-----------|
| [`dqt/features.sql`](dqt/features.sql) | DQT feature store → parquet | `{start_date}`, `{end_date}` | `make features` / `scripts/build_features.py` |
| [`etp/big_miss_shipments.sql`](etp/big_miss_shipments.sql) | Large \|ETP50 − cost\| miss analysis | none | `notebooks/dqt.ipynb` |
| [`etp-slider/etp-feature-history.sql`](etp-slider/etp-feature-history.sql) | Snapshot-grain ETP + curated `features_used` for analytics mart | `{ship_date_start}`, `{ship_date_end}` | `build_etp_lake` mart stage |
| [`etp-slider/etp-feature-keys-inventory.sql`](etp-slider/etp-feature-keys-inventory.sql) | Sample `OBJECT_KEYS(features_used)` for catalog | `{ship_date_start}`, `{ship_date_end}`, `{sample_limit}` | catalog refresh / `etp-features.ipynb` |
| [`etp-slider/etp-slider-cte.sql`](etp-slider/etp-slider-cte.sql) | Shared slider CTE (source) | `{ship_date_start}`, `{ship_date_end}` | `dqt.etp_slider.slider_sql` |
| [`etp-slider/etp-slider-shift-select.sql`](etp-slider/etp-slider-shift-select.sql) | Aggregate shift SELECT tail | none (uses CTE) | `dqt.etp_slider` |
| [`etp-slider/etp-slider-per-load-select.sql`](etp-slider/etp-slider-per-load-select.sql) | Per-load SELECT tail | none (uses CTE) | `dqt.etp_slider` |
| [`etp-slider/etp-slider-history-select.sql`](etp-slider/etp-slider-history-select.sql) | History SELECT tail (incomplete alone) | none (uses CTE) | `dqt.etp_slider` |
| [`etp-slider/etp-slider*.sql`](etp-slider/) (composites) | **Generated** CTE+tail with dates baked | n/a — regenerate | ad-hoc `query_sf(Path)`; do not hand-edit |
| [`ds-monitoring-queries/dqt_alt_percentiles.sql`](ds-monitoring-queries/dqt_alt_percentiles.sql) | Latest DQT alt row per `valid_date` | none | Tableau; `dqt.model.global_dqt.GlobalDQT.history`; `notebooks/ds-monitoring.ipynb` |
| [`ds-monitoring-queries/dqt_vs_non_adjusted.sql`](ds-monitoring-queries/dqt_vs_non_adjusted.sql) | ETP vs DQT attainment | none | Tableau; `notebooks/ds-monitoring.ipynb` |
| [`ds-monitoring-queries/etp_dashboard_v2.sql`](ds-monitoring-queries/etp_dashboard_v2.sql) | ETP Dashboard v2 financial metrics | none | Tableau; `notebooks/ds-monitoring.ipynb` |
| [`ds-monitoring-queries/etp_monitoring_monthly_report.sql`](ds-monitoring-queries/etp_monitoring_monthly_report.sql) | ETP monitoring monthly report | none | Tableau; `notebooks/ds-monitoring.ipynb` |

Regenerate slider composites:

```bash
uv run python -m dqt.etp_slider
```

Playbook: [`.ai/plans/sql-version-control.md`](../.ai/plans/sql-version-control.md).
