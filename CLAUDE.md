# etp-dqt — agent guide

DQT quantile pipeline only. Sibling repos: `../etp-lake`, `../etp-explanations`.

## Commands

```bash
make install && make lint && make test
make features && make hybrid && make sarima-wf
DQT_DATA_DIR=data make shadow-dqt PUBLISH_MODE=cap MAX_WEEKLY_MOVE=0.05
make shadow-bakeoff && make smoke-pipeline
DQT_DATA_DIR=data make guardrails-assess QUICK=1
```

## Key modules

- `src/dqt/score/constants.py` — canonical column names; never recompute `date` from `loaddate` in Python
- `src/dqt/panel.py` — implied percentile panel
- `src/dqt/hybrid.py` / `conformal.py` / `sarima_dial.py` — quantile pipeline
- `SQL/dqt/features.sql` — feature store (lint with `make lint-sql`)

## Data

Everything under `$DQT_DATA_DIR` (default `data/`). Never commit `data/` or `.env`.
