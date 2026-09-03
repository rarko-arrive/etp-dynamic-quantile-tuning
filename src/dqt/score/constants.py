"""Feature-store key names and the standard 5..95-by-5 quantile grid.

``Cols`` is names only — ``id`` / ``time`` / ``date`` are computed in
``SQL/dqt/features.sql`` (Chicago wall time → calendar date). Change the
attributes here if the SQL aliases change; do not recompute ``date`` in Python.
"""

from __future__ import annotations


class Cols:
    id = "loadnumber"
    time = "loaddate"
    date = "date"
    cost = "carrier_shipment_charges_total"


QUANTILES = tuple(range(5, 100, 5))  # 5, 10, ..., 95 (int, percent)
NOMINAL = tuple(q / 100 for q in QUANTILES)  # 0.05, ..., 0.95 (float, fraction)

ID_COL = Cols.id
TIME_COL = Cols.time
DATE_COL = Cols.date
COST_COL = Cols.cost
