"""Freight-relevant holiday calendar.

Used by the SARIMA dial as an exogenous Monday/Friday/holiday flag.
Nothing is baked into the panel or features parquet — dates are resolved
at analysis time from :data:`DEFAULT_HOLIDAYS`.

DOT Blitz Week is CVSA International Roadcheck (72-hour inspection blitz).
Dates are explicit per year via ``DOT_BLITZ_WEEKS``; disabled by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass(frozen=True, slots=True)
class HolidaySpec:
    """One configurable holiday (or multi-day event)."""

    name: str
    kind: str  # federal | dot | custom
    rule: str  # fixed | nth_weekday | explicit_dates
    month: int | None = None
    day: int | None = None  # fixed: calendar day
    nth: int | None = None  # nth_weekday: 1=first, …, -1=last
    weekday: int | None = None  # nth_weekday: Mon=0 … Sun=6
    explicit_dates: tuple[date, ...] = ()
    enabled: bool = True
    pre_days: int = 3
    post_days: int = 3
    observe_weekend: bool = True


def _observe(d: date) -> date:
    """US federal observed-day shift: Sat → Fri, Sun → Mon."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, nth: int, weekday: int) -> date:
    """Nth weekday in month (weekday: Mon=0…Sun=6). ``nth=-1`` → last."""
    if nth == -1:
        if month == 12:
            d = date(year, 12, 31)
        else:
            d = date(year, month + 1, 1) - timedelta(days=1)
        while d.weekday() != weekday:
            d -= timedelta(days=1)
        return d
    if nth < 1:
        raise ValueError(f"nth must be >= 1 or -1, got {nth}")
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    d += timedelta(weeks=nth - 1)
    if d.month != month:
        raise ValueError(f"no {nth}-th weekday={weekday} in {year}-{month:02d}")
    return d


def _statutory_date(spec: HolidaySpec, year: int) -> date:
    if spec.rule == "fixed":
        if spec.month is None or spec.day is None:
            raise ValueError(f"{spec.name}: fixed rule needs month and day")
        return date(year, spec.month, spec.day)
    if spec.rule == "nth_weekday":
        if spec.month is None or spec.nth is None or spec.weekday is None:
            raise ValueError(
                f"{spec.name}: nth_weekday rule needs month, nth, weekday"
            )
        return _nth_weekday(year, spec.month, spec.nth, spec.weekday)
    raise ValueError(f"{spec.name}: unsupported rule {spec.rule!r}")


def _contiguous_spans(dates: Sequence[date]) -> list[tuple[date, date]]:
    """Group sorted dates into inclusive (start, end) runs of consecutive days."""
    if not dates:
        return []
    ordered = sorted(dates)
    spans: list[tuple[date, date]] = []
    start = prev = ordered[0]
    for d in ordered[1:]:
        if d == prev + timedelta(days=1):
            prev = d
            continue
        spans.append((start, prev))
        start = prev = d
    spans.append((start, prev))
    return spans


def date_range(start: date, end: date) -> tuple[date, ...]:
    """Inclusive calendar days from ``start`` through ``end``."""
    if end < start:
        raise ValueError(f"date_range: end {end} before start {start}")
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return tuple(out)


# CVSA International Roadcheck ("DOT Blitz Week") — 72h inspection event.
DOT_BLITZ_WEEKS: tuple[tuple[date, date], ...] = (
    (date(2016, 6, 7), date(2016, 6, 9)),
    (date(2017, 6, 6), date(2017, 6, 8)),
    (date(2018, 6, 5), date(2018, 6, 7)),
    (date(2019, 6, 4), date(2019, 6, 6)),
    (date(2020, 9, 9), date(2020, 9, 11)),
    (date(2021, 5, 4), date(2021, 5, 6)),
    (date(2022, 5, 17), date(2022, 5, 19)),
    (date(2023, 5, 16), date(2023, 5, 18)),
    (date(2024, 5, 14), date(2024, 5, 16)),
    (date(2025, 5, 13), date(2025, 5, 15)),
    (date(2026, 5, 12), date(2026, 5, 14)),
)


def _flatten_week_ranges(weeks: Sequence[tuple[date, date]]) -> tuple[date, ...]:
    days: list[date] = []
    for start, end in weeks:
        days.extend(date_range(start, end))
    return tuple(days)


DEFAULT_HOLIDAYS: tuple[HolidaySpec, ...] = (
    HolidaySpec(name="New Year's Day", kind="federal", rule="fixed", month=1, day=1),
    HolidaySpec(
        name="Memorial Day",
        kind="federal",
        rule="nth_weekday",
        month=5,
        nth=-1,
        weekday=0,
    ),
    HolidaySpec(name="Independence Day", kind="federal", rule="fixed", month=7, day=4),
    HolidaySpec(
        name="Labor Day",
        kind="federal",
        rule="nth_weekday",
        month=9,
        nth=1,
        weekday=0,
    ),
    HolidaySpec(
        name="Thanksgiving",
        kind="federal",
        rule="nth_weekday",
        month=11,
        nth=4,
        weekday=3,
    ),
    HolidaySpec(name="Christmas Day", kind="federal", rule="fixed", month=12, day=25),
    HolidaySpec(
        name="DOT Blitz Week",
        kind="dot",
        rule="explicit_dates",
        explicit_dates=_flatten_week_ranges(DOT_BLITZ_WEEKS),
        enabled=False,
        pre_days=0,
        post_days=0,
        observe_weekend=False,
    ),
)


def enabled_specs(*, include_dot: bool = True) -> tuple[HolidaySpec, ...]:
    """Holiday specs with ``enabled=True``; DOT Blitz optional."""
    out: list[HolidaySpec] = []
    for spec in DEFAULT_HOLIDAYS:
        if not spec.enabled:
            continue
        if spec.kind == "dot" and not include_dot:
            continue
        out.append(spec)
    return tuple(out)


def resolve_occurrences(
    specs: Sequence[HolidaySpec] | None = None,
    *,
    year_start: int,
    year_end: int,
) -> pl.DataFrame:
    """One row per holiday occurrence in ``[year_start, year_end]``.

    Columns: ``name``, ``kind``, ``observed_date``, ``event_start``,
    ``event_end``, ``window_start``, ``window_end``.
    """
    if year_end < year_start:
        raise ValueError("year_end must be >= year_start")
    specs = tuple(specs if specs is not None else DEFAULT_HOLIDAYS)
    rows: list[dict] = []
    for spec in specs:
        if not spec.enabled:
            continue
        if spec.rule == "explicit_dates":
            dates = [d for d in spec.explicit_dates if year_start <= d.year <= year_end]
            for event_start, event_end in _contiguous_spans(dates):
                rows.append(
                    {
                        "name": spec.name,
                        "kind": spec.kind,
                        "observed_date": event_start,
                        "event_start": event_start,
                        "event_end": event_end,
                        "window_start": event_start - timedelta(days=spec.pre_days),
                        "window_end": event_end + timedelta(days=spec.post_days),
                    }
                )
            continue
        for year in range(year_start, year_end + 1):
            statutory = _statutory_date(spec, year)
            observed = _observe(statutory) if spec.observe_weekend else statutory
            rows.append(
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "observed_date": observed,
                    "event_start": observed,
                    "event_end": observed,
                    "window_start": observed - timedelta(days=spec.pre_days),
                    "window_end": observed + timedelta(days=spec.post_days),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "name": pl.Utf8,
                "kind": pl.Utf8,
                "observed_date": pl.Date,
                "event_start": pl.Date,
                "event_end": pl.Date,
                "window_start": pl.Date,
                "window_end": pl.Date,
            }
        )
    return pl.DataFrame(rows).sort(["observed_date", "name"])
