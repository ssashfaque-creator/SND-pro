"""Classify closed months vs in-progress MTD from an SSRS extract."""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from typing import Any, Optional

import pandas as pd


def parse_execution_date(params: dict | None) -> Optional[datetime]:
    if not params:
        return None
    raw = params.get("execution_date")
    if not raw:
        return None
    time_part = params.get("execution_time") or "00:00:00"
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            if "%H" in fmt:
                return datetime.strptime(f"{raw} {time_part}", fmt)
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def open_mtd_period(periods: list[str], execution: Optional[datetime]) -> Optional[str]:
    """A period is open MTD when the extract was run during that calendar month, before month-end."""
    if execution is None or not periods:
        return None
    ym = f"{execution.year:04d}-{execution.month:02d}"
    if ym not in set(periods):
        return None
    last = monthrange(execution.year, execution.month)[1]
    if execution.day < last:
        return ym
    return None


def run_rate_factor(execution: Optional[datetime], period: str) -> tuple[float, int, int]:
    """Return (factor, as_of_day, days_in_month). Factor scales MTD to a full-month pace."""
    year, month = int(period[:4]), int(period[5:7])
    days = monthrange(year, month)[1]
    if execution is None or f"{execution.year:04d}-{execution.month:02d}" != period:
        return 1.0, days, days
    day = min(max(execution.day, 1), days)
    return (days / day), day, days


def format_period_label(
    period: str | None,
    *,
    open_: bool = False,
    as_of_day: int | None = None,
    days_in_month: int | None = None,
) -> str:
    """Human month label. Never '8/30' — that reads as 30 August."""
    if not period:
        return ""
    try:
        dt = datetime.strptime(f"{period}-01", "%Y-%m-%d")
    except ValueError:
        return str(period)
    month = dt.strftime("%b %Y")
    mon = dt.strftime("%b")
    if open_ and as_of_day and days_in_month:
        return f"{month} MTD · billed through {int(as_of_day)} {mon} ({int(days_in_month)}-day month)"
    return f"{month} · closed month"


def period_state(ledger: pd.DataFrame | None, period: str | None) -> dict[str, Any]:
    """Normalised closed vs open-MTD state for one calendar month."""
    empty: dict[str, Any] = {
        "period": period,
        "status": "closed",
        "open": False,
        "as_of_day": None,
        "days_in_month": None,
        "factor": 1.0,
        "execution_date": None,
        "source_file": None,
        "label": period or "",
    }
    if not period or ledger is None or ledger.empty or "period" not in ledger.columns:
        return empty
    part = ledger[ledger["period"].astype(str) == str(period)]
    if part.empty:
        return empty
    row = part.iloc[0]
    status = str(row.get("status") or "closed")
    as_of = row.get("as_of_day")
    days = row.get("days_in_month")
    as_of_i = int(as_of) if pd.notna(as_of) else None
    days_i = int(days) if pd.notna(days) else None
    open_ = status == "mtd_open"
    factor = 1.0
    if open_ and as_of_i and days_i:
        factor = days_i / as_of_i
    label = str(period)
    if open_ and as_of_i and days_i:
        label = format_period_label(period, open_=True, as_of_day=as_of_i, days_in_month=days_i)
    elif period:
        label = format_period_label(period, open_=False, as_of_day=as_of_i, days_in_month=days_i)
    exec_raw = row.get("execution_date")
    return {
        "period": period,
        "status": status,
        "open": open_,
        "as_of_day": as_of_i,
        "days_in_month": days_i,
        "factor": factor,
        "execution_date": None if pd.isna(exec_raw) else str(exec_raw),
        "source_file": None if pd.isna(row.get("source_file")) else str(row.get("source_file")),
        "label": label,
    }


def banner_text(ledger: pd.DataFrame | None, period: str | None) -> str:
    state = period_state(ledger, period)
    if not period:
        return ""
    if state["open"] and state["as_of_day"] and state["days_in_month"]:
        return (
            f"{state.get('label') or period} — later extracts replace this month in full; "
            "closed months stay as they are."
        )
    return f"{period} is a closed month. Insights cover the full warehouse, not only the latest file."
