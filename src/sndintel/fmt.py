"""Number formatting shared by every pack.

Rules: kilograms under 0.1 MT (never "0.00 MT" for a 4 kg drop), two decimals
under 10 MT, one decimal from 10 MT up. Integer MT is never printed — at city
level a 1 MT rounding step is a whole DSR's week.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_mt(value: Any, *, zero: str = "0 kg", signed: bool = False) -> str:
    v = to_float(value)
    sign = "+" if signed and v > 0 else ""
    if abs(v) < 0.0005:
        return zero
    if abs(v) < 0.1:
        return f"{sign}{v * 1000:,.0f} kg"
    if abs(v) < 10:
        return f"{sign}{v:,.2f} MT"
    return f"{sign}{v:,.1f} MT"


def kg(value: Any) -> int:
    return int(round(to_float(value) * 1000))


def fmt_kg(value: Any, *, signed: bool = False) -> str:
    """Shop-row prose: always kilograms, so the sentence matches the kg columns beside it."""
    n = kg(value)
    if signed and n > 0:
        return f"+{n:,} kg"
    return f"{n:,} kg"


# Excel number formats for "(MT)" table columns. Two decimals = 10 kg resolution,
# which is the carton grain the extracts are quantised at.
MT_FORMAT = "#,##0.00"
MT_SIGNED_FORMAT = "+#,##0.00;-#,##0.00;0.00"
COUNT_FORMAT = "#,##0"

_SIGNED_PREFIXES = ("vs ", "From ", "Extra", "MoM", "YoY", "Change")


def round_mt(value: Any) -> float | None:
    """Table MT value: numeric (Excel can sum it) at two decimals; None when missing."""
    try:
        if value is None or pd.isna(value):
            return None
        v = round(float(value), 2)
        return 0.0 if v == 0 else v  # never a "-0.00"
    except (TypeError, ValueError):
        return None


def signed_header(header: str) -> bool:
    """Columns that carry a direction (vs Target, From drop size, Extra …) print a sign."""
    return str(header).startswith(_SIGNED_PREFIXES)


def is_mt_header(header: str) -> bool:
    return "(MT)" in str(header)


def is_kg_header(header: str) -> bool:
    h = str(header)
    return "(kg)" in h or "(KG)" in h


def is_count_header(header: str) -> bool:
    return str(header) in {
        "Billed shops",
        "Visited shops",
        "Universe",
        "Visits MTD",
        "Doors",
        "Billed doors",
        "Usual billed doors",
        "Doors short",
        "Shops",
        "Days since",
        "Days since bill",
        "Days overdue",
        "Usual cycle (days)",
        "Cover left (days)",
    }


def is_pct_header(header: str) -> bool:
    h = str(header)
    if is_mt_header(h) or is_kg_header(h):
        return False
    return h.endswith("%") or h.endswith("(%)") or "Strike" in h


def fmt_cell(value: Any, header: str) -> str:
    """Text for one PDF table cell, driven by the column header.

    "(MT)" columns print two decimals (signed for vs/From/Extra columns), kg and
    count columns print whole numbers with thousands separators, percent columns
    print whole percents. Everything else is passed through as text. Integer MT
    never appears: at city level a 1 MT rounding step is a whole DSR-week.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    v = float(value)
    if is_mt_header(header):
        return f"{v:+,.2f}" if signed_header(header) else f"{v:,.2f}"
    if is_kg_header(header) or is_count_header(header):
        return f"{int(round(v)):+,}" if signed_header(header) else f"{int(round(v)):,}"
    if is_pct_header(header):
        return f"{int(round(v))}"
    if isinstance(value, float):
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}"
    return f"{value:,}"


def excel_number_format(header: str) -> str | None:
    """openpyxl number_format for a table column, or None to leave Excel's default."""
    if is_mt_header(header):
        return MT_SIGNED_FORMAT if signed_header(header) else MT_FORMAT
    if is_kg_header(header) or is_count_header(header):
        return COUNT_FORMAT
    if is_pct_header(header):
        return "0"
    return None


def fmt_pct(value: Any, *, signed: bool = True, decimals: int = 1) -> str:
    v = to_float(value, float("nan"))
    if pd.isna(v):
        return "n/a"
    v = round(v, decimals)
    if v == 0:
        return f"{0.0:.{decimals}f}%"  # never "-0%" or "+0%"
    return f"{v:+.{decimals}f}%" if signed else f"{v:.{decimals}f}%"
