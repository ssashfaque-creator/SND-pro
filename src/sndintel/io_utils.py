"""Helpers for messy Excel / CSV tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def read_raw_table(path: PathLike, sheet: Union[str, int, None] = 0) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, header=None, dtype=object, sheet_name=sheet)
    elif suffix in {".csv", ".txt"}:
        df = _read_ragged_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if isinstance(df, dict):
        df = next(iter(df.values()))
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    df = df.reset_index(drop=True)
    df.columns = list(range(df.shape[1]))
    return df


def _read_ragged_csv(path: Path) -> pd.DataFrame:
    """SSRS CSVs are jagged: parameter rows have ~18 fields, the tablix has 40+.

    pandas' C engine rejects that. Pad every row to the max width.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return pd.DataFrame()
    width = max(len(row) for row in rows)
    padded = [row + [None] * (width - len(row)) for row in rows]
    df = pd.DataFrame(padded, dtype=object)
    df.replace("", None, inplace=True)
    return df


def cell_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def norm_key(value) -> str:
    text = cell_str(value).lower()
    for ch in ("\n", "\r", "_", "-", ".", "/", "\\"):
        text = text.replace(ch, " ")
    return " ".join(text.split())


def parse_month(value) -> int | None:
    text = cell_str(value)
    if not text:
        return None
    key = text.lower().replace(".", "")
    if key in MONTH_MAP:
        return MONTH_MAP[key]
    try:
        num = int(float(text))
        if 1 <= num <= 12:
            return num
    except ValueError:
        pass
    return None


def parse_year(value) -> int | None:
    text = cell_str(value)
    if not text:
        return None
    try:
        num = int(float(text))
        if 1990 <= num <= 2100:
            return num
    except ValueError:
        return None
    return None


def parse_volume(value) -> float | None:
    text = cell_str(value)
    if not text or text.lower() in {"nan", "none", "-", "null"}:
        return None
    text = text.replace(",", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


def looks_like_store_id(value) -> bool:
    text = cell_str(value)
    if not text or " " in text:
        return False
    if text.lower().endswith("total"):
        return False
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    return digits >= 6 and letters <= 4 and 6 <= len(text) <= 32


_TOTAL_LABEL = re.compile(r"^(grand|sub|running|page|net)?\s*totals?\s*:?$")


def looks_like_total(value) -> bool:
    """A total / subtotal *label*, not any name that happens to contain 'total'.

    "Total", "Grand Total", "Karachi Total", "Total for Ali" are labels.
    "TOTAL PUMP", "STAR MART TOTAL PUMP" and "TOTAL M/STORE" are shops
    (Total is a fuel brand) and must be kept.
    """
    text = " ".join(cell_str(value).lower().split())
    if not text or "total" not in text:
        return False
    if _TOTAL_LABEL.match(text):
        return True
    if text.endswith(" total") or text.endswith(" totals") or text.endswith(" total:"):
        return True
    return text.startswith("total for ") or text.startswith("total of ") or text.startswith("totals for ")


def period_key(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def shift_period(period: str, months: int) -> str:
    year = int(period[:4])
    month = int(period[5:7]) + months
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    return period_key(year, month)


def prior_periods(period: str, n: int = 3) -> list[str]:
    """n calendar months immediately before ``period``, oldest first.

    Scoring 2026-08 with n=3 → 2026-05, 2026-06, 2026-07. Does not skip a
    missing May and pull in 2025-07 to fill the window.
    """
    if not period or n <= 0:
        return []
    return [shift_period(str(period), -i) for i in range(int(n), 0, -1)]


def panel_start(shop_month: pd.DataFrame | None) -> str | None:
    """First period on file, or None when nothing is loaded."""
    if shop_month is None or shop_month.empty or "period" not in shop_month.columns:
        return None
    periods = shop_month["period"].dropna().astype(str)
    periods = periods[periods.str.len() >= 7]
    return str(periods.min()) if not periods.empty else None


def window_periods(period: str, n: int, shop_month: pd.DataFrame | None = None, start: str | None = None) -> list[str]:
    """``prior_periods`` clipped to the months the data on file can speak for.

    A month before the first period on file is *unknown*, not a zero: with
    extracts from May onward, July's last-3 window is May–June (divisor 2),
    not April–June with April counted as nothing sold. Inside the panel a
    missing month is still a real zero. ``start`` overrides the panel start
    read from ``shop_month``.
    """
    window = prior_periods(period, n)
    first = start or panel_start(shop_month)
    if not first:
        return window
    return [p for p in window if p >= str(first)]
