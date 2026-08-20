"""Helpers for messy Excel / CSV tables."""

from __future__ import annotations

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
        df = pd.read_csv(path, header=None, dtype=object, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if isinstance(df, dict):
        df = next(iter(df.values()))
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
    df = df.reset_index(drop=True)
    df.columns = list(range(df.shape[1]))
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
    return digits >= 6 and letters <= 4 and 6 <= len(text) <= 20


def looks_like_total(value) -> bool:
    text = cell_str(value).lower()
    return "total" in text


def period_key(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"
