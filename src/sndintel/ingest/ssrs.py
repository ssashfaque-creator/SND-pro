"""Parse SSRS Shop SKU Wise Execution Report exports (Excel or CSV).

These files are not tidy tables. Typical layout:

* Rows 1-3: report chrome (textbox names, parameters, culture).
* A field-id row such as ``txt_cDISTRIB``, ``txt_cPOP_Co``, ``uval_MTD_Se``.
* Human labels (DISTRIBUTOR, DSR NAME, ...) often repeating in columns A-F
  of every data row, with the real values starting around column G.
* Matrix totals (shop / section / DSR / distributor / grand) as extra
  columns to the right — those are discarded.

The parser tries, in order:

1. Named SSRS field ids
2. Human header labels
3. Positional inference from a distributor + store-id + year/month pattern
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from sndintel.io_utils import (
    cell_str,
    looks_like_store_id,
    looks_like_total,
    norm_key,
    parse_month,
    parse_year,
    parse_volume,
    period_key,
    read_raw_table,
)

SALES_COLUMNS = [
    "distributor",
    "dsr_name",
    "section",
    "store_id",
    "store_name",
    "sku",
    "year",
    "month",
    "period",
    "volume_mt",
]

FIELD_ID_MAP = {
    "txt_cdistrib": "distributor",
    "txt_cdsr": "dsr_name",
    "txt_cdsr_na": "dsr_name",
    "txt_csectio": "section",
    "txt_csection": "section",
    "txt_cpop_co": "store_id",
    "txt_cpop_c": "store_id",
    "txt_cpop_na": "store_name",
    "txt_csku_lo": "sku",
    "txt_csku": "sku",
    "txt_calendar": "calendar",
    "uval_mtd_se": "volume_mt",
    "uval_mtd": "volume_mt",
}

HUMAN_HEADER_MAP = {
    "distributor": "distributor",
    "dsr name": "dsr_name",
    "dsr": "dsr_name",
    "salesperson": "dsr_name",
    "section lon": "section",
    "section long": "section",
    "section": "section",
    "area": "section",
    "pop code": "store_id",
    "popcode": "store_id",
    "store id": "store_id",
    "store code": "store_id",
    "outlet code": "store_id",
    "pop name": "store_name",
    "store name": "store_name",
    "outlet name": "store_name",
    "sku long di": "sku",
    "sku long": "sku",
    "sku": "sku",
    "product": "sku",
    "year": "year",
    "calendar year": "year",
    "month": "month",
    "calendar month": "month",
    "mtd sales": "volume_mt",
    "mtd": "volume_mt",
    "volume": "volume_mt",
    "uom tons": "volume_mt",
}

LABEL_VALUES = {
    "distributor",
    "dsr name",
    "dsr",
    "section lon",
    "section",
    "pop code",
    "pop name",
    "sku long di",
    "sku long",
    "sku",
    "year",
    "month",
}


@dataclass
class ParseReport:
    strategy: str
    source_file: str
    n_raw_rows: int
    n_clean_rows: int
    header_row: Optional[int]
    data_start_row: Optional[int]
    column_map: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def parse_sales_file(path: str | Path) -> tuple[pd.DataFrame, ParseReport]:
    path = Path(path)
    raw = read_raw_table(path)
    report = ParseReport(
        strategy="unknown",
        source_file=str(path),
        n_raw_rows=len(raw),
        n_clean_rows=0,
        header_row=None,
        data_start_row=None,
        params=_extract_params(raw),
    )
    skip_cols = _total_columns(raw)
    field_row, field_map = _detect_field_id_row(raw, skip_cols)
    human_row, human_map = _detect_human_header_row(raw, skip_cols)

    if field_map:
        report.strategy = "ssrs_field_ids"
        report.header_row = field_row
        report.column_map = field_map
        data_start = (human_row + 1) if human_row is not None and human_row > field_row else field_row + 1
        mapped = _project_columns(raw, field_map, data_start)
    elif human_map:
        report.strategy = "human_headers"
        report.header_row = human_row
        report.column_map = human_map
        mapped = _project_columns(raw, human_map, human_row + 1)
    else:
        report.strategy = "positional"
        mapped, cmap, data_start = _positional_project(raw, skip_cols)
        report.column_map = cmap
        report.data_start_row = data_start
        if mapped.empty:
            raise ValueError(
                f"Could not detect a sales table in {path.name}. "
                "Expected distributor, store id, SKU, year, month, and MTD volume."
            )

    if report.data_start_row is None:
        report.data_start_row = report.header_row + 1 if report.header_row is not None else 0

    clean = _normalize_sales(mapped, report)
    report.n_clean_rows = len(clean)
    if clean.empty:
        report.warnings.append("Parser found headers but no usable fact rows")
    return clean, report


def _extract_params(raw: pd.DataFrame) -> dict:
    params: dict[str, str] = {}
    blob = " ".join(cell_str(v) for v in raw.head(8).to_numpy().ravel()[:200])
    for key in ("Report", "User ID", "Region", "Area", "Territory", "Town", "UOM", "Culture"):
        m = re.search(rf"{key}\s*:\s*([^|]+)", blob, flags=re.I)
        if m:
            params[key.lower().replace(" ", "_")] = m.group(1).strip()
    if "Shop SKU Wise" in blob:
        params["report_name"] = "Shop SKU Wise Execution Report"
    return params


def _total_columns(raw: pd.DataFrame) -> set[int]:
    skip: set[int] = set()
    scan = raw.head(12)
    for col in raw.columns:
        values = [cell_str(v) for v in scan[col].tolist()]
        if any(looks_like_total(v) for v in values):
            # Keep a column if it is the actual volume field containing the word
            # only as a rare cell; totals in SSRS usually have "Total" in the header.
            headerish = " ".join(values[:8]).lower()
            if "total" in headerish:
                skip.add(int(col))
    return skip


def _detect_field_id_row(raw: pd.DataFrame, skip_cols: set[int]) -> tuple[Optional[int], dict[int, str]]:
    for idx, row in raw.head(25).iterrows():
        mapping: dict[int, str] = {}
        hits = 0
        for col, val in row.items():
            if int(col) in skip_cols:
                continue
            key = norm_key(val).replace(" ", "_")
            mapped = None
            if key in FIELD_ID_MAP:
                mapped = FIELD_ID_MAP[key]
            else:
                for prefix, role in FIELD_ID_MAP.items():
                    if key.startswith(prefix):
                        mapped = role
                        break
            if mapped:
                mapping[int(col)] = mapped
                hits += 1
        if hits >= 4:
            mapping = _dedupe_roles(raw, idx, mapping)
            return int(idx), mapping
    return None, {}


def _detect_human_header_row(raw: pd.DataFrame, skip_cols: set[int]) -> tuple[Optional[int], dict[int, str]]:
    best: tuple[int, dict[int, str]] | None = None
    best_hits = 0
    for idx, row in raw.head(30).iterrows():
        mapping: dict[int, str] = {}
        hits = 0
        for col, val in row.items():
            if int(col) in skip_cols:
                continue
            key = norm_key(val)
            role = HUMAN_HEADER_MAP.get(key)
            if role:
                mapping[int(col)] = role
                hits += 1
        if hits > best_hits and hits >= 4:
            best_hits = hits
            best = (int(idx), mapping)
    if best:
        return best[0], _dedupe_roles(raw, best[0], best[1])
    return None, {}


def _dedupe_roles(raw: pd.DataFrame, header_row: int, mapping: dict[int, str]) -> dict[int, str]:
    """If labels repeat in A-F and again in the data block, keep the rightmost copy of each role."""
    by_role: dict[str, list[int]] = {}
    for col, role in mapping.items():
        by_role.setdefault(role, []).append(col)
    chosen: dict[int, str] = {}
    data = raw.iloc[header_row + 1 : header_row + 40]
    for role, cols in by_role.items():
        if len(cols) == 1:
            chosen[cols[0]] = role
            continue
        if role == "calendar":
            cols_sorted = sorted(cols)
            chosen[cols_sorted[0]] = "year"
            if len(cols_sorted) > 1:
                chosen[cols_sorted[1]] = "month"
            continue
        scored = []
        for col in cols:
            series = data[col] if col in data.columns else pd.Series(dtype=object)
            nonempty = series.map(cell_str).ne("").sum()
            labelish = series.map(lambda v: norm_key(v) in LABEL_VALUES or norm_key(v).startswith("txt ")).sum()
            storeish = series.map(looks_like_store_id).sum() if role == "store_id" else 0
            numeric = series.map(lambda v: parse_volume(v) is not None).sum() if role == "volume_mt" else 0
            scored.append((storeish + numeric + nonempty - 2 * labelish, col))
        scored.sort()
        chosen[scored[-1][1]] = role
    return chosen


def _project_columns(raw: pd.DataFrame, mapping: dict[int, str], data_start: int) -> pd.DataFrame:
    body = raw.iloc[data_start:].copy()
    frame = pd.DataFrame()
    for col, role in sorted(mapping.items()):
        if col in body.columns:
            if role not in frame.columns:
                frame[role] = body[col].values
            elif role in {"year", "month", "calendar"}:
                # SSRS calendar can occupy two physical columns.
                alt = f"{role}_2" if f"{role}_2" not in frame.columns else f"{role}_extra"
                frame[alt] = body[col].values
    return frame.reset_index(drop=True)


def _positional_project(
    raw: pd.DataFrame, skip_cols: set[int]
) -> tuple[pd.DataFrame, dict[int, str], int]:
    """Find a run of 9 columns: dist, dsr, section, store_id, store_name, sku, year, month, volume."""
    for start_row in range(min(20, len(raw))):
        window = raw.iloc[start_row : start_row + 80]
        for col0 in range(max(0, raw.shape[1] - 8)):
            cols = [c for c in range(col0, min(col0 + 12, raw.shape[1])) if c not in skip_cols]
            if len(cols) < 8:
                continue
            store_hits = window[cols[3]].map(looks_like_store_id).sum() if cols[3] in window.columns else 0
            year_hits = window[cols[6]].map(lambda v: parse_year(v) is not None).sum() if len(cols) > 6 else 0
            month_hits = window[cols[7]].map(lambda v: parse_month(v) is not None).sum() if len(cols) > 7 else 0
            vol_idx = cols[8] if len(cols) > 8 else None
            vol_hits = window[vol_idx].map(lambda v: parse_volume(v) is not None).sum() if vol_idx is not None else 0
            if store_hits >= 5 and year_hits >= 5 and month_hits >= 3 and vol_hits >= 5:
                mapping = {
                    cols[0]: "distributor",
                    cols[1]: "dsr_name",
                    cols[2]: "section",
                    cols[3]: "store_id",
                    cols[4]: "store_name",
                    cols[5]: "sku",
                    cols[6]: "year",
                    cols[7]: "month",
                    vol_idx: "volume_mt",
                }
                projected = _project_columns(raw, mapping, start_row)
                return projected, mapping, start_row
    return pd.DataFrame(), {}, 0


def _normalize_sales(mapped: pd.DataFrame, report: ParseReport) -> pd.DataFrame:
    df = mapped.copy()
    if "calendar" in df.columns:
        years = df["calendar"].map(parse_year)
        months = df["calendar"].map(parse_month)
        if "year" not in df.columns:
            df["year"] = years
        else:
            df["year"] = df["year"].where(df["year"].map(parse_year).notna(), years)
        if "month" not in df.columns:
            df["month"] = months
        else:
            df["month"] = df["month"].where(df["month"].map(parse_month).notna(), months)
    if "year_2" in df.columns:
        df["year"] = df.get("year", pd.Series([None] * len(df)))
        df["year"] = df["year"].where(df["year"].map(parse_year).notna(), df["year_2"])
    if "month_2" in df.columns:
        df["month"] = df.get("month", pd.Series([None] * len(df)))
        df["month"] = df["month"].where(df["month"].map(parse_month).notna(), df["month_2"])

    for col in ("distributor", "dsr_name", "section", "store_id", "store_name", "sku"):
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].map(lambda v: cell_str(v) or None)

    df["year"] = df.get("year", pd.Series([None] * len(df))).map(parse_year)
    df["month"] = df.get("month", pd.Series([None] * len(df))).map(parse_month)
    if "volume_mt" not in df.columns:
        report.warnings.append("No volume column detected")
        df["volume_mt"] = None
    df["volume_mt"] = df["volume_mt"].map(parse_volume)

    # Drop repeating header labels and SSRS total rows.
    def _is_junk(row) -> bool:
        sid = row.get("store_id") or ""
        dist = (row.get("distributor") or "").lower()
        sku = (row.get("sku") or "").lower()
        if not sid or looks_like_total(sid) or looks_like_total(row.get("store_name")):
            return True
        if looks_like_total(row.get("distributor")) or looks_like_total(row.get("dsr_name")):
            return True
        if dist in LABEL_VALUES or sku in LABEL_VALUES:
            return True
        if not looks_like_store_id(sid) and not re.match(r"^[A-Za-z0-9_\-]{4,20}$", sid):
            return True
        if row.get("year") is None or row.get("month") is None:
            return True
        if row.get("volume_mt") is None:
            return True
        if not row.get("sku"):
            return True
        return False

    keep = df.apply(_is_junk, axis=1)
    df = df.loc[~keep].copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["period"] = [period_key(y, m) for y, m in zip(df["year"], df["month"])]
    df["volume_mt"] = df["volume_mt"].astype(float)
    df["store_id"] = df["store_id"].astype(str).str.strip()
    grouped = (
        df.groupby(["store_id", "sku", "period"], as_index=False)
        .agg(
            distributor=("distributor", "last"),
            dsr_name=("dsr_name", "last"),
            section=("section", "last"),
            store_name=("store_name", "last"),
            year=("year", "last"),
            month=("month", "last"),
            volume_mt=("volume_mt", "sum"),
        )
    )
    return grouped[SALES_COLUMNS].reset_index(drop=True)
