"""Parse SSRS Shop Visited Calls (MTD visit history).

Same chrome as the Shop SKU Wise extract: footer field-ids, parameter row,
then a tablix with repeated labels in columns A–F and values to the right.
POP code joins to the universe. Visit count is MTD_VISITED_CALLS.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from sndintel.ingest.ssrs import ParseReport, _dedupe_roles, _extract_params, _hierarchy_total_columns
from sndintel.io_utils import (
    MONTH_MAP,
    cell_str,
    looks_like_store_id,
    looks_like_total,
    norm_key,
    parse_volume,
    period_key,
    read_raw_table,
)

VISIT_COLUMNS = [
    "store_id",
    "store_name",
    "distributor",
    "dsr_name",
    "city",
    "section",
    "visits",
    "period",
]

FIELD_ID_MAP = {
    "txt_cdistributor_name": "distributor",
    "txt_cdistrib": "distributor",
    "txt_carea": "city",
    "txt_cdsr_name": "dsr_name",
    "txt_cdsr_na": "dsr_name",
    "txt_cdsr": "dsr_name",
    "txt_csection_long_description": "section",
    "txt_csection": "section",
    "txt_csectio": "section",
    "txt_cpop_code": "store_id",
    "txt_cpop_co": "store_id",
    "txt_cpop_c": "store_id",
    "txt_cpop_name": "store_name",
    "txt_cpop_na": "store_name",
    "uval_mtd_visited_calls": "visits",
    "uval_mtd_visited": "visits",
}

HUMAN_HEADER_MAP = {
    "distributor": "distributor",
    "distributor name": "distributor",
    "area": "city",
    "city": "city",
    "town": "city",
    "dsr name": "dsr_name",
    "dsr": "dsr_name",
    "section long description": "section",
    "section long": "section",
    "section": "section",
    "pop code": "store_id",
    "popcode": "store_id",
    "store id": "store_id",
    "pop name": "store_name",
    "store name": "store_name",
    "mtd visited calls": "visits",
    "mtd_visited_calls": "visits",
    "visited calls": "visits",
    "visits": "visits",
}


def parse_visit_calls(path: str | Path, period: str | None = None) -> tuple[pd.DataFrame, ParseReport]:
    path = Path(path)
    raw = read_raw_table(path)
    params = _extract_params(raw)
    params.update(_extract_calendar(raw))
    report = ParseReport(
        strategy="unknown",
        source_file=str(path),
        n_raw_rows=len(raw),
        n_clean_rows=0,
        header_row=None,
        data_start_row=None,
        params=params,
    )
    skip_cols = _hierarchy_total_columns(raw)
    field_row, field_map = _detect_field_row(raw, skip_cols)
    human_row, human_map = _detect_human_row(raw, skip_cols)
    if field_map:
        report.strategy = "ssrs_field_ids"
        report.header_row = field_row
        report.column_map = field_map
        data_start = field_row + 1
        if human_row is not None and human_row > field_row:
            store_cols = [c for c, role in field_map.items() if role == "store_id"]
            looks_like_data = False
            if store_cols and human_row < len(raw):
                looks_like_data = looks_like_store_id(raw.iloc[human_row][store_cols[0]])
            data_start = human_row if looks_like_data else human_row + 1
        mapped = _project(raw, field_map, data_start)
    elif human_map:
        report.strategy = "human_headers"
        report.header_row = human_row
        report.column_map = human_map
        mapped = _project(raw, human_map, human_row + 1)
    else:
        raise ValueError(
            f"Could not detect visit-call columns in {path.name}. "
            "Expected POP code and MTD visited calls."
        )
    inferred = period or _period_from_params(params)
    clean = _normalize(mapped, inferred, report)
    report.n_clean_rows = len(clean)
    if not inferred:
        report.warnings.append("Could not read Calendar Month/Year from the visit file; pass the sales period.")
    if clean.empty:
        report.warnings.append("Visit file had headers but no usable POP rows")
    return clean, report


def _extract_calendar(raw: pd.DataFrame) -> dict[str, str]:
    blob = " ".join(cell_str(v) for v in raw.head(8).to_numpy().ravel()[:240])
    out: dict[str, str] = {}
    m = re.search(r"Calendar Month\s*:\s*([A-Za-z]+)", blob, flags=re.I)
    if m:
        out["calendar_month"] = m.group(1).strip()
    y = re.search(r"Calendar Year\s*:\s*(\d{4})", blob, flags=re.I)
    if y:
        out["calendar_year"] = y.group(1)
    fd = re.search(r"FullDate\s*:\s*(\d{4}-\d{2}-\d{2})", blob, flags=re.I)
    if fd:
        out["full_date"] = fd.group(1)
    if "Shops Visit" in blob or "Visit Call" in blob:
        out["report_name"] = "Shops Visit Calls"
    return out


def _period_from_params(params: dict) -> Optional[str]:
    full = params.get("full_date") or ""
    if re.match(r"\d{4}-\d{2}-\d{2}$", full):
        return full[:7]
    year = None
    month = None
    try:
        year = int(params.get("calendar_year") or "")
    except (TypeError, ValueError):
        year = None
    month = MONTH_MAP.get(str(params.get("calendar_month") or "").lower().replace(".", ""))
    if year and month:
        return period_key(year, month)
    return None


def _detect_field_row(raw: pd.DataFrame, skip_cols: set[int]) -> tuple[Optional[int], dict[int, str]]:
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
        if hits >= 4 and "store_id" in mapping.values():
            return int(idx), _dedupe_roles(raw, idx, mapping)
    return None, {}


def _detect_human_row(raw: pd.DataFrame, skip_cols: set[int]) -> tuple[Optional[int], dict[int, str]]:
    best: tuple[int, dict[int, str]] | None = None
    best_hits = 0
    for idx, row in raw.head(30).iterrows():
        mapping: dict[int, str] = {}
        hits = 0
        for col, val in row.items():
            if int(col) in skip_cols:
                continue
            role = HUMAN_HEADER_MAP.get(norm_key(val))
            if role and role not in mapping.values():
                mapping[int(col)] = role
                hits += 1
        if hits > best_hits and hits >= 4:
            best_hits = hits
            best = (int(idx), mapping)
    if best:
        return best[0], _dedupe_roles(raw, best[0], best[1])
    return None, {}


def _project(raw: pd.DataFrame, mapping: dict[int, str], data_start: int) -> pd.DataFrame:
    body = raw.iloc[data_start:].copy()
    frame = pd.DataFrame()
    for col, role in sorted(mapping.items()):
        if col in body.columns and role not in frame.columns:
            frame[role] = body[col].values
    return frame.reset_index(drop=True)


def _normalize(df: pd.DataFrame, period: Optional[str], report: ParseReport) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in VISIT_COLUMNS:
        if col == "period":
            continue
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = None
    out["store_id"] = out["store_id"].map(lambda v: cell_str(v) or None)
    out = out[out["store_id"].notna()].copy()
    out = out[~out["store_id"].map(looks_like_total)]
    out = out[out["store_id"].map(looks_like_store_id)]
    out["store_id"] = out["store_id"].astype(str).str.strip()
    for col in ("store_name", "distributor", "dsr_name", "city", "section"):
        out[col] = out[col].map(lambda v: cell_str(v) or None)
        out = out[~out[col].fillna("").map(looks_like_total)]
    visits = out["visits"].map(parse_volume).fillna(0.0)
    out["visits"] = visits.clip(lower=0).astype(int)
    out["period"] = period
    if out.empty:
        return out.reset_index(drop=True)
    g = out.groupby("store_id", as_index=False).agg(
        store_name=("store_name", "last"),
        distributor=("distributor", "last"),
        dsr_name=("dsr_name", "last"),
        city=("city", "last"),
        section=("section", "last"),
        visits=("visits", "sum"),
        period=("period", "last"),
    )
    dropped = int(len(out) - len(g))
    if dropped:
        report.warnings.append(f"Collapsed {dropped} duplicate visit rows (same POP)")
    return g.reset_index(drop=True)
