"""Parse the outlet universe / shop master list.

Expected (but not required) order from the business file:

    distributor, DSR name, store id, store name,
    four categorization columns, zone, city, section, then two unused columns.

Column detection is heuristic so extra/missing columns do not break ingest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sndintel.io_utils import (
    cell_str,
    looks_like_store_id,
    looks_like_total,
    norm_key,
    read_raw_table,
)

STORE_COLUMNS = [
    "store_id",
    "store_name",
    "distributor",
    "dsr_name",
    "zone",
    "city",
    "section",
    "category_1",
    "category_2",
    "category_3",
    "category_4",
]

HEADER_ALIASES = {
    "store id": "store_id",
    "store code": "store_id",
    "pop code": "store_id",
    "outlet code": "store_id",
    "outlet id": "store_id",
    "store name": "store_name",
    "pop name": "store_name",
    "outlet name": "store_name",
    "distributor": "distributor",
    "dsr name": "dsr_name",
    "dsr": "dsr_name",
    "salesperson": "dsr_name",
    "zone": "zone",
    "division": "zone",
    "region": "zone",
    "city": "city",
    "town": "city",
    "section": "section",
    "area": "section",
    "beat": "section",
    "channel": "category_1",
    "channel type": "category_1",
    "outlet type": "category_1",
    "category": "category_1",
    "class": "category_2",
    "channel class": "category_2",
    "locality": "category_3",
    "sub channel": "category_3",
    "tier": "category_4",
}


@dataclass
class ShopParseReport:
    strategy: str
    source_file: str
    n_raw_rows: int
    n_clean_rows: int
    column_map: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def parse_shop_master(path: str | Path) -> tuple[pd.DataFrame, ShopParseReport]:
    path = Path(path)
    raw = read_raw_table(path)
    report = ShopParseReport(
        strategy="unknown",
        source_file=str(path),
        n_raw_rows=len(raw),
        n_clean_rows=0,
    )
    header_row, mapping = _detect_headers(raw)
    if mapping and "store_id" in mapping.values():
        report.strategy = "headers"
        report.column_map = {v: k for k, v in mapping.items()}
        body = raw.iloc[header_row + 1 :].copy()
        projected = pd.DataFrame({role: body[col].values for col, role in mapping.items() if col in body.columns})
    else:
        report.strategy = "positional"
        projected, mapping = _positional_shops(raw)
        report.column_map = mapping
        if projected.empty:
            raise ValueError(f"Could not detect shop master columns in {path.name}")

    clean = _normalize_shops(projected, report)
    report.n_clean_rows = len(clean)
    return clean, report


def _detect_headers(raw: pd.DataFrame) -> tuple[int, dict[int, str]]:
    best_row, best_map, best_hits = 0, {}, 0
    for idx, row in raw.head(15).iterrows():
        mapping: dict[int, str] = {}
        hits = 0
        for col, val in row.items():
            role = HEADER_ALIASES.get(norm_key(val))
            if role and role not in mapping.values():
                mapping[int(col)] = role
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_row, best_map = int(idx), mapping
    if best_hits >= 3:
        return best_row, best_map
    return 0, {}


def _positional_shops(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """distributor, dsr, store_id, store_name, 4 cats, zone, city, section, junk, junk."""
    for start in range(min(8, len(raw))):
        window = raw.iloc[start : start + 60]
        for col0 in range(max(0, raw.shape[1] - 6)):
            if col0 + 3 >= raw.shape[1]:
                continue
            id_col = col0 + 2
            hits = window[id_col].map(looks_like_store_id).sum()
            if hits < 8:
                continue
            # Need at least 11 columns for the documented layout; tolerate shorter.
            names = [
                "distributor",
                "dsr_name",
                "store_id",
                "store_name",
                "category_1",
                "category_2",
                "category_3",
                "category_4",
                "zone",
                "city",
                "section",
            ]
            mapping = {}
            cols_out = {}
            for offset, role in enumerate(names):
                col = col0 + offset
                if col < raw.shape[1]:
                    mapping[role] = col
                    cols_out[role] = raw.iloc[start:, col].values
            projected = pd.DataFrame(cols_out)
            return projected, mapping
    return pd.DataFrame(), {}


def _normalize_shops(df: pd.DataFrame, report: ShopParseReport) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in STORE_COLUMNS:
        if col in df.columns:
            out[col] = df[col].map(lambda v: cell_str(v) or None)
        else:
            out[col] = None
    out = out[out["store_id"].notna()].copy()
    out = out[~out["store_id"].map(looks_like_total)]
    out = out[~out["store_name"].fillna("").map(looks_like_total)]
    # Filter obvious header echoes
    out = out[out["store_id"].str.lower() != "store id"]
    out = out[out["store_id"].map(lambda v: looks_like_store_id(v) or (isinstance(v, str) and len(v) >= 4))]
    out["store_id"] = out["store_id"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["store_id"], keep="last")
    missing_geo = out["city"].isna().sum() + out["zone"].isna().sum()
    if missing_geo:
        report.warnings.append(f"{missing_geo} shops missing zone/city after parse")
    return out.reset_index(drop=True)
