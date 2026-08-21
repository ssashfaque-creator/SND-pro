"""Parse the live universe shop list.

Column order from the business file:

    distributor, area/city, section, DSR, POP code, POP name
    (optional TOTAL UNIVERSE OUTLETS).

Excel exports merge distributor / city / section / DSR cells down the group.
Those blanks are forward-filled. POP code is the only stable shop identity —
names, DSR, distributor, and city can all change; the current row is the
assignment the briefing uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sndintel.ingest.shops import ShopParseReport, _normalize_shops
from sndintel.io_utils import cell_str, looks_like_store_id, norm_key, read_raw_table

HEADER_ALIASES = {
    "distributor": "distributor",
    "distributor name": "distributor",
    "txt cdistributor name": "distributor",
    "area": "city",
    "city": "city",
    "town": "city",
    "section": "section",
    "section long description": "section",
    "section long": "section",
    "txt csection long description": "section",
    "beat": "section",
    "dsr": "dsr_name",
    "dsr name": "dsr_name",
    "salesperson": "dsr_name",
    "txt cdsr name": "dsr_name",
    "pop code": "store_id",
    "popcode": "store_id",
    "store id": "store_id",
    "store code": "store_id",
    "outlet code": "store_id",
    "txt cpop code": "store_id",
    "pop name": "store_name",
    "store name": "store_name",
    "outlet name": "store_name",
    "txt cpop name": "store_name",
    "zone": "zone",
    "division": "zone",
}

POSITIONAL_ROLES = ["distributor", "city", "section", "dsr_name", "store_id", "store_name"]


def parse_universe(path: str | Path) -> tuple[pd.DataFrame, ShopParseReport]:
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
        report.strategy = "universe_headers"
        report.column_map = {v: k for k, v in mapping.items()}
        body = raw.iloc[header_row + 1 :].copy()
        projected = pd.DataFrame({role: body[col].values for col, role in mapping.items() if col in body.columns})
    else:
        report.strategy = "universe_positional"
        projected, mapping = _positional(raw)
        report.column_map = mapping
        if projected.empty:
            raise ValueError(
                f"Could not detect universe columns in {path.name}. "
                "Expected distributor, area/city, section, DSR, POP code, POP name."
            )
    projected = _ffill_hierarchy(projected)
    clean = _normalize_shops(projected, report)
    report.n_clean_rows = len(clean)
    if report.n_clean_rows == 0:
        report.warnings.append("Universe file had headers but no POP codes")
    return clean, report


def fill_zone_from_legacy(universe: pd.DataFrame, legacy: pd.DataFrame | None) -> pd.DataFrame:
    """Universe file has no zone. Borrow it from the old master or prior stores, by POP then by city."""
    out = universe.copy()
    if "zone" not in out.columns:
        out["zone"] = None
    if legacy is None or legacy.empty or "store_id" not in legacy.columns:
        return out
    leg = legacy.copy()
    leg["store_id"] = leg["store_id"].astype(str).str.strip()
    if "zone" in leg.columns:
        by_id = (
            leg.dropna(subset=["zone"])
            .drop_duplicates("store_id")
            .set_index("store_id")["zone"]
        )
        out["zone"] = out["store_id"].map(by_id).combine_first(out["zone"])
    if "city" in leg.columns and "zone" in leg.columns:
        city_zone = (
            leg.dropna(subset=["city", "zone"])
            .groupby(leg["city"].astype(str).str.strip())["zone"]
            .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        )
        missing = out["zone"].isna() | (out["zone"].astype(str).str.strip() == "")
        if missing.any() and "city" in out.columns:
            out.loc[missing, "zone"] = out.loc[missing, "city"].astype(str).str.strip().map(city_zone)
    return out


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


def _positional(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    for start in range(min(6, len(raw))):
        window = raw.iloc[start : start + 80]
        for id_col in range(min(6, raw.shape[1])):
            hits = window[id_col].map(looks_like_store_id).sum()
            if hits < 8:
                continue
            # Prefer the documented 6-col layout ending at POP code.
            if id_col >= 4:
                col0 = id_col - 4
            else:
                col0 = 0
            mapping = {}
            cols_out = {}
            for offset, role in enumerate(POSITIONAL_ROLES):
                col = col0 + offset
                if col < raw.shape[1]:
                    mapping[role] = col
                    cols_out[role] = raw.iloc[start:, col].values
            return pd.DataFrame(cols_out), mapping
    return pd.DataFrame(), {}


def _ffill_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("distributor", "city", "zone", "section", "dsr_name"):
        if col not in out.columns:
            continue
        series = out[col].map(lambda v: cell_str(v) or None)
        out[col] = series.ffill()
    return out
