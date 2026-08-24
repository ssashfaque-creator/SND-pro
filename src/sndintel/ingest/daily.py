"""Parse Outlet Date Wise Sale (daily store matrix) into shop-month facts.

DSS export: one row per POP, date columns of secondary tons. There is no
distributor on the sheet — POP codes are mapped from the universe / legacy
shop list at ingest. Daily cells are summed to calendar months.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from sndintel.ingest.ssrs import ParseReport, SALES_COLUMNS
from sndintel.io_utils import (
    cell_str,
    looks_like_store_id,
    looks_like_total,
    parse_volume,
    period_key,
    read_raw_table,
)

DAILY_SKU = "ALL"
DATE_WISE_TITLE = "outlet date wise"


def looks_like_outlet_date_wise(path: str | Path) -> bool:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            xl = pd.ExcelFile(path)
            if any(DATE_WISE_TITLE in str(name).lower() for name in xl.sheet_names):
                return True
            peek = pd.read_excel(xl, sheet_name=0, header=None, nrows=4, dtype=object)
            return _header_is_outlet_date_wise(peek)
        peek = read_raw_table(path).head(5)
        return _header_is_outlet_date_wise(peek)
    except Exception:
        return False


def parse_outlet_date_wise(path: str | Path) -> tuple[pd.DataFrame, ParseReport]:
    path = Path(path)
    raw, params = _load_date_wise(path)
    report = ParseReport(
        strategy="outlet_date_wise",
        source_file=str(path),
        n_raw_rows=max(0, len(raw) - 2),
        n_clean_rows=0,
        header_row=0,
        data_start_row=2,
        params=params,
        column_map={"0": "store_id", "1": "store_name", "dates": "volume_mt"},
    )
    if raw is None or raw.empty:
        report.warnings.append("Outlet Date Wise file was empty")
        return pd.DataFrame(columns=SALES_COLUMNS), report

    date_row, pop_row, id_col, name_col, date_cols = _locate_layout(raw)
    report.header_row = pop_row
    report.data_start_row = pop_row + 1
    if not date_cols:
        raise ValueError(
            f"Could not find date columns in {path.name}. "
            "Expected Outlet Date Wise Sale with POP Code and daily Secondary Sales UOM."
        )

    body = raw.iloc[pop_row + 1 :].copy()
    store_id = body[id_col].map(cell_str)
    store_name = body[name_col].map(cell_str) if name_col is not None else pd.Series("", index=body.index)
    keep = store_id.map(looks_like_store_id) & ~store_id.map(looks_like_total) & ~store_name.map(looks_like_total)
    body = body.loc[keep]
    store_id = store_id.loc[keep]
    store_name = store_name.loc[keep]
    report.daily_dates = sorted(
        {
            (d.isoformat() if hasattr(d, "isoformat") else str(d)[:10])
            for d in date_cols.values()
            if d is not None
        }
    )
    report.daily_store_ids = store_id.astype(str).str.strip().drop_duplicates().tolist()
    if body.empty:
        report.warnings.append("No POP codes in the Outlet Date Wise sheet")
        return pd.DataFrame(columns=SALES_COLUMNS), report

    vol = body.loc[:, list(date_cols)].apply(pd.to_numeric, errors="coerce")
    vol.columns = pd.Index([date_cols[c] for c in vol.columns])
    vol.index = pd.MultiIndex.from_arrays([store_id.to_numpy(), store_name.to_numpy()], names=["store_id", "store_name"])
    stacked = vol.stack()
    stacked = stacked[stacked.notna() & (stacked > 0)]
    if stacked.empty:
        report.warnings.append("Outlet Date Wise had POPs but no positive daily volume")
        return pd.DataFrame(columns=SALES_COLUMNS), report

    long = stacked.reset_index()
    long.columns = ["store_id", "store_name", "sale_date", "volume_mt"]
    long["sale_date"] = pd.to_datetime(long["sale_date"], errors="coerce")
    long = long.loc[long["sale_date"].notna()].copy()
    long["year"] = long["sale_date"].dt.year.astype(int)
    long["month"] = long["sale_date"].dt.month.astype(int)
    long["period"] = [period_key(y, m) for y, m in zip(long["year"], long["month"])]
    long["store_id"] = long["store_id"].astype(str).str.strip()
    long["store_name"] = long["store_name"].fillna("").map(cell_str)
    long["day"] = long["sale_date"].dt.day.astype(int)
    monthly = (
        long.groupby(["store_id", "period"], as_index=False)
        .agg(
            store_name=("store_name", "last"),
            year=("year", "last"),
            month=("month", "last"),
            volume_mt=("volume_mt", "sum"),
        )
    )
    monthly["sku"] = DAILY_SKU
    monthly["distributor"] = None
    monthly["dsr_name"] = None
    monthly["section"] = None
    monthly = monthly[SALES_COLUMNS]

    if not params.get("execution_date") and not long.empty:
        last = long["sale_date"].max()
        params["execution_date"] = last.strftime("%d/%m/%Y")
        params["execution_source"] = "last_billed_day"
        report.params = params

    n_dates = len({d.date() if hasattr(d, "date") else d for d in date_cols.values()})
    report.params["n_date_columns"] = str(n_dates)
    report.params["report_name"] = "Outlet Date Wise Sale"
    report.n_clean_rows = len(monthly)
    report.daily = long[
        ["store_id", "store_name", "sale_date", "year", "month", "day", "period", "volume_mt"]
    ].copy()
    return monthly, report


def overlay_store_attrs(facts: pd.DataFrame, stores: pd.DataFrame | None) -> pd.DataFrame:
    """Fill distributor / DSR / section / name from the shop book. Universe rows win."""
    if facts is None or facts.empty or stores is None or stores.empty:
        return facts if facts is not None else pd.DataFrame()
    if "store_id" not in stores.columns:
        return facts
    st = stores.copy()
    st["store_id"] = st["store_id"].astype(str).str.strip()
    if "in_universe" in st.columns:
        st["_live"] = pd.to_numeric(st["in_universe"], errors="coerce").fillna(0)
        st = st.sort_values("_live", ascending=False)
    st = st.drop_duplicates("store_id", keep="first")
    cols = [c for c in ("store_id", "distributor", "dsr_name", "section", "store_name", "city") if c in st.columns]
    geo = st[cols]
    out = facts.copy()
    out["store_id"] = out["store_id"].astype(str).str.strip()
    merged = out.merge(geo, on="store_id", how="left", suffixes=("", "_m"))
    for col in ("distributor", "dsr_name", "section", "store_name", "city"):
        master = f"{col}_m"
        if master in merged.columns:
            merged[col] = merged[master].combine_first(merged[col]) if col in merged.columns else merged[master]
            merged = merged.drop(columns=[master])
    return merged


def _header_is_outlet_date_wise(raw: pd.DataFrame) -> bool:
    if raw is None or raw.empty:
        return False
    blob = " ".join(cell_str(v) for v in raw.head(4).to_numpy().ravel()[:120]).lower()
    if DATE_WISE_TITLE in blob:
        return True
    cells = [cell_str(v) for v in raw.head(3).to_numpy().ravel()]
    has_pop = any(c.lower().replace(" ", "") in {"popcode", "pop code"} or c.lower() == "pop code" for c in cells)
    has_date = any(_as_date(v) is not None for v in raw.head(2).to_numpy().ravel())
    return has_pop and has_date


def _load_date_wise(path: Path) -> tuple[pd.DataFrame, dict]:
    params: dict[str, str] = {}
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(path)
        sheet = 0
        for name in xl.sheet_names:
            if DATE_WISE_TITLE in str(name).lower():
                sheet = name
                break
        raw = pd.read_excel(xl, sheet_name=sheet, header=None, dtype=object)
        for name in xl.sheet_names:
            if "filter" in str(name).lower() or "selected" in str(name).lower():
                filt = pd.read_excel(xl, sheet_name=name, header=None, dtype=object)
                params.update(_params_from_frame(filt))
        if not params:
            params.update(_params_from_frame(raw.head(8)))
        raw = raw.dropna(how="all", axis=0).reset_index(drop=True)
        return raw, params
    raw = read_raw_table(path)
    params.update(_params_from_frame(raw.head(8)))
    return raw, params


def _params_from_frame(raw: pd.DataFrame) -> dict[str, str]:
    import re

    params: dict[str, str] = {}
    blob = " ".join(cell_str(v) for v in raw.to_numpy().ravel()[:400])
    exec_m = re.search(
        r"Execution Date(?:\s*&\s*Time)?\s*:\s*(\d{1,2}/\d{1,2}/\d{4})(?:\s+(\d{1,2}:\d{2}:\d{2}))?",
        blob,
        flags=re.I,
    )
    if exec_m:
        params["execution_date"] = exec_m.group(1)
        if exec_m.group(2):
            params["execution_time"] = exec_m.group(2)
    uom = re.search(r"UOM\s*:\s*([A-Za-z]+)", blob, flags=re.I)
    if uom:
        params["uom"] = uom.group(1).strip()
    year = re.search(r"Calendar Year\s*:\s*(\d{4})", blob, flags=re.I)
    if year:
        params["calendar_year"] = year.group(1)
    if DATE_WISE_TITLE in blob.lower():
        params["report_name"] = "Outlet Date Wise Sale"
    return params


def _locate_layout(
    raw: pd.DataFrame,
) -> tuple[int, int, int, Optional[int], dict[int, date]]:
    pop_row = 0
    id_col = 0
    name_col: Optional[int] = 1 if raw.shape[1] > 1 else None
    for idx, row in raw.head(8).iterrows():
        for col, val in row.items():
            key = cell_str(val).lower().replace(" ", "")
            if key in {"popcode"} or cell_str(val).lower() == "pop code":
                pop_row = int(idx)
                id_col = int(col)
            if key in {"popname"} or cell_str(val).lower() == "pop name":
                name_col = int(col)
    date_row = pop_row - 1 if pop_row > 0 else 0
    date_cols: dict[int, date] = {}
    for scan in (date_row, pop_row, 0):
        if scan < 0 or scan >= len(raw):
            continue
        found: dict[int, date] = {}
        for col, val in raw.iloc[scan].items():
            dt = _as_date(val)
            if dt is not None:
                found[int(col)] = dt
        if len(found) >= 3:
            date_cols = found
            date_row = scan
            break
        if len(found) > len(date_cols):
            date_cols = found
            date_row = scan
    return date_row, pop_row, id_col, name_col, date_cols


def _as_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()
    text = cell_str(value)
    if not text:
        return None
    text = text.replace(" 00:00:00", "")[:10]
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    serial = parse_volume(value)
    if serial is not None and 20000 < serial < 60000:
        try:
            return (pd.Timestamp("1899-12-30") + pd.to_timedelta(int(serial), unit="D")).date()
        except (ValueError, OverflowError):
            return None
    return None
