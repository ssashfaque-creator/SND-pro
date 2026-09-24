"""Shop-wise billed lists to compare the warehouse with a Shop SKU Wise extract."""

from __future__ import annotations

import pandas as pd


def period_totals(shop_month: pd.DataFrame) -> pd.DataFrame:
    """Billed MT by calendar month — compare each row to that month’s Grand Total on the extract.

    ``volume_mt`` is everything on file for the month (this is the number that
    must tie to the extract). ``off_master_shops`` / ``off_master_mt`` show how
    much of it came from POPs that are not on the live universe list.
    """
    cols = ["period", "shops", "volume_mt", "off_master_shops", "off_master_mt"]
    if shop_month is None or shop_month.empty:
        return pd.DataFrame(columns=cols)
    work = shop_month.copy()
    work["period"] = work["period"].astype(str)
    work["volume_mt"] = pd.to_numeric(work.get("volume_mt"), errors="coerce").fillna(0.0)
    work = work.loc[work["volume_mt"] > 0]
    if "in_universe" in work.columns:
        off = pd.to_numeric(work["in_universe"], errors="coerce").fillna(1).astype(int) == 0
    else:
        off = pd.Series(False, index=work.index)
    work = work.assign(_off_mt=work["volume_mt"].where(off, 0.0), _off_shop=off)
    out = (
        work.groupby("period", as_index=False)
        .agg(
            shops=("store_id", "nunique"),
            volume_mt=("volume_mt", "sum"),
            off_master_shops=("_off_shop", "sum"),
            off_master_mt=("_off_mt", "sum"),
        )
        .sort_values("period")
    )
    out["off_master_shops"] = out["off_master_shops"].astype(int)
    return out.reset_index(drop=True)


def off_master_shops(shop_month: pd.DataFrame, period: str | None = None) -> pd.DataFrame:
    """Billed POPs that are not on the live universe list, biggest first."""
    cols = ["store_id", "store_name", "distributor", "dsr_name", "period", "volume_mt"]
    if shop_month is None or shop_month.empty or "in_universe" not in shop_month.columns:
        return pd.DataFrame(columns=cols)
    work = shop_month.copy()
    work["period"] = work["period"].astype(str)
    work["volume_mt"] = pd.to_numeric(work.get("volume_mt"), errors="coerce").fillna(0.0)
    off = pd.to_numeric(work["in_universe"], errors="coerce").fillna(1).astype(int) == 0
    part = work.loc[off & (work["volume_mt"] > 0)]
    if period:
        part = part.loc[part["period"] == str(period)]
    keep = [c for c in cols if c in part.columns]
    return part[keep].sort_values("volume_mt", ascending=False).reset_index(drop=True)


def distributor_shop_sales(
    shop_month: pd.DataFrame,
    distributor: str,
    period: str,
) -> pd.DataFrame:
    """Every billed shop for one distributor and month.

    Sum ``volume_mt`` and compare to that distributor’s total on the Shop SKU
    Wise row for the same year and month (the figure next to
    ``Agha Traders (Quetta) Total`` on a 2026 July line, for example).
    """
    cols = ["store_id", "store_name", "dsr_name", "section", "sku_count", "volume_mt"]
    if shop_month is None or shop_month.empty or not distributor or not period:
        return pd.DataFrame(columns=cols)
    work = shop_month.copy()
    work["period"] = work["period"].astype(str)
    if "distributor" not in work.columns:
        return pd.DataFrame(columns=cols)
    needle = str(distributor).strip().lower()
    dist = work["distributor"].fillna("").astype(str)
    matched = dist.str.lower() == needle
    if not matched.any():
        matched = dist.str.lower().str.contains(needle, regex=False)
    part = work.loc[matched & (work["period"] == str(period))].copy()
    if part.empty:
        return pd.DataFrame(columns=cols)
    part["volume_mt"] = pd.to_numeric(part.get("volume_mt"), errors="coerce").fillna(0.0)
    keep = [c for c in cols if c in part.columns]
    out = part[keep].sort_values("volume_mt", ascending=False)
    return out.loc[out["volume_mt"] > 0].reset_index(drop=True)


def match_distributors(shop_month: pd.DataFrame) -> list[str]:
    if shop_month is None or shop_month.empty or "distributor" not in shop_month.columns:
        return []
    names = shop_month["distributor"].fillna("").astype(str).str.strip()
    return sorted({n for n in names.tolist() if n and n.lower() != "nan"})
