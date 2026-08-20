"""Feature engineering on shop-month secondary sales."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sndintel.io_utils import period_key


def rebuild_shop_month(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    if sales.empty:
        return pd.DataFrame()
    facts = sales.copy()
    agg = (
        facts.groupby(["store_id", "period"], as_index=False)
        .agg(
            year=("year", "last"),
            month=("month", "last"),
            volume_mt=("volume_mt", "sum"),
            sku_count=("sku", "nunique"),
            distributor=("distributor", "last"),
            dsr_name=("dsr_name", "last"),
            section=("section", "last"),
            store_name=("store_name", "last"),
        )
    )
    agg["billed"] = (agg["volume_mt"] > 0).astype(int)
    if stores is not None and not stores.empty:
        geo = stores[
            ["store_id", "store_name", "distributor", "dsr_name", "section", "zone", "city"]
        ].drop_duplicates("store_id")
        agg = agg.merge(geo, on="store_id", how="left", suffixes=("", "_m"))
        for col in ("store_name", "distributor", "dsr_name", "section"):
            master_col = f"{col}_m"
            if master_col in agg.columns:
                agg[col] = agg[master_col].combine_first(agg[col])
                agg = agg.drop(columns=[master_col])
        if "zone" not in agg.columns:
            agg["zone"] = None
        if "city" not in agg.columns:
            agg["city"] = None
    else:
        agg["zone"] = None
        agg["city"] = None
    return agg


def add_calendar_panel(shop_month: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Fill missing shop-months with zeros so recency and strike-rate are honest."""
    if shop_month.empty:
        return shop_month
    attr_cols = ["distributor", "dsr_name", "section", "store_name", "zone", "city"]
    periods = sorted(shop_month["period"].unique())
    store_ids = set(shop_month["store_id"])
    if stores is not None and not stores.empty:
        store_ids |= set(stores["store_id"])
    grid = pd.MultiIndex.from_product(
        [sorted(store_ids), periods], names=["store_id", "period"]
    ).to_frame(index=False)
    value_cols = ["store_id", "period", "volume_mt", "sku_count"]
    merged = grid.merge(shop_month[value_cols], on=["store_id", "period"], how="left")
    merged["volume_mt"] = merged["volume_mt"].fillna(0.0)
    merged["sku_count"] = merged["sku_count"].fillna(0).astype(int)
    merged["billed"] = (merged["volume_mt"] > 0).astype(int)
    merged["year"] = merged["period"].str.slice(0, 4).astype(int)
    merged["month"] = merged["period"].str.slice(5, 7).astype(int)

    from_sales = (
        shop_month.sort_values("period")
        .groupby("store_id")[[c for c in attr_cols if c in shop_month.columns]]
        .last()
        .reset_index()
    )
    if stores is not None and not stores.empty:
        geo_cols = ["store_id"] + [c for c in attr_cols if c in stores.columns]
        attr = stores[geo_cols].drop_duplicates("store_id").merge(
            from_sales, on="store_id", how="outer", suffixes=("", "_s")
        )
        for col in attr_cols:
            other = f"{col}_s"
            if col not in attr.columns and other in attr.columns:
                attr[col] = attr[other]
            elif other in attr.columns:
                attr[col] = attr[col].combine_first(attr[other])
            if other in attr.columns:
                attr = attr.drop(columns=[other])
    else:
        attr = from_sales
    return merged.merge(attr, on="store_id", how="left")


def build_features(shop_month: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    if shop_month.empty:
        return pd.DataFrame()
    df = shop_month.sort_values(["store_id", "period"]).copy()
    g = df.groupby("store_id", group_keys=False)
    df["lag_1"] = g["volume_mt"].shift(1)
    df["lag_2"] = g["volume_mt"].shift(2)
    df["lag_3"] = g["volume_mt"].shift(3)
    df["lag_12"] = g["volume_mt"].shift(12)
    df["roll_mean_3"] = g["volume_mt"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["roll_mean_6"] = g["volume_mt"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    df["roll_median_6"] = g["volume_mt"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).median())
    df["cv_6m"] = g["volume_mt"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=3).std() / s.shift(1).rolling(6, min_periods=3).mean().replace(0, np.nan)
    )
    own_mean = g["volume_mt"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
    own_std = g["volume_mt"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).std()).replace(0, np.nan)
    df["zscore_own"] = (df["volume_mt"] - own_mean) / own_std
    df["mom_pct"] = np.where(df["lag_1"] > 0, (df["volume_mt"] - df["lag_1"]) / df["lag_1"] * 100, np.nan)
    df["yoy_pct"] = np.where(df["lag_12"] > 0, (df["volume_mt"] - df["lag_12"]) / df["lag_12"] * 100, np.nan)
    df["billed_rate_12"] = g["billed"].transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())

    def _recency(s: pd.Series) -> pd.Series:
        last = -1
        out = []
        for i, billed in enumerate(s.astype(int).tolist()):
            if billed:
                last = i
                out.append(0)
            else:
                out.append((i - last) if last >= 0 else 99)
        return pd.Series(out, index=s.index)

    df["recency_months"] = g["billed"].transform(_recency)

    section_period = df.groupby(["section", "period"])["volume_mt"].transform("mean")
    city_period = df.groupby(["city", "period"])["volume_mt"].transform("mean")
    df["vs_section_pct"] = np.where(section_period > 0, (df["volume_mt"] / section_period - 1) * 100, np.nan)
    df["vs_city_pct"] = np.where(city_period > 0, (df["volume_mt"] / city_period - 1) * 100, np.nan)

    top_share = _top_sku_share(sales)
    df = df.merge(top_share, on=["store_id", "period"], how="left")
    df["top_sku_share"] = df["top_sku_share"].fillna(0)
    keep = [
        "store_id",
        "period",
        "volume_mt",
        "sku_count",
        "roll_mean_3",
        "roll_mean_6",
        "roll_median_6",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "mom_pct",
        "yoy_pct",
        "zscore_own",
        "vs_section_pct",
        "vs_city_pct",
        "cv_6m",
        "recency_months",
        "billed_rate_12",
        "top_sku_share",
    ]
    return df[keep]


def _top_sku_share(sales: pd.DataFrame) -> pd.DataFrame:
    if sales is None or sales.empty:
        return pd.DataFrame(columns=["store_id", "period", "top_sku_share"])
    totals = sales.groupby(["store_id", "period"])["volume_mt"].transform("sum").replace(0, np.nan)
    tmp = sales.copy()
    tmp["share"] = tmp["volume_mt"] / totals
    top = tmp.groupby(["store_id", "period"], as_index=False)["share"].max()
    top = top.rename(columns={"share": "top_sku_share"})
    return top


def latest_period(df: pd.DataFrame, col: str = "period") -> str | None:
    if df is None or df.empty or col not in df.columns:
        return None
    return str(sorted(df[col].dropna().unique())[-1])


def previous_period(period: str) -> str:
    year = int(period[:4])
    month = int(period[5:7])
    month -= 1
    if month == 0:
        month = 12
        year -= 1
    return period_key(year, month)
