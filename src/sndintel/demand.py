"""Demand-driven Ask: shop depletion cycle, cold start, lapse cut-off, 4-pillar bridge.

Official Expected (last-3 AMS blended with last-6 median, then the national day
curve) is unchanged. This module sizes **Ask** and the pipeline identity:

    Pipeline Expected = Billed + Due unvisited + Drop variance + Not yet due

Ask is decoupled from monthly pacing. A due shop is asked for its expected drop,
not remaining-to-Expected, and not a next-drop ML prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

WINDOW_DAYS = 90
DUE_RATIO = 0.8
LAPSE_MULTIPLIER = 3.0
DEFAULT_API_DAYS = 14.0
MIN_GAP_DAYS = 2
MAX_GAP_DAYS = 120
MONTHLY_API_DAYS = 30.0

ACTION_CALL = "Due"
ACTION_CONVERT = "Due · visited"
ACTION_LIFT = "Another visit"
ACTION_RECOVER = "Lapsing"
ACTION_HOLD = "Hold"

REC_REORDER = "Reorder due"
REC_RECOVER = "Recover lost volume"
REC_HOLD = "Hold — not due"
REC_LAPSED = "Lapsed — lost door"
REC_NEW = "New door — no history"
REC_COMING = "Coming due this month"


def attach_demand_cycles(shops: pd.DataFrame, shop_day: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """API and expected drop from a rolling 90-day window, with cold-start rules.

    * 0 purchases ever → expected drop 0, Ask 0 (stays in universe).
    * 0 purchases in the window but older history → lapsed (no invented run-rate).
    * 1 purchase → expected drop = that invoice; API = 14 days until a second bill.
    * 2 purchases → mean drop; API = the one interval (else 14).
    * 3+ in the window → median drop and median gap (2–120 days).
    """
    out = shops.copy()
    out["cycle_days"] = np.nan
    out["typical_drop_mt"] = np.nan
    out["expected_drop_mt"] = 0.0
    out["last_drop_mt"] = np.nan
    out["last_bill_date"] = ""
    out["days_since_bill"] = np.nan
    out["cover_left_days"] = np.nan
    out["n_intervals"] = 0
    out["n_purchases_90d"] = 0
    out["n_purchases_ever"] = 0
    out["is_cold_start"] = False
    out["is_lapsed"] = False
    out["depletion_ratio"] = np.nan
    out["api_days"] = np.nan

    daily = _purchases(shop_day, as_of_ts)
    window_start = as_of_ts - pd.Timedelta(days=WINDOW_DAYS)
    by_shop: dict[str, pd.DataFrame] = {}
    if not daily.empty:
        for sid, g in daily.groupby(daily["store_id"].astype(str)):
            by_shop[str(sid)] = g.sort_values("sale_date")

    for i, row in out.iterrows():
        sid = str(row["store_id"])
        hist = by_shop.get(sid)
        if hist is None or hist.empty:
            hist = _monthly_as_purchases(row, as_of_ts)
        stats = _shop_demand_stats(hist, as_of_ts, window_start)
        out.at[i, "cycle_days"] = stats["api"]
        out.at[i, "api_days"] = stats["api"]
        out.at[i, "typical_drop_mt"] = stats["expected_drop"]
        out.at[i, "expected_drop_mt"] = stats["expected_drop"]
        out.at[i, "last_drop_mt"] = stats["last_drop"]
        out.at[i, "last_bill_date"] = stats["last_date_s"]
        out.at[i, "days_since_bill"] = stats["dslp"]
        out.at[i, "n_intervals"] = stats["n_intervals"]
        out.at[i, "n_purchases_90d"] = stats["n_90d"]
        out.at[i, "n_purchases_ever"] = stats["n_ever"]
        out.at[i, "is_cold_start"] = stats["cold"]
        out.at[i, "is_lapsed"] = stats["lapsed"]
        out.at[i, "depletion_ratio"] = stats["ratio"]
        out.at[i, "typical_bill_day"] = stats["api"]
        ams = float(row.get("ams_3m") or 0) or float(row.get("expected_mt") or 0)
        drop = stats["expected_drop"]
        api = stats["api"]
        daily_rate = max(ams / 30.0, drop / max(api, 1.0) if drop and api else 0.01, 0.01)
        last_drop = stats["last_drop"]
        dslp = stats["dslp"]
        stock_days = float(last_drop) / daily_rate if last_drop and daily_rate else np.nan
        cover = (stock_days - dslp) if pd.notna(stock_days) and pd.notna(dslp) else np.nan
        out.at[i, "cover_left_days"] = cover

    days_since = pd.to_numeric(out["days_since_bill"], errors="coerce")
    api = pd.to_numeric(out["api_days"], errors="coerce")
    out["days_overdue"] = (days_since - api).clip(lower=0)
    out["pace_frac"] = pd.to_numeric(out["depletion_ratio"], errors="coerce")
    return out


def _purchases(shop_day: pd.DataFrame | None, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    work = shop_day.copy()
    work["store_id"] = work["store_id"].astype(str)
    work["sale_date"] = pd.to_datetime(work["sale_date"], errors="coerce")
    work["volume_mt"] = pd.to_numeric(work["volume_mt"], errors="coerce").fillna(0.0)
    work = work[work["sale_date"].notna() & (work["sale_date"] <= as_of_ts) & (work["volume_mt"] > 0)]
    if work.empty:
        return work
    # One purchase per shop-day (sum SKUs if a daily file ever splits them).
    return work.groupby(["store_id", "sale_date"], as_index=False)["volume_mt"].sum()


def _monthly_as_purchases(row: pd.Series, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """When Outlet Date Wise is missing, treat the last billed month as one purchase."""
    last_period = str(row.get("last_billed_period") or "")
    last_vol = float(row.get("last_billed_mt") or row.get("last_month_mt") or 0)
    if not last_period or len(last_period) < 7 or last_vol <= 0:
        return pd.DataFrame(columns=["store_id", "sale_date", "volume_mt"])
    try:
        year, month = int(last_period[:4]), int(last_period[5:7])
        from calendar import monthrange

        day = monthrange(year, month)[1]
        ts = pd.Timestamp(year=year, month=month, day=day)
    except (TypeError, ValueError):
        return pd.DataFrame(columns=["store_id", "sale_date", "volume_mt"])
    if ts > as_of_ts:
        ts = as_of_ts
    return pd.DataFrame([{"store_id": str(row["store_id"]), "sale_date": ts, "volume_mt": last_vol}])


def _shop_demand_stats(hist: pd.DataFrame, as_of_ts: pd.Timestamp, window_start: pd.Timestamp) -> dict[str, Any]:
    empty = {
        "api": np.nan,
        "expected_drop": 0.0,
        "last_drop": np.nan,
        "last_date_s": "",
        "dslp": np.nan,
        "n_intervals": 0,
        "n_90d": 0,
        "n_ever": 0,
        "cold": False,
        "lapsed": False,
        "ratio": np.nan,
    }
    if hist is None or hist.empty:
        return empty
    hist = hist.sort_values("sale_date")
    n_ever = int(len(hist))
    last_row = hist.iloc[-1]
    last_date = pd.Timestamp(last_row["sale_date"])
    last_drop = float(last_row["volume_mt"])
    dslp = float((as_of_ts - last_date).days)
    in_window = hist[hist["sale_date"] >= window_start]
    n_90d = int(len(in_window))
    vols = pd.to_numeric(in_window["volume_mt"], errors="coerce").dropna() if n_90d else pd.Series(dtype=float)
    gaps = pd.Series(dtype=float)
    if n_90d >= 2:
        gaps = in_window["sale_date"].drop_duplicates().sort_values().diff().dt.days.dropna()
        gaps = gaps[(gaps >= MIN_GAP_DAYS) & (gaps <= MAX_GAP_DAYS)]

    cold = n_90d in (1, 2)
    if n_90d == 0:
        # Older history, nothing in the window — dead door, do not invent a drop.
        api = np.nan
        expected_drop = 0.0
        lapsed = True
        ratio = np.nan
    elif n_90d == 1:
        expected_drop = float(vols.iloc[0]) if len(vols) else 0.0
        api = DEFAULT_API_DAYS
        lapsed = bool(pd.notna(dslp) and dslp > LAPSE_MULTIPLIER * api)
        ratio = dslp / api if api else np.nan
    elif n_90d == 2:
        expected_drop = float(vols.mean()) if len(vols) else 0.0
        api = float(gaps.iloc[0]) if len(gaps) else DEFAULT_API_DAYS
        lapsed = bool(pd.notna(dslp) and dslp > LAPSE_MULTIPLIER * api)
        ratio = dslp / api if api else np.nan
    else:
        expected_drop = float(vols.median()) if len(vols) else 0.0
        api = float(gaps.median()) if len(gaps) else DEFAULT_API_DAYS
        lapsed = bool(pd.notna(dslp) and dslp > LAPSE_MULTIPLIER * api)
        ratio = dslp / api if api else np.nan

    return {
        "api": float(api) if pd.notna(api) else np.nan,
        "expected_drop": float(expected_drop) if expected_drop else 0.0,
        "last_drop": last_drop,
        "last_date_s": last_date.strftime("%Y-%m-%d"),
        "dslp": dslp,
        "n_intervals": int(len(gaps)),
        "n_90d": n_90d,
        "n_ever": n_ever,
        "cold": cold,
        "lapsed": lapsed,
        "ratio": float(ratio) if pd.notna(ratio) else np.nan,
    }


def classify_demand_actions(shops: pd.DataFrame, open_mtd: bool) -> pd.DataFrame:
    """Due / due-visited / another-visit / lapsed / hold from the depletion ratio.

    Lapsed doors are tagged Lapsing and must have Ask 0. Never-billed universe
    doors stay Due with Ask 0 so they still sit on the beat for visit %.
    """
    del open_mtd  # labels do not flip when the month closes; Ask is zeroed in the pipeline
    out = shops.copy()
    ratio = pd.to_numeric(out.get("depletion_ratio"), errors="coerce")
    billed = pd.to_numeric(out["billed_mt"], errors="coerce").fillna(0)
    drop = pd.to_numeric(out.get("expected_drop_mt"), errors="coerce").fillna(0)
    lapsed = out["is_lapsed"].fillna(False).astype(bool) if "is_lapsed" in out.columns else pd.Series(False, index=out.index)
    n_ever = pd.to_numeric(out.get("n_purchases_ever"), errors="coerce").fillna(0)
    visited = out["call_status"].eq("Visited · not billed")
    unvisited = out["call_status"].eq("Unvisited")
    unbilled = billed <= 0.005
    whitespace = unvisited & unbilled & (n_ever <= 0)
    due = ratio.notna() & (ratio >= DUE_RATIO) & ~lapsed & (drop > 0)

    out["action"] = ACTION_HOLD
    out.loc[due & unvisited & unbilled, "action"] = ACTION_CALL
    out.loc[due & visited & unbilled, "action"] = ACTION_CONVERT
    out.loc[due & ~unbilled, "action"] = ACTION_LIFT
    out.loc[lapsed, "action"] = ACTION_RECOVER
    out.loc[whitespace, "action"] = ACTION_CALL
    out["instruction"] = ""
    return out


def attach_pipeline(shops: pd.DataFrame, days_left: int = 0, open_mtd: bool = True) -> pd.DataFrame:
    """Daily Ask plus the four pillars that add back to pipeline Expected.

    Ask = expected drop when 0.8 ≤ ratio ≤ 3× (due, not lapsed); else 0.
    Not-yet-due volume is *not* Ask — it is the future pipeline pillar.
    """
    out = shops.copy()
    drop = pd.to_numeric(out.get("expected_drop_mt"), errors="coerce").fillna(0)
    billed = pd.to_numeric(out.get("billed_mt"), errors="coerce").fillna(0)
    ratio = pd.to_numeric(out.get("depletion_ratio"), errors="coerce")
    api = pd.to_numeric(out.get("api_days"), errors="coerce").replace(0, np.nan)
    dslp = pd.to_numeric(out.get("days_since_bill"), errors="coerce")
    lapsed = out["is_lapsed"].fillna(False).astype(bool) if "is_lapsed" in out.columns else pd.Series(False, index=out.index)
    n_ever = pd.to_numeric(out.get("n_purchases_ever"), errors="coerce").fillna(0)
    visited = out["call_status"].isin({"Visited · not billed", "Billed"})
    unvisited = out["call_status"].eq("Unvisited")
    unbilled = billed <= 0.005
    days = max(int(days_left), 0)

    due = ratio.notna() & (ratio >= DUE_RATIO) & ~lapsed & (drop > 0)
    # Days until the shop crosses the 0.8 reorder window.
    until = (DUE_RATIO * api - dslp.fillna(0)).clip(lower=0)
    will_due = (~due) & (~lapsed) & (drop > 0) & api.notna() & (until <= float(days)) & (days > 0)
    if not open_mtd:
        will_due = pd.Series(False, index=out.index)

    ask = pd.Series(0.0, index=out.index)
    if open_mtd:
        ask = pd.Series(np.where(due, drop, 0.0), index=out.index)

    due_unvisited = pd.Series(np.where(due & unvisited & unbilled, drop, 0.0), index=out.index)
    # Visited (or billed) and due: performance gap versus the typical drop.
    drop_var = pd.Series(np.where(due & visited, np.maximum(drop - billed, 0.0), 0.0), index=out.index)
    not_yet = pd.Series(np.where(will_due, drop, 0.0), index=out.index)

    pipeline = billed + due_unvisited + drop_var + not_yet

    out["week_target_mt"] = ask
    out["n_orders_left"] = np.where(due & open_mtd, 1.0, 0.0)
    out["days_until_due"] = np.where(due, 0.0, until)
    out["coming_due"] = will_due & ~due
    out["due_unvisited_mt"] = due_unvisited
    out["drop_variance_mt"] = drop_var
    out["not_yet_due_mt"] = not_yet
    out["pipeline_expected_mt"] = pipeline
    out["recommended_action"] = [
        _recommend(r, bool(d), bool(w), bool(l), bool(ws))
        for r, d, w, l, ws in zip(
            out.itertuples(index=False),
            due.tolist(),
            will_due.tolist(),
            lapsed.tolist(),
            (unvisited & unbilled & (n_ever <= 0)).tolist(),
        )
    ]
    return out


def _recommend(row: Any, due: bool, coming: bool, lapsed: bool, whitespace: bool) -> str:
    if whitespace:
        return REC_NEW
    if lapsed:
        return REC_LAPSED
    billed = float(getattr(row, "billed_mt", 0) or 0)
    drop = float(getattr(row, "expected_drop_mt", 0) or 0)
    if due:
        if billed > 0.005 and billed + 1e-9 < drop:
            return REC_RECOVER
        return REC_REORDER
    if coming:
        return REC_COMING
    return REC_HOLD


def pipeline_identity_holds(shops: pd.DataFrame, atol: float = 1e-6) -> bool:
    if shops is None or shops.empty:
        return True
    billed = pd.to_numeric(shops.get("billed_mt"), errors="coerce").fillna(0)
    due_u = pd.to_numeric(shops.get("due_unvisited_mt"), errors="coerce").fillna(0)
    var = pd.to_numeric(shops.get("drop_variance_mt"), errors="coerce").fillna(0)
    nyd = pd.to_numeric(shops.get("not_yet_due_mt"), errors="coerce").fillna(0)
    pipe = pd.to_numeric(shops.get("pipeline_expected_mt"), errors="coerce").fillna(0)
    return bool(np.allclose((billed + due_u + var + nyd).to_numpy(), pipe.to_numpy(), atol=atol))
