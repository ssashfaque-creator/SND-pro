"""This-week action engine: who to call, who to push, how much is still due.

Daily shop history is used for one job the monthly scorecard cannot do: decide
whether a quiet door is *behind its own intra-month shape* or simply
back-loaded. Expected stays the last-three-closed-month run-rate (same recipe
as the scorecard). The curve only times that Expected through the month.

Hierarchy (empirical Bayes): shop → DSR → city → country. Sparse doors inherit
the parent shape. Typical drop is a TSB-style median of billed days, shrunk
the same way.

No black-box forecast replaces Expected. The engine is scored with a walk-forward
cut (day 15 of closed months): does ranking by curve-behind catch shops that
actually miss Expected, better than ranking by calendar-day pace?
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sndintel.io_utils import prior_periods, shift_period
from sndintel.mtd import period_state
from sndintel.season import _recent_level, fit_seasonality, fit_shop_expected

SHOP_FLOOR_MT = 0.25
CURVE_K = 6.0
DROP_K = 4.0
SUMMARY_DIST_N = 15
SUMMARY_DSR_N = 15
SUMMARY_CALL_N = 80
SUMMARY_CONVERT_N = 40
SUMMARY_LIFT_N = 40
BACKTEST_CUT_DAY = 15
BACKTEST_MONTHS = 6
BACKTEST_TOP_N = 50
MAX_DAY = 31

ACTION_CALL = "Call"
ACTION_CONVERT = "Convert"
ACTION_LIFT = "Lift drop"
ACTION_HOLD = "Hold"
ACTION_RECOVER = "Recover"


@dataclass
class ActionPack:
    period: str
    label: str = ""
    as_of_day: int = 0
    days_in_month: int = 0
    days_left: int = 0
    open_mtd: bool = False
    has_daily: bool = False
    headline: str = ""
    source: str = "calendar"
    country: pd.DataFrame = field(default_factory=pd.DataFrame)
    distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    converts: pd.DataFrame = field(default_factory=pd.DataFrame)
    lifts: pd.DataFrame = field(default_factory=pd.DataFrame)
    holds: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    brief: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    raw_shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_backtest: pd.DataFrame = field(default_factory=pd.DataFrame)


def empty_action_pack(period: str = "") -> ActionPack:
    return ActionPack(period=period or "", label=period or "")


def build_action_pack(
    shop_month: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    shop_day: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
    ledger: pd.DataFrame | None = None,
    period: str | None = None,
) -> ActionPack:
    """Score every AMS>0 door for this week's call / convert / lift list."""
    if shop_month is None or shop_month.empty:
        return empty_action_pack(period or "")
    period = period or str(shop_month["period"].dropna().astype(str).max() or "")
    if not period:
        return empty_action_pack("")
    mtd = period_state(ledger, period)
    days_in_month = int(mtd.get("days_in_month") or _days_in_period(period))
    as_of = _as_of_day(shop_day, period, mtd, days_in_month)
    days_left = max(0, days_in_month - as_of)
    open_mtd = bool(mtd.get("open"))

    expected = _shop_expected_full(shop_month, period)
    billed = _period_billed(shop_month, period)
    shops = _shop_frame(expected, billed, stores, shop_month, period)
    if shops.empty:
        pack = empty_action_pack(period)
        pack.as_of_day = as_of
        pack.days_in_month = days_in_month
        pack.days_left = days_left
        pack.open_mtd = open_mtd
        pack.label = mtd.get("label") or period
        return pack

    daily = _prepare_daily(shop_day)
    has_daily = not daily.empty and (daily["period"].astype(str) == period).any()
    hist_daily = daily[daily["period"].astype(str) != period] if not daily.empty else daily
    curves = fit_delivery_curves(hist_daily, shops) if hist_daily is not None and not hist_daily.empty else {}
    source = "daily_curve" if curves.get("national") is not None else "calendar"

    pace = _lookup_pace(shops, curves, as_of, days_in_month)
    shops["pace_frac"] = pace
    shops["should_have_mt"] = shops["expected_mt"] * shops["pace_frac"]
    shops["behind_pace_mt"] = (shops["should_have_mt"] - shops["billed_mt"]).clip(lower=0)
    shops["remaining_mt"] = (shops["expected_mt"] - shops["billed_mt"]).clip(lower=0)
    shops["week_target_mt"] = _week_target(shops, curves, as_of, days_in_month, days_left)
    drops, bill_days = typical_drop_and_bill_day(hist_daily, shops)
    shops["typical_drop_mt"] = drops
    shops["typical_bill_day"] = bill_days
    last_bill, days_since = _last_bill(daily, shops, period, as_of)
    shops["last_bill_date"] = last_bill
    shops["days_since_bill"] = days_since
    shops = _attach_calls(shops, visits, period)
    shops = _classify_actions(shops, open_mtd, days_left)
    shops["value_score"] = _value_score(shops)
    shops = shops.sort_values(["value_score", "week_target_mt", "remaining_mt"], ascending=False)

    dist = _roll_units(shops, "distributor", extra_city=True)
    dsr = _roll_units(shops, "dsr_name", extra_city=True, extra_dist=True)

    country = _country_row(shops, as_of, days_in_month, days_left, open_mtd, source)
    backtest = backtest_delivery_curve(daily, shop_month, stores, period) if hist_daily is not None and not hist_daily.empty else pd.DataFrame()
    headline = _headline(country, shops, days_left, open_mtd, has_daily)

    pack = ActionPack(
        period=period,
        label=mtd.get("label") or period,
        as_of_day=as_of,
        days_in_month=days_in_month,
        days_left=days_left,
        open_mtd=open_mtd,
        has_daily=bool(has_daily or source == "daily_curve"),
        headline=headline,
        source=source,
        country=_present_country(country),
        distributors=_present_units(dist.head(SUMMARY_DIST_N), "Distributor"),
        dsrs=_present_units(dsr.head(SUMMARY_DSR_N), "DSR"),
        calls=_present_shops(_take_action(shops, ACTION_CALL, SUMMARY_CALL_N)),
        converts=_present_shops(_take_action(shops, ACTION_CONVERT, SUMMARY_CONVERT_N)),
        lifts=_present_shops(_take_action(shops, ACTION_LIFT, SUMMARY_LIFT_N)),
        holds=_present_shops(shops[shops["action"] == ACTION_HOLD].head(20)),
        all_shops=_present_shops(shops),
        all_distributors=_present_units(dist, "Distributor"),
        all_dsrs=_present_units(dsr, "DSR"),
        backtest=_present_backtest(backtest),
        brief=country,
        curves=curves,
        raw_shops=shops,
        raw_distributors=dist,
        raw_dsrs=dsr,
        raw_backtest=backtest,
    )
    return pack


def fit_delivery_curves(shop_day: pd.DataFrame, shops: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Volume-weighted cumulative fraction by day-of-month, shrunk down the tree."""
    panel = _month_cumfrac(shop_day)
    if panel.empty:
        return {}
    national = _weighted_curve(panel, None)
    if national is None:
        return {}
    out: dict[str, np.ndarray] = {"national": national}
    if "city" in panel.columns:
        for city, g in panel.groupby(panel["city"].astype(str)):
            local = _weighted_curve(g, None)
            cred = float(g["period"].nunique()) / (float(g["period"].nunique()) + CURVE_K)
            out[f"city::{city}"] = _mix_curve(local, national, cred)
    if "dsr_name" in panel.columns:
        for (city, dsr), g in panel.groupby([panel["city"].astype(str), panel["dsr_name"].astype(str)]):
            parent = _curve_or(out.get(f"city::{city}"), national)
            local = _weighted_curve(g, None)
            cred = float(g["period"].nunique()) / (float(g["period"].nunique()) + CURVE_K)
            out[f"dsr::{city}::{dsr}"] = _mix_curve(local, parent, cred)
    if "store_id" in panel.columns:
        for (sid, city, dsr), g in panel.groupby(
            [panel["store_id"].astype(str), panel["city"].astype(str), panel["dsr_name"].astype(str)]
        ):
            parent = _curve_or(out.get(f"dsr::{city}::{dsr}"), out.get(f"city::{city}"), national)
            local = _weighted_curve(g, None)
            cred = float(len(g)) / (float(len(g)) + CURVE_K)
            out[f"shop::{sid}"] = _mix_curve(local, parent, cred)
    return out


def typical_drop_and_bill_day(
    shop_day: pd.DataFrame, shops: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Median billed-day drop and typical bill day, shrunk shop → DSR → city."""
    idx = shops.index
    drop = pd.Series(np.nan, index=idx, dtype=float)
    day = pd.Series(np.nan, index=idx, dtype=float)
    if shop_day is None or shop_day.empty:
        fallback = (shops["expected_mt"] / 4.0).clip(lower=0.05)
        return fallback, pd.Series(15.0, index=idx)
    pos = shop_day[pd.to_numeric(shop_day["volume_mt"], errors="coerce").fillna(0) > 0].copy()
    if pos.empty:
        fallback = (shops["expected_mt"] / 4.0).clip(lower=0.05)
        return fallback, pd.Series(15.0, index=idx)
    pos["store_id"] = pos["store_id"].astype(str)
    pos["city"] = pos.get("city", pd.Series("", index=pos.index)).fillna("").astype(str)
    pos["dsr_name"] = pos.get("dsr_name", pd.Series("", index=pos.index)).fillna("").astype(str)
    pos["day"] = pd.to_numeric(pos["day"], errors="coerce")
    shop_drop = pos.groupby("store_id")["volume_mt"].median()
    shop_day_med = pos.groupby("store_id")["day"].median()
    shop_n = pos.groupby("store_id").size()
    dsr_drop = pos.groupby(["city", "dsr_name"])["volume_mt"].median()
    city_drop = pos.groupby("city")["volume_mt"].median()
    nat_drop = float(pos["volume_mt"].median())
    dsr_day = pos.groupby(["city", "dsr_name"])["day"].median()
    city_day = pos.groupby("city")["day"].median()
    nat_day = float(pos["day"].median()) if pos["day"].notna().any() else 15.0

    for i, row in shops.iterrows():
        sid = str(row["store_id"])
        city = str(row.get("city") or "")
        dsr = str(row.get("dsr_name") or "")
        parent_drop = dsr_drop.get((city, dsr), city_drop.get(city, nat_drop))
        parent_day = dsr_day.get((city, dsr), city_day.get(city, nat_day))
        n = float(shop_n.get(sid, 0))
        cred = n / (n + DROP_K) if n else 0.0
        raw_drop = shop_drop.get(sid)
        raw_day = shop_day_med.get(sid)
        drop.loc[i] = cred * float(raw_drop) + (1 - cred) * float(parent_drop) if pd.notna(raw_drop) else float(parent_drop)
        day.loc[i] = cred * float(raw_day) + (1 - cred) * float(parent_day) if pd.notna(raw_day) else float(parent_day)
    drop = drop.fillna((shops["expected_mt"] / 4.0).clip(lower=0.05))
    day = day.fillna(15.0)
    return drop, day


def backtest_delivery_curve(
    shop_day: pd.DataFrame,
    shop_month: pd.DataFrame,
    stores: pd.DataFrame | None,
    period: str,
    cut_day: int = BACKTEST_CUT_DAY,
    n_months: int = BACKTEST_MONTHS,
) -> pd.DataFrame:
    """Walk-forward: at day 15 of each closed month, did curve-behind beat calendar-behind?"""
    if shop_day is None or shop_day.empty or shop_month is None or shop_month.empty:
        return pd.DataFrame()
    daily = _prepare_daily(shop_day)
    months = sorted(p for p in daily["period"].astype(str).unique() if p < str(period))
    if len(months) < 4:
        return pd.DataFrame()
    test_months = months[-n_months:]
    rows = []
    for test in test_months:
        hist_day = daily[daily["period"].astype(str) < test]
        if hist_day.empty:
            continue
        expected = _shop_expected_full(shop_month, test)
        billed_full = _period_billed(shop_month, test)
        shops = _shop_frame(expected, billed_full, stores, shop_month, test)
        if shops.empty:
            continue
        cut = daily[(daily["period"].astype(str) == test) & (pd.to_numeric(daily["day"], errors="coerce") <= cut_day)]
        billed_cut = cut.groupby(cut["store_id"].astype(str))["volume_mt"].sum() if not cut.empty else pd.Series(dtype=float)
        shops["billed_cut"] = shops["store_id"].map(billed_cut).fillna(0.0)
        days = _days_in_period(test)
        curves = fit_delivery_curves(hist_day, shops)
        curve_frac = _lookup_pace(shops, curves, cut_day, days)
        cal_frac = cut_day / float(days) if days else 1.0
        shops["curve_behind"] = (shops["expected_mt"] * curve_frac - shops["billed_cut"]).clip(lower=0)
        shops["cal_behind"] = (shops["expected_mt"] * cal_frac - shops["billed_cut"]).clip(lower=0)
        shops["missed"] = (shops["expected_mt"] - shops["billed_mt"]).clip(lower=0) >= SHOP_FLOOR_MT
        scored = shops[shops["expected_mt"] >= SHOP_FLOOR_MT].copy()
        if scored.empty:
            continue
        curve_top = scored.sort_values("curve_behind", ascending=False).head(BACKTEST_TOP_N)
        cal_top = scored.sort_values("cal_behind", ascending=False).head(BACKTEST_TOP_N)
        n = min(BACKTEST_TOP_N, len(scored))
        curve_p = float(curve_top["missed"].mean()) if n else 0.0
        cal_p = float(cal_top["missed"].mean()) if n else 0.0
        hold = scored[(scored["cal_behind"] >= SHOP_FLOOR_MT) & (scored["curve_behind"] < SHOP_FLOOR_MT)]
        rows.append(
            {
                "period": test,
                "cut_day": cut_day,
                "n_shops": int(len(scored)),
                "curve_precision_at_50": curve_p,
                "calendar_precision_at_50": cal_p,
                "curve_catch_mt": float(curve_top.loc[curve_top["missed"], "expected_mt"].sum() - curve_top.loc[curve_top["missed"], "billed_mt"].sum()),
                "calendar_catch_mt": float(cal_top.loc[cal_top["missed"], "expected_mt"].sum() - cal_top.loc[cal_top["missed"], "billed_mt"].sum()),
                "n_backloaded_hold": int(len(hold)),
                "n_backloaded_ok": int((~hold["missed"]).sum()) if not hold.empty else 0,
                "remaining_mae_mt": float(
            (
                (scored["expected_mt"] - scored["billed_cut"]).clip(lower=0)
                - (scored["expected_mt"] - scored["billed_mt"]).clip(lower=0)
            )
            .abs()
            .mean()
        ),
            }
        )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = {
        "period": period,
        "cut_day": cut_day,
        "n_months": int(len(detail)),
        "n_shops": int(detail["n_shops"].sum()),
        "curve_precision_at_50": float(detail["curve_precision_at_50"].mean()),
        "calendar_precision_at_50": float(detail["calendar_precision_at_50"].mean()),
        "curve_catch_mt": float(detail["curve_catch_mt"].sum()),
        "calendar_catch_mt": float(detail["calendar_catch_mt"].sum()),
        "n_backloaded_hold": int(detail["n_backloaded_hold"].sum()),
        "n_backloaded_ok": int(detail["n_backloaded_ok"].sum()),
        "remaining_mae_mt": float(detail["remaining_mae_mt"].mean()),
        "notes": (
            "At day 15 of each closed month, rank shops by how far they are behind their own "
            "delivery curve versus a flat calendar pace. Precision@50 is the share of that list "
            "that finished the month at least 0.25 MT below Expected. Back-loaded hold = shops "
            "calendar would call but the curve left alone; OK means they finished on Expected."
        ),
    }
    return pd.DataFrame([summary])


def persist_action_pack(conn, pack: ActionPack) -> None:
    """Write scored lists into the warehouse for the UI and exports."""
    from sndintel.storage import replace_table

    period = pack.period
    replace_table(conn, "action_shops", _raw_shops_to_sql(pack.raw_shops, period))
    units = []
    if pack.raw_distributors is not None and not pack.raw_distributors.empty:
        units.append(_raw_units_to_sql(pack.raw_distributors, period, "distributor"))
    if pack.raw_dsrs is not None and not pack.raw_dsrs.empty:
        units.append(_raw_units_to_sql(pack.raw_dsrs, period, "dsr"))
    replace_table(conn, "action_units", pd.concat(units, ignore_index=True) if units else pd.DataFrame())
    bt = pack.raw_backtest if pack.raw_backtest is not None and not pack.raw_backtest.empty else pack.backtest
    replace_table(conn, "action_backtest", _backtest_to_sql(bt, period))
    replace_table(conn, "action_brief", pd.DataFrame([_brief_to_sql(pack)]))


def load_action_pack(conn, period: str | None = None) -> ActionPack:
    """Rebuild a pack from warehouse tables (no recompute)."""
    from sndintel.storage import read_sql

    brief = read_sql(conn, "SELECT * FROM action_brief")
    if brief is None or brief.empty:
        return empty_action_pack(period or "")
    if period:
        brief = brief[brief["period"].astype(str) == str(period)]
    if brief.empty:
        brief = read_sql(conn, "SELECT * FROM action_brief")
    row = brief.iloc[-1]
    period = str(row["period"])
    shops = read_sql(conn, "SELECT * FROM action_shops WHERE period = ?", (period,))
    units = read_sql(conn, "SELECT * FROM action_units WHERE period = ?", (period,))
    back = read_sql(conn, "SELECT * FROM action_backtest WHERE period = ?", (period,))
    dists = units[units["grain"] == "distributor"] if units is not None and not units.empty else pd.DataFrame()
    dsrs = units[units["grain"] == "dsr"] if units is not None and not units.empty else pd.DataFrame()
    shops = _sql_shops_to_present(shops)
    dist_p = _sql_units_to_present(dists, "Distributor")
    dsr_p = _sql_units_to_present(dsrs, "DSR")
    return ActionPack(
        period=period,
        label=period,
        as_of_day=int(row.get("as_of_day") or 0),
        days_in_month=int(row.get("days_in_month") or 0),
        days_left=int(row.get("days_left") or 0),
        open_mtd=bool(int(row.get("days_left") or 0) > 0),
        has_daily=bool(int(row.get("has_daily") or 0)),
        headline=str(row.get("headline") or ""),
        source=str(row.get("source") or ""),
        country=_present_country(
            {
                "billed_mt": row.get("billed_mt"),
                "expected_mt": row.get("expected_mt"),
                "should_have_mt": row.get("should_have_mt"),
                "behind_pace_mt": row.get("behind_pace_mt"),
                "remaining_mt": row.get("remaining_mt"),
                "week_target_mt": row.get("week_target_mt"),
                "as_of_day": row.get("as_of_day"),
                "days_in_month": row.get("days_in_month"),
                "days_left": row.get("days_left"),
                "n_call": row.get("n_call"),
                "n_convert": row.get("n_convert"),
                "n_lift": row.get("n_lift"),
                "source": row.get("source"),
            }
        ),
        distributors=dist_p.head(SUMMARY_DIST_N),
        dsrs=dsr_p.head(SUMMARY_DSR_N),
        calls=_take_present(shops, ACTION_CALL, SUMMARY_CALL_N),
        converts=_take_present(shops, ACTION_CONVERT, SUMMARY_CONVERT_N),
        lifts=_take_present(shops, ACTION_LIFT, SUMMARY_LIFT_N),
        holds=_take_present(shops, ACTION_HOLD, 20),
        all_shops=shops,
        all_distributors=dist_p,
        all_dsrs=dsr_p,
        backtest=_present_backtest(back),
        brief=dict(row),
    )


def _days_in_period(period: str) -> int:
    year, month = int(period[:4]), int(period[5:7])
    return monthrange(year, month)[1]


def _as_of_day(shop_day: pd.DataFrame | None, period: str, mtd: dict[str, Any], days: int) -> int:
    if mtd.get("as_of_day"):
        return int(min(max(int(mtd["as_of_day"]), 1), days))
    if shop_day is not None and not shop_day.empty:
        cur = shop_day[shop_day["period"].astype(str) == str(period)]
        if not cur.empty and "day" in cur.columns:
            last = pd.to_numeric(cur["day"], errors="coerce").max()
            if pd.notna(last):
                return int(min(max(int(last), 1), days))
        if not cur.empty and "sale_date" in cur.columns:
            last = pd.to_datetime(cur["sale_date"], errors="coerce").max()
            if pd.notna(last):
                return int(min(max(int(last.day), 1), days))
    return days


def _prepare_daily(shop_day: pd.DataFrame | None) -> pd.DataFrame:
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    out = shop_day.copy()
    out["store_id"] = out["store_id"].astype(str)
    out["volume_mt"] = pd.to_numeric(out["volume_mt"], errors="coerce").fillna(0.0)
    out["period"] = out["period"].astype(str)
    if "day" not in out.columns or out["day"].isna().all():
        out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce")
        out["day"] = out["sale_date"].dt.day
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out = out[out["day"].notna() & (out["volume_mt"] > 0)]
    for col in ("city", "dsr_name", "distributor", "store_name", "section"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


def _month_cumfrac(shop_day: pd.DataFrame) -> pd.DataFrame:
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    rows = []
    for (sid, period), g in shop_day.groupby(["store_id", "period"]):
        days = _days_in_period(str(period))
        by_day = g.groupby("day")["volume_mt"].sum()
        total = float(by_day.sum())
        if total <= 0:
            continue
        running = 0.0
        frac = np.zeros(MAX_DAY + 1, dtype=float)
        for d in range(1, days + 1):
            running += float(by_day.get(d, 0.0))
            frac[d] = running / total
        if days < MAX_DAY:
            frac[days + 1 :] = 1.0
        city = str(g["city"].iloc[-1]) if "city" in g.columns else ""
        dsr = str(g["dsr_name"].iloc[-1]) if "dsr_name" in g.columns else ""
        rows.append(
            {
                "store_id": str(sid),
                "period": str(period),
                "city": city,
                "dsr_name": dsr,
                "month_total": total,
                "frac": frac,
            }
        )
    return pd.DataFrame(rows)


def _weighted_curve(panel: pd.DataFrame, _unused) -> np.ndarray | None:
    if panel is None or panel.empty:
        return None
    weights = pd.to_numeric(panel["month_total"], errors="coerce").fillna(0.0).to_numpy()
    if float(weights.sum()) <= 0:
        return None
    stacked = np.vstack(panel["frac"].to_list())
    curve = np.average(stacked, axis=0, weights=weights)
    return _monotone_unit(curve)


def _curve_or(*candidates) -> np.ndarray | None:
    for item in candidates:
        if item is not None:
            return item
    return None


def _mix_curve(local: np.ndarray | None, parent: np.ndarray, cred: float) -> np.ndarray:
    if local is None:
        return parent
    cred = float(min(max(cred, 0.0), 1.0))
    return _monotone_unit(cred * local + (1.0 - cred) * parent)


def _monotone_unit(curve: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(curve, dtype=float), 0.0, 1.0)
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    if out[MAX_DAY] < 1.0:
        # Last observed day should finish at 1 if any mass exists.
        last = int(np.max(np.where(out > 0)[0])) if np.any(out > 0) else MAX_DAY
        if last >= 1:
            out[last:] = np.maximum(out[last:], 1.0)
    out[-1] = 1.0
    return out


def _lookup_pace(shops: pd.DataFrame, curves: dict[str, np.ndarray], as_of: int, days: int) -> pd.Series:
    calendar = as_of / float(days) if days else 1.0
    calendar = float(min(max(calendar, 0.0), 1.0))
    if not curves:
        return pd.Series(calendar, index=shops.index)
    national = curves.get("national")
    vals = []
    d = int(min(max(as_of, 1), MAX_DAY))
    for _, row in shops.iterrows():
        sid = str(row["store_id"])
        city = str(row.get("city") or "")
        dsr = str(row.get("dsr_name") or "")
        curve = _curve_or(
            curves.get(f"shop::{sid}"),
            curves.get(f"dsr::{city}::{dsr}"),
            curves.get(f"city::{city}"),
            national,
        )
        if curve is None:
            vals.append(calendar)
        else:
            vals.append(float(curve[d]))
    return pd.Series(vals, index=shops.index)


def _week_target(
    shops: pd.DataFrame,
    curves: dict[str, np.ndarray],
    as_of: int,
    days: int,
    days_left: int,
) -> pd.Series:
    horizon = min(7, days_left) if days_left > 0 else 0
    if horizon <= 0:
        return shops["remaining_mt"].copy()
    future = _lookup_pace(shops, curves, min(as_of + horizon, days), days)
    now = shops["pace_frac"]
    denom = (1.0 - now).clip(lower=1e-6)
    week_frac = ((future - now) / denom).clip(lower=0, upper=1)
    on_curve = shops["remaining_mt"] * week_frac
    catch_up = shops["behind_pace_mt"]
    return (on_curve + catch_up).clip(upper=shops["remaining_mt"])


def _shop_expected_full(shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    season = fit_seasonality(shop_month, period)
    exp = fit_shop_expected(shop_month, period, season, intra_frac=1.0)
    if exp is None or exp.empty:
        return pd.DataFrame(columns=["store_id", "expected_mt", "ams_3m"])
    out = exp.copy()
    out["store_id"] = out["store_id"].astype(str)
    full = pd.to_numeric(out.get("expected_full_mt"), errors="coerce")
    paced = pd.to_numeric(out.get("expected_mt"), errors="coerce")
    out["expected_mt"] = full.where(full.notna() & (full > 0), paced).fillna(0.0)
    recent = prior_periods(period, 3)
    hist = shop_month[shop_month["period"].astype(str).isin(recent)].copy()
    if hist.empty:
        out["ams_3m"] = 0.0
        return out[["store_id", "expected_mt", "ams_3m"]]
    pt = hist.pivot_table(index=hist["store_id"].astype(str), columns="period", values="volume_mt", aggfunc="sum")
    pt = pt.reindex(columns=recent, fill_value=0).fillna(0.0)
    ams = pt.mean(axis=1).rename("ams_3m")
    out = out.merge(ams.reset_index(), on="store_id", how="left")
    out["ams_3m"] = pd.to_numeric(out["ams_3m"], errors="coerce").fillna(0.0)
    return out[["store_id", "expected_mt", "ams_3m"]]


def _period_billed(shop_month: pd.DataFrame, period: str) -> pd.Series:
    cur = shop_month[shop_month["period"].astype(str) == str(period)]
    if cur.empty:
        return pd.Series(dtype=float)
    return cur.groupby(cur["store_id"].astype(str))["volume_mt"].sum()


def _shop_frame(
    expected: pd.DataFrame,
    billed: pd.Series,
    stores: pd.DataFrame | None,
    shop_month: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    if expected is None or expected.empty:
        return pd.DataFrame()
    out = expected.copy()
    out["billed_mt"] = out["store_id"].map(billed).fillna(0.0)
    out = out[(out["ams_3m"] > 0) | (out["expected_mt"] >= SHOP_FLOOR_MT)]
    attrs = _latest_attrs(shop_month, stores, period)
    if not attrs.empty:
        out = out.merge(attrs, on="store_id", how="left")
    for col in ("store_name", "city", "distributor", "dsr_name", "section"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out.reset_index(drop=True)


def _latest_attrs(shop_month: pd.DataFrame, stores: pd.DataFrame | None, period: str) -> pd.DataFrame:
    cols = ["store_id", "store_name", "city", "distributor", "dsr_name", "section"]
    frames = []
    if stores is not None and not stores.empty:
        st = stores.copy()
        st["store_id"] = st["store_id"].astype(str)
        have = [c for c in cols if c in st.columns]
        frames.append(st[have].drop_duplicates("store_id", keep="first"))
    if shop_month is not None and not shop_month.empty:
        sm = shop_month.copy()
        sm["store_id"] = sm["store_id"].astype(str)
        sm = sm.sort_values("period")
        have = [c for c in cols if c in sm.columns]
        frames.append(sm[have].drop_duplicates("store_id", keep="last"))
    if not frames:
        return pd.DataFrame(columns=cols)
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="store_id", how="outer", suffixes=("", "_x"))
        for col in cols:
            if col == "store_id":
                continue
            other = f"{col}_x"
            if other in out.columns:
                out[col] = out[col].combine_first(out[other]) if col in out.columns else out[other]
                out = out.drop(columns=[other])
    return out


def _last_bill(
    daily: pd.DataFrame, shops: pd.DataFrame, period: str, as_of: int
) -> tuple[pd.Series, pd.Series]:
    last = pd.Series("", index=shops.index)
    since = pd.Series(np.nan, index=shops.index)
    if daily is None or daily.empty:
        return last, since
    work = daily.copy()
    work["sale_date"] = pd.to_datetime(work.get("sale_date"), errors="coerce")
    latest = work.sort_values("sale_date").groupby("store_id").tail(1)
    mapped = dict(zip(latest["store_id"].astype(str), latest["sale_date"]))
    year, month = int(period[:4]), int(period[5:7])
    as_of_ts = pd.Timestamp(year=year, month=month, day=min(max(as_of, 1), _days_in_period(period)))
    for i, row in shops.iterrows():
        dt = mapped.get(str(row["store_id"]))
        if dt is None or pd.isna(dt):
            continue
        last.loc[i] = pd.Timestamp(dt).strftime("%Y-%m-%d")
        since.loc[i] = int((as_of_ts - pd.Timestamp(dt)).days)
    return last, since


def _attach_calls(shops: pd.DataFrame, visits: pd.DataFrame | None, period: str) -> pd.DataFrame:
    out = shops.copy()
    out["visits"] = 0
    if visits is not None and not visits.empty and "store_id" in visits.columns:
        v = visits.copy()
        if "period" in v.columns:
            v = v[v["period"].astype(str) == str(period)]
        if not v.empty:
            g = v.groupby(v["store_id"].astype(str), as_index=False)["visits"].sum() if "visits" in v.columns else v
            if "visits" in g.columns:
                out["visits"] = out["store_id"].map(dict(zip(g["store_id"].astype(str), pd.to_numeric(g["visits"], errors="coerce").fillna(0)))).fillna(0)
    billed = pd.to_numeric(out["billed_mt"], errors="coerce").fillna(0) > 0
    visited = pd.to_numeric(out["visits"], errors="coerce").fillna(0) > 0
    out["call_status"] = "Unvisited"
    out.loc[visited & ~billed, "call_status"] = "Visited · not billed"
    out.loc[billed, "call_status"] = "Billed"
    return out


def _classify_actions(shops: pd.DataFrame, open_mtd: bool, days_left: int) -> pd.DataFrame:
    out = shops.copy()
    remaining = pd.to_numeric(out["remaining_mt"], errors="coerce").fillna(0)
    behind = pd.to_numeric(out["behind_pace_mt"], errors="coerce").fillna(0)
    billed = pd.to_numeric(out["billed_mt"], errors="coerce").fillna(0)
    material = (remaining >= SHOP_FLOOR_MT) | (behind >= SHOP_FLOOR_MT)
    out["action"] = ACTION_HOLD
    recover = (not open_mtd) or days_left == 0
    call_mask = material & (billed <= 0) & (out["call_status"] == "Unvisited")
    convert_mask = material & (billed <= 0) & (out["call_status"] == "Visited · not billed")
    lift_mask = material & (billed > 0) & (behind >= SHOP_FLOOR_MT)
    out.loc[call_mask, "action"] = ACTION_RECOVER if recover else ACTION_CALL
    out.loc[convert_mask, "action"] = ACTION_CONVERT
    out.loc[lift_mask, "action"] = ACTION_LIFT
    out["instruction"] = [_instruction(r, open_mtd, days_left) for r in out.itertuples(index=False)]
    return out


def _instruction(row: Any, open_mtd: bool, days_left: int) -> str:
    name = str(getattr(row, "store_name", "") or getattr(row, "store_id", ""))
    remaining = float(getattr(row, "remaining_mt", 0) or 0)
    behind = float(getattr(row, "behind_pace_mt", 0) or 0)
    billed = float(getattr(row, "billed_mt", 0) or 0)
    should = float(getattr(row, "should_have_mt", 0) or 0)
    week = float(getattr(row, "week_target_mt", 0) or 0)
    drop = float(getattr(row, "typical_drop_mt", 0) or 0)
    bill_day = getattr(row, "typical_bill_day", None)
    last = str(getattr(row, "last_bill_date", "") or "")
    action = str(getattr(row, "action", "") or "")
    typical = f"Typical drop {drop:.2f} MT"
    if pd.notna(bill_day):
        typical += f", usually bills around day {int(round(float(bill_day)))}"
    last_bit = f" Last billed {last}." if last else ""
    if action == ACTION_CALL:
        return (
            f"Call {name} this week. Need {week:.2f} MT in the next 7 days "
            f"({remaining:.2f} MT still to Expected). Behind pace by {behind:.2f} MT. {typical}.{last_bit}"
        )
    if action == ACTION_RECOVER:
        return (
            f"Recover {name} in the first week of next month. Finished {behind:.2f} MT behind Expected. {typical}.{last_bit}"
        )
    if action == ACTION_CONVERT:
        return (
            f"{name} was visited and did not buy. Close {week:.2f} MT this week "
            f"({remaining:.2f} MT still to Expected). {typical}."
        )
    if action == ACTION_LIFT:
        return (
            f"{name} billed {billed:.2f} MT vs {should:.2f} MT that should already be in. "
            f"Get another {week:.2f} MT this week ({remaining:.2f} MT still to Expected). {typical}."
        )
    if remaining < SHOP_FLOOR_MT and behind < SHOP_FLOOR_MT:
        if days_left > 0 and open_mtd:
            return f"{name} is on its own delivery curve. Do not pull the beat here — {remaining:.2f} MT can wait."
        return f"{name} finished on Expected."
    return f"{name}: billed {billed:.2f} of {should:.2f} expected by now."


def _value_score(shops: pd.DataFrame) -> pd.Series:
    remaining = pd.to_numeric(shops["remaining_mt"], errors="coerce").fillna(0)
    behind = pd.to_numeric(shops["behind_pace_mt"], errors="coerce").fillna(0)
    should = pd.to_numeric(shops["should_have_mt"], errors="coerce").fillna(0)
    week = pd.to_numeric(shops["week_target_mt"], errors="coerce").fillna(0)
    urgency = (behind / should.clip(lower=SHOP_FLOOR_MT)).clip(upper=2.0)
    score = week * (1.0 + 0.5 * urgency)
    recency = pd.to_numeric(shops.get("days_since_bill"), errors="coerce")
    score = score.where(~(recency > 21), score * 1.15)
    score = score.where(shops["action"] != ACTION_CONVERT, score * 0.9)
    score = score.where(shops["action"] != ACTION_LIFT, score * 0.75)
    score = score.where(~shops["action"].isin({ACTION_HOLD}), 0.0)
    return score.fillna(0.0)


def _roll_units(
    shops: pd.DataFrame,
    key: str,
    extra_city: bool = False,
    extra_dist: bool = False,
) -> pd.DataFrame:
    if shops is None or shops.empty or key not in shops.columns:
        return pd.DataFrame()
    work = shops[shops[key].astype(str).str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame()
    rows = []
    for name, g in work.groupby(key):
        billed = float(g["billed_mt"].sum())
        expected = float(g["expected_mt"].sum())
        should = float(g["should_have_mt"].sum())
        behind = float(g["behind_pace_mt"].sum())
        remaining = float(g["remaining_mt"].sum())
        week = float(g["week_target_mt"].sum())
        n_call = int((g["action"].isin({ACTION_CALL, ACTION_RECOVER})).sum())
        n_convert = int((g["action"] == ACTION_CONVERT).sum())
        n_lift = int((g["action"] == ACTION_LIFT).sum())
        n_hold = int((g["action"] == ACTION_HOLD).sum())
        instruction = (
            f"Push {name}: {n_call} shops to call ({g.loc[g['action'].isin({ACTION_CALL, ACTION_RECOVER}), 'week_target_mt'].sum():.1f} MT this week), "
            f"{n_convert} visited-not-billed ({g.loc[g['action'] == ACTION_CONVERT, 'week_target_mt'].sum():.1f} MT), "
            f"{n_lift} billed but behind curve ({g.loc[g['action'] == ACTION_LIFT, 'week_target_mt'].sum():.1f} MT). "
            f"Total {week:.1f} MT this week, {remaining:.1f} MT still to Expected."
        )
        row = {
            "grain_id": str(name),
            "n_call": n_call,
            "n_convert": n_convert,
            "n_lift": n_lift,
            "n_hold": n_hold,
            "billed_mt": billed,
            "expected_mt": expected,
            "should_have_mt": should,
            "behind_pace_mt": behind,
            "remaining_mt": remaining,
            "week_target_mt": week,
            "instruction": instruction,
            "value_score": float(g["value_score"].sum()),
        }
        if extra_city and "city" in g.columns:
            row["city"] = str(g["city"].mode().iloc[0]) if not g["city"].mode().empty else ""
        if extra_dist and "distributor" in g.columns:
            row["distributor"] = str(g["distributor"].mode().iloc[0]) if not g["distributor"].mode().empty else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["value_score", "week_target_mt"], ascending=False).reset_index(drop=True)


def _country_row(
    shops: pd.DataFrame,
    as_of: int,
    days: int,
    days_left: int,
    open_mtd: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "billed_mt": float(shops["billed_mt"].sum()),
        "expected_mt": float(shops["expected_mt"].sum()),
        "should_have_mt": float(shops["should_have_mt"].sum()),
        "behind_pace_mt": float(shops["behind_pace_mt"].sum()),
        "remaining_mt": float(shops["remaining_mt"].sum()),
        "week_target_mt": float(shops["week_target_mt"].sum()),
        "as_of_day": as_of,
        "days_in_month": days,
        "days_left": days_left,
        "open_mtd": open_mtd,
        "n_call": int(shops["action"].isin({ACTION_CALL, ACTION_RECOVER}).sum()),
        "n_convert": int((shops["action"] == ACTION_CONVERT).sum()),
        "n_lift": int((shops["action"] == ACTION_LIFT).sum()),
        "n_hold": int((shops["action"] == ACTION_HOLD).sum()),
        "source": source,
    }


def _headline(country: dict[str, Any], shops: pd.DataFrame, days_left: int, open_mtd: bool, has_daily: bool) -> str:
    week = float(country.get("week_target_mt") or 0)
    n_call = int(country.get("n_call") or 0)
    n_convert = int(country.get("n_convert") or 0)
    n_lift = int(country.get("n_lift") or 0)
    behind = float(country.get("behind_pace_mt") or 0)
    if not has_daily:
        return (
            f"No daily billed days in the warehouse — lists use calendar pace. "
            f"{n_call} shops to call, {n_convert} to convert, {n_lift} to lift drop ({week:.0f} MT this week)."
        )
    if open_mtd and days_left > 0:
        return (
            f"Day {country.get('as_of_day')}/{country.get('days_in_month')}: country is {behind:.0f} MT behind its own "
            f"delivery curve. This week: call {n_call} doors, convert {n_convert}, lift drop on {n_lift} "
            f"({week:.0f} MT)."
        )
    return (
        f"Month closed {behind:.0f} MT behind Expected. First-week recover list: {n_call} shops, "
        f"{n_convert} still to convert, {n_lift} drop-size recoveries ({week:.0f} MT)."
    )


def _take_action(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    part = shops[shops["action"] == action]
    if action == ACTION_CALL:
        part = shops[shops["action"].isin({ACTION_CALL, ACTION_RECOVER})]
    return part.head(n)


def _round_mt(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _round_drop(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


COUNTRY_VIEW = [
    ("billed_mt", "Billed (MT)"),
    ("expected_mt", "Expected this month (MT)"),
    ("should_have_mt", "Should have by today (MT)"),
    ("behind_pace_mt", "Behind pace (MT)"),
    ("remaining_mt", "Still to Expected (MT)"),
    ("week_target_mt", "This week (MT)"),
    ("as_of_day", "As of day"),
    ("days_left", "Days left"),
    ("n_call", "Shops to call"),
    ("n_convert", "Visited · not billed"),
    ("n_lift", "Lift drop"),
]


SHOP_VIEW = [
    ("store_name", "Shop"),
    ("store_id", "POP"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("dsr_name", "DSR"),
    ("action", "Action"),
    ("week_target_mt", "This week (MT)"),
    ("remaining_mt", "Still to Expected (MT)"),
    ("behind_pace_mt", "Behind pace (MT)"),
    ("billed_mt", "Billed (MT)"),
    ("expected_mt", "Expected (MT)"),
    ("should_have_mt", "Should have (MT)"),
    ("typical_drop_mt", "Typical drop (MT)"),
    ("typical_bill_day", "Usual bill day"),
    ("last_bill_date", "Last billed"),
    ("call_status", "Call"),
    ("instruction", "Do this"),
]


UNIT_VIEW_DIST = [
    ("grain_id", "Distributor"),
    ("city", "City"),
    ("n_call", "Call"),
    ("n_convert", "Convert"),
    ("n_lift", "Lift drop"),
    ("week_target_mt", "This week (MT)"),
    ("remaining_mt", "Still to Expected (MT)"),
    ("behind_pace_mt", "Behind pace (MT)"),
    ("billed_mt", "Billed (MT)"),
    ("expected_mt", "Expected (MT)"),
    ("instruction", "Do this"),
]


UNIT_VIEW_DSR = [
    ("grain_id", "DSR"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("n_call", "Call"),
    ("n_convert", "Convert"),
    ("n_lift", "Lift drop"),
    ("week_target_mt", "This week (MT)"),
    ("remaining_mt", "Still to Expected (MT)"),
    ("behind_pace_mt", "Behind pace (MT)"),
    ("instruction", "Do this"),
]


BACKTEST_VIEW = [
    ("n_months", "Closed months tested"),
    ("cut_day", "Cut day"),
    ("curve_precision_at_50", "Curve precision@50"),
    ("calendar_precision_at_50", "Calendar precision@50"),
    ("curve_catch_mt", "Curve catch (MT)"),
    ("calendar_catch_mt", "Calendar catch (MT)"),
    ("n_backloaded_hold", "Back-loaded left alone"),
    ("n_backloaded_ok", "Of those, finished OK"),
    ("remaining_mae_mt", "Remaining MAE (MT)"),
    ("notes", "How to read"),
]


def _present(df: pd.DataFrame, view: list[tuple[str, str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[label for _, label in view])
    out = {}
    mt_int = {
        "Billed (MT)",
        "Expected this month (MT)",
        "Should have by today (MT)",
        "Behind pace (MT)",
        "Still to Expected (MT)",
        "This week (MT)",
        "Expected (MT)",
        "Should have (MT)",
        "Curve catch (MT)",
        "Calendar catch (MT)",
        "Remaining MAE (MT)",
    }
    for src, label in view:
        if src not in df.columns:
            out[label] = [None] * len(df)
            continue
        col = df[src]
        if label in mt_int:
            out[label] = [_round_mt(v) for v in col]
        elif label == "Typical drop (MT)":
            out[label] = [_round_drop(v) for v in col]
        elif label == "Usual bill day":
            out[label] = [None if pd.isna(v) else int(round(float(v))) for v in col]
        elif label in {"Curve precision@50", "Calendar precision@50"}:
            out[label] = [None if pd.isna(v) else round(float(v) * 100) for v in col]
        else:
            out[label] = list(col)
    return pd.DataFrame(out)


def _present_country(country: dict[str, Any]) -> pd.DataFrame:
    return _present(pd.DataFrame([country]), COUNTRY_VIEW)


def _present_shops(df: pd.DataFrame) -> pd.DataFrame:
    return _present(df, SHOP_VIEW)


def _present_units(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    view = UNIT_VIEW_DIST if grain == "Distributor" else UNIT_VIEW_DSR
    return _present(df, view)


def _present_backtest(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[l for _, l in BACKTEST_VIEW])
    return _present(df, BACKTEST_VIEW)


def _take_present(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    if shops is None or shops.empty or "Action" not in shops.columns:
        return shops if shops is not None else pd.DataFrame()
    if action == ACTION_CALL:
        part = shops[shops["Action"].isin({ACTION_CALL, ACTION_RECOVER})]
    else:
        part = shops[shops["Action"] == action]
    return part.head(n)


def _raw_shops_to_sql(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["period"] = period
    cols = [
        "period",
        "store_id",
        "store_name",
        "city",
        "distributor",
        "dsr_name",
        "section",
        "action",
        "instruction",
        "billed_mt",
        "expected_mt",
        "ams_3m",
        "should_have_mt",
        "behind_pace_mt",
        "remaining_mt",
        "week_target_mt",
        "typical_drop_mt",
        "typical_bill_day",
        "pace_frac",
        "last_bill_date",
        "days_since_bill",
        "call_status",
        "visits",
        "value_score",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = None
    return out[cols]


def _raw_units_to_sql(df: pd.DataFrame, period: str, grain: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["period"] = period
    out["grain"] = grain
    if "distributor" not in out.columns:
        out["distributor"] = out["grain_id"] if grain == "distributor" else ""
    cols = [
        "period",
        "grain",
        "grain_id",
        "city",
        "distributor",
        "n_call",
        "n_convert",
        "n_lift",
        "n_hold",
        "billed_mt",
        "expected_mt",
        "should_have_mt",
        "behind_pace_mt",
        "remaining_mt",
        "week_target_mt",
        "instruction",
        "value_score",
    ]
    for col in cols:
        if col not in out.columns:
            out[col] = None
    return out[cols]


def _backtest_to_sql(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # May already be presented.
    if "Closed months tested" in df.columns:
        return pd.DataFrame(
            [
                {
                    "period": period,
                    "cut_day": df.iloc[0].get("Cut day"),
                    "n_months": df.iloc[0].get("Closed months tested"),
                    "n_shops": None,
                    "curve_precision_at_50": (df.iloc[0].get("Curve precision@50") or 0) / 100.0
                    if df.iloc[0].get("Curve precision@50") is not None
                    else None,
                    "calendar_precision_at_50": (df.iloc[0].get("Calendar precision@50") or 0) / 100.0
                    if df.iloc[0].get("Calendar precision@50") is not None
                    else None,
                    "curve_catch_mt": df.iloc[0].get("Curve catch (MT)"),
                    "calendar_catch_mt": df.iloc[0].get("Calendar catch (MT)"),
                    "n_backloaded_hold": df.iloc[0].get("Back-loaded left alone"),
                    "n_backloaded_ok": df.iloc[0].get("Of those, finished OK"),
                    "remaining_mae_mt": df.iloc[0].get("Remaining MAE (MT)"),
                    "notes": df.iloc[0].get("How to read"),
                }
            ]
        )
    out = df.copy()
    out["period"] = period
    return out


def _brief_to_sql(pack: ActionPack) -> dict[str, Any]:
    b = pack.brief or {}
    return {
        "period": pack.period,
        "as_of_day": pack.as_of_day,
        "days_in_month": pack.days_in_month,
        "days_left": pack.days_left,
        "billed_mt": b.get("billed_mt"),
        "expected_mt": b.get("expected_mt"),
        "should_have_mt": b.get("should_have_mt"),
        "behind_pace_mt": b.get("behind_pace_mt"),
        "remaining_mt": b.get("remaining_mt"),
        "week_target_mt": b.get("week_target_mt"),
        "n_call": b.get("n_call"),
        "n_convert": b.get("n_convert"),
        "n_lift": b.get("n_lift"),
        "has_daily": int(pack.has_daily),
        "headline": pack.headline,
        "source": pack.source,
        "metrics_json": pd.Series(b).to_json(),
    }


def _sql_shops_to_present(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[l for _, l in SHOP_VIEW])
    return _present(df, SHOP_VIEW)


def _sql_units_to_present(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return _present_units(df, grain)


# Keep unused import referenced for type checkers / future expected blend.
_ = shift_period
