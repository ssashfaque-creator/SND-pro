"""This-week action engine: who is due to order, who still has cover, who is fading.

Daily billed days teach each door’s replenishment cycle (days between drops) and
typical drop size, shrunk shop → DSR → city so thin history borrows the parent.
Last drop ÷ daily run-rate is remaining cover. A shop that took two months of
stock last time is not due. A shop that usually buys every 15 days and is on
day 16 with no bill is due — especially if nobody visited.

Expected stays the last-three-closed-month run-rate. This engine does not use
day-of-month seasonality.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sndintel.io_utils import prior_periods, shift_period
from sndintel.mtd import period_state
from sndintel.season import fit_seasonality, fit_shop_expected

SHOP_FLOOR_MT = 0.25
SHRINK_K = 4.0
SUMMARY_DIST_N = 15
SUMMARY_DSR_N = 15
SUMMARY_CALL_N = 80
SUMMARY_CONVERT_N = 40
SUMMARY_AGAIN_N = 40
SUMMARY_LAPSE_N = 40
BACKTEST_CUT_DAY = 15
BACKTEST_MONTHS = 6
COVER_HOLD_DAYS = 7
LOADED_MONTHS = 1.6
DECLINE_PCT = -0.25
LAPSE_CYCLES = 2.0

ACTION_CALL = "Due"
ACTION_CONVERT = "Due · visited"
ACTION_LIFT = "Another visit"
ACTION_RECOVER = "Lapsing"
ACTION_HOLD = "Hold"


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
    source: str = "monthly"
    country: pd.DataFrame = field(default_factory=pd.DataFrame)
    distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    converts: pd.DataFrame = field(default_factory=pd.DataFrame)
    lifts: pd.DataFrame = field(default_factory=pd.DataFrame)
    holds: pd.DataFrame = field(default_factory=pd.DataFrame)
    lapses: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    all_dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    brief: dict[str, Any] = field(default_factory=dict)
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
    """Score every AMS>0 door for due / another-visit / lapsing / hold."""
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
    as_of_ts = _as_of_timestamp(period, as_of, days_in_month)

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
    has_daily = not daily.empty
    source = "cycle" if has_daily else "monthly"
    shops = _attach_trend(shops, shop_month, period)
    shops = _attach_last_month(shops, shop_month, period)
    shops = _attach_last_billed(shops, shop_month, period)
    shops = attach_cycles(shops, daily, as_of_ts)
    shops["remaining_mt"] = (shops["expected_mt"] - shops["billed_mt"]).clip(lower=0)
    shops["light_mt"] = shops["remaining_mt"]
    shops["should_have_mt"] = shops["expected_mt"]
    shops["behind_pace_mt"] = shops["light_mt"]
    shops = _attach_calls(shops, visits, period)
    shops = _classify_actions(shops, open_mtd)
    shops["week_target_mt"] = _ask_mt(shops)
    shops["value_score"] = _value_score(shops)
    shops = shops.sort_values(["value_score", "week_target_mt", "days_overdue"], ascending=False)

    dist = _units_with_work(_roll_units(shops, "distributor", extra_city=True))
    dsr = _units_with_work(_roll_units(shops, "dsr_name", extra_city=True, extra_dist=True))
    country = _country_row(shops, as_of, days_in_month, days_left, open_mtd, source)
    backtest = backtest_cycles(daily, shop_month, stores, period) if has_daily else pd.DataFrame()
    headline = _headline(country, open_mtd, has_daily)

    pack = ActionPack(
        period=period,
        label=mtd.get("label") or period,
        as_of_day=as_of,
        days_in_month=days_in_month,
        days_left=days_left,
        open_mtd=open_mtd,
        has_daily=has_daily,
        headline=headline,
        source=source,
        country=_present_country(country),
        distributors=_present_units(dist.head(SUMMARY_DIST_N), "Distributor"),
        dsrs=_present_units(dsr.head(SUMMARY_DSR_N), "DSR"),
        calls=_present_shops(_take_action(shops, ACTION_CALL, SUMMARY_CALL_N)),
        converts=_present_shops(_take_action(shops, ACTION_CONVERT, SUMMARY_CONVERT_N)),
        lifts=_present_shops(_take_action(shops, ACTION_LIFT, SUMMARY_AGAIN_N)),
        lapses=_present_shops(_take_action(shops, ACTION_RECOVER, SUMMARY_LAPSE_N)),
        holds=_present_shops(shops[shops["action"] == ACTION_HOLD].head(20)),
        all_shops=_present_shops(shops),
        all_distributors=_present_units(dist, "Distributor"),
        all_dsrs=_present_units(dsr, "DSR"),
        backtest=_present_backtest(backtest),
        brief=country,
        raw_shops=shops,
        raw_distributors=dist,
        raw_dsrs=dsr,
        raw_backtest=backtest,
    )
    return pack


def attach_cycles(shops: pd.DataFrame, shop_day: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """Usual days between drops, last drop, cover left, days since last bill."""
    out = shops.copy()
    out["cycle_days"] = np.nan
    out["typical_drop_mt"] = np.nan
    out["last_drop_mt"] = np.nan
    out["last_bill_date"] = ""
    out["days_since_bill"] = np.nan
    out["cover_left_days"] = np.nan
    out["n_intervals"] = 0
    raw: dict[str, dict[str, float]] = {}
    if shop_day is not None and not shop_day.empty:
        work = shop_day.copy()
        work["sale_date"] = pd.to_datetime(work["sale_date"], errors="coerce")
        work = work[work["sale_date"].notna() & (work["sale_date"] <= as_of_ts)]
        work = work[pd.to_numeric(work["volume_mt"], errors="coerce").fillna(0) > 0]
        for sid, g in work.groupby(work["store_id"].astype(str)):
            g = g.sort_values("sale_date")
            dates = g["sale_date"].drop_duplicates()
            gaps = dates.diff().dt.days.dropna()
            gaps = gaps[(gaps >= 2) & (gaps <= 120)]
            last_row = g.iloc[-1]
            raw[str(sid)] = {
                "cycle": float(gaps.median()) if len(gaps) else np.nan,
                "n": int(len(gaps)),
                "drop": float(g["volume_mt"].median()),
                "last_drop": float(last_row["volume_mt"]),
                "last_date": last_row["sale_date"],
            }

    parents = _parent_cycle_priors(out, raw)
    for i, row in out.iterrows():
        sid = str(row["store_id"])
        city = str(row.get("city") or "")
        dsr = str(row.get("dsr_name") or "")
        prior_cycle, prior_drop = parents.get((city, dsr), parents.get((city, ""), (30.0, float(row.get("ams_3m") or 1.0) / 2)))
        got = raw.get(sid)
        n = float(got["n"]) if got else 0.0
        cred = n / (n + SHRINK_K) if n else 0.0
        cycle = cred * got["cycle"] + (1 - cred) * prior_cycle if got and pd.notna(got["cycle"]) else prior_cycle
        drop = cred * got["drop"] + (1 - cred) * prior_drop if got and pd.notna(got["drop"]) else prior_drop
        last_drop = got["last_drop"] if got else float(row.get("last_billed_mt") or row.get("last_month_mt") or 0)
        last_date = got["last_date"] if got else None
        if last_date is None:
            last_date = _month_end_or_as_of(str(row.get("last_billed_period") or ""), as_of_ts)
        days_since = float((as_of_ts - pd.Timestamp(last_date)).days) if last_date is not None else np.nan
        ams = float(row.get("ams_3m") or 0) or float(row.get("expected_mt") or 0)
        daily_rate = max(ams / 30.0, drop / max(cycle, 1.0) if drop and cycle else 0.01, 0.01)
        stock_days = float(last_drop) / daily_rate if last_drop and daily_rate else np.nan
        cover_left = (stock_days - days_since) if pd.notna(stock_days) and pd.notna(days_since) else np.nan
        out.at[i, "cycle_days"] = float(cycle)
        out.at[i, "typical_drop_mt"] = float(drop) if pd.notna(drop) else np.nan
        out.at[i, "last_drop_mt"] = float(last_drop) if last_drop else np.nan
        out.at[i, "last_bill_date"] = pd.Timestamp(last_date).strftime("%Y-%m-%d") if last_date is not None else ""
        out.at[i, "days_since_bill"] = days_since
        out.at[i, "cover_left_days"] = cover_left
        out.at[i, "n_intervals"] = int(n)
        out.at[i, "typical_bill_day"] = float(cycle)

    last_month = pd.to_numeric(out.get("last_month_mt"), errors="coerce")
    ams = pd.to_numeric(out["ams_3m"], errors="coerce").replace(0, np.nan)
    loaded_cover = last_month / (ams / 30.0)
    days_since = pd.to_numeric(out["days_since_bill"], errors="coerce")
    from_month = loaded_cover - days_since
    use_month = last_month >= (LOADED_MONTHS * ams.fillna(0))
    out["cover_left_days"] = np.where(use_month.fillna(False) & from_month.notna(), from_month, out["cover_left_days"])
    out["days_overdue"] = (days_since - pd.to_numeric(out["cycle_days"], errors="coerce")).clip(lower=0)
    out["pace_frac"] = np.nan
    return out


def _parent_cycle_priors(shops: pd.DataFrame, raw: dict[str, dict[str, float]]) -> dict[tuple[str, str], tuple[float, float]]:
    rows = []
    for i, row in shops.iterrows():
        sid = str(row["store_id"])
        got = raw.get(sid)
        if not got or pd.isna(got.get("cycle")):
            continue
        rows.append(
            {
                "city": str(row.get("city") or ""),
                "dsr_name": str(row.get("dsr_name") or ""),
                "cycle": got["cycle"],
                "drop": got["drop"],
            }
        )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    nat_c = float(frame["cycle"].median())
    nat_d = float(frame["drop"].median())
    city = frame.groupby("city")[["cycle", "drop"]].median()
    dsr = frame.groupby(["city", "dsr_name"])[["cycle", "drop"]].median()
    out: dict[tuple[str, str], tuple[float, float]] = {("", ""): (nat_c, nat_d)}
    for c, row in city.iterrows():
        out[(str(c), "")] = (float(row["cycle"]), float(row["drop"]))
    for _, row in shops.iterrows():
        c = str(row.get("city") or "")
        d = str(row.get("dsr_name") or "")
        if (c, d) in dsr.index:
            out[(c, d)] = (float(dsr.loc[(c, d), "cycle"]), float(dsr.loc[(c, d), "drop"]))
        elif (c, "") in out:
            out[(c, d)] = out[(c, "")]
        else:
            out[(c, d)] = (nat_c, nat_d)
    return out


def backtest_cycles(
    shop_day: pd.DataFrame,
    shop_month: pd.DataFrame,
    stores: pd.DataFrame | None,
    period: str,
    cut_day: int = BACKTEST_CUT_DAY,
    n_months: int = BACKTEST_MONTHS,
) -> pd.DataFrame:
    """At day 15 of closed months: did 'due' shops actually bill in the next 14 days?"""
    if shop_day is None or shop_day.empty or shop_month is None or shop_month.empty:
        return pd.DataFrame()
    daily = _prepare_daily(shop_day)
    months = sorted(p for p in daily["period"].astype(str).unique() if p < str(period))
    if len(months) < 4:
        return pd.DataFrame()
    rows = []
    for test in months[-n_months:]:
        days = _days_in_period(test)
        cut_ts = _as_of_timestamp(test, min(cut_day, days), days)
        hist = daily[pd.to_datetime(daily["sale_date"], errors="coerce") <= cut_ts]
        expected = _shop_expected_full(shop_month, test)
        billed = _period_billed(shop_month, test)
        shops = _shop_frame(expected, billed, stores, shop_month, test)
        if shops.empty:
            continue
        shops = _attach_last_month(shops, shop_month, test)
        shops = _attach_last_billed(shops, shop_month, test)
        shops = attach_cycles(shops, hist, cut_ts)
        shops["remaining_mt"] = (shops["expected_mt"] - shops["billed_mt"]).clip(lower=0)
        future = daily[
            (daily["period"].astype(str) == test)
            & (pd.to_numeric(daily["day"], errors="coerce") > cut_day)
            & (pd.to_numeric(daily["day"], errors="coerce") <= cut_day + 14)
        ]
        billed_next = set(future["store_id"].astype(str)) if not future.empty else set()
        due = shops[
            (pd.to_numeric(shops["days_since_bill"], errors="coerce") >= pd.to_numeric(shops["cycle_days"], errors="coerce") * 0.9)
            & (pd.to_numeric(shops["cover_left_days"], errors="coerce").fillna(0) <= COVER_HOLD_DAYS)
            & (pd.to_numeric(shops["expected_mt"], errors="coerce") >= SHOP_FLOOR_MT)
        ]
        loaded = shops[
            (pd.to_numeric(shops["last_month_mt"], errors="coerce") >= LOADED_MONTHS * pd.to_numeric(shops["ams_3m"], errors="coerce"))
            & (pd.to_numeric(shops["cover_left_days"], errors="coerce") > COVER_HOLD_DAYS)
        ]
        n_due = int(len(due))
        hit = int(due["store_id"].astype(str).isin(billed_next).sum()) if n_due else 0
        base = int(shops["store_id"].astype(str).isin(billed_next).mean() * n_due) if n_due and len(shops) else 0
        quiet = 0
        if not loaded.empty:
            rest = daily[(daily["period"].astype(str) == test) & (pd.to_numeric(daily["day"], errors="coerce") > cut_day)]
            later = rest.groupby(rest["store_id"].astype(str))["volume_mt"].sum() if not rest.empty else pd.Series(dtype=float)
            quiet = int((loaded["store_id"].astype(str).map(later).fillna(0) < SHOP_FLOOR_MT).sum())
        rows.append(
            {
                "period": test,
                "cut_day": cut_day,
                "n_shops": int(len(shops)),
                "n_due": n_due,
                "due_precision": (hit / n_due) if n_due else 0.0,
                "baseline_precision": (base / n_due) if n_due else 0.0,
                "n_loaded_hold": int(len(loaded)),
                "n_loaded_quiet": quiet,
            }
        )
    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "period": period,
                "cut_day": cut_day,
                "n_months": int(len(detail)),
                "n_shops": int(detail["n_shops"].sum()),
                "due_precision": float(detail["due_precision"].mean()),
                "baseline_precision": float(detail["baseline_precision"].mean()),
                "n_loaded_hold": int(detail["n_loaded_hold"].sum()),
                "n_loaded_quiet": int(detail["n_loaded_quiet"].sum()),
                "notes": (
                    "At day 15 of each closed month, mark shops whose usual cycle has elapsed and "
                    "who no longer have cover from the last drop. Precision is the share that billed "
                    "in the next 14 days. Baseline is the same hit-rate if we picked that many shops "
                    "at random. Loaded hold = last month was ≥1.6× AMS; quiet means they stayed below "
                    "0.25 MT for the rest of the month."
                ),
            }
        ]
    )


def persist_action_pack(conn, pack: ActionPack) -> None:
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
    from sndintel.storage import read_sql

    brief = read_sql(conn, "SELECT * FROM action_brief")
    if brief is None or brief.empty:
        return empty_action_pack(period or "")
    if period:
        part = brief[brief["period"].astype(str) == str(period)]
        if not part.empty:
            brief = part
    row = brief.iloc[-1]
    period = str(row["period"])
    shops = read_sql(conn, "SELECT * FROM action_shops WHERE period = ?", (period,))
    units = read_sql(conn, "SELECT * FROM action_units WHERE period = ?", (period,))
    back = read_sql(conn, "SELECT * FROM action_backtest WHERE period = ?", (period,))
    dists = units[units["grain"] == "distributor"] if units is not None and not units.empty else pd.DataFrame()
    dsrs = units[units["grain"] == "dsr"] if units is not None and not units.empty else pd.DataFrame()
    shops_p = _present_shops(shops) if shops is not None and not shops.empty and "store_name" in shops.columns else _sql_shops_to_present(shops)
    if shops is not None and not shops.empty and "Shop" in shops.columns:
        shops_p = shops
    elif shops is not None and not shops.empty and "action" in shops.columns:
        shops_p = _present_shops(shops)
    dist_p = _present_units(dists, "Distributor") if dists is not None and not dists.empty and "grain_id" in dists.columns else pd.DataFrame()
    dsr_p = _present_units(dsrs, "DSR") if dsrs is not None and not dsrs.empty and "grain_id" in dsrs.columns else pd.DataFrame()
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
        country=_present_country(dict(row)),
        distributors=dist_p.head(SUMMARY_DIST_N),
        dsrs=dsr_p.head(SUMMARY_DSR_N),
        calls=_take_present(shops_p, ACTION_CALL, SUMMARY_CALL_N),
        converts=_take_present(shops_p, ACTION_CONVERT, SUMMARY_CONVERT_N),
        lifts=_take_present(shops_p, ACTION_LIFT, SUMMARY_AGAIN_N),
        lapses=_take_present(shops_p, ACTION_RECOVER, SUMMARY_LAPSE_N),
        holds=_take_present(shops_p, ACTION_HOLD, 20),
        all_shops=shops_p,
        all_distributors=dist_p,
        all_dsrs=dsr_p,
        backtest=_present_backtest(back),
        brief=dict(row),
    )


def _days_in_period(period: str) -> int:
    return monthrange(int(period[:4]), int(period[5:7]))[1]


def _month_end_or_as_of(period: str, as_of_ts: pd.Timestamp) -> pd.Timestamp | None:
    if not period or len(period) < 7 or not period[:4].isdigit():
        return None
    try:
        year, month = int(period[:4]), int(period[5:7])
        last = pd.Timestamp(year=year, month=month, day=monthrange(year, month)[1])
    except (TypeError, ValueError):
        return None
    return as_of_ts if last > as_of_ts else last


def _as_of_timestamp(period: str, as_of: int, days: int) -> pd.Timestamp:
    year, month = int(period[:4]), int(period[5:7])
    day = int(min(max(as_of, 1), days))
    return pd.Timestamp(year=year, month=month, day=day)


def _as_of_day(shop_day: pd.DataFrame | None, period: str, mtd: dict[str, Any], days: int) -> int:
    if mtd.get("as_of_day"):
        return int(min(max(int(mtd["as_of_day"]), 1), days))
    if shop_day is not None and not shop_day.empty:
        cur = shop_day[shop_day["period"].astype(str) == str(period)]
        if not cur.empty and "day" in cur.columns:
            last = pd.to_numeric(cur["day"], errors="coerce").max()
            if pd.notna(last):
                return int(min(max(int(last), 1), days))
    return days


def _prepare_daily(shop_day: pd.DataFrame | None) -> pd.DataFrame:
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    out = shop_day.copy()
    out["store_id"] = out["store_id"].astype(str)
    out["volume_mt"] = pd.to_numeric(out["volume_mt"], errors="coerce").fillna(0.0)
    out["period"] = out["period"].astype(str)
    out["sale_date"] = pd.to_datetime(out.get("sale_date"), errors="coerce")
    if "day" not in out.columns or out["day"].isna().all():
        out["day"] = out["sale_date"].dt.day
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out = out[out["sale_date"].notna() & (out["volume_mt"] > 0)]
    for col in ("city", "dsr_name", "distributor", "store_name", "section"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out


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
    del period
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


def _attach_trend(shops: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    out = shops.copy()
    recent = prior_periods(period, 3)
    prior = prior_periods(shift_period(period, -3), 3)
    hist = shop_month.copy()
    hist["store_id"] = hist["store_id"].astype(str)
    r = hist[hist["period"].astype(str).isin(recent)].groupby("store_id")["volume_mt"].mean()
    p = hist[hist["period"].astype(str).isin(prior)].groupby("store_id")["volume_mt"].mean()
    out["recent3_mt"] = out["store_id"].map(r).fillna(0.0)
    out["prior3_mt"] = out["store_id"].map(p)
    out["trend_pct"] = np.where(out["prior3_mt"] > 0, (out["recent3_mt"] - out["prior3_mt"]) / out["prior3_mt"], np.nan)
    return out


def _attach_last_month(shops: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    out = shops.copy()
    prev = shift_period(period, -1)
    last = shop_month[shop_month["period"].astype(str) == prev]
    if last.empty:
        out["last_month_mt"] = 0.0
        return out
    vol = last.groupby(last["store_id"].astype(str))["volume_mt"].sum()
    out["last_month_mt"] = out["store_id"].map(vol).fillna(0.0)
    return out


def _attach_last_billed(shops: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    """Most recent billed month — so a door quiet for 60 days is not treated as unknown."""
    out = shops.copy()
    out["last_billed_period"] = ""
    out["last_billed_mt"] = 0.0
    if shop_month is None or shop_month.empty:
        return out
    hist = shop_month.copy()
    hist["store_id"] = hist["store_id"].astype(str)
    hist["volume_mt"] = pd.to_numeric(hist["volume_mt"], errors="coerce").fillna(0.0)
    hist = hist[(hist["period"].astype(str) <= str(period)) & (hist["volume_mt"] > 0)]
    if hist.empty:
        return out
    last = hist.sort_values("period").drop_duplicates("store_id", keep="last")
    out["last_billed_period"] = out["store_id"].map(dict(zip(last["store_id"], last["period"].astype(str)))).fillna("")
    out["last_billed_mt"] = out["store_id"].map(dict(zip(last["store_id"], last["volume_mt"]))).fillna(0.0)
    return out


def _attach_calls(shops: pd.DataFrame, visits: pd.DataFrame | None, period: str) -> pd.DataFrame:
    out = shops.copy()
    out["visits"] = 0
    if visits is not None and not visits.empty and "store_id" in visits.columns:
        v = visits.copy()
        if "period" in v.columns:
            v = v[v["period"].astype(str) == str(period)]
        if not v.empty and "visits" in v.columns:
            g = v.groupby(v["store_id"].astype(str), as_index=False)["visits"].sum()
            out["visits"] = out["store_id"].map(dict(zip(g["store_id"].astype(str), pd.to_numeric(g["visits"], errors="coerce").fillna(0)))).fillna(0)
    billed = pd.to_numeric(out["billed_mt"], errors="coerce").fillna(0) > 0
    visited = pd.to_numeric(out["visits"], errors="coerce").fillna(0) > 0
    out["call_status"] = "Unvisited"
    out.loc[visited & ~billed, "call_status"] = "Visited · not billed"
    out.loc[billed, "call_status"] = "Billed"
    return out


def _classify_actions(shops: pd.DataFrame, open_mtd: bool) -> pd.DataFrame:
    out = shops.copy()
    cycle = pd.to_numeric(out["cycle_days"], errors="coerce").replace(0, np.nan).fillna(30)
    days_since = pd.to_numeric(out["days_since_bill"], errors="coerce")
    cover = pd.to_numeric(out["cover_left_days"], errors="coerce")
    billed = pd.to_numeric(out["billed_mt"], errors="coerce").fillna(0)
    remaining = pd.to_numeric(out["remaining_mt"], errors="coerce").fillna(0)
    drop = pd.to_numeric(out["typical_drop_mt"], errors="coerce").fillna(0)
    trend = pd.to_numeric(out.get("trend_pct"), errors="coerce")
    last_month = pd.to_numeric(out.get("last_month_mt"), errors="coerce").fillna(0)
    ams = pd.to_numeric(out["ams_3m"], errors="coerce").fillna(0)
    visited = out["call_status"].eq("Visited · not billed")
    unvisited = out["call_status"].eq("Unvisited")

    expected = pd.to_numeric(out["expected_mt"], errors="coerce").fillna(0)
    has_cover = cover.fillna(0) > COVER_HOLD_DAYS
    loaded = (last_month >= (LOADED_MONTHS * ams)) & has_cover
    due = days_since.notna() & (days_since >= cycle * 0.9) & ~has_cover
    material = (remaining >= SHOP_FLOOR_MT) | (drop >= SHOP_FLOOR_MT) | (ams >= SHOP_FLOOR_MT)
    prior3 = pd.to_numeric(out.get("prior3_mt"), errors="coerce").fillna(0)
    unbilled = billed <= 0.005
    lapse = (days_since.fillna(0) >= (cycle * LAPSE_CYCLES)) & unbilled & ~has_cover
    lapse = lapse | (
        (trend <= DECLINE_PCT)
        & (prior3 >= SHOP_FLOOR_MT)
        & unbilled
        & (days_since.fillna(99) >= cycle)
        & ~has_cover
    )
    short_month = (~unbilled) & (remaining >= SHOP_FLOOR_MT) & (billed < 0.5 * expected.clip(lower=SHOP_FLOOR_MT))
    cycle_again = days_since.fillna(0) >= cycle * 0.8
    just_bought = days_since.fillna(0) < np.maximum(5.0, cycle * 0.5)
    stub = (~unbilled) & (drop > 0) & (billed < 0.4 * drop) & ~just_bought
    light = short_month & ~has_cover & (cycle_again | stub)

    out["action"] = ACTION_HOLD
    out.loc[material & due & unvisited & ~lapse, "action"] = ACTION_CALL
    out.loc[material & due & visited & ~lapse, "action"] = ACTION_CONVERT
    out.loc[material & light, "action"] = ACTION_LIFT
    out.loc[material & lapse, "action"] = ACTION_RECOVER
    out.loc[loaded & ~lapse, "action"] = ACTION_HOLD
    if not open_mtd:
        out.loc[out["action"].isin({ACTION_CALL, ACTION_CONVERT}), "action"] = ACTION_RECOVER
    out["instruction"] = [_instruction(r) for r in out.itertuples(index=False)]
    return out


def _instruction(row: Any) -> str:
    name = str(getattr(row, "store_name", "") or getattr(row, "store_id", ""))
    action = str(getattr(row, "action", "") or "")
    cycle = getattr(row, "cycle_days", None)
    since = getattr(row, "days_since_bill", None)
    cover = getattr(row, "cover_left_days", None)
    drop = float(getattr(row, "typical_drop_mt", 0) or 0)
    last_drop = float(getattr(row, "last_drop_mt", 0) or 0)
    billed = float(getattr(row, "billed_mt", 0) or 0)
    expected = float(getattr(row, "expected_mt", 0) or 0)
    last_month = float(getattr(row, "last_month_mt", 0) or 0)
    ams = float(getattr(row, "ams_3m", 0) or 0)
    last = str(getattr(row, "last_bill_date", "") or "")
    trend = getattr(row, "trend_pct", None)
    cycle_s = f"{int(round(float(cycle)))} days" if pd.notna(cycle) else "its usual gap"
    since_s = f"{int(round(float(since)))} days" if pd.notna(since) else "unknown"
    cover_s = f"{int(round(float(cover)))} days of cover left" if pd.notna(cover) else "cover unknown"
    last_bit = f" Last billed {last}." if last else ""
    if action == ACTION_CALL:
        return (
            f"Call {name}. Usually buys every {cycle_s}; it has been {since_s} with no bill. "
            f"Close about {_kg_text(drop)}. Not visited this month.{last_bit}"
        )
    if action == ACTION_CONVERT:
        return (
            f"{name} was visited and did not buy. Cycle is {cycle_s} and it has been {since_s}. "
            f"Close {_kg_text(drop)}."
        )
    if action == ACTION_LIFT:
        return (
            f"{name} billed only {_kg_text(billed)} this month versus {_kg_text(expected)} Expected. "
            f"Usual drop {_kg_text(drop)} every {cycle_s}; last drop {_kg_text(last_drop)} {since_s} ago. "
            f"Worth another visit."
        )
    if action == ACTION_RECOVER:
        fading = pd.notna(trend) and float(trend) <= DECLINE_PCT
        verb = "is fading" if fading else "has been quiet"
        trend_s = f" Last three months are {float(trend)*100:.0f}% versus the three before." if pd.notna(trend) else ""
        return (
            f"{name} {verb} — {since_s} since the last bill (usual cycle {cycle_s}).{trend_s} "
            f"Get this door back on the beat.{last_bit}"
        )
    if pd.notna(cover) and float(cover) > COVER_HOLD_DAYS:
        why = f"Last drop {_kg_text(last_drop)}" if last_drop else f"Last month {_kg_text(last_month)} versus AMS {_kg_text(ams)}"
        return f"Hold {name}. {why} — {cover_s}. Do not pull the beat here.{last_bit}"
    return f"{name} is inside its {cycle_s} cycle ({since_s} since last bill). Leave it."


def _kg_text(mt: Any) -> str:
    try:
        if mt is None or pd.isna(mt):
            return "0 KG"
        return f"{int(round(float(mt) * 1000)):,} KG"
    except (TypeError, ValueError):
        return "0 KG"


def _ask_mt(shops: pd.DataFrame) -> pd.Series:
    drop = pd.to_numeric(shops["typical_drop_mt"], errors="coerce").fillna(0)
    light = pd.to_numeric(shops["remaining_mt"], errors="coerce").fillna(0)
    ask = drop.where(shops["action"].isin({ACTION_CALL, ACTION_CONVERT, ACTION_RECOVER}), light)
    ask = ask.where(shops["action"] != ACTION_HOLD, 0.0)
    return ask.clip(lower=0)


def _value_score(shops: pd.DataFrame) -> pd.Series:
    ask = pd.to_numeric(shops["week_target_mt"], errors="coerce").fillna(0)
    overdue = pd.to_numeric(shops.get("days_overdue"), errors="coerce").fillna(0)
    cycle = pd.to_numeric(shops.get("cycle_days"), errors="coerce").replace(0, np.nan).fillna(30)
    cover = pd.to_numeric(shops.get("cover_left_days"), errors="coerce").fillna(0)
    trend = pd.to_numeric(shops.get("trend_pct"), errors="coerce").fillna(0)
    unvisited = shops["call_status"].eq("Unvisited")
    score = ask * (1.0 + overdue / cycle.clip(lower=7))
    score = score + (-trend.clip(upper=0) * pd.to_numeric(shops["ams_3m"], errors="coerce").fillna(0))
    score = score.where(~unvisited, score * 1.35)
    score = score.where(cover <= COVER_HOLD_DAYS, score * 0.1)
    score = score.where(shops["action"] != ACTION_HOLD, 0.0)
    return score.fillna(0.0)


def _roll_units(shops: pd.DataFrame, key: str, extra_city: bool = False, extra_dist: bool = False) -> pd.DataFrame:
    if shops is None or shops.empty or key not in shops.columns:
        return pd.DataFrame()
    work = shops[shops[key].astype(str).str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame()
    rows = []
    for name, g in work.groupby(key):
        n_call = int((g["action"] == ACTION_CALL).sum())
        n_convert = int((g["action"] == ACTION_CONVERT).sum())
        n_lift = int((g["action"] == ACTION_LIFT).sum())
        n_lapse = int((g["action"] == ACTION_RECOVER).sum())
        week = float(g["week_target_mt"].sum())
        instruction = (
            f"Push {name}: {n_call} due and unvisited, {n_convert} due but already visited, "
            f"{n_lift} need another visit (light this month), {n_lapse} lapsing. "
            f"Ask {_kg_text(week)} this week."
        )
        row = {
            "grain_id": str(name),
            "n_call": n_call,
            "n_convert": n_convert,
            "n_lift": n_lift,
            "n_lapse": n_lapse,
            "n_hold": int((g["action"] == ACTION_HOLD).sum()),
            "ams_3m": float(pd.to_numeric(g["ams_3m"], errors="coerce").fillna(0).sum()) if "ams_3m" in g.columns else 0.0,
            "billed_mt": float(g["billed_mt"].sum()),
            "expected_mt": float(g["expected_mt"].sum()),
            "should_have_mt": float(g["expected_mt"].sum()),
            "behind_pace_mt": float(g["remaining_mt"].sum()),
            "remaining_mt": float(g["remaining_mt"].sum()),
            "week_target_mt": week,
            "instruction": instruction,
            "value_score": float(g["value_score"].sum()),
        }
        if extra_city and "city" in g.columns:
            row["city"] = str(g["city"].mode().iloc[0]) if not g["city"].mode().empty else ""
        if extra_dist and "distributor" in g.columns:
            row["distributor"] = str(g["distributor"].mode().iloc[0]) if not g["distributor"].mode().empty else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["value_score", "week_target_mt"], ascending=False).reset_index(drop=True)


def _units_with_work(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    work = (
        pd.to_numeric(df.get("n_call"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_convert"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_lift"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_lapse"), errors="coerce").fillna(0)
    )
    return df.loc[work > 0].copy()


def _country_row(shops: pd.DataFrame, as_of: int, days: int, days_left: int, open_mtd: bool, source: str) -> dict[str, Any]:
    return {
        "billed_mt": float(shops["billed_mt"].sum()),
        "expected_mt": float(shops["expected_mt"].sum()),
        "ams_3m": float(pd.to_numeric(shops.get("ams_3m"), errors="coerce").fillna(0).sum()),
        "should_have_mt": float(shops["expected_mt"].sum()),
        "behind_pace_mt": float(shops["remaining_mt"].sum()),
        "remaining_mt": float(shops["remaining_mt"].sum()),
        "week_target_mt": float(shops["week_target_mt"].sum()),
        "as_of_day": as_of,
        "days_in_month": days,
        "days_left": days_left,
        "open_mtd": open_mtd,
        "n_call": int((shops["action"] == ACTION_CALL).sum()),
        "n_convert": int((shops["action"] == ACTION_CONVERT).sum()),
        "n_lift": int((shops["action"] == ACTION_LIFT).sum()),
        "n_lapse": int((shops["action"] == ACTION_RECOVER).sum()),
        "n_hold": int((shops["action"] == ACTION_HOLD).sum()),
        "source": source,
    }


def _headline(country: dict[str, Any], open_mtd: bool, has_daily: bool) -> str:
    n_call = int(country.get("n_call") or 0)
    n_convert = int(country.get("n_convert") or 0)
    n_lift = int(country.get("n_lift") or 0)
    n_lapse = int(country.get("n_lapse") or 0)
    week = float(country.get("week_target_mt") or 0)
    if not has_daily:
        return (
            f"No billed days in the warehouse — cycles fall back to monthly gaps. "
            f"{n_call} due, {n_convert} due and already visited, {n_lift} another visit, {n_lapse} lapsing ({_kg_text(week)})."
        )
    when = f"Day {country.get('as_of_day')}/{country.get('days_in_month')}" if open_mtd else "Month closed"
    return (
        f"{when}: {n_call} shops are due and unvisited, {n_convert} due but already seen, "
        f"{n_lift} bought too little and need another visit, {n_lapse} are lapsing ({_kg_text(week)} to ask)."
    )


def _take_action(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    return shops[shops["action"] == action].head(n)


COUNTRY_VIEW = [
    ("billed_mt", "Billed (KG)"),
    ("expected_mt", "Expected this month (KG)"),
    ("ams_3m", "AMS (KG)"),
    ("remaining_mt", "Still to Expected (KG)"),
    ("week_target_mt", "Ask this week (KG)"),
    ("as_of_day", "As of day"),
    ("days_left", "Days left"),
    ("n_call", "Due · unvisited"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
]

SHOP_VIEW = [
    ("store_name", "Shop"),
    ("store_id", "POP"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("dsr_name", "DSR"),
    ("action", "Action"),
    ("week_target_mt", "Ask (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("ams_3m", "AMS (KG)"),
    ("days_since_bill", "Days since bill"),
    ("cycle_days", "Usual cycle (days)"),
    ("days_overdue", "Days overdue"),
    ("cover_left_days", "Cover left (days)"),
    ("last_drop_mt", "Last drop (KG)"),
    ("typical_drop_mt", "Typical drop (KG)"),
    ("expected_mt", "Expected (KG)"),
    ("last_month_mt", "Last month (KG)"),
    ("trend_pct", "Trend vs prior 3m"),
    ("last_bill_date", "Last billed"),
    ("call_status", "Call"),
    ("instruction", "Do this"),
]

SHOP_PDF_COLS = [
    "Shop",
    "City",
    "DSR",
    "Action",
    "Ask (KG)",
    "Billed (KG)",
    "AMS (KG)",
    "Days since bill",
    "Usual cycle (days)",
    "Cover left (days)",
    "Do this",
]

UNIT_VIEW_DIST = [
    ("grain_id", "Distributor"),
    ("city", "City"),
    ("n_call", "Due"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
    ("ams_3m", "AMS (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("week_target_mt", "Ask this week (KG)"),
    ("remaining_mt", "Still to Expected (KG)"),
    ("instruction", "Do this"),
]

UNIT_VIEW_DSR = [
    ("grain_id", "DSR"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("n_call", "Due"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
    ("ams_3m", "AMS (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("week_target_mt", "Ask this week (KG)"),
    ("instruction", "Do this"),
]

BACKTEST_VIEW = [
    ("n_months", "Closed months tested"),
    ("cut_day", "Cut day"),
    ("due_precision", "Due precision %"),
    ("baseline_precision", "Random precision %"),
    ("n_loaded_hold", "Loaded left alone"),
    ("n_loaded_quiet", "Of those, stayed quiet"),
]


def _present(df: pd.DataFrame, view: list[tuple[str, str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[label for _, label in view])
    out = {}
    kg_cols = {
        "Billed (KG)",
        "Expected this month (KG)",
        "Still to Expected (KG)",
        "Ask this week (KG)",
        "Expected (KG)",
        "Last month (KG)",
        "Ask (KG)",
        "AMS (KG)",
        "Last drop (KG)",
        "Typical drop (KG)",
    }
    for src, label in view:
        if src not in df.columns:
            out[label] = [None] * len(df)
            continue
        col = df[src]
        if label in kg_cols:
            out[label] = [_round_kg(v) for v in col]
        elif label in {"Usual cycle (days)", "Days since bill", "Days overdue", "Cover left (days)"}:
            out[label] = [None if pd.isna(v) else int(round(float(v))) for v in col]
        elif label == "Trend vs prior 3m":
            out[label] = [None if pd.isna(v) else int(round(float(v) * 100)) for v in col]
        elif label in {"Due precision %", "Random precision %"}:
            out[label] = [None if pd.isna(v) else int(round(float(v) * 100)) for v in col]
        else:
            out[label] = list(col)
    return pd.DataFrame(out)


def _round_kg(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None


def _present_country(country: dict[str, Any]) -> pd.DataFrame:
    return _present(pd.DataFrame([country]), COUNTRY_VIEW)


def _present_shops(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[l for _, l in SHOP_VIEW])
    if "Shop" in df.columns:
        return df
    return _present(df, SHOP_VIEW)


def _present_units(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if grain == "Distributor" and "Distributor" in df.columns:
        return df
    if grain == "DSR" and "DSR" in df.columns:
        return df
    view = UNIT_VIEW_DIST if grain == "Distributor" else UNIT_VIEW_DSR
    return _present(df, view)


def _present_backtest(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[l for _, l in BACKTEST_VIEW])
    if "Due precision %" in df.columns or "Due precision" in df.columns:
        return df
    return _present(df, BACKTEST_VIEW)


def _take_present(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame()
    col = "Action" if "Action" in shops.columns else "action"
    if col not in shops.columns:
        return shops.head(0)
    return shops[shops[col] == action].head(n)


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
        "cycle_days",
        "days_overdue",
        "last_drop_mt",
        "cover_left_days",
        "light_mt",
        "trend_pct",
        "last_month_mt",
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
        "n_lapse",
        "n_hold",
        "ams_3m",
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
    if "Due precision %" in df.columns or "Due precision" in df.columns:
        row = df.iloc[0]
        due = row.get("Due precision %")
        if due is None:
            due = row.get("Due precision")
        rnd = row.get("Random precision %")
        if rnd is None:
            rnd = row.get("Random precision")
        return pd.DataFrame(
            [
                {
                    "period": period,
                    "cut_day": row.get("Cut day"),
                    "n_months": row.get("Closed months tested"),
                    "n_shops": None,
                    "curve_precision_at_50": (due or 0) / 100.0 if due is not None else None,
                    "calendar_precision_at_50": (rnd or 0) / 100.0 if rnd is not None else None,
                    "curve_catch_mt": None,
                    "calendar_catch_mt": None,
                    "n_backloaded_hold": row.get("Loaded left alone"),
                    "n_backloaded_ok": row.get("Of those, stayed quiet"),
                    "remaining_mae_mt": None,
                    "notes": row.get("How to read"),
                }
            ]
        )
    out = df.copy()
    out["period"] = period
    # Map new names onto existing columns so old warehouses still load.
    if "due_precision" in out.columns and "curve_precision_at_50" not in out.columns:
        out["curve_precision_at_50"] = out["due_precision"]
    if "baseline_precision" in out.columns and "calendar_precision_at_50" not in out.columns:
        out["calendar_precision_at_50"] = out["baseline_precision"]
    if "n_loaded_hold" in out.columns and "n_backloaded_hold" not in out.columns:
        out["n_backloaded_hold"] = out["n_loaded_hold"]
    if "n_loaded_quiet" in out.columns and "n_backloaded_ok" not in out.columns:
        out["n_backloaded_ok"] = out["n_loaded_quiet"]
    keep = [
        "period",
        "cut_day",
        "n_months",
        "n_shops",
        "curve_precision_at_50",
        "calendar_precision_at_50",
        "curve_catch_mt",
        "calendar_catch_mt",
        "n_backloaded_hold",
        "n_backloaded_ok",
        "remaining_mae_mt",
        "notes",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def _brief_to_sql(pack: ActionPack) -> dict[str, Any]:
    b = pack.brief or {}
    return {
        "period": pack.period,
        "as_of_day": pack.as_of_day,
        "days_in_month": pack.days_in_month,
        "days_left": pack.days_left,
        "billed_mt": b.get("billed_mt"),
        "expected_mt": b.get("expected_mt"),
        "ams_3m": b.get("ams_3m"),
        "should_have_mt": b.get("should_have_mt"),
        "behind_pace_mt": b.get("behind_pace_mt"),
        "remaining_mt": b.get("remaining_mt"),
        "week_target_mt": b.get("week_target_mt"),
        "n_call": b.get("n_call"),
        "n_convert": b.get("n_convert"),
        "n_lift": b.get("n_lift"),
        "n_lapse": b.get("n_lapse"),
        "has_daily": int(pack.has_daily),
        "headline": pack.headline,
        "source": pack.source,
        "metrics_json": pd.Series(b).to_json(),
    }


def _sql_shops_to_present(df: pd.DataFrame) -> pd.DataFrame:
    return _present_shops(df)
