"""This-week action engine: demand-driven Ask from each shop’s depletion cycle.

Ask is decoupled from monthly pacing. A rolling 90-day window teaches API
(median days between purchases) and expected drop (median invoice). Cold-start
shops with 0 history contribute 0; 1–2 purchases use whatever invoices exist
(default API = 14 days after a first bill). Due = depletion ratio ≥ 0.8.
Lapsed = DSLP > 3 × API — Ask is reset to 0 and the door leaves the beat.

Official Expected (last-3 AMS blended with last-6 median, then the national
day curve) is unchanged and is not multiplied into Ask. Pipeline identity:

    Pipeline Expected = Billed + Due unvisited + Drop variance + Not yet due
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sndintel.demand import (
    ACTION_CALL,
    ACTION_CONVERT,
    ACTION_HOLD,
    ACTION_LIFT,
    ACTION_RECOVER,
    DUE_RATIO,
    LAPSE_MULTIPLIER,
    attach_demand_cycles,
    attach_pipeline,
    classify_demand_actions,
)
from sndintel.io_utils import prior_periods, shift_period
from sndintel.mtd import period_state
from sndintel.nextdrop import attach_next_drop
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
LAPSE_CYCLES = LAPSE_MULTIPLIER

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
    pipeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    sales_head: pd.DataFrame = field(default_factory=pd.DataFrame)
    beat: pd.DataFrame = field(default_factory=pd.DataFrame)
    lost_doors: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    """Score every universe door for due / another-visit / lapsing / hold.

    Ask is the 90-day expected drop when the depletion ratio is ≥ 0.8 and the
    shop is not lapsed. Official Expected is still last-3 / last-6 for Gap.
    0-history universe doors stay Due with Ask 0 so they count in visit %.
    """
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
    source = "demand" if has_daily else "monthly"
    shops = _attach_trend(shops, shop_month, period)
    shops = _attach_last_month(shops, shop_month, period)
    shops = _attach_last_billed(shops, shop_month, period)
    shops = attach_cycles(shops, daily, as_of_ts)
    shops["remaining_mt"] = (shops["expected_mt"] - shops["billed_mt"]).clip(lower=0)
    shops["light_mt"] = shops["remaining_mt"]
    shops["should_have_mt"] = shops["expected_mt"]
    shops["behind_pace_mt"] = shops["light_mt"]
    shops = attach_next_drop(shops, daily, as_of_ts, shop_month)
    shops = _attach_calls(shops, visits, period)
    shops = _classify_actions(shops, open_mtd)
    from sndintel.capacity import apply_city_driver_priority, cap_shops_per_dsr, score_dsr_capacity

    shops = apply_city_driver_priority(shops)
    shops = _attach_rest_of_month(shops, days_left, open_mtd, days_in_month)
    shops["instruction"] = [_instruction(r) for r in shops.itertuples(index=False)]
    shops["value_score"] = _value_score(shops)
    shops = shops.sort_values(["value_score", "week_target_mt", "days_overdue"], ascending=False)

    dist = _units_with_work(_roll_units(shops, "distributor", extra_city=True))
    dsr = _units_with_work(_roll_dsrs(shops))
    cap = score_dsr_capacity(shops, as_of, days_in_month, days_left)
    if not dsr.empty and not cap.empty:
        dsr = dsr.merge(cap[["grain_id", "label", "span_unique", "day_cap"]], on="grain_id", how="left")
        if "label" in dsr.columns:
            dsr["instruction"] = [
                f"{lab} · {inst}" if pd.notna(lab) and str(lab).strip() else inst
                for lab, inst in zip(dsr["label"], dsr["instruction"])
            ]
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
        pipeline=_present_pipeline(shops),
        sales_head=_present_sales_head(shops),
        beat=_present_beat_plan(shops),
        lost_doors=_present_lost_doors(shops),
    )
    return pack


def attach_cycles(shops: pd.DataFrame, shop_day: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """Usual days between drops and expected drop from a rolling 90-day window."""
    return attach_demand_cycles(shops, shop_day, as_of_ts)


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
        shops = _attach_calls(shops, None, test)
        shops = classify_demand_actions(shops, open_mtd=True)
        future = daily[
            (daily["period"].astype(str) == test)
            & (pd.to_numeric(daily["day"], errors="coerce") > cut_day)
            & (pd.to_numeric(daily["day"], errors="coerce") <= cut_day + 14)
        ]
        billed_next = set(future["store_id"].astype(str)) if not future.empty else set()
        due = shops[shops["action"].isin({ACTION_CALL, ACTION_CONVERT})]
        loaded = shops[shops["action"] == ACTION_HOLD]
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
    from sndintel.storage import read_sql, replace_table

    period = pack.period
    try:
        previous = read_sql(conn, "SELECT * FROM action_shops WHERE period = ?", (period,))
    except Exception:
        previous = pd.DataFrame()
    if previous is not None and not previous.empty:
        from sndintel.ops import persist_outcomes, score_closed_loop

        try:
            shop_month = read_sql(conn, "SELECT * FROM shop_month")
        except Exception:
            shop_month = pd.DataFrame()
        try:
            visits = read_sql(conn, "SELECT * FROM shop_visits")
        except Exception:
            visits = pd.DataFrame()
        persist_outcomes(conn, score_closed_loop(previous, shop_month, visits, period))
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
        raw_shops=_sql_shops_to_raw(shops),
        raw_distributors=dists if dists is not None else pd.DataFrame(),
        raw_dsrs=dsrs if dsrs is not None else pd.DataFrame(),
        raw_backtest=back if back is not None else pd.DataFrame(),
        pipeline=_present_pipeline(_sql_shops_to_raw(shops)),
        sales_head=_present_sales_head(_sql_shops_to_raw(shops)),
        beat=_present_beat_plan(_sql_shops_to_raw(shops)),
        lost_doors=_present_lost_doors(_sql_shops_to_raw(shops)),
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


def _universe_store_ids(stores: pd.DataFrame | None) -> pd.DataFrame:
    if stores is None or stores.empty or "store_id" not in stores.columns:
        return pd.DataFrame(columns=["store_id"])
    uni = stores.copy()
    uni["store_id"] = uni["store_id"].astype(str).str.strip()
    uni = uni.loc[uni["store_id"].ne("")].copy()
    if "in_universe" in uni.columns and (pd.to_numeric(uni["in_universe"], errors="coerce").fillna(0) == 1).any():
        uni = uni[pd.to_numeric(uni["in_universe"], errors="coerce").fillna(0) == 1]
    return uni.drop_duplicates("store_id")


def _shop_frame(
    expected: pd.DataFrame,
    billed: pd.Series,
    stores: pd.DataFrame | None,
    shop_month: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    if expected is None or expected.empty:
        out = pd.DataFrame(columns=["store_id", "expected_mt", "ams_3m"])
    else:
        out = expected.copy()
    if "store_id" in out.columns:
        out["store_id"] = out["store_id"].astype(str)
    if "expected_mt" not in out.columns:
        out["expected_mt"] = 0.0
    if "ams_3m" not in out.columns:
        out["ams_3m"] = 0.0
    uni = _universe_store_ids(stores)
    if not uni.empty:
        have = set(out["store_id"].astype(str)) if not out.empty else set()
        missing = uni.loc[~uni["store_id"].astype(str).isin(have), ["store_id"]].copy()
        if not missing.empty:
            extra = pd.DataFrame(
                {
                    "store_id": missing["store_id"].astype(str),
                    "expected_mt": 0.0,
                    "ams_3m": 0.0,
                }
            )
            out = pd.concat([out, extra], ignore_index=True) if not out.empty else extra
    if out is None or out.empty:
        return pd.DataFrame()
    out["billed_mt"] = out["store_id"].map(billed).fillna(0.0)
    out["expected_mt"] = pd.to_numeric(out["expected_mt"], errors="coerce").fillna(0.0)
    out["ams_3m"] = pd.to_numeric(out["ams_3m"], errors="coerce").fillna(0.0)
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
    return classify_demand_actions(shops, open_mtd)


def _instruction(row: Any) -> str:
    name = str(getattr(row, "store_name", "") or getattr(row, "store_id", ""))
    action = str(getattr(row, "action", "") or "")
    cycle = getattr(row, "cycle_days", None)
    since = getattr(row, "days_since_bill", None)
    cover = getattr(row, "cover_left_days", None)
    drop = float(getattr(row, "expected_drop_mt", 0) or getattr(row, "typical_drop_mt", 0) or 0)
    last_drop = float(getattr(row, "last_drop_mt", 0) or 0)
    billed = float(getattr(row, "billed_mt", 0) or 0)
    ams = float(getattr(row, "ams_3m", 0) or 0)
    last = str(getattr(row, "last_bill_date", "") or "")
    cycle_s = f"{int(round(float(cycle)))} days" if pd.notna(cycle) else "its usual gap"
    since_s = f"{int(round(float(since)))} days" if pd.notna(since) else "unknown"
    last_bit = f" Last billed {last}." if last else ""
    if action == ACTION_CALL:
        if ams <= 1e-9 and float(getattr(row, "expected_drop_mt", 0) or 0) <= 1e-9:
            return (
                f"Call {name}. Universe door with no purchase history. "
                f"Unvisited this month — still part of the beat.{last_bit}"
            )
        rec = str(getattr(row, "recommended_action", "") or "Reorder due")
        return (
            f"{rec}: {name}. Usually buys every {cycle_s}; it has been {since_s} with no bill. "
            f"Ask {_kg_text(drop)} (90-day expected drop). Not visited this month.{last_bit}"
        )
    if action == ACTION_CONVERT:
        rec = str(getattr(row, "recommended_action", "") or "Reorder due")
        return (
            f"{rec}: {name} was visited and did not buy. Cycle is {cycle_s} and it has been {since_s}. "
            f"Ask {_kg_text(drop)}."
        )
    if action == ACTION_LIFT:
        rec = str(getattr(row, "recommended_action", "") or "Recover lost volume")
        return (
            f"{rec}: {name} billed {_kg_text(billed)} this month versus {_kg_text(drop)} expected drop. "
            f"Usual drop {_kg_text(drop)} every {cycle_s}; last drop {_kg_text(last_drop)} {since_s} ago."
        )
    if action == ACTION_RECOVER:
        if pd.notna(cycle):
            cut = f"(cut-off is {int(LAPSE_MULTIPLIER)}× the {cycle_s} cycle)"
        else:
            cut = "(no purchase in the last 90 days)"
        return f"Lapsed — lost door: {name} has been quiet {since_s} {cut}. Ask is 0.{last_bit}"
    coming = bool(getattr(row, "coming_due", False))
    until = getattr(row, "days_until_due", None)
    if coming and action == ACTION_HOLD:
        until_s = f"{int(round(float(until)))} days" if until is not None and pd.notna(until) else "a few days"
        return (
            f"{name} comes due in {until_s} (cycle {cycle_s}). "
            f"Expected drop {_kg_text(drop)} sits in Not yet due — Ask today is 0.{last_bit}"
        )
    return f"{name} is inside its {cycle_s} cycle ({since_s} since last bill). Leave it."


def _kg_text(mt: Any) -> str:
    try:
        if mt is None or pd.isna(mt):
            return "0 KG"
        return f"{int(round(float(mt) * 1000)):,} KG"
    except (TypeError, ValueError):
        return "0 KG"


def _ask_mt(
    shops: pd.DataFrame,
    days_left: int = 0,
    open_mtd: bool = True,
    days_in_month: int = 31,
) -> pd.Series:
    """Rest-of-month closable volume. Prefer `_attach_rest_of_month` which also flags coming-due doors."""
    return _attach_rest_of_month(shops, days_left, open_mtd, days_in_month)["week_target_mt"]


def _attach_rest_of_month(
    shops: pd.DataFrame,
    days_left: int = 0,
    open_mtd: bool = True,
    days_in_month: int = 31,
) -> pd.DataFrame:
    """Ask = expected drop when due; not-yet-due volume is a pipeline pillar, not Ask."""
    del days_in_month
    return attach_pipeline(shops, days_left=days_left, open_mtd=open_mtd)


def _value_score(shops: pd.DataFrame) -> pd.Series:
    ask = pd.to_numeric(shops["week_target_mt"], errors="coerce").fillna(0)
    overdue = pd.to_numeric(shops.get("days_overdue"), errors="coerce").fillna(0)
    cycle = pd.to_numeric(shops.get("cycle_days"), errors="coerce").replace(0, np.nan).fillna(30)
    trend = pd.to_numeric(shops.get("trend_pct"), errors="coerce").fillna(0)
    unvisited = shops["call_status"].eq("Unvisited")
    score = ask * (1.0 + overdue / cycle.clip(lower=7))
    score = score + (-trend.clip(upper=0) * pd.to_numeric(shops["ams_3m"], errors="coerce").fillna(0))
    score = score.where(~unvisited, score * 1.35)
    hold = shops["action"].eq(ACTION_HOLD)
    coming = shops["coming_due"].fillna(False) if "coming_due" in shops.columns else False
    nyd = pd.to_numeric(shops.get("not_yet_due_mt"), errors="coerce").fillna(0) if "not_yet_due_mt" in shops.columns else 0.0
    score = score.where(~hold, 0.0)
    score = score.where(~(hold & coming), nyd * 0.45 if isinstance(nyd, pd.Series) else 0.0)
    return score.fillna(0.0)


def action_buckets(shops: pd.DataFrame) -> dict[str, float]:
    """Immediate Ask plus the four pipeline pillars.

    Immediate Ask = Due unvisited + Due visited + Another visit.
    Lapsed Ask is 0. Coming due is Not yet due, not Ask.

    Pipeline Expected = Billed + Due unvisited + Drop variance + Not yet due.
    """
    empty = {
        "n_call": 0,
        "n_convert": 0,
        "n_lift": 0,
        "n_lapse": 0,
        "n_coming": 0,
        "n_doors": 0,
        "n_hold": 0,
        "n_due": 0,
        "n_lapsed": 0,
        "n_universe": 0,
        "ask_call": 0.0,
        "ask_convert": 0.0,
        "ask_lift": 0.0,
        "ask_lapse": 0.0,
        "ask_coming": 0.0,
        "ask_doors": 0.0,
        "week_target_mt": 0.0,
        "expected_mt": 0.0,
        "ams_3m": 0.0,
        "billed_mt": 0.0,
        "remaining_mt": 0.0,
        "due_unvisited_mt": 0.0,
        "drop_variance_mt": 0.0,
        "not_yet_due_mt": 0.0,
        "pipeline_expected_mt": 0.0,
        "n_due_visited": 0,
    }
    if shops is None or shops.empty:
        return empty
    g = shops
    ask = pd.to_numeric(g["week_target_mt"], errors="coerce").fillna(0) if "week_target_mt" in g.columns else pd.Series(0.0, index=g.index)
    action = g["action"] if "action" in g.columns else pd.Series("", index=g.index)
    coming = g["coming_due"].fillna(False) if "coming_due" in g.columns else pd.Series(False, index=g.index)
    coming = coming.astype(bool)

    def _n_ask(mask) -> tuple[int, float]:
        m = mask.fillna(False) if hasattr(mask, "fillna") else mask
        return int(m.sum()), float(ask[m].sum())

    def _sum(name: str) -> float:
        if name not in g.columns:
            return 0.0
        return float(pd.to_numeric(g[name], errors="coerce").fillna(0).sum())

    n_call, ask_call = _n_ask(action.eq(ACTION_CALL))
    n_convert, ask_convert = _n_ask(action.eq(ACTION_CONVERT))
    n_lift, ask_lift = _n_ask(action.eq(ACTION_LIFT))
    n_lapse, _ = _n_ask(action.eq(ACTION_RECOVER))
    n_coming, _ = _n_ask(coming)
    n_hold, _ = _n_ask(action.eq(ACTION_HOLD))
    due_mask = action.isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT})
    visited_due = due_mask & g["call_status"].isin({"Visited · not billed", "Billed"}) if "call_status" in g.columns else due_mask & action.eq(ACTION_CONVERT)
    nyd = _sum("not_yet_due_mt")
    billed = _sum("billed_mt")
    due_u = _sum("due_unvisited_mt")
    var = _sum("drop_variance_mt")
    pipe = _sum("pipeline_expected_mt")
    if pipe <= 0 and (billed or due_u or var or nyd):
        pipe = billed + due_u + var + nyd
    return {
        "n_call": n_call,
        "n_convert": n_convert,
        "n_lift": n_lift,
        "n_lapse": n_lapse,
        "n_coming": n_coming,
        "n_doors": n_call + n_convert + n_lift,
        "n_hold": n_hold,
        "n_due": int(due_mask.sum()),
        "n_lapsed": n_lapse,
        "n_universe": int(len(g)),
        "n_due_visited": int(visited_due.sum()) if hasattr(visited_due, "sum") else 0,
        "ask_call": ask_call,
        "ask_convert": ask_convert,
        "ask_lift": ask_lift,
        "ask_lapse": 0.0,
        "ask_coming": nyd,
        "ask_doors": ask_call + ask_convert + ask_lift,
        "week_target_mt": float(ask.sum()),
        "expected_mt": _sum("expected_mt"),
        "ams_3m": _sum("ams_3m"),
        "billed_mt": billed,
        "remaining_mt": _sum("remaining_mt"),
        "due_unvisited_mt": due_u,
        "drop_variance_mt": var,
        "not_yet_due_mt": nyd,
        "pipeline_expected_mt": pipe,
    }


def _roll_units(shops: pd.DataFrame, key: str, extra_city: bool = False, extra_dist: bool = False) -> pd.DataFrame:
    if shops is None or shops.empty or key not in shops.columns:
        return pd.DataFrame()
    work = shops[shops[key].astype(str).str.len() > 0].copy()
    if work.empty:
        return pd.DataFrame()
    rows = []
    for name, g in work.groupby(key):
        buckets = action_buckets(g)
        n_call = int(buckets["n_call"])
        n_convert = int(buckets["n_convert"])
        n_lift = int(buckets["n_lift"])
        n_lapse = int(buckets["n_lapse"])
        n_coming = int(buckets["n_coming"])
        week = float(buckets["week_target_mt"])
        remaining = float(buckets["remaining_mt"])
        n_doors = int(buckets["n_doors"])
        coming_bit = f", {n_coming} more come due before month-end" if n_coming else ""
        lapse_bit = f", {n_lapse} lapsed (Ask 0)" if n_lapse else ""
        instruction = (
            f"Push {name}: {n_doors} doors to work now "
            f"({n_call} due and unvisited, {n_convert} due but already visited, "
            f"{n_lift} another visit){coming_bit}{lapse_bit}. "
            f"Immediate Ask {_kg_text(week)} "
            f"(unvisited due {_kg_text(buckets['due_unvisited_mt'])}, "
            f"variance {_kg_text(buckets['drop_variance_mt'])}, "
            f"not yet due {_kg_text(buckets['not_yet_due_mt'])})."
        )
        row = {
            "grain_id": str(name),
            "n_doors": n_doors,
            "n_coming": n_coming,
            "n_call": n_call,
            "n_convert": n_convert,
            "n_lift": n_lift,
            "n_lapse": n_lapse,
            "n_hold": int(buckets["n_hold"]),
            "n_due": int(buckets["n_due"]),
            "n_universe": int(buckets["n_universe"]),
            "n_due_visited": int(buckets["n_due_visited"]),
            "ask_call": buckets["ask_call"],
            "ask_convert": buckets["ask_convert"],
            "ask_lift": buckets["ask_lift"],
            "ask_lapse": buckets["ask_lapse"],
            "ask_coming": buckets["ask_coming"],
            "ask_doors": buckets["ask_doors"],
            "ams_3m": float(buckets["ams_3m"]),
            "billed_mt": float(buckets["billed_mt"]),
            "expected_mt": float(buckets["expected_mt"]),
            "should_have_mt": float(buckets["expected_mt"]),
            "behind_pace_mt": remaining,
            "remaining_mt": remaining,
            "week_target_mt": week,
            "due_unvisited_mt": float(buckets["due_unvisited_mt"]),
            "drop_variance_mt": float(buckets["drop_variance_mt"]),
            "not_yet_due_mt": float(buckets["not_yet_due_mt"]),
            "pipeline_expected_mt": float(buckets["pipeline_expected_mt"]),
            "instruction": instruction,
            "value_score": float(pd.to_numeric(g["value_score"], errors="coerce").fillna(0).sum()) if "value_score" in g.columns else 0.0,
        }
        if extra_city and "city" in g.columns:
            row["city"] = str(g["city"].mode().iloc[0]) if not g["city"].mode().empty else ""
        if extra_dist and "distributor" in g.columns:
            row["distributor"] = str(g["distributor"].mode().iloc[0]) if not g["distributor"].mode().empty else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["value_score", "week_target_mt"], ascending=False).reset_index(drop=True)


def _roll_dsrs(shops: pd.DataFrame) -> pd.DataFrame:
    """One DSR row per unique (city, distributor, name) — never merge namesakes."""
    from sndintel.identity import dsr_display_name, dsr_unit_id

    if shops is None or shops.empty or "dsr_name" not in shops.columns:
        return pd.DataFrame()
    work = shops.copy()
    for col, default in (("city", ""), ("distributor", ""), ("dsr_name", "")):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default).astype(str)
    work["_dsr_id"] = [
        dsr_unit_id(c, d, n) for c, d, n in zip(work["city"], work["distributor"], work["dsr_name"])
    ]
    out = _roll_units(work, "_dsr_id", extra_city=True, extra_dist=True)
    if out.empty:
        return out
    out["dsr_name"] = [dsr_display_name(g) for g in out["grain_id"]]
    out["instruction"] = [
        inst.replace(f"Push {gid}:", f"Push {name}:", 1)
        for gid, name, inst in zip(out["grain_id"], out["dsr_name"], out["instruction"])
    ]
    return out


def _units_with_work(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    work = (
        pd.to_numeric(df.get("n_call"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_convert"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_lift"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_lapse"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("n_coming"), errors="coerce").fillna(0)
    )
    ask = pd.to_numeric(df.get("week_target_mt"), errors="coerce").fillna(0)
    return df.loc[(work > 0) | (ask > 0)].copy()


def _country_row(shops: pd.DataFrame, as_of: int, days: int, days_left: int, open_mtd: bool, source: str) -> dict[str, Any]:
    b = action_buckets(shops)
    return {
        "billed_mt": b["billed_mt"],
        "expected_mt": b["expected_mt"],
        "ams_3m": b["ams_3m"],
        "should_have_mt": b["expected_mt"],
        "behind_pace_mt": b["remaining_mt"],
        "remaining_mt": b["remaining_mt"],
        "week_target_mt": b["week_target_mt"],
        "due_unvisited_mt": b["due_unvisited_mt"],
        "drop_variance_mt": b["drop_variance_mt"],
        "not_yet_due_mt": b["not_yet_due_mt"],
        "pipeline_expected_mt": b["pipeline_expected_mt"],
        "as_of_day": as_of,
        "days_in_month": days,
        "days_left": days_left,
        "open_mtd": open_mtd,
        "n_call": b["n_call"],
        "n_convert": b["n_convert"],
        "n_lift": b["n_lift"],
        "n_lapse": b["n_lapse"],
        "n_hold": b["n_hold"],
        "n_doors": b["n_doors"],
        "n_coming": b["n_coming"],
        "n_due": b["n_due"],
        "n_universe": b["n_universe"],
        "source": source,
    }


def _headline(country: dict[str, Any], open_mtd: bool, has_daily: bool) -> str:
    n_call = int(country.get("n_call") or 0)
    n_convert = int(country.get("n_convert") or 0)
    n_lift = int(country.get("n_lift") or 0)
    n_lapse = int(country.get("n_lapse") or 0)
    n_coming = int(country.get("n_coming") or 0)
    week = float(country.get("week_target_mt") or 0)
    billed = float(country.get("billed_mt") or 0)
    pipe = float(country.get("pipeline_expected_mt") or country.get("expected_mt") or 0)
    due_u = float(country.get("due_unvisited_mt") or 0)
    var = float(country.get("drop_variance_mt") or 0)
    nyd = float(country.get("not_yet_due_mt") or 0)
    coming_bit = f", {n_coming} come due before month-end" if n_coming else ""
    hole = (
        f"billed {_kg_text(billed)} of {_kg_text(pipe)} pipeline "
        f"(due unvisited {_kg_text(due_u)}, variance {_kg_text(var)}, "
        f"not yet due {_kg_text(nyd)}). "
    )
    if not has_daily:
        return (
            f"No billed days in the warehouse — cycles fall back to monthly gaps. "
            f"{hole}{n_call} due, {n_convert} due and already visited, {n_lift} another visit, "
            f"{n_lapse} lapsed{coming_bit}. Immediate Ask {_kg_text(week)}."
        )
    when = f"Day {country.get('as_of_day')}/{country.get('days_in_month')}" if open_mtd else "Month closed"
    return (
        f"{when}: {hole}{n_call} due and unvisited, {n_convert} due but already seen, "
        f"{n_lift} another visit, {n_lapse} lapsed{coming_bit}. "
        f"Immediate Ask {_kg_text(week)} (expected drop when ratio ≥ {DUE_RATIO:.1f}, not remaining-to-Expected)."
    )


def _take_action(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    from sndintel.capacity import cap_shops_per_dsr

    if shops is None or shops.empty or "action" not in shops.columns:
        return shops if shops is not None else pd.DataFrame()
    subset = shops[shops["action"] == action]
    if subset.empty:
        return subset
    return cap_shops_per_dsr(subset).head(n)


COUNTRY_VIEW = [
    ("billed_mt", "Billed (KG)"),
    ("pipeline_expected_mt", "Pipeline expected (KG)"),
    ("week_target_mt", "Ask rest of month (KG)"),
    ("due_unvisited_mt", "Due unvisited (KG)"),
    ("drop_variance_mt", "Volume lost to variance (KG)"),
    ("not_yet_due_mt", "Not yet due (KG)"),
    ("expected_mt", "Expected this month (KG)"),
    ("ams_3m", "AMS (KG)"),
    ("remaining_mt", "Still to Expected (KG)"),
    ("as_of_day", "As of day"),
    ("days_left", "Days left"),
    ("n_doors", "Doors"),
    ("n_coming", "Coming due"),
    ("n_call", "Due · unvisited"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
]

SHOP_VIEW = [
    ("store_name", "Shop"),
    ("store_id", "POP"),
    ("city", "City"),
    ("section", "Area"),
    ("distributor", "Distributor"),
    ("dsr_name", "DSR"),
    ("action", "Action"),
    ("recommended_action", "Recommended action"),
    ("coming_due", "Coming due"),
    ("week_target_mt", "Ask rest of month (KG)"),
    ("expected_drop_mt", "Target drop (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("ams_3m", "AMS (KG)"),
    ("days_since_bill", "Days since bill"),
    ("cycle_days", "Usual cycle (days)"),
    ("days_overdue", "Days overdue"),
    ("depletion_ratio", "Depletion ratio"),
    ("cover_left_days", "Cover left (days)"),
    ("last_drop_mt", "Last drop (KG)"),
    ("typical_drop_mt", "Typical drop (KG)"),
    ("next_drop_mt", "Next order (KG)"),
    ("expected_mt", "Expected (KG)"),
    ("last_month_mt", "Last month (KG)"),
    ("trend_pct", "Trend vs prior 3m"),
    ("last_bill_date", "Last billed"),
    ("call_status", "Call"),
    ("instruction", "Do this"),
]

SHOP_PDF_COLS = [
    "Shop",
    "Area",
    "DSR",
    "Action",
    "Recommended action",
    "Ask rest of month (KG)",
    "Target drop (KG)",
    "Billed (KG)",
    "Days overdue",
    "Last billed",
    "Do this",
]

COUNTRY_PDF_COLS = [
    "Billed (KG)",
    "Pipeline expected (KG)",
    "Ask rest of month (KG)",
    "Due unvisited (KG)",
    "Volume lost to variance (KG)",
    "Not yet due (KG)",
    "Expected this month (KG)",
    "AMS (KG)",
    "Doors",
    "Coming due",
    "Due · unvisited",
    "Due · visited",
    "Another visit",
    "Lapsing",
]

DIST_PDF_COLS = [
    "Distributor",
    "City",
    "Doors",
    "Coming due",
    "Due",
    "Due · visited",
    "Another visit",
    "Lapsing",
    "AMS (KG)",
    "Billed (KG)",
    "Ask rest of month (KG)",
    "Do this",
]

DSR_PDF_COLS = [
    "DSR",
    "City",
    "Doors",
    "Coming due",
    "Due",
    "Due · visited",
    "Another visit",
    "Lapsing",
    "AMS (KG)",
    "Billed (KG)",
    "Ask rest of month (KG)",
    "Do this",
]

UNIT_VIEW_DIST = [
    ("grain_id", "Distributor"),
    ("city", "City"),
    ("n_doors", "Doors"),
    ("n_coming", "Coming due"),
    ("n_call", "Due"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
    ("ams_3m", "AMS (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("week_target_mt", "Ask rest of month (KG)"),
    ("remaining_mt", "Still to Expected (KG)"),
    ("instruction", "Do this"),
]

UNIT_VIEW_DSR = [
    ("dsr_name", "DSR"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("label", "Label"),
    ("n_doors", "Doors"),
    ("n_coming", "Coming due"),
    ("n_call", "Due"),
    ("n_convert", "Due · visited"),
    ("n_lift", "Another visit"),
    ("n_lapse", "Lapsing"),
    ("ams_3m", "AMS (KG)"),
    ("billed_mt", "Billed (KG)"),
    ("week_target_mt", "Ask rest of month (KG)"),
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
        "Ask rest of month (KG)",
        "Expected (KG)",
        "Last month (KG)",
        "AMS (KG)",
        "Last drop (KG)",
        "Typical drop (KG)",
        "Next order (KG)",
        "Target drop (KG)",
        "Pipeline expected (KG)",
        "Due unvisited (KG)",
        "Volume lost to variance (KG)",
        "Not yet due (KG)",
    }
    for src, label in view:
        if src not in df.columns:
            out[label] = [None] * len(df)
            continue
        col = df[src]
        if label in kg_cols:
            out[label] = [_round_kg(v) for v in col]
        elif src == "coming_due":
            out[label] = ["Yes" if bool(v) and not (isinstance(v, float) and pd.isna(v)) else "" for v in col]
        elif label in {"Usual cycle (days)", "Days since bill", "Days overdue", "Cover left (days)"}:
            out[label] = [None if pd.isna(v) else int(round(float(v))) for v in col]
        elif label == "Depletion ratio":
            out[label] = [None if pd.isna(v) else round(float(v), 2) for v in col]
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
    work = df
    if grain == "DSR" and "dsr_name" not in df.columns and "grain_id" in df.columns:
        from sndintel.identity import dsr_display_name

        work = df.copy()
        work["dsr_name"] = [dsr_display_name(v) for v in work["grain_id"]]
    view = UNIT_VIEW_DIST if grain == "Distributor" else UNIT_VIEW_DSR
    return _present(work, view)


def _present_backtest(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[l for _, l in BACKTEST_VIEW])
    if "Due precision %" in df.columns or "Due precision" in df.columns:
        return df
    return _present(df, BACKTEST_VIEW)


def _present_pipeline(shops: pd.DataFrame) -> pd.DataFrame:
    """Tier 1 — city pipeline bridge."""
    if shops is None or shops.empty or "city" not in shops.columns:
        return pd.DataFrame(
            columns=[
                "City",
                "Pipeline expected (MT)",
                "Billed (MT)",
                "Immediate Due / Ask (MT)",
                "Not yet due (MT)",
                "Volume lost to variance (MT)",
            ]
        )
    rows = []
    for city, g in shops.groupby(shops["city"].astype(str)):
        b = action_buckets(g)
        rows.append(
            {
                "City": str(city),
                "Pipeline expected (MT)": round(float(b["pipeline_expected_mt"]), 1),
                "Billed (MT)": round(float(b["billed_mt"]), 1),
                "Immediate Due / Ask (MT)": round(float(b["week_target_mt"]), 1),
                "Not yet due (MT)": round(float(b["not_yet_due_mt"]), 1),
                "Volume lost to variance (MT)": round(float(b["drop_variance_mt"]), 1),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("Pipeline expected (MT)", ascending=False).reset_index(drop=True)


def _present_sales_head(shops: pd.DataFrame) -> pd.DataFrame:
    """Tier 2 — distributor × DSR execution."""
    from sndintel.identity import dsr_display_name, dsr_unit_id

    cols = [
        "Distributor",
        "DSR",
        "City",
        "Active universe",
        "Shops due for reorder",
        "Due shops visited %",
        "Unvisited due Ask (MT)",
        "Drop variance (MT)",
        "Lapsed / churned",
    ]
    if shops is None or shops.empty or "dsr_name" not in shops.columns:
        return pd.DataFrame(columns=cols)
    work = shops.copy()
    for col, default in (("city", ""), ("distributor", ""), ("dsr_name", "")):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default).astype(str)
    work["_dsr_id"] = [
        dsr_unit_id(c, d, n) for c, d, n in zip(work["city"], work["distributor"], work["dsr_name"])
    ]
    rows = []
    for gid, g in work.groupby("_dsr_id"):
        b = action_buckets(g)
        n_due = int(b["n_due"])
        visited_pct = (100.0 * b["n_due_visited"] / n_due) if n_due else None
        lapsed = g["is_lapsed"].fillna(False).astype(bool) if "is_lapsed" in g.columns else g["action"].eq(ACTION_RECOVER)
        active = int((~lapsed).sum()) if hasattr(lapsed, "sum") else int(b["n_universe"])
        rows.append(
            {
                "Distributor": str(g["distributor"].iloc[0]),
                "DSR": dsr_display_name(gid),
                "City": str(g["city"].iloc[0]),
                "Active universe": active,
                "Shops due for reorder": n_due,
                "Due shops visited %": None if visited_pct is None else int(round(visited_pct)),
                "Unvisited due Ask (MT)": round(float(b["due_unvisited_mt"]), 1),
                "Drop variance (MT)": round(float(b["drop_variance_mt"]), 1),
                "Lapsed / churned": int(b["n_lapse"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["Unvisited due Ask (MT)", "Drop variance (MT)"], ascending=False).reset_index(drop=True)


def _present_beat_plan(shops: pd.DataFrame) -> pd.DataFrame:
    """Tier 3 — today's reorder list (due, not lapsed)."""
    cols = [
        "Shop",
        "Area",
        "City",
        "DSR",
        "Last purchased",
        "Days overdue",
        "Target drop (KG)",
        "Recommended action",
    ]
    if shops is None or shops.empty:
        return pd.DataFrame(columns=cols)
    due = shops[shops["action"].isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT})].copy()
    if due.empty:
        return pd.DataFrame(columns=cols)
    due = due.sort_values(["week_target_mt", "days_overdue"], ascending=False)
    return pd.DataFrame(
        {
            "Shop": list(due.get("store_name", due.get("store_id"))),
            "Area": list(due.get("section", [""] * len(due))),
            "City": list(due.get("city", [""] * len(due))),
            "DSR": list(due.get("dsr_name", [""] * len(due))),
            "Last purchased": list(due.get("last_bill_date", [""] * len(due))),
            "Days overdue": [None if pd.isna(v) else int(round(float(v))) for v in due.get("days_overdue", [])],
            "Target drop (KG)": [_round_kg(v) for v in due.get("expected_drop_mt", due.get("typical_drop_mt", []))],
            "Recommended action": list(due.get("recommended_action", due.get("action", []))),
        }
    )


def _present_lost_doors(shops: pd.DataFrame) -> pd.DataFrame:
    cols = ["Shop", "City", "DSR", "Distributor", "Last purchased", "Days since bill", "Usual cycle (days)", "Do this"]
    if shops is None or shops.empty:
        return pd.DataFrame(columns=cols)
    lost = shops[shops["action"] == ACTION_RECOVER].copy()
    if lost.empty:
        return pd.DataFrame(columns=cols)
    lost = lost.sort_values("days_since_bill", ascending=False)
    return pd.DataFrame(
        {
            "Shop": list(lost.get("store_name", lost.get("store_id"))),
            "City": list(lost.get("city", [""] * len(lost))),
            "DSR": list(lost.get("dsr_name", [""] * len(lost))),
            "Distributor": list(lost.get("distributor", [""] * len(lost))),
            "Last purchased": list(lost.get("last_bill_date", [""] * len(lost))),
            "Days since bill": [None if pd.isna(v) else int(round(float(v))) for v in lost.get("days_since_bill", [])],
            "Usual cycle (days)": [None if pd.isna(v) else int(round(float(v))) for v in lost.get("cycle_days", [])],
            "Do this": list(lost.get("instruction", [""] * len(lost))),
        }
    )


def _take_present(shops: pd.DataFrame, action: str, n: int) -> pd.DataFrame:
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame()
    col = "Action" if "Action" in shops.columns else "action"
    if col not in shops.columns:
        return shops.head(0)
    return shops[shops[col] == action].head(n)


def _sql_shops_to_raw(df: pd.DataFrame | None) -> pd.DataFrame:
    """Reload persisted action_shops so Monday / beat packs can use them."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "coming_due" in out.columns:
        out["coming_due"] = pd.to_numeric(out["coming_due"], errors="coerce").fillna(0).astype(bool)
    for flag in ("is_lapsed", "is_cold_start"):
        if flag in out.columns:
            out[flag] = pd.to_numeric(out[flag], errors="coerce").fillna(0).astype(bool)
    return out


def _raw_shops_to_sql(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["period"] = period
    for flag in ("coming_due", "is_lapsed", "is_cold_start"):
        if flag in out.columns:
            out[flag] = pd.to_numeric(out[flag], errors="coerce").fillna(0).astype(int)
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
        "next_drop_mt",
        "next_drop_model",
        "coming_due",
        "days_until_due",
        "n_orders_left",
        "expected_drop_mt",
        "api_days",
        "depletion_ratio",
        "n_purchases_90d",
        "n_purchases_ever",
        "is_cold_start",
        "is_lapsed",
        "due_unvisited_mt",
        "drop_variance_mt",
        "not_yet_due_mt",
        "pipeline_expected_mt",
        "recommended_action",
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
    if "dsr_name" not in out.columns and grain == "dsr":
        from sndintel.identity import dsr_display_name

        out["dsr_name"] = [dsr_display_name(v) for v in out["grain_id"]]
    cols = [
        "period",
        "grain",
        "grain_id",
        "city",
        "distributor",
        "dsr_name",
        "label",
        "span_unique",
        "day_cap",
        "n_call",
        "n_convert",
        "n_lift",
        "n_lapse",
        "n_hold",
        "n_doors",
        "n_coming",
        "ams_3m",
        "billed_mt",
        "expected_mt",
        "should_have_mt",
        "behind_pace_mt",
        "remaining_mt",
        "week_target_mt",
        "due_unvisited_mt",
        "drop_variance_mt",
        "not_yet_due_mt",
        "pipeline_expected_mt",
        "n_due",
        "n_universe",
        "n_due_visited",
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
        "n_doors": b.get("n_doors"),
        "n_coming": b.get("n_coming"),
        "pipeline_expected_mt": b.get("pipeline_expected_mt"),
        "due_unvisited_mt": b.get("due_unvisited_mt"),
        "drop_variance_mt": b.get("drop_variance_mt"),
        "not_yet_due_mt": b.get("not_yet_due_mt"),
        "has_daily": int(pack.has_daily),
        "headline": pack.headline,
        "source": pack.source,
        "metrics_json": pd.Series(b).to_json(),
    }


def _sql_shops_to_present(df: pd.DataFrame) -> pd.DataFrame:
    return _present_shops(df)
