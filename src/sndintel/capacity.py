"""DSR capacity, whale doors, and visit-file quality.

Answers the four operating questions without another model:

* Overloaded — universe does not fit the working-day call budget
* Not working the beat — capacity exists, visit % is weak
* Not converting — visits happened, billed / visited is weak
* Not lifting drop — billed well, drop vs expected is light
* Fine — on or ahead of Expected
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sndintel.config import (
    CALLS_PER_DAY,
    DSR_DAY_CAP,
    SPAN_OVERLOAD,
    VISIT_SUSPECT_RATE,
    VISIT_SUSPECT_UNIVERSE,
    WHALE_AMS_MT,
    YOY_MIN_LY_MT,
)
from sndintel.identity import dsr_display_name, dsr_unit_id

LABEL_OVERLOADED = "Overloaded"
LABEL_NOT_WORKING = "Not working the beat"
LABEL_NOT_CONVERTING = "Not converting"
LABEL_NOT_LIFTING = "Not lifting drop"
LABEL_FINE = "Fine"

WHALE_N = 20


def score_dsr_capacity_from_units(
    dsrs: pd.DataFrame,
    as_of_day: int,
    days_in_month: int,
    calls_per_day: float | None = None,
) -> pd.DataFrame:
    """Capacity labels from DSR scorecards when shop-day is not in this pack."""
    if dsrs is None or dsrs.empty:
        return pd.DataFrame()
    rate = float(calls_per_day if calls_per_day is not None else CALLS_PER_DAY)
    days_done = max(int(as_of_day or 0), 1)
    feasible = days_done * rate
    out = dsrs.copy()
    if "dsr_name" not in out.columns:
        out["dsr_name"] = out.get("grain_id", "").map(dsr_display_name)
    universe = pd.to_numeric(out.get("universe"), errors="coerce").fillna(0)
    visited = pd.to_numeric(out.get("visited"), errors="coerce")
    billed = pd.to_numeric(out.get("billed"), errors="coerce").fillna(0)
    visit_rate = pd.to_numeric(out.get("visit_rate"), errors="coerce")
    visit_rate = visit_rate.where(visit_rate.notna(), visited / universe.replace(0, np.nan))
    visited = visited.fillna(visit_rate.fillna(0) * universe)
    strike_v = np.where(visited.fillna(0) > 0, billed / visited.replace(0, np.nan), np.nan)
    span = universe / feasible if feasible else np.inf
    vol = pd.to_numeric(out.get("volume_mt"), errors="coerce").fillna(0)
    exp = pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0)
    drop_now = vol / billed.replace(0, np.nan)
    exp_drop = pd.to_numeric(out.get("expected_drop_size_mt"), errors="coerce")
    drop_index = drop_now / exp_drop.replace(0, np.nan)
    remaining = (exp - vol).clip(lower=0)
    labels, whys = [], []
    for i in range(len(out)):
        lab, why = _label(
            span_unique=float(span.iloc[i]) if np.isfinite(span.iloc[i]) else 9.0,
            visit_rate=float(visit_rate.iloc[i]) if pd.notna(visit_rate.iloc[i]) else 0.0,
            strike_of_visits=float(strike_v[i]) if pd.notna(strike_v[i]) else 0.0,
            drop_index=float(drop_index.iloc[i]) if pd.notna(drop_index.iloc[i]) else 1.0,
            remaining=float(remaining.iloc[i]),
            expected=float(exp.iloc[i]),
        )
        labels.append(lab)
        whys.append(why)
    out = out.assign(
        label=labels,
        why=whys,
        span_unique=span,
        visit_rate=visit_rate,
        strike_of_visits=strike_v,
        remaining_mt=remaining,
        week_target_mt=remaining,
        billed_mt=vol,
        day_cap=int(DSR_DAY_CAP),
        feasible_mtd=feasible,
    )
    order = {
        LABEL_OVERLOADED: 0,
        LABEL_NOT_WORKING: 1,
        LABEL_NOT_CONVERTING: 2,
        LABEL_NOT_LIFTING: 3,
        LABEL_FINE: 4,
    }
    out["_ord"] = out["label"].map(order).fillna(9)
    return out.sort_values(["_ord", "remaining_mt"], ascending=[True, False]).drop(columns=["_ord"])


def present_capacity_table(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    cols = ["DSR", "City", "Distributor", "Label", "Universe", "Visit %", "Strike of visits %", "Span ×", "Gap (MT)", "Why"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    show = df.head(int(n))
    name = show["dsr_name"] if "dsr_name" in show.columns else show.get("grain_id")
    rec = pd.to_numeric(show.get("remaining_mt"), errors="coerce")
    if rec is None or rec.isna().all():
        rec = pd.to_numeric(show.get("recoverable_mt"), errors="coerce")
    return pd.DataFrame(
        {
            "DSR": [dsr_display_name(v) for v in name],
            "City": list(show.get("city", [])),
            "Distributor": list(show.get("distributor", [])),
            "Label": list(show.get("label", [])),
            "Universe": [int(v) if pd.notna(v) else None for v in show.get("universe", [])],
            "Visit %": [None if pd.isna(v) else int(round(float(v) * 100)) for v in show.get("visit_rate", [])],
            "Strike of visits %": [None if pd.isna(v) else int(round(float(v) * 100)) for v in show.get("strike_of_visits", [])],
            "Span ×": [None if pd.isna(v) else round(float(v), 1) for v in show.get("span_unique", [])],
            "Gap (MT)": [None if pd.isna(v) else round(float(v), 0) for v in rec],
            "Why": list(show.get("why", [])),
        }
    )


def yoy_is_printable(ly_mt: Any) -> bool:
    try:
        return float(ly_mt) >= float(YOY_MIN_LY_MT)
    except (TypeError, ValueError):
        return False


def visit_quality_warnings(
    units: pd.DataFrame | None,
    visits: pd.DataFrame | None = None,
    period: str | None = None,
) -> list[str]:
    """Flag visit files that would flip unbilled vs unvisited."""
    warnings: list[str] = []
    has_visits = visits is not None and not visits.empty
    if not has_visits:
        warnings.append(
            "No visit file for this period. Unvisited vs unbilled cannot be split — "
            "missed doors are booked as unbilled. Do not hire or fire on coverage."
        )
        return warnings
    if units is None or units.empty:
        return warnings
    cities = units[units["grain"] == "city"].copy() if "grain" in units.columns else pd.DataFrame()
    if cities.empty:
        return warnings
    visit_rate = pd.to_numeric(cities.get("visit_rate"), errors="coerce")
    universe = pd.to_numeric(cities.get("universe"), errors="coerce")
    has_file = cities.get("has_visit_file")
    suspect = (visit_rate >= float(VISIT_SUSPECT_RATE)) & (universe >= float(VISIT_SUSPECT_UNIVERSE))
    if has_file is not None:
        try:
            suspect = suspect & (pd.to_numeric(has_file, errors="coerce").fillna(0) > 0)
        except (TypeError, ValueError):
            pass
    hits = cities.loc[suspect.fillna(False)]
    for _, row in hits.iterrows():
        name = str(row.get("grain_id") or row.get("city") or "City")
        pct = float(row.get("visit_rate") or 0) * 100
        uni = int(row.get("universe") or 0)
        warnings.append(
            f"{name}: visit {pct:.0f}% of {uni} doors. Confirm the visit file is actual calls "
            f"(not planned beats or any GPS ping) before treating the hole as conversion."
        )
    return warnings


def whale_shops(shops: pd.DataFrame, n: int = WHALE_N, floor_mt: float | None = None) -> pd.DataFrame:
    """Material doors for volume — AMS / expected / last drop at or above the whale floor."""
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame()
    floor = float(floor_mt if floor_mt is not None else WHALE_AMS_MT)
    out = shops.copy()
    ams = pd.to_numeric(out.get("ams_3m"), errors="coerce").fillna(0)
    exp = pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0)
    last = pd.to_numeric(out.get("last_drop_mt"), errors="coerce")
    if last is None or last.isna().all():
        last = pd.to_numeric(out.get("ly_mt"), errors="coerce")
    last = last.fillna(0) if last is not None else 0
    rec = pd.to_numeric(out.get("recoverable_mt"), errors="coerce")
    if rec is None or rec.isna().all():
        rec = pd.to_numeric(out.get("remaining_mt"), errors="coerce")
    rec = rec.fillna(0) if rec is not None else pd.Series(0.0, index=out.index)
    ask = pd.to_numeric(out.get("week_target_mt"), errors="coerce").fillna(0)
    whale = (ams >= floor) | (exp >= floor) | (last >= floor)
    hole = rec > 0.05
    picked = out.loc[whale & hole].copy()
    if picked.empty:
        picked = out.loc[whale].copy()
    if picked.empty:
        return out.iloc[0:0].copy()
    rank = pd.concat(
        [ask.reindex(picked.index).fillna(0), rec.reindex(picked.index).fillna(0)],
        axis=1,
    ).max(axis=1)
    picked = picked.assign(_whale_rank=rank).sort_values("_whale_rank", ascending=False)
    return picked.head(int(n)).drop(columns=["_whale_rank"], errors="ignore")


def score_dsr_capacity(
    shops: pd.DataFrame,
    as_of_day: int,
    days_in_month: int,
    days_left: int = 0,
    calls_per_day: float | None = None,
) -> pd.DataFrame:
    """One row per DSR: span, visit/strike, drop index, operating label."""
    if shops is None or shops.empty:
        return pd.DataFrame()
    work = shops.copy()
    for col, default in (("city", "(unmapped)"), ("distributor", "(unmapped)"), ("dsr_name", "(unnamed)")):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default).astype(str).replace("", default)
    work["dsr_id"] = [
        dsr_unit_id(c, d, n) for c, d, n in zip(work["city"], work["distributor"], work["dsr_name"])
    ]
    rate = float(calls_per_day if calls_per_day is not None else CALLS_PER_DAY)
    days_done = max(int(as_of_day or 0), 1)
    days_month = max(int(days_in_month or 0), days_done)
    left = max(int(days_left or 0), 0)
    feasible_mtd = days_done * rate
    feasible_left = max(left, 1) * rate
    rows = []
    for did, g in work.groupby("dsr_id"):
        universe = int(g["store_id"].nunique()) if "store_id" in g.columns else int(len(g))
        visited = int((g.get("call_status", pd.Series(dtype=str)) != "Unvisited").sum()) if "call_status" in g.columns else universe
        billed_n = int((pd.to_numeric(g.get("billed_mt"), errors="coerce").fillna(0) > 0.005).sum()) if "billed_mt" in g.columns else 0
        if billed_n == 0 and "volume_mt" in g.columns:
            billed_n = int((pd.to_numeric(g["volume_mt"], errors="coerce").fillna(0) > 0.005).sum())
        cycle = pd.to_numeric(g.get("cycle_days"), errors="coerce").replace(0, np.nan).fillna(30)
        required_freq = float((days_done / cycle.clip(lower=7)).sum())
        span_unique = universe / feasible_mtd if feasible_mtd else np.inf
        span_freq = required_freq / feasible_mtd if feasible_mtd else np.inf
        visit_rate = visited / universe if universe else np.nan
        strike_of_visits = billed_n / visited if visited else np.nan
        billed_mt = float(pd.to_numeric(g.get("billed_mt"), errors="coerce").fillna(0).sum()) if "billed_mt" in g.columns else float(
            pd.to_numeric(g.get("volume_mt"), errors="coerce").fillna(0).sum()
        )
        expected = float(pd.to_numeric(g.get("expected_mt"), errors="coerce").fillna(0).sum())
        ams = float(pd.to_numeric(g.get("ams_3m"), errors="coerce").fillna(0).sum()) if "ams_3m" in g.columns else expected
        billed_shops = max(billed_n, 1)
        drop_now = billed_mt / billed_shops if billed_n else np.nan
        exp_drop = expected / max(int((pd.to_numeric(g.get("expected_mt"), errors="coerce").fillna(0) > 0).sum()), 1)
        if "typical_drop_mt" in g.columns:
            typ = pd.to_numeric(g["typical_drop_mt"], errors="coerce")
            if typ.notna().any():
                exp_drop = float(typ.median())
        drop_index = (drop_now / exp_drop) if exp_drop and drop_now == drop_now and exp_drop == exp_drop and exp_drop > 0 else np.nan
        remaining = float(pd.to_numeric(g.get("remaining_mt"), errors="coerce").fillna(0).sum()) if "remaining_mt" in g.columns else max(
            0.0, expected - billed_mt
        )
        ask = float(pd.to_numeric(g.get("week_target_mt"), errors="coerce").fillna(0).sum()) if "week_target_mt" in g.columns else remaining
        label, why = _label(
            span_unique=span_unique,
            visit_rate=visit_rate,
            strike_of_visits=strike_of_visits,
            drop_index=drop_index,
            remaining=remaining,
            expected=expected,
        )
        cap = int(min(DSR_DAY_CAP, max(8, round(feasible_left / max(left, 1))))) if left else int(DSR_DAY_CAP)
        rows.append(
            {
                "grain_id": did,
                "dsr_name": dsr_display_name(did),
                "city": str(g["city"].iloc[0]),
                "distributor": str(g["distributor"].iloc[0]),
                "universe": universe,
                "visited": visited,
                "billed": billed_n,
                "visit_rate": visit_rate,
                "strike_of_visits": strike_of_visits,
                "span_unique": span_unique,
                "span_freq": span_freq,
                "feasible_mtd": feasible_mtd,
                "feasible_left": feasible_left,
                "day_cap": cap,
                "drop_index": drop_index,
                "billed_mt": billed_mt,
                "expected_mt": expected,
                "ams_3m": ams,
                "remaining_mt": remaining,
                "week_target_mt": ask,
                "label": label,
                "why": why,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {
        LABEL_OVERLOADED: 0,
        LABEL_NOT_WORKING: 1,
        LABEL_NOT_CONVERTING: 2,
        LABEL_NOT_LIFTING: 3,
        LABEL_FINE: 4,
    }
    out["_ord"] = out["label"].map(order).fillna(9)
    out = out.sort_values(["_ord", "week_target_mt", "remaining_mt"], ascending=[True, False, False])
    return out.drop(columns=["_ord"])


def _label(
    span_unique: float,
    visit_rate: float,
    strike_of_visits: float,
    drop_index: float,
    remaining: float,
    expected: float,
) -> tuple[str, str]:
    span = float(span_unique) if span_unique == span_unique else 0.0
    visit = float(visit_rate) if visit_rate == visit_rate else 0.0
    strike = float(strike_of_visits) if strike_of_visits == strike_of_visits else 0.0
    drop = float(drop_index) if drop_index == drop_index else 1.0
    if span >= float(SPAN_OVERLOAD):
        return (
            LABEL_OVERLOADED,
            f"Universe does not fit the call budget (span {span:.1f}×). Split the beat or add a DSR — "
            "this is headcount, not effort.",
        )
    if visit < 0.70 and span < float(SPAN_OVERLOAD):
        return (
            LABEL_NOT_WORKING,
            f"Visit {visit*100:.0f}% with spare capacity (span {span:.1f}×). Ride-with and audit callage.",
        )
    if visit >= 0.85 and strike < 0.55:
        return (
            LABEL_NOT_CONVERTING,
            f"Visit {visit*100:.0f}% but billed only {strike*100:.0f}% of calls. Named convert list, not more coverage.",
        )
    if strike >= 0.70 and drop < 0.65 and remaining >= 0.25:
        return (
            LABEL_NOT_LIFTING,
            f"Strike is fine; drop is {drop*100:.0f}% of typical. Order size / stock / scheme on billed doors.",
        )
    if remaining <= 0.05 * max(expected, 0.25):
        return LABEL_FINE, "On or ahead of Expected. Leave this beat."
    return LABEL_FINE, "Inside the normal band. Do not raid this beat for a city firefight."


def cap_shops_per_dsr(shops: pd.DataFrame, per_dsr: int | None = None) -> pd.DataFrame:
    """Keep the top Ask doors each DSR can actually work. Remainder is a waiting list."""
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame()
    cap = int(per_dsr if per_dsr is not None else DSR_DAY_CAP)
    out = shops.copy()
    if "dsr_name" not in out.columns:
        return out.head(0)
    for col, default in (("city", ""), ("distributor", "")):
        if col not in out.columns:
            out[col] = default
    key = [
        dsr_unit_id(c, d, n)
        for c, d, n in zip(out["city"], out["distributor"], out["dsr_name"])
    ]
    out["_dsr_id"] = key
    sort_cols = [c for c in ("value_score", "week_target_mt", "remaining_mt") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    kept = out.groupby("_dsr_id", sort=False, group_keys=False).head(cap)
    return kept.drop(columns=["_dsr_id"], errors="ignore")


def apply_city_driver_priority(shops: pd.DataFrame) -> pd.DataFrame:
    """Karachi-style (high visit) boosts convert/lift; Islamabad-style boosts unvisited due."""
    if shops is None or shops.empty or "city" not in shops.columns:
        return shops if shops is not None else pd.DataFrame()
    out = shops.copy()
    if "value_score" not in out.columns:
        out["value_score"] = pd.to_numeric(out.get("week_target_mt"), errors="coerce").fillna(0)
    if "call_status" not in out.columns:
        return out
    visit_rate = (
        out.assign(_v=out["call_status"].ne("Unvisited").astype(float))
        .groupby("city")["_v"]
        .mean()
    )
    out["_city_visit"] = out["city"].map(visit_rate)
    score = pd.to_numeric(out["value_score"], errors="coerce").fillna(0)
    unvisited = out["call_status"].eq("Unvisited")
    convert = out["call_status"].eq("Visited · not billed")
    high = out["_city_visit"].fillna(0) >= 0.90
    low = out["_city_visit"].fillna(1) < 0.70
    score = np.where(high & unvisited, score * 0.70, score)
    score = np.where(high & convert, score * 1.30, score)
    score = np.where(low & unvisited, score * 1.30, score)
    out["value_score"] = score
    return out.drop(columns=["_city_visit"], errors="ignore")
