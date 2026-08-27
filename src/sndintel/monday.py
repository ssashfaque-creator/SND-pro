"""Monday NSM pack: summary + city/store detail with count-and-ask cells."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

from sndintel.action import (
    ACTION_CALL,
    ACTION_CONVERT,
    ACTION_LIFT,
    ACTION_RECOVER,
    action_buckets,
)
from sndintel.capacity import (
    LABEL_FINE,
    LABEL_NOT_CONVERTING,
    LABEL_NOT_LIFTING,
    LABEL_NOT_WORKING,
    LABEL_OVERLOADED,
)
from sndintel.identity import dsr_display_name, dsr_unit_id

HIGHLIGHT_DIST_N = 15
HIGHLIGHT_STORE_N = 30
WHO_TO_PUSH_N = 15

SUMMARY_NOTE = (
    "Pipeline Expected = Billed + Due unvisited + Drop variance + Not yet due. "
    "Immediate Ask is the 90-day expected drop for shops whose depletion ratio is ≥ 0.8 "
    "and who are not lapsed (DSLP > 3× API). Not yet due is pipeline, not today's Ask. "
    "Scorecard Expected (last-3 / last-6 + national day curve) is unchanged."
)


def fmt_mt(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "0.0"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "0.0"


def fmt_kg(value: Any) -> str:
    """Shop tables stay KG with thousands separators."""
    try:
        if value is None or pd.isna(value):
            return "0"
        kg = int(round(float(value) * 1000))
        return f"{kg:,}"
    except (TypeError, ValueError):
        return "0"


def count_ask(n: Any, mt: Any) -> str:
    doors = 0
    try:
        doors = int(n or 0)
    except (TypeError, ValueError):
        doors = 0
    return f"{doors} - {fmt_mt(mt)} MT"


def anchor_id(kind: str, name: str) -> str:
    raw = f"{kind}:{name}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-")[:24] or "x"
    return f"{kind}-{slug}-{digest}"


def monday_summary_sheets(sheets: list[tuple[str, str, str, pd.DataFrame]]):
    return [s for s in sheets if str(s[0])[:2].isdigit()]


def present_action_row(name: str, name_label: str, buckets: dict[str, float], extra: dict | None = None) -> dict[str, Any]:
    unvis = round(float(buckets.get("ask_call") or 0), 1)
    due_v = round(float(buckets.get("ask_convert") or 0), 1)
    again = round(float(buckets.get("ask_lift") or 0), 1)
    coming = round(float(buckets.get("not_yet_due_mt") or buckets.get("ask_coming") or 0), 1)
    doors_mt = round(unvis + due_v + again, 1)
    ask_mt = round(float(buckets.get("week_target_mt") or (unvis + due_v + again)), 1)
    row = {
        name_label: name,
        "Expected this month (MT)": round(float(buckets.get("expected_mt") or 0), 1),
        "Pipeline expected (MT)": round(float(buckets.get("pipeline_expected_mt") or 0), 1),
        "AMS (MT)": round(float(buckets.get("ams_3m") or 0), 1),
        "Billed (MT)": round(float(buckets.get("billed_mt") or 0), 1),
        "Ask rest of month (MT)": ask_mt,
        "Not yet due (MT)": round(float(buckets.get("not_yet_due_mt") or 0), 1),
        "Volume lost to variance (MT)": round(float(buckets.get("drop_variance_mt") or 0), 1),
        "Doors to visit": count_ask(buckets.get("n_doors"), doors_mt),
        "Unvisited": count_ask(buckets.get("n_call"), unvis),
        "Due visited": count_ask(buckets.get("n_convert"), due_v),
        "Another visit": count_ask(buckets.get("n_lift"), again),
        "Lapsing": count_ask(buckets.get("n_lapse"), 0.0),
        "Coming due": count_ask(buckets.get("n_coming"), coming),
    }
    if extra:
        # Keep name first; insert extra after name.
        out = {name_label: row[name_label]}
        out.update(extra)
        for k, v in row.items():
            if k != name_label:
                out[k] = v
        return out
    return row


def country_action_table(shops: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([present_action_row("Country", "Scope", action_buckets(shops))])


def city_action_table(shops: pd.DataFrame) -> pd.DataFrame:
    if shops is None or shops.empty or "city" not in shops.columns:
        return pd.DataFrame()
    rows = []
    for city, g in shops.groupby(shops["city"].astype(str)):
        b = action_buckets(g)
        if b["week_target_mt"] <= 0 and b["n_doors"] <= 0 and float(b.get("not_yet_due_mt") or 0) <= 0 and int(b.get("n_lapse") or 0) <= 0:
            continue
        rows.append(present_action_row(str(city), "City", b))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("Ask rest of month (MT)", ascending=False).reset_index(drop=True)


def distributor_action_table(shops: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    if shops is None or shops.empty or "distributor" not in shops.columns:
        return pd.DataFrame()
    rows = []
    for dist, g in shops.groupby(shops["distributor"].astype(str)):
        b = action_buckets(g)
        if b["week_target_mt"] <= 0 and b["n_doors"] <= 0 and float(b.get("not_yet_due_mt") or 0) <= 0 and int(b.get("n_lapse") or 0) <= 0:
            continue
        city = ""
        if "city" in g.columns and not g["city"].mode().empty:
            city = str(g["city"].mode().iloc[0])
        rows.append(present_action_row(str(dist), "Distributor", b, extra={"City": city}))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("Ask rest of month (MT)", ascending=False).reset_index(drop=True)
    if n is not None:
        out = out.head(int(n))
    return out


def distributors_in_city(shops: pd.DataFrame, city: str) -> pd.DataFrame:
    if shops is None or shops.empty:
        return pd.DataFrame()
    part = shops[shops["city"].astype(str) == str(city)]
    return distributor_action_table(part, n=None)


def store_table(shops: pd.DataFrame) -> pd.DataFrame:
    if shops is None or shops.empty:
        return pd.DataFrame(
            columns=["Shop", "City", "DSR", "Distributor", "Ask rest of month (KG)", "AMS (KG)", "Billed (KG)", "Call", "Do this"]
        )
    work = _sort_stores(shops)
    name = work["store_name"] if "store_name" in work.columns else work.get("Shop")
    return pd.DataFrame(
        {
            "Shop": list(name),
            "City": list(work.get("city", [])),
            "DSR": list(work.get("dsr_name", [])),
            "Distributor": list(work.get("distributor", [])),
            "Ask rest of month (KG)": [fmt_kg(v) for v in work.get("week_target_mt", [])],
            "AMS (KG)": [fmt_kg(v) for v in work.get("ams_3m", [])],
            "Billed (KG)": [fmt_kg(v) for v in work.get("billed_mt", work.get("volume_mt", []))],
            "Call": list(work.get("call_status", work.get("Call", []))),
            "Do this": list(work.get("instruction", [])),
        }
    )


def _sort_stores(shops: pd.DataFrame) -> pd.DataFrame:
    """DSR with the higher total Ask first; shops inside a DSR by Ask desc."""
    work = shops.copy()
    for col, default in (("city", ""), ("distributor", ""), ("dsr_name", "")):
        if col not in work.columns:
            work[col] = default
        work[col] = work[col].fillna(default).astype(str)
    ask = pd.to_numeric(work.get("week_target_mt"), errors="coerce").fillna(0)
    work["_ask"] = ask
    work["_dsr_id"] = [
        dsr_unit_id(c, d, n) for c, d, n in zip(work["city"], work["distributor"], work["dsr_name"])
    ]
    dsr_ask = work.groupby("_dsr_id")["_ask"].transform("sum")
    work["_dsr_ask"] = dsr_ask
    work = work.sort_values(["_dsr_ask", "_dsr_id", "_ask"], ascending=[False, True, False])
    return work.drop(columns=["_ask", "_dsr_id", "_dsr_ask"], errors="ignore")


def city_driver_table(units: pd.DataFrame, shops: pd.DataFrame | None = None) -> pd.DataFrame:
    if units is None or units.empty:
        return pd.DataFrame()
    cities = units[units["grain"] == "city"].copy()
    if cities.empty:
        return pd.DataFrame()
    rec = pd.to_numeric(cities.get("isolated_mt"), errors="coerce").fillna(0).clip(upper=0).abs()
    unb = pd.to_numeric(cities.get("from_unbilled_mt"), errors="coerce").fillna(0)
    unv = pd.to_numeric(cities.get("from_unvisited_mt"), errors="coerce").fillna(0)
    drop = pd.to_numeric(cities.get("from_drop_size_mt"), errors="coerce").fillna(0)
    driver = []
    for u, v, d in zip(unb, unv, drop):
        parts = [("unbilled", float(u)), ("unvisited", float(v)), ("drop size", float(d))]
        parts.sort(key=lambda x: abs(x[1]), reverse=True)
        driver.append(parts[0][0] if parts[0][1] else "on expected")
    if "ams_3m" in cities.columns:
        ams = pd.to_numeric(cities["ams_3m"], errors="coerce")
    else:
        ams = pd.Series(float("nan"), index=cities.index, dtype="float64")
    if shops is not None and not shops.empty and "city" in shops.columns:
        shop_ams = pd.to_numeric(shops.get("ams_3m"), errors="coerce").fillna(0)
        rolled = shops.assign(_city=shops["city"].astype(str), _ams=shop_ams).groupby("_city")["_ams"].sum()
        from_shops = cities["grain_id"].astype(str).map(rolled)
        empty = ams.isna() | (ams.fillna(0) <= 0)
        ams = ams.where(~empty, from_shops)
    if "universe" in cities.columns:
        universe = pd.to_numeric(cities["universe"], errors="coerce")
    else:
        universe = pd.Series(float("nan"), index=cities.index, dtype="float64")
    if shops is not None and not shops.empty and "city" in shops.columns and "store_id" in shops.columns:
        shop_uni = shops.groupby(shops["city"].astype(str))["store_id"].nunique()
        from_shops_n = cities["grain_id"].astype(str).map(shop_uni)
        missing_uni = universe.isna() | (universe.fillna(0) <= 0)
        universe = universe.where(~missing_uni, from_shops_n)
    out = pd.DataFrame(
        {
            "City": cities["grain_id"].astype(str),
            "AMS (MT)": pd.to_numeric(ams, errors="coerce").fillna(0).round(1),
            "Billed (MT)": pd.to_numeric(cities.get("volume_mt"), errors="coerce").round(1),
            "Expected (MT)": pd.to_numeric(cities.get("expected_mt"), errors="coerce").round(1),
            "Gap (MT)": rec.round(1),
            "Universe": pd.to_numeric(universe, errors="coerce").fillna(0).astype(int),
            "Visit %": (pd.to_numeric(cities.get("visit_rate"), errors="coerce") * 100).round(0),
            "Strike %": (pd.to_numeric(cities.get("strike_rate"), errors="coerce") * 100).round(0),
            "Driver": driver,
        }
    )
    return out.sort_values("Gap (MT)", ascending=False)


def who_to_push_table(cap: pd.DataFrame, shops: pd.DataFrame, n: int = WHO_TO_PUSH_N) -> pd.DataFrame:
    cols = [
        "DSR",
        "City",
        "Distributor",
        "Label",
        "Universe",
        "Visit %",
        "Strike of visits %",
        "AMS (MT)",
        "Billed (MT)",
        "Ask rest of month (MT)",
        "Why",
    ]
    if cap is None or cap.empty:
        return pd.DataFrame(columns=cols)
    work = cap[cap["label"] != LABEL_FINE] if "label" in cap.columns else cap
    if work.empty:
        work = cap
    show = work.head(int(n))
    rows = []
    for rec in show.itertuples(index=False):
        dsr = dsr_display_name(getattr(rec, "dsr_name", "") or getattr(rec, "grain_id", ""))
        city = str(getattr(rec, "city", "") or "")
        dist = str(getattr(rec, "distributor", "") or "")
        part = _dsr_shops(shops, city, dist, getattr(rec, "dsr_name", dsr))
        why = _dsr_why(str(getattr(rec, "label", "") or ""), part, fallback=str(getattr(rec, "why", "") or ""))
        visit = getattr(rec, "visit_rate", None)
        strike = getattr(rec, "strike_of_visits", None)
        rows.append(
            {
                "DSR": dsr,
                "City": city,
                "Distributor": dist,
                "Label": getattr(rec, "label", ""),
                "Universe": int(getattr(rec, "universe", 0) or 0),
                "Visit %": None if visit is None or pd.isna(visit) else int(round(float(visit) * 100)),
                "Strike of visits %": None if strike is None or pd.isna(strike) else int(round(float(strike) * 100)),
                "AMS (MT)": float(fmt_mt(getattr(rec, "ams_3m", 0))),
                "Billed (MT)": float(fmt_mt(getattr(rec, "billed_mt", 0))),
                "Ask rest of month (MT)": float(fmt_mt(getattr(rec, "week_target_mt", 0))),
                "Why": why,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _dsr_shops(shops: pd.DataFrame, city: str, dist: str, dsr_name: str) -> pd.DataFrame:
    if shops is None or shops.empty:
        return pd.DataFrame()
    work = shops
    if "city" in work.columns:
        work = work[work["city"].astype(str) == str(city)]
    if "distributor" in work.columns:
        work = work[work["distributor"].astype(str) == str(dist)]
    if "dsr_name" in work.columns:
        want = dsr_display_name(dsr_name)
        work = work[work["dsr_name"].map(dsr_display_name) == want]
    return work


def _dsr_why(label: str, shops: pd.DataFrame, fallback: str = "") -> str:
    b = action_buckets(shops)
    issues = [
        ("unvisited", int(b["n_call"]), float(b["ask_call"]), "not covering the beat"),
        ("visited · not billed", int(b["n_convert"]), float(b["ask_convert"]), "not converting calls"),
        ("another visit", int(b["n_lift"]), float(b["ask_lift"]), "not lifting drop size"),
        ("lapsing", int(b["n_lapse"]), float(b["ask_lapse"]), "lapsing doors"),
        ("coming due", int(b["n_coming"]), float(b["ask_coming"]), "volume still sitting in coming-due"),
    ]
    issues.sort(key=lambda x: x[2], reverse=True)
    top_name, top_n, top_ask, top_issue = issues[0]
    if top_ask <= 0 and fallback:
        text = fallback
        text = re.sub(r" \(span [0-9.]+×\)", "", text)
        return text
    lead = {
        LABEL_OVERLOADED: "Beat does not fit the call budget — this is headcount.",
        LABEL_NOT_WORKING: "Spare capacity, weak visit % — ride-with.",
        LABEL_NOT_CONVERTING: "Calls happened; shops did not buy.",
        LABEL_NOT_LIFTING: "Billed, but order size is light.",
        LABEL_FINE: "On Expected.",
    }.get(label, label)
    return (
        f"{lead} Key issue: {top_issue} "
        f"({top_n} {top_name}, {fmt_mt(top_ask)} MT still asked)."
    )


def operating_shops(shops: pd.DataFrame) -> pd.DataFrame:
    """Doors that still have rest-of-month Ask (work now + coming due)."""
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame()
    ask = pd.to_numeric(shops.get("week_target_mt"), errors="coerce").fillna(0)
    work = shops["action"].isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT}) if "action" in shops.columns else False
    coming = shops["coming_due"].fillna(False) if "coming_due" in shops.columns else False
    ams = pd.to_numeric(shops.get("ams_3m"), errors="coerce").fillna(0) if "ams_3m" in shops.columns else 1.0
    return shops.loc[(ask > 0.0005) | ((work | coming) & (ams > 1e-9))].copy()
