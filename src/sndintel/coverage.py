"""Active-universe coverage: unvisited vs visited-unbilled vs drop size.

Identity (one shop, then summed to a grain):

    opportunity = this shop's Expected (typical same month + trend, shrunk toward
    the city, paced). Fallback AMS × pace, else last-year × pace. 0 if the door
    has no history.
    visited     = visit count > 0 OR billed this period
    billed      = volume > 0

    from unvisited = −opportunity  if not visited
    from unbilled  = −opportunity  if visited and not billed
    from drop size = volume − opportunity  if billed

These three sum to billed volume − Σ opportunity, so they do not double-count.
Never-billed whitespace has opportunity 0 (no MT to recover) but still sits in
visit % and strike %. If the visit file is missing for the period, unvisited is
left at 0 and every missed door is booked as unbilled — we will not invent calls.

Remarks compare visit rate, billed/visited (productivity), and drop size.
Expected drop is Expected volume ÷ Expected billed shops (same last-3 / last-6
run-rate as Expected sales). National average is this period's country drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sndintel.isolate import robust_z
from sndintel.io_utils import shift_period


COVERAGE_UNIT_COLS = [
    "visited",
    "visit_rate",
    "productivity",
    "from_unvisited_mt",
    "from_unbilled_mt",
    "from_drop_size_mt",
    "visits",
    "opportunity_mt",
    "has_visit_file",
]


def build_coverage_book(
    stores: pd.DataFrame,
    shop_month: pd.DataFrame,
    visits: pd.DataFrame | None,
    period: str,
    pace: float = 1.0,
    ledger: pd.DataFrame | None = None,
    shop_expected: pd.DataFrame | None = None,
    city_expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if stores is None or stores.empty or not period:
        return pd.DataFrame()
    uni = stores.copy()
    if "in_universe" in uni.columns and (uni["in_universe"] == 1).any():
        uni = uni[uni["in_universe"] == 1]
    uni = uni.drop_duplicates("store_id")
    if uni.empty:
        return pd.DataFrame()
    uni["store_id"] = uni["store_id"].astype(str)
    yoy = shift_period(period, -12)
    sm = shop_month.copy() if shop_month is not None and not shop_month.empty else pd.DataFrame()
    if not sm.empty:
        sm["store_id"] = sm["store_id"].astype(str)
    now = (
        sm[sm["period"].astype(str) == str(period)][["store_id", "volume_mt"]]
        if not sm.empty and "period" in sm.columns
        else pd.DataFrame(columns=["store_id", "volume_mt"])
    )
    ly = (
        sm[sm["period"].astype(str) == str(yoy)][["store_id", "volume_mt"]].rename(columns={"volume_mt": "ly_mt"})
        if not sm.empty and "period" in sm.columns
        else pd.DataFrame(columns=["store_id", "ly_mt"])
    )
    from sndintel.briefing import ams_last_n

    ams = ams_last_n(sm, period, ["store_id"], ledger=ledger) if not sm.empty else pd.DataFrame(columns=["store_id", "ams_3m"])
    vis = pd.DataFrame(columns=["store_id", "visits"])
    has_visit_file = False
    if visits is not None and not visits.empty and "store_id" in visits.columns:
        v = visits.copy()
        v["store_id"] = v["store_id"].astype(str)
        if "period" in v.columns:
            v = v[v["period"].astype(str) == str(period)]
        if not v.empty:
            vis = v.groupby("store_id", as_index=False)["visits"].sum()
            has_visit_file = True

    cols = [c for c in ["store_id", "store_name", "distributor", "dsr_name", "city", "zone", "section"] if c in uni.columns]
    book = uni[cols].copy()
    book = book.merge(now, on="store_id", how="left")
    book = book.merge(ly, on="store_id", how="left")
    if not ams.empty:
        book = book.merge(ams, on="store_id", how="left")
    else:
        book["ams_3m"] = np.nan
    book = book.merge(vis, on="store_id", how="left")
    book["volume_mt"] = pd.to_numeric(book.get("volume_mt"), errors="coerce").fillna(0.0)
    book["ly_mt"] = pd.to_numeric(book.get("ly_mt"), errors="coerce").fillna(0.0)
    book["ams_3m"] = pd.to_numeric(book.get("ams_3m"), errors="coerce")
    book["visits"] = pd.to_numeric(book.get("visits"), errors="coerce").fillna(0.0)
    book["billed"] = (book["volume_mt"] > 0).astype(int)
    if has_visit_file:
        book["visited"] = ((book["visits"] > 0) | (book["billed"] == 1)).astype(int)
    else:
        book["visited"] = book["billed"]
    ams_v = book["ams_3m"].fillna(0.0)
    pace = float(pace or 1.0)
    book["opportunity_mt"] = np.where(ams_v > 1e-9, ams_v * pace, book["ly_mt"] * pace)
    if shop_expected is not None and not shop_expected.empty and "store_id" in shop_expected.columns:
        se = shop_expected[["store_id", "expected_mt"]].copy()
        se["store_id"] = se["store_id"].astype(str)
        book = book.merge(se, on="store_id", how="left", suffixes=("", "_se"))
        learned = pd.to_numeric(book.get("expected_mt"), errors="coerce")
        book["opportunity_mt"] = np.where(learned.fillna(0) > 1e-9, learned, book["opportunity_mt"])
        if "expected_mt_se" in book.columns:
            book = book.drop(columns=["expected_mt_se"])
    if city_expected is not None and not city_expected.empty and "city" in book.columns:
        targets = {}
        id_col = "grain_id" if "grain_id" in city_expected.columns else "city"
        for _, r in city_expected.iterrows():
            targets[str(r.get(id_col) or r.get("city") or "")] = float(r.get("expected_mt") or 0.0)
        scaled = book["opportunity_mt"].copy()
        for city, g in book.groupby(book["city"].astype(str), dropna=False):
            target = targets.get(str(city))
            total = float(pd.to_numeric(g["opportunity_mt"], errors="coerce").fillna(0).sum())
            if target is not None and target > 1e-9 and total > 1e-9:
                scaled.loc[g.index] = g["opportunity_mt"] * (target / total)
        book["opportunity_mt"] = scaled
    book["from_unvisited_mt"] = np.where(
        (book["visited"] == 0) & has_visit_file, -book["opportunity_mt"], 0.0
    )
    if has_visit_file:
        unbilled = (book["visited"] == 1) & (book["billed"] == 0)
    else:
        unbilled = book["billed"] == 0
    book["from_unbilled_mt"] = np.where(unbilled, -book["opportunity_mt"], 0.0)
    book["from_drop_size_mt"] = np.where(book["billed"] == 1, book["volume_mt"] - book["opportunity_mt"], 0.0)
    book["has_visit_file"] = has_visit_file
    book["call_status"] = np.where(
        book["billed"] == 1,
        "Billed",
        np.where(book["visited"] == 1, "Visited · not billed", "Unvisited"),
    )
    return book


def rollup_coverage(book: pd.DataFrame, keys: list[str] | None) -> pd.DataFrame:
    if book is None or book.empty:
        return pd.DataFrame()
    work = book.copy()
    if not keys:
        work["_all"] = "ALL"
        keys = ["_all"]
    for k in keys:
        if k not in work.columns:
            work[k] = "(unmapped)"
        work[k] = work[k].fillna("(unmapped)").astype(str).replace("", "(unmapped)")
    g = work.groupby(keys, dropna=False, as_index=False).agg(
        universe=("store_id", "nunique"),
        billed=("billed", "sum"),
        visited=("visited", "sum"),
        visits=("visits", "sum"),
        from_unvisited_mt=("from_unvisited_mt", "sum"),
        from_unbilled_mt=("from_unbilled_mt", "sum"),
        from_drop_size_mt=("from_drop_size_mt", "sum"),
        opportunity_mt=("opportunity_mt", "sum"),
        volume_mt=("volume_mt", "sum"),
        has_visit_file=("has_visit_file", "max"),
    )
    uni = g["universe"].replace(0, np.nan)
    vis = g["visited"].replace(0, np.nan)
    g["visit_rate"] = g["visited"] / uni
    g["strike_rate"] = g["billed"] / uni
    g["productivity"] = g["billed"] / vis
    if "_all" in g.columns:
        g = g.drop(columns=["_all"])
    return g


def allocate_recoverable_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Partition recoverable across drop size / unvisited / unbilled.

    Recoverable is not recalculated. The three From columns keep the same
    coverage identity as weights, then scale so they add to the hole versus
    Expected:

    * behind (recoverable > 0) — positive pieces of the hole, summing to recoverable
    * ahead (billed > Expected) — negative pieces of the surplus; recoverable stays 0
    * in line — zeros
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    drop = pd.to_numeric(out.get("from_drop_size_mt"), errors="coerce").fillna(0.0)
    unv = pd.to_numeric(out.get("from_unvisited_mt"), errors="coerce").fillna(0.0)
    unb = pd.to_numeric(out.get("from_unbilled_mt"), errors="coerce").fillna(0.0)
    rec = pd.to_numeric(out.get("recoverable_mt"), errors="coerce").fillna(0.0)
    iso = pd.to_numeric(out.get("isolated_mt"), errors="coerce")
    vol = pd.to_numeric(out.get("volume_mt"), errors="coerce")
    fair = pd.to_numeric(out.get("share_expected_mt"), errors="coerce")
    out["from_drop_size_raw_mt"] = drop
    out["from_unvisited_raw_mt"] = unv
    out["from_unbilled_raw_mt"] = unb

    new_drop: list[float] = []
    new_unv: list[float] = []
    new_unb: list[float] = []
    for i in range(len(out)):
        r = float(rec.iloc[i] or 0.0)
        d, u, b = float(drop.iloc[i]), float(unv.iloc[i]), float(unb.iloc[i])
        isolated = iso.iloc[i] if i < len(iso) else np.nan
        isolated_f = float(isolated) if pd.notna(isolated) else 0.0
        billed = vol.iloc[i] if i < len(vol) else np.nan
        share = fair.iloc[i] if i < len(fair) else np.nan
        if pd.isna(share) and "expected_mt" in out.columns:
            share = pd.to_numeric(out["expected_mt"], errors="coerce").iloc[i]
        ahead = isolated_f > 1e-9
        if not ahead and pd.notna(billed) and pd.notna(share):
            ahead = float(billed) > float(share) + 1e-9
        if r > 1e-9:
            target = r
            weights = [max(0.0, -d), max(0.0, -u), max(0.0, -b)]
        elif ahead:
            surplus = isolated_f
            if surplus <= 1e-9 and pd.notna(billed) and pd.notna(share):
                surplus = max(0.0, float(billed) - float(share))
            target = -max(0.0, surplus)
            weights = [max(0.0, d), max(0.0, u), max(0.0, b)]
        else:
            target = 0.0
            weights = [0.0, 0.0, 0.0]
        total_w = sum(weights)
        if abs(target) < 1e-12:
            vals = [0.0, 0.0, 0.0]
        elif total_w < 1e-12:
            vals = [target, 0.0, 0.0]
        else:
            vals = [target * w / total_w for w in weights]
        new_drop.append(vals[0])
        new_unv.append(vals[1])
        new_unb.append(vals[2])
    out["from_drop_size_mt"] = new_drop
    out["from_unvisited_mt"] = new_unv
    out["from_unbilled_mt"] = new_unb
    return out


def attach_coverage_split(units: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """Merge grain-level unvisited / unbilled / drop-size onto unit scorecards."""
    if units is None or units.empty:
        return units
    out = units.copy()
    if book is None or book.empty:
        for col in COVERAGE_UNIT_COLS:
            if col not in out.columns:
                out[col] = None
        return out
    city_r = rollup_coverage(book, ["city"])
    dist_r = rollup_coverage(book, ["city", "distributor"])
    dsr_r = rollup_coverage(book, ["city", "dsr_name"])
    nat_r = rollup_coverage(book, None)
    out = _merge_grain(out, "city", city_r, {"city": "grain_id"})
    out = _merge_grain(out, "distributor", dist_r, {"city": "parent_id", "distributor": "grain_id"})
    out = _merge_grain(out, "dsr", dsr_r, {"city": "parent_id", "dsr_name": "grain_id"})
    out = _merge_grain(out, "national", nat_r, {})
    return out


def _merge_grain(units: pd.DataFrame, grain: str, rolled: pd.DataFrame, key_map: dict[str, str]) -> pd.DataFrame:
    mask = units["grain"] == grain
    if not mask.any() or rolled is None or rolled.empty:
        return units
    keep = [
        c
        for c in [
            "universe",
            "billed",
            "visited",
            "visits",
            "from_unvisited_mt",
            "from_unbilled_mt",
            "from_drop_size_mt",
            "opportunity_mt",
            "visit_rate",
            "strike_rate",
            "productivity",
            "has_visit_file",
        ]
        if c in rolled.columns
    ]
    if grain == "national":
        if rolled.empty:
            return units
        row = rolled.iloc[0]
        for c in keep:
            out_col = c
            units.loc[mask, out_col] = row.get(c)
        return units
    right = rolled.copy()
    rename = {src: dst for src, dst in key_map.items() if src in right.columns}
    right = right.rename(columns=rename)
    join_on = list(rename.values())
    left = units.loc[mask].copy()
    left["_cv_i"] = np.arange(len(left))
    for c in join_on:
        left[c] = left[c].astype(str)
        right[c] = right[c].astype(str)
    payload = right[join_on + keep].copy()
    payload = payload.rename(columns={c: f"_cv_{c}" for c in keep})
    merged = left.merge(payload, on=join_on, how="left").sort_values("_cv_i")
    for c in keep:
        src = f"_cv_{c}"
        if src in merged.columns:
            units.loc[mask, c] = merged[src].to_numpy()
    return units


def attach_remarks(df: pd.DataFrame, parent: dict[str, Any] | None, sibling_z: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Four-point remarks: trend, visit coverage, productivity, drop size."""
    if df is None or df.empty:
        return df
    out = df.copy()
    parent = parent or {}
    z_visit = sibling_z.get("visit_rate") if sibling_z else None
    z_prod = sibling_z.get("productivity") if sibling_z else None
    z_drop = sibling_z.get("drop") if sibling_z else None
    remarks = []
    for i, r in out.iterrows():
        remarks.append(
            _remark_row(
                r,
                parent,
                z_visit.loc[i] if z_visit is not None and i in z_visit.index else None,
                z_prod.loc[i] if z_prod is not None and i in z_prod.index else None,
                z_drop.loc[i] if z_drop is not None and i in z_drop.index else None,
            )
        )
    out["remarks"] = remarks
    return out


def sibling_z_frame(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df is None or df.empty:
        return {}
    visit = pd.to_numeric(df.get("visit_rate"), errors="coerce")
    prod = pd.to_numeric(df.get("productivity"), errors="coerce")
    drop = pd.to_numeric(df.get("from_drop_size_raw_mt"), errors="coerce")
    if drop is None or drop.isna().all():
        drop = pd.to_numeric(df.get("from_drop_size_mt"), errors="coerce")
    billed = pd.to_numeric(df.get("billed"), errors="coerce").replace(0, np.nan)
    drop_per = drop / billed
    return {
        "visit_rate": robust_z(visit),
        "productivity": robust_z(prod),
        "drop": robust_z(drop_per),
    }


def _remark_row(r: pd.Series, parent: dict[str, Any], z_visit, z_prod, z_drop) -> str:
    bits = [_trend_bit(r, parent), _coverage_bit(r, parent, z_visit), _productivity_bit(r, parent, z_prod), _drop_bit(r, parent, z_drop)]
    return "\n".join(f"• {b}" for b in bits if b)


def _trend_bit(r: pd.Series, parent: dict[str, Any]) -> str:
    vol = _num(r.get("volume_mt"))
    ams = _num(r.get("ams_3m"))
    vs_ams = _num(r.get("vs_ams_mt"))
    ly = _num(r.get("ly_mt"))
    exp = _num(r.get("expected_mt"))
    parts = ["Trend:"]
    if ams is not None and ams > 1e-9 and vs_ams is not None:
        pct = 100.0 * vs_ams / ams
        parts.append(f"{vs_ams:+.0f} MT vs AMS ({pct:+.0f}%).")
    elif vol is not None:
        parts.append(f"billed {vol:.0f} MT.")
    if exp is not None and vol is not None:
        gap = vol - exp
        parts.append(f"{gap:+.0f} MT vs Expected.")
    if ly is not None and ly > 1e-9 and vol is not None:
        yoy = 100.0 * (vol - ly) / ly
        p_yoy = parent.get("yoy_pct")
        if p_yoy is not None:
            parts.append(f"YoY {yoy:+.0f}% vs country {p_yoy:+.0f}%.")
        else:
            parts.append(f"YoY {yoy:+.0f}%.")
    return " ".join(parts) if len(parts) > 1 else ""


def _coverage_bit(r: pd.Series, parent: dict[str, Any], z) -> str:
    uni = _num(r.get("universe"))
    visited = _num(r.get("visited"))
    rate = _num(r.get("visit_rate"))
    has_vis = r.get("has_visit_file")
    try:
        has_vis = bool(has_vis) if pd.notna(has_vis) else False
    except (TypeError, ValueError):
        has_vis = bool(has_vis)
    if not has_vis or uni is None or uni <= 0:
        return "Coverage: no visit file this week — unvisited vs unbilled cannot be split."
    pct = 100.0 * (rate if rate is not None else (visited or 0) / uni)
    p_rate = _num(parent.get("visit_rate"))
    vs = f" vs country {100.0 * p_rate:.0f}%" if p_rate is not None else ""
    tag = _z_tag(z)
    return f"Coverage: visited {pct:.0f}% of {int(uni)} doors{vs}{tag}."


def _productivity_bit(r: pd.Series, parent: dict[str, Any], z) -> str:
    billed = _num(r.get("billed"))
    visited = _num(r.get("visited"))
    uni = _num(r.get("universe"))
    prod = _num(r.get("productivity"))
    strike = _num(r.get("strike_rate"))
    if billed is None or uni is None:
        return ""
    if prod is None and visited and visited > 0:
        prod = billed / visited
    p_prod = _num(parent.get("productivity"))
    vs = f" vs country {100.0 * p_prod:.0f}%" if p_prod is not None else ""
    tag = _z_tag(z)
    strike_s = f"; strike {100.0 * strike:.0f}% of universe" if strike is not None else ""
    if prod is not None and visited:
        return f"Productivity: billed {100.0 * prod:.0f}% of calls{vs}{tag}{strike_s}."
    return f"Productivity: billed {int(billed)} of {int(uni)} universe doors{strike_s}."


def _drop_bit(r: pd.Series, parent: dict[str, Any], z) -> str:
    del z
    drop = _num(r.get("drop_size_mt"))
    billed = _num(r.get("billed"))
    vol = _num(r.get("volume_mt"))
    if drop is None and billed and billed > 0 and vol is not None:
        drop = vol / billed
    if drop is None:
        return ""
    exp_drop = _num(r.get("expected_drop_size_mt"))
    nat = _num(parent.get("drop_size_mt"))
    if exp_drop is not None and nat is not None:
        return (
            f"Drop size: {drop:.2f} vs expected {exp_drop:.2f} vs national average {nat:.2f} "
            f"(MT per billed shop)."
        )
    if exp_drop is not None and nat is None:
        return f"Drop size: {drop:.2f} vs expected {exp_drop:.2f} (MT per billed shop, national average)."
    if nat is None:
        return f"Drop size: {drop:.2f} MT per billed shop (national average)."
    return f"Drop size: {drop:.2f} vs national average {nat:.2f} (MT per billed shop)."


def _z_tag(z) -> str:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return ""
    z = float(z)
    if z <= -1.0:
        return " (weak vs country)"
    if z >= 1.0:
        return " (ahead of country)"
    return " (in line with country)"


def _num(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
