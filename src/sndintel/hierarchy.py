"""Hierarchical volume gap engine: city → distributor → DSR → shop.

Not a canned strategy. Every row is computed from the warehouse:

* Expected is recent run-rate at every grain: last three closed months (same
  window as AMS), blended with the last-six-month median, paced if MTD is
  open. Calendar-month seasonality is not applied. Children add to the parent
  Expected. Last year is one input, not the call.
* Gap = hole versus that Expected. From drop / unvisited / unbilled
  partition it.
* National hole = billed versus national Expected (additive after reconcile).
* Inside a city, hole = like-for-like drop size + lost-shop volume − new volume.
* Diagnosis is whichever component dominates — drop size vs coverage vs whitespace.
* Targets are the named distributors / DSRs / shops that contribute most to that hole.

Tuned on the real Shop SKU Wise extract: ~18k billed POPs, ~30k universe,
Karachi ~50% of volume, median shop drop ~0.02 MT, Eva 5L pouches dominate mix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sndintel.config import MIN_MATERIAL_MT
from sndintel.coverage import attach_coverage_split, build_coverage_book
from sndintel.features import latest_period
from sndintel.io_utils import shift_period
from sndintel.isolate import (
    apply_coverage_velocity,
    apply_expected_gap,
    attach_mix,
    attach_seasonal_mom,
    empirical_bayes,
    k_from_ly,
    rewrite_action,
    seasonal_mom_expected,
    situation_brief,
    sku_industry_mix,
)
from sndintel.mtd import period_state
from sndintel.season import (
    apply_child_expected,
    apply_city_expected,
    fit_seasonality,
    fit_shop_expected,
    intra_month_fraction,
    reconcile_expected,
)
from sndintel.storage import dumps


UNIT_COLUMNS = [
    "period",
    "grain",
    "grain_id",
    "parent_grain",
    "parent_id",
    "zone",
    "city",
    "volume_mt",
    "ly_mt",
    "expected_mt",
    "run_rate_mt",
    "gap_mt",
    "gap_vs_ly_mt",
    "gap_pct",
    "lfl_now",
    "lfl_ly",
    "lfl_gap",
    "lost_n",
    "lost_mt",
    "new_n",
    "new_mt",
    "billed",
    "billed_ly",
    "universe",
    "strike_rate",
    "diagnosis",
    "verdict",
    "do_this_week",
    "contrib_national_gap",
    "share_expected_mt",
    "competitive_mt",
    "isolated_mt",
    "z_score",
    "focus_score",
    "coverage_effect_mt",
    "velocity_effect_mt",
    "interaction_effect_mt",
    "mix_effect_mt",
    "wd",
    "nd",
    "parent_index",
    "intra_month_frac",
    "seasonal_mom_index",
    "mom_expected_mt",
    "mom_gap_mt",
    "seasonal_index",
    "seasonal_typical_mt",
    "situation",
    "metrics_json",
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

TARGET_COLUMNS = [
    "period",
    "rank",
    "grain",
    "entity_id",
    "entity_name",
    "city",
    "zone",
    "distributor",
    "dsr_name",
    "section",
    "volume_mt",
    "ly_mt",
    "gap_mt",
    "diagnosis",
    "action",
    "why",
    "competitive_mt",
    "isolated_mt",
    "z_score",
    "focus_score",
    "situation",
]


@dataclass
class HierarchyPack:
    period: str
    yoy_period: str
    mtd: dict[str, Any]
    national: dict[str, Any]
    units: pd.DataFrame = field(default_factory=pd.DataFrame)
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)
    seasonality: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_hierarchy_pack(
    shop_month: pd.DataFrame,
    stores: pd.DataFrame | None,
    features: pd.DataFrame | None = None,  # noqa: ARG001 — kept for pipeline signature
    ledger: pd.DataFrame | None = None,
    facts: pd.DataFrame | None = None,
    mtd_obs: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
) -> HierarchyPack:
    period = latest_period(shop_month)
    if not period or shop_month is None or shop_month.empty:
        return HierarchyPack(period=period or "", yoy_period="", mtd={}, national={})
    mtd = period_state(ledger, period)
    yoy_p = shift_period(period, -12)
    intra_frac, intra_src = intra_month_fraction(
        mtd.get("as_of_day"),
        mtd.get("days_in_month"),
        mtd_obs,
        open_mtd=bool(mtd.get("open")),
    )
    pace = float(intra_frac)
    factor = (1.0 / pace) if pace > 1e-9 else 1.0
    if not mtd.get("open"):
        factor = 1.0

    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str)
    for col in ("city", "zone", "distributor", "dsr_name", "section"):
        if col not in sm.columns:
            sm[col] = "(unmapped)"
        sm[col] = sm[col].fillna("(unmapped)").replace("", "(unmapped)")
    if "store_name" not in sm.columns:
        sm["store_name"] = ""
    sm["store_name"] = sm["store_name"].fillna("")

    cur = sm[sm["period"] == period].copy()
    ly = sm[sm["period"] == yoy_p].copy()
    if cur.empty:
        return HierarchyPack(period=period, yoy_period=yoy_p, mtd=mtd, national={})

    universe = _universe_by_keys(stores, ["city"], fallback=sm)
    city_units = _grain_bridge(cur, ly, ["city"], universe, pace, factor)
    if "zone" in cur.columns:
        zone_map = cur.groupby("city")["zone"].agg(_mode_or_first)
        city_units["zone"] = city_units["grain_id"].map(zone_map)
    city_units["grain"] = "city"
    city_units["parent_grain"] = "national"
    city_units["parent_id"] = "ALL"
    city_units["city"] = city_units["grain_id"]
    season = fit_seasonality(sm, period)
    city_units = apply_city_expected(city_units, season, pace)
    nat_expected = (season.expected_full_national * pace) if season.expected_full_national else float(city_units["expected_mt"].sum())
    nat_parent = pd.DataFrame([{"grain_id": "ALL", "expected_mt": nat_expected}])
    city_units["parent_id"] = "ALL"
    city_units = reconcile_expected(city_units, nat_parent, intra_frac=pace)
    nat_now = float(city_units["volume_mt"].sum())
    nat_ly = float(city_units["ly_mt"].sum())
    city_k = k_from_ly(city_units["ly_mt"], default=1.0)
    city_units = apply_coverage_velocity(city_units)
    city_units = apply_expected_gap(city_units, city_k)
    facts_city = _facts_with_geo(facts, stores, sm, period, yoy_p)
    if facts_city is not None:
        fnow, fly = facts_city
        mix = sku_industry_mix(fnow, fly, ["city"])
        city_units = attach_mix(city_units, mix.rename(columns={"city": "grain_id"}), ["grain_id"])
    mom = seasonal_mom_expected(sm, period, ["city"])
    city_units = attach_seasonal_mom(
        city_units, mom.rename(columns={"city": "grain_id"}) if not mom.empty else mom, ["grain_id"]
    )
    city_units["intra_month_frac"] = pace
    city_units["contrib_national_gap"] = city_units["isolated_mt"]
    city_units = _annotate_units(city_units, mtd, grain_label="city")
    city_units = _rewrite_actions(city_units, mtd, "city")

    dist_units = _grain_bridge(
        cur, ly, ["city", "distributor"], _universe_by_keys(stores, ["city", "distributor"], fallback=sm), pace, factor
    )
    dist_units["grain"] = "distributor"
    dist_units["parent_grain"] = "city"
    dist_units["parent_id"] = dist_units["city"].astype(str)
    dist_units["grain_id"] = dist_units["distributor"].astype(str)
    dist_units = apply_child_expected(dist_units, sm, period, ["city", "distributor"], season, pace)
    dist_units = reconcile_expected(dist_units, city_units, intra_frac=pace)
    dist_units = _enrich_children(dist_units, default_k=0.5)
    dist_units["intra_month_frac"] = pace
    dist_units["contrib_national_gap"] = dist_units["isolated_mt"]
    dist_units = _annotate_units(dist_units, mtd, grain_label="distributor")
    dist_units = _rewrite_actions(dist_units, mtd, "distributor")

    dsr_units = _grain_bridge(
        cur, ly, ["city", "dsr_name"], _universe_by_keys(stores, ["city", "dsr_name"], fallback=sm), pace, factor
    )
    dsr_units["grain"] = "dsr"
    dsr_units["parent_grain"] = "city"
    dsr_units["parent_id"] = dsr_units["city"].astype(str)
    dsr_units["grain_id"] = dsr_units["dsr_name"].astype(str)
    dsr_units = apply_child_expected(dsr_units, sm, period, ["city", "dsr_name"], season, pace)
    dsr_units = reconcile_expected(dsr_units, city_units, intra_frac=pace)
    dsr_units = _enrich_children(dsr_units, default_k=0.2)
    dsr_units["intra_month_frac"] = pace
    dsr_units["contrib_national_gap"] = dsr_units["isolated_mt"]
    dsr_units = _annotate_units(dsr_units, mtd, grain_label="dsr")
    dsr_units = _rewrite_actions(dsr_units, mtd, "dsr")

    section_units = _grain_bridge(
        cur, ly, ["city", "section"], _universe_by_keys(stores, ["city", "section"], fallback=sm), pace, factor
    )
    section_units["grain"] = "section"
    section_units["parent_grain"] = "city"
    section_units["parent_id"] = section_units["city"].astype(str)
    section_units["grain_id"] = section_units["section"].astype(str)
    section_units = apply_child_expected(section_units, sm, period, ["city", "section"], season, pace)
    section_units = reconcile_expected(section_units, city_units, intra_frac=pace)
    section_units = _enrich_children(section_units, default_k=0.3)
    section_units["intra_month_frac"] = pace
    section_units["contrib_national_gap"] = section_units["isolated_mt"]
    section_units = _annotate_units(section_units, mtd, grain_label="section")
    section_units = _rewrite_actions(section_units, mtd, "section")

    nat_gap = float(city_units["gap_mt"].sum())
    nat_row = _national_unit(city_units, period, mtd, pace, factor)
    if not nat_row.empty:
        nat_row["parent_index"] = 1.0
        nat_row["expected_mt"] = nat_expected
        nat_row["seasonal_typical_mt"] = season.expected_full_national
        nat_row["seasonal_index"] = season.national_index.get(season.month, 1.0)
        nat_row["gap_mt"] = nat_row["volume_mt"].fillna(0) - nat_row["expected_mt"]
        nat_row = apply_coverage_velocity(nat_row)
        nat_row = apply_expected_gap(nat_row, 1.0)
        nat_row["intra_month_frac"] = pace
        nat_row = _annotate_units(nat_row, mtd, grain_label="national")
        nat_row = _rewrite_actions(nat_row, mtd, "national")

    units = pd.concat(
        [nat_row, city_units, dist_units, dsr_units, section_units],
        ignore_index=True,
        sort=False,
    )
    units["period"] = period
    shop_exp = fit_shop_expected(sm, period, season, pace)
    if shop_exp is not None and not shop_exp.empty and "city" in shop_exp.columns and not city_units.empty:
        shop_exp = shop_exp.copy()
        shop_exp["parent_id"] = shop_exp["city"].astype(str)
        shop_exp = reconcile_expected(shop_exp, city_units, intra_frac=pace)
    book = build_coverage_book(
        stores, sm, visits, period, pace=pace, ledger=ledger, shop_expected=shop_exp, city_expected=city_units
    )
    units = attach_coverage_split(units, book)
    units = _fit_unit_columns(units)

    targets = _focus_targets(
        cur, ly, city_units, dist_units, dsr_units, section_units, stores, period, mtd, pace, shop_exp
    )
    national = {
        "volume_mt": nat_now,
        "ly_mt": nat_ly,
        "expected_mt": nat_expected
        if nat_expected
        else nat_ly * pace,
        "run_rate_mt": nat_now * factor,
        "gap_mt": nat_gap,
        "n_cities": int(city_units["grain_id"].nunique()) if not city_units.empty else 0,
        "open_mtd": bool(mtd.get("open")),
        "label": mtd.get("label") or period,
        "period": period,
        "diagnosis": str(nat_row.iloc[0]["diagnosis"]) if not nat_row.empty else "mixed",
        "verdict": str(nat_row.iloc[0]["verdict"]) if not nat_row.empty else "watch",
        "intra_month_frac": pace,
        "intra_month_source": intra_src,
        "n_history_periods": season.n_periods,
        "n_same_month": season.n_same_month,
        "seasonality_source": season.source,
        "seasonal_index": season.national_index.get(season.month, 1.0),
        "as_of_day": mtd.get("as_of_day"),
    }
    national.update(situation_brief(national, city_units))
    return HierarchyPack(
        period=period,
        yoy_period=yoy_p,
        mtd=mtd,
        national=national,
        units=units,
        targets=targets,
        seasonality=season.table,
    )


def _mode_or_first(s: pd.Series):
    m = s.dropna()
    if m.empty:
        return "(unmapped)"
    mode = m.mode()
    return str(mode.iloc[0]) if len(mode) else str(m.iloc[0])


def _rewrite_actions(df: pd.DataFrame, mtd: dict[str, Any], grain_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["do_this_week"] = [rewrite_action(r, mtd, grain_label) for _, r in out.iterrows()]
    return out


def _enrich_children(df: pd.DataFrame, default_k: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    k = k_from_ly(df["ly_mt"], default=default_k)
    out = apply_coverage_velocity(df)
    return apply_expected_gap(out, k)


def _facts_with_geo(
    facts: pd.DataFrame | None,
    stores: pd.DataFrame | None,
    sm: pd.DataFrame,
    period: str,
    yoy_p: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if facts is None or facts.empty:
        return None
    f = facts.copy()
    if "city" not in f.columns or f["city"].isna().all():
        if stores is not None and not stores.empty and "city" in stores.columns:
            geo = stores[["store_id", "city"]].drop_duplicates("store_id")
            geo["store_id"] = geo["store_id"].astype(str)
            f["store_id"] = f["store_id"].astype(str)
            f = f.merge(geo, on="store_id", how="left", suffixes=("", "_g"))
            if "city_g" in f.columns:
                f["city"] = f["city"].fillna(f["city_g"]) if "city" in f.columns else f["city_g"]
        if "city" not in f.columns or f["city"].isna().all():
            cmap = sm.groupby("store_id")["city"].last()
            f["store_id"] = f["store_id"].astype(str)
            f["city"] = f["store_id"].map(cmap)
    f["city"] = f["city"].fillna("(unmapped)").replace("", "(unmapped)")
    now = f[f["period"] == period]
    ly = f[f["period"] == yoy_p]
    if now.empty or ly.empty:
        return None
    return now, ly


def _universe_by_city(stores: pd.DataFrame | None) -> pd.Series:
    return _universe_by_keys(stores, ["city"])


def _universe_by_keys(
    stores: pd.DataFrame | None,
    keys: list[str],
    fallback: pd.DataFrame | None = None,
) -> pd.Series:
    """Unique shops on the master (or billed history) at this grain."""
    src = None
    if stores is not None and not stores.empty and "store_id" in stores.columns and all(k in stores.columns for k in keys):
        src = stores
    elif fallback is not None and not fallback.empty and "store_id" in fallback.columns and all(
        k in fallback.columns for k in keys
    ):
        src = fallback
    if src is None:
        return pd.Series(dtype=float)
    work = src.copy()
    for k in keys:
        work[k] = work[k].fillna("(unmapped)").replace("", "(unmapped)")
    return work.groupby(keys, dropna=False)["store_id"].nunique()


def _lookup_universe(universe: pd.Series | None, rec: dict, keys: list[str]) -> int:
    if universe is None or len(universe) == 0:
        return 0
    if len(keys) == 1:
        val = rec.get(keys[0], rec.get("grain_id"))
        if val in universe.index:
            return int(universe.loc[val])
        tup = (val,)
        if tup in universe.index:
            return int(universe.loc[tup])
        return 0
    key = tuple(rec.get(k) for k in keys)
    if key in universe.index:
        return int(universe.loc[key])
    return 0


def _billed_ids(part: pd.DataFrame) -> set[str]:
    if part is None or part.empty or "store_id" not in part.columns:
        return set()
    ids = part["store_id"].astype(str)
    if "billed" in part.columns:
        billed = pd.to_numeric(part["billed"], errors="coerce").fillna(0)
        return set(ids[billed == 1])
    if "volume_mt" in part.columns:
        vol = pd.to_numeric(part["volume_mt"], errors="coerce").fillna(0)
        return set(ids[vol > 0])
    return set(ids)


def _grain_bridge(
    cur: pd.DataFrame,
    ly: pd.DataFrame,
    keys: list[str],
    universe: pd.Series | None,
    pace: float,
    factor: float,
) -> pd.DataFrame:
    def _canon(k) -> tuple:
        if isinstance(k, tuple):
            return tuple(k)
        return (k,)

    empty = pd.DataFrame(columns=cur.columns)

    def _parts(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
        if df is None or df.empty:
            return {}
        out: dict[tuple, pd.DataFrame] = {}
        for raw, part in df.groupby(keys, dropna=False):
            out[_canon(raw)] = part
        return out

    cur_map = _parts(cur)
    ly_map = _parts(ly)
    rows = []
    for key in set(cur_map) | set(ly_map):
        rows.append(
            _bridge_row(
                key,
                keys,
                cur_map.get(key, empty),
                ly_map.get(key, empty),
                universe,
                pace,
                factor,
            )
        )
    return pd.DataFrame(rows)


def _bridge_row(
    key: tuple,
    keys: list[str],
    cpart: pd.DataFrame,
    lpart: pd.DataFrame,
    universe: pd.Series | None,
    pace: float,
    factor: float,
) -> dict[str, Any]:
    rec = {k: key[i] for i, k in enumerate(keys)}
    rec["grain_id"] = str(key[-1]) if key else ""
    c_ids = _billed_ids(cpart)
    l_ids = _billed_ids(lpart)
    both, lost, new = c_ids & l_ids, l_ids - c_ids, c_ids - l_ids
    vol = float(cpart["volume_mt"].sum()) if not cpart.empty and "volume_mt" in cpart.columns else 0.0
    ly_mt = float(lpart["volume_mt"].sum()) if not lpart.empty and "volume_mt" in lpart.columns else 0.0
    lfl_now = (
        float(cpart.loc[cpart["store_id"].astype(str).isin(both), "volume_mt"].sum())
        if not cpart.empty and both
        else 0.0
    )
    lfl_ly = (
        float(lpart.loc[lpart["store_id"].astype(str).isin(both), "volume_mt"].sum())
        if not lpart.empty and both
        else 0.0
    )
    lost_mt = (
        float(lpart.loc[lpart["store_id"].astype(str).isin(lost), "volume_mt"].sum())
        if not lpart.empty and lost
        else 0.0
    )
    new_mt = (
        float(cpart.loc[cpart["store_id"].astype(str).isin(new), "volume_mt"].sum())
        if not cpart.empty and new
        else 0.0
    )
    expected = ly_mt * pace
    rec.update(
        {
            "volume_mt": vol,
            "ly_mt": ly_mt,
            "expected_mt": expected,
            "run_rate_mt": vol * factor,
            "gap_mt": vol - expected,
            "gap_vs_ly_mt": vol * factor - ly_mt,
            "gap_pct": ((vol - expected) / expected * 100) if expected else None,
            "lfl_now": lfl_now,
            "lfl_ly": lfl_ly,
            "lfl_gap": lfl_now - lfl_ly,
            "lost_n": len(lost),
            "lost_mt": lost_mt,
            "new_n": len(new),
            "new_mt": new_mt,
            "billed": len(c_ids),
            "billed_ly": len(l_ids),
            "universe": _lookup_universe(universe, rec, keys),
            "strike_rate": None,
        }
    )
    uni = rec["universe"]
    rec["strike_rate"] = (rec["billed"] / uni) if uni else None
    return rec


def _national_unit(
    city_units: pd.DataFrame, period: str, mtd: dict[str, Any], pace: float, factor: float
) -> pd.DataFrame:
    if city_units.empty:
        return pd.DataFrame()
    hole = city_units.copy()
    hole["hole"] = hole["gap_mt"].clip(upper=0).abs()
    top_diag = "mixed"
    if hole["hole"].sum() > 0:
        by = hole.groupby("diagnosis")["hole"].sum().sort_values(ascending=False)
        top_diag = str(by.index[0])
    rec = {
        "period": period,
        "grain": "national",
        "grain_id": "ALL",
        "parent_grain": "",
        "parent_id": "-",
        "zone": "ALL",
        "city": "ALL",
        "volume_mt": float(city_units["volume_mt"].sum()),
        "ly_mt": float(city_units["ly_mt"].sum()),
        "expected_mt": float(city_units["expected_mt"].sum()),
        "run_rate_mt": float(city_units["volume_mt"].sum()) * factor,
        "gap_mt": float(city_units["gap_mt"].sum()),
        "gap_vs_ly_mt": float(city_units["gap_vs_ly_mt"].sum()),
        "gap_pct": None,
        "lfl_now": float(city_units["lfl_now"].sum()),
        "lfl_ly": float(city_units["lfl_ly"].sum()),
        "lfl_gap": float(city_units["lfl_gap"].sum()),
        "lost_n": int(city_units["lost_n"].sum()),
        "lost_mt": float(city_units["lost_mt"].sum()),
        "new_n": int(city_units["new_n"].sum()),
        "new_mt": float(city_units["new_mt"].sum()),
        "billed": int(city_units["billed"].sum()),
        "billed_ly": int(city_units["billed_ly"].sum()),
        "universe": int(city_units["universe"].sum()),
        "strike_rate": None,
        "contrib_national_gap": float(city_units["gap_mt"].sum()),
        "diagnosis": top_diag,
    }
    expected = rec["expected_mt"]
    rec["gap_pct"] = ((rec["volume_mt"] - expected) / expected * 100) if expected else None
    uni = rec["universe"]
    rec["strike_rate"] = (rec["billed"] / uni) if uni else None
    frame = pd.DataFrame([rec])
    return _annotate_units(frame, mtd, grain_label="national")


def _annotate_units(df: pd.DataFrame, mtd: dict[str, Any], grain_label: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    diagnoses, verdicts, actions = [], [], []
    for _, r in out.iterrows():
        d, v, a = _diagnose(r, mtd, grain_label)
        diagnoses.append(d)
        verdicts.append(v)
        actions.append(a)
    if "diagnosis" not in out.columns or out["diagnosis"].isna().all():
        out["diagnosis"] = diagnoses
    else:
        # Preserve a pre-set national diagnosis when present; still fill blanks.
        out["diagnosis"] = out["diagnosis"].where(out["diagnosis"].notna() & (out["diagnosis"] != ""), diagnoses)
    out["verdict"] = verdicts
    out["do_this_week"] = actions
    return out


def _diagnose(r: pd.Series, mtd: dict[str, Any], grain_label: str) -> tuple[str, str, str]:
    gap = float(r.get("gap_mt") or 0)
    ly = float(r.get("ly_mt") or 0)
    vol = float(r.get("volume_mt") or 0)
    expected = float(r.get("expected_mt") or 0)
    lfl_gap = float(r.get("lfl_gap") or 0)
    lost_mt = float(r.get("lost_mt") or 0)
    strike = r.get("strike_rate")
    uni = int(r.get("universe") or 0)
    name = str(r.get("grain_id") or r.get("city") or grain_label)
    if name == "ALL":
        name = "National"
    hole = max(0.0, -gap)
    lfl_share = (max(0.0, -lfl_gap) / hole) if hole >= 0.05 else 0.0
    lost_share = (lost_mt / hole) if hole >= 0.05 else 0.0

    preset = str(r.get("diagnosis") or "")
    if grain_label == "national" and preset in {"drop_size", "coverage", "whitespace", "mixed", "holding"}:
        diagnosis = preset
    elif uni >= 30 and strike is not None and not pd.isna(strike) and strike < 0.15:
        diagnosis = "whitespace"
    elif hole < 0.05 and gap >= -0.05:
        diagnosis = "holding"
    elif lfl_share >= 0.55:
        diagnosis = "drop_size"
    elif lost_share >= 0.40:
        diagnosis = "coverage"
    else:
        diagnosis = "mixed"

    pct = float(r["gap_pct"]) if pd.notna(r.get("gap_pct")) else None
    if pct is None:
        verdict = "watch"
    elif pct <= -25 and hole >= 0.5:
        verdict = "collapsing"
    elif pct <= -8:
        verdict = "behind"
    elif pct >= 8:
        verdict = "ahead"
    else:
        verdict = "on_pace"

    if mtd.get("open") and mtd.get("as_of_day"):
        should = (
            f"should have billed ~{expected:.1f} MT by day {mtd['as_of_day']} "
            f"(last year closed at {ly:.1f} MT)"
        )
    else:
        should = f"should be at last year's {ly:.1f} MT"
    doing = f"{name} billed {vol:.1f} MT"
    if diagnosis == "drop_size":
        action = (
            f"{doing}; {should}. {lfl_share*100:.0f}% of the hole is like-for-like drop size "
            f"on shops still billing — recover drop size, do not print a lost-shop list."
        )
    elif diagnosis == "coverage":
        action = (
            f"{doing}; {should}. {int(r.get('lost_n') or 0)} shops / {lost_mt:.1f} MT last year are quiet. "
            "Must-visit the material doors; cadence the tail."
        )
    elif diagnosis == "whitespace":
        action = (
            f"{doing} on {int(r.get('billed') or 0)} of {uni} universe shops "
            f"({(strike or 0)*100:.0f}% strike). Open a coverage plan; volume will not appear from callage on the same 20 doors."
        )
    elif diagnosis == "holding":
        action = f"{doing}; {should}. Hold drop size; do not load."
    else:
        action = (
            f"{doing}; {should}. Split the week: drop size on continuing shops "
            f"({max(0, -lfl_gap):.1f} MT) and recover material unbilled ({lost_mt:.1f} MT)."
        )
    return diagnosis, verdict, action


def _fit_unit_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=UNIT_COLUMNS)
    out = df.copy()
    if "metrics_json" not in out.columns:
        out["metrics_json"] = out.apply(
            lambda r: dumps(
                {
                    "lfl_share_of_hole": None
                    if not r.get("gap_mt") or r["gap_mt"] >= 0
                    else max(0.0, -float(r.get("lfl_gap") or 0)) / max(0.05, -float(r["gap_mt"])),
                }
            ),
            axis=1,
        )
    for col in UNIT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[UNIT_COLUMNS]


def _focus_targets(
    cur: pd.DataFrame,
    ly: pd.DataFrame,
    cities: pd.DataFrame,
    dists: pd.DataFrame,
    dsrs: pd.DataFrame,
    sections: pd.DataFrame,
    stores: pd.DataFrame | None,
    period: str,
    mtd: dict[str, Any],
    pace: float,
    shop_expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if cities.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    sort_col = "isolated_mt" if "isolated_mt" in cities.columns else "gap_mt"
    ranked_cities = cities.sort_values(sort_col)
    if "situation" in ranked_cities.columns:
        lagging = ranked_cities[ranked_cities["situation"] == "lagging"]
        focus_cities = lagging if not lagging.empty else ranked_cities.head(8)
    else:
        focus_cities = ranked_cities.head(10)
        extra = ranked_cities[ranked_cities["gap_mt"] <= -1.0]
        focus_cities = pd.concat([focus_cities, extra]).drop_duplicates("grain_id")

    shop_gap = _shop_gaps(cur, ly, pace, shop_expected=shop_expected)
    if not shop_gap.empty:
        shop_gap["competitive_mt"] = shop_gap["volume_mt"] - shop_gap["expected_mt"]
        k_shop = k_from_ly(shop_gap["ly_mt"], 0.05)
        size = np.maximum(
            pd.to_numeric(shop_gap["ly_mt"], errors="coerce").fillna(0.0),
            pd.to_numeric(shop_gap["expected_mt"], errors="coerce").fillna(0.0),
        )
        shop_gap["isolated_mt"] = [
            empirical_bayes(c, s, k_shop) for c, s in zip(shop_gap["competitive_mt"], size)
        ]
    else:
        shop_gap["competitive_mt"] = shop_gap.get("gap_mt", 0)
        shop_gap["isolated_mt"] = shop_gap.get("gap_mt", 0)

    for _, city in focus_cities.iterrows():
        city_name = str(city["grain_id"])
        diagnosis = str(city["diagnosis"])
        zone = city.get("zone")
        cd = dists[dists["parent_id"] == city_name].copy() if not dists.empty else dists
        if not cd.empty:
            cd = cd.sort_values("isolated_mt" if "isolated_mt" in cd.columns else "gap_mt")
        for rec in cd.itertuples(index=False):
            iso = float(getattr(rec, "isolated_mt", rec.gap_mt) or 0)
            sit = str(getattr(rec, "situation", "") or "")
            if iso > 0.15:
                continue
            if len([r for r in rows if r["city"] == city_name and r["grain"] == "distributor"]) >= 5:
                break
            rows.append(
                _target(
                    period,
                    "distributor",
                    rec.grain_id,
                    rec.grain_id,
                    city_name,
                    zone,
                    rec.grain_id,
                    None,
                    None,
                    rec.volume_mt,
                    rec.ly_mt,
                    rec.gap_mt,
                    rec.diagnosis,
                    rec.do_this_week,
                    f"{rec.grain_id} is {iso:+.1f} MT vs Expected (billed {float(rec.volume_mt):.1f} vs {float(getattr(rec, 'expected_mt', rec.ly_mt) or 0):.1f}).",
                    iso,
                    iso,
                    float(getattr(rec, "z_score", 0) or 0),
                    float(getattr(rec, "focus_score", 0) or 0),
                    sit or None,
                )
            )
        cs = dsrs[dsrs["parent_id"] == city_name].copy() if not dsrs.empty else dsrs
        if not cs.empty:
            cs = cs.sort_values("isolated_mt" if "isolated_mt" in cs.columns else "gap_mt")
        for rec in cs.itertuples(index=False):
            iso = float(getattr(rec, "isolated_mt", rec.gap_mt) or 0)
            sit = str(getattr(rec, "situation", "") or "")
            if iso > 0.15:
                continue
            if len([r for r in rows if r["city"] == city_name and r["grain"] == "dsr"]) >= 4:
                break
            rows.append(
                _target(
                    period,
                    "dsr",
                    rec.grain_id,
                    rec.grain_id,
                    city_name,
                    zone,
                    None,
                    rec.grain_id,
                    None,
                    rec.volume_mt,
                    rec.ly_mt,
                    rec.gap_mt,
                    rec.diagnosis,
                    f"Ride-with {rec.grain_id} this week. {rec.do_this_week}",
                    f"{rec.grain_id} is {iso:+.1f} MT vs Expected.",
                    iso,
                    iso,
                    float(getattr(rec, "z_score", 0) or 0),
                    float(getattr(rec, "focus_score", 0) or 0),
                    sit or None,
                )
            )
        sec = sections[sections["parent_id"] == city_name].copy() if not sections.empty else sections
        if not sec.empty:
            sec = sec.sort_values("isolated_mt" if "isolated_mt" in sec.columns else "gap_mt")
        for rec in sec.itertuples(index=False):
            iso = float(getattr(rec, "isolated_mt", rec.gap_mt) or 0)
            sit = str(getattr(rec, "situation", "") or "")
            if iso > 0.15:
                continue
            if len([r for r in rows if r["city"] == city_name and r["grain"] == "section"]) >= 4:
                break
            rows.append(
                _target(
                    period,
                    "section",
                    rec.grain_id,
                    rec.grain_id,
                    city_name,
                    zone,
                    None,
                    None,
                    rec.grain_id,
                    rec.volume_mt,
                    rec.ly_mt,
                    rec.gap_mt,
                    rec.diagnosis,
                    f"Beat {rec.grain_id}: {rec.do_this_week}",
                    f"Section {rec.grain_id} is {iso:+.1f} MT vs Expected.",
                    iso,
                    iso,
                    float(getattr(rec, "z_score", 0) or 0),
                    float(getattr(rec, "focus_score", 0) or 0),
                    sit or None,
                )
            )
        pick = _pick_city_shops(city_name, diagnosis, shop_gap, stores, cur)
        for rec in pick.itertuples(index=False):
            vol = float(rec.volume_mt)
            iso = float(getattr(rec, "isolated_mt", rec.gap_mt) or 0)
            action = (
                "Must-visit: this door is behind its own Expected. Confirm stock, credit, competitor fill-in."
                if vol > 0
                else "Must-visit: billed last year, quiet this period — and behind its own Expected."
            )
            if diagnosis == "whitespace" and vol <= 0:
                action = "Universe door, not billed this period. Put it on a coverage beat — do not wait for inbound."
            if mtd.get("open") and vol == 0:
                action = "Not billed yet this MTD — still recoverable before month-end. Call this week."
            rows.append(
                _target(
                    period,
                    "shop",
                    rec.store_id,
                    rec.store_name or rec.store_id,
                    city_name,
                    zone,
                    rec.distributor,
                    rec.dsr_name,
                    rec.section,
                    rec.volume_mt,
                    rec.ly_mt,
                    rec.gap_mt,
                    diagnosis,
                    action,
                    f"{rec.store_name or rec.store_id} {float(rec.volume_mt):.2f} MT vs {float(rec.ly_mt):.2f} last year "
                    f"({iso:+.2f} MT vs Expected).",
                    iso,
                    iso,
                    0.0,
                    -iso,
                    "lagging" if iso < -0.05 else "with_market",
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    sort_t = "isolated_mt" if "isolated_mt" in out.columns else "gap_mt"
    out = out.sort_values(sort_t).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[TARGET_COLUMNS]


def _pick_city_shops(
    city_name: str,
    diagnosis: str,
    shop_gap: pd.DataFrame,
    stores: pd.DataFrame | None,
    cur: pd.DataFrame,
) -> pd.DataFrame:
    city_shops = shop_gap[shop_gap["city"] == city_name].copy() if not shop_gap.empty else pd.DataFrame()
    empty_cols = ["store_id", "store_name", "distributor", "dsr_name", "section", "volume_mt", "ly_mt", "gap_mt"]
    if diagnosis == "whitespace" and stores is not None and not stores.empty and "city" in stores.columns:
        billed = _billed_ids(cur[cur["city"] == city_name] if not cur.empty and "city" in cur.columns else cur)
        universe = stores.copy()
        universe["city"] = universe["city"].fillna("(unmapped)").replace("", "(unmapped)")
        universe["store_id"] = universe["store_id"].astype(str)
        unbilled = universe[(universe["city"] == city_name) & (~universe["store_id"].isin(billed))].copy()
        if not unbilled.empty:
            extra = shop_gap[["store_id", "ly_mt", "volume_mt", "gap_mt"]].drop_duplicates("store_id") if not shop_gap.empty else pd.DataFrame()
            if extra.empty:
                unbilled["ly_mt"] = 0.0
                unbilled["volume_mt"] = 0.0
                unbilled["gap_mt"] = 0.0
            else:
                extra["store_id"] = extra["store_id"].astype(str)
                unbilled = unbilled.merge(extra, on="store_id", how="left")
                unbilled["ly_mt"] = unbilled["ly_mt"].fillna(0)
                unbilled["volume_mt"] = unbilled["volume_mt"].fillna(0)
                unbilled["gap_mt"] = unbilled["gap_mt"].fillna(0)
            for col in ("store_name", "distributor", "dsr_name", "section"):
                if col not in unbilled.columns:
                    unbilled[col] = ""
                unbilled[col] = unbilled[col].fillna("")
            return unbilled.sort_values(["ly_mt", "store_name"], ascending=[False, True]).head(8)
    if city_shops.empty:
        return pd.DataFrame(columns=empty_cols)
    material = city_shops[(city_shops["ly_mt"] >= MIN_MATERIAL_MT) | (city_shops["gap_mt"] <= -0.15)]
    if material.empty:
        material = city_shops
    if diagnosis == "drop_size":
        cont = material[material["volume_mt"] > 0]
        pool = cont if not cont.empty else material
    elif diagnosis == "coverage":
        quiet = material[material["volume_mt"] <= 0]
        pool = quiet if not quiet.empty else material
    else:
        pool = material
    sort_col = "isolated_mt" if "isolated_mt" in pool.columns else "gap_mt"
    return pool.nsmallest(8, sort_col)


def _shop_gaps(
    cur: pd.DataFrame, ly: pd.DataFrame, pace: float, shop_expected: pd.DataFrame | None = None
) -> pd.DataFrame:
    cols = {
        "store_name": "last",
        "city": "last",
        "zone": "last",
        "distributor": "last",
        "dsr_name": "last",
        "section": "last",
    }

    def _side(df: pd.DataFrame, vol_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["store_id", vol_name, *cols.keys()])
        work = df.copy()
        for col in cols:
            if col not in work.columns:
                work[col] = ""
        out = work.groupby("store_id", as_index=False).agg(
            store_name=("store_name", "last"),
            city=("city", "last"),
            zone=("zone", "last"),
            distributor=("distributor", "last"),
            dsr_name=("dsr_name", "last"),
            section=("section", "last"),
            **{vol_name: ("volume_mt", "sum")},
        )
        out["store_id"] = out["store_id"].astype(str)
        return out

    now = _side(cur, "volume_mt")
    ref = _side(ly, "ly_mt")
    # last-year names live under the same columns; prefix before merge
    if not ref.empty:
        ref = ref.rename(
            columns={c: f"{c}_ly" for c in cols if c in ref.columns}
        )
    m = now.merge(ref, on="store_id", how="outer")
    m["volume_mt"] = m["volume_mt"].fillna(0)
    m["ly_mt"] = m["ly_mt"].fillna(0)
    m["expected_mt"] = m["ly_mt"] * pace
    if shop_expected is not None and not shop_expected.empty and "store_id" in shop_expected.columns:
        se = shop_expected[["store_id", "expected_mt"]].copy()
        se["store_id"] = se["store_id"].astype(str)
        m["store_id"] = m["store_id"].astype(str)
        m = m.merge(se, on="store_id", how="left", suffixes=("", "_learned"))
        learned = pd.to_numeric(m.get("expected_mt_learned"), errors="coerce")
        if learned is not None:
            base = pd.to_numeric(m["expected_mt"], errors="coerce")
            m["expected_mt"] = learned.where(learned.notna() & (learned > 1e-9), base)
            m = m.drop(columns=["expected_mt_learned"])
    m["gap_mt"] = m["volume_mt"] - m["expected_mt"]
    for col in cols:
        ly_col = f"{col}_ly"
        if col not in m.columns:
            m[col] = None
        if ly_col in m.columns:
            m[col] = m[col].fillna(m[ly_col])
        m[col] = m[col].fillna("")
    return m


def _target(
    period: str,
    grain: str,
    entity_id: Any,
    entity_name: Any,
    city: Any,
    zone: Any,
    distributor: Any,
    dsr_name: Any,
    section: Any,
    volume_mt: Any,
    ly_mt: Any,
    gap_mt: Any,
    diagnosis: Any,
    action: Any,
    why: Any,
    competitive_mt: Any = 0,
    isolated_mt: Any = 0,
    z_score: Any = 0,
    focus_score: Any = 0,
    situation: Any = None,
) -> dict[str, Any]:
    def _s(v: Any) -> str | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        text = str(v)
        return None if text in {"", "nan", "None"} else text

    return {
        "period": period,
        "rank": 0,
        "grain": grain,
        "entity_id": _s(entity_id) or "",
        "entity_name": _s(entity_name) or _s(entity_id) or "",
        "city": _s(city),
        "zone": _s(zone),
        "distributor": _s(distributor),
        "dsr_name": _s(dsr_name),
        "section": _s(section),
        "volume_mt": float(volume_mt or 0),
        "ly_mt": float(ly_mt or 0),
        "gap_mt": float(gap_mt or 0),
        "diagnosis": _s(diagnosis),
        "action": _s(action) or "",
        "why": _s(why) or "",
        "competitive_mt": float(competitive_mt or 0),
        "isolated_mt": float(isolated_mt or 0),
        "z_score": float(z_score or 0),
        "focus_score": float(focus_score or 0),
        "situation": _s(situation),
    }


_PLAY_COLS = [
    "run_id",
    "slot",
    "theme",
    "title",
    "why",
    "do_this_week",
    "owner",
    "period",
    "metric_value",
    "shops_json",
    "metrics_json",
]


def plays_from_pack(run_id: int, pack: HierarchyPack) -> pd.DataFrame:
    """Turn the hierarchy into a short named briefing — still data-driven, not a template."""
    if pack.units is None or pack.units.empty:
        return pd.DataFrame(columns=_PLAY_COLS)
    cities = pack.units[pack.units["grain"] == "city"].copy()
    if "isolated_mt" in cities.columns:
        cities = cities.sort_values("isolated_mt")
    else:
        cities = cities.sort_values("gap_mt")
    plays: list[dict[str, Any]] = []
    nat = pack.national
    mtd = pack.mtd
    if nat:
        plays.append(
            {
                "theme": "close_month" if mtd.get("open") else "recover",
                "title": nat.get("headline") or (
                    f"National hole {nat['gap_mt']:+.0f} MT vs expected"
                    if nat.get("gap_mt", 0) < -1
                    else f"National {nat['volume_mt']:.0f} MT"
                ),
                "why": nat.get("weather") or "",
                "do_this_week": nat.get("action_summary") or nat.get("problem") or "",
                "owner": "NSM",
                "metric_value": abs(float(nat.get("gap_mt") or 0)),
                "shops": [],
                "metrics": nat,
            }
        )
    targets = pack.targets if pack.targets is not None else pd.DataFrame()
    for _, city in cities.head(6).iterrows():
        city_name = str(city["grain_id"])
        shops = []
        if not targets.empty:
            hit = targets[(targets["city"] == city_name) & (targets["grain"] == "shop")].head(8)
            for rec in hit.itertuples(index=False):
                shops.append(
                    {
                        "store_id": rec.entity_id,
                        "store_name": rec.entity_name,
                        "dsr_name": rec.dsr_name,
                        "section": rec.section,
                        "distributor": rec.distributor,
                        "volume_mt": rec.volume_mt,
                        "ly_mt": rec.ly_mt,
                        "gap_mt": rec.gap_mt,
                        "tier": rec.diagnosis,
                    }
                )
        dist_names = []
        if not targets.empty:
            dist_names = (
                targets[(targets["city"] == city_name) & (targets["grain"] == "distributor")]
                .head(3)["entity_name"]
                .tolist()
            )
        theme = {
            "drop_size": "recover",
            "coverage": "coverage",
            "whitespace": "coverage",
            "mixed": "fix_beat",
            "holding": "protect",
        }.get(str(city["diagnosis"]), "fix_beat")
        plays.append(
            {
                "theme": theme,
                "title": f"{city_name}: {city['verdict']} · {str(city['diagnosis']).replace('_', ' ')} ({city['gap_mt']:+.0f} MT)",
                "why": city["do_this_week"],
                "do_this_week": (
                    ("Focus distributors: " + ", ".join(str(d) for d in dist_names) + ". " if dist_names else "")
                    + ("Must-visit the named shops — not the tail." if shops else "See the city scorecard.")
                ),
                "owner": f"{city_name} / {city.get('zone') or ''}".strip(" /"),
                "metric_value": abs(float(city["gap_mt"] or 0)),
                "shops": shops,
                "metrics": {
                    "city": city_name,
                    "diagnosis": city["diagnosis"],
                    "verdict": city["verdict"],
                    "volume_mt": city["volume_mt"],
                    "expected_mt": city["expected_mt"],
                    "ly_mt": city["ly_mt"],
                    "lfl_gap": city["lfl_gap"],
                    "lost_mt": city["lost_mt"],
                    "lost_n": city["lost_n"],
                },
            }
        )
    rows = []
    for i, p in enumerate(plays[:8], start=1):
        rows.append(
            {
                "run_id": run_id,
                "slot": i,
                "theme": p["theme"],
                "title": p["title"],
                "why": p["why"],
                "do_this_week": p["do_this_week"],
                "owner": p.get("owner") or "NSM",
                "period": pack.period,
                "metric_value": float(p.get("metric_value") or 0),
                "shops_json": dumps(p.get("shops") or []),
                "metrics_json": dumps(p.get("metrics") or {}),
            }
        )
    return pd.DataFrame(rows, columns=_PLAY_COLS)
