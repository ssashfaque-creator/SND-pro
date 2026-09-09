"""Attach sales-team shop targets onto unit scorecards without touching Expected.

Oracle SVP / SAP trade analytics keep Actual, Baseline, and Target as three
key figures. Stretch = max(0, paced Target − Expected) is ambition, not a
coverage miss. Execution hole = max(0, Expected − billed) is the operating Gap.

National Target is the submitted book (every numeric shop row). City /
distributor / DSR Target is that book rolled on live geography: matched shops
use universe identity; unmatched shops still count if Area / distributor / DSR
fold onto a unique live name. Shop-level identity still requires a conservative
name match — we never assign a whale quota to the wrong POP.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sndintel.identity import dsr_display_name
from sndintel.ingest.targets import fold_name, fold_place

PLAN_COLS = [
    "target_mt",
    "target_paced_mt",
    "vs_target_mt",
    "gap_to_target_mt",
    "attain_pct",
    "stretch_mt",
    "plan_quality",
    "plan_status",
    "n_target_shops",
    "target_matched_mt",
    "target_book_mt",
    "n_target_unmatched",
]

AGGRESSIVE_X = 2.0
SOFT_X = 0.5


def attach_plan(
    units: pd.DataFrame,
    shop_targets: pd.DataFrame | None,
    pace: float = 1.0,
) -> pd.DataFrame:
    """Roll shop targets to national / city / distributor / DSR / section grains."""
    if units is None or units.empty:
        return units if units is not None else pd.DataFrame()
    out = units.copy()
    for col in PLAN_COLS:
        if col not in out.columns:
            out[col] = np.nan
    pace = float(pace or 1.0) or 1.0
    if shop_targets is None or shop_targets.empty:
        return out

    book = _live_book(shop_targets, out)
    if book.empty:
        return out
    book_mt = float(book["target_mt"].sum())
    matched_mask = book["_matched"]
    matched_mt = float(book.loc[matched_mask, "target_mt"].sum()) if matched_mask.any() else 0.0
    unmatched_n = int((~matched_mask).sum())

    city_g = _group_sum(book, ["live_city"])
    dist_g = _group_sum(book, ["live_city", "live_dist"])
    dist_name_g = _group_sum(book, ["live_dist"])
    dsr_g = _group_sum(book, ["live_city", "live_dist", "live_dsr"])
    section_g = _group_sum(book, ["live_city", "live_section"])

    pace_s = pd.to_numeric(out.get("intra_month_frac"), errors="coerce").fillna(pace)
    grain = out["grain"].astype(str)
    vol = pd.to_numeric(out.get("volume_mt"), errors="coerce").fillna(0.0)
    exp = pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0.0)

    target = pd.Series(np.nan, index=out.index, dtype=float)
    n_shops = pd.Series(np.nan, index=out.index, dtype=float)

    nat = grain == "national"
    if nat.any():
        target.loc[nat] = book_mt
        n_shops.loc[nat] = float(len(book))

    city = grain == "city"
    if city.any():
        rec = out.loc[city, "grain_id"].astype(str).str.strip().map(lambda c: city_g.get((c,)))
        target.loc[city] = rec.map(_amt)
        n_shops.loc[city] = rec.map(_n)

    dist = grain == "distributor"
    if dist.any():
        city_k = out.loc[dist].apply(
            lambda r: str(r.get("city") or r.get("parent_id") or "").strip(), axis=1
        )
        dist_k = out.loc[dist, "grain_id"].astype(str).str.strip()
        rec = pd.Series(
            [dist_g.get((c, d)) or dist_name_g.get((d,)) for c, d in zip(city_k, dist_k)],
            index=out.index[dist],
        )
        target.loc[dist] = rec.map(_amt)
        n_shops.loc[dist] = rec.map(_n)

    dsr = grain == "dsr"
    if dsr.any():
        rec = out.loc[dsr].apply(lambda r: dsr_g.get(_dsr_key(r)), axis=1)
        target.loc[dsr] = rec.map(_amt)
        n_shops.loc[dsr] = rec.map(_n)

    section = grain == "section"
    if section.any() and section_g:
        rec = out.loc[section].apply(
            lambda r: section_g.get(
                (
                    str(r.get("city") or r.get("parent_id") or "").strip(),
                    str(r.get("grain_id") or r.get("section") or "").strip(),
                )
            ),
            axis=1,
        )
        target.loc[section] = rec.map(_amt)
        n_shops.loc[section] = rec.map(_n)

    paced = target * pace_s
    out["target_mt"] = target
    out["target_paced_mt"] = paced
    out["vs_target_mt"] = vol - paced
    out["gap_to_target_mt"] = (paced - vol).clip(lower=0)
    out["attain_pct"] = np.where(paced > 1e-9, vol / paced, np.nan)
    out["stretch_mt"] = (paced - exp).clip(lower=0)
    out["n_target_shops"] = n_shops
    out["plan_quality"] = [_quality(t, e) for t, e in zip(paced.fillna(0), exp)]
    out["plan_status"] = [_status(v, e, t) for v, e, t in zip(vol, exp, paced.fillna(0))]
    if nat.any():
        out.loc[nat, "target_book_mt"] = book_mt
        out.loc[nat, "target_matched_mt"] = matched_mt
        out.loc[nat, "n_target_unmatched"] = unmatched_n
        # Do not mark national as no_plan when the book is on file.
        out.loc[nat, "plan_status"] = [
            _status(v, e, t) for v, e, t in zip(vol.loc[nat], exp.loc[nat], paced.loc[nat].fillna(0))
        ]
    return out


def _dsr_key(row: pd.Series) -> tuple[str, str, str]:
    city = str(row.get("city") or "").strip()
    dist = str(row.get("distributor") or "").strip()
    dsr = str(row.get("dsr_name") or "").strip()
    if not dsr:
        dsr = dsr_display_name(row.get("grain_id"))
    return (city, dist, dsr)


def _live_book(shop_targets: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    book = shop_targets.copy()
    book["target_mt"] = pd.to_numeric(book.get("target_mt"), errors="coerce")
    book = book.loc[book["target_mt"].notna()].copy()
    if book.empty:
        return book
    for col in ("city", "distributor", "dsr_name", "section", "store_id", "match_method"):
        if col not in book.columns:
            book[col] = ""
        book[col] = book[col].fillna("").astype(str)
    method = book["match_method"].astype(str)
    store_id = book["store_id"].astype(str).replace({"nan": "", "None": ""})
    book["_matched"] = method.ne("unmatched") & store_id.ne("")

    city_map, dist_map, dsr_map, section_map = _unit_geo_maps(units)

    city_k = book["city"].map(fold_place)
    dist_k = book["distributor"].map(fold_name)
    dsr_k = book["dsr_name"].map(fold_name)
    mapped_city = city_k.map(lambda k: city_map.get(k, ""))
    mapped_dist = [dist_map.get((c, d)) for c, d in zip(city_k, dist_k)]
    mapped_dsr = [dsr_map.get((c, d, s)) for c, d, s in zip(city_k, dist_k, dsr_k)]

    live_city, live_dist, live_dsr, live_section = [], [], [], []
    for i, matched in enumerate(book["_matched"].tolist()):
        row = book.iloc[i]
        if matched:
            live_city.append(str(row["city"]).strip())
            live_dist.append(str(row["distributor"]).strip())
            live_dsr.append(str(row["dsr_name"]).strip())
            live_section.append(str(row.get("section") or "").strip())
            continue
        dist_hit = mapped_dist[i]
        dsr_hit = mapped_dsr[i]
        city = mapped_city.iloc[i] if hasattr(mapped_city, "iloc") else mapped_city[i]
        dist = dist_hit[1] if dist_hit else ""
        dsr = ""
        if dsr_hit:
            city = city or dsr_hit[0]
            dist = dist or dsr_hit[1]
            dsr = dsr_hit[2]
        sec_hit = section_map.get((fold_place(city), fold_name(row.get("section") or "")))
        live_city.append(str(city or ""))
        live_dist.append(str(dist or ""))
        live_dsr.append(str(dsr or ""))
        live_section.append(sec_hit[1] if sec_hit else "")
    book["live_city"] = live_city
    book["live_dist"] = live_dist
    book["live_dsr"] = live_dsr
    book["live_section"] = live_section
    return book


def _unit_geo_maps(
    units: pd.DataFrame,
) -> tuple[dict[str, str], dict[tuple[str, str], tuple[str, str]], dict[tuple[str, str, str], tuple[str, str, str]], dict[tuple[str, str], tuple[str, str]]]:
    grain = units["grain"].astype(str) if "grain" in units.columns else pd.Series("", index=units.index)
    city_map: dict[str, str] = {}
    dist_map: dict[tuple[str, str], tuple[str, str]] = {}
    dsr_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    section_map: dict[tuple[str, str], tuple[str, str]] = {}

    cities = units.loc[grain == "city"]
    if not cities.empty:
        city_map = _unique_fold_map(cities["grain_id"], fold_place)

    dists = units.loc[grain == "distributor"]
    if not dists.empty:
        for _, r in dists.iterrows():
            city = str(r.get("city") or r.get("parent_id") or "").strip()
            dist = str(r.get("grain_id") or r.get("distributor") or "").strip()
            if not city or not dist:
                continue
            key = (fold_place(city), fold_name(dist))
            if key in dist_map and dist_map[key] != (city, dist):
                dist_map[key] = ("", "")  # ambiguous
            else:
                dist_map[key] = (city, dist)
        dist_map = {k: v for k, v in dist_map.items() if v[0]}

    dsrs = units.loc[grain == "dsr"]
    if not dsrs.empty:
        for _, r in dsrs.iterrows():
            city = str(r.get("city") or "").strip()
            dist = str(r.get("distributor") or "").strip()
            dsr = str(r.get("dsr_name") or "").strip() or dsr_display_name(r.get("grain_id"))
            if not city or not dsr:
                continue
            key = (fold_place(city), fold_name(dist), fold_name(dsr))
            val = (city, dist, dsr)
            if key in dsr_map and dsr_map[key] != val:
                dsr_map[key] = ("", "", "")
            else:
                dsr_map[key] = val
        dsr_map = {k: v for k, v in dsr_map.items() if v[0]}

    sections = units.loc[grain == "section"]
    if not sections.empty:
        for _, r in sections.iterrows():
            city = str(r.get("city") or r.get("parent_id") or "").strip()
            section = str(r.get("grain_id") or r.get("section") or "").strip()
            if not city or not section:
                continue
            key = (fold_place(city), fold_name(section))
            val = (city, section)
            if key in section_map and section_map[key] != val:
                section_map[key] = ("", "")
            else:
                section_map[key] = val
        section_map = {k: v for k, v in section_map.items() if v[0]}
    return city_map, dist_map, dsr_map, section_map


def _unique_fold_map(values: pd.Series, folder) -> dict[str, str]:
    work = pd.DataFrame({"raw": values.fillna("").astype(str).str.strip()})
    work = work.loc[work["raw"] != ""]
    if work.empty:
        return {}
    work["k"] = work["raw"].map(folder)
    grouped = work.groupby("k")["raw"].agg(["nunique", "first"])
    return grouped.loc[grouped["nunique"] == 1, "first"].to_dict()


def _group_sum(book: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    work = book.copy()
    for k in keys:
        work[k] = work[k].fillna("").astype(str).str.strip()
    work = work.loc[work[keys].replace("", np.nan).notna().all(axis=1)]
    if work.empty:
        return {}
    g = work.groupby(keys, dropna=False)["target_mt"].agg(["sum", "count"])
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, rec in g.iterrows():
        if not isinstance(key, tuple):
            key = (key,)
        if any(not str(x).strip() for x in key):
            continue
        out[tuple(str(x) for x in key)] = {"target_mt": float(rec["sum"]), "n": int(rec["count"])}
    return out


def _amt(rec: Any) -> float:
    return float(rec["target_mt"]) if isinstance(rec, dict) else np.nan


def _n(rec: Any) -> float:
    return float(rec["n"]) if isinstance(rec, dict) else np.nan


def _quality(target: float, expected: float) -> str:
    if expected < 0.1 or target <= 0:
        return "aligned"
    ratio = target / expected
    if ratio >= AGGRESSIVE_X:
        return "aggressive"
    if ratio <= SOFT_X:
        return "soft"
    return "aligned"


def _status(volume: float, expected: float, target: float) -> str:
    if target <= 1e-9:
        return "no_plan"
    if volume + 0.05 >= target:
        return "beating_plan"
    if volume + 0.05 < expected:
        return "missing_run_rate"
    if volume + 0.05 < target:
        return "missing_stretch"
    return "on_plan"
