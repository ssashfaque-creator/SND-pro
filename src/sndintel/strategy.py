"""Turn warehouse facts into a short strategy pack — not a dump of every flag.

IRI / Nielsen / FireAI-style S&D tools rank *plays* (protect, recover, cover, mix,
people) and attach a must-visit list. Shop-level noise stays in Evidence.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from sndintel.features import latest_period
from sndintel.materiality import classify_gap_shops, must_visit_recoveries
from sndintel.mtd import period_state
from sndintel.storage import dumps


PLAY_COLUMNS = [
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


def compile_plays(
    run_id: int,
    shop_month: pd.DataFrame,
    stores: pd.DataFrame,
    features: pd.DataFrame,
    insights: pd.DataFrame,
    kpis: pd.DataFrame,
    anomalies: pd.DataFrame,
    ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    period = latest_period(shop_month)
    if not period:
        return pd.DataFrame(columns=PLAY_COLUMNS)
    mtd = period_state(ledger, period)
    gap = classify_gap_shops(shop_month, period, features)
    plays: list[dict[str, Any]] = []

    plays += _play_close_month(kpis, period, mtd)
    plays += _play_protect_core(anomalies, shop_month, period, gap)
    plays += _play_recover_material(gap, mtd, period)
    plays += _play_fix_beat(insights, period)
    plays += _play_long_tail(gap, mtd, period)
    plays += _play_mix(insights, period)
    plays += _play_people(insights, kpis, period)

    # Hard cap: a sales head can brief five plays, not forty flags.
    ranked = _rank_plays(plays)[:5]
    rows = []
    for i, p in enumerate(ranked, start=1):
        shops = p.get("shops") or []
        rows.append(
            {
                "run_id": run_id,
                "slot": i,
                "theme": p["theme"],
                "title": p["title"],
                "why": p["why"],
                "do_this_week": p["do_this_week"],
                "owner": p.get("owner") or "NSM",
                "period": period,
                "metric_value": float(p.get("metric_value") or 0),
                "shops_json": dumps(shops) if not isinstance(shops, str) else shops,
                "metrics_json": dumps(p.get("metrics") or {}),
            }
        )
    return pd.DataFrame(rows, columns=PLAY_COLUMNS)


def _rank_plays(plays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    weight = {
        "close_month": 100,
        "protect": 90,
        "recover": 85,
        "fix_beat": 80,
        "mix": 55,
        "people": 50,
        "coverage": 45,
    }
    return sorted(plays, key=lambda p: weight.get(p["theme"], 10) + min(abs(float(p.get("metric_value") or 0)), 40), reverse=True)


def _play_close_month(kpis: pd.DataFrame, period: str, mtd: dict[str, Any]) -> list[dict]:
    if not mtd.get("open") or kpis is None or kpis.empty:
        return []
    nat = kpis[(kpis["grain"] == "national") & (kpis["period"] == period)]
    if nat.empty:
        return []
    row = nat.iloc[0]
    run_yoy = row.get("run_rate_yoy_pct")
    if pd.isna(run_yoy):
        return []
    pace = f"day {mtd.get('as_of_day')}/{mtd.get('days_in_month')}"
    if float(run_yoy) >= -3:
        return [
            {
                "theme": "close_month",
                "title": f"Hold the month — {period} MTD is on pace",
                "why": (
                    f"Through {pace}, billed {float(row['volume_mt']):.1f} MT. "
                    f"Run-rate vs last year's closed month is {float(run_yoy):+.1f}%."
                ),
                "do_this_week": "Do not load extra stock to 'make the month'. Protect drop size on core shops and keep the call cadence.",
                "owner": "NSM",
                "metric_value": abs(float(run_yoy)),
                "metrics": {"run_rate_yoy_pct": float(run_yoy), "volume_mt": float(row["volume_mt"])},
            }
        ]
    return [
        {
            "theme": "close_month",
            "title": f"Close {period} — run-rate is {float(run_yoy):+.1f}% vs last year",
            "why": (
                f"This is still open MTD ({pace}), not a closed miss. "
                f"Billed {float(row['volume_mt']):.1f} MT; full-month pace {float(row.get('run_rate_mt') or 0):.1f} MT."
            ),
            "do_this_week": (
                "Prioritise unbilled *material* shops (Recover play), not the long tail. "
                "A later extract this month replaces MTD in full."
            ),
            "owner": "NSM + DSRs",
            "metric_value": abs(float(run_yoy)),
            "metrics": {"run_rate_yoy_pct": float(run_yoy), "volume_mt": float(row["volume_mt"])},
        }
    ]


def _play_protect_core(anomalies: pd.DataFrame, shop_month: pd.DataFrame, period: str, gap: dict[str, Any]) -> list[dict]:
    if anomalies is None or anomalies.empty:
        return []
    tiers = gap.get("tiers")
    core_ids = set(tiers.loc[tiers["tier"] == "core", "store_id"]) if tiers is not None and not tiers.empty else set()
    dumps = anomalies[anomalies["kind"] == "trade_loading"].copy()
    if core_ids:
        dumps = dumps[dumps["store_id"].astype(str).isin(core_ids) | (dumps["volume_mt"] >= 0.2)]
    dumps = dumps.sort_values("volume_mt", ascending=False).head(8)
    if dumps.empty:
        return []
    names = shop_month.drop_duplicates("store_id").set_index("store_id")
    shops = []
    for _, r in dumps.iterrows():
        sid = r["store_id"]
        name = names.loc[sid, "store_name"] if sid in names.index else sid
        shops.append({"store_id": sid, "store_name": str(name), "volume_mt": float(r["volume_mt"])})
    n = len(dumps)
    mt = float(dumps["volume_mt"].sum())
    return [
        {
            "theme": "protect",
            "title": f"Protect the base — {n} core shops look loaded ({mt:.1f} MT)",
            "why": (
                "Spikes vs each shop's own history, not vs a national average. "
                "If this is forward-buy it will reverse next month and fake this month's win."
            ),
            "do_this_week": "Check invoice vs storage and scheme on the must-visit list. Do not add more stock to these doors.",
            "owner": "Distributor + DSR",
            "metric_value": mt,
            "shops": shops,
            "metrics": {"shops": n, "volume_mt": mt},
        }
    ]


def _play_recover_material(gap: dict[str, Any], mtd: dict[str, Any], period: str) -> list[dict]:
    visit = must_visit_recoveries(gap, limit=25)
    core_n = gap.get("counts", {}).get("lost_core", 0)
    mid_n = gap.get("counts", {}).get("lost_middle", 0)
    core_mt = gap.get("volumes", {}).get("lost_core", 0.0)
    mid_mt = gap.get("volumes", {}).get("lost_middle", 0.0)
    material_n = core_n + mid_n
    material_mt = core_mt + mid_mt
    if material_n == 0 or material_mt < 0.05:
        return []
    still = "not yet billed this MTD" if mtd.get("open") else "unbilled vs last year"
    shops = []
    if not visit.empty:
        for _, r in visit.iterrows():
            shops.append(
                {
                    "store_id": r["store_id"],
                    "store_name": str(r.get("store_name") or r["store_id"]),
                    "dsr_name": str(r.get("dsr_name") or ""),
                    "section": str(r.get("section") or ""),
                    "volume_mt": float(r["volume_mt"]),
                    "tier": r.get("tier"),
                }
            )
    by_dsr = ""
    if shops:
        dsr = pd.DataFrame(shops).groupby("dsr_name")["volume_mt"].sum().sort_values(ascending=False).head(3)
        by_dsr = " Biggest holes: " + ", ".join(f"{k or 'unassigned'} {v:.1f} MT" for k, v in dsr.items()) + "."
    return [
        {
            "theme": "recover",
            "title": f"Recover {material_n} material shops ({material_mt:.1f} MT last year)",
            "why": (
                f"{material_n} shops in the core/middle of last year's volume are {still}. "
                f"This is weighted distribution, not a headcount of every quiet kiryana."
                f"{by_dsr}"
            ),
            "do_this_week": (
                "Print the must-visit list (top 25 by last-year drop). Ride-with those DSRs this week. "
                "Do not put the long-tail shops on this list — they have their own coverage play."
            ),
            "owner": "DSRs on the list",
            "metric_value": material_mt,
            "shops": shops,
            "metrics": {"shops": material_n, "volume_mt": material_mt, "core_n": core_n, "middle_n": mid_n},
        }
    ]


def _play_fix_beat(insights: pd.DataFrame, period: str) -> list[dict]:
    if insights is None or insights.empty:
        return []
    div = insights[insights["type"] == "divergence"].sort_values("rank_score", ascending=False)
    if div.empty:
        return []
    rec = div.iloc[0]
    return [
        {
            "theme": "fix_beat",
            "title": rec["title"],
            "why": rec["narrative"],
            "do_this_week": rec["action"],
            "owner": f"DSR / section {rec.get('entity_name') or rec.get('entity_id')}",
            "metric_value": float(rec.get("metric_value") or 0),
            "metrics": {"entity_id": rec.get("entity_id")},
        }
    ]


def _play_long_tail(gap: dict[str, Any], mtd: dict[str, Any], period: str) -> list[dict]:
    n = gap.get("counts", {}).get("lost_tail", 0)
    mt = gap.get("volumes", {}).get("lost_tail", 0.0)
    occ_n = gap.get("counts", {}).get("lost_occasional", 0)
    occ_mt = gap.get("volumes", {}).get("lost_occasional", 0.0)
    if n == 0 and occ_n == 0:
        return []
    label = "not yet billed this MTD" if mtd.get("open") else "quiet vs last year"
    tail = gap.get("lost_tail")
    pocket = ""
    if tail is not None and not tail.empty and "section" in tail.columns:
        top = tail.groupby("section", as_index=False)["volume_mt"].sum().sort_values("volume_mt", ascending=False).head(3)
        pocket = " Pockets: " + ", ".join(f"{r.section} {r.volume_mt:.2f} MT" for r in top.itertuples()) + "."
    return [
        {
            "theme": "coverage",
            "title": f"Long-tail coverage — {n} micro shops ({mt:.1f} MT last year)",
            "why": (
                f"{n} shops that were tiny last year are {label}. Together they are {mt:.1f} MT — "
                f"worth a coverage KPI, not {n} recovery calls. "
                f"{occ_n} more shops ({occ_mt:.1f} MT) were already irregular billers; missing a month is their cadence."
                f"{pocket}"
            ),
            "do_this_week": (
                "Give each DSR a monthly coverage target on micro shops (billed / universe in the tail), "
                "not a must-visit list of hundreds of doors. Do not load them."
            ),
            "owner": "Area sales manager",
            "metric_value": mt,
            "metrics": {"shops": n, "volume_mt": mt, "occasional_shops": occ_n, "occasional_mt": occ_mt},
        }
    ]


def _play_mix(insights: pd.DataFrame, period: str) -> list[dict]:
    if insights is None or insights.empty:
        return []
    mix = insights[insights["type"].isin(["cannibalization", "sku_substitution"])].sort_values("rank_score", ascending=False)
    if mix.empty:
        return []
    rec = mix.iloc[0]
    return [
        {
            "theme": "mix",
            "title": rec["title"],
            "why": rec["narrative"],
            "do_this_week": rec["action"],
            "owner": "Category / sales ops",
            "metric_value": float(rec.get("metric_value") or 0),
        }
    ]


def _play_people(insights: pd.DataFrame, kpis: pd.DataFrame, period: str) -> list[dict]:
    if insights is None or insights.empty:
        return []
    weak = insights[insights["type"].isin(["dsr_underperformance", "coverage_gap"])].sort_values("rank_score", ascending=False)
    win = insights[insights["type"].isin(["dsr_win", "local_outperformance"])].sort_values("rank_score", ascending=False)
    if weak.empty and win.empty:
        return []
    if not weak.empty:
        rec = weak.iloc[0]
        copy = f" Copy {win.iloc[0]['title']}." if not win.empty else ""
        return [
            {
                "theme": "people",
                "title": rec["title"],
                "why": rec["narrative"] + copy,
                "do_this_week": rec["action"],
                "owner": str(rec.get("entity_name") or rec.get("entity_id")),
                "metric_value": float(rec.get("metric_value") or 0),
            }
        ]
    rec = win.iloc[0]
    return [
        {
            "theme": "people",
            "title": rec["title"],
            "why": rec["narrative"],
            "do_this_week": rec["action"],
            "owner": str(rec.get("entity_name") or rec.get("entity_id")),
            "metric_value": float(rec.get("metric_value") or 0),
        }
    ]


def plays_to_visit_frame(plays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if plays is None or plays.empty:
        return pd.DataFrame()
    for rec in plays.itertuples(index=False):
        raw = rec.shops_json
        try:
            shops = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            shops = []
        if not shops:
            continue
        for s in shops:
            rows.append(
                {
                    "play": rec.title,
                    "theme": rec.theme,
                    "store_id": s.get("store_id"),
                    "store_name": s.get("store_name"),
                    "dsr_name": s.get("dsr_name"),
                    "section": s.get("section"),
                    "volume_mt": s.get("volume_mt"),
                    "tier": s.get("tier"),
                }
            )
    return pd.DataFrame(rows)
