"""Diagnostic modules that turn facts + scores into ranked strategic insights."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from sndintel.config import DIVERGENCE_GAP_PP, MIN_VOLUME_FLAG_MT
from sndintel.features import latest_period, previous_period
from sndintel.io_utils import shift_period
from sndintel.storage import dumps


SEVERITY_WEIGHT = {"critical": 100, "high": 70, "medium": 40, "low": 15, "positive": 25}


def compile_insights(
    run_id: int,
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    shop_month: pd.DataFrame,
    features: pd.DataFrame,
    forecasts: pd.DataFrame,
    anomalies: pd.DataFrame,
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (insights, kpi_snapshots)."""
    period = latest_period(shop_month)
    if not period:
        return pd.DataFrame(), pd.DataFrame()
    prev = previous_period(period)
    kpis = build_kpi_snapshots(shop_month, stores, sales, period, prev)
    rows: list[dict[str, Any]] = []
    rows += _coverage_insights(kpis, stores, shop_month, period)
    rows += _divergence_insights(kpis, shop_month, period)
    rows += _anomaly_insights(anomalies, shop_month, period)
    rows += _segment_insights(segments, shop_month, period)
    rows += _cannibalization_insights(sales, period, prev)
    rows += _pareto_insights(shop_month, period)
    rows += _forecast_gap_insights(forecasts, shop_month, period)
    rows += _positive_insights(kpis, segments, shop_month, period)
    rows += _volume_bridge_insights(shop_month, period)
    rows += _whitespace_insights(stores, shop_month, period)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, kpis
    frame["run_id"] = run_id
    frame["period"] = frame["period"].fillna(period)
    frame["metrics_json"] = frame["metrics"].apply(lambda m: dumps(m) if not isinstance(m, str) else m)
    frame["rank_score"] = [
        SEVERITY_WEIGHT.get(s, 10) + min(abs(float(v or 0)) * 5, 40)
        for s, v in zip(frame["severity"], frame["metric_value"])
    ]
    # Boost volume-backed operational issues.
    frame.loc[frame["type"].isin(["trade_loading", "drop_off", "divergence", "coverage_gap"]), "rank_score"] += 10
    frame.loc[frame["severity"] == "positive", "rank_score"] = frame.loc[frame["severity"] == "positive", "rank_score"].clip(upper=45)
    frame = frame.sort_values("rank_score", ascending=False).reset_index(drop=True)
    keep = [
        "run_id",
        "type",
        "severity",
        "entity_type",
        "entity_id",
        "entity_name",
        "period",
        "title",
        "narrative",
        "action",
        "metric_value",
        "metrics_json",
        "rank_score",
    ]
    return frame[keep], kpis


def build_kpi_snapshots(
    shop_month: pd.DataFrame,
    stores: pd.DataFrame,
    sales: pd.DataFrame,
    period: str,
    prev: str,
) -> pd.DataFrame:
    frames = []
    current = shop_month[shop_month["period"] == period]
    previous = shop_month[shop_month["period"] == prev]
    yoy_period = f"{int(period[:4]) - 1}-{period[5:]}"
    yoy = shop_month[shop_month["period"] == yoy_period]

    universe = stores.copy() if stores is not None else pd.DataFrame(columns=["store_id", "zone", "city", "section", "dsr_name", "distributor"])
    if universe.empty:
        universe = current[["store_id", "zone", "city", "section", "dsr_name", "distributor"]].drop_duplicates("store_id")

    sku_depth = (
        sales[sales["period"] == period].groupby("store_id")["sku"].nunique().rename("sku_n")
        if not sales.empty
        else pd.Series(dtype=float)
    )

    grains = {
        "national": None,
        "zone": "zone",
        "city": "city",
        "section": "section",
        "dsr": "dsr_name",
        "distributor": "distributor",
    }
    for grain, col in grains.items():
        if col is None:
            groups = [("ALL", current, previous, yoy, universe)]
        else:
            if col not in universe.columns and col not in current.columns:
                continue
            uni_keys = set(universe[col].dropna().unique()) if col in universe.columns else set()
            cur_keys = set(current[col].dropna().unique()) if col in current.columns else set()
            keys = sorted({str(k) for k in (uni_keys | cur_keys)})
            groups = []
            for key in keys:
                groups.append(
                    (
                        str(key),
                        current[current[col] == key],
                        previous[previous[col] == key] if not previous.empty else previous,
                        yoy[yoy[col] == key] if not yoy.empty else yoy,
                        universe[universe[col] == key] if col in universe.columns else universe,
                    )
                )
        for grain_id, cur, prv, yy, uni in groups:
            billed = int((cur["billed"] == 1).sum()) if not cur.empty else 0
            vol = float(cur["volume_mt"].sum()) if not cur.empty else 0.0
            uni_n = int(uni["store_id"].nunique()) if not uni.empty else int(cur["store_id"].nunique() if not cur.empty else 0)
            strike = billed / uni_n if uni_n else 0.0
            drop = vol / billed if billed else 0.0
            depth = float(cur.merge(sku_depth, left_on="store_id", right_index=True, how="left")["sku_n"].mean()) if billed and len(sku_depth) else 0.0
            prev_vol = float(prv["volume_mt"].sum()) if prv is not None and not prv.empty else 0.0
            yoy_vol = float(yy["volume_mt"].sum()) if yy is not None and not yy.empty else 0.0
            mom = (vol - prev_vol) / prev_vol * 100 if prev_vol else None
            yoy_pct = (vol - yoy_vol) / yoy_vol * 100 if yoy_vol else None
            frames.append(
                {
                    "period": period,
                    "grain": grain,
                    "grain_id": grain_id,
                    "volume_mt": vol,
                    "billed_outlets": billed,
                    "universe_outlets": uni_n,
                    "strike_rate": strike,
                    "drop_size": drop,
                    "sku_depth": depth if depth == depth else 0.0,
                    "mom_pct": mom,
                    "yoy_pct": yoy_pct,
                }
            )

    # SKU grain
    if not sales.empty:
        cur_s = sales[sales["period"] == period]
        prv_s = sales[sales["period"] == prev]
        yy_s = sales[sales["period"] == yoy_period]
        for sku, part in cur_s.groupby("sku"):
            vol = float(part["volume_mt"].sum())
            prev_vol = float(prv_s.loc[prv_s["sku"] == sku, "volume_mt"].sum()) if not prv_s.empty else 0.0
            yoy_vol = float(yy_s.loc[yy_s["sku"] == sku, "volume_mt"].sum()) if not yy_s.empty else 0.0
            frames.append(
                {
                    "period": period,
                    "grain": "sku",
                    "grain_id": str(sku),
                    "volume_mt": vol,
                    "billed_outlets": int(part["store_id"].nunique()),
                    "universe_outlets": int(universe["store_id"].nunique()) if not universe.empty else 0,
                    "strike_rate": part["store_id"].nunique() / max(universe["store_id"].nunique(), 1),
                    "drop_size": vol / max(part["store_id"].nunique(), 1),
                    "sku_depth": 1.0,
                    "mom_pct": (vol - prev_vol) / prev_vol * 100 if prev_vol else None,
                    "yoy_pct": (vol - yoy_vol) / yoy_vol * 100 if yoy_vol else None,
                }
            )
    return pd.DataFrame(frames)


def _insight(**kwargs) -> dict[str, Any]:
    kwargs.setdefault("metrics", {})
    kwargs.setdefault("metric_value", 0.0)
    kwargs.setdefault("entity_name", kwargs.get("entity_id"))
    kwargs.setdefault("action", "")
    kwargs.setdefault("period", None)
    return kwargs


def _coverage_insights(kpis: pd.DataFrame, stores: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    dsr = kpis[(kpis["grain"] == "dsr") & (kpis["period"] == period)].copy()
    if dsr.empty:
        return rows
    dsr = dsr.sort_values("strike_rate")
    weak = dsr[(dsr["universe_outlets"] >= 8) & (dsr["strike_rate"] <= 0.45)]
    for _, r in weak.head(8).iterrows():
        rows.append(
            _insight(
                type="coverage_gap",
                severity="high" if r["strike_rate"] < 0.3 else "medium",
                entity_type="dsr",
                entity_id=r["grain_id"],
                entity_name=r["grain_id"],
                title=f"Low strike rate on {r['grain_id']}'s beat",
                narrative=(
                    f"{r['grain_id']} billed {int(r['billed_outlets'])} of {int(r['universe_outlets'])} "
                    f"universe outlets ({r['strike_rate']*100:.0f}% strike). Volume {r['volume_mt']:.2f} MT. "
                    "This beat is underdeveloped versus a healthy 60%+ productive-outlet rate."
                ),
                action="Audit beat plan, unbilled high-potential shops, and callage vs. order conversion.",
                metric_value=float(1 - r["strike_rate"]),
                metrics={"strike_rate": r["strike_rate"], "universe": r["universe_outlets"], "volume_mt": r["volume_mt"]},
            )
        )
    saturated = dsr[(dsr["strike_rate"] >= 0.8) & (dsr["mom_pct"].fillna(0) < -8) & (dsr["volume_mt"] >= 0.5)]
    for _, r in saturated.head(5).iterrows():
        rows.append(
            _insight(
                type="saturated_beat",
                severity="medium",
                entity_type="dsr",
                entity_id=r["grain_id"],
                title=f"{r['grain_id']} is covering the universe but losing volume",
                narrative=(
                    f"Strike rate is {r['strike_rate']*100:.0f}% yet MoM volume is {r['mom_pct']:.1f}%. "
                    "Coverage is not the constraint — drop size or mix is slipping."
                ),
                action="Protect drop size: check stock-outs, competitor schemes, and over-frequent small drops.",
                metric_value=abs(float(r["mom_pct"] or 0)),
                metrics={"strike_rate": r["strike_rate"], "mom_pct": r["mom_pct"]},
            )
        )
    return rows


def _divergence_insights(kpis: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    """Flag local execution misses: a section/DSR down while its city/distributor is not.

    Parent rates exclude shops whose latest drop is > 4x their own prior month so a
    single stock dump cannot make every other beat look like a failure.
    """
    rows = []
    prev = previous_period(period)
    cur = shop_month[shop_month["period"] == period].copy()
    prv = shop_month[shop_month["period"] == prev].copy()
    if cur.empty or prv.empty:
        return rows

    merged = cur.merge(prv[["store_id", "volume_mt"]], on="store_id", how="left", suffixes=("", "_prev"))
    merged["volume_mt_prev"] = merged["volume_mt_prev"].fillna(0)
    dump_ids = set(
        merged.loc[
            (merged["volume_mt_prev"] > 0.02) & (merged["volume_mt"] >= merged["volume_mt_prev"] * 4),
            "store_id",
        ]
    )
    parent = merged[~merged["store_id"].isin(dump_ids)] if dump_ids else merged

    def _mom(frame: pd.DataFrame) -> float | None:
        a = float(frame["volume_mt"].sum())
        b = float(frame["volume_mt_prev"].sum()) if "volume_mt_prev" in frame.columns else 0.0
        if b <= 0:
            return None
        return (a - b) / b * 100

    nat_mom = _mom(parent)
    for section, part in parent.groupby("section"):
        if part["volume_mt"].sum() < 0.25:
            continue
        mom = _mom(part)
        if mom is None:
            continue
        city = part["city"].dropna().mode()
        city_name = str(city.iloc[0]) if len(city) else None
        city_part = parent[parent["city"] == city_name] if city_name else parent
        parent_mom = _mom(city_part)
        parent_label = city_name or "national"
        if parent_mom is None:
            parent_mom = nat_mom or 0.0
            parent_label = "national"
        gap = float(mom) - float(parent_mom)
        if gap <= -DIVERGENCE_GAP_PP and mom < 0:
            rows.append(
                _insight(
                    type="divergence",
                    severity="critical" if gap <= -25 else "high",
                    entity_type="section",
                    entity_id=str(section),
                    title=f"{section} is diverging vs {parent_label}",
                    narrative=(
                        f"Section {section} is {mom:.1f}% MoM while {parent_label} is {parent_mom:.1f}% "
                        f"(gap {gap:.1f} pp) on {part['volume_mt'].sum():.2f} MT. This looks like a local "
                        "execution miss, not a category headwind."
                    ),
                    action="Ride-with the DSR, check competitor activity, and compare billed vs universe shops in this section.",
                    metric_value=abs(gap),
                    metrics={
                        "section_mom": mom,
                        "parent_mom": parent_mom,
                        "parent": parent_label,
                        "gap_pp": gap,
                        "volume_mt": float(part["volume_mt"].sum()),
                    },
                )
            )
        elif gap >= DIVERGENCE_GAP_PP and mom > 5:
            rows.append(
                _insight(
                    type="local_outperformance",
                    severity="positive",
                    entity_type="section",
                    entity_id=str(section),
                    title=f"{section} is outrunning {parent_label}",
                    narrative=(
                        f"Section {section} grew {mom:.1f}% MoM vs {parent_mom:.1f}% in {parent_label}. "
                        "Replicate callage, assortment, and scheme execution from this pocket."
                    ),
                    action="Document what the DSR changed this month and copy it to peer sections.",
                    metric_value=gap,
                    metrics={"section_mom": mom, "parent_mom": parent_mom, "gap_pp": gap},
                )
            )

    dsrs = kpis[(kpis["grain"] == "dsr") & (kpis["period"] == period)]
    for _, r in dsrs.iterrows():
        if pd.isna(r["mom_pct"]) or r["volume_mt"] < 0.4:
            continue
        part = parent[parent["dsr_name"] == r["grain_id"]]
        dist_name = part["distributor"].dropna().mode()
        dist_label = str(dist_name.iloc[0]) if len(dist_name) else "peers"
        dist_part = parent[parent["distributor"] == dist_label] if dist_label != "peers" else parent
        dist_mom = _mom(dist_part)
        if dist_mom is None:
            continue
        gap = float(r["mom_pct"]) - float(dist_mom)
        # DSR MoM still includes dumps; recompute from stripped parent.
        dsr_mom = _mom(part)
        if dsr_mom is None:
            continue
        gap = dsr_mom - dist_mom
        if gap <= -DIVERGENCE_GAP_PP and dsr_mom < 0:
            rows.append(
                _insight(
                    type="dsr_underperformance",
                    severity="high",
                    entity_type="dsr",
                    entity_id=r["grain_id"],
                    title=f"{r['grain_id']} is lagging {dist_label}",
                    narrative=(
                        f"{r['grain_id']} is {dsr_mom:.1f}% MoM versus {dist_label} {dist_mom:.1f}%. "
                        f"Strike {r['strike_rate']*100:.0f}%, drop size {r['drop_size']:.2f} MT."
                    ),
                    action="Coaching focus: unbilled regulars first, then drop-size on billed shops.",
                    metric_value=abs(gap),
                    metrics={"mom_pct": dsr_mom, "parent_mom": dist_mom, "strike_rate": r["strike_rate"], "drop_size": r["drop_size"]},
                )
            )
    return rows


def _anomaly_insights(anomalies: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    if anomalies is None or anomalies.empty:
        return rows
    names = shop_month.drop_duplicates("store_id").set_index("store_id")
    for _, r in anomalies.iterrows():
        sid = r["store_id"]
        meta = names.loc[sid] if sid in names.index else None
        store_name = meta["store_name"] if meta is not None else sid
        dsr = meta["dsr_name"] if meta is not None else ""
        section = meta["section"] if meta is not None else ""
        details = {}
        if isinstance(r.get("details_json"), str):
            try:
                details = json.loads(r["details_json"])
            except json.JSONDecodeError:
                details = {}
        kind = r["kind"]
        vol, exp = float(r["volume_mt"] or 0), float(r["expected_mt"] or 0)
        if kind == "trade_loading":
            multiple = vol / exp if exp else 99
            rows.append(
                _insight(
                    type="trade_loading",
                    severity=r["severity"],
                    entity_type="shop",
                    entity_id=sid,
                    entity_name=store_name,
                    title=f"Possible stock dump at {store_name}",
                    narrative=(
                        f"{store_name} ({sid}) took {vol:.2f} MT in {period} versus a typical drop of {exp:.2f} MT "
                        f"({multiple:.1f}x). DSR {dsr}, section {section}. This pattern is classic trade-loading — "
                        "volume that will likely reverse next month if it is not genuine offtake."
                    ),
                    action="Check invoice vs. shop storage, scheme-driven forward buy, and next-month returns/zero bill.",
                    metric_value=vol - exp,
                    metrics={"volume_mt": vol, "expected_mt": exp, "multiple": multiple, "dsr": dsr, "section": section, **details},
                )
            )
        elif kind in {"drop_off", "lapse"}:
            rows.append(
                _insight(
                    type="drop_off",
                    severity=r["severity"],
                    entity_type="shop",
                    entity_id=sid,
                    entity_name=store_name,
                    title=f"{store_name} has fallen off its baseline",
                    narrative=(
                        f"{store_name} billed {vol:.2f} MT vs expected {exp:.2f} MT in {period}. "
                        f"DSR {dsr} / {section}. Treat as a recovery call, not a lost account, until proven otherwise."
                    ),
                    action="Must-visit this week. Confirm stock, credit, and competitor fill-in.",
                    metric_value=exp - vol,
                    metrics={"volume_mt": vol, "expected_mt": exp, "dsr": dsr, "section": section},
                )
            )
        elif kind == "lumpy":
            rows.append(
                _insight(
                    type="lumpy",
                    severity="medium",
                    entity_type="shop",
                    entity_id=sid,
                    entity_name=store_name,
                    title=f"{store_name} buys in irregular spikes",
                    narrative=(
                        f"Coefficient of variation is high and latest drop is {vol:.2f} MT (baseline {exp:.2f}). "
                        "Irregular spikes often hide loading or skipped calls."
                    ),
                    action="Move the account onto a fixed replenishment cadence.",
                    metric_value=vol,
                    metrics={"volume_mt": vol, "expected_mt": exp},
                )
            )
        else:
            rows.append(
                _insight(
                    type="anomaly",
                    severity=r.get("severity", "medium"),
                    entity_type="shop",
                    entity_id=sid,
                    entity_name=store_name,
                    title=f"Unusual pattern at {store_name}",
                    narrative=f"Isolation Forest flagged {store_name} ({sid}) in {period}: {vol:.2f} MT vs {exp:.2f} expected.",
                    action="Review the shop scorecard before month-end cut-off.",
                    metric_value=abs(vol - exp),
                    metrics={"volume_mt": vol, "expected_mt": exp, "kind": kind},
                )
            )
    return rows


def _segment_insights(segments: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    if segments is None or segments.empty:
        return rows
    names = shop_month.drop_duplicates("store_id").set_index("store_id")
    counts = segments["segment"].value_counts()
    rows.append(
        _insight(
            type="portfolio_mix",
            severity="low",
            entity_type="national",
            entity_id="ALL",
            title="Outlet portfolio mix this month",
            narrative=" | ".join(f"{k}: {int(v)}" for k, v in counts.items()),
            action="Put Growth Target and Churn Risk shops on the DSR must-visit list.",
            metric_value=float(counts.get("Churn Risk", 0) + counts.get("Dormant", 0)),
            metrics=counts.to_dict(),
        )
    )
    focus = segments[segments["segment"].isin(["Churn Risk", "Growth Target", "Lumpy / Loaded"])]
    focus = focus.sort_values("monetary", ascending=False).head(12)
    for _, r in focus.iterrows():
        sid = r["store_id"]
        name = names.loc[sid, "store_name"] if sid in names.index else sid
        rows.append(
            _insight(
                type="segment_focus",
                severity="high" if r["segment"] == "Churn Risk" else "medium",
                entity_type="shop",
                entity_id=sid,
                entity_name=name,
                title=f"{name} tagged {r['segment']}",
                narrative=(
                    f"{name} — recency {r['recency_months']:.0f}m, billed {r['frequency']*100:.0f}% of months, "
                    f"avg {r['monetary']:.2f} MT, trend {r['trend']*100:.0f}%."
                ),
                action="Churn Risk: recovery call. Growth Target: expand SKU depth. Lumpy: stop dumping, fix cadence.",
                metric_value=float(r["monetary"] or 0),
                metrics={"segment": r["segment"], "frequency": r["frequency"], "trend": r["trend"]},
            )
        )
    return rows


def _pack_family(sku: str) -> str:
    text = str(sku)
    m = re.search(r"(\d+\s?(kg|l|ltr|litre|liter|ml|g)\b)", text, flags=re.I)
    if m:
        family = re.sub(r"\d+\s?(kg|l|ltr|litre|liter|ml|g)\b", "", text, flags=re.I)
        return family.strip(" -_/") or "Unknown"
    # Strip trailing pack tokens
    return re.sub(r"\b(tin|pouch|pet|jar)\b", "", text, flags=re.I).strip() or text


def _cannibalization_insights(sales: pd.DataFrame, period: str, prev: str) -> list[dict]:
    rows = []
    if sales is None or sales.empty:
        return rows
    cur = sales[sales["period"] == period].groupby("sku")["volume_mt"].sum()
    prv = sales[sales["period"] == prev].groupby("sku")["volume_mt"].sum()
    if cur.empty or prv.empty:
        return rows
    total_cur, total_prv = cur.sum(), prv.sum()
    if total_cur <= 0 or total_prv <= 0:
        return rows
    share_cur = cur / total_cur
    share_prv = prv / total_prv
    delta = share_cur.subtract(share_prv, fill_value=0)
    vol_delta = cur.subtract(prv, fill_value=0)
    # Pairs: one SKU share up, another down, same family
    gainers = delta[delta > 0.03].sort_values(ascending=False)
    losers = delta[delta < -0.03].sort_values()
    for g_sku, g_share in gainers.items():
        fam = _pack_family(g_sku)
        for l_sku, l_share in losers.items():
            if _pack_family(l_sku) != fam and fam == _pack_family(g_sku):
                # still allow cross-pack if names share a stem
                stem_g = re.split(r"\d", g_sku)[0].strip()[:10]
                if stem_g and stem_g.lower() not in l_sku.lower():
                    continue
            # Require loser actual volume down, gainer volume up
            if vol_delta.get(g_sku, 0) <= 0 or vol_delta.get(l_sku, 0) >= 0:
                continue
            net = float(vol_delta.get(g_sku, 0) + vol_delta.get(l_sku, 0))
            severity = "medium" if net >= 0 else "high"
            rows.append(
                _insight(
                    type="cannibalization",
                    severity=severity,
                    entity_type="sku_pair",
                    entity_id=f"{g_sku}||{l_sku}",
                    entity_name=f"{g_sku} vs {l_sku}",
                    title=f"{g_sku} is eating {l_sku}",
                    narrative=(
                        f"{g_sku} share {share_prv.get(g_sku, 0)*100:.1f}% → {share_cur.get(g_sku, 0)*100:.1f}% "
                        f"({vol_delta.get(g_sku, 0):+.2f} MT) while {l_sku} {share_prv.get(l_sku, 0)*100:.1f}% → "
                        f"{share_cur.get(l_sku, 0)*100:.1f}% ({vol_delta.get(l_sku, 0):+.2f} MT). "
                        f"Net family movement {net:+.2f} MT — "
                        + ("mix shift with little new demand." if net < 0.05 else "some incremental volume on top of the switch.")
                    ),
                    action="Check pricing/scheme stacking. Do not celebrate gainer SKU if family volume is flat.",
                    metric_value=abs(float(l_share)),
                    metrics={
                        "gainer": g_sku,
                        "loser": l_sku,
                        "gainer_delta_mt": float(vol_delta.get(g_sku, 0)),
                        "loser_delta_mt": float(vol_delta.get(l_sku, 0)),
                        "net_mt": net,
                    },
                )
            )
            break
        if len(rows) >= 8:
            break
    # Correlation on first differences at national monthly grain (needs history)
    wide = sales.groupby(["period", "sku"])["volume_mt"].sum().unstack(fill_value=0)
    if wide.shape[0] >= 8 and wide.shape[1] >= 2:
        diff = wide.diff().dropna()
        corr = diff.corr()
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                c = corr.loc[a, b]
                if pd.notna(c) and c <= -0.55:
                    pairs.append((c, a, b))
        pairs.sort()
        for c, a, b in pairs[:4]:
            rows.append(
                _insight(
                    type="sku_substitution",
                    severity="medium",
                    entity_type="sku_pair",
                    entity_id=f"{a}||{b}",
                    entity_name=f"{a} vs {b}",
                    title=f"Persistent substitution: {a} ↔ {b}",
                    narrative=(
                        f"Month-to-month volume changes of {a} and {b} move in opposite directions "
                        f"(correlation {c:.2f}). Treat them as one demand pool when setting targets."
                    ),
                    action="Set a combined family target; avoid dual schemes on both SKUs in the same month.",
                    metric_value=abs(float(c)),
                    metrics={"correlation": float(c), "sku_a": a, "sku_b": b},
                )
            )
    return rows


def _pareto_insights(shop_month: pd.DataFrame, period: str) -> list[dict]:
    cur = shop_month[(shop_month["period"] == period) & (shop_month["volume_mt"] > 0)].copy()
    if cur.empty:
        return []
    cur = cur.sort_values("volume_mt", ascending=False)
    total = cur["volume_mt"].sum()
    n = max(len(cur), 1)
    top20 = cur.head(max(int(round(n * 0.2)), 1))
    share = float(top20["volume_mt"].sum() / total) if total else 0
    rows = [
        _insight(
            type="concentration",
            severity="medium" if share >= 0.65 else "low",
            entity_type="national",
            entity_id="ALL",
            title=f"Top 20% of billed shops deliver {share*100:.0f}% of volume",
            narrative=(
                f"{len(top20)} shops (of {n} billed) produced {share*100:.0f}% of {total:.1f} MT. "
                + (
                    "High concentration means a handful of accounts can make or break the month — protect them and build the middle tier."
                    if share >= 0.65
                    else "Volume is reasonably spread; keep expanding the productive middle."
                )
            ),
            action="Build a 'protect' list of top accounts and a 'develop' list from the next quintile.",
            metric_value=share,
            metrics={"top20_share": share, "billed": n, "volume_mt": total},
        )
    ]
    return rows


def _forecast_gap_insights(forecasts: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    if forecasts is None or forecasts.empty:
        return []
    f = forecasts[(forecasts["entity_type"] == "shop") & (forecasts["period"] == period)].copy()
    if f.empty:
        return []
    names = shop_month.drop_duplicates("store_id").set_index("store_id")
    f["abs_resid"] = f["residual"].abs()
    big = f[f["abs_resid"] >= max(MIN_VOLUME_FLAG_MT, 0.08)].sort_values("abs_resid", ascending=False).head(10)
    rows = []
    for _, r in big.iterrows():
        sid = r["entity_id"]
        name = names.loc[sid, "store_name"] if sid in names.index else sid
        direction = "above" if r["residual"] > 0 else "below"
        rows.append(
            _insight(
                type="forecast_gap",
                severity="medium",
                entity_type="shop",
                entity_id=sid,
                entity_name=name,
                title=f"{name} is {direction} the expected baseline",
                narrative=(
                    f"Actual {r['actual']:.2f} MT vs model {r['predicted']:.2f} MT "
                    f"({r['residual']:+.2f} MT, {r.get('residual_pct') or 0:+.0f}%)."
                ),
                action="If above: confirm genuine offtake. If below: recovery call before month close.",
                metric_value=float(r["abs_resid"]),
                metrics={"actual": r["actual"], "predicted": r["predicted"], "model": r.get("model")},
            )
        )
    return rows


def _positive_insights(kpis: pd.DataFrame, segments: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    dsr = kpis[(kpis["grain"] == "dsr") & (kpis["period"] == period)].copy()
    winners = dsr[(dsr["mom_pct"].fillna(0) >= 10) & (dsr["volume_mt"] >= 0.5)].sort_values("mom_pct", ascending=False)
    for _, r in winners.head(5).iterrows():
        rows.append(
            _insight(
                type="dsr_win",
                severity="positive",
                entity_type="dsr",
                entity_id=r["grain_id"],
                title=f"{r['grain_id']} is having a strong month",
                narrative=(
                    f"+{r['mom_pct']:.1f}% MoM, {r['volume_mt']:.2f} MT, strike {r['strike_rate']*100:.0f}%, "
                    f"drop size {r['drop_size']:.2f} MT."
                ),
                action="Ask what changed (new shops, mix, scheme) and brief peer DSRs.",
                metric_value=float(r["mom_pct"]),
                metrics={"mom_pct": r["mom_pct"], "volume_mt": r["volume_mt"]},
            )
        )
    sku = kpis[(kpis["grain"] == "sku") & (kpis["period"] == period)].copy()
    sku_w = sku[(sku["mom_pct"].fillna(0) >= 15) & (sku["volume_mt"] >= 1)].sort_values("mom_pct", ascending=False)
    for _, r in sku_w.head(4).iterrows():
        rows.append(
            _insight(
                type="sku_win",
                severity="positive",
                entity_type="sku",
                entity_id=r["grain_id"],
                title=f"{r['grain_id']} is carrying growth",
                narrative=f"{r['grain_id']} {r['mom_pct']:+.1f}% MoM on {r['volume_mt']:.2f} MT across {int(r['billed_outlets'])} shops.",
                action="Ensure supply; do not starve a winning SKU to push a slower pack.",
                metric_value=float(r["mom_pct"]),
                metrics={"mom_pct": r["mom_pct"], "volume_mt": r["volume_mt"]},
            )
        )
    return rows


def _volume_bridge_insights(shop_month: pd.DataFrame, period: str) -> list[dict]:
    """Split YoY movement into continuing shops vs new areas vs lost shops.

    A store that first appears this year is expansion, not 'growth from zero'.
    """
    yoy_p = shift_period(period, -12)
    cur = shop_month[shop_month["period"] == period]
    ly = shop_month[shop_month["period"] == yoy_p]
    if cur.empty or ly.empty:
        return []
    cur_ids = set(cur.loc[cur["billed"] == 1, "store_id"])
    ly_ids = set(ly.loc[ly["billed"] == 1, "store_id"])
    continuing = cur_ids & ly_ids
    new_ids = cur_ids - ly_ids
    lost_ids = ly_ids - cur_ids
    cur_vol = float(cur["volume_mt"].sum())
    ly_vol = float(ly["volume_mt"].sum())
    cont_now = float(cur.loc[cur["store_id"].isin(continuing), "volume_mt"].sum())
    cont_ly = float(ly.loc[ly["store_id"].isin(continuing), "volume_mt"].sum())
    new_vol = float(cur.loc[cur["store_id"].isin(new_ids), "volume_mt"].sum())
    lost_vol = float(ly.loc[ly["store_id"].isin(lost_ids), "volume_mt"].sum())
    like_pct = (cont_now - cont_ly) / cont_ly * 100 if cont_ly else None
    headline = (cur_vol - ly_vol) / ly_vol * 100 if ly_vol else None
    rows = [
        _insight(
            type="volume_bridge",
            severity="high" if (headline or 0) < -10 else "medium",
            entity_type="national",
            entity_id="ALL",
            title=f"YoY volume bridge vs {yoy_p}",
            narrative=(
                f"{period} is {cur_vol:.1f} MT vs {ly_vol:.1f} MT in {yoy_p} "
                f"({headline:+.1f}% headline). "
                f"Like-for-like (shops billed in both years): {cont_now:.1f} vs {cont_ly:.1f} MT"
                + (f" ({like_pct:+.1f}%). " if like_pct is not None else ". ")
                + f"New shops (not billed in {yoy_p}): {len(new_ids)} / {new_vol:.1f} MT. "
                f"Lost shops: {len(lost_ids)} / {lost_vol:.1f} MT. "
                "New areas are counted as expansion, not as a recovery from zero."
            ),
            action="Coach lost-shop recovery separately from like-for-like drop size. Do not target YoY on shops that were not on file last year.",
            metric_value=abs(float(headline or 0)),
            metrics={
                "headline_yoy_pct": headline,
                "like_for_like_pct": like_pct,
                "continuing_shops": len(continuing),
                "new_shops": len(new_ids),
                "lost_shops": len(lost_ids),
                "new_mt": new_vol,
                "lost_mt": lost_vol,
                "current_mt": cur_vol,
                "ly_mt": ly_vol,
            },
        )
    ]
    # Sections that are genuinely new this year (no like-for-like base).
    if new_ids:
        sec = (
            cur.loc[cur["store_id"].isin(new_ids)]
            .groupby("section", as_index=False)
            .agg(new_mt=("volume_mt", "sum"), shops=("store_id", "nunique"))
            .sort_values("new_mt", ascending=False)
            .head(5)
        )
        for _, r in sec.iterrows():
            if r["new_mt"] < 0.2:
                continue
            rows.append(
                _insight(
                    type="new_area",
                    severity="positive",
                    entity_type="section",
                    entity_id=str(r["section"]),
                    title=f"{r['section']} is adding shops that were not on file last year",
                    narrative=(
                        f"{int(r['shops'])} shops in {r['section']} billed {r['new_mt']:.2f} MT in {period} "
                        f"with no bill in {yoy_p}. Treat as distribution expansion, not like-for-like growth."
                    ),
                    action="Keep the opening cadence; do not load extra stock just to make the first months look big.",
                    metric_value=float(r["new_mt"]),
                    metrics={"shops": int(r["shops"]), "new_mt": float(r["new_mt"])},
                )
            )
    return rows


def _whitespace_insights(stores: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> list[dict]:
    if stores is None or stores.empty:
        return []
    billed_ids = set(shop_month.loc[(shop_month["period"] == period) & (shop_month["billed"] == 1), "store_id"])
    ever = set(shop_month.loc[shop_month["volume_mt"] > 0, "store_id"])
    never = stores[~stores["store_id"].isin(ever)]
    missed = stores[~stores["store_id"].isin(billed_ids)]
    rows = []
    if len(never):
        by_city = never.groupby("city").size().sort_values(ascending=False)
        top_city = by_city.index[0] if len(by_city) else "unknown"
        pocket_n = int(by_city.iloc[0]) if len(by_city) else 0
        rows.append(
            _insight(
                type="whitespace",
                severity="high" if len(never) >= 20 else "medium",
                entity_type="national",
                entity_id="ALL",
                title=f"{len(never)} universe shops have never been billed",
                narrative=(
                    f"{len(never)} outlets sit on the master list with zero history. "
                    f"Largest pocket: {top_city} ({pocket_n}) shops. "
                    f"{len(missed)} shops (including lapsed) were not billed in {period}."
                ),
                action="Give each DSR a top-20 never-billed list in their own section — do not spray the whole universe.",
                metric_value=float(len(never)),
                metrics={"never_billed": int(len(never)), "unbilled_this_period": int(len(missed))},
            )
        )
    return rows
