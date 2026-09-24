"""POP code lineage: retired codes and the live codes that continue them.

The universe file is the complete list of doors. A POP code with billing
history that is *not* on the current universe is a **retired code**: the
company re-coded the shop (route change, distributor change, duplicate
clean-up) or the shop was dropped. Left alone, a retired code carries its
run-rate into Expected while the successor code starts from zero, so one
physical door is counted twice — once as a "lost door" hole and once as a
"new door" beating Expected.

Resolution is conservative and structural:

* Candidates for a retired code are live codes with the same folded name
  whose first bill comes after the retired code's first bill and lands within
  ``MIGRATION_WINDOW_DAYS`` of its last bill (a short overlap is allowed —
  the route often bills both codes in the changeover week).
* Corroboration adds to the score: same distributor, same DSR, same city
  (when the retired code's old assignment is known), and a shared code
  prefix (the code family the company issues per region / distributor).
* A pair is accepted when the best candidate is unambiguous — either it is
  the only candidate, or it beats the runner-up by a clear margin.

Accepted pairs are persisted in ``pop_lineage`` and applied **in memory**
to shop-days and facts before scorecards are built: the retired code's
history becomes the live code's history. The warehouse tables keep the raw
codes so the extract still reconciles line by line.

Retired codes with no accepted successor stay on the panel as their own
bucket ("Retired POP code") — never as a lost door or a miss.

A second pass catches **superseded codes the universe file has not retired
yet**: a listed code that went quiet well before the panel end exactly when a
same-name code under the same distributor, city and DSR (or code route)
started. Those are merged the same way and reported as "Superseded code" so
the universe can be cleaned; until then they are not counted as doors.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sndintel.dupes import fold_name

MIGRATION_WINDOW_DAYS = 75
MAX_OVERLAP_DAYS = 45
CLEAN_HANDOVER_DAYS = 14
PREFIX_FAMILY = 9
PREFIX_ROUTE = 12
ACCEPT_MARGIN = 1.0
MAX_CHAIN = 3
# A code that is *still on the universe* but stopped billing when a same-name
# code under the same distributor started is a superseded code the universe
# file has not retired yet. Both codes are listed as doors, so the evidence
# bar is higher: quiet for this long, a clean handover, the same distributor
# and city, *and* one piece of local evidence — the same DSR or the same code
# route. Shop names ("Khan Store", "Madina Store") repeat many times per city,
# so distributor + city alone is not enough to merge two listed doors.
SUPERSEDED_QUIET_DAYS = 30
SUPERSEDED_MIN_SCORE = 5.0
SUPERSEDED_LOCAL_EVIDENCE = {"same DSR", "same code route"}
METHOD_RETIRED = "name+handover"
METHOD_SUPERSEDED = "superseded on universe"

LINEAGE_COLUMNS = [
    "old_id",
    "new_id",
    "method",
    "confidence",
    "score",
    "gap_days",
    "old_first",
    "old_last",
    "new_first",
    "old_name",
    "new_name",
    "detail",
]

RETIRED_COLUMNS = [
    "store_id",
    "store_name",
    "distributor",
    "dsr_name",
    "city",
    "first_bill",
    "last_bill",
    "n_candidates",
    "best_candidate",
    "reason",
]


def _norm_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _fold_col(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index)
    return frame[col].fillna("").astype(str).str.strip().str.casefold()


def universe_ids(stores: pd.DataFrame | None) -> set[str] | None:
    """Live universe codes, or None when no universe has been loaded."""
    if stores is None or stores.empty or "store_id" not in stores.columns or "in_universe" not in stores.columns:
        return None
    live = pd.to_numeric(stores["in_universe"], errors="coerce").fillna(0) == 1
    if not live.any():
        return None
    return set(_norm_ids(stores.loc[live, "store_id"]))


def _spans(shop_month: pd.DataFrame, shop_day: pd.DataFrame | None) -> pd.DataFrame:
    """First / last billed date per code (day precision where shop_day has it)."""
    sm = shop_month.copy()
    sm["store_id"] = _norm_ids(sm["store_id"])
    sm["period"] = sm["period"].astype(str)
    sm["volume_mt"] = pd.to_numeric(sm["volume_mt"], errors="coerce").fillna(0.0)
    billed = sm.loc[sm["volume_mt"] > 0]
    if billed.empty:
        return pd.DataFrame(columns=["first", "last", "n_months", "recent_mt"])
    g = billed.groupby("store_id")
    first = pd.to_datetime(g["period"].min() + "-01", errors="coerce")
    last = pd.to_datetime(g["period"].max() + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"first": first, "last": last, "n_months": g["period"].nunique()})
    last3 = sorted(sm["period"].unique())[-3:]
    recent = billed.loc[billed["period"].isin(last3)].groupby("store_id")["volume_mt"].sum() / max(len(last3), 1)
    out["recent_mt"] = recent.reindex(out.index).fillna(0.0)
    if shop_day is None or shop_day.empty or "sale_date" not in shop_day.columns:
        return out
    sd = shop_day.copy()
    sd["store_id"] = _norm_ids(sd["store_id"])
    sd["sale_date"] = pd.to_datetime(sd["sale_date"], errors="coerce")
    sd["volume_mt"] = pd.to_numeric(sd["volume_mt"], errors="coerce").fillna(0.0)
    sd = sd.loc[sd["sale_date"].notna() & (sd["volume_mt"] > 0)]
    if sd.empty:
        return out
    dg = sd.groupby("store_id")["sale_date"]
    d_first = dg.min().reindex(out.index)
    d_last = dg.max().reindex(out.index)
    same_first = d_first.notna() & (d_first.dt.to_period("M") == out["first"].dt.to_period("M"))
    same_last = d_last.notna() & (d_last.dt.to_period("M") == out["last"].dt.to_period("M"))
    out.loc[same_first, "first"] = d_first[same_first]
    out.loc[same_last, "last"] = d_last[same_last]
    return out


def _attrs(shop_month: pd.DataFrame, stores: pd.DataFrame | None) -> pd.DataFrame:
    """Name / distributor / DSR / city per code. Stores first, shop_month fills gaps."""
    cols = ["store_name", "distributor", "dsr_name", "city"]
    frames = []
    if stores is not None and not stores.empty and "store_id" in stores.columns:
        st = stores.copy()
        st["store_id"] = _norm_ids(st["store_id"])
        if "in_universe" in st.columns:
            st = st.sort_values("in_universe", ascending=False, kind="mergesort")
        st = st.drop_duplicates("store_id", keep="first")
        for c in cols:
            if c not in st.columns:
                st[c] = None
        frames.append(st.set_index("store_id")[cols])
    sm = shop_month.copy()
    sm["store_id"] = _norm_ids(sm["store_id"])
    sm = sm.sort_values("period", kind="mergesort").drop_duplicates("store_id", keep="last")
    for c in cols:
        if c not in sm.columns:
            sm[c] = None
    frames.append(sm.set_index("store_id")[cols])
    out = frames[0]
    for extra in frames[1:]:
        out = out.combine_first(extra)
    for c in cols:
        out[c] = out[c].fillna("").astype(str).str.strip()
    out["_name"] = out["store_name"].map(fold_name)
    return out


def _score_pair(old: pd.Series, new: pd.Series) -> tuple[float, list[str]]:
    score = 0.0
    why: list[str] = []
    old_dist, new_dist = str(old["distributor"]).casefold(), str(new["distributor"]).casefold()
    old_dsr, new_dsr = str(old["dsr_name"]).casefold(), str(new["dsr_name"]).casefold()
    old_city, new_city = str(old["city"]).casefold(), str(new["city"]).casefold()
    if old_dist and new_dist:
        if old_dist == new_dist:
            score += 3.0
            why.append("same distributor")
        else:
            score -= 1.0
    if old_dsr and new_dsr and old_dist == new_dist:
        if old_dsr == new_dsr:
            score += 1.0
            why.append("same DSR")
    if old_city and new_city:
        if old_city == new_city:
            score += 1.0
            why.append("same city")
        else:
            score -= 2.0
    old_id, new_id = str(old.name), str(new.name)
    if old_id[:PREFIX_ROUTE] == new_id[:PREFIX_ROUTE]:
        score += 2.0
        why.append("same code route")
    elif old_id[:PREFIX_FAMILY] == new_id[:PREFIX_FAMILY]:
        score += 1.0
        why.append("same code family")
    gap = int(old["gap_days"])
    if abs(gap) <= CLEAN_HANDOVER_DAYS:
        score += 1.0
        why.append(f"handover within {abs(gap)} days")
    else:
        score -= min(1.0, abs(gap) / MIGRATION_WINDOW_DAYS)
    return score, why


def build_lineage(
    shop_month: pd.DataFrame | None,
    stores: pd.DataFrame | None,
    shop_day: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accepted pairs, retired codes without an accepted successor)."""
    empty = pd.DataFrame(columns=LINEAGE_COLUMNS)
    empty_retired = pd.DataFrame(columns=RETIRED_COLUMNS)
    if shop_month is None or shop_month.empty:
        return empty, empty_retired
    live = universe_ids(stores)
    if live is None:
        return empty, empty_retired
    spans = _spans(shop_month, shop_day)
    if spans.empty:
        return empty, empty_retired
    attrs = _attrs(shop_month, stores)
    work = attrs.join(spans, how="inner")
    work = work.loc[work["first"].notna() & work["last"].notna()]
    ids = pd.Index(work.index.astype(str))
    work.index = ids
    work["_live"] = ids.isin(live)
    live_rows = work.loc[work["_live"] & (work["_name"].str.len() >= 3)]
    by_name = {name: part for name, part in live_rows.groupby("_name")} if not live_rows.empty else {}

    accepted: list[dict] = []
    unresolved: list[dict] = []

    # Pass 1 — codes off the universe: retired by the universe file itself.
    retired = work.loc[~work["_live"]]
    for old_id, old in retired.iterrows():
        pair, left = _match_successor(old_id, old, by_name, work, method=METHOD_RETIRED)
        if pair is not None:
            accepted.append(pair)
        elif left is not None:
            unresolved.append(left)

    # Pass 2 — codes still on the universe that went quiet exactly when a
    # same-name code under the same distributor started. The universe file
    # lists both as doors, so only a clean, well-corroborated handover counts.
    panel_end = work["last"].max()
    quiet_cut = panel_end - pd.Timedelta(days=SUPERSEDED_QUIET_DAYS)
    superseded_pool = work.loc[work["_live"] & (work["last"] <= quiet_cut) & (work["_name"].str.len() >= 3)]
    for old_id, old in superseded_pool.iterrows():
        pair, _left = _match_successor(
            old_id,
            old,
            by_name,
            work,
            method=METHOD_SUPERSEDED,
            max_gap=MIGRATION_WINDOW_DAYS,
            max_overlap=CLEAN_HANDOVER_DAYS,
            min_score=SUPERSEDED_MIN_SCORE,
            exclude={str(old_id)},
        )
        if pair is not None:
            accepted.append(pair)

    pairs = pd.DataFrame(accepted, columns=LINEAGE_COLUMNS)
    pairs = _collapse_chains(pairs)
    left = pd.DataFrame(unresolved, columns=RETIRED_COLUMNS)
    return pairs.reset_index(drop=True), left.reset_index(drop=True)


def _match_successor(
    old_id: str,
    old: pd.Series,
    by_name: dict[str, pd.DataFrame],
    work: pd.DataFrame,
    *,
    method: str,
    max_gap: int = MIGRATION_WINDOW_DAYS,
    max_overlap: int = MAX_OVERLAP_DAYS,
    min_score: float | None = None,
    exclude: set[str] | None = None,
) -> tuple[dict | None, dict | None]:
    """Best live successor for one code, or the reason none was accepted."""
    name = str(old["_name"])
    cands = by_name.get(name) if len(name) >= 3 else None
    if cands is None or cands.empty:
        return None, _retired_row(old_id, old, 0, "", "no live code with this name")
    if exclude:
        cands = cands.loc[~cands.index.astype(str).isin(exclude)]
        if cands.empty:
            return None, _retired_row(old_id, old, 0, "", "no other live code with this name")
    gap_days = (cands["first"] - old["last"]).dt.days
    started_after = cands["first"] > old["first"]
    window = (gap_days >= -int(max_overlap)) & (gap_days <= int(max_gap))
    pool = cands.loc[started_after & window].copy()
    if pool.empty:
        return None, _retired_row(
            old_id, old, int(len(cands)), "", "same-name live code(s) did not start when this one stopped"
        )
    pool["gap_days"] = gap_days.loc[pool.index]
    scored = []
    for new_id, new in pool.iterrows():
        probe = old.copy()
        probe["gap_days"] = int(new["gap_days"])
        score, why = _score_pair(probe.rename(old_id), new.rename(new_id))
        scored.append((score, str(new_id), int(new["gap_days"]), why))
    scored.sort(key=lambda t: (-t[0], abs(t[2]), t[1]))
    best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else None
    unique = runner is None
    clear = unique or (best[0] - runner) >= ACCEPT_MARGIN
    if not clear:
        return None, _retired_row(
            old_id,
            old,
            int(len(scored)),
            best[1],
            f"{len(scored)} live codes share this name; none is clearly the successor",
        )
    if min_score is not None:
        if best[0] < float(min_score) or not (set(best[3]) & SUPERSEDED_LOCAL_EVIDENCE):
            return None, _retired_row(old_id, old, int(len(scored)), best[1], "handover not corroborated")
        confidence = "high"
    else:
        if unique and best[0] < 0:
            return None, _retired_row(old_id, old, 1, best[1], "only same-name code sits in another city/distributor")
        confidence = "high" if (best[0] >= 3.0) else ("medium" if unique or best[0] >= 1.0 else "low")
        if confidence == "low":
            return None, _retired_row(old_id, old, int(len(scored)), best[1], "weak evidence — check before merging")
    new = work.loc[best[1]]
    gap = best[2]
    when = "after" if gap >= 0 else "before"
    why = "; ".join(best[3]) if best[3] else "same name"
    return (
        {
            "old_id": str(old_id),
            "new_id": str(best[1]),
            "method": method,
            "confidence": confidence,
            "score": round(float(best[0]), 2),
            "gap_days": int(gap),
            "old_first": old["first"].strftime("%Y-%m-%d"),
            "old_last": old["last"].strftime("%Y-%m-%d"),
            "new_first": new["first"].strftime("%Y-%m-%d"),
            "old_name": str(old["store_name"]),
            "new_name": str(new["store_name"]),
            "detail": f"{best[1]} first billed {abs(gap)} days {when} {old_id} last billed; {why}",
        },
        None,
    )


def _retired_row(old_id: str, old: pd.Series, n_cands: int, best: str, reason: str) -> dict:
    return {
        "store_id": str(old_id),
        "store_name": str(old.get("store_name") or ""),
        "distributor": str(old.get("distributor") or ""),
        "dsr_name": str(old.get("dsr_name") or ""),
        "city": str(old.get("city") or ""),
        "first_bill": old["first"].strftime("%Y-%m-%d") if pd.notna(old.get("first")) else "",
        "last_bill": old["last"].strftime("%Y-%m-%d") if pd.notna(old.get("last")) else "",
        "n_candidates": int(n_cands),
        "best_candidate": str(best or ""),
        "reason": reason,
    }


def _collapse_chains(pairs: pd.DataFrame) -> pd.DataFrame:
    """old → mid → new becomes old → new (successors are live codes, so chains are rare)."""
    if pairs.empty:
        return pairs
    nxt = dict(zip(pairs["old_id"].astype(str), pairs["new_id"].astype(str)))
    out = pairs.copy()
    for i, row in out.iterrows():
        target = str(row["new_id"])
        hops = 0
        while target in nxt and hops < MAX_CHAIN:
            target = nxt[target]
            hops += 1
        out.at[i, "new_id"] = target
    return out


def lineage_map(pairs: pd.DataFrame | None) -> dict[str, str]:
    if pairs is None or pairs.empty or "old_id" not in pairs.columns:
        return {}
    return {str(o): str(n) for o, n in zip(pairs["old_id"], pairs["new_id"]) if str(o) != str(n)}


def apply_lineage(frame: pd.DataFrame | None, pairs: pd.DataFrame | dict | None, id_col: str = "store_id") -> pd.DataFrame:
    """Re-key rows from retired codes to their live successor. Idempotent."""
    if frame is None or frame.empty or id_col not in frame.columns:
        return frame if frame is not None else pd.DataFrame()
    mapping = pairs if isinstance(pairs, dict) else lineage_map(pairs)
    if not mapping:
        return frame
    out = frame.copy()
    ids = _norm_ids(out[id_col])
    hit = ids.isin(mapping.keys())
    if not hit.any():
        return out
    out[id_col] = ids.where(~hit, ids.map(mapping))
    return out


def continues_map(pairs: pd.DataFrame | None) -> dict[str, list[str]]:
    """new_id → [old codes it continues]."""
    out: dict[str, list[str]] = {}
    for old, new in lineage_map(pairs).items():
        out.setdefault(new, []).append(old)
    return out


def load_lineage(conn) -> pd.DataFrame:
    from sndintel.storage import read_sql

    try:
        pairs = read_sql(conn, "SELECT * FROM pop_lineage")
    except Exception:
        return pd.DataFrame(columns=LINEAGE_COLUMNS)
    return pairs if pairs is not None else pd.DataFrame(columns=LINEAGE_COLUMNS)


def load_shop_day(conn) -> pd.DataFrame:
    """shop_day with retired codes re-keyed to their live successor (what the pipeline scored)."""
    from sndintel.storage import read_sql

    try:
        shop_day = read_sql(conn, "SELECT * FROM shop_day")
    except Exception:
        return pd.DataFrame()
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    return apply_lineage(shop_day, load_lineage(conn))


def retired_ids(stores: pd.DataFrame | None, ids: Iterable[str]) -> set[str]:
    """Codes in ``ids`` that are off the live universe (empty when no universe is loaded)."""
    live = universe_ids(stores)
    if live is None:
        return set()
    return {str(s).strip() for s in ids if str(s).strip() not in live}
