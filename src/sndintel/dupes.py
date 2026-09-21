"""Duplicate-code and migrated-POP detection.

Two failure modes make a live shop look like a lost door and inflate the
universe:

* **Duplicate code** — the same physical shop bills under two POP codes at
  the same time. Signature: the two codes post identical (date, volume)
  invoices, or the same name under the same DSR with overlapping bills.
* **Migrated code** — a shop was re-coded; the old code stops exactly when
  the new one starts. Signature: same folded name under the same DSR or
  distributor, and the new code's first bill falls within a short window of
  the old code's last bill.

Nothing here changes volume. The output is a flag per POP that the shop book
prints beside the row so a "lost door" is checked before it is chased.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

MIGRATION_WINDOW_DAYS = 45
MIN_MATCHING_INVOICES = 2
LOOKBACK_DAYS = 120

FLAG_DUPLICATE = "Possible duplicate code"
FLAG_MIGRATED_FROM = "Code migrated (old)"
FLAG_MIGRATED_TO = "Code migrated (new)"

_NOISE = re.compile(r"[^a-z0-9 ]+")
_STOP = {"gs", "g s", "store", "shop", "mart", "traders", "general", "karyana", "kiryana", "the", "and", "&"}


def fold_name(value: object) -> str:
    text = _NOISE.sub(" ", str(value or "").casefold())
    words = [w for w in text.split() if w and w not in _STOP]
    return " ".join(words)


def _prep_shop_month(shop_month: pd.DataFrame | None) -> pd.DataFrame:
    if shop_month is None or shop_month.empty:
        return pd.DataFrame()
    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str).str.strip()
    sm["period"] = sm["period"].astype(str)
    sm["volume_mt"] = pd.to_numeric(sm["volume_mt"], errors="coerce").fillna(0.0)
    return sm.loc[sm["volume_mt"] > 0]


def _prep_shop_day(shop_day: pd.DataFrame | None) -> pd.DataFrame:
    if shop_day is None or shop_day.empty:
        return pd.DataFrame()
    sd = shop_day.copy()
    sd["store_id"] = sd["store_id"].astype(str).str.strip()
    sd["sale_date"] = pd.to_datetime(sd["sale_date"], errors="coerce")
    sd["volume_mt"] = pd.to_numeric(sd["volume_mt"], errors="coerce").fillna(0.0)
    sd = sd.loc[sd["sale_date"].notna() & (sd["volume_mt"] > 0)]
    return sd.groupby(["store_id", "sale_date"], as_index=False)["volume_mt"].sum()


def _attrs(shop_month: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("store_name", "distributor", "dsr_name", "city") if c in shop_month.columns]
    latest = shop_month.sort_values("period").drop_duplicates("store_id", keep="last")
    out = latest[["store_id"] + cols].copy()
    for c in ("store_name", "distributor", "dsr_name", "city"):
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["_name"] = out["store_name"].map(fold_name)
    out["_dsr"] = (out["distributor"].str.casefold().str.strip() + " | " + out["dsr_name"].str.casefold().str.strip())
    out["_dist"] = out["distributor"].str.casefold().str.strip()
    return out.set_index("store_id")


def _bill_spans(shop_month: pd.DataFrame, shop_day: pd.DataFrame) -> pd.DataFrame:
    """First and last bill date per POP.

    Month bounds (1st / month-end) come from shop_month. Where daily rows
    exist for the same first / last month they replace the bound with the
    exact day; daily history that starts later than the month history is not
    allowed to move the first bill forward.
    """
    if shop_month.empty:
        return pd.DataFrame(columns=["first", "last"])
    g = shop_month.groupby("store_id")["period"]
    first_p = pd.to_datetime(g.min() + "-01", errors="coerce")
    last_p = pd.to_datetime(g.max() + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    spans = pd.DataFrame({"first": first_p, "last": last_p})
    if shop_day.empty:
        return spans
    dg = shop_day.groupby("store_id")["sale_date"]
    d_first = dg.min().reindex(spans.index)
    d_last = dg.max().reindex(spans.index)
    same_first = d_first.notna() & (d_first.dt.to_period("M") == spans["first"].dt.to_period("M"))
    same_last = d_last.notna() & (d_last.dt.to_period("M") == spans["last"].dt.to_period("M"))
    spans.loc[same_first, "first"] = d_first[same_first]
    spans.loc[same_last, "last"] = d_last[same_last]
    return spans


def find_duplicate_pairs(shop_day: pd.DataFrame, attrs: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.DataFrame:
    """POP pairs under one DSR that posted the same (date, volume) at least twice recently."""
    cols = ["store_id_a", "store_id_b", "matches", "detail"]
    if shop_day.empty or attrs.empty:
        return pd.DataFrame(columns=cols)
    sd = shop_day.copy()
    if as_of is not None:
        sd = sd.loc[sd["sale_date"] >= as_of - pd.Timedelta(days=LOOKBACK_DAYS)]
    sd = sd.loc[sd["store_id"].isin(attrs.index)]
    if sd.empty:
        return pd.DataFrame(columns=cols)
    sd["_dsr"] = sd["store_id"].map(attrs["_dsr"])
    sd["_vol"] = sd["volume_mt"].round(4)
    rows = []
    for (_dsr, _date, _vol), part in sd.groupby(["_dsr", "sale_date", "_vol"]):
        ids = sorted(set(part["store_id"]))
        if len(ids) < 2 or len(ids) > 6:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                rows.append((ids[i], ids[j]))
    if not rows:
        return pd.DataFrame(columns=cols)
    pairs = pd.DataFrame(rows, columns=["store_id_a", "store_id_b"]).value_counts().reset_index(name="matches")
    pairs = pairs.loc[pairs["matches"] >= MIN_MATCHING_INVOICES].copy()
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    pairs["detail"] = pairs.apply(
        lambda r: f"{int(r['matches'])} identical (date, volume) invoices with {r['store_id_b']} in the last {LOOKBACK_DAYS} days",
        axis=1,
    )
    return pairs[cols].reset_index(drop=True)


def find_migration_pairs(attrs: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Same folded name under one DSR (or distributor) where the new code starts as the old one stops."""
    cols = ["old_id", "new_id", "gap_days", "detail"]
    if attrs.empty or spans.empty:
        return pd.DataFrame(columns=cols)
    work = attrs.join(spans, how="inner")
    work = work.loc[(work["_name"].str.len() >= 3) & work["first"].notna() & work["last"].notna()]
    if work.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key_col in ("_dsr", "_dist"):
        for (_k, _n), part in work.groupby([key_col, "_name"]):
            if len(part) < 2 or len(part) > 6:
                continue
            # Across DSRs of one distributor a single generic word ("Bismillah")
            # is not evidence; require a two-word name there.
            if key_col == "_dist" and len(str(_n).split()) < 2:
                continue
            part = part.sort_values("first")
            ids = list(part.index)
            for i in range(len(ids)):
                for j in range(len(ids)):
                    if i == j:
                        continue
                    old, new = ids[i], ids[j]
                    old_last = part.loc[old, "last"]
                    new_first = part.loc[new, "first"]
                    if pd.isna(old_last) or pd.isna(new_first):
                        continue
                    gap = (new_first - old_last).days
                    if -MIGRATION_WINDOW_DAYS <= gap <= MIGRATION_WINDOW_DAYS and new_first > part.loc[old, "first"]:
                        rows.append((old, new, int(gap)))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=["old_id", "new_id", "gap_days"]).drop_duplicates(["old_id", "new_id"])
    out["detail"] = out.apply(
        lambda r: (
            f"same name, same DSR/distributor; {r['new_id']} first billed "
            f"{abs(int(r['gap_days']))} days {'after' if r['gap_days'] >= 0 else 'before'} {r['old_id']} last billed"
        ),
        axis=1,
    )
    return out[cols].reset_index(drop=True)


def flag_shops(
    shop_month: pd.DataFrame | None,
    shop_day: pd.DataFrame | None,
    period: str | None,
    scope_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per flagged POP: ``store_id, flag, flag_pair, flag_detail``."""
    cols = ["store_id", "flag", "flag_pair", "flag_detail"]
    sm = _prep_shop_month(shop_month)
    if sm.empty:
        return pd.DataFrame(columns=cols)
    sd = _prep_shop_day(shop_day)
    attrs = _attrs(sm)
    as_of = None
    if period:
        try:
            as_of = pd.to_datetime(f"{str(period)[:7]}-01") + pd.offsets.MonthEnd(0)
        except Exception:
            as_of = None
    spans = _bill_spans(sm, sd)
    dupes = find_duplicate_pairs(sd, attrs, as_of)
    migr = find_migration_pairs(attrs, spans)
    rows: dict[str, dict[str, str]] = {}

    def _put(sid: str, flag: str, pair: str, detail: str) -> None:
        if sid in rows:
            return
        rows[sid] = {"store_id": sid, "flag": flag, "flag_pair": pair, "flag_detail": detail}

    for _, r in migr.iterrows():
        _put(str(r["old_id"]), FLAG_MIGRATED_FROM, str(r["new_id"]), str(r["detail"]))
        _put(str(r["new_id"]), FLAG_MIGRATED_TO, str(r["old_id"]), f"continues {r['old_id']}: {r['detail']}")
    for _, r in dupes.iterrows():
        _put(str(r["store_id_a"]), FLAG_DUPLICATE, str(r["store_id_b"]), str(r["detail"]))
        _put(str(r["store_id_b"]), FLAG_DUPLICATE, str(r["store_id_a"]), str(r["detail"]).replace(str(r["store_id_b"]), str(r["store_id_a"])))
    out = pd.DataFrame(list(rows.values()), columns=cols)
    if scope_ids is not None and not out.empty:
        wanted = {str(s) for s in scope_ids}
        out = out.loc[out["store_id"].isin(wanted)]
    return out.reset_index(drop=True)
