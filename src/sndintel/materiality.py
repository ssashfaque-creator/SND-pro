"""Volume-weighted shop tiers — how Nielsen/IRI treat numeric vs weighted distribution.

A shop that billed 8 kg last August is not a 'lost account' in the NSM pack.
Together, hundreds of those shops *are* a coverage KPI. This module splits them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sndintel.config import (
    CORE_VOLUME_SHARE,
    MIN_MATERIAL_MT,
    OCCASIONAL_BILLED_RATE,
)
from sndintel.io_utils import shift_period


def tier_from_volumes(volumes: pd.Series) -> pd.DataFrame:
    """Pareto tiers on a reference month (usually last year).

    * core   — shops that make up the first 80% of volume (weighted distribution)
    * middle — next 15% of volume, plus any shop still above MIN_MATERIAL_MT
    * tail   — the bottom ~5% of volume, usually most of the shop count
    """
    s = volumes.fillna(0)
    s = s[s > 0].sort_values(ascending=False)
    if s.empty:
        return pd.DataFrame(columns=["store_id", "volume_mt", "share", "cum_share", "tier"])
    total = float(s.sum()) or 1.0
    out = pd.DataFrame({"store_id": s.index.astype(str), "volume_mt": s.to_numpy(dtype=float)})
    out["share"] = out["volume_mt"] / total
    out["cum_share"] = out["share"].cumsum()
    out["tier"] = "tail"
    # The shop that crosses 80% still counts as core (standard Pareto cut).
    core_cut = out["cum_share"] <= CORE_VOLUME_SHARE
    crossed = (~core_cut) & (out["cum_share"].shift(1).fillna(0) <= CORE_VOLUME_SHARE)
    out.loc[core_cut | crossed, "tier"] = "core"
    # Productive middle: still a real drop, but not in the 80% core.
    # Do not let the 95% volume band pull micro shops into "middle".
    out.loc[(out["tier"] != "core") & (out["volume_mt"] >= MIN_MATERIAL_MT), "tier"] = "middle"
    return out


def classify_gap_shops(
    shop_month: pd.DataFrame,
    period: str,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Split current vs year-ago billed sets into material vs long-tail vs occasional."""
    yoy_p = shift_period(period, -12)
    cur = shop_month[shop_month["period"] == period]
    ly = shop_month[shop_month["period"] == yoy_p]
    empty = {
        "yoy_period": yoy_p,
        "tiers": pd.DataFrame(),
        "lost_core": pd.DataFrame(),
        "lost_middle": pd.DataFrame(),
        "lost_tail": pd.DataFrame(),
        "lost_occasional": pd.DataFrame(),
        "new_material": pd.DataFrame(),
        "new_tail": pd.DataFrame(),
        "counts": {},
        "volumes": {},
    }
    if cur.empty or ly.empty:
        return empty

    ly_vol = ly.groupby("store_id")["volume_mt"].sum()
    tiers = tier_from_volumes(ly_vol)
    tier_map = dict(zip(tiers["store_id"], tiers["tier"])) if not tiers.empty else {}

    billed_rate = {}
    if features is not None and not features.empty and "billed_rate_12" in features.columns:
        feat = features[features["period"] == period]
        billed_rate = dict(zip(feat["store_id"].astype(str), feat["billed_rate_12"]))

    cur_ids = set(cur.loc[cur["billed"] == 1, "store_id"].astype(str))
    ly_ids = set(ly.loc[ly["billed"] == 1, "store_id"].astype(str))
    lost_ids = ly_ids - cur_ids
    new_ids = cur_ids - ly_ids

    def _lost_frame(ids: set[str]) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame()
        part = ly[ly["store_id"].astype(str).isin(ids)].copy()
        part["store_id"] = part["store_id"].astype(str)
        part["tier"] = part["store_id"].map(lambda s: tier_map.get(s, "tail"))
        part["billed_rate_12"] = part["store_id"].map(lambda s: billed_rate.get(s))
        part["occasional"] = part["billed_rate_12"].fillna(1.0) < OCCASIONAL_BILLED_RATE
        return part

    lost = _lost_frame(lost_ids)
    if lost.empty:
        lost_core = lost_middle = lost_tail = lost_occ = lost
    else:
        lost_occ = lost[lost["occasional"]]
        regular = lost[~lost["occasional"]]
        lost_core = regular[regular["tier"] == "core"]
        lost_middle = regular[regular["tier"] == "middle"]
        lost_tail = regular[regular["tier"] == "tail"]

    new_cur = cur[cur["store_id"].astype(str).isin(new_ids)].copy()
    if new_cur.empty:
        new_material = new_tail = new_cur
    else:
        new_material = new_cur[new_cur["volume_mt"] >= MIN_MATERIAL_MT]
        new_tail = new_cur[new_cur["volume_mt"] < MIN_MATERIAL_MT]

    def _vol(frame: pd.DataFrame) -> float:
        return float(frame["volume_mt"].sum()) if frame is not None and not frame.empty else 0.0

    def _n(frame: pd.DataFrame) -> int:
        if frame is None or frame.empty:
            return 0
        return int(frame["store_id"].nunique())

    return {
        "yoy_period": yoy_p,
        "tiers": tiers,
        "lost_core": lost_core,
        "lost_middle": lost_middle,
        "lost_tail": lost_tail,
        "lost_occasional": lost_occ,
        "new_material": new_material,
        "new_tail": new_tail,
        "counts": {
            "lost_core": _n(lost_core),
            "lost_middle": _n(lost_middle),
            "lost_tail": _n(lost_tail),
            "lost_occasional": _n(lost_occ),
            "new_material": _n(new_material),
            "new_tail": _n(new_tail),
            "lost_all": len(lost_ids),
            "new_all": len(new_ids),
        },
        "volumes": {
            "lost_core": _vol(lost_core),
            "lost_middle": _vol(lost_middle),
            "lost_tail": _vol(lost_tail),
            "lost_occasional": _vol(lost_occ),
            "new_material": _vol(new_material),
            "new_tail": _vol(new_tail),
        },
    }


def must_visit_recoveries(gap: dict[str, Any], limit: int = 25) -> pd.DataFrame:
    """Material unbilled shops, largest last-year volume first — the recovery list."""
    frames = [gap.get("lost_core"), gap.get("lost_middle")]
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    keep = [c for c in ("store_id", "store_name", "dsr_name", "section", "city", "volume_mt", "tier") if c in out.columns]
    out = out[keep].sort_values("volume_mt", ascending=False).head(limit)
    return out.reset_index(drop=True)
