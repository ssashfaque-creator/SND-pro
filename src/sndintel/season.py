"""Expected is recent run-rate, not a calendar-month seasonal index.

A city that never billed in August (LY = 0) can still be running 40 MT/month
in May–July. Multiplying destationalized trend by an August index of ~0
produces a 4 MT Expected against a 44 MT AMS — a false “ahead”. Expected
therefore uses the last three closed months (same window as AMS), blended
with the last-six-month median, then paced if MTD is open. Children still
add to the parent Expected.

A 12-month index is stored for diagnostics. It is not applied to Expected.
Open-MTD pace is the country’s usual billed share by that calendar day,
learned from Outlet Date Wise across closed months. One national curve
is applied at every grain — stores and cities are too thin to fit their
own day shape. Fallback is learned MTD snapshot cuts, then elapsed days.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sndintel.io_utils import prior_periods

NATIONAL_DAY_MIN_MONTHS = 2
NATIONAL_DAY_MIN_DAYS = 8


def _finite(value: Any, default: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


SEASON_COLUMNS = [
    "grain",
    "grain_id",
    "month",
    "seasonal_index",
    "typical_mt",
    "n_obs",
    "credibility",
]


@dataclass
class SeasonFit:
    period: str
    month: int
    n_periods: int
    n_same_month: int
    national_index: dict[int, float] = field(default_factory=dict)
    expected_full_national: float = 0.0
    expected_drop_national: float = float("nan")
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    city_expected: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "history"


def fit_seasonality(shop_month: pd.DataFrame, period: str) -> SeasonFit:
    """Fit calendar-month seasonality from every period except the one being scored."""
    month = int(str(period)[5:7]) if period and len(period) >= 7 else 0
    empty = SeasonFit(period=period or "", month=month, n_periods=0, n_same_month=0)
    if shop_month is None or shop_month.empty or not period:
        return empty
    sm = shop_month.copy()
    if "month" not in sm.columns:
        sm["month"] = sm["period"].astype(str).str.slice(5, 7).astype(int)
    if "city" not in sm.columns:
        sm["city"] = "(unmapped)"
    sm["city"] = sm["city"].fillna("(unmapped)").replace("", "(unmapped)")
    hist = sm[sm["period"] != period]
    if hist.empty:
        return empty
    n_periods = int(hist["period"].nunique())
    nat = hist.groupby("period", as_index=False)["volume_mt"].sum()
    nat["month"] = nat["period"].astype(str).str.slice(5, 7).astype(int)
    nat_idx = _iterative_month_index(nat)
    n_same = int((nat["month"] == month).sum())
    recent_ps = prior_periods(period, 3)
    longer_ps = prior_periods(period, 6)
    expected_nat, _, _, _ = _recent_level(nat, recent_ps, longer_ps)
    nat_metrics = _period_volume_and_shops(hist, [])
    expected_drop_nat = _expected_drop_size(nat_metrics, recent_ps, longer_ps)

    city_period = hist.groupby(["city", "period"], as_index=False)["volume_mt"].sum()
    city_period["month"] = city_period["period"].astype(str).str.slice(5, 7).astype(int)
    city_metrics = _period_volume_and_shops(hist, ["city"])
    city_idx_rows = []
    city_exp_rows = []
    for city, g in city_period.groupby("city"):
        raw = _iterative_month_index(g)
        n_city = int(g["period"].nunique())
        cred = n_city / (n_city + 4)
        mixed = {}
        for m in range(1, 13):
            local = raw.get(m)
            natv = nat_idx.get(m, 1.0)
            if local is None or not np.isfinite(_finite(local, default=float("nan")) if local is not None else float("nan")):
                mixed[m] = _finite(natv)
            else:
                mixed[m] = _finite(cred * float(local) + (1 - cred) * _finite(natv))
        same = g[g["month"] == month]
        n_s = int(len(same))
        expected, ams, trend, _ = _recent_level(g, recent_ps, longer_ps)
        gm = (
            city_metrics[city_metrics["city"].astype(str) == str(city)]
            if city_metrics is not None and not city_metrics.empty
            else g
        )
        drop_e = _expected_drop_size(gm, recent_ps, longer_ps)
        city_exp_rows.append(
            {
                "city": city,
                "expected_full_mt": float(expected) if pd.notna(expected) else 0.0,
                "typical_mt": ams if pd.notna(ams) else None,
                "trend_mt": trend if pd.notna(trend) else None,
                "n_same_month": n_s,
                "seasonal_index": 1.0,
                "expected_drop_size_mt": drop_e if pd.notna(drop_e) else None,
            }
        )
        for m, val in mixed.items():
            typ_m = g.loc[g["month"] == m, "volume_mt"]
            typical_m = float(typ_m.mean()) if len(typ_m) else None
            if typical_m is not None and not np.isfinite(typical_m):
                typical_m = None
            city_idx_rows.append(
                {
                    "grain": "city",
                    "grain_id": str(city),
                    "month": int(m),
                    "seasonal_index": _finite(val),
                    "typical_mt": typical_m,
                    "n_obs": int(len(typ_m)),
                    "credibility": _finite(cred, 0.0),
                }
            )
    table_nat = []
    for m in range(1, 13):
        typ = nat.loc[nat["month"] == m, "volume_mt"]
        table_nat.append(
            {
                "grain": "national",
                "grain_id": "ALL",
                "month": m,
                "seasonal_index": _finite(nat_idx.get(m, 1.0)),
                "typical_mt": float(typ.mean()) if len(typ) else None,
                "n_obs": int(len(typ)),
                "credibility": 1.0,
            }
        )
    table = pd.DataFrame(table_nat + city_idx_rows)
    if not table.empty:
        table["period"] = period
    city_expected = pd.DataFrame(city_exp_rows)
    source = "history"
    if n_periods >= 12:
        source = "history_full_year"
    if n_same >= 2:
        source = "history_repeat_month"
    return SeasonFit(
        period=period,
        month=month,
        n_periods=n_periods,
        n_same_month=n_same,
        national_index=nat_idx,
        expected_full_national=float(expected_nat) if pd.notna(expected_nat) else 0.0,
        expected_drop_national=float(expected_drop_nat) if pd.notna(expected_drop_nat) else float("nan"),
        table=table,
        city_expected=city_expected,
        source=source,
    )


def _iterative_month_index(frame: pd.DataFrame, rounds: int = 2) -> dict[int, float]:
    """Classical seasonal index: month mean vs destationalized trend, mean 1.0."""
    if frame.empty or "volume_mt" not in frame.columns:
        return {m: 1.0 for m in range(1, 13)}
    df = frame.dropna(subset=["volume_mt", "month"]).copy()
    if df.empty:
        return {m: 1.0 for m in range(1, 13)}
    grand = float(df["volume_mt"].mean()) or 1.0
    idx = df.groupby("month")["volume_mt"].mean() / grand
    idx = idx.to_dict()
    df = df.sort_values("period") if "period" in df.columns else df
    for _ in range(rounds):
        df["idx"] = df["month"].map(idx).fillna(1.0).replace(0, 1.0)
        df["dest"] = df["volume_mt"] / df["idx"]
        if "period" in df.columns:
            df["trend"] = df["dest"].rolling(3, min_periods=1).median()
        else:
            df["trend"] = df["dest"]
        df["trend"] = df["trend"].replace(0, np.nan)
        ratio = df["volume_mt"] / df["trend"]
        new = ratio.groupby(df["month"]).mean()
        if new.empty or float(new.mean() or 0) == 0:
            break
        new = new / float(new.mean())
        idx = {int(m): _finite(v) for m, v in new.items()}
    for m in range(1, 13):
        idx[m] = _finite(idx.get(m, 1.0))
    return idx


def _trend_seasonal(frame: pd.DataFrame, idx: dict[int, float], month: int) -> float:
    """Kept for diagnostics. Expected does not use this."""
    if frame.empty:
        return float("nan")
    df = frame.copy()
    df["idx"] = df["month"].map(idx).fillna(1.0).replace(0, 1.0)
    df["dest"] = df["volume_mt"] / df["idx"]
    if "period" in df.columns:
        df = df.sort_values("period")
        tail = df.tail(6)
    else:
        tail = df
    if tail.empty:
        return float("nan")
    trend = float(tail["dest"].median())
    return trend * float(idx.get(month, 1.0))


def _tail_periods(frame: pd.DataFrame, n: int) -> list[str]:
    if frame is None or frame.empty or "period" not in frame.columns:
        return []
    return sorted(frame["period"].astype(str).unique())[-int(n) :]


def _recent_level(
    frame: pd.DataFrame,
    recent_periods: list[str] | None = None,
    longer_periods: list[str] | None = None,
    n_recent: int = 3,
    n_longer: int = 6,
) -> tuple[float, float, float, int]:
    """Full-month Expected from recent run-rate — no calendar-month index.

    Uses the caller's recent window (last three / six periods on the parent
    panel). A distributor with no volume in May–Jul gets 0, not last August.
    Periods the unit actually billed inside that window are averaged the same
    way AMS is — missing months are not zero-filled, but months *outside* the
    window are ignored.
    """
    nan = float("nan")
    if frame is None or frame.empty or "volume_mt" not in frame.columns:
        return nan, nan, nan, 0
    df = frame.copy()
    if "period" in df.columns:
        per = df.groupby("period", as_index=False)["volume_mt"].sum()
        per["period"] = per["period"].astype(str)
    else:
        return nan, nan, nan, 0
    if not recent_periods:
        recent_periods = _tail_periods(per, n_recent)
    if not longer_periods:
        longer_periods = _tail_periods(per, n_longer)
    recent_list = list(recent_periods or [])
    if recent_list:
        recent = per.set_index("period")["volume_mt"].reindex(recent_list).fillna(0.0)
        ams = float(recent.mean())
    else:
        recent = per.iloc[0:0]
        ams = 0.0
    longer = per[per["period"].isin(list(longer_periods or []))]
    n = int(per["period"].nunique())
    if recent_list and float(recent.abs().sum()) <= 1e-12 and longer.empty:
        return 0.0, 0.0, 0.0, n
    if not recent_list:
        return 0.0, 0.0, 0.0, n
    longer_m = float(longer["volume_mt"].median()) if not longer.empty else ams
    expected = _blend(ams, longer_m, int(len(recent_list)))
    return expected, ams, longer_m, n


def _period_volume_and_shops(frame: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """One row per key×period: volume, billed shops, drop size."""
    keys = [k for k in (keys or []) if k]
    cols = keys + ["period", "volume_mt", "billed_shops", "drop_size_mt"]
    if frame is None or frame.empty or "period" not in frame.columns or "volume_mt" not in frame.columns:
        return pd.DataFrame(columns=cols)
    df = frame.copy()
    df["period"] = df["period"].astype(str)
    for k in keys:
        if k not in df.columns:
            df[k] = "(unmapped)"
        df[k] = df[k].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
    group_cols = keys + ["period"]
    vol = df.groupby(group_cols, as_index=False)["volume_mt"].sum()
    if "store_id" in df.columns:
        billed = df.loc[pd.to_numeric(df["volume_mt"], errors="coerce").fillna(0) > 0].copy()
        if billed.empty:
            shops = vol[group_cols].copy()
            shops["billed_shops"] = 0.0
        else:
            billed["store_id"] = billed["store_id"].astype(str)
            shops = (
                billed.groupby(group_cols, as_index=False)["store_id"]
                .nunique()
                .rename(columns={"store_id": "billed_shops"})
            )
    else:
        billed_col = pd.to_numeric(df.get("billed"), errors="coerce") if "billed" in df.columns else None
        if billed_col is not None and billed_col.notna().any():
            tmp = df.copy()
            tmp["_b"] = billed_col.fillna(0)
            shops = tmp.groupby(group_cols, as_index=False)["_b"].sum().rename(columns={"_b": "billed_shops"})
        else:
            shops = vol[group_cols].copy()
            shops["billed_shops"] = np.where(vol["volume_mt"] > 0, 1.0, 0.0)
    out = vol.merge(shops, on=group_cols, how="left")
    out["billed_shops"] = pd.to_numeric(out["billed_shops"], errors="coerce").fillna(0.0)
    out["drop_size_mt"] = np.where(out["billed_shops"] > 1e-9, out["volume_mt"] / out["billed_shops"], np.nan)
    return out


def _expected_drop_size(
    panel: pd.DataFrame,
    recent_periods: list[str] | None = None,
    longer_periods: list[str] | None = None,
) -> float:
    """Expected MT per billed shop: Expected volume ÷ Expected billed shops.

    Same window and blend as Expected sales. Not paced, and not this month's shop count.
    """
    if panel is None or panel.empty or "period" not in panel.columns or "volume_mt" not in panel.columns:
        return float("nan")
    if "billed_shops" not in panel.columns:
        return float("nan")
    vol = panel[["period", "volume_mt"]].copy()
    shops = panel[["period", "billed_shops"]].rename(columns={"billed_shops": "volume_mt"})
    exp_vol, _, _, _ = _recent_level(vol, recent_periods, longer_periods)
    exp_shops, _, _, _ = _recent_level(shops, recent_periods, longer_periods)
    if pd.isna(exp_vol) or pd.isna(exp_shops) or float(exp_shops) <= 1e-9:
        return float("nan")
    return float(exp_vol) / float(exp_shops)


def _blend(typical: float, trend: float, n_same: int) -> float:
    t_ok = pd.notna(typical) and typical > 0
    r_ok = pd.notna(trend) and trend > 0
    if t_ok and r_ok:
        w = n_same / (n_same + 1.5)
        return w * float(typical) + (1 - w) * float(trend)
    if t_ok:
        return float(typical)
    if r_ok:
        return float(trend)
    return float("nan")


def month_index_by_grain(fit: SeasonFit, grain: str) -> dict[str, dict[int, float]]:
    """{grain_id: {month: seasonal_index}} from a fitted season table."""
    out: dict[str, dict[int, float]] = {}
    if fit is None or fit.table is None or fit.table.empty:
        return out
    part = fit.table[fit.table["grain"].astype(str) == str(grain)]
    for _, r in part.iterrows():
        gid = str(r.get("grain_id") or "")
        try:
            m = int(r.get("month") or 0)
        except (TypeError, ValueError):
            continue
        if not gid or m < 1:
            continue
        out.setdefault(gid, {})[m] = _finite(r.get("seasonal_index"), 1.0)
    return out


def expected_for_keys(
    shop_month: pd.DataFrame,
    period: str,
    keys: list[str],
    parent_index: dict[str, dict[int, float]] | None = None,
    national_index: dict[int, float] | None = None,
    parent_key: str | None = None,
    shrink_k: float = 4.0,
) -> pd.DataFrame:
    """Recent run-rate Expected per key tuple. No calendar-month seasonal index.

    parent_index / national_index are ignored (kept so callers do not change).
    """
    del parent_index, national_index, parent_key, shrink_k
    cols = list(keys) + [
        "expected_full_mt",
        "typical_mt",
        "trend_mt",
        "seasonal_index",
        "n_same_month",
        "n_periods",
        "credibility",
        "expected_drop_size_mt",
    ]
    empty = pd.DataFrame(columns=cols)
    if shop_month is None or shop_month.empty or not period or not keys:
        return empty
    sm = shop_month.copy()
    if "month" not in sm.columns:
        sm["month"] = sm["period"].astype(str).str.slice(5, 7).astype(int)
    month = int(str(period)[5:7]) if len(str(period)) >= 7 else 0
    for k in keys:
        if k not in sm.columns:
            sm[k] = "(unmapped)"
        sm[k] = sm[k].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
    hist = sm[sm["period"].astype(str) != str(period)]
    if hist.empty:
        return empty
    grouped = _period_volume_and_shops(hist, keys)
    if grouped.empty:
        return empty
    grouped["month"] = grouped["period"].astype(str).str.slice(5, 7).astype(int)
    recent_ps = prior_periods(period, 3)
    longer_ps = prior_periods(period, 6)
    rows = []
    for key_vals, g in grouped.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = {k: key_vals[i] for i, k in enumerate(keys)}
        n_g = int(g["period"].nunique())
        n_s = int((g["month"] == month).sum())
        expected, ams, trend, _ = _recent_level(g, recent_ps, longer_ps)
        drop_e = _expected_drop_size(g, recent_ps, longer_ps)
        rec.update(
            {
                "expected_full_mt": float(expected) if pd.notna(expected) else 0.0,
                "typical_mt": ams if pd.notna(ams) else None,
                "trend_mt": trend if pd.notna(trend) else None,
                "seasonal_index": 1.0,
                "n_same_month": n_s,
                "n_periods": n_g,
                "credibility": n_g / (n_g + 4.0) if n_g else 0.0,
                "expected_drop_size_mt": drop_e if pd.notna(drop_e) else None,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows) if rows else empty


def apply_child_expected(
    children: pd.DataFrame,
    shop_month: pd.DataFrame,
    period: str,
    keys: list[str],
    fit: SeasonFit,
    intra_frac: float,
    parent_key: str = "city",
    shrink_k: float = 4.0,
) -> pd.DataFrame:
    """Own-history Expected at this grain from recent run-rate, then paced."""
    if children is None or children.empty:
        return children
    out = children.copy()
    parent_idx = month_index_by_grain(fit, "city") if parent_key == "city" else {}
    learned = expected_for_keys(
        shop_month,
        period,
        keys,
        parent_index=parent_idx,
        national_index=fit.national_index if fit else None,
        parent_key=parent_key,
        shrink_k=shrink_k,
    )
    frac = float(intra_frac or 1.0)
    if learned is None or learned.empty:
        ly = pd.to_numeric(out.get("ly_mt"), errors="coerce").fillna(0.0)
        out["seasonal_typical_mt"] = ly
        out["expected_mt"] = ly * frac
        out["gap_mt"] = pd.to_numeric(out.get("volume_mt"), errors="coerce").fillna(0.0) - out["expected_mt"]
        out["expected_drop_size_mt"] = np.nan
        return out
    keep = keys + ["expected_full_mt", "seasonal_index", "typical_mt", "credibility", "expected_drop_size_mt"]
    learned = learned[[c for c in keep if c in learned.columns]].copy()
    for k in keys:
        if k in out.columns:
            out[k] = out[k].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
        if k in learned.columns:
            learned[k] = learned[k].astype(str)
    drop_learned = [
        c
        for c in ["expected_full_mt", "seasonal_index", "typical_mt", "credibility", "expected_drop_size_mt"]
        if c in out.columns
    ]
    if drop_learned:
        out = out.drop(columns=drop_learned)
    out = out.merge(learned, on=keys, how="left")
    full = pd.to_numeric(out.get("expected_full_mt"), errors="coerce")
    ly = pd.to_numeric(out.get("ly_mt"), errors="coerce").fillna(0.0)
    full = full.where(full.notna(), ly)
    out["seasonal_typical_mt"] = full.fillna(0.0)
    out["expected_mt"] = out["seasonal_typical_mt"] * frac
    out["gap_mt"] = pd.to_numeric(out["volume_mt"], errors="coerce").fillna(0) - out["expected_mt"]
    out["gap_pct"] = np.where(
        out["expected_mt"] > 1e-9,
        out["gap_mt"] / out["expected_mt"] * 100,
        np.nan,
    )
    return out


def reconcile_expected(
    children: pd.DataFrame,
    parents: pd.DataFrame,
    parent_col: str = "parent_id",
    parent_id_col: str = "grain_id",
    exp_col: str = "expected_mt",
    intra_frac: float = 1.0,
) -> pd.DataFrame:
    """Scale children's own Expected so they add to the parent's Expected.

    Forecast-based proportions (independent estimates, then add up), not last-year
    mix × what the parent billed now.
    """
    if children is None or children.empty or parents is None or parents.empty:
        return children
    if parent_col not in children.columns or exp_col not in children.columns:
        return children
    targets = {}
    for _, r in parents.iterrows():
        targets[str(r.get(parent_id_col) or "")] = float(r.get(exp_col) or 0.0)
    parts = []
    frac = float(intra_frac or 1.0) or 1.0
    for pid, g in children.groupby(parent_col, dropna=False):
        g = g.copy()
        target = targets.get(str(pid))
        series = pd.to_numeric(g[exp_col], errors="coerce").fillna(0.0)
        total = float(series.sum())
        if target is not None and target > 1e-9 and total > 1e-9:
            g[exp_col] = series * (target / total)
            g["seasonal_typical_mt"] = g[exp_col] / frac
            if "volume_mt" in g.columns:
                vol = pd.to_numeric(g["volume_mt"], errors="coerce").fillna(0.0)
                g["gap_mt"] = vol - g[exp_col]
                g["gap_pct"] = np.where(g[exp_col] > 1e-9, g["gap_mt"] / g[exp_col] * 100, np.nan)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else children


def fit_shop_expected(
    shop_month: pd.DataFrame,
    period: str,
    fit: SeasonFit,
    intra_frac: float,
) -> pd.DataFrame:
    """Per-shop Expected: last-3-month AMS blended with last-6-month median, shrunk toward the city.

    Sparse doors borrow the city's Expected via AMS mix. Never-billed whitespace stays 0.
    No calendar-month seasonal index.
    """
    empty = pd.DataFrame(columns=["store_id", "city", "expected_full_mt", "expected_mt", "credibility"])
    if shop_month is None or shop_month.empty or not period:
        return empty
    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str)
    if "city" not in sm.columns:
        sm["city"] = "(unmapped)"
    sm["city"] = sm["city"].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
    hist = sm[sm["period"].astype(str) != str(period)]
    if hist.empty:
        return empty
    city_full: dict[str, float] = {}
    if fit is not None and fit.city_expected is not None and not fit.city_expected.empty:
        for _, r in fit.city_expected.iterrows():
            city_full[str(r["city"])] = float(r.get("expected_full_mt") or 0.0)

    hist = hist.copy()
    recent_list = prior_periods(period, 3)
    recent_ps = set(recent_list)
    longer_ps = set(prior_periods(period, 6))
    n_per = hist.groupby("store_id")["period"].nunique().rename("n_periods")
    in_recent = hist[hist["period"].astype(str).isin(recent_ps)]
    in_longer = hist[hist["period"].astype(str).isin(longer_ps)]
    if in_recent.empty:
        ams = pd.DataFrame(columns=["store_id", "ams"])
        n_recent = pd.DataFrame(columns=["store_id", "n_recent"])
    else:
        pt = in_recent.pivot_table(index="store_id", columns="period", values="volume_mt", aggfunc="sum")
        pt = pt.reindex(columns=recent_list, fill_value=0).fillna(0.0)
        ams = pt.mean(axis=1).rename("ams").reset_index()
        n_recent = (pt.reindex(columns=recent_list).fillna(0.0) > 1e-12).sum(axis=1).rename("n_recent").reset_index()
    longer = in_longer.groupby("store_id")["volume_mt"].median().rename("longer_mt")
    cities = sm.groupby("store_id")["city"].last()
    from sndintel.io_utils import shift_period

    yoy = shift_period(period, -12)
    ly_s = (
        sm[sm["period"].astype(str) == str(yoy)].groupby("store_id")["volume_mt"].sum().rename("ly_mt")
        if yoy
        else pd.Series(dtype=float, name="ly_mt")
    )
    ids = pd.Index(sorted(set(hist["store_id"].astype(str))))
    out = pd.DataFrame({"store_id": ids})
    out = out.merge(ams, on="store_id", how="left")
    out = out.merge(n_recent, on="store_id", how="left")
    out = out.merge(n_per.reset_index(), on="store_id", how="left")
    out = out.merge(longer.reset_index(), on="store_id", how="left")
    if ly_s is not None and not ly_s.empty:
        out = out.merge(ly_s.reset_index(), on="store_id", how="left")
    else:
        out["ly_mt"] = np.nan
    out["city"] = out["store_id"].map(cities).fillna("(unmapped)")
    out["n_recent"] = pd.to_numeric(out.get("n_recent"), errors="coerce").fillna(0)
    out["n_periods"] = pd.to_numeric(out.get("n_periods"), errors="coerce").fillna(0)
    out["local_mt"] = [
        _blend(a, lng, int(n))
        for a, lng, n in zip(
            pd.to_numeric(out.get("ams"), errors="coerce"),
            pd.to_numeric(out.get("longer_mt"), errors="coerce"),
            out["n_recent"].fillna(0),
        )
    ]
    city_ly = out.groupby("city")["ly_mt"].transform("sum")
    city_ams = out.groupby("city")["ams"].transform("sum")
    prior = []
    for i in range(len(out)):
        r = out.iloc[i]
        cexp = float(city_full.get(str(r["city"]), 0.0) or 0.0)
        shop_ams = float(r["ams"]) if pd.notna(r.get("ams")) else 0.0
        c_ams = float(city_ams.iloc[i]) if pd.notna(city_ams.iloc[i]) else 0.0
        shop_ly = float(r["ly_mt"]) if pd.notna(r.get("ly_mt")) else 0.0
        c_ly = float(city_ly.iloc[i]) if pd.notna(city_ly.iloc[i]) else 0.0
        if cexp > 1e-9 and c_ams > 1e-9 and shop_ams > 1e-9:
            prior.append(cexp * (shop_ams / c_ams))
        elif cexp > 1e-9 and shop_ly > 1e-9 and c_ly > 1e-9:
            prior.append(cexp * (shop_ly / c_ly))
        else:
            prior.append(0.0)
    out["prior_mt"] = prior
    cred = out["n_periods"] / (out["n_periods"] + 6.0)
    out["credibility"] = cred
    local_s = pd.to_numeric(out["local_mt"], errors="coerce")
    prior_s = pd.to_numeric(out["prior_mt"], errors="coerce").fillna(0.0)
    blended = cred * local_s.fillna(prior_s) + (1 - cred) * prior_s
    blended = blended.where(local_s.notna() | (prior_s > 0), 0.0)
    out["expected_full_mt"] = blended.fillna(0.0).clip(lower=0.0)
    frac = float(intra_frac or 1.0)
    out["expected_mt"] = out["expected_full_mt"] * frac
    return out[["store_id", "city", "expected_full_mt", "expected_mt", "credibility"]]

def apply_city_expected(city_units: pd.DataFrame, fit: SeasonFit, intra_frac: float) -> pd.DataFrame:
    """Replace last-year×pace expected with recent run-rate Expected, then pace."""
    if city_units is None or city_units.empty:
        return city_units
    out = city_units.copy()
    lookup = {}
    drop_lookup: dict[str, float] = {}
    if fit.city_expected is not None and not fit.city_expected.empty:
        lookup = {str(k): v for k, v in fit.city_expected.set_index("city")["expected_full_mt"].to_dict().items()}
        if "expected_drop_size_mt" in fit.city_expected.columns:
            drop_lookup = {
                str(k): v
                for k, v in fit.city_expected.set_index("city")["expected_drop_size_mt"].to_dict().items()
            }
    full = []
    idx = []
    drops = []
    for _, r in out.iterrows():
        city = str(r.get("grain_id") or r.get("city") or "")
        # 0 is a valid Expected (no volume in the AMS window). Only a missing
        # city falls back to last year — never treat a zero run-rate as LY.
        if city in lookup:
            try:
                exp = float(lookup[city])
            except (TypeError, ValueError):
                exp = float("nan")
            if pd.isna(exp):
                exp = float(r.get("ly_mt") or 0)
        else:
            exp = float(r.get("ly_mt") or 0)
        full.append(float(exp) if pd.notna(exp) else 0.0)
        raw_drop = drop_lookup.get(city)
        try:
            drops.append(float(raw_drop) if raw_drop is not None and pd.notna(raw_drop) else float("nan"))
        except (TypeError, ValueError):
            drops.append(float("nan"))
        si = fit.national_index.get(fit.month, 1.0)
        if fit.city_expected is not None and not fit.city_expected.empty:
            hit = fit.city_expected[fit.city_expected["city"] == city]
            if not hit.empty:
                si = float(hit.iloc[0]["seasonal_index"] or si)
        idx.append(si)
    out["seasonal_typical_mt"] = full
    out["seasonal_index"] = idx
    out["expected_drop_size_mt"] = drops
    frac = float(intra_frac or 1.0)
    out["expected_mt"] = out["seasonal_typical_mt"] * frac
    out["gap_mt"] = out["volume_mt"].fillna(0) - out["expected_mt"]
    out["gap_pct"] = np.where(
        out["expected_mt"] > 1e-9,
        (out["volume_mt"] - out["expected_mt"]) / out["expected_mt"] * 100,
        np.nan,
    )
    return out


def scale_children_expected(children: pd.DataFrame, parents: pd.DataFrame, intra_frac: float) -> pd.DataFrame:
    """Children inherit the parent's learned seasonal scale vs last year."""
    if children is None or children.empty or parents is None or parents.empty:
        return children
    out = children.copy()
    scales = {}
    for _, r in parents.iterrows():
        ly = float(r.get("ly_mt") or 0)
        full = float(r.get("seasonal_typical_mt") or r.get("expected_mt") or 0)
        if ly > 1e-9:
            scales[str(r["grain_id"])] = full / ly
        else:
            scales[str(r["grain_id"])] = 1.0
    scale_s = out["parent_id"].astype(str).map(scales).fillna(1.0)
    out["seasonal_typical_mt"] = out["ly_mt"].fillna(0) * scale_s
    idxs = {}
    for _, r in parents.iterrows():
        idxs[str(r["grain_id"])] = r.get("seasonal_index")
    if "parent_id" in out.columns:
        out["seasonal_index"] = out["parent_id"].astype(str).map(idxs)
    frac = float(intra_frac or 1.0)
    out["expected_mt"] = out["seasonal_typical_mt"] * frac
    out["gap_mt"] = out["volume_mt"].fillna(0) - out["expected_mt"]
    out["gap_pct"] = np.where(
        out["expected_mt"] > 1e-9,
        (out["volume_mt"] - out["expected_mt"]) / out["expected_mt"] * 100,
        np.nan,
    )
    return out


def elapsed_month_frac(as_of_day: int | float, days_in_month: int | float) -> float:
    """Straight calendar share: day 20 of 31 is 20/31."""
    as_of = float(as_of_day)
    days = float(days_in_month)
    if days <= 0:
        return 1.0
    return min(max(as_of / days, 0.02), 1.0)


def intra_month_fraction(
    as_of_day: int | float | None,
    days_in_month: int | float | None,
    observations: pd.DataFrame | None = None,
    open_mtd: bool = False,
    shop_day: pd.DataFrame | None = None,
    period: str | None = None,
) -> tuple[float, str]:
    """Fraction of a full month billed by as_of_day.

    Prefer the national Outlet Date Wise curve (one country shape, applied
    everywhere). Then mid-month MTD snapshots. Then elapsed calendar days.
    """
    if not open_mtd or not as_of_day or not days_in_month:
        return 1.0, "closed"
    as_of = int(as_of_day)
    days = int(days_in_month)
    nat = national_day_frac(shop_day, period, as_of, days)
    if nat is not None:
        return nat, "national_day_curve"
    emp = _empirical_mtd_frac(observations, as_of, days)
    if emp is not None:
        return emp, "learned_mtd_cuts"
    return elapsed_month_frac(as_of, days), "elapsed_days"


def national_day_frac(
    shop_day: pd.DataFrame | None,
    period: str | None,
    as_of_day: int,
    days_in_month: int,
) -> float | None:
    """Median share of a closed month the country had billed by this calendar day."""
    curve = fit_national_day_curve(shop_day, period)
    if not curve:
        return None
    day = int(min(max(as_of_day, 1), max(days_in_month, 1)))
    return float(curve[day - 1])


def fit_national_day_curve(shop_day: pd.DataFrame | None, period: str | None) -> list[float] | None:
    """31-day cumulative billed share for the country. Median across closed months."""
    daily = _national_daily(shop_day, period)
    if daily.empty:
        return None
    months = []
    for per, g in daily.groupby("period"):
        days_used = int(g["day"].nunique())
        if days_used < NATIONAL_DAY_MIN_DAYS:
            continue
        year, month = int(str(per)[:4]), int(str(per)[5:7])
        last = monthrange(year, month)[1]
        by_day = g.groupby(g["day"].astype(int))["volume_mt"].sum()
        vols = np.array([float(by_day.get(int(d), 0.0) or 0.0) for d in range(1, last + 1)], dtype=float)
        total = float(vols.sum())
        if total <= 0:
            continue
        cum = np.cumsum(vols) / total
        # Stretch to 31: days after this month’s last day stay at 1.0.
        padded = np.ones(31, dtype=float)
        padded[:last] = np.clip(cum, 0.0, 1.0)
        months.append(padded)
    if len(months) < NATIONAL_DAY_MIN_MONTHS:
        return None
    stacked = np.vstack(months)
    med = np.median(stacked, axis=0)
    med = np.maximum.accumulate(np.clip(med, 0.02, 1.0))
    med[-1] = 1.0
    return [float(x) for x in med]


def _national_daily(shop_day: pd.DataFrame | None, period: str | None) -> pd.DataFrame:
    if shop_day is None or shop_day.empty or not period:
        return pd.DataFrame()
    out = shop_day.copy()
    out["volume_mt"] = pd.to_numeric(out.get("volume_mt"), errors="coerce").fillna(0.0)
    out["period"] = out["period"].astype(str)
    out = out[out["period"] < str(period)]
    if "day" not in out.columns or out["day"].isna().all():
        sale = pd.to_datetime(out.get("sale_date"), errors="coerce")
        out["day"] = sale.dt.day
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out = out[out["day"].notna() & (out["day"] >= 1) & (out["day"] <= 31) & (out["volume_mt"] > 0)]
    return out[["period", "day", "volume_mt"]] if not out.empty else pd.DataFrame()


def _empirical_mtd_frac(obs: pd.DataFrame | None, as_of: int, days: int) -> float | None:
    """Need ≥2 distinct as-of days in a month *and* a close. Month-end-only rows cannot teach day 20."""
    if obs is None or obs.empty or "period" not in obs.columns:
        return None
    need = {"as_of_day", "volume_mt"}
    if not need.issubset(obs.columns):
        return None
    fracs = []
    for _, g in obs.groupby("period"):
        g = g.dropna(subset=["as_of_day", "volume_mt"]).sort_values("as_of_day")
        if g.empty or g["as_of_day"].nunique() < 2:
            continue
        closed = g[pd.to_numeric(g["as_of_day"], errors="coerce") >= days]
        if closed.empty:
            continue
        final = float(closed["volume_mt"].iloc[-1])
        if final <= 0:
            continue
        days_s = pd.to_numeric(g["as_of_day"], errors="coerce")
        vols = pd.to_numeric(g["volume_mt"], errors="coerce")
        mask = days_s.notna() & vols.notna()
        if mask.sum() < 2:
            continue
        xs = days_s[mask].to_numpy(dtype=float)
        ys = vols[mask].to_numpy(dtype=float)
        if as_of < xs.min() or as_of > xs.max():
            # do not clamp to the month-end total
            if as_of < xs.min():
                v = ys[0] * (as_of / xs.min()) if xs.min() else 0.0
            else:
                continue
        else:
            v = float(np.interp(as_of, xs, ys))
        fracs.append(min(max(v / final, 0.01), 0.99))
    if len(fracs) < 1:
        return None
    return float(np.median(fracs))
