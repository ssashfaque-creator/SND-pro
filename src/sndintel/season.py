"""Learn seasonality from the warehouse panel — not a curve we shipped.

Nineteen months of shop-month totals are enough to estimate:

* A 12-month seasonal index (August vs January) nationally and by city
* A typical same-month level (mean of every August on file, not only last year)
* A destationalized recent trend × this month's index

They are **not** enough to estimate a day-of-month loading curve. A closed
August row is one number (the month total). Day 20's share of August only
appears if mid-month MTD cuts were ingested. Those we learn when present;
otherwise open MTD uses elapsed calendar days of the learned typical August.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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
    typical_nat = float(nat.loc[nat["month"] == month, "volume_mt"].mean()) if n_same else float("nan")
    trend_nat = _trend_seasonal(nat, nat_idx, month)
    expected_nat = _blend(typical_nat, trend_nat, n_same)

    city_period = hist.groupby(["city", "period"], as_index=False)["volume_mt"].sum()
    city_period["month"] = city_period["period"].astype(str).str.slice(5, 7).astype(int)
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
        typical = float(same["volume_mt"].mean()) if not same.empty else float("nan")
        n_s = int(len(same))
        trend = _trend_seasonal(g, mixed, month)
        expected = _blend(typical, trend, n_s)
        city_exp_rows.append(
            {
                "city": city,
                "expected_full_mt": expected,
                "typical_mt": typical if pd.notna(typical) else None,
                "trend_mt": trend if pd.notna(trend) else None,
                "n_same_month": n_s,
                "seasonal_index": _finite(mixed.get(month, 1.0)),
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
    """Learned typical same-month + destationalized trend, per key tuple.

    Seasonal indexes shrink toward the parent (city → national, dist/DSR → city).
    Same formula as the national/city Expected the pack already prints.
    """
    cols = list(keys) + [
        "expected_full_mt",
        "typical_mt",
        "trend_mt",
        "seasonal_index",
        "n_same_month",
        "n_periods",
        "credibility",
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
    nat_idx = national_index or {m: 1.0 for m in range(1, 13)}
    parent_index = parent_index or {}
    parent_key = parent_key or (keys[0] if keys else None)
    grouped = hist.groupby(keys + ["period"], as_index=False)["volume_mt"].sum()
    grouped["month"] = grouped["period"].astype(str).str.slice(5, 7).astype(int)
    rows = []
    for key_vals, g in grouped.groupby(keys, dropna=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        rec = {k: key_vals[i] for i, k in enumerate(keys)}
        raw = _iterative_month_index(g)
        n_g = int(g["period"].nunique())
        cred = n_g / (n_g + shrink_k)
        parent_id = str(rec.get(parent_key) or "") if parent_key else ""
        pidx = parent_index.get(parent_id) or nat_idx
        mixed: dict[int, float] = {}
        for m in range(1, 13):
            local = raw.get(m)
            prior = _finite((pidx or {}).get(m, nat_idx.get(m, 1.0)))
            if local is None or not np.isfinite(_finite(local, default=float("nan"))):
                mixed[m] = prior
            else:
                mixed[m] = _finite(cred * float(local) + (1 - cred) * prior)
        same = g[g["month"] == month]
        typical = float(same["volume_mt"].mean()) if not same.empty else float("nan")
        n_s = int(len(same))
        trend = _trend_seasonal(g, mixed, month)
        expected = _blend(typical, trend, n_s)
        rec.update(
            {
                "expected_full_mt": float(expected) if pd.notna(expected) else 0.0,
                "typical_mt": typical if pd.notna(typical) else None,
                "trend_mt": trend if pd.notna(trend) else None,
                "seasonal_index": _finite(mixed.get(month, 1.0)),
                "n_same_month": n_s,
                "n_periods": n_g,
                "credibility": _finite(cred, 0.0),
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
    """Own-history Expected at this grain, shrunk toward the parent's month index."""
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
        return out
    keep = keys + ["expected_full_mt", "seasonal_index", "typical_mt", "credibility"]
    learned = learned[[c for c in keep if c in learned.columns]].copy()
    for k in keys:
        if k in out.columns:
            out[k] = out[k].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
        if k in learned.columns:
            learned[k] = learned[k].astype(str)
    drop_learned = [c for c in ["expected_full_mt", "seasonal_index", "typical_mt", "credibility"] if c in out.columns]
    if drop_learned:
        out = out.drop(columns=drop_learned)
    out = out.merge(learned, on=keys, how="left")
    full = pd.to_numeric(out.get("expected_full_mt"), errors="coerce")
    ly = pd.to_numeric(out.get("ly_mt"), errors="coerce").fillna(0.0)
    full = full.where(full.notna() & (full > 0), ly)
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
    """Per-shop Expected: same-month typical + destationalized trend, shrunk toward the city.

    Sparse doors borrow the city's learned August. Never-billed whitespace stays 0.
    """
    empty = pd.DataFrame(columns=["store_id", "city", "expected_full_mt", "expected_mt", "credibility"])
    if shop_month is None or shop_month.empty or not period:
        return empty
    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str)
    if "month" not in sm.columns:
        sm["month"] = sm["period"].astype(str).str.slice(5, 7).astype(int)
    if "city" not in sm.columns:
        sm["city"] = "(unmapped)"
    sm["city"] = sm["city"].fillna("(unmapped)").replace("", "(unmapped)").astype(str)
    month = int(str(period)[5:7]) if len(str(period)) >= 7 else 0
    hist = sm[sm["period"].astype(str) != str(period)]
    if hist.empty:
        return empty
    city_full: dict[str, float] = {}
    city_si: dict[str, float] = {}
    if fit is not None and fit.city_expected is not None and not fit.city_expected.empty:
        for _, r in fit.city_expected.iterrows():
            city_full[str(r["city"])] = float(r.get("expected_full_mt") or 0.0)
            city_si[str(r["city"])] = _finite(r.get("seasonal_index"), 1.0)
    nat_idx = (fit.national_index if fit else None) or {m: 1.0 for m in range(1, 13)}
    city_month = month_index_by_grain(fit, "city") if fit else {}

    def _si(city: str, m: int) -> float:
        local = (city_month.get(str(city)) or {}).get(int(m))
        if local is not None:
            return _finite(local)
        return _finite(nat_idx.get(int(m), 1.0))

    hist = hist.copy()
    hist["si"] = [_si(c, m) for c, m in zip(hist["city"], hist["month"])]
    hist["si"] = hist["si"].replace(0, 1.0)
    hist["dest"] = pd.to_numeric(hist["volume_mt"], errors="coerce") / hist["si"]
    typical = hist[hist["month"] == month].groupby("store_id")["volume_mt"].mean().rename("typical_mt")
    n_same = hist[hist["month"] == month].groupby("store_id").size().rename("n_same")
    n_per = hist.groupby("store_id")["period"].nunique().rename("n_periods")
    hist["prank"] = hist.groupby("store_id")["period"].rank(method="first", ascending=False)
    tail = hist[hist["prank"] <= 6]
    trend_dest = tail.groupby("store_id")["dest"].median().rename("trend_dest")
    ams = hist.sort_values("period").groupby("store_id").tail(3).groupby("store_id")["volume_mt"].mean().rename("ams")
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
    out = out.merge(typical.reset_index(), on="store_id", how="left")
    out = out.merge(n_same.reset_index(), on="store_id", how="left")
    out = out.merge(n_per.reset_index(), on="store_id", how="left")
    out = out.merge(trend_dest.reset_index(), on="store_id", how="left")
    out = out.merge(ams.reset_index(), on="store_id", how="left")
    if ly_s is not None and not ly_s.empty:
        out = out.merge(ly_s.reset_index(), on="store_id", how="left")
    else:
        out["ly_mt"] = np.nan
    out["city"] = out["store_id"].map(cities).fillna("(unmapped)")
    out["n_same"] = pd.to_numeric(out.get("n_same"), errors="coerce").fillna(0)
    out["n_periods"] = pd.to_numeric(out.get("n_periods"), errors="coerce").fillna(0)
    this_si = out["city"].map(lambda c: city_si.get(str(c), _finite(nat_idx.get(month, 1.0))))
    out["trend_mt"] = pd.to_numeric(out.get("trend_dest"), errors="coerce") * this_si
    out["local_mt"] = [
        _blend(t, tr, int(n))
        for t, tr, n in zip(
            pd.to_numeric(out.get("typical_mt"), errors="coerce"),
            pd.to_numeric(out.get("trend_mt"), errors="coerce"),
            out["n_same"].fillna(0),
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
    """Replace last-year×pace expected with the warehouse-learned seasonal expected."""
    if city_units is None or city_units.empty:
        return city_units
    out = city_units.copy()
    lookup = {}
    if fit.city_expected is not None and not fit.city_expected.empty:
        lookup = fit.city_expected.set_index("city")["expected_full_mt"].to_dict()
    full = []
    idx = []
    for _, r in out.iterrows():
        city = str(r.get("grain_id") or r.get("city") or "")
        exp = lookup.get(city)
        if exp is None or (isinstance(exp, float) and (pd.isna(exp) or exp <= 0)):
            exp = float(r.get("ly_mt") or 0)
        full.append(float(exp or 0.0))
        si = fit.national_index.get(fit.month, 1.0)
        if fit.city_expected is not None and not fit.city_expected.empty:
            hit = fit.city_expected[fit.city_expected["city"] == city]
            if not hit.empty:
                si = float(hit.iloc[0]["seasonal_index"] or si)
        idx.append(si)
    out["seasonal_typical_mt"] = full
    out["seasonal_index"] = idx
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


def intra_month_fraction(
    as_of_day: int | float | None,
    days_in_month: int | float | None,
    observations: pd.DataFrame | None = None,
    open_mtd: bool = False,
) -> tuple[float, str]:
    """Fraction of a full month billed by as_of_day — learned, never a shipped knot curve."""
    if not open_mtd or not as_of_day or not days_in_month:
        return 1.0, "closed"
    as_of = int(as_of_day)
    days = int(days_in_month)
    emp = _empirical_mtd_frac(observations, as_of, days)
    if emp is not None:
        return emp, "learned_mtd_cuts"
    # No mid-month history: elapsed calendar days. Not a loading curve we invented.
    return min(max(as_of / days, 0.02), 1.0), "elapsed_days"


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
