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
