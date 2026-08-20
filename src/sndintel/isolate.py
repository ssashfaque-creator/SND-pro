"""Exception-based isolation: parent-adjusted residuals, seasonality, drivers.

Excel compares each unit to last year. That flags every city when the country
is down. Enterprise S&D tools (Nielsen Business Drivers, IRI volume decomp,
shift-share, exception-based selling) ask a different question:

    After the parent moved, is this unit still a problem?

* National declining → cities that declined *more* than national are the problem.
* National growing → cities whose growth is *slower* than national are the problem.
* A city that moved with the market is weather, not a local fire.

Residuals are additive (they sum to zero at each parent), shrunk with empirical
Bayes so a 0.02 MT shop cannot outrank Eva Foods on a percentage, and scored
with a robust z so we only brief statistically unusual units.

Seasonality:

* Calendar-month shape is **learned from every month in the warehouse** (typical
  August, city indices shrunk toward national) — not last year alone, and not a
  curve shipped in the code.
* Intra-month day shape is learned from mid-month MTD cuts when they exist.
  Month-end totals cannot teach day 20. If those cuts are missing, open MTD
  uses elapsed calendar days of the *learned* typical month.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sndintel.io_utils import shift_period
from sndintel.season import intra_month_fraction  # re-export for callers / tests


def parent_index(parent_now: float, parent_ly: float, floor: float = 1e-6) -> float:
    if parent_ly is None or parent_ly <= floor:
        return 1.0
    return float(parent_now) / float(parent_ly)


def share_expected(unit_ly: float, index: float) -> float:
    return float(unit_ly or 0.0) * float(index)


def competitive_mt(unit_now: float, unit_ly: float, parent_now: float, parent_ly: float) -> float:
    """Volume this unit billed minus the fair share of the parent's current book.

    Fair share = last-year volume mix × parent's current volume.
    Sums to ~0 across siblings.
    """
    return float(unit_now or 0.0) - share_expected(unit_ly, parent_index(parent_now, parent_ly))


def empirical_bayes(residual: float, ly: float, k: float) -> float:
    """Shrink a residual toward 0 when last-year volume is small.

    k is a prior equivalent sample (median shop LY works). Credibility = ly/(ly+k).
    """
    ly = max(0.0, float(ly or 0.0))
    k = max(1e-9, float(k))
    return float(residual) * (ly / (ly + k))


def robust_z(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").fillna(0.0)
    med = float(s.median()) if len(s) else 0.0
    mad = float((s - med).abs().median()) if len(s) else 0.0
    if mad < 1e-9:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def coverage_velocity(
    n_now: float, n_ly: float, vol_now: float, vol_ly: float
) -> tuple[float, float, float, float, float]:
    """CPG identity: ΔV = ΔN × drop_ly + N_ly × Δdrop + ΔN × Δdrop.

    coverage (numeric distribution) vs velocity (drop size). Interaction is the rest.
    """
    n_now = float(n_now or 0.0)
    n_ly = float(n_ly or 0.0)
    vol_now = float(vol_now or 0.0)
    vol_ly = float(vol_ly or 0.0)
    drop_ly = (vol_ly / n_ly) if n_ly else 0.0
    drop_now = (vol_now / n_now) if n_now else 0.0
    coverage = (n_now - n_ly) * drop_ly
    velocity = n_ly * (drop_now - drop_ly)
    interaction = (n_now - n_ly) * (drop_now - drop_ly)
    return coverage, velocity, interaction, drop_now, drop_ly


def weighted_distribution(lost_mt: float, ly_mt: float) -> float | None:
    """Share of last-year volume still billed (WD). 1 - lost/ly."""
    ly_mt = float(ly_mt or 0.0)
    if ly_mt <= 1e-9:
        return None
    return 1.0 - max(0.0, float(lost_mt or 0.0)) / ly_mt


def apply_shift_share(
    df: pd.DataFrame,
    parent_now: float,
    parent_ly: float,
    k: float,
    z_clip: float = 4.0,
) -> pd.DataFrame:
    """Attach fair-share expected, competitive residual, shrinkage, robust z."""
    out = df.copy()
    idx = parent_index(parent_now, parent_ly)
    out["parent_index"] = idx
    out["share_expected_mt"] = out["ly_mt"].fillna(0) * idx
    out["competitive_mt"] = out["volume_mt"].fillna(0) - out["share_expected_mt"]
    out["isolated_mt"] = [
        empirical_bayes(c, ly, k)
        for c, ly in zip(out["competitive_mt"], out["ly_mt"].fillna(0))
    ]
    out["z_score"] = robust_z(pd.Series(out["isolated_mt"], index=out.index)).clip(-z_clip, z_clip)
    out["focus_score"] = -pd.Series(out["isolated_mt"]) * (1.0 + out["z_score"].abs() / 4.0)
    out["situation"] = out.apply(_situation_label, axis=1)
    return out


def _situation_label(r: pd.Series) -> str:
    iso = float(r.get("isolated_mt") or 0)
    ly = float(r.get("ly_mt") or 0)
    # Additive hole vs a percentage: 0.5 MT (or 2% of own LY) is a city-scale exception.
    material = max(0.5, 0.02 * ly) if ly >= 8 else max(0.15, 0.05 * max(ly, 0.0))
    if iso <= -material:
        return "lagging"
    if iso >= material:
        return "outperforming"
    return "with_market"


def apply_coverage_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cov, vel, inter, wd, nd = [], [], [], [], []
    for _, r in out.iterrows():
        c, v, i, _dn, _dl = coverage_velocity(
            r.get("billed") or 0,
            r.get("billed_ly") or 0,
            r.get("volume_mt") or 0,
            r.get("ly_mt") or 0,
        )
        cov.append(c)
        vel.append(v)
        inter.append(i)
        wd.append(weighted_distribution(r.get("lost_mt") or 0, r.get("ly_mt") or 0))
        uni = r.get("universe")
        billed = r.get("billed") or 0
        nd.append((billed / uni) if uni else None)
    out["coverage_effect_mt"] = cov
    out["velocity_effect_mt"] = vel
    out["interaction_effect_mt"] = inter
    out["wd"] = wd
    out["nd"] = nd
    return out


def seasonal_mom_expected(shop_month: pd.DataFrame, period: str, keys: list[str]) -> pd.DataFrame:
    """Expected this month from last year's same month-to-month shape.

    Aug expected = Jul_this_year × (Aug_last_year / Jul_last_year).
    Controls within-year seasonality that a raw MoM misses.
    """
    prev = shift_period(period, -1)
    yoy = shift_period(period, -12)
    yoy_prev = shift_period(prev, -12)
    cols = keys + ["volume_mt"]
    if shop_month is None or shop_month.empty:
        return pd.DataFrame(columns=keys + ["seasonal_mom_index", "mom_expected_mt"])

    def _g(p: str) -> pd.DataFrame:
        part = shop_month[shop_month["period"] == p]
        if part.empty:
            return pd.DataFrame(columns=keys + ["volume_mt"])
        return part.groupby(keys, dropna=False, as_index=False)["volume_mt"].sum()

    now_prev = _g(prev).rename(columns={"volume_mt": "prev_mt"})
    ly_now = _g(yoy).rename(columns={"volume_mt": "ly_mt"})
    ly_prev = _g(yoy_prev).rename(columns={"volume_mt": "ly_prev_mt"})
    m = now_prev.merge(ly_now, on=keys, how="outer").merge(ly_prev, on=keys, how="outer")
    for c in ("prev_mt", "ly_mt", "ly_prev_mt"):
        if c not in m.columns:
            m[c] = 0.0
        m[c] = m[c].fillna(0)
    prev = m["ly_prev_mt"].to_numpy(dtype=float)
    ly = m["ly_mt"].to_numpy(dtype=float)
    idx = np.full(len(m), np.nan)
    np.divide(ly, prev, out=idx, where=prev > 0)
    m["seasonal_mom_index"] = idx
    m["mom_expected_mt"] = m["prev_mt"] * m["seasonal_mom_index"]
    return m[keys + ["seasonal_mom_index", "mom_expected_mt"]]


def sku_industry_mix(
    facts_now: pd.DataFrame,
    facts_ly: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Shift-share industry-mix: extra volume from SKUs that grew faster nationally."""
    empty = pd.DataFrame(columns=keys + ["mix_effect_mt"])
    if facts_now is None or facts_now.empty or facts_ly is None or facts_ly.empty:
        return empty
    if "sku" not in facts_now.columns or "sku" not in facts_ly.columns:
        return empty
    nat_now = float(facts_now["volume_mt"].sum())
    nat_ly = float(facts_ly["volume_mt"].sum())
    nat_idx = parent_index(nat_now, nat_ly)
    sku_now = facts_now.groupby("sku", dropna=False)["volume_mt"].sum()
    sku_ly = facts_ly.groupby("sku", dropna=False)["volume_mt"].sum()
    sku_idx = (sku_now / sku_ly.replace(0, np.nan)).fillna(nat_idx)

    rows = []
    grouped = facts_ly.groupby(keys, dropna=False)
    for raw, part in grouped:
        if not isinstance(raw, tuple):
            raw = (raw,)
        rec = {k: raw[i] for i, k in enumerate(keys)}
        mix = 0.0
        for sku, ly_v in part.groupby("sku", dropna=False)["volume_mt"].sum().items():
            mix += float(ly_v) * (float(sku_idx.get(sku, nat_idx)) - nat_idx)
        rec["mix_effect_mt"] = mix
        rows.append(rec)
    return pd.DataFrame(rows) if rows else empty


def attach_mix(df: pd.DataFrame, mix: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    if mix is None or mix.empty:
        out["mix_effect_mt"] = 0.0
        return out
    out = out.merge(mix, on=keys, how="left")
    if "mix_effect_mt" not in out.columns:
        out["mix_effect_mt"] = 0.0
    out["mix_effect_mt"] = out["mix_effect_mt"].fillna(0.0)
    return out


def attach_seasonal_mom(df: pd.DataFrame, mom: pd.DataFrame, keys: list[str], vol_col: str = "volume_mt") -> pd.DataFrame:
    out = df.copy()
    if mom is None or mom.empty:
        out["seasonal_mom_index"] = np.nan
        out["mom_expected_mt"] = np.nan
        out["mom_gap_mt"] = np.nan
        return out
    out = out.merge(mom, on=keys, how="left")
    for col in ("seasonal_mom_index", "mom_expected_mt"):
        if col not in out.columns:
            out[col] = np.nan
    out["mom_gap_mt"] = out[vol_col] - out["mom_expected_mt"]
    return out


def k_from_ly(ly: pd.Series, default: float = 0.05) -> float:
    s = pd.to_numeric(ly, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return default
    return float(max(default, s.median()))


def situation_brief(national: dict[str, Any], cities: pd.DataFrame) -> dict[str, Any]:
    """One-screen narrative: weather vs exceptions vs what to do."""
    vol = float(national.get("volume_mt") or 0)
    expected = float(national.get("expected_mt") or 0)
    ly = float(national.get("ly_mt") or 0)
    gap = float(national.get("gap_mt") or 0)
    label = national.get("label") or national.get("period") or ""
    weather_dir = "declining" if gap < -1 else ("growing" if gap > 1 else "flat")
    pct = (gap / expected * 100) if expected else None
    intra_src = national.get("intra_month_source") or "closed"
    if weather_dir == "declining":
        weather = (
            f"{label}: the country billed {vol:.1f} MT against {expected:.1f} expected "
            f"from {int(national.get('n_history_periods') or 0)} months of history "
            f"(last year {ly:.1f} MT, {pct:+.0f}% vs expected). That is the weather. "
            "A city that is simply down with the country is not a local fire."
        )
    elif weather_dir == "growing":
        weather = (
            f"{label}: the country is ahead of last year ({vol:.1f} vs {expected:.1f} MT). "
            "The problem is units whose growth is slower than that fair share — not anyone still growing."
        )
    else:
        weather = (
            f"{label}: the country is roughly on last year's book ({vol:.1f} vs {expected:.1f} MT). "
            "Focus on units that are off that fair share."
        )
    if intra_src == "learned_mtd_cuts":
        weather += (
            f" Open MTD is paced from mid-month cuts already in your warehouse "
            f"(day {national.get('as_of_day')}: {float(national.get('intra_month_frac') or 0)*100:.0f}% of a full month)."
        )
    elif intra_src == "elapsed_days":
        weather += (
            f" Typical {label[:7] if label else 'month'} is learned from "
            f"{int(national.get('n_history_periods') or 0)} months on file. "
            "No mid-month MTD cuts are stored, so the open month is elapsed calendar days "
            "of that learned typical month — not a loading curve we specified."
        )
    elif intra_src == "empirical_mtd_curve":
        weather += " Open MTD is paced off your own historical intra-month billing curve."

    lag = pd.DataFrame()
    beat = pd.DataFrame()
    if cities is not None and not cities.empty and "situation" in cities.columns:
        lag = cities[cities["situation"] == "lagging"].sort_values("isolated_mt")
        beat = cities[cities["situation"] == "outperforming"].sort_values("isolated_mt", ascending=False)

    if lag.empty:
        problem = (
            "No city is a statistical exception versus the national index. "
            "Do not send extra people to the biggest city just because it is big. "
            "The miss is national — mix, drop size, supply, or the category."
        )
        action = (
            "Hold city firefights. Work the national driver split (coverage vs velocity vs mix) "
            "and the named shops that lag their own city, not a city hit-list."
        )
        headline = f"National is {weather_dir}. No city is an exception."
    else:
        names = ", ".join(str(x) for x in lag["grain_id"].head(4).tolist())
        extra = float(lag["isolated_mt"].sum())
        problem = (
            f"After national weather, extra hole is {extra:.1f} MT in {names}. "
            "Those cities declined more (or grew slower) than the country — that is the local problem."
        )
        action = (
            f"This week: {names}. Inside each, do the driver the card names "
            "(velocity / coverage / mix), on the named distributors and shops — not the tail."
        )
        if weather_dir == "growing":
            headline = f"National is growing. Slow-growth exceptions: {names} ({extra:.1f} MT vs fair share)."
        else:
            headline = f"National is {weather_dir}. Extra hole after weather: {names} ({extra:.1f} MT)."
    if not beat.empty:
        winners = ", ".join(str(x) for x in beat["grain_id"].head(3).tolist())
        problem += f" Holding / beating fair share: {winners} — copy, do not raid."

    return {
        "headline": headline,
        "weather": weather,
        "problem": problem,
        "action_summary": action,
        "weather_dir": weather_dir,
        "n_lagging": int(len(lag)),
        "n_outperforming": int(len(beat)),
        "extra_hole_mt": float(lag["isolated_mt"].sum()) if not lag.empty else 0.0,
    }


def rewrite_action(r: pd.Series, mtd: dict[str, Any], grain_label: str) -> str:
    """Action text that leads with parent-adjusted residual, then the driver."""
    name = str(r.get("grain_id") or grain_label)
    if name == "ALL":
        name = "National"
    sit = str(r.get("situation") or "with_market")
    iso = float(r.get("isolated_mt") or 0)
    share = float(r.get("share_expected_mt") or 0)
    vol = float(r.get("volume_mt") or 0)
    cov = float(r.get("coverage_effect_mt") or 0)
    vel = float(r.get("velocity_effect_mt") or 0)
    mix = float(r.get("mix_effect_mt") or 0)
    diagnosis = str(r.get("diagnosis") or "")
    z = float(r.get("z_score") or 0)
    if sit == "with_market" and grain_label != "national":
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} fair share of its parent (z {z:+.1f}). "
            "It moved with the market — not a local exception."
        )
    elif sit == "outperforming":
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} fair share ({iso:+.1f} MT, z {z:+.1f}). "
            "Beating the parent. Protect drop size; do not load; copy the beat."
        )
    else:
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} fair share of its parent "
            f"({iso:+.1f} MT extra hole, z {z:+.1f}). This is the local problem."
        )
    driver = f" Drivers: velocity {vel:+.1f} MT, coverage {cov:+.1f} MT, mix {mix:+.1f} MT."
    if diagnosis == "drop_size":
        how = " Recover drop size on continuing doors; do not print a lost-shop list."
    elif diagnosis == "coverage":
        how = " Must-visit material quiet doors; cadence the tail."
    elif diagnosis == "whitespace":
        how = " Open universe doors; volume will not appear from the same 20 billed shops."
    else:
        how = ""
    return base + driver + how
