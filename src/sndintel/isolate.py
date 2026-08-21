"""Expected-based isolation: billed vs recent run-rate.

Excel compares each unit to last year. Peer fair share compares each unit to
its parent. Calendar-month seasonality zeros a city that never billed in
August even when AMS is 44 MT.

The call in this pack is:

    After this unit's own Expected (last three closed months, blended with
    the last-six-month median, paced if MTD is open), is it still a problem?

Expected is learned at every grain. Children's Expecteds are scaled so they
add to the parent Expected. Gap is the hole versus Expected. From drop /
unvisited / unbilled partition that hole.

Residuals are shrunk with empirical Bayes so a 0.02 MT shop cannot outrank
Eva Foods on a percentage, and scored with a robust z.

Intra-month day shape is learned from mid-month MTD cuts when they exist.
Month-end totals cannot teach day 20. If those cuts are missing, open MTD
uses elapsed calendar days of the recent run-rate.
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


VITAL_COVERAGE = 0.80
VITAL_MIN_Z = 1.0
VITAL_MIN_SHARE = 0.05


def isolate_key_holes(
    df: pd.DataFrame,
    value_col: str = "recoverable_mt",
    group_col: str | None = None,
    coverage: float = VITAL_COVERAGE,
    min_z: float = VITAL_MIN_Z,
    min_share: float = VITAL_MIN_SHARE,
    abs_floor: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Keep the vital few holes; roll the rest into a remainder.

    Among lagging units (optionally within a parent city/distributor):

    1. Iglewicz–Hoaglin modified z of Gap versus other lagging Gaps.
       Keep z ≥ ``min_z`` (unusually large vs peers on the lagging list).
    2. Then walk largest-first and add a unit only if it is at least
       ``min_share`` of that group's hole (default 5%) until named rows
       cover ``coverage`` of the hole (default 80%). The tail is not listed
       just to hit 80% — a city of equal 0.3 MT doors is a coverage KPI.

    Always keeps the single largest hole in a group when it clears ``abs_floor``.
    """
    empty_meta = {
        "n_kept": 0,
        "n_hidden": 0,
        "hidden_mt": 0.0,
        "n_pool": 0,
        "coverage": float(coverage),
        "min_z": float(min_z),
        "min_share": float(min_share),
        "abs_floor": float(abs_floor),
    }
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame(), empty_meta
    work = df.copy()
    val = pd.to_numeric(work.get(value_col), errors="coerce").fillna(0.0)
    work = work.assign(_vital_gap=val)
    pool = work.loc[val > float(abs_floor)].copy()
    below = work.loc[val <= float(abs_floor)].copy()
    if pool.empty:
        hidden_mt = float(val.sum())
        meta = {**empty_meta, "n_hidden": int(len(work)), "hidden_mt": hidden_mt, "n_pool": int(len(work))}
        return work.iloc[0:0].drop(columns=["_vital_gap"], errors="ignore"), meta

    kept_parts: list[pd.DataFrame] = []
    rest_parts: list[pd.DataFrame] = [below] if not below.empty else []
    if group_col and group_col in pool.columns:
        for _, g in pool.groupby(pool[group_col].astype(str), dropna=False):
            k, r = _vital_few_split(g, coverage=coverage, min_z=min_z, min_share=min_share)
            if not k.empty:
                kept_parts.append(k)
            if not r.empty:
                rest_parts.append(r)
    else:
        k, r = _vital_few_split(pool, coverage=coverage, min_z=min_z, min_share=min_share)
        if not k.empty:
            kept_parts.append(k)
        if not r.empty:
            rest_parts.append(r)

    kept = pd.concat(kept_parts, ignore_index=False) if kept_parts else pool.iloc[0:0]
    rest = pd.concat(rest_parts, ignore_index=False) if rest_parts else pool.iloc[0:0]
    if "_vital_gap" in kept.columns:
        kept = kept.drop(columns=["_vital_gap"])
    hidden_mt = float(pd.to_numeric(rest.get(value_col), errors="coerce").fillna(0).sum()) if not rest.empty else 0.0
    meta = {
        **empty_meta,
        "n_kept": int(len(kept)),
        "n_hidden": int(len(rest)),
        "hidden_mt": hidden_mt,
        "n_pool": int(len(work)),
    }
    return kept, meta


def _vital_few_split(
    g: pd.DataFrame,
    coverage: float,
    min_z: float,
    min_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.to_numeric(g["_vital_gap"], errors="coerce").fillna(0.0)
    if x.empty:
        return g.iloc[0:0], g
    if len(g) == 1:
        return g, g.iloc[0:0]
    keep = pd.Series(False, index=g.index)
    total = float(x.sum()) or 1.0
    hurdle = max(0.0, float(min_share) * total)
    leader = x.idxmax()
    if len(g) == 1 or float(x.loc[leader]) + 1e-12 >= hurdle:
        keep.loc[leader] = True
    med = float(x.median())
    mad = float((x - med).abs().median())
    if mad >= 1e-12:
        z = 0.6745 * (x - med) / mad
        keep.loc[z >= float(min_z)] = True
    running = float(x.loc[keep].sum()) if keep.any() else 0.0
    if running / total < float(coverage):
        for idx, v in x.sort_values(ascending=False).items():
            if keep.loc[idx]:
                continue
            if float(v) + 1e-12 < hurdle:
                break
            keep.loc[idx] = True
            running += float(v)
            if running / total >= float(coverage):
                break
    return g.loc[keep].copy(), g.loc[~keep].copy()


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


def apply_expected_gap(df: pd.DataFrame, k: float, z_clip: float = 4.0) -> pd.DataFrame:
    """Hole versus this unit's own Expected. share_expected_mt tracks Expected so From-columns add to Gap."""
    out = df.copy()
    exp = pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0.0)
    vol = pd.to_numeric(out.get("volume_mt"), errors="coerce").fillna(0.0)
    ly = pd.to_numeric(out.get("ly_mt"), errors="coerce").fillna(0.0)
    out["share_expected_mt"] = exp
    out["competitive_mt"] = vol - exp
    size = np.maximum(ly.to_numpy(dtype=float), exp.to_numpy(dtype=float))
    out["isolated_mt"] = [empirical_bayes(c, s, k) for c, s in zip(out["competitive_mt"], size)]
    out["gap_mt"] = vol - exp
    out["gap_pct"] = np.where(exp > 1e-9, (vol - exp) / exp * 100, np.nan)
    out["z_score"] = robust_z(pd.Series(out["isolated_mt"], index=out.index)).clip(-z_clip, z_clip)
    out["focus_score"] = -pd.Series(out["isolated_mt"]) * (1.0 + out["z_score"].abs() / 4.0)
    out["situation"] = out.apply(_situation_label, axis=1)
    return out


def apply_shift_share(
    df: pd.DataFrame,
    parent_now: float,
    parent_ly: float,
    k: float,
    z_clip: float = 4.0,
) -> pd.DataFrame:
    """Deprecated peer residual. Kept for tests; scoring uses apply_expected_gap."""
    out = df.copy()
    idx = parent_index(parent_now, parent_ly)
    out["parent_index"] = idx
    if "expected_mt" not in out.columns or pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0).eq(0).all():
        out["expected_mt"] = out["ly_mt"].fillna(0) * idx
    return apply_expected_gap(out, k, z_clip=z_clip)


def _situation_label(r: pd.Series) -> str:
    iso = float(r.get("isolated_mt") or 0)
    exp = float(r.get("expected_mt") or r.get("ly_mt") or 0)
    material = max(0.5, 0.02 * exp) if exp >= 8 else max(0.15, 0.05 * max(exp, 0.0))
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
    """One-screen narrative: billed vs Expected, then the units behind their own Expected."""
    vol = float(national.get("volume_mt") or 0)
    expected = float(national.get("expected_mt") or 0)
    ly = float(national.get("ly_mt") or 0)
    gap = float(national.get("gap_mt") or 0)
    label = national.get("label") or national.get("period") or ""
    weather_dir = "declining" if gap < -1 else ("growing" if gap > 1 else "flat")
    pct = (gap / expected * 100) if expected else None
    intra_src = national.get("intra_month_source") or "closed"
    miss = max(0.0, -gap)
    if weather_dir == "declining":
        weather = (
            f"{label}: the country billed {vol:.1f} MT against {expected:.1f} expected "
            f"from {int(national.get('n_history_periods') or 0)} months of history "
            f"(last year {ly:.1f} MT, {pct:+.0f}% vs expected). Gap is that miss "
            f"({miss:.1f} MT) — not a leftover versus peers."
        )
    elif weather_dir == "growing":
        weather = (
            f"{label}: the country billed {vol:.1f} MT against {expected:.1f} expected "
            f"(last year {ly:.1f} MT). Ahead of the recent run-rate."
        )
    else:
        weather = (
            f"{label}: the country is on its recent run-rate ({vol:.1f} vs {expected:.1f} MT). "
            "Focus on units that are off their own Expected."
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
            "No city is behind its own Expected. "
            "Do not send extra people to the biggest city just because it is big. "
            "If the country is also on Expected, hold. If the country is behind, the miss is national — mix, drop size, supply, or the category."
        )
        action = (
            "Hold city firefights. Work the national driver split (coverage vs velocity vs mix) "
            "and the named shops that lag their own Expected, not a city hit-list."
        )
        headline = f"National is {weather_dir} vs Expected. No city is behind its own Expected."
    else:
        names = ", ".join(str(x) for x in lag["grain_id"].head(4).tolist())
        extra = float(lag["isolated_mt"].sum())
        problem = (
            f"Cities behind their own Expected: {names} ({extra:.1f} MT). "
            "Each hole is billed versus that city's recent run-rate, not versus last August or the country's current book."
        )
        action = (
            f"This week: {names}. Inside each, do the driver the card names "
            "(unbilled / unvisited / drop size), on the named distributors and shops — not the tail."
        )
        headline = f"National is {weather_dir} vs Expected. Behind their Expected: {names} ({extra:.1f} MT)."
    if not beat.empty:
        winners = ", ".join(str(x) for x in beat["grain_id"].head(3).tolist())
        problem += f" Ahead of Expected: {winners} — copy, do not raid."

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
    """Action text that leads with billed versus this unit's Expected, then the driver."""
    name = str(r.get("grain_id") or grain_label)
    if name == "ALL":
        name = "National"
    sit = str(r.get("situation") or "with_market")
    iso = float(r.get("isolated_mt") or 0)
    share = float(r.get("expected_mt") or r.get("share_expected_mt") or 0)
    vol = float(r.get("volume_mt") or 0)
    cov = float(r.get("coverage_effect_mt") or 0)
    vel = float(r.get("velocity_effect_mt") or 0)
    mix = float(r.get("mix_effect_mt") or 0)
    diagnosis = str(r.get("diagnosis") or "")
    z = float(r.get("z_score") or 0)
    if sit == "with_market" and grain_label != "national":
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} Expected (z {z:+.1f}). "
            "On its recent run-rate — not a local exception."
        )
    elif sit == "outperforming":
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} Expected ({iso:+.1f} MT, z {z:+.1f}). "
            "Ahead of its typical month. Protect drop size; do not load; copy the beat."
        )
    else:
        base = (
            f"{name} billed {vol:.1f} MT vs {share:.1f} Expected "
            f"({iso:+.1f} MT hole, z {z:+.1f}). This is the local problem."
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
