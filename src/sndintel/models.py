"""Shop-month baseline, rule-based anomaly tags, and outlet segments.

There is one Expected in this system: the shop's own run-rate (last-three-month
AMS blended with the last-six-month median, winsorised) from ``season``. This
module used to fit a second, global XGBoost forecast, an Isolation Forest and a
K-Means clustering on top of it. Those produced numbers that contradicted the
pack ("model 0.31 MT" beside "Expected 0.27 MT"), flagged doors nobody could
explain, and re-labelled the same shop differently month to month when a
random-seeded fit moved. They are gone.

* ``forecast_shop_month`` — the run-rate baseline for every observed shop-month
  (same formula as the pack), so the shop chart line *is* the official Expected.
* ``detect_anomalies``   — four explainable rules against that baseline:
  trade loading, drop-off, lumpy buying, and (closed month only) a quiet month
  on a regular biller. No statistical outlier bucket.
* ``cluster_shops``      — RFM-style segments from deterministic thresholds.
  The name is kept for the pipeline; ``cluster_id`` is a stable code per segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sndintel.config import MIN_MATERIAL_MT
from sndintel.season import SHOP_WINSOR_MULT, fit_shop_expected

BASELINE_MODEL = "run_rate_baseline"

# Anomaly rules. Trade loading uses the same multiple the winsoriser caps at —
# a month the Expected engine refuses to believe is the month worth a call.
LOADING_MULT = SHOP_WINSOR_MULT
DROP_OFF_SHARE = 0.4  # billed under this share of Expected on a closed month
LUMPY_CV = 1.2
QUIET_MIN_BILL_RATE = 0.5  # billed in at least half of the prior 12 months
QUIET_MIN_EXPECTED_MT = 0.01  # ... and with a run-rate of at least 10 kg
CRITICAL_MT = 0.2

FORECAST_COLS = ["entity_type", "entity_id", "period", "actual", "predicted", "residual", "residual_pct", "model"]

SEGMENT_CODES = {
    "New / Ramp-up": 0,
    "Churn Risk": 1,
    "Dormant": 2,
    "Lumpy / Loaded": 3,
    "Star Account": 4,
    "Growth Target": 5,
    "Declining Core": 6,
    "Long Tail": 7,
    "Stable Core": 8,
}


def _baseline_by_period(shop_month: pd.DataFrame, periods: list[str]) -> pd.DataFrame:
    """Run-rate Expected for every shop for each period in ``periods``."""
    parts = []
    for p in periods:
        exp = fit_shop_expected(shop_month, str(p), None, 1.0)
        if exp is None or exp.empty:
            continue
        part = exp[["store_id", "expected_full_mt"]].copy()
        part["period"] = str(p)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["store_id", "period", "expected_full_mt"])
    return pd.concat(parts, ignore_index=True)


def forecast_shop_month(features: pd.DataFrame, shop_month: pd.DataFrame) -> pd.DataFrame:
    """Score every observed shop-month against its own run-rate baseline.

    ``features`` is accepted for signature stability; the baseline comes from
    ``season.fit_shop_expected`` so it matches the pack exactly. Periods with no
    prior history (the first month on file) have no baseline and are omitted.
    """
    if shop_month is None or shop_month.empty:
        return pd.DataFrame(columns=FORECAST_COLS)
    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str)
    sm["period"] = sm["period"].astype(str)
    sm["volume_mt"] = pd.to_numeric(sm.get("volume_mt"), errors="coerce").fillna(0.0)
    periods = sorted(sm["period"].unique())
    if len(periods) < 2:
        return pd.DataFrame(columns=FORECAST_COLS)
    base = _baseline_by_period(sm, periods[1:])
    if base.empty:
        return pd.DataFrame(columns=FORECAST_COLS)
    actual = sm.groupby(["store_id", "period"], as_index=False)["volume_mt"].sum()
    df = base.merge(actual, on=["store_id", "period"], how="left")
    df["volume_mt"] = df["volume_mt"].fillna(0.0)
    df = df.loc[(df["expected_full_mt"] > 1e-9) | (df["volume_mt"] > 1e-9)]
    out = pd.DataFrame(
        {
            "entity_type": "shop",
            "entity_id": df["store_id"].to_numpy(),
            "period": df["period"].to_numpy(),
            "actual": df["volume_mt"].to_numpy(dtype=float),
            "predicted": np.clip(df["expected_full_mt"].to_numpy(dtype=float), 0, None),
        }
    )
    out["residual"] = out["actual"] - out["predicted"]
    out["residual_pct"] = np.where(out["predicted"] > 0.01, out["residual"] / out["predicted"] * 100, np.nan)
    out["model"] = BASELINE_MODEL
    return out.reset_index(drop=True)


def detect_anomalies(
    features: pd.DataFrame,
    shop_month: pd.DataFrame,
    period: str,
    mtd_open: bool = False,
) -> pd.DataFrame:
    """Rule tags on the latest month against the shop's run-rate Expected.

    * ``trade_loading`` — billed ≥ 2.5× Expected (and ≥ 50 kg). Same multiple
      the winsoriser caps, so this is the month the Expected engine distrusts.
    * ``drop_off``      — closed month, Expected ≥ 50 kg, billed under 40% of it.
    * ``quiet_month``   — closed month, billed 0, a regular biller (≥ half of the
      prior twelve months) with a run-rate of at least 10 kg but under the
      drop-off materiality. The shop book decides *lost door* by days since last
      bill; this is only the monthly early-warning list.
    * ``lumpy``         — coefficient of variation ≥ 1.2 over six months, ≥ 50 kg.

    Open MTD months never raise drop-off or quiet flags — a quiet shop may still
    bill before month-end.
    """
    if features is None or features.empty or shop_month is None or shop_month.empty or not period:
        return pd.DataFrame()
    latest = features[features["period"].astype(str) == str(period)].copy()
    if latest.empty:
        return pd.DataFrame()
    latest["store_id"] = latest["store_id"].astype(str)
    sm = shop_month.copy()
    sm["store_id"] = sm["store_id"].astype(str)
    meta_cols = [c for c in ("volume_mt", "section", "city", "dsr_name", "store_name") if c in sm.columns]
    meta = sm.loc[sm["period"].astype(str) == str(period), ["store_id", *meta_cols]].drop_duplicates("store_id")
    latest = latest.drop(columns=[c for c in meta_cols if c in latest.columns], errors="ignore")
    latest = latest.merge(meta, on="store_id", how="left")
    exp = fit_shop_expected(sm, str(period), None, 1.0)
    exp_map = exp.set_index("store_id")["expected_full_mt"].to_dict() if exp is not None and not exp.empty else {}

    rows = []
    for _, row in latest.iterrows():
        sid = str(row["store_id"])
        expected = float(exp_map.get(sid, 0.0) or 0.0)
        volume = float(pd.to_numeric(row.get("volume_mt"), errors="coerce") or 0.0)
        months_on = int(pd.to_numeric(row.get("months_on_file"), errors="coerce") or 0)
        if months_on <= 1:
            continue  # first month on file: nothing to compare against
        kinds: list[str] = []
        if expected > 1e-9 and volume >= expected * LOADING_MULT and volume >= MIN_MATERIAL_MT:
            kinds.append("trade_loading")
        if not mtd_open:
            if expected >= MIN_MATERIAL_MT and volume <= expected * DROP_OFF_SHARE:
                kinds.append("drop_off")
            bill_rate = pd.to_numeric(row.get("billed_rate_12"), errors="coerce")
            if (
                volume <= 1e-9
                and expected >= QUIET_MIN_EXPECTED_MT
                and pd.notna(bill_rate)
                and float(bill_rate) >= QUIET_MIN_BILL_RATE
                and "drop_off" not in kinds
            ):
                kinds.append("quiet_month")
        cv = pd.to_numeric(row.get("cv_6m"), errors="coerce")
        if pd.notna(cv) and float(cv) >= LUMPY_CV and volume >= MIN_MATERIAL_MT:
            kinds.append("lumpy")
        if not kinds:
            continue
        primary = kinds[0]
        big = volume >= CRITICAL_MT or expected >= CRITICAL_MT
        if primary in {"trade_loading", "drop_off", "quiet_month"}:
            severity = "critical" if big else "high"
        else:
            severity = "medium"
        rel = abs(volume - expected) / expected if expected > 1e-9 else (volume / MIN_MATERIAL_MT if volume else 0.0)
        z = pd.to_numeric(row.get("zscore_own"), errors="coerce")
        rows.append(
            {
                "store_id": sid,
                "period": str(period),
                "kind": primary,
                "score": float(rel),
                "severity": severity,
                "volume_mt": volume,
                "expected_mt": expected,
                "details_json": pd.Series(
                    {
                        "kinds": kinds,
                        "zscore_own": float(z) if pd.notna(z) else None,
                        "mom_pct": _opt(row.get("mom_pct")),
                        "cv_6m": _opt(cv),
                        "top_sku_share": _opt(row.get("top_sku_share")),
                        "dsr_name": row.get("dsr_name"),
                        "section": row.get("section"),
                        "store_name": row.get("store_name"),
                        "rule": primary,
                    }
                ).to_json(),
            }
        )
    return pd.DataFrame(rows)


def _opt(value) -> float | None:
    v = pd.to_numeric(value, errors="coerce")
    return None if v is None or pd.isna(v) else float(v)


def cluster_shops(features: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    """RFM-style outlet segments from fixed, explainable thresholds.

    Recency (months since last bill), frequency (share of the last twelve
    months billed), monetary (mean monthly MT), trend (last three months vs
    the three before, labelled relative to the market median so a seasonal
    dip does not turn every door into Declining Core), breadth (SKU count)
    and cv. The label rules
    are in ``_label_segment``; ``cluster_id`` is a stable code for that label so
    the same shop gets the same segment on every run with the same data.
    """
    if shop_month is None or shop_month.empty or not period:
        return pd.DataFrame()
    hist = shop_month.copy()
    hist["store_id"] = hist["store_id"].astype(str)
    hist["volume_mt"] = pd.to_numeric(hist.get("volume_mt"), errors="coerce").fillna(0.0)
    if "billed" not in hist.columns:
        hist["billed"] = (hist["volume_mt"] > 0).astype(int)
    if "sku_count" not in hist.columns:
        hist["sku_count"] = 0
    periods = sorted(hist["period"].astype(str).unique())
    last12 = periods[-12:]
    window = hist[hist["period"].astype(str).isin(last12)]
    snap = (
        window.groupby("store_id", as_index=False)
        .agg(
            monetary=("volume_mt", "mean"),
            frequency=("billed", "mean"),
            breadth=("sku_count", "mean"),
            lifetime_volume=("volume_mt", "sum"),
            last_volume=("volume_mt", "last"),
            cv=("volume_mt", lambda s: float(s.std() / s.mean()) if s.mean() else 0.0),
        )
    )
    rec_cols = [c for c in ("recency_months", "months_on_file") if c in features.columns] if features is not None else []
    if rec_cols and not features.empty:
        rec = features.loc[features["period"].astype(str) == str(period), ["store_id", *rec_cols]].copy()
        rec["store_id"] = rec["store_id"].astype(str)
        snap = snap.merge(rec.drop_duplicates("store_id"), on="store_id", how="left")
    for c in ("recency_months", "months_on_file"):
        if c not in snap.columns:
            snap[c] = np.nan
    ordered = hist.sort_values(["store_id", "period"])
    last3 = ordered.groupby("store_id").tail(3).groupby("store_id")["volume_mt"].mean()

    def _prior3(s: pd.Series) -> float:
        if len(s) >= 6:
            return float(s.iloc[-6:-3].mean())
        if len(s) > 3:
            return float(s.iloc[:-3].mean())
        return float("nan")

    prior = ordered.groupby("store_id")["volume_mt"].apply(_prior3)
    snap["recent3"] = snap["store_id"].map(last3)
    snap["prior3"] = snap["store_id"].map(prior)
    snap["trend"] = np.where(snap["prior3"] > 0, (snap["recent3"] - snap["prior3"]) / snap["prior3"], 0.0)
    # Label on the trend *relative to the market*: when every shop is down 25%
    # because the season turned, nobody is "Declining Core" — the ones that are
    # down 25% more than their peers are.
    market = float(pd.Series(snap["trend"]).replace([np.inf, -np.inf], np.nan).dropna().median() or 0.0) if len(snap) >= 8 else 0.0
    snap["trend_vs_market"] = snap["trend"] - market
    snap["recency_months"] = pd.to_numeric(snap["recency_months"], errors="coerce").fillna(99)
    snap["months_on_file"] = pd.to_numeric(snap["months_on_file"], errors="coerce").fillna(99)
    snap["cv"] = snap["cv"].replace([np.inf, -np.inf], 0).fillna(0)
    snap["segment"] = [_label_segment(r) for r in snap.itertuples(index=False)]
    snap["cluster_id"] = snap["segment"].map(SEGMENT_CODES).fillna(len(SEGMENT_CODES)).astype(int)
    snap["last_period"] = str(period)
    return snap[
        [
            "store_id",
            "cluster_id",
            "segment",
            "recency_months",
            "frequency",
            "monetary",
            "trend",
            "breadth",
            "cv",
            "lifetime_volume",
            "last_period",
            "last_volume",
        ]
    ].reset_index(drop=True)


def _num_attr(row, name: str, default: float) -> float:
    v = pd.to_numeric(getattr(row, name, None), errors="coerce")
    return default if v is None or pd.isna(v) else float(v)


def _label_segment(row) -> str:
    # A shop that billed this month has recency 0 — that is the freshest value,
    # not a missing one, so defaults apply only to None / NaN.
    recency = _num_attr(row, "recency_months", 99)
    freq = _num_attr(row, "frequency", 0)
    money = _num_attr(row, "monetary", 0)
    trend = _num_attr(row, "trend_vs_market", _num_attr(row, "trend", 0))
    cv = _num_attr(row, "cv", 0)
    months_on = _num_attr(row, "months_on_file", 99)
    if months_on <= 2:
        return "New / Ramp-up"
    if recency >= 4 and freq >= 0.3:
        return "Churn Risk"
    if recency >= 4 and freq < 0.3:
        return "Dormant"
    if cv >= 1.3 and money >= MIN_MATERIAL_MT:
        return "Lumpy / Loaded"
    if money >= 0.15 and freq >= 0.6 and trend >= 0:
        return "Star Account"
    if money >= 0.08 and trend >= 0.15:
        return "Growth Target"
    if trend <= -0.2 and freq >= 0.4:
        return "Declining Core"
    if money < 0.03 and freq < 0.4:
        return "Long Tail"
    if trend >= 0.1:
        return "Growth Target"
    return "Stable Core"
