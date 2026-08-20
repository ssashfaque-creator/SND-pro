"""Forecasting, anomaly detection, and outlet clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sndintel.config import (
    CLUSTER_RANDOM_STATE,
    FORECAST_MIN_PERIODS,
    ISOLATION_CONTAMINATION,
    MODEL_DIR,
    ensure_dirs,
)

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


FEATURE_COLS = [
    "month_num",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "roll_mean_3",
    "roll_mean_6",
    "sku_count",
    "billed_rate_12",
]


def _xgb() -> Optional[object]:
    if XGBRegressor is None:
        return None
    return XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=2,
        random_state=CLUSTER_RANDOM_STATE,
    )


def forecast_shop_month(features: pd.DataFrame, shop_month: pd.DataFrame) -> pd.DataFrame:
    """Train a global shop-month model and score every observed period.

    Sparse shops fall back to a seasonal-naive / rolling-median baseline so
    the residual is still useful for delta scoring.
    """
    if features.empty or shop_month.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "period", "actual", "predicted", "residual", "residual_pct", "model"])
    df = features.merge(
        shop_month[["store_id", "period", "month", "billed"]],
        on=["store_id", "period"],
        how="left",
    )
    df["month_num"] = df["month"].fillna(df["period"].str.slice(5, 7).astype(int))
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["actual"] = df["volume_mt"].fillna(0)
    # Prefer last-year same month when the shop actually existed then.
    df["baseline"] = df["lag_12"].where(df["lag_12"].notna(), df["roll_median_6"])
    df["baseline"] = df["baseline"].fillna(df["roll_mean_3"]).fillna(df["lag_1"]).fillna(0)

    model_name = "seasonal_naive"
    preds = df["baseline"].to_numpy()
    train = df.dropna(subset=FEATURE_COLS).copy()
    train = train[train[FEATURE_COLS].notna().all(axis=1)]
    n_periods = df["period"].nunique()
    model = _xgb()
    if model is not None and len(train) >= 80 and n_periods >= FORECAST_MIN_PERIODS:
        X = train[FEATURE_COLS]
        y = train["actual"]
        # Time-aware-ish fit: drop the latest period from training when possible.
        latest = sorted(train["period"].unique())[-1]
        mask = train["period"] != latest
        if mask.sum() >= 60:
            model.fit(X.loc[mask], y.loc[mask])
        else:
            model.fit(X, y)
        preds = model.predict(df[FEATURE_COLS])
        model_name = "xgboost_shop_month"
        ensure_dirs()
        joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_DIR / "forecast_xgb.joblib")

    out = pd.DataFrame(
        {
            "entity_type": "shop",
            "entity_id": df["store_id"],
            "period": df["period"],
            "actual": df["actual"],
            "predicted": np.clip(preds, 0, None),
        }
    )
    out["residual"] = out["actual"] - out["predicted"]
    out["residual_pct"] = np.where(out["predicted"] > 0.01, out["residual"] / out["predicted"] * 100, np.nan)
    out["model"] = model_name
    return out


def detect_anomalies(features: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    """Isolation Forest on shop-normalized latest-month vectors, plus rule tags."""
    if features.empty:
        return pd.DataFrame()
    latest = features[features["period"] == period].copy()
    if latest.empty:
        return pd.DataFrame()
    latest = latest.merge(
        shop_month[["store_id", "period", "volume_mt", "section", "city", "dsr_name", "store_name"]],
        on=["store_id", "period"],
        how="left",
        suffixes=("", "_sm"),
    )
    cols = [
        "volume_mt",
        "sku_count",
        "zscore_own",
        "mom_pct",
        "cv_6m",
        "vs_section_pct",
        "top_sku_share",
        "recency_months",
        "roll_median_6",
    ]
    for c in cols:
        if c not in latest.columns:
            latest[c] = 0
    X = latest[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scores = np.zeros(len(latest))
    if_flag = np.zeros(len(latest), dtype=bool)
    if len(latest) >= 15:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(
            contamination=min(ISOLATION_CONTAMINATION, 0.2),
            random_state=CLUSTER_RANDOM_STATE,
            n_estimators=200,
        )
        pred = iso.fit_predict(Xs)
        scores = -iso.score_samples(Xs)
        if_flag = pred == -1
        ensure_dirs()
        joblib.dump({"model": iso, "scaler": scaler, "cols": cols}, MODEL_DIR / "isolation_forest.joblib")

    rows = []
    for i, row in latest.reset_index(drop=True).iterrows():
        expected = row.get("roll_median_6")
        expected = float(expected) if pd.notna(expected) else 0.0
        ly = row.get("lag_12")
        if pd.notna(ly):
            expected = float(ly)
        volume = float(row.get("volume_mt") or 0)
        months_on = int(row.get("months_on_file") or 0)
        comparable = int(row.get("yoy_comparable") or 0)
        z = row.get("zscore_own")
        z = float(z) if pd.notna(z) else 0.0
        kinds = []
        # New / newly listed shops are not dumps or lapses.
        if months_on <= 1:
            continue
        if expected > 0 and volume >= expected * 2.5 and volume >= 0.05:
            kinds.append("trade_loading")
        if expected > 0.05 and volume <= expected * 0.4:
            kinds.append("drop_off")
        if volume == 0 and (row.get("billed_rate_12") or 0) >= 0.5 and comparable:
            kinds.append("lapse")
        cv = row.get("cv_6m")
        if pd.notna(cv) and float(cv) >= 1.2 and volume >= 0.05:
            kinds.append("lumpy")
        if if_flag[i] and not kinds:
            kinds.append("statistical_outlier")
        if not kinds:
            continue
        primary = kinds[0]
        severity = "high"
        if primary in {"trade_loading", "lapse"} and (volume >= 0.2 or expected >= 0.2 or abs(z) >= 2.5):
            severity = "critical"
        elif primary == "drop_off" and expected >= 0.15:
            severity = "critical"
        elif scores[i] < np.quantile(scores, 0.85) if len(scores) else True:
            severity = "medium"
        rows.append(
            {
                "store_id": row["store_id"],
                "period": period,
                "kind": primary,
                "score": float(scores[i]),
                "severity": severity,
                "volume_mt": volume,
                "expected_mt": expected,
                "details_json": pd.Series(
                    {
                        "kinds": kinds,
                        "zscore_own": z,
                        "mom_pct": row.get("mom_pct"),
                        "cv_6m": row.get("cv_6m"),
                        "top_sku_share": row.get("top_sku_share"),
                        "dsr_name": row.get("dsr_name"),
                        "section": row.get("section"),
                        "store_name": row.get("store_name"),
                    }
                ).to_json(),
            }
        )
    return pd.DataFrame(rows)


def cluster_shops(features: pd.DataFrame, shop_month: pd.DataFrame, period: str) -> pd.DataFrame:
    if shop_month.empty:
        return pd.DataFrame()
    hist = shop_month.copy()
    periods = sorted(hist["period"].unique())
    last12 = periods[-12:]
    window = hist[hist["period"].isin(last12)]
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
    rec = features[features["period"] == period][["store_id", "recency_months", "roll_mean_3", "lag_3", "months_on_file", "yoy_comparable"]]
    snap = snap.merge(rec, on="store_id", how="left")
    # Trend: last 3 billed months vs prior 3
    ordered = hist.sort_values(["store_id", "period"])
    last3 = ordered.groupby("store_id").tail(3).groupby("store_id")["volume_mt"].mean()

    def _prior3(s: pd.Series) -> float:
        if len(s) >= 6:
            return float(s.iloc[-6:-3].mean())
        if len(s) > 3:
            return float(s.iloc[:-3].mean())
        return float("nan")

    prior = ordered.groupby("store_id")["volume_mt"].apply(_prior3)
    snap = snap.merge(last3.rename("recent3"), on="store_id", how="left")
    snap["prior3"] = snap["store_id"].map(prior)
    snap["trend"] = np.where(snap["prior3"] > 0, (snap["recent3"] - snap["prior3"]) / snap["prior3"], 0)
    snap["recency_months"] = snap["recency_months"].fillna(99)
    snap["cv"] = snap["cv"].replace([np.inf, -np.inf], 0).fillna(0)
    feat_cols = ["recency_months", "frequency", "monetary", "trend", "breadth", "cv"]
    X = snap[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = _choose_k(Xs)
    km = KMeans(n_clusters=k, random_state=CLUSTER_RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xs) if len(snap) >= k else np.zeros(len(snap), dtype=int)
    snap["cluster_id"] = labels
    snap["segment"] = [_label_segment(r) for r in snap.itertuples(index=False)]
    snap["last_period"] = period
    ensure_dirs()
    if len(snap) >= k:
        joblib.dump({"model": km, "scaler": scaler, "k": k}, MODEL_DIR / "kmeans.joblib")
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
    ]


def _choose_k(Xs: np.ndarray) -> int:
    n = len(Xs)
    if n < 12:
        return 2 if n >= 4 else 1
    best_k, best_score = 4, -1
    for k in range(3, min(7, n // 4 + 1)):
        km = KMeans(n_clusters=k, random_state=CLUSTER_RANDOM_STATE, n_init=8)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _label_segment(row) -> str:
    recency = getattr(row, "recency_months", 99) or 99
    freq = getattr(row, "frequency", 0) or 0
    money = getattr(row, "monetary", 0) or 0
    trend = getattr(row, "trend", 0) or 0
    cv = getattr(row, "cv", 0) or 0
    months_on = getattr(row, "months_on_file", 99) or 99
    if months_on <= 2:
        return "New / Ramp-up"
    if recency >= 4 and freq >= 0.3:
        return "Churn Risk"
    if recency >= 4 and freq < 0.3:
        return "Dormant"
    if cv >= 1.3 and money >= 0.05:
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
