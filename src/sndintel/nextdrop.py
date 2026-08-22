"""Next-order size from billed-day sequences.

Median invoice is the wrong ask when the last drop was fat (shop is stocked)
or thin / late (shop should catch up). This module trains a pooled gradient
boosted tree on every historical next-bill, using only information known
before that bill, then scores the next order as of today.

XGBoost is the primary learner (already in the stack). HistGradientBoosting
is the fallback. Thin history still uses the hierarchical median drop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sndintel.config import CLUSTER_RANDOM_STATE
from sndintel.io_utils import shift_period

FEATURE_COLS = [
    "last_drop",
    "prev_drop",
    "prev2_drop",
    "drop_p25",
    "drop_p50",
    "drop_p75",
    "last_over_p50",
    "last_over_ams",
    "median_gap",
    "last_gap",
    "days_since",
    "overdue_ratio",
    "cover_days",
    "n_prior",
    "billed_mtd",
    "ams",
    "remaining_proxy",
    "last_month",
    "day_of_month",
    "month_sin",
    "month_cos",
    "city_p50",
    "dsr_p50",
]

MIN_TRAIN_ROWS = 80
MIN_HISTORY = 2


def attach_next_drop(
    shops: pd.DataFrame,
    shop_day: pd.DataFrame | None,
    as_of_ts: pd.Timestamp,
    shop_month: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Set ``next_drop_mt`` from the next-order model; leave median typical drop alone."""
    out = shops.copy()
    out["next_drop_mt"] = pd.to_numeric(out.get("typical_drop_mt"), errors="coerce")
    out["next_drop_model"] = "median"
    if shop_day is None or shop_day.empty or out.empty:
        return out
    daily = _daily(shop_day, as_of_ts)
    if daily.empty:
        return out
    train, city_p50, dsr_p50 = _transition_frame(daily, shop_month)
    model, name = _fit(train)
    scored = _score_shops(out, daily, as_of_ts, shop_month, city_p50, dsr_p50)
    if model is None or scored.empty:
        return out
    pred = _predict(model, scored)
    out = out.merge(pred, on="store_id", how="left")
    have = pd.to_numeric(out.get("pred_next_drop_mt"), errors="coerce")
    typ = pd.to_numeric(out.get("typical_drop_mt"), errors="coerce").fillna(0)
    ams = pd.to_numeric(out.get("ams_3m"), errors="coerce").fillna(0)
    last = pd.to_numeric(out.get("last_drop_mt"), errors="coerce").fillna(0)
    hi = pd.concat([3.0 * ams, 3.0 * typ, 2.0 * last], axis=1).max(axis=1).clip(lower=0.25)
    clipped = have.clip(lower=0.02, upper=hi)
    use = clipped.notna()
    out.loc[use, "next_drop_mt"] = clipped[use]
    out.loc[use, "next_drop_model"] = name
    out = out.drop(columns=["pred_next_drop_mt"], errors="ignore")
    return out


def _daily(shop_day: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    out = shop_day.copy()
    out["store_id"] = out["store_id"].astype(str)
    out["volume_mt"] = pd.to_numeric(out["volume_mt"], errors="coerce").fillna(0.0)
    out["sale_date"] = pd.to_datetime(out.get("sale_date"), errors="coerce")
    out = out[out["sale_date"].notna() & (out["volume_mt"] > 0) & (out["sale_date"] <= pd.Timestamp(as_of_ts))]
    if out.empty:
        return out
    if "period" not in out.columns or out["period"].isna().all():
        out["period"] = out["sale_date"].dt.strftime("%Y-%m")
    out["period"] = out["period"].astype(str)
    for col in ("city", "dsr_name"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out.sort_values(["store_id", "sale_date"])


def _month_volumes(shop_month: pd.DataFrame | None, daily: pd.DataFrame) -> dict[tuple[str, str], float]:
    frames = []
    if shop_month is not None and not shop_month.empty:
        sm = shop_month.copy()
        sm["store_id"] = sm["store_id"].astype(str)
        sm["period"] = sm["period"].astype(str)
        sm["volume_mt"] = pd.to_numeric(sm["volume_mt"], errors="coerce").fillna(0.0)
        frames.append(sm.groupby(["store_id", "period"], as_index=False)["volume_mt"].sum())
    if daily is not None and not daily.empty:
        frames.append(daily.groupby(["store_id", "period"], as_index=False)["volume_mt"].sum())
    if not frames:
        return {}
    g = pd.concat(frames, ignore_index=True).groupby(["store_id", "period"], as_index=False)["volume_mt"].max()
    return {(str(r.store_id), str(r.period)): float(r.volume_mt) for r in g.itertuples(index=False)}


def _trailing_ams(month_vol: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    by_shop: dict[str, list[tuple[str, float]]] = {}
    for (sid, period), vol in month_vol.items():
        by_shop.setdefault(sid, []).append((period, vol))
    out: dict[tuple[str, str], float] = {}
    for sid, rows in by_shop.items():
        rows = sorted(rows)
        vols = [v for _, v in rows]
        pers = [p for p, _ in rows]
        for i, period in enumerate(pers):
            win = vols[max(0, i - 3) : i]
            out[(sid, period)] = float(np.mean(win)) if win else 0.0
    return out


def _transition_frame(
    daily: pd.DataFrame,
    shop_month: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    month_vol = _month_volumes(shop_month, daily)
    ams_map = _trailing_ams(month_vol)
    rows: list[dict[str, Any]] = []
    city_drops: dict[str, list[float]] = {}
    dsr_drops: dict[str, list[float]] = {}
    for sid, g in daily.groupby("store_id", sort=False):
        g = g.sort_values("sale_date")
        vols = g["volume_mt"].to_numpy(dtype=float)
        dates = pd.to_datetime(g["sale_date"])
        periods = g["period"].astype(str).to_numpy()
        city = str(g["city"].iloc[-1] or "")
        dsr = str(g["dsr_name"].iloc[-1] or "")
        if len(vols) < MIN_HISTORY:
            continue
        for i in range(1, len(vols)):
            target_date = pd.Timestamp(dates.iloc[i])
            period = str(periods[i])
            feat = _features(
                vols[:i],
                dates.iloc[:i],
                as_of=target_date,
                ams=ams_map.get((str(sid), period), 0.0),
                last_month=month_vol.get((str(sid), shift_period(period, -1)), 0.0),
                billed_mtd=float(vols[:i][periods[:i] == period].sum()) if i else 0.0,
                city_p50=0.0,
                dsr_p50=0.0,
            )
            feat["y"] = float(vols[i])
            feat["city"] = city
            feat["dsr_name"] = dsr
            feat["store_id"] = str(sid)
            rows.append(feat)
            city_drops.setdefault(city, []).append(float(vols[i - 1]))
            dsr_drops.setdefault(dsr, []).append(float(vols[i - 1]))
    city_p50 = {k: float(np.median(v)) for k, v in city_drops.items() if v}
    dsr_p50 = {k: float(np.median(v)) for k, v in dsr_drops.items() if v}
    train = pd.DataFrame(rows)
    if train.empty:
        return train, city_p50, dsr_p50
    train["city_p50"] = train["city"].map(city_p50).fillna(train["drop_p50"])
    train["dsr_p50"] = train["dsr_name"].map(dsr_p50).fillna(train["drop_p50"])
    return train, city_p50, dsr_p50


def _features(
    vols: np.ndarray,
    dates: pd.Series,
    as_of: pd.Timestamp,
    ams: float,
    last_month: float,
    billed_mtd: float,
    city_p50: float,
    dsr_p50: float,
) -> dict[str, float]:
    vols = np.asarray(vols, dtype=float)
    last = float(vols[-1])
    prev = float(vols[-2]) if len(vols) >= 2 else last
    prev2 = float(vols[-3]) if len(vols) >= 3 else prev
    p25, p50, p75 = (float(x) for x in np.percentile(vols, [25, 50, 75]))
    if len(dates) >= 2:
        delta = pd.to_datetime(dates).diff().dt.days.dropna()
        gaps = delta[(delta >= 2) & (delta <= 120)]
        med_gap = float(gaps.median()) if len(gaps) else 30.0
        last_gap = float((pd.Timestamp(dates.iloc[-1]) - pd.Timestamp(dates.iloc[-2])).days)
    else:
        med_gap = 30.0
        last_gap = 30.0
    days_since = float((pd.Timestamp(as_of) - pd.Timestamp(dates.iloc[-1])).days)
    daily = max(float(ams) / 30.0 if ams else 0.0, p50 / max(med_gap, 1.0), 0.01)
    cover = last / daily - days_since
    ams = float(ams or 0.0)
    return {
        "last_drop": last,
        "prev_drop": prev,
        "prev2_drop": prev2,
        "drop_p25": p25,
        "drop_p50": p50,
        "drop_p75": p75,
        "last_over_p50": last / p50 if p50 > 0 else 1.0,
        "last_over_ams": last / ams if ams > 0 else 1.0,
        "median_gap": med_gap,
        "last_gap": last_gap,
        "days_since": days_since,
        "overdue_ratio": days_since / med_gap if med_gap > 0 else 1.0,
        "cover_days": cover,
        "n_prior": float(len(vols)),
        "billed_mtd": float(billed_mtd or 0.0),
        "ams": ams,
        "remaining_proxy": max(0.0, ams - float(billed_mtd or 0.0)),
        "last_month": float(last_month or 0.0),
        "day_of_month": float(pd.Timestamp(as_of).day),
        "month_sin": float(np.sin(2 * np.pi * pd.Timestamp(as_of).month / 12)),
        "month_cos": float(np.cos(2 * np.pi * pd.Timestamp(as_of).month / 12)),
        "city_p50": float(city_p50 or p50),
        "dsr_p50": float(dsr_p50 or p50),
    }


def _regressor() -> tuple[Any, str]:
    try:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.06,
                min_child_weight=4,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.4,
                objective="reg:squarederror",
                n_jobs=2,
                random_state=CLUSTER_RANDOM_STATE,
            ),
            "xgboost",
        )
    except Exception:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            HistGradientBoostingRegressor(
                max_depth=5,
                max_iter=180,
                learning_rate=0.08,
                min_samples_leaf=8,
                random_state=CLUSTER_RANDOM_STATE,
            ),
            "histgb",
        )


def _fit(train: pd.DataFrame) -> tuple[Any, str]:
    if train is None or train.empty or len(train) < MIN_TRAIN_ROWS:
        return None, "median"
    model, name = _regressor()
    X = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.log1p(pd.to_numeric(train["y"], errors="coerce").clip(lower=0).fillna(0))
    model.fit(X, y)
    return model, name


def _score_shops(
    shops: pd.DataFrame,
    daily: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    shop_month: pd.DataFrame | None,
    city_p50: dict[str, float],
    dsr_p50: dict[str, float],
) -> pd.DataFrame:
    month_vol = _month_volumes(shop_month, daily)
    as_of = pd.Timestamp(as_of_ts)
    period = as_of.strftime("%Y-%m")
    hist = {sid: g.sort_values("sale_date") for sid, g in daily.groupby("store_id", sort=False)}
    rows = []
    for row in shops.itertuples(index=False):
        sid = str(getattr(row, "store_id", ""))
        g = hist.get(sid)
        if g is None or len(g) < 1:
            continue
        vols = g["volume_mt"].to_numpy(dtype=float)
        dates = pd.to_datetime(g["sale_date"])
        city = str(getattr(row, "city", "") or "")
        dsr = str(getattr(row, "dsr_name", "") or "")
        ams = float(getattr(row, "ams_3m", 0) or 0)
        billed = float(getattr(row, "billed_mt", 0) or 0)
        last_month = float(getattr(row, "last_month_mt", 0) or 0)
        if last_month <= 0:
            last_month = month_vol.get((sid, shift_period(period, -1)), 0.0)
        feat = _features(
            vols,
            dates,
            as_of=as_of,
            ams=ams,
            last_month=last_month,
            billed_mtd=billed,
            city_p50=city_p50.get(city, 0.0),
            dsr_p50=dsr_p50.get(dsr, 0.0),
        )
        feat["store_id"] = sid
        rows.append(feat)
    return pd.DataFrame(rows)


def _predict(model: Any, scored: pd.DataFrame) -> pd.DataFrame:
    X = scored[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    raw = np.asarray(model.predict(X), dtype=float)
    pred = np.expm1(np.clip(raw, 0, None))
    return pd.DataFrame({"store_id": scored["store_id"].astype(str), "pred_next_drop_mt": pred})
