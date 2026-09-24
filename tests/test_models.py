"""One Expected, explainable rules, deterministic segments."""

from __future__ import annotations

import pandas as pd

from sndintel.features import build_features
from sndintel.materiality import unit_material_mt
from sndintel.models import BASELINE_MODEL, cluster_shops, detect_anomalies, forecast_shop_month
from sndintel.season import fit_shop_expected


def _panel() -> pd.DataFrame:
    periods = [f"2026-{m:02d}" for m in range(1, 8)]
    rows = []
    # Steady door: 0.20 every month, then loads 0.70 in July.
    for i, p in enumerate(periods):
        rows.append(_row("S1", p, 0.70 if p == "2026-07" else 0.20, i))
    # Regular door that goes quiet in July.
    for i, p in enumerate(periods):
        rows.append(_row("Q1", p, 0.0 if p == "2026-07" else 0.15, i))
    # Door that fell to a third of its run-rate in July.
    for i, p in enumerate(periods):
        rows.append(_row("D1", p, 0.05 if p == "2026-07" else 0.30, i))
    # Occasional tiny biller: never material, never flagged.
    for i, p in enumerate(periods):
        rows.append(_row("T1", p, 0.004 if i % 2 else 0.0, i))
    # A dozen filler doors so segment medians are meaningful.
    for k in range(12):
        for i, p in enumerate(periods):
            rows.append(_row(f"F{k:02d}", p, 0.10 + 0.01 * k, i))
    return pd.DataFrame(rows)


def _row(sid: str, period: str, vol: float, i: int) -> dict:
    return {
        "store_id": sid,
        "store_name": f"Shop {sid}",
        "period": period,
        "year": int(period[:4]),
        "month": int(period[5:7]),
        "volume_mt": vol,
        "billed": int(vol > 0),
        "sku_count": 2 if vol > 0 else 0,
        "city": "Karachi",
        "distributor": "Coastal",
        "dsr_name": "Nadia",
        "section": "Saddar",
    }


def test_forecast_is_the_pack_expected_not_a_second_model():
    sm = _panel()
    feats = build_features(sm, pd.DataFrame())
    fc = forecast_shop_month(feats, sm)
    assert not fc.empty
    assert set(fc["model"]) == {BASELINE_MODEL}
    july = fc[fc["period"] == "2026-07"].set_index("entity_id")
    official = fit_shop_expected(sm, "2026-07", None, 1.0).set_index("store_id")["expected_full_mt"]
    for sid in ("S1", "Q1", "D1"):
        assert abs(float(july.loc[sid, "predicted"]) - float(official.loc[sid])) < 1e-9
    # First month on file has no baseline and is not scored.
    assert "2026-01" not in set(fc["period"])


def test_anomaly_rules_name_themselves_and_use_the_run_rate_expected():
    sm = _panel()
    feats = build_features(sm, pd.DataFrame())
    out = detect_anomalies(feats, sm, "2026-07", mtd_open=False)
    kinds = out.set_index("store_id")["kind"].to_dict()
    assert kinds["S1"] == "trade_loading"
    assert kinds["Q1"] in {"drop_off", "quiet_month"}
    assert kinds["D1"] == "drop_off"
    assert "T1" not in kinds  # 4 kg doors are not anomalies
    assert "statistical_outlier" not in set(out["kind"])
    official = fit_shop_expected(sm, "2026-07", None, 1.0).set_index("store_id")["expected_full_mt"]
    row = out.set_index("store_id").loc["S1"]
    assert abs(float(row["expected_mt"]) - float(official.loc["S1"])) < 1e-9
    # Open MTD: a quiet shop may still bill, so no drop-off / quiet flags.
    open_out = detect_anomalies(feats, sm, "2026-07", mtd_open=True)
    open_kinds = set(open_out["kind"]) if not open_out.empty else set()
    assert not open_kinds & {"drop_off", "quiet_month"}
    assert "trade_loading" in open_kinds


def test_segments_are_deterministic_and_a_shop_billed_this_month_is_not_churn_risk():
    sm = _panel()
    feats = build_features(sm, pd.DataFrame())
    a = cluster_shops(feats, sm, "2026-07")
    b = cluster_shops(feats, sm, "2026-07")
    pd.testing.assert_frame_equal(a, b)
    seg = a.set_index("store_id")["segment"]
    assert seg["S1"] != "Churn Risk"
    assert seg["F00"] != "Churn Risk"
    # A steady 0.11 MT door in a flat market is not "Declining Core" just because
    # every door dipped together — trend is judged against the market median.
    assert seg["F01"] in {"Stable Core", "Star Account", "Growth Target", "Long Tail"}
    codes = a.set_index("segment")["cluster_id"].to_dict()
    assert len(set(codes.values())) == len(codes)


def test_unit_materiality_is_relative_with_a_floor():
    assert unit_material_mt(0) == 0.15
    assert unit_material_mt(1.3) == 0.15  # 5% would be 65 kg; the floor holds
    assert abs(unit_material_mt(4.0) - 0.20) < 1e-9
    assert abs(unit_material_mt(10.0) - 0.50) < 1e-9
    assert abs(unit_material_mt(20.0) - 0.50) < 1e-9
    assert abs(unit_material_mt(100.0) - 2.0) < 1e-9
    assert unit_material_mt(None) == 0.15
    # Continuous: no jump around the old 8 MT switch.
    assert unit_material_mt(7.99) <= unit_material_mt(8.01) + 1e-9
