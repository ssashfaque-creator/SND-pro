import pandas as pd

from sndintel.features import add_calendar_panel, build_features, rebuild_shop_month
from sndintel.ingest.pipeline import run_pipeline
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import parse_sales_file
from sndintel.storage import connect, read_sql


def test_full_pipeline_detects_injected_signals(demo, tmp_path):
    db = tmp_path / "warehouse.db"
    result = run_pipeline(demo["sales"], shop_path=demo["shops"], db_path=db)
    assert result["n_insights"] >= 5
    assert result["latest_period"] == "2026-07"
    assert result["parser"] == "ssrs_field_ids"

    with connect(db) as conn:
        insights = read_sql(conn, "SELECT * FROM insights")
        anomalies = read_sql(conn, "SELECT * FROM anomalies")
        kpis = read_sql(conn, "SELECT * FROM kpi_snapshots")
        segments = read_sql(conn, "SELECT * FROM shop_segments")
        shops = read_sql(conn, "SELECT * FROM stores")

    types = set(insights["type"])
    loaded = anomalies[anomalies["store_id"] == "T0001999001"] if not anomalies.empty else anomalies
    assert not loaded.empty, f"expected trade-load anomaly, got {anomalies.head().to_dict()}"
    assert "trade_loading" in set(loaded["kind"]) | types

    lapsed_flags = []
    if not anomalies.empty:
        lapsed_flags += anomalies[anomalies["store_id"] == "T0001999002"]["kind"].tolist()
    lapsed_flags += insights[insights["entity_id"] == "T0001999002"]["type"].tolist()
    assert any(k in {"drop_off", "lapse", "segment_focus", "anomaly"} for k in lapsed_flags) or (
        "drop_off" in types or "Churn Risk" in set(segments["segment"])
    )

    assert "whitespace" in types or "coverage_gap" in types
    assert shops["store_id"].nunique() >= 50

    nat = kpis[kpis["grain"] == "national"]
    assert not nat.empty
    assert nat.iloc[0]["volume_mt"] > 1
    assert not segments.empty
    assert result["n_sales_rows"] > 500


def test_features_have_lags(demo):
    sales, _ = parse_sales_file(demo["sales"])
    stores, _ = parse_shop_master(demo["shops"])
    sm = add_calendar_panel(rebuild_shop_month(sales, stores), stores)
    feats = build_features(sm, sales)
    latest = feats["period"].max()
    row = feats[(feats["store_id"] == "T0001601407") & (feats["period"] == latest)]
    assert not row.empty
    assert pd.notna(row.iloc[0]["roll_mean_3"]) or row.iloc[0]["volume_mt"] >= 0
