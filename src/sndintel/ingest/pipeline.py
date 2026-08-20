"""End-to-end ingest → features → models → insights."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from sndintel.config import DB_PATH, PROCESSED_DIR, ensure_dirs
from sndintel.features import add_calendar_panel, build_features, latest_period, rebuild_shop_month
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import parse_sales_file
from sndintel.insights import compile_insights
from sndintel.models import cluster_shops, detect_anomalies, forecast_shop_month
from sndintel.storage import (
    connect,
    init_db,
    read_sql,
    replace_table,
    upsert_dataframe,
    utcnow,
)


def run_pipeline(
    sales_path: str | Path,
    shop_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
) -> dict:
    ensure_dirs()
    db_path = Path(db_path or DB_PATH)
    init_db(db_path)
    sales_path = Path(sales_path)
    started = utcnow()

    sales, sales_report = parse_sales_file(sales_path)
    shops = pd.DataFrame()
    shop_report = None
    if shop_path:
        shops, shop_report = parse_shop_master(shop_path)

    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO pipeline_runs (started_at, status, sales_file, shop_file)
               VALUES (?, 'running', ?, ?)""",
            (started, str(sales_path), str(shop_path) if shop_path else None),
        )
        run_id = int(cur.lastrowid)

        if not shops.empty:
            store_rows = shops.copy()
            store_rows["in_universe"] = 1
            store_rows["extra_json"] = None
            store_rows["updated_at"] = utcnow()
            upsert_dataframe(
                conn,
                "stores",
                store_rows[
                    [
                        "store_id",
                        "store_name",
                        "distributor",
                        "dsr_name",
                        "zone",
                        "city",
                        "section",
                        "category_1",
                        "category_2",
                        "category_3",
                        "category_4",
                        "in_universe",
                        "extra_json",
                        "updated_at",
                    ]
                ],
                ["store_id"],
            )

        if not sales.empty:
            discovered = (
                sales.groupby("store_id", as_index=False)
                .agg(
                    store_name=("store_name", "last"),
                    distributor=("distributor", "last"),
                    dsr_name=("dsr_name", "last"),
                    section=("section", "last"),
                )
            )
            existing = set(read_sql(conn, "SELECT store_id FROM stores")["store_id"])
            new_ids = discovered[~discovered["store_id"].isin(existing)].copy()
            if not new_ids.empty:
                for col in ("zone", "city", "category_1", "category_2", "category_3", "category_4"):
                    new_ids[col] = None
                new_ids["in_universe"] = 0
                new_ids["extra_json"] = None
                new_ids["updated_at"] = utcnow()
                upsert_dataframe(
                    conn,
                    "stores",
                    new_ids[
                        [
                            "store_id",
                            "store_name",
                            "distributor",
                            "dsr_name",
                            "zone",
                            "city",
                            "section",
                            "category_1",
                            "category_2",
                            "category_3",
                            "category_4",
                            "in_universe",
                            "extra_json",
                            "updated_at",
                        ]
                    ],
                    ["store_id"],
                )

            facts = sales.copy()
            facts["source_file"] = sales_path.name
            facts["ingested_at"] = utcnow()
            upsert_dataframe(
                conn,
                "sales_facts",
                facts[
                    [
                        "store_id",
                        "sku",
                        "period",
                        "year",
                        "month",
                        "volume_mt",
                        "distributor",
                        "dsr_name",
                        "section",
                        "store_name",
                        "source_file",
                        "ingested_at",
                    ]
                ],
                ["store_id", "sku", "period"],
            )

        stores_df = read_sql(conn, "SELECT * FROM stores")
        facts_df = read_sql(conn, "SELECT * FROM sales_facts")
        shop_month = rebuild_shop_month(facts_df, stores_df)
        shop_month = add_calendar_panel(shop_month, stores_df)
        shop_cols = [
            "store_id",
            "period",
            "year",
            "month",
            "volume_mt",
            "sku_count",
            "billed",
            "distributor",
            "dsr_name",
            "section",
            "store_name",
            "zone",
            "city",
        ]
        replace_table(conn, "shop_month", shop_month[shop_cols])

        feats = build_features(shop_month, facts_df)
        replace_table(conn, "features_shop_month", feats)

        period = latest_period(shop_month)
        forecasts = forecast_shop_month(feats, shop_month)
        replace_table(conn, "forecasts", forecasts)

        anomalies = detect_anomalies(feats, shop_month, period) if period else pd.DataFrame()
        replace_table(conn, "anomalies", anomalies)

        segments = cluster_shops(feats, shop_month, period) if period else pd.DataFrame()
        replace_table(conn, "shop_segments", segments)

        insights, kpis = compile_insights(
            run_id, facts_df, stores_df, shop_month, feats, forecasts, anomalies, segments
        )
        conn.execute("DELETE FROM insights")
        if not insights.empty:
            insights.to_sql("insights", conn, if_exists="append", index=False)
        replace_table(conn, "kpi_snapshots", kpis)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = started.replace(":", "").replace("-", "")
        dest = PROCESSED_DIR / f"{stamp}_{sales_path.name}"
        try:
            Path(dest).write_bytes(Path(sales_path).read_bytes())
        except OSError:
            dest = None

        notes = {
            "sales_strategy": sales_report.strategy,
            "sales_clean_rows": sales_report.n_clean_rows,
            "sales_warnings": sales_report.warnings,
            "shop_strategy": shop_report.strategy if shop_report else None,
            "shop_clean_rows": shop_report.n_clean_rows if shop_report else None,
        }
        conn.execute(
            """UPDATE pipeline_runs
               SET finished_at=?, status=?, n_sales_rows=?, n_stores=?, latest_period=?, notes=?
               WHERE run_id=?""",
            (
                utcnow(),
                "success",
                int(len(facts_df)),
                int(len(stores_df)),
                period,
                pd.Series(notes).to_json(),
                run_id,
            ),
        )

    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "latest_period": period,
        "n_sales_rows": int(len(facts_df)),
        "n_stores": int(len(stores_df)),
        "n_insights": int(len(insights)) if insights is not None else 0,
        "n_anomalies": int(len(anomalies)) if anomalies is not None else 0,
        "parser": sales_report.strategy,
        "processed_copy": str(dest) if dest else None,
        "warnings": sales_report.warnings,
    }


def load_brief(db_path: Optional[str | Path] = None, limit: int = 25) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return read_sql(
            conn,
            "SELECT * FROM insights ORDER BY rank_score DESC LIMIT ?",
            (limit,),
        )


def load_kpis(db_path: Optional[str | Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return read_sql(conn, "SELECT * FROM kpi_snapshots")
