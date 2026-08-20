"""End-to-end ingest → features → models → insights."""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import Optional

import pandas as pd

from sndintel.config import DATA_DIR, DB_PATH, PROCESSED_DIR, ensure_dirs
from sndintel.features import add_calendar_panel, build_features, latest_period, rebuild_shop_month
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import parse_sales_file
from sndintel.hierarchy import HierarchyPack, build_hierarchy_pack, plays_from_pack
from sndintel.insights import compile_insights
from sndintel.models import cluster_shops, detect_anomalies, forecast_shop_month
from sndintel.mtd import open_mtd_period, parse_execution_date, run_rate_factor
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

    touched_periods: list[str] = []
    open_period: Optional[str] = None
    execution = None

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
            touched_periods = sorted(facts["period"].dropna().unique().tolist())
            # Snapshot replace: the extract is the new truth for every month it contains.
            # An August-only file updates August MTD and leaves July (and older) untouched.
            if touched_periods:
                placeholders = ", ".join("?" * len(touched_periods))
                conn.execute(
                    f"DELETE FROM sales_facts WHERE period IN ({placeholders})",
                    tuple(touched_periods),
                )
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
            execution = parse_execution_date(sales_report.params)
            open_period = open_mtd_period(touched_periods, execution)
            for per in touched_periods:
                part = facts[facts["period"] == per]
                status = "mtd_open" if per == open_period else "closed"
                as_of = days = None
                if status == "mtd_open":
                    _factor, as_of, days = run_rate_factor(execution, per)
                else:
                    year, month = int(str(per)[:4]), int(str(per)[5:7])
                    days = monthrange(year, month)[1]
                    as_of = days
                conn.execute(
                    """INSERT INTO period_ledger
                       (period, status, source_file, execution_date, ingested_at, n_fact_rows, volume_mt, as_of_day, days_in_month)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(period) DO UPDATE SET
                         status=excluded.status,
                         source_file=excluded.source_file,
                         execution_date=excluded.execution_date,
                         ingested_at=excluded.ingested_at,
                         n_fact_rows=excluded.n_fact_rows,
                         volume_mt=excluded.volume_mt,
                         as_of_day=excluded.as_of_day,
                         days_in_month=excluded.days_in_month
                    """,
                    (
                        per,
                        status,
                        sales_path.name,
                        execution.strftime("%Y-%m-%d") if execution else None,
                        utcnow(),
                        int(len(part)),
                        float(part["volume_mt"].sum()),
                        as_of,
                        days,
                    ),
                )
                if as_of and days:
                    conn.execute(
                        """INSERT INTO mtd_observations
                           (period, as_of_day, days_in_month, volume_mt, source_file, ingested_at)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(period, as_of_day) DO UPDATE SET
                             volume_mt=excluded.volume_mt,
                             days_in_month=excluded.days_in_month,
                             source_file=excluded.source_file,
                             ingested_at=excluded.ingested_at
                        """,
                        (
                            per,
                            int(as_of),
                            int(days),
                            float(part["volume_mt"].sum()),
                            sales_path.name,
                            utcnow(),
                        ),
                    )

        scored = _rebuild_intelligence(conn, run_id)
        period = scored["latest_period"]
        facts_df = scored["_facts"]
        stores_df = scored["_stores"]
        insights = scored["_insights"]
        anomalies = scored["_anomalies"]
        plays = scored["_plays"]
        pack = scored["_pack"]

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
            "replaced_periods": touched_periods,
            "open_mtd_period": open_period,
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
        "replaced_periods": touched_periods,
        "open_mtd_period": open_period,
        "n_plays": int(len(plays)) if plays is not None else 0,
        "n_cities": int(pack.national.get("n_cities") or 0) if pack is not None else 0,
        "n_targets": int(len(pack.targets)) if pack is not None and pack.targets is not None else 0,
        "data_dir": str(DATA_DIR),
    }


SHOP_MONTH_COLS = [
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


def _rebuild_intelligence(conn, run_id: int) -> dict:
    """Rebuild features, insights, and the city→shop hierarchy from warehouse facts."""
    stores_df = read_sql(conn, "SELECT * FROM stores")
    facts_df = read_sql(conn, "SELECT * FROM sales_facts")
    shop_month = rebuild_shop_month(facts_df, stores_df)
    shop_month = add_calendar_panel(shop_month, stores_df)
    if not shop_month.empty:
        replace_table(conn, "shop_month", shop_month[SHOP_MONTH_COLS])
    else:
        replace_table(conn, "shop_month", pd.DataFrame(columns=SHOP_MONTH_COLS))

    feats = build_features(shop_month, facts_df)
    replace_table(conn, "features_shop_month", feats)

    period = latest_period(shop_month)
    forecasts = forecast_shop_month(feats, shop_month)
    replace_table(conn, "forecasts", forecasts)

    ledger = read_sql(conn, "SELECT * FROM period_ledger")
    latest_open = None
    if not ledger.empty and "status" in ledger.columns:
        open_rows = ledger[ledger["status"] == "mtd_open"]
        if not open_rows.empty:
            latest_open = str(open_rows["period"].max())
    anomalies = (
        detect_anomalies(feats, shop_month, period, mtd_open=bool(period and period == latest_open))
        if period
        else pd.DataFrame()
    )
    replace_table(conn, "anomalies", anomalies)

    segments = cluster_shops(feats, shop_month, period) if period else pd.DataFrame()
    replace_table(conn, "shop_segments", segments)

    insights, kpis = compile_insights(
        run_id,
        facts_df,
        stores_df,
        shop_month,
        feats,
        forecasts,
        anomalies,
        segments,
        ledger=ledger,
    )
    conn.execute("DELETE FROM insights")
    if not insights.empty:
        insights.to_sql("insights", conn, if_exists="append", index=False)
    replace_table(conn, "kpi_snapshots", kpis)

    pack = build_hierarchy_pack(
        shop_month,
        stores_df,
        feats,
        ledger,
        facts=facts_df,
        mtd_obs=read_sql(conn, "SELECT * FROM mtd_observations"),
    )
    replace_table(conn, "unit_scorecards", pack.units)
    replace_table(conn, "focus_targets", pack.targets)
    season_df = pack.seasonality if pack.seasonality is not None else pd.DataFrame()
    if season_df is not None and not season_df.empty and "period" not in season_df.columns:
        season_df = season_df.copy()
        season_df["period"] = pack.period
    replace_table(conn, "seasonality_index", season_df)
    sit = pd.DataFrame(
        [
            {
                "period": pack.period,
                "headline": pack.national.get("headline"),
                "weather": pack.national.get("weather"),
                "problem": pack.national.get("problem"),
                "action_summary": pack.national.get("action_summary"),
                "metrics_json": pd.Series(pack.national).to_json(),
            }
        ]
    ) if pack.national else pd.DataFrame()
    if not sit.empty:
        replace_table(conn, "situation_brief", sit)
    plays = plays_from_pack(run_id, pack)
    conn.execute("DELETE FROM strategy_plays")
    if not plays.empty:
        plays.to_sql("strategy_plays", conn, if_exists="append", index=False)

    return {
        "latest_period": period,
        "n_sales_rows": int(len(facts_df)),
        "n_stores": int(len(stores_df)),
        "n_insights": int(len(insights)) if insights is not None else 0,
        "n_anomalies": int(len(anomalies)) if anomalies is not None else 0,
        "n_plays": int(len(plays)) if plays is not None else 0,
        "n_cities": int(pack.national.get("n_cities") or 0) if pack.national else 0,
        "n_targets": int(len(pack.targets)) if pack.targets is not None else 0,
        "_facts": facts_df,
        "_stores": stores_df,
        "_insights": insights,
        "_anomalies": anomalies,
        "_plays": plays,
        "_pack": pack,
    }


def rescore_warehouse(db_path: Optional[str | Path] = None) -> dict:
    """Rebuild scorecards from facts already in the warehouse. No file upload."""
    ensure_dirs()
    db_path = Path(db_path or DB_PATH)
    init_db(db_path)
    started = utcnow()
    pack = HierarchyPack(period="", yoy_period="", mtd={}, national={})
    with connect(db_path) as conn:
        facts = read_sql(conn, "SELECT 1 AS x FROM sales_facts LIMIT 1")
        if facts.empty:
            raise ValueError("Warehouse has no sales facts yet. Upload a Shop SKU Wise extract first.")
        cur = conn.execute(
            """INSERT INTO pipeline_runs (started_at, status, sales_file, shop_file)
               VALUES (?, 'running', ?, ?)""",
            (started, "rescore", None),
        )
        run_id = int(cur.lastrowid)
        scored = _rebuild_intelligence(conn, run_id)
        pack = scored["_pack"]
        notes = {"rescore": True, "n_cities": scored["n_cities"], "n_targets": scored["n_targets"]}
        conn.execute(
            """UPDATE pipeline_runs
               SET finished_at=?, status=?, n_sales_rows=?, n_stores=?, latest_period=?, notes=?
               WHERE run_id=?""",
            (
                utcnow(),
                "success",
                scored["n_sales_rows"],
                scored["n_stores"],
                scored["latest_period"],
                pd.Series(notes).to_json(),
                run_id,
            ),
        )
    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "latest_period": scored["latest_period"],
        "n_sales_rows": scored["n_sales_rows"],
        "n_stores": scored["n_stores"],
        "n_insights": scored["n_insights"],
        "n_anomalies": scored["n_anomalies"],
        "parser": "rescore",
        "processed_copy": None,
        "warnings": [],
        "replaced_periods": [],
        "open_mtd_period": None,
        "n_plays": scored["n_plays"],
        "n_cities": scored["n_cities"],
        "n_targets": int(len(pack.targets)) if pack.targets is not None else 0,
        "data_dir": str(DATA_DIR),
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


def load_ledger(db_path: Optional[str | Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return read_sql(conn, "SELECT * FROM period_ledger ORDER BY period")


def load_plays(db_path: Optional[str | Path] = None) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return read_sql(conn, "SELECT * FROM strategy_plays ORDER BY slot")
