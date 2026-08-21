"""End-to-end ingest → features → models → insights."""

from __future__ import annotations

from calendar import monthrange
from pathlib import Path
from typing import Optional
import os

import pandas as pd

from sndintel.config import DATA_DIR, DB_PATH, PROCESSED_DIR, ensure_dirs
from sndintel.features import add_calendar_panel, build_features, latest_period, rebuild_shop_month
from sndintel.ingest.daily import overlay_store_attrs
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import collapse_sales_facts, parse_sales_file
from sndintel.ingest.universe import fill_zone_from_legacy, parse_universe
from sndintel.ingest.visits import parse_visit_calls
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
    sales_path: Optional[str | Path] = None,
    shop_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
    visits_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
) -> dict:
    """Ingest sales (optional if warehouse already has facts), live universe, visits, then rescore."""
    ensure_dirs()
    db_path = Path(db_path or DB_PATH)
    init_db(db_path)
    started = utcnow()
    sales_path = Path(sales_path) if sales_path else None
    shop_path = Path(shop_path) if shop_path else None
    universe_path = Path(universe_path) if universe_path else None
    visits_path = Path(visits_path) if visits_path else None

    sales, sales_report = (pd.DataFrame(), None)
    if sales_path:
        sales, sales_report = parse_sales_file(sales_path)
    shops = pd.DataFrame()
    shop_report = None
    if shop_path:
        shops, shop_report = parse_shop_master(shop_path)
    universe = pd.DataFrame()
    universe_report = None
    if universe_path:
        universe, universe_report = parse_universe(universe_path)
    visits = pd.DataFrame()
    visit_report = None
    if visits_path:
        visits, visit_report = parse_visit_calls(visits_path)

    touched_periods: list[str] = []
    open_period: Optional[str] = None
    execution = None

    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO pipeline_runs (started_at, status, sales_file, shop_file)
               VALUES (?, 'running', ?, ?)""",
            (
                started,
                str(sales_path) if sales_path else None,
                str(universe_path or shop_path) if (universe_path or shop_path) else None,
            ),
        )
        run_id = int(cur.lastrowid)

        existing_stores = read_sql(conn, "SELECT * FROM stores")
        legacy_for_zone = shops if not shops.empty else existing_stores
        if not universe.empty:
            universe = fill_zone_from_legacy(universe, legacy_for_zone)
            conn.execute("UPDATE stores SET in_universe = 0")
            store_rows = universe.copy()
            store_rows["in_universe"] = 1
            store_rows["source"] = "universe"
            store_rows["extra_json"] = None
            store_rows["updated_at"] = utcnow()
            _upsert_stores(conn, store_rows)
        elif not shops.empty:
            store_rows = shops.copy()
            store_rows["in_universe"] = 1
            store_rows["source"] = "legacy_master"
            store_rows["extra_json"] = None
            store_rows["updated_at"] = utcnow()
            _upsert_stores(conn, store_rows)

        if not sales.empty:
            live_book = read_sql(conn, "SELECT * FROM stores")
            if not shops.empty:
                extra = shops.copy()
                if "in_universe" not in extra.columns:
                    extra["in_universe"] = 0
                live_book = pd.concat([live_book, extra], ignore_index=True)
            sales = overlay_store_attrs(sales, live_book)
            discovered = (
                sales.groupby("store_id", as_index=False)
                .agg(
                    store_name=("store_name", "last"),
                    distributor=("distributor", "last"),
                    dsr_name=("dsr_name", "last"),
                    section=("section", "last"),
                )
            )
            existing = set(read_sql(conn, "SELECT store_id FROM stores")["store_id"].astype(str))
            live = read_sql(conn, "SELECT store_id FROM stores WHERE in_universe = 1")
            has_universe = not live.empty
            new_ids = discovered[~discovered["store_id"].astype(str).isin(existing)].copy()
            if not new_ids.empty and not has_universe:
                for col in ("zone", "city", "category_1", "category_2", "category_3", "category_4"):
                    new_ids[col] = None
                new_ids["in_universe"] = 0
                new_ids["source"] = "sales"
                new_ids["extra_json"] = None
                new_ids["updated_at"] = utcnow()
                _upsert_stores(conn, new_ids)

            facts = collapse_sales_facts(sales.copy())
            facts["source_file"] = sales_path.name
            facts["ingested_at"] = utcnow()
            touched_periods = sorted(facts["period"].dropna().unique().tolist())
            # Snapshot replace: the extract is the new truth for every month it contains.
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
            execution = parse_execution_date(sales_report.params if sales_report else None)
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

        if not visits.empty:
            if visits["period"].isna().all() or (visits["period"].astype(str) == "None").all():
                fallback = open_period or (max(touched_periods) if touched_periods else None)
                if not fallback:
                    sm_p = read_sql(conn, "SELECT MAX(period) AS p FROM shop_month")
                    fallback = str(sm_p.iloc[0]["p"]) if not sm_p.empty and sm_p.iloc[0]["p"] else None
                if fallback:
                    visits["period"] = fallback
                    if visit_report is not None:
                        visit_report.warnings.append(f"Visit period set to {fallback} from sales/warehouse")
            vis_periods = sorted(visits["period"].dropna().astype(str).unique().tolist())
            if vis_periods:
                placeholders = ", ".join("?" * len(vis_periods))
                conn.execute(
                    f"DELETE FROM shop_visits WHERE period IN ({placeholders})",
                    tuple(vis_periods),
                )
            vis_rows = visits.copy()
            vis_rows["source_file"] = visits_path.name if visits_path else None
            vis_rows["ingested_at"] = utcnow()
            upsert_dataframe(
                conn,
                "shop_visits",
                vis_rows[
                    [
                        c
                        for c in [
                            "store_id",
                            "period",
                            "visits",
                            "distributor",
                            "dsr_name",
                            "city",
                            "section",
                            "store_name",
                            "source_file",
                            "ingested_at",
                        ]
                        if c in vis_rows.columns
                    ]
                ],
                ["store_id", "period"],
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
        dest = None
        if sales_path is not None:
            stamp = started.replace(":", "").replace("-", "")
            dest = PROCESSED_DIR / f"{stamp}_{sales_path.name}"
            try:
                Path(dest).write_bytes(Path(sales_path).read_bytes())
            except OSError:
                dest = None

        notes = {
            "sales_strategy": sales_report.strategy if sales_report else None,
            "sales_clean_rows": sales_report.n_clean_rows if sales_report else None,
            "sales_warnings": sales_report.warnings if sales_report else [],
            "shop_strategy": shop_report.strategy if shop_report else None,
            "shop_clean_rows": shop_report.n_clean_rows if shop_report else None,
            "universe_strategy": universe_report.strategy if universe_report else None,
            "universe_clean_rows": universe_report.n_clean_rows if universe_report else None,
            "visit_strategy": visit_report.strategy if visit_report else None,
            "visit_clean_rows": visit_report.n_clean_rows if visit_report else None,
            "replaced_periods": touched_periods,
            "open_mtd_period": open_period,
        }
        warnings = []
        for rep in (sales_report, shop_report, universe_report, visit_report):
            if rep is not None:
                warnings.extend(rep.warnings or [])
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
        "parser": (sales_report.strategy if sales_report else None) or (universe_report.strategy if universe_report else "rescore"),
        "processed_copy": str(dest) if dest else None,
        "warnings": warnings,
        "replaced_periods": touched_periods,
        "open_mtd_period": open_period,
        "n_plays": int(len(plays)) if plays is not None else 0,
        "n_cities": int(pack.national.get("n_cities") or 0) if pack is not None else 0,
        "n_targets": int(len(pack.targets)) if pack is not None and pack.targets is not None else 0,
        "n_universe": int(universe_report.n_clean_rows) if universe_report else None,
        "n_visits": int(visit_report.n_clean_rows) if visit_report else None,
        "exec_ok": bool(scored.get("exec_ok")),
        "exec_error": scored.get("exec_error") or "",
        "exec_model": scored.get("exec_model") or "",
        "data_dir": str(DATA_DIR),
    }


SALES_FACT_COLS = [
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


STORE_UPSERT_COLS = [
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
    "source",
    "extra_json",
    "updated_at",
]


def _upsert_stores(conn, store_rows: pd.DataFrame) -> None:
    if store_rows is None or store_rows.empty:
        return
    work = store_rows.copy()
    for col in STORE_UPSERT_COLS:
        if col not in work.columns:
            work[col] = None if col not in {"in_universe"} else 1
    upsert_dataframe(conn, "stores", work[STORE_UPSERT_COLS], ["store_id"])


def _active_universe(stores_df: pd.DataFrame) -> pd.DataFrame:
    if stores_df is None or stores_df.empty:
        return stores_df if stores_df is not None else pd.DataFrame()
    if "in_universe" in stores_df.columns and (stores_df["in_universe"] == 1).any():
        return stores_df[stores_df["in_universe"] == 1].copy()
    return stores_df


def _rebuild_intelligence(conn, run_id: int) -> dict:
    """Rebuild features, insights, and the city→shop hierarchy from warehouse facts."""
    stores_all = read_sql(conn, "SELECT * FROM stores")
    facts_all = read_sql(conn, "SELECT * FROM sales_facts")
    if facts_all is not None and not facts_all.empty:
        n_before = len(facts_all)
        vol_before = float(pd.to_numeric(facts_all["volume_mt"], errors="coerce").fillna(0).sum())
        cleaned = collapse_sales_facts(facts_all)
        vol_after = (
            float(pd.to_numeric(cleaned["volume_mt"], errors="coerce").fillna(0).sum())
            if cleaned is not None and not cleaned.empty
            else 0.0
        )
        if len(cleaned) != n_before or abs(vol_after - vol_before) > 1e-6:
            # Duplicate SKU lines were stored as separate keys (whitespace, copies).
            # Persist the collapsed facts so SKU mix and billed are not still 2×.
            conn.execute("DELETE FROM sales_facts")
            cols = [c for c in SALES_FACT_COLS if c in cleaned.columns]
            if cols and not cleaned.empty:
                upsert_dataframe(conn, "sales_facts", cleaned[cols], ["store_id", "sku", "period"])
        facts_all = cleaned
    stores_df = _active_universe(stores_all)
    facts_df = facts_all
    if stores_df is not None and not stores_df.empty and stores_all is not None and len(stores_df) < len(stores_all):
        ids = set(stores_df["store_id"].astype(str))
        if not facts_all.empty:
            facts_df = facts_all[facts_all["store_id"].astype(str).isin(ids)].copy()
    try:
        visits_df = read_sql(conn, "SELECT * FROM shop_visits")
    except Exception:
        visits_df = pd.DataFrame()
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
        visits=visits_df,
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

    exec_meta = _write_national_exec(conn, shop_month, visits_df, ledger, pack.period)

    return {
        "latest_period": period,
        "n_sales_rows": int(len(facts_df)),
        "n_stores": int(len(stores_df)),
        "n_insights": int(len(insights)) if insights is not None else 0,
        "n_anomalies": int(len(anomalies)) if anomalies is not None else 0,
        "n_plays": int(len(plays)) if plays is not None else 0,
        "n_cities": int(pack.national.get("n_cities") or 0) if pack.national else 0,
        "n_targets": int(len(pack.targets)) if pack.targets is not None else 0,
        "exec_ok": bool(exec_meta.get("ok")),
        "exec_error": exec_meta.get("error") or "",
        "exec_model": exec_meta.get("model") or "",
        "_facts": facts_df,
        "_stores": stores_df,
        "_insights": insights,
        "_anomalies": anomalies,
        "_plays": plays,
        "_pack": pack,
    }


def _write_national_exec(conn, shop_month, visits_df, ledger, period) -> dict:
    """Rebuild the national pack and ask the model for the executive summary.

    Missing keys or API failures are stored as exec_error. Ingest still succeeds.
    """
    from sndintel.briefing import build_strategy_pack
    from sndintel.narrative import refresh_exec_summary

    if not period:
        return {"ok": False, "error": "No period to summarise.", "skipped": True}
    if "PYTEST_CURRENT_TEST" in os.environ:
        from sndintel.narrative import store_exec_summary

        store_exec_summary(
            conn,
            period,
            {
                "ok": False,
                "error": "OpenAI skipped under pytest",
                "model": "",
                "brief": {},
                "situation": [],
                "focus": [],
                "raw": "",
            },
        )
        return {"ok": False, "error": "OpenAI skipped under pytest", "skipped": True}
    units = read_sql(conn, "SELECT * FROM unit_scorecards")
    try:
        situation = read_sql(conn, "SELECT * FROM situation_brief")
    except Exception:
        situation = pd.DataFrame()
    report = build_strategy_pack(
        units,
        shop_month,
        situation=situation,
        ledger=ledger,
        period=period,
        visits=visits_df,
    )
    return refresh_exec_summary(conn, report)


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
        "exec_ok": bool(scored.get("exec_ok")),
        "exec_error": scored.get("exec_error") or "",
        "exec_model": scored.get("exec_model") or "",
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
