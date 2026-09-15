"""End-to-end ingest → features → models → insights."""

from __future__ import annotations

from calendar import monthrange
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

import pandas as pd

from sndintel.config import DATA_DIR, DB_PATH, PROCESSED_DIR, ensure_dirs
from sndintel.features import add_calendar_panel, build_features, latest_period, rebuild_shop_month
from sndintel.action import build_action_pack, persist_action_pack
from sndintel.ingest.daily import overlay_store_attrs
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.ssrs import ParseReport, collapse_sales_facts, parse_sales_file
from sndintel.ingest.targets import match_shop_targets, parse_shop_targets
from sndintel.ingest.universe import fill_zone_from_legacy, parse_universe
from sndintel.ingest.visits import parse_visit_calls
from sndintel.plan import attach_plan
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


def _normalize_sales_paths(
    sales_path: Optional[str | Path],
    sales_paths: Optional[list[str | Path]],
) -> list[Path]:
    paths: list[Path] = []
    if sales_paths:
        paths.extend(Path(p) for p in sales_paths if p)
    elif sales_path:
        paths.append(Path(sales_path))
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _sales_label(paths: list[Path]) -> str:
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0].name
    return f"{paths[0].name} + {len(paths) - 1} more"


def _parse_sales_paths(paths: list[Path]) -> tuple[pd.DataFrame, ParseReport | None]:
    frames: list[pd.DataFrame] = []
    reports: list[ParseReport] = []
    dailies: list[pd.DataFrame] = []
    for path in paths:
        df, report = parse_sales_file(path)
        frames.append(df)
        reports.append(report)
        daily = getattr(report, "daily", None)
        if daily is not None and not daily.empty:
            dailies.append(daily)
    merged = merge_sales_reports(reports)
    if merged is not None:
        merged.daily = combine_daily_frames(dailies)
    return combine_sales_frames(frames), merged


def combine_daily_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Later files in the same drop win on the same shop-day (override, not add)."""
    parts = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["store_id"] = out["store_id"].astype(str).str.strip()
    out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce")
    out = out.loc[out["sale_date"].notna()].copy()
    key = ["store_id", "sale_date"]
    extra = [c for c in out.columns if c not in key]
    agg = {c: "last" for c in extra}
    grouped = out.groupby(key, as_index=False, sort=False).agg(agg)
    grouped = grouped.loc[pd.to_numeric(grouped["volume_mt"], errors="coerce").fillna(0) > 0].copy()
    return grouped.reset_index(drop=True)


def combine_sales_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Collapse copies inside each file, then add the same POP+SKU+month across files."""
    parts = [collapse_sales_facts(frame.copy()) for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    key = [c for c in ("store_id", "sku", "period") if c in out.columns]
    if len(key) < 3:
        return collapse_sales_facts(out)
    extra = [c for c in out.columns if c not in key]
    agg = {c: "last" for c in extra}
    if "volume_mt" in agg:
        agg["volume_mt"] = "sum"
    grouped = out.groupby(key, as_index=False, sort=False).agg(agg)
    grouped = grouped.loc[pd.to_numeric(grouped["volume_mt"], errors="coerce").fillna(0) > 0].copy()
    return grouped.reset_index(drop=True)


def merge_sales_reports(reports: list[ParseReport]) -> ParseReport | None:
    reports = [r for r in reports if r is not None]
    if not reports:
        return None
    if len(reports) == 1:
        return reports[0]
    best = reports[0]
    best_dt = parse_execution_date(best.params) or datetime.min
    for report in reports[1:]:
        dt = parse_execution_date(report.params)
        if dt and dt > best_dt:
            best_dt = dt
            best = report
    strategies = {r.strategy for r in reports}
    merged = ParseReport(
        strategy=next(iter(strategies)) if len(strategies) == 1 else "combined",
        source_file="; ".join(Path(r.source_file).name for r in reports),
        n_raw_rows=sum(int(r.n_raw_rows or 0) for r in reports),
        n_clean_rows=sum(int(r.n_clean_rows or 0) for r in reports),
        header_row=best.header_row,
        data_start_row=best.data_start_row,
        column_map=dict(best.column_map or {}),
        warnings=[w for r in reports for w in (r.warnings or [])],
        params=dict(best.params or {}),
    )
    merged.params["n_files"] = str(len(reports))
    dates: list[str] = []
    stores: list[str] = []
    for report in reports:
        dates.extend(getattr(report, "daily_dates", None) or [])
        stores.extend(getattr(report, "daily_store_ids", None) or [])
    merged.daily_dates = sorted({str(d)[:10] for d in dates if d})
    merged.daily_store_ids = sorted({str(s).strip() for s in stores if s})
    return merged


def _clear_billed_sales(conn) -> None:
    """Drop billed facts and month ledger. Stores and visits stay."""
    conn.execute("DELETE FROM sales_facts")
    conn.execute("DELETE FROM period_ledger")
    conn.execute("DELETE FROM mtd_observations")
    try:
        conn.execute("DELETE FROM shop_day")
    except Exception:
        pass


def clear_billed_sales(db_path: Optional[str | Path] = None) -> dict:
    """Wipe billed sales only so a new Outlet Date Wise set can replace Shop SKU Wise."""
    ensure_dirs()
    db_path = Path(db_path or DB_PATH)
    init_db(db_path)
    with connect(db_path) as conn:
        stores = read_sql(conn, "SELECT COUNT(*) AS n FROM stores")
        visits = read_sql(conn, "SELECT COUNT(*) AS n FROM shop_visits")
        _clear_billed_sales(conn)
        for table in (
            "shop_month",
            "features_shop_month",
            "forecasts",
            "anomalies",
            "shop_segments",
            "insights",
            "strategy_plays",
            "unit_scorecards",
            "seasonality_index",
            "focus_targets",
            "situation_brief",
            "exec_summary",
            "kpi_snapshots",
            "shop_day",
            "action_shops",
            "action_units",
            "action_backtest",
            "action_brief",
        ):
            conn.execute(f"DELETE FROM {table}")
    return {
        "cleared": "sales",
        "n_stores": int(stores.iloc[0]["n"]) if stores is not None and not stores.empty else 0,
        "n_visits": int(visits.iloc[0]["n"]) if visits is not None and not visits.empty else 0,
        "db_path": str(db_path),
    }


def run_pipeline(
    sales_path: Optional[str | Path] = None,
    shop_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
    visits_path: Optional[str | Path] = None,
    targets_path: Optional[str | Path] = None,
    db_path: Optional[str | Path] = None,
    sales_paths: Optional[list[str | Path]] = None,
    replace_sales: bool = False,
) -> dict:
    """Ingest sales (optional if warehouse already has facts), live universe, visits, then rescore.

    ``sales_paths`` accepts several Outlet Date Wise (or Shop SKU Wise) files in
    one go — split by shops or by date range. Outlet Date Wise **overrides** the
    shop-days present in the files and leaves other days in the warehouse.
    Month totals are rebuilt from the combined daily rows. Shop SKU Wise still
    replaces each calendar month the file contains. ``replace_sales`` wipes
    previous billed rows and the month ledger; stores (universe) and visit
    calls stay.
    """
    ensure_dirs()
    db_path = Path(db_path or DB_PATH)
    init_db(db_path)
    started = utcnow()
    shop_path = Path(shop_path) if shop_path else None
    universe_path = Path(universe_path) if universe_path else None
    visits_path = Path(visits_path) if visits_path else None
    targets_path = Path(targets_path) if targets_path else None
    file_paths = _normalize_sales_paths(sales_path, sales_paths)
    sales_path = file_paths[0] if file_paths else None

    sales, sales_report = (pd.DataFrame(), None)
    if file_paths:
        sales, sales_report = _parse_sales_paths(file_paths)
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
    targets = pd.DataFrame()
    target_report = None
    if targets_path:
        targets, target_report = parse_shop_targets(targets_path)

    touched_periods: list[str] = []
    open_period: Optional[str] = None
    execution = None
    ingest_mode = ""
    overridden_dates: list[str] = []

    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO pipeline_runs (started_at, status, sales_file, shop_file)
               VALUES (?, 'running', ?, ?)""",
            (
                started,
                _sales_label(file_paths) if file_paths else None,
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

            label = _sales_label(file_paths)
            execution = parse_execution_date(sales_report.params if sales_report else None)
            daily = getattr(sales_report, "daily", None) if sales_report else None
            has_daily = daily is not None and not daily.empty
            ingest_mode = "daily_overlay" if has_daily else "month_replace"
            if replace_sales:
                _clear_billed_sales(conn)
            if has_daily:
                daily = overlay_store_attrs(daily, live_book)
                daily = _prepare_shop_day(daily, label)
                coverage_dates = _daily_coverage_dates(sales_report, daily)
                coverage_stores = _daily_coverage_stores(sales_report, daily)
                if coverage_stores and coverage_dates:
                    _delete_shop_days(conn, coverage_stores, coverage_dates)
                if not daily.empty:
                    upsert_dataframe(conn, "shop_day", daily[SHOP_DAY_COLS], ["store_id", "sale_date"])
                touched_periods = _rebuild_facts_from_shop_day(
                    conn,
                    coverage_stores,
                    _periods_from_daily(coverage_dates, daily),
                    label,
                    live_book,
                )
                overridden_dates = coverage_dates
            else:
                facts = collapse_sales_facts(sales.copy())
                facts["source_file"] = label
                facts["ingested_at"] = utcnow()
                touched_periods = sorted(facts["period"].dropna().unique().tolist())
                if not replace_sales and touched_periods:
                    placeholders = ", ".join("?" * len(touched_periods))
                    conn.execute(
                        f"DELETE FROM sales_facts WHERE period IN ({placeholders})",
                        tuple(touched_periods),
                    )
                upsert_dataframe(
                    conn,
                    "sales_facts",
                    facts[[c for c in SALES_FACT_COLS if c in facts.columns]],
                    ["store_id", "sku", "period"],
                )
            open_period = open_mtd_period(touched_periods, execution)
            _refresh_period_ledger(conn, touched_periods, label, execution, open_period)

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

        if not targets.empty:
            live_book = read_sql(conn, "SELECT * FROM stores")
            matched, target_report = match_shop_targets(targets, live_book)
            if target_report is not None:
                target_report.source_file = str(targets_path)
            replace_table(conn, "shop_targets", matched)

        facts_probe = read_sql(conn, "SELECT 1 AS x FROM sales_facts LIMIT 1")
        if (facts_probe is None or facts_probe.empty) and sales.empty:
            conn.execute(
                """UPDATE pipeline_runs SET finished_at=?, status=?, notes=? WHERE run_id=?""",
                (utcnow(), "error", "No sales files and warehouse is empty.", run_id),
            )
            return {
                "ok": False,
                "error": "No sales files and warehouse is empty.",
                "run_id": run_id,
                "db_path": str(db_path),
            }

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
            "target_strategy": target_report.strategy if target_report else None,
            "target_clean_rows": target_report.n_clean_rows if target_report else None,
            "target_matched": target_report.n_matched if target_report else None,
            "replaced_periods": touched_periods,
            "open_mtd_period": open_period,
            "replace_sales": replace_sales,
            "n_sales_files": len(file_paths),
            "ingest_mode": ingest_mode,
            "overridden_dates": overridden_dates,
        }
        warnings = []
        for rep in (sales_report, shop_report, universe_report, visit_report, target_report):
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
        "replace_sales": replace_sales,
        "n_sales_files": len(file_paths),
        "ingest_mode": ingest_mode,
        "overridden_dates": overridden_dates,
        "n_plays": int(len(plays)) if plays is not None else 0,
        "n_cities": int(pack.national.get("n_cities") or 0) if pack is not None else 0,
        "n_targets": int(len(pack.targets)) if pack is not None and pack.targets is not None else 0,
        "n_universe": int(universe_report.n_clean_rows) if universe_report else None,
        "n_visits": int(visit_report.n_clean_rows) if visit_report else None,
        "n_plan_shops": int(target_report.n_clean_rows) if target_report else None,
        "n_plan_matched": int(target_report.n_matched) if target_report else None,
        "plan_book_mt": float(target_report.book_mt) if target_report else None,
        "plan_matched_mt": float(target_report.matched_mt) if target_report else None,
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


SHOP_DAY_COLS = [
    "store_id",
    "sale_date",
    "period",
    "year",
    "month",
    "day",
    "volume_mt",
    "distributor",
    "dsr_name",
    "section",
    "store_name",
    "city",
    "source_file",
    "ingested_at",
]


def _prepare_shop_day(daily: pd.DataFrame, source_file: str) -> pd.DataFrame:
    out = daily.copy()
    out["sale_date"] = pd.to_datetime(out["sale_date"], errors="coerce")
    out = out.loc[out["sale_date"].notna()].copy()
    out["year"] = out["sale_date"].dt.year.astype(int) if "year" not in out.columns or out["year"].isna().any() else out["year"]
    out["month"] = out["sale_date"].dt.month.astype(int)
    out["day"] = out["sale_date"].dt.day.astype(int)
    if "period" not in out.columns or out["period"].isna().any():
        from sndintel.io_utils import period_key

        out["period"] = [period_key(int(y), int(m)) for y, m in zip(out["year"], out["month"])]
    out["sale_date"] = out["sale_date"].dt.strftime("%Y-%m-%d")
    out["source_file"] = source_file
    out["ingested_at"] = utcnow()
    for col in SHOP_DAY_COLS:
        if col not in out.columns:
            out[col] = None
    return out[SHOP_DAY_COLS]


SQL_IN_CHUNK = 400


def _chunked(items: list, size: int = SQL_IN_CHUNK):
    seq = list(items)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _as_day_key(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "T" in text:
        text = text.split("T", 1)[0]
    return text[:10]


def _daily_coverage_dates(report, daily: pd.DataFrame) -> list[str]:
    dates = [ _as_day_key(d) for d in (getattr(report, "daily_dates", None) or []) ]
    if daily is not None and not daily.empty and "sale_date" in daily.columns:
        dates.extend(_as_day_key(v) for v in daily["sale_date"].tolist())
    return sorted({d for d in dates if d})


def _daily_coverage_stores(report, daily: pd.DataFrame) -> list[str]:
    stores = [str(s).strip() for s in (getattr(report, "daily_store_ids", None) or [])]
    if daily is not None and not daily.empty and "store_id" in daily.columns:
        stores.extend(daily["store_id"].astype(str).str.strip().tolist())
    return sorted({s for s in stores if s})


def _periods_from_daily(dates: list[str], daily: pd.DataFrame) -> list[str]:
    periods: set[str] = set()
    if daily is not None and not daily.empty and "period" in daily.columns:
        periods.update(daily["period"].dropna().astype(str).str.strip().tolist())
    for day in dates:
        key = _as_day_key(day)
        if len(key) >= 7:
            periods.add(key[:7])
    return sorted(p for p in periods if p)


def _delete_shop_days(conn, store_ids: list[str], dates: list[str]) -> None:
    if not store_ids or not dates:
        return
    date_list = [_as_day_key(d) for d in dates if _as_day_key(d)]
    if not date_list:
        return
    dph = ", ".join("?" * len(date_list))
    try:
        for part in _chunked(store_ids):
            sph = ", ".join("?" * len(part))
            conn.execute(
                f"DELETE FROM shop_day WHERE store_id IN ({sph}) AND sale_date IN ({dph})",
                tuple(part) + tuple(date_list),
            )
    except Exception:
        pass


def _delete_sales_facts(conn, store_ids: list[str], periods: list[str]) -> None:
    if not store_ids or not periods:
        return
    pph = ", ".join("?" * len(periods))
    for part in _chunked(store_ids):
        sph = ", ".join("?" * len(part))
        conn.execute(
            f"DELETE FROM sales_facts WHERE store_id IN ({sph}) AND period IN ({pph})",
            tuple(part) + tuple(periods),
        )


def _read_shop_day_slice(conn, store_ids: list[str], periods: list[str]) -> pd.DataFrame:
    if not store_ids or not periods:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    pph = ", ".join("?" * len(periods))
    for part in _chunked(store_ids):
        sph = ", ".join("?" * len(part))
        try:
            chunk = read_sql(
                conn,
                f"SELECT * FROM shop_day WHERE store_id IN ({sph}) AND period IN ({pph})",
                tuple(part) + tuple(periods),
            )
        except Exception:
            chunk = pd.DataFrame()
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _month_facts_from_days(days: pd.DataFrame, label: str) -> pd.DataFrame:
    work = days.copy()
    work["store_id"] = work["store_id"].astype(str).str.strip()
    work["period"] = work["period"].astype(str).str.strip()
    work["volume_mt"] = pd.to_numeric(work["volume_mt"], errors="coerce").fillna(0)
    grouped = work.groupby(["store_id", "period"], as_index=False).agg(
        store_name=("store_name", "last") if "store_name" in work.columns else ("store_id", "last"),
        year=("year", "last") if "year" in work.columns else ("period", "last"),
        month=("month", "last") if "month" in work.columns else ("period", "last"),
        volume_mt=("volume_mt", "sum"),
        distributor=("distributor", "last") if "distributor" in work.columns else ("store_id", "last"),
        dsr_name=("dsr_name", "last") if "dsr_name" in work.columns else ("store_id", "last"),
        section=("section", "last") if "section" in work.columns else ("store_id", "last"),
    )
    if "year" not in grouped.columns or grouped["year"].isna().any():
        grouped["year"] = grouped["period"].astype(str).str.slice(0, 4).astype(int)
    if "month" not in grouped.columns or grouped["month"].isna().any():
        grouped["month"] = grouped["period"].astype(str).str.slice(5, 7).astype(int)
    grouped["year"] = pd.to_numeric(grouped["year"], errors="coerce").fillna(0).astype(int)
    grouped["month"] = pd.to_numeric(grouped["month"], errors="coerce").fillna(0).astype(int)
    grouped["sku"] = "ALL"
    grouped["source_file"] = label
    grouped["ingested_at"] = utcnow()
    return grouped


def _rebuild_facts_from_shop_day(
    conn,
    store_ids: list[str],
    periods: list[str],
    label: str,
    live_book: pd.DataFrame,
) -> list[str]:
    periods = sorted({str(p) for p in periods if p})
    if not periods:
        return []
    if store_ids:
        _delete_sales_facts(conn, store_ids, periods)
    days = _read_shop_day_slice(conn, store_ids, periods) if store_ids else pd.DataFrame()
    if days is None or days.empty:
        return periods
    facts = _month_facts_from_days(days, label)
    facts = overlay_store_attrs(facts, live_book)
    facts = collapse_sales_facts(facts)
    if facts is not None and not facts.empty:
        facts["source_file"] = label
        facts["ingested_at"] = utcnow()
        cols = [c for c in SALES_FACT_COLS if c in facts.columns]
        upsert_dataframe(conn, "sales_facts", facts[cols], ["store_id", "sku", "period"])
    return periods


def _refresh_period_ledger(conn, periods: list[str], label: str, execution, open_period: Optional[str]) -> None:
    for per in periods:
        part = read_sql(conn, "SELECT volume_mt FROM sales_facts WHERE period = ?", (per,))
        vol = (
            float(pd.to_numeric(part["volume_mt"], errors="coerce").fillna(0).sum())
            if part is not None and not part.empty
            else 0.0
        )
        n_rows = int(len(part)) if part is not None and not part.empty else 0
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
                label,
                execution.strftime("%Y-%m-%d") if execution else None,
                utcnow(),
                n_rows,
                vol,
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
                (per, int(as_of), int(days), vol, label, utcnow()),
            )


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

    try:
        shop_day_df = read_sql(conn, "SELECT * FROM shop_day")
    except Exception:
        shop_day_df = pd.DataFrame()
    pack = build_hierarchy_pack(
        shop_month,
        stores_df,
        feats,
        ledger,
        facts=facts_df,
        mtd_obs=read_sql(conn, "SELECT * FROM mtd_observations"),
        visits=visits_df,
        shop_day=shop_day_df,
    )
    try:
        shop_targets = read_sql(conn, "SELECT * FROM shop_targets")
    except Exception:
        shop_targets = pd.DataFrame()
    if shop_targets is not None and not shop_targets.empty:
        rematched, _ = match_shop_targets(shop_targets, stores_df)
        replace_table(conn, "shop_targets", rematched)
        shop_targets = rematched
    pace = 1.0
    if pack.units is not None and not pack.units.empty and "intra_month_frac" in pack.units.columns:
        pace = float(pd.to_numeric(pack.units["intra_month_frac"], errors="coerce").dropna().max() or 1.0)
    pack.units = attach_plan(pack.units, shop_targets, pace=pace)
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

    try:
        shop_day = read_sql(conn, "SELECT * FROM shop_day")
    except Exception:
        shop_day = pd.DataFrame()
    actions = build_action_pack(
        shop_month,
        stores_df,
        shop_day=shop_day,
        visits=visits_df,
        ledger=ledger,
        period=period,
    )
    persist_action_pack(conn, actions)

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
