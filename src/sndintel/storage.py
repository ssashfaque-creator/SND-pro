"""SQLite warehouse for facts, features, scores, and insights."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

from sndintel.config import DB_PATH, ensure_dirs

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,
    sales_file TEXT,
    shop_file TEXT,
    n_sales_rows INTEGER,
    n_stores INTEGER,
    latest_period TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    store_name TEXT,
    distributor TEXT,
    dsr_name TEXT,
    zone TEXT,
    city TEXT,
    section TEXT,
    category_1 TEXT,
    category_2 TEXT,
    category_3 TEXT,
    category_4 TEXT,
    in_universe INTEGER DEFAULT 1,
    source TEXT,
    extra_json TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS sales_facts (
    store_id TEXT NOT NULL,
    sku TEXT NOT NULL,
    period TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    volume_mt REAL NOT NULL,
    distributor TEXT,
    dsr_name TEXT,
    section TEXT,
    store_name TEXT,
    source_file TEXT,
    ingested_at TEXT,
    PRIMARY KEY (store_id, sku, period)
);

CREATE INDEX IF NOT EXISTS idx_sales_period ON sales_facts(period);
CREATE INDEX IF NOT EXISTS idx_sales_store ON sales_facts(store_id);
CREATE INDEX IF NOT EXISTS idx_sales_section ON sales_facts(section);
CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales_facts(sku);

CREATE TABLE IF NOT EXISTS shop_visits (
    store_id TEXT NOT NULL,
    period TEXT NOT NULL,
    visits INTEGER NOT NULL DEFAULT 0,
    distributor TEXT,
    dsr_name TEXT,
    city TEXT,
    section TEXT,
    store_name TEXT,
    source_file TEXT,
    ingested_at TEXT,
    PRIMARY KEY (store_id, period)
);

CREATE INDEX IF NOT EXISTS idx_visits_period ON shop_visits(period);

CREATE TABLE IF NOT EXISTS shop_month (
    store_id TEXT NOT NULL,
    period TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    volume_mt REAL,
    sku_count INTEGER,
    billed INTEGER,
    distributor TEXT,
    dsr_name TEXT,
    section TEXT,
    store_name TEXT,
    zone TEXT,
    city TEXT,
    PRIMARY KEY (store_id, period)
);

CREATE INDEX IF NOT EXISTS idx_sm_period ON shop_month(period);
CREATE INDEX IF NOT EXISTS idx_sm_city ON shop_month(city);
CREATE INDEX IF NOT EXISTS idx_sm_section ON shop_month(section);
CREATE INDEX IF NOT EXISTS idx_sm_dsr ON shop_month(dsr_name);

CREATE TABLE IF NOT EXISTS features_shop_month (
    store_id TEXT NOT NULL,
    period TEXT NOT NULL,
    volume_mt REAL,
    sku_count INTEGER,
    roll_mean_3 REAL,
    roll_mean_6 REAL,
    roll_median_6 REAL,
    lag_1 REAL,
    lag_2 REAL,
    lag_3 REAL,
    lag_12 REAL,
    mom_pct REAL,
    yoy_pct REAL,
    zscore_own REAL,
    vs_section_pct REAL,
    vs_city_pct REAL,
    cv_6m REAL,
    recency_months INTEGER,
    billed_rate_12 REAL,
    top_sku_share REAL,
    first_period TEXT,
    months_on_file INTEGER,
    yoy_comparable INTEGER,
    PRIMARY KEY (store_id, period)
);

CREATE TABLE IF NOT EXISTS forecasts (
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    period TEXT NOT NULL,
    actual REAL,
    predicted REAL,
    residual REAL,
    residual_pct REAL,
    model TEXT,
    PRIMARY KEY (entity_type, entity_id, period)
);

CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id TEXT,
    period TEXT,
    kind TEXT,
    score REAL,
    severity TEXT,
    volume_mt REAL,
    expected_mt REAL,
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS shop_segments (
    store_id TEXT PRIMARY KEY,
    cluster_id INTEGER,
    segment TEXT,
    recency_months REAL,
    frequency REAL,
    monetary REAL,
    trend REAL,
    breadth REAL,
    cv REAL,
    lifetime_volume REAL,
    last_period TEXT,
    last_volume REAL
);

CREATE TABLE IF NOT EXISTS insights (
    insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    type TEXT,
    severity TEXT,
    entity_type TEXT,
    entity_id TEXT,
    entity_name TEXT,
    period TEXT,
    title TEXT,
    narrative TEXT,
    action TEXT,
    metric_value REAL,
    metrics_json TEXT,
    rank_score REAL
);

CREATE INDEX IF NOT EXISTS idx_insights_rank ON insights(rank_score DESC);
CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(type);

CREATE TABLE IF NOT EXISTS period_ledger (
    period TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    source_file TEXT,
    execution_date TEXT,
    ingested_at TEXT,
    n_fact_rows INTEGER,
    volume_mt REAL,
    as_of_day INTEGER,
    days_in_month INTEGER
);

CREATE TABLE IF NOT EXISTS strategy_plays (
    play_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    slot INTEGER,
    theme TEXT,
    title TEXT,
    why TEXT,
    do_this_week TEXT,
    owner TEXT,
    period TEXT,
    metric_value REAL,
    shops_json TEXT,
    metrics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_plays_slot ON strategy_plays(slot);

CREATE TABLE IF NOT EXISTS unit_scorecards (
    period TEXT NOT NULL,
    grain TEXT NOT NULL,
    grain_id TEXT NOT NULL,
    parent_grain TEXT,
    parent_id TEXT,
    zone TEXT,
    city TEXT,
    distributor TEXT,
    dsr_name TEXT,
    volume_mt REAL,
    ly_mt REAL,
    expected_mt REAL,
    run_rate_mt REAL,
    gap_mt REAL,
    gap_vs_ly_mt REAL,
    gap_pct REAL,
    lfl_now REAL,
    lfl_ly REAL,
    lfl_gap REAL,
    lost_n INTEGER,
    lost_mt REAL,
    new_n INTEGER,
    new_mt REAL,
    billed INTEGER,
    billed_ly INTEGER,
    universe INTEGER,
    strike_rate REAL,
    diagnosis TEXT,
    verdict TEXT,
    do_this_week TEXT,
    contrib_national_gap REAL,
    share_expected_mt REAL,
    competitive_mt REAL,
    isolated_mt REAL,
    z_score REAL,
    focus_score REAL,
    coverage_effect_mt REAL,
    velocity_effect_mt REAL,
    interaction_effect_mt REAL,
    mix_effect_mt REAL,
    wd REAL,
    nd REAL,
    parent_index REAL,
    intra_month_frac REAL,
    seasonal_mom_index REAL,
    mom_expected_mt REAL,
    mom_gap_mt REAL,
    seasonal_index REAL,
    seasonal_typical_mt REAL,
    expected_drop_size_mt REAL,
    situation TEXT,
    metrics_json TEXT,
    PRIMARY KEY (period, grain, grain_id, parent_id)
);

CREATE TABLE IF NOT EXISTS shop_targets (
    store_id TEXT,
    store_name TEXT NOT NULL,
    distributor TEXT,
    dsr_name TEXT,
    zone TEXT,
    city TEXT,
    section TEXT,
    target_mt REAL NOT NULL,
    match_method TEXT,
    match_score REAL,
    source_file TEXT,
    ingested_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_shop_targets_store ON shop_targets(store_id);
CREATE INDEX IF NOT EXISTS idx_shop_targets_city ON shop_targets(city);

CREATE TABLE IF NOT EXISTS seasonality_index (
    period TEXT NOT NULL,
    grain TEXT NOT NULL,
    grain_id TEXT NOT NULL DEFAULT '',
    month INTEGER NOT NULL,
    seasonal_index REAL NOT NULL,
    typical_mt REAL,
    n_obs INTEGER,
    credibility REAL,
    PRIMARY KEY (period, grain, grain_id, month)
);

CREATE INDEX IF NOT EXISTS idx_units_grain ON unit_scorecards(grain, gap_mt);
CREATE INDEX IF NOT EXISTS idx_units_city ON unit_scorecards(city);

CREATE TABLE IF NOT EXISTS focus_targets (
    period TEXT,
    rank INTEGER,
    grain TEXT,
    entity_id TEXT,
    entity_name TEXT,
    city TEXT,
    zone TEXT,
    distributor TEXT,
    dsr_name TEXT,
    section TEXT,
    volume_mt REAL,
    ly_mt REAL,
    gap_mt REAL,
    diagnosis TEXT,
    action TEXT,
    why TEXT,
    competitive_mt REAL,
    isolated_mt REAL,
    z_score REAL,
    focus_score REAL,
    situation TEXT
);

CREATE INDEX IF NOT EXISTS idx_targets_rank ON focus_targets(rank);
CREATE INDEX IF NOT EXISTS idx_targets_city ON focus_targets(city);

CREATE TABLE IF NOT EXISTS situation_brief (
    period TEXT PRIMARY KEY,
    headline TEXT,
    weather TEXT,
    problem TEXT,
    action_summary TEXT,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS exec_summary (
    period TEXT PRIMARY KEY,
    model TEXT,
    situation_json TEXT,
    focus_json TEXT,
    brief_json TEXT,
    raw_json TEXT,
    error TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS mtd_observations (
    period TEXT NOT NULL,
    as_of_day INTEGER NOT NULL,
    days_in_month INTEGER,
    volume_mt REAL,
    source_file TEXT,
    ingested_at TEXT,
    PRIMARY KEY (period, as_of_day)
);

CREATE TABLE IF NOT EXISTS kpi_snapshots (
    period TEXT NOT NULL,
    grain TEXT NOT NULL,
    grain_id TEXT NOT NULL,
    volume_mt REAL,
    billed_outlets INTEGER,
    universe_outlets INTEGER,
    strike_rate REAL,
    drop_size REAL,
    sku_depth REAL,
    mom_pct REAL,
    yoy_pct REAL,
    period_status TEXT,
    as_of_day INTEGER,
    days_in_month INTEGER,
    run_rate_mt REAL,
    comparable_mom_pct REAL,
    run_rate_yoy_pct REAL,
    PRIMARY KEY (period, grain, grain_id)
);

CREATE TABLE IF NOT EXISTS shop_day (
    store_id TEXT NOT NULL,
    sale_date TEXT NOT NULL,
    period TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    volume_mt REAL NOT NULL,
    distributor TEXT,
    dsr_name TEXT,
    section TEXT,
    store_name TEXT,
    city TEXT,
    source_file TEXT,
    ingested_at TEXT,
    PRIMARY KEY (store_id, sale_date)
);

CREATE INDEX IF NOT EXISTS idx_shop_day_period ON shop_day(period);
CREATE INDEX IF NOT EXISTS idx_shop_day_store ON shop_day(store_id);
CREATE INDEX IF NOT EXISTS idx_shop_day_dsr ON shop_day(dsr_name);

CREATE TABLE IF NOT EXISTS action_shops (
    period TEXT NOT NULL,
    store_id TEXT NOT NULL,
    store_name TEXT,
    city TEXT,
    distributor TEXT,
    dsr_name TEXT,
    section TEXT,
    action TEXT,
    instruction TEXT,
    billed_mt REAL,
    expected_mt REAL,
    ams_3m REAL,
    should_have_mt REAL,
    behind_pace_mt REAL,
    remaining_mt REAL,
    week_target_mt REAL,
    typical_drop_mt REAL,
    typical_bill_day REAL,
    pace_frac REAL,
    last_bill_date TEXT,
    days_since_bill INTEGER,
    call_status TEXT,
    visits INTEGER,
    value_score REAL,
    cycle_days REAL,
    days_overdue REAL,
    last_drop_mt REAL,
    cover_left_days REAL,
    light_mt REAL,
    trend_pct REAL,
    last_month_mt REAL,
    next_drop_mt REAL,
    next_drop_model TEXT,
    coming_due INTEGER,
    days_until_due REAL,
    n_orders_left REAL,
    PRIMARY KEY (period, store_id)
);

CREATE INDEX IF NOT EXISTS idx_action_shops_action ON action_shops(action, value_score DESC);

CREATE TABLE IF NOT EXISTS action_units (
    period TEXT NOT NULL,
    grain TEXT NOT NULL,
    grain_id TEXT NOT NULL,
    city TEXT,
    distributor TEXT,
    dsr_name TEXT,
    label TEXT,
    span_unique REAL,
    day_cap INTEGER,
    n_call INTEGER,
    n_convert INTEGER,
    n_lift INTEGER,
    n_lapse INTEGER,
    n_hold INTEGER,
    n_doors INTEGER,
    n_coming INTEGER,
    ams_3m REAL,
    billed_mt REAL,
    expected_mt REAL,
    should_have_mt REAL,
    behind_pace_mt REAL,
    remaining_mt REAL,
    week_target_mt REAL,
    instruction TEXT,
    value_score REAL,
    PRIMARY KEY (period, grain, grain_id)
);

CREATE INDEX IF NOT EXISTS idx_action_units_grain ON action_units(grain, value_score DESC);

CREATE TABLE IF NOT EXISTS action_backtest (
    period TEXT NOT NULL,
    cut_day INTEGER NOT NULL,
    n_months INTEGER,
    n_shops INTEGER,
    curve_precision_at_50 REAL,
    calendar_precision_at_50 REAL,
    curve_catch_mt REAL,
    calendar_catch_mt REAL,
    n_backloaded_hold INTEGER,
    n_backloaded_ok INTEGER,
    remaining_mae_mt REAL,
    notes TEXT,
    PRIMARY KEY (period, cut_day)
);

CREATE TABLE IF NOT EXISTS action_brief (
    period TEXT PRIMARY KEY,
    as_of_day INTEGER,
    days_in_month INTEGER,
    days_left INTEGER,
    billed_mt REAL,
    expected_mt REAL,
    ams_3m REAL,
    should_have_mt REAL,
    behind_pace_mt REAL,
    remaining_mt REAL,
    week_target_mt REAL,
    n_call INTEGER,
    n_convert INTEGER,
    n_lift INTEGER,
    n_lapse INTEGER,
    n_doors INTEGER,
    n_coming INTEGER,
    has_daily INTEGER,
    headline TEXT,
    source TEXT,
    metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS action_outcomes (
    period TEXT NOT NULL,
    listed_at TEXT NOT NULL,
    store_id TEXT NOT NULL,
    store_name TEXT,
    city TEXT,
    distributor TEXT,
    dsr_name TEXT,
    action TEXT,
    ask_mt REAL,
    billed_mt_listed REAL,
    visits_listed REAL,
    billed_mt_now REAL,
    visits_now REAL,
    gained_mt REAL,
    outcome TEXT,
    PRIMARY KEY (period, store_id, listed_at)
);

CREATE INDEX IF NOT EXISTS idx_action_outcomes_period ON action_outcomes(period, listed_at);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    ensure_dirs()
    db_path = Path(path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Optional[Path] = None) -> Path:
    db_path = Path(path or DB_PATH)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        _ensure_column(conn, "features_shop_month", "first_period", "TEXT")
        _ensure_column(conn, "features_shop_month", "months_on_file", "INTEGER")
        _ensure_column(conn, "features_shop_month", "yoy_comparable", "INTEGER")
        for col, ddl in (
            ("period_status", "TEXT"),
            ("as_of_day", "INTEGER"),
            ("days_in_month", "INTEGER"),
            ("run_rate_mt", "REAL"),
            ("comparable_mom_pct", "REAL"),
            ("run_rate_yoy_pct", "REAL"),
        ):
            _ensure_column(conn, "kpi_snapshots", col, ddl)
        for col, ddl in (
            ("share_expected_mt", "REAL"),
            ("competitive_mt", "REAL"),
            ("isolated_mt", "REAL"),
            ("z_score", "REAL"),
            ("focus_score", "REAL"),
            ("coverage_effect_mt", "REAL"),
            ("velocity_effect_mt", "REAL"),
            ("interaction_effect_mt", "REAL"),
            ("mix_effect_mt", "REAL"),
            ("wd", "REAL"),
            ("nd", "REAL"),
            ("parent_index", "REAL"),
            ("intra_month_frac", "REAL"),
            ("seasonal_mom_index", "REAL"),
            ("mom_expected_mt", "REAL"),
            ("mom_gap_mt", "REAL"),
            ("seasonal_index", "REAL"),
            ("seasonal_typical_mt", "REAL"),
            ("expected_drop_size_mt", "REAL"),
            ("situation", "TEXT"),
            ("visited", "INTEGER"),
            ("visit_rate", "REAL"),
            ("productivity", "REAL"),
            ("from_unvisited_mt", "REAL"),
            ("from_unbilled_mt", "REAL"),
            ("from_drop_size_mt", "REAL"),
            ("visits", "REAL"),
            ("opportunity_mt", "REAL"),
            ("has_visit_file", "INTEGER"),
            ("distributor", "TEXT"),
            ("dsr_name", "TEXT"),
            ("target_mt", "REAL"),
            ("target_paced_mt", "REAL"),
            ("vs_target_mt", "REAL"),
            ("gap_to_target_mt", "REAL"),
            ("attain_pct", "REAL"),
            ("stretch_mt", "REAL"),
            ("plan_quality", "TEXT"),
            ("plan_status", "TEXT"),
            ("n_target_shops", "INTEGER"),
            ("target_matched_mt", "REAL"),
            ("target_book_mt", "REAL"),
            ("n_target_unmatched", "INTEGER"),
        ):
            _ensure_column(conn, "unit_scorecards", col, ddl)
        _ensure_column(conn, "stores", "source", "TEXT")
        _ensure_column(conn, "shop_targets", "section", "TEXT")
        for col, ddl in (
            ("competitive_mt", "REAL"),
            ("isolated_mt", "REAL"),
            ("z_score", "REAL"),
            ("focus_score", "REAL"),
            ("situation", "TEXT"),
        ):
            _ensure_column(conn, "focus_targets", col, ddl)
        for col, ddl in (
            ("period", "TEXT"),
            ("grain", "TEXT"),
            ("grain_id", "TEXT"),
            ("month", "INTEGER"),
            ("seasonal_index", "REAL"),
            ("typical_mt", "REAL"),
            ("n_obs", "INTEGER"),
            ("credibility", "REAL"),
        ):
            _ensure_column(conn, "seasonality_index", col, ddl)
        for col, ddl in (
            ("cycle_days", "REAL"),
            ("days_overdue", "REAL"),
            ("last_drop_mt", "REAL"),
            ("cover_left_days", "REAL"),
            ("light_mt", "REAL"),
            ("trend_pct", "REAL"),
            ("last_month_mt", "REAL"),
            ("next_drop_mt", "REAL"),
            ("next_drop_model", "TEXT"),
        ("coming_due", "INTEGER"),
        ("days_until_due", "REAL"),
        ("n_orders_left", "REAL"),
        ("expected_drop_mt", "REAL"),
        ("api_days", "REAL"),
        ("depletion_ratio", "REAL"),
        ("n_purchases_90d", "INTEGER"),
        ("n_purchases_ever", "INTEGER"),
        ("is_cold_start", "INTEGER"),
        ("is_lapsed", "INTEGER"),
        ("due_unvisited_mt", "REAL"),
        ("drop_variance_mt", "REAL"),
        ("not_yet_due_mt", "REAL"),
        ("pipeline_expected_mt", "REAL"),
        ("recommended_action", "TEXT"),
        ):
            _ensure_column(conn, "action_shops", col, ddl)
        _ensure_column(conn, "action_units", "n_lapse", "INTEGER")
        _ensure_column(conn, "action_units", "n_doors", "INTEGER")
        _ensure_column(conn, "action_units", "n_coming", "INTEGER")
        _ensure_column(conn, "action_units", "ams_3m", "REAL")
        _ensure_column(conn, "action_units", "dsr_name", "TEXT")
        _ensure_column(conn, "action_units", "label", "TEXT")
        _ensure_column(conn, "action_units", "span_unique", "REAL")
        _ensure_column(conn, "action_units", "day_cap", "INTEGER")
        for col, ddl in (
            ("due_unvisited_mt", "REAL"),
            ("drop_variance_mt", "REAL"),
            ("not_yet_due_mt", "REAL"),
            ("pipeline_expected_mt", "REAL"),
            ("n_due", "INTEGER"),
            ("n_universe", "INTEGER"),
            ("n_due_visited", "INTEGER"),
        ):
            _ensure_column(conn, "action_units", col, ddl)
        _ensure_column(conn, "action_brief", "n_lapse", "INTEGER")
        _ensure_column(conn, "action_brief", "n_doors", "INTEGER")
        _ensure_column(conn, "action_brief", "n_coming", "INTEGER")
        _ensure_column(conn, "action_brief", "ams_3m", "REAL")
        for col, ddl in (
            ("pipeline_expected_mt", "REAL"),
            ("due_unvisited_mt", "REAL"),
            ("drop_variance_mt", "REAL"),
            ("not_yet_due_mt", "REAL"),
        ):
            _ensure_column(conn, "action_brief", col, ddl)
    return db_path


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _table_columns(conn: sqlite3.Connection, name: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({name})")]


def _sql_cell(value: Any) -> Any:
    """SQLite-safe Python scalars. NaN becomes NULL (inf is dropped too)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (bytes, str, dict, list)):
        try:
            return _sql_cell(value.item())
        except Exception:
            pass
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def replace_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    """Replace a table without pandas.to_sql (which hides sqlite errors as DatabaseError)."""
    conn.execute(f"DELETE FROM {name}")
    if df is None or df.empty:
        return
    existing = _table_columns(conn, name)
    if not existing:
        raise RuntimeError(f"Warehouse table {name} does not exist. Restart the app so init_db can create it.")
    cols = [c for c in df.columns if c in existing]
    if not cols:
        raise RuntimeError(f"No overlapping columns when writing {name}: {list(df.columns)}")
    work = df.loc[:, cols].copy()
    if name == "seasonality_index" and "seasonal_index" in work.columns:
        work["seasonal_index"] = pd.to_numeric(work["seasonal_index"], errors="coerce").fillna(1.0)
        if "grain_id" in work.columns:
            work["grain_id"] = work["grain_id"].fillna("(unmapped)").astype(str)
        if "grain" in work.columns:
            work["grain"] = work["grain"].fillna("national").astype(str)
        if "period" in work.columns:
            work["period"] = work["period"].fillna("").astype(str)
        if "month" in work.columns:
            work["month"] = pd.to_numeric(work["month"], errors="coerce").fillna(0).astype(int)
    placeholders = ", ".join("?" * len(cols))
    col_sql = ", ".join(cols)
    sql = f"INSERT INTO {name} ({col_sql}) VALUES ({placeholders})"
    rows = [tuple(_sql_cell(v) for v in rec) for rec in work.itertuples(index=False, name=None)]
    try:
        conn.executemany(sql, rows)
    except Exception as exc:
        raise RuntimeError(f"Could not write {name} ({len(rows)} rows): {exc}") from exc


def upsert_dataframe(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    key_cols: list[str],
) -> int:
    if df is None or df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join(cols)
    update_cols = [c for c in cols if c not in key_cols]
    if update_cols:
        set_sql = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
        sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT({', '.join(key_cols)}) DO UPDATE SET {set_sql}"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT({', '.join(key_cols)}) DO NOTHING"
        )
    rows = [
        tuple(None if pd.isna(v) else v for v in rec)
        for rec in df.itertuples(index=False, name=None)
    ]
    conn.executemany(sql, rows)
    return len(rows)


def read_sql(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
