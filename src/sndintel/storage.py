"""SQLite warehouse for facts, features, scores, and insights."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

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
    PRIMARY KEY (period, grain, grain_id)
);
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
    return db_path


def replace_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    conn.execute(f"DELETE FROM {name}")
    if df is None or df.empty:
        return
    df.to_sql(name, conn, if_exists="append", index=False)


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
