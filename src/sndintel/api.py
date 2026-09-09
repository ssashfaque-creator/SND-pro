"""Read-only JSON API over the SQLite intelligence warehouse."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Query

from sndintel import __version__
from sndintel.storage import connect, init_db, read_sql

app = FastAPI(
    title="SND Intelligence API",
    version=__version__,
    description="Deterministic queries over ranked FMCG secondary-sales insights.",
)


@app.on_event("startup")
def _startup():
    init_db()


def _rows(sql: str, params: tuple = ()):
    with connect() as conn:
        df = read_sql(conn, sql, params)
    return df.where(df.notna(), None).to_dict(orient="records")


@app.get("/health")
def health():
    return {"ok": True, "version": __version__}


@app.get("/brief")
def brief(limit: int = 25):
    return _rows("SELECT * FROM insights ORDER BY rank_score DESC LIMIT ?", (limit,))


@app.get("/insights")
def insights(
    type: Optional[str] = None,
    severity: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 100,
):
    clauses = ["1=1"]
    params: list = []
    if type:
        clauses.append("type = ?")
        params.append(type)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    params.append(limit)
    sql = f"SELECT * FROM insights WHERE {' AND '.join(clauses)} ORDER BY rank_score DESC LIMIT ?"
    return _rows(sql, tuple(params))


@app.get("/kpis")
def kpis(grain: Optional[str] = None, period: Optional[str] = None):
    clauses = ["1=1"]
    params: list = []
    if grain:
        clauses.append("grain = ?")
        params.append(grain)
    if period:
        clauses.append("period = ?")
        params.append(period)
    sql = f"SELECT * FROM kpi_snapshots WHERE {' AND '.join(clauses)} ORDER BY volume_mt DESC"
    return _rows(sql, tuple(params))


@app.get("/anomalies")
def anomalies(kind: Optional[str] = None):
    if kind:
        return _rows("SELECT * FROM anomalies WHERE kind = ? ORDER BY score DESC", (kind,))
    return _rows("SELECT * FROM anomalies ORDER BY score DESC")


@app.get("/segments")
def segments(segment: Optional[str] = None):
    if segment:
        return _rows("SELECT * FROM shop_segments WHERE segment = ?", (segment,))
    return _rows("SELECT * FROM shop_segments")


@app.get("/shops/{store_id}")
def shop(store_id: str):
    profile = _rows("SELECT * FROM stores WHERE store_id = ?", (store_id,))
    history = _rows(
        "SELECT * FROM shop_month WHERE store_id = ? ORDER BY period",
        (store_id,),
    )
    features = _rows(
        "SELECT * FROM features_shop_month WHERE store_id = ? ORDER BY period",
        (store_id,),
    )
    ins = _rows("SELECT * FROM insights WHERE entity_id = ? ORDER BY rank_score DESC", (store_id,))
    return {"store": profile[0] if profile else None, "history": history, "features": features, "insights": ins}


@app.get("/waterfall")
def waterfall():
    """City scorecards sorted by gap vs expected — the national hole, additively."""
    return _rows("SELECT * FROM unit_scorecards WHERE grain = 'city' ORDER BY gap_mt")


@app.get("/targets")
def targets(city: Optional[str] = None, grain: Optional[str] = None, limit: int = 100):
    clauses = ["1=1"]
    params: list = []
    if city:
        clauses.append("city = ?")
        params.append(city)
    if grain:
        clauses.append("grain = ?")
        params.append(grain)
    params.append(limit)
    sql = f"SELECT * FROM focus_targets WHERE {' AND '.join(clauses)} ORDER BY rank LIMIT ?"
    return _rows(sql, tuple(params))


@app.get("/scorecards")
def scorecards(grain: Optional[str] = None, city: Optional[str] = None):
    clauses = ["1=1"]
    params: list = []
    if grain:
        clauses.append("grain = ?")
        params.append(grain)
    if city:
        clauses.append("city = ?")
        params.append(city)
    sql = f"SELECT * FROM unit_scorecards WHERE {' AND '.join(clauses)} ORDER BY gap_mt"
    return _rows(sql, tuple(params))


@app.get("/focus")
def focus():
    return {
        "churn_risk": _rows("SELECT * FROM shop_segments WHERE segment = 'Churn Risk' ORDER BY monetary DESC LIMIT 30"),
        "growth": _rows("SELECT * FROM shop_segments WHERE segment = 'Growth Target' ORDER BY trend DESC LIMIT 30"),
        "trade_loading": _rows("SELECT * FROM anomalies WHERE kind = 'trade_loading' ORDER BY volume_mt DESC"),
        "whitespace": _rows(
            """
            SELECT s.* FROM stores s
            WHERE s.store_id NOT IN (SELECT DISTINCT store_id FROM sales_facts)
            LIMIT 50
            """
        ),
    }


@app.get("/situation")
def situation(
    scope: str = "national",
    city: Optional[str] = None,
    distributor: Optional[str] = None,
    period: Optional[str] = None,
):
    """Sendable situation cascade: under/over performers and next actions."""
    from sndintel.situation_report import build_situation_pack, load_units_for_period

    with connect() as conn:
        ledger = read_sql(conn, "SELECT * FROM period_ledger ORDER BY period")
        period_df = read_sql(conn, "SELECT MAX(period) AS period FROM shop_month")
        latest = str(period_df.iloc[0]["period"]) if period_df is not None and not period_df.empty else ""
        period = period or latest
        units = load_units_for_period(conn, period)
    pack = build_situation_pack(
        units, ledger=ledger, period=period, scope=scope, city=city, distributor=distributor
    )
    return {
        "period": pack.period,
        "label": pack.label,
        "scope": pack.scope,
        "scope_label": pack.scope_label,
        "headline": pack.headline,
        "weather": pack.weather,
        "situation": pack.situation,
        "plan": pack.plan_lines,
        "kpis": pack.kpis,
        "steps": pack.steps.to_dict(orient="records") if pack.steps is not None else [],
        "lagging_cities": pack.lagging_cities.to_dict(orient="records") if pack.lagging_cities is not None else [],
        "ahead_cities": pack.ahead_cities.to_dict(orient="records") if pack.ahead_cities is not None else [],
        "lagging_distributors": pack.lagging_distributors.to_dict(orient="records")
        if pack.lagging_distributors is not None
        else [],
        "ahead_distributors": pack.ahead_distributors.to_dict(orient="records") if pack.ahead_distributors is not None else [],
        "lagging_people": pack.lagging_people.to_dict(orient="records") if pack.lagging_people is not None else [],
        "ahead_people": pack.ahead_people.to_dict(orient="records") if pack.ahead_people is not None else [],
        "copy_from": pack.copy_from,
    }



@app.get("/search")
def search(q: str = Query(..., min_length=2)):
    like = f"%{q}%"
    return _rows(
        """
        SELECT * FROM insights
        WHERE title LIKE ? OR narrative LIKE ? OR entity_name LIKE ? OR entity_id LIKE ? OR type LIKE ?
        ORDER BY rank_score DESC LIMIT 50
        """,
        (like, like, like, like, like),
    )
