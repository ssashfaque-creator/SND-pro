"""Warehouse-learned seasonality: month shape from history, not a shipped curve."""

from __future__ import annotations

import pandas as pd

from sndintel.hierarchy import build_hierarchy_pack
from sndintel.season import fit_seasonality, intra_month_fraction


def _shop_month(store_id: str, period: str, volume: float, city: str = "Karachi") -> dict:
    return {
        "store_id": store_id,
        "period": period,
        "year": int(period[:4]),
        "month": int(period[5:7]),
        "volume_mt": volume,
        "sku_count": 1,
        "billed": 1,
        "distributor": "Eva Foods",
        "dsr_name": "Amir Surveyor",
        "section": "Nazimabad",
        "store_name": "Shop K",
        "zone": "South",
        "city": city,
    }


def _period(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def test_nineteen_months_learn_august_is_a_high_month():
    """~19 months of month totals teach calendar shape (August vs January)."""
    rows = []
    # Feb 2025 through July 2026 = 18 history months, plus current Aug 2026.
    # August is 1.4× the other months; January is 0.7×.
    start_y, start_m = 2025, 2
    for i in range(18):
        m = (start_m - 1 + i) % 12 + 1
        y = start_y + (start_m - 1 + i) // 12
        base = 100.0
        if m == 8:
            vol = 140.0
        elif m == 1:
            vol = 70.0
        else:
            vol = base
        rows.append(_shop_month("K1", _period(y, m), vol))
    rows.append(_shop_month("K1", "2026-08", 100.0))
    sm = pd.DataFrame(rows)
    fit = fit_seasonality(sm, "2026-08")
    assert fit.n_periods == 18
    assert fit.n_same_month == 1  # only Aug 2025 is a prior August
    assert fit.national_index[8] > 1.05
    assert fit.national_index[1] < 0.95
    # Expected uses typical August (140), not a curve we specified.
    assert fit.expected_full_national > 120


def test_typical_august_is_mean_of_every_august_not_last_year_only():
    rows = []
    rows.append(_shop_month("K1", "2024-08", 100.0))
    rows.append(_shop_month("K1", "2025-08", 80.0))
    for m in range(1, 8):
        rows.append(_shop_month("K1", _period(2026, m), 90.0))
    rows.append(_shop_month("K1", "2026-08", 70.0))
    sm = pd.DataFrame(rows)
    fit = fit_seasonality(sm, "2026-08")
    assert fit.n_same_month == 2
    # Mean of 100 and 80 is 90 — last year alone is 80.
    typical = fit.city_expected.iloc[0]["typical_mt"]
    assert abs(typical - 90.0) < 0.5
    pack = build_hierarchy_pack(
        sm,
        sm.drop_duplicates("store_id")[
            ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
        ],
        ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]),
    )
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    # Expected closer to 90 than to last year's 80.
    assert float(khi["expected_mt"]) > 84
    assert abs(float(khi["expected_mt"]) - 80.0) > 3


def test_month_end_only_observations_do_not_invent_day20_at_full_month():
    obs = pd.DataFrame(
        [
            {"period": "2025-07", "as_of_day": 31, "days_in_month": 31, "volume_mt": 90.0},
            {"period": "2025-08", "as_of_day": 31, "days_in_month": 31, "volume_mt": 100.0},
        ]
    )
    frac, src = intra_month_fraction(20, 31, obs, open_mtd=True)
    assert src == "elapsed_days"
    assert abs(frac - 20 / 31) < 1e-9
    assert frac < 0.9


def test_mid_month_plus_close_learns_intra_month_frac():
    obs = pd.DataFrame(
        [
            {"period": "2025-08", "as_of_day": 10, "days_in_month": 31, "volume_mt": 30.0},
            {"period": "2025-08", "as_of_day": 31, "days_in_month": 31, "volume_mt": 100.0},
        ]
    )
    frac, src = intra_month_fraction(10, 31, obs, open_mtd=True)
    assert src == "learned_mtd_cuts"
    assert abs(frac - 0.30) < 0.02
    frac20, src20 = intra_month_fraction(20, 31, obs, open_mtd=True)
    assert src20 == "learned_mtd_cuts"
    # Linear interp 10→31: day 20 is between 0.30 and 1.0, not clamped to 1.0.
    assert 0.35 < frac20 < 0.85


def test_replace_table_accepts_nan_seasonal_index(tmp_path):
    """NaN used to become SQL NULL and fail NOT NULL as pandas DatabaseError."""
    from sndintel.storage import connect, init_db, read_sql, replace_table

    db = tmp_path / "warehouse.db"
    init_db(db)
    messy = pd.DataFrame(
        [
            {
                "period": "2026-08",
                "grain": "city",
                "grain_id": "Karachi",
                "month": 8,
                "seasonal_index": float("nan"),
                "typical_mt": float("nan"),
                "n_obs": 0,
                "credibility": 0.2,
            }
        ]
    )
    with connect(db) as conn:
        replace_table(conn, "seasonality_index", messy)
        out = read_sql(conn, "SELECT * FROM seasonality_index")
    assert len(out) == 1
    assert abs(float(out.iloc[0]["seasonal_index"]) - 1.0) < 1e-9


def test_rescore_persists_seasonality_with_sparse_cities(tmp_path):
    """A tiny city with one billed month must not crash warehouse writes."""
    from sndintel.ingest.pipeline import rescore_warehouse
    from sndintel.storage import connect, init_db, read_sql, replace_table, utcnow

    db = tmp_path / "warehouse.db"
    init_db(db)
    rows = []
    start_y, start_m = 2025, 2
    for i in range(18):
        m = (start_m - 1 + i) % 12 + 1
        y = start_y + (start_m - 1 + i) // 12
        vol = 140.0 if m == 8 else (70.0 if m == 1 else 100.0)
        rows.append(_shop_month("K1", f"{y:04d}-{m:02d}", vol, "Karachi"))
    rows.append(_shop_month("F1", "2026-01", 0.01, "Faisalabad"))
    rows.append(_shop_month("K1", "2026-08", 90.0, "Karachi"))
    rows.append(_shop_month("F1", "2026-08", 0.02, "Faisalabad"))
    sm = pd.DataFrame(rows)
    stores = sm.drop_duplicates("store_id")[
        ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
    ]
    stores = stores.copy()
    stores["category_1"] = stores["category_2"] = stores["category_3"] = stores["category_4"] = None
    stores["in_universe"] = 1
    stores["extra_json"] = None
    stores["updated_at"] = utcnow()
    facts = pd.DataFrame(
        {
            "store_id": sm["store_id"],
            "sku": "Eva 5L",
            "period": sm["period"],
            "year": sm["year"],
            "month": sm["month"],
            "volume_mt": sm["volume_mt"],
            "distributor": sm["distributor"],
            "dsr_name": sm["dsr_name"],
            "section": sm["section"],
            "store_name": sm["store_name"],
            "source_file": "test.xlsx",
            "ingested_at": utcnow(),
        }
    )
    with connect(db) as conn:
        replace_table(conn, "stores", stores)
        replace_table(conn, "sales_facts", facts)
        conn.execute(
            """INSERT INTO period_ledger (period, status, as_of_day, days_in_month)
               VALUES ('2026-08', 'mtd_open', 20, 31)"""
        )
    scored = rescore_warehouse(db)
    assert scored["n_cities"] >= 1
    with connect(db) as conn:
        season = read_sql(conn, "SELECT * FROM seasonality_index")
        units = read_sql(conn, "SELECT * FROM unit_scorecards WHERE grain = 'city'")
    assert not season.empty
    assert season["seasonal_index"].notna().all()
    assert not units.empty
