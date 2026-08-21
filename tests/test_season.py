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
    # Expected uses last-3-month run-rate (May–Jul 2026 are 100), not typical August (140).
    assert 90 < fit.expected_full_national < 115


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
    # Last three closed months (May–Jul) are 90. Same-month Augusts are not the call.
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
    # Expected follows recent 90 MT months, not last August 80.
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


def test_distributor_expected_is_own_typical_august_not_parent_scale():
    """Eva's August typical is 80. City typical/LY scale would push Eva toward 107.

    Forecast-based proportions keep Eva near its own typical month, then scale
    so distributors add to the city Expected — not last-year mix × city billed now.
    """
    rows = []
    def add(store_id, dist, period, volume, name):
        rows.append(
            {
                "store_id": store_id,
                "period": period,
                "year": int(period[:4]),
                "month": int(period[5:7]),
                "volume_mt": volume,
                "sku_count": 1,
                "billed": 1,
                "distributor": dist,
                "dsr_name": "Amir Surveyor" if dist == "Eva Foods" else "South Rep",
                "section": "Nazimabad",
                "store_name": name,
                "zone": "South",
                "city": "Karachi",
            }
        )

    add("E1", "Eva Foods", "2024-08", 80.0, "Eva Shop")
    add("S1", "South Dist", "2024-08", 120.0, "South Shop")
    add("E1", "Eva Foods", "2025-08", 80.0, "Eva Shop")
    add("S1", "South Dist", "2025-08", 40.0, "South Shop")
    for m in range(1, 8):
        add("E1", "Eva Foods", f"2026-{m:02d}", 70.0, "Eva Shop")
        add("S1", "South Dist", f"2026-{m:02d}", 50.0, "South Shop")
    add("E1", "Eva Foods", "2026-08", 70.0, "Eva Shop")
    add("S1", "South Dist", "2026-08", 30.0, "South Shop")
    sm = pd.DataFrame(rows)
    stores = sm.drop_duplicates("store_id")[
        ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
    ]
    pack = build_hierarchy_pack(
        sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    )
    eva = pack.units[(pack.units["grain"] == "distributor") & (pack.units["grain_id"] == "Eva Foods")].iloc[0]
    south = pack.units[(pack.units["grain"] == "distributor") & (pack.units["grain_id"] == "South Dist")].iloc[0]
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    # Parent-scale fair share: 80 × (city typical 160 / city LY 120) ≈ 107.
    parent_scale = 80.0 * (160.0 / 120.0)
    assert abs(float(eva["expected_mt"]) - parent_scale) > 15
    assert 60.0 < float(eva["expected_mt"]) < 95.0
    assert abs(float(eva["expected_mt"]) + float(south["expected_mt"]) - float(khi["expected_mt"])) < 0.2
    # Eva billed 70 vs ~80 typical: on or slightly behind Expected, not a 37 MT hole vs 107.
    assert float(eva["volume_mt"]) - float(eva["expected_mt"]) > -25


def test_expected_follows_recent_ams_when_august_history_is_empty():
    """Larkana-style: no August last year, but May–Jul run-rate is 44 MT.

    Seasonality used to print Expected ≈ 4 and mark the city Ahead. Expected
    must stay near the 44 MT AMS, then pace if MTD is open.
    """
    rows = []
    for per, vol in [
        ("2026-05", 44.0),
        ("2026-06", 44.0),
        ("2026-07", 44.0),
        ("2026-08", 18.0),
    ]:
        rows.append(_shop_month("L1", per, vol, "Larkana"))
    sm = pd.DataFrame(rows)
    fit = fit_seasonality(sm, "2026-08")
    assert abs(float(fit.city_expected.iloc[0]["expected_full_mt"]) - 44.0) < 3
    stores = sm.drop_duplicates("store_id")[
        ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
    ]
    pack = build_hierarchy_pack(
        sm,
        stores,
        ledger=pd.DataFrame(
            [
                {"period": "2026-05", "status": "closed"},
                {"period": "2026-06", "status": "closed"},
                {"period": "2026-07", "status": "closed"},
                {
                    "period": "2026-08",
                    "status": "mtd_open",
                    "as_of_day": 20,
                    "days_in_month": 31,
                },
            ]
        ),
    )
    city = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Larkana")].iloc[0]
    expected = float(city["expected_mt"])
    # Paced 44 × 20/31 ≈ 28, not 4.
    assert 20.0 < expected < 36.0
    assert float(city["volume_mt"]) < expected
    assert city["situation"] == "lagging"


def test_city_with_only_old_august_does_not_steal_expected_from_live_cities():
    """A city that billed last August but nothing in May–Jul must Expected 0.

    Treating that 0 as 'missing' used to fall back to last August, then
    reconcile stole Expected from cities that actually have a run-rate.
    """
    rows = []
    for per, vol in [("2026-05", 40.0), ("2026-06", 40.0), ("2026-07", 40.0), ("2026-08", 30.0)]:
        rows.append(_shop_month("K1", per, vol, "Karachi"))
    rows.append(_shop_month("G1", "2025-08", 40.0, "GhostCity"))
    rows.append(_shop_month("G1", "2026-08", 1.0, "GhostCity"))
    sm = pd.DataFrame(rows)
    stores = sm.drop_duplicates("store_id")[
        ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
    ]
    pack = build_hierarchy_pack(
        sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    )
    ghost = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "GhostCity")].iloc[0]
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    assert float(ghost["expected_mt"]) < 5.0
    assert float(khi["expected_mt"]) > 30.0

