"""The hierarchy engine must name cities, people, and shops from the data.

It is not a canned strategy pack: diagnosis flips with the shape of the gap.
"""

from __future__ import annotations

import pandas as pd

from sndintel.hierarchy import build_hierarchy_pack
from sndintel.ingest.pipeline import rescore_warehouse, run_pipeline
from sndintel.storage import connect, read_sql


def _row(
    store_id: str,
    period: str,
    volume: float,
    city: str,
    dist: str,
    dsr: str,
    section: str,
    name: str,
    billed: int | None = None,
    zone: str = "South",
) -> dict:
    return {
        "store_id": store_id,
        "period": period,
        "year": int(period[:4]),
        "month": int(period[5:7]),
        "volume_mt": volume,
        "sku_count": 1,
        "billed": 1 if billed is None and volume > 0 else (0 if billed is None else billed),
        "distributor": dist,
        "dsr_name": dsr,
        "section": section,
        "store_name": name,
        "zone": zone,
        "city": city,
    }


def _stores(rows: list[dict], extra_ids: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    seen = {}
    for r in rows:
        seen[r["store_id"]] = r
    recs = []
    for r in seen.values():
        recs.append(
            {
                "store_id": r["store_id"],
                "store_name": r["store_name"],
                "distributor": r["distributor"],
                "dsr_name": r["dsr_name"],
                "zone": r["zone"],
                "city": r["city"],
                "section": r["section"],
            }
        )
    for sid, city in extra_ids or []:
        recs.append(
            {
                "store_id": sid,
                "store_name": sid,
                "distributor": "Whitespace Dist",
                "dsr_name": "Whitespace DSR",
                "zone": "Central",
                "city": city,
                "section": "Open beat",
            }
        )
    return pd.DataFrame(recs)


def test_drop_size_city_names_continuing_shops():
    """Same doors, smaller drops → drop_size, and the named shop/DSR/dist are the targets."""
    rows = []
    for sid, name, now, ly in [
        ("K1", "Kifaya Mart", 5.0, 21.0),
        ("K2", "Diamond Super", 11.0, 23.0),
        ("K3", "Kifaya KDA", 12.0, 24.0),
    ]:
        rows.append(_row(sid, "2026-08", now, "Karachi", "Eva Foods", "Amir Surveyor", "Nazimabad", name))
        rows.append(_row(sid, "2025-08", ly, "Karachi", "Eva Foods", "Amir Surveyor", "Nazimabad", name))
    sm = pd.DataFrame(rows)
    stores = _stores(rows)
    pack = build_hierarchy_pack(sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    cities = pack.units[pack.units["grain"] == "city"]
    khi = cities[cities["grain_id"] == "Karachi"].iloc[0]
    assert khi["diagnosis"] == "drop_size"
    assert khi["gap_mt"] < -30
    dists = pack.targets[pack.targets["grain"] == "distributor"]
    assert "Eva Foods" in set(dists["entity_name"])
    dsrs = pack.targets[pack.targets["grain"] == "dsr"]
    assert "Amir Surveyor" in set(dsrs["entity_name"])
    shops = pack.targets[pack.targets["grain"] == "shop"]
    names = set(shops["entity_name"])
    assert "Kifaya Mart" in names
    # Drop-size lists continuing doors, not a lost-shop dump.
    assert (shops["volume_mt"] > 0).all()


def test_coverage_city_names_quiet_material_doors():
    rows = []
    # Continuing door holds drop size.
    rows.append(_row("M0", "2026-08", 2.0, "Multan", "Danial", "Ali DSR", "Cantt", "Stable Mart", zone="Central"))
    rows.append(_row("M0", "2025-08", 2.0, "Multan", "Danial", "Ali DSR", "Cantt", "Stable Mart", zone="Central"))
    for i in range(1, 6):
        rows.append(
            _row(
                f"M{i}",
                "2025-08",
                2.0,
                "Multan",
                "Danial",
                "Ali DSR",
                "Cantt",
                f"Lost Shop {i}",
                zone="Central",
            )
        )
        rows.append(
            _row(
                f"M{i}",
                "2026-08",
                0.0,
                "Multan",
                "Danial",
                "Ali DSR",
                "Cantt",
                f"Lost Shop {i}",
                billed=0,
                zone="Central",
            )
        )
    sm = pd.DataFrame(rows)
    pack = build_hierarchy_pack(sm, _stores(rows), ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    mul = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Multan")].iloc[0]
    assert mul["diagnosis"] == "coverage"
    assert mul["lost_n"] >= 5
    shops = pack.targets[(pack.targets["city"] == "Multan") & (pack.targets["grain"] == "shop")]
    assert "Lost Shop 1" in set(shops["entity_name"])
    assert "Danial" in set(pack.targets[pack.targets["grain"] == "distributor"]["entity_name"])


def test_whitespace_city_uses_universe_not_the_same_20_doors():
    rows = []
    rows.append(_row("F1", "2026-08", 0.04, "Faisalabad", "Local Dist", "Local DSR", "A", "Tiny 1", zone="Central"))
    rows.append(_row("F2", "2026-08", 0.03, "Faisalabad", "Local Dist", "Local DSR", "A", "Tiny 2", zone="Central"))
    extra = [(f"U{i:03d}", "Faisalabad") for i in range(40)]
    sm = pd.DataFrame(rows)
    stores = _stores(rows, extra_ids=extra)
    pack = build_hierarchy_pack(sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    fsd = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Faisalabad")].iloc[0]
    assert fsd["diagnosis"] == "whitespace"
    assert fsd["universe"] >= 40
    shops = pack.targets[(pack.targets["city"] == "Faisalabad") & (pack.targets["grain"] == "shop")]
    assert not shops.empty
    assert shops["entity_id"].str.startswith("U").any()


def test_national_hole_is_sum_of_city_holes():
    rows = []
    rows.append(_row("K1", "2026-08", 10.0, "Karachi", "Eva Foods", "Amir Surveyor", "A", "Shop K"))
    rows.append(_row("K1", "2025-08", 30.0, "Karachi", "Eva Foods", "Amir Surveyor", "A", "Shop K"))
    rows.append(_row("L1", "2026-08", 8.0, "Lahore", "Lahore Dist", "Lahore DSR", "B", "Shop L", zone="Central"))
    rows.append(_row("L1", "2025-08", 12.0, "Lahore", "Lahore Dist", "Lahore DSR", "B", "Shop L", zone="Central"))
    pack = build_hierarchy_pack(
        pd.DataFrame(rows),
        _stores(rows),
        ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]),
    )
    cities = pack.units[pack.units["grain"] == "city"]
    nat = pack.units[pack.units["grain"] == "national"].iloc[0]
    assert abs(float(nat["gap_mt"]) - float(cities["gap_mt"].sum())) < 1e-6
    assert pack.national["n_cities"] == 2


def test_open_mtd_prorates_expected():
    rows = []
    rows.append(_row("K1", "2026-08", 10.0, "Karachi", "Eva Foods", "Amir Surveyor", "A", "Shop K"))
    rows.append(_row("K1", "2025-08", 31.0, "Karachi", "Eva Foods", "Amir Surveyor", "A", "Shop K"))
    ledger = pd.DataFrame(
        [
            {
                "period": "2026-08",
                "status": "mtd_open",
                "as_of_day": 10,
                "days_in_month": 31,
                "execution_date": "2026-08-10",
            }
        ]
    )
    pack = build_hierarchy_pack(pd.DataFrame(rows), _stores(rows), ledger=ledger)
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    expected = 31.0 * (10 / 31)
    assert abs(float(khi["expected_mt"]) - expected) < 0.05
    assert pack.national["intra_month_source"] == "elapsed_days"
    # Elapsed days of last August, not a hole vs the full closed month.
    assert float(khi["gap_mt"]) > -21


def test_pipeline_persists_scorecards_and_rescore_does_not_need_a_file(demo, tmp_path):
    db = tmp_path / "warehouse.db"
    result = run_pipeline(demo["sales"], shop_path=demo["shops"], db_path=db)
    assert result["n_cities"] >= 1
    with connect(db) as conn:
        units = read_sql(conn, "SELECT * FROM unit_scorecards")
        targets = read_sql(conn, "SELECT * FROM focus_targets")
        season = read_sql(conn, "SELECT * FROM seasonality_index")
        conn.execute("DELETE FROM unit_scorecards")
        conn.execute("DELETE FROM focus_targets")
    assert "city" in set(units["grain"])
    assert not season.empty
    assert "national" in set(season["grain"])
    assert result["n_targets"] == len(targets) or not targets.empty or result["n_cities"] >= 1
    scored = rescore_warehouse(db)
    assert scored["parser"] == "rescore"
    assert scored["n_cities"] >= 1
    with connect(db) as conn:
        units2 = read_sql(conn, "SELECT * FROM unit_scorecards WHERE grain = 'city'")
    assert not units2.empty
