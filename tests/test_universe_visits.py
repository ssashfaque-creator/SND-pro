"""Live universe + visit calls: identity, ffill, unvisited vs unbilled."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sndintel.coverage import build_coverage_book
from sndintel.features import rebuild_shop_month
from sndintel.hierarchy import build_hierarchy_pack
from sndintel.ingest.universe import parse_universe
from sndintel.ingest.visits import parse_visit_calls
from sndintel.briefing import build_strategy_pack

DRIVE = Path("/tmp/sndintel-drive")


def _write_universe(path: Path) -> None:
    rows = [
        ["DISTRIBUTOR NAME", "Area", "SECTION LONG DESCRIPTION", "DSR NAME", "POP Code", "POP NAME", "TOTAL UNIVERSE OUTLETS"],
        [" SA Traders", "Lahore", "Bahria Town", "Said", "T0000511000100017231", "MARVELA MART", 1],
        [None, None, None, None, "T0000511000100017232", "IQBAL STORE", 1],
        [None, None, "EME SOCIETY", "Said", "T0000511000100017277", "Elegant Store", 1],
        ["Agha Traders (Quetta)", "Quetta", "Alamdar Road", "ASHRAF KHAN", "T0001601401000016136", "Hameed GS", 1],
    ]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def _write_visits(path: Path) -> None:
    lines = [
        "txtRectangle_FooterReport_Name,txtRectangle_FooterExecution_Date___Time,txtFooter_DimDatesCalendarMonth,txtFooter_DimDatesCalendarYear,txtFooter_DimDatesFullDate",
        "Shops Visit Calls,Execution Date & Time: 21/08/2026 12:07:24,Calendar Month: August,Calendar Year: 2026,FullDate: 2026-08-21",
        "txtCorner_0_0,txtCorner_1_0,txt_cDISTRIBUTOR_NAME,txt_cArea,txt_cDSR_NAME,txt_cSECTION_LONG_DESCRIPTION,txt_cPOP_Code,txt_cPOP_NAME,txt_MTD_VISITED_CALLS,uval_MTD_VISITED_CALLS",
        "DISTRIBUTOR NAME,Area,Agha Traders (Quetta),Quetta,ASHRAF KHAN,Alamdar Road,T0001601401000016136,Hameed GS,MTD_VISITED_CALLS,2.00",
        "DISTRIBUTOR NAME,Area,SA Traders,Lahore,Said,Bahria Town,T0000511000100017231,MARVELA MART,MTD_VISITED_CALLS,1.00",
        "DISTRIBUTOR NAME,Area,SA Traders,Lahore,Said,Bahria Town,T0000511000100017232,IQBAL STORE,MTD_VISITED_CALLS,0.00",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_universe_ffills_merged_cells_and_maps_area_to_city(tmp_path):
    path = tmp_path / "universe.xlsx"
    _write_universe(path)
    shops, report = parse_universe(path)
    assert report.n_clean_rows == 4
    by_id = shops.set_index("store_id")
    iqbal = by_id.loc["T0000511000100017232"]
    assert iqbal["distributor"].strip() == "SA Traders"
    assert iqbal["city"] == "Lahore"
    assert iqbal["dsr_name"] == "Said"
    assert iqbal["section"] == "Bahria Town"
    elegant = by_id.loc["T0000511000100017277"]
    assert elegant["section"] == "EME SOCIETY"
    assert elegant["city"] == "Lahore"
    hameed = by_id.loc["T0001601401000016136"]
    assert hameed["city"] == "Quetta"


def test_visit_parser_reads_ssrs_chrome_and_calendar_month(tmp_path):
    path = tmp_path / "visits.csv"
    _write_visits(path)
    visits, report = parse_visit_calls(path)
    assert report.strategy == "ssrs_field_ids"
    assert set(visits["period"]) == {"2026-08"}
    h = visits.set_index("store_id").loc["T0001601401000016136"]
    assert int(h["visits"]) == 2
    assert h["city"] == "Quetta"


def test_unvisited_unbilled_drop_size_do_not_double_count():
    stores = pd.DataFrame(
        [
            {"store_id": "A", "city": "Karachi", "distributor": "D1", "dsr_name": "X", "in_universe": 1},
            {"store_id": "B", "city": "Karachi", "distributor": "D1", "dsr_name": "X", "in_universe": 1},
            {"store_id": "C", "city": "Karachi", "distributor": "D1", "dsr_name": "X", "in_universe": 1},
        ]
    )
    shop_month = pd.DataFrame(
        [
            {"store_id": "A", "period": "2026-08", "volume_mt": 0.8, "billed": 1},
            {"store_id": "A", "period": "2026-07", "volume_mt": 1.0, "billed": 1},
            {"store_id": "A", "period": "2026-06", "volume_mt": 1.0, "billed": 1},
            {"store_id": "A", "period": "2026-05", "volume_mt": 1.0, "billed": 1},
            {"store_id": "B", "period": "2026-07", "volume_mt": 0.5, "billed": 1},
            {"store_id": "B", "period": "2026-06", "volume_mt": 0.5, "billed": 1},
            {"store_id": "B", "period": "2026-05", "volume_mt": 0.5, "billed": 1},
            {"store_id": "C", "period": "2026-07", "volume_mt": 0.4, "billed": 1},
            {"store_id": "C", "period": "2026-06", "volume_mt": 0.4, "billed": 1},
            {"store_id": "C", "period": "2026-05", "volume_mt": 0.4, "billed": 1},
        ]
    )
    visits = pd.DataFrame(
        [
            {"store_id": "A", "period": "2026-08", "visits": 2},
            {"store_id": "B", "period": "2026-08", "visits": 1},
        ]
    )
    book = build_coverage_book(stores, shop_month, visits, "2026-08", pace=1.0)
    by = book.set_index("store_id")
    assert by.loc["A", "call_status"] == "Billed"
    assert by.loc["B", "call_status"] == "Visited · not billed"
    assert by.loc["C", "call_status"] == "Unvisited"
    # A billed 0.8 vs AMS 1.0 → drop size -0.2; B unbilled −0.5; C unvisited −0.4
    assert abs(float(by.loc["A", "from_drop_size_mt"]) - (0.8 - 1.0)) < 1e-9
    assert abs(float(by.loc["B", "from_unbilled_mt"]) + 0.5) < 1e-9
    assert abs(float(by.loc["C", "from_unvisited_mt"]) + 0.4) < 1e-9
    total = book["from_drop_size_mt"].sum() + book["from_unvisited_mt"].sum() + book["from_unbilled_mt"].sum()
    assert abs(total - (book["volume_mt"].sum() - book["opportunity_mt"].sum())) < 1e-9


def test_closed_shop_history_is_dropped_from_shop_month():
    sales = pd.DataFrame(
        [
            {"store_id": "LIVE", "period": "2026-08", "year": 2026, "month": 8, "volume_mt": 5.0, "sku": "Oil", "distributor": "D", "dsr_name": "X", "section": "A", "store_name": "Live"},
            {"store_id": "DEAD", "period": "2026-08", "year": 2026, "month": 8, "volume_mt": 9.0, "sku": "Oil", "distributor": "D", "dsr_name": "X", "section": "A", "store_name": "Dead"},
            {"store_id": "DEAD", "period": "2025-08", "year": 2025, "month": 8, "volume_mt": 9.0, "sku": "Oil", "distributor": "D", "dsr_name": "X", "section": "A", "store_name": "Dead"},
        ]
    )
    stores = pd.DataFrame(
        [
            {"store_id": "LIVE", "store_name": "Live", "distributor": "D", "dsr_name": "X", "section": "A", "city": "Karachi", "zone": "South", "in_universe": 1},
            {"store_id": "DEAD", "store_name": "Dead", "distributor": "D", "dsr_name": "X", "section": "A", "city": "Karachi", "zone": "South", "in_universe": 0},
        ]
    )
    live_facts = sales[sales["store_id"].isin(stores.loc[stores["in_universe"] == 1, "store_id"])]
    sm = rebuild_shop_month(live_facts, stores[stores["in_universe"] == 1])
    assert set(sm["store_id"]) == {"LIVE"}
    assert "DEAD" not in set(sm["store_id"])


def test_country_row_and_remarks_on_city_table():
    rows = []
    for sid, city, dist, now, ly in [
        ("K1", "Karachi", "Eva", 10.0, 40.0),
        ("L1", "Lahore", "Hold", 30.0, 32.0),
    ]:
        rows.append(
            {
                "store_id": sid,
                "period": "2026-08",
                "year": 2026,
                "month": 8,
                "volume_mt": now,
                "sku_count": 1,
                "billed": 1,
                "distributor": dist,
                "dsr_name": "A",
                "section": "S",
                "store_name": sid,
                "zone": "South" if city == "Karachi" else "Central",
                "city": city,
            }
        )
        rows.append({**rows[-1], "period": "2025-08", "year": 2025, "volume_mt": ly})
        for per, y, m in [("2026-05", 2026, 5), ("2026-06", 2026, 6), ("2026-07", 2026, 7)]:
            rows.append({**rows[-2], "period": per, "year": y, "month": m, "volume_mt": ly * 0.9})
    sm = pd.DataFrame(rows)
    stores = sm.drop_duplicates("store_id")[["store_id", "store_name", "distributor", "dsr_name", "section", "zone", "city"]].copy()
    stores["in_universe"] = 1
    visits = pd.DataFrame(
        [
            {"store_id": "K1", "period": "2026-08", "visits": 1},
            {"store_id": "L1", "period": "2026-08", "visits": 2},
        ]
    )
    pack_h = build_hierarchy_pack(sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]), visits=visits)
    report = build_strategy_pack(pack_h.units, sm, period="2026-08", visits=visits)
    assert report.cities.iloc[0]["City"] == "Country"
    assert "Remarks" in report.cities.columns
    assert "Main driver" not in report.cities.columns
    remarks = " ".join(report.cities["Remarks"].dropna().astype(str))
    assert "Trend:" in remarks
    assert "Coverage:" in remarks or "visit" in remarks.lower()
    assert "From unvisited shops (MT)" in report.lagging_distributors.columns or report.lagging_distributors.empty or "From unvisited shops (MT)" in report.all_distributors.columns


def test_real_drive_universe_and_visits_if_present():
    uni = DRIVE / "Universe Shop List.xlsx"
    vis = DRIVE / "Shops_Visit_Calls.csv"
    if not uni.exists() or not vis.exists():
        return
    shops, urep = parse_universe(uni)
    assert urep.n_clean_rows > 10000
    assert shops["city"].notna().mean() > 0.95
    assert shops["distributor"].notna().mean() > 0.95
    visits, vrep = parse_visit_calls(vis)
    assert vrep.n_clean_rows > 5000
    assert set(visits["period"]) == {"2026-08"}
    overlap = set(shops["store_id"]) & set(visits["store_id"])
    assert len(overlap) > 5000
