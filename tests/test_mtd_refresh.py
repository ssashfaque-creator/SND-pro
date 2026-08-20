"""Incremental MTD ingest: a new August extract replaces August and leaves July alone."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sndintel.ingest.pipeline import run_pipeline
from sndintel.ingest.ssrs import parse_sales_file
from sndintel.storage import connect, read_sql


MONTH_NAME = {7: "July", 8: "August"}


def _write_ssrs_csv(path: Path, rows: list[dict], execution: str) -> Path:
    lines = [
        "txtRectangle,txtFooter,txtCulture",
        f"Shop SKU Wise Sale,User ID: tester,UOM: Tons,Execution Date & Time: {execution}",
        "",
        "txtCorner_0,txt_cDISTRIBUTOR_NAME,txt_cDSR_NAME,txt_cSECTION_LONG_DESCRIPTION,"
        "txt_cPOP_Code,txt_cPOP_NAME,txt_cSKU_LONG_DESCRIPTION,txt_Calendar_Year,"
        "txt_Calendar_Month,uval_MTD_Secondary_Sales_UOM",
    ]
    for r in rows:
        month = r["month"] if isinstance(r["month"], str) else MONTH_NAME[int(r["month"])]
        lines.append(
            ",".join(
                [
                    "x",
                    r["distributor"],
                    r["dsr_name"],
                    r["section"],
                    r["store_id"],
                    r["store_name"],
                    r["sku"],
                    str(r["year"]),
                    month,
                    str(r["volume_mt"]),
                ]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _fact(**kwargs) -> dict:
    base = {
        "distributor": "Agha Traders",
        "dsr_name": "ASHRAF KHAN",
        "section": "Alamdar Road",
        "sku": "Maan Banaspati Pouch 1X5Kg",
    }
    base.update(kwargs)
    return base


def _shops_xlsx(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "distributor": "Agha Traders",
                "dsr_name": "ASHRAF KHAN",
                "store_id": "T0000000001",
                "store_name": "Hameed GS",
                "category_1": "GT",
                "category_2": "A",
                "category_3": "GT",
                "category_4": "",
                "zone": "West",
                "city": "Quetta",
                "section": "Alamdar Road",
            },
            {
                "distributor": "Agha Traders",
                "dsr_name": "ASHRAF KHAN",
                "store_id": "T0000000002",
                "store_name": "Old Mart",
                "category_1": "GT",
                "category_2": "B",
                "category_3": "GT",
                "category_4": "",
                "zone": "West",
                "city": "Quetta",
                "section": "Alamdar Road",
            },
            {
                "distributor": "Coastal Foods",
                "dsr_name": "BILAL SHAIKH",
                "store_id": "T0000000003",
                "store_name": "New Mart",
                "category_1": "GT",
                "category_2": "A",
                "category_3": "GT",
                "category_4": "",
                "zone": "South",
                "city": "Karachi",
                "section": "Korangi",
            },
        ]
    ).to_excel(path, index=False)
    return path


HISTORY = [
    _fact(store_id="T0000000001", store_name="Hameed GS", year=2025, month=7, volume_mt=1.00),
    _fact(store_id="T0000000001", store_name="Hameed GS", year=2025, month=8, volume_mt=1.20),
    _fact(store_id="T0000000001", store_name="Hameed GS", year=2026, month=7, volume_mt=1.10),
    _fact(store_id="T0000000001", store_name="Hameed GS", year=2026, month=8, volume_mt=0.60),
    _fact(store_id="T0000000002", store_name="Old Mart", year=2025, month=7, volume_mt=2.00),
    _fact(store_id="T0000000002", store_name="Old Mart", year=2025, month=8, volume_mt=2.10),
    _fact(store_id="T0000000002", store_name="Old Mart", year=2026, month=7, volume_mt=1.80),
    _fact(store_id="T0000000002", store_name="Old Mart", year=2026, month=8, volume_mt=0.90),
]

AUGUST_REFRESH = [
    # MTD grew vs the earlier August extract.
    _fact(store_id="T0000000001", store_name="Hameed GS", year=2026, month=8, volume_mt=0.95),
    # New shop billed during the rest of the month.
    _fact(
        store_id="T0000000003",
        store_name="New Mart",
        year=2026,
        month=8,
        volume_mt=0.40,
        distributor="Coastal Foods",
        dsr_name="BILAL SHAIKH",
        section="Korangi",
    ),
    # T0000000002 is intentionally absent — stale August MTD must be removed.
]


def test_execution_date_parsed_from_ssrs_chrome(tmp_path):
    path = _write_ssrs_csv(tmp_path / "sale.csv", HISTORY, "20/08/2026 14:03:11")
    _, report = parse_sales_file(path)
    assert report.params.get("execution_date") == "20/08/2026"
    assert report.params.get("execution_time") == "14:03:11"


def test_august_only_file_replaces_august_mtd_and_keeps_july(tmp_path):
    shops = _shops_xlsx(tmp_path / "shops.xlsx")
    first = _write_ssrs_csv(tmp_path / "jul_aug.csv", HISTORY, "15/08/2026 09:00:00")
    db = tmp_path / "warehouse.db"
    r1 = run_pipeline(first, shop_path=shops, db_path=db)
    assert r1["latest_period"] == "2026-08"
    assert r1["open_mtd_period"] == "2026-08"
    assert set(r1["replaced_periods"]) >= {"2025-07", "2025-08", "2026-07", "2026-08"}

    with connect(db) as conn:
        facts1 = read_sql(conn, "SELECT * FROM sales_facts")
        july1 = float(facts1.loc[facts1["period"] == "2026-07", "volume_mt"].sum())
        aug1 = float(facts1.loc[facts1["period"] == "2026-08", "volume_mt"].sum())
    assert abs(july1 - 2.90) < 1e-6
    assert abs(aug1 - 1.50) < 1e-6

    second = _write_ssrs_csv(tmp_path / "aug_only.csv", AUGUST_REFRESH, "20/08/2026 14:03:11")
    r2 = run_pipeline(second, shop_path=shops, db_path=db)
    assert r2["replaced_periods"] == ["2026-08"]
    assert r2["open_mtd_period"] == "2026-08"
    assert r2["latest_period"] == "2026-08"

    with connect(db) as conn:
        facts = read_sql(conn, "SELECT * FROM sales_facts")
        shop_month = read_sql(conn, "SELECT * FROM shop_month")
        ledger = read_sql(conn, "SELECT * FROM period_ledger")
        insights = read_sql(conn, "SELECT * FROM insights")
        kpis = read_sql(conn, "SELECT * FROM kpi_snapshots")

    july2 = float(facts.loc[facts["period"] == "2026-07", "volume_mt"].sum())
    aug2 = float(facts.loc[facts["period"] == "2026-08", "volume_mt"].sum())
    assert abs(july2 - 2.90) < 1e-6, "July must stay frozen when the new file is August-only"
    assert abs(aug2 - 1.35) < 1e-6, "August MTD must become the new extract (0.95 + 0.40)"

    aug_ids = set(facts.loc[facts["period"] == "2026-08", "store_id"])
    assert "T0000000001" in aug_ids
    assert "T0000000003" in aug_ids
    assert "T0000000002" not in aug_ids, "stale August shop-SKU lines must be deleted"
    assert "T0000000002" in set(facts.loc[facts["period"] == "2026-07", "store_id"])

    led = ledger.set_index("period")
    assert led.loc["2026-07", "status"] == "closed"
    assert led.loc["2026-08", "status"] == "mtd_open"
    assert int(led.loc["2026-08", "as_of_day"]) == 20
    assert int(led.loc["2026-08", "days_in_month"]) == 31

    types = set(insights["type"])
    assert "warehouse_position" in types
    pos = insights[insights["type"] == "warehouse_position"].iloc[0]
    assert "open MTD" in pos["narrative"] or "day 20" in pos["narrative"]
    assert "2.9" in pos["narrative"] or "July" in pos["narrative"] or "closed YTD" in pos["narrative"].lower()
    # Insights are from the whole warehouse, not only the new file.
    assert "2026-07" in set(shop_month["period"])
    assert abs(float(shop_month.loc[shop_month["period"] == "2026-07", "volume_mt"].sum()) - 2.90) < 1e-6

    nat = kpis[(kpis["grain"] == "national") & (kpis["period"] == "2026-08")]
    assert not nat.empty
    assert nat.iloc[0]["period_status"] == "mtd_open"
    assert abs(float(nat.iloc[0]["volume_mt"]) - 1.35) < 1e-6
    # Run-rate uses 31/20 of MTD, not raw MTD vs full August last year.
    assert float(nat.iloc[0]["run_rate_mt"]) > float(nat.iloc[0]["volume_mt"])
    bridge = insights[insights["type"] == "volume_bridge"]
    assert not bridge.empty
    assert "run-rate" in bridge.iloc[0]["narrative"].lower() or "pace" in bridge.iloc[0]["narrative"].lower()
