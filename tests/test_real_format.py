"""Tests against the real SSRS CSV shape and staggered store onboarding."""

from pathlib import Path

import pandas as pd

from sndintel.features import add_calendar_panel, build_features, rebuild_shop_month
from sndintel.ingest.pipeline import rescore_warehouse
from sndintel.ingest.ssrs import collapse_sales_facts, parse_sales_file
from sndintel.storage import connect, init_db, read_sql, upsert_dataframe


def _write_ragged_csv(path: Path) -> Path:
    # Parameter rows are narrower than the tablix — this is what pandas' C engine rejects.
    lines = [
        "txtRectangle,txtFooter,txtCulture",
        "Shop SKU Wise Sale,User ID: shahmir,UOM: Tons,Execution Date & Time: 20/08/2026 14:03:11",
        "",
        "txtCorner_0,txtCorner_1,txt_cDISTRIBUTOR_NAME,txt_cDSR_NAME,txt_cSECTION_LONG_DESCRIPTION,txt_cPOP_Code,txt_cPOP_NAME,txt_cSKU_LONG_DESCRIPTION,txt_Calendar_Year,txt_Calendar_Month,uval_MTD_Secondary_Sales_UOM,txt_GrandTotal_Calendar_Year_1,val_TotalC_0_0,val_TotalC_1_0,txt_totalPOP_NAME_14,val_shop_total",
        "DISTRIBUTOR NAME,DSR NAME,Agha Traders,ASHRAF KHAN,Alamdar Road,T0001601401000016136,Hameed GS,Maan Banaspati Pouch 1X5Kg,2025,July,,2025 Total,,0.05,Hameed GS Total,0.05",
        "DISTRIBUTOR NAME,DSR NAME,Agha Traders,ASHRAF KHAN,Alamdar Road,T0001601401000016136,Hameed GS,Maan Banaspati Pouch 1X5Kg,2026,July,0.04,2026 Total,0.04,0.05,Hameed GS Total,0.04",
        "DISTRIBUTOR NAME,DSR NAME,Coastal Foods,BILAL SHAIKH,Korangi,T0000100100100009849,New Area Mart,Eva-Cooking Oil Stand Up Pouch 1X5Ltr,2026,July,0.20,2026 Total,0.20,,New Area Mart Total,0.20",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_ragged_csv_year_group_volume(tmp_path):
    csv_path = _write_ragged_csv(tmp_path / "sale.csv")
    df, report = parse_sales_file(csv_path)
    assert report.strategy == "ssrs_field_ids"
    assert report.params.get("execution_date") == "20/08/2026"
    hameed = df[df["store_id"] == "T0001601401000016136"].sort_values("period")
    assert list(hameed["period"]) == ["2025-07", "2026-07"]
    # 2025 MTD column empty — must read val_TotalC_1_0, not drop the row.
    assert abs(float(hameed.iloc[0]["volume_mt"]) - 0.05) < 1e-9
    assert abs(float(hameed.iloc[1]["volume_mt"]) - 0.04) < 1e-9
    newbie = df[df["store_id"] == "T0000100100100009849"]
    assert list(newbie["period"]) == ["2026-07"]
    assert abs(float(newbie.iloc[0]["volume_mt"]) - 0.20) < 1e-9


def test_staggered_stores_are_not_prefilled_as_zeros(tmp_path):
    csv_path = _write_ragged_csv(tmp_path / "sale.csv")
    sales, _ = parse_sales_file(csv_path)
    stores = pd.DataFrame(
        [
            {
                "store_id": "T0001601401000016136",
                "store_name": "Hameed GS",
                "distributor": "Agha Traders",
                "dsr_name": "ASHRAF KHAN",
                "section": "Alamdar Road",
                "zone": "West",
                "city": "Quetta",
                "category_1": None,
                "category_2": None,
                "category_3": "GT",
                "category_4": None,
            },
            {
                "store_id": "T0000100100100009849",
                "store_name": "New Area Mart",
                "distributor": "Coastal Foods",
                "dsr_name": "BILAL SHAIKH",
                "section": "Korangi",
                "zone": "South",
                "city": "Karachi",
                "category_1": None,
                "category_2": None,
                "category_3": "GT",
                "category_4": None,
            },
        ]
    )
    sm = add_calendar_panel(rebuild_shop_month(sales, stores), stores)
    new_hist = sm[sm["store_id"] == "T0000100100100009849"]
    # Must not invent 2025-07 zeros for a shop that was not on file yet.
    assert "2025-07" not in set(new_hist["period"])
    feats = build_features(sm, sales)
    row = feats[(feats["store_id"] == "T0000100100100009849") & (feats["period"] == "2026-07")].iloc[0]
    assert int(row["yoy_comparable"]) == 0
    assert int(row["months_on_file"]) == 1
    old = feats[(feats["store_id"] == "T0001601401000016136") & (feats["period"] == "2026-07")].iloc[0]
    assert int(old["yoy_comparable"]) == 1
    assert abs(float(old["lag_12"]) - 0.05) < 1e-9


def _ssrs_line(store_id, store_name, sku, year, month, volume, *, dist="Agha Traders", dsr="ASHRAF KHAN", section="Alamdar Road"):
    mtd = "" if volume is None else str(volume)
    return (
        f"DISTRIBUTOR NAME,DSR NAME,{dist},{dsr},{section},{store_id},{store_name},"
        f"{sku},{year},{month},{mtd},{year} Total,{mtd},,{store_name} Total,{mtd}"
    )


def _ssrs_csv(path: Path, body_lines: list[str]) -> Path:
    header = [
        "txtRectangle,txtFooter,txtCulture",
        "Shop SKU Wise Sale,User ID: tester,UOM: Tons,Execution Date & Time: 20/08/2026 14:03:11",
        "",
        "txtCorner_0,txtCorner_1,txt_cDISTRIBUTOR_NAME,txt_cDSR_NAME,txt_cSECTION_LONG_DESCRIPTION,"
        "txt_cPOP_Code,txt_cPOP_NAME,txt_cSKU_LONG_DESCRIPTION,txt_Calendar_Year,txt_Calendar_Month,"
        "uval_MTD_Secondary_Sales_UOM,txt_GrandTotal_Calendar_Year_1,val_TotalC_0_0,val_TotalC_1_0,"
        "txt_totalPOP_NAME_14,val_shop_total",
    ]
    path.write_text("\n".join(header + body_lines), encoding="utf-8")
    return path


def test_duplicate_identical_sku_lines_are_not_summed(tmp_path):
    """SSRS often repeats the same shop/SKU/month. That is a copy, not 2× volume."""
    line = _ssrs_line(
        "T0001601401000016136", "Hameed GS", "Maan Banaspati Pouch 1X5Kg", 2026, "July", 0.04
    )
    csv_path = _ssrs_csv(tmp_path / "dup.csv", [line, line])
    df, report = parse_sales_file(csv_path)
    hameed = df[(df["store_id"] == "T0001601401000016136") & (df["period"] == "2026-07")]
    assert len(hameed) == 1
    assert abs(float(hameed["volume_mt"].sum()) - 0.04) < 1e-9
    assert any("duplicate" in w.lower() for w in report.warnings)


def test_sku_named_like_shop_total_is_dropped(tmp_path):
    csv_path = _ssrs_csv(
        tmp_path / "tot.csv",
        [
            _ssrs_line("T0001601401000016136", "Hameed GS", "Hameed GS Total", 2026, "July", 0.04),
            _ssrs_line("T0001601401000016136", "Hameed GS", "Maan Banaspati Pouch 1X5Kg", 2026, "July", 0.04),
        ],
    )
    df, _ = parse_sales_file(csv_path)
    assert df["sku"].str.contains("Total", case=False).sum() == 0
    assert abs(float(df["volume_mt"].sum()) - 0.04) < 1e-9


def test_collapse_does_not_sum_whitespace_sku_copies():
    df = pd.DataFrame(
        [
            {"store_id": " T1 ", "sku": "Maan 1kg", "period": "2026-07", "volume_mt": 0.04, "year": 2026, "month": 7},
            {"store_id": "T1", "sku": "Maan  1kg", "period": "2026-07", "volume_mt": 0.04, "year": 2026, "month": 7},
        ]
    )
    out = collapse_sales_facts(df)
    assert len(out) == 1
    assert abs(float(out["volume_mt"].sum()) - 0.04) < 1e-9


def test_collapse_prefers_mtd_over_year_group_backfill():
    df = pd.DataFrame(
        [
            {
                "store_id": "T1",
                "sku": "Maan 1kg",
                "period": "2026-07",
                "volume_mt": 0.05,
                "_prefer_mtd": False,
            },
            {
                "store_id": "T1",
                "sku": "Maan 1kg",
                "period": "2026-07",
                "volume_mt": 0.04,
                "_prefer_mtd": True,
            },
        ]
    )
    out = collapse_sales_facts(df)
    assert len(out) == 1
    assert abs(float(out.iloc[0]["volume_mt"]) - 0.04) < 1e-9


def test_embedded_shop_total_sku_is_dropped_but_two_equal_skus_are_kept():
    tot = pd.DataFrame(
        [
            {"store_id": "T1", "sku": "Oil A", "period": "2026-07", "volume_mt": 0.10},
            {"store_id": "T1", "sku": "Oil B", "period": "2026-07", "volume_mt": 0.20},
            {"store_id": "T1", "sku": "Combo Pack", "period": "2026-07", "volume_mt": 0.30},
        ]
    )
    out = collapse_sales_facts(tot)
    assert set(out["sku"]) == {"Oil A", "Oil B"}
    assert abs(float(out["volume_mt"].sum()) - 0.30) < 1e-9

    twins = pd.DataFrame(
        [
            {"store_id": "T1", "sku": "Oil A", "period": "2026-07", "volume_mt": 0.10},
            {"store_id": "T1", "sku": "Oil B", "period": "2026-07", "volume_mt": 0.10},
        ]
    )
    kept = collapse_sales_facts(twins)
    assert len(kept) == 2
    assert abs(float(kept["volume_mt"].sum()) - 0.20) < 1e-9


def test_rescore_collapses_duplicate_sku_keys_in_warehouse(tmp_path):
    """Whitespace SKU copies already in SQLite must not stay as 2× billed after rebuild."""
    db = tmp_path / "warehouse.db"
    init_db(db)
    rows = pd.DataFrame(
        [
            {
                "store_id": "T0000000001",
                "sku": "Maan 1kg",
                "period": "2026-07",
                "year": 2026,
                "month": 7,
                "volume_mt": 0.04,
                "distributor": "Agha Traders",
                "dsr_name": "ASHRAF KHAN",
                "section": "Alamdar Road",
                "store_name": "Hameed GS",
                "source_file": "old.csv",
                "ingested_at": "2026-08-20T00:00:00Z",
            },
            {
                "store_id": "T0000000001",
                "sku": "Maan  1kg",
                "period": "2026-07",
                "year": 2026,
                "month": 7,
                "volume_mt": 0.04,
                "distributor": "Agha Traders",
                "dsr_name": "ASHRAF KHAN",
                "section": "Alamdar Road",
                "store_name": "Hameed GS",
                "source_file": "old.csv",
                "ingested_at": "2026-08-20T00:00:00Z",
            },
        ]
    )
    with connect(db) as conn:
        upsert_dataframe(conn, "sales_facts", rows, ["store_id", "sku", "period"])
    rescore_warehouse(db)
    with connect(db) as conn:
        facts = read_sql(conn, "SELECT * FROM sales_facts")
        shop_month = read_sql(conn, "SELECT * FROM shop_month")
    assert len(facts) == 1
    assert abs(float(facts["volume_mt"].sum()) - 0.04) < 1e-9
    july = shop_month[shop_month["period"].astype(str) == "2026-07"]
    assert abs(float(july["volume_mt"].sum()) - 0.04) < 1e-9
