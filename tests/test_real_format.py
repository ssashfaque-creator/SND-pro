"""Tests against the real SSRS CSV shape and staggered store onboarding."""

from pathlib import Path

import pandas as pd

from sndintel.features import add_calendar_panel, build_features, rebuild_shop_month
from sndintel.ingest.ssrs import parse_sales_file


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
