"""Outlet Date Wise daily sales: monthly rollup + shop-list mapping."""

from datetime import date, datetime

import pandas as pd

from sndintel.ingest.daily import overlay_store_attrs, parse_outlet_date_wise
from sndintel.ingest.pipeline import run_pipeline
from sndintel.ingest.ssrs import parse_sales_file
from sndintel.storage import connect, read_sql


def _write_daily(path, extra_shop=None):
    dates = [date(2026, 5, 1), date(2026, 5, 31), date(2026, 6, 15), date(2026, 7, 10), date(2026, 8, 15)]
    header = ["Outlet Date Wise Sale", None, *dates]
    labels = ["POP Code", "POP NAME", *["Secondary Sales UOM"] * len(dates)]
    rows = [
        header,
        labels,
        ["T0001601401000016136", "Hameed GS", 0.10, 0.20, 0.30, 0.40, 0.05],
        ["T0000100100100009849", "B.MART", 1.00, 2.00, 3.00, 4.00, 1.50],
        ["Grand Total", "Grand Total", 1.10, 2.20, 3.30, 4.40, 1.55],
    ]
    if extra_shop:
        rows.append(extra_shop)
    sales = pd.DataFrame(rows)
    filt = pd.DataFrame(
        [
            [None, "Outlet Date Wise Sale", None, "Execution Date & Time: 21/08/2026 19:04:50", None, "User ID: shahmir"],
            [None, "Calendar Month: Multiple", None, "Calendar Year: 2026", None, "UOM: Tons"],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sales.to_excel(writer, sheet_name="Outlet Date Wise Sale", header=False, index=False)
        filt.to_excel(writer, sheet_name="Selected Filters", header=False, index=False)


def _write_universe(path):
    rows = [
        ["DISTRIBUTOR NAME", "Area", "SECTION LONG DESCRIPTION", "DSR NAME", "POP Code", "POP NAME"],
        ["Agha Traders (Quetta)", "Quetta", "Alamdar Road", "ASHRAF KHAN", "T0001601401000016136", "Hameed GS"],
        ["S.M Traders (F.B Area)", "Karachi", "F.B Area", "IMAD", "T0000100100100009849", "B.MART"],
    ]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def test_daily_parser_sums_days_to_months_and_skips_totals(tmp_path):
    path = tmp_path / "daily.xlsx"
    _write_daily(path)
    df, report = parse_sales_file(path)
    assert report.strategy == "outlet_date_wise"
    assert report.params.get("execution_date") == "21/08/2026"
    assert report.params.get("execution_time") == "19:04:50"
    assert "Grand Total" not in set(df["store_id"])
    h = df[df["store_id"] == "T0001601401000016136"].set_index("period")["volume_mt"]
    assert abs(float(h["2026-05"]) - 0.30) < 1e-9
    assert abs(float(h["2026-06"]) - 0.30) < 1e-9
    assert abs(float(h["2026-07"]) - 0.40) < 1e-9
    assert abs(float(h["2026-08"]) - 0.05) < 1e-9
    assert set(df["sku"]) == {"ALL"}


def test_daily_maps_pop_to_distributor_via_universe(tmp_path):
    sales_path = tmp_path / "daily.xlsx"
    uni_path = tmp_path / "universe.xlsx"
    _write_daily(sales_path)
    _write_universe(uni_path)
    facts, _ = parse_outlet_date_wise(sales_path)
    assert facts["distributor"].isna().all() or (facts["distributor"].astype(str) == "None").all()
    uni = pd.DataFrame(
        [
            {
                "store_id": "T0000100100100009849",
                "distributor": "S.M Traders (F.B Area)",
                "dsr_name": "IMAD",
                "section": "F.B Area",
                "store_name": "B.MART",
                "in_universe": 1,
            }
        ]
    )
    mapped = overlay_store_attrs(facts, uni)
    sm = mapped[mapped["store_id"] == "T0000100100100009849"]
    assert set(sm["distributor"]) == {"S.M Traders (F.B Area)"}
    assert set(sm["dsr_name"]) == {"IMAD"}


def test_daily_pipeline_maps_and_tags_open_mtd(tmp_path):
    sales_path = tmp_path / "daily.xlsx"
    uni_path = tmp_path / "universe.xlsx"
    db = tmp_path / "wh.db"
    _write_daily(sales_path)
    _write_universe(uni_path)
    result = run_pipeline(sales_path, universe_path=uni_path, db_path=db)
    assert result["parser"] == "outlet_date_wise"
    assert result["open_mtd_period"] == "2026-08"
    with connect(db) as conn:
        facts = read_sql(conn, "SELECT * FROM sales_facts")
        sm = read_sql(conn, "SELECT * FROM shop_month")
        ledger = read_sql(conn, "SELECT * FROM period_ledger")
    bm = facts[facts["store_id"] == "T0000100100100009849"]
    assert set(bm["distributor"]) == {"S.M Traders (F.B Area)"}
    may = float(bm.loc[bm["period"] == "2026-05", "volume_mt"].sum())
    assert abs(may - 3.0) < 1e-9
    aug = sm[(sm["store_id"] == "T0000100100100009849") & (sm["period"] == "2026-08")]
    assert abs(float(aug["volume_mt"].sum()) - 1.50) < 1e-9
    assert "Karachi" in set(sm["city"].dropna().astype(str))
    open_row = ledger[ledger["period"] == "2026-08"].iloc[0]
    assert open_row["status"] == "mtd_open"
    assert int(open_row["as_of_day"]) == 21
