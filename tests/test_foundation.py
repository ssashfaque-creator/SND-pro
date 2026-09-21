"""Warehouse invariants: the numbers the packs print must tie to the extract and never inflate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from sndintel.ingest.pipeline import combine_sales_frames, run_pipeline
from sndintel.mtd import parse_execution_date
from sndintel.storage import connect, read_sql

from tests.test_mtd_refresh import HISTORY, _fact, _shops_xlsx, _write_ssrs_csv

COASTAL = dict(distributor="Coastal Foods", dsr_name="BILAL SHAIKH", section="Korangi")


def _write_universe(path: Path, rows: list[list[str]]) -> Path:
    header = ["DISTRIBUTOR NAME", "Area", "SECTION LONG DESCRIPTION", "DSR NAME", "POP Code", "POP NAME"]
    pd.DataFrame([header, *rows]).to_excel(path, header=False, index=False)
    return path


def _facts(db: Path) -> pd.DataFrame:
    with connect(db) as conn:
        return read_sql(conn, "SELECT * FROM sales_facts")


def _ledger(db: Path) -> pd.DataFrame:
    with connect(db) as conn:
        return read_sql(conn, "SELECT * FROM period_ledger").set_index("period")


def test_regional_extract_replaces_only_its_own_distributors(tmp_path):
    shops = _shops_xlsx(tmp_path / "shops.xlsx")
    db = tmp_path / "warehouse.db"
    national = HISTORY + [
        _fact(store_id="T0000000003", store_name="New Mart", year=2026, month=8, volume_mt=0.40, **COASTAL)
    ]
    run_pipeline(_write_ssrs_csv(tmp_path / "all.csv", national, "15/08/2026 09:00:00"), shop_path=shops, db_path=db)
    before = _facts(db)
    assert abs(float(before.loc[before["period"] == "2026-08", "volume_mt"].sum()) - 1.90) < 1e-6

    coastal_only = [
        _fact(store_id="T0000000003", store_name="New Mart", year=2026, month=8, volume_mt=0.70, **COASTAL)
    ]
    run_pipeline(_write_ssrs_csv(tmp_path / "coastal.csv", coastal_only, "20/08/2026 09:00:00"), shop_path=shops, db_path=db)
    after = _facts(db)
    aug = after.loc[after["period"] == "2026-08"].set_index("store_id")["volume_mt"]
    assert abs(float(aug["T0000000003"]) - 0.70) < 1e-6, "the Coastal shop takes the newer cut"
    assert abs(float(aug["T0000000001"]) - 0.60) < 1e-6, "Agha Traders' August is untouched by a Coastal-only file"
    assert abs(float(aug["T0000000002"]) - 0.90) < 1e-6


def test_as_of_day_never_moves_backwards_on_an_older_cut(tmp_path):
    shops = _shops_xlsx(tmp_path / "shops.xlsx")
    db = tmp_path / "warehouse.db"
    run_pipeline(_write_ssrs_csv(tmp_path / "d20.csv", HISTORY, "20/08/2026 18:00:00"), shop_path=shops, db_path=db)
    assert int(_ledger(db).loc["2026-08", "as_of_day"]) == 20

    run_pipeline(_write_ssrs_csv(tmp_path / "d12.csv", HISTORY, "12/08/2026 18:00:00"), shop_path=shops, db_path=db)
    led = _ledger(db)
    assert led.loc["2026-08", "status"] == "mtd_open"
    assert int(led.loc["2026-08", "as_of_day"]) == 20, "an older cut must not shrink the elapsed days (run-rate would inflate)"


def test_open_month_closes_when_a_later_month_arrives(tmp_path):
    shops = _shops_xlsx(tmp_path / "shops.xlsx")
    db = tmp_path / "warehouse.db"
    run_pipeline(_write_ssrs_csv(tmp_path / "aug.csv", HISTORY, "20/08/2026 18:00:00"), shop_path=shops, db_path=db)
    assert _ledger(db).loc["2026-08", "status"] == "mtd_open"

    sept = [_fact(store_id="T0000000001", store_name="Hameed GS", year=2026, month="September", volume_mt=0.30)]
    run_pipeline(_write_ssrs_csv(tmp_path / "sep.csv", sept, "10/09/2026 09:00:00"), shop_path=shops, db_path=db)
    led = _ledger(db)
    assert led.loc["2026-09", "status"] == "mtd_open"
    assert led.loc["2026-08", "status"] == "closed"
    assert int(led.loc["2026-08", "as_of_day"]) == 31

    # Re-dropping the August cut later cannot re-open a finished month.
    run_pipeline(_write_ssrs_csv(tmp_path / "aug2.csv", HISTORY, "20/08/2026 18:00:00"), shop_path=shops, db_path=db)
    assert _ledger(db).loc["2026-08", "status"] == "closed"


def test_off_master_billed_pops_are_kept_flagged_and_national_ties_to_extract(tmp_path):
    db = tmp_path / "warehouse.db"
    universe = _write_universe(
        tmp_path / "universe.xlsx",
        [["Agha Traders", "Quetta", "Alamdar Road", "ASHRAF KHAN", "T0000000001", "Hameed GS"]],
    )
    run_pipeline(
        _write_ssrs_csv(tmp_path / "hist.csv", HISTORY, "31/08/2026 23:00:00"),
        universe_path=universe,
        db_path=db,
    )
    with connect(db) as conn:
        sm = read_sql(conn, "SELECT * FROM shop_month")
        stores = read_sql(conn, "SELECT * FROM stores").set_index("store_id")
        kpis = read_sql(conn, "SELECT * FROM kpi_snapshots")
        units = read_sql(conn, "SELECT * FROM unit_scorecards")

    aug = sm.loc[sm["period"] == "2026-08"]
    assert abs(float(aug["volume_mt"].sum()) - 1.50) < 1e-6, "off-master volume stays on file"
    flags = aug.set_index("store_id")["in_universe"].astype(int)
    assert flags["T0000000001"] == 1
    assert flags["T0000000002"] == 0
    assert int(stores.loc["T0000000002", "in_universe"]) == 0
    assert stores.loc["T0000000002", "source"] == "sales"
    # Off-master door is reported under its distributor's city, not "(unmapped)".
    assert aug.set_index("store_id").loc["T0000000002", "city"] == "Quetta"

    nat = kpis[(kpis["grain"] == "national") & (kpis["period"] == "2026-08")].iloc[0]
    assert abs(float(nat["volume_mt"]) - 1.50) < 1e-6
    assert float(nat["strike_rate"]) <= 1.0
    if not units.empty and "strike_rate" in units.columns:
        assert pd.to_numeric(units["strike_rate"], errors="coerce").fillna(0).max() <= 1.0


def test_two_cuts_of_the_same_month_in_one_drop_do_not_add():
    old = pd.DataFrame(
        [
            {"store_id": "T1", "sku": "Oil A", "period": "2026-08", "volume_mt": 0.30},
            {"store_id": "T2", "sku": "Oil A", "period": "2026-08", "volume_mt": 0.10},
        ]
    )
    new = pd.DataFrame([{"store_id": "T1", "sku": "Oil A", "period": "2026-08", "volume_mt": 0.45}])
    out = combine_sales_frames([old, new]).set_index("store_id")["volume_mt"]
    assert abs(float(out["T1"]) - 0.45) < 1e-9, "later file wins for the same shop-month"
    assert abs(float(out["T2"]) - 0.10) < 1e-9, "shops only in the earlier file are kept"


def test_execution_date_handles_month_first_culture():
    assert parse_execution_date({"execution_date": "21/08/2026", "execution_time": "19:04:50"}) == datetime(
        2026, 8, 21, 19, 4, 50
    )
    assert parse_execution_date({"execution_date": "08/21/2026", "execution_time": "19:04:50"}) == datetime(
        2026, 8, 21, 19, 4, 50
    )
    # Ambiguous: day-first reads 3 Aug, month-first reads 8 Mar. The file holds August, so it was run in August.
    assert parse_execution_date({"execution_date": "08/03/2026"}, ["2026-07", "2026-08"]) == datetime(2026, 8, 3)
    assert parse_execution_date({"execution_date": "03/08/2026"}, ["2026-07", "2026-08"]) == datetime(2026, 8, 3)
