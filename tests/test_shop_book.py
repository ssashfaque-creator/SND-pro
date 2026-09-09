"""Shop-wise issues pack: reuse action Expected / Ask; MTD vs closed cuts."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from sndintel.action import ActionPack, build_action_pack
from sndintel.plan import attach_plan
from sndintel.shop_book import build_shop_book, excel_bytes, list_shop_book_entities, pdf_bytes
from sndintel.situation_report import build_situation_pack

from test_situation_report import _mtd_world, _stores, _world


def test_closed_month_lists_missed_expected_largest_hole_first():
    sm, _hier = _world()
    ledger = pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    action = build_action_pack(sm, _stores(sm.to_dict("records")), ledger=ledger, period="2026-08")
    book = build_shop_book(action=action, ledger=ledger, period="2026-08", scope="national")
    assert book.open_mtd is False
    assert "this week" not in (book.headline + book.weather).lower()
    issues = book.issues
    assert not issues.empty
    assert "Kifaya" in set(issues["Shop"].astype(str))
    assert "Quiet K" in set(issues["Shop"].astype(str))
    # Lahore Ace beat or held Expected — not an issue.
    assert "Big L" not in set(issues["Shop"].astype(str))
    gap = pd.to_numeric(issues["Gap (MT)"], errors="coerce").fillna(0)
    assert gap.is_monotonic_decreasing or len(issues) <= 1
    blob = " ".join(issues["Comment"].astype(str).tolist()).lower()
    assert "this week" not in blob
    assert "next month" in blob
    drops = pd.to_numeric(book.raw.get("expected_drop_mt"), errors="coerce").fillna(0)
    if float(drops.max()) > 0.0005:
        assert "usual drop is" in blob
        cycles = pd.to_numeric(book.raw.get("cycle_days"), errors="coerce")
        if cycles.notna().any() and float(cycles.fillna(0).max()) > 0:
            assert "every" in blob
            assert "days" in blob
    assert "Ask (KG)" not in issues.columns
    assert "Usual drop (MT)" in issues.columns
    assert "Gap (MT)" in issues.columns


def test_closed_expected_matches_action_pack():
    sm, _hier = _world()
    ledger = pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    action = build_action_pack(sm, _stores(sm.to_dict("records")), ledger=ledger, period="2026-08")
    book = build_shop_book(action=action, ledger=ledger, period="2026-08")
    left = action.raw_shops.set_index("store_id")["expected_mt"].astype(float)
    right = book.raw.set_index("store_id")["expected_mt"].astype(float)
    assert set(left.index) == set(right.index)
    pd.testing.assert_series_equal(left.sort_index(), right.sort_index(), check_names=False)


def test_city_and_dsr_scope():
    sm, _hier = _world()
    ledger = pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    action = build_action_pack(sm, _stores(sm.to_dict("records")), ledger=ledger, period="2026-08")
    karachi = build_shop_book(
        action=action, ledger=ledger, period="2026-08", scope="city", city="Karachi"
    )
    assert set(karachi.raw["city"].astype(str)) <= {"Karachi"}
    assert "Big L" not in set(karachi.shops["Shop"].astype(str))
    assert "Distributor" in karachi.shops.columns
    assert "City" not in karachi.shops.columns
    amir = build_shop_book(
        action=action,
        ledger=ledger,
        period="2026-08",
        scope="dsr",
        city="Karachi",
        distributor="Eva Foods",
        dsr="Amir",
    )
    assert set(amir.shops["Shop"].astype(str)) <= {"Kifaya"}
    cities = list_shop_book_entities(action.raw_shops, "city")
    assert "Karachi" in cities and "Lahore" in cities
    dsrs = list_shop_book_entities(action.raw_shops, "dsr")
    assert any("Amir" in x for x in dsrs)


def test_mtd_does_not_flag_still_to_expected_as_the_issue_list():
    sm, _units, ledger, _pace = _mtd_world()
    stores = _stores(sm.to_dict("records"))
    action = build_action_pack(sm, stores, ledger=ledger, period="2026-09")
    book = build_shop_book(action=action, ledger=ledger, period="2026-09")
    assert book.open_mtd is True
    assert "Ask (KG)" in book.shops.columns
    assert "Do this" in book.shops.columns
    assert "Gap (MT)" not in book.shops.columns
    remaining = pd.to_numeric(book.raw["remaining_mt"], errors="coerce").fillna(0)
    # Mid-month almost every shop still has a full-month hole. That must not be the issue list.
    behind = int((remaining > 0.05).sum())
    n_issues = 0 if book.issues.empty else len(book.issues)
    assert behind >= n_issues
    if not book.issues.empty:
        assert set(book.issues["Issue"].astype(str)) <= {
            "Due · unvisited",
            "Due · no bill",
            "Due · light drop",
            "Visited, no bill",
            "Lapsed",
            "Unvisited",
        }
    ask_left = action.raw_shops.set_index("store_id")["week_target_mt"].astype(float)
    ask_right = book.raw.set_index("store_id")["week_target_mt"].astype(float)
    pd.testing.assert_series_equal(ask_left.sort_index(), ask_right.sort_index(), check_names=False)


def test_mtd_attaches_matched_shop_target_without_replacing_expected():
    sm, _units, ledger, _pace = _mtd_world()
    stores = _stores(sm.to_dict("records"))
    action = build_action_pack(sm, stores, ledger=ledger, period="2026-09")
    targets = pd.DataFrame(
        [
            {"store_id": "K1", "store_name": "Kifaya", "target_mt": 50.0, "match_method": "id"},
            {"store_id": "L1", "store_name": "Big L", "target_mt": 55.0, "match_method": "id"},
        ]
    )
    book = build_shop_book(action=action, shop_targets=targets, ledger=ledger, period="2026-09")
    raw = book.raw.set_index("store_id")
    assert float(raw.loc["K1", "shop_target_mt"]) == 50.0
    assert float(raw.loc["K1", "expected_mt"]) != 50.0
    assert float(raw.loc["K1", "expected_mt"]) == float(action.raw_shops.set_index("store_id").loc["K1", "expected_mt"])


def _closed_pack(rows):
    shops = pd.DataFrame(rows)
    if "remaining_mt" not in shops.columns:
        shops["remaining_mt"] = (
            pd.to_numeric(shops["expected_mt"], errors="coerce").fillna(0)
            - pd.to_numeric(shops["billed_mt"], errors="coerce").fillna(0)
        ).clip(lower=0)
    return ActionPack(period="2026-08", label="Aug 2026", open_mtd=False, raw_shops=shops)


def test_unbilled_is_billed_zero_not_a_tiny_invoice():
    """0.05 MT was used as 'no bill', so Unbilled showed billed volume on the Karachi mix."""
    pack = _closed_pack(
        [
            {
                "store_id": "U1",
                "store_name": "Visited Zero",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 0.0,
                "expected_mt": 2.0,
                "call_status": "Visited · not billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
            {
                "store_id": "T1",
                "store_name": "Tiny Invoice",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 0.02,
                "expected_mt": 2.92,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
                "expected_drop_mt": 1.71,
                "cycle_days": 14,
                "last_bill_date": "2025-10-25",
            },
        ]
    )
    book = build_shop_book(action=pack, period="2026-08")
    by_name = book.raw.set_index("store_name")["issue"].astype(str)
    assert by_name["Visited Zero"] == "Unbilled"
    assert by_name["Tiny Invoice"] == "Missed Expected"
    mix = book.mix.set_index("Issue")
    assert float(mix.loc["Unbilled", "Billed (MT)"]) == 0.0
    assert float(mix.loc["Unbilled", "Expected (MT)"]) == 2.0
    assert "visited but not billed" in str(
        book.issues.loc[book.issues["Shop"] == "Visited Zero", "Comment"].iloc[0]
    ).lower()
    tiny_comment = str(book.issues.loc[book.issues["Shop"] == "Tiny Invoice", "Comment"].iloc[0]).lower()
    assert "visited but not billed" not in tiny_comment
    assert "2.92" in tiny_comment
    assert "usual drop is 1.71 mt every 14 days" in tiny_comment
    last = str(book.issues.loc[book.issues["Shop"] == "Tiny Invoice", "Last billed"].iloc[0])
    assert "2025" in last


def test_no_run_rate_is_not_a_row_with_expected():
    """0.05 MT was used as 'no Expected', so the mix showed Expected on No Expected."""
    pack = _closed_pack(
        [
            {
                "store_id": "Q1",
                "store_name": "Quiet",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 0.02,
                "expected_mt": 0.03,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
            {
                "store_id": "Z1",
                "store_name": "Universe",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 0.0,
                "expected_mt": 0.0,
                "call_status": "Unvisited",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
            {
                "store_id": "M1",
                "store_name": "Real Miss",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 1.0,
                "expected_mt": 4.0,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
        ]
    )
    book = build_shop_book(action=pack, period="2026-08")
    by_name = book.raw.set_index("store_name")["issue"].astype(str)
    assert by_name["Universe"] == "No run-rate"
    assert by_name["Quiet"] == "On Expected"
    assert by_name["Real Miss"] == "Missed Expected"
    assert "No Expected" not in set(book.mix["Issue"].astype(str))
    assert "No run-rate" not in set(book.mix["Issue"].astype(str))
    assert "On Expected" not in set(book.mix["Issue"].astype(str))
    assert book.kpis.get("n_quiet") == 1


def test_pdf_and_excel_are_real_files():
    sm, _hier = _world()
    ledger = pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    action = build_action_pack(sm, _stores(sm.to_dict("records")), ledger=ledger, period="2026-08")
    book = build_shop_book(action=action, ledger=ledger, period="2026-08")
    pdf = pdf_bytes(book)
    assert pdf[:4] == b"%PDF"
    xls = excel_bytes(book)
    assert xls[:2] == b"PK"
    wb = load_workbook(BytesIO(xls))
    assert "00 Cover" in wb.sheetnames
    assert "02 Issues" in wb.sheetnames
    assert "03 All shops" in wb.sheetnames


def _karachi_cover_units(*, volume: float, expected: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grain": "national",
                "grain_id": "ALL",
                "city": "",
                "distributor": "",
                "dsr_name": "",
                "volume_mt": volume,
                "expected_mt": expected,
                "intra_month_frac": 1.0,
            },
            {
                "grain": "city",
                "grain_id": "Karachi",
                "city": "Karachi",
                "distributor": "",
                "dsr_name": "",
                "volume_mt": volume,
                "expected_mt": expected,
                "intra_month_frac": 1.0,
            },
        ]
    )


def test_cover_gap_is_net_not_sum_of_shop_remainings():
    """Beat shops must not inflate Gap vs Expected. Cover Gap is Expected − billed, floored at 0."""
    pack = _closed_pack(
        [
            {
                "store_id": "M1",
                "store_name": "Miss",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 1.0,
                "expected_mt": 5.0,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
            {
                "store_id": "B1",
                "store_name": "Beat",
                "city": "Karachi",
                "distributor": "Dist",
                "dsr_name": "Ali",
                "billed_mt": 10.0,
                "expected_mt": 3.0,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
        ]
    )
    book = build_shop_book(action=pack, period="2026-08", scope="city", city="Karachi")
    # Shop billed 11 vs shop Expected 8 → net cover Gap 0, even though the miss still has a 4 MT hole.
    assert abs(float(book.kpis["billed_mt"]) - 11.0) < 1e-9
    assert abs(float(book.kpis["expected_mt"]) - 8.0) < 1e-9
    assert abs(float(book.kpis["gap_mt"])) < 1e-9
    assert abs(float(book.kpis["issue_mt"]) - 4.0) < 1e-9
    assert "gap 0.0 mt" in book.headline.lower()


def test_cover_target_is_plan_book_not_matched_pop_sum():
    """City Target is the full plan book (unmatched names still count), same as Situation cascade."""
    pack = _closed_pack(
        [
            {
                "store_id": "K1",
                "store_name": "Kifaya",
                "city": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "billed_mt": 1.0,
                "expected_mt": 5.0,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
            {
                "store_id": "K2",
                "store_name": "Beat K",
                "city": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "billed_mt": 10.0,
                "expected_mt": 3.0,
                "call_status": "Billed",
                "is_lapsed": False,
                "week_target_mt": 0.0,
            },
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "store_id": "K1",
                "store_name": "Kifaya",
                "city": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "target_mt": 2.0,
                "match_method": "id",
            },
            {
                "store_id": "",
                "store_name": "Unknown Kiryana",
                "city": "Karachi",
                "distributor": "Ghost Dist",
                "dsr_name": "Ghost",
                "target_mt": 8.0,
                "match_method": "unmatched",
            },
        ]
    )
    units = _karachi_cover_units(volume=20.0, expected=30.0)
    book = build_shop_book(
        action=pack,
        shop_targets=targets,
        units=units,
        period="2026-08",
        scope="city",
        city="Karachi",
    )
    assert abs(float(book.kpis["matched_target_mt"]) - 2.0) < 1e-9
    assert abs(float(book.kpis["target_mt"]) - 10.0) < 1e-9
    assert abs(float(book.kpis["billed_mt"]) - 20.0) < 1e-9
    assert abs(float(book.kpis["expected_mt"]) - 30.0) < 1e-9
    assert abs(float(book.kpis["gap_mt"]) - 10.0) < 1e-9
    assert abs(float(book.kpis["vs_target_mt"]) - 10.0) < 1e-9
    assert book.kpis.get("cover_from_scorecard") is True
    # Shop Target on the row is still the matched POP only.
    assert abs(float(book.raw.set_index("store_id").loc["K1", "shop_target_mt"]) - 2.0) < 1e-9


def test_city_cover_matches_situation_scorecard():
    sm, hier = _world()
    ledger = pd.DataFrame([{"period": "2026-08", "status": "closed"}])
    targets = pd.DataFrame(
        [
            {
                "store_id": "K1",
                "store_name": "Kifaya",
                "city": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "target_mt": 10.0,
                "match_method": "id",
            },
            {
                "store_id": "",
                "store_name": "Unknown Kiryana",
                "city": "Karachi",
                "distributor": "Ghost Dist",
                "dsr_name": "Ghost",
                "target_mt": 90.0,
                "match_method": "unmatched",
            },
        ]
    )
    units = attach_plan(hier.units, targets, pace=1.0)
    sit = build_situation_pack(units, ledger=ledger, period="2026-08", scope="city", city="Karachi")
    action = build_action_pack(sm, _stores(sm.to_dict("records")), ledger=ledger, period="2026-08")
    book = build_shop_book(
        action=action,
        shop_targets=targets,
        units=units,
        ledger=ledger,
        period="2026-08",
        scope="city",
        city="Karachi",
    )
    sit_expected = float(sit.kpis.get("expected_full_mt") or sit.kpis.get("expected_mt") or 0)
    assert abs(float(book.kpis["target_mt"]) - 100.0) < 1e-6
    assert abs(float(book.kpis["target_mt"]) - float(sit.kpis["target_mt"])) < 1e-6
    assert abs(float(book.kpis["matched_target_mt"]) - 10.0) < 1e-6
    assert abs(float(book.kpis["billed_mt"]) - float(sit.kpis["billed_mt"])) < 0.6
    assert abs(float(book.kpis["expected_mt"]) - sit_expected) < 0.6
    assert abs(float(book.kpis["gap_mt"]) - float(sit.kpis["gap_mt"])) < 0.6
    assert abs(float(book.kpis["gap_mt"]) - max(0.0, float(book.kpis["expected_mt"]) - float(book.kpis["billed_mt"]))) < 1e-6
    assert float(book.kpis["target_mt"]) > float(book.kpis["matched_target_mt"]) + 1.0
