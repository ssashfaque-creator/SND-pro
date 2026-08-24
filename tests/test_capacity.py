"""Capacity labels, whale doors, YoY gate, and per-DSR caps."""

import pandas as pd

from sndintel.capacity import (
    LABEL_NOT_CONVERTING,
    LABEL_NOT_WORKING,
    LABEL_OVERLOADED,
    cap_shops_per_dsr,
    score_dsr_capacity,
    visit_quality_warnings,
    whale_shops,
    yoy_is_printable,
)
from sndintel.coverage import attach_remarks
from sndintel.identity import dsr_unit_id


def test_yoy_is_printable_needs_half_a_ton():
    assert yoy_is_printable(0.5)
    assert yoy_is_printable(12)
    assert not yoy_is_printable(0.49)
    assert not yoy_is_printable(None)


def test_tiny_ly_does_not_print_yoy_in_remarks():
    df = pd.DataFrame(
        [
            {
                "grain_id": "Tiny",
                "volume_mt": 0.2,
                "ly_mt": 0.1,
                "expected_mt": 0.3,
                "ams_3m": 0.3,
                "vs_ams_mt": -0.1,
                "universe": 10,
                "visited": 8,
                "visit_rate": 0.8,
                "has_visit_file": 1,
                "billed": 2,
                "strike_rate": 0.2,
                "productivity": 0.1,
                "from_drop_size_mt": 0.05,
                "expected_drop_size_mt": 0.15,
            }
        ]
    )
    out = attach_remarks(df, {})
    assert "YoY" not in str(out.iloc[0]["remarks"])


def test_overloaded_vs_not_working_vs_not_converting():
    overloaded = _shops(n=80, visited=80, billed=60, city="Karachi", dist="A", dsr="Big")
    skip = _shops(n=20, visited=4, billed=2, city="Islamabad", dist="B", dsr="Skip")
    convert = _shops(n=20, visited=20, billed=6, city="Lahore", dist="C", dsr="Talk")
    cap = score_dsr_capacity(pd.concat([overloaded, skip, convert], ignore_index=True), as_of_day=1, days_in_month=31)
    labels = {row.dsr_name: row.label for row in cap.itertuples(index=False)}
    assert labels["Big"] == LABEL_OVERLOADED
    assert labels["Skip"] == LABEL_NOT_WORKING
    assert labels["Talk"] == LABEL_NOT_CONVERTING


def test_two_shahids_get_separate_capacity_rows():
    a = _shops(n=10, visited=10, billed=8, city="Karachi", dist="Eva", dsr="Shahid")
    b = _shops(n=10, visited=2, billed=1, city="Lahore", dist="Other", dsr="Shahid")
    cap = score_dsr_capacity(pd.concat([a, b], ignore_index=True), as_of_day=21, days_in_month=31)
    assert len(cap) == 2
    assert cap["grain_id"].nunique() == 2
    ids = set(cap["grain_id"])
    assert dsr_unit_id("Karachi", "Eva", "Shahid") in ids
    assert dsr_unit_id("Lahore", "Other", "Shahid") in ids


def test_whale_shops_keep_ton_doors():
    shops = pd.DataFrame(
        [
            {"store_id": "W", "store_name": "Whale", "ams_3m": 2.0, "expected_mt": 2.0, "remaining_mt": 1.0, "week_target_mt": 1.0},
            {"store_id": "K", "store_name": "Kiryana", "ams_3m": 0.05, "expected_mt": 0.05, "remaining_mt": 0.04, "week_target_mt": 0.04},
        ]
    )
    whales = whale_shops(shops)
    assert list(whales["store_id"]) == ["W"]


def test_cap_shops_per_dsr_keeps_top_ask():
    rows = []
    for i in range(20):
        rows.append(
            {
                "store_id": f"S{i}",
                "dsr_name": "Amir",
                "city": "Karachi",
                "distributor": "Eva",
                "week_target_mt": 1.0 - i * 0.01,
                "value_score": 20 - i,
            }
        )
    kept = cap_shops_per_dsr(pd.DataFrame(rows), per_dsr=12)
    assert len(kept) == 12
    assert kept["store_id"].tolist() == [f"S{i}" for i in range(12)]


def test_visit_quality_warns_on_100_percent_huge_universe():
    units = pd.DataFrame(
        [
            {
                "grain": "city",
                "grain_id": "Karachi",
                "visit_rate": 1.0,
                "universe": 800,
                "has_visit_file": 1,
            }
        ]
    )
    visits = pd.DataFrame([{"store_id": "X", "period": "2026-08", "visits": 1}])
    warnings = visit_quality_warnings(units, visits, "2026-08")
    assert any("Karachi" in w for w in warnings)


def _shops(n: int, visited: int, billed: int, city: str, dist: str, dsr: str) -> pd.DataFrame:
    rows = []
    for i in range(n):
        call = "Visited · billed" if i < visited else "Unvisited"
        billed_mt = 0.1 if i < billed else 0.0
        rows.append(
            {
                "store_id": f"{dsr}-{city}-{i}",
                "city": city,
                "distributor": dist,
                "dsr_name": dsr,
                "call_status": call,
                "billed_mt": billed_mt,
                "expected_mt": 0.2,
                "remaining_mt": max(0.0, 0.2 - billed_mt),
                "week_target_mt": 0.1,
                "cycle_days": 15,
                "typical_drop_mt": 0.2,
            }
        )
    return pd.DataFrame(rows)
