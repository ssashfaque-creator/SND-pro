"""Isolation is billed versus this unit's own Expected — not peer fair share."""

from __future__ import annotations

import pandas as pd

from sndintel.hierarchy import build_hierarchy_pack
from sndintel.isolate import (
    apply_expected_gap,
    coverage_velocity,
    empirical_bayes,
    intra_month_fraction,
    situation_brief,
)


def test_billed_in_line_with_expected_is_on_expected():
    df = pd.DataFrame(
        {
            "grain_id": ["Karachi", "Lahore"],
            "volume_mt": [50.0, 50.0],
            "ly_mt": [100.0, 100.0],
            "expected_mt": [50.0, 50.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_expected_gap(df, k=1.0)
    assert (out["situation"] == "with_market").all()
    brief = situation_brief(
        {"volume_mt": 100, "expected_mt": 100, "ly_mt": 200, "gap_mt": 0, "label": "2026-08"},
        out,
    )
    assert "No city is behind" in brief["problem"] or "No city is behind" in brief["headline"]


def test_equal_miss_versus_expected_is_lagging_everywhere():
    """Old fair share called equal decline 'weather'. Both cities missed their typical month."""
    df = pd.DataFrame(
        {
            "grain_id": ["Karachi", "Lahore"],
            "volume_mt": [50.0, 50.0],
            "ly_mt": [100.0, 100.0],
            "expected_mt": [100.0, 100.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_expected_gap(df, k=1.0)
    assert (out["situation"] == "lagging").all()
    brief = situation_brief(
        {"volume_mt": 100, "expected_mt": 200, "ly_mt": 200, "gap_mt": -100, "label": "2026-08"},
        out,
    )
    assert "Karachi" in brief["headline"]
    assert "Lahore" in brief["headline"] or "Lahore" in brief["problem"]


def test_behind_own_expected_is_the_local_problem():
    df = pd.DataFrame(
        {
            "grain_id": ["Karachi", "Lahore"],
            "volume_mt": [20.0, 100.0],
            "ly_mt": [100.0, 100.0],
            "expected_mt": [100.0, 100.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_expected_gap(df, k=1.0)
    khi = out[out["grain_id"] == "Karachi"].iloc[0]
    lhe = out[out["grain_id"] == "Lahore"].iloc[0]
    assert khi["situation"] == "lagging"
    assert lhe["situation"] == "with_market"
    assert khi["isolated_mt"] < -20
    brief = situation_brief(
        {"volume_mt": 120, "expected_mt": 200, "ly_mt": 200, "gap_mt": -80, "label": "2026-08"},
        out,
    )
    assert "Karachi" in brief["headline"]
    assert "Lahore" not in brief["headline"] or "Ahead" in brief["problem"]


def test_grew_versus_last_year_but_missed_expected_is_lagging():
    df = pd.DataFrame(
        {
            "grain_id": ["Rawalpindi", "Multan"],
            "volume_mt": [102.0, 118.0],
            "ly_mt": [100.0, 100.0],
            "expected_mt": [118.0, 118.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_expected_gap(df, k=1.0)
    rwp = out[out["grain_id"] == "Rawalpindi"].iloc[0]
    mul = out[out["grain_id"] == "Multan"].iloc[0]
    assert rwp["volume_mt"] > rwp["ly_mt"]  # still grew versus last year
    assert rwp["situation"] == "lagging"  # but missed its own Expected
    assert mul["situation"] == "with_market"
    brief = situation_brief(
        {"volume_mt": 220, "expected_mt": 236, "ly_mt": 200, "gap_mt": -16, "label": "2026-08"},
        out,
    )
    assert "Rawalpindi" in brief["headline"]


def test_coverage_velocity_identity():
    c, v, i, *_ = coverage_velocity(n_now=8, n_ly=10, vol_now=16, vol_ly=20)
    assert abs((c + v + i) - (16 - 20)) < 1e-9
    assert c < 0  # lost doors
    assert abs(v) < 1e-9  # drop size held at 2.0


def test_empirical_bayes_shrinks_tiny_shops():
    big = empirical_bayes(-10, ly=50, k=0.05)
    tiny = empirical_bayes(-10, ly=0.02, k=0.05)
    assert abs(tiny) < 4
    assert abs(big) > 9


def test_open_mtd_uses_elapsed_days_not_a_shipped_curve():
    """Month-end totals cannot teach day 20. No handmade GT knot curve."""
    frac, src = intra_month_fraction(20, 31, None, open_mtd=True)
    assert src == "elapsed_days"
    assert abs(frac - 20 / 31) < 1e-9
    closed, src2 = intra_month_fraction(20, 31, None, open_mtd=False)
    assert closed == 1.0 and src2 == "closed"
    # A single month-end observation must not interpolate day 20 to 100%.
    month_end_only = pd.DataFrame(
        [{"period": "2025-08", "as_of_day": 31, "days_in_month": 31, "volume_mt": 100.0}]
    )
    frac3, src3 = intra_month_fraction(20, 31, month_end_only, open_mtd=True)
    assert src3 == "elapsed_days"
    assert abs(frac3 - 20 / 31) < 1e-9


def test_hierarchy_isolates_shop_inside_a_city():
    rows = []
    for sid, name, now, ly in [
        ("K1", "Kifaya Mart", 5.0, 21.0),
        ("K2", "Diamond Super", 11.0, 23.0),
        ("K3", "Kifaya KDA", 12.0, 24.0),
    ]:
        for period, vol in [("2026-08", now), ("2025-08", ly)]:
            rows.append(
                {
                    "store_id": sid,
                    "period": period,
                    "year": int(period[:4]),
                    "month": int(period[5:7]),
                    "volume_mt": vol,
                    "sku_count": 1,
                    "billed": 1,
                    "distributor": "Eva Foods",
                    "dsr_name": "Amir Surveyor",
                    "section": "Nazimabad",
                    "store_name": name,
                    "zone": "South",
                    "city": "Karachi",
                }
            )
    sm = pd.DataFrame(rows)
    stores = sm.drop_duplicates("store_id")[
        ["store_id", "store_name", "distributor", "dsr_name", "zone", "city", "section"]
    ]
    pack = build_hierarchy_pack(sm, stores, ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    shops = pack.targets[pack.targets["grain"] == "shop"]
    assert "Kifaya Mart" in set(shops["entity_name"])
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    assert float(khi["expected_mt"]) > float(khi["volume_mt"])
    assert khi["situation"] == "lagging"
