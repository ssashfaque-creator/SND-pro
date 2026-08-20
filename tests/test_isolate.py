"""Parent-adjusted isolation is the difference vs Excel YoY."""

from __future__ import annotations

import pandas as pd

from sndintel.hierarchy import build_hierarchy_pack
from sndintel.isolate import (
    apply_shift_share,
    coverage_velocity,
    empirical_bayes,
    intra_month_fraction,
    situation_brief,
)


def test_equal_decline_is_weather_not_a_city_fire():
    """National −50% and every city −50% → residuals ~0, no lagging city."""
    df = pd.DataFrame(
        {
            "grain_id": ["Karachi", "Lahore"],
            "volume_mt": [50.0, 50.0],
            "ly_mt": [100.0, 100.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_shift_share(df, parent_now=100, parent_ly=200, k=1.0)
    assert abs(float(out["competitive_mt"].sum())) < 1e-6
    assert (out["situation"] == "with_market").all()
    brief = situation_brief(
        {"volume_mt": 100, "expected_mt": 200, "ly_mt": 200, "gap_mt": -100, "label": "2026-08"},
        out,
    )
    assert "No city is a statistical exception" in brief["problem"] or "No city is an exception" in brief["headline"]


def test_extra_decline_is_the_local_problem():
    """National −50%. Karachi −80%, Lahore −20% → only Karachi is lagging."""
    df = pd.DataFrame(
        {
            "grain_id": ["Karachi", "Lahore"],
            "volume_mt": [20.0, 80.0],
            "ly_mt": [100.0, 100.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_shift_share(df, parent_now=100, parent_ly=200, k=1.0)
    assert abs(float(out["competitive_mt"].sum())) < 1e-6
    khi = out[out["grain_id"] == "Karachi"].iloc[0]
    lhe = out[out["grain_id"] == "Lahore"].iloc[0]
    assert khi["situation"] == "lagging"
    assert lhe["situation"] == "outperforming"
    assert khi["isolated_mt"] < -20
    brief = situation_brief(
        {"volume_mt": 100, "expected_mt": 200, "ly_mt": 200, "gap_mt": -100, "label": "2026-08"},
        out,
    )
    assert "Karachi" in brief["headline"]
    assert "Lahore" in brief["problem"] or "beating" in brief["problem"].lower() or "Holding" in brief["problem"]


def test_slower_growth_is_the_problem_when_national_is_up():
    df = pd.DataFrame(
        {
            "grain_id": ["Rawalpindi", "Multan"],
            "volume_mt": [102.0, 118.0],
            "ly_mt": [100.0, 100.0],
            "billed": [10, 10],
            "billed_ly": [10, 10],
            "lost_mt": [0, 0],
            "universe": [20, 20],
        }
    )
    out = apply_shift_share(df, parent_now=220, parent_ly=200, k=1.0)
    rwp = out[out["grain_id"] == "Rawalpindi"].iloc[0]
    assert rwp["volume_mt"] > rwp["ly_mt"]  # still grew
    assert rwp["situation"] == "lagging"  # but slower than national +10%
    brief = situation_brief(
        {"volume_mt": 220, "expected_mt": 200, "ly_mt": 200, "gap_mt": 20, "label": "2026-08"},
        out,
    )
    assert "growing" in brief["headline"]
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
    # Single city: city residual is ~0 (it IS the nation). Isolation still names the door.
    khi = pack.units[(pack.units["grain"] == "city") & (pack.units["grain_id"] == "Karachi")].iloc[0]
    assert abs(float(khi["competitive_mt"])) < 0.05
