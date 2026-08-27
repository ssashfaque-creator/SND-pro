"""Demand-driven Ask: 90-day API, cold start, 3× lapse, 4-pillar identity."""

from datetime import date, timedelta

import pandas as pd

from sndintel.action import (
    ACTION_CALL,
    ACTION_CONVERT,
    ACTION_HOLD,
    ACTION_LIFT,
    ACTION_RECOVER,
    build_action_pack,
)
from sndintel.demand import (
    DEFAULT_API_DAYS,
    DUE_RATIO,
    LAPSE_MULTIPLIER,
    attach_demand_cycles,
    attach_pipeline,
    classify_demand_actions,
    pipeline_identity_holds,
)
from sndintel.io_utils import shift_period


def _period_range(end: str, n: int) -> list[str]:
    out = [end]
    cur = end
    for _ in range(n - 1):
        cur = shift_period(cur, -1)
        out.append(cur)
    return list(reversed(out))


def test_cold_start_zero_history_ask_is_zero():
    as_of = pd.Timestamp("2026-08-22")
    shops = pd.DataFrame(
        [
            {
                "store_id": "NEW1",
                "store_name": "New Mart",
                "city": "Karachi",
                "distributor": "Eva",
                "dsr_name": "Amir",
                "ams_3m": 0.0,
                "expected_mt": 0.0,
                "billed_mt": 0.0,
                "last_billed_period": "",
                "last_billed_mt": 0.0,
                "last_month_mt": 0.0,
            }
        ]
    )
    shops = attach_demand_cycles(shops, pd.DataFrame(), as_of)
    shops["call_status"] = "Unvisited"
    shops = classify_demand_actions(shops, True)
    shops = attach_pipeline(shops, days_left=9, open_mtd=True)
    assert float(shops.loc[0, "expected_drop_mt"]) == 0.0
    assert float(shops.loc[0, "week_target_mt"]) == 0.0
    assert shops.loc[0, "action"] == ACTION_CALL
    assert shops.loc[0, "recommended_action"] == "New door — no history"


def test_cold_start_one_purchase_uses_default_api():
    as_of = pd.Timestamp("2026-08-22")
    daily = pd.DataFrame(
        [
            {
                "store_id": "ONE1",
                "sale_date": date(2026, 8, 8),
                "volume_mt": 0.8,
            }
        ]
    )
    shops = pd.DataFrame(
        [
            {
                "store_id": "ONE1",
                "ams_3m": 0.8,
                "expected_mt": 0.8,
                "billed_mt": 0.8,
                "last_billed_period": "2026-08",
                "last_billed_mt": 0.8,
                "last_month_mt": 0.8,
            }
        ]
    )
    shops = attach_demand_cycles(shops, daily, as_of)
    assert float(shops.loc[0, "api_days"]) == DEFAULT_API_DAYS
    assert abs(float(shops.loc[0, "expected_drop_mt"]) - 0.8) < 1e-9
    assert bool(shops.loc[0, "is_cold_start"])
    # DSLP = 14, API = 14, ratio = 1.0 → due, but billed this month so another visit
    shops["call_status"] = "Billed"
    shops = classify_demand_actions(shops, True)
    shops = attach_pipeline(shops, days_left=9, open_mtd=True)
    assert float(shops.loc[0, "depletion_ratio"]) == DUE_RATIO + (14 / 14 - DUE_RATIO)
    assert shops.loc[0, "action"] == ACTION_LIFT
    assert abs(float(shops.loc[0, "week_target_mt"]) - 0.8) < 1e-9


def test_ratio_below_due_ask_is_zero_and_not_yet_due_fills_pipeline():
    as_of = pd.Timestamp("2026-08-22")
    days = [date(2026, 8, 12) - timedelta(days=15 * i) for i in range(8)]
    daily = pd.DataFrame(
        [{"store_id": "MID1", "sale_date": d, "volume_mt": 1.0} for d in days]
    )
    shops = pd.DataFrame(
        [
            {
                "store_id": "MID1",
                "ams_3m": 2.0,
                "expected_mt": 2.0,
                "billed_mt": 1.0,
                "last_billed_period": "2026-08",
                "last_billed_mt": 1.0,
                "last_month_mt": 1.0,
                "call_status": "Billed",
            }
        ]
    )
    shops = attach_demand_cycles(shops, daily, as_of)
    shops = classify_demand_actions(shops, True)
    shops = attach_pipeline(shops, days_left=9, open_mtd=True)
    assert float(shops.loc[0, "depletion_ratio"]) < DUE_RATIO
    assert shops.loc[0, "action"] == ACTION_HOLD
    assert float(shops.loc[0, "week_target_mt"]) == 0.0
    assert bool(shops.loc[0, "coming_due"])
    assert abs(float(shops.loc[0, "not_yet_due_mt"]) - float(shops.loc[0, "expected_drop_mt"])) < 1e-9
    assert pipeline_identity_holds(shops)


def test_lapse_beyond_three_api_zeros_ask():
    as_of = pd.Timestamp("2026-08-22")
    last = date(2026, 6, 20)
    days = [last - timedelta(days=15 * i) for i in range(10)]
    daily = pd.DataFrame([{"store_id": "DEAD1", "sale_date": d, "volume_mt": 1.0} for d in days])
    shops = pd.DataFrame(
        [
            {
                "store_id": "DEAD1",
                "ams_3m": 2.0,
                "expected_mt": 2.0,
                "billed_mt": 0.0,
                "last_billed_period": "2026-06",
                "last_billed_mt": 1.0,
                "last_month_mt": 0.0,
                "call_status": "Unvisited",
            }
        ]
    )
    shops = attach_demand_cycles(shops, daily, as_of)
    shops = classify_demand_actions(shops, True)
    shops = attach_pipeline(shops, days_left=9, open_mtd=True)
    api = float(shops.loc[0, "api_days"])
    dslp = float(shops.loc[0, "days_since_bill"])
    assert dslp > LAPSE_MULTIPLIER * api
    assert bool(shops.loc[0, "is_lapsed"])
    assert shops.loc[0, "action"] == ACTION_RECOVER
    assert float(shops.loc[0, "week_target_mt"]) == 0.0
    assert shops.loc[0, "recommended_action"] == "Lapsed — lost door"


def test_due_unvisited_ask_equals_expected_drop():
    as_of = pd.Timestamp("2026-08-22")
    last = date(2026, 7, 28)
    days = [last - timedelta(days=15 * i) for i in range(10)]
    daily = pd.DataFrame([{"store_id": "DUE1", "sale_date": d, "volume_mt": 1.2} for d in days])
    shops = pd.DataFrame(
        [
            {
                "store_id": "DUE1",
                "ams_3m": 2.4,
                "expected_mt": 8.0,
                "billed_mt": 0.0,
                "last_billed_period": "2026-07",
                "last_billed_mt": 1.2,
                "last_month_mt": 2.4,
                "call_status": "Unvisited",
            }
        ]
    )
    shops = attach_demand_cycles(shops, daily, as_of)
    shops = classify_demand_actions(shops, True)
    shops = attach_pipeline(shops, days_left=9, open_mtd=True)
    drop = float(shops.loc[0, "expected_drop_mt"])
    assert shops.loc[0, "action"] == ACTION_CALL
    assert abs(float(shops.loc[0, "week_target_mt"]) - drop) < 1e-9
    assert abs(float(shops.loc[0, "due_unvisited_mt"]) - drop) < 1e-9
    # Ask is the drop, not remaining-to-Expected (8.0).
    assert float(shops.loc[0, "week_target_mt"]) < 7.0
    assert pipeline_identity_holds(shops)


def test_four_pillar_identity_on_mixed_panel():
    from tests.test_action import _panel

    shop_month, stores, shop_day, ledger, visits = _panel()
    pack = build_action_pack(shop_month, stores, shop_day, visits=visits, ledger=ledger, period="2026-08")
    raw = pack.raw_shops
    assert pipeline_identity_holds(raw)
    billed = float(raw["billed_mt"].sum())
    due_u = float(raw["due_unvisited_mt"].sum())
    var = float(raw["drop_variance_mt"].sum())
    nyd = float(raw["not_yet_due_mt"].sum())
    pipe = float(raw["pipeline_expected_mt"].sum())
    assert abs(billed + due_u + var + nyd - pipe) < 1e-6
    lapsed = raw[raw["action"] == ACTION_RECOVER]
    if not lapsed.empty:
        assert (pd.to_numeric(lapsed["week_target_mt"], errors="coerce").fillna(0) == 0).all()
    assert not pack.pipeline.empty
    assert list(pack.pipeline.columns)[:3] == ["City", "Pipeline expected (MT)", "Billed (MT)"]
    assert "Unvisited due Ask (MT)" in pack.sales_head.columns
    assert "Target drop (KG)" in pack.beat.columns
    assert "Recommended action" in pack.beat.columns
