"""This-week action engine: delivery curve, call lists, backtest."""

from calendar import monthrange
from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from sndintel.action import (
    ACTION_CALL,
    ACTION_HOLD,
    SHOP_FLOOR_MT,
    build_action_pack,
    fit_delivery_curves,
)
from sndintel.action_report import excel_bytes, iter_action_sheets
from sndintel.io_utils import period_key, shift_period


def _period_range(end: str, n: int) -> list[str]:
    out = [end]
    cur = end
    for _ in range(n - 1):
        cur = shift_period(cur, -1)
        out.append(cur)
    return list(reversed(out))


def _daily_rows(store_id, name, city, dist, dsr, periods, pattern, current_as_of=None, current_frac=None):
    """pattern: 'front' bills 80% by day 10; 'back' bills 80% after day 20."""
    rows = []
    month_rows = []
    for period in periods:
        year, month = int(period[:4]), int(period[5:7])
        days = monthrange(year, month)[1]
        last = days
        if current_as_of and period == periods[-1]:
            last = current_as_of
        total = 2.0
        if current_frac is not None and period == periods[-1]:
            billed = total * current_frac
        else:
            billed = total
        if pattern == "front":
            day_a, day_b = 4, 9
            split = (0.8, 0.2) if last >= 9 else (1.0, 0.0)
        else:
            day_a, day_b = 22, 27
            if last < 20:
                day_a, day_b = 8, 14
                split = (0.5, 0.5)
            else:
                split = (0.2, 0.8)
        if current_frac is not None and period == periods[-1] and last < 20 and pattern == "back":
            day_a, day_b = 8, 14
            split = (0.5, 0.5)
        vols = [billed * split[0], billed * split[1]]
        for day, vol in ((day_a, vols[0]), (day_b, vols[1])):
            if day > last:
                continue
            rows.append(
                {
                    "store_id": store_id,
                    "store_name": name,
                    "city": city,
                    "distributor": dist,
                    "dsr_name": dsr,
                    "section": city,
                    "sale_date": f"{year:04d}-{month:02d}-{day:02d}",
                    "year": year,
                    "month": month,
                    "day": day,
                    "period": period,
                    "volume_mt": vol,
                }
            )
        month_rows.append(
            {
                "store_id": store_id,
                "store_name": name,
                "city": city,
                "distributor": dist,
                "dsr_name": dsr,
                "section": city,
                "period": period,
                "year": year,
                "month": month,
                "volume_mt": billed if period != periods[-1] or current_frac is None else billed,
                "sku_count": 1,
                "billed": 1,
                "zone": None,
            }
        )
    return rows, month_rows


def _panel(as_of=15, front_frac=0.20, back_frac=0.20):
    periods = _period_range("2026-08", 18)
    hist = periods[:-1]
    stores = pd.DataFrame(
        [
            {"store_id": "FRONT1", "store_name": "Front Mart", "city": "Karachi", "distributor": "Eva Foods", "dsr_name": "Amir", "section": "Clifton", "in_universe": 1},
            {"store_id": "BACK1", "store_name": "Back Mart", "city": "Karachi", "distributor": "Eva Foods", "dsr_name": "Amir", "section": "Clifton", "in_universe": 1},
            {"store_id": "MISS1", "store_name": "Miss Mart", "city": "Karachi", "distributor": "Eva Foods", "dsr_name": "Amir", "section": "Clifton", "in_universe": 1},
            {"store_id": "GHOST", "store_name": "Ghost", "city": "Lahore", "distributor": "Ghost Dist", "dsr_name": "Ghost DSR", "section": "X", "in_universe": 1},
        ]
    )
    day_rows, month_rows = [], []
    a, b = _daily_rows("FRONT1", "Front Mart", "Karachi", "Eva Foods", "Amir", hist + ["2026-08"], "front", as_of, front_frac)
    c, d = _daily_rows("BACK1", "Back Mart", "Karachi", "Eva Foods", "Amir", hist + ["2026-08"], "back", as_of, back_frac)
    e, f = _daily_rows("MISS1", "Miss Mart", "Karachi", "Eva Foods", "Amir", hist + ["2026-08"], "front", as_of, 0.0)
    day_rows.extend(a + c + e)
    month_rows.extend(b + d + f)
    shop_day = pd.DataFrame(day_rows)
    shop_month = pd.DataFrame(month_rows)
    ledger = pd.DataFrame(
        [
            {"period": p, "status": "closed" if p != "2026-08" else "mtd_open", "as_of_day": 31 if p != "2026-08" else as_of, "days_in_month": monthrange(int(p[:4]), int(p[5:7]))[1]}
            for p in periods
        ]
    )
    return shop_month, stores, shop_day, ledger


def test_front_loaded_shop_is_called_back_loaded_is_held():
    """Same MTD share: the early buyer is late; the late buyer is on its own curve."""
    shop_month, stores, shop_day, ledger = _panel(as_of=15, front_frac=0.20, back_frac=0.20)
    pack = build_action_pack(shop_month, stores, shop_day, ledger=ledger, period="2026-08")
    raw = pack.raw_shops.set_index("store_id")
    assert "FRONT1" in raw.index
    assert "BACK1" in raw.index
    assert "MISS1" in raw.index
    assert "GHOST" not in raw.index
    assert raw.loc["FRONT1", "action"] == "Lift drop"
    assert raw.loc["BACK1", "action"] == ACTION_HOLD
    assert raw.loc["MISS1", "action"] == ACTION_CALL
    assert float(raw.loc["FRONT1", "behind_pace_mt"]) > float(raw.loc["BACK1", "behind_pace_mt"])
    assert float(raw.loc["FRONT1", "week_target_mt"]) >= SHOP_FLOOR_MT
    assert "Front Mart" in str(raw.loc["FRONT1", "instruction"])
    assert "Call Miss Mart" in str(raw.loc["MISS1", "instruction"])
    assert pack.source == "daily_curve"
    assert not pack.calls.empty
    assert "Miss Mart" in set(pack.calls["Shop"].astype(str))
    assert "Front Mart" in set(pack.lifts["Shop"].astype(str))
    assert "Back Mart" not in set(pack.calls["Shop"].astype(str))


def test_curve_is_below_calendar_for_back_loaded_shop():
    shop_month, stores, shop_day, ledger = _panel()
    hist = shop_day[shop_day["period"] != "2026-08"]
    shops = pd.DataFrame(
        [{"store_id": "BACK1", "city": "Karachi", "dsr_name": "Amir", "expected_mt": 2.0, "billed_mt": 0.4}]
    )
    curves = fit_delivery_curves(hist, shops)
    shop_curve = curves["shop::BACK1"]
    calendar = 15 / 31
    assert float(shop_curve[15]) < calendar - 0.10
    front = pd.DataFrame(
        [{"store_id": "FRONT1", "city": "Karachi", "dsr_name": "Amir", "expected_mt": 2.0, "billed_mt": 0.4}]
    )
    curves_f = fit_delivery_curves(hist, front)
    assert float(curves_f["shop::FRONT1"][15]) > calendar + 0.10


def test_dsr_instruction_names_counts_and_tonnes():
    shop_month, stores, shop_day, ledger = _panel()
    pack = build_action_pack(shop_month, stores, shop_day, ledger=ledger, period="2026-08")
    assert not pack.dsrs.empty
    row = pack.dsrs.iloc[0]
    assert row["DSR"] == "Amir"
    assert "Push Amir" in str(row["Do this"])
    assert "this week" in str(row["Do this"]).lower()


def test_visited_unbilled_is_convert():
    shop_month, stores, shop_day, ledger = _panel(front_frac=0.0, back_frac=0.20)
    # Front shop billed 0 this month — still in daily history for prior months.
    shop_month.loc[(shop_month["store_id"] == "FRONT1") & (shop_month["period"] == "2026-08"), "volume_mt"] = 0.0
    shop_month.loc[(shop_month["store_id"] == "FRONT1") & (shop_month["period"] == "2026-08"), "billed"] = 0
    shop_day = shop_day[~((shop_day["store_id"] == "FRONT1") & (shop_day["period"] == "2026-08"))]
    visits = pd.DataFrame([{"store_id": "FRONT1", "period": "2026-08", "visits": 2}])
    pack = build_action_pack(shop_month, stores, shop_day, visits=visits, ledger=ledger, period="2026-08")
    raw = pack.raw_shops.set_index("store_id")
    assert raw.loc["FRONT1", "action"] == "Convert"
    assert "Front Mart" in set(pack.converts["Shop"].astype(str))


def test_action_workbook_has_one_table_per_grain():
    shop_month, stores, shop_day, ledger = _panel()
    pack = build_action_pack(shop_month, stores, shop_day, ledger=ledger, period="2026-08")
    names = [s[0] for s in iter_action_sheets(pack)]
    assert names == ["01 Country", "02 Distributors", "03 DSRs", "04 Call", "05 Convert", "06 Lift drop", "07 Backtest"]
    raw = excel_bytes(pack)
    wb = load_workbook(BytesIO(raw))
    assert "00 Cover" in wb.sheetnames
    assert "04 Call" in wb.sheetnames
    assert "02 Distributors" in wb.sheetnames
    detailed = excel_bytes(pack, detailed=True)
    wb_d = load_workbook(BytesIO(detailed))
    assert "04 Shops" in wb_d.sheetnames
    assert pack.backtest is not None
    if not pack.backtest.empty and "Curve precision@50" in pack.backtest.columns:
        curve = pack.backtest.iloc[0]["Curve precision@50"]
        cal = pack.backtest.iloc[0]["Calendar precision@50"]
        assert curve is None or cal is None or float(curve) >= float(cal) - 5


def test_period_key_stable():
    assert period_key(2026, 8) == "2026-08"
