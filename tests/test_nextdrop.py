"""Next-order model: fat last drop → smaller next; thin / late → catch-up."""

from datetime import date, timedelta

import pandas as pd

from sndintel.nextdrop import MIN_TRAIN_ROWS, attach_next_drop


def _bill(store_id: str, day: date, vol: float, city: str = "Karachi", dsr: str = "Amir") -> dict:
    return {
        "store_id": store_id,
        "store_name": store_id,
        "city": city,
        "dsr_name": dsr,
        "sale_date": day.isoformat(),
        "period": f"{day.year:04d}-{day.month:02d}",
        "volume_mt": vol,
    }


def _walk(store_id: str, last: date, every: int, vols: list[float]) -> list[dict]:
    rows = []
    cur = last
    for vol in reversed(vols):
        rows.append(_bill(store_id, cur, vol))
        cur = cur - timedelta(days=every)
    return list(reversed(rows))


def _universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Enough billed-day transitions for the tree to learn both directions."""
    rows: list[dict] = []
    as_of = date(2026, 8, 22)
    # Stable 1.0 every 15 days — many shops so the model sees a baseline.
    for i in range(24):
        vols = [1.0] * 16
        last = as_of - timedelta(days=10)
        rows.extend(_walk(f"S{i:02d}", last, 15, vols))
    # After a fat 2.5 drop the next bill is small (0.45).
    for i in range(16):
        vols = [1.0] * 10 + [2.5, 0.45]
        last = as_of - timedelta(days=8)
        rows.extend(_walk(f"F{i:02d}", last, 15, vols))
    # After a stub 0.2 and a long gap the next bill catches up (1.6).
    for i in range(16):
        vols = [1.0] * 10 + [0.2, 1.6]
        last = as_of - timedelta(days=6)
        rows.extend(_walk(f"T{i:02d}", last, 18, vols))
    daily = pd.DataFrame(rows)
    # Score shops sitting in those two states, plus a stable one.
    fat_hist = _walk("FAT1", as_of - timedelta(days=8), 15, [1.0] * 10 + [2.5])
    thin_hist = _walk("THIN1", as_of - timedelta(days=22), 15, [1.0] * 10 + [0.2])
    stable_hist = _walk("STAB1", as_of - timedelta(days=10), 15, [1.0] * 12)
    daily = pd.concat([daily, pd.DataFrame(fat_hist + thin_hist + stable_hist)], ignore_index=True)
    shops = pd.DataFrame(
        [
            {
                "store_id": "FAT1",
                "city": "Karachi",
                "dsr_name": "Amir",
                "ams_3m": 2.0,
                "billed_mt": 2.5,
                "last_month_mt": 2.0,
                "typical_drop_mt": 1.0,
                "last_drop_mt": 2.5,
            },
            {
                "store_id": "THIN1",
                "city": "Karachi",
                "dsr_name": "Amir",
                "ams_3m": 2.0,
                "billed_mt": 0.2,
                "last_month_mt": 1.0,
                "typical_drop_mt": 1.0,
                "last_drop_mt": 0.2,
            },
            {
                "store_id": "STAB1",
                "city": "Karachi",
                "dsr_name": "Amir",
                "ams_3m": 2.0,
                "billed_mt": 1.0,
                "last_month_mt": 2.0,
                "typical_drop_mt": 1.0,
                "last_drop_mt": 1.0,
            },
        ]
    )
    return shops, daily, pd.Timestamp(as_of)


def test_next_drop_goes_both_ways():
    shops, daily, as_of = _universe()
    assert len(daily) > MIN_TRAIN_ROWS
    out = attach_next_drop(shops, daily, as_of, shop_month=None).set_index("store_id")
    assert out.loc["FAT1", "next_drop_model"] == "xgboost"
    fat = float(out.loc["FAT1", "next_drop_mt"])
    thin = float(out.loc["THIN1", "next_drop_mt"])
    stab = float(out.loc["STAB1", "next_drop_mt"])
    assert fat < 2.5
    assert fat < stab
    assert thin > 0.2
    assert thin > fat
    assert 0.6 < stab < 1.6


def test_thin_history_keeps_median():
    as_of = pd.Timestamp("2026-08-22")
    daily = pd.DataFrame([_bill("X", date(2026, 8, 1), 0.4)])
    shops = pd.DataFrame(
        [{"store_id": "X", "city": "", "dsr_name": "", "ams_3m": 0.4, "billed_mt": 0.4, "typical_drop_mt": 0.4, "last_drop_mt": 0.4}]
    )
    out = attach_next_drop(shops, daily, as_of)
    assert float(out.iloc[0]["next_drop_mt"]) == 0.4
    assert out.iloc[0]["next_drop_model"] == "median"
