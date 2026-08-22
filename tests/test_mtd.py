from datetime import datetime

from sndintel.mtd import banner_text, open_mtd_period, parse_execution_date, period_state, run_rate_factor
import pandas as pd


def test_parse_execution_date():
    dt = parse_execution_date({"execution_date": "20/08/2026", "execution_time": "14:03:11"})
    assert dt == datetime(2026, 8, 20, 14, 3, 11)


def test_open_mtd_is_the_execution_month_before_month_end():
    exec_dt = datetime(2026, 8, 20, 14, 3, 11)
    assert open_mtd_period(["2026-07", "2026-08"], exec_dt) == "2026-08"
    assert open_mtd_period(["2026-07"], exec_dt) is None
    month_end = datetime(2026, 8, 31, 23, 0, 0)
    assert open_mtd_period(["2026-08"], month_end) is None


def test_run_rate_factor_scales_mtd():
    factor, day, days = run_rate_factor(datetime(2026, 8, 20), "2026-08")
    assert day == 20
    assert days == 31
    assert abs(factor - 31 / 20) < 1e-9
    closed, _, _ = run_rate_factor(datetime(2026, 8, 20), "2026-07")
    assert closed == 1.0


def test_period_state_and_banner():
    ledger = pd.DataFrame(
        [
            {
                "period": "2026-07",
                "status": "closed",
                "as_of_day": None,
                "days_in_month": None,
                "execution_date": "2026-08-20",
                "source_file": "july_aug.csv",
            },
            {
                "period": "2026-08",
                "status": "mtd_open",
                "as_of_day": 20,
                "days_in_month": 31,
                "execution_date": "2026-08-20",
                "source_file": "aug.csv",
            },
        ]
    )
    aug = period_state(ledger, "2026-08")
    assert aug["open"] is True
    assert abs(aug["factor"] - 31 / 20) < 1e-9
    assert "day 20/31" in aug["label"]
    text = banner_text(ledger, "2026-08")
    assert "open MTD" in text
    assert "day 20" in text
    jul = period_state(ledger, "2026-07")
    assert jul["open"] is False
