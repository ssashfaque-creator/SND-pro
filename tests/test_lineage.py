"""POP lineage: retired and superseded codes fold into the live code that continues the door."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sndintel.action import ACTION_SUPERSEDED, build_action_pack
from sndintel.io_utils import looks_like_total, window_periods
from sndintel.lineage import (
    METHOD_RETIRED,
    METHOD_SUPERSEDED,
    apply_lineage,
    build_lineage,
    continues_map,
    lineage_map,
)
from sndintel.season import fit_seasonality, fit_shop_expected
from sndintel.shop_book import ISSUE_SUPERSEDED, MIX_TOTAL, build_shop_book

from test_daily import _write_daily

OLD = "T0000508100100015724"
NEW = "T0000508100100015902"  # same 12-char route as OLD
FAR = "T0000509300400099001"  # other route, other DSR
OTHER = "T0000508100100015777"


def _stores(rows):
    return pd.DataFrame(rows)


def _months(store_id, name, vols: dict[str, float], dist="Rubina Shaheen (LHR)", dsr="Umair", city="Lahore"):
    return [
        {
            "store_id": store_id,
            "store_name": name,
            "period": p,
            "volume_mt": v,
            "distributor": dist,
            "dsr_name": dsr,
            "city": city,
        }
        for p, v in vols.items()
    ]


def _days(store_id, days: dict[str, float]):
    return [{"store_id": store_id, "sale_date": d, "volume_mt": v} for d, v in days.items()]


def test_retired_code_off_universe_merges_into_same_name_live_code():
    stores = _stores(
        [
            {"store_id": NEW, "store_name": "Khan Store", "distributor": "Rubina Shaheen (LHR)", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
            {"store_id": OLD, "store_name": "Khan Store", "distributor": "Rubina Shaheen (LHR)", "dsr_name": "Umair", "city": "Lahore", "in_universe": 0},
        ]
    )
    sm = pd.DataFrame(
        _months(OLD, "Khan Store", {"2026-05": 0.4, "2026-06": 0.4})
        + _months(NEW, "Khan Store", {"2026-07": 0.4, "2026-08": 0.4})
    )
    sd = pd.DataFrame(_days(OLD, {"2026-06-20": 0.4}) + _days(NEW, {"2026-07-03": 0.4}))
    pairs, left = build_lineage(sm, stores, sd)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["old_id"], row["new_id"], row["method"]) == (OLD, NEW, METHOD_RETIRED)
    assert row["confidence"] == "high"
    assert left.empty

    merged = apply_lineage(sm, pairs)
    assert OLD not in set(merged["store_id"])
    assert abs(float(merged.loc[merged["store_id"] == NEW, "volume_mt"].sum()) - 1.6) < 1e-9
    # Idempotent: re-applying changes nothing.
    again = apply_lineage(merged, pairs)
    pd.testing.assert_frame_equal(again.reset_index(drop=True), merged.reset_index(drop=True))
    assert lineage_map(pairs) == {OLD: NEW}
    assert continues_map(pairs) == {NEW: [OLD]}


def test_retired_code_with_ambiguous_same_name_codes_stays_unresolved():
    stores = _stores(
        [
            {"store_id": NEW, "store_name": "Madina Store", "distributor": "D1", "dsr_name": "A", "city": "Lahore", "in_universe": 1},
            {"store_id": OTHER, "store_name": "Madina Store", "distributor": "D1", "dsr_name": "A", "city": "Lahore", "in_universe": 1},
            {"store_id": OLD, "store_name": "Madina Store", "distributor": "D1", "dsr_name": "A", "city": "Lahore", "in_universe": 0},
        ]
    )
    sm = pd.DataFrame(
        _months(OLD, "Madina Store", {"2026-05": 0.2, "2026-06": 0.2}, dist="D1", dsr="A")
        + _months(NEW, "Madina Store", {"2026-07": 0.2}, dist="D1", dsr="A")
        + _months(OTHER, "Madina Store", {"2026-07": 0.2}, dist="D1", dsr="A")
    )
    pairs, left = build_lineage(sm, stores, None)
    assert pairs.empty
    assert len(left) == 1
    assert left.iloc[0]["store_id"] == OLD
    assert "2 live codes share this name" in left.iloc[0]["reason"]
    assert int(left.iloc[0]["n_candidates"]) == 2


def test_superseded_code_still_on_universe_merges_with_same_dsr_and_clean_handover():
    stores = _stores(
        [
            {"store_id": OLD, "store_name": "Butt Store", "distributor": "D1", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
            {"store_id": NEW, "store_name": "Butt Store", "distributor": "D1", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
        ]
    )
    sm = pd.DataFrame(
        _months(OLD, "Butt Store", {"2026-05": 0.3, "2026-06": 0.3}, dist="D1")
        + _months(NEW, "Butt Store", {"2026-06": 0.1, "2026-07": 0.3, "2026-08": 0.3}, dist="D1")
    )
    sd = pd.DataFrame(
        _days(OLD, {"2026-05-10": 0.3, "2026-06-12": 0.3})
        + _days(NEW, {"2026-06-20": 0.1, "2026-07-15": 0.3, "2026-08-20": 0.3})
    )
    pairs, left = build_lineage(sm, stores, sd)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["old_id"], row["new_id"], row["method"]) == (OLD, NEW, METHOD_SUPERSEDED)
    assert "same DSR" in row["detail"]
    assert left.empty  # nothing is *retired*; the superseded code is a universe clean-up item


def test_superseded_needs_local_evidence_not_just_distributor_and_city():
    # Same distributor and city, same name, clean handover — but a different DSR
    # and a different code route. "Khan Store" repeats across a city; not merged.
    stores = _stores(
        [
            {"store_id": OLD, "store_name": "Khan Store", "distributor": "D1", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
            {"store_id": FAR, "store_name": "Khan Store", "distributor": "D1", "dsr_name": "Shakeel", "city": "Lahore", "in_universe": 1},
        ]
    )
    sm = pd.DataFrame(
        _months(OLD, "Khan Store", {"2026-05": 0.3, "2026-06": 0.3}, dist="D1")
        + _months(FAR, "Khan Store", {"2026-07": 0.3, "2026-08": 0.3}, dist="D1", dsr="Shakeel")
    )
    sd = pd.DataFrame(_days(OLD, {"2026-06-25": 0.3}) + _days(FAR, {"2026-07-02": 0.3, "2026-08-20": 0.3}))
    pairs, _left = build_lineage(sm, stores, sd)
    assert pairs.empty


def test_superseded_code_that_is_still_billing_is_not_merged():
    stores = _stores(
        [
            {"store_id": OLD, "store_name": "Butt Store", "distributor": "D1", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
            {"store_id": NEW, "store_name": "Butt Store", "distributor": "D1", "dsr_name": "Umair", "city": "Lahore", "in_universe": 1},
        ]
    )
    # Both codes bill through the last month on file: two doors (or a duplicate), not a handover.
    sm = pd.DataFrame(
        _months(OLD, "Butt Store", {"2026-05": 0.3, "2026-06": 0.3, "2026-07": 0.3, "2026-08": 0.3}, dist="D1")
        + _months(NEW, "Butt Store", {"2026-06": 0.3, "2026-07": 0.3, "2026-08": 0.3}, dist="D1")
    )
    pairs, _left = build_lineage(sm, stores, None)
    assert pairs.empty


def test_pipeline_merges_superseded_code_and_shop_book_counts_the_door_once(tmp_path):
    from sndintel.ingest.pipeline import run_pipeline
    from sndintel.lineage import load_lineage, load_shop_day
    from sndintel.storage import connect, read_sql

    sales = tmp_path / "daily.xlsx"
    universe = tmp_path / "universe.xlsx"
    dates = [date(2026, 5, 10), date(2026, 6, 12), date(2026, 6, 20), date(2026, 7, 15), date(2026, 8, 10)]
    shops = [(OLD, "Butt Store"), (NEW, "Butt Store"), (OTHER, "Ali Traders")]
    volumes = [
        [0.30, 0.30, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.10, 0.30, 0.30],
        [0.20, 0.20, 0.00, 0.20, 0.20],
    ]
    _write_daily(sales, shops=shops, dates=dates, volumes=volumes)
    rows = [["DISTRIBUTOR NAME", "Area", "SECTION LONG DESCRIPTION", "DSR NAME", "POP Code", "POP NAME"]]
    for sid, name in shops:
        rows.append(["Rubina Shaheen (LHR)", "Lahore", "Shadman", "Umair", sid, name])
    pd.DataFrame(rows).to_excel(universe, header=False, index=False)
    db = tmp_path / "wh.db"
    run_pipeline(sales, universe_path=universe, db_path=db)

    with connect(db) as conn:
        lineage = load_lineage(conn)
        sm = read_sql(conn, "SELECT * FROM shop_month")
        stores = read_sql(conn, "SELECT * FROM stores")
        ledger = read_sql(conn, "SELECT * FROM period_ledger")
        facts = read_sql(conn, "SELECT * FROM sales_facts")
        action_rows = read_sql(conn, "SELECT * FROM action_shops")
        shop_day = load_shop_day(conn)

    assert len(lineage) == 1
    assert (lineage.iloc[0]["old_id"], lineage.iloc[0]["new_id"], lineage.iloc[0]["method"]) == (OLD, NEW, METHOD_SUPERSEDED)
    # Raw facts keep the old code; derived tables carry the door under the live code.
    assert OLD in set(facts["store_id"].astype(str))
    assert OLD not in set(sm["store_id"].astype(str))
    new_hist = sm[sm["store_id"].astype(str) == NEW].set_index("period")["volume_mt"]
    assert abs(float(new_hist["2026-05"]) - 0.30) < 1e-9
    assert abs(float(new_hist["2026-06"]) - 0.40) < 1e-9
    assert OLD not in set(shop_day["store_id"].astype(str))
    old_row = action_rows[action_rows["store_id"].astype(str) == OLD]
    assert not old_row.empty
    assert set(old_row["action"]) == {ACTION_SUPERSEDED}
    assert set(old_row["superseded_by"].astype(str)) == {NEW}
    assert float(pd.to_numeric(old_row["expected_mt"], errors="coerce").fillna(0).sum()) == 0.0

    action = build_action_pack(sm, stores, shop_day=shop_day, ledger=ledger, period="2026-07", lineage=lineage)
    book = build_shop_book(action=action, shop_month=sm, ledger=ledger, period="2026-07", scope="city", city="Lahore")
    raw = book.raw
    assert set(raw.loc[raw["store_id"].astype(str) == OLD, "issue"]) == {ISSUE_SUPERSEDED}
    kpis = book.kpis
    assert kpis["n_superseded"] == 1
    assert kpis["n_merged"] == 1
    assert kpis["n_universe"] == 2  # OLD is listed but is not a door
    assert kpis["n_shops"] == 2
    mix = book.mix
    total = mix[mix["Issue"] == MIX_TOTAL].iloc[0]
    body = mix[mix["Issue"] != MIX_TOTAL]
    assert int(total["Shops"]) == int(body["Shops"].sum())
    assert abs(float(total["Billed (MT)"]) - float(kpis["billed_mt"])) < 0.01
    assert not book.retired.empty
    sup = book.retired[book.retired["POP"] == OLD].iloc[0]
    assert sup["Status"].startswith("Superseded")
    assert NEW in sup["What to do"]
    roll = book.rollup
    tot = roll[roll[roll.columns[0]] == MIX_TOTAL].iloc[0]
    assert int(tot["Universe doors"]) == 2
    assert int(tot["Retired codes"]) == 1
    assert "superseded" in book.weather.lower()


def test_returns_net_at_month_level_and_totals_tie(tmp_path):
    from sndintel.ingest.ssrs import parse_sales_file

    sales = tmp_path / "daily.xlsx"
    dates = [date(2026, 5, 10), date(2026, 5, 20), date(2026, 6, 5)]
    shops = [(OLD, "Butt Store")]
    volumes = [[0.50, -0.20, 0.30]]
    _write_daily(sales, shops=shops, dates=dates, volumes=volumes)
    df, report = parse_sales_file(sales)
    h = df[df["store_id"] == OLD].set_index("period")["volume_mt"]
    assert abs(float(h["2026-05"]) - 0.30) < 1e-9
    assert abs(float(h["2026-06"]) - 0.30) < 1e-9
    daily = report.daily
    assert (daily["volume_mt"] < 0).any()
    assert abs(float(daily["volume_mt"].sum()) - 0.60) < 1e-9


def test_total_matcher_keeps_shops_whose_name_contains_total():
    for name in ("TOTAL PUMP", "STAR MART TOTAL PUMP", "TOTAL TAKE SHOP", "TOTAL M/STORE"):
        assert not looks_like_total(name), name
    for label in ("Total", "Grand Total", "GRAND TOTAL:", "Umair Total", "Total for Lahore", "Sub Total"):
        assert looks_like_total(label), label


def test_window_stops_at_the_first_month_on_file():
    sm = pd.DataFrame(_months(NEW, "Butt Store", {"2026-05": 1.0, "2026-06": 1.0, "2026-07": 1.0}))
    assert window_periods("2026-07", 3, sm) == ["2026-05", "2026-06"]
    assert window_periods("2026-08", 3, sm) == ["2026-05", "2026-06", "2026-07"]
    assert window_periods("2026-08", 6, sm) == ["2026-05", "2026-06", "2026-07"]
    assert window_periods("2026-07", 3, None) == ["2026-04", "2026-05", "2026-06"]


def test_expected_does_not_treat_months_before_the_extract_as_zero():
    # Flat shop, data on file from May. Scoring July must not average April as 0.
    rows = _months(NEW, "Butt Store", {"2026-05": 1.0, "2026-06": 1.0, "2026-07": 1.0})
    rows += _months(OTHER, "Ali Traders", {"2026-05": 0.5, "2026-06": 0.5, "2026-07": 0.5})
    sm = pd.DataFrame(rows)
    fit = fit_seasonality(sm, "2026-07")
    exp = fit_shop_expected(sm, "2026-07", fit, intra_frac=1.0).set_index("store_id")
    assert abs(float(exp.loc[NEW, "expected_full_mt"]) - 1.0) < 1e-6
    assert abs(float(exp.loc[OTHER, "expected_full_mt"]) - 0.5) < 1e-6
    city = fit.city_expected.set_index("city")
    assert abs(float(city.loc["Lahore", "expected_full_mt"]) - 1.5) < 1e-6
