"""Monday / DSR / Friday operating packs and closed-loop scoring."""

import pandas as pd

from sndintel.action import ACTION_CALL, ACTION_CONVERT, persist_action_pack
from sndintel.ops import (
    beat_owner_options,
    build_dsr_beat_pack,
    build_friday_pack,
    build_monday_pack,
    filter_beat_by_owner,
    load_outcomes,
    score_closed_loop,
)
from sndintel.storage import connect, init_db


def _listed(sid, billed=0.0, visits=0, action=ACTION_CALL, ask=1.0):
    return {
        "store_id": sid,
        "store_name": sid,
        "action": action,
        "billed_mt": billed,
        "visits": visits,
        "week_target_mt": ask,
        "city": "Karachi",
        "distributor": "Eva",
        "dsr_name": "Amir",
    }


def test_closed_loop_billed_visited_unbilled_and_not_visited():
    previous = pd.DataFrame(
        [
            _listed("BILL", billed=0.0, visits=0),
            _listed("SEEN", billed=0.0, visits=1, action=ACTION_CONVERT),
            _listed("SKIP", billed=0.0, visits=0),
        ]
    )
    shop_month = pd.DataFrame(
        [
            {"store_id": "BILL", "period": "2026-08", "volume_mt": 0.8},
            {"store_id": "SEEN", "period": "2026-08", "volume_mt": 0.0},
            {"store_id": "SKIP", "period": "2026-08", "volume_mt": 0.0},
        ]
    )
    visits = pd.DataFrame(
        [
            {"store_id": "BILL", "period": "2026-08", "visits": 2},
            {"store_id": "SEEN", "period": "2026-08", "visits": 3},
            {"store_id": "SKIP", "period": "2026-08", "visits": 0},
        ]
    )
    out = score_closed_loop(previous, shop_month, visits, "2026-08")
    got = {row.store_id: row.outcome for row in out.itertuples(index=False)}
    assert got["BILL"] == "billed"
    assert got["SEEN"] == "visited_unbilled"
    assert got["SKIP"] == "not_visited"
    friday = build_friday_pack(out)
    assert "billed after the list" in friday.headline.lower()
    assert not friday.sheets[0][3].empty


def test_dsr_beat_pack_caps_per_person():
    from sndintel.action import ActionPack
    from sndintel.config import DSR_DAY_CAP

    rows = []
    for i in range(30):
        rows.append(
            {
                "store_id": f"S{i}",
                "store_name": f"Shop {i}",
                "dsr_name": "Amir",
                "city": "Karachi",
                "distributor": "Eva",
                "action": ACTION_CALL,
                "week_target_mt": 1.0 - i * 0.01,
                "value_score": 30 - i,
                "instruction": "Call",
                "next_drop_mt": 0.5,
                "days_since_bill": 20,
                "cover_left_days": 0,
            }
        )
    pack = ActionPack(period="2026-08", label="August 2026", raw_shops=pd.DataFrame(rows), headline="test")
    beat = build_dsr_beat_pack(pack)
    listed = beat.sheets[0][3]
    assert len(listed) == DSR_DAY_CAP


def test_beat_filter_keeps_two_people_with_the_same_name_apart():
    from sndintel.action import ActionPack

    rows = []
    for city, dist, n in (("Karachi", "Eva", 5), ("Lahore", "Punjab", 5)):
        for i in range(n):
            rows.append(
                {
                    "store_id": f"{city[:1]}{i}",
                    "store_name": f"{city} shop {i}",
                    "dsr_name": "Shahid",
                    "city": city,
                    "distributor": dist,
                    "action": ACTION_CALL,
                    "week_target_mt": 1.0,
                    "value_score": n - i,
                    "instruction": "Call",
                    "next_drop_mt": 0.4,
                    "days_since_bill": 18,
                    "cover_left_days": 0,
                }
            )
    pack = ActionPack(period="2026-08", label="August 2026", raw_shops=pd.DataFrame(rows), headline="test")
    beat = build_dsr_beat_pack(pack, per_dsr=12)
    table = beat.sheets[0][3]
    owners = beat_owner_options(table)
    assert owners == ["Shahid · Karachi · Eva", "Shahid · Lahore · Punjab"]
    karachi = filter_beat_by_owner(table, owners[0])
    lahore = filter_beat_by_owner(table, owners[1])
    assert set(karachi["City"]) == {"Karachi"}
    assert set(lahore["City"]) == {"Lahore"}
    assert len(karachi) == 5
    assert len(lahore) == 5


def test_monday_pack_has_capacity_and_whales():
    from sndintel.action import ActionPack

    shops = pd.DataFrame(
        [
            {
                "store_id": "W",
                "store_name": "Whale Mart",
                "city": "Karachi",
                "distributor": "Eva",
                "dsr_name": "Amir",
                "ams_3m": 2.0,
                "expected_mt": 2.0,
                "billed_mt": 0.2,
                "remaining_mt": 1.8,
                "week_target_mt": 1.0,
                "call_status": "Unvisited",
                "action": ACTION_CALL,
                "instruction": "Call Whale Mart",
                "value_score": 10,
            }
        ]
    )
    units = pd.DataFrame(
        [
            {
                "grain": "city",
                "grain_id": "Karachi",
                "volume_mt": 10,
                "expected_mt": 20,
                "isolated_mt": -10,
                "from_unbilled_mt": 8,
                "from_unvisited_mt": 1,
                "from_drop_size_mt": 1,
                "visit_rate": 0.99,
                "strike_rate": 0.4,
                "universe": 800,
                "has_visit_file": 1,
            }
        ]
    )
    pack = ActionPack(
        period="2026-08",
        label="August 2026",
        headline="Karachi is the hole",
        as_of_day=21,
        days_in_month=31,
        days_left=10,
        country=pd.DataFrame([{"Billed (KG)": 200}]),
        distributors=pd.DataFrame([{"Distributor": "Eva"}]),
        raw_shops=shops,
    )
    monday = build_monday_pack(pack, units, visits=pd.DataFrame([{"store_id": "W", "period": "2026-08", "visits": 1}]))
    titles = [s[0] for s in monday.sheets]
    assert "03 DSR labels" in titles
    assert "04 Whales" in titles
    whales = monday.sheets[3][3]
    assert "Whale Mart" in set(whales["Shop"].astype(str))
    assert any("Karachi" in w for w in monday.warnings)


def test_persist_action_pack_scores_previous_list(tmp_path):
    from sndintel.action import ActionPack

    db = tmp_path / "w.db"
    init_db(db)
    first = ActionPack(
        period="2026-08",
        headline="first",
        raw_shops=pd.DataFrame(
            [
                {
                    "store_id": "S1",
                    "store_name": "Shop 1",
                    "city": "Karachi",
                    "distributor": "Eva",
                    "dsr_name": "Amir",
                    "action": ACTION_CALL,
                    "instruction": "Call",
                    "billed_mt": 0.0,
                    "visits": 0,
                    "week_target_mt": 1.0,
                    "expected_mt": 1.0,
                    "ams_3m": 1.0,
                }
            ]
        ),
    )
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO shop_month (store_id, period, volume_mt) VALUES (?, ?, ?)",
            ("S1", "2026-08", 0.0),
        )
        persist_action_pack(conn, first)
        conn.execute("UPDATE shop_month SET volume_mt = 0.6 WHERE store_id = 'S1'")
        persist_action_pack(conn, first)
        outcomes = load_outcomes(conn, "2026-08")
    assert not outcomes.empty
    assert outcomes.iloc[0]["outcome"] == "billed"


def test_loaded_pack_still_builds_beat_lists(tmp_path):
    from sndintel.action import ActionPack, persist_action_pack, load_action_pack
    from sndintel.config import DSR_DAY_CAP

    db = tmp_path / "w.db"
    init_db(db)
    rows = []
    for i in range(20):
        rows.append(
            {
                "store_id": f"S{i}",
                "store_name": f"Shop {i}",
                "dsr_name": "Amir",
                "city": "Karachi",
                "distributor": "Eva",
                "action": ACTION_CALL,
                "instruction": "Call",
                "billed_mt": 0.0,
                "visits": 0,
                "week_target_mt": 1.0 - i * 0.02,
                "value_score": 20 - i,
                "coming_due": False,
                "ams_3m": 0.4,
                "expected_mt": 0.5,
            }
        )
    pack = ActionPack(period="2026-08", label="August 2026", headline="Karachi is the hole", raw_shops=pd.DataFrame(rows))
    with connect(db) as conn:
        persist_action_pack(conn, pack)
        loaded = load_action_pack(conn, "2026-08")
    assert loaded.headline == "Karachi is the hole"
    assert loaded.raw_shops is not None and not loaded.raw_shops.empty
    beat = build_dsr_beat_pack(loaded)
    listed = beat.sheets[0][3]
    assert len(listed) == DSR_DAY_CAP
    assert "Shop 0" in set(listed["Shop"].astype(str))
