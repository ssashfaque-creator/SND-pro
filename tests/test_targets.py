"""Shop-wise sales-team targets stay a plan layer. Expected does not change."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sndintel.ingest.pipeline import run_pipeline
from sndintel.ingest.shops import parse_shop_master
from sndintel.ingest.targets import match_shop_targets, parse_shop_targets
from sndintel.plan import attach_plan
from sndintel.situation_report import build_situation_pack
from sndintel.storage import connect, read_sql
from sndintel.watch import _is_shop_file, _is_target_file


def _ssrs_csv(path: Path, rows: list[tuple]) -> Path:
    header = "txt_cRegion,txt_cArea,txt_cDISTRIBUTOR_NAME,txt_cDSR_NAME,txt_cPOP_NAME,txt_TARGET_UOM,uval_TARGET_UOM"
    lines = [header]
    for zone, city, dist, dsr, name, tgt in rows:
        lines.append(f"{zone},{city},{dist},{dsr},{name},TARGET_UOM,{tgt}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _stores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "store_id": "T0001601407",
                "store_name": "Hameed GS",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "zone": "South",
                "city": "Karachi",
                "section": "Clifton",
            },
            {
                "store_id": "T0001601408",
                "store_name": "Kifaya",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "zone": "South",
                "city": "Karachi",
                "section": "Clifton",
            },
            {
                "store_id": "T0001601409",
                "store_name": "Diamond Super Market",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "zone": "South",
                "city": "Karachi",
                "section": "Clifton",
            },
            {
                "store_id": "T0001700001",
                "store_name": "Kifaya",
                "distributor": "Lahore Dist",
                "dsr_name": "Bilal",
                "zone": "Central",
                "city": "Lahore",
                "section": "Gulberg",
            },
        ]
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "grain": "national",
                "grain_id": "ALL",
                "city": "",
                "distributor": "",
                "dsr_name": "",
                "volume_mt": 10.0,
                "expected_mt": 12.0,
                "gap_mt": 2.0,
                "situation": "lagging",
                "intra_month_frac": 1.0,
            },
            {
                "grain": "city",
                "grain_id": "Karachi",
                "city": "Karachi",
                "parent_id": "ALL",
                "distributor": "",
                "dsr_name": "",
                "volume_mt": 6.0,
                "expected_mt": 8.0,
                "gap_mt": 2.0,
                "situation": "lagging",
                "intra_month_frac": 1.0,
            },
            {
                "grain": "city",
                "grain_id": "Lahore",
                "city": "Lahore",
                "parent_id": "ALL",
                "distributor": "",
                "dsr_name": "",
                "volume_mt": 4.0,
                "expected_mt": 4.0,
                "gap_mt": 0.0,
                "situation": "with_market",
                "intra_month_frac": 1.0,
            },
            {
                "grain": "distributor",
                "grain_id": "Eva Foods",
                "city": "Karachi",
                "parent_id": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "",
                "volume_mt": 6.0,
                "expected_mt": 8.0,
                "gap_mt": 2.0,
                "situation": "lagging",
                "intra_month_frac": 1.0,
            },
            {
                "grain": "dsr",
                "grain_id": "Amir | Karachi | Eva Foods",
                "city": "Karachi",
                "parent_id": "Karachi",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "volume_mt": 6.0,
                "expected_mt": 8.0,
                "gap_mt": 2.0,
                "situation": "lagging",
                "intra_month_frac": 1.0,
            },
        ]
    )


def test_parse_ssrs_field_ids_drops_na_keeps_zero(tmp_path):
    path = _ssrs_csv(
        tmp_path / "targets.csv",
        [
            ("South", "Karachi", "Eva Foods", "Amir", "Hameed GS", 1.5),
            ("South", "Karachi", "Eva Foods", "Amir", "NA", 9.0),
            ("South", "Karachi", "Eva Foods", "Amir", "Quiet Door", 0.0),
        ],
    )
    df, report = parse_shop_targets(path)
    assert report.strategy == "ssrs_or_headers"
    names = set(df["store_name"].str.upper())
    assert "HAMEED GS" in names
    assert "NA" not in names
    quiet = df[df["store_name"].str.lower() == "quiet door"]
    assert len(quiet) == 1
    assert float(quiet.iloc[0]["target_mt"]) == 0.0
    assert abs(report.book_mt - 1.5) < 1e-9


def test_exact_and_unique_city_name_match():
    stores = _stores()
    targets = pd.DataFrame(
        [
            {
                "store_name": "Hameed GS",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "city": "Karachi",
                "target_mt": 1.2,
            },
            {
                "store_name": "Kifaya",
                "distributor": "Lahore Dist",
                "dsr_name": "Bilal",
                "city": "Lahore",
                "target_mt": 0.4,
            },
        ]
    )
    matched, report = match_shop_targets(targets, stores)
    by_name = matched.set_index("store_name")
    assert by_name.loc["Hameed GS", "store_id"] == "T0001601407"
    assert by_name.loc["Hameed GS", "match_method"] == "city_dist_dsr_name"
    assert by_name.loc["Kifaya", "store_id"] == "T0001700001"
    assert report.n_matched == 2


def test_whale_is_not_fuzzy_matched_to_the_wrong_shop():
    stores = _stores()
    targets = pd.DataFrame(
        [
            {
                "store_name": "Diamond Super Mart",
                "distributor": "Someone Else",
                "dsr_name": "Other",
                "city": "Karachi",
                "target_mt": 21.87,
            }
        ]
    )
    matched, report = match_shop_targets(targets, stores)
    assert report.n_matched == 0
    assert matched.iloc[0]["match_method"] == "unmatched"
    assert str(matched.iloc[0]["store_id"] or "") == ""


def test_attach_plan_rolls_full_book_and_leaves_expected():
    units = _units()
    before = units["expected_mt"].tolist()
    stores = _stores()
    targets = pd.DataFrame(
        [
            {
                "store_name": "Hameed GS",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "city": "Karachi",
                "target_mt": 2.0,
            },
            {
                "store_name": "Unknown Kiryana",
                "distributor": "Ghost Dist",
                "dsr_name": "Ghost",
                "city": "Karachi",
                "target_mt": 3.0,
            },
        ]
    )
    matched, _ = match_shop_targets(targets, stores)
    out = attach_plan(units, matched, pace=1.0)
    assert out["expected_mt"].tolist() == before
    nat = out[out["grain"] == "national"].iloc[0]
    assert abs(float(nat["target_mt"]) - 5.0) < 1e-9
    assert abs(float(nat["target_book_mt"]) - 5.0) < 1e-9
    assert float(nat["target_matched_mt"]) == 2.0
    assert int(nat["n_target_unmatched"]) == 1
    khi = out[out["grain_id"] == "Karachi"].iloc[0]
    assert abs(float(khi["target_mt"]) - 5.0) < 1e-9  # unmatched still rolls on Area
    assert abs(float(khi["stretch_mt"])) < 1e-9
    eva = out[out["grain"] == "distributor"].iloc[0]
    assert abs(float(eva["target_mt"]) - 2.0) < 1e-9  # unmatched dist name does not land on Eva


def test_situation_pack_shows_target_when_plan_exists():
    units = attach_plan(_units(), pd.DataFrame(
        [
            {
                "store_id": "T0001601407",
                "store_name": "Hameed GS",
                "distributor": "Eva Foods",
                "dsr_name": "Amir",
                "city": "Karachi",
                "match_method": "city_dist_dsr_name",
                "target_mt": 20.0,
            }
        ]
    ))
    pack = build_situation_pack(units, period="2026-08")
    assert float(pack.kpis.get("target_mt") or 0) > 0
    assert "Target (MT)" in pack.lagging_cities.columns or float(pack.kpis["target_mt"]) >= 20
    text = " ".join(pack.situation)
    assert "plan" in text.lower() or "Plan" in text or pack.kpis.get("stretch_mt") is not None


def test_watch_does_not_treat_shopwise_targets_as_universe():
    path = Path("shopwise_targets_5d46.csv")
    assert _is_target_file(path)
    assert not _is_shop_file(path)


def test_pipeline_optional_targets(demo, tmp_path):
    db = tmp_path / "warehouse.db"
    shops, _ = parse_shop_master(demo["shops"])
    row = shops.iloc[0]
    path = _ssrs_csv(
        tmp_path / "plan.csv",
        [
            (
                str(row.get("zone") or "South"),
                str(row.get("city") or "Karachi"),
                str(row["distributor"]),
                str(row["dsr_name"]),
                str(row["store_name"]),
                1.25,
            )
        ],
    )
    result = run_pipeline(demo["sales"], shop_path=demo["shops"], targets_path=path, db_path=db)
    assert result.get("ok") is not False
    assert result.get("n_plan_shops") == 1
    with connect(db) as conn:
        units = read_sql(conn, "SELECT * FROM unit_scorecards WHERE grain = 'national'")
        stored = read_sql(conn, "SELECT * FROM shop_targets")
    assert not stored.empty
    assert float(units.iloc[0]["target_mt"]) > 0
    expected_before = float(units.iloc[0]["expected_mt"])
    again = run_pipeline(targets_path=path, db_path=db)
    with connect(db) as conn:
        units2 = read_sql(conn, "SELECT * FROM unit_scorecards WHERE grain = 'national'")
    assert abs(float(units2.iloc[0]["expected_mt"]) - expected_before) < 1e-6
    assert again.get("n_plan_matched") is not None
