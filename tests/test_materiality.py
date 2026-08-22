import pandas as pd

from sndintel.materiality import classify_gap_shops, must_visit_recoveries, tier_from_volumes


def test_pareto_tiers_separate_micro_shops():
    vols = {}
    for i in range(20):
        vols[f"C{i:03d}"] = 1.0  # 20 MT
    for i in range(10):
        vols[f"M{i:03d}"] = 0.2  # 2 MT
    for i in range(70):
        vols[f"T{i:03d}"] = 0.01  # 0.70 MT
    tiers = tier_from_volumes(pd.Series(vols))
    counts = tiers.groupby("tier").size().to_dict()
    assert counts["core"] < 30
    assert counts["tail"] >= 50
    assert float(tiers.loc[tiers["tier"] == "tail", "volume_mt"].sum()) < 2.0
    assert float(tiers.loc[tiers["tier"] == "core", "volume_mt"].sum()) >= 0.75 * float(tiers["volume_mt"].sum())


def test_lost_micro_shops_are_tail_not_must_visit():
    rows = []
    # Core shop billed both years
    rows.append(_sm("CORE1", "2025-08", 5.0, "A"))
    rows.append(_sm("CORE1", "2026-08", 4.5, "A"))
    # Material shop lost
    rows.append(_sm("BIG1", "2025-08", 2.0, "A"))
    rows.append(_sm("BIG1", "2026-08", 0.0, "A", billed=0))
    # 40 micro shops billed last year only
    for i in range(40):
        rows.append(_sm(f"TINY{i:02d}", "2025-08", 0.01, "B"))
        rows.append(_sm(f"TINY{i:02d}", "2026-08", 0.0, "B", billed=0))
    sm = pd.DataFrame(rows)
    gap = classify_gap_shops(sm, "2026-08")
    assert gap["counts"]["lost_core"] + gap["counts"]["lost_middle"] == 1
    assert gap["counts"]["lost_tail"] == 40
    visit = must_visit_recoveries(gap)
    assert list(visit["store_id"]) == ["BIG1"]
    assert "TINY00" not in set(visit["store_id"])
    # Tail volume is still counted so it is not ignored.
    assert abs(gap["volumes"]["lost_tail"] - 0.40) < 1e-6


def _sm(store_id, period, volume, section, billed=None):
    year, month = period.split("-")
    return {
        "store_id": store_id,
        "store_name": store_id,
        "period": period,
        "year": int(year),
        "month": int(month),
        "volume_mt": volume,
        "billed": 1 if billed is None and volume > 0 else (billed if billed is not None else 0),
        "section": section,
        "dsr_name": "DSR-A",
        "city": "Quetta",
        "distributor": "Agha",
        "sku_count": 1,
    }
