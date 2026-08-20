"""Strategy pack: AMS, recoverable volume, layered lists, Excel export."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from openpyxl import load_workbook

from sndintel.briefing import ams_last_n, build_strategy_pack, excel_bytes, render_html
from sndintel.hierarchy import build_hierarchy_pack


def _row(store_id, period, volume, city, dist, dsr, section="A", name="Shop"):
    return {
        "store_id": store_id,
        "period": period,
        "year": int(period[:4]),
        "month": int(period[5:7]),
        "volume_mt": volume,
        "sku_count": 1,
        "billed": 1 if volume > 0 else 0,
        "distributor": dist,
        "dsr_name": dsr,
        "section": section,
        "store_name": name,
        "zone": "South" if city == "Karachi" else "Central",
        "city": city,
    }


def _stores(rows):
    seen = {}
    for r in rows:
        seen[r["store_id"]] = r
    return pd.DataFrame(
        [
            {
                "store_id": r["store_id"],
                "store_name": r["store_name"],
                "distributor": r["distributor"],
                "dsr_name": r["dsr_name"],
                "zone": r["zone"],
                "city": r["city"],
                "section": r["section"],
            }
            for r in seen.values()
        ]
    )


def test_ams_is_mean_of_last_three_closed_months():
    rows = []
    for per, vol in [("2026-05", 10.0), ("2026-06", 20.0), ("2026-07", 30.0), ("2026-08", 5.0)]:
        rows.append(_row("K1", per, vol, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    sm = pd.DataFrame(rows)
    ams = ams_last_n(sm, "2026-08", ["city"])
    assert abs(float(ams.iloc[0]["ams_3m"]) - 20.0) < 1e-9


def test_pack_layers_cities_then_those_dists_then_all_dists():
    """Karachi lags the country. Inside Lahore (ahead), Local Dist still lags the city."""
    rows = []
    # Karachi −80% vs LY, Lahore −20% vs LY, national −50%.
    rows.append(_row("K1", "2026-08", 5.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2025-08", 80.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K2", "2026-08", 15.0, "Karachi", "South Dist", "Amir", name="Hold K"))
    rows.append(_row("K2", "2025-08", 20.0, "Karachi", "South Dist", "Amir", name="Hold K"))
    rows.append(_row("L1", "2026-08", 90.0, "Lahore", "Holding Dist", "Lahore Ace", name="Big L"))
    rows.append(_row("L1", "2025-08", 100.0, "Lahore", "Holding Dist", "Lahore Ace", name="Big L"))
    rows.append(_row("L2", "2026-08", 10.0, "Lahore", "Local Dist", "Lahore Weak", name="Small L"))
    rows.append(_row("L2", "2025-08", 100.0, "Lahore", "Local Dist", "Lahore Weak", name="Small L"))
    sm = pd.DataFrame(rows)
    pack_h = build_hierarchy_pack(sm, _stores(rows), ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    report = build_strategy_pack(pack_h.units, sm, period="2026-08")
    cities = report.cities
    assert "City" in cities.columns
    assert "AMS last 3 months (MT)" in cities.columns
    assert "Recoverable (MT)" in cities.columns
    assert "Extra vs country (MT)" in cities.columns
    khi = cities[cities["City"] == "Karachi"].iloc[0]
    assert khi["Situation"] == "Lagging"
    assert float(khi["Recoverable (MT)"]) > 0
    # Drill-down distributors only in lagging cities → Karachi, not Lahore.
    drill = report.city_distributors
    assert set(drill["City"]) <= {"Karachi"}
    assert "Eva Foods" in set(drill["Distributor"].astype(str))
    assert "Local Dist" not in set(drill["Distributor"].astype(str))
    # Full lagging-distributor list includes Local Dist in Lahore (city is ahead).
    names = set(report.lagging_distributors["Distributor"].astype(str))
    assert "Local Dist" in names
    assert "Eva Foods" in names


def test_excel_and_html_are_readable_packs():
    rows = []
    rows.append(_row("K1", "2026-08", 10.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2025-08", 30.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2026-07", 28.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2026-06", 26.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2026-05", 24.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    sm = pd.DataFrame(rows)
    pack_h = build_hierarchy_pack(sm, _stores(rows), ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    report = build_strategy_pack(pack_h.units, sm, period="2026-08")
    raw = excel_bytes(report)
    wb = load_workbook(BytesIO(raw))
    names = wb.sheetnames
    assert names[0] == "00 Cover"
    assert "01 Country by city" in names
    assert "04 All lagging distributors" in names
    assert "06 All lagging shops" in names
    cover = wb["00 Cover"]
    assert "Glossary" in [c.value for row in cover.iter_rows(min_col=1, max_col=1, values_only=False) for c in row]
    html = render_html(report)
    assert "AMS last 3 months" in html or "Recoverable" in html
    assert "Glossary" in html
    assert "Fair share" in html or "Extra vs" in html
