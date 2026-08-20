"""Synthetic but realistic Pakistani edible-oil S&D data, including a messy SSRS export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

from sndintel.config import SAMPLE_DIR, ensure_dirs
from sndintel.io_utils import period_key

SKUS = [
    "Maan Banaspati 1kg",
    "Maan Banaspati 5kg",
    "Maan Banaspati 16kg Tin",
    "Maan Cooking Oil 1L",
    "Maan Cooking Oil 5L",
    "Maan Cooking Oil 16L",
    "Maan Super Banaspati 5kg",
    "Maan Ghee 1kg",
]

# Relative mix; 5L oil and 16kg tin will be swapped later to mimic cannibalization.
SKU_SHARE = np.array([0.10, 0.16, 0.22, 0.10, 0.14, 0.12, 0.10, 0.06])

ZONES = {
    "West": {
        "Quetta": {
            "distributor": "Agha Traders (Quetta)",
            "sections": {
                "Alamdar Road": ["ASHRAF KHAN"],
                "Toghi Road": ["ASHRAF KHAN"],
                "Satellite Town": ["IMRAN BALOCH"],
                "Jinnah Road": ["IMRAN BALOCH"],
            },
        },
        "Pishin": {
            "distributor": "Kakar Brothers",
            "sections": {"Pishin Bazaar": ["SADIQ KAKAR"], "Khanozai": ["SADIQ KAKAR"]},
        },
    },
    "South": {
        "Karachi": {
            "distributor": "Coastal Foods",
            "sections": {
                "Korangi": ["BILAL SHAIKH"],
                "Orangi": ["BILAL SHAIKH"],
                "Saddar": ["NADIA RIZVI"],
                "Malir": ["NADIA RIZVI"],
            },
        },
        "Hyderabad": {
            "distributor": "Sindh Distributors",
            "sections": {"Latifabad": ["KAMRAN QURESHI"], "Qasimabad": ["KAMRAN QURESHI"]},
        },
    },
    "North": {
        "Lahore": {
            "distributor": "Punjab Trading Co",
            "sections": {
                "Allama Iqbal Town": ["USMAN RIAZ"],
                "Model Town": ["USMAN RIAZ"],
                "Ravi Road": ["FARAH MALIK"],
            },
        },
        "Faisalabad": {
            "distributor": "Lyallpur Stores",
            "sections": {"Clock Tower": ["HAMZA TARIQ"], "Madina Town": ["HAMZA TARIQ"]},
        },
    },
}

CHANNELS = ["General Trade", "Wholesaler", "Kiriana", "Hotel / Khoka"]
CLASSES = ["A", "B", "C", "D"]


def _store_id(n: int) -> str:
    return f"T{n:010d}"


def build_universe(n_shops: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    slots = []
    for zone, cities in ZONES.items():
        for city, spec in cities.items():
            for section, dsrs in spec["sections"].items():
                for dsr in dsrs:
                    slots.append((zone, city, spec["distributor"], section, dsr))
    rows = []
    for i in range(n_shops):
        zone, city, dist, section, dsr = slots[i % len(slots)]
        # Sprinkle extra shops onto Alamdar Road so divergence is visible.
        if i % 11 == 0:
            zone, city, dist, section, dsr = (
                "West",
                "Quetta",
                "Agha Traders (Quetta)",
                "Alamdar Road",
                "ASHRAF KHAN",
            )
        sid = _store_id(1_600_000 + i)
        rows.append(
            {
                "distributor": dist,
                "dsr_name": dsr,
                "store_id": sid,
                "store_name": f"{city[:3].upper()}-{section.split()[0]} Shop {i+1:03d}",
                "category_1": CHANNELS[int(rng.integers(0, len(CHANNELS)))],
                "category_2": CLASSES[int(rng.choice(len(CLASSES), p=[0.15, 0.35, 0.35, 0.15]))],
                "category_3": f"{section} locality",
                "category_4": rng.choice(["Gold", "Silver", "Bronze"], p=[0.2, 0.5, 0.3]),
                "zone": zone,
                "city": city,
                "section": section,
                "gps_dummy": "",
                "legacy_code": f"OLD{i:04d}",
            }
        )
    # Named fixtures the tests and briefing can latch onto.
    rows[0].update(
        {
            "store_id": "T0001601407",
            "store_name": "Hameed GS",
            "distributor": "Agha Traders (Quetta)",
            "dsr_name": "ASHRAF KHAN",
            "zone": "West",
            "city": "Quetta",
            "section": "Alamdar Road",
            "category_1": "General Trade",
            "category_2": "B",
        }
    )
    rows[1].update(
        {
            "store_id": "T0001999001",
            "store_name": "Dump Corner Store",
            "distributor": "Agha Traders (Quetta)",
            "dsr_name": "ASHRAF KHAN",
            "zone": "West",
            "city": "Quetta",
            "section": "Alamdar Road",
        }
    )
    rows[2].update(
        {
            "store_id": "T0001999002",
            "store_name": "Lapsed Mart",
            "distributor": "Coastal Foods",
            "dsr_name": "BILAL SHAIKH",
            "zone": "South",
            "city": "Karachi",
            "section": "Korangi",
        }
    )
    # Never-billed whitespace shops.
    for j in range(3, 12):
        rows[j]["store_name"] = f"Unlisted {rows[j]['city']} {j}"
        rows[j]["_never"] = True
    return pd.DataFrame(rows)


def _season_factor(month: int) -> float:
    # Edible oil: winter + Ramadan-ish spring bump (approx).
    table = {1: 1.15, 2: 1.10, 3: 1.20, 4: 1.12, 5: 0.95, 6: 0.90, 7: 0.92, 8: 0.95, 9: 1.00, 10: 1.05, 11: 1.12, 12: 1.18}
    return table[month]


def simulate_sales(stores: pd.DataFrame, start: str = "2024-01", end: str = "2026-07", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start, end, freq="M")
    if "_never" in stores.columns:
        never = set(stores.loc[stores["_never"].fillna(False), "store_id"])
    else:
        never = set()
    # Shop latent size ~ lognormal (Pareto-ish).
    sizes = rng.lognormal(mean=-2.2, sigma=0.85, size=len(stores))
    size_map = dict(zip(stores["store_id"], sizes))
    # Make Hameed GS small-regular; Dump Corner tiny then spiked; Lapsed previously healthy.
    size_map["T0001601407"] = 0.04
    size_map["T0001999001"] = 0.05
    size_map["T0001999002"] = 0.22

    # DSR skill
    dsr_mult = {d: float(rng.uniform(0.75, 1.25)) for d in stores["dsr_name"].unique()}
    dsr_mult["ASHRAF KHAN"] = 0.88
    dsr_mult["NADIA RIZVI"] = 1.22
    dsr_mult["USMAN RIAZ"] = 1.15

    facts = []
    last_period = str(periods[-1])
    for p in periods:
        year, month = p.year, p.month
        period = period_key(year, month)
        months_from_end = (periods[-1] - p).n
        for _, shop in stores.iterrows():
            sid = shop["store_id"]
            if sid in never:
                continue
            base = size_map[sid] * dsr_mult.get(shop["dsr_name"], 1.0) * _season_factor(month)
            # Slow growth with noise
            base *= 1.0 + 0.004 * (p - periods[0]).n
            base *= float(rng.lognormal(0, 0.18))
            # Alamdar Road local failure in last 4 months while Quetta still OK.
            if shop["section"] == "Alamdar Road" and months_from_end <= 3 and sid not in {"T0001999001"}:
                base *= 0.55
            # Lapsed mart: last 4 months zero after being regular.
            if sid == "T0001999002" and months_from_end <= 3:
                continue
            # Dump corner: tiny then last month huge trade load.
            if sid == "T0001999001":
                if period == last_period:
                    base = 2.55
                else:
                    base = float(rng.uniform(0.03, 0.07))
            # Random skip (not billed this month)
            skip_p = 0.18 if shop["category_2"] in {"C", "D"} else 0.08
            if sid in {"T0001601407", "T0001999001"}:
                skip_p = 0.02
            if rng.random() < skip_p and sid != "T0001999001":
                continue
            # SKU mix
            share = SKU_SHARE.copy()
            if months_from_end <= 2:
                # 5L oil eats 16kg tin
                share[4] += 0.10  # 5L
                share[2] -= 0.10  # 16kg tin
                share = np.clip(share, 0.01, None)
                share = share / share.sum()
            n_skus = int(rng.integers(2, 6))
            picks = rng.choice(len(SKUS), size=n_skus, replace=False, p=share)
            remaining = base
            for j, idx in enumerate(picks):
                if j == len(picks) - 1:
                    vol = remaining
                else:
                    piece = float(share[idx] / share[list(picks)].sum()) * base * float(rng.uniform(0.7, 1.3))
                    vol = min(piece, remaining * 0.9)
                    remaining -= vol
                if vol < 0.002:
                    continue
                facts.append(
                    {
                        "distributor": shop["distributor"],
                        "dsr_name": shop["dsr_name"],
                        "section": shop["section"],
                        "store_id": sid,
                        "store_name": shop["store_name"],
                        "sku": SKUS[int(idx)],
                        "year": year,
                        "month": month,
                        "period": period,
                        "volume_mt": round(float(vol), 4),
                    }
                )
    return pd.DataFrame(facts)


def write_shop_master(stores: pd.DataFrame, path: Path) -> Path:
    cols = [
        "distributor",
        "dsr_name",
        "store_id",
        "store_name",
        "category_1",
        "category_2",
        "category_3",
        "category_4",
        "zone",
        "city",
        "section",
        "gps_dummy",
        "legacy_code",
    ]
    out = stores.copy()
    if "_never" in out.columns:
        out = out.drop(columns=["_never"])
    out[cols].to_excel(path, index=False)
    return path


def write_ssrs_sales(sales: pd.DataFrame, path: Path) -> Path:
    """Reproduce the Shop SKU Wise Execution Report chrome the parser must survive."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Execution"

    # Row 1: textbox names
    ws.append(["txtRectangle", "txtFooter_Di", "txtFooter_Culture"] + [None] * 15)
    # Row 2: parameters
    ws.append(
        [
            "Shop SKU Wise Execution Report",
            "User ID: shah",
            "Region: All",
            "Area: All",
            "Territory: All",
            "Town: All",
            "UOM: Tons",
            "Culture: English",
        ]
    )
    ws.append([])
    # Row 4: field ids — real table starts at column G (index 7)
    field_ids = [
        None,
        None,
        None,
        None,
        None,
        None,
        "txt_cDISTRIB",
        "txt_cDSR_NA",
        "txt_cSECTIO",
        "txt_cPOP_Co",
        "txt_cPOP_NA",
        "txt_cSKU_LO",
        "txt_Calendar",
        "txt_Calendar",
        "uval_MTD_Se",
        "txtShopTotal",
        "uvalShopTotal",
        "txtSectionTotal",
        "uvalSectionTotal",
        "txtDSRTotal",
        "uvalDSRTotal",
        "txtDistTotal",
        "uvalDistTotal",
        "txtGrandTotal",
        "uvalGrandTotal",
    ]
    ws.append(field_ids)
    human = [
        None,
        None,
        None,
        None,
        None,
        None,
        "DISTRIBUTOR",
        "DSR NAME",
        "SECTION LON",
        "POP Code",
        "POP NAME",
        "SKU LONG DI",
        "Year",
        "Month",
        "MTD Sales",
        "Shop Total",
        None,
        "Section Total",
        None,
        "DSR Total",
        None,
        "Distributor Total",
        None,
        "Grand Total",
        None,
    ]
    ws.append(human)

    month_name = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    shop_tot = sales.groupby(["store_id", "period"])["volume_mt"].transform("sum")
    sec_tot = sales.groupby(["section", "period"])["volume_mt"].transform("sum")
    dsr_tot = sales.groupby(["dsr_name", "period"])["volume_mt"].transform("sum")
    dist_tot = sales.groupby(["distributor", "period"])["volume_mt"].transform("sum")
    grand = float(sales["volume_mt"].sum())
    labels_left = ["DISTRIBUTOR", "DSR NAME", "SECTION LON", "POP Code", "POP NAME", "SKU LONG DI"]

    for i, row in enumerate(sales.itertuples(index=False)):
        rec = row._asdict() if hasattr(row, "_asdict") else None
        if rec is None:
            data = list(row)
            rec = dict(
                zip(
                    [
                        "distributor",
                        "dsr_name",
                        "section",
                        "store_id",
                        "store_name",
                        "sku",
                        "year",
                        "month",
                        "period",
                        "volume_mt",
                    ],
                    data,
                )
            )
        ws.append(
            labels_left
            + [
                rec["distributor"],
                rec["dsr_name"],
                rec["section"],
                rec["store_id"],
                rec["store_name"],
                rec["sku"],
                rec["year"],
                month_name[int(rec["month"])],
                rec["volume_mt"],
                f"{rec['store_name']} Total",
                float(shop_tot.iloc[i]),
                f"{rec['section']} Total",
                float(sec_tot.iloc[i]),
                f"{rec['dsr_name']} Total",
                float(dsr_tot.iloc[i]),
                f"{rec['distributor']} Total",
                float(dist_tot.iloc[i]),
                "Grand Total",
                grand,
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def generate_demo_files(
    out_dir: Path | None = None,
    n_shops: int = 140,
    seed: int = 7,
    start: str = "2024-01",
    end: str = "2026-07",
) -> dict:
    ensure_dirs()
    out_dir = Path(out_dir or SAMPLE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stores = build_universe(n_shops=n_shops, seed=seed)
    sales = simulate_sales(stores, start=start, end=end, seed=seed)
    shop_path = out_dir / "shop_master.xlsx"
    sales_path = out_dir / "Shop_SKU_Wise_Execution_Report.xlsx"
    csv_path = out_dir / "Shop_SKU_Wise_Execution_Report.csv"
    write_shop_master(stores, shop_path)
    write_ssrs_sales(sales, sales_path)
    # CSV of the same workbook-like grid for drop-folder tests.
    raw = pd.read_excel(sales_path, header=None)
    raw.to_csv(csv_path, index=False, header=False)
    clean_path = out_dir / "sales_clean_reference.csv"
    sales.to_csv(clean_path, index=False)
    return {
        "shops": shop_path,
        "sales": sales_path,
        "sales_csv": csv_path,
        "clean_reference": clean_path,
        "n_sales_rows": len(sales),
        "n_shops": len(stores),
    }
