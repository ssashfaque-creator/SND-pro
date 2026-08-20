"""One strategy pack: country → lagging cities → people → doors, plus full lagging lists.

Built for a sales head to print or filter. Column names are in English; jargon lives
in the glossary, not the headers. Excel is the working file. HTML is the print-to-PDF
board pack (open it, File → Print → Save as PDF).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.worksheet.worksheet import Worksheet

from sndintel.hierarchy import _shop_gaps
from sndintel.isolate import empirical_bayes, k_from_ly
from sndintel.io_utils import shift_period


SITUATION_LABEL = {
    "lagging": "Lagging",
    "with_market": "With the country",
    "outperforming": "Ahead",
}

DRIVER_LABEL = {
    "drop_size": "Drop size — same doors, smaller drops",
    "coverage": "Coverage — fewer billed doors",
    "whitespace": "Whitespace — universe not billed",
    "mixed": "Mixed — more than one driver",
    "holding": "Holding",
    "mix": "SKU mix",
}

GLOSSARY = [
    ("Billed this period", "Secondary volume in the month being scored (MTD if the month is still open)."),
    ("AMS last 3 months", "Average monthly secondary volume over the last three *closed* months. A typical recent month, not last year."),
    ("vs AMS", "This period minus AMS × fraction of the month elapsed. Negative = behind the recent run-rate."),
    ("Same month last year", "What this unit billed in the same calendar month a year ago (full closed month)."),
    ("Expected this month", "Warehouse-learned typical same calendar month (every August on file, not last year alone), paced if MTD is open."),
    ("Fair share of country / city", "This unit’s last-year mix × what the parent billed now. The volume it would have if it only moved with its parent."),
    ("Extra vs country / city", "Billed minus fair share. Negative = worse than the parent after national (or city) weather. This is the local problem."),
    ("Recoverable", "The extra hole expressed as a positive number — volume that could come back if this unit merely matched its parent. max(0, −extra)."),
    ("Shop lists (visit-worthy)", "A door is listed only if it is a real account (size ≥ 0.5 MT AMS/last year, or the 80th percentile of shops if that is higher) AND the hole is worth a call (recoverable ≥ 0.25 MT and at least 8% of that shop’s size). Smaller doors are rolled into one remainder line — coverage, not must-visit."),
    ("Situation: Lagging", "Worse than the parent’s current book. A city can be down with the country and *not* lagging."),
    ("Situation: With the country", "Moved in line with the parent. Weather, not a local fire."),
    ("Situation: Ahead", "Better than the parent’s current book."),
    ("Main driver", "The identity that explains most of the hole: drop size, coverage (doors), whitespace, or mix."),
    ("Strike %", "Billed shops ÷ universe shops on the master list."),
]


CITY_VIEW = [
    ("grain_id", "City"),
    ("zone", "Zone"),
    ("volume_mt", "Billed this period (MT)"),
    ("ams_3m", "AMS last 3 months (MT)"),
    ("vs_ams_mt", "vs AMS (MT)"),
    ("ly_mt", "Same month last year (MT)"),
    ("expected_mt", "Expected this month (MT)"),
    ("share_expected_mt", "Fair share of country (MT)"),
    ("isolated_mt", "Extra vs country (MT)"),
    ("recoverable_mt", "Recoverable (MT)"),
    ("situation_label", "Situation"),
    ("driver_label", "Main driver"),
    ("billed", "Billed shops"),
    ("universe", "Universe"),
    ("strike_pct", "Strike %"),
]

DIST_IN_CITY_VIEW = [
    ("city", "City"),
    ("grain_id", "Distributor"),
    ("volume_mt", "Billed this period (MT)"),
    ("ams_3m", "AMS last 3 months (MT)"),
    ("vs_ams_mt", "vs AMS (MT)"),
    ("ly_mt", "Same month last year (MT)"),
    ("share_expected_mt", "Fair share of this city (MT)"),
    ("isolated_mt", "Extra vs city (MT)"),
    ("recoverable_mt", "Recoverable (MT)"),
    ("situation_label", "Situation"),
    ("driver_label", "Main driver"),
    ("billed", "Billed shops"),
]

DIST_ALL_VIEW = [
    ("grain_id", "Distributor"),
    ("city", "City"),
    ("zone", "Zone"),
    ("volume_mt", "Billed this period (MT)"),
    ("ams_3m", "AMS last 3 months (MT)"),
    ("vs_ams_mt", "vs AMS (MT)"),
    ("ly_mt", "Same month last year (MT)"),
    ("share_expected_mt", "Fair share of its city (MT)"),
    ("isolated_mt", "Extra vs city (MT)"),
    ("recoverable_mt", "Recoverable (MT)"),
    ("situation_label", "Situation"),
    ("driver_label", "Main driver"),
    ("billed", "Billed shops"),
    ("do_this_week", "What to do"),
]

DSR_VIEW = [
    ("grain_id", "DSR"),
    ("city", "City"),
    ("volume_mt", "Billed this period (MT)"),
    ("ams_3m", "AMS last 3 months (MT)"),
    ("vs_ams_mt", "vs AMS (MT)"),
    ("ly_mt", "Same month last year (MT)"),
    ("share_expected_mt", "Fair share of its city (MT)"),
    ("isolated_mt", "Extra vs city (MT)"),
    ("recoverable_mt", "Recoverable (MT)"),
    ("situation_label", "Situation"),
    ("driver_label", "Main driver"),
    ("billed", "Billed shops"),
    ("do_this_week", "What to do"),
]

SHOP_VIEW = [
    ("store_name", "Shop"),
    ("city", "City"),
    ("distributor", "Distributor"),
    ("dsr_name", "DSR"),
    ("section", "Beat"),
    ("volume_mt", "Billed this period (MT)"),
    ("ams_3m", "AMS last 3 months (MT)"),
    ("vs_ams_mt", "vs AMS (MT)"),
    ("ly_mt", "Same month last year (MT)"),
    ("fair_share_mt", "Fair share of its city (MT)"),
    ("isolated_mt", "Extra vs city (MT)"),
    ("recoverable_mt", "Recoverable (MT)"),
]


@dataclass
class StrategyPack:
    period: str
    label: str
    headline: str = ""
    weather: str = ""
    problem: str = ""
    action: str = ""
    kpis: dict[str, Any] = field(default_factory=dict)
    cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    city_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    city_distributor_shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_dsrs: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_city_names: list[str] = field(default_factory=list)
    lagging_distributor_names: list[str] = field(default_factory=list)
    shop_note: str = ""


def build_strategy_pack(
    units: pd.DataFrame,
    shop_month: pd.DataFrame,
    situation: pd.DataFrame | None = None,
    ledger: pd.DataFrame | None = None,
    national: dict[str, Any] | None = None,
    period: str | None = None,
) -> StrategyPack:
    empty = StrategyPack(period=period or "", label=period or "")
    if units is None or units.empty:
        return empty
    period = period or str(units["period"].dropna().astype(str).max() or "")
    nat_u = units[units["grain"] == "national"]
    nat_row = nat_u.iloc[0] if not nat_u.empty else None
    pace = 1.0
    if nat_row is not None and pd.notna(nat_row.get("intra_month_frac")):
        pace = float(nat_row["intra_month_frac"] or 1.0) or 1.0
    elif "intra_month_frac" in units.columns:
        pace = float(pd.to_numeric(units["intra_month_frac"], errors="coerce").dropna().max() or 1.0)

    sit = _situation_row(situation, national, nat_row)
    label = sit.get("label") or period
    kpis = _kpis(nat_row, units, sit)

    cities = _grain(units, "city")
    dists = _grain(units, "distributor")
    dsrs = _grain(units, "dsr")
    if not cities.empty:
        cities["city"] = cities["grain_id"]
        cities = _attach_ams(cities, shop_month, period, ["city"], ledger, pace)
    if not dists.empty:
        dists["city"] = dists["parent_id"]
        dists["distributor"] = dists["grain_id"]
        dists = _attach_ams(dists, shop_month, period, ["city", "distributor"], ledger, pace)
    if not dsrs.empty:
        dsrs["city"] = dsrs["parent_id"]
        dsrs["dsr_name"] = dsrs["grain_id"]
        dsrs = _attach_ams(dsrs, shop_month, period, ["city", "dsr_name"], ledger, pace)

    lagging_cities = cities[cities["situation"] == "lagging"] if not cities.empty else cities
    lagging_city_names = [str(x) for x in lagging_cities["grain_id"].tolist()] if not lagging_cities.empty else []

    city_dists = pd.DataFrame()
    if not dists.empty and lagging_city_names:
        city_dists = dists[dists["parent_id"].astype(str).isin(lagging_city_names)].copy()
        city_dists = city_dists[city_dists["situation"] == "lagging"] if "situation" in city_dists.columns else city_dists
        city_dists = _sort_focus(city_dists)

    lagging_dist_names: list[str] = []
    if not city_dists.empty:
        lagging_dist_names = sorted({str(x) for x in city_dists["grain_id"].tolist()})

    shops = score_shops(shop_month, cities, period, pace)
    shops = _attach_ams(shops, shop_month, period, ["store_id"], ledger, pace)
    floors = visit_shop_floors(shop_month, period)

    city_dist_shops = pd.DataFrame()
    city_dist_meta = {"n_hidden": 0, "hidden_mt": 0.0}
    if not shops.empty and lagging_dist_names:
        city_dist_shops = shops[
            shops["distributor"].astype(str).isin(lagging_dist_names)
            & shops["city"].astype(str).isin(lagging_city_names)
        ].copy()
        city_dist_shops, city_dist_meta = keep_visit_shops(city_dist_shops, floors, max_rows=50)

    all_lag_dist = dists[dists["situation"] == "lagging"].copy() if not dists.empty else dists
    all_lag_dist = _sort_focus(all_lag_dist)
    all_lag_dsr = dsrs[dsrs["situation"] == "lagging"].copy() if not dsrs.empty else dsrs
    all_lag_dsr = _sort_focus(all_lag_dsr)

    lag_shops = pd.DataFrame()
    lag_meta = {"n_hidden": 0, "hidden_mt": 0.0}
    if not shops.empty:
        lag_shops, lag_meta = keep_visit_shops(shops, floors, max_rows=75)

    kpis["n_lagging_distributors"] = int(len(all_lag_dist)) if all_lag_dist is not None else 0
    kpis["n_lagging_dsrs"] = int(len(all_lag_dsr)) if all_lag_dsr is not None else 0
    kpis["n_lagging_shops"] = int(lag_meta.get("n_kept") or 0)
    kpis["shop_size_floor_mt"] = floors["size_floor"]
    kpis["shop_hole_floor_mt"] = floors["hole_floor"]
    kpis["n_shops_hidden"] = int(lag_meta.get("n_hidden") or 0)
    kpis["hidden_shop_recoverable_mt"] = float(lag_meta.get("hidden_mt") or 0)
    shop_note = (
        f"Visit-worthy shops only: size ≥ {floors['size_floor']:.2f} MT (AMS or last year) and "
        f"recoverable ≥ {floors['hole_floor']:.2f} MT (and ≥ 8% of that shop’s size). "
        f"{int(lag_meta.get('n_hidden') or 0)} smaller doors totalling "
        f"{float(lag_meta.get('hidden_mt') or 0):.1f} MT recoverable are not listed."
    )

    return StrategyPack(
        period=period,
        label=label,
        headline=sit.get("headline") or "",
        weather=sit.get("weather") or "",
        problem=sit.get("problem") or "",
        action=sit.get("action") or sit.get("action_summary") or "",
        kpis=kpis,
        cities=_present(cities, CITY_VIEW),
        city_distributors=_present(city_dists, DIST_IN_CITY_VIEW),
        city_distributor_shops=_present_shops(city_dist_shops, city_dist_meta),
        lagging_distributors=_present(all_lag_dist, DIST_ALL_VIEW),
        lagging_dsrs=_present(all_lag_dsr.head(250), DSR_VIEW),
        lagging_shops=_present_shops(lag_shops, lag_meta),
        lagging_city_names=lagging_city_names,
        lagging_distributor_names=lagging_dist_names,
        shop_note=shop_note,
    )


def score_shops(shop_month: pd.DataFrame, cities: pd.DataFrame, period: str, pace: float) -> pd.DataFrame:
    """Shop extra vs its city's current book × last-year mix. Recoverable = the extra hole."""
    if shop_month is None or shop_month.empty or not period:
        return pd.DataFrame()
    yoy = shift_period(period, -12)
    cur = shop_month[shop_month["period"] == period]
    ly = shop_month[shop_month["period"] == yoy]
    gaps = _shop_gaps(cur, ly, pace)
    if gaps is None or gaps.empty:
        return pd.DataFrame()
    idx_map = {}
    if cities is not None and not cities.empty:
        for _, r in cities.iterrows():
            ly_v = float(r.get("ly_mt") or 0)
            now_v = float(r.get("volume_mt") or 0)
            idx_map[str(r.get("grain_id") or r.get("city") or "")] = (now_v / ly_v) if ly_v > 1e-9 else 1.0
    gaps["parent_index"] = gaps["city"].astype(str).map(idx_map).fillna(1.0)
    gaps["fair_share_mt"] = gaps["ly_mt"].fillna(0) * gaps["parent_index"]
    gaps["competitive_mt"] = gaps["volume_mt"].fillna(0) - gaps["fair_share_mt"]
    k_shop = k_from_ly(gaps["ly_mt"], 0.05)
    gaps["isolated_mt"] = [
        empirical_bayes(c, ly_v, k_shop) for c, ly_v in zip(gaps["competitive_mt"], gaps["ly_mt"])
    ]
    gaps["recoverable_mt"] = gaps["isolated_mt"].clip(upper=0).abs()
    gaps["store_name"] = gaps["store_name"].replace("", pd.NA).fillna(gaps["store_id"])
    return gaps.loc[gaps["recoverable_mt"] > 0].copy()


def visit_shop_floors(shop_month: pd.DataFrame, period: str) -> dict[str, float]:
    """Size a must-visit door from the warehouse, not a kiryana tail.

    Median billed shop on this book is often ~0.02 MT. A ride-with only pays if the
    door is a real account (~0.5 MT, or the 80th percentile of shops if the book is
    larger) and the hole is at least 0.25 MT *and* 8% of that shop’s own size.
    """
    size_floor = 0.50
    hole_floor = 0.25
    if shop_month is None or shop_month.empty or not period:
        return {"size_floor": size_floor, "hole_floor": hole_floor, "rel": 0.08}
    yoy = shift_period(period, -12)
    parts = []
    for per in (period, yoy):
        part = shop_month[shop_month["period"].astype(str) == str(per)]
        if not part.empty:
            parts.append(part.groupby("store_id")["volume_mt"].sum())
    if not parts:
        return {"size_floor": size_floor, "hole_floor": hole_floor, "rel": 0.08}
    size = pd.concat(parts, axis=1).max(axis=1)
    size = size[size > 0]
    if len(size) >= 40:
        p80 = float(size.quantile(0.80))
        size_floor = float(min(max(0.50, p80), 2.00))
    return {"size_floor": size_floor, "hole_floor": hole_floor, "rel": 0.08}


def _account_size(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(df["ams_3m"], errors="coerce") if "ams_3m" in df.columns else pd.Series(0.0, index=df.index),
            pd.to_numeric(df["ly_mt"], errors="coerce") if "ly_mt" in df.columns else pd.Series(0.0, index=df.index),
            pd.to_numeric(df["volume_mt"], errors="coerce") if "volume_mt" in df.columns else pd.Series(0.0, index=df.index),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)


def keep_visit_shops(
    shops: pd.DataFrame,
    floors: dict[str, float],
    max_rows: int = 75,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Keep real accounts with a hole worth a visit; roll the rest into a remainder."""
    empty_meta = {"n_kept": 0, "n_hidden": 0, "hidden_mt": 0.0, "n_pool": 0}
    if shops is None or shops.empty:
        return shops if shops is not None else pd.DataFrame(), empty_meta
    out = shops.copy()
    rec = pd.to_numeric(out.get("recoverable_mt"), errors="coerce").fillna(0.0)
    pool = out.loc[rec > 0].copy()
    rec = rec.loc[pool.index]
    size = _account_size(pool)
    size_floor = float(floors.get("size_floor") or 0.50)
    hole_floor = float(floors.get("hole_floor") or 0.25)
    rel = float(floors.get("rel") or 0.08)
    need = pd.concat([pd.Series(hole_floor, index=pool.index), size * rel], axis=1).max(axis=1)
    keep = (size >= size_floor) & (rec >= need)
    kept = pool.loc[keep].copy()
    hidden = pool.loc[~keep]
    # Among visit-worthy doors, keep those that make 90% of that recoverable so
    # a long list of similar medium shops does not bury the briefing.
    kept = _sort_focus(kept)
    if not kept.empty and len(kept) > 8:
        total = float(kept["recoverable_mt"].sum()) or 1.0
        cum = kept["recoverable_mt"].cumsum() / total
        keep_p = cum.shift(1, fill_value=0) < 0.90
        extra = kept.loc[~keep_p]
        if not extra.empty:
            hidden = pd.concat([hidden, extra], ignore_index=True)
        kept = kept.loc[keep_p]
    if len(kept) > max_rows:
        extra = kept.iloc[max_rows:]
        hidden = pd.concat([hidden, extra], ignore_index=True)
        kept = kept.iloc[:max_rows]
    meta = {
        "n_kept": int(len(kept)),
        "n_hidden": int(len(hidden)),
        "hidden_mt": float(pd.to_numeric(hidden.get("recoverable_mt"), errors="coerce").fillna(0).sum()) if not hidden.empty else 0.0,
        "n_pool": int(len(pool)),
        "size_floor": size_floor,
        "hole_floor": hole_floor,
    }
    return kept, meta


def _present_shops(df: pd.DataFrame, meta: dict[str, float]) -> pd.DataFrame:
    table = _present(df, SHOP_VIEW)
    n_hidden = int(meta.get("n_hidden") or 0)
    hidden_mt = float(meta.get("hidden_mt") or 0)
    if n_hidden <= 0:
        return table
    rest = {label: None for _, label in SHOP_VIEW}
    rest["Shop"] = (
        f"Not listed — {n_hidden} smaller / shallower doors "
        f"({hidden_mt:.1f} MT recoverable). Coverage KPI, not must-visit."
    )
    rest["Recoverable (MT)"] = hidden_mt
    return pd.concat([table, pd.DataFrame([rest])], ignore_index=True)


def ams_last_n(
    shop_month: pd.DataFrame,
    period: str,
    keys: list[str],
    n: int = 3,
    ledger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mean volume of the last n closed months, excluding the period being scored."""
    cols = keys + ["ams_3m"]
    if shop_month is None or shop_month.empty or not period:
        return pd.DataFrame(columns=cols)
    hist = shop_month[shop_month["period"].astype(str) != str(period)].copy()
    if hist.empty:
        return pd.DataFrame(columns=cols)
    if ledger is not None and not ledger.empty and "status" in ledger.columns:
        closed = set(ledger.loc[ledger["status"].astype(str) == "closed", "period"].astype(str))
        if closed:
            hist = hist[hist["period"].astype(str).isin(closed)]
    if hist.empty:
        return pd.DataFrame(columns=cols)
    for k in keys:
        if k not in hist.columns:
            hist[k] = "(unmapped)"
        hist[k] = hist[k].fillna("(unmapped)").replace("", "(unmapped)")
    periods = sorted(hist["period"].astype(str).unique())[-n:]
    hist = hist[hist["period"].astype(str).isin(periods)]
    g = hist.groupby(keys + ["period"], as_index=False)["volume_mt"].sum()
    out = g.groupby(keys, as_index=False)["volume_mt"].mean().rename(columns={"volume_mt": "ams_3m"})
    return out


def excel_bytes(pack: StrategyPack) -> bytes:
    buf = BytesIO()
    write_excel(pack, buf)
    return buf.getvalue()


def html_bytes(pack: StrategyPack) -> bytes:
    return render_html(pack).encode("utf-8")


def write_excel(pack: StrategyPack, path: Path | str | BytesIO) -> None:
    wb = Workbook()
    _sheet_cover(wb, pack)
    _sheet_table(
        wb, "01 Country by city",
        "Every city versus the country",
        "Negative Extra vs country = worse than the national book after weather. Recoverable is that hole as a positive number.",
        pack.cities, freeze="A2",
        bar_col="Extra vs country (MT)", cat_col="City",
    )
    _sheet_table(
        wb, "02 Lagging cities-dists",
        "Distributors inside lagging cities",
        "Only cities on the lagging list. A distributor here is behind its city — that is who to call first.",
        pack.city_distributors,
    )
    _sheet_table(
        wb, "03 Those dists-shops",
        "Lagging shops under those distributors",
        (pack.shop_note or "Visit-worthy doors only.")
        + " Recoverable is volume that comes back if the door matches the city.",
        pack.city_distributor_shops,
    )
    _sheet_table(
        wb, "04 All lagging distributors",
        "Every lagging distributor (all cities)",
        "Includes distributors that are behind a city even when the city itself moved with the country. Sheet 02 only showed distributors in lagging cities.",
        pack.lagging_distributors,
    )
    _sheet_table(
        wb, "05 All lagging DSRs",
        "Every lagging DSR (all cities)",
        "Salespeople behind their city. Ride-with this list; do not build a city hit-list from national weather.",
        pack.lagging_dsrs,
    )
    _sheet_table(
        wb, "06 All lagging shops",
        "Every lagging shop worth a visit",
        pack.shop_note or "Visit-worthy accounts only. Smaller doors are the remainder line.",
        pack.lagging_shops,
    )
    # First default sheet was cover
    if path is not None:
        wb.save(path)


def render_html(pack: StrategyPack) -> str:
    k = pack.kpis
    sections = [
        _html_cover(pack, k),
        _html_section("1. The country — every city", "Negative Extra vs country is the local problem. Recoverable is that hole as a positive number.", pack.cities),
        _html_section(
            "2. Lagging cities — distributors",
            "Cities on the lagging list, broken by distributor. These are the first calls.",
            pack.city_distributors,
        ),
        _html_section(
            "3. Those distributors — lagging shops",
            "Doors behind their city, only under the distributors above. Visit-worthy accounts only; remainder line is the tail.",
            pack.city_distributor_shops,
        ),
        _html_section(
            "4. Every lagging distributor (all cities)",
            "Distributors behind their city even when the city is not a national exception.",
            pack.lagging_distributors,
        ),
        _html_section(
            "5. Every lagging DSR (all cities)",
            "Salespeople behind their city.",
            pack.lagging_dsrs,
        ),
        _html_section(
            "6. Every lagging shop worth a visit",
            "Visit-worthy doors behind their city. Smaller shops are rolled into the last row.",
            pack.lagging_shops,
        ),
        _html_glossary(),
    ]
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>SND strategy pack · {html.escape(pack.label)}</title>
<style>
  :root {{ --ink:#0f172a; --muted:#475569; --line:#cbd5e1; --red:#b91c1c; --green:#15803d; --wash:#f8fafc; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: var(--ink); margin: 24px; background: white; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; margin: 28px 0 6px; page-break-after: avoid; }}
  .kicker {{ letter-spacing: .12em; text-transform: uppercase; font-size: 11px; font-weight: 700; color: var(--muted); }}
  .headline {{ font-size: 20px; font-weight: 700; margin: 8px 0 10px; }}
  .lead {{ color: var(--muted); line-height: 1.45; margin: 0 0 8px; max-width: 920px; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 8px; }}
  .kpi {{ background: var(--wash); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; min-width: 120px; }}
  .kpi b {{ display: block; font-size: 18px; }}
  .kpi span {{ font-size: 11px; color: var(--muted); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0 4px; }}
  th {{ background: var(--ink); color: white; text-align: left; padding: 6px 8px; font-weight: 600; }}
  td {{ border-bottom: 1px solid var(--line); padding: 5px 8px; vertical-align: top; }}
  tr.lagging td {{ background: #fef2f2; }}
  tr.ahead td {{ background: #f0fdf4; }}
  .note {{ font-size: 12px; color: var(--muted); margin: 0 0 8px; }}
  dl.glossary {{ font-size: 12px; line-height: 1.4; }}
  dl.glossary dt {{ font-weight: 700; margin-top: 8px; }}
  dl.glossary dd {{ margin: 2px 0 0 0; color: var(--muted); }}
  @page {{ size: A4 landscape; margin: 12mm; }}
  @media print {{
    body {{ margin: 0; }}
    section {{ page-break-before: always; }}
    section.cover {{ page-break-before: auto; }}
    h2 {{ page-break-after: avoid; }}
    tr {{ page-break-inside: avoid; }}
  }}
</style></head><body>
{''.join(sections)}
</body></html>
"""


# ----- internals -----

def _grain(units: pd.DataFrame, grain: str) -> pd.DataFrame:
    out = units[units["grain"] == grain].copy()
    if out.empty:
        return out
    out["recoverable_mt"] = pd.to_numeric(out.get("isolated_mt"), errors="coerce").fillna(0).clip(upper=0).abs()
    sit = out["situation"] if "situation" in out.columns else pd.Series("", index=out.index)
    out["situation_label"] = sit.map(SITUATION_LABEL).fillna(sit)
    diag = out["diagnosis"] if "diagnosis" in out.columns else pd.Series("", index=out.index)
    out["driver_label"] = diag.map(DRIVER_LABEL).fillna(diag.astype(str).str.replace("_", " "))
    strike = pd.to_numeric(out.get("strike_rate"), errors="coerce")
    out["strike_pct"] = strike * 100
    return out


def _attach_ams(
    df: pd.DataFrame,
    shop_month: pd.DataFrame,
    period: str,
    keys: list[str],
    ledger: pd.DataFrame | None,
    pace: float,
) -> pd.DataFrame:
    out = df.copy()
    ams = ams_last_n(shop_month, period, keys, n=3, ledger=ledger)
    if ams.empty:
        out["ams_3m"] = pd.NA
        out["vs_ams_mt"] = pd.NA
        return out
    left = out.copy()
    for k in keys:
        if k not in left.columns:
            left[k] = "(unmapped)"
        left[k] = left[k].fillna("(unmapped)").astype(str).replace("", "(unmapped)")
    right = ams.copy()
    for k in keys:
        right[k] = right[k].fillna("(unmapped)").astype(str)
    left["_row"] = range(len(left))
    merged = left.merge(right, on=keys, how="left")
    merged = merged.sort_values("_row").drop(columns=["_row"])
    ams_v = pd.to_numeric(merged["ams_3m"], errors="coerce")
    vol = pd.to_numeric(merged.get("volume_mt"), errors="coerce")
    merged["vs_ams_mt"] = vol - ams_v * float(pace or 1.0)
    return merged


def _sort_focus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "recoverable_mt" in out.columns:
        return out.sort_values(["recoverable_mt", "isolated_mt"], ascending=[False, True])
    if "isolated_mt" in out.columns:
        return out.sort_values("isolated_mt")
    return out


def _present(df: pd.DataFrame, view: list[tuple[str, str]]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[label for _, label in view])
    cols = []
    data = {}
    for src, label in view:
        cols.append(label)
        data[label] = df[src] if src in df.columns else pd.NA
    out = pd.DataFrame(data)[cols]
    return out.reset_index(drop=True)


def _situation_row(situation: pd.DataFrame | None, national: dict | None, nat_row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if national:
        out.update(national)
    if nat_row is not None:
        out.setdefault("label", None)
    if situation is not None and not situation.empty:
        row = situation.iloc[0]
        out["headline"] = row.get("headline") or out.get("headline")
        out["weather"] = row.get("weather") or out.get("weather")
        out["problem"] = row.get("problem") or out.get("problem")
        out["action"] = row.get("action_summary") or out.get("action")
        try:
            import json as _json
            meta = _json.loads(row["metrics_json"]) if row.get("metrics_json") else {}
            out.update({k: v for k, v in meta.items() if v is not None})
        except Exception:
            pass
    return out


def _kpis(nat_row, units: pd.DataFrame, sit: dict[str, Any]) -> dict[str, Any]:
    cities = units[units["grain"] == "city"] if units is not None else pd.DataFrame()
    n_lag = int((cities["situation"] == "lagging").sum()) if not cities.empty and "situation" in cities.columns else 0
    extra = sit.get("extra_hole_mt")
    if extra is None and not cities.empty and "isolated_mt" in cities.columns:
        extra = float(cities.loc[cities.get("situation", "") == "lagging", "isolated_mt"].sum()) if "situation" in cities.columns else 0.0
    billed = float(nat_row["volume_mt"]) if nat_row is not None else None
    expected = float(nat_row["expected_mt"]) if nat_row is not None else None
    gap = float(nat_row["gap_mt"]) if nat_row is not None else None
    ly = float(nat_row["ly_mt"]) if nat_row is not None else None
    return {
        "billed_mt": billed,
        "expected_mt": expected,
        "gap_mt": gap,
        "ly_mt": ly,
        "extra_hole_mt": float(extra or 0),
        "n_lagging_cities": n_lag,
        "n_history_periods": sit.get("n_history_periods"),
        "n_same_month": sit.get("n_same_month"),
        "intra_month_frac": sit.get("intra_month_frac") or (nat_row.get("intra_month_frac") if nat_row is not None else None),
        "open_mtd": sit.get("open_mtd"),
    }


NAVY = "0F172A"
RED = "B91C1C"
GREEN = "15803D"
SLATE = "64748B"
WASH = "F8FAFC"
WHITE = "FFFFFF"
THIN = Border(
    left=Side(style="thin", color="E2E8F0"),
    right=Side(style="thin", color="E2E8F0"),
    top=Side(style="thin", color="E2E8F0"),
    bottom=Side(style="thin", color="E2E8F0"),
)


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _sheet_cover(wb: Workbook, pack: StrategyPack) -> Worksheet:
    ws = wb.active
    ws.title = "00 Cover"
    ws.sheet_properties.tabColor = NAVY
    ws["A1"] = "SND Intelligence"
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=NAVY)
    ws["A2"] = f"Strategy pack · {pack.label}"
    ws["A2"].font = Font(name="Calibri", size=18, bold=True, color=NAVY)
    ws["A3"] = pack.headline or "Scorecards ready"
    ws["A3"].font = Font(name="Calibri", size=14, bold=True)
    ws.merge_cells("A3:H3")
    ws["A5"] = pack.weather or ""
    ws.merge_cells("A5:H6")
    ws["A5"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A7"] = "The problem"
    ws["A7"].font = Font(bold=True, color=RED)
    ws["A8"] = pack.problem or ""
    ws.merge_cells("A8:H9")
    ws["A8"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A10"] = "Do this week"
    ws["A10"].font = Font(bold=True, color=GREEN)
    ws["A11"] = pack.action or ""
    ws.merge_cells("A11:H12")
    ws["A11"].alignment = Alignment(wrap_text=True, vertical="top")

    k = pack.kpis
    labels = [
        ("Billed (MT)", k.get("billed_mt"), "0.0"),
        ("Expected (MT)", k.get("expected_mt"), "0.0"),
        ("Gap vs expected (MT)", k.get("gap_mt"), "+0.0;-0.0;0.0"),
        ("Extra hole after weather (MT)", k.get("extra_hole_mt"), "+0.0;-0.0;0.0"),
        ("Lagging cities", k.get("n_lagging_cities"), "0"),
    ]
    for i, (lab, val, fmt) in enumerate(labels, start=1):
        cell_l = ws.cell(14, i, lab)
        cell_l.fill = _fill(WASH)
        cell_l.font = Font(size=9, color=SLATE, bold=True)
        cell_v = ws.cell(15, i, val if val is not None else "—")
        cell_v.font = Font(size=14, bold=True, color=NAVY)
        if isinstance(val, (int, float)):
            cell_v.number_format = fmt

    ws["A17"] = "How to read this pack"
    ws["A17"].font = Font(bold=True, size=12)
    steps = [
        "01 Country by city — every city versus national weather. Start here. Red Extra vs country is a local fire; grey is weather.",
        "02 Lagging cities-dists — only the cities that showed up as lagging, broken by distributor. First calls.",
        "03 Those dists-shops — lagging shops under those distributors that are worth a visit. Remainder line is the kiryana tail.",
        "04 All lagging distributors — every distributor behind its own city, including cities that are not national exceptions.",
        "05 All lagging DSRs — every salesperson behind their city.",
        "06 All lagging shops — visit-worthy doors only (size ≥ ~0.5 MT and a hole ≥ 0.25 MT). Smaller doors are one remainder line.",
    ]
    for i, line in enumerate(steps):
        ws.cell(18 + i, 1, line)
        ws.merge_cells(start_row=18 + i, start_column=1, end_row=18 + i, end_column=8)
        ws.cell(18 + i, 1).alignment = Alignment(wrap_text=True)

    ws["A25"] = "Glossary"
    ws["A25"].font = Font(bold=True, size=12)
    ws["A26"] = "Term"
    ws["B26"] = "Meaning"
    ws["A26"].font = Font(bold=True, color=WHITE)
    ws["B26"].font = Font(bold=True, color=WHITE)
    ws["A26"].fill = _fill(NAVY)
    ws["B26"].fill = _fill(NAVY)
    for i, (term, meaning) in enumerate(GLOSSARY, start=27):
        ws.cell(i, 1, term).font = Font(bold=True)
        ws.cell(i, 2, meaning)
        ws.merge_cells(start_row=i, start_column=2, end_row=i, end_column=8)
        ws.cell(i, 2).alignment = Alignment(wrap_text=True)
        ws.row_dimensions[i].height = 28
    ws.column_dimensions["A"].width = 36
    for col in "BCDEFGH":
        ws.column_dimensions[col].width = 18
    ws.row_dimensions[5].height = 48
    ws.row_dimensions[8].height = 48
    ws.row_dimensions[11].height = 48
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = "1:2"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    return ws


def _sheet_table(
    wb: Workbook,
    title: str,
    heading: str,
    note: str,
    df: pd.DataFrame,
    freeze: str = "A5",
    bar_col: str | None = None,
    cat_col: str | None = None,
) -> Worksheet:
    ws = wb.create_sheet(title[:31])
    ws.sheet_properties.tabColor = RED if "lagging" in title.lower() or "dists" in title.lower() or "shops" in title.lower() or "DSR" in title else NAVY
    ws["A1"] = heading
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=NAVY)
    ws.merge_cells("A1:L1")
    ws["A2"] = note
    ws["A2"].font = Font(size=10, italic=True, color=SLATE)
    ws.merge_cells("A2:L2")
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[2].height = 32
    if df is None or df.empty:
        ws["A4"] = "No rows at this layer for this period."
        ws.column_dimensions["A"].width = 40
        return ws
    start = 4
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), start=start):
        for c_idx, value in enumerate(row, start=1):
            cell = ws.cell(r_idx, c_idx, _excel_value(value))
            cell.border = THIN
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            if r_idx == start:
                cell.fill = _fill(NAVY)
                cell.font = Font(bold=True, color=WHITE, size=10)
            else:
                cell.font = Font(size=10)
                _format_metric_cell(cell, df.columns[c_idx - 1] if c_idx - 1 < len(df.columns) else "")
                sit = _row_situation(df, r_idx - start - 1)
                if sit == "Lagging":
                    cell.fill = PatternFill("solid", fgColor="FEF2F2")
                elif sit == "Ahead":
                    cell.fill = PatternFill("solid", fgColor="F0FDF4")
    headers = list(df.columns)
    for i, name in enumerate(headers, start=1):
        width = min(max(len(str(name)) + 2, 12), 28)
        if name in {"What to do", "Shop"}:
            width = 32
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.auto_filter.ref = f"A{start}:{get_column_letter(len(headers))}{start + len(df)}"
    ws.freeze_panes = f"A{start + 1}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.print_title_rows = f"{start}:{start}"
    ws.page_setup.horizontalCentered = True
    if bar_col and cat_col and bar_col in df.columns and cat_col in df.columns and len(df) > 0:
        cat_idx = headers.index(cat_col) + 1
        bar_idx = headers.index(bar_col) + 1
        chart = BarChart()
        chart.type = "bar"
        chart.style = 10
        chart.title = bar_col
        chart.y_axis.title = None
        data = Reference(ws, min_col=bar_idx, min_row=start, max_row=start + len(df))
        cats = Reference(ws, min_col=cat_idx, min_row=start + 1, max_row=start + len(df))
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.shape = 4
        chart.height = 8
        chart.width = 15
        ws.add_chart(chart, "Q4")
    return ws


def _excel_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _excel_value(value.item())
        except Exception:
            return str(value)
    return value


def _format_metric_cell(cell, header: str) -> None:
    h = str(header)
    if "(MT)" in h:
        cell.number_format = "+0.00;-0.00;0.00" if h.startswith("Extra") or h.startswith("vs ") or "Gap" in h else "0.00"
    elif h.endswith("%") or "Strike" in h:
        cell.number_format = "0.0"


def _row_situation(df: pd.DataFrame, idx: int) -> str:
    if "Situation" not in df.columns:
        return ""
    if idx < 0 or idx >= len(df):
        return ""
    val = df.iloc[idx]["Situation"]
    return str(val) if pd.notna(val) else ""


def _html_cover(pack: StrategyPack, k: dict[str, Any]) -> str:
    def fmt(val, spec="{:.1f}"):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "—"
        if isinstance(val, (int, float)):
            return spec.format(val)
        return str(val)

    kpis = [
        ("Billed", fmt(k.get("billed_mt")) + " MT"),
        ("Expected", fmt(k.get("expected_mt")) + " MT"),
        ("Gap vs expected", fmt(k.get("gap_mt"), "{:+.1f}") + " MT"),
        ("Extra hole after weather", fmt(k.get("extra_hole_mt"), "{:+.1f}") + " MT"),
        ("Lagging cities", str(k.get("n_lagging_cities") or 0)),
    ]
    kpi_html = "".join(f'<div class="kpi"><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>' for a, b in kpis)
    steps = [
        "Country by city — every city versus national weather.",
        "Lagging cities → distributors — first calls.",
        "Those distributors → lagging shops — visit-worthy doors only; remainder line is the tail.",
        "Every lagging distributor, including cities that are not national exceptions.",
        "Every lagging DSR.",
        "Every lagging shop worth a visit (kiryana tail rolled into the last row).",
    ]
    ol = "".join(f"<li>{html.escape(s)}</li>" for s in steps)
    return f"""<section class="cover">
  <div class="kicker">SND Intelligence · strategy pack</div>
  <h1>{html.escape(pack.label)}</h1>
  <p class="headline">{html.escape(pack.headline or "Scorecards ready")}</p>
  <p class="lead">{html.escape(pack.weather or "")}</p>
  <p class="lead"><b>The problem.</b> {html.escape(pack.problem or "")}</p>
  <p class="lead"><b>Do this week.</b> {html.escape(pack.action or "")}</p>
  <div class="kpis">{kpi_html}</div>
  <p class="note">How to read this pack</p>
  <ol class="note">{ol}</ol>
</section>
"""


def _html_section(title: str, note: str, df: pd.DataFrame) -> str:
    return f"""<section>
  <h2>{html.escape(title)}</h2>
  <p class="note">{html.escape(note)}</p>
  {_df_html(df)}
</section>
"""


def _html_glossary() -> str:
    items = "".join(
        f"<dt>{html.escape(t)}</dt><dd>{html.escape(d)}</dd>" for t, d in GLOSSARY
    )
    return f"""<section>
  <h2>Glossary</h2>
  <dl class="glossary">{items}</dl>
</section>
"""


def _df_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='note'>No rows at this layer for this period.</p>"
    heads = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    sit_col = "Situation" if "Situation" in df.columns else None
    for _, row in df.iterrows():
        klass = ""
        if sit_col:
            sit = str(row[sit_col] or "")
            if sit == "Lagging":
                klass = "lagging"
            elif sit == "Ahead":
                klass = "ahead"
        tds = "".join(f"<td>{html.escape(_html_cell(row[c], c))}</td>" for c in df.columns)
        body.append(f"<tr class='{klass}'>{tds}</tr>")
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _html_cell(val: Any, col: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (int, float)) and "(MT)" in str(col):
        if str(col).startswith("Extra") or str(col).startswith("vs "):
            return f"{val:+.2f}"
        return f"{val:.2f}"
    if isinstance(val, (int, float)) and "Strike" in str(col):
        return f"{val:.0f}"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def write_html(pack: StrategyPack, path: Path | str) -> None:
    Path(path).write_text(render_html(pack), encoding="utf-8")
