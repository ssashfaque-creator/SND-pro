"""Sendable situation cascade: national HQ pack, then one pack per city / distributor.

The existing strategy PDF is a scorecard dump (glossary first, lagging-only
top-N, no overperformers, no path to potential). This module answers the
operating questions in order:

1. What has billed so far, what will month-end look like, and what is the quota?
2. Who is underperforming, who is overperforming?
3. What are the few next actions?

National pack goes to the sales head. City and distributor packs are the
same skeleton, scoped, so they can be emailed without a covering note.
Narratives are deterministic from scorecards — no API key required.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from sndintel.briefing import DRIVER_LABEL, SITUATION_LABEL, _sheet_table
from sndintel.capacity import (
    LABEL_FINE,
    score_dsr_capacity_from_units,
)
from sndintel.config import EXPECTED_FORMULA
from sndintel.identity import dsr_display_name
from sndintel.mtd import period_state

LAG_N = 12
AHEAD_N = 10
PEOPLE_LAG_N = 15
PEOPLE_AHEAD_N = 10
SECTION_N = 12
DOOR_N = 25
STEP_N = 5
RUN_RATE_FLOOR = 0.05

GLOSSARY = [
    (
        "Expected / potential",
        "Full-month run-rate (last-three-month AMS blended with last-six-month median). On an open month the cover also shows where we should be by today (that run-rate × the national day curve).",
    ),
    (
        "Projected month-end",
        "Open month only. Billed so far ÷ the same national day-curve fraction used for Expected. If we keep this pace, this is month-end billed.",
    ),
    (
        "Monthly target",
        "Sales-team shop-wise quota for the full month. Not paced. Attainment on an open month is projected month-end ÷ this target.",
    ),
    (
        "Lagging / Ahead / On expected",
        "Behind its own Expected, ahead of it, or on it. A city that declined with the country is still lagging if it missed its own run-rate.",
    ),
    (
        "Driver",
        "Why the hole exists: drop size (same doors, smaller orders), coverage (fewer billed doors), whitespace (universe never billed), or mixed.",
    ),
    (
        "Capacity label",
        "Overloaded (too many doors for the call budget), Not working the beat (spare capacity, weak visit %), Not converting (visits happened, shops did not buy), Not lifting drop (billed, order size is light), Fine.",
    ),
    (
        "Plan and next actions",
        "What to do this week, in order, with the MT at stake. Short bullets — not a scorecard dump.",
    ),
]


@dataclass
class SituationPack:
    period: str
    label: str
    scope: str = "national"
    scope_label: str = "Country"
    headline: str = ""
    weather: str = ""
    situation: list[str] = field(default_factory=list)
    kpis: dict[str, Any] = field(default_factory=dict)
    steps: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    ahead_cities: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    ahead_distributors: pd.DataFrame = field(default_factory=pd.DataFrame)
    lagging_people: pd.DataFrame = field(default_factory=pd.DataFrame)
    ahead_people: pd.DataFrame = field(default_factory=pd.DataFrame)
    weak_areas: pd.DataFrame = field(default_factory=pd.DataFrame)
    this_week: pd.DataFrame = field(default_factory=pd.DataFrame)
    copy_from: list[str] = field(default_factory=list)
    plan_lines: list[str] = field(default_factory=list)


def empty_situation_pack(period: str = "") -> SituationPack:
    return SituationPack(period=period or "", label=period or "")


def scorecards_for_period(
    shop_month: pd.DataFrame,
    stores: pd.DataFrame | None,
    ledger: pd.DataFrame | None,
    period: str,
    shop_targets: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
    shop_day: pd.DataFrame | None = None,
    facts: pd.DataFrame | None = None,
    mtd_obs: pd.DataFrame | None = None,
    cached_units: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Units for one calendar month. Reuses the live scorecards when they already match."""
    period = str(period or "")
    if cached_units is not None and not cached_units.empty and "period" in cached_units.columns:
        stored = str(cached_units["period"].dropna().astype(str).max() or "")
        if stored == period:
            return cached_units
    from sndintel.hierarchy import build_hierarchy_pack
    from sndintel.plan import attach_plan

    pack = build_hierarchy_pack(
        shop_month,
        stores,
        None,
        ledger,
        facts=facts,
        mtd_obs=mtd_obs,
        visits=visits,
        shop_day=shop_day,
        period=period,
    )
    mtd = period_state(ledger, period)
    pace = 1.0
    if pack.units is not None and not pack.units.empty and "intra_month_frac" in pack.units.columns:
        pace = float(pd.to_numeric(pack.units["intra_month_frac"], errors="coerce").dropna().max() or 1.0)
    if not mtd.get("open"):
        pace = 1.0
    return attach_plan(pack.units, shop_targets, pace=pace)


def load_units_for_period(conn: Any, period: str) -> pd.DataFrame:
    """Warehouse scorecards for one month. Rebuilds when the cached units are a different month."""
    from sndintel.storage import read_sql

    period = str(period or "")
    try:
        units = read_sql(conn, "SELECT * FROM unit_scorecards")
    except Exception:
        units = pd.DataFrame()
    if units is not None and not units.empty and "period" in units.columns:
        stored = str(units["period"].dropna().astype(str).max() or "")
        if stored == period:
            return units

    def _table(sql: str) -> pd.DataFrame:
        try:
            return read_sql(conn, sql)
        except Exception:
            return pd.DataFrame()

    return scorecards_for_period(
        _table("SELECT * FROM shop_month"),
        _table("SELECT * FROM stores"),
        _table("SELECT * FROM period_ledger ORDER BY period"),
        period,
        shop_targets=_table("SELECT * FROM shop_targets"),
        visits=_table("SELECT * FROM shop_visits"),
        shop_day=_table("SELECT * FROM shop_day"),
        mtd_obs=_table("SELECT * FROM mtd_observations"),
        cached_units=units,
    )


def build_situation_pack(
    units: pd.DataFrame,
    action: Any | None = None,
    ledger: pd.DataFrame | None = None,
    period: str | None = None,
    scope: str = "national",
    city: str | None = None,
    distributor: str | None = None,
) -> SituationPack:
    """Build one sendable pack. ``action`` is optional (enriches this-week doors)."""
    if units is None or units.empty:
        return empty_situation_pack(period or "")
    period = period or str(units["period"].dropna().astype(str).max() or "")
    scoped = _scope_units(units, scope, city, distributor)
    if scoped.empty:
        return empty_situation_pack(period)
    mtd = period_state(ledger, period)
    as_of = int(mtd.get("as_of_day") or mtd.get("days_in_month") or 21)
    days_m = int(mtd.get("days_in_month") or 31)
    label = str(mtd.get("label") or period)

    cities = _active(_grain(scoped, "city"))
    dists = _active(_grain(scoped, "distributor"))
    dsrs = _active(_grain(scoped, "dsr"))
    nat = _grain(units, "national")
    focus = _focus_row(scoped, units, scope, city, distributor)
    open_mtd = bool(mtd.get("open"))

    cap = (
        score_dsr_capacity_from_units(dsrs, as_of, days_m)
        if dsrs is not None and not dsrs.empty
        else pd.DataFrame()
    )

    lag_c = _board(cities, "city", lagging=True, n=LAG_N if scope == "national" else None, open_mtd=open_mtd)
    ahead_c = _board(cities, "city", lagging=False, n=AHEAD_N if scope == "national" else None, open_mtd=open_mtd)
    lag_d = _board(dists, "distributor", lagging=True, n=LAG_N if scope == "national" else None, open_mtd=open_mtd)
    ahead_d = _board(dists, "distributor", lagging=False, n=AHEAD_N if scope == "national" else None, open_mtd=open_mtd)
    lag_p = _people_board(dsrs, cap, lagging=True, n=PEOPLE_LAG_N if scope == "national" else None, open_mtd=open_mtd)
    ahead_p = _people_board(dsrs, cap, lagging=False, n=PEOPLE_AHEAD_N if scope == "national" else None, open_mtd=open_mtd)
    doors = _this_week_doors(action, scope, city, distributor) if open_mtd else pd.DataFrame()
    steps = _steps_to_potential(focus, lag_p, cap, action, scope, city, distributor, open_mtd=open_mtd)
    kpis = _kpis(focus, nat, cities, dists, dsrs, lag_p, ahead_p, mtd, scope)
    copy_from = _copy_lines(ahead_c, ahead_d, ahead_p, scope)
    headline, weather, paras = _narrative(
        focus, kpis, lag_c, ahead_c, lag_d, ahead_d, lag_p, ahead_p, steps, scope, city, distributor, label
    )
    plan_lines = _plan_lines(kpis, steps, open_mtd)
    scope_label = {
        "national": "Country",
        "city": city or "City",
        "distributor": distributor or "Distributor",
    }.get(scope, scope)

    return SituationPack(
        period=period,
        label=label,
        scope=scope,
        scope_label=str(scope_label),
        headline=headline,
        weather=weather,
        situation=paras,
        kpis=kpis,
        steps=steps,
        lagging_cities=lag_c,
        ahead_cities=ahead_c,
        lagging_distributors=lag_d,
        ahead_distributors=ahead_d,
        lagging_people=lag_p,
        ahead_people=ahead_p,
        weak_areas=pd.DataFrame(),
        this_week=doors,
        copy_from=copy_from,
        plan_lines=plan_lines,
    )


def list_situation_entities(units: pd.DataFrame, kind: str) -> list[str]:
    kind = (kind or "").strip().lower()
    if units is None or units.empty:
        return []
    if kind == "city":
        src = _active(_grain(units, "city"))
        if src.empty:
            return []
        return sorted({str(x) for x in src["grain_id"].dropna().astype(str) if str(x) not in {"", "ALL", "Country"}})
    if kind == "distributor":
        src = _active(_grain(units, "distributor"))
        if src.empty:
            return []
        out = []
        for _, r in src.iterrows():
            name = str(r.get("distributor") or r.get("grain_id") or "").strip()
            city = str(r.get("city") or "").strip()
            if not name:
                continue
            out.append(f"{city} · {name}" if city and city not in {"nan", "(unmapped)"} else name)
        return sorted(set(out))
    return []


def build_field_packs(
    units: pd.DataFrame,
    action: Any | None = None,
    ledger: pd.DataFrame | None = None,
    period: str | None = None,
    kind: str = "city",
) -> list[tuple[str, SituationPack]]:
    """One pack per city or distributor, ready to zip and send."""
    kind = (kind or "city").strip().lower()
    names = list_situation_entities(units, kind)
    out: list[tuple[str, SituationPack]] = []
    for name in names:
        city, dist = _split_entity(name, kind)
        pack = build_situation_pack(
            units,
            action=action,
            ledger=ledger,
            period=period,
            scope=kind,
            city=city,
            distributor=dist,
        )
        if pack.headline:
            out.append((name, pack))
    return out


def zip_field_packs(packs: list[tuple[str, SituationPack]], fmt: str = "pdf") -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, pack in packs:
            safe = _safe_name(name)
            period = pack.period or "period"
            if fmt == "excel":
                payload = excel_bytes(pack)
                zf.writestr(f"SND_{pack.scope}_{safe}_{period}.xlsx", payload)
            else:
                payload = pdf_bytes(pack)
                zf.writestr(f"SND_{pack.scope}_{safe}_{period}.pdf", payload)
    return buf.getvalue()


def how_to_read(pack: SituationPack) -> list[str]:
    scope = (pack.scope or "national").lower()
    kpis = pack.kpis or {}
    open_mtd = bool(kpis.get("open_mtd"))
    if open_mtd:
        lines = [
            "This is the in-month (MTD) pack. Billed is so far; projected month-end is that billed ÷ the national day curve.",
            "Monthly target is the full sales-team quota, not a paced slice.",
            "A unit can be ahead of today’s Expected and still miss the month-end target.",
        ]
    else:
        lines = [
            "This is the closed-month pack. Billed, Expected, and Target are full-month numbers.",
            "No month-end projection — the month is finished.",
        ]
    if scope == "city":
        lines += [
            f"{pack.scope_label} versus its own Expected.",
            "Distributors lagging first, then who is ahead (copy, do not raid).",
            "Plan and next actions are the week’s work, in order.",
        ]
    elif scope == "distributor":
        lines += [
            f"{pack.scope_label} versus its own Expected.",
            "DSRs on this book — who is lagging, who is ahead.",
            "Plan and next actions close this distributor’s Gap, not the country’s.",
        ]
    else:
        lines += [
            "Cities and distributors behind their own run-rate, then who is ahead.",
            "Salespeople: the label is the coaching script (too many doors / not visiting / not converting / light orders).",
            "Plan and next actions are ranked by MT. Send the matching city or distributor pack to the field.",
        ]
    return lines


def iter_situation_sheets(pack: SituationPack) -> list[tuple[str, str, str, pd.DataFrame]]:
    scope = (pack.scope or "national").lower()
    label = pack.scope_label or scope
    open_mtd = bool((pack.kpis or {}).get("open_mtd"))
    plan_note = (
        "What to do this week, in order. MT is volume at stake."
        if open_mtd
        else "What to do next on this closed month. MT is volume at stake."
    )
    sheets: list[tuple[str, str, str, pd.DataFrame]] = [
        (
            "01 Plan and next actions",
            "Plan and next actions",
            plan_note,
            pack.steps,
        )
    ]
    if scope == "national":
        sheets.extend(
            [
                (
                    "02 Cities lagging",
                    "Cities behind their own Expected",
                    "Key cities that missed their recent run-rate. Not a list of the biggest books.",
                    pack.lagging_cities,
                ),
                (
                    "03 Cities ahead",
                    "Cities ahead of Expected",
                    "Copy these beats. Do not raid people or stock from them to prop up a lagging city.",
                    pack.ahead_cities,
                ),
                (
                    "04 Distributors lagging",
                    "Distributors behind Expected",
                    "Send each of these the matching distributor pack.",
                    pack.lagging_distributors,
                ),
                (
                    "05 Distributors ahead",
                    "Distributors ahead of Expected",
                    "Protect drop size. Copy call cadence and mix.",
                    pack.ahead_distributors,
                ),
            ]
        )
    elif scope == "city":
        sheets.extend(
            [
                (
                    "02 Distributors lagging",
                    f"Distributors behind in {label}",
                    "These close this city’s Gap.",
                    pack.lagging_distributors,
                ),
                (
                    "03 Distributors ahead",
                    f"Distributors ahead in {label}",
                    "Copy, do not raid.",
                    pack.ahead_distributors,
                ),
            ]
        )
    sheets.extend(
        [
            (
                "06 People lagging" if scope == "national" else "04 People lagging",
                "Underperforming sales staff",
                "A DSR is city + distributor + name. Label is capacity vs skill, not a ranking by tons.",
                pack.lagging_people,
            ),
            (
                "07 People ahead" if scope == "national" else "05 People ahead",
                "Overperforming sales staff",
                "Ride with these names. Copy the beat; do not load extra stock.",
                pack.ahead_people,
            ),
        ]
    )
    if pack.this_week is not None and not pack.this_week.empty:
        sheets.append(
            (
                "08 This week" if scope == "national" else "06 This week",
                "This-week doors",
                "Call these shops this week. Ask is the reorder size.",
                pack.this_week,
            )
        )
    return sheets


def excel_bytes(pack: SituationPack) -> bytes:
    buf = BytesIO()
    write_excel(pack, buf)
    return buf.getvalue()


def write_excel(pack: SituationPack, path: Path | str | BytesIO) -> None:
    wb = Workbook()
    _excel_cover(wb, pack)
    for sheet, heading, note, df in iter_situation_sheets(pack):
        _sheet_table(wb, sheet, heading, note, df)
    if isinstance(path, BytesIO):
        wb.save(path)
        path.seek(0)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        wb.save(path)


def pdf_bytes(pack: SituationPack) -> bytes:
    buf = BytesIO()
    write_pdf(pack, buf)
    return buf.getvalue()


def write_pdf(pack: SituationPack, path: Path | str | BytesIO) -> None:
    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        path if not isinstance(path, (str, Path)) else str(path),
        pagesize=pagesize,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        title=f"SND Intelligence · {pack.scope_label} situation",
        author="SND Intelligence",
    )
    styles = _pdf_styles()
    story: list[Any] = []
    story.extend(_cover_flowables(pack, styles))
    usable = pagesize[0] - doc.leftMargin - doc.rightMargin
    for _sheet, heading, note, df in iter_situation_sheets(pack):
        story.append(PageBreak())
        block = [
            Paragraph(xml_escape(heading), styles["h2"]),
            Paragraph(xml_escape(note or ""), styles["note"]),
            Spacer(1, 4),
            _pdf_table(df, styles, usable),
        ]
        story.append(KeepTogether(block[:2]))
        story.extend(block[2:])
    story.append(PageBreak())
    story.extend(_glossary_end(styles))
    scope = pack.scope_label or pack.scope or "national"
    label = pack.label or ""

    def _on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, pagesize[1] - 9 * mm, pagesize[0], 9 * mm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(10 * mm, pagesize[1] - 6.2 * mm, "SND Intelligence · Situation")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            pagesize[0] - 10 * mm,
            pagesize[1] - 6.2 * mm,
            f"{scope}  ·  {label}  ·  {doc_.page}",
        )
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(10 * mm, 5 * mm, EXPECTED_FORMULA)
        canvas.drawRightString(pagesize[0] - 10 * mm, 5 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


# --- builders -----------------------------------------------------------------


def _num(row: pd.Series | dict[str, Any], name: str, default: float = 0.0) -> float:
    try:
        val = row.get(name) if hasattr(row, "get") else row[name]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return float(val)
    except (KeyError, TypeError, ValueError):
        return default


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if df is None or df.empty or name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _grain(units: pd.DataFrame, grain: str) -> pd.DataFrame:
    if units is None or units.empty or "grain" not in units.columns:
        return pd.DataFrame()
    return units[units["grain"].astype(str) == grain].copy()


def _active(df: pd.DataFrame) -> pd.DataFrame:
    """Hide units with no recent run-rate and no billed volume."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    exp = _col(df, "expected_mt").fillna(0)
    vol = _col(df, "volume_mt").fillna(0)
    return df.loc[(exp >= RUN_RATE_FLOOR) | (vol >= RUN_RATE_FLOOR)].copy()


def _scope_units(units: pd.DataFrame, scope: str, city: str | None, distributor: str | None) -> pd.DataFrame:
    scope = (scope or "national").lower()
    if scope in {"", "national", "country"}:
        return units
    work = units.copy()
    if city:
        if "city" in work.columns:
            work = work[work["city"].astype(str) == str(city)]
        elif scope == "city" and "grain_id" in work.columns:
            work = work[(work["grain"] != "city") | (work["grain_id"].astype(str) == str(city))]
    if distributor and "distributor" in work.columns:
        work = work[work["distributor"].astype(str) == str(distributor)]
    return work


def _focus_row(
    scoped: pd.DataFrame,
    units: pd.DataFrame,
    scope: str,
    city: str | None,
    distributor: str | None,
) -> pd.Series:
    scope = (scope or "national").lower()
    if scope == "city" and city:
        src = _grain(scoped, "city")
        hit = src[src["grain_id"].astype(str) == str(city)] if not src.empty else src
        if hit is not None and not hit.empty:
            return hit.iloc[0]
    if scope == "distributor" and distributor:
        src = _grain(scoped, "distributor")
        if not src.empty:
            mask = src["grain_id"].astype(str) == str(distributor)
            if city and "city" in src.columns:
                mask = mask & (src["city"].astype(str) == str(city))
            hit = src.loc[mask]
            if not hit.empty:
                return hit.iloc[0]
            if "distributor" in src.columns:
                mask = src["distributor"].astype(str) == str(distributor)
                if city:
                    mask = mask & (src["city"].astype(str) == str(city))
                hit = src.loc[mask]
                if not hit.empty:
                    return hit.iloc[0]
    nat = _grain(units, "national")
    if not nat.empty:
        return nat.iloc[0]
    return pd.Series(dtype=object)


def _sit_label(value: Any) -> str:
    key = str(value or "with_market")
    return SITUATION_LABEL.get(key, key.replace("_", " ").title())


def _driver_label(value: Any) -> str:
    key = str(value or "mixed")
    short = {
        "drop_size": "Drop size",
        "coverage": "Coverage",
        "whitespace": "Whitespace",
        "mixed": "Mixed",
        "holding": "Holding",
        "mix": "SKU mix",
    }
    return short.get(key, DRIVER_LABEL.get(key, key.replace("_", " ").title()))


def _short_action(text: Any, limit: int = 140) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _pct(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return None


def _mt(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _pace(row: pd.Series | dict[str, Any]) -> float:
    frac = _num(row, "intra_month_frac", 1.0)
    return frac if frac > 1e-6 else 1.0


def _projected(row: pd.Series, open_mtd: bool) -> float:
    billed = _num(row, "volume_mt")
    if not open_mtd:
        return billed
    return billed / _pace(row)


def _expected_full(row: pd.Series, open_mtd: bool) -> float:
    exp = _num(row, "expected_mt")
    if not open_mtd:
        return exp
    return exp / _pace(row)


def _monthly_target(row: pd.Series) -> float:
    return _num(row, "target_mt")


def _plan_phrase(projected: float, expected_full: float, target: float) -> str:
    if target <= 0.05:
        return ""
    if projected + 0.5 >= target:
        return "On track for target"
    if projected + 0.5 >= expected_full:
        return "On run-rate, short of target"
    return "Behind run-rate"


def _unit_do_this(row: pd.Series, open_mtd: bool) -> str:
    sit = str(row.get("situation") or "")
    diag = str(row.get("diagnosis") or "")
    projected = _projected(row, open_mtd)
    expected_full = _expected_full(row, open_mtd)
    target = _monthly_target(row)
    if sit == "lagging":
        if diag == "drop_size":
            return "Orders on billed shops are light. Lift drop this week."
        if diag == "coverage":
            return "Too few shops billed. Call the named doors."
        if diag == "whitespace":
            return "Universe not billed. Cover new doors this week."
        return "Behind Expected. Work the named doors this week."
    if sit == "outperforming":
        if target > 0.05 and projected + 0.5 < target:
            return "Ahead of Expected, still short of the monthly target. Keep drop. Hit due shops."
        return "Ahead. Protect drop. Do not load extra stock."
    if target > 0.05 and projected + 0.5 < target:
        return "On Expected. Still short of the monthly target."
    return "On Expected. Hold drop. Do not load."


def _people_do_this(label: str, sit: str) -> str:
    lab = str(label or "")
    if lab == "Overloaded":
        return "Too many doors for the call budget. Split the beat or add a DSR."
    if lab == "Not working the beat":
        return "Visit % is low. Ride with this DSR and audit calls."
    if lab == "Not converting":
        return "Visiting but not selling. Convert named shops — do not add coverage."
    if lab == "Not lifting drop":
        return "Billed, but orders are light. Lift drop on those doors."
    if sit == "lagging":
        return "Behind Expected. Coach from the label on this row."
    return "On or ahead. Leave this beat. Do not load."


def _door_do_this(call: str, ask_kg: int | None) -> str:
    ask = f"Ask {ask_kg:,} KG." if ask_kg else "Call this week."
    text = str(call or "").lower()
    if "not billed" in text or "visited" in text:
        return f"Visited, did not buy. Call again this week. {ask}"
    if "unvisited" in text or "unvisit" in text:
        return f"Due and not visited. Call this week. {ask}"
    return f"Call this week. {ask}"


def _board(df: pd.DataFrame, grain: str, lagging: bool, n: int | None, open_mtd: bool = False) -> pd.DataFrame:
    has_plan = _has_plan(df)
    cols = _board_columns(grain, has_plan=has_plan, open_mtd=open_mtd)
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    work = df.copy()
    sit = work["situation"].astype(str) if "situation" in work.columns else pd.Series("", index=work.index)
    if lagging:
        work = work[sit == "lagging"]
        sort_col = "gap_mt" if "gap_mt" in work.columns else "isolated_mt"
        if sort_col in work.columns:
            work = work.sort_values(sort_col, ascending=False)
    else:
        work = work[sit == "outperforming"]
        iso = _col(work, "isolated_mt")
        if iso.empty:
            work = work.sort_values("gap_mt") if "gap_mt" in work.columns else work
        else:
            work = work.assign(_iso=iso).sort_values("_iso", ascending=False)
    if n:
        work = work.head(int(n))
    rows = []
    for _, r in work.iterrows():
        rec: dict[str, Any] = {}
        if grain == "city":
            rec["City"] = str(r.get("grain_id") or r.get("city") or "")
        elif grain == "distributor":
            rec["Distributor"] = str(r.get("distributor") or r.get("grain_id") or "")
            rec["City"] = str(r.get("city") or "")
        rec["Situation"] = _sit_label(r.get("situation"))
        rec["Billed (MT)"] = _mt(r.get("volume_mt"))
        projected = _projected(r, open_mtd)
        expected_full = _expected_full(r, open_mtd)
        target = _monthly_target(r)
        if open_mtd:
            rec["Projected month-end (MT)"] = _mt(projected)
        rec["Expected (MT)"] = _mt(expected_full)
        rec["Driver"] = _driver_label(r.get("diagnosis"))
        rec["Visit %"] = _pct(r.get("visit_rate"))
        if has_plan:
            rec["Monthly target (MT)"] = _mt(target)
            rec["vs Target (MT)"] = _mt(projected - target)
            rec["Plan"] = _plan_phrase(projected, expected_full, target)
        rec["Do this"] = _unit_do_this(r, open_mtd)
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _has_plan(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "target_mt" not in df.columns:
        return False
    return float(pd.to_numeric(df["target_mt"], errors="coerce").fillna(0).sum()) > 0.05


def _board_columns(grain: str, has_plan: bool = False, open_mtd: bool = False) -> list[str]:
    if grain == "distributor":
        head = ["Distributor", "City"]
    else:
        head = ["City"]
    mid = ["Situation", "Billed (MT)"]
    if open_mtd:
        mid.append("Projected month-end (MT)")
    mid += ["Expected (MT)", "Driver", "Visit %"]
    if has_plan:
        mid += ["Monthly target (MT)", "vs Target (MT)", "Plan"]
    return head + mid + ["Do this"]


def _people_columns(has_plan: bool, open_mtd: bool) -> list[str]:
    cols = ["DSR", "City", "Distributor", "Situation", "Label", "Billed (MT)"]
    if open_mtd:
        cols.append("Projected month-end (MT)")
    cols += ["Expected (MT)", "Visit %"]
    if has_plan:
        cols += ["Monthly target (MT)", "vs Target (MT)", "Plan"]
    cols.append("Do this")
    return cols


def _people_board(dsrs: pd.DataFrame, cap: pd.DataFrame, lagging: bool, n: int | None, open_mtd: bool = False) -> pd.DataFrame:
    has_plan = _has_plan(dsrs)
    cols = _people_columns(has_plan, open_mtd)
    if dsrs is None or dsrs.empty:
        return pd.DataFrame(columns=cols)
    work = dsrs.copy()
    if cap is not None and not cap.empty:
        key = "grain_id" if "grain_id" in work.columns and "grain_id" in cap.columns else None
        if key:
            labels = cap.set_index(key)["label"] if "label" in cap.columns else pd.Series(dtype=object)
            work["_label"] = work[key].map(labels)
        else:
            work["_label"] = cap["label"].tolist()[: len(work)] if "label" in cap.columns else ""
    else:
        work["_label"] = ""
    sit = work["situation"].astype(str) if "situation" in work.columns else pd.Series("", index=work.index)
    lab = work["_label"].astype(str)
    gap_s = _col(work, "gap_mt").fillna(0)
    if lagging:
        keep = (sit == "lagging") | lab.isin(
            ["Overloaded", "Not working the beat", "Not converting", "Not lifting drop"]
        )
        work = work.loc[keep]
        work = work.sort_values("gap_mt" if "gap_mt" in work.columns else "volume_mt", ascending=False)
    else:
        keep = (sit == "outperforming") | ((lab == LABEL_FINE) & (gap_s <= 0) & (sit != "lagging"))
        work = work.loc[keep]
        iso = _col(work, "isolated_mt")
        work = work.assign(_iso=iso.fillna(0)).sort_values("_iso", ascending=False)
    if n:
        work = work.head(int(n))
    rows = []
    for _, r in work.iterrows():
        name = r.get("dsr_name") or dsr_display_name(r.get("grain_id"))
        projected = _projected(r, open_mtd)
        expected_full = _expected_full(r, open_mtd)
        target = _monthly_target(r)
        rec: dict[str, Any] = {
            "DSR": str(name or ""),
            "City": str(r.get("city") or ""),
            "Distributor": str(r.get("distributor") or ""),
            "Situation": _sit_label(r.get("situation")),
            "Label": str(r.get("_label") or ""),
            "Billed (MT)": _mt(r.get("volume_mt")),
            "Expected (MT)": _mt(expected_full),
            "Visit %": _pct(r.get("visit_rate")),
            "Do this": _people_do_this(str(r.get("_label") or ""), str(r.get("situation") or "")),
        }
        if open_mtd:
            rec["Projected month-end (MT)"] = _mt(projected)
        if has_plan:
            rec["Monthly target (MT)"] = _mt(target)
            rec["vs Target (MT)"] = _mt(projected - target)
            rec["Plan"] = _plan_phrase(projected, expected_full, target)
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _this_week_doors(action: Any | None, scope: str, city: str | None, distributor: str | None) -> pd.DataFrame:
    cols = ["Shop", "City", "Distributor", "DSR", "Call", "Ask (KG)", "Do this"]
    if action is None:
        return pd.DataFrame(columns=cols)
    src = getattr(action, "raw_shops", None)
    if src is None or not isinstance(src, pd.DataFrame) or src.empty:
        src = getattr(action, "all_shops", None)
    if src is None or not isinstance(src, pd.DataFrame) or src.empty:
        return pd.DataFrame(columns=cols)
    work = src.copy()
    city_col = "city" if "city" in work.columns else ("City" if "City" in work.columns else None)
    dist_col = "distributor" if "distributor" in work.columns else ("Distributor" if "Distributor" in work.columns else None)
    if scope in {"city", "distributor"} and city and city_col:
        work = work[work[city_col].astype(str) == str(city)]
    if scope == "distributor" and distributor and dist_col:
        work = work[work[dist_col].astype(str) == str(distributor)]
    if work.empty:
        return pd.DataFrame(columns=cols)
    ask_col = "week_target_mt" if "week_target_mt" in work.columns else None
    if ask_col is None and "Ask rest of month (KG)" in work.columns:
        work["_ask_mt"] = pd.to_numeric(work["Ask rest of month (KG)"], errors="coerce").fillna(0) / 1000.0
        ask_col = "_ask_mt"
    if ask_col:
        work = work.assign(_ask=pd.to_numeric(work[ask_col], errors="coerce").fillna(0))
        work = work[work["_ask"] > 0].sort_values("_ask", ascending=False)
    work = work.head(DOOR_N)
    rows = []
    for _, r in work.iterrows():
        shop = r.get("store_name") or r.get("Shop") or r.get("store_id") or ""
        dsr = r.get("dsr_name") or r.get("DSR") or ""
        call = r.get("call_status") or r.get("Call") or r.get("recommended_action") or ""
        ask_mt = _num(r, "_ask") if "_ask" in r.index else _num(r, "week_target_mt")
        ask_kg = int(round(ask_mt * 1000)) if ask_mt else None
        call = str(call)
        rows.append(
            {
                "Shop": str(shop),
                "City": str(r.get("city") or r.get("City") or ""),
                "Distributor": str(r.get("distributor") or r.get("Distributor") or ""),
                "DSR": str(dsr),
                "Call": call,
                "Ask (KG)": ask_kg,
                "Do this": _door_do_this(call, ask_kg),
            }
        )
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _action_ask_mt(action: Any | None, scope: str, city: str | None, distributor: str | None) -> tuple[float, int]:
    if action is None:
        return 0.0, 0
    src = getattr(action, "raw_shops", None)
    if src is None or not isinstance(src, pd.DataFrame) or src.empty:
        return 0.0, 0
    work = src.copy()
    if scope in {"city", "distributor"} and city and "city" in work.columns:
        work = work[work["city"].astype(str) == str(city)]
    if scope == "distributor" and distributor and "distributor" in work.columns:
        work = work[work["distributor"].astype(str) == str(distributor)]
    ask = pd.to_numeric(work.get("week_target_mt"), errors="coerce").fillna(0) if not work.empty else pd.Series(dtype=float)
    due = pd.to_numeric(work.get("due_unvisited_mt"), errors="coerce").fillna(0) if "due_unvisited_mt" in work.columns else ask
    n = int((ask > 0).sum()) if not ask.empty else 0
    return float(due.sum() if due is not None and not due.empty else ask.sum()), n


def _steps_to_potential(
    focus: pd.Series,
    lag_people: pd.DataFrame,
    cap: pd.DataFrame,
    action: Any | None,
    scope: str,
    city: str | None,
    distributor: str | None,
    open_mtd: bool = False,
) -> pd.DataFrame:
    cols = ["Step", "Volume at stake (MT)", "Who", "Do this"]
    if focus is None or focus.empty:
        return pd.DataFrame(columns=cols)
    unvis = max(0.0, _num(focus, "from_unvisited_mt"))
    unbill = max(0.0, _num(focus, "from_unbilled_mt"))
    drop = max(0.0, _num(focus, "from_drop_size_mt"))
    gap = max(0.0, _num(focus, "gap_mt"))
    stretch = max(0.0, _num(focus, "stretch_mt"))
    gap_to_target = max(0.0, _num(focus, "gap_to_target_mt"))
    ask_mt, ask_n = (0.0, 0)
    if open_mtd:
        ask_mt, ask_n = _action_ask_mt(action, scope, city, distributor)
    owner = {"national": "NSM", "city": "City manager", "distributor": "Distributor / ASM"}.get(scope, "NSM")
    when = "this week" if open_mtd else "next"
    candidates: list[dict[str, Any]] = []
    if ask_mt >= 0.05:
        candidates.append(
            {
                "title": "Call due shops that have not been visited",
                "mt": ask_mt,
                "owner": "DSRs",
                "do": f"{ask_n} shops are due. Call them {when}. Ask {ask_mt:.1f} MT.",
            }
        )
    if unvis >= 0.25:
        candidates.append(
            {
                "title": "Cover shops not yet visited",
                "mt": unvis,
                "owner": owner,
                "do": f"{unvis:.1f} MT sits on unvisited shops. Ride with DSRs who are not working the beat.",
            }
        )
    if unbill >= 0.25:
        candidates.append(
            {
                "title": "Convert shops that were visited but did not buy",
                "mt": unbill,
                "owner": "DSRs",
                "do": f"{unbill:.1f} MT is visits with no bill. Call those shops again {when}.",
            }
        )
    if drop >= 0.25:
        candidates.append(
            {
                "title": "Lift drop size on billed shops",
                "mt": drop,
                "owner": owner,
                "do": f"{drop:.1f} MT is smaller orders on shops that already billed. Increase drop. Do not dump stock.",
            }
        )
    if lag_people is not None and not lag_people.empty:
        names = [str(x) for x in lag_people["DSR"].head(4).tolist()] if "DSR" in lag_people.columns else []
        people_gap = _people_stake_mt(lag_people)
        labels = []
        if cap is not None and not cap.empty and "label" in cap.columns:
            for lab in ("Overloaded", "Not working the beat", "Not converting", "Not lifting drop"):
                n_lab = int((cap["label"].astype(str) == lab).sum())
                if n_lab:
                    labels.append(f"{n_lab} {lab.lower()}")
        start = f"Start with {', '.join(names)}." if names else "Coach the named DSRs."
        why = f"{', '.join(labels)}. " if labels else ""
        candidates.append(
            {
                "title": "Coach lagging salespeople",
                "mt": people_gap,
                "owner": owner,
                "do": f"{why}{start}",
            }
        )
    if stretch >= 0.5 and gap_to_target > gap + 0.25:
        candidates.append(
            {
                "title": "Close the stretch to the monthly target",
                "mt": stretch,
                "owner": owner,
                "do": f"Target is {stretch:.1f} MT above Expected. Lift drop and hit due shops. Do not print a lost-shop list.",
            }
        )
    if gap >= 0.25 and not candidates:
        candidates.append(
            {
                "title": "Close the Gap versus Expected",
                "mt": gap,
                "owner": owner,
                "do": f"{gap:.1f} MT behind Expected. Work the named distributors and DSRs on this pack.",
            }
        )
    candidates.append(
        {
            "title": "Protect and copy overperformers",
            "mt": 0.01,
            "owner": owner,
            "do": "Do not pull people or stock from winning beats. Copy their call cadence and mix.",
        }
    )
    ranked = sorted(candidates, key=lambda p: float(p.get("mt") or 0), reverse=True)[:STEP_N]
    rows = []
    for i, step in enumerate(ranked, start=1):
        rows.append(
            {
                "Step": f"{i}. {step['title']}",
                "Volume at stake (MT)": _mt(step["mt"]) if float(step["mt"]) >= 0.5 else "—",
                "Who": step["owner"],
                "Do this": step["do"],
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _people_stake_mt(lag_people: pd.DataFrame) -> float:
    if lag_people is None or lag_people.empty:
        return 0.0
    if "vs Target (MT)" in lag_people.columns:
        vs = pd.to_numeric(lag_people["vs Target (MT)"], errors="coerce").fillna(0)
        return float((-vs.clip(upper=0)).head(8).sum())
    if "Gap (MT)" in lag_people.columns:
        return float(pd.to_numeric(lag_people["Gap (MT)"], errors="coerce").fillna(0).head(8).sum())
    return 0.0


def _plan_lines(kpis: dict[str, Any], steps: pd.DataFrame, open_mtd: bool) -> list[str]:
    billed = float(kpis.get("billed_mt") or 0)
    projected = float(kpis.get("projected_mt") or billed)
    target = float(kpis.get("target_mt") or 0)
    expected_today = float(kpis.get("expected_today_mt") or 0)
    expected_full = float(kpis.get("expected_full_mt") or kpis.get("expected_mt") or 0)
    vs_t = float(kpis.get("vs_target_mt") if kpis.get("vs_target_mt") is not None else (projected - target))
    lines: list[str] = []
    if open_mtd:
        lines.append(f"Billed so far {billed:.0f} MT. Projected month-end {projected:.0f} MT.")
        if expected_today > 0.05:
            lines.append(f"Should have billed {expected_today:.0f} MT by today.")
        if target > 0.05:
            if vs_t >= -0.5:
                lines.append(f"On track for the monthly target of {target:.0f} MT.")
            else:
                lines.append(f"Short of the monthly target ({target:.0f} MT) by {abs(vs_t):.0f} MT at this pace.")
    else:
        lines.append(f"Closed month billed {billed:.0f} MT. Expected was {expected_full:.0f} MT.")
        if target > 0.05:
            if billed + 0.5 >= target:
                lines.append(f"Hit the monthly target of {target:.0f} MT.")
            else:
                lines.append(f"Missed the monthly target ({target:.0f} MT) by {max(0.0, target - billed):.0f} MT.")
    return lines


def _kpis(
    focus: pd.Series,
    nat: pd.DataFrame,
    cities: pd.DataFrame,
    dists: pd.DataFrame,
    dsrs: pd.DataFrame,
    lag_p: pd.DataFrame,
    ahead_p: pd.DataFrame,
    mtd: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    sit = str(focus.get("situation") or "with_market") if focus is not None and not focus.empty else "with_market"
    n_lag_c = int((_grain_sit(cities, "lagging"))) if cities is not None else 0
    n_ahead_c = int((_grain_sit(cities, "outperforming"))) if cities is not None else 0
    n_lag_d = int((_grain_sit(dists, "lagging"))) if dists is not None else 0
    n_ahead_d = int((_grain_sit(dists, "outperforming"))) if dists is not None else 0
    billed = _num(focus, "volume_mt") if focus is not None and not focus.empty else 0.0
    expected = _num(focus, "expected_mt") if focus is not None and not focus.empty else 0.0
    gap = max(0.0, _num(focus, "gap_mt") if focus is not None and not focus.empty else 0.0)
    open_mtd = bool(mtd.get("open"))
    projected = _projected(focus, open_mtd) if focus is not None and not focus.empty else billed
    expected_full = _expected_full(focus, open_mtd) if focus is not None and not focus.empty else expected
    target = _num(focus, "target_mt") if focus is not None and not focus.empty else 0.0
    vs_target = projected - target
    attain = (projected / target) if target > 0.05 else None
    return {
        "scope": scope,
        "situation": sit,
        "situation_label": _sit_label(sit),
        "billed_mt": billed,
        "expected_mt": expected,
        "expected_today_mt": expected if open_mtd else expected_full,
        "expected_full_mt": expected_full,
        "projected_mt": projected,
        "gap_mt": gap,
        "ly_mt": _num(focus, "ly_mt") if focus is not None and not focus.empty else 0.0,
        "visit_pct": _pct(focus.get("visit_rate") if focus is not None and not focus.empty else None),
        "strike_pct": _pct(focus.get("strike_rate") if focus is not None and not focus.empty else None),
        "driver": _driver_label(focus.get("diagnosis") if focus is not None and not focus.empty else "mixed"),
        "n_lagging_cities": n_lag_c,
        "n_ahead_cities": n_ahead_c,
        "n_lagging_distributors": n_lag_d,
        "n_ahead_distributors": n_ahead_d,
        "n_lagging_people": int(len(lag_p)) if lag_p is not None else 0,
        "n_ahead_people": int(len(ahead_p)) if ahead_p is not None else 0,
        "n_dsrs": int(len(dsrs)) if dsrs is not None and not dsrs.empty else 0,
        "open_mtd": open_mtd,
        "as_of_day": mtd.get("as_of_day"),
        "days_in_month": mtd.get("days_in_month"),
        "country_billed_mt": float(_num(nat.iloc[0], "volume_mt")) if nat is not None and not nat.empty else billed,
        "country_gap_mt": max(0.0, float(_num(nat.iloc[0], "gap_mt"))) if nat is not None and not nat.empty else gap,
        "target_mt": target,
        "target_full_mt": target,
        "vs_target_mt": vs_target,
        "gap_to_target_mt": max(0.0, -vs_target),
        "stretch_mt": max(0.0, _num(focus, "stretch_mt") if focus is not None and not focus.empty else 0.0),
        "attain_pct": attain,
        "plan_quality": str(focus.get("plan_quality") or "") if focus is not None and not focus.empty else "",
        "plan_status": str(focus.get("plan_status") or "") if focus is not None and not focus.empty else "",
        "n_target_unmatched": int(_num(focus, "n_target_unmatched")) if focus is not None and not focus.empty else 0,
        "target_book_mt": _num(focus, "target_book_mt") if focus is not None and not focus.empty else 0.0,
    }


def _grain_sit(df: pd.DataFrame, sit: str) -> int:
    if df is None or df.empty or "situation" not in df.columns:
        return 0
    return int((df["situation"].astype(str) == sit).sum())


def _copy_lines(
    ahead_c: pd.DataFrame, ahead_d: pd.DataFrame, ahead_p: pd.DataFrame, scope: str
) -> list[str]:
    lines: list[str] = []
    if scope == "national" and ahead_c is not None and not ahead_c.empty and "City" in ahead_c.columns:
        names = ", ".join(str(x) for x in ahead_c["City"].head(4).tolist())
        lines.append(f"Copy city beats: {names}.")
    if ahead_d is not None and not ahead_d.empty and "Distributor" in ahead_d.columns:
        names = ", ".join(str(x) for x in ahead_d["Distributor"].head(4).tolist())
        lines.append(f"Copy distributors: {names}.")
    if ahead_p is not None and not ahead_p.empty and "DSR" in ahead_p.columns:
        names = ", ".join(str(x) for x in ahead_p["DSR"].head(4).tolist())
        lines.append(f"Ride with: {names}.")
    if not lines:
        lines.append("No named overperformers this period — hold winning doors and do not load.")
    return lines


def _narrative(
    focus: pd.Series,
    kpis: dict[str, Any],
    lag_c: pd.DataFrame,
    ahead_c: pd.DataFrame,
    lag_d: pd.DataFrame,
    ahead_d: pd.DataFrame,
    lag_p: pd.DataFrame,
    ahead_p: pd.DataFrame,
    steps: pd.DataFrame,
    scope: str,
    city: str | None,
    distributor: str | None,
    label: str,
) -> tuple[str, str, list[str]]:
    billed = float(kpis.get("billed_mt") or 0)
    expected_today = float(kpis.get("expected_today_mt") or kpis.get("expected_mt") or 0)
    expected_full = float(kpis.get("expected_full_mt") or kpis.get("expected_mt") or 0)
    gap = float(kpis.get("gap_mt") or 0)
    projected = float(kpis.get("projected_mt") or billed)
    target = float(kpis.get("target_mt") or 0)
    vs_t = float(kpis.get("vs_target_mt") if kpis.get("vs_target_mt") is not None else (projected - target))
    sit = str(kpis.get("situation_label") or "On expected")
    open_mtd = bool(kpis.get("open_mtd"))
    who = {"national": "Country", "city": city or "City", "distributor": distributor or "Distributor"}.get(
        scope, "Country"
    )
    attain = kpis.get("attain_pct")
    attain_txt = f"{float(attain)*100:.0f}%" if attain is not None else "—"

    if open_mtd:
        weather = (
            f"{label}: {who} billed {billed:.0f} MT so far. "
            f"Projected month-end {projected:.0f} MT"
        )
        if target > 0.05:
            weather += f" versus monthly target {target:.0f} MT ({attain_txt})."
        else:
            weather += "."
        if expected_today > 0.05:
            weather += f" Should have billed {expected_today:.0f} MT by today ({sit.lower()})."
        else:
            weather += f" {sit}."
    else:
        weather = (
            f"{label}: {who} billed {billed:.0f} MT versus Expected {expected_full:.0f} MT "
            f"({sit.lower()})."
        )
        if target > 0.05:
            weather += f" Monthly target {target:.0f} MT ({attain_txt})."

    short_target = target > 0.05 and vs_t < -0.5
    if short_target:
        headline = f"{who} is short of the monthly target by {abs(vs_t):.0f} MT at this pace."
    elif sit == "Lagging":
        headline = f"{who} is behind Expected by {gap:.0f} MT."
    elif sit == "Ahead":
        headline = f"{who} is ahead of Expected. Protect the base."
    else:
        headline = f"{who} is on Expected. Work the local exceptions."

    paras: list[str] = [weather]
    if scope == "national":
        lag_names = _join_col(lag_c, "City")
        ahead_names = _join_col(ahead_c, "City")
        paras.append(
            f"{int(kpis.get('n_lagging_cities') or 0)} cities lagging"
            + (f" ({lag_names})" if lag_names else "")
            + f"; {int(kpis.get('n_ahead_cities') or 0)} ahead"
            + (f" ({ahead_names})" if ahead_names else "")
            + "."
        )
        dist_lag = _join_col(lag_d, "Distributor")
        dist_ahead = _join_col(ahead_d, "Distributor")
        paras.append(
            f"Distributors: {int(kpis.get('n_lagging_distributors') or 0)} behind"
            + (f" — {dist_lag}" if dist_lag else "")
            + f"; {int(kpis.get('n_ahead_distributors') or 0)} ahead"
            + (f" — {dist_ahead}" if dist_ahead else "")
            + ". Send each lagging distributor its own pack."
        )
    else:
        dist_lag = _join_col(lag_d, "Distributor")
        dist_ahead = _join_col(ahead_d, "Distributor")
        if scope == "city":
            paras.append(
                f"In {who}: distributors behind — {dist_lag or 'none named'}; "
                f"ahead — {dist_ahead or 'none named'}."
            )
        country_gap = float(kpis.get("country_gap_mt") or 0)
        if country_gap > 0 and gap > 0:
            share = 100.0 * gap / country_gap
            paras.append(f"This book is {share:.0f}% of the country Gap ({country_gap:.0f} MT).")
    people_lag = _join_col(lag_p, "DSR")
    people_ahead = _join_col(ahead_p, "DSR")
    paras.append(
        f"Sales staff: {int(kpis.get('n_lagging_people') or 0)} underperforming"
        + (f" ({people_lag})" if people_lag else "")
        + f"; {int(kpis.get('n_ahead_people') or 0)} to copy"
        + (f" ({people_ahead})" if people_ahead else "")
        + "."
    )
    if steps is not None and not steps.empty and "Step" in steps.columns:
        titles = [str(x).split(". ", 1)[-1] for x in steps["Step"].tolist()[:4]]
        paras.append("Next actions: " + "; ".join(titles) + ".")
    driver = str(kpis.get("driver") or "Mixed")
    paras.append(f"Main driver on this book: {driver}.")
    if target > 0.05:
        unmatched = int(kpis.get("n_target_unmatched") or 0)
        extra = f" {unmatched} target shops did not match the universe." if unmatched else ""
        paras.append("Expected is the run-rate. Target is the quota." + extra)
    return headline, weather, paras


def _join_col(df: pd.DataFrame, col: str, n: int = 4) -> str:
    if df is None or df.empty or col not in df.columns:
        return ""
    names = [str(x) for x in df[col].head(n).tolist() if str(x).strip()]
    extra = int(len(df) - n)
    text = ", ".join(names)
    if extra > 0:
        text += f" +{extra}"
    return text


def _split_entity(entity: str, kind: str) -> tuple[str | None, str | None]:
    parts = [p.strip() for p in str(entity or "").split(" · ") if str(p).strip()]
    if kind == "city":
        return (parts[-1] if parts else entity), None
    if kind == "distributor":
        if len(parts) >= 2:
            return parts[0], " · ".join(parts[1:])
        return None, parts[0] if parts else entity
    return None, None


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return (text[:60] or "pack").strip("_")


# --- Excel cover --------------------------------------------------------------


def _excel_cover(wb: Workbook, pack: SituationPack) -> None:
    from openpyxl.styles import Alignment, Font
    from sndintel.briefing import NAVY as HEX_NAVY, SLATE as HEX_SLATE, _fill

    ws = wb.active
    ws.title = "00 Situation"
    ws["A1"] = "SND Intelligence · Situation cascade"
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=HEX_NAVY)
    ws["A2"] = f"{pack.scope_label} · {pack.label}"
    ws["A2"].font = Font(name="Calibri", size=18, bold=True, color=HEX_NAVY)
    ws["A3"] = pack.headline or ""
    ws["A3"].font = Font(name="Calibri", size=12, bold=True, color=HEX_NAVY)
    ws.merge_cells("A3:H3")
    row = 5
    kpis = pack.kpis or {}
    open_mtd = bool(kpis.get("open_mtd"))
    has_plan = float(kpis.get("target_mt") or 0) > 0.05
    if open_mtd:
        metric_row = [
            ("Billed so far (MT)", kpis.get("billed_mt")),
            ("Projected month-end (MT)", kpis.get("projected_mt")),
            ("Monthly target (MT)", kpis.get("target_mt") if has_plan else None),
            ("vs Target (MT)", kpis.get("vs_target_mt") if has_plan else None),
            ("Situation", kpis.get("situation_label")),
            ("Driver", kpis.get("driver")),
        ]
    else:
        metric_row = [
            ("Billed (MT)", kpis.get("billed_mt")),
            ("Expected (MT)", kpis.get("expected_full_mt") or kpis.get("expected_mt")),
            ("Monthly target (MT)", kpis.get("target_mt") if has_plan else None),
            ("vs Target (MT)", kpis.get("vs_target_mt") if has_plan else None),
            ("Situation", kpis.get("situation_label")),
            ("Driver", kpis.get("driver")),
        ]
    metric_row += [
        ("Lagging people", kpis.get("n_lagging_people")),
        ("Ahead people", kpis.get("n_ahead_people")),
    ]
    for i, (name, val) in enumerate(metric_row, start=1):
        cell = ws.cell(row, i, name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = _fill(HEX_NAVY)
        show_mt = isinstance(val, (int, float)) and name.endswith("(MT)")
        v = ws.cell(row + 1, i, _mt(val) if show_mt else val)
        v.font = Font(size=12, bold=True)
    row = 8
    if open_mtd and float(kpis.get("expected_today_mt") or 0) > 0.05:
        ws.cell(row, 1, f"Should have billed {float(kpis.get('expected_today_mt') or 0):.0f} MT by today (Expected × day curve).")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).font = Font(size=9, italic=True, color=HEX_SLATE)
        row += 1
    ws.cell(row, 1, "Current situation")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=HEX_NAVY)
    row += 1
    for para in pack.situation:
        ws.cell(row, 1, para)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 48
        row += 1
    if pack.plan_lines:
        row += 1
        ws.cell(row, 1, "Plan")
        ws.cell(row, 1).font = Font(bold=True, size=12, color=HEX_NAVY)
        row += 1
        for line in pack.plan_lines:
            ws.cell(row, 1, "• " + line)
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
            row += 1
        row += 1
    row += 1
    ws.cell(row, 1, "Copy from overperformers")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=HEX_NAVY)
    row += 1
    for line in pack.copy_from:
        ws.cell(row, 1, line)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        row += 1
    row += 1
    ws.cell(row, 1, "How to read this pack")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=HEX_NAVY)
    row += 1
    for step in how_to_read(pack):
        ws.cell(row, 1, step)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).font = Font(size=10, italic=True, color=HEX_SLATE)
        row += 1
    ws.column_dimensions["A"].width = 28


# --- PDF ----------------------------------------------------------------------

NAVY = colors.HexColor("#0F172A")
SLATE = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
WASH = colors.HexColor("#F8FAFC")
LAG = colors.HexColor("#FEF2F2")
AHEAD = colors.HexColor("#F0FDF4")
WHITE = colors.white

TEXT_COLS = {
    "City",
    "Distributor",
    "DSR",
    "Shop",
    "Area",
    "Situation",
    "Label",
    "Driver",
    "Plan",
    "Call",
    "Do this",
    "Do this week",
    "Step",
    "Owner",
    "Who",
}


def _pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "kicker", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=SLATE, spaceAfter=4
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=18, textColor=NAVY, spaceAfter=6, leading=22
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, spaceBefore=0, spaceAfter=4
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=3
        ),
        "headline": ParagraphStyle(
            "headline", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=NAVY, leading=16, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica", fontSize=9, textColor=NAVY, leading=13, spaceAfter=4
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8, textColor=SLATE, leading=11, spaceAfter=2
        ),
        "th": ParagraphStyle(
            "th", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7, textColor=WHITE, leading=9, alignment=TA_LEFT
        ),
        "td": ParagraphStyle(
            "td", parent=base["Normal"], fontName="Helvetica", fontSize=7.5, textColor=NAVY, leading=10, alignment=TA_LEFT
        ),
        "td_right": ParagraphStyle(
            "td_right", parent=base["Normal"], fontName="Helvetica", fontSize=7.5, textColor=NAVY, leading=10, alignment=TA_RIGHT
        ),
        "kpi_l": ParagraphStyle(
            "kpi_l", parent=base["Normal"], fontName="Helvetica", fontSize=7, textColor=SLATE, leading=9, alignment=TA_LEFT
        ),
        "kpi_v": ParagraphStyle(
            "kpi_v", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=12, textColor=NAVY, leading=14, alignment=TA_LEFT
        ),
        "gloss": ParagraphStyle(
            "gloss", parent=base["Normal"], fontName="Helvetica", fontSize=8, textColor=SLATE, leading=11, spaceAfter=3
        ),
    }


def _cover_flowables(pack: SituationPack, styles: dict[str, ParagraphStyle]) -> list[Any]:
    kicker = {
        "national": "NATIONAL SITUATION",
        "city": "CITY PACK — SEND TO THE CITY",
        "distributor": "DISTRIBUTOR PACK — SEND TO THE DISTRIBUTOR",
    }.get(pack.scope, "SITUATION PACK")
    story: list[Any] = [
        Paragraph(kicker, styles["kicker"]),
        Paragraph(xml_escape(f"{pack.scope_label} · {pack.label}"), styles["h1"]),
        Paragraph(xml_escape(pack.headline or ""), styles["headline"]),
    ]
    kpis = pack.kpis or {}
    open_mtd = bool(kpis.get("open_mtd"))
    has_plan = float(kpis.get("target_mt") or 0) > 0.05
    billed = f"{float(kpis.get('billed_mt') or 0):.0f} MT"
    projected = f"{float(kpis.get('projected_mt') or 0):.0f} MT"
    expected = f"{float(kpis.get('expected_full_mt') or kpis.get('expected_mt') or 0):.0f} MT"
    target = f"{float(kpis.get('target_mt') or 0):.0f} MT" if has_plan else "—"
    vs_t = f"{float(kpis.get('vs_target_mt') or 0):.0f} MT" if has_plan else "—"
    if open_mtd:
        kpi_cells = [
            ("Billed so far", billed),
            ("Projected month-end", projected),
            ("Monthly target", target),
            ("vs Target", vs_t),
            ("Situation", str(kpis.get("situation_label") or "")),
        ]
    else:
        kpi_cells = [
            ("Billed", billed),
            ("Expected", expected),
            ("Monthly target", target),
            ("vs Target", vs_t),
            ("Situation", str(kpis.get("situation_label") or "")),
        ]
    kpi_data = [
        [Paragraph(xml_escape(n), styles["kpi_l"]) for n, _ in kpi_cells],
        [Paragraph(xml_escape(v), styles["kpi_v"]) for _, v in kpi_cells],
    ]
    kpi_table = Table(kpi_data, colWidths=[270 * mm / max(len(kpi_cells), 1)] * len(kpi_cells))
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WASH),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(kpi_table)
    if open_mtd and float(kpis.get("expected_today_mt") or 0) > 0.05:
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                xml_escape(
                    f"Should have billed {float(kpis.get('expected_today_mt') or 0):.0f} MT by today "
                    "(Expected × the national day curve)."
                ),
                styles["note"],
            )
        )
    story.append(Spacer(1, 8))
    story.append(Paragraph("Current situation", styles["h2"]))
    for para in pack.situation:
        story.append(Paragraph(xml_escape(para), styles["body"]))
    if pack.plan_lines:
        story.append(Paragraph("Plan", styles["h3"]))
        for line in pack.plan_lines:
            story.append(Paragraph(xml_escape("• " + line), styles["body"]))
    if pack.copy_from:
        story.append(Paragraph("Copy from overperformers", styles["h3"]))
        for line in pack.copy_from:
            story.append(Paragraph(xml_escape(line), styles["body"]))
    story.append(Paragraph("How to read this pack", styles["h3"]))
    for step in how_to_read(pack):
        story.append(Paragraph(xml_escape("• " + step), styles["note"]))
    return story


def _glossary_end(styles: dict[str, ParagraphStyle]) -> list[Any]:
    story = [
        Paragraph("Short glossary", styles["h2"]),
        Paragraph("Full scorecard definitions stay on the detailed national pack.", styles["note"]),
    ]
    rows = [[Paragraph("Term", styles["th"]), Paragraph("Meaning", styles["th"])]]
    for term, meaning in GLOSSARY:
        rows.append([Paragraph(xml_escape(term), styles["td"]), Paragraph(xml_escape(meaning), styles["gloss"])])
    table = Table(rows, colWidths=[55 * mm, 212 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, WASH]),
            ]
        )
    )
    story.append(table)
    return story


def _pdf_table(df: pd.DataFrame, styles: dict[str, ParagraphStyle], usable: float) -> Table:
    if df is None or df.empty:
        return Paragraph("No rows at this layer for this period.", styles["note"])
    headers = [str(c) for c in df.columns]
    widths = _col_widths(headers, usable)
    data = [[Paragraph(xml_escape(h.replace(" this period", "").replace(" this month", "")), styles["th"]) for h in headers]]
    tones: list[str] = [""]
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row.get(h)
            text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
            style = styles["td"] if h in TEXT_COLS else styles["td_right"]
            cells.append(Paragraph(xml_escape(text).replace("\n", "<br/>"), style))
        data.append(cells)
        sit = str(row.get("Situation") or "")
        if sit == "Lagging":
            tones.append("lag")
        elif sit == "Ahead":
            tones.append("ahead")
        else:
            tones.append("")
    table = Table(data, colWidths=widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, tone in enumerate(tones):
        if i == 0:
            continue
        if tone == "lag":
            cmds.append(("BACKGROUND", (0, i), (-1, i), LAG))
        elif tone == "ahead":
            cmds.append(("BACKGROUND", (0, i), (-1, i), AHEAD))
        elif i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), WASH))
    table.setStyle(TableStyle(cmds))
    return table


def _col_widths(headers: list[str], usable: float) -> list[float]:
    n = len(headers)
    if n == 0:
        return [usable]
    weights = []
    for h in headers:
        if h in {"Do this", "Do this week", "Step"}:
            weights.append(3.2)
        elif h in {"Shop", "Distributor", "DSR", "Area"}:
            weights.append(1.6)
        elif h in {"Situation", "Label", "Driver", "Call", "Who", "Plan"}:
            weights.append(1.3)
        elif "Projected" in h or "Monthly target" in h:
            weights.append(1.15)
        else:
            weights.append(1.0)
    total = sum(weights) or 1.0
    return [usable * w / total for w in weights]
