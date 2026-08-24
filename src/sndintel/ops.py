"""Monday / DSR / Friday operating packs on top of the this-week engine.

Monday — NSM one-pager: hole, driver, whales, DSR labels, Ask KG.
DSR beat — capacity-capped call list with owner and ask.
Friday — closed loop: listed → visited → billed vs Ask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import re
from typing import Any

import pandas as pd

from sndintel.action import (
    ACTION_CALL,
    ACTION_CONVERT,
    ACTION_HOLD,
    ACTION_LIFT,
    ACTION_RECOVER,
    ActionPack,
    _kg_text,
    _round_kg,
)
from sndintel.capacity import (
    cap_shops_per_dsr,
    score_dsr_capacity,
    visit_quality_warnings,
    whale_shops,
)
from sndintel.monday import (
    HIGHLIGHT_DIST_N,
    HIGHLIGHT_STORE_N,
    SUMMARY_NOTE,
    city_action_table,
    city_driver_table,
    country_action_table,
    distributor_action_table,
    distributors_in_city,
    operating_shops,
    store_table,
    who_to_push_table,
    anchor_id,
)
from sndintel.config import DSR_DAY_CAP, EXPECTED_FORMULA
from sndintel.identity import dsr_display_name

MONDAY_WHALES = 30
MONDAY_DSRS = 15


@dataclass
class OpsPack:
    period: str
    label: str = ""
    kind: str = "monday"
    headline: str = ""
    sheets: list[tuple[str, str, str, pd.DataFrame]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exec_situation: list[str] = field(default_factory=list)
    exec_focus: list[dict[str, str]] = field(default_factory=list)
    exec_error: str = ""
    exec_model: str = ""


def build_monday_pack(
    action: ActionPack,
    units: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
    exec_summary: dict[str, Any] | None = None,
) -> OpsPack:
    """NSM Monday: summary (country → cities → DSR → dist → stores) then city/store detail."""
    shops = action.raw_shops if action.raw_shops is not None and not action.raw_shops.empty else pd.DataFrame()
    warnings = visit_quality_warnings(units, visits, action.period)
    cap = score_dsr_capacity(shops, action.as_of_day, action.days_in_month, action.days_left) if not shops.empty else pd.DataFrame()
    whales = whale_shops(shops, n=MONDAY_WHALES) if not shops.empty else pd.DataFrame()
    drivers = city_driver_table(units, shops) if units is not None else pd.DataFrame()
    headline = action.headline or f"{action.label}: Monday dispatch"
    work = operating_shops(shops)
    city_actions = city_action_table(shops)
    highlighted_dist = distributor_action_table(shops, n=HIGHLIGHT_DIST_N)
    highlighted_stores = store_table(whales if whales is not None and not whales.empty else work.head(HIGHLIGHT_STORE_N))
    sheets = [
        (
            "01 Country",
            "1. Country",
            SUMMARY_NOTE,
            country_action_table(shops) if not shops.empty else (action.country if action.country is not None else pd.DataFrame()),
        ),
        (
            "02 City drivers",
            "2. City drivers",
            "Unbilled = conversion. Unvisited = coverage. Drop = order size. Do not send coverage actions into a 100% visit city.",
            drivers,
        ),
        (
            "03 City actions",
            "3. City-wise required actions",
            "Click a city in the PDF to jump to that city's distributor list. " + SUMMARY_NOTE,
            city_actions,
        ),
        (
            "04 Who to push",
            "4. Who to push — DSR",
            "Overloaded = headcount. Not working = effort. Not converting / not lifting = skill or commercial. Fine = leave them.",
            who_to_push_table(cap, shops, n=MONDAY_DSRS),
        ),
        (
            "05 Highlighted distributors",
            "5. Highlighted distributors",
            "Highest rest-of-month Ask. Click a distributor in the PDF to jump to its shops. " + SUMMARY_NOTE,
            highlighted_dist,
        ),
        (
            "06 Highlighted stores",
            "6. Highlighted stores",
            "Volume doors (AMS or last drop ≥ 1 MT). Shop figures are KG.",
            highlighted_stores,
        ),
    ]
    cities = []
    if city_actions is not None and not city_actions.empty and "City" in city_actions.columns:
        cities = [str(c) for c in city_actions["City"].tolist()]
    elif not shops.empty and "city" in shops.columns:
        cities = sorted(shops["city"].astype(str).unique())
    for city in cities:
        dist_tbl = distributors_in_city(shops, city)
        if dist_tbl is None or dist_tbl.empty:
            continue
        sheets.append(
            (
                f"C {city}",
                f"{city} — distributors",
                f"Every distributor in {city} with rest-of-month Ask. " + SUMMARY_NOTE,
                dist_tbl,
            )
        )
    dist_order = []
    if highlighted_dist is not None and not highlighted_dist.empty:
        dist_order = [str(d) for d in highlighted_dist["Distributor"].tolist()]
    if not shops.empty and "distributor" in shops.columns:
        by_ask = (
            shops.assign(_ask=pd.to_numeric(shops.get("week_target_mt"), errors="coerce").fillna(0))
            .groupby(shops["distributor"].astype(str))["_ask"]
            .sum()
            .sort_values(ascending=False)
        )
        for name in by_ask.index.astype(str):
            if name not in dist_order:
                dist_order.append(name)
    for dist in dist_order:
        part = work[work["distributor"].astype(str) == str(dist)] if not work.empty and "distributor" in work.columns else pd.DataFrame()
        if part is None or part.empty:
            continue
        sheets.append(
            (
                f"S {dist}",
                f"{dist} — stores",
                "Doors with rest-of-month Ask. DSR with the higher Ask is grouped first; shops inside a DSR by Ask. KG with thousands separators.",
                store_table(part),
            )
        )
    pack = OpsPack(period=action.period, label=action.label, kind="monday", headline=headline, sheets=sheets, warnings=warnings)
    if exec_summary:
        from sndintel.briefing import _parse_exec_focus, _parse_exec_list

        pack.exec_situation = _parse_exec_list(exec_summary.get("situation_json") or exec_summary.get("situation"))
        pack.exec_focus = _parse_exec_focus(exec_summary.get("focus_json") or exec_summary.get("focus"))
        pack.exec_model = str(exec_summary.get("model") or "")
        pack.exec_error = str(exec_summary.get("error") or "")
    return pack


def build_dsr_beat_pack(action: ActionPack, per_dsr: int | None = None) -> OpsPack:
    """Every DSR’s today-list, capped at what they can actually call."""
    shops = action.raw_shops if action.raw_shops is not None and not action.raw_shops.empty else pd.DataFrame()
    if shops.empty:
        return OpsPack(period=action.period, label=action.label, kind="dsr", headline="No DSR beat list.")
    work = shops[shops["action"].isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT, ACTION_RECOVER})].copy()
    coming = shops[shops.get("coming_due").fillna(False)] if "coming_due" in shops.columns else shops.iloc[0:0]
    work = pd.concat([work, coming], ignore_index=True)
    if "store_id" in work.columns:
        work = work.drop_duplicates("store_id")
    work = cap_shops_per_dsr(work, per_dsr=per_dsr or DSR_DAY_CAP)
    if "value_score" in work.columns:
        work = work.sort_values(["dsr_name", "city", "value_score"], ascending=[True, True, False])
    cap = int(per_dsr or DSR_DAY_CAP)
    headline = (
        f"DSR beat lists — {cap} doors each (capacity cap). "
        f"{len(work)} calls on the sheet. Waiting-list doors stay in the detailed action pack."
    )
    sheets = [
        (
            "01 Beat lists",
            "Capacity-capped calls",
            f"Max {cap} doors per DSR, ranked by Ask KG. Owner = DSR. Do this is the instruction.",
            _present_beat(work),
        )
    ]
    return OpsPack(period=action.period, label=action.label, kind="dsr", headline=headline, sheets=sheets)


def beat_owner_options(df: pd.DataFrame) -> list[str]:
    """Picker labels: display name · city · distributor so two Shahids never collide."""
    if df is None or df.empty:
        return []
    labels = []
    for _, row in df.iterrows():
        dsr = str(row.get("DSR") or "").strip()
        city = str(row.get("City") or "").strip()
        dist = str(row.get("Distributor") or "").strip()
        if not dsr:
            continue
        labels.append(f"{dsr} · {city} · {dist}".strip(" ·"))
    return sorted(set(labels))


def filter_beat_by_owner(df: pd.DataFrame, owner: str | None) -> pd.DataFrame:
    """Keep one DSR's doors. `owner` is `Name · City · Distributor` from beat_owner_options."""
    if df is None or df.empty or not owner:
        return df if df is not None else pd.DataFrame()
    parts = [p.strip() for p in str(owner).split("·")]
    dsr = parts[0] if parts else ""
    city = parts[1] if len(parts) > 1 else ""
    dist = parts[2] if len(parts) > 2 else ""
    out = df.copy()
    if "DSR" in out.columns and dsr:
        out = out[out["DSR"].astype(str).str.strip() == dsr]
    if "City" in out.columns and city:
        out = out[out["City"].astype(str).str.strip() == city]
    if "Distributor" in out.columns and dist:
        out = out[out["Distributor"].astype(str).str.strip() == dist]
    return out


def build_friday_pack(outcomes: pd.DataFrame, action: ActionPack | None = None) -> OpsPack:
    """Close the loop on the list we printed, not a new ranking."""
    if outcomes is None or outcomes.empty:
        headline = (
            "No closed-loop yet. Score the warehouse twice in the same month "
            "(a later MTD cut) and Friday will compare the previous list to new bills and visits."
        )
        empty = pd.DataFrame(
            [
                {
                    "Listed": 0,
                    "Visited after list": 0,
                    "Billed after list": 0,
                    "Not visited": 0,
                    "Visited · still unbilled": 0,
                    "Ask listed (KG)": 0,
                    "Billed vs ask (KG)": 0,
                }
            ]
        )
        return OpsPack(
            period=action.period if action else "",
            label=action.label if action else "",
            kind="friday",
            headline=headline,
            sheets=[("01 Close", "Friday close", headline, empty)],
        )
    summary = _outcome_summary(outcomes)
    by_action = _outcome_by_action(outcomes)
    misses = outcomes[outcomes["outcome"].isin({"not_visited", "visited_unbilled"})].copy()
    if "ask_mt" in misses.columns:
        misses = misses.sort_values("ask_mt", ascending=False).head(40)
    headline = str(summary.get("headline") or "Friday close")
    sheets = [
        ("01 Close", "Listed → visited → billed", "This is whether the list moved volume, not a new ranking.", summary["table"]),
        ("02 By action", "By Due / Convert / Lift / Lapsing", "Visited-unbilled is commercial. Not-visited is management.", by_action),
        ("03 Still open", "Doors still open from the list", "Highest Ask first.", _present_open(misses)),
    ]
    return OpsPack(
        period=str(outcomes["period"].iloc[0]) if "period" in outcomes.columns else "",
        label=action.label if action else "",
        kind="friday",
        headline=headline,
        sheets=sheets,
    )


def score_closed_loop(
    previous: pd.DataFrame,
    shop_month: pd.DataFrame,
    visits: pd.DataFrame | None,
    period: str,
    listed_at: str | None = None,
) -> pd.DataFrame:
    """Compare a previously listed action_shops snapshot to current billed / visits."""
    if previous is None or previous.empty or not period:
        return pd.DataFrame()
    prev = previous.copy()
    work = prev[prev.get("action", pd.Series(dtype=str)).isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT, ACTION_RECOVER})]
    if "Action" in prev.columns and work.empty:
        work = prev[prev["Action"].isin({ACTION_CALL, ACTION_CONVERT, ACTION_LIFT, ACTION_RECOVER})]
    if work.empty:
        work = prev
    billed_now = _period_billed(shop_month, period)
    visits_now = _period_visits(visits, period)
    rows = []
    stamp = listed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for _, row in work.iterrows():
        sid = str(row.get("store_id") or "")
        if not sid:
            continue
        billed_listed = float(pd.to_numeric(pd.Series([row.get("billed_mt")]), errors="coerce").fillna(0).iloc[0] or 0)
        visits_listed = float(pd.to_numeric(pd.Series([row.get("visits")]), errors="coerce").fillna(0).iloc[0] or 0)
        billed = float(billed_now.get(sid, 0.0) or 0)
        vis = float(visits_now.get(sid, 0.0) or 0)
        ask = float(pd.to_numeric(pd.Series([row.get("week_target_mt")]), errors="coerce").fillna(0).iloc[0] or 0)
        action = str(row.get("action") or row.get("Action") or "")
        gained = max(0.0, billed - billed_listed)
        if billed > billed_listed + 0.005:
            outcome = "billed"
        elif vis > visits_listed + 0.5:
            outcome = "visited_unbilled"
        else:
            outcome = "not_visited"
        rows.append(
            {
                "period": period,
                "listed_at": stamp,
                "store_id": sid,
                "store_name": row.get("store_name") or row.get("Shop") or sid,
                "city": row.get("city") or row.get("City") or "",
                "distributor": row.get("distributor") or row.get("Distributor") or "",
                "dsr_name": row.get("dsr_name") or row.get("DSR") or "",
                "action": action,
                "ask_mt": ask,
                "billed_mt_listed": billed_listed,
                "visits_listed": visits_listed,
                "billed_mt_now": billed,
                "visits_now": vis,
                "gained_mt": gained,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def persist_outcomes(conn, outcomes: pd.DataFrame) -> None:
    from sndintel.storage import replace_table

    conn.execute(
        """CREATE TABLE IF NOT EXISTS action_outcomes (
            period TEXT NOT NULL,
            listed_at TEXT NOT NULL,
            store_id TEXT NOT NULL,
            store_name TEXT,
            city TEXT,
            distributor TEXT,
            dsr_name TEXT,
            action TEXT,
            ask_mt REAL,
            billed_mt_listed REAL,
            visits_listed REAL,
            billed_mt_now REAL,
            visits_now REAL,
            gained_mt REAL,
            outcome TEXT,
            PRIMARY KEY (period, store_id, listed_at)
        )"""
    )
    if outcomes is None or outcomes.empty:
        return
    existing = None
    try:
        from sndintel.storage import read_sql

        existing = read_sql(conn, "SELECT * FROM action_outcomes")
    except Exception:
        existing = None
    if existing is not None and not existing.empty:
        frame = pd.concat([existing, outcomes], ignore_index=True)
        if "listed_at" in frame.columns and "store_id" in frame.columns:
            frame = frame.drop_duplicates(["period", "store_id", "listed_at"], keep="last")
    else:
        frame = outcomes
    replace_table(conn, "action_outcomes", frame)


def load_outcomes(conn, period: str | None = None) -> pd.DataFrame:
    from sndintel.storage import read_sql

    try:
        df = read_sql(conn, "SELECT * FROM action_outcomes")
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if period:
        part = df[df["period"].astype(str) == str(period)]
        if not part.empty:
            latest = str(part["listed_at"].max()) if "listed_at" in part.columns else None
            if latest:
                return part[part["listed_at"].astype(str) == latest].copy()
            return part.copy()
    return df


def excel_bytes(pack: OpsPack) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    from sndintel.briefing import NAVY, SLATE, _sheet_table

    buf = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "00 Cover"
    ws["A1"] = "SND Intelligence"
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=NAVY)
    title = {"monday": "Monday NSM pack", "dsr": "DSR beat lists", "friday": "Friday close"}.get(pack.kind, pack.kind)
    ws["A2"] = f"{title} · {pack.label}"
    ws["A2"].font = Font(name="Calibri", size=18, bold=True, color=NAVY)
    ws["A3"] = pack.headline or ""
    ws["A3"].font = Font(name="Calibri", size=11, italic=True, color=SLATE)
    ws.merge_cells("A3:H3")
    ws["A4"] = EXPECTED_FORMULA
    ws["A4"].font = Font(name="Calibri", size=9, color=SLATE)
    row = 6
    for warning in pack.warnings:
        ws.cell(row, 1, warning)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True)
        row += 1
    used: set[str] = {"00 Cover"}
    key_to_sheet: dict[str, str] = {}
    if pack.kind == "monday":
        _excel_monday_exec(wb, pack)
        used.add("00 Exec")
    for sheet, heading, note, df in pack.sheets:
        name = _unique_sheet(sheet, used)
        _sheet_table(wb, name, heading, note, df)
        key_to_sheet[sheet] = name
        used.add(name)
    if pack.kind == "monday":
        _excel_monday_links(wb, pack, key_to_sheet)
    wb.save(buf)
    return buf.getvalue()


def _unique_sheet(title: str, used: set[str]) -> str:
    base = re.sub(r"[\\/*?:\[\]]", " ", str(title)).strip()[:31] or "Sheet"
    name = base
    i = 2
    while name in used:
        suffix = f" {i}"
        name = (base[: 31 - len(suffix)] + suffix).strip()
        i += 1
    return name


def _excel_monday_exec(wb, pack: OpsPack) -> None:
    from openpyxl.styles import Alignment, Font

    from sndintel.briefing import NAVY, SLATE

    ws = wb.create_sheet("00 Exec", 1)
    ws["A1"] = "Executive summary"
    ws["A1"].font = Font(name="Calibri", size=16, bold=True, color=NAVY)
    ws["A2"] = pack.label or pack.period
    ws["A2"].font = Font(name="Calibri", size=11, italic=True, color=SLATE)
    if pack.exec_model and pack.exec_situation:
        ws["A3"] = f"Written from this period’s scorecards ({pack.exec_model})."
        ws["A3"].font = Font(name="Calibri", size=9, color=SLATE)
    row = 5
    ws.cell(row, 1, "Summary of current situation").font = Font(name="Calibri", size=13, bold=True, color=NAVY)
    row += 1
    paragraphs = pack.exec_situation or [
        pack.exec_error
        or "No national executive summary is stored for this period. Paste an OpenAI key on Upload files and rebuild."
    ]
    for para in paragraphs:
        ws.cell(row, 1, para)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row].height = 48
        row += 1
    row += 1
    ws.cell(row, 1, "Key focus areas").font = Font(name="Calibri", size=13, bold=True, color=NAVY)
    row += 1
    if pack.exec_focus:
        for i, item in enumerate(pack.exec_focus, start=1):
            title = item.get("title") or f"Focus {i}"
            why = item.get("why") or ""
            do = item.get("do") or ""
            ws.cell(row, 1, f"{i}. {title}").font = Font(name="Calibri", size=11, bold=True, color=NAVY)
            row += 1
            if why:
                ws.cell(row, 1, why)
                ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
                ws.cell(row, 1).alignment = Alignment(wrap_text=True)
                row += 1
            if do:
                ws.cell(row, 1, f"Do this week. {do}")
                ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
                ws.cell(row, 1).alignment = Alignment(wrap_text=True)
                row += 1
            row += 1
    else:
        ws.cell(row, 1, "Focus areas appear here after the national executive summary is generated.")
    ws.column_dimensions["A"].width = 110


def _excel_monday_links(wb, pack: OpsPack, key_to_sheet: dict[str, str]) -> None:
    from openpyxl.styles import Font

    city_sheet = {k[2:]: v for k, v in key_to_sheet.items() if k.startswith("C ")}
    dist_sheet = {k[2:]: v for k, v in key_to_sheet.items() if k.startswith("S ")}
    link_font = Font(name="Calibri", size=10, color="0563C1", underline="single")
    for key, sheet_name in key_to_sheet.items():
        ws = wb[sheet_name]
        header = [ws.cell(4, c).value for c in range(1, ws.max_column + 1)]
        if key == "03 City actions" and "City" in header:
            col = header.index("City") + 1
            for r in range(5, ws.max_row + 1):
                city = str(ws.cell(r, col).value or "")
                target = city_sheet.get(city)
                if target:
                    ws.cell(r, col).hyperlink = f"#'{target}'!A1"
                    ws.cell(r, col).font = link_font
        if key in {"05 Highlighted distributors"} or key.startswith("C "):
            if "Distributor" in header:
                col = header.index("Distributor") + 1
                for r in range(5, ws.max_row + 1):
                    dist = str(ws.cell(r, col).value or "")
                    target = dist_sheet.get(dist)
                    if target:
                        ws.cell(r, col).hyperlink = f"#'{target}'!A1"
                        ws.cell(r, col).font = link_font


def pdf_bytes(pack: OpsPack) -> bytes:
    if pack.kind == "monday":
        return _monday_pdf(pack)
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    from sndintel.action_report import _xml

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=f"SND Intelligence · {pack.kind} · {pack.label}",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#0F172A"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#0F172A"), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#334155"), leading=11)
    kind = {"monday": "Monday NSM pack", "dsr": "DSR beat lists", "friday": "Friday close"}.get(pack.kind, pack.kind)
    story = [
        Paragraph(f"SND Intelligence · {kind}", body),
        Paragraph(_xml(pack.headline or pack.label), title),
        Paragraph(_xml(EXPECTED_FORMULA), body),
    ]
    for warning in pack.warnings:
        story.append(Paragraph(_xml(warning), body))
    for i, (_sheet, heading, note, df) in enumerate(pack.sheets, start=1):
        story.append(Paragraph(f"{i}. {_xml(heading)}", h2))
        story.append(Paragraph(_xml(note), body))
        story.append(Spacer(1, 4))
        story.append(_pdf_table(df, body, max_rows=80))
    doc.build(story)
    return buf.getvalue()


def _monday_pdf(pack: OpsPack) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    from sndintel.action_report import _xml

    buf = BytesIO()
    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        buf,
        pagesize=pagesize,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        title=f"SND Intelligence · Monday NSM pack · {pack.label}",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#0F172A"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0F172A"), spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading2"], fontSize=10, textColor=colors.HexColor("#1D4ED8"), spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=7.5, textColor=colors.HexColor("#334155"), leading=10)
    exec_body = ParagraphStyle("eb", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#0F172A"), leading=14, spaceAfter=8)
    link = ParagraphStyle("lnk", parent=body, textColor=colors.HexColor("#1D4ED8"), leading=10)
    city_anchor = {k[2:]: anchor_id("city", k[2:]) for k, *_ in pack.sheets if str(k).startswith("C ")}
    dist_anchor = {k[2:]: anchor_id("dist", k[2:]) for k, *_ in pack.sheets if str(k).startswith("S ")}
    story = _monday_exec_flowables(pack, title, h2, exec_body, body)
    story.extend(
        [
            Paragraph("SND Intelligence · Monday NSM pack", body),
            Paragraph(_xml(pack.headline or pack.label), title),
            Paragraph(_xml(EXPECTED_FORMULA), body),
        ]
    )
    for warning in pack.warnings:
        story.append(Paragraph(_xml(warning), body))
    story.append(Paragraph("Contents", h2))
    story.append(Paragraph('<a href="#sec-summary" color="#1D4ED8"><u>Summary</u></a>', link))
    story.append(Paragraph('<a href="#sec-cities" color="#1D4ED8"><u>Distributors by city</u></a>', link))
    story.append(Paragraph('<a href="#sec-stores" color="#1D4ED8"><u>Stores by distributor</u></a>', link))
    for city, dest in city_anchor.items():
        story.append(Paragraph(f'<a href="#{dest}" color="#1D4ED8"><u>{_xml(city)}</u></a>', link))

    first_city = True
    first_store = True
    for key, heading, note, df in pack.sheets:
        if str(key)[:2].isdigit():
            if key.startswith("01"):
                story.append(Paragraph('<a name="sec-summary"/>Summary', h2))
            dest = None
            link_col = None
            link_map = None
            if key.startswith("03"):
                link_col, link_map = "City", city_anchor
            elif key.startswith("05"):
                link_col, link_map = "Distributor", dist_anchor
            story.append(Paragraph(_xml(heading), h2))
            story.append(Paragraph(_xml(note), body))
            story.append(Spacer(1, 3))
            story.append(_pdf_table(df, body, max_rows=None, link_col=link_col, link_map=link_map))
            continue
        if str(key).startswith("C "):
            if first_city:
                story.append(PageBreak())
                story.append(Paragraph('<a name="sec-cities"/>Distributors by city', h2))
                first_city = False
            city = str(key)[2:]
            dest = city_anchor.get(city, anchor_id("city", city))
            story.append(Paragraph(f'<a name="{dest}"/>{_xml(heading)}', h3))
            story.append(Paragraph(_xml(note), body))
            story.append(_pdf_table(df, body, max_rows=None, link_col="Distributor", link_map=dist_anchor))
            continue
        if str(key).startswith("S "):
            if first_store:
                story.append(PageBreak())
                story.append(Paragraph('<a name="sec-stores"/>Stores by distributor', h2))
                first_store = False
            dist = str(key)[2:]
            dest = dist_anchor.get(dist, anchor_id("dist", dist))
            story.append(Paragraph(f'<a name="{dest}"/>{_xml(heading)}', h3))
            story.append(Paragraph(_xml(note), body))
            story.append(_pdf_table(df, body, max_rows=None))

    def _on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#1D4ED8"))
        canvas.setFont("Helvetica", 8)
        text = "Top of report"
        x = pagesize[0] / 2
        y = 6 * mm
        canvas.drawCentredString(x, y, text)
        width = canvas.stringWidth(text, "Helvetica", 8)
        canvas.linkRect("top", "top", (x - width / 2 - 4, y - 3, x + width / 2 + 4, y + 11), relative=0, thickness=0)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.setFont("Helvetica", 7)
        canvas.drawRightString(pagesize[0] - 10 * mm, y, f"Page {doc_.page}")
        canvas.restoreState()

    def _on_first(canvas, doc_):
        canvas.bookmarkPage("top")
        _on_page(canvas, doc_)

    doc.build(story, onFirstPage=_on_first, onLaterPages=_on_page)
    return buf.getvalue()


def _monday_exec_flowables(pack: OpsPack, title, h2, exec_body, body) -> list:
    from reportlab.platypus import PageBreak, Paragraph, Spacer

    from sndintel.action_report import _xml

    story = [
        Paragraph('<a name="top"/>SND Intelligence · Monday NSM pack', body),
        Paragraph("Executive summary", title),
        Paragraph(_xml(pack.label or pack.period or "This period"), exec_body),
    ]
    if pack.exec_model and pack.exec_situation:
        story.append(
            Paragraph(
                f"Written from this period’s scorecards ({_xml(pack.exec_model)}). "
                "Every figure matches the tables that follow. Nothing here is estimated by the model.",
                body,
            )
        )
    story.append(Paragraph("Summary of current situation", h2))
    if pack.exec_situation:
        for para in pack.exec_situation:
            story.append(Paragraph(_xml(para), exec_body))
            story.append(Spacer(1, 6))
    elif pack.exec_error:
        story.append(
            Paragraph(
                "The national executive summary was not generated. "
                f"{_xml(pack.exec_error)} "
                "Paste an OpenAI key on Upload files and rebuild scorecards.",
                exec_body,
            )
        )
    else:
        story.append(
            Paragraph(
                "No national executive summary is stored for this period. "
                "Paste an OpenAI API key on Upload files, then upload data or rebuild scorecards. "
                "The same key used for the national pack is reused here.",
                exec_body,
            )
        )
    story.append(PageBreak())
    story.append(Paragraph("Key focus areas", h2))
    if pack.exec_focus:
        for i, item in enumerate(pack.exec_focus, start=1):
            title_t = _xml(item.get("title") or f"Focus {i}")
            why = _xml(item.get("why") or "")
            do = _xml(item.get("do") or "")
            bits = [f"<b>{i}. {title_t}</b>"]
            if why:
                bits.append(why)
            if do:
                bits.append(f"<b>Do this week.</b> {do}")
            story.append(Paragraph("<br/>".join(bits), exec_body))
            story.append(Spacer(1, 8))
    else:
        story.append(
            Paragraph(
                "Focus areas appear here after the national executive summary is generated on Upload files.",
                exec_body,
            )
        )
    story.append(PageBreak())
    return story


def _pdf_table(df: pd.DataFrame, style, max_rows: int | None = 40, link_col: str | None = None, link_map: dict[str, str] | None = None):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    from sndintel.action_report import _xml

    if df is None or df.empty:
        return Paragraph("No rows at this layer.", style)
    show = df if max_rows is None else df.head(max_rows)
    header = [Paragraph(f"<b>{_xml(str(c))}</b>", style) for c in show.columns]
    data = [header]
    for _, rec in show.iterrows():
        cells = []
        for c in show.columns:
            raw = rec[c]
            text = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw)
            if link_col and link_map and str(c) == str(link_col):
                dest = link_map.get(text)
                if dest:
                    cells.append(Paragraph(f'<a href="#{dest}" color="#1D4ED8"><u>{_xml(text)}</u></a>', style))
                    continue
            cells.append(Paragraph(_xml(text), style))
        data.append(cells)
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _city_driver_table(units: pd.DataFrame) -> pd.DataFrame:
    if units is None or units.empty:
        return pd.DataFrame()
    cities = units[units["grain"] == "city"].copy()
    if cities.empty:
        return pd.DataFrame()
    rec = pd.to_numeric(cities.get("isolated_mt"), errors="coerce").fillna(0).clip(upper=0).abs()
    unb = pd.to_numeric(cities.get("from_unbilled_mt"), errors="coerce").fillna(0)
    unv = pd.to_numeric(cities.get("from_unvisited_mt"), errors="coerce").fillna(0)
    drop = pd.to_numeric(cities.get("from_drop_size_mt"), errors="coerce").fillna(0)
    driver = []
    for u, v, d in zip(unb, unv, drop):
        parts = [("unbilled", float(u)), ("unvisited", float(v)), ("drop size", float(d))]
        parts.sort(key=lambda x: abs(x[1]), reverse=True)
        driver.append(parts[0][0] if parts[0][1] else "on expected")
    out = pd.DataFrame(
        {
            "City": cities["grain_id"].astype(str),
            "Billed (MT)": pd.to_numeric(cities.get("volume_mt"), errors="coerce").round(0),
            "Expected (MT)": pd.to_numeric(cities.get("expected_mt"), errors="coerce").round(0),
            "Gap (MT)": rec.round(0),
            "Visit %": (pd.to_numeric(cities.get("visit_rate"), errors="coerce") * 100).round(0),
            "Strike %": (pd.to_numeric(cities.get("strike_rate"), errors="coerce") * 100).round(0),
            "Driver": driver,
        }
    )
    return out.sort_values("Gap (MT)", ascending=False)


def _present_capacity(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["DSR", "City", "Distributor", "Label", "Universe", "Visit %", "Strike of visits %", "Span ×", "Ask rest of month (KG)", "Why"]
        )
    return pd.DataFrame(
        {
            "DSR": [dsr_display_name(v) for v in df.get("dsr_name", df.get("grain_id"))],
            "City": list(df.get("city", [])),
            "Distributor": list(df.get("distributor", [])),
            "Label": list(df.get("label", [])),
            "Universe": list(df.get("universe", [])),
            "Visit %": [None if pd.isna(v) else int(round(float(v) * 100)) for v in df.get("visit_rate", [])],
            "Strike of visits %": [None if pd.isna(v) else int(round(float(v) * 100)) for v in df.get("strike_of_visits", [])],
            "Span ×": [None if pd.isna(v) else round(float(v), 1) for v in df.get("span_unique", [])],
            "Ask rest of month (KG)": [_round_kg(v) for v in df.get("week_target_mt", [])],
            "Day cap": list(df.get("day_cap", [])),
            "Why": list(df.get("why", [])),
        }
    )


def _present_whale_ops(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Shop", "City", "DSR", "Distributor", "Ask rest of month (KG)", "AMS (KG)", "Call", "Do this"])
    name = df["store_name"] if "store_name" in df.columns else df.get("Shop")
    return pd.DataFrame(
        {
            "Shop": list(name),
            "City": list(df.get("city", [])),
            "DSR": list(df.get("dsr_name", [])),
            "Distributor": list(df.get("distributor", [])),
            "Ask rest of month (KG)": [_round_kg(v) for v in df.get("week_target_mt", [])],
            "AMS (KG)": [_round_kg(v) for v in df.get("ams_3m", [])],
            "Billed (KG)": [_round_kg(v) for v in df.get("billed_mt", df.get("volume_mt", []))],
            "Call": list(df.get("call_status", df.get("Call", []))),
            "Do this": list(df.get("instruction", [])),
        }
    )


def _present_beat(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "DSR": list(df.get("dsr_name", [])),
            "City": list(df.get("city", [])),
            "Distributor": list(df.get("distributor", [])),
            "Shop": list(df.get("store_name", [])),
            "Action": list(df.get("action", [])),
            "Ask rest of month (KG)": [_round_kg(v) for v in df.get("week_target_mt", [])],
            "Next order (KG)": [_round_kg(v) for v in df.get("next_drop_mt", [])],
            "Days since bill": [None if pd.isna(v) else int(round(float(v))) for v in df.get("days_since_bill", [])],
            "Cover left (days)": [None if pd.isna(v) else int(round(float(v))) for v in df.get("cover_left_days", [])],
            "Owner": list(df.get("dsr_name", [])),
            "Do this": list(df.get("instruction", [])),
        }
    )
    return out


def _period_billed(shop_month: pd.DataFrame, period: str) -> dict[str, float]:
    if shop_month is None or shop_month.empty:
        return {}
    cur = shop_month[shop_month["period"].astype(str) == str(period)]
    if cur.empty:
        return {}
    g = cur.groupby(cur["store_id"].astype(str))["volume_mt"].sum()
    return {str(k): float(v) for k, v in g.items()}


def _period_visits(visits: pd.DataFrame | None, period: str) -> dict[str, float]:
    if visits is None or visits.empty or "store_id" not in visits.columns:
        return {}
    v = visits.copy()
    if "period" in v.columns:
        v = v[v["period"].astype(str) == str(period)]
    if v.empty:
        return {}
    col = "visits" if "visits" in v.columns else None
    if col is None:
        return {str(s): 1.0 for s in v["store_id"].astype(str)}
    g = v.groupby(v["store_id"].astype(str))[col].sum()
    return {str(k): float(v) for k, v in g.items()}


def _outcome_summary(outcomes: pd.DataFrame) -> dict[str, Any]:
    n = int(len(outcomes))
    billed = int((outcomes["outcome"] == "billed").sum())
    vis_u = int((outcomes["outcome"] == "visited_unbilled").sum())
    skip = int((outcomes["outcome"] == "not_visited").sum())
    ask = float(pd.to_numeric(outcomes.get("ask_mt"), errors="coerce").fillna(0).sum())
    gained = float(pd.to_numeric(outcomes.get("gained_mt"), errors="coerce").fillna(0).sum())
    headline = (
        f"{billed} of {n} listed doors billed after the list "
        f"({_kg_text(gained)} against {_kg_text(ask)} Ask). "
        f"{skip} were not visited — management. {vis_u} were visited and still unbilled — commercial."
    )
    table = pd.DataFrame(
        [
            {
                "Listed": n,
                "Billed after list": billed,
                "Visited · still unbilled": vis_u,
                "Not visited": skip,
                "Ask listed (KG)": _round_kg(ask),
                "Billed vs ask (KG)": _round_kg(gained),
            }
        ]
    )
    return {"headline": headline, "table": table}


def _outcome_by_action(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, g in outcomes.groupby(outcomes["action"].astype(str)):
        rows.append(
            {
                "Action": action,
                "Listed": int(len(g)),
                "Billed": int((g["outcome"] == "billed").sum()),
                "Visited · unbilled": int((g["outcome"] == "visited_unbilled").sum()),
                "Not visited": int((g["outcome"] == "not_visited").sum()),
                "Ask (KG)": _round_kg(g["ask_mt"].sum()) if "ask_mt" in g.columns else 0,
                "Gained (KG)": _round_kg(g["gained_mt"].sum()) if "gained_mt" in g.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def _present_open(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Shop", "DSR", "City", "Action", "Outcome", "Ask (KG)"])
    return pd.DataFrame(
        {
            "Shop": list(df.get("store_name", [])),
            "DSR": list(df.get("dsr_name", [])),
            "City": list(df.get("city", [])),
            "Action": list(df.get("action", [])),
            "Outcome": list(df.get("outcome", [])),
            "Ask (KG)": [_round_kg(v) for v in df.get("ask_mt", [])],
        }
    )
