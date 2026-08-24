"""Board-pack PDF for the strategy report.

Landscape A4, navy headers, whole-number MT, remarks as wrapped bullets
in the last column. Built with reportlab (no browser print step).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
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

from sndintel.briefing import (
    CALCULATION_NOTES,
    GLOSSARY,
    StrategyPack,
    how_to_read_steps,
    is_national_pack,
    iter_report_sheets,
    row_tone,
)
from sndintel.config import EXPECTED_FORMULA

NAVY = colors.HexColor("#0F172A")
SLATE = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
WASH = colors.HexColor("#F8FAFC")
RED = colors.HexColor("#B91C1C")
GREEN = colors.HexColor("#15803D")
LAG = colors.HexColor("#FEF2F2")
AHEAD = colors.HexColor("#F0FDF4")
COUNTRY = colors.HexColor("#E2E8F0")
WHITE = colors.white

HEADER_ALIAS = {
    "Billed this period (MT)": "Billed<br/>(MT)",
    "AMS last 3 months (MT)": "AMS 3m<br/>(MT)",
    "vs AMS (MT)": "vs AMS<br/>(MT)",
    "Same month last year (MT)": "LY same<br/>month",
    "Expected this month (MT)": "Expected<br/>(MT)",
    "Gap (MT)": "Gap<br/>(MT)",
    "Drop size (MT)": "Drop size<br/>(MT)",
    "From drop size (MT)": "From drop<br/>(MT)",
    "From unvisited shops (MT)": "From<br/>unvisited",
    "From unbilled shops (MT)": "From<br/>unbilled",
    "Billed shops": "Billed<br/>shops",
    "Visited shops": "Visited<br/>shops",
    "Strike %": "Strike<br/>%",
    "Visit %": "Visit<br/>%",
    "Visits MTD": "Visits<br/>MTD",
}

TEXT_COLS = {
    "City",
    "Distributor",
    "DSR",
    "Shop",
    "Beat",
    "Call",
    "Situation",
    "Remarks",
}
SIGNED_MT = {
    "vs AMS (MT)",
    "From drop size (MT)",
    "From unvisited shops (MT)",
    "From unbilled shops (MT)",
}


def render_pdf(pack: StrategyPack, detailed: bool = False) -> bytes:
    buf = BytesIO()
    write_pdf(pack, buf, detailed=detailed)
    return buf.getvalue()


def write_pdf(pack: StrategyPack, path: Path | str | BytesIO, detailed: bool = False) -> None:
    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        path if not isinstance(path, (str, Path)) else str(path),
        pagesize=pagesize,
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=14 * mm,
        bottomMargin=12 * mm,
        title=f"SND Intelligence · {pack.label}",
        author="SND Intelligence",
    )
    styles = _styles()
    story: list[Any] = []
    story.extend(_glossary_flowables(pack, styles, detailed=detailed))
    if is_national_pack(pack):
        story.append(PageBreak())
        story.extend(_exec_flowables(pack, styles))
    else:
        story.append(PageBreak())
        story.extend(_cover_flowables(pack, styles, detailed=detailed))
    usable = pagesize[0] - doc.leftMargin - doc.rightMargin
    for _sheet, heading, note, df in iter_report_sheets(pack, detailed=detailed):
        story.append(PageBreak())
        block = [
            Paragraph(xml_escape(heading), styles["h2"]),
            Paragraph(xml_escape(note or ""), styles["note"]),
            Spacer(1, 4),
            _table_flowable(df, styles, usable),
        ]
        story.append(KeepTogether(block[:2]))
        story.extend(block[2:])
    label = pack.label or ""
    scope = pack.scope_label or pack.scope or "national"

    def _on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, pagesize[1] - 9 * mm, pagesize[0], 9 * mm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(10 * mm, pagesize[1] - 6.2 * mm, "SND Intelligence")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            pagesize[0] - 10 * mm,
            pagesize[1] - 6.2 * mm,
            f"{scope}  ·  {label}  ·  {doc_.page}",
        )
        canvas.setFillColor(SLATE)
        canvas.setFont("Helvetica", 7)
        canvas.drawString(
            10 * mm,
            5 * mm,
            f"Figures in MT are whole numbers. From drop / unvisited / unbilled add to Gap. {EXPECTED_FORMULA}.",
        )
        canvas.drawRightString(pagesize[0] - 10 * mm, 5 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle(
            "kicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=SLATE,
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=NAVY,
            spaceAfter=6,
            leading=22,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=4,
            leading=14,
        ),
        "headline": ParagraphStyle(
            "headline",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=NAVY,
            leading=16,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=NAVY,
            leading=13,
            spaceAfter=4,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=SLATE,
            leading=11,
            spaceAfter=2,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            textColor=WHITE,
            leading=8.5,
            alignment=TA_LEFT,
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=NAVY,
            leading=9,
            alignment=TA_LEFT,
        ),
        "td_right": ParagraphStyle(
            "td_right",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=NAVY,
            leading=9,
            alignment=TA_RIGHT,
        ),
        "td_remarks": ParagraphStyle(
            "td_remarks",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=6.5,
            textColor=NAVY,
            leading=8.5,
            alignment=TA_LEFT,
        ),
        "gloss": ParagraphStyle(
            "gloss",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=SLATE,
            leading=11,
            spaceAfter=3,
        ),
    }


def _glossary_flowables(pack: StrategyPack, styles: dict[str, ParagraphStyle], detailed: bool = False) -> list[Any]:
    kicker = "NATIONAL PACK" if is_national_pack(pack) else f"{(pack.scope or 'report').upper()} PACK"
    if detailed and is_national_pack(pack):
        kicker = "DETAILED NATIONAL PACK"
    story: list[Any] = [
        Paragraph(kicker, styles["kicker"]),
        Paragraph(xml_escape(pack.label or "Scorecards"), styles["h1"]),
        Paragraph("Glossary", styles["h2"]),
        Paragraph(
            "Read this page first. Every later table uses these words. "
            "Figures in MT are whole numbers; drop size is two decimals. "
            "From drop / unvisited / unbilled add to Gap.",
            styles["note"],
        ),
        Spacer(1, 4),
    ]
    rows = [[Paragraph("Term", styles["th"]), Paragraph("Meaning", styles["th"])]]
    for term, meaning in GLOSSARY:
        rows.append(
            [
                Paragraph(xml_escape(term), styles["td"]),
                Paragraph(xml_escape(meaning), styles["gloss"]),
            ]
        )
    gloss = Table(rows, colWidths=[55 * mm, 212 * mm])
    gloss.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BACKGROUND", (0, 1), (-1, -1), WHITE),
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
    story.append(gloss)
    story.append(Spacer(1, 10))
    story.append(Paragraph("How the figures are calculated", styles["h2"]))
    for term, meaning in CALCULATION_NOTES:
        story.append(Paragraph(f"<b>{xml_escape(term)}.</b> {xml_escape(meaning)}", styles["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("How to read the tables that follow", styles["h2"]))
    for i, step in enumerate(how_to_read_steps(pack, detailed=detailed), start=1):
        story.append(Paragraph(f"<b>{i}.</b>  {xml_escape(step)}", styles["body"]))
    warnings = getattr(pack, "visit_warnings", None) or []
    if warnings:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Visit file quality", styles["h2"]))
        for warning in warnings:
            story.append(Paragraph(xml_escape(warning), styles["body"]))
    return story


def _exec_flowables(pack: StrategyPack, styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = [
        Paragraph("EXECUTIVE SUMMARY · NATIONAL", styles["kicker"]),
        Paragraph(xml_escape(pack.label or "Scorecards"), styles["h1"]),
        Paragraph("Executive summary", styles["h2"]),
    ]
    if pack.exec_model and pack.exec_situation:
        story.append(
            Paragraph(
                f"Written from this period’s scorecards ({xml_escape(pack.exec_model)}). "
                "Every figure matches the tables that follow. Nothing here is estimated by the model.",
                styles["note"],
            )
        )
    if pack.exec_situation:
        story.append(Paragraph("Summary of current situation", styles["h3"]))
        for para in pack.exec_situation:
            story.append(Paragraph(xml_escape(para), styles["body"]))
        if pack.exec_focus:
            story.append(Paragraph("Key focus areas", styles["h3"]))
            for i, item in enumerate(pack.exec_focus, start=1):
                title = xml_escape(item.get("title") or f"Focus {i}")
                why = xml_escape(item.get("why") or "")
                do = xml_escape(item.get("do") or "")
                bits = [f"<b>{i}. {title}</b>"]
                if why:
                    bits.append(why)
                if do:
                    bits.append(f"<b>Do this week.</b> {do}")
                story.append(Paragraph("<br/>".join(bits), styles["body"]))
                story.append(Spacer(1, 4))
        return story
    if pack.exec_error:
        story.append(
            Paragraph(
                "The national executive summary was not generated. "
                f"{xml_escape(pack.exec_error)} "
                "Paste an OpenAI key on Upload files and rebuild scorecards (or use Generate on that page).",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                "No national executive summary is stored for this period. "
                "Paste an OpenAI API key on Upload files, then upload data or rebuild scorecards. "
                "The model is given the same rounded country, city, distributor, DSR, and shop figures as this pack.",
                styles["body"],
            )
        )
    return story


def _cover_flowables(pack: StrategyPack, styles: dict[str, ParagraphStyle], detailed: bool = False) -> list[Any]:
    kicker = f"{(pack.scope or 'report').upper()} PACK"
    story: list[Any] = [
        Paragraph(kicker, styles["kicker"]),
        Paragraph(xml_escape(pack.scope_label or pack.label or "Scorecards"), styles["h1"]),
        Paragraph(xml_escape(pack.headline or "Scorecards ready"), styles["headline"]),
    ]
    if pack.weather:
        story.append(Paragraph(xml_escape(pack.weather), styles["body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("How to read this pack", styles["headline"]))
    for i, step in enumerate(how_to_read_steps(pack, detailed=detailed), start=1):
        story.append(Paragraph(f"<b>{i}.</b>  {xml_escape(step)}", styles["body"]))
    return story


def _table_flowable(df: pd.DataFrame, styles: dict[str, ParagraphStyle], usable: float) -> Any:
    if df is None or df.empty:
        return Paragraph("No rows at this layer for this period.", styles["note"])
    cols = list(df.columns)
    widths = _col_widths(cols, usable)
    header = [Paragraph(HEADER_ALIAS.get(c, xml_escape(str(c))), styles["th"]) for c in cols]
    data = [header]
    for _, row in df.iterrows():
        cells = [_cell(row[c], c, styles) for c in cols]
        data.append(cells)
    tbl = Table(data, colWidths=widths, repeatRows=1)
    cmds: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
    ]
    for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
        sit = row_tone(row)
        if sit == "lagging":
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), LAG))
        elif sit == "ahead":
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), AHEAD))
        elif sit == "country":
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), COUNTRY))
        elif r_idx % 2 == 0:
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), WASH))
    tbl.setStyle(TableStyle(cmds))
    return tbl


def _col_widths(cols: list[str], usable: float) -> list[float]:
    weights = []
    for c in cols:
        if c == "Remarks":
            weights.append(3.6)
        elif c in {"Shop"}:
            weights.append(2.2)
        elif c in {"Distributor", "DSR"}:
            weights.append(1.6)
        elif c in {"City", "Beat", "Call", "Situation"}:
            weights.append(1.15)
        elif "(MT)" in c:
            weights.append(0.85)
        else:
            weights.append(0.7)
    total = sum(weights) or 1.0
    return [usable * w / total for w in weights]


def _cell(val: Any, col: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    name = str(col)
    if name == "Remarks":
        text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
        lines = [xml_escape(x) for x in text.split("\n") if x.strip()]
        html = "<br/>".join(lines) if lines else "—"
        return Paragraph(html, styles["td_remarks"])
    style = styles["td"] if name in TEXT_COLS else styles["td_right"]
    return Paragraph(xml_escape(_pdf_cell_text(val, name)), style)


def _pdf_cell_text(val: Any, col: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (int, float)) and col == "Drop size (MT)":
        return f"{float(val):.2f}"
    if isinstance(val, (int, float)) and "(MT)" in col:
        n = int(round(float(val)))
        if col in SIGNED_MT or col.startswith("vs ") or col.startswith("From "):
            return f"{n:+,}"
        return f"{n:,}"
    if isinstance(val, (int, float)) and (col.endswith("%") or "Strike" in col):
        return f"{int(round(float(val)))}"
    if isinstance(val, (int, float)) and col in {"Billed shops", "Visited shops", "Universe", "Visits MTD"}:
        return f"{int(round(float(val))):,}"
    if isinstance(val, float):
        return f"{int(round(val)):,}"
    return str(val)
