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

from sndintel.briefing import GLOSSARY, StrategyPack, iter_report_sheets

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
    "Fair share of country (MT)": "Fair share<br/>(MT)",
    "Fair share of this city (MT)": "Fair share<br/>(MT)",
    "Fair share of its city (MT)": "Fair share<br/>(MT)",
    "Recoverable (MT)": "Recoverable<br/>(MT)",
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
    "Zone",
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
    story.append(PageBreak())
    story.append(Paragraph("Glossary", styles["h2"]))
    for term, meaning in GLOSSARY:
        story.append(Paragraph(f"<b>{xml_escape(term)}</b> — {xml_escape(meaning)}", styles["gloss"]))
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
        canvas.drawString(10 * mm, 5 * mm, "Figures in MT are rounded to whole numbers. From drop / unvisited / unbilled add to Recoverable.")
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
        "kpi_lab": ParagraphStyle(
            "kpi_lab",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=SLATE,
            alignment=TA_LEFT,
        ),
        "kpi_val": ParagraphStyle(
            "kpi_val",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=NAVY,
            alignment=TA_LEFT,
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


def _fmt_kpi(val: Any, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        n = int(round(float(val)))
    except (TypeError, ValueError):
        return str(val)
    return f"{n:+,}" if signed else f"{n:,}"


def _cover_flowables(pack: StrategyPack, styles: dict[str, ParagraphStyle], detailed: bool = False) -> list[Any]:
    k = pack.kpis or {}
    kicker = "DETAILED PACK" if detailed else "STRATEGY PACK"
    if pack.scope and pack.scope != "national":
        kicker = f"{pack.scope.upper()} PACK"
    story: list[Any] = [
        Paragraph(kicker, styles["kicker"]),
        Paragraph(xml_escape(pack.label or "Scorecards"), styles["h1"]),
        Paragraph(xml_escape(pack.headline or "Scorecards ready"), styles["headline"]),
        Paragraph(xml_escape(pack.weather or ""), styles["body"]),
        Paragraph(f"<b>The problem.</b> {xml_escape(pack.problem or '')}", styles["body"]),
        Paragraph(f"<b>Do this week.</b> {xml_escape(pack.action or '')}", styles["body"]),
        Spacer(1, 8),
    ]
    kpi_rows = [
        ("Billed (MT)", _fmt_kpi(k.get("billed_mt"))),
        ("Expected (MT)", _fmt_kpi(k.get("expected_mt"))),
        ("Gap vs expected", _fmt_kpi(k.get("gap_mt"), signed=True)),
        ("Extra hole after weather", _fmt_kpi(k.get("extra_hole_mt"), signed=True)),
        ("Lagging cities", str(k.get("n_lagging_cities") or 0)),
    ]
    kpi_data = [
        [Paragraph(xml_escape(a), styles["kpi_lab"]) for a, _ in kpi_rows],
        [Paragraph(xml_escape(b), styles["kpi_val"]) for _, b in kpi_rows],
    ]
    kpi_table = Table(kpi_data, colWidths=[36 * mm] * 5)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WASH),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("How to read this pack", styles["headline"]))
    if pack.scope == "city":
        steps = [
            "City scorecard versus the country. Recoverable is the local hole after national weather.",
            "Every distributor in this city with AMS greater than 0.",
            "Every DSR in this city with AMS greater than 0.",
            "Shops in this city with recoverable greater than 0.25 MT.",
        ]
    elif pack.scope == "distributor":
        steps = [
            "Distributor scorecard versus its city.",
            "DSRs on this distributor’s doors.",
            "Shops under this distributor with recoverable greater than 0.25 MT.",
        ]
    elif pack.scope == "dsr":
        steps = [
            "DSR scorecard versus its city.",
            "Shops on this beat with recoverable greater than 0.25 MT.",
        ]
    elif detailed:
        steps = [
            "City detail — every city, highest recoverable first.",
            "Distributor detail — every distributor with AMS greater than 0.",
            "DSR detail — every DSR with AMS greater than 0.",
            "National shops — every door with recoverable greater than 0.25 MT.",
        ]
    else:
        steps = [
            "Country by city — every city versus national weather. Highest recoverable first.",
            "Lagging cities → distributors — first calls. AMS = 0 is hidden.",
            "Those distributors → shops with recoverable greater than 0.25 MT.",
            "Every lagging distributor (AMS > 0), including cities that are not national exceptions.",
            "Every lagging DSR (AMS > 0).",
            "Every shop with recoverable greater than 0.25 MT (shallower doors rolled into the last row).",
        ]
    for i, step in enumerate(steps, start=1):
        story.append(Paragraph(f"<b>{i}.</b>  {xml_escape(step)}", styles["body"]))
    return story


def _table_flowable(df: pd.DataFrame, styles: dict[str, ParagraphStyle], usable: float) -> Any:
    if df is None or df.empty:
        return Paragraph("No rows at this layer for this period.", styles["note"])
    cols = list(df.columns)
    widths = _col_widths(cols, usable)
    header = [Paragraph(HEADER_ALIAS.get(c, xml_escape(str(c))), styles["th"]) for c in cols]
    data = [header]
    sit_col = "Situation" if "Situation" in cols else None
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
        sit = str(row[sit_col] or "") if sit_col else ""
        if sit == "Lagging":
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), LAG))
        elif sit == "Ahead":
            cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), AHEAD))
        elif sit == "Country":
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
        elif c in {"City", "Zone", "Beat", "Call", "Situation"}:
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
