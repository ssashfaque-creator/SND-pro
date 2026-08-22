"""This-week action pack: Excel, PDF, and HTML. One table per grain."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.worksheet import Worksheet

from sndintel.action import COUNTRY_PDF_COLS, DIST_PDF_COLS, DSR_PDF_COLS, SHOP_PDF_COLS, ActionPack
from sndintel.briefing import NAVY, SLATE, _excel_value, _sheet_table

GLOSSARY = [
    (
        "Expected this month",
        "Same full-month recipe as the scorecard: last three closed calendar months blended with the last-six-month median. The scorecard paces that run-rate by the national billed-by-day curve when the month is open; this pack still uses the full-month number so remaining-to-Expected is what the door still owes.",
    ),
    (
        "Usual cycle / typical drop",
        "Median days between billed days and median billed-day volume, shrunk shop → DSR → city so a thin history borrows the beat. Store Y that bills every 15 days is due when it has been ~15 days with no bill. Volumes print in KG.",
    ),
    (
        "Cover left",
        "Last drop ÷ daily run-rate, minus days since that bill. If last month was ≥1.6× AMS (two months of stock), leftover cover uses last-month volume instead — that door is less likely to buy now.",
    ),
    (
        "AMS",
        "Average of the three closed calendar months before this period (May + June + July when scoring August). Full-month run-rate, in KG. On every table so you can see the book next to the ask.",
    ),
    (
        "Still to Expected",
        "max(0, Expected − billed). What the door still owes the month, in KG.",
    ),
    (
        "Ask rest of month",
        "Typical drops that can still land before month-end, capped at remaining-to-Expected. Not the whole hole: day 22 with 9 days left cannot close 500 MT of full-month miss. Includes doors that are due now and doors that come due before month-end (once-a-month shops that last billed late last month). Zero when leftover cover lasts past month-end, or the door already hit Expected. Printed in KG.",
    ),
    (
        "Coming due",
        "Doors that are not due today but whose cycle or leftover cover runs out before month-end. They are most of a late-month country hole. Doors is today's work list only (Due / Due visited / Another visit / Lapsing).",
    ),
    (
        "Due / Due · visited / Another visit / Lapsing / Hold",
        "Due = cycle has elapsed, cover is gone, nobody visited. Due · visited = same, but the beat already called and still no bill. Another visit = billed this month but too little, and the cycle says they should have bought again (not a drop from yesterday). Lapsing = two cycles with no bill this month, or declining last-3 vs prior-3 and unbilled this cycle. Hold = leftover cover — do not pull the beat.",
    ),
    (
        "Backtest",
        "At day 15 of each closed month, mark shops whose cycle has elapsed and who no longer have cover. Precision is the share that billed in the next 14 days versus picking that many shops at random. Loaded hold = last month ≥1.6× AMS; quiet means they stayed below 0.25 MT for the rest of the month.",
    ),
]


def how_to_read(pack: ActionPack, detailed: bool = False) -> list[str]:
    day = f"Day {pack.as_of_day} of {pack.days_in_month}" if pack.as_of_day else pack.period
    if detailed:
        return [
            f"{day}. Full lists — every AMS > 0 distributor, DSR, and shop the engine scored.",
            "Distributors ranked by rest-of-month ask KG (not Gap tons).",
            "DSRs ranked the same way.",
            "Every shop with an action. Do this names the door and why it is due, light, lapsing, or hold.",
        ]
    return [
        f"{day}. Country: billed vs Expected is the hole (Still to Expected). Ask rest of month is drops that can still land — including Coming due, not only today's call list.",
        "One distributor push list — doors to work now, how many come due before month-end, and closable KG.",
        "One DSR push list — ride-with names, not nested under the distributors.",
        "Due — cycle elapsed, no leftover cover, not visited this month.",
        "Due · visited — same clock, already called, still no bill.",
        "Another visit — bought too little this month and the cycle says they should have bought again.",
        "Lapsing — long gap or declining volume. Unvisited overdue doors rank first.",
        "Backtest — whether 'due' shops actually billed in the next 14 days on closed months.",
    ]


def iter_action_sheets(pack: ActionPack, detailed: bool = False) -> list[tuple[str, str, str, pd.DataFrame]]:
    if detailed:
        return [
            (
                "01 Country",
                "Country this week",
                pack.headline or "Who is due, light, or lapsing, rolled to the country.",
                pack.country,
            ),
            (
                "02 Distributors",
                "Every distributor to push",
                "AMS = 0 is hidden. Ranked by rest-of-month ask KG.",
                pack.all_distributors,
            ),
            (
                "03 DSRs",
                "Every DSR to push",
                "National list, not nested under the distributors.",
                pack.all_dsrs,
            ),
            (
                "04 Shops",
                "Every scored shop",
                "Due / Due · visited / Another visit / Lapsing / Hold. Do this is the instruction.",
                pack.all_shops,
            ),
            (
                "05 Backtest",
                "Did due shops bill in the next 14 days?",
                "Walk-forward cut at day 15 of closed months.",
                pack.backtest,
            ),
        ]
    return [
        (
            "01 Country",
            "Country this week",
            pack.headline or "Who is due, light, or lapsing, rolled to the country.",
            pack.country,
        ),
        (
            "02 Distributors",
            "Push these distributors",
            "One list. Ranked by rest-of-month ask. Doors is the work count — Due can be 0 when those shops were already visited.",
            pack.distributors,
        ),
        (
            "03 DSRs",
            "Push these DSRs",
            "One national list — a DSR can appear even if its distributor is not in the fifteen.",
            pack.dsrs,
        ),
        (
            "04 Due",
            "Due and unvisited",
            "Usual cycle has elapsed, leftover cover is gone, nobody visited this month. Ask rest of month is the typical drop (or two if another cycle still fits).",
            pack.calls,
        ),
        (
            "05 Due visited",
            "Due · already visited",
            "Cycle says they should have billed. The beat already called. Close the order.",
            pack.converts,
        ),
        (
            "06 Another visit",
            "Bought too little this month",
            "Billed once (or a stub) and still short of Expected. Cycle says they should have bought again.",
            pack.lifts,
        ),
        (
            "07 Lapsing",
            "Quiet too long or declining",
            "Two cycles with no bill, or last-three months down ≥25% versus the three before and unbilled this cycle.",
            pack.lapses,
        ),
        (
            "08 Backtest",
            "Did due shops bill in the next 14 days?",
            "If daily history is thin this sheet stays empty. Rebuild after Outlet Date Wise is in the warehouse.",
            pack.backtest,
        ),
    ]


def excel_bytes(pack: ActionPack, detailed: bool = False) -> bytes:
    buf = BytesIO()
    write_excel(pack, buf, detailed=detailed)
    return buf.getvalue()


def write_excel(pack: ActionPack, path: Path | str | BytesIO, detailed: bool = False) -> None:
    wb = Workbook()
    _cover(wb, pack, detailed=detailed)
    for sheet, heading, note, df in iter_action_sheets(pack, detailed=detailed):
        kwargs: dict[str, Any] = {}
        if sheet.startswith("02") and df is not None and not df.empty and "Ask rest of month (KG)" in df.columns:
            cat = "Distributor" if "Distributor" in df.columns else "DSR"
            kwargs = dict(bar_col="Ask rest of month (KG)", cat_col=cat)
        _sheet_table(wb, sheet, heading, note, df, **kwargs)
    if path is not None:
        wb.save(path)


def html_bytes(pack: ActionPack, detailed: bool = False) -> bytes:
    return render_html(pack, detailed=detailed).encode("utf-8")


def render_html(pack: ActionPack, detailed: bool = False) -> str:
    import html as html_lib

    bits = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>",
        f"<title>This week · {html_lib.escape(pack.label)}</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;color:#0f172a}",
        "h1{font-size:22px;margin:0 0 6px}h2{font-size:16px;margin:28px 0 8px}",
        "p,li{color:#334155;max-width:920px}table{border-collapse:collapse;font-size:13px;margin:8px 0 24px}",
        "th{background:#0f172a;color:#fff;padding:6px 8px;text-align:left}td{border:1px solid #e2e8f0;padding:6px 8px}",
        ".note{color:#64748b;font-style:italic}</style></head><body>",
        "<div>SND Intelligence · this week</div>",
        f"<h1>{html_lib.escape(pack.headline or pack.label)}</h1>",
        "<h2>Glossary</h2><dl>",
    ]
    for term, meaning in GLOSSARY:
        bits.append(f"<dt><b>{html_lib.escape(term)}</b></dt><dd>{html_lib.escape(meaning)}</dd>")
    bits.append("</dl><h2>How to read</h2><ol>")
    for line in how_to_read(pack, detailed=detailed):
        bits.append(f"<li>{html_lib.escape(line)}</li>")
    bits.append("</ol>")
    for i, (_sheet, heading, note, df) in enumerate(iter_action_sheets(pack, detailed=detailed), start=1):
        bits.append(f"<h2>{i}. {html_lib.escape(heading)}</h2>")
        bits.append(f"<p class='note'>{html_lib.escape(note)}</p>")
        bits.append(_html_table(df))
    bits.append("</body></html>")
    return "".join(bits)


def pdf_bytes(pack: ActionPack, detailed: bool = False) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=10 * mm, rightMargin=10 * mm, topMargin=10 * mm, bottomMargin=10 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#0F172A"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, textColor=colors.HexColor("#0F172A"), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#334155"), leading=11)
    story = [
        Paragraph("SND Intelligence · this week", body),
        Paragraph(_xml(pack.headline or pack.label), title),
        Paragraph("Glossary", h2),
    ]
    for term, meaning in GLOSSARY:
        story.append(Paragraph(f"<b>{_xml(term)}.</b> {_xml(meaning)}", body))
    story.append(Paragraph("How to read", h2))
    for i, line in enumerate(how_to_read(pack, detailed=detailed), start=1):
        story.append(Paragraph(f"{i}. {_xml(line)}", body))
    for i, (_sheet, heading, note, df) in enumerate(iter_action_sheets(pack, detailed=detailed), start=1):
        story.append(Paragraph(f"{i}. {_xml(heading)}", h2))
        story.append(Paragraph(_xml(note), body))
        story.append(Spacer(1, 4))
        story.append(_pdf_table(df, body))
    doc.build(story)
    return buf.getvalue()


def _cover(wb: Workbook, pack: ActionPack, detailed: bool = False) -> Worksheet:
    ws = wb.active
    ws.title = "00 Cover"
    ws["A1"] = "SND Intelligence"
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=NAVY)
    ws["A2"] = f"{'Detailed action pack' if detailed else 'This week'} · {pack.label}"
    ws["A2"].font = Font(name="Calibri", size=18, bold=True, color=NAVY)
    ws["A3"] = pack.headline or "Call list from each door’s purchase cycle and leftover cover."
    ws["A3"].font = Font(name="Calibri", size=11, italic=True, color=SLATE)
    ws.merge_cells("A3:H3")
    row = 5
    ws.cell(row, 1, "Glossary")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=NAVY)
    row += 1
    for term, meaning in GLOSSARY:
        ws.cell(row, 1, term)
        ws.cell(row, 1).font = Font(bold=True, color=NAVY)
        ws.cell(row, 2, meaning)
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=8)
        ws.cell(row, 2).alignment = Alignment(wrap_text=True)
        ws.row_dimensions[row].height = 36
        row += 1
    row += 1
    ws.cell(row, 1, "How to read")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=NAVY)
    row += 1
    for i, line in enumerate(how_to_read(pack, detailed=detailed), start=1):
        ws.cell(row, 1, f"{i:02d}. {line}")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).alignment = Alignment(wrap_text=True)
        row += 1
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 80
    return ws


def _html_table(df: pd.DataFrame) -> str:
    import html as html_lib

    if df is None or df.empty:
        return "<p class='note'>No rows at this layer.</p>"
    head = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, rec in df.iterrows():
        tds = "".join(f"<td>{html_lib.escape('' if rec[c] is None or pd.isna(rec[c]) else str(rec[c]))}</td>" for c in df.columns)
        rows.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def pdf_columns(df: pd.DataFrame) -> list[str]:
    """Keep Ask / AMS / Billed / Lapsing on the PDF instead of the first eight fields."""
    if df is None or df.empty:
        return []
    if "Shop" in df.columns:
        wanted = SHOP_PDF_COLS
    elif "Distributor" in df.columns and "DSR" not in df.columns:
        wanted = DIST_PDF_COLS
    elif "DSR" in df.columns:
        wanted = DSR_PDF_COLS
    elif "Due · unvisited" in df.columns or "Ask rest of month (KG)" in df.columns:
        wanted = COUNTRY_PDF_COLS
    else:
        wanted = list(df.columns)
    return [c for c in wanted if c in df.columns]


def _pdf_table(df: pd.DataFrame, style) -> Any:
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    if df is None or df.empty:
        return Paragraph("No rows at this layer.", style)
    show = df.copy()
    keep = pdf_columns(show)
    if keep:
        show = show[keep]
    header = [Paragraph(f"<b>{_xml(str(c))}</b>", style) for c in show.columns]
    data = [header]
    for _, rec in show.head(40).iterrows():
        data.append([Paragraph(_xml("" if rec[c] is None or pd.isna(rec[c]) else str(rec[c])), style) for c in show.columns])
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
            ]
        )
    )
    return table


def _xml(text: str) -> str:
    from xml.sax.saxutils import escape

    return escape(str(text or ""))


# silence unused import warning if a linter flags _excel_value
_ = _excel_value
