"""Shop-wise issues pack: every door in a scope, MTD or a closed month.

Does not invent Expected, Ask, last bill, or Target. Those come from the
existing action shop row (last-3 / last-6 Expected, 90-day Ask / usual drop,
last billed date and drop) plus the matched shop-plan quota.

Closed month and open MTD share one skeleton (KPIs → issue mix → shop list)
and differ in what counts as an issue:

- Closed month is a result. The hole is Expected − billed. Issues are shops
  that missed that Expected (light orders, visited with no bill, unvisited,
  lapsed). Sort by hole. Comment is next month, never “this week”. Ask is 0
  on a closed month, so the next-order number is the usual drop.
- Open MTD is the beat. Full-month Expected − billed so far is not a miss
  mid-month, so it is not the issue list. Issues are due this week (Ask),
  visited with no bill, and lost doors. Comment is the existing this-week
  instruction. Next-order number is Ask (KG).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from sndintel.action import ActionPack, build_action_pack
from sndintel.briefing import _sheet_table
from sndintel.mtd import period_state

HOLE_MT = 0.05
ASK_MT = 0.0005
PDF_ISSUE_N = 200

CLOSED_ISSUE_RANK = {
    "Unvisited": 0,
    "Unbilled": 1,
    "Missed Expected": 2,
    "Lapsed": 3,
    "Beat Expected": 8,
    "On Expected": 9,
    "No run-rate": 10,
}
MTD_ISSUE_RANK = {
    "Due · unvisited": 0,
    "Due · no bill": 1,
    "Due · light drop": 2,
    "Visited, no bill": 3,
    "Lapsed": 4,
    "Unvisited": 6,
    "On cycle": 9,
    "No run-rate": 10,
}
MIX_SKIP = {"No run-rate"}
ISSUE_LAG = {
    "Due · unvisited",
    "Due · no bill",
    "Due · light drop",
    "Visited, no bill",
    "Missed Expected",
    "Unbilled",
    "Unvisited",
    "Lapsed",
}


@dataclass
class ShopBook:
    period: str
    label: str
    scope: str = "national"
    scope_label: str = "Country"
    open_mtd: bool = False
    headline: str = ""
    weather: str = ""
    how_to_read: list[str] = field(default_factory=list)
    kpis: dict[str, Any] = field(default_factory=dict)
    mix: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: pd.DataFrame = field(default_factory=pd.DataFrame)
    shops: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw: pd.DataFrame = field(default_factory=pd.DataFrame)


def empty_shop_book(period: str = "") -> ShopBook:
    return ShopBook(period=period or "", label=period or "")


def build_shop_book(
    *,
    action: ActionPack | None = None,
    shop_month: pd.DataFrame | None = None,
    stores: pd.DataFrame | None = None,
    shop_day: pd.DataFrame | None = None,
    visits: pd.DataFrame | None = None,
    ledger: pd.DataFrame | None = None,
    shop_targets: pd.DataFrame | None = None,
    period: str | None = None,
    scope: str = "national",
    city: str | None = None,
    distributor: str | None = None,
    dsr: str | None = None,
) -> ShopBook:
    """Classify existing shop rows. Does not recompute Expected or Ask."""
    period = str(period or (getattr(action, "period", None) or "") or "")
    if action is None or getattr(action, "raw_shops", None) is None or action.raw_shops.empty:
        if shop_month is None or shop_month.empty:
            return empty_shop_book(period)
        action = build_action_pack(
            shop_month,
            stores,
            shop_day=shop_day,
            visits=visits,
            ledger=ledger,
            period=period or None,
        )
    period = str(action.period or period or "")
    shops = action.raw_shops.copy()
    if shops.empty:
        book = empty_shop_book(period)
        book.label = action.label or period
        book.open_mtd = bool(action.open_mtd)
        return book

    shops = _attach_shop_targets(shops, shop_targets)
    scope = (scope or "national").strip().lower()
    if scope in {"country", "national hq", ""}:
        scope = "national"
    shops = _scope_shops(shops, scope, city, distributor, dsr)
    open_mtd = bool(action.open_mtd)
    if ledger is not None:
        open_mtd = bool(period_state(ledger, period).get("open"))
    shops = _classify_rows(shops, open_mtd)
    shops = _sort_rows(shops, open_mtd)

    scope_label = _scope_label(scope, city, distributor, dsr)
    kpis = _kpis(shops, open_mtd)
    mix = _issue_mix(shops, open_mtd)
    headline, weather = _headline(kpis, scope_label, action.label or period, open_mtd)
    presented = _present(shops, scope, open_mtd, bool(kpis.get("has_plan")))
    issues = presented[presented["Issue"].isin(ISSUE_LAG)].copy() if not presented.empty else presented
    return ShopBook(
        period=period,
        label=action.label or period,
        scope=scope,
        scope_label=scope_label,
        open_mtd=open_mtd,
        headline=headline,
        weather=weather,
        how_to_read=_how_to_read(open_mtd),
        kpis=kpis,
        mix=mix,
        issues=issues,
        shops=presented,
        raw=shops,
    )


def list_shop_book_entities(shops: pd.DataFrame | None, kind: str) -> list[str]:
    kind = (kind or "").strip().lower()
    if shops is None or shops.empty:
        return []
    work = shops.copy()
    if kind == "city":
        return sorted({str(x).strip() for x in work.get("city", pd.Series(dtype=str)).dropna().astype(str) if str(x).strip() not in {"", "nan", "(unmapped)"}})
    if kind == "distributor":
        out = []
        for _, r in work.iterrows():
            name = str(r.get("distributor") or "").strip()
            city = str(r.get("city") or "").strip()
            if not name or name in {"nan", "(unmapped)"}:
                continue
            out.append(f"{city} · {name}" if city and city not in {"nan", "(unmapped)"} else name)
        return sorted(set(out))
    if kind == "dsr":
        out = []
        for _, r in work.iterrows():
            name = str(r.get("dsr_name") or "").strip()
            city = str(r.get("city") or "").strip()
            dist = str(r.get("distributor") or "").strip()
            if not name or name in {"nan", "(unnamed)"}:
                continue
            if city and city not in {"nan", "(unmapped)"} and dist and dist not in {"nan", "(unmapped)"}:
                out.append(f"{city} · {dist} · {name}")
            elif city and city not in {"nan", "(unmapped)"}:
                out.append(f"{city} · {name}")
            else:
                out.append(name)
        return sorted(set(out))
    return []


def parse_shop_scope(kind: str, entity: str | None) -> dict[str, str | None]:
    parts = [p.strip() for p in str(entity or "").split(" · ") if str(p).strip()]
    kind = (kind or "national").strip().lower()
    if kind == "city":
        return {"city": entity, "distributor": None, "dsr": None}
    if kind == "distributor":
        if len(parts) >= 2:
            return {"city": parts[0], "distributor": " · ".join(parts[1:]), "dsr": None}
        return {"city": None, "distributor": entity, "dsr": None}
    if kind == "dsr":
        if len(parts) >= 3:
            return {"city": parts[0], "distributor": parts[1], "dsr": " · ".join(parts[2:])}
        if len(parts) == 2:
            return {"city": parts[0], "distributor": None, "dsr": parts[1]}
        return {"city": None, "distributor": None, "dsr": entity}
    return {"city": None, "distributor": None, "dsr": None}


def _attach_shop_targets(shops: pd.DataFrame, shop_targets: pd.DataFrame | None) -> pd.DataFrame:
    out = shops.copy()
    out["shop_target_mt"] = 0.0
    if shop_targets is None or shop_targets.empty or "store_id" not in shop_targets.columns:
        return out
    work = shop_targets.copy()
    work["store_id"] = work["store_id"].astype(str).str.strip()
    method = work["match_method"].fillna("unmatched").astype(str) if "match_method" in work.columns else pd.Series("id", index=work.index)
    matched = work.loc[method.ne("unmatched") & work["store_id"].ne("") & work["store_id"].ne("nan")]
    if matched.empty or "target_mt" not in matched.columns:
        return out
    tgt = matched.groupby(matched["store_id"].astype(str))["target_mt"].sum()
    out["shop_target_mt"] = out["store_id"].astype(str).map(tgt).fillna(0.0)
    return out


def _scope_shops(
    shops: pd.DataFrame,
    scope: str,
    city: str | None,
    distributor: str | None,
    dsr: str | None,
) -> pd.DataFrame:
    out = shops.copy()
    if scope == "national" or not scope:
        return out
    if city:
        out = out[out["city"].astype(str) == str(city)]
    if scope in {"distributor", "dsr"} and distributor:
        out = out[out["distributor"].astype(str) == str(distributor)]
    if scope == "dsr" and dsr:
        out = out[out["dsr_name"].astype(str) == str(dsr)]
    return out.reset_index(drop=True)


def _scope_label(scope: str, city: str | None, distributor: str | None, dsr: str | None) -> str:
    if scope == "dsr":
        bits = [b for b in (city, distributor, dsr) if b]
        return " · ".join(bits) if bits else "DSR"
    if scope == "distributor":
        return f"{city} · {distributor}" if city and distributor else (distributor or city or "Distributor")
    if scope == "city":
        return city or "City"
    return "Country"


def _classify_rows(shops: pd.DataFrame, open_mtd: bool) -> pd.DataFrame:
    out = shops.copy()
    billed = pd.to_numeric(out.get("billed_mt"), errors="coerce").fillna(0.0)
    expected = pd.to_numeric(out.get("expected_mt"), errors="coerce").fillna(0.0)
    remaining = pd.to_numeric(out.get("remaining_mt"), errors="coerce")
    if remaining.isna().all():
        remaining = (expected - billed).clip(lower=0)
    remaining = remaining.fillna((expected - billed).clip(lower=0))
    ask = pd.to_numeric(out.get("week_target_mt"), errors="coerce").fillna(0.0)
    lapsed = out["is_lapsed"].fillna(False).astype(bool) if "is_lapsed" in out.columns else pd.Series(False, index=out.index)
    call = out["call_status"].astype(str) if "call_status" in out.columns else pd.Series("", index=out.index)
    unvisited = call.eq("Unvisited")
    visited_no_bill = call.eq("Visited · not billed")
    issues = []
    flags = []
    comments = []
    for i, row in out.iterrows():
        rec = row.to_dict()
        rec["billed_mt"] = float(billed.loc[i])
        rec["expected_mt"] = float(expected.loc[i])
        rec["remaining_mt"] = float(remaining.loc[i])
        rec["week_target_mt"] = float(ask.loc[i])
        rec["is_lapsed"] = bool(lapsed.loc[i])
        rec["call_status"] = str(call.loc[i])
        issue = _issue(rec, open_mtd, bool(unvisited.loc[i]), bool(visited_no_bill.loc[i]))
        issues.append(issue)
        flags.append(issue in ISSUE_LAG)
        comments.append(_comment(rec, issue, open_mtd))
    out["remaining_mt"] = remaining
    out["issue"] = issues
    out["is_issue"] = flags
    out["comment"] = comments
    rank_map = MTD_ISSUE_RANK if open_mtd else CLOSED_ISSUE_RANK
    out["issue_rank"] = out["issue"].map(lambda x: rank_map.get(str(x), 7))
    return out


def _issue(row: dict[str, Any], open_mtd: bool, unvisited: bool, visited_no_bill: bool) -> str:
    billed = float(row.get("billed_mt") or 0)
    expected = float(row.get("expected_mt") or 0)
    remaining = float(row.get("remaining_mt") or 0)
    ask = float(row.get("week_target_mt") or 0)
    lapsed = bool(row.get("is_lapsed"))
    has_bill = billed > ASK_MT
    has_exp = expected > HOLE_MT
    if open_mtd:
        if lapsed and not has_bill:
            return "Lapsed"
        if ask > ASK_MT and unvisited:
            return "Due · unvisited"
        if ask > ASK_MT and visited_no_bill:
            return "Due · no bill"
        if ask > ASK_MT:
            return "Due · light drop"
        if visited_no_bill or (not has_bill and not unvisited):
            return "Visited, no bill"
        if not has_exp and not has_bill:
            return "No run-rate"
        return "On cycle"
    # Closed month: billed means any invoice (same as action call_status).
    # 0.05 MT is the material hole, not a "did they bill / do they have Expected" cutoff.
    if not has_exp and not has_bill:
        return "No run-rate"
    if not has_bill and lapsed:
        return "Lapsed"
    if not has_bill and unvisited:
        return "Unvisited"
    if not has_bill:
        return "Unbilled"
    if remaining > HOLE_MT:
        return "Missed Expected"
    if billed > expected + HOLE_MT:
        return "Beat Expected"
    return "On Expected"


def _comment(row: dict[str, Any], issue: str, open_mtd: bool) -> str:
    if open_mtd:
        text = str(row.get("instruction") or "").strip()
        if text:
            return text
        rec = str(row.get("recommended_action") or "").strip()
        name = str(row.get("store_name") or row.get("store_id") or "Shop")
        return rec if rec else f"{name} is inside its cycle. Leave it."
    name = str(row.get("store_name") or row.get("store_id") or "Shop")
    billed = float(row.get("billed_mt") or 0)
    expected = float(row.get("expected_mt") or 0)
    remaining = float(row.get("remaining_mt") or 0)
    last = _date_text(row.get("last_bill_date"))
    last_drop = float(row.get("last_drop_mt") or 0)
    usual = float(row.get("expected_drop_mt") or row.get("typical_drop_mt") or 0)
    last_bit = last or "no billed sale"
    drop_bit = f"{last_drop:.2f} MT" if last_drop > ASK_MT else "—"
    usual_bit = f"{usual:.2f} MT" if usual > ASK_MT else "the usual drop"
    if issue == "Unvisited":
        return (
            f"{name} was not visited. Last billed {last_bit}. "
            f"Next month: put on the beat before the first drop is due."
        )
    if issue == "Unbilled":
        return (
            f"{name} was visited but not billed. Last billed {last_bit} ({drop_bit}). "
            f"Next month: do not leave without {usual_bit}."
        )
    if issue == "Lapsed":
        return f"{name} is a lost door. Last billed {last_bit}. Next month: recover or drop from the beat."
    if issue == "Missed Expected":
        return (
            f"{name} billed {billed:.2f} MT versus Expected {expected:.2f} MT. "
            f"Last drop {drop_bit} on {last_bit}. Next month: recover the {remaining:.2f} MT hole."
        )
    if issue == "Beat Expected":
        return f"{name} billed {billed:.2f} MT versus Expected {expected:.2f} MT. Keep the same drop next month."
    if issue == "No run-rate":
        return f"{name} has no material Expected this month. Last billed {last_bit}."
    return f"{name} landed on Expected ({billed:.2f} MT). Last billed {last_bit}."


def _sort_rows(shops: pd.DataFrame, open_mtd: bool) -> pd.DataFrame:
    out = shops.copy()
    if open_mtd:
        out["_ask"] = pd.to_numeric(out.get("week_target_mt"), errors="coerce").fillna(0.0)
        out = out.sort_values(
            ["is_issue", "issue_rank", "_ask", "remaining_mt"],
            ascending=[False, True, False, False],
        )
        return out.drop(columns=["_ask"]).reset_index(drop=True)
    out = out.sort_values(
        ["is_issue", "remaining_mt", "issue_rank"],
        ascending=[False, False, True],
    )
    return out.reset_index(drop=True)


def _kpis(shops: pd.DataFrame, open_mtd: bool) -> dict[str, Any]:
    billed = float(pd.to_numeric(shops.get("billed_mt"), errors="coerce").fillna(0).sum()) if not shops.empty else 0.0
    expected = float(pd.to_numeric(shops.get("expected_mt"), errors="coerce").fillna(0).sum()) if not shops.empty else 0.0
    remaining = float(pd.to_numeric(shops.get("remaining_mt"), errors="coerce").fillna(0).sum()) if not shops.empty else 0.0
    target = float(pd.to_numeric(shops.get("shop_target_mt"), errors="coerce").fillna(0).sum()) if not shops.empty else 0.0
    ask = float(pd.to_numeric(shops.get("week_target_mt"), errors="coerce").fillna(0).sum()) if not shops.empty else 0.0
    issues = shops[shops["is_issue"]] if not shops.empty and "is_issue" in shops.columns else shops.iloc[0:0]
    issue_mt = float(pd.to_numeric(issues.get("remaining_mt"), errors="coerce").fillna(0).sum()) if not issues.empty else 0.0
    quiet_n = int((shops["issue"] == "No run-rate").sum()) if not shops.empty and "issue" in shops.columns else 0
    due_n = int((pd.to_numeric(shops.get("week_target_mt"), errors="coerce").fillna(0) > ASK_MT).sum()) if open_mtd and not shops.empty else 0
    return {
        "open_mtd": open_mtd,
        "billed_mt": billed,
        "expected_mt": expected,
        "gap_mt": remaining,
        "target_mt": target,
        "has_plan": target > HOLE_MT,
        "ask_mt": ask,
        "n_shops": int(len(shops)),
        "n_issues": int(len(issues)),
        "issue_mt": issue_mt,
        "n_quiet": quiet_n,
        "n_due": due_n,
        "n_unvisited": int((shops["issue"] == "Unvisited").sum()) if not shops.empty else 0,
        "n_unbilled": int((shops["issue"].isin(["Unbilled", "Visited, no bill", "Due · no bill"])).sum()) if not shops.empty else 0,
        "n_lapsed": int((shops["issue"] == "Lapsed").sum()) if not shops.empty else 0,
    }


def _issue_mix(shops: pd.DataFrame, open_mtd: bool) -> pd.DataFrame:
    if shops is None or shops.empty:
        return pd.DataFrame()
    order = list(MTD_ISSUE_RANK) if open_mtd else list(CLOSED_ISSUE_RANK)
    rows = []
    for issue in order:
        if issue in MIX_SKIP:
            continue
        part = shops[shops["issue"] == issue]
        if part.empty:
            continue
        hole = float(pd.to_numeric(part["remaining_mt"], errors="coerce").fillna(0).sum())
        if issue not in ISSUE_LAG:
            hole = 0.0
        rec = {
            "Issue": issue,
            "Shops": int(len(part)),
            "Billed (MT)": round(float(pd.to_numeric(part["billed_mt"], errors="coerce").fillna(0).sum()), 2),
            "Expected (MT)": round(float(pd.to_numeric(part["expected_mt"], errors="coerce").fillna(0).sum()), 2),
            "Hole (MT)": round(hole, 2),
        }
        if open_mtd:
            rec["Ask (KG)"] = int(round(float(pd.to_numeric(part["week_target_mt"], errors="coerce").fillna(0).sum()) * 1000))
        rows.append(rec)
    return pd.DataFrame(rows)


def _headline(kpis: dict[str, Any], scope_label: str, label: str, open_mtd: bool) -> tuple[str, str]:
    billed = float(kpis.get("billed_mt") or 0)
    expected = float(kpis.get("expected_mt") or 0)
    n_iss = int(kpis.get("n_issues") or 0)
    issue_mt = float(kpis.get("issue_mt") or 0)
    if open_mtd:
        n_due = int(kpis.get("n_due") or 0)
        ask_kg = int(round(float(kpis.get("ask_mt") or 0) * 1000))
        headline = (
            f"{scope_label} · {label}: billed {billed:.1f} MT so far. "
            f"{n_due} shops due this week (Ask {ask_kg:,} KG)."
        )
        weather = (
            "Issues are due this week, visited with no bill, and lost doors. "
            "Still-to-Expected is the full month, not a miss yet."
        )
        return headline, weather
    headline = (
        f"{scope_label} closed {label}: billed {billed:.1f} MT versus Expected {expected:.1f} MT. "
        f"{n_iss} shops missed Expected ({issue_mt:.1f} MT hole)."
    )
    weather = (
        "This is a finished month. Issues are shops that missed Expected — "
        "light orders, visited with no bill, unvisited, lapsed — largest hole first."
    )
    return headline, weather


def _how_to_read(open_mtd: bool) -> list[str]:
    if open_mtd:
        return [
            "This is the in-month shop list. Billed is so far; Expected is the full-month run-rate (same last-3 / last-6 as every other pack).",
            "A shop that has not yet billed its full Expected is not a miss mid-month. Issues are due this week (Ask), visited with no bill, and lost doors.",
            "Ask (KG) is the 90-day expected drop when the depletion ratio is ≥ 0.8. That is the next-order number for this week. Official Expected is not multiplied into Ask.",
            "Target is the matched shop quota when a plan row exists. It does not replace Expected.",
        ]
    return [
        "This is a closed month. Billed and Expected are the full month. Gap on a shop is Expected − billed when that is positive.",
        "Unbilled means visited and billed 0 this month. A 10–50 KG invoice is billed (light order / missed Expected), not unbilled.",
        "Issues are shops that missed Expected: light orders, visited with no bill, unvisited, lapsed. Largest hole first. Universe doors with no material Expected are omitted from the mix.",
        "Usual drop is the 90-day expected drop (what they typically take). Ask is 0 on a closed month, so it is not on this list.",
        "Comment is next month, not this week. Target is the matched shop quota when a plan row exists.",
    ]


def _present(shops: pd.DataFrame, scope: str, open_mtd: bool, has_plan: bool) -> pd.DataFrame:
    if shops is None or shops.empty:
        return pd.DataFrame()
    rows = []
    for _, r in shops.iterrows():
        rec: dict[str, Any] = {"Shop": str(r.get("store_name") or r.get("store_id") or "")}
        if scope == "national":
            rec["City"] = str(r.get("city") or "")
            rec["Distributor"] = str(r.get("distributor") or "")
            rec["DSR"] = str(r.get("dsr_name") or "")
        elif scope == "city":
            rec["Distributor"] = str(r.get("distributor") or "")
            rec["DSR"] = str(r.get("dsr_name") or "")
        elif scope == "distributor":
            rec["DSR"] = str(r.get("dsr_name") or "")
        rec["Billed (MT)"] = round(float(r.get("billed_mt") or 0), 2)
        rec["Expected (MT)"] = round(float(r.get("expected_mt") or 0), 2)
        if not open_mtd:
            rec["Gap (MT)"] = round(float(r.get("remaining_mt") or 0), 2)
            if has_plan:
                rec["Target (MT)"] = round(float(r.get("shop_target_mt") or 0), 2)
        rec["Last billed"] = _date_text(r.get("last_bill_date"))
        if open_mtd:
            days = r.get("days_since_bill")
            rec["Days since"] = int(round(float(days))) if days is not None and pd.notna(days) else None
            rec["Ask (KG)"] = int(round(float(r.get("week_target_mt") or 0) * 1000))
        else:
            rec["Last drop (MT)"] = round(float(r.get("last_drop_mt") or 0), 2)
            rec["Usual drop (MT)"] = round(float(r.get("expected_drop_mt") or r.get("typical_drop_mt") or 0), 2)
        rec["Issue"] = str(r.get("issue") or "")
        rec["Comment" if not open_mtd else "Do this"] = str(r.get("comment") or "")
        rows.append(rec)
    return pd.DataFrame(rows)


def _date_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    ts = pd.to_datetime(text, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%d %b")
    return text


# --- Excel / PDF --------------------------------------------------------------


def excel_bytes(book: ShopBook) -> bytes:
    from openpyxl.styles import Alignment, Font
    from sndintel.briefing import NAVY as HEX_NAVY, SLATE as HEX_SLATE, _fill

    wb = Workbook()
    ws = wb.active
    ws.title = "00 Cover"
    cut = "MTD" if book.open_mtd else "Closed month"
    ws["A1"] = "SND Intelligence · Shop-wise issues"
    ws["A1"].font = Font(name="Calibri", size=14, bold=True, color=HEX_NAVY)
    ws["A2"] = f"{book.scope_label} · {book.label} · {cut}"
    ws["A2"].font = Font(name="Calibri", size=18, bold=True, color=HEX_NAVY)
    ws["A3"] = book.headline or ""
    ws["A3"].font = Font(name="Calibri", size=12, bold=True, color=HEX_NAVY)
    ws.merge_cells("A3:H3")
    kpis = book.kpis or {}
    if book.open_mtd:
        metric_row = [
            ("Billed so far (MT)", kpis.get("billed_mt")),
            ("Expected (MT)", kpis.get("expected_mt")),
            ("Due shops", kpis.get("n_due")),
            ("Ask (KG)", int(round(float(kpis.get("ask_mt") or 0) * 1000))),
            ("Issue shops", kpis.get("n_issues")),
        ]
    else:
        metric_row = [
            ("Billed (MT)", kpis.get("billed_mt")),
            ("Expected (MT)", kpis.get("expected_mt")),
            ("Gap (MT)", kpis.get("gap_mt")),
            ("Issue shops", kpis.get("n_issues")),
            ("Hole (MT)", kpis.get("issue_mt")),
        ]
    if kpis.get("has_plan"):
        metric_row.append(("Target (MT)", kpis.get("target_mt")))
    for i, (name, val) in enumerate(metric_row, start=1):
        cell = ws.cell(5, i, name)
        cell.font = Font(bold=True, color="FFFFFF", size=9)
        cell.fill = _fill(HEX_NAVY)
        show_mt = isinstance(val, (int, float)) and str(name).endswith("(MT)")
        v = ws.cell(6, i, round(float(val), 2) if show_mt else val)
        v.font = Font(size=12, bold=True)
    ws.cell(8, 1, book.weather or "")
    ws.merge_cells("A8:H8")
    ws.cell(8, 1).font = Font(size=9, italic=True, color=HEX_SLATE)
    row = 10
    ws.cell(row, 1, "How to read")
    ws.cell(row, 1).font = Font(bold=True, size=12, color=HEX_NAVY)
    row += 1
    for step in book.how_to_read:
        ws.cell(row, 1, step)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.cell(row, 1).font = Font(size=10, italic=True, color=HEX_SLATE)
        row += 1
    ws.column_dimensions["A"].width = 28
    _sheet_table(
        wb,
        "01 Issue mix",
        "Issue mix",
        "Unbilled = visited and billed 0. A small invoice is Missed Expected. Universe doors with no run-rate are omitted.",
        book.mix,
    )
    note_iss = "Shops that are an issue on this cut, largest hole / Ask first."
    _sheet_table(wb, "02 Issues", "Issues", note_iss, book.issues)
    note_all = (
        "Every shop in this scope. Mid-month still-to-Expected is not a miss. "
        if book.open_mtd
        else "Every shop in this scope, including those that landed on Expected. "
    )
    _sheet_table(wb, "03 All shops", "All shops", note_all + "Excel is the full list; the PDF is issues only.", book.shops)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def pdf_bytes(book: ShopBook) -> bytes:
    buf = BytesIO()
    write_pdf(book, buf)
    return buf.getvalue()


def write_pdf(book: ShopBook, path: Path | str | BytesIO) -> None:
    from sndintel.situation_report import (
        LINE,
        NAVY,
        WASH,
        WHITE,
        _pdf_styles,
        _pdf_table,
    )

    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        path if not isinstance(path, (str, Path)) else str(path),
        pagesize=pagesize,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=16 * mm,
        bottomMargin=12 * mm,
        title=f"SND Intelligence · {book.scope_label} shops",
        author="SND Intelligence",
    )
    styles = _pdf_styles()
    usable = pagesize[0] - doc.leftMargin - doc.rightMargin
    cut = "MTD SHOP ISSUES" if book.open_mtd else "CLOSED-MONTH SHOP ISSUES"
    story: list[Any] = [
        Paragraph(cut, styles["kicker"]),
        Paragraph(xml_escape(f"{book.scope_label} · {book.label}"), styles["h1"]),
        Paragraph(xml_escape(book.headline or ""), styles["headline"]),
    ]
    kpis = book.kpis or {}
    if book.open_mtd:
        kpi_cells = [
            ("Billed so far", f"{float(kpis.get('billed_mt') or 0):.1f} MT"),
            ("Expected", f"{float(kpis.get('expected_mt') or 0):.1f} MT"),
            ("Due shops", str(int(kpis.get("n_due") or 0))),
            ("Ask", f"{int(round(float(kpis.get('ask_mt') or 0) * 1000)):,} KG"),
            ("Issue shops", str(int(kpis.get("n_issues") or 0))),
        ]
    else:
        kpi_cells = [
            ("Billed", f"{float(kpis.get('billed_mt') or 0):.1f} MT"),
            ("Expected", f"{float(kpis.get('expected_mt') or 0):.1f} MT"),
            ("Gap vs Expected", f"{float(kpis.get('gap_mt') or 0):.1f} MT"),
            ("Issue shops", str(int(kpis.get("n_issues") or 0))),
            ("Hole", f"{float(kpis.get('issue_mt') or 0):.1f} MT"),
        ]
    if kpis.get("has_plan"):
        kpi_cells.append(("Target", f"{float(kpis.get('target_mt') or 0):.1f} MT"))
    labels = [Paragraph(xml_escape(n), styles["kpi_l"]) for n, _ in kpi_cells]
    values = [Paragraph(xml_escape(v), styles["kpi_v"]) for _, v in kpi_cells]
    n = max(len(kpi_cells), 1)
    kpi_table = Table([labels, values], colWidths=[usable / n] * n)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, 1), WASH),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 1), (-1, 1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(xml_escape(book.weather or ""), styles["lead"]))
    story.append(Paragraph("How to read", styles["h3"]))
    for line in book.how_to_read:
        story.append(Paragraph(xml_escape("- " + line), styles["body"]))
    mix = book.mix
    if mix is not None and not mix.empty:
        heading = Paragraph("Issue mix", styles["h2"])
        note = Paragraph(
            "Unbilled = visited and billed 0 this month. A small invoice is Missed Expected, not unbilled. "
            "Universe doors with no material Expected are omitted.",
            styles["note"],
        )
        table = _pdf_table(mix, styles, usable)
        story.append(KeepTogether([heading, note, Spacer(1, 2), table]))
    issues = book.issues
    n_iss = 0 if issues is None or issues.empty else int(len(issues))
    shown = issues.head(PDF_ISSUE_N) if n_iss else issues
    heading = Paragraph("Shop issues", styles["h2"])
    extra = f" Showing {PDF_ISSUE_N} of {n_iss}. Excel has every shop." if n_iss > PDF_ISSUE_N else " Excel has every shop in this scope."
    note = Paragraph(
        ("Largest Ask first. " if book.open_mtd else "Largest hole versus Expected first. ") + extra.strip(),
        styles["note"],
    )
    table = _pdf_table(shown, styles, usable)
    story.append(PageBreak())
    story.append(KeepTogether([heading, note, Spacer(1, 2)]))
    story.append(table)
    doc.build(story)


def write_excel(book: ShopBook, path: Path | str) -> None:
    Path(path).write_bytes(excel_bytes(book))
