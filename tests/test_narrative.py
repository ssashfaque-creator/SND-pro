"""Grounded national executive summary: brief contents, mocked OpenAI, PDF order."""

from __future__ import annotations

import base64
import json
import re
import zlib
from unittest.mock import patch

import pandas as pd

from sndintel.briefing import build_strategy_pack, pdf_bytes
from sndintel.hierarchy import build_hierarchy_pack
from sndintel.narrative import build_grounded_brief, generate_exec_summary, store_exec_summary
from sndintel.storage import connect, init_db
from tests.test_briefing import _row, _stores, _with_recent_ams


def _pdf_text(pdf: bytes) -> str:
    chunks: list[str] = []
    for m in re.finditer(rb"/Length (\d+)\s*>>\s*stream\n", pdf):
        raw = pdf[m.end() : m.end() + int(m.group(1))]
        try:
            decoded = base64.a85decode(raw.strip(), adobe=True)
            chunks.append(zlib.decompress(decoded).decode("latin-1", "ignore"))
        except Exception:
            continue
    return "\n".join(chunks)


def _pack():
    rows = []
    rows.append(_row("K1", "2026-08", 5.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K1", "2025-08", 80.0, "Karachi", "Eva Foods", "Amir", name="Kifaya"))
    rows.append(_row("K2", "2026-08", 15.0, "Karachi", "South Dist", "Amir", name="Hold K"))
    rows.append(_row("K2", "2025-08", 20.0, "Karachi", "South Dist", "Amir", name="Hold K"))
    rows.append(_row("L1", "2026-08", 90.0, "Lahore", "Holding Dist", "Lahore Ace", name="Big L"))
    rows.append(_row("L1", "2025-08", 100.0, "Lahore", "Holding Dist", "Lahore Ace", name="Big L"))
    rows.append(_row("L2", "2026-08", 10.0, "Lahore", "Local Dist", "Lahore Weak", name="Small L"))
    rows.append(_row("L2", "2025-08", 100.0, "Lahore", "Local Dist", "Lahore Weak", name="Small L"))
    rows = _with_recent_ams(rows, volume_by_store={"K1": 8.0, "K2": 18.0, "L1": 95.0, "L2": 80.0})
    sm = pd.DataFrame(rows)
    pack_h = build_hierarchy_pack(sm, _stores(rows), ledger=pd.DataFrame([{"period": "2026-08", "status": "closed"}]))
    return build_strategy_pack(pack_h.units, sm, period="2026-08")


def test_grounded_brief_contains_country_and_lagging_cities():
    report = _pack()
    brief = build_grounded_brief(report)
    assert brief["period"] == "2026-08"
    country = brief["country"]
    assert country is not None
    assert country["City"] == "Country"
    assert country["Billed this period (MT)"] is not None
    lag_names = {row["City"] for row in brief["lagging_cities"]}
    assert "Karachi" in lag_names
    billed = {row["city"]: row["billed_mt"] for row in brief["city_share_of_country_billed"]}
    assert "Karachi" in billed
    assert all(not str(s.get("Shop") or "").startswith("Not listed") for s in brief["lagging_shops_top"])


def test_generate_exec_summary_uses_mocked_openai_and_does_not_invent():
    report = _pack()
    payload = {
        "situation": [
            "Country billed the volume in the brief against expected. That is weather.",
            "Karachi holds the extra hole after weather.",
        ],
        "focus": [
            {
                "title": "Karachi extra hole",
                "why": "Recoverable is concentrated here versus the country.",
                "do": "Call Eva Foods first.",
            }
        ],
    }

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch("sndintel.narrative.urllib.request.urlopen", return_value=_Resp()):
        result = generate_exec_summary(report, api_key="sk-test", model="gpt-4.1")
    assert result["ok"] is True
    assert result["situation"][0].startswith("Country billed")
    assert result["focus"][0]["title"] == "Karachi extra hole"
    assert "Eva Foods" in result["focus"][0]["do"]


def test_generate_without_key_does_not_call_network():
    report = _pack()
    with patch("sndintel.narrative.load_openai_settings", return_value={"api_key": "", "model": "gpt-4.1"}):
        with patch("sndintel.narrative.urllib.request.urlopen") as mocked:
            result = generate_exec_summary(report, api_key="", model="gpt-4.1")
            mocked.assert_not_called()
    assert result["ok"] is False
    assert "key" in result["error"].lower()


def test_national_pdf_starts_with_glossary_then_exec():
    report = _pack()
    report.exec_situation = ["Country billed 120 MT against 200 expected. That is weather."]
    report.exec_focus = [
        {"title": "Karachi 80 MT recoverable", "why": "From unbilled shops.", "do": "Ride with Eva Foods."}
    ]
    report.exec_model = "gpt-4.1"
    pdf = pdf_bytes(report)
    assert pdf.startswith(b"%PDF")
    text = _pdf_text(pdf)
    g = text.find("Glossary")
    e = text.find("Executive summary")
    sit = text.find("Summary of current situation")
    tables = text.find("Every city versus the country")
    assert g != -1
    assert e != -1
    assert sit != -1
    assert tables != -1
    assert g < e < tables
    assert "How the figures are calculated" in text
    assert "Karachi 80 MT recoverable" in text
    assert text.rfind("Glossary") == g


def test_store_exec_summary_roundtrip(tmp_path):
    db = tmp_path / "wh.db"
    init_db(db)
    with connect(db) as conn:
        store_exec_summary(
            conn,
            "2026-08",
            {
                "ok": True,
                "model": "gpt-4.1",
                "situation": ["Weather paragraph."],
                "focus": [{"title": "Karachi", "why": "Hole", "do": "Call"}],
                "brief": {"period": "2026-08"},
                "raw": "{}",
                "error": "",
            },
        )
        row = conn.execute("SELECT * FROM exec_summary WHERE period='2026-08'").fetchone()
    assert row["model"] == "gpt-4.1"
    assert "Weather paragraph" in row["situation_json"]
