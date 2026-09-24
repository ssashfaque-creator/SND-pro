"""Number formatting: kg under 0.1 MT, never integer MT, signs only on directional columns."""

from __future__ import annotations

import pandas as pd

from sndintel.fmt import (
    MT_FORMAT,
    MT_SIGNED_FORMAT,
    excel_number_format,
    fmt_cell,
    fmt_mt,
    kg,
    round_mt,
)


def test_fmt_mt_uses_kg_under_a_tenth_and_never_prints_integer_mt():
    assert fmt_mt(0.004) == "4 kg"
    assert fmt_mt(0.0995) == "100 kg"
    assert fmt_mt(0.1) == "0.10 MT"
    assert fmt_mt(2.5) == "2.50 MT"
    assert fmt_mt(12.34) == "12.3 MT"
    assert fmt_mt(1234.56) == "1,234.6 MT"
    assert fmt_mt(0) == "0 kg"
    assert fmt_mt(None) == "0 kg"
    assert fmt_mt(float("nan")) == "0 kg"


def test_fmt_mt_signed_marks_direction_both_ways():
    assert fmt_mt(0.5, signed=True) == "+0.50 MT"
    assert fmt_mt(-0.5, signed=True) == "-0.50 MT"
    assert fmt_mt(-0.02, signed=True) == "-20 kg"
    assert fmt_mt(0.0, signed=True) == "0 kg"


def test_round_mt_and_kg_keep_tables_numeric():
    assert round_mt(1.23456) == 1.23
    assert round_mt(None) is None
    assert round_mt(float("nan")) is None
    assert kg(0.0045) == 4  # banker's rounding on .5 is fine at 1 kg


def test_fmt_cell_follows_the_column_header():
    assert fmt_cell(1.5, "Billed (MT)") == "1.50"
    assert fmt_cell(1234.5, "Expected (MT)") == "1,234.50"
    assert fmt_cell(-0.4, "vs Target (MT)") == "-0.40"
    assert fmt_cell(0.4, "vs Target (MT)") == "+0.40"
    assert fmt_cell(0.25, "From drop size (MT)") == "+0.25"
    assert fmt_cell(0.25, "Gap (MT)") == "0.25"
    assert fmt_cell(1250, "Billed (kg)") == "1,250"
    assert fmt_cell(1250, "Ask (KG)") == "1,250"
    assert fmt_cell(12, "Billed shops") == "12"
    assert fmt_cell(37.6, "Strike %") == "38"
    assert fmt_cell(37.6, "ECO (%)") == "38"
    assert fmt_cell(None, "Gap (MT)") == "—"
    assert fmt_cell(float("nan"), "Gap (MT)") == "—"
    assert fmt_cell("Lagging", "Situation") == "Lagging"
    assert fmt_cell(True, "Coming due") == "True"


def test_excel_number_format_is_two_decimal_mt_and_whole_counts():
    assert excel_number_format("Billed (MT)") == MT_FORMAT
    assert excel_number_format("vs Target (MT)") == MT_SIGNED_FORMAT
    assert excel_number_format("From unvisited shops (MT)") == MT_SIGNED_FORMAT
    assert excel_number_format("Gap (MT)") == MT_FORMAT
    assert excel_number_format("Billed (kg)") == "#,##0"
    assert excel_number_format("Billed shops") == "#,##0"
    assert excel_number_format("Visit %") == "0"
    assert excel_number_format("City") is None
    assert MT_FORMAT.endswith("0.00")


def test_no_integer_mt_left_in_a_situation_pack_table():
    """A DSR that billed 0.4 MT against 0.9 Expected must not show as 0 vs 1."""
    from sndintel.situation_report import _body_cell, _pdf_styles

    styles = _pdf_styles()
    cell = _body_cell("Billed (MT)", 0.4, styles)
    assert "0.40" in cell.text
    cell = _body_cell("Gap (MT)", 0.5, styles)
    assert "0.50" in cell.text
    cell = _body_cell("vs Target (MT)", -0.5, styles)
    assert "-0.50" in cell.text
    assert pd.isna(round_mt(pd.NA))
