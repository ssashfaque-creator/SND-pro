"""Stable people identity.

A DSR name is not a person. The same first name appears under more than one
distributor and city. Scorecards, action lists, and capacity labels all use
``dsr_unit_id`` so two Shahids never share a row.
"""

from __future__ import annotations

from typing import Any

SEP = " | "


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "(unmapped)"}:
        return ""
    return text.replace(SEP, " / ")


def dsr_unit_id(city: Any, distributor: Any, dsr_name: Any) -> str:
    """Unique grain_id: ``Name | City | Distributor``."""
    name = _clean(dsr_name) or "(unnamed)"
    city_s = _clean(city) or "(unmapped)"
    dist = _clean(distributor) or "(unmapped)"
    return f"{name}{SEP}{city_s}{SEP}{dist}"


def dsr_display_name(unit_id: Any) -> str:
    text = "" if unit_id is None else str(unit_id)
    if SEP in text:
        return text.split(SEP, 1)[0]
    return text


def parse_dsr_unit_id(unit_id: Any) -> tuple[str, str, str]:
    """Return (dsr_name, city, distributor)."""
    text = "" if unit_id is None else str(unit_id)
    parts = text.split(SEP)
    if len(parts) >= 3:
        return parts[0], parts[1], SEP.join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return text, "", ""


def attach_dsr_identity(df, city_col: str = "city", dist_col: str = "distributor", name_col: str = "dsr_name"):
    """Set grain_id / dsr_name / city / distributor from the three parts."""
    import pandas as pd

    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    cities = out[city_col] if city_col in out.columns else ""
    dists = out[dist_col] if dist_col in out.columns else ""
    names = out[name_col] if name_col in out.columns else out.get("grain_id", "")
    out["dsr_name"] = [_clean(v) or "(unnamed)" for v in names]
    out["city"] = [_clean(v) or "(unmapped)" for v in cities]
    out["distributor"] = [_clean(v) or "(unmapped)" for v in dists]
    out["grain_id"] = [dsr_unit_id(c, d, n) for c, d, n in zip(out["city"], out["distributor"], out["dsr_name"])]
    return out
