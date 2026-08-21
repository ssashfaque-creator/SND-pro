"""National executive summary from the scorecards, via OpenAI.

The model only sees code-computed figures (the same rounded numbers as the PDF).
It cannot invent volumes, shops, or percentages. Generation runs after every
upload or scorecard rebuild and is stored on the warehouse. Missing API keys
do not fail ingest.
"""

from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from sndintel.briefing import CALCULATION_NOTES, GLOSSARY, StrategyPack, is_national_pack
from sndintel.config import DATA_DIR, ensure_dirs
from sndintel.storage import utcnow

DEFAULT_MODEL = os.environ.get("SNDINTEL_OPENAI_MODEL", "gpt-4.1")
FALLBACK_MODEL = "gpt-4o"
OPENAI_URL = os.environ.get("SNDINTEL_OPENAI_URL", "https://api.openai.com/v1/chat/completions")
MAX_LAGGING_DISTS = 15
MAX_LAGGING_DSRS = 12
MAX_SHOPS = 15

SYSTEM_PROMPT = """You write the national executive summary for an FMCG secondary-sales board pack (edible oil S&D, Pakistan).

You receive a BRIEF JSON. Every number in that JSON was computed by the scorecard engine and already rounded the same way the PDF prints (whole MT, whole shop counts, whole percents; drop size to two decimals). You may cite only numbers that appear in the BRIEF. If a figure is not in the BRIEF, omit it. Do not invent shops, distributors, DSRs, cities, volumes, or percentages. Do not pad with generic advice.

Return JSON only:
{
  "situation": ["paragraph", "..."],
  "focus": [{"title": "...", "why": "...", "do": "..."}]
}

Two sections, and only these two:

1. situation — 2 to 4 short paragraphs. Summary of the current situation.
   Paragraph 1: the country. Billed versus AMS, last year, and Expected. Country Recoverable is that miss versus Expected (not a leftover versus peers). If MTD is open, say the day/fraction. Visit % versus Strike % at country level.
   Paragraph 2: cities versus their own Expected. Rank lagging cities by Recoverable and, when you can, each city's share of country Recoverable. Lagging-city Recoverable can sum to more than country Recoverable when other cities are Ahead. Name Ahead cities so leadership does not raid them. Cities "On expected" billed their typical month — they are not local fires even if they are down versus last year.
   Paragraph 3: coverage versus conversion versus drop size. Read country Visit % against Strike %, and the country From drop / From unvisited / From unbilled split (they add to Recoverable). High visit and low strike means unbilled shops, not unvisited. Confirm with the From columns.
   Optional paragraph 4: concentration. If one city is both a large share of billed volume and most of the country miss versus Expected, say so. Do not tour every city that is slightly down.

2. focus — 4 to 8 key focus areas, ranked by Recoverable MT, each with:
   title: named entity plus the Recoverable MT (city, then distributors, then DSRs or a material shop).
   why: the From-columns, Visit/Strike versus parent, and why this unit is behind its own Expected.
   do: one specific next action this week (who to call, what to inspect). Not slogans.

How the pack is built (use these definitions; do not redefine them in the prose):

- Expected = typical same calendar month from every matching month in the warehouse, blended with destationalized recent trend × month index, paced if MTD is open. Same method at country, city, distributor, DSR, and shop. Thin series shrink toward the parent month index; children's Expecteds are then scaled so they add to the parent Expected. It is not last year alone, and it is not last-year mix × what the parent billed now.
- Lagging = behind this unit's own Expected by a material amount, not merely down YoY. Ahead = ahead of own Expected. On expected = billed in line with the typical month.
- Recoverable = the hole versus this unit's own Expected as a positive number. Country Recoverable is the country miss versus Expected. Do not treat peer mix or "fair share of parent" as the call.
- From drop / unvisited / unbilled add to Recoverable. Positive = part of the hole. Negative = billed more than Expected.
- Drop size (MT) = billed MT ÷ billed shops. It is not From drop size.
- Visit % = visited ÷ universe (a billed shop counts as visited). Strike % = billed ÷ universe.
- AMS = 0 distributors/DSRs are already hidden. Shop lists already drop doors with Recoverable ≤ 0.25 MT; the remainder line is the tail.
- Last-year = 0 can produce a huge YoY %. That is an artifact, not a win.

Quality bar (pattern, not numbers to copy):

If the country is far below AMS and Expected, that miss is Country Recoverable. Rank lagging cities by Recoverable versus their own Expected. Read Visit % vs Strike %: high visit and low strike means the hole is unbilled, confirmed when From unbilled dominates From unvisited. Name Ahead cities. Then name the lagging distributors and DSRs that own the city hole, and any single shop whose Recoverable is material (for example a visited-not-billed door). A distributor can lag inside a city that is Ahead or On expected — mention that pocket; do not send the city into the lagging-city list. If visit is already ~90%+, do not recommend "visit more"; recommend converting unbilled doors or lifting drop size.

Write for a sales head. Plain English. Named entities. No filler. No bullet salad inside situation paragraphs. Do not mention that you are an AI, the prompt, or the BRIEF JSON.
"""


def secrets_dir() -> Path:
    ensure_dirs()
    path = DATA_DIR / "secrets"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass
    return path


def openai_settings_path() -> Path:
    return secrets_dir() / "openai.json"


def load_openai_settings() -> dict[str, str]:
    env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    env_model = (os.environ.get("SNDINTEL_OPENAI_MODEL") or "").strip()
    stored: dict[str, Any] = {}
    path = openai_settings_path()
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
    key = env_key or str(stored.get("api_key") or "").strip()
    model = env_model or str(stored.get("model") or "").strip() or DEFAULT_MODEL
    return {"api_key": key, "model": model, "has_stored_key": bool(str(stored.get("api_key") or "").strip())}


def save_openai_settings(api_key: str | None = None, model: str | None = None) -> dict[str, str]:
    current = load_openai_settings()
    key = (api_key if api_key is not None else current.get("api_key") or "").strip()
    chosen = (model if model is not None else current.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    payload = {"api_key": key, "model": chosen}
    path = openai_settings_path()
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return {"api_key": key, "model": chosen, "has_stored_key": bool(key)}


def has_openai_key() -> bool:
    return bool(load_openai_settings().get("api_key"))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _records(df: pd.DataFrame, n: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    use = df.head(int(n)) if n is not None else df
    out: list[dict[str, Any]] = []
    for _, row in use.iterrows():
        rec: dict[str, Any] = {}
        for col in use.columns:
            rec[str(col)] = _jsonable(row[col])
        out.append(rec)
    return out


def _city_split(cities: pd.DataFrame) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cities is None or cities.empty or "City" not in cities.columns:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    country_df = cities[cities["City"].astype(str) == "Country"]
    rest = cities[cities["City"].astype(str) != "Country"].copy()
    country = _records(country_df)[0] if not country_df.empty else None
    if "Situation" in rest.columns:
        lag = rest[rest["Situation"].astype(str) == "Lagging"]
        ahead = rest[rest["Situation"].astype(str) == "Ahead"]
        with_c = rest[rest["Situation"].astype(str).str.contains("On expected", case=False, na=False)]
    else:
        lag = rest
        ahead = pd.DataFrame()
        with_c = pd.DataFrame()
    return country, rest, lag, ahead, with_c


def build_grounded_brief(pack: StrategyPack) -> dict[str, Any]:
    """Compact, already-rounded facts for the model. No invented numbers."""
    cities = pack.cities if pack.cities is not None else pd.DataFrame()
    country, rest, lag, ahead, with_c = _city_split(cities)
    kpis = pack.kpis or {}
    extra = kpis.get("extra_hole_mt")
    rec_country = None
    if country and "Recoverable (MT)" in country:
        rec_country = country.get("Recoverable (MT)")
    lag_share: list[dict[str, Any]] = []
    denom = abs(float(rec_country)) if rec_country not in (None, "") else abs(float(extra or 0) or 0)
    for rec in _records(lag):
        rec_mt = rec.get("Recoverable (MT)")
        share = None
        try:
            if denom and rec_mt is not None:
                share = round(100.0 * float(rec_mt) / denom)
        except (TypeError, ValueError):
            share = None
        lag_share.append(
            {
                "city": rec.get("City"),
                "recoverable_mt": rec_mt,
                "share_of_country_recoverable_pct": share,
                "billed_mt": rec.get("Billed this period (MT)"),
                "visit_pct": rec.get("Visit %"),
                "strike_pct": rec.get("Strike %"),
                "from_drop_mt": rec.get("From drop size (MT)"),
                "from_unvisited_mt": rec.get("From unvisited shops (MT)"),
                "from_unbilled_mt": rec.get("From unbilled shops (MT)"),
            }
        )
    billed_country = country.get("Billed this period (MT)") if country else kpis.get("billed_mt")
    city_billed_share: list[dict[str, Any]] = []
    try:
        billed_f = float(billed_country) if billed_country is not None else 0.0
    except (TypeError, ValueError):
        billed_f = 0.0
    for rec in _records(rest):
        vol = rec.get("Billed this period (MT)")
        share = None
        try:
            if billed_f and vol is not None:
                share = round(100.0 * float(vol) / billed_f)
        except (TypeError, ValueError):
            share = None
        city_billed_share.append({"city": rec.get("City"), "billed_mt": vol, "share_of_country_billed_pct": share})
    dists = pack.lagging_distributors if pack.lagging_distributors is not None else pd.DataFrame()
    dsrs = pack.lagging_dsrs if pack.lagging_dsrs is not None else pd.DataFrame()
    shops = pack.lagging_shops if pack.lagging_shops is not None else pd.DataFrame()
    shop_rows = _records(shops, MAX_SHOPS)
    shop_rows = [r for r in shop_rows if not str(r.get("Shop") or "").startswith("Not listed")]
    return {
        "period": pack.period,
        "label": pack.label,
        "pace": {
            "open_mtd": kpis.get("open_mtd"),
            "intra_month_frac": kpis.get("intra_month_frac"),
            "n_history_periods": kpis.get("n_history_periods"),
            "n_same_month": kpis.get("n_same_month"),
        },
        "country": country,
        "country_kpis": {
            "billed_mt": kpis.get("billed_mt"),
            "expected_mt": kpis.get("expected_mt"),
            "gap_vs_expected_mt": kpis.get("gap_mt"),
            "ly_mt": kpis.get("ly_mt"),
            "country_recoverable_mt": rec_country,
            "n_lagging_cities": kpis.get("n_lagging_cities") if kpis.get("n_lagging_cities") is not None else int(len(lag)),
        },
        "lagging_cities": _records(lag),
        "lagging_city_share_of_country_recoverable": lag_share,
        "ahead_cities": _records(ahead),
        "on_expected_cities": _records(with_c),
        "city_share_of_country_billed": city_billed_share,
        "lagging_distributors_top": _records(dists, MAX_LAGGING_DISTS),
        "n_lagging_distributors": int(len(dists)) if dists is not None else 0,
        "lagging_dsrs_top": _records(dsrs, MAX_LAGGING_DSRS),
        "n_lagging_dsrs": int(len(dsrs)) if dsrs is not None else 0,
        "lagging_shops_top": shop_rows,
        "n_lagging_shops_listed": int(len(shops)) if shops is not None else 0,
        "shop_remainder": pack.shop_note,
        "hidden_shop_recoverable_mt": kpis.get("hidden_shop_recoverable_mt"),
        "n_shops_hidden": kpis.get("n_shops_hidden"),
    }


def _strip_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _validate_summary(payload: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    situation_raw = payload.get("situation") or payload.get("current_situation") or []
    if isinstance(situation_raw, str):
        situation = [p.strip() for p in situation_raw.split("\n\n") if p.strip()]
    elif isinstance(situation_raw, list):
        situation = [str(x).strip() for x in situation_raw if str(x).strip()]
    else:
        situation = []
    focus_raw = payload.get("focus") or payload.get("key_focus_areas") or []
    focus: list[dict[str, str]] = []
    if isinstance(focus_raw, list):
        for item in focus_raw:
            if isinstance(item, str) and item.strip():
                focus.append({"title": item.strip(), "why": "", "do": ""})
                continue
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("name") or "").strip()
            why = str(item.get("why") or item.get("because") or "").strip()
            do = str(item.get("do") or item.get("action") or item.get("next") or "").strip()
            if title or why or do:
                focus.append({"title": title, "why": why, "do": do})
    if not situation:
        raise ValueError("Model returned no situation paragraphs.")
    if not focus:
        raise ValueError("Model returned no focus areas.")
    return situation, focus


def _chat(api_key: str, model: str, brief: dict[str, Any]) -> tuple[dict[str, Any], str]:
    user = (
        "Write the national executive summary from this BRIEF JSON. "
        "Cite only these figures. Two sections: situation paragraphs, then focus areas.\n\n"
        + json.dumps(brief, ensure_ascii=False, default=str)
    )
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Column glossary:\n"
                + "\n".join(f"- {t}: {d}" for t, d in GLOSSARY)
                + "\nHow values are calculated:\n"
                + "\n".join(f"- {t}: {d}" for t, d in CALCULATION_NOTES),
            },
            {"role": "user", "content": user},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {err_body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI returned no choices.")
    content = ((choices[0] or {}).get("message") or {}).get("content") or ""
    parsed = json.loads(_strip_fence(content))
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI JSON was not an object.")
    return parsed, json.dumps(raw, ensure_ascii=False, default=str)


def generate_exec_summary(pack: StrategyPack, api_key: str | None = None, model: str | None = None) -> dict[str, Any]:
    settings = load_openai_settings()
    key = (api_key or settings.get("api_key") or "").strip()
    chosen = (model or settings.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    brief = build_grounded_brief(pack)
    if not key:
        return {
            "ok": False,
            "error": "No OpenAI API key. Paste one on Upload files, then rebuild scorecards.",
            "model": chosen,
            "brief": brief,
            "situation": [],
            "focus": [],
            "raw": "",
        }
    models = [chosen]
    if FALLBACK_MODEL not in models:
        models.append(FALLBACK_MODEL)
    last_error = ""
    for attempt in models:
        try:
            parsed, raw = _chat(key, attempt, brief)
            situation, focus = _validate_summary(parsed)
            return {
                "ok": True,
                "error": "",
                "model": attempt,
                "brief": brief,
                "situation": situation,
                "focus": focus,
                "raw": raw,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{attempt}: {exc}"
            continue
    return {
        "ok": False,
        "error": last_error or "OpenAI call failed.",
        "model": chosen,
        "brief": brief,
        "situation": [],
        "focus": [],
        "raw": "",
    }


def store_exec_summary(conn: Any, period: str, result: dict[str, Any]) -> None:
    conn.execute(
        """INSERT INTO exec_summary(period, model, situation_json, focus_json, brief_json, raw_json, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(period) DO UPDATE SET
             model=excluded.model,
             situation_json=excluded.situation_json,
             focus_json=excluded.focus_json,
             brief_json=excluded.brief_json,
             raw_json=excluded.raw_json,
             error=excluded.error,
             created_at=excluded.created_at""",
        (
            period,
            result.get("model") or "",
            json.dumps(result.get("situation") or [], ensure_ascii=False),
            json.dumps(result.get("focus") or [], ensure_ascii=False),
            json.dumps(result.get("brief") or {}, ensure_ascii=False, default=str),
            result.get("raw") or "",
            result.get("error") or "",
            utcnow(),
        ),
    )


def load_exec_summary_row(conn: Any, period: str | None) -> dict[str, Any] | None:
    if not period:
        return None
    try:
        cur = conn.execute("SELECT * FROM exec_summary WHERE period = ?", (str(period),))
    except Exception:
        return None
    row = cur.fetchone()
    if row is None:
        return None
    if hasattr(row, "keys"):
        return {k: row[k] for k in row.keys()}
    return None


def exec_row_from_frame(df: pd.DataFrame | None, period: str | None) -> dict[str, Any] | None:
    if df is None or df.empty or not period or "period" not in df.columns:
        return None
    hit = df[df["period"].astype(str) == str(period)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def refresh_exec_summary(conn: Any, pack: StrategyPack) -> dict[str, Any]:
    """Call the model for a national pack and persist. Never raises for API/key issues."""
    if not is_national_pack(pack) or not pack.period:
        return {"ok": False, "error": "Executive summary is national only.", "skipped": True}
    try:
        result = generate_exec_summary(pack)
        store_exec_summary(conn, pack.period, result)
        return {
            "ok": bool(result.get("ok")),
            "error": result.get("error") or "",
            "model": result.get("model") or "",
            "skipped": False,
        }
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        store_exec_summary(
            conn,
            pack.period,
            {
                "ok": False,
                "error": err,
                "model": load_openai_settings().get("model") or DEFAULT_MODEL,
                "brief": {},
                "situation": [],
                "focus": [],
                "raw": "",
            },
        )
        return {"ok": False, "error": err, "skipped": False}
