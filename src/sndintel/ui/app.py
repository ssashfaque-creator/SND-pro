"""Local briefing app: upload files, then city → distributor → DSR → shop scorecards.

Warehouse lives in the user data directory so reinstalling the code does not
require re-uploading history.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sndintel.action import build_action_pack, load_action_pack
from sndintel.action_report import GLOSSARY as ACTION_GLOSSARY
from sndintel.action_report import excel_bytes as action_excel_bytes
from sndintel.action_report import pdf_bytes as action_pdf_bytes
from sndintel.briefing import (
    CALCULATION_NOTES,
    GLOSSARY,
    build_strategy_pack,
    excel_bytes,
    excel_bytes_detailed,
    focus_pack,
    list_report_entities,
    pdf_bytes,
    pdf_bytes_detailed,
)
from sndintel.situation_report import (
    build_field_packs,
    build_situation_pack,
    excel_bytes as situation_excel_bytes,
    list_situation_entities,
    pdf_bytes as situation_pdf_bytes,
    scorecards_for_period,
    zip_field_packs,
    _gap_sentence,
)
from sndintel import __version__
from sndintel.config import DATA_DIR, DB_PATH, INCOMING_DIR, MASTER_DIR, ensure_dirs
from sndintel.ingest.pipeline import clear_billed_sales, rescore_warehouse, run_pipeline
from sndintel.mtd import banner_text, format_period_label, period_state
from sndintel.monday import monday_summary_sheets
from sndintel.ops import (
    beat_owner_options,
    build_dsr_beat_pack,
    build_friday_pack,
    build_monday_pack,
    excel_bytes as ops_excel_bytes,
    filter_beat_by_owner,
    load_outcomes,
    pdf_bytes as ops_pdf_bytes,
)
from sndintel.storage import connect, init_db, read_sql

DIAGNOSIS_COLOR = {
    "drop_size": "#b91c1c",
    "coverage": "#c2410c",
    "whitespace": "#334155",
    "mixed": "#7c3aed",
    "holding": "#15803d",
}

SITUATION_COLOR = {
    "lagging": "#b91c1c",
    "with_market": "#64748b",
    "outperforming": "#15803d",
}


def _save_upload(file, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(file.getvalue())
    return dest


def _remember(kind: str, path: Path) -> None:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    (MASTER_DIR / f"last_{kind}.json").write_text(json.dumps({"path": str(path)}), encoding="utf-8")


def _last(kind: str) -> Path | None:
    marker = MASTER_DIR / f"last_{kind}.json"
    if not marker.exists():
        return None
    try:
        path = Path(json.loads(marker.read_text(encoding="utf-8"))["path"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return path if path.exists() else None


def _remember_shop(path: Path) -> None:
    _remember("shop", path)


def _last_shop() -> Path | None:
    return _last("shop")


def _warehouse_has_stores() -> bool:
    try:
        with connect() as conn:
            rows = read_sql(conn, "SELECT 1 AS x FROM stores LIMIT 1")
        return rows is not None and not rows.empty
    except Exception:
        return False


def _exec_row(data, period: str):
    from sndintel.narrative import exec_row_from_frame

    return exec_row_from_frame(data.get("exec_summary"), period)


def _strategy_pack(data, period, ledger):
    pack = build_strategy_pack(
        data.get("units", pd.DataFrame()),
        data.get("shop_month", pd.DataFrame()),
        situation=data.get("situation", pd.DataFrame()),
        ledger=ledger,
        period=period,
        visits=data.get("visits", pd.DataFrame()),
        exec_summary=_exec_row(data, period),
    )
    return pack


@st.cache_data(ttl=15)
def load_all():
    ensure_dirs()
    init_db()
    with connect() as conn:
        data = {
            "insights": read_sql(conn, "SELECT * FROM insights ORDER BY rank_score DESC"),
            "kpis": read_sql(conn, "SELECT * FROM kpi_snapshots"),
            "shop_month": read_sql(conn, "SELECT * FROM shop_month"),
            "anomalies": read_sql(conn, "SELECT * FROM anomalies"),
            "segments": read_sql(conn, "SELECT * FROM shop_segments"),
            "stores": read_sql(conn, "SELECT * FROM stores"),
            "forecasts": read_sql(conn, "SELECT * FROM forecasts"),
            "runs": read_sql(conn, "SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT 8"),
            "ledger": read_sql(conn, "SELECT * FROM period_ledger ORDER BY period"),
        }
        try:
            data["plays"] = read_sql(conn, "SELECT * FROM strategy_plays ORDER BY slot")
        except Exception:
            data["plays"] = pd.DataFrame()
        try:
            data["units"] = read_sql(conn, "SELECT * FROM unit_scorecards")
        except Exception:
            data["units"] = pd.DataFrame()
        try:
            data["targets"] = read_sql(conn, "SELECT * FROM focus_targets ORDER BY rank")
        except Exception:
            data["targets"] = pd.DataFrame()
        try:
            data["situation"] = read_sql(conn, "SELECT * FROM situation_brief")
        except Exception:
            data["situation"] = pd.DataFrame()
        try:
            data["seasonality"] = read_sql(conn, "SELECT * FROM seasonality_index")
        except Exception:
            data["seasonality"] = pd.DataFrame()
        try:
            data["visits"] = read_sql(conn, "SELECT * FROM shop_visits")
        except Exception:
            data["visits"] = pd.DataFrame()
        try:
            data["exec_summary"] = read_sql(conn, "SELECT * FROM exec_summary")
        except Exception:
            data["exec_summary"] = pd.DataFrame()
        try:
            data["action_brief"] = read_sql(conn, "SELECT * FROM action_brief")
            data["action_shops"] = read_sql(conn, "SELECT * FROM action_shops")
            data["action_units"] = read_sql(conn, "SELECT * FROM action_units")
            data["action_backtest"] = read_sql(conn, "SELECT * FROM action_backtest")
        except Exception:
            data["action_brief"] = pd.DataFrame()
            data["action_shops"] = pd.DataFrame()
            data["action_units"] = pd.DataFrame()
            data["action_backtest"] = pd.DataFrame()
        try:
            data["shop_day"] = read_sql(conn, "SELECT * FROM shop_day")
        except Exception:
            data["shop_day"] = pd.DataFrame()
        try:
            data["shop_targets"] = read_sql(conn, "SELECT * FROM shop_targets")
        except Exception:
            data["shop_targets"] = pd.DataFrame()
        try:
            data["mtd_obs"] = read_sql(conn, "SELECT * FROM mtd_observations")
        except Exception:
            data["mtd_obs"] = pd.DataFrame()
    return data


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.0rem; max-width: 1480px;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        .sit-card {border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: 0.9rem; color: #0f172a;}
        .sit-kicker {font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; margin-bottom: 0.35rem;}
        .sit-headline {font-size: 1.55rem; line-height: 1.25; font-weight: 700; margin: 0 0 0.55rem 0;}
        .sit-body {font-size: 1.02rem; line-height: 1.45; margin: 0 0 0.4rem 0;}
        .sit-action {font-size: 1.02rem; line-height: 1.45; margin: 0; font-weight: 600;}
        [data-testid="stDataFrame"] td {white-space: pre-wrap;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="SND Intelligence", layout="wide", page_icon="▣")
    _inject_css()
    ensure_dirs()

    data = load_all()
    empty = data["shop_month"].empty or data["kpis"].empty

    st.sidebar.title("SND Intelligence")
    st.sidebar.caption("Secondary sales · shop × SKU · Pakistan S&D")
    page = st.sidebar.radio(
        "Workspace",
        ["Strategy", "This week", "Report", "Upload files", "Focus", "People", "Mix", "Shops", "Warehouse"],
        index=1,
    )
    st.sidebar.divider()
    st.sidebar.markdown(f"**Warehouse** `{DB_PATH}`")
    st.sidebar.caption(
        "History lives here, not inside the app folder. Updating the code does not wipe months you already ingested."
    )
    runs = data["runs"]
    if not runs.empty:
        st.sidebar.write(f"Last run `{runs.iloc[0]['status']}`")
        st.sidebar.caption(str(runs.iloc[0].get("sales_file") or ""))

    if page == "Upload files" or empty:
        _page_upload(empty)
        if empty or page == "Upload files":
            return

    kpis = data["kpis"]
    national = kpis[kpis["grain"] == "national"].sort_values("period")
    latest = national.iloc[-1]
    period = latest["period"]
    ledger = data.get("ledger", pd.DataFrame())
    mtd = period_state(ledger, period)

    if page == "Strategy":
        _page_strategy(data, latest, period, mtd, ledger)
    elif page == "This week":
        _page_this_week(data, period, mtd, ledger)
    elif page == "Report":
        _page_report(data, latest, period, mtd, ledger)
    elif page == "Focus":
        _page_focus(data, period)
    elif page == "People":
        _page_people(data, period, mtd)
    elif page == "Mix":
        _page_mix(data, period)
    elif page == "Shops":
        _page_shops(data, period)
    elif page == "Warehouse":
        _page_warehouse(data)


def _page_upload(empty: bool):
    st.title("Upload files")
    st.caption(
        "POP code is the shop. Names, DSR, distributor, and city change — the **Universe shop list** is the live book. "
        "Sales history of POPs not on that list is ignored. Visit calls land weekly with the sales extract."
    )
    if empty:
        st.warning("No billed months yet. Universe can stay as it is — upload one or more Outlet Date Wise files.")
    else:
        st.success("Shop lists already in the warehouse. Upload sales to replace billed rows (universe stays unless you re-upload it).")

    last_uni = _last("universe")
    last_shop = _last_shop()
    last_vis = _last("visits")
    last_tgt = _last("targets")
    has_stores = _warehouse_has_stores()
    c1, c2 = st.columns(2)
    with c1:
        universe = st.file_uploader(
            "Universe shop list (xlsx) — live distributors, DSRs, shops",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="universe",
        )
        if last_uni and universe is None:
            st.caption(f"Using saved universe: `{last_uni.name}`")
        elif has_stores and universe is None:
            st.caption("Universe already in the warehouse — no need to re-upload.")
        shops = st.file_uploader(
            "Legacy shop list (optional — zone / historical map)",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="shops",
        )
        if last_shop and shops is None:
            st.caption(f"Saved legacy master: `{last_shop.name}`")
    with c2:
        sales_files = st.file_uploader(
            "Sales extracts — one or more Outlet Date Wise / Shop SKU Wise files",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="sales",
            accept_multiple_files=True,
            help="Split a large daily file by shops or by date range. Later files override overlapping shop-days; other days stay.",
        )
        visits = st.file_uploader(
            "Shop visit calls (csv / xlsx) — MTD visits, weekly with sales",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="visits",
        )
        if last_vis and visits is None:
            st.caption(f"Last visit file: `{last_vis.name}` (re-upload each week)")
        targets = st.file_uploader(
            "Shop-wise targets (csv / xlsx) — sales-team quota / plan",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="targets",
            help="Region, area, distributor, DSR, shop name, target MT. Does not replace Expected. Matched to the universe by name.",
        )
        if last_tgt and targets is None:
            st.caption(f"Plan on file: `{last_tgt.name}` (re-upload when quotas change)")

    st.markdown(
        "- **Universe** can stay in the warehouse. Re-upload only when shops/DSRs move.\n"
        "- **Outlet Date Wise:** days in these files **replace** the same shop-days already stored "
        "(a later file that includes 20 Aug overrides an earlier incomplete 20 Aug). "
        "Days and shops not in the file stay. Shop-split files in one drop are combined. "
        "Do not tick replace-all for a weekly MTD refresh.\n"
        "- **Shop SKU Wise:** months in the file replace those months (August-only keeps July).\n"
        "- Every later week: **sales + visit calls**. Tick replace-all only to wipe billed history "
        "(for example switching from Shop SKU Wise to Outlet Date Wise).\n"
        "- **Shop-wise targets** are the sales-team plan (quota), not Expected. Expected stays the "
        "statistical run-rate. The situation pack then shows billed vs Expected vs plan, and whether "
        "a miss is execution (behind run-rate) or stretch (behind quota)."
    )
    replace_sales = st.checkbox(
        "Replace all billed sales (keep universe and visits)",
        value=False,
        help="Off (default): Outlet Date Wise overrides overlapping shop-days and rebuilds those months from the warehouse; other days stay. Shop SKU Wise still replaces each month the file contains. On: wipe every billed row first, then load only these files.",
    )

    st.divider()
    st.subheader("National executive summary")
    st.caption(
        "After each upload or rebuild, the national PDF opens with a glossary, then an AI summary of the current situation and key focus areas. "
        "The model is given the same rounded country / city / distributor / DSR / shop figures as the pack. It cannot invent volumes. "
        "City, distributor, and DSR reports skip the AI page. The key stays on this machine — it is not written into the warehouse or git."
    )
    from sndintel.narrative import (
        DEFAULT_MODEL,
        has_openai_key,
        load_openai_settings,
        refresh_exec_summary,
        save_openai_settings,
    )

    settings = load_openai_settings()
    models = ["gpt-4.1", "gpt-4o"]
    stored_model = settings.get("model") or DEFAULT_MODEL
    if stored_model not in models:
        models = [stored_model, *models]
    ckey, cmodel = st.columns([2, 1])
    with ckey:
        new_key = st.text_input(
            "OpenAI API key",
            type="password",
            value="",
            placeholder="Key saved — paste a new one to replace" if settings.get("has_stored_key") or has_openai_key() else "sk-…",
            help="Stored under the SND Intelligence data directory (chmod 600). Environment OPENAI_API_KEY also works.",
        )
    with cmodel:
        model_choice = st.selectbox("Model", models, index=models.index(stored_model) if stored_model in models else 0)
    k1, k2 = st.columns(2)
    with k1:
        if st.button("Save API key"):
            if new_key.strip():
                save_openai_settings(api_key=new_key.strip(), model=model_choice)
                st.success(f"Key saved. National summaries will use {model_choice} after the next score or generate.")
            else:
                save_openai_settings(model=model_choice)
                if has_openai_key():
                    st.success(f"Model set to {model_choice}. Existing key kept.")
                else:
                    st.warning("No key yet. Paste an OpenAI key, then save.")
    with k2:
        if st.button("Generate national summary now"):
            units = None
            try:
                with connect() as conn:
                    units = read_sql(conn, "SELECT * FROM unit_scorecards")
                    shop_month = read_sql(conn, "SELECT * FROM shop_month")
                    try:
                        situation = read_sql(conn, "SELECT * FROM situation_brief")
                    except Exception:
                        situation = pd.DataFrame()
                    try:
                        visits = read_sql(conn, "SELECT * FROM shop_visits")
                    except Exception:
                        visits = pd.DataFrame()
                    ledger = read_sql(conn, "SELECT * FROM period_ledger ORDER BY period")
                    if units is None or units.empty:
                        st.error("No scorecards yet. Upload files or rebuild first.")
                    else:
                        from sndintel.features import latest_period

                        period = latest_period(shop_month) if shop_month is not None and not shop_month.empty else ""
                        pack = build_strategy_pack(
                            units,
                            shop_month,
                            situation=situation,
                            ledger=ledger,
                            period=period,
                            visits=visits,
                        )
                        with st.spinner("Writing the national executive summary from this period’s scorecards."):
                            meta = refresh_exec_summary(conn, pack)
                        st.cache_data.clear()
                        if meta.get("ok"):
                            st.success(f"National executive summary stored ({meta.get('model') or model_choice}).")
                        else:
                            st.warning(meta.get("error") or "Summary was not generated.")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    go = st.button("Score warehouse", type="primary", disabled=not sales_files and empty and targets is None)
    if not go:
        return
    universe_path = None
    if universe is not None:
        universe_path = _save_upload(universe, MASTER_DIR / universe.name)
        _remember("universe", universe_path)
    elif not has_stores:
        universe_path = last_uni
    shop_path = None
    if shops is not None:
        shop_path = _save_upload(shops, MASTER_DIR / shops.name)
        _remember_shop(shop_path)
    elif not has_stores:
        shop_path = last_shop
    if empty and universe_path is None and shop_path is None and not has_stores:
        st.error("Upload the universe shop list once, or keep the one already in the warehouse.")
        return
    sales_paths = []
    if sales_files:
        for i, uploaded in enumerate(sales_files):
            dest = INCOMING_DIR / f"{i:02d}_{uploaded.name}"
            sales_paths.append(_save_upload(uploaded, dest))
    if empty and not sales_paths:
        st.error("Drop one or more sales files (Outlet Date Wise can be split).")
        return
    visits_path = None
    if visits is not None:
        visits_path = _save_upload(visits, INCOMING_DIR / visits.name)
        _remember("visits", visits_path)
    targets_path = None
    if targets is not None:
        targets_path = _save_upload(targets, MASTER_DIR / targets.name)
        _remember("targets", targets_path)
    with st.spinner("Parsing files and rebuilding city → distributor → DSR → shop scorecards. Large files take a few minutes."):
        try:
            result = run_pipeline(
                sales_paths=sales_paths or None,
                shop_path=shop_path,
                universe_path=universe_path,
                visits_path=visits_path,
                targets_path=targets_path,
                replace_sales=bool(replace_sales and sales_paths),
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return
    st.cache_data.clear()
    overlay_note = ""
    if result.get("ingest_mode") == "daily_overlay" and not result.get("replace_sales"):
        n_days = len(result.get("overridden_dates") or [])
        overlay_note = (
            f"Daily overlay: {n_days} billed day(s) in these files replaced matching warehouse shop-days; "
            "other days stayed. "
        )
    st.success(
        f"Scored {result.get('n_sales_rows')} fact rows · {result.get('n_sales_files') or 0} sales file(s) · "
        f"latest {result.get('latest_period')} · "
        f"{result.get('n_cities', 0)} cities · universe {result.get('n_universe') or '—'} · "
        f"visits {result.get('n_visits') or '—'} · "
        f"plan {result.get('n_plan_matched') or '—'} shops / {result.get('plan_matched_mt') or 0:.0f} MT matched. "
        f"{'Replaced all billed sales. ' if result.get('replace_sales') else overlay_note}"
        f"Months: {', '.join(result.get('replaced_periods') or []) or '—'}"
    )
    if result.get("open_mtd_period"):
        st.info(f"Open MTD: {result['open_mtd_period']}")
    if result.get("exec_ok"):
        st.success(f"National executive summary written ({result.get('exec_model') or 'OpenAI'}).")
    elif result.get("exec_error"):
        st.warning(f"Scorecards rebuilt; executive summary skipped. {result.get('exec_error')}")
    st.rerun()


def _page_strategy(data, _latest, period, mtd, ledger):
    st.title("Briefing")
    st.caption(
        f"**{mtd['label'] or period}** · expected is the recent run-rate (last three closed months), "
        "paced if the month is still open by the country’s usual billed share through that day. "
        "An empty August last year does not zero Expected."
    )
    if mtd["open"]:
        st.info(banner_text(ledger, period))

    units = data.get("units", pd.DataFrame())
    if units is None or units.empty:
        st.warning(
            "No scorecards yet. If sales are already in the warehouse, rebuild below. "
            "Otherwise upload the shop list and sales extract."
        )
        _rescore_button()
        return

    pack = _strategy_pack(data, period, ledger)
    for warning in getattr(pack, "visit_warnings", None) or []:
        st.warning(warning)
    if pack.exec_situation:
        st.markdown(
            '<div class="sit-card" style="background:#f8fafc;border-left:8px solid #0f172a">'
            '<div class="sit-kicker">Executive summary · national</div>'
            "<h3 style='margin:0 0 0.4rem 0;'>Summary of current situation</h3>"
            + "".join(f'<p class="sit-body">{html.escape(p)}</p>' for p in pack.exec_situation)
            + "<h3 style='margin:0.8rem 0 0.4rem 0;'>Key focus areas</h3>"
            + "".join(
                "<p class='sit-body'><b>"
                + html.escape(item.get("title") or "")
                + "</b> "
                + html.escape(item.get("why") or "")
                + ((" Do this week. " + html.escape(item.get("do") or "")) if item.get("do") else "")
                + "</p>"
                for item in pack.exec_focus
            )
            + (
                f'<p class="sit-body" style="opacity:0.7">Model {html.escape(pack.exec_model)}. Figures match the tables below.</p>'
                if pack.exec_model
                else ""
            )
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        national = units[units["grain"] == "national"]
        nat = national.iloc[0] if not national.empty else None
        sit_df = data.get("situation", pd.DataFrame())
        sit = sit_df.iloc[0] if sit_df is not None and not sit_df.empty else None
        headline = (sit["headline"] if sit is not None else None) or (
            nat.get("do_this_week") if nat is not None else None
        )
        weather = sit["weather"] if sit is not None else ""
        problem = sit["problem"] if sit is not None else ""
        action = sit["action_summary"] if sit is not None else ""
        weather_dir = "declining"
        if nat is not None and float(nat.get("gap_mt") or 0) > 1:
            weather_dir = "growing"
        elif nat is not None and abs(float(nat.get("gap_mt") or 0)) <= 1:
            weather_dir = "flat"
        bar = {"declining": "#b91c1c", "growing": "#15803d", "flat": "#334155"}[weather_dir]
        miss = html.escape(
            pack.exec_error
            or "Paste an OpenAI key on Upload files, then rebuild, to write the national executive summary."
        )
        st.markdown(
            f'<div class="sit-card" style="background:#f8fafc;border-left:8px solid {bar}">'
            f'<div class="sit-kicker">Situation</div>'
            f'<p class="sit-headline">{html.escape(str(headline or "Scorecards ready"))}</p>'
            f'<p class="sit-body">{html.escape(str(weather or ""))}</p>'
            f'<p class="sit-body"><b>The problem.</b> {html.escape(str(problem or ""))}</p>'
            f'<p class="sit-action">Do this week. {html.escape(str(action or ""))}</p>'
            f'<p class="sit-body" style="opacity:0.75">{miss}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    cities = units[units["grain"] == "city"].copy()
    if "recoverable_mt" in cities.columns:
        rec = pd.to_numeric(cities["isolated_mt"], errors="coerce").fillna(0).clip(upper=0).abs()
        cities["recoverable_mt"] = rec
        cities = cities.sort_values("recoverable_mt", ascending=False)
    elif "isolated_mt" in cities.columns:
        cities = cities.sort_values("isolated_mt")
    else:
        cities = cities.sort_values("gap_mt")
    cdl, cdr = st.columns(2)
    with cdl:
        st.download_button(
            "Download strategy pack (Excel)",
            excel_bytes(pack),
            file_name=f"SND_strategy_{period}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
    with cdr:
        st.download_button(
            "Download strategy pack (PDF)",
            pdf_bytes(pack),
            file_name=f"SND_strategy_{period}.pdf",
            mime="application/pdf",
        )
    ddl, ddr = st.columns(2)
    with ddl:
        st.download_button(
            "Download detailed pack (Excel)",
            excel_bytes_detailed(pack),
            file_name=f"SND_strategy_{period}_detailed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with ddr:
        st.download_button(
            "Download detailed pack (PDF)",
            pdf_bytes_detailed(pack),
            file_name=f"SND_strategy_{period}_detailed.pdf",
            mime="application/pdf",
        )
    st.caption(
        "Excel is the working file (filters, one sheet per layer). "
        "PDF is the board pack — whole numbers, remarks as bullets in the last column. "
        "For a city / distributor / DSR pack, use **Report**. "
        "National summary has two rankings per grain: Gap tons (the books that close the month) "
        "and seriousness (Gap ÷ √Expected — collapsed mid-size names). Whales are AMS / last drop ≥ 1 MT. "
        "A DSR is city + distributor + name. Detailed pack = every city, every distributor and DSR with AMS > 0, "
        "and every shop with gap > 0.25 MT."
    )

    st.markdown("##### 1. The country — every city")
    st.caption(
        "Start here. The first row is the **country**. **Gap** is billed versus this unit’s own Expected (recent run-rate, paced if MTD is open). "
        "**From drop size / unvisited / unbilled** add to Gap (positive = hole; negative = billed more than Expected). "
        "**Remarks** (last column) are four bullets: trend, coverage, productivity, drop size. "
        "Distributors and DSRs with AMS = 0 are hidden later."
    )
    left, right = st.columns((1.4, 1))
    with left:
        if cities.empty:
            st.write("No city rows.")
        else:
            chart = cities.head(16).copy()
            chart["city"] = chart["grain_id"]
            if "recoverable_mt" in chart.columns:
                ycol = "recoverable_mt"
                ylab = "Gap (MT)"
            else:
                ycol = "isolated_mt" if "isolated_mt" in chart.columns else "gap_mt"
                ylab = "Gap (MT)"
            color = "situation" if "situation" in chart.columns else "diagnosis"
            cmap = SITUATION_COLOR if color == "situation" else DIAGNOSIS_COLOR
            fig = px.bar(
                chart,
                x="city",
                y=ycol,
                color=color,
                color_discrete_map=cmap,
                labels={ycol: ylab, "city": ""},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
            fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        _strategy_table(pack.cities)
    with right:
        st.markdown("**Why the hole**")
        st.caption("Coverage = fewer billed doors. Drop size = smaller drops on the same doors. Mix = SKU shift.")
        lag = cities[cities["situation"] == "lagging"] if "situation" in cities.columns else cities.head(4)
        if lag.empty:
            lag = cities.head(4)
        if not lag.empty and {"coverage_effect_mt", "velocity_effect_mt"}.issubset(lag.columns):
            long = lag.melt(
                id_vars=["grain_id"],
                value_vars=[c for c in ["coverage_effect_mt", "velocity_effect_mt", "mix_effect_mt"] if c in lag.columns],
                var_name="driver",
                value_name="mt",
            )
            long["driver"] = long["driver"].str.replace("_effect_mt", "")
            fig = px.bar(
                long,
                x="grain_id",
                y="mt",
                color="driver",
                barmode="relative",
                labels={"grain_id": "", "mt": "MT"},
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        trend = data["shop_month"].groupby("period", as_index=False)["volume_mt"].sum().sort_values("period")
        fig = px.line(trend, x="period", y="volume_mt", markers=True, labels={"volume_mt": "MT", "period": ""})
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), title="National volume")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 2. Top 10 distributors by Gap tons")
    st.caption("The books that close the month. Ranked by Gap MT, not seriousness. AMS = 0 is hidden.")
    vol_dist = pack.volume_distributors if pack.volume_distributors is not None and not pack.volume_distributors.empty else pack.lagging_distributors
    if vol_dist is None or vol_dist.empty:
        st.info("No lagging distributor with a recent run-rate.")
    else:
        _strategy_table(vol_dist)

    st.markdown("##### 3. Top 10 most serious lagging distributors")
    st.caption(
        "Ranked by Gap ÷ √Expected — collapsed mid-size names beat large books that are only slightly light. Remainder line is everyone after the tenth."
    )
    if pack.lagging_distributors.empty:
        st.info("No lagging distributor with a recent run-rate.")
    else:
        _strategy_table(pack.lagging_distributors)

    st.markdown("##### 4. Top 10 DSRs by Gap tons")
    st.caption("City + distributor + name. Ranked by Gap MT. Two people with the same first name never share a row.")
    vol_dsr = pack.volume_dsrs if pack.volume_dsrs is not None and not pack.volume_dsrs.empty else pack.lagging_dsrs
    if vol_dsr is None or vol_dsr.empty:
        st.info("No lagging DSR with a recent run-rate.")
    else:
        _strategy_table(vol_dsr)

    st.markdown("##### 5. Top 10 most serious lagging DSRs")
    st.caption(
        "One national list — not nested under the ten distributors. Ranked by miss versus own Expected given size. AMS = 0 is hidden."
    )
    if pack.lagging_dsrs.empty:
        st.info("No lagging DSR with a recent run-rate.")
    else:
        _strategy_table(pack.lagging_dsrs)

    st.markdown("##### 6. Whales")
    st.caption("AMS or last drop ≥ 1 MT. These close the month. Kiryana seriousness lists do not.")
    if pack.whales is None or pack.whales.empty:
        st.info("No whale doors at the 1 MT floor this period.")
    else:
        _strategy_table(pack.whales, height=280)

    st.markdown("##### 7. Top 50 most serious shops")
    st.caption(pack.shop_note or "One national list of the 50 most serious doors after the 0.25 MT floor. The rest of the hole is the remainder line.")
    _strategy_table(pack.lagging_shops, height=420)

    st.markdown("##### 8. DSR capacity")
    st.caption("Overloaded = not enough DSRs. Not working the beat = spare capacity, weak visit %. Not converting = visits happened, shops did not buy. Not lifting drop = billed, order size is light.")
    if pack.capacity_dsrs is None or pack.capacity_dsrs.empty:
        st.info("No DSR capacity labels yet.")
    else:
        _strategy_table(pack.capacity_dsrs, height=320)

    with st.expander("How to read the columns", expanded=False):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}.** {meaning}")
        st.markdown("##### How the figures are calculated")
        for term, meaning in CALCULATION_NOTES:
            st.markdown(f"**{term}.** {meaning}")

    _rescore_button()

    with st.expander("Evidence — ranked flags behind the briefing"):
        ins = data["insights"]
        if ins.empty:
            st.write("No insights stored.")
        else:
            show = ins[~ins["type"].isin(["portfolio_mix"])].head(20)
            for rec in show.itertuples(index=False):
                st.markdown(f"**{rec.title}** · _{rec.severity} · {rec.type}_")
                st.write(rec.narrative)
                st.caption(rec.action)


def _strategy_table(df: pd.DataFrame, height: int = 320):
    if df is None or df.empty:
        st.caption("No rows at this layer.")
        return
    cfg = {}
    for col in df.columns:
        name = str(col)
        if name in {"Remarks", "Do this", "Do this week", "Step", "Comment"}:
            cfg[col] = st.column_config.TextColumn(name, width="large")
        elif name == "Drop size (MT)":
            cfg[col] = st.column_config.NumberColumn(name, format="%.2f")
        elif "(MT)" in name:
            cfg[col] = st.column_config.NumberColumn(name, format="%.0f")
        elif name.endswith("%") or name == "Strike %":
            cfg[col] = st.column_config.NumberColumn(name, format="%.0f")
        elif name in {"Billed shops", "Visited shops", "Universe", "Visits MTD"}:
            cfg[col] = st.column_config.NumberColumn(name, format="%.0f")
    n = max(3, len(df))
    row_h = 56 if "Remarks" in df.columns else (40 if "Do this" in df.columns else 28)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(max(height, 160), 80 + row_h * min(n, 12)),
        column_config=cfg,
    )


def _ops_downloads(pack, period: str, stem: str, key_prefix: str):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Excel",
            ops_excel_bytes(pack),
            file_name=f"SND_{stem}_{period}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xlsx",
        )
    with c2:
        st.download_button(
            "PDF",
            ops_pdf_bytes(pack),
            file_name=f"SND_{stem}_{period}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
        )


def _render_ops_sheets(sheets, height: int = 280):
    if not sheets:
        st.caption("No rows at this layer.")
        return
    for _sheet, heading, note, df in sheets:
        st.markdown(f"**{heading}**")
        if note:
            st.caption(note)
        _strategy_table(df, height=height)


def _page_this_week(data, period, mtd, ledger):
    st.title("This week")
    st.caption(
        f"**{mtd['label'] or period}** · Monday dispatch is the operating view. "
        "Ask is the shop’s 90-day expected drop when the depletion ratio is ≥ 0.8. "
        "Official Expected on Gap cards is still last-3 / last-6 paced by the national day curve."
    )
    if mtd.get("open"):
        st.info(banner_text(ledger, period))
    pack = None
    brief = data.get("action_brief", pd.DataFrame())
    if brief is not None and not brief.empty:
        with connect() as conn:
            pack = load_action_pack(conn, period)
    if pack is None or not pack.headline:
        shop_month = data.get("shop_month", pd.DataFrame())
        if shop_month is None or shop_month.empty:
            st.warning("No scorecards yet. Upload Outlet Date Wise and rebuild.")
            _rescore_button()
            return
        pack = build_action_pack(
            shop_month,
            data.get("stores"),
            shop_day=data.get("shop_day"),
            visits=data.get("visits"),
            ledger=ledger,
            period=period,
        )
    if not pack.headline and pack.country.empty:
        st.warning("No action list yet. Rebuild after daily sales are in the warehouse.")
        _rescore_button()
        return

    st.markdown(f"**{pack.headline}**")
    if pack.source == "monthly":
        st.caption("Daily billed days were not found — cycles fall back to monthly gaps. Upload Outlet Date Wise for exact days-between-bills.")
    else:
        st.caption("90-day purchase cycle (API and expected drop). Ask is the expected drop when due; lapsed doors have Ask 0.")

    units = data.get("units", pd.DataFrame())
    visits = data.get("visits", pd.DataFrame())
    monday = build_monday_pack(pack, units, visits, exec_summary=_exec_row(data, period))
    beat = build_dsr_beat_pack(pack)
    with connect() as conn:
        outcomes = load_outcomes(conn, period)
    friday = build_friday_pack(outcomes, pack)

    view = st.radio(
        "View",
        ["Monday dispatch", "DSR beat lists", "Friday close", "Full this-week lists"],
        horizontal=True,
        index=0,
        key="this_week_view",
    )

    if view == "Monday dispatch":
        st.caption(
            "Summary: country, city drivers, city actions, DSRs, highlighted distributors, highlighted stores. "
            "The PDF/Excel download has every city and every store with Ask, with links from the summary."
        )
        for warning in monday.warnings:
            st.warning(warning)
        _render_ops_sheets(monday_summary_sheets(monday.sheets), height=300)
        with st.expander("Download Monday NSM pack"):
            _ops_downloads(monday, period, "monday", "monday")
    elif view == "DSR beat lists":
        st.caption(beat.headline)
        beat_df = beat.sheets[0][3] if beat.sheets else pd.DataFrame()
        owners = beat_owner_options(beat_df)
        pick = st.selectbox("DSR", ["All DSRs"] + owners, key="beat_dsr")
        shown = beat_df if pick == "All DSRs" else filter_beat_by_owner(beat_df, pick)
        st.caption(f"{len(shown)} doors on this list. Waiting-list doors stay in Full this-week lists.")
        _strategy_table(shown, height=480)
        with st.expander("Download DSR beat lists"):
            _ops_downloads(beat, period, "dsr_beat", "beat")
    elif view == "Friday close":
        st.caption(
            "This is whether last week’s list moved volume — not a new ranking. "
            "Score the warehouse twice in the same month to fill it."
        )
        st.markdown(f"**{friday.headline}**")
        _render_ops_sheets(friday.sheets, height=240)
        with st.expander("Download Friday close"):
            _ops_downloads(friday, period, "friday", "friday")
    else:
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download this week (Excel)",
                action_excel_bytes(pack),
                file_name=f"SND_this_week_{period}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                key="full_week_xlsx",
            )
        with right:
            st.download_button(
                "Download this week (PDF)",
                action_pdf_bytes(pack),
                file_name=f"SND_this_week_{period}.pdf",
                mime="application/pdf",
                key="full_week_pdf",
            )
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download detailed action pack (Excel)",
                action_excel_bytes(pack, detailed=True),
                file_name=f"SND_this_week_{period}_detailed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="full_week_detail_xlsx",
            )
        with d2:
            st.download_button(
                "Download detailed action pack (PDF)",
                action_pdf_bytes(pack, detailed=True),
                file_name=f"SND_this_week_{period}_detailed.pdf",
                mime="application/pdf",
                key="full_week_detail_pdf",
            )
        st.markdown("##### 1. Country this week")
        _strategy_table(pack.country, height=140)
        st.markdown("##### Pipeline (management)")
        st.caption("Billed + Due unvisited + Drop variance + Not yet due = Pipeline Expected.")
        _strategy_table(pack.pipeline, height=220)
        st.markdown("##### Sales head")
        st.caption("Due shops, visit compliance, unvisited Ask, drop variance, lapsed count.")
        _strategy_table(pack.sales_head, height=280)
        st.markdown("##### 2. Push these distributors")
        st.caption("Ranked by immediate Ask. Doors = due now. Not yet due is pipeline, not Ask.")
        _strategy_table(pack.distributors, height=320)
        st.markdown("##### 3. Push these DSRs")
        st.caption("One national list. A DSR can appear even if its distributor is not above.")
        _strategy_table(pack.dsrs, height=320)
        st.markdown("##### Beat — today")
        st.caption("Shop, area, last purchased, days overdue, target drop, recommended action.")
        _strategy_table(pack.beat, height=360)
        st.markdown("##### 4. Due — ratio ≥ 0.8, not visited")
        st.caption("Usually buys every N days; it has been N days with no bill. Nobody visited this month. Ask is the 90-day expected drop.")
        _strategy_table(pack.calls, height=420)
        st.markdown("##### 5. Due · visited — called, still no bill")
        _strategy_table(pack.converts, height=280)
        st.markdown("##### 6. Another visit — billed and already due again")
        st.caption("Billed this month and depletion ratio is already ≥ 0.8.")
        _strategy_table(pack.lifts, height=280)
        st.markdown("##### 7. Lost doors — lapsed (Ask 0)")
        st.caption("Days since last purchase > 3× API. Recovery drive, not the daily beat.")
        _strategy_table(pack.lost_doors if pack.lost_doors is not None and not pack.lost_doors.empty else pack.lapses, height=280)
        st.markdown("##### 8. Backtest")
        st.caption("At day 15 of closed months: did shops marked due actually bill in the next 14 days?")
        _strategy_table(pack.backtest, height=160)
        with st.expander("How to read this pack", expanded=False):
            for term, meaning in ACTION_GLOSSARY:
                st.markdown(f"**{term}.** {meaning}")

    _rescore_button()


def _load_action(data, period, ledger):
    brief = data.get("action_brief", pd.DataFrame())
    if brief is not None and not brief.empty:
        with connect() as conn:
            pack = load_action_pack(conn, period)
        if pack is not None and pack.headline:
            return pack
    shop_month = data.get("shop_month", pd.DataFrame())
    if shop_month is None or shop_month.empty:
        return None
    return build_action_pack(
        shop_month,
        data.get("stores"),
        shop_day=data.get("shop_day"),
        visits=data.get("visits"),
        ledger=ledger,
        period=period,
    )


def _page_report(data, _latest, period, mtd, ledger):
    st.title("Report")
    st.caption(
        f"**{mtd['label'] or period}** · Situation cascade is the pack you send. "
        "MTD is the in-month cut. Monthly closing is a finished month you pick."
    )
    units = data.get("units", pd.DataFrame())
    if units is None or units.empty:
        st.warning("No scorecards yet. Rebuild from the warehouse or upload files.")
        _rescore_button()
        return
    mode = st.radio(
        "Pack",
        ["Situation cascade", "Detailed scorecards"],
        horizontal=True,
        index=0,
        key="report_mode",
    )
    if mode == "Situation cascade":
        _page_situation_cascade(data, units, period, mtd, ledger)
    else:
        _page_detailed_scorecard(data, period, mtd, ledger)
    _rescore_button()


def _closed_periods(ledger, shop_month) -> list[str]:
    periods: list[str] = []
    if shop_month is not None and not shop_month.empty and "period" in shop_month.columns:
        periods = sorted({str(p) for p in shop_month["period"].dropna().astype(str)})
    elif ledger is not None and not ledger.empty and "period" in ledger.columns:
        periods = sorted({str(p) for p in ledger["period"].dropna().astype(str)})
    return [p for p in periods if not period_state(ledger, p).get("open")]


def _open_period(ledger, fallback: str) -> str | None:
    if ledger is not None and not ledger.empty and "status" in ledger.columns:
        open_rows = ledger[ledger["status"].astype(str) == "mtd_open"]
        if not open_rows.empty and "period" in open_rows.columns:
            return str(open_rows["period"].astype(str).max())
    if fallback and period_state(ledger, fallback).get("open"):
        return str(fallback)
    return None


def _units_for_situation(data, live_units, live_period: str, selected: str):
    selected = str(selected or "")
    if selected and selected == str(live_period or "") and live_units is not None and not live_units.empty:
        return live_units
    shop_month = data.get("shop_month", pd.DataFrame())
    fp = f"{selected}:{0 if shop_month is None or shop_month.empty else len(shop_month)}"
    cached = st.session_state.get("sit_units_cache")
    if isinstance(cached, dict) and cached.get("fp") == fp and cached.get("units") is not None:
        return cached["units"]
    with st.spinner(f"Building {selected} scorecards…"):
        units = scorecards_for_period(
            shop_month,
            data.get("stores"),
            data.get("ledger"),
            selected,
            shop_targets=data.get("shop_targets"),
            visits=data.get("visits"),
            shop_day=data.get("shop_day"),
            mtd_obs=data.get("mtd_obs"),
            cached_units=live_units,
        )
    st.session_state["sit_units_cache"] = {"fp": fp, "units": units}
    return units


def _page_situation_cascade(data, units, period, mtd, ledger):
    open_period = _open_period(ledger, period)
    closed = _closed_periods(ledger, data.get("shop_month"))
    cut_options = ["MTD (this month)", "Monthly closing"]
    default_cut = 0 if open_period else 1
    cut = st.radio("Report cut", cut_options, horizontal=True, index=default_cut, key="sit_cut")

    sit_period = period
    if cut == "MTD (this month)":
        if not open_period:
            st.info("No open month in the warehouse. Use Monthly closing to pick a finished month.")
            return
        sit_period = open_period
    else:
        if not closed:
            st.info("No closed month in the warehouse yet.")
            return
        labels = {p: format_period_label(p, open_=False) or p for p in closed}
        default_i = len(closed) - 1
        sit_period = st.selectbox(
            "Month",
            closed,
            index=default_i,
            format_func=lambda p: labels.get(p, p),
            key="sit_closed_month",
        )

    sit_units = _units_for_situation(data, units, period, sit_period)
    sit_mtd = period_state(ledger, sit_period)
    open_mtd = bool(sit_mtd.get("open"))
    action = _load_action(data, sit_period, ledger) if open_mtd else None

    c1, c2, c3 = st.columns(3)
    with c1:
        report_type = st.selectbox("1. Send to", ["National HQ", "City", "Distributor"], key="sit_report_type")
    kind = {"National HQ": "national", "City": "city", "Distributor": "distributor"}[report_type]
    entity = None
    with c2:
        if kind == "national":
            st.selectbox("2. Scope", ["Whole country"], disabled=True, key="sit_scope_national")
        else:
            options = list_situation_entities(sit_units, kind)
            q = st.text_input("Search", placeholder=f"Type to filter {kind}s", key="sit_search")
            filtered = [o for o in options if not q or q.lower() in o.lower()]
            if not options:
                st.selectbox(f"2. {report_type}", ["No options in the warehouse"], disabled=True, key="sit_entity_empty")
            elif not filtered:
                st.selectbox(f"2. {report_type}", [f"No match for “{q}”"], disabled=True, key="sit_entity_nomatch")
            else:
                entity = st.selectbox(f"2. {report_type}", filtered, index=0, key="sit_entity")
    with c3:
        fmt = st.selectbox("3. Format", ["PDF", "Excel"], key="sit_format")

    if kind != "national" and not entity:
        st.info("Choose a city or distributor — type in Search to narrow the list.")
        return

    city = dist = None
    if kind == "city":
        city = entity
    elif kind == "distributor" and entity:
        parts = [p.strip() for p in str(entity).split(" · ") if p.strip()]
        if len(parts) >= 2:
            city, dist = parts[0], " · ".join(parts[1:])
        else:
            dist = entity

    pack = build_situation_pack(
        sit_units, action=action, ledger=ledger, period=sit_period, scope=kind, city=city, distributor=dist
    )
    kpis = pack.kpis or {}
    st.markdown(f"**{pack.headline}**")
    st.caption(pack.weather)
    has_plan = float(kpis.get("target_mt") or 0) > 0.05
    if open_mtd:
        cols = st.columns(6 if has_plan else 4)
        cols[0].metric("Billed so far (MT)", f"{float(kpis.get('billed_mt') or 0):.0f}")
        cols[1].metric("Projected month-end (MT)", f"{float(kpis.get('projected_mt') or 0):.0f}")
        if has_plan:
            attain = kpis.get("attain_pct")
            cols[2].metric(
                "Monthly target (MT)",
                f"{float(kpis.get('target_mt') or 0):.0f}",
                delta=None if attain is None else f"{float(attain)*100:.0f}% of target",
            )
            cols[3].metric("vs Target (MT)", f"{float(kpis.get('vs_target_mt') or 0):.0f}")
            cols[4].metric("Situation", str(kpis.get("situation_label") or "—"))
            cols[5].metric("Lagging people", int(kpis.get("n_lagging_people") or 0))
        else:
            cols[2].metric("Situation", str(kpis.get("situation_label") or "—"))
            cols[3].metric("Lagging people", int(kpis.get("n_lagging_people") or 0))
        today = float(kpis.get("expected_today_mt") or 0)
        if today > 0.05:
            st.caption(f"Should have billed {today:.0f} MT by today (Expected × the national day curve).")
    else:
        cols = st.columns(6 if has_plan else 4)
        cols[0].metric("Billed (MT)", f"{float(kpis.get('billed_mt') or 0):.0f}")
        cols[1].metric("Expected (MT)", f"{float(kpis.get('expected_full_mt') or kpis.get('expected_mt') or 0):.0f}")
        cols[2].metric("Gap vs Expected (MT)", f"{float(kpis.get('gap_mt') or 0):.0f}")
        if has_plan:
            attain = kpis.get("attain_pct")
            cols[3].metric(
                "Monthly target (MT)",
                f"{float(kpis.get('target_mt') or 0):.0f}",
                delta=None if attain is None else f"{float(attain)*100:.0f}% of target",
            )
            cols[4].metric("vs Target (MT)", f"{float(kpis.get('vs_target_mt') or 0):.0f}")
            cols[5].metric("Situation", str(kpis.get("situation_label") or "—"))
        else:
            cols[3].metric("Situation", str(kpis.get("situation_label") or "—"))
    st.caption(_gap_sentence(kpis, pack.scope_label or "Country"))

    st.markdown("##### Current situation")
    for para in pack.situation:
        st.write(para)
    if pack.plan_lines:
        st.markdown("##### Plan")
        for line in pack.plan_lines:
            st.write("• " + line)
    if pack.copy_from:
        st.markdown("##### Copy from overperformers")
        for line in pack.copy_from:
            st.caption(line)

    st.markdown("##### Next actions" if open_mtd else "##### Results and next month")
    _strategy_table(pack.steps, height=220)

    if pack.gap_breakdown is not None and not pack.gap_breakdown.empty:
        st.markdown("##### City gap breakdown" if kind == "national" else "##### Distributor gap breakdown")
        _strategy_table(pack.gap_breakdown, height=320)

    if kind == "national":
        left, right = st.columns(2)
        with left:
            st.markdown("##### Distributors lagging")
            _strategy_table(pack.lagging_distributors, height=220)
        with right:
            st.markdown("##### Distributors ahead")
            _strategy_table(pack.ahead_distributors, height=220)
    elif kind == "city" and (pack.gap_breakdown is None or pack.gap_breakdown.empty):
        st.markdown("##### Distributors lagging")
        _strategy_table(pack.lagging_distributors, height=220)
        st.markdown("##### Distributors ahead")
        _strategy_table(pack.ahead_distributors, height=180)

    left, right = st.columns(2)
    with left:
        st.markdown("##### Underperforming sales staff")
        _strategy_table(pack.lagging_people, height=260)
    with right:
        st.markdown("##### Overperforming sales staff")
        _strategy_table(pack.ahead_people, height=260)

    if pack.this_week is not None and not pack.this_week.empty:
        st.markdown("##### This-week doors")
        _strategy_table(pack.this_week, height=200)

    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(pack.scope_label))[:60]
    if fmt == "Excel":
        payload = situation_excel_bytes(pack)
        name = f"SND_situation_{kind}_{safe}_{sit_period}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        payload = situation_pdf_bytes(pack)
        name = f"SND_situation_{kind}_{safe}_{sit_period}.pdf"
        mime = "application/pdf"
    st.download_button(
        f"Download {report_type.lower()} situation pack ({fmt})",
        payload,
        file_name=name,
        mime=mime,
        type="primary",
    )
    if open_mtd:
        st.caption("Situation PDF starts with billed, projected month-end, and the monthly target — not the glossary.")
    else:
        st.caption(
            "Closed-month PDF starts with billed, Expected, and Gap versus Expected "
            "(split into light orders / unbilled / unvisited) — not the glossary."
        )

    z1, z2 = st.columns(2)
    with z1:
        if st.button("Build ZIP of every city pack (PDF)", key="sit_zip_cities"):
            with st.spinner("Writing one PDF per city."):
                packs = build_field_packs(sit_units, action=action, ledger=ledger, period=sit_period, kind="city")
                st.session_state["sit_city_zip"] = zip_field_packs(packs, fmt="pdf")
                st.session_state["sit_city_zip_n"] = len(packs)
        if st.session_state.get("sit_city_zip"):
            n = int(st.session_state.get("sit_city_zip_n") or 0)
            st.download_button(
                f"Download {n} city packs",
                st.session_state["sit_city_zip"],
                file_name=f"SND_city_packs_{sit_period}.zip",
                mime="application/zip",
                key="sit_zip_cities_dl",
            )
    with z2:
        if st.button("Build ZIP of every distributor pack (PDF)", key="sit_zip_dists"):
            with st.spinner("Writing one PDF per distributor."):
                packs = build_field_packs(sit_units, action=action, ledger=ledger, period=sit_period, kind="distributor")
                st.session_state["sit_dist_zip"] = zip_field_packs(packs, fmt="pdf")
                st.session_state["sit_dist_zip_n"] = len(packs)
        if st.session_state.get("sit_dist_zip"):
            n = int(st.session_state.get("sit_dist_zip_n") or 0)
            st.download_button(
                f"Download {n} distributor packs",
                st.session_state["sit_dist_zip"],
                file_name=f"SND_distributor_packs_{sit_period}.zip",
                mime="application/zip",
                key="sit_zip_dists_dl",
            )


def _page_detailed_scorecard(data, period, mtd, ledger):
    st.caption(
        "Working file: every city / distributor / DSR / shop versus Expected. "
        "National PDF still starts with the glossary. Use Situation cascade to send a briefing."
    )
    pack = _strategy_pack(data, period, ledger)
    c1, c2, c3 = st.columns(3)
    with c1:
        report_type = st.selectbox("1. Report type", ["National", "City", "Distributor", "DSR"], key="report_type")
    entity = None
    with c2:
        if report_type == "National":
            st.selectbox("2. Scope", ["Whole country"], disabled=True, key="report_scope_national")
        else:
            options = list_report_entities(pack, report_type)
            q = st.text_input("Search", placeholder=f"Type to filter {report_type.lower()}s", key="report_search")
            filtered = [o for o in options if not q or q.lower() in o.lower()]
            if not options:
                st.selectbox(f"2. {report_type}", ["No options in the warehouse"], disabled=True, key="report_entity_empty")
            elif not filtered:
                st.selectbox(f"2. {report_type}", [f"No match for “{q}”"], disabled=True, key="report_entity_nomatch")
            else:
                entity = st.selectbox(f"2. {report_type}", filtered, index=0, key="report_entity")
    with c3:
        fmt = st.selectbox("3. Format", ["PDF", "Excel"], key="report_format")

    if report_type != "National" and not entity:
        st.info("Choose a city, distributor, or DSR — type in Search to narrow the list.")
        return

    focused = pack if report_type == "National" else focus_pack(pack, report_type, entity)
    kind = (focused.scope or "national").lower()
    if kind == "city":
        preview = focused.cities
        preview_label = "City scorecard"
    elif kind == "distributor":
        preview = focused.all_distributors
        preview_label = "Distributor scorecard"
    elif kind == "dsr":
        preview = focused.all_dsrs
        preview_label = "DSR scorecard"
    else:
        preview = focused.cities
        preview_label = "Country by city"

    st.markdown(f"**{focused.headline or focused.scope_label or 'National briefing'}**")
    st.caption(focused.weather or "")
    if kind == "national":
        if focused.exec_situation:
            st.markdown("##### Summary of current situation")
            for para in focused.exec_situation:
                st.write(para)
            st.markdown("##### Key focus areas")
            for item in focused.exec_focus:
                st.markdown(f"**{item.get('title') or ''}.** {item.get('why') or ''}")
                if item.get("do"):
                    st.caption(f"Do this week. {item['do']}")
            if focused.exec_model:
                st.caption(f"Model {focused.exec_model}. Figures match the tables in the download.")
        else:
            st.info(
                focused.exec_error
                or "No national executive summary stored. Paste an OpenAI key on Upload files, then generate or rebuild."
            )
            if st.button("Generate national executive summary"):
                from sndintel.narrative import refresh_exec_summary

                with connect() as conn:
                    with st.spinner("Writing the national executive summary from this period’s scorecards."):
                        meta = refresh_exec_summary(conn, pack)
                st.cache_data.clear()
                if meta.get("ok"):
                    st.success(f"Stored ({meta.get('model')}). Download again.")
                    st.rerun()
                else:
                    st.warning(meta.get("error") or "Summary was not generated.")
    st.markdown(f"##### Preview — {preview_label}")
    _strategy_table(preview, height=280)

    if kind == "city":
        st.markdown("##### Distributors in this city")
        _strategy_table(focused.all_distributors, height=240)
    elif kind == "distributor":
        st.markdown("##### Shops under this distributor")
        _strategy_table(focused.all_shops, height=280)
    elif kind == "dsr":
        st.markdown("##### Shops on this beat")
        _strategy_table(focused.all_shops, height=280)

    label = focused.scope_label or "national"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(label))[:60]
    if fmt == "Excel":
        payload = excel_bytes(focused)
        name = f"SND_{report_type.lower()}_{safe}_{period}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        payload = pdf_bytes(focused)
        name = f"SND_{report_type.lower()}_{safe}_{period}.pdf"
        mime = "application/pdf"
    st.download_button(
        f"Download {report_type.lower()} report ({fmt})",
        payload,
        file_name=name,
        mime=mime,
        type="primary",
    )
    st.caption("PDF is a real PDF (not HTML). Excel keeps one sheet per layer for filters.")


def _rescore_button():
    st.divider()
    st.caption(
        "Code updates do not wipe the warehouse. Rebuild scorecards from facts already on disk. "
        "If billed is double your extract, re-upload the sales file — duplicate lines used to be summed into one row."
    )
    if st.button("Rebuild scorecards from warehouse"):
        with st.spinner("Rebuilding city → distributor → DSR → shop scorecards from the warehouse."):
            try:
                result = rescore_warehouse()
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
                return
        st.cache_data.clear()
        st.success(
            f"Rebuilt {result.get('n_cities', 0)} cities · {result.get('n_targets', 0)} named targets · "
            f"latest {result.get('latest_period')}"
        )
        if result.get("exec_ok"):
            st.success(f"National executive summary written ({result.get('exec_model') or 'OpenAI'}).")
        elif result.get("exec_error"):
            st.warning(f"Scorecards rebuilt; executive summary skipped. {result.get('exec_error')}")
        st.rerun()


def _page_focus(data, period):
    st.title("Where to put people")
    units = data.get("units", pd.DataFrame())
    kpis = data["kpis"]
    grains = st.selectbox("Slice", ["city", "distributor", "dsr", "section", "zone"])
    if units is not None and not units.empty and grains in set(units["grain"].dropna()):
        slice_df = units[(units["grain"] == grains) & (units["period"] == period)].copy()
        if "isolated_mt" in slice_df.columns:
            slice_df = slice_df.sort_values("isolated_mt")
        else:
            slice_df = slice_df.sort_values("gap_mt")
        ycol = "isolated_mt" if "isolated_mt" in slice_df.columns else "gap_mt"
        color = "situation" if "situation" in slice_df.columns else "diagnosis"
        cmap = SITUATION_COLOR if color == "situation" else DIAGNOSIS_COLOR
        fig = px.bar(
            slice_df.head(20),
            x="grain_id",
            y=ycol,
            color=color,
            color_discrete_map=cmap,
            labels={"grain_id": grains, ycol: "vs Expected (MT)"},
        )
        fig.update_layout(height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        cols = [
            c
            for c in [
                "grain_id",
                "city",
                "volume_mt",
                "expected_mt",
                "ly_mt",
                "gap_mt",
                "lfl_gap",
                "lost_n",
                "lost_mt",
                "billed",
                "universe",
                "strike_rate",
                "diagnosis",
                "verdict",
                "do_this_week",
            ]
            if c in slice_df.columns
        ]
        st.dataframe(slice_df[cols].rename(columns={"grain_id": grains}), use_container_width=True, hide_index=True)
    else:
        slice_df = kpis[(kpis["grain"] == grains) & (kpis["period"] == period)].copy().sort_values("volume_mt", ascending=False)
        fig = px.bar(
            slice_df.head(20),
            x="grain_id",
            y="volume_mt",
            color="comparable_mom_pct" if "comparable_mom_pct" in slice_df.columns else "mom_pct",
            color_continuous_scale="RdYlGn",
            labels={"grain_id": grains, "volume_mt": "MT"},
        )
        fig.update_layout(height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        cols = [
            c
            for c in [
                "grain_id",
                "volume_mt",
                "billed_outlets",
                "universe_outlets",
                "strike_rate",
                "drop_size",
                "mom_pct",
                "comparable_mom_pct",
                "yoy_pct",
                "run_rate_yoy_pct",
            ]
            if c in slice_df.columns
        ]
        st.dataframe(slice_df[cols].rename(columns={"grain_id": grains}), use_container_width=True, hide_index=True)
    segs = data["segments"]
    if not segs.empty:
        st.subheader("Outlet segments")
        mix = segs["segment"].value_counts().rename_axis("segment").reset_index(name="shops")
        fig = px.pie(mix, names="segment", values="shops", hole=0.35)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def _page_people(data, period, mtd):
    st.title("People scorecards")
    st.caption(
        "A DSR is city + distributor + name. Two people with the same first name never share a row. "
        "Label is capacity vs skill: Overloaded / Not working the beat / Not converting / Not lifting drop / Fine."
    )
    units = data.get("units", pd.DataFrame())
    kpis = data["kpis"]
    as_of = int(mtd.get("as_of_day") or mtd.get("days_in_month") or 21)
    days_m = int(mtd.get("days_in_month") or 31)
    st.markdown("#### Salespeople")
    if units is not None and not units.empty and "dsr" in set(units["grain"].dropna()):
        from sndintel.capacity import present_capacity_table, score_dsr_capacity_from_units
        from sndintel.identity import dsr_display_name

        dsrs = units[(units["grain"] == "dsr") & (units["period"] == period)].copy()
        if "dsr_name" not in dsrs.columns or dsrs["dsr_name"].isna().all():
            dsrs["dsr_name"] = dsrs["grain_id"].map(dsr_display_name)
        cap = score_dsr_capacity_from_units(dsrs, as_of, days_m)
        if not cap.empty:
            st.markdown("##### Capacity labels")
            _strategy_table(present_capacity_table(cap, n=40), height=360)
        show = dsrs.sort_values("gap_mt")
        cols = [
            c
            for c in [
                "dsr_name",
                "city",
                "distributor",
                "volume_mt",
                "expected_mt",
                "ly_mt",
                "gap_mt",
                "visit_rate",
                "strike_rate",
                "diagnosis",
                "verdict",
                "do_this_week",
            ]
            if c in show.columns
        ]
        rename = {
            "dsr_name": "DSR",
            "city": "City",
            "distributor": "Distributor",
            "volume_mt": "Billed (MT)",
            "expected_mt": "Expected (MT)",
            "ly_mt": "Last year (MT)",
            "gap_mt": "Gap (MT)",
            "visit_rate": "Visit %",
            "strike_rate": "Strike %",
            "diagnosis": "Diagnosis",
            "verdict": "Verdict",
            "do_this_week": "Do this week",
        }
        table = show[cols].rename(columns=rename)
        if "Visit %" in table.columns:
            table["Visit %"] = (pd.to_numeric(table["Visit %"], errors="coerce") * 100).round(0)
        if "Strike %" in table.columns:
            table["Strike %"] = (pd.to_numeric(table["Strike %"], errors="coerce") * 100).round(0)
        st.markdown("##### Scorecards")
        _strategy_table(table, height=420)
    else:
        df = kpis[(kpis["grain"] == "dsr") & (kpis["period"] == period)].sort_values("volume_mt", ascending=False)
        cols = [c for c in ["grain_id", "volume_mt", "strike_rate", "drop_size", "billed_outlets", "universe_outlets"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Distributors")
    if units is not None and not units.empty and "distributor" in set(units["grain"].dropna()):
        df = units[(units["grain"] == "distributor") & (units["period"] == period)].sort_values("gap_mt")
        cols = [
            c
            for c in ["grain_id", "city", "volume_mt", "expected_mt", "ly_mt", "gap_mt", "diagnosis", "verdict", "do_this_week"]
            if c in df.columns
        ]
        st.dataframe(
            df[cols].rename(columns={"grain_id": "Distributor", "city": "City"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        df = kpis[(kpis["grain"] == "distributor") & (kpis["period"] == period)].sort_values("volume_mt", ascending=False)
        cols = [c for c in ["grain_id", "volume_mt", "strike_rate", "drop_size"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)


def _page_mix(data, period):
    st.title("SKU mix")
    sku = data["kpis"][(data["kpis"]["grain"] == "sku") & (data["kpis"]["period"] == period)].copy()
    if sku.empty:
        st.write("No SKU grain yet.")
        return
    fig = px.bar(
        sku.sort_values("volume_mt", ascending=False).head(30),
        x="grain_id",
        y="volume_mt",
        color="mom_pct",
        color_continuous_scale="RdYlGn",
        labels={"grain_id": "SKU", "volume_mt": "MT"},
    )
    fig.update_layout(height=360, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    cann = data["insights"][data["insights"]["type"].isin(["cannibalization", "sku_substitution", "sku_win"])]
    for rec in cann.itertuples(index=False):
        st.markdown(f"**{rec.title}**")
        st.write(rec.narrative)


def _page_shops(data, period):
    st.title("Shop explorer")
    stores = data["stores"]
    if stores.empty:
        st.write("No stores.")
        return
    options = (stores["store_id"].astype(str) + " · " + stores["store_name"].fillna("")).tolist()
    pick = st.selectbox("Store", options)
    sid = pick.split(" · ", 1)[0]
    plan = data.get("shop_targets", pd.DataFrame())
    if plan is not None and not plan.empty and "store_id" in plan.columns:
        hit = plan[plan["store_id"].astype(str) == sid]
        if not hit.empty:
            rec = hit.iloc[0]
            tgt = float(pd.to_numeric(pd.Series([rec.get("target_mt")]), errors="coerce").fillna(0).iloc[0])
            method = str(rec.get("match_method") or "")
            st.metric("Sales-team plan (MT)", f"{tgt:.2f}", help="Quota for this POP. Not Expected.")
            st.caption(f"Matched as `{method}`. Expected on the board pack stays the statistical run-rate.")
        else:
            unmatched = plan[plan["store_name"].fillna("").astype(str).str.lower() == pick.split(" · ", 1)[-1].strip().lower()]
            if not unmatched.empty:
                st.caption("A target row exists for this name but did not match this POP — check Warehouse → Shop plan.")
    hist = data["shop_month"][data["shop_month"]["store_id"] == sid].sort_values("period")
    fig = px.bar(hist, x="period", y="volume_mt", labels={"volume_mt": "MT"})
    fc = data["forecasts"]
    fc = fc[(fc["entity_type"] == "shop") & (fc["entity_id"] == sid)]
    if not fc.empty:
        fig.add_scatter(x=fc["period"], y=fc["predicted"], name="ML forecast (not official Expected)", mode="lines+markers")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Official Expected on the board pack is last-three-month run-rate, paced. "
        "The line above is a shop-month ML forecast and is not used for Gap or this-week Ask."
    )
    st.dataframe(hist, use_container_width=True, hide_index=True)


def _page_warehouse(data):
    st.title("Warehouse")
    st.markdown(
        f"- App version **{__version__}**. If this is still 0.9.3 (not 0.9.4), curl did not land the new ZIP.\n"
        f"- Code can be replaced any time. **Do not** keep `warehouse.db` inside the unzipped app folder.\n"
        f"- Data directory: `{DATA_DIR}`\n"
        f"- Database: `{DB_PATH}`"
    )
    with st.expander("Update the app on this Mac (curl, keep .venv)", expanded=True):
        from sndintel.update import mac_update_commands, zip_url

        st.markdown(
            "Same method as before: `curl` the branch ZIP, `rsync` over `~/sndintel`, "
            "keep `.venv`, `pip install -e .`. Warehouse stays in Application Support. "
            f"URL: `{zip_url()}`"
        )
        st.code(mac_update_commands().rstrip(), language="bash")
    ledger = data.get("ledger", pd.DataFrame())
    if not ledger.empty:
        st.subheader("Months on file")
        st.dataframe(ledger, use_container_width=True, hide_index=True)
        st.caption(
            f"{len(ledger)} months in the warehouse. Scorecards learn calendar-month seasonality from these totals. "
            "A day-of-month loading curve is learned only when mid-month MTD cuts exist for the same month."
        )
    season = data.get("seasonality", pd.DataFrame())
    if season is not None and not season.empty:
        st.subheader("Learned seasonal index")
        show = season[season["grain"].isin(["national", "city"])].copy()
        st.dataframe(show.sort_values(["grain", "grain_id", "month"]), use_container_width=True, hide_index=True)
    plan = data.get("shop_targets", pd.DataFrame())
    if plan is not None and not plan.empty:
        st.subheader("Shop plan (sales-team targets)")
        method = plan["match_method"].fillna("unmatched").astype(str) if "match_method" in plan.columns else pd.Series("unmatched", index=plan.index)
        matched = plan.loc[method.ne("unmatched") & plan.get("store_id", pd.Series("", index=plan.index)).astype(str).ne("")]
        book_mt = float(pd.to_numeric(plan.get("target_mt"), errors="coerce").fillna(0).sum())
        matched_mt = float(pd.to_numeric(matched.get("target_mt"), errors="coerce").fillna(0).sum()) if not matched.empty else 0.0
        coverage = (matched_mt / book_mt) if book_mt else 0.0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plan shops", f"{len(plan):,}")
        c2.metric("Matched shops", f"{len(matched):,}")
        c3.metric("Book (MT)", f"{book_mt:.0f}")
        c4.metric("Matched (MT)", f"{matched_mt:.0f}", delta=f"{coverage:.0%} of book")
        st.caption(
            "Target is the sales-team quota, not Expected. Country / city / DSR plan uses the full submitted book "
            "(unmatched names still roll if Area matches a live city). Shop identity is conservative — whales need an exact name."
        )
        if "match_method" in plan.columns:
            by = (
                plan.assign(_m=method, _mt=pd.to_numeric(plan["target_mt"], errors="coerce").fillna(0))
                .groupby("_m", dropna=False)
                .agg(shops=("_mt", "count"), mt=("_mt", "sum"))
                .reset_index()
                .rename(columns={"_m": "Match"})
            )
            st.dataframe(by, use_container_width=True, hide_index=True)
        leftover = plan.loc[method.eq("unmatched")].copy()
        if not leftover.empty:
            st.markdown("##### Unmatched sample")
            show = leftover.sort_values("target_mt", ascending=False).head(40)
            cols = [c for c in ("store_name", "city", "distributor", "dsr_name", "target_mt") if c in show.columns]
            st.dataframe(show[cols], use_container_width=True, hide_index=True)
            st.download_button(
                "Download unmatched plan rows (CSV)",
                leftover.to_csv(index=False).encode("utf-8"),
                file_name="shop_plan_unmatched.csv",
                mime="text/csv",
            )
    st.caption("Updating the app does not wipe this warehouse. Use Strategy → Rebuild scorecards if the briefing looks stale.")
    st.subheader("Replace billed sales")
    st.caption(
        "Clears billed rows and month scorecards. **Universe, legacy shops, and visit calls stay.** "
        "Then go to Upload files and drop the Outlet Date Wise parts in one go."
    )
    if st.button("Clear billed sales only", type="secondary"):
        try:
            cleared = clear_billed_sales()
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
        else:
            st.cache_data.clear()
            st.success(
                f"Billed sales cleared. Universe shops still in the warehouse: {cleared.get('n_stores')}. "
                "Upload the new daily files next."
            )
            st.rerun()
    sm = data.get("shop_month", pd.DataFrame())
    if sm is not None and not sm.empty:
        from sndintel.reconcile import distributor_shop_sales, match_distributors, period_totals

        st.subheader("Check billed vs your extract")
        st.caption(
            "Pick a distributor and a **closed** month (July). Sum the shops and compare to that "
            "distributor’s total on Outlet Date Wise (or Shop SKU Wise) for the same year and month. "
            "Country totals below should match the extract’s Grand Total for that month — not ~2×."
        )
        nat = period_totals(sm)
        if not nat.empty:
            st.dataframe(
                nat.rename(columns={"period": "Month", "shops": "Billed shops", "volume_mt": "Billed (MT)"}),
                use_container_width=True,
                hide_index=True,
            )
        names = match_distributors(sm)
        periods = sorted(sm["period"].astype(str).unique().tolist())
        if names and periods:
            c1, c2 = st.columns(2)
            with c1:
                dist = st.selectbox("Distributor", names, key="check_dist")
            with c2:
                default_i = periods.index("2026-07") if "2026-07" in periods else max(0, len(periods) - 2)
                per = st.selectbox("Month", periods, index=default_i, key="check_period")
            shops = distributor_shop_sales(sm, dist, per)
            total = float(shops["volume_mt"].sum()) if not shops.empty else 0.0
            st.markdown(
                f"**{dist}** · **{per}** · **{len(shops)} shops** · **{total:.2f} MT**. "
                "This is what the scorecards use. It should match your list."
            )
            if not shops.empty:
                show = shops.rename(
                    columns={
                        "store_name": "Shop",
                        "store_id": "POP",
                        "dsr_name": "DSR",
                        "section": "Section",
                        "sku_count": "SKUs",
                        "volume_mt": "Billed (MT)",
                    }
                )
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download this shop list (CSV)",
                    show.to_csv(index=False).encode("utf-8"),
                    file_name=f"shops_{per}_{dist[:40]}.csv",
                    mime="text/csv",
                )
    runs = data["runs"]
    if not runs.empty:
        st.subheader("Recent runs")
        st.dataframe(runs, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
