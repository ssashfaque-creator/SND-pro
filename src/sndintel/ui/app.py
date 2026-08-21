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
from sndintel.config import DATA_DIR, DB_PATH, INCOMING_DIR, MASTER_DIR, ensure_dirs
from sndintel.ingest.pipeline import rescore_warehouse, run_pipeline
from sndintel.mtd import banner_text, period_state
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
        ["Strategy", "Report", "Upload files", "Focus", "People", "Mix", "Shops", "Warehouse"],
        index=1 if empty else 0,
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
    elif page == "Report":
        _page_report(data, latest, period, mtd, ledger)
    elif page == "Focus":
        _page_focus(data, period)
    elif page == "People":
        _page_people(data, period)
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
        st.warning("No warehouse yet. Upload the universe list and a sales extract. Visit calls can come with the sales file.")
    else:
        st.success("Warehouse already has history. Typical week: **sales + visit calls**. Re-upload universe when the active book changes.")

    last_uni = _last("universe")
    last_shop = _last_shop()
    last_vis = _last("visits")
    c1, c2 = st.columns(2)
    with c1:
        universe = st.file_uploader(
            "Universe shop list (xlsx) — live distributors, DSRs, shops",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="universe",
        )
        if last_uni and universe is None:
            st.caption(f"Using saved universe: `{last_uni.name}`")
        shops = st.file_uploader(
            "Legacy shop list (optional — zone / historical map)",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="shops",
        )
        if last_shop and shops is None:
            st.caption(f"Saved legacy master: `{last_shop.name}`")
    with c2:
        sales = st.file_uploader("Sales extract (xlsx / csv)", type=["xlsx", "xls", "xlsm", "csv"], key="sales")
        visits = st.file_uploader(
            "Shop visit calls (csv / xlsx) — MTD visits, weekly with sales",
            type=["xlsx", "xls", "xlsm", "csv"],
            key="visits",
        )
        if last_vis and visits is None:
            st.caption(f"Last visit file: `{last_vis.name}` (re-upload each week)")

    st.markdown(
        "- **Universe** columns: distributor, area/city, section, DSR, POP code, POP name. Merged cells are filled down.\n"
        "- First time: universe **and** sales. Optional legacy shop list fills **zone**.\n"
        "- Every later week: **sales + visit calls**. Closed months stay. Re-upload universe when shops/DSRs move."
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

    go = st.button("Score warehouse", type="primary", disabled=sales is None and empty)
    if not go:
        return
    universe_path = last_uni
    if universe is not None:
        universe_path = _save_upload(universe, MASTER_DIR / universe.name)
        _remember("universe", universe_path)
    shop_path = last_shop
    if shops is not None:
        shop_path = _save_upload(shops, MASTER_DIR / shops.name)
        _remember_shop(shop_path)
    if empty and universe_path is None and shop_path is None:
        st.error("Upload the universe shop list (or a legacy shop list) once.")
        return
    sales_path = _save_upload(sales, INCOMING_DIR / sales.name) if sales is not None else None
    visits_path = None
    if visits is not None:
        visits_path = _save_upload(visits, INCOMING_DIR / visits.name)
        _remember("visits", visits_path)
    with st.spinner("Parsing files and rebuilding city → distributor → DSR → shop scorecards. Large files take a few minutes."):
        try:
            result = run_pipeline(
                sales_path,
                shop_path=shop_path,
                universe_path=universe_path,
                visits_path=visits_path,
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return
    st.cache_data.clear()
    st.success(
        f"Scored {result.get('n_sales_rows')} fact rows · latest {result.get('latest_period')} · "
        f"{result.get('n_cities', 0)} cities · universe {result.get('n_universe') or '—'} · "
        f"visits {result.get('n_visits') or '—'}. "
        f"Replaced months: {', '.join(result.get('replaced_periods') or []) or '—'}"
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
        f"**{mtd['label'] or period}** · AMS is DSS L3M (this month + two before). "
        "Expected is the last three *closed* months, paced if MTD is open."
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
        "National summary names the top 10 lagging distributors by Gap, the top 10 lagging DSRs "
        "from those distributors, and the top 50 lagging shops (after the 0.25 MT floor). "
        "Detailed pack = every city, every distributor and DSR with AMS > 0, and every shop with gap > 0.25 MT."
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

    st.markdown("##### 2. Top 10 lagging distributors")
    st.caption(
        "Biggest Gap first, nationally. AMS = 0 is hidden. Remainder line is everyone after the tenth."
    )
    if pack.city_distributors.empty:
        st.info("No lagging distributor with a recent run-rate.")
    else:
        _strategy_table(pack.city_distributors)

    st.markdown("##### 3. Those distributors — top 50 shops")
    st.caption(
        (pack.shop_note or "Top 50 shops by Gap under those distributors.")
        + " **Gap** is the volume you get back if the door billed its own Expected."
    )
    if pack.city_distributor_shops.empty:
        st.info("No material lagging shops under those distributors.")
    else:
        _strategy_table(pack.city_distributor_shops, height=360)

    st.markdown("##### 4. Top 10 lagging distributors (all cities)")
    st.caption(
        "Same ten names as section 2 — national rank by Gap, including pockets inside cities that are on expected."
    )
    _strategy_table(pack.lagging_distributors)

    st.markdown("##### 5. Top 10 lagging DSRs from those distributors")
    st.caption("DSRs whose shops sit under the top 10 distributors, then the ten biggest Gaps. Ride-with this list.")
    _strategy_table(pack.lagging_dsrs)

    st.markdown("##### 6. Top 50 lagging shops")
    st.caption(pack.shop_note or "Top 50 doors by Gap after the 0.25 MT floor. The rest of the hole is the remainder line.")
    _strategy_table(pack.lagging_shops, height=420)

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
        if name == "Remarks":
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
    row_h = 56 if "Remarks" in df.columns else 28
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(max(height, 160), 80 + row_h * min(n, 12)),
        column_config=cfg,
    )


def _page_report(data, _latest, period, mtd, ledger):
    st.title("Report")
    st.caption(
        f"**{mtd['label'] or period}** · Pick a report, then a city / distributor / DSR if needed, then PDF or Excel. "
        "National PDF starts with the glossary, then the AI executive summary, then the tables. "
        "Figures are whole numbers. Remarks are bullets in the last column."
    )
    units = data.get("units", pd.DataFrame())
    if units is None or units.empty:
        st.warning("No scorecards yet. Rebuild from the warehouse or upload files.")
        _rescore_button()
        return
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
    _rescore_button()


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


def _page_people(data, period):
    st.title("People scorecards")
    units = data.get("units", pd.DataFrame())
    kpis = data["kpis"]
    for grain, label, key in [("dsr", "Salespeople", "dsr"), ("distributor", "Distributors", "distributor")]:
        st.markdown(f"#### {label}")
        if units is not None and not units.empty and grain in set(units["grain"].dropna()):
            df = units[(units["grain"] == grain) & (units["period"] == period)].sort_values("gap_mt")
            cols = [
                c
                for c in ["grain_id", "city", "volume_mt", "expected_mt", "ly_mt", "gap_mt", "diagnosis", "verdict", "do_this_week"]
                if c in df.columns
            ]
            st.dataframe(df[cols].rename(columns={"grain_id": key}), use_container_width=True, hide_index=True)
        else:
            df = kpis[(kpis["grain"] == grain) & (kpis["period"] == period)].sort_values("volume_mt", ascending=False)
            cols = [
                c
                for c in [
                    "grain_id",
                    "volume_mt",
                    "mom_pct",
                    "comparable_mom_pct",
                    "yoy_pct",
                    "run_rate_yoy_pct",
                    "strike_rate",
                    "drop_size",
                    "sku_depth",
                    "billed_outlets",
                    "universe_outlets",
                ]
                if c in df.columns
            ]
            st.dataframe(df[cols].rename(columns={"grain_id": label[:-1]}), use_container_width=True, hide_index=True)


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
    hist = data["shop_month"][data["shop_month"]["store_id"] == sid].sort_values("period")
    fig = px.bar(hist, x="period", y="volume_mt", labels={"volume_mt": "MT"})
    fc = data["forecasts"]
    fc = fc[(fc["entity_type"] == "shop") & (fc["entity_id"] == sid)]
    if not fc.empty:
        fig.add_scatter(x=fc["period"], y=fc["predicted"], name="expected", mode="lines+markers")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hist, use_container_width=True, hide_index=True)


def _page_warehouse(data):
    st.title("Warehouse")
    st.markdown(
        f"- Code can be replaced any time. **Do not** keep `warehouse.db` inside the unzipped app folder.\n"
        f"- Data directory: `{DATA_DIR}`\n"
        f"- Database: `{DB_PATH}`"
    )
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
    st.caption("Updating the app does not wipe this warehouse. Use Strategy → Rebuild scorecards if the briefing looks stale.")
    sm = data.get("shop_month", pd.DataFrame())
    if sm is not None and not sm.empty:
        from sndintel.reconcile import distributor_shop_sales, match_distributors, period_totals

        st.subheader("Check billed vs your extract")
        st.caption(
            "Pick a distributor and a **closed** month (July). Sum the shops and compare to that "
            "distributor’s total on Shop SKU Wise for the same year and month. "
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
