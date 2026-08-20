"""Local briefing app: upload files, then city → distributor → DSR → shop scorecards.

Warehouse lives in the user data directory so reinstalling the code does not
require re-uploading history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sndintel.briefing import GLOSSARY, build_strategy_pack, excel_bytes, html_bytes
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


def _remember_shop(path: Path) -> None:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    (MASTER_DIR / "last_shop.json").write_text(json.dumps({"path": str(path)}), encoding="utf-8")


def _last_shop() -> Path | None:
    marker = MASTER_DIR / "last_shop.json"
    if not marker.exists():
        return None
    try:
        path = Path(json.loads(marker.read_text(encoding="utf-8"))["path"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return path if path.exists() else None


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
    return data


def metric_or_dash(val, fmt="{:.1f}", suffix=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return fmt.format(val) + suffix


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.0rem; max-width: 1440px;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        .sit-card {border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: 0.9rem; color: #0f172a;}
        .sit-kicker {font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; margin-bottom: 0.35rem;}
        .sit-headline {font-size: 1.55rem; line-height: 1.25; font-weight: 700; margin: 0 0 0.55rem 0;}
        .sit-body {font-size: 1.02rem; line-height: 1.45; margin: 0 0 0.4rem 0;}
        .sit-action {font-size: 1.02rem; line-height: 1.45; margin: 0; font-weight: 600;}
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
        ["Strategy", "Upload files", "Focus", "People", "Mix", "Shops", "Warehouse"],
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
        "Shop list = outlet universe. Sales file = Shop SKU Wise SSRS extract (CSV or Excel). "
        "A new file replaces only the months it contains. July stays when you upload August."
    )
    if empty:
        st.warning("No warehouse yet. Upload the shop list once, then the sales extract (full history or current MTD).")
    else:
        st.success("Warehouse already has history. You can upload **sales only** for the latest month.")

    last = _last_shop()
    c1, c2 = st.columns(2)
    with c1:
        shops = st.file_uploader("Shop list (xlsx / csv)", type=["xlsx", "xls", "xlsm", "csv"], key="shops")
        if last and shops is None:
            st.caption(f"Using saved universe: `{last.name}`")
    with c2:
        sales = st.file_uploader("Sales extract (xlsx / csv)", type=["xlsx", "xls", "xlsm", "csv"], key="sales")

    st.markdown(
        "- First time: shop list **and** the full history extract (or the Drive sample).\n"
        "- Every later month: **sales only**. August MTD grows; closed months do not change.\n"
        "- Re-upload the shop list only when the universe changed (new DSR, new area)."
    )
    go = st.button("Score warehouse", type="primary", disabled=sales is None)
    if not go:
        return
    if sales is None:
        st.error("Sales file is required.")
        return
    shop_path = last
    if shops is not None:
        shop_path = _save_upload(shops, MASTER_DIR / shops.name)
        _remember_shop(shop_path)
    elif shop_path is None:
        st.error("No shop list on file yet. Upload it once.")
        return
    sales_path = _save_upload(sales, INCOMING_DIR / sales.name)
    with st.spinner("Parsing the extract and rebuilding city → distributor → DSR → shop scorecards. Large files take a few minutes."):
        try:
            result = run_pipeline(sales_path, shop_path=shop_path)
        except Exception as exc:  # noqa: BLE001 — show parse errors in the UI
            st.exception(exc)
            return
    st.cache_data.clear()
    st.success(
        f"Scored {result.get('n_sales_rows')} fact rows · latest {result.get('latest_period')} · "
        f"{result.get('n_cities', 0)} cities · {result.get('n_targets', 0)} named targets. "
        f"Replaced months: {', '.join(result.get('replaced_periods') or []) or '—'}"
    )
    if result.get("open_mtd_period"):
        st.info(f"Open MTD: {result['open_mtd_period']}")
    st.rerun()


def _kpi_row(latest, mtd):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Volume (MTD)" if mtd["open"] else "Volume", metric_or_dash(latest["volume_mt"], "{:.1f}", " MT"))
    mom_val = latest["comparable_mom_pct"] if mtd["open"] and "comparable_mom_pct" in latest.index else latest["mom_pct"]
    c2.metric("MoM (run-rate)" if mtd["open"] else "MoM", metric_or_dash(mom_val, "{:+.1f}", "%"))
    yoy_val = latest["run_rate_yoy_pct"] if mtd["open"] and "run_rate_yoy_pct" in latest.index else latest["yoy_pct"]
    c3.metric("YoY (run-rate)" if mtd["open"] else "YoY", metric_or_dash(yoy_val, "{:+.1f}", "%"))
    c4.metric("Strike rate", metric_or_dash(latest["strike_rate"] * 100, "{:.0f}", "%"))
    c5.metric("Drop size", metric_or_dash(latest["drop_size"], "{:.2f}", " MT"))


def _page_strategy(data, latest, period, mtd, ledger):
    st.title("Briefing")
    st.caption(
        f"**{mtd['label'] or period}** · expected is the typical same calendar month from every month "
        "in your warehouse — not last year alone, and not a loading curve we specified. "
        "Cities that moved with the country are not a local fire."
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

    cities = units[units["grain"] == "city"].copy()
    if "isolated_mt" in cities.columns:
        cities = cities.sort_values("isolated_mt")
    else:
        cities = cities.sort_values("gap_mt")
    national = units[units["grain"] == "national"]
    nat = national.iloc[0] if not national.empty else None
    sit_df = data.get("situation", pd.DataFrame())
    sit = sit_df.iloc[0] if sit_df is not None and not sit_df.empty else None

    headline = (sit["headline"] if sit is not None else None) or (nat.get("do_this_week") if nat is not None else None)
    weather = sit["weather"] if sit is not None else ""
    problem = sit["problem"] if sit is not None else ""
    action = sit["action_summary"] if sit is not None else ""
    extra = float(nat["isolated_mt"]) if nat is not None and "isolated_mt" in nat.index else 0.0
    n_hist = 0
    n_same = 0
    intra_src = ""
    if sit is not None:
        try:
            import json as _json

            meta = _json.loads(sit["metrics_json"]) if sit.get("metrics_json") else {}
            extra = float(meta.get("extra_hole_mt") or extra)
            n_hist = int(meta.get("n_history_periods") or 0)
            n_same = int(meta.get("n_same_month") or 0)
            intra_src = str(meta.get("intra_month_source") or "")
        except Exception:
            pass

    weather_dir = "declining"
    if nat is not None and float(nat.get("gap_mt") or 0) > 1:
        weather_dir = "growing"
    elif nat is not None and abs(float(nat.get("gap_mt") or 0)) <= 1:
        weather_dir = "flat"
    bar = {"declining": "#b91c1c", "growing": "#15803d", "flat": "#334155"}[weather_dir]
    st.markdown(
        f'<div class="sit-card" style="background:#f8fafc;border-left:8px solid {bar}">'
        f'<div class="sit-kicker">Situation</div>'
        f'<p class="sit-headline">{headline or "Scorecards ready"}</p>'
        f'<p class="sit-body">{weather}</p>'
        f'<p class="sit-body"><b>The problem.</b> {problem}</p>'
        f'<p class="sit-action">Do this week. {action}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if nat is not None:
        c1.metric("Billed", metric_or_dash(nat["volume_mt"], "{:.1f}", " MT"))
        hist_help = (
            f"Typical same calendar month from {n_hist} months in the warehouse"
            if n_hist
            else "Typical same calendar month from warehouse history"
        )
        if n_same:
            hist_help += f" ({n_same} prior {period[5:7] if period else 'same'} months, not last year alone)"
        c2.metric(
            "Seasonal expected",
            metric_or_dash(nat["expected_mt"], "{:.1f}", " MT"),
            help=hist_help,
        )
        c3.metric("Gap vs expected", metric_or_dash(nat["gap_mt"], "{:+.1f}", " MT"))
        extra_val = extra if extra else (
            float(cities.loc[cities["situation"] == "lagging", "isolated_mt"].sum())
            if "situation" in cities.columns
            else 0.0
        )
        c4.metric("Extra hole after weather", metric_or_dash(extra_val, "{:+.1f}", " MT"))
        n_lag = int((cities["situation"] == "lagging").sum()) if "situation" in cities.columns else 0
        c5.metric("Exception cities", str(n_lag))
    else:
        _kpi_row(latest, mtd)

    if n_hist:
        pace_note = ""
        if intra_src == "elapsed_days":
            pace_note = (
                " Open month is elapsed calendar days of that typical month — "
                "month-end totals cannot teach day-of-month loading."
            )
        elif intra_src == "learned_mtd_cuts":
            pace_note = " Open month is paced from mid-month MTD cuts already in the warehouse."
        st.caption(
            f"Seasonality fitted on **{n_hist} months** already in the warehouse"
            + (f", including **{n_same} prior same calendar months**." if n_same else ".")
            + pace_note
        )

    pack = build_strategy_pack(
        units,
        data.get("shop_month", pd.DataFrame()),
        situation=sit_df,
        ledger=ledger,
        period=period,
    )
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
            "Download printable briefing (HTML → Print to PDF)",
            html_bytes(pack),
            file_name=f"SND_strategy_{period}.html",
            mime="text/html",
        )
    st.caption(
        "Excel is the working file (filters, one sheet per layer). "
        "HTML opens in a browser — File → Print → Save as PDF for a board pack."
    )

    st.markdown("##### 1. The country — every city")
    st.caption(
        "Start here. **Extra vs country** is the local problem after national weather. "
        "**Recoverable** is that hole as a positive number. **AMS** is the average of the last three closed months."
    )
    left, right = st.columns((1.4, 1))
    with left:
        if cities.empty:
            st.write("No city rows.")
        else:
            chart = cities.head(16).copy()
            chart["city"] = chart["grain_id"]
            ycol = "isolated_mt" if "isolated_mt" in chart.columns else "gap_mt"
            color = "situation" if "situation" in chart.columns else "diagnosis"
            cmap = SITUATION_COLOR if color == "situation" else DIAGNOSIS_COLOR
            fig = px.bar(
                chart,
                x="city",
                y=ycol,
                color=color,
                color_discrete_map=cmap,
                labels={ycol: "Extra vs country (MT)", "city": ""},
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
            fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
            st.plotly_chart(fig, use_container_width=True)
        _strategy_table(pack.cities)
    with right:
        st.markdown("**Why the extra hole**")
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

    st.markdown("##### 2. Lagging cities — distributors")
    st.caption(
        "Only cities on the lagging list, broken by distributor. "
        "These are the first calls — not every distributor in the country."
    )
    if pack.city_distributors.empty:
        st.info("No lagging city, or no distributor inside those cities is behind the city.")
    else:
        _strategy_table(pack.city_distributors)

    st.markdown("##### 3. Those distributors — lagging shops")
    st.caption(
        (pack.shop_note or "Visit-worthy doors only.")
        + " **Recoverable** is the volume you get back if the door merely matches the city."
    )
    if pack.city_distributor_shops.empty:
        st.info("No material lagging shops under those distributors.")
    else:
        _strategy_table(pack.city_distributor_shops, height=360)

    st.markdown("##### 4. Every lagging distributor (all cities)")
    st.caption(
        "Distributors behind their own city even when the city moved with the country. "
        "Section 2 only showed distributors in lagging cities."
    )
    _strategy_table(pack.lagging_distributors)

    st.markdown("##### 5. Every lagging DSR (all cities)")
    st.caption("Salespeople behind their city. Ride-with this list.")
    _strategy_table(pack.lagging_dsrs.head(60))

    st.markdown("##### 6. Every lagging shop worth a visit")
    st.caption(pack.shop_note or "Visit-worthy doors behind their city. Tiny kiryana is a coverage KPI, not this list.")
    _strategy_table(pack.lagging_shops, height=420)

    with st.expander("How to read the columns", expanded=False):
        for term, meaning in GLOSSARY:
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
        if "(MT)" in str(col):
            cfg[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif str(col).endswith("%") or str(col) == "Strike %":
            cfg[col] = st.column_config.NumberColumn(col, format="%.0f")
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(height, 80 + 28 * max(3, len(df))), column_config=cfg)


def _rescore_button():
    st.divider()
    st.caption("Code updates do not wipe the warehouse. Rebuild scorecards from facts already on disk — no re-upload.")
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
            labels={"grain_id": grains, ycol: "vs fair share (MT)"},
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
    runs = data["runs"]
    if not runs.empty:
        st.subheader("Recent runs")
        st.dataframe(runs, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
