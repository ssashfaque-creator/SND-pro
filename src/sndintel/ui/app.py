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
    return data


def metric_or_dash(val, fmt="{:.1f}", suffix=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return fmt.format(val) + suffix


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; max-width: 1400px;}
        div[data-testid="stMetricValue"] {font-size: 1.35rem;}
        .play-card {border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.85rem; background: #fff;}
        .play-kicker {font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;}
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
    st.title("What to do this week")
    st.caption(
        f"**{mtd['label'] or period}** · city → distributor → DSR → shop. "
        "The engine reads the warehouse and names the units that move the number. "
        "It does not paste a canned plan."
    )
    if mtd["open"]:
        st.info(banner_text(ledger, period))
    _kpi_row(latest, mtd)

    units = data.get("units", pd.DataFrame())
    targets = data.get("targets", pd.DataFrame())
    if units is None or units.empty:
        st.warning(
            "No city scorecards yet. If sales are already in the warehouse, rebuild below. "
            "Otherwise upload the shop list and sales extract."
        )
        _rescore_button()
        return

    cities = units[units["grain"] == "city"].copy().sort_values("gap_mt")
    national = units[units["grain"] == "national"]
    nat = national.iloc[0] if not national.empty else None
    if nat is not None:
        st.markdown(
            f"**National billed {nat['volume_mt']:.1f} MT** vs **{nat['expected_mt']:.1f} expected** "
            f"(last year {nat['ly_mt']:.1f} MT) · hole **{nat['gap_mt']:+.1f} MT** · "
            f"{str(nat['diagnosis']).replace('_', ' ')} · {nat['verdict']}"
        )
        st.caption(nat["do_this_week"])

    left, right = st.columns((1.45, 1))
    with left:
        st.subheader("City waterfall (gap vs expected)")
        if cities.empty:
            st.write("No city rows.")
        else:
            chart = cities.head(16).copy()
            chart["city"] = chart["grain_id"]
            fig = px.bar(
                chart,
                x="city",
                y="gap_mt",
                color="diagnosis",
                color_discrete_map=DIAGNOSIS_COLOR,
                labels={"gap_mt": "Gap vs expected (MT)", "city": ""},
            )
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            show = cities[
                [
                    "grain_id",
                    "zone",
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
                ]
            ].rename(columns={"grain_id": "city"})
            st.dataframe(show, use_container_width=True, hide_index=True)
    with right:
        st.subheader("National trend")
        trend = data["shop_month"].groupby("period", as_index=False)["volume_mt"].sum().sort_values("period")
        fig = px.line(trend, x="period", y="volume_mt", markers=True, labels={"volume_mt": "MT", "period": ""})
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if targets is not None and not targets.empty:
            shops = targets[targets["grain"] == "shop"][
                ["rank", "city", "entity_name", "distributor", "dsr_name", "volume_mt", "ly_mt", "gap_mt", "action"]
            ].rename(columns={"entity_name": "shop"})
            st.subheader("Must-visit (named shops)")
            st.dataframe(shops.head(40), use_container_width=True, hide_index=True)
            st.download_button(
                "Download named targets CSV",
                targets.to_csv(index=False).encode("utf-8"),
                file_name=f"focus_targets_{period}.csv",
                mime="text/csv",
            )

    st.subheader("Open a city — named people and doors")
    for i, city in enumerate(cities.itertuples(index=False)):
        hole = float(city.gap_mt)
        if i >= 12 and hole > -1.0:
            continue
        color = DIAGNOSIS_COLOR.get(str(city.diagnosis), "#334155")
        label = (
            f"{city.grain_id}: billed {city.volume_mt:.1f} MT · should be {city.expected_mt:.1f} · "
            f"{hole:+.1f} MT · {str(city.diagnosis).replace('_', ' ')} · {city.verdict}"
        )
        with st.expander(label, expanded=i == 0):
            st.markdown(
                f"<span style='color:{color};font-weight:600;text-transform:uppercase;letter-spacing:0.06em'>"
                f"{str(city.diagnosis).replace('_', ' ')}</span>",
                unsafe_allow_html=True,
            )
            st.write(city.do_this_week)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Distributors**")
                dist = units[(units["grain"] == "distributor") & (units["parent_id"] == city.grain_id)].sort_values("gap_mt")
                if dist.empty:
                    st.caption("No distributor slice.")
                else:
                    st.dataframe(
                        dist[["grain_id", "volume_mt", "ly_mt", "gap_mt", "diagnosis", "verdict"]].rename(
                            columns={"grain_id": "distributor"}
                        ).head(12),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown("**Sections / areas**")
                sec = units[(units["grain"] == "section") & (units["parent_id"] == city.grain_id)].sort_values("gap_mt")
                if sec.empty:
                    st.caption("No section slice.")
                else:
                    st.dataframe(
                        sec[["grain_id", "volume_mt", "ly_mt", "gap_mt", "lost_n", "diagnosis"]].rename(
                            columns={"grain_id": "section"}
                        ).head(10),
                        use_container_width=True,
                        hide_index=True,
                    )
            with c2:
                st.markdown("**DSRs**")
                dsr = units[(units["grain"] == "dsr") & (units["parent_id"] == city.grain_id)].sort_values("gap_mt")
                if dsr.empty:
                    st.caption("No DSR slice.")
                else:
                    st.dataframe(
                        dsr[["grain_id", "volume_mt", "ly_mt", "gap_mt", "diagnosis", "verdict"]].rename(
                            columns={"grain_id": "dsr"}
                        ).head(12),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown("**Must-visit shops**")
                if targets is None or targets.empty:
                    st.caption("No named shops.")
                else:
                    hit = targets[(targets["city"] == city.grain_id) & (targets["grain"] == "shop")]
                    if hit.empty:
                        st.caption("No material shop gap in this city.")
                    else:
                        st.dataframe(
                            hit[["entity_name", "distributor", "dsr_name", "section", "volume_mt", "ly_mt", "gap_mt", "action"]].rename(
                                columns={"entity_name": "shop"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

    _rescore_button()

    with st.expander("Evidence — ranked flags behind the scorecards"):
        ins = data["insights"]
        if ins.empty:
            st.write("No insights stored.")
        else:
            show = ins[~ins["type"].isin(["portfolio_mix"])].head(20)
            for rec in show.itertuples(index=False):
                st.markdown(f"**{rec.title}** · _{rec.severity} · {rec.type}_")
                st.write(rec.narrative)
                st.caption(rec.action)


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
        slice_df = units[(units["grain"] == grains) & (units["period"] == period)].copy().sort_values("gap_mt")
        fig = px.bar(
            slice_df.head(20),
            x="grain_id",
            y="gap_mt",
            color="diagnosis",
            color_discrete_map=DIAGNOSIS_COLOR,
            labels={"grain_id": grains, "gap_mt": "Gap vs expected (MT)"},
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
    st.caption("Updating the app does not wipe this warehouse. Use Strategy → Rebuild scorecards if the briefing looks stale.")
    runs = data["runs"]
    if not runs.empty:
        st.subheader("Recent runs")
        st.dataframe(runs, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
