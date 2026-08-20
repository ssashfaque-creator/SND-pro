"""Streamlit briefing app — run via `snd-intel dashboard`."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from sndintel.mtd import banner_text, period_state
from sndintel.storage import connect, init_db, read_sql

st.set_page_config(page_title="SND Intelligence", layout="wide", page_icon="▣")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30)
def load_all():
    init_db()
    with connect() as conn:
        return {
            "insights": read_sql(conn, "SELECT * FROM insights ORDER BY rank_score DESC"),
            "kpis": read_sql(conn, "SELECT * FROM kpi_snapshots"),
            "shop_month": read_sql(conn, "SELECT * FROM shop_month"),
            "anomalies": read_sql(conn, "SELECT * FROM anomalies"),
            "segments": read_sql(conn, "SELECT * FROM shop_segments"),
            "stores": read_sql(conn, "SELECT * FROM stores"),
            "forecasts": read_sql(conn, "SELECT * FROM forecasts"),
            "runs": read_sql(conn, "SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT 5"),
            "ledger": read_sql(conn, "SELECT * FROM period_ledger ORDER BY period"),
        }


def metric_or_dash(val, fmt="{:.1f}", suffix=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    return fmt.format(val) + suffix


data = load_all()
kpis = data["kpis"]
insights = data["insights"]
shop_month = data["shop_month"]

if kpis.empty or shop_month.empty:
    st.title("SND Intelligence")
    st.info("No warehouse yet. In a terminal run `snd-intel demo` or `snd-intel ingest REPORT.xlsx --shops shop_master.xlsx`.")
    st.stop()

national = kpis[kpis["grain"] == "national"].sort_values("period")
latest = national.iloc[-1]
period = latest["period"]
ledger = data.get("ledger", pd.DataFrame())
mtd = period_state(ledger, period)

st.title("SND Intelligence")
st.caption(
    f"Secondary sales briefing for **{mtd['label'] or period}** · volumes in metric tons · "
    "shop × SKU execution data scored against each outlet's own history. "
    "Insights cover the **full warehouse**, not only the latest file."
)
if mtd["open"]:
    st.info(banner_text(ledger, period))
elif not ledger.empty:
    st.caption(banner_text(ledger, period))

c1, c2, c3, c4, c5 = st.columns(5)
vol_help = "Billed MTD in this extract" if mtd["open"] else "Closed-month volume"
c1.metric(
    "Volume (MTD)" if mtd["open"] else "Volume",
    metric_or_dash(latest["volume_mt"], "{:.1f}", " MT"),
    help=vol_help,
)
mom_val = latest["comparable_mom_pct"] if mtd["open"] and "comparable_mom_pct" in latest.index else latest["mom_pct"]
c2.metric(
    "MoM (run-rate)" if mtd["open"] else "MoM",
    metric_or_dash(mom_val, "{:+.1f}", "%"),
)
yoy_val = latest["run_rate_yoy_pct"] if mtd["open"] and "run_rate_yoy_pct" in latest.index else latest["yoy_pct"]
c3.metric(
    "YoY (run-rate)" if mtd["open"] else "YoY",
    metric_or_dash(yoy_val, "{:+.1f}", "%"),
    help="Run-rate vs last year's closed month when the latest period is still open MTD",
)
c4.metric(
    "Strike rate",
    metric_or_dash(latest["strike_rate"] * 100, "{:.0f}", "%"),
    help="Billed shops / universe shops on the master list",
)
c5.metric("Drop size", metric_or_dash(latest["drop_size"], "{:.2f}", " MT"))

tabs = st.tabs(
    [
        "Briefing",
        "Focus map",
        "Anomalies",
        "Coverage",
        "SKU mix",
        "People",
        "Shop explorer",
    ]
)

with tabs[0]:
    left, right = st.columns((1.4, 1))
    with left:
        st.subheader("What needs attention")
        issues = insights[insights["severity"].isin(["critical", "high"])]
        if issues.empty:
            st.success("No critical flags this period.")
        for rec in issues.head(12).itertuples(index=False):
            st.markdown(f"**{rec.title}**")
            st.write(rec.narrative)
            st.caption(f"{rec.severity} · {rec.type} · {rec.action}")
            st.divider()
    with right:
        st.subheader("What is working")
        wins = insights[insights["severity"] == "positive"]
        if wins.empty:
            st.write("No outperformance flags this period.")
        for rec in wins.head(8).itertuples(index=False):
            st.markdown(f"**{rec.title}**")
            st.write(rec.narrative)
        st.subheader("National trend")
        trend = (
            shop_month.groupby("period", as_index=False)["volume_mt"]
            .sum()
            .sort_values("period")
        )
        fig = px.line(trend, x="period", y="volume_mt", markers=True, labels={"volume_mt": "MT", "period": ""})
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Where to put people this week")
    grains = st.selectbox("Slice", ["city", "section", "dsr", "distributor", "zone"])
    slice_df = kpis[(kpis["grain"] == grains) & (kpis["period"] == period)].copy()
    slice_df = slice_df.sort_values("volume_mt", ascending=False)
    fig = px.bar(
        slice_df.head(20),
        x="grain_id",
        y="volume_mt",
        color="mom_pct",
        color_continuous_scale="RdYlGn",
        labels={"grain_id": grains, "volume_mt": "MT", "mom_pct": "MoM %"},
    )
    fig.update_layout(height=360, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        slice_df[
            ["grain_id", "volume_mt", "billed_outlets", "universe_outlets", "strike_rate", "drop_size", "mom_pct", "yoy_pct"]
        ].rename(columns={"grain_id": grains}),
        use_container_width=True,
        hide_index=True,
    )
    segs = data["segments"]
    if not segs.empty:
        st.subheader("Outlet segments")
        mix = segs["segment"].value_counts().rename_axis("segment").reset_index(name="shops")
        fig = px.pie(mix, names="segment", values="shops", hole=0.35)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(segs.sort_values("monetary", ascending=False), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Trade loading, drop-offs, and statistical outliers")
    an = data["anomalies"]
    if an.empty:
        st.write("No anomalies stored.")
    else:
        kind = st.multiselect("Kind", sorted(an["kind"].dropna().unique()), default=list(an["kind"].unique()))
        view = an[an["kind"].isin(kind)] if kind else an
        merged = view.merge(data["stores"][["store_id", "store_name", "dsr_name", "city", "section"]], on="store_id", how="left")
        fig = px.scatter(
            merged,
            x="expected_mt",
            y="volume_mt",
            color="kind",
            hover_data=["store_name", "dsr_name", "section", "severity"],
            labels={"expected_mt": "Typical drop (MT)", "volume_mt": "This month (MT)"},
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(merged, use_container_width=True, hide_index=True)
    flagged = insights[insights["type"].isin(["trade_loading", "drop_off", "lumpy", "anomaly"])]
    st.subheader("Narratives")
    for rec in flagged.head(15).itertuples(index=False):
        st.markdown(f"**{rec.title}** — {rec.narrative}")

with tabs[3]:
    st.subheader("Strike rate: billed shops vs universe")
    cov = kpis[(kpis["grain"] == "dsr") & (kpis["period"] == period)].copy()
    cov["unbilled"] = cov["universe_outlets"] - cov["billed_outlets"]
    fig = px.bar(
        cov.sort_values("strike_rate"),
        x="strike_rate",
        y="grain_id",
        orientation="h",
        color="volume_mt",
        labels={"grain_id": "DSR", "strike_rate": "Strike rate", "volume_mt": "MT"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    city = kpis[(kpis["grain"] == "city") & (kpis["period"] == period)]
    st.dataframe(
        city[["grain_id", "volume_mt", "billed_outlets", "universe_outlets", "strike_rate", "drop_size", "sku_depth", "mom_pct"]],
        use_container_width=True,
        hide_index=True,
    )
    ws = insights[insights["type"] == "whitespace"]
    for rec in ws.itertuples(index=False):
        st.warning(rec.narrative)

with tabs[4]:
    st.subheader("SKU contribution and mix shift")
    sku = kpis[(kpis["grain"] == "sku") & (kpis["period"] == period)].copy()
    fig = px.bar(
        sku.sort_values("volume_mt", ascending=False),
        x="grain_id",
        y="volume_mt",
        color="mom_pct",
        color_continuous_scale="RdYlGn",
        labels={"grain_id": "SKU", "volume_mt": "MT"},
    )
    fig.update_layout(height=360, xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    cann = insights[insights["type"].isin(["cannibalization", "sku_substitution", "sku_win"])]
    for rec in cann.itertuples(index=False):
        st.markdown(f"**{rec.title}**")
        st.write(rec.narrative)

with tabs[5]:
    st.subheader("DSR and distributor scorecards")
    for grain, label in [("dsr", "Salespeople"), ("distributor", "Distributors")]:
        st.markdown(f"#### {label}")
        df = kpis[(kpis["grain"] == grain) & (kpis["period"] == period)].sort_values("volume_mt", ascending=False)
        st.dataframe(
            df[
                ["grain_id", "volume_mt", "mom_pct", "yoy_pct", "strike_rate", "drop_size", "sku_depth", "billed_outlets", "universe_outlets"]
            ].rename(columns={"grain_id": label[:-1]}),
            use_container_width=True,
            hide_index=True,
        )

with tabs[6]:
    st.subheader("Shop deep dive")
    stores = data["stores"]
    options = (stores["store_id"] + " · " + stores["store_name"].fillna("")).tolist()
    pick = st.selectbox("Store", options)
    sid = pick.split(" · ", 1)[0]
    hist = shop_month[shop_month["store_id"] == sid].sort_values("period")
    fig = px.bar(hist, x="period", y="volume_mt", labels={"volume_mt": "MT"})
    fc = data["forecasts"]
    fc = fc[(fc["entity_type"] == "shop") & (fc["entity_id"] == sid)]
    if not fc.empty:
        fig.add_scatter(x=fc["period"], y=fc["predicted"], name="expected", mode="lines+markers")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    shop_ins = insights[insights["entity_id"] == sid]
    for rec in shop_ins.itertuples(index=False):
        st.markdown(f"**{rec.title}** — {rec.narrative}")

st.sidebar.markdown("### Pipeline")
runs = data["runs"]
if not runs.empty:
    st.sidebar.write(f"Last run `{runs.iloc[0]['status']}` · {runs.iloc[0]['finished_at']}")
    st.sidebar.caption(str(runs.iloc[0]["sales_file"]))
st.sidebar.markdown("Drop a new SSRS file in `data/incoming` and run `snd-intel watch --once`.")
