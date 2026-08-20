"""Command-line interface for the secondary sales intelligence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sndintel import __version__
from sndintel.config import SAMPLE_DIR, DATA_DIR, DB_PATH
from sndintel.ingest.pipeline import load_brief, load_kpis, load_ledger, rescore_warehouse, run_pipeline
from sndintel.mtd import banner_text, period_state
from sndintel.sampledata import generate_demo_files
from sndintel.storage import connect, init_db, read_sql
from sndintel.watch import scan_once, watch_forever

app = typer.Typer(help="FMCG store-wise secondary sales intelligence.")
console = Console()


@app.callback()
def _version(version: bool = typer.Option(False, "--version", help="Show version")):
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def ingest(
    sales: Path = typer.Argument(..., exists=True, help="SSRS Shop SKU Wise Execution Report (xlsx/csv)"),
    shops: Optional[Path] = typer.Option(None, "--shops", exists=True, help="Outlet universe / shop master"),
):
    """Clean a sales export, merge the shop list, train models, and write insights."""
    result = run_pipeline(sales, shop_path=shops)
    console.print_json(data=result)


@app.command()
def rescore():
    """Rebuild city → shop scorecards from the warehouse. Does not re-read a sales file."""
    result = rescore_warehouse()
    console.print_json(data=result)
    brief()


@app.command()
def demo(
    n_shops: int = typer.Option(140, help="Universe size for the synthetic company"),
    seed: int = 7,
):
    """Generate a realistic messy SSRS file, run the full pipeline, print the briefing."""
    paths = generate_demo_files(n_shops=n_shops, seed=seed)
    console.print(f"Wrote sample files to {SAMPLE_DIR}")
    result = run_pipeline(paths["sales"], shop_path=paths["shops"])
    console.print_json(data={k: v for k, v in result.items() if k != "warnings"})
    brief()


@app.command()
def brief(limit: int = typer.Option(20, help="How many ranked insights to show")):
    """Print the current executive briefing from the warehouse."""
    df = load_brief(limit=limit)
    if df.empty:
        console.print("No insights yet. Run [bold]snd-intel demo[/] or [bold]snd-intel ingest[/].")
        raise typer.Exit(1)
    kpis = load_kpis()
    ledger = load_ledger()
    national = kpis[kpis["grain"] == "national"].sort_values("period")
    if not national.empty:
        row = national.iloc[-1]
        state = period_state(ledger, row["period"])
        vol_label = f"{row['volume_mt']:.1f} MT"
        if state["open"] and pd.notna(row.get("run_rate_mt")):
            vol_label = f"{row['volume_mt']:.1f} MT MTD (run-rate {row['run_rate_mt']:.1f})"
        yoy_txt = row["yoy_pct"] if pd.notna(row["yoy_pct"]) else "n/a"
        if state["open"] and pd.notna(row.get("run_rate_yoy_pct")):
            yoy_txt = f"run-rate {row['run_rate_yoy_pct']:+.1f}%"
        console.print(banner_text(ledger, row["period"]))
        console.print(
            f"[bold]Period {row['period']}[/]  volume {vol_label}  "
            f"strike {row['strike_rate']*100:.0f}%  billed {int(row['billed_outlets'])}/{int(row['universe_outlets'])}  "
            f"MoM {row['comparable_mom_pct'] if pd.notna(row.get('comparable_mom_pct')) else (row['mom_pct'] if pd.notna(row['mom_pct']) else 'n/a')}  "
            f"YoY {yoy_txt}"
        )
    with connect() as conn:
        try:
            sit = read_sql(conn, "SELECT * FROM situation_brief ORDER BY period DESC LIMIT 1")
        except Exception:
            sit = pd.DataFrame()
        cities = read_sql(conn, "SELECT * FROM unit_scorecards WHERE grain = 'city' ORDER BY isolated_mt")
        shops = read_sql(conn, "SELECT * FROM focus_targets WHERE grain = 'shop' ORDER BY rank LIMIT 12")
    if not sit.empty:
        row_s = sit.iloc[0]
        console.print(f"\n[bold]{row_s['headline']}[/]")
        console.print(row_s["weather"])
        console.print(f"[red]{row_s['problem']}[/]")
        console.print(f"[green]{row_s['action_summary']}[/]\n")
    if not cities.empty:
        waterfall = Table(title="City exceptions vs fair share of national")
        waterfall.add_column("City", width=16)
        waterfall.add_column("Billed", justify="right")
        waterfall.add_column("Fair share", justify="right")
        waterfall.add_column("vs parent", justify="right")
        waterfall.add_column("Situation", width=14)
        waterfall.add_column("Driver", width=12)
        for rec in cities.head(12).itertuples(index=False):
            fair = getattr(rec, "share_expected_mt", rec.expected_mt)
            iso = getattr(rec, "isolated_mt", rec.gap_mt)
            sit_l = getattr(rec, "situation", rec.verdict)
            waterfall.add_row(
                str(rec.grain_id)[:16],
                f"{rec.volume_mt:.1f}",
                f"{fair:.1f}",
                f"{iso:+.1f}",
                str(sit_l),
                str(rec.diagnosis),
            )
        console.print(waterfall)
    if not shops.empty:
        console.print("\n[bold]Named must-visit shops[/]")
        for rec in shops.itertuples(index=False):
            console.print(
                f"• {rec.entity_name} ({rec.city}) {rec.volume_mt:.2f} vs {rec.ly_mt:.2f} LY "
                f"· {rec.dsr_name or '—'} · {rec.distributor or '—'}"
            )
    table = Table(title="Ranked insights", show_lines=False)
    table.add_column("Sev", width=8)
    table.add_column("Type", width=16)
    table.add_column("Where", width=22)
    table.add_column("Insight")
    for rec in df.itertuples(index=False):
        table.add_row(str(rec.severity), str(rec.type), str(rec.entity_name)[:22], str(rec.title))
    console.print(table)
    console.print("\n[bold]Top actions[/]")
    for rec in df.head(8).itertuples(index=False):
        console.print(f"• [bold]{rec.title}[/]\n  {rec.narrative}\n  → {rec.action}\n")


@app.command("watch")
def watch_cmd(
    once: bool = typer.Option(False, "--once", help="Process the drop folder once and exit"),
):
    """Watch data/incoming for new sales files and score them automatically."""
    if once:
        results = scan_once()
        console.print_json(data=results)
        return
    watch_forever()


@app.command()
def dashboard(
    port: int = typer.Option(8501, help="Streamlit port"),
):
    """Launch the interactive briefing app (same as `snd-intel app`)."""
    _launch_app(port)


@app.command("app")
def app_cmd(
    port: int = typer.Option(8501, help="Streamlit port"),
):
    """Open the local app: upload files and the strategy briefing."""
    _launch_app(port)


def _launch_app(port: int) -> None:
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    raise typer.Exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
                "--server.headless",
                "false",
                "--server.maxUploadSize",
                "500",
                "--browser.gatherUsageStats",
                "false",
            ]
        )
    )


@app.command("where")
def where_cmd():
    """Print where the warehouse is stored (survives app updates)."""
    console.print(f"data_dir  {DATA_DIR}")
    console.print(f"warehouse {DB_PATH}")


@app.command("serve-api")
def serve_api(port: int = 8080):
    """Serve the JSON API so a later ReAct agent can query insights deterministically."""
    import uvicorn

    uvicorn.run("sndintel.api:app", host="0.0.0.0", port=port, reload=False)


@app.command("export-excel")
def export_excel(path: Path = typer.Argument(Path("SND_intelligence_brief.xlsx"))):
    """Write a multi-sheet Excel briefing for sales managers."""
    init_db()
    with connect() as conn:
        insights = read_sql(conn, "SELECT * FROM insights ORDER BY rank_score DESC")
        kpis = read_sql(conn, "SELECT * FROM kpi_snapshots")
        anomalies = read_sql(conn, "SELECT * FROM anomalies")
        segments = read_sql(conn, "SELECT * FROM shop_segments")
        shops = read_sql(
            conn,
            """
            SELECT s.*, g.segment, m.volume_mt AS last_volume, m.period AS last_period
            FROM stores s
            LEFT JOIN shop_segments g ON g.store_id = s.store_id
            LEFT JOIN shop_month m ON m.store_id = s.store_id
            WHERE m.period = (SELECT MAX(period) FROM shop_month)
            """,
        )
        try:
            units = read_sql(conn, "SELECT * FROM unit_scorecards ORDER BY grain, gap_mt")
        except Exception:
            units = pd.DataFrame()
        try:
            targets = read_sql(conn, "SELECT * FROM focus_targets ORDER BY rank")
        except Exception:
            targets = pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        insights.to_excel(writer, sheet_name="Insights", index=False)
        kpis.to_excel(writer, sheet_name="KPIs", index=False)
        if not units.empty:
            units.to_excel(writer, sheet_name="City_to_DSR", index=False)
        if not targets.empty:
            targets.to_excel(writer, sheet_name="Named_targets", index=False)
        anomalies.to_excel(writer, sheet_name="Anomalies", index=False)
        segments.to_excel(writer, sheet_name="Segments", index=False)
        shops.to_excel(writer, sheet_name="Shop Scorecard", index=False)
    console.print(f"Wrote {path}")


@app.command()
def query(text: str = typer.Argument(..., help="Plain-language filter, e.g. 'trade loading Quetta'")):
    """Filter stored insights without an LLM — deterministic keyword search for a future ReAct agent."""
    init_db()
    tokens = [t.lower() for t in text.split() if len(t) > 2]
    with connect() as conn:
        df = read_sql(conn, "SELECT * FROM insights ORDER BY rank_score DESC")
    if df.empty:
        console.print("No insights in the warehouse yet.")
        raise typer.Exit(1)
    blob = (
        df["title"].fillna("")
        + " "
        + df["narrative"].fillna("")
        + " "
        + df["type"].fillna("")
        + " "
        + df["entity_name"].fillna("")
        + " "
        + df["entity_id"].fillna("")
    ).str.lower()
    mask = pd.Series(True, index=df.index)
    for tok in tokens:
        mask &= blob.str.contains(tok, regex=False)
    hits = df[mask] if tokens else df
    console.print_json(data=json.loads(hits.head(30).to_json(orient="records")))


if __name__ == "__main__":
    app()
