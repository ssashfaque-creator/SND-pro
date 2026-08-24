# SND Intelligence

Store-wise secondary sales intelligence for FMCG (edible oil / cooking fat) teams.

The system reads the messy **Shop SKU Wise Execution Report** your SSRS stack exports, joins it to the outlet universe, and every time a new file lands it rebuilds a baseline of *what each shop, beat, and SKU should have done*. It then ranks the few things a sales head should actually act on: where volume is real, where a DSR dumped stock, which sections are failing while the city grows, which packs are eating each other, and which universe shops you have never billed.

## What your files actually contain

### Sales extract (Outlet Date Wise or Shop SKU Wise)

**Outlet Date Wise Sale** (daily tons by POP code) is the preferred billed file. Date columns are summed to calendar months. There is no distributor on that sheet — POP codes are mapped from the **Universe shop list** (then the legacy shop list). Upload universe once, then the daily file.

Shop SKU Wise Execution Report still parses if that is what you have. Opening it in Excel looks broken. That is normal. The tablix is a matrix, not a table:

| Region of the sheet | What it is | Keep? |
|---|---|---|
| Rows 1–2 | Textbox names (`txtRectangle`) and parameters (report name, user, UOM = Tons) | No |
| Field-id row | `txt_cDISTRIB`, `txt_cDSR_NA`, `txt_cSECTIO`, `txt_cPOP_Co`, `txt_cPOP_NA`, `txt_cSKU_LO`, `txt_Calendar`, `uval_MTD_Se` | Used only to map columns |
| Columns A–F on data rows | Repeated labels (`DISTRIBUTOR`, `DSR NAME`, …) | No |
| Starting at column G | Distributor → DSR → section → POP code → POP name → SKU → year → month → **MTD volume in MT** | **Yes** |
| Everything to the right (`Hameed GS Total`, section / DSR / distributor / grand totals) | Matrix subtotals | No |

One row = one shop × one SKU × one month of secondary (distributor → retailer) volume. This is the gold-standard S&D fact, not primary factory billing.

The Google Drive files are a **format sample** (July–August 2025 vs 2026). The pipeline is built for the full extract: every month and year you drop in later. Areas that were listed later are not treated as 2024 churn — a shop only enters the history at its first billed month.

SSRS CSVs are jagged (parameter rows have fewer columns than the tablix) and hide last year’s number in extra `val_TotalC_*` columns. The parser pads the CSV, then maps each year-group measure to the row’s calendar year so 2025 volume is not copied onto empty 2026 months.

### Shop master

| Order | Field | Use |
|---|---|---|
| 1 | Distributor | Hierarchy |
| 2 | DSR / salesperson | Beat ownership |
| 3 | Store id (POP code, e.g. `T0001601407`) | Primary key |
| 4 | Store name | Display |
| 5–8 | Categorization | Kept, not used in models |
| 9 | Zone (one of three divisions) | Macro slice |
| 10 | City | Macro vs micro |
| 11 | Section | Beat / area |
| 12–13 | Unused | Dropped |

The master list is the **universe**. Strike rate = billed shops / universe shops. Shops on the master that never appear in sales are whitespace.

## What the machine actually learns

Monthly shop data is sparse. The pipeline does **not** pretend a neural net can “understand Excel”. It builds a stack that companies like Nielsen / IRI / FireAI-style S&D platforms use, adapted to one monthly MTD file:

1. **Baselines (XGBoost, with a rolling-median fallback)**  
   For every shop-month: lags, 3/6-month rolling means, seasonality (month sine/cosine), SKU depth, billed rate. The model answers: *given this outlet’s history, what volume was expected this month?* Residual = surprise.

2. **Anomalies (Isolation Forest + rules)**  
   A shop-month vector (own z-score, MoM, vs section, drop-size vs median, mix concentration, recency) is scored. Rules then *name* the surprise:
   - **Trade loading / stock dump** — this month ≥ 2.5× the shop’s typical drop
   - **Drop-off / lapse** — regular billed shop went quiet
   - **Lumpy** — high coefficient of variation (spike-and-skip cadence)

3. **Outlet clustering (K-Means on RFM + trend)**  
   Recency, billed-month frequency, average drop, 3-vs-3 trend, SKU breadth, lumpiness. Clusters are relabelled into language a NSM can use: Star Account, Growth Target, Declining Core, Churn Risk, Dormant, Lumpy / Loaded, Long Tail, Stable Core.

4. **Diagnostics (no model required, but they use the baselines)**  
   - Micro vs macro divergence (section −10% while national +5% = local execution miss)
   - SKU cannibalization (share shift + negative correlation of month-to-month changes)
   - Strike rate / numeric distribution vs universe
   - Pareto concentration (top 20% shops)
   - DSR scorecards (volume, strike, drop size)
   - Whitespace (never billed)

Every run writes a ranked `insights` table so a later ReAct agent can query *facts*, not re-parse Excel.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Generate a realistic messy SSRS file + shop master, train, brief
snd-intel demo

# Your own files
snd-intel ingest path/to/Shop_SKU_Wise_Execution_Report.xlsx --shops path/to/shop_master.xlsx
snd-intel brief
snd-intel actions
snd-intel dashboard
snd-intel export-excel SND_intelligence_brief.xlsx

# Drop-folder: copy new extracts into data/incoming/
snd-intel watch --once
```

JSON API (for a future agent): `snd-intel serve-api` then `GET /brief`, `/insights`, `/focus`, `/shops/{id}`, `/search?q=trade+loading`.

## Install on a Mac (curl, keep `.venv`)

The warehouse is **not** stored in the app folder. It lives at:

`~/Library/Application Support/SND Intelligence/warehouse.db`

Git is not required. Update is `curl` the branch ZIP, `rsync` over `~/sndintel`, keep `.venv`, `pip install -e .`.

**Install once**

```bash
rm -rf /tmp/sndintel-dl
mkdir -p /tmp/sndintel-dl "$HOME/sndintel"
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/actionable-ops-layer-2f34.zip" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
rsync -a --delete --exclude '.venv' "$SRC/" ~/sndintel/
cd ~/sndintel
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
snd-intel app
```

In the app: **Upload files** → shop list once, then the sales extract. Later months: sales file only.

**Update later** (warehouse stays — do not re-upload July). Same `.venv` method as before; only the branch in the URL changed:

```bash
rm -rf /tmp/sndintel-dl
mkdir -p /tmp/sndintel-dl
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/actionable-ops-layer-2f34.zip" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
rsync -a --delete --exclude '.venv' "$SRC/" ~/sndintel/
cd ~/sndintel && source .venv/bin/activate && pip install -e . && snd-intel app
```

Do not pick a ZIP from Downloads — an old `SND-pro*.zip` will silently install the previous branch. After this landing, **Warehouse** should show version **0.6.1**. The app opens on **This week → Monday dispatch**.

Open the app → **Upload files**. Universe can stay in the warehouse. Drop **one or more Outlet Date Wise** files (split by shops or dates) and leave **Replace all billed sales** ticked so Shop SKU Wise rows go away. Score warehouse. AMS is the last three *closed* months (May+June+July when scoring August), paced vs billed if MTD is open.

Warehouse → **Clear billed sales only** also wipes billed rows and keeps shop lists. Do not upload the same POP+day in two files — those volumes are added.

## How a new file is applied

The Google Drive sample is July + August in one extract. The next file you drop will often be **August only** (or a later cut of the same month as MTD grows).

- Every calendar month **present in the file** is replaced in full. That is how MTD works: 20 Aug 0.60 MT at a shop becomes 0.95 MT when the 20 Aug extract is superseded. Shops that drop off that month’s extract are removed from that month, not left as stale MTD.
- Months **not** in the file stay as they are. July does not change when you upload August.
- Insights are rebuilt from the **whole warehouse** (closed July + open August MTD + any earlier history), not from the new file in isolation.
- Focus is **Expected-based**. Every grain is scored billed versus its own typical same calendar month (history + destationalized trend, then children scaled so they add to the parent Expected). A city that declined with the country is still a hole if it missed that typical month. Intra-month pace uses elapsed calendar days of the learned typical month until successive MTD cuts train your own curve.
- If the SSRS header has `Execution Date & Time` before month-end, that month is tagged `mtd_open`. Briefings use run-rate vs last year’s **closed** August instead of comparing 20 days to 31.

## Strategy questions this is built to answer

| Question | Where it shows up |
|---|---|
| How is each city doing vs what it should be billing? | Strategy city waterfall |
| Which distributor / DSR / shop in that city? | Open the city card; Named targets CSV |
| Which areas/shops must we focus on this week? | Strategy must-visit + Focus slice by gap |
| What are we doing well, so we can copy it? | Positive insights, DSR wins, local outperformance |
| Did salespeople dump stock into a shop? | Trade-loading anomalies (spike vs that shop’s own 6-month median) |
| Is a section dying while the city is fine? | Divergence insights |
| Are we covering the universe or just the same 30 shops? | Strike rate by DSR, whitespace |
| Is 5L oil growth real or stealing 16kg tin? | Cannibalization / substitution |
| Are we over-dependent on a handful of accounts? | Pareto / concentration |
| Who on the team is actually productive? | DSR scorecard (strike × drop size × MoM) |

## Project layout

```
src/sndintel/
  ingest/ssrs.py       SSRS chrome stripper + column inference
  ingest/shops.py      Universe parser
  ingest/pipeline.py   Snapshot-replace months in the file → features → models → insights
  materiality.py       Pareto core / middle / long-tail (not every quiet shop is 'lost')
  strategy.py          Five-play briefing
  ui/app.py            Local app: upload + strategy pack
  mtd.py               Closed vs open MTD from SSRS execution date
  features.py          Shop-month panel, lags, z-scores
  models.py            XGBoost, Isolation Forest, K-Means
  insights.py          Ranked narratives + KPI snapshots
  dashboard_app.py     Streamlit briefing
  api.py               FastAPI
  watch.py             data/incoming drop folder
  sampledata.py        Synthetic company (Quetta / Karachi / Lahore edible oil)
```

Warehouse: `data/warehouse.db`. Models: `data/models/`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full blueprint (ingestion, scoring, agent contract).
