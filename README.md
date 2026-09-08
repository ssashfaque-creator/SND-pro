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

### Shop-wise targets (sales-team plan)

Optional. Region / area / distributor / DSR / shop name / target MT (SSRS field ids like `txt_cPOP_NAME`, `uval_TARGET_UOM`). There is usually **no POP code** and **no calendar month** — treat the file as the live quota for the period being scored.

This is **not** a forecast. The model keeps three numbers the way Oracle SVP / SAP trade analytics do:

| Number | Meaning |
|---|---|
| **Actual** | Billed secondary |
| **Expected** (baseline) | Last-3 AMS blended with last-6 median, paced if MTD is open. Unchanged by the target file. |
| **Target** (quota) | What the sales team wrote, rolled to city / distributor / DSR. Open MTD uses the same national day curve as Expected. |

Stretch = max(0, paced Target − Expected) is ambition, not a coverage miss. Gap on the board is still billed vs Expected. Shop matching is conservative (exact city+distributor+name; whales ≥ 1 MT are never fuzzy-matched). Unmatched names still count in the city/DSR book if Area folds onto a unique live city. Warehouse → Shop plan shows match rate and unmatched rows.

No extra LLM or ML is used on this file. Name-matching models mis-assign kiryana whales; XGBoost Expected already carries seasonality and national day shape.

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
snd-intel ingest path/to/Shop_SKU_Wise_Execution_Report.xlsx --shops path/to/shop_master.xlsx --targets path/to/shopwise_targets.csv
snd-intel brief
snd-intel actions
snd-intel dashboard
snd-intel export-excel SND_intelligence_brief.xlsx
snd-intel situation
snd-intel export-situation SND_situation.pdf

# Drop-folder: copy new extracts into data/incoming/
snd-intel watch --once
```

JSON API (for a future agent): `snd-intel serve-api` then `GET /brief`, `/insights`, `/focus`, `/situation`, `/shops/{id}`, `/search?q=trade+loading`.

## Install on a Mac (curl, keep `.venv`)

The warehouse is **not** stored in the app folder. It lives at:

`~/Library/Application Support/SND Intelligence/warehouse.db`

Git is not required. Update is `curl` the branch ZIP, `rsync` over `~/sndintel`, keep `.venv`, `pip install -e .`.

**Install once**

```bash
rm -rf /tmp/sndintel-dl
mkdir -p /tmp/sndintel-dl "$HOME/sndintel"
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/situation-cascade-eccd.zip" -o /tmp/sndintel-dl/app.zip
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
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/situation-cascade-eccd.zip" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
rsync -a --delete --exclude '.venv' "$SRC/" ~/sndintel/
cd ~/sndintel && source .venv/bin/activate && pip install -e . && snd-intel app
```

Do not pick a ZIP from Downloads — an old `SND-pro*.zip` will silently install the previous branch. After this landing, **Warehouse** should show version **0.9.2**. Open **Report → Situation cascade**. Choose **MTD (this month)** for the in-month pack (billed so far, projected month-end, full monthly target) or **Monthly closing** for a finished month (results, not a live tracker). The national cover explains Gap versus Expected in MT: light orders (under-billed), unbilled, unvisited — and how much of the quota miss is stretch. National HQ has a city gap table (Target, Expected, Billed, Gap, vs Target, then that split). Comments name both drivers and the next action (city / DSR). Underperforming sales staff is unchanged. Weak-area pages are gone. This week → Monday dispatch is still the operating call list. Ask is the shop’s 90-day expected drop when the depletion ratio is ≥ 0.8; official Expected on Gap cards is still last-3 / last-6 paced by the national day curve.

Open the app → **Upload files**. Universe can stay in the warehouse. Drop **one or more Outlet Date Wise** files (split by shops or dates). Optionally drop **shop-wise targets** (quota / plan) — they do not replace Expected. Leave **Replace all billed sales** unticked for a weekly refresh: days in the new file override the same shop-days (a later 20 Aug file replaces an incomplete 20 Aug); other days stay. Tick replace-all only when switching from Shop SKU Wise or wiping billed history. Score warehouse. AMS is the last three *closed* months (May+June+July when scoring August), paced vs billed if MTD is open.

Warehouse → **Clear billed sales only** also wipes billed rows and keeps shop lists.

## How a new file is applied

The Google Drive sample is July + August in one extract. The next file you drop will often be **later days of August** (or a later cut of the same month as MTD grows).

- **Outlet Date Wise:** each shop-day present in the file **replaces** that shop-day in the warehouse. A 20 Aug extract that was incomplete is overwritten when the next file includes 20 Aug. Days not in the new file (1–19 Aug, or July) stay. Month totals are rebuilt from the combined daily rows for shops the file touched.
- **Shop SKU Wise:** every calendar month **present in the file** is replaced in full. Months **not** in the file stay. July does not change when you upload August.
- Insights are rebuilt from the **whole warehouse** (closed July + open August MTD + any earlier history), not from the new file in isolation.
- Insights are rebuilt from the **whole warehouse** (closed July + open August MTD + any earlier history), not from the new file in isolation.
- Focus is **Expected-based**. Every grain is scored billed versus its own typical same calendar month (history + destationalized trend, then children scaled so they add to the parent Expected). A city that declined with the country is still a hole if it missed that typical month. Intra-month pace uses elapsed calendar days of the learned typical month until successive MTD cuts train your own curve.
- If the SSRS header has `Execution Date & Time` before month-end, that month is tagged `mtd_open`. Briefings use run-rate vs last year’s **closed** August instead of comparing 20 days to 31.

## Strategy questions this is built to answer

| Question | Where it shows up |
|---|---|
| What is the national situation, and who is under / over? | Report → Situation cascade → National HQ (MTD or Monthly closing) |
| Are we on the monthly target, or only on Expected? | Situation KPIs: projected month-end vs full monthly target |
| What should we send each city / distributor? | Same page: City pack / Distributor pack, or ZIP of every pack |
| Which salespeople are lagging, and why? | Situation pack People sheets; capacity label is the coaching script |
| What is the plan, and what do we do next? | Situation pack → Plan and next actions |
| How is each city doing vs what it should be billing? | Situation city gap table (Target / Expected / Billed / Gap + light orders / unbilled / unvisited) |
| Which distributor / DSR / shop in that city? | City pack; named targets; detailed scorecards |
| Which areas/shops must we focus on this week? | Situation this-week doors + Monday dispatch |
| What are we doing well, so we can copy it? | Situation “ahead” sheets and Copy from overperformers |
| Did salespeople dump stock into a shop? | Trade-loading anomalies (spike vs that shop’s own 6-month median) |
| Is a section dying while the city is fine? | Situation weak areas; divergence insights |
| Are we covering the universe or just the same 30 shops? | Strike rate by DSR, whitespace, coverage driver |
| Is 5L oil growth real or stealing 16kg tin? | Cannibalization / substitution |
| Are we over-dependent on a handful of accounts? | Pareto / concentration |
| Who on the team is actually productive? | People ahead + DSR capacity Fine |

## Project layout

```
src/sndintel/
  ingest/ssrs.py       SSRS chrome stripper + column inference
  ingest/shops.py      Universe parser
  ingest/targets.py    Shop-wise sales-team quota (not a forecast)
  ingest/pipeline.py   Daily shop-day overlay (or SKU-wise month replace) → features → models → insights
  plan.py              Attach Actual / Expected / Target without mixing them
  materiality.py       Pareto core / middle / long-tail (not every quiet shop is 'lost')
  situation_report.py  National / city / distributor sendable packs
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
