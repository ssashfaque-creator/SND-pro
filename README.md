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

No extra LLM or ML is used on this file. Name-matching models mis-assign kiryana whales; the run-rate Expected already carries the national day shape.

## What the machine actually computes

Monthly shop data is sparse. The pipeline does **not** pretend a neural net can “understand Excel”, and it does not run a second model beside the pack's Expected. There is **one Expected** and every number in every pack is derived from it:

1. **Expected (run-rate baseline)**  
   For every unit and every shop: last-three-month AMS blended with the last-six-month median, paced by the national day curve when the month is open. Shop months are **winsorised** (capped at 2.5× the shop's median billed month) before the run-rate is taken, so one loading month does not become next quarter's Expected. Sparse doors shrink toward their city. The shop history chart draws this same line — there is no separate ML forecast.

2. **Gap = Expected − Billed, floored at 0**  
   The same definition on the briefing, the Monday dispatch, the situation cascade and the shop book. A unit that beat Expected shows Gap 0 and the surplus under *Ahead by*. The Empirical-Bayes shrunk residual only **ranks** lists; it is never printed as an amount.

3. **Materiality is relative**  
   A unit (DSR / distributor / city / country) is off Expected when the difference exceeds 5% of Expected for small books, 2% for large ones, never under 150 kg (`materiality.unit_material_mt`). A shop is off Expected past 20% and 10 kg. Within each DSR the doors outside the top 80% of size are **tail** and are judged as a coverage panel (doors billed vs usual), not as individual holes.

4. **Anomalies (rules only, against that Expected)**  
   - **Trade loading / stock dump** — this month ≥ 2.5× the shop’s run-rate Expected (the same multiple the winsoriser distrusts)
   - **Drop-off** — closed month, billed under 40% of a material Expected
   - **Quiet month** — closed month, billed 0 on a regular small biller (billed in at least half of the prior twelve months, run-rate ≥ 10 kg). The shop book decides *lost door* by days since last bill, `max(3 × cycle, 45 days)`; this is the early-warning list
   - **Lumpy** — high coefficient of variation (spike-and-skip cadence)
   No unsupervised outlier bucket: every flag names the rule that raised it.

5. **Outlet segments (fixed thresholds on RFM + trend)**  
   Recency, billed-month frequency, average drop, 3-vs-3 trend **relative to the market median**, SKU breadth, lumpiness → Star Account, Growth Target, Declining Core, Churn Risk, Dormant, Lumpy / Loaded, Long Tail, Stable Core. The same shop gets the same label on the same data; nothing is seeded.

6. **Next-order size (`nextdrop`)**  
   The only fitted model left: a pooled gradient-boosted regressor on billed-day sequences, using only what was known before each bill, that sizes this week's Ask when the last drop was fat or thin. Thin history falls back to the median drop. It never touches Expected or Gap.

7. **POP code hygiene (`dupes`)**  
   Two codes under one DSR with the same invoices on the same days are flagged *duplicate*; a same-name door that started billing when another stopped is *migrated*. Flagged codes are shown on the shop book instead of being called lost.

8. **Diagnostics (no model required, but they use the same Expected)**  
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
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/gap-engine-hardening-f506.zip" -o /tmp/sndintel-dl/app.zip
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
curl -L --fail "https://github.com/ssashfaque-creator/SND-pro/archive/refs/heads/cursor/gap-engine-hardening-f506.zip" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
rsync -a --delete --exclude '.venv' "$SRC/" ~/sndintel/
cd ~/sndintel && source .venv/bin/activate && pip install -e . && snd-intel app
```

Do not pick a ZIP from Downloads — an old `SND-pro*.zip` will silently install the previous branch. After this landing, **Warehouse** should show version **0.11.0**. Open **Report → Situation cascade** for the briefing pack, or **Report → Shop-wise issues** for every door in a scope (National / City / Distributor / DSR). Same MTD vs Monthly closing cut. The shop pack opens with the DSR roll-up, then the door bridge (who stopped, who started), then the shops. **Tail shops** are the doors outside the top 80% of size within their DSR (and every door under 10 kg); they are judged in the coverage panel, not one by one. **Missed Expected** = a live shop under its run-rate by more than 20% (and at least 10 kg); billed 0 is a miss, not a separate Unbilled row. **Lapsed (lost door)** = quiet past max(3× the measured cycle, 45 days); a door with no measured cycle needs 60 quiet days. Gap is Expected − billed floored at 0 everywhere; a scope that beat Expected shows "ahead by". Shop rows print kg; cover, roll-up and mix print MT to two decimals. The shop pack's Expected is the sum of each shop's own run-rate; the Situation cascade Expected for the scope is printed beside it with the reconciliation factor. **Flags** mark POP codes that look like a duplicate or a migrated code — check before calling them lost. “Every N days” is printed only when a purchase gap was measured. Open MTD does **not** treat still-to-Expected as a miss — issues are due this week (Ask), visited with no bill, and lost doors. This week → Monday dispatch is still the operating call list. Ask is the shop’s 90-day expected drop when the depletion ratio is ≥ 0.8; official Expected on Gap cards is last-3 / last-6 paced by the national day curve. Sales rows for POPs not on the shop list are kept and flagged off-master: they count in national volume but not in coverage.

**Re-coded shops (POP lineage).** The universe file is the complete door list. A code with billing history that is not on it is a **retired** code; when a live code with the same name started billing as it stopped (same distributor and DSR / code route, clean handover) the old history is merged into the live code, which is flagged *Continues retired code* — one shop, counted once, never a lost door plus a new door. A **superseded** code is the same handover where the old code is still on the universe file: its history is merged the same way and the old code is not counted as a door; the shop pack's *Retired codes* panel lists it with "Superseded — re-coded to …" so the universe file can be fixed. Retired codes with no clear successor sit in the same panel (never an issue, run-rate outside Expected). Expected windows stop at the first month on file: with data from May, July's *Last-2 avg* is (May + June) ÷ 2, and the cover prints that average next to Expected. Returns (negative cells in Outlet Date Wise) are netted at the month so the warehouse ties to the extract.

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
| Which shops missed Expected, and what should we do next month? | Report → Shop-wise issues → Monthly closing (city / distributor / DSR) |
| Which shops are due this week, visited with no bill, or lost? | Report → Shop-wise issues → MTD |
| Which areas/shops must we focus on this week? | Situation this-week doors + Monday dispatch + shop-wise MTD |
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
  shop_book.py         Shop-wise issues (MTD due list / closed-month misses)
  strategy.py          Five-play briefing
  ui/app.py            Local app: upload + strategy pack
  mtd.py               Closed vs open MTD from SSRS execution date
  features.py          Shop-month panel, lags, z-scores
  models.py            Run-rate baseline per shop-month, rule anomalies, RFM segments
  nextdrop.py          Next-order size (pooled GBM on billed-day sequences, median fallback)
  dupes.py             Duplicate / migrated POP-code detection
  demand.py            Depletion cycle, due / not due, lapse cut-off
  season.py            Expected engine (winsorised last-3 / last-6 run-rate, day curve)
  fmt.py               Number formatting: kg under 0.1 MT, two-decimal MT, never whole tons
  insights.py          Ranked narratives + KPI snapshots
  dashboard_app.py     Streamlit briefing
  api.py               FastAPI
  watch.py             data/incoming drop folder
  sampledata.py        Synthetic company (Quetta / Karachi / Lahore edible oil)
```

Warehouse: `data/warehouse.db`.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full blueprint (ingestion, scoring, agent contract).
