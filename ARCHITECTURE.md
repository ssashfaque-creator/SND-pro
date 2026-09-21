# Architecture — SND Intelligence

Blueprint for a repeatable, file-drop secondary-sales intelligence pipeline.

## 1. Data foundation

```
SSRS xlsx/csv  ─┐
Universe       ├─► parsers ─► SQLite facts ─► shop-month panel ─► features
Shop targets ──┘
```

- Parsers are heuristic. They do not assume column G; they look for SSRS field ids, then human headers, then a positional distributor + POP-code + year/month + numeric MTD pattern.
- Totals are recognised **structurally** (an SKU label that reads as a total, repeated A–F labels). No arithmetic guess: with carton-quantised volumes a real SKU equal to the sum of two others is routine, and deleting it silently understates the shop.
- One fact per shop + SKU + month. An identical line (same hierarchy, same volume) is a copy and is dropped; the same POP under two DSRs in one month is real and adds up to the extract's Grand Total. When the MTD measure exists for a key, year-group fallback rows for that key never add to it.
- Facts for months **in the incoming file** are replaced **only for the distributors and shops the file carries** (DELETE then insert, scoped). A regional cut of August replaces that region's August and leaves the other regions' August — and every other month — untouched. Two cuts of the same shop-month in one drop: the later file wins, they never add.
- Billed POPs that are not on the universe master are **kept and flagged off-master** (`stores.in_universe = 0`, `shop_month.in_universe`). They count in national and distributor volume so the warehouse ties to the extract's Grand Total; they are excluded from strike-rate denominators and coverage. Reconciliation shows their count and MT per period.
- `period_ledger` records whether each month is `closed` or `mtd_open` (SSRS execution date before month-end). `as_of_day` never moves backwards on an older cut; an open month closes when a later month arrives. Insights for an open month use run-rate vs last year's closed month, not raw MTD vs a full August.
- Indexes: `store_id`, `section`, `city`, `period`.
- Shop-month panel starts at each outlet's **first billed month**. Leading zeros are not invented for areas that were not on the file yet. Later gaps *after* that first bill are real zeros (skipped / lapsed).
- On ingest we materialise composite metrics: calendar-aware lags (MoM and YoY, not "12 rows back"), 3/6-month rolling mean and median, own z-score, recency, 12-month billed rate, top-SKU share, vs-section / vs-city, `yoy_comparable` flag.

SQLite is the right first backbone: one file, WAL mode, portable to a sales laptop, queryable by a ReAct agent without a server. Move to Postgres later if several users write concurrently.

## 2. Expected engine and the analytical layer

| Component | Grain | Job |
|---|---|---|
| Run-rate Expected (`season`) | every unit and shop | Last-three-month AMS blended with the last-six-month median; shop months winsorised at 2.5× the shop's median billed month first; sparse doors shrink toward the city. Paced by the national day curve when the month is open. Children are reconciled to the parent on the cascade; the shop book shows the unscaled shop sum beside the cascade figure with its factor. |
| Gap | every unit and shop | `max(0, Expected − Billed)`. One definition on every pack. Empirical-Bayes shrinkage orders lists only. |
| Unit materiality (`materiality.unit_material_mt`) | DSR / distributor / city / country | `max(0.15, min(5% × E, max(0.5, 2% × E)))` MT. Shared by the situation label, the boards, the KPI colours and the step list, so a row's label never contradicts its printed Gap. |
| Shop materiality (`shop_book`) | shop | Off Expected past 20% and 10 kg. Pareto tail per DSR (outside the top 80% by size, or under 10 kg) is judged as a coverage panel. |
| Depletion cycle (`demand`) | shop | Average purchase interval from billed days; due / not due; lost door past `max(3 × cycle, 45 days)` (60 quiet days with no measured cycle). A shop that billed this month is never lapsed. |
| Rule anomalies (`models.detect_anomalies`) | latest shop-month | Trade loading (≥ 2.5× Expected), drop-off (closed month, < 40% of a material Expected), quiet month (closed month, billed 0 on a regular biller), lumpy (CV ≥ 1.2). Every flag names its rule. |
| Segments (`models.cluster_shops`) | shop snapshot | RFM + market-relative trend + breadth + CV → fixed-threshold labels. Deterministic. |
| Next-order size (`nextdrop`) | shop, this week | Pooled gradient-boosted regressor on billed-day sequences (features known before each bill), median fallback under 80 training rows. Sizes the Ask; never Expected or Gap. |
| POP hygiene (`dupes`) | shop | Duplicate codes (same DSR, same invoices on the same days) and migrated codes (same folded name, one stops as the other starts). |

Why no global forecast model, Isolation Forest or K-Means any more? Each produced a second number beside the pack's Expected ("model 0.31 MT" against "Expected 0.27 MT"), flagged doors nobody could explain, or relabelled the same shop when a random-seeded fit moved. A field manager acts on one Expected, one Gap and a rule she can repeat; the system now has exactly that.

### Presentation rules (`fmt`)

MT prints to two decimals in tables and prose under 10 MT, one decimal above; never whole tons (a 1 MT rounding step is a DSR-week at city grain). Volumes under 0.1 MT are written in kg; shop rows are entirely in kg. Signs appear only on directional columns (vs Target, From …). Excel cells stay numeric with matching number formats so they still sum.

## 3. Diagnostic modules

Each module writes rows into `insights` with `type`, `severity`, `entity_*`, `narrative`, `action`, `metrics_json`, `rank_score`.

- **Fair-share isolation (shift-share)** — city / distributor / DSR / shop residual versus parent current book × last-year mix. Residuals sum to ~0. National declining → extra-declining units are the problem. National growing → slower-growth units are the problem. Empirical-Bayes shrinkage stops 0.02 MT shops from ranking above Eva Foods.
- **Coverage × velocity × mix** — CPG volume identity plus SKU industry-mix versus national SKU trends.
- **Seasonality** — calendar-month index and typical same-month level are **learned from every month in the warehouse** (city indices shrink toward national). Last year is one input, not the only one. Intra-month day shape is learned from mid-month MTD cuts when they exist; month-end totals cannot teach day 20, so open MTD then uses elapsed calendar days of the learned typical month. No shipped GT loading curve.
- **Exception briefing** — `situation_brief` headline / weather / problem / do-this-week. Cities that moved with the market are not a hit-list.

- **Micro vs macro** — section/DSR MoM minus national MoM. Gap ≥ 15 pp and local decline → execution failure, not category weather.
- **SKU cannibalization** — share shift this month vs last, plus Pearson correlation of first differences across the full history (≤ −0.55 ⇒ persistent substitution).
- **Strike-rate efficiency** — billed / universe by DSR. Low strike + large universe = underdeveloped beat. High strike + falling drop size = saturated coverage, mix/stock problem.
- **Trade loading** — spike vs *that shop’s* median drop, not vs a national average (otherwise large wholesalers always look like dumps).
- **Whitespace** — master shops with zero sales history.
- **Pareto** — top 20% billed shops’ volume share.
- **Positive copy-ables** — DSR/SKU/section outperformance.
- **Warehouse position** — closed YTD plus open MTD run-rate vs last year. Always ranked first so the briefing is the overall book, not “what was in the latest file”.
- **Volume bridge** — like-for-like vs new vs lost, split by Pareto **core / middle / tail**. Micro shops are one coverage KPI (weighted distribution), not a lost-account dump. Irregular billers are not treated as lapses.
- **Strategy plays** — at most five: close the month, protect the base, recover material volume, fix the beat, long-tail coverage / mix / people. Must-visit lists are material shops only.
- **Situation cascade** — sendable packs, no API key. National HQ names under- and over-performing cities, distributors, and DSRs, then five steps that close Gap versus Expected. Each city and each distributor gets the same skeleton, scoped, so it can be emailed. Detailed scorecards stay as the working file.
- **Sales-team plan** — shop-wise quota is a third number (Target), never a replacement for Expected. National Target is the submitted book. City / DSR Target rolls the book on live geography (unmatched kiryana names still count if Area matches). Shop identity is conservative; whales are never fuzzy-matched. Stretch above Expected is ambition. The optional national GPT brief may *cite* plan figures already on the scorecard; it is not used to match names or to forecast quota.

## 4. Execution

- CLI `snd-intel app` / `ingest` / `watch` / `demo` / `brief` / `situation` / `where`.
- On a Mac the warehouse is `~/Library/Application Support/SND Intelligence/warehouse.db` so unzipping a new app build does not wipe history.
- Local UI: upload shop list once, then incremental sales extracts. Drop folder `incoming/` still works.
- Delta scoring is implicit: months in the new file replace that month’s facts, features use the full history as baseline, and insights are rebuilt for the **overall warehouse**.
- Agent contract: FastAPI read models over SQLite. Do not let an LLM parse Excel; let it `GET /search?q=` and `GET /shops/{id}`.

## 5. What this cannot see (and should not fake)

No primary billing, no on-hand stock, no visit GPS, no scheme calendar, no competitor prices. Trade loading is inferred from *shape* (spike vs own baseline), not from primary−secondary gap. When those files exist, add them as extra facts and reuse the same insight table.

## 6. Promotion path

1. Run `demo` on synthetic data, confirm briefing quality.
2. Ingest 12+ months of real execution reports plus the current shop master.
3. Sit with sales to retune `SNDINTEL_TRADE_LOAD_X` (default 2.5) and divergence gap (15 pp).
4. Put the drop folder on the shared drive the MIS dump uses.
5. Only then wire a ReAct agent on `/brief` and `/search`.
