# Architecture — SND Intelligence

Blueprint for a repeatable, file-drop secondary-sales intelligence pipeline.

## 1. Data foundation

```
SSRS xlsx/csv  ─┐
                ├─► parsers ─► SQLite facts ─► shop-month panel ─► features
Shop master ───┘
```

- Parsers are heuristic. They do not assume column G; they look for SSRS field ids, then human headers, then a positional distributor + POP-code + year/month + numeric MTD pattern.
- Totals (any header containing `Total`) and repeated A–F labels are dropped.
- Facts for months **in the incoming file** are snapshot-replaced (DELETE then insert). An August-only extract is the new truth for August MTD and **leaves July (and every other month) untouched**. Shop–SKU lines that disappear from the new extract are removed, not left as stale MTD.
- `period_ledger` records whether each month is `closed` or `mtd_open` (SSRS execution date before month-end). Insights for an open month use run-rate vs last year's closed month, not raw MTD vs a full August.
- Indexes: `store_id`, `section`, `city`, `period`.
- Shop-month panel starts at each outlet's **first billed month**. Leading zeros are not invented for areas that were not on the file yet. Later gaps *after* that first bill are real zeros (skipped / lapsed).
- On ingest we materialise composite metrics: calendar-aware lags (MoM and YoY, not "12 rows back"), 3/6-month rolling mean and median, own z-score, recency, 12-month billed rate, top-SKU share, vs-section / vs-city, `yoy_comparable` flag.

SQLite is the right first backbone: one file, WAL mode, portable to a sales laptop, queryable by a ReAct agent without a server. Move to Postgres later if several users write concurrently.

## 2. Machine learning engine

| Model | Grain | Job |
|---|---|---|
| XGBoost regressor | shop-month | Expected volume. Trained on all shops together with shop-normalised lags (global model, local history). Latest period held out of the fit when enough history exists. |
| Seasonal / rolling-median fallback | shop-month | Used when history < 8 periods or the booster cannot fit. |
| Isolation Forest | latest shop-month vector | Unsupervised surprise. Contamination ~6%. |
| Rule layer | same | Names the surprise: trade load, drop-off, lapse, lumpy. |
| K-Means (k via silhouette 3–6) | shop snapshot | RFM + trend + breadth + CV → business segment labels. |

Why not a shop-specific Isolation Forest? Monthly history is too short. A global forest on *normalised* features is the standard pattern for retail exception detection.

Why XGBoost not ARIMA/Prophet per shop? Hundreds of shops × short series. A pooled tree model with calendar features and lags is the usual CPG demand-planning choice at this grain.

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

## 4. Execution

- CLI `snd-intel app` / `ingest` / `watch` / `demo` / `brief` / `where`.
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
