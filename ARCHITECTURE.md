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
- Facts are upserted on `(store_id, sku, period)` so a new monthly extract overwrites that month and leaves history intact.
- Indexes: `store_id`, `section`, `city`, `period`.
- On ingest we materialise composite metrics: 3/6-month rolling mean and median, own z-score, MoM/YoY, recency, 12-month billed rate, top-SKU share, vs-section / vs-city.

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

- **Micro vs macro** — section/DSR MoM minus national MoM. Gap ≥ 15 pp and local decline → execution failure, not category weather.
- **SKU cannibalization** — share shift this month vs last, plus Pearson correlation of first differences across the full history (≤ −0.55 ⇒ persistent substitution).
- **Strike-rate efficiency** — billed / universe by DSR. Low strike + large universe = underdeveloped beat. High strike + falling drop size = saturated coverage, mix/stock problem.
- **Trade loading** — spike vs *that shop’s* median drop, not vs a national average (otherwise large wholesalers always look like dumps).
- **Whitespace** — master shops with zero sales history.
- **Pareto** — top 20% billed shops’ volume share.
- **Positive copy-ables** — DSR/SKU/section outperformance.

## 4. Execution

- CLI `snd-intel ingest` / `watch` / `demo` / `brief` / `dashboard` / `serve-api` / `query` / `export-excel`.
- Drop folder `data/incoming`. Marker files `*.done` prevent double processing.
- Delta scoring is implicit: the new file is upserted, features use prior months as baseline, only residuals and new flags are ranked.
- Agent contract: FastAPI read models over SQLite. Do not let an LLM parse Excel; let it `GET /search?q=` and `GET /shops/{id}`.

## 5. What this cannot see (and should not fake)

No primary billing, no on-hand stock, no visit GPS, no scheme calendar, no competitor prices. Trade loading is inferred from *shape* (spike vs own baseline), not from primary−secondary gap. When those files exist, add them as extra facts and reuse the same insight table.

## 6. Promotion path

1. Run `demo` on synthetic data, confirm briefing quality.
2. Ingest 12+ months of real execution reports plus the current shop master.
3. Sit with sales to retune `SNDINTEL_TRADE_LOAD_X` (default 2.5) and divergence gap (15 pp).
4. Put the drop folder on the shared drive the MIS dump uses.
5. Only then wire a ReAct agent on `/brief` and `/search`.
