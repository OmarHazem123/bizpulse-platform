# Technical Requirements

Requirements are numbered so tests and docs can reference them (e.g. `TR-ING-03`).

## 1. Environment & engineering standards

| ID | Requirement |
| --- | --- |
| TR-ENV-01 | Python ≥ 3.11 (pinned to 3.12 via `.python-version`), dependencies locked in `uv.lock`. |
| TR-ENV-02 | `uv run python tasks.py <cmd>` is the canonical runner. The `Makefile` mirrors every command 1:1 (verified by `tasks.py makefile-sync`). |
| TR-ENV-03 | All text files are LF-only (`.gitattributes` + `tasks.py eol` scans working-tree bytes). |
| TR-ENV-04 | `ruff check`, `ruff format --check` and `mypy --strict` pass with zero errors across `src/`, `tests/` and `tasks.py`. |
| TR-ENV-05 | No secrets, personal emails or absolute local paths in the repo (`tasks.py hygiene`). |
| TR-ENV-06 | Generated artifacts (`data/`, `artifacts/`) are git-ignored and stay < 50 MB (`tasks.py size`). |
| TR-ENV-07 | MIT license, git branch `main`. |

## 2. Canonical transaction contract (line-item grain)

| Field | Type | Req. | Rule (reason code on failure) |
| --- | --- | --- | --- |
| `transaction_id` | str | ✔ | non-empty, ≤ 128 chars (`transaction_id.missing`); unique per dataset (`transaction_id.conflicting_duplicate`) |
| `order_id` | str | | defaults to `transaction_id` |
| `order_date` | datetime | ✔ | parseable (`order_date.unparseable`); `2000-01-01 ≤ d ≤ as_of + 1 day` (`order_date.out_of_range`) |
| `customer_id` | str | ✔ | non-empty (`customer_id.missing`) |
| `product_id` | str | ✔ | non-empty (`product_id.missing`) |
| `product_name` | str | | defaults to `product_id` |
| `category` | str | | defaults to `Uncategorized` |
| `channel` | str | | lower snake case, defaults to `unknown` |
| `region` | str | | defaults to `unknown` |
| `quantity` | float | ✔ | parseable (`quantity.unparseable`); `0 < q ≤ 1e6` (`quantity.non_positive`, `quantity.out_of_range`) |
| `unit_price` | float | ✔* | parseable (`unit_price.unparseable`); `≥ 0` (`unit_price.negative`). *Or derive it from a mapped `line_total / quantity`. |
| `discount_amount` | float | | `≥ 0` (`discount_amount.negative`); `≤ quantity × unit_price` (`discount_amount.exceeds_gross`); default 0 |
| `unit_cost` | float | | `≥ 0` (`unit_cost.negative`). NULL when unknown, so margin is NULL, never 0. |
| `fee_amount` | float | | `≥ 0` (`fee_amount.negative`); default 0 |
| `currency` | str | | ISO-4217 code present in the FX table (`currency.unsupported`); default = dataset default |
| `status` | enum | | `completed` / `refunded` / `cancelled` after synonym mapping (`status.invalid`); default `completed` |

Row-level reasons: `row.empty` (every field blank), `row.duplicate` (exact duplicate of an
earlier row in the batch). A quarantined row carries **every** applicable reason, each
with `field`, `code`, `message` and the offending raw `value`. Cross-field rules only run
when their inputs are individually valid, so one defect yields one reason.

Every amount column is parsed independently, so an amount that is unparseable
or negative in a non-required column (`discount_amount`, `unit_cost`, `fee_amount`)
also quarantines the row.

## 3. Ingestion

| ID | Requirement |
| --- | --- |
| TR-ING-01 | Read `.csv`/`.txt` (delimiter sniffed among `, ; \t \|`, UTF-8 with BOM fallback to latin-1), `.xlsx`/`.xls` (first sheet or configured sheet), `.parquet`. |
| TR-ING-02 | Column mapping from YAML/JSON (`MappingConfig`). Unmapped canonical fields are auto-detected in this order: header synonyms, then fuzzy match (ratio ≥ 0.88), then content-based date detection. |
| TR-ING-03 | One source column maps to at most one canonical field. Ties are reported as ambiguous and left unmapped. |
| TR-ING-04 | Composite keys: `composite_keys.transaction_id: [col_a, col_b]` joins columns with `-`. |
| TR-ING-05 | Amount parsing handles currency symbols and codes, thousand separators, decimal commas (`1.234,56`, `45,90`), accounting negatives `(12.00)` and native numerics. |
| TR-ING-06 | Date parsing handles ISO-8601, mixed formats (honouring `dayfirst`), Excel serial numbers and tz-aware strings (converted to naive UTC). |
| TR-ING-07 | Amounts are converted to the dataset's `reporting_currency` using `fx_rates.yaml`. The original currency and rate are stored on the fact row. |
| TR-ING-08 | Batch idempotency: SHA-256 of file bytes. A previously *successful* identical batch is `skipped` unless forced. |
| TR-ING-09 | Row idempotency: `transaction_key = sha256(dataset ∣ transaction_id)`, `row_hash = sha256(canonical normalized values)`. Same key and same hash means unchanged. Same key and a new hash means update. |
| TR-ING-10 | `negative_quantity_as_refund: true` turns `q < 0` into `abs(q)` with `status = refunded`. |

## 4. Data-quality audit report

JSON plus Markdown per run, at `data/reports/<dataset>/<run_id>.{json,md}`. Contents: run
metadata (source *file name* only, never an absolute path), row counts (read, valid,
quarantined, inserted, updated, unchanged), pass rate, resolved column mapping with the
detection method, unmapped source columns, reason-code counts, field completeness,
currency mix and conversions, date parsing methods, warnings, and a sample of
quarantined rows (≤ 25).

## 5. Warehouse (DuckDB)

| Table | Grain / key | Notes |
| --- | --- | --- |
| `dim_date` | one row per day, `date_key = yyyymmdd` | ISO week, quarter, weekday, weekend flag |
| `dim_customers` | `customer_key = sha256(dataset∣customer_id)` | first/last seen, first channel, region, `is_anonymous` |
| `dim_products` | `product_key = sha256(dataset∣product_id)` | latest name and category, first/last seen |
| `fact_transactions` | `transaction_key` | measures in reporting currency, `row_hash`, `run_id`, lineage timestamps |
| `meta_pipeline_runs` | `run_id` (UUID4) | status, counts, file hash, mapping, timings, error |
| `meta_quarantine` | `(run_id, source_row)` | reasons JSON + raw record JSON |
| `meta_datasets` | `dataset` | display name, domain, reporting currency, latest mapping |

### Fact measures (reporting currency)

- `gross_amount = quantity × unit_price × fx_rate`
- `discount_amount = discount × fx_rate`
- `net_amount = gross_amount − discount_amount`
- `cost_amount = quantity × unit_cost × fx_rate`, NULL when cost is unknown
- `fee_amount = fee × fx_rate`
- `is_refund = (status = 'refunded')`

## 6. KPI definitions (Phase 2, fixed here)

Let **S** be the non-cancelled rows (`completed ∪ refunded`), **C** the completed rows
and **R** the refunded rows. Cancelled rows are stored but excluded from every KPI.

| KPI | Formula |
| --- | --- |
| Gross Revenue | `Σ_S gross_amount`, all sales before discounts and refunds |
| Discounts | `Σ_S discount_amount` |
| Refunds | `Σ_R net_amount` |
| **Net Revenue** | `Gross − Discounts − Refunds = Σ_C net_amount` |
| COGS | `Σ_C cost_amount`. Refunded goods are assumed returned, so no COGS. |
| Gross Profit | `Σ_C (net_amount − cost_amount)` over rows with known cost |
| **Gross Margin %** | `Gross Profit / Σ_C net_amount` over rows with known cost. NULL if no costs. The report states cost coverage. |
| Fees | `Σ_S fee_amount`. Processing fees are not returned on refunds. |
| **Net Profit (contribution)** | `Net Revenue − COGS − Fees`. Contribution-level, because operating expenses are not in transactional data. |
| Order Count | `COUNT DISTINCT order_id` over **S** |
| **AOV** | `(Gross Revenue − Discounts) / Order Count`, the value at purchase time |
| Refund Rate | `Refunds / (Gross Revenue − Discounts)`, value-based. The order-based rate is reported alongside. |
| MoM / WoW growth | `(X_t − X_{t−1}) / X_{t−1}` on complete calendar months / ISO weeks only |
| Pareto 80/20 | products (or categories) ranked by Net Revenue; smallest set reaching 80 % of the total |
| Active customers | distinct non-anonymous customers with ≥ 1 row in **S** in the period |

## 7. Analytics & ML requirements (Phase 2)

| ID | Requirement |
| --- | --- |
| TR-AN-01 | RFM: quintile scores (R reversed), ties broken by rank. The R×F grid partitions into Champions / Loyal Customers / Potential Loyalists / New Customers / At Risk / Hibernating-Lost, each with an action playbook. Anonymous customers are excluded. |
| TR-AN-02 | Cohorts: monthly acquisition cohort (first order in **S**), retention matrix `active(c, k) / size(c)`, cumulative net-revenue LTV per customer at months 1, 3, 6 and 12. |
| TR-ML-01 | Daily net-revenue forecast, 30-day horizon, recursive LightGBM on lag/rolling/calendar features vs. a Seasonal-Naive (7-day) baseline. |
| TR-ML-02 | Backtest with `TimeSeriesSplit` (fixed 30-day test folds). Features at time *t* use data `< t` only. Multi-step predictions are recursive, never fed actuals. |
| TR-ML-03 | Metrics MAE, RMSE, MAPE (non-zero actuals), sMAPE for model and baseline. 80 %/95 % intervals from empirical out-of-fold residual quantiles per horizon step. |
| TR-ML-04 | Anomalies on daily KPIs (net revenue, orders, AOV, refund rate): rolling z-score and IQR on a trailing window that excludes the current day, plus IsolationForest. Severity is `CRITICAL` or `WARNING`. Root cause ranks category/channel/segment slices by deviation from their own baseline. |
| TR-ML-05 | What-if: 6-month projection from a trailing baseline with levers for retention, AOV, discount, new-customer volume and order frequency. Outputs revenue, gross profit, contribution, customers, LTV, LTV:CAC and payback. |

## 8. Performance targets (local laptop)

| Operation | Target |
| --- | --- |
| Generate all demo data | < 30 s |
| Pipeline, ~50k rows | < 20 s end to end |
| Full test suite | < 3 min |
| Generated artifacts | < 50 MB total |
