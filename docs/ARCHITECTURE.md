# Architecture

BizPulse is a single-process, file-based analytics platform: **files in, DuckDB star
schema in the middle, decisions out**. Everything runs locally or in one container. There
are no external services, no network calls at runtime and no secrets to manage.

## 1. Layered view

```
            ┌───────────────────────────────────────────────────────────────┐
 Layer 4    │  Presentation: Streamlit dashboard (5 tabs) · CLI · Docker    │
            └───────────────▲───────────────────────────────▲───────────────┘
                            │ read-only queries             │ upload → pipeline
            ┌───────────────┴──────────────┐ ┌──────────────┴───────────────┐
 Layer 3    │  Data science                │ │  Layer 2: Analytics / BI     │
            │  forecast · anomaly · what-if│ │  KPIs · RFM · cohorts · LTV  │
            └───────────────▲──────────────┘ └──────────────▲───────────────┘
                            └───────────────┬───────────────┘
            ┌───────────────────────────────┴───────────────────────────────┐
 Layer 1    │  Data engineering                                             │
            │  read → map → normalize → validate (contract) → dedupe → load │
            │  DuckDB: dim_date · dim_customers · dim_products ·            │
            │          fact_transactions · meta_pipeline_runs ·             │
            │          meta_quarantine · meta_datasets                      │
            └───────────────────────────────▲───────────────────────────────┘
                                            │ CSV · Excel · Parquet
                            ┌───────────────┴───────────────┐
                            │ company files / demo generator│
                            └───────────────────────────────┘
```

Layers only depend downward. Analytics and ML read the warehouse through SQL and never
touch raw files. The dashboard never writes to the warehouse except through the same
`run_pipeline()` entry point the CLI uses.

## 2. Package layout

```
src/bizpulse/
├── config.py              # Settings (paths from BIZPULSE_DATA_DIR), constants
├── logging_utils.py       # structured logging
├── resources/             # packaged defaults shipped inside the wheel
│   ├── fx_rates.yaml      # static reference FX rates (to USD)
│   └── mappings/          # ecommerce.yaml · saas.yaml · retail_fb.yaml
├── ingestion/
│   ├── readers.py         # CSV (sniffed delimiter/encoding) · Excel · Parquet → raw str frame
│   ├── mapping.py         # MappingConfig + column auto-detection heuristics
│   ├── normalize.py       # amount/currency/date/id/status normalization
│   └── hashing.py         # SHA-256 file, row and surrogate keys
├── quality/
│   ├── contracts.py       # Pydantic TransactionRecord + reason codes
│   ├── validator.py       # contract enforcement, dedupe, quarantine
│   └── report.py          # JSON + Markdown data-quality audit report
├── warehouse/
│   ├── schema.py          # DDL, table registry
│   ├── connection.py      # connection factory (always closed, Windows-safe)
│   └── loader.py          # staged upserts into the star schema
├── pipeline.py            # orchestration + run tracking (meta_pipeline_runs)
├── demo/                  # deterministic synthetic generator (seed=42)
├── analytics/             # Phase 2: kpis · rfm · cohorts
├── ml/                    # Phase 2: forecasting · anomaly · whatif
├── dashboard/             # Phase 3: Streamlit app
└── cli.py                 # `bizpulse` entry point
```

## 3. Data flow of one pipeline run

1. **Register run.** Insert a `meta_pipeline_runs` row (`status='running'`), compute the
   file's SHA-256. If a *successful* run already loaded the same bytes for the same
   dataset, the run ends as `skipped` (batch-level idempotency; `--force` overrides).
2. **Read.** Every column is read as a string (or native Excel value) so nothing is
   silently coerced before the contract sees it.
3. **Map.** Explicit mapping, then synonym match, then fuzzy match, then content-based
   detection (dates only). One source column can never satisfy two canonical fields.
   Ambiguity is reported, never guessed. A missing required field fails the run with
   an actionable message.
4. **Normalize.** Amounts (currency symbols, thousand separators, decimal commas),
   currency codes, dates (ISO, mixed, Excel serials, tz-aware to UTC), IDs, statuses.
   Every failure is captured as a reason code. Nothing is dropped silently.
5. **Validate.** Each row is checked against the Pydantic `TransactionRecord` contract.
   Then exact duplicates and conflicting duplicates are detected. Invalid rows go to
   quarantine with every reason attached. Valid rows continue.
6. **Quality gate.** If the quarantine ratio exceeds `max_quarantine_ratio` (default
   0.5), the run fails without loading anything, because the mapping is probably wrong.
7. **Load.** In one transaction: `dim_date`, `dim_customers`, `dim_products` and
   `fact_transactions` are upserted from staged temp tables (anti-join insert plus
   update-when-hash-changed). Unchanged rows are counted, not rewritten.
8. **Report.** JSON and Markdown audit reports are written to `data/reports/` and
   quarantined rows are written to `meta_quarantine`. The run row is finalised with
   counts and status.

## 4. Storage layout (all git-ignored, budget < 50 MB)

```
data/
├── raw/                    # demo inputs + manifests (ground truth for tests)
├── warehouse/bizpulse.duckdb
└── reports/<dataset>/<run_id>.{json,md}
```

`BIZPULSE_DATA_DIR` relocates everything (used by tests and by the container volume).

## 5. Key architectural decisions

| Decision | Rationale |
| --- | --- |
| DuckDB single file | Columnar OLAP speed, zero ops, ACID transactions, native pandas interop. |
| One warehouse, `dataset` column on every table | Multi-company comparisons and a single dashboard connection. Keys are hashed with the dataset name so tenants never collide. |
| SHA-256 surrogate keys | Deterministic and idempotent across re-runs and machines. No sequences to drift. |
| No declared foreign keys | DuckDB restricts updates on FK-referenced rows. Referential integrity is enforced by the loader and asserted in tests instead. |
| Staged upsert (anti-join insert + update) instead of `INSERT OR REPLACE` | Deterministic insert/update/unchanged counts and no delete-reinsert of primary keys inside a transaction. |
| Read everything as text first | Type coercion is a quality decision. The contract must see the raw value to give an exact reason. |
| Static FX table | Reproducible results offline. Rates are configuration, not live data. |
| Line-item grain | `order_id` is separate from `transaction_id`, so Order Count/AOV are exact while category/product analytics keep full detail. |

## 6. Non-functional properties

- **Deterministic**: demo data, keys, reports (except timestamps) and models are seeded.
- **Idempotent**: re-running the same file is `skipped`. Re-loading the same rows under a
  new file yields `0 inserted / 0 updated / N unchanged`.
- **Fault-isolated**: a bad row never fails a run, only a bad file or a failed quality
  gate does. A failed load rolls back completely.
- **Portable**: LF everywhere, no absolute paths, Python ≥ 3.11, non-root container.
