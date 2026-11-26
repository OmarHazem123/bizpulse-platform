# Application Flow

## 1. Personas

| Persona | Goal | Primary surface |
| --- | --- | --- |
| Founder / CEO | "Are we growing profitably? What needs my attention?" | Executive Overview, Anomaly Alerts |
| Growth / marketing lead | "Which customers do we keep, win back or upsell?" | Customer & Cohort Intelligence |
| Finance / ops | "What will next month look like? What if we change pricing?" | Forecasts, What-If Simulator |
| Data engineer / analyst | "Can I trust this data? How do I load our files?" | Upload & Quality Audit, CLI |

## 2. CLI flows

```
bizpulse generate-demo [--out DIR] [--seed 42] [--scale 1.0]
    → data/raw/{ecommerce_orders.csv, saas_invoices.xlsx, retail_fb_pos.parquet}
    → data/raw/manifests/<dataset>.json      (dirty-row + anomaly ground truth)

bizpulse run-pipeline --all                    # the three demo datasets
bizpulse run-pipeline --file my.csv --mapping my_mapping.yaml [--force]
bizpulse run-pipeline --file my.csv --dataset acme   # auto-detect columns
    → data/warehouse/bizpulse.duckdb
    → data/reports/<dataset>/<run_id>.{json,md}
    → exit 0 on success / success_with_warnings / skipped, 1 on failed

bizpulse runs [--dataset X]                    # run history from meta_pipeline_runs
bizpulse smoke                                 # Phase 3: end-to-end self-test (used by docker-smoke)
```

## 3. Pipeline state machine (one run)

```
        ┌─────────┐ same file hash already loaded ┌─────────┐
start ─►│ running ├──────────────────────────────►│ skipped │
        └────┬────┘                               └─────────┘
             │ read / map error, or quality gate breached
             ├─────────────────────────────────────►┌────────┐
             │                                      │ failed │ (nothing loaded,
             │ load ok                              └────────┘  transaction rolled back)
             ▼
   quarantined == 0 ? ──yes──► success
             │ no
             └──────────────► success_with_warnings
```

## 4. Dashboard flow (Phase 3)

```
Sidebar: [dataset ▼] [date range] [granularity: day/week/month]
│
├─ 1. Executive Overview
│     KPI tiles (Net Revenue, Gross Margin %, Net Profit, Orders, AOV, Refund Rate,
│     MoM growth) → revenue trend + forecast preview → Pareto chart → channel/category mix
│
├─ 2. Customer & Cohort Intelligence
│     RFM segment treemap + table → per-segment action playbook → cohort retention
│     heatmap → LTV curves by cohort
│
├─ 3. ML Forecasts & Anomaly Alerts
│     30-day forecast with 80/95 % bands → backtest metrics vs. Seasonal Naive →
│     anomaly timeline (CRITICAL/WARNING) → drill-down root-cause bars per alert
│
├─ 4. What-If Growth Simulator
│     sliders (retention, AOV, discount, acquisition, frequency, CAC) → 6-month
│     baseline vs. scenario chart → unit economics table (LTV, LTV:CAC, payback)
│
└─ 5. Company Data Upload & Quality Audit
      upload CSV/XLSX/Parquet → preview + auto-detected mapping (editable) →
      run pipeline → audit report (pass rate, reason table, quarantine sample,
      downloadable JSON/MD) → dataset appears in the sidebar selector
```

## 5. Error handling contract

| Situation | User-visible behaviour |
| --- | --- |
| Unsupported file type | Error naming the supported extensions. Nothing is written. |
| Required field unmappable | Run `failed`, listing the missing fields and the candidate headers. |
| Some rows invalid | Run `success_with_warnings`. Valid rows load and the report lists reasons. |
| > 50 % rows invalid | Run `failed` by the quality gate. The report is still written for diagnosis. |
| Same file uploaded twice | Run `skipped`, pointing to the original run. |
| Empty warehouse | Dashboard tabs show the "generate demo or upload" call-to-action. |
