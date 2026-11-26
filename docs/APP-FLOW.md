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

bizpulse report [--dataset X] [--json PATH] [--no-forecast] [--no-anomalies]
    → KPIs (all time, last complete month, MoM / WoW), Pareto, channel / category mix,
      RFM segments with first actions, LTV and retention, peak local hours,
      forecast backtest summary, anomaly counts and the last 30 days of alerts
bizpulse forecast --dataset X [--horizon 30] [--csv PATH]
    → backtest metrics vs seasonal naive + daily yhat with 80 / 95 % intervals
bizpulse whatif --dataset X [--retention-pp 2] [--aov-pct 5] [--discount-pp -1]
                [--new-customers-pct 10] [--frequency-pct 3] [--cac 40] [--months 6]
    → month-by-month scenario vs status quo, totals, unit economics
    → exit 2 when a lever is outside its bounds

bizpulse smoke [--scale 0.1] [--work-dir DIR]  # end-to-end self-test in an isolated temp dir
bizpulse dashboard [--port 8501] [--address 0.0.0.0] [--headless] [--demo-if-empty]
```

`report`, `forecast` and `whatif` exit 1 with a "load data first" hint when the
warehouse is empty, and with the list of available datasets for an unknown name.

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

## 4. Dashboard flow

```
Sidebar: [dataset ▼] [date range] [granularity: day/week/month] [refresh]
│   The date range scopes KPIs, trends, Pareto, mix and alert lists. Segments, cohorts,
│   forecasts, anomaly scoring and the simulator always use the full history.
│
├─ 1. Executive Overview
│     KPI tiles with sparklines and the change vs. the previous equal-length period
│     (Net Revenue, Gross Margin %, Net Profit, Orders, AOV, Refund Rate) → MoM growth
│     and a 30-day forecast preview tile → trend chart (metric selector) → Pareto
│     cumulative-share curve → orders by local weekday × hour → channel / category /
│     region mix
│
├─ 2. Customer & Cohort Intelligence
│     RFM share-of-customers vs. share-of-revenue bars + table → per-segment action
│     playbook + customer-list CSV download → cohort retention heatmap → LTV tiles
│     (1/3/6/12 months) → LTV curves (each cohort in gray, size-weighted average in color)
│
├─ 3. ML Forecasts & Anomaly Alerts
│     30-day forecast with 80/95 % bands and the seasonal-naive baseline → backtest
│     tiles, per-fold MAE, metrics table, top features → alert filters (scope,
│     severity, metric) → CRITICAL / WARNING counts → timeline (total or a store /
│     channel slice) → alert table → root-cause bars for the selected alert
│
├─ 4. What-If Growth Simulator
│     sliders (retention, AOV, discount, acquisition, frequency) + CAC input (default
│     0: not in the data) → 6-month status quo vs. scenario charts (net revenue,
│     contribution) → totals and unit economics (LTV, LTV:CAC, payback)
│
└─ 5. Company Data Upload & Quality Audit
      upload CSV / XLSX / Parquet → dataset key, display name, business type,
      currency, timezone, day-first → preview + auto-detected mapping (editable; a
      changed source column is pinned explicitly) → validate and load → result
      (status, pass rate, row reconciliation, reason chart, quarantine sample,
      JSON / Markdown downloads) → dataset appears in the sidebar selector →
      run history with each run's audit report
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
