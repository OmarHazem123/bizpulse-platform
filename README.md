# BizPulse Platform

A domain-agnostic **decision intelligence platform** for startups and small companies.
It ingests messy sales or billing exports (CSV, Excel, Parquet) through strict data
contracts into a DuckDB star schema. From there it answers the questions founders,
growth, finance and data people actually ask:

- Are we growing profitably?
- Which customers do we keep, win back or upsell?
- What will next month look like?
- What changed, and where?
- What if we change pricing or retention?

It works for any vertical through column mapping. Three demo businesses ship with it:
an e-commerce shop, a SaaS company and a cafe chain.

| Layer | What it does |
| --- | --- |
| **Data engineering** | Multi-format ingestion with column auto-detection, currency / date / type normalization, a Pydantic data contract that quarantines bad rows with exact reasons, idempotent SHA-256-keyed loads, a DuckDB star schema with run lineage, and JSON + Markdown data-quality audit reports. |
| **Analytics / BI** | Gross/net revenue, gross margin, net profit, orders, AOV, MoM/WoW growth, refund rate, Pareto 80/20, RFM segments with action playbooks, monthly cohort retention, empirical LTV, hour-of-day in each business's timezone. |
| **Data science** | Leak-free LightGBM revenue forecast vs. a seasonal-naive baseline (`TimeSeriesSplit` backtest, 80/95 % intervals); KPI anomaly detection (robust z-score, IQR and an expanding-window IsolationForest) with CRITICAL/WARNING severities, store/channel slices and root-cause attribution; a 6-month what-if and unit-economics simulator. |
| **Product** | `bizpulse` CLI, a 5-tab Streamlit + Plotly dashboard, and a non-root Docker image with a health check. |

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned and installed by
uv). Docker is optional.

```bash
uv run python tasks.py setup           # sync the environment, verify imports
uv run python tasks.py generate-demo   # 3 deterministic demo datasets (seed 42, 18 months)
uv run python tasks.py run-pipeline    # ingest -> validate -> quarantine -> DuckDB star schema
uv run python tasks.py dashboard       # http://localhost:8501
```

Or in Docker. The image ships with the demo warehouse, so the dashboard has data
immediately:

```bash
uv run python tasks.py docker-build    # or: docker build -t bizpulse-platform .
docker run --rm -p 8501:8501 bizpulse-platform
docker run --rm bizpulse-platform bizpulse report --dataset ecommerce
```

A `Makefile` mirrors every `tasks.py` command (`make check`, `make dashboard`, ...).
Parity between the two is itself a checked gate.

## Load your own data

Files are mapped to one canonical order-line contract: `transaction_id`, `order_date`,
`customer_id`, `product_id`, `quantity`, plus `unit_price` or `line_total`. Optional
fields are `order_id`, `product_name`, `category`, `channel`, `region`, `status`,
`discount_amount`, `unit_cost`, `fee_amount` and `currency`. Headers are matched by
synonym, fuzzy matching or date sniffing, and anything can be pinned in a YAML/JSON
mapping.

```bash
uv run bizpulse run-pipeline --file my_orders.csv --dataset acme          # auto-detect
uv run bizpulse run-pipeline --file my_orders.xlsx --mapping acme.yaml    # pinned mapping
uv run bizpulse runs                                                      # audit trail
```

The dashboard's **Company Data Upload & Quality Audit** tab does the same interactively:
upload, preview, edit the detected mapping, load, and read the audit report.

How a run handles bad data:

- Invalid rows are quarantined with reason codes such as `order_date.unparseable` or
  `quantity.non_positive`, and valid rows still load.
- If more than 50 % of the rows fail, nothing loads, because the mapping is probably
  wrong.
- Re-sending the same file is `skipped`.
- Re-sending changed rows updates only those rows.

## CLI

| Command | Purpose |
| --- | --- |
| `bizpulse generate-demo [--scale 1.0] [--seed 42]` | deterministic demo data: CSV, Excel, Parquet, plus ground-truth manifests |
| `bizpulse run-pipeline --all \| --file F (--dataset D \| --mapping M) [--force]` | ingest, validate, quarantine, load, compact |
| `bizpulse runs [--dataset D]` | pipeline run history (lineage) |
| `bizpulse report [--dataset D] [--json PATH]` | KPIs, growth, Pareto, mix, RFM with first actions, LTV/retention, peak hours, forecast backtest, recent alerts |
| `bizpulse forecast --dataset D [--horizon 30] [--csv PATH]` | daily forecast with 80/95 % bands and backtest metrics |
| `bizpulse whatif --dataset D --retention-pp 2 --aov-pct 5 --cac 40` | 6-month scenario vs. status quo and unit economics |
| `bizpulse smoke` | end-to-end self-test on small demo data in an isolated temp dir |
| `bizpulse dashboard [--port 8501] [--demo-if-empty]` | start the dashboard |

## Dashboard

1. **Executive Overview.**
   - KPI tiles with sparklines and the change vs. the previous period.
   - MoM growth and a 30-day forecast tile.
   - A trend chart (metric selector).
   - A Pareto curve.
   - Orders by local weekday and hour.
   - Channel, category and region mix.
2. **Customer & Cohort Intelligence.**
   - RFM segments, comparing share of customers with share of revenue.
   - An action playbook per segment, plus a customer-list CSV.
   - A cohort retention heatmap.
   - LTV curves.
3. **ML Forecasts & Anomaly Alerts.**
   - The forecast with its intervals and the seasonal-naive baseline.
   - Backtest folds, metrics and top features.
   - Alerts filtered by scope, severity and metric.
   - A timeline per business total or per store/channel slice.
   - Root-cause bars for the selected alert.
4. **What-If Growth Simulator.**
   - Retention, AOV, discount, acquisition and frequency levers, plus a CAC input.
   - Status quo vs. scenario for net revenue and contribution.
   - Unit economics: LTV, LTV:CAC and payback.
5. **Company Data Upload & Quality Audit.**
   - Upload, mapping editor, validation and load.
   - The audit report with JSON/Markdown downloads.
   - Run history.

How the dashboard behaves:

- **Rendering.** Tabs render lazily, and results are cached per warehouse version.
- **Colors.**
  - Each color has one job: identity, magnitude, or status.
  - Red and amber are reserved for CRITICAL and WARNING, always with an icon and a label.
  - Light and dark mode use separately validated steps.
- **Accessibility.** Every chart has a table view.

## Measured results

The data is the full-scale demo data (seed 42, 2025-05-01 → 2026-10-31). Timings were
measured on a Windows 11 laptop with Python 3.12.

**Forecasting.** 6 rolling 30-day `TimeSeriesSplit` folds; criteria registered before
the first run, no tuning:

| Dataset | LightGBM MAE | Seasonal-naive MAE | Ratio | 80 % interval coverage | 95 % interval coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ecommerce (USD/day) | 876 | 922 | **0.95** | 73.9 % | 91.1 % |
| retail_fb (EUR/day) | 85.7 | 104.5 | **0.82** | 71.7 % | 89.4 % |
| saas (USD/day) | 5,763 | 7,835 | **0.74** | 72.8 % | 90.0 % |

**Anomaly detection.** The generator plants 7 day-level ground-truth incidents:

- a checkout outage, a refund spike and a viral social day (e-commerce);
- a store closure, a POS outage and a BOGO promo (cafes);
- a one-off enterprise deal (SaaS).

An eighth incident, a month-level SaaS churn spike, is documented but out of daily and
weekly scope.

- **Recall: 7/7.** Each is detected on its metric inside its window, with the correct root
  cause or scope.
- **Immutability:** a stored severity never changes when new data arrives. This is
  asserted by a test.
- **Quiet data:** synthetic data with no incidents never produces a CRITICAL. Also
  asserted.

**Data quality.**

- The generator plants about 1–2 % dirty rows, each with exactly one defect.
- The quarantine equals the ground-truth manifest **row for row, reason for reason**.
- Pass rates: 98.80 % (ecommerce), 98.84 % (saas), 98.82 % (retail_fb).
- The generator is byte-identical across runs.

**KPI snapshot.**

| Dataset | Net revenue | Gross margin | Other figures |
| --- | ---: | ---: | --- |
| ecommerce | $1.92M | 45.6 % | AOV $72.35; 26 of 120 products = 80 % of revenue |
| saas | $3.56M | 76.9 % | |
| retail_fb | €305k | 73.7 % | peak local hours 08:00 and 13:00 (Europe/Berlin; UTC storage would suggest 06:00 / 11:00) |

**Performance.**

| Operation | Time |
| --- | ---: |
| Generate the demo data (123k rows, 3 formats) | about 9–12 s |
| Load all three datasets incl. validation and compaction | about 26 s |
| Forecast with backtest | 1.2–1.4 s per dataset |
| Anomaly scoring | 2.5–5.8 s per dataset |
| `bizpulse smoke` | about 18 s |

The warehouse is 22 MB after compaction. All generated artifacts total about 29 MB, under
the 50 MB budget.

**Container.**

- The image is 1.11 GB, of which the Python environment is 659 MB and pyarrow alone 166 MB.
- It reports healthy about 12 s after start.
- `tasks.py docker-smoke` finishes in about 55 s with a warm build cache, or 3–4 min
  cold.

**Tests.** 242 pytest tests pass in about 2.5 minutes, with no `xfail`. They cover
unit, component and integration tests, the CLI, the dashboard (Streamlit `AppTest`)
and the full-scale ML acceptance criteria. `tasks.py docker-smoke` comes on top.

## Engineering standards

`uv run python tasks.py check` runs every gate, and each must pass:

- **eol:** LF line endings, checked on the working-tree bytes.
- **hygiene:** no secrets, personal emails or local machine paths.
- **makefile-sync:** Makefile parity with `tasks.py`.
- **lint and format:** `ruff`.
- **typecheck:** `mypy --strict` over `src/`, `tests/` and `tasks.py`.
- **size:** generated data stays under 50 MB.

Beyond that, `tasks.py test` runs pytest and `tasks.py docker-smoke` builds the image
and verifies it. Everything is reproducible:

- The demo data is seeded and its window is fixed.
- Surrogate keys are deterministic: 64-bit keys derived from SHA-256, plus a full SHA-256
  row hash.
- The models are seeded.
- The wall clock is never used for analysis dates.

## Limitations

- **The anomaly rule was amended after the fact.** The pre-registered rule reached 4/7.
  Two amendments, approved and documented in `docs/TECHNICAL-REQUIREMENTS.md` §7.1,
  brought recall to 7/7:
  - two-tailed log-scale scoring;
  - weekly monitoring for SaaS, an expanding-window IsolationForest, calibrated
    severities, and slice monitoring.

  Severity agreement with the planted expectations is **3/7**. Detection is reliable;
  the CRITICAL vs. WARNING grading is not yet.
- **The intervals are slightly narrow.** About 72–74 % of days fall inside the nominal
  80 % band, and about 89–91 % inside the 95 % band.
- **18 months cannot teach a model a once-a-year event.** For example, a Black Friday
  peak is learned from a single occurrence.
- **The what-if is a steady-state model,** built on trailing 3-month averages. It is a
  reasoning tool, not a forecast.
- **CAC is not in the data.** It is an explicit input, and LTV:CAC and payback depend
  entirely on it.
- **FX rates are a static reference table,** for reproducibility.
- **Single-node design.** DuckDB is a single file. The dashboard is intended for one
  team, not for concurrent multi-tenant writes.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Design](docs/DESIGN.md): data contract, mapping heuristics, keys, dashboard design.
- [Technical requirements](docs/TECHNICAL-REQUIREMENTS.md): KPI formulas, ML rules and
  amendments.
- [App flow](docs/APP-FLOW.md)
- [Implementation plan](docs/IMPLEMENTATION-PLAN.md): with measured results per phase.
- [Testing guide](docs/TESTING-GUIDE.md)

## License

MIT. See [LICENSE](LICENSE).
