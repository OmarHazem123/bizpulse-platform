# Implementation Plan

Each phase ends with a **gate**: `tasks.py check` (eol, hygiene, makefile-sync, ruff
lint, ruff format, mypy --strict, size) and `tasks.py test` are green, followed by a
review pause.

## Phase 0: Foundation ✅

- [x] git `main`, repo-local author, `core.autocrlf=false`, `.gitattributes` LF rules
- [x] `pyproject.toml` (uv, hatchling, ruff, mypy strict, pytest), `uv.lock`, Python 3.12 pin
- [x] `tasks.py` + mirrored `Makefile`, with parity enforced by `makefile-sync`
- [x] Gates: eol (working-tree bytes), hygiene (secrets/emails/paths), size (< 50 MB)
- [x] MIT `LICENSE`, `.gitignore`, `.dockerignore`, `.editorconfig`
- [x] Specification docs (this folder)

## Phase 1: Ingestion, Quality & DuckDB Star Schema ✅

All steps below are done and gated (`tasks.py check` + `tasks.py test` green).

Findings and decisions made during the build:

- **Timezone leak in pandas ISO parsing.** A `+02:00` value shifted later naive values
  by two hours. Fixed by parsing offset-aware and naive strings separately, with a
  regression test.
- **Warehouse over budget (58 MB).** Measurement showed PK indexes on 64-char hex keys
  plus free blocks were the cause. The fix was BIGINT `key64` keys plus compaction after
  data-changing runs, giving 21.8 MB.
- **Loose currency-code stripping** made `ELE-001` parse as −1. Stripping is now
  restricted to known ISO-4217 codes.
- **Duplicate CSV headers** were silently renamed by pandas. They are now detected on
  the raw header line.

| Step | Module | Exit criteria |
| --- | --- | --- |
| 1.1 | `config.py`, `logging_utils.py`, `resources/` | settings resolve from `BIZPULSE_DATA_DIR`. FX + mapping YAML load and validate. |
| 1.2 | `ingestion/hashing.py` | stable SHA-256 keys. Float/format-insensitive row hash. |
| 1.3 | `ingestion/readers.py` | CSV (sniffed), Excel, Parquet all read to a string-typed frame. |
| 1.4 | `ingestion/mapping.py` | explicit / synonym / fuzzy / content detection. No double-claims. Ambiguity reported. |
| 1.5 | `ingestion/normalize.py` | amounts, currencies, dates, ids, statuses, with reason codes. |
| 1.6 | `quality/contracts.py`, `quality/validator.py` | Pydantic contract and dedupe. One defect gives one reason. |
| 1.7 | `quality/report.py` | JSON + Markdown audit reports. |
| 1.8 | `warehouse/*` | DDL, staged upserts, insert/update/unchanged counts. Connections always closed. |
| 1.9 | `pipeline.py` | run tracking, batch skip, quality gate, transactional load. |
| 1.10 | `demo/*` | deterministic 18-month generator for 3 verticals + manifests. Byte-identical reruns. |
| 1.11 | `cli.py` (minimal) | `generate-demo`, `run-pipeline`, `runs`. |
| 1.12 | tests | unit tests per module. Integration test: quarantine set == manifest. Idempotency (rerun → 0/0/N, mutate one row → 1 updated). |

## Phase 2: Analytics & ML ✅

Gate: `tasks.py check` green, with 211 tests passing (no `xfail`) in about 95 s.

Measured on the full-scale demo data (backtests: 6 × 30-day `TimeSeriesSplit` folds):

| Dataset | Model MAE | Seasonal-naive MAE | Ratio | 80 % coverage (LOFO) | 95 % coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| ecommerce (USD) | 876 | 922 | **0.95** | 73.9 % | 91.1 % |
| retail_fb (EUR) | 85.7 | 104.5 | **0.82** | 71.7 % | 89.4 % |
| saas (USD) | 5,763 | 7,835 | 0.74 | 72.8 % | 90.0 % |

Findings and decisions:

- **Pre-registered forecast criteria passed on the first run.** No tuning was done.
  The intervals are slightly narrow (about 72–74 % vs. the nominal 80 %).
- **The anomaly rule needed two post-hoc amendments, both accepted by the user.**
  A1 adds two-tailed scoring. A2 adds weekly saas monitoring, an expanding-window
  IsolationForest so severities are immutable, CRITICAL calibration, and per-store and
  per-channel slice monitoring. Recall: 4/7 pre-registered, 6/7 with A1, **7/7 with A2**.
  Severity agreement is 3/7. The full record is in TECHNICAL-REQUIREMENTS §7.1.
- **The what-if baseline reproduces trailing actuals exactly**, and zero levers give the
  baseline exactly. retail_fb carries anonymous guest revenue (~59 %) as its own stream.
- **Time of day:** retail_fb peaks at 8:00 and 13:00 local time (Europe/Berlin), while the
  UTC storage would suggest 6:00 and 11:00.

| Step | Module | Exit criteria |
| --- | --- | --- |
| 2.1 | `analytics/kpis.py` | KPI formulas exactly as in TECHNICAL-REQUIREMENTS §6. Hand-computed fixtures. |
| 2.2 | `analytics/rfm.py` | quintiles, full R×F partition, playbooks. |
| 2.3 | `analytics/cohorts.py` | retention matrix and empirical LTV. |
| 2.4 | `ml/forecasting.py` | leak-free features, recursive LightGBM vs. Seasonal Naive, TimeSeriesSplit backtest, intervals. |
| 2.5 | `ml/anomaly.py` | z-score/IQR/IsolationForest, severity, root cause. Recall on injected anomalies. |
| 2.6 | `ml/whatif.py` | 6-month simulator. Zero-change scenario equals baseline. |

## Phase 3: CLI, Dashboard, Docker & README ✅

Gate: `tasks.py check` green, `tasks.py test` green (253 tests, no `xfail`, 95 % statement + branch coverage) and
`tasks.py docker-smoke` green.

| Step | Deliverable | Exit criteria | Result |
| --- | --- | --- | --- |
| 3.1 | full CLI: `report`, `forecast`, `whatif`, `smoke`, `dashboard` (plus the Phase 1 data commands) | `bizpulse --help` documents every command | asserted by `test_help_documents_every_command` |
| 3.2 | Streamlit + Plotly dashboard, 5 tabs | `streamlit.testing.AppTest` renders every tab without exceptions | every tab × 3 demo datasets, empty state and the upload flow are tested; screenshots reviewed in light and dark mode |
| 3.3 | multi-stage `Dockerfile`, uid 10001, `HEALTHCHECK` | `tasks.py docker-smoke` passes | healthy about 12 s after start; image 1.11 GB (venv 659 MB, of which pyarrow 166 MB) |
| 3.4 | `README.md` | real benchmark and backtest numbers from this machine | done |

Findings and decisions:

- **A real bug found by the dashboard tests.** `cohort_analysis` crashed on NaN when an
  upload covered only part of one month, so no cohort was complete yet. It now returns
  the empty result, with a regression test.
- **Lazy tabs.** `st.tabs(on_change="rerun")` runs only the open tab. Cold ML costs per
  dataset are a forecast of about 1.3 s and anomaly scoring of 2.5–6 s. They are cached
  on `(warehouse, mtime, dataset)`.
- **Theme.**
  - Setting a global `theme.primaryColor` turned off Streamlit's automatic light/dark
    switching, so the blue accent is set per mode with `theme.light` / `theme.dark`.
  - The red default accent was replaced because it collided with the CRITICAL status
    color.
- **CAC defaults to 0 in the simulator and the CLI.** Acquisition cost is not in the
  data. An invented default made the status quo's contribution negative.
- **Second pass against the final Phase 3 deliverable list.**
  - The dashboard package moved to `src/bizpulse/app/`.
  - New CLI command: `status`. New report option: `--markdown` (executive summary). New
    task: `tasks.py report`.
  - Tab 1 gained Gross Revenue and WoW growth tiles, a revenue-and-profit trend, Pareto
    by product or category, and an executive-summary download.
  - Tab 3 gained store/channel and company-wide filters, plus a severity-ranked
    **incident feed**: `ml/incidents.py` groups alerts on adjacent periods. In retail_fb,
    55 alerts become 33 incidents.
  - Tab 5 now shows the full quarantined-rows table with the original values.
  - Missing optional columns are labelled. A single-valued dimension is no longer
    monitored as a slice, because its alerts duplicated the total's. Demo results are
    unchanged, since the demo data has no missing dimensions.
  - `bizpulse smoke` (and therefore `docker-smoke`) now checks the 50 % quality gate and
    a hostile file: NaN, inf, 1e400, impossible dates, negatives, blanks and no optional
    columns.
- **Container.**
  - The demo warehouse is built into the image (about 30 MB) for an instant, deterministic
    first start.
  - `--demo-if-empty` covers a fresh volume mounted over `/app/data`.
  - The venv is root-owned and read-only. Only `/app/data` belongs to uid 10001.

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| C: drive at 90 % | uv cache already on E:. Data under the repo. Slim image + `.dockerignore`. |
| `mypy --strict` with pandas | `pandas-stubs`, typed helper boundaries, narrow `cast` at library edges only. |
| DuckDB file locks on Windows | connection context managers. Tests use `tmp_path` with explicit close. |
| xlsx non-determinism (zip timestamps) | rewrite the archive with fixed timestamps and core properties. |
| Forecast leakage | recursive prediction, features shifted ≥ 1, fold-wise fitting only. |
