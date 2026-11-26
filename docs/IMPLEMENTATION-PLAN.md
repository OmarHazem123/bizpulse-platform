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

## Phase 2: Analytics & ML

| Step | Module | Exit criteria |
| --- | --- | --- |
| 2.1 | `analytics/kpis.py` | KPI formulas exactly as in TECHNICAL-REQUIREMENTS §6. Hand-computed fixtures. |
| 2.2 | `analytics/rfm.py` | quintiles, full R×F partition, playbooks. |
| 2.3 | `analytics/cohorts.py` | retention matrix and empirical LTV. |
| 2.4 | `ml/forecasting.py` | leak-free features, recursive LightGBM vs. Seasonal Naive, TimeSeriesSplit backtest, intervals. |
| 2.5 | `ml/anomaly.py` | z-score/IQR/IsolationForest, severity, root cause. Recall on injected anomalies. |
| 2.6 | `ml/whatif.py` | 6-month simulator. Zero-change scenario equals baseline. |

## Phase 3: CLI, Dashboard, Docker & README

| Step | Deliverable | Exit criteria |
| --- | --- | --- |
| 3.1 | full CLI (`smoke`, `report`, `forecast`) | `bizpulse --help` documents every command |
| 3.2 | Streamlit + Plotly dashboard, 5 tabs | `streamlit.testing.AppTest` renders every tab without exceptions |
| 3.3 | multi-stage `Dockerfile`, uid 10001, `HEALTHCHECK` | `tasks.py docker-smoke` passes |
| 3.4 | `README.md` | real benchmark and backtest numbers from this machine |

## Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| C: drive at 90 % | uv cache already on E:. Data under the repo. Slim image + `.dockerignore`. |
| `mypy --strict` with pandas | `pandas-stubs`, typed helper boundaries, narrow `cast` at library edges only. |
| DuckDB file locks on Windows | connection context managers. Tests use `tmp_path` with explicit close. |
| xlsx non-determinism (zip timestamps) | rewrite the archive with fixed timestamps and core properties. |
| Forecast leakage | recursive prediction, features shifted ≥ 1, fold-wise fitting only. |
