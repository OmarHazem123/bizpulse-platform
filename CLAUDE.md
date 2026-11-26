# CLAUDE.md: bizpulse-platform engineering constitution

This file is the permanent rule set and quick reference for every Claude Code session
and handoff in this repository. The specifications in `docs/` are authoritative for
behaviour; this file is authoritative for **how work is done**. If the two disagree,
stop and ask the user. Do not silently pick one.

## Current state

Keep this section current at every phase boundary.

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repo foundation: git, LF, uv, `tasks.py` + Makefile, gates, LICENSE, specs | done |
| 1 | Ingestion, data contracts, quarantine, DuckDB star schema, demo generator | done |
| 2 | Analytics (KPIs, Pareto, RFM, cohorts / LTV) and ML (forecast, anomalies, what-if), incl. anomaly amendments A1 + A2 | done |
| 3 | Full CLI, 5-tab Streamlit dashboard, non-root Docker image, README | done; approved by the user on 2026-11-26 |

All phases are complete. Any new work starts only when the user asks for it.

The last verified gate results:
- `tasks.py check`: all 7 gates pass.
- `tasks.py test`: 253 tests pass (95 % statement + branch coverage), none marked `xfail`.
- `tasks.py docker-smoke`: passes.

Commits are local on `main`. There is no remote, so never push unless the user asks.

## 1. Project overview and mission

`bizpulse-platform` is a domain-agnostic, end-to-end **decision intelligence platform**
for startups and companies in any vertical: `ecommerce`, `saas`, `retail_fb`, or a custom
company dataset mapped onto one canonical order-line contract. It has four layers.

**1. Data engineering**
- Multi-format ingestion: CSV, `.xlsx` and Parquet.
- Column mapping and auto-detection.
- Normalization of currency, dates and types.
- A Pydantic data contract, with quarantine of invalid rows.
- Idempotent SHA-256-keyed loads into a DuckDB star schema with run lineage.
- JSON and Markdown data-quality audit reports.

**2. Data analytics / BI**
- Executive KPIs: gross / net revenue, gross margin %, net profit, orders, AOV, MoM / WoW
  growth and refund rate.
- Pareto 80/20.
- RFM segmentation with action playbooks.
- Monthly cohort retention and empirical LTV.
- Hour-of-day views in each business's local timezone.

**3. Data science / ML**
- Leak-free time-series revenue forecasting: LightGBM against a seasonal-naive baseline,
  forward `TimeSeriesSplit` backtests and 80 % / 95 % prediction intervals.
- KPI anomaly detection with CRITICAL / WARNING severities, store / channel slices and
  root-cause attribution.
- A 6-month what-if growth and unit-economics simulator.

**4. Product**
- The `bizpulse` CLI.
- A 5-tab Streamlit + Plotly dashboard.
- A multi-stage non-root Docker image.

## 2. Non-negotiable engineering rules (strict gates)

- **OS and line endings.**
  - The environment is Windows with Git Bash.
  - Every text, code and config file uses **LF**. `.gitattributes` enforces it, and
    git is configured with `core.autocrlf=false` and `core.eol=lf`.
  - `tasks.py eol` scans the working-tree bytes for CR. Never commit CRLF.
- **Environment and tooling.**
  - Use `uv` (`pyproject.toml` + `uv.lock`) with Python 3.11+. The project pins 3.12 in
    `.python-version`; the system default (3.14) must not be used.
  - Always run tools through `uv run ...`.
  - Add dependencies with `uv add`, which keeps `uv.lock` in sync.
- **Code quality and static typing.**
  - `ruff check`, `ruff format --check` and `mypy --strict` must pass with **zero
    errors** across `src/`, `tests/` and `tasks.py`.
  - Never weaken a gate to get it to pass: no blanket `type: ignore`, no new per-file
    ignores without a reason.
- **Git and licensing.**
  - The default branch is `main`. The license is MIT
    (`Copyright (c) 2026 Omar Hazem`).
  - The repo-local git identity is `user.name = "Omar Hazem"` and
    `user.email = "OmarHazem123@users.noreply.github.com"`.
  - Zero secrets, zero personal emails and zero hardcoded local machine paths: no
    drive-letter paths (a `C:`, `D:`, `E:` or `G:` prefix followed by a separator) and
    no home-directory paths.
  - `tasks.py hygiene` enforces all of this on every tracked or unignored text file. In
    tests, build fake secrets by concatenation.
- **Storage hygiene.**
  - Generated artifacts (`data/` + `artifacts/`: warehouse, demo data, reports) stay
    **under 50 MB** and are git-ignored. `tasks.py size` enforces the budget.
  - The C: drive is nearly full, so keep caches, temp files and pytest temp off it.
    The uv cache already lives off the system drive, and pytest temp goes to the
    git-ignored `.pytest_tmp/`.
  - Docker Desktop's disk image lives off the system drive, on the D: drive. It was
    moved there on 2026-11-26. Still keep images slim, and never prune Docker
    resources without asking.

## 3. Architecture and source layout

Read these docs before changing a module:

| Doc | Answers |
| --- | --- |
| `docs/ARCHITECTURE.md` | layers, package layout, pipeline data flow, runtime and deployment, key decisions |
| `docs/DESIGN.md` | canonical contract, mapping heuristics, normalization, keys, quarantine, dashboard design, demo data |
| `docs/TECHNICAL-REQUIREMENTS.md` | functional requirements, KPI formulas (§6), ML rules and the A1 / A2 amendment record (§7.1) |
| `docs/APP-FLOW.md` | personas, CLI flows, pipeline state machine, dashboard flow, error contract |
| `docs/IMPLEMENTATION-PLAN.md` | phases, exit criteria, measured results and findings |
| `docs/TESTING-GUIDE.md` | test pyramid, ground-truth manifests, pre-registered ML criteria, smoke tests |

```
src/bizpulse/
├── config.py        Settings: everything under BIZPULSE_DATA_DIR (default ./data); DEMO_FILES
├── schema.py        canonical fields, required fields, QualityIssue
├── pipeline.py      run_pipeline (run tracking, batch skip, quality gate, transactional load),
│                    run_demo_pipelines, load_demo, compact_if_changed
├── resources/       fx_rates.yaml; mappings/{ecommerce,saas,retail_fb}.yaml (shipped in the wheel)
├── ingestion/       readers (CSV sniffing, .xlsx, Parquet; .xls rejected), mapping (MappingConfig +
│                    synonym / fuzzy / content auto-detection), normalize, fx, hashing (SHA-256, key64)
├── quality/         contracts (Pydantic TransactionRecord -> reason codes), validator (dedupe,
│                    quarantine, max_quarantine_ratio=0.5 gate), report (JSON + Markdown audit)
├── warehouse/       schema (DDL: fact_transactions, dim_customers, dim_products, dim_date,
│                    meta_pipeline_runs, meta_quarantine, meta_datasets), connection, loader
│                    (staged upserts), transform, maintenance (compaction)
├── analytics/       data (load_transactions), kpis (KPIs, growth, breakdown, pareto),
│                    rfm (1-5 scores, segments, PLAYBOOKS), cohorts (retention, LTV), timeofday
├── ml/              forecasting (leak-free LightGBM vs seasonal naive, TimeSeriesSplit, intervals),
│                    anomaly (z / IQR / expanding IsolationForest, severities, slices, root causes),
│                    incidents (adjacent alerts -> severity-ranked incidents),
│                    whatif (6-month simulator, unit economics)
├── demo/            deterministic seed=42, 18-month generator (2025-05-01 -> 2026-10-31) for the
│                    3 verticals: seasonality, retention, ~1-2 % dirty rows, planted anomalies,
│                    ground-truth manifests
├── reporting.py     headless decision report (CLI `report`: text, JSON, Markdown summary)
├── smoke.py         end-to-end self-test in an isolated temp dir (CLI `smoke`, docker-smoke)
├── app/             Streamlit + Plotly: app.py (sidebar, 5 lazy tabs), state.py (cached read-only
│                    loads), charts.py (pure figures), palette.py (validated colors), ingest.py
│                    (upload helpers), views/ (one module per tab)
└── cli.py           `bizpulse` entry point
tests/   unit/ · component/ · integration/ (integration and slow markers) · conftest.py fixtures
```

Other top-level files:

- `tasks.py` is the source of truth for every command. `Makefile` mirrors it, and parity
  is a gate.
- `Dockerfile` builds the multi-stage image.

## 4. Domain and ML safety invariants (never violate)

- **Zero temporal leakage.**
  - Every lag, rolling statistic, scaling step and IsolationForest fit uses only past
    observations (`t < target_date`).
  - `features_for(history, date)` receives only the history before `date`.
  - Backtests are forward `TimeSeriesSplit` folds that fit on training rows only.
  - Analysis dates come from the data and never from the wall clock.
  - Leakage tests must keep passing.
- **Idempotency and quarantine.**
  - Re-ingesting an identical file is `skipped`. Identical rows under a new file are
    `0 inserted / 0 updated / N unchanged`. A modified row updates exactly that row.
  - Invalid rows are quarantined with explicit reason codes and audit reports. Nothing
    is dropped silently.
  - Counts reconcile: `read = valid + quarantined` and
    `valid = inserted + updated + unchanged`.
  - If more than `max_quarantine_ratio` (0.5) of the rows fail, the run fails and nothing
    loads.
- **Determinism.** Generators, splits and models propagate explicit seeds (default 42).
  Regenerating the demo data is byte-identical.
- **Immutable alert history.** Appending data never changes an earlier alert or its
  severity. Quiet data never produces a CRITICAL. Both are asserted by tests.
- **Warehouse access.**
  - Always close DuckDB connections, using the `connect()` context manager.
  - Readers (dashboard, analytics) use short-lived read-only connections, because
    compaction replaces the file.
  - Every data-changing run is followed by compaction.
- **Container security.**
  - The image is multi-stage, with minimal runtime packages (`libgomp1` only).
  - It runs as non-root uid / gid 10001 with home `/app`. The venv is root-owned and
    read-only; only `/app/data` is writable.
  - It has an explicit `HEALTHCHECK` on `/_stcore/health` and no `ENTRYPOINT`.

### Accepted decisions (approved by the user; do not re-open without a new reason)

1. **Keys.** Surrogate and join keys are 64-bit (`key64` = the first 8 bytes of
   SHA-256), and `row_hash` remains a full SHA-256 change fingerprint. This took the
   warehouse from 58 MB to 22 MB. Collision odds are about N^2 / 2^65.
2. **Quality gate.** `max_quarantine_ratio = 0.5`, a safety guard against broken
   mappings or wrong files.
3. **Timezones.** Timestamps are stored as naive UTC. Each dataset has a `timezone`
   (`retail_fb` = Europe/Berlin), used only for hour-of-day views.
4. **Anomaly amendment A1.** Two-tailed scoring: spikes on the ratio deviation, drops on
   a log-scale (`log1p`) deviation.
5. **Anomaly amendment A2.**
   - `saas` is monitored weekly.
   - IsolationForest is refit on an expanding window every 28 days / 4 weeks, with a
     past-only 1st-percentile threshold.
   - CRITICAL iff |deviation| >= 30 % and (|z| >= 5 or all three methods agree).
   - Per-store and per-channel slices are monitored.
6. **Transparency.** Every post-hoc refinement is documented, with every run recorded in
   TECHNICAL-REQUIREMENTS §7.1. Severity agreement (3/7) is reported honestly, never
   tuned away.
7. **Phase 3 defaults** (approved at the Phase 3 review on 2026-11-26).
   - The demo data is baked into the Docker image. `--demo-if-empty` bootstraps an
     empty volume.
   - Dashboard uploads persist into the shared warehouse.
   - What-if CAC defaults to 0.
   - RFM is shown as share-of-customers vs. share-of-revenue bars, not a treemap.
   - The accent color is blue in each mode, set with `theme.light` / `theme.dark`, so
     red stays reserved for CRITICAL.

Other standing decisions:
- No declared foreign keys; integrity is asserted in tests.
- When cost data is partial, profit uses an estimated COGS.
- What-if CAC is an explicit input with default 0, because it is not in the data.
- Chart colors follow `app/palette.py`: red and amber are reserved for alert
  severity, and there are no dual-axis charts.

## 5. Standard commands

```bash
uv run python tasks.py setup          # uv sync --all-groups + import check
uv run python tasks.py check          # eol, hygiene, makefile-sync, lint, format-check, typecheck, size
uv run python tasks.py test           # pytest (add -m "not slow" for the fast loop)
uv run python tasks.py generate-demo  # deterministic demo data -> data/raw (+ manifests)
uv run python tasks.py run-pipeline   # load the demo data -> data/warehouse/bizpulse.duckdb + reports
uv run python tasks.py dashboard      # Streamlit on http://localhost:8501
uv run python tasks.py docker-build   # build bizpulse-platform:latest
uv run python tasks.py docker-smoke   # build + uid 10001 + CLI + smoke + healthy + reachable
```

Single gates are also available: `eol`, `hygiene`, `size`, `makefile-sync`, `lint`,
`format` (writes files), `format-check` and `typecheck`. Cleanup commands are `clean` and
`clean-data`.

Each command has a matching `make <command>` target. `make` may not be installed, so
`tasks.py` is the primary interface.

CLI commands: `bizpulse generate-demo | run-pipeline | runs | status | report | forecast | whatif | smoke | dashboard`.
Run `bizpulse <command> --help` for the options.

## 6. Workflow and verification protocol

1. **Orient.** Read the relevant `docs/` sections and the code you will touch. Check
   `git status` / `git log`. At session start, resume from the newest handoff if one
   exists.
2. **Respect phase pauses.** The user reviews at every phase boundary. Present any
   decision with trade-offs and wait for approval. Do not start the next phase
   unprompted.
3. **Implement.** Make small typed functions around thin I/O. Keep specs in sync: update
   `docs/` in the same change whenever behaviour, commands or decisions change.
4. **Verify before every commit.**
   - Run `uv run python tasks.py check` and `uv run python tasks.py test`. Both must be
     green.
   - Also run `uv run python tasks.py docker-smoke` when you touch the Dockerfile,
     dependencies, the CLI or the dashboard.
   - Report failures honestly, with their output.
5. **Commit.** Keep commits clean and atomic on `main`, with conventional messages
   (`feat(phase-N): ...`, `fix(...)`, `docs(...)`, `test(...)`, `chore: ...`). Never
   use `--no-verify`, and never push without being asked.
6. **Hand off.** When context grows large, write a handoff (`.claude/handoffs/`, which is
   git-ignored) and update the project memory.

## Gotchas (learned the hard way)

**pandas, numpy and typing**
- Write `pd.Timedelta(n, unit="D")`, never `pd.Timedelta(days=n)`. The latter emits a
  NumPy DeprecationWarning, and the pytest config turns that into an error.
- `mypy --strict` with pandas-stubs:
  - Prefer `np.asarray(s.to_numpy(), dtype=np.float64)`.
  - Read `.loc` scalars via `float(series[key])`.
  - Narrow `Optional` values with `if x is None: raise`. Check helpers like
    `_require(...)` do not narrow the type for mypy.
- `ruff --fix` can rewrite code. Run `ruff format` after edits.

**DuckDB**
- `ATTACH` cannot take bound parameters; use `maintenance._literal()`.
- Windows holds file locks while a connection is open, so close everything before
  `tmp_path` cleanup or compaction.

**pytest**
- pytest **wipes `.pytest_tmp/` at session start**. Never write logs or anything you
  want to keep there.
- The `demo_settings` fixture is a shared read-only warehouse at scale 0.1.
- `test_ml_demo.py` needs full scale; it is marked `slow`.

**Streamlit tests**
- Tabs use `st.tabs(..., key="nav", on_change="rerun")`.
- In `AppTest`, switch tabs with `at.session_state["nav"] = "<tab label>"`, then call
  `at.run()`.
- Use `default_timeout` of 120 s or more for the ML tab.

**Editing and repo hygiene**
- Editing tools can mangle literal non-ASCII characters (NBSP, BOM, symbols, emoji). In
  source code, use `\uXXXX` escapes. When patching files with non-ASCII content, use
  ASCII-only Python scripts that build those characters with `chr()`.
- Never write real drive-letter or home-directory paths in tracked files, docs
  included. The hygiene gate will fail.

**Demo data**
- The demo window is fixed. The SaaS export ends 2026-10-28, so October is
  "incomplete" for `saas` by design.
