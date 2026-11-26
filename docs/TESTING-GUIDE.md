# Testing Guide

## 1. Commands

```bash
uv run python tasks.py test                      # full suite
uv run python tasks.py test -m "not slow"        # fast inner loop
uv run python tasks.py test --cov --cov-report=term-missing
uv run python tasks.py check                     # every static gate
```

`make test ARGS="-k mapping"` works where `make` exists.

## 2. Test pyramid

| Layer | Location | What it proves |
| --- | --- | --- |
| Unit | `tests/unit/` | pure functions: parsing, mapping, hashing, contract rules, KPI math |
| Component | `tests/component/` | a module with real I/O: readers on real files, loader on a temp DuckDB |
| Integration | `tests/integration/` (`@pytest.mark.integration`) | generator → pipeline → warehouse → reports, end to end |
| Tooling | `tests/unit/test_tasks.py` | the gates themselves (eol, hygiene and Makefile parity detect violations) |

## 3. Ground-truth testing with the demo manifest

The generator writes `manifests/<dataset>.json`:

```json
{"dirty_rows": [{"source_row": 118, "codes": ["order_date.unparseable"]}, ...],
 "anomalies": [{"name": "checkout_outage", "start": "2026-03-10", "end": "2026-03-12", ...}]}
```

The integration suite asserts:

1. **Exact quarantine**: the set of quarantined `source_row` values equals the manifest,
   and each row's reason codes equal the manifest codes. Nothing extra, nothing missing.
2. **Messy-but-valid rows load**: currency strings, mixed date formats, status casing.
3. **Reconciliation**: `read = valid + quarantined` and `valid = inserted + updated + unchanged`.
4. **Referential integrity**: every fact `customer_key`/`product_key`/`date_key` exists in its dimension.

## 4. Idempotency tests

| Scenario | Expected |
| --- | --- |
| Same file twice | second run `skipped`, warehouse unchanged |
| Same rows, different file bytes (forced) | `0 inserted / 0 updated / N unchanged` |
| One row's quantity changed | `1 updated`, all others unchanged |
| New rows appended | only the new rows are inserted |
| Re-upsert the same key (loader unit test) | no PK violation, exactly one row |

## 5. Determinism tests

- Generating twice with `seed=42` yields **byte-identical** files (CSV, XLSX, Parquet).
- A different seed gives different data.
- Window bounds are fixed (2025-05-01 → 2026-10-31) and never depend on today's date.

## 6. ML evaluation tests (Phase 2)

These criteria were registered **before** the models ran on demo data. A failure is
reported as a finding. Thresholds are never tuned to make it pass.

- **Leakage guard**: features at *t* are unchanged when y at or after *t* is perturbed.
  Forecasts are identical when future actuals are replaced by NaN or garbage. Every
  backtest fold trains strictly before its test window, and the model is fitted on
  train rows only (asserted by instrumenting `fit`).
- **Backtest sanity**: on the full-scale `ecommerce` and `retail_fb` demo data, the
  model's MAE ≤ 1.1 × Seasonal-Naive MAE. `saas` (lumpy anniversary billing) is
  reported, not asserted.
- **Intervals**: ordered `lo95 ≤ lo80 ≤ yhat ≤ hi80 ≤ hi95` with non-negative bounds.
  Leave-one-fold-out coverage of the 80 % band is within ±15 pp of nominal.
- **Anomaly recall**: every manifest anomaly with `granularity: day` yields ≥ 1 alert on
  its metric inside its window. `checkout_outage` must be `CRITICAL`. Agreement with the
  other expected severities is reported, not asserted. Root cause for slice-specific
  anomalies ranks the manifest slice first (Riverside, social, Apparel).
  **Result:** the pre-registered rule reached 4/7. Post-hoc amendment A1 reached 6/7.
  Amendment A2 (user-approved: weekly saas, expanding-window IsolationForest, severity
  calibration, slice monitoring) reaches **7/7** with correct root causes or scopes and
  **no known-failure markers**. Severity agreement is 3/7 (reported, not asserted). All
  amendments are post-hoc and are documented in TECHNICAL-REQUIREMENTS §7.1.
- **Immutability**: appending data never changes any earlier alert, including its
  severity (asserted). Quiet synthetic data never produces a CRITICAL alert (asserted).
- **What-if identity**: a zero-lever scenario reproduces the baseline exactly. The
  baseline month equals the trailing 3-month actual average within 15 %.
- **Determinism**: two forecast runs give identical numbers.

## 7. Conventions

- Tests are typed (`mypy --strict` covers `tests/`).
- Use the `tmp_path` / `settings` fixtures. Never write into the real `data/`.
- Always close DuckDB connections (context managers), because Windows file locks break
  `tmp_path` cleanup.
- Build fake secrets in tests by concatenation, so the hygiene scanner never flags them.
- Integration fixtures generate demo data at `scale=0.25` once per session for speed.
