# BizPulse Platform

Domain-agnostic **Business & Startup Decision Intelligence Platform** unifying data
engineering, BI analytics and machine learning.

> Work in progress: see [docs/IMPLEMENTATION-PLAN.md](docs/IMPLEMENTATION-PLAN.md) for the
> phased roadmap. The full README with benchmarks lands in Phase 3.

## Quick start

```bash
uv run python tasks.py setup          # sync the environment, verify imports
uv run python tasks.py generate-demo  # 3 deterministic datasets (CSV, Excel, Parquet)
uv run python tasks.py run-pipeline   # ingest -> validate -> quarantine -> DuckDB star schema
uv run python tasks.py check          # eol, hygiene, ruff, mypy --strict, size budget
uv run python tasks.py test           # pytest
```

Load your own file with column auto-detection:

```bash
uv run bizpulse run-pipeline --file my_orders.csv --dataset acme
uv run bizpulse run-pipeline --file my_orders.xlsx --mapping my_mapping.yaml
uv run bizpulse runs
```

Each run writes a data-quality audit report to `data/reports/<dataset>/<run_id>.{json,md}`.

## License

MIT, see [LICENSE](LICENSE).
