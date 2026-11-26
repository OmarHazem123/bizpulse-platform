"""End-to-end self-test (``bizpulse smoke``, used by ``tasks.py docker-smoke``).

Runs every layer on small demo data inside an isolated work directory (never the
configured ``BIZPULSE_DATA_DIR``): generate -> ingest/validate/load (CSV, Excel,
Parquet) -> idempotent re-run -> compaction -> BI + forecast + anomalies -> what-if.
Checks are structural (they hold for any seed/scale), not tuned numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from bizpulse.analytics.data import load_transactions
from bizpulse.config import Settings
from bizpulse.demo.generator import DEFAULT_SEED, generate_demo_data
from bizpulse.ml.anomaly import granularity_for
from bizpulse.ml.forecasting import HORIZON
from bizpulse.ml.whatif import Levers, compute_baseline, simulate
from bizpulse.pipeline import SUCCESS_STATUSES, compact_if_changed, run_demo_pipelines
from bizpulse.reporting import build_report

DEFAULT_SMOKE_SCALE = 0.1


class SmokeError(RuntimeError):
    """A smoke check failed."""


@dataclass
class SmokeResult:
    checks: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def run_smoke(
    work_dir: Path,
    *,
    scale: float = DEFAULT_SMOKE_SCALE,
    seed: int = DEFAULT_SEED,
    echo: Callable[[str], None] | None = None,
) -> SmokeResult:
    """Run the self-test in *work_dir*; raise :class:`SmokeError` on the first failure."""
    result = SmokeResult()
    started = time.perf_counter()

    def ok(message: str) -> None:
        result.checks.append(message)
        if echo is not None:
            echo(f"[ok] {message}")

    settings = Settings(data_dir=work_dir.resolve())
    settings.ensure_dirs()
    files = generate_demo_data(settings.raw_dir, seed=seed, scale=scale)
    expected_dirty = {f.dataset: f.dirty_rows for f in files}
    ok(f"generated {len(files)} demo datasets (CSV, Excel, Parquet) at scale {scale}")

    runs = run_demo_pipelines(settings)
    for run in runs:
        rows = run.report.rows
        _require(run.status in SUCCESS_STATUSES, f"{run.dataset}: pipeline {run.status}")
        _require(rows.read == rows.valid + rows.quarantined, f"{run.dataset}: rows don't add up")
        _require(
            rows.quarantined == expected_dirty[run.dataset],
            f"{run.dataset}: quarantined {rows.quarantined}, manifest has "
            f"{expected_dirty[run.dataset]} dirty rows",
        )
        _require(run.md_report.is_file(), f"{run.dataset}: audit report missing")
    ok("pipelines loaded; quarantine matches the manifest exactly; audit reports written")

    reruns = run_demo_pipelines(settings)
    _require(all(r.status == "skipped" for r in reruns), "re-running identical files must skip")
    ok("identical re-run skipped (idempotent batches)")

    compacted = compact_if_changed(runs, settings)
    if compacted is None:
        raise SmokeError("warehouse was not compacted after loading")
    ok(f"warehouse compacted to {compacted[1] / 1048576:.1f} MB")

    for run in runs:
        info, tx = load_transactions(settings.warehouse_path, run.dataset)
        report = build_report(info, tx)
        kpis = report["kpis"]["all_time"]
        _require(kpis["net_revenue"] > 0, f"{info.dataset}: net revenue must be positive")
        _require(kpis["orders"] > 0, f"{info.dataset}: no orders")
        daily = report["forecast"]["daily"]
        _require(len(daily) == HORIZON, f"{info.dataset}: forecast must have {HORIZON} days")
        for row in daily:
            bands = [row["lo95"], row["lo80"], row["yhat"], row["hi80"], row["hi95"]]
            _require(
                bands[0] >= 0.0 and bands == sorted(bands),
                f"{info.dataset}: forecast bands not ordered on {row['date']}",
            )
        _require(report["forecast"]["backtest"]["mae_ratio"] is not None, "backtest failed")
        granularity = report["anomalies"]["granularity"]
        _require(
            granularity == granularity_for(info.domain),
            f"{info.dataset}: monitored at {granularity}",
        )

        whatif = simulate(compute_baseline(tx), Levers())
        change = whatif.totals()["change"].abs().max()
        _require(float(change) < 1e-6, f"{info.dataset}: zero-lever what-if must equal baseline")
        ok(
            f"{info.dataset}: KPIs, Pareto, RFM, cohorts, {HORIZON}-day forecast, "
            f"{report['anomalies']['alerts_total']} anomaly alerts ({granularity}), what-if"
        )

    result.seconds = time.perf_counter() - started
    return result
