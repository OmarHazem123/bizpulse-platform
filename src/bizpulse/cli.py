"""``bizpulse`` command-line interface.

Data commands (``generate-demo``, ``run-pipeline``, ``runs``), analysis commands
(``report``, ``forecast``, ``whatif``), the end-to-end ``smoke`` test and ``dashboard``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from bizpulse import __version__
from bizpulse.analytics.data import DatasetInfo, list_datasets, load_transactions
from bizpulse.config import ENV_DATA_DIR, Settings
from bizpulse.demo.generator import DEFAULT_SEED, SIMULATORS, generate_demo_data
from bizpulse.ingestion.mapping import MappingConfig
from bizpulse.logging_utils import configure_logging
from bizpulse.ml.forecasting import HORIZON, daily_revenue, forecast_series
from bizpulse.ml.whatif import MONTHS, Levers, compute_baseline, simulate
from bizpulse.pipeline import (
    PipelineResult,
    compact_if_changed,
    load_demo,
    run_demo_pipelines,
    run_pipeline,
)
from bizpulse.reporting import build_report, render_text
from bizpulse.smoke import DEFAULT_SMOKE_SCALE, SmokeError, run_smoke
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.schema import META_RUNS

DASHBOARD_APP = Path(__file__).resolve().parent / "dashboard" / "app.py"


def _print_result(result: PipelineResult, settings: Settings) -> None:
    rows = result.report.rows
    rate = result.report.pass_rate
    print(
        f"{result.dataset:<12} {result.status:<22} read={rows.read:>7,} valid={rows.valid:>7,} "
        f"quarantined={rows.quarantined:>5,} inserted={rows.inserted:>7,} "
        f"updated={rows.updated:>5,} unchanged={rows.unchanged:>7,} "
        f"pass={'n/a' if rate is None else f'{rate:.2%}'}"
    )
    if result.report.error:
        print(f"  error: {result.report.error}")
    if result.report.skipped_reason:
        print(f"  skipped: {result.report.skipped_reason}")
    print(f"  report: {settings.relative(result.md_report)}")


def _compact(results: Sequence[PipelineResult], settings: Settings) -> None:
    try:
        sizes = compact_if_changed(results, settings)
    except (OSError, duckdb.Error) as exc:  # e.g. another process holds a Windows lock
        print(f"warning: warehouse not compacted ({exc}); data is intact", file=sys.stderr)
        return
    if sizes is not None:
        before, after = sizes
        print(f"warehouse compacted: {before / 1048576:.1f} MB -> {after / 1048576:.1f} MB")


def _datasets(settings: Settings, wanted: str | None) -> list[DatasetInfo] | None:
    """Datasets to analyse, or ``None`` (after printing why) when there are none."""
    available = list_datasets(settings.warehouse_path)
    if not available:
        print(
            "no data yet: run `bizpulse generate-demo` and `bizpulse run-pipeline --all`, "
            "or load a file with `bizpulse run-pipeline --file`",
            file=sys.stderr,
        )
        return None
    if wanted is None:
        return available
    match = [info for info in available if info.dataset == wanted]
    if not match:
        names = ", ".join(info.dataset for info in available)
        print(f"error: unknown dataset {wanted!r}; available: {names}", file=sys.stderr)
        return None
    return match


# --------------------------------------------------------------------------- data
def cmd_generate_demo(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    out_dir = Path(args.out) if args.out else settings.raw_dir
    start = time.perf_counter()
    files = generate_demo_data(out_dir, seed=args.seed, scale=args.scale, datasets=args.datasets)
    for item in files:
        size_mb = item.path.stat().st_size / 1024 / 1024
        print(
            f"{item.dataset:<12} {item.rows:>7,} rows ({item.dirty_rows:,} dirty) "
            f"{size_mb:6.2f} MB  {item.path.name}"
        )
    print(f"generated {len(files)} datasets in {time.perf_counter() - start:.1f}s")
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    if args.all:
        try:
            results = run_demo_pipelines(settings, force=args.force)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.file:
            print("error: pass --all or --file PATH", file=sys.stderr)
            return 2
        source = Path(args.file)
        if args.mapping:
            config = MappingConfig.from_file(Path(args.mapping))
        elif args.dataset:
            config = MappingConfig(dataset=args.dataset)
        else:
            print("error: --file needs --mapping or --dataset", file=sys.stderr)
            return 2
        results = [run_pipeline(source, config, settings, force=args.force)]
    for result in results:
        _print_result(result, settings)
    _compact(results, settings)
    return 0 if all(r.ok for r in results) else 1


def cmd_runs(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    if not settings.warehouse_path.exists():
        print("no warehouse yet: run `bizpulse run-pipeline --all`")
        return 0
    where = "WHERE dataset = ?" if args.dataset else ""
    params = [args.dataset] if args.dataset else []
    with connect(settings.warehouse_path, read_only=True) as con:
        rows = con.execute(
            f"""SELECT started_at, dataset, status, rows_read, rows_quarantined, rows_inserted,
                       rows_updated, rows_unchanged, duration_seconds, run_id
                FROM {META_RUNS} {where} ORDER BY started_at DESC LIMIT ?""",
            [*params, args.limit],
        ).fetchall()
    print(
        f"{'started (UTC)':<20} {'dataset':<12} {'status':<22} {'read':>7} {'quar.':>6} "
        f"{'ins.':>7} {'upd.':>5} {'same':>7} {'secs':>6}  run_id"
    )
    for r in rows:
        print(
            f"{r[0]:%Y-%m-%d %H:%M:%S}  {r[1]:<12} {r[2]:<22} {r[3] or 0:>7,} {r[4] or 0:>6,} "
            f"{r[5] or 0:>7,} {r[6] or 0:>5,} {r[7] or 0:>7,} {r[8] or 0:>6.2f}  {r[9]}"
        )
    return 0


# --------------------------------------------------------------------------- analysis
def cmd_report(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    infos = _datasets(settings, args.dataset)
    if infos is None:
        return 1
    reports: dict[str, object] = {}
    for info in infos:
        _, tx = load_transactions(settings.warehouse_path, info.dataset)
        report = build_report(
            info,
            tx,
            include_forecast=not args.no_forecast,
            include_anomalies=not args.no_anomalies,
        )
        reports[info.dataset] = report
        if len(reports) > 1:
            print("\n" + "=" * 100 + "\n")
        print(render_text(report))
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = reports[infos[0].dataset] if args.dataset else reports
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report: {target.name}")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    infos = _datasets(settings, args.dataset)
    if infos is None:
        return 1
    info = infos[0]
    _, tx = load_transactions(settings.warehouse_path, info.dataset)
    try:
        result = forecast_series(daily_revenue(tx), horizon=args.horizon)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    bt = result.backtest
    print(f"{info.display_name}: net revenue forecast ({info.currency}/day)")
    print(
        f"backtest: {len(bt.folds)} TimeSeriesSplit folds x {args.horizon} days | "
        f"MAE ratio vs seasonal naive {bt.mae_ratio:.3f} | interval coverage "
        f"80% {bt.coverage80 or 0:.1%}, 95% {bt.coverage95 or 0:.1%}"
    )
    print(f"{'metric':<8} {'LightGBM':>12} {'naive':>12}")
    for name in ("mae", "rmse", "mape", "smape"):
        model, base = bt.model[name], bt.baseline[name]
        print(
            f"{name:<8} {'n/a' if model is None else f'{model:,.2f}':>12} "
            f"{'n/a' if base is None else f'{base:,.2f}':>12}"
        )
    print()
    print(f"{'date':<11} {'yhat':>11} {'80% band':>23} {'95% band':>23} {'naive':>11}")
    for day, row in result.forecast.iterrows():
        print(
            f"{pd.Timestamp(str(day)):%Y-%m-%d} {row['yhat']:>11,.0f} "
            f"{row['lo80']:>11,.0f}-{row['hi80']:<11,.0f} "
            f"{row['lo95']:>11,.0f}-{row['hi95']:<11,.0f} {row['baseline']:>11,.0f}"
        )
    if args.csv:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.forecast.round(2).to_csv(target, lineterminator="\n")
        print(f"\nforecast CSV: {target.name}")
    return 0


def cmd_whatif(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    infos = _datasets(settings, args.dataset)
    if infos is None:
        return 1
    info = infos[0]
    _, tx = load_transactions(settings.warehouse_path, info.dataset)
    levers = Levers(
        retention_pp=args.retention_pp / 100,
        aov_pct=args.aov_pct / 100,
        discount_pp=args.discount_pp / 100,
        new_customers_pct=args.new_customers_pct / 100,
        frequency_pct=args.frequency_pct / 100,
        cac=args.cac,
    )
    try:
        baseline = compute_baseline(tx)
        result = simulate(baseline, levers, months=args.months)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cur = info.currency
    print(
        f"{info.display_name}: baseline = average of {', '.join(baseline.months)} | "
        f"retention {baseline.retention:.1%}, {baseline.new_customers:,.0f} new and "
        f"{baseline.active_customers:,.0f} active customers/month"
    )
    print(f"\n{'month':>5} {'baseline net':>14} {'scenario net':>14} {'contribution':>14}")
    for month in result.scenario_projection.index:
        base = result.baseline_projection.loc[month]
        scen = result.scenario_projection.loc[month]
        print(
            f"{month:>5} {base['net_revenue']:>14,.0f} {scen['net_revenue']:>14,.0f} "
            f"{scen['contribution']:>14,.0f}"
        )
    print(f"\n{args.months}-month totals ({cur})")
    for name, row in result.totals().iterrows():
        pct = row["change_pct"]
        print(
            f"  {name!s:<13} {row['baseline']:>14,.0f} -> {row['scenario']:>14,.0f} "
            f"({'n/a' if pd.isna(pct) else f'{pct:+.1%}'})"
        )
    print("\nunit economics          baseline     scenario")
    for label, attr in (
        ("margin/customer-month", "margin_per_customer_month"),
        ("lifetime (months)", "expected_lifetime_months"),
        ("LTV", "ltv"),
        ("LTV:CAC", "ltv_to_cac"),
        ("payback (months)", "payback_months"),
    ):
        base_value = getattr(result.baseline_units, attr)
        scen_value = getattr(result.scenario_units, attr)
        print(
            f"  {label:<21} {'n/a' if base_value is None else f'{base_value:,.2f}':>10} "
            f"{'n/a' if scen_value is None else f'{scen_value:,.2f}':>12}"
        )
    if args.cac == 0:
        print("  (pass --cac to see LTV:CAC and payback; CAC is not in the data)")
    return 0


# --------------------------------------------------------------------------- ops
def cmd_smoke(args: argparse.Namespace) -> int:
    print(f"bizpulse {__version__} smoke test (scale {args.scale})")
    try:
        if args.work_dir:
            result = run_smoke(Path(args.work_dir), scale=args.scale, echo=print)
        else:
            # Isolated from BIZPULSE_DATA_DIR on purpose: never touches real data.
            with tempfile.TemporaryDirectory(
                prefix="bizpulse-smoke-", ignore_cleanup_errors=True
            ) as tmp:
                result = run_smoke(Path(tmp), scale=args.scale, echo=print)
    except SmokeError as exc:
        print(f"smoke: FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"smoke: OK ({len(result.checks)} checks in {result.seconds:.1f}s)")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.data_dir)
    if args.demo_if_empty and not list_datasets(settings.warehouse_path):
        print("no data found: generating and loading the demo datasets (~40 s) ...")
        for result in load_demo(settings):
            _print_result(result, settings)
    os.environ[ENV_DATA_DIR] = str(settings.data_dir)  # the app resolves the same directory
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(DASHBOARD_APP),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        str(args.headless).lower(),
        "--browser.gatherUsageStats",
        "false",
        # Blue accents per mode (light/dark switching stays automatic): Streamlit's default
        # red would collide with the CRITICAL status color.
        "--theme.light.primaryColor",
        "#257de0",
        "--theme.dark.primaryColor",
        "#338ced",
    ]
    streamlit_cli.main()  # click entry point: runs the server and exits the process
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bizpulse",
        description="BizPulse decision intelligence platform: ingestion and data contracts, "
        "a DuckDB star schema, BI analytics, forecasting, anomaly detection and what-if "
        "simulation.",
    )
    parser.add_argument("--version", action="version", version=f"bizpulse {__version__}")
    parser.add_argument("--data-dir", help="data directory (default: $BIZPULSE_DATA_DIR or ./data)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    gen = sub.add_parser("generate-demo", help="generate deterministic demo datasets")
    gen.add_argument("--out", help="output directory (default: <data-dir>/raw)")
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gen.add_argument("--scale", type=float, default=1.0, help="volume multiplier (default 1.0)")
    gen.add_argument("--datasets", nargs="+", choices=sorted(SIMULATORS))
    gen.set_defaults(func=cmd_generate_demo)

    pipe = sub.add_parser("run-pipeline", help="ingest, validate and load data into DuckDB")
    pipe.add_argument("--all", action="store_true", help="load the three demo datasets")
    pipe.add_argument("--file", help="CSV / Excel / Parquet file to ingest")
    pipe.add_argument("--mapping", help="YAML/JSON mapping config for --file")
    pipe.add_argument("--dataset", help="dataset name for --file (auto-detect columns)")
    pipe.add_argument("--force", action="store_true", help="reload even if already loaded")
    pipe.set_defaults(func=cmd_run_pipeline)

    runs = sub.add_parser("runs", help="show pipeline run history")
    runs.add_argument("--dataset")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    rep = sub.add_parser(
        "report", help="KPIs, Pareto, RFM, cohorts/LTV, forecast and anomalies per dataset"
    )
    rep.add_argument("--dataset", help="one dataset (default: every dataset)")
    rep.add_argument("--json", help="also write the report as JSON to this path")
    rep.add_argument("--no-forecast", action="store_true", help="skip the forecast backtest")
    rep.add_argument("--no-anomalies", action="store_true", help="skip anomaly detection")
    rep.set_defaults(func=cmd_report)

    fc = sub.add_parser(
        "forecast", help="daily net-revenue forecast with 80/95%% intervals and backtest"
    )
    fc.add_argument("--dataset", required=True)
    fc.add_argument("--horizon", type=int, default=HORIZON, help=f"days (default {HORIZON})")
    fc.add_argument("--csv", help="write the forecast table to this CSV path")
    fc.set_defaults(func=cmd_forecast)

    wi = sub.add_parser(
        "whatif", help=f"{MONTHS}-month growth and unit-economics scenario vs status quo"
    )
    wi.add_argument("--dataset", required=True)
    wi.add_argument("--retention-pp", type=float, default=0.0, help="retention change, pp")
    wi.add_argument("--aov-pct", type=float, default=0.0, help="average order value change, %%")
    wi.add_argument("--discount-pp", type=float, default=0.0, help="discount rate change, pp")
    wi.add_argument(
        "--new-customers-pct", type=float, default=0.0, help="acquisition volume change, %%"
    )
    wi.add_argument("--frequency-pct", type=float, default=0.0, help="order frequency change, %%")
    wi.add_argument("--cac", type=float, default=0.0, help="acquisition cost per new customer")
    wi.add_argument("--months", type=int, default=MONTHS)
    wi.set_defaults(func=cmd_whatif)

    smoke = sub.add_parser(
        "smoke", help="end-to-end self-test on small demo data in an isolated temp directory"
    )
    smoke.add_argument("--scale", type=float, default=DEFAULT_SMOKE_SCALE)
    smoke.add_argument("--work-dir", help="use this (empty) directory instead of a temp dir")
    smoke.set_defaults(func=cmd_smoke)

    dash = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    dash.add_argument("--port", type=int, default=8501)
    dash.add_argument("--address", default="localhost", help="bind address (0.0.0.0 in Docker)")
    dash.add_argument("--headless", action="store_true", help="don't open a browser")
    dash.add_argument(
        "--demo-if-empty",
        action="store_true",
        help="generate and load the demo datasets first when the warehouse has no data",
    )
    dash.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING")
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
