"""``bizpulse`` command-line interface."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bizpulse import __version__
from bizpulse.config import Settings
from bizpulse.demo.generator import DEFAULT_SEED, SIMULATORS, generate_demo_data
from bizpulse.ingestion.mapping import MappingConfig
from bizpulse.logging_utils import configure_logging
from bizpulse.pipeline import PipelineResult, run_demo_pipelines, run_pipeline
from bizpulse.warehouse.connection import connect
from bizpulse.warehouse.maintenance import compact
from bizpulse.warehouse.schema import META_RUNS


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
    # DuckDB never shrinks files and under-reports reclaimable index blocks, so compact
    # after every run that changed data (about a second; keeps the 50 MB budget).
    changed = any(r.report.rows.inserted or r.report.rows.updated for r in results)
    if changed and settings.warehouse_path.exists():
        before, after = compact(settings.warehouse_path)
        print(f"warehouse compacted: {before / 1048576:.1f} MB -> {after / 1048576:.1f} MB")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bizpulse", description="BizPulse decision intelligence platform"
    )
    parser.add_argument("--version", action="version", version=f"bizpulse {__version__}")
    parser.add_argument("--data-dir", help="data directory (default: $BIZPULSE_DATA_DIR or ./data)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING")
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
