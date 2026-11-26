"""Project task runner.

Usage::

    uv run python tasks.py <command> [extra args]
    uv run python tasks.py --help

Every command here is mirrored 1:1 by a ``Makefile`` target (verified by
``makefile-sync``), but this file is the source of truth so that the project works on
machines without ``make`` (e.g. Windows + Git Bash).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

SIZE_BUDGET_BYTES = 50 * 1024 * 1024
GENERATED_DIRS: tuple[str, ...] = ("data", "artifacts")
DOCKER_IMAGE = "bizpulse-platform:latest"
DOCKER_SMOKE_CONTAINER = "bizpulse-smoke"
DOCKER_SMOKE_PORT = 18501

BINARY_SUFFIXES = frozenset(
    {".xlsx", ".xls", ".parquet", ".duckdb", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf"}
)
CACHE_DIRS: tuple[str, ...] = (".mypy_cache", ".ruff_cache", ".pytest_cache", "htmlcov")


class TaskError(RuntimeError):
    """A task failed; the message is shown to the user."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run(cmd: Sequence[str], *, check: bool = True) -> int:
    """Run *cmd* from the repo root, streaming output."""
    print(f"$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(list(cmd), cwd=ROOT, check=False)
    if check and completed.returncode != 0:
        raise TaskError(f"command failed with exit code {completed.returncode}: {cmd[0]}")
    return completed.returncode


def capture(cmd: Sequence[str]) -> str:
    completed = subprocess.run(list(cmd), cwd=ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise TaskError(f"command failed: {' '.join(cmd)}\n{completed.stderr.strip()}")
    return completed.stdout


def project_files() -> list[Path]:
    """Tracked + untracked-but-not-ignored files (works before the first commit)."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted({p for p in out.decode("utf-8").split("\0") if p})
    return [ROOT / p for p in paths if (ROOT / p).is_file()]


def is_binary(path: Path, head: bytes) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES or b"\0" in head


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# line endings
# ---------------------------------------------------------------------------
def find_eol_violations(files: Iterable[Path]) -> list[str]:
    """Return ``path: reason`` for text files containing CR characters (working-tree bytes)."""
    problems: list[str] = []
    for path in files:
        data = path.read_bytes()
        if is_binary(path, data[:8192]):
            continue
        crlf_count = data.count(b"\r\n")
        if crlf_count:
            problems.append(f"{rel(path)}: CRLF line endings ({crlf_count} lines)")
        elif b"\r" in data:
            problems.append(f"{rel(path)}: bare CR line endings")
    return problems


def cmd_eol(_: list[str]) -> None:
    files = project_files()
    problems = find_eol_violations(files)
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    if "eol=lf" not in attributes:
        problems.append(".gitattributes: missing 'eol=lf' rule")
    autocrlf = subprocess.run(
        ["git", "config", "--get", "core.autocrlf"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if autocrlf.lower() == "true":
        problems.append(
            "git config core.autocrlf=true (run: git config --local core.autocrlf false)"
        )
    if problems:
        raise TaskError("line-ending violations:\n  " + "\n  ".join(problems))
    print(f"eol: OK ({len(files)} files are LF-only)")


# ---------------------------------------------------------------------------
# hygiene: secrets, personal emails, local machine paths
# ---------------------------------------------------------------------------
_SEP = r"[\\/]"
HYGIENE_PATTERNS: dict[str, re.Pattern[str]] = {
    # A drive letter, a colon, a path separator, then a path segment.
    "windows_abs_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:" + _SEP + r"(?=[A-Za-z0-9_.])"),
    # Git-Bash / macOS style home directories.
    "posix_user_home": re.compile(r"(?<![\w.])/(?:[a-zA-Z]/)?Users/[A-Za-z0-9_.-]+"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "api_secret_key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    "credential_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED_EMAIL_DOMAINS = frozenset(
    {"example.com", "example.org", "example.net", "users.noreply.github.com"}
)


def email_allowed(address: str) -> bool:
    local, _, domain = address.lower().partition("@")
    return (
        domain in ALLOWED_EMAIL_DOMAINS
        or domain.endswith(".example.com")
        or local.startswith("noreply")
        or local.startswith("no-reply")
    )


def scan_text(text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, rule, match)`` hygiene findings for *text*."""
    findings: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in HYGIENE_PATTERNS.items():
            for match in pattern.finditer(line):
                findings.append((line_no, rule, match.group(0)))
        for match in EMAIL_RE.finditer(line):
            if not email_allowed(match.group(0)):
                findings.append((line_no, "personal_email", match.group(0)))
    return findings


def cmd_hygiene(_: list[str]) -> None:
    files = project_files()
    problems: list[str] = []
    for path in files:
        name = path.name
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            problems.append(f"{rel(path)}: environment file must not be committed")
            continue
        data = path.read_bytes()
        if is_binary(path, data[:8192]):
            continue
        for line_no, rule, match in scan_text(data.decode("utf-8", errors="replace")):
            problems.append(f"{rel(path)}:{line_no}: {rule}: {match[:60]}")
    if problems:
        raise TaskError("hygiene violations:\n  " + "\n  ".join(problems))
    print(f"hygiene: OK ({len(files)} files scanned: no secrets, personal emails or local paths)")


# ---------------------------------------------------------------------------
# disk budget
# ---------------------------------------------------------------------------
def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cmd_size(_: list[str]) -> None:
    total = 0
    for name in GENERATED_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            size = child.stat().st_size if child.is_file() else dir_size(child)
            total += size
            print(f"  {name}/{child.name:<28} {size / 1024 / 1024:8.2f} MB")
    budget_mb = SIZE_BUDGET_BYTES / 1024 / 1024
    print(f"size: {total / 1024 / 1024:.2f} MB of generated artifacts (budget {budget_mb:.0f} MB)")
    if total > SIZE_BUDGET_BYTES:
        raise TaskError("generated artifacts exceed the 50 MB budget: run 'tasks.py clean-data'")


# ---------------------------------------------------------------------------
# Makefile <-> tasks.py parity
# ---------------------------------------------------------------------------
MAKE_TARGET_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:(?!=)", re.MULTILINE)


def makefile_problems(makefile_text: str, commands: Iterable[str]) -> list[str]:
    problems: list[str] = []
    targets = set(MAKE_TARGET_RE.findall(makefile_text)) - {"help"}
    expected = set(commands)
    for missing in sorted(expected - targets):
        problems.append(f"Makefile is missing target '{missing}'")
    for extra in sorted(targets - expected):
        problems.append(f"Makefile target '{extra}' has no tasks.py command")
    lines = makefile_text.splitlines()
    for idx, line in enumerate(lines):
        match = MAKE_TARGET_RE.match(line)
        if not match or match.group(1) == "help":
            continue
        target = match.group(1)
        recipe = lines[idx + 1] if idx + 1 < len(lines) else ""
        delegations = (f"tasks.py {target}", f"$(RUN) {target}")
        if not recipe.startswith("\t"):
            problems.append(f"target '{target}': recipe must start with a TAB")
        elif not any(recipe.rstrip().endswith(d) or f"{d} " in recipe for d in delegations):
            problems.append(f"target '{target}': recipe must delegate to tasks.py {target}")
    return problems


def cmd_makefile_sync(_: list[str]) -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    problems = makefile_problems(text, COMMANDS)
    if problems:
        raise TaskError("Makefile out of sync:\n  " + "\n  ".join(problems))
    print(f"makefile-sync: OK ({len(COMMANDS)} targets mirror tasks.py)")


# ---------------------------------------------------------------------------
# quality gates
# ---------------------------------------------------------------------------
def cmd_lint(args: list[str]) -> None:
    run([PY, "-m", "ruff", "check", ".", *args])


def cmd_format(args: list[str]) -> None:
    run([PY, "-m", "ruff", "check", "--fix", "."])
    run([PY, "-m", "ruff", "format", ".", *args])


def cmd_format_check(_: list[str]) -> None:
    run([PY, "-m", "ruff", "format", "--check", "."])


def cmd_typecheck(args: list[str]) -> None:
    run([PY, "-m", "mypy", *args])


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    detail: str = field(default="")


def cmd_check(_: list[str]) -> None:
    steps: list[tuple[str, Callable[[list[str]], None]]] = [
        ("eol", cmd_eol),
        ("hygiene", cmd_hygiene),
        ("makefile-sync", cmd_makefile_sync),
        ("lint", cmd_lint),
        ("format-check", cmd_format_check),
        ("typecheck", cmd_typecheck),
        ("size", cmd_size),
    ]
    results: list[StepResult] = []
    for name, step in steps:
        print(f"\n=== {name} ===", flush=True)
        start = time.perf_counter()
        try:
            step([])
            results.append(StepResult(name, True, time.perf_counter() - start))
        except TaskError as exc:
            print(f"FAILED: {exc}", flush=True)
            results.append(StepResult(name, False, time.perf_counter() - start, str(exc)))
    print("\n=== summary ===")
    for result in results:
        print(f"  {'PASS' if result.ok else 'FAIL'}  {result.name:<14} {result.seconds:6.1f}s")
    failed = [r.name for r in results if not r.ok]
    if failed:
        raise TaskError(f"check failed: {', '.join(failed)}")
    print("check: all gates passed")


def cmd_test(args: list[str]) -> None:
    run([PY, "-m", "pytest", *args])


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
def cmd_setup(_: list[str]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise TaskError("uv is not installed: see https://docs.astral.sh/uv/")
    run([uv, "sync", "--all-groups"])
    run(
        [
            PY,
            "-c",
            "import duckdb, lightgbm, openpyxl, pandas, plotly, pyarrow, pydantic, sklearn, "
            "streamlit, yaml; print('setup: all runtime dependencies import cleanly')",
        ]
    )


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def cmd_clean(_: list[str]) -> None:
    for name in CACHE_DIRS:
        _remove(ROOT / name)
    for path in [
        *ROOT.joinpath("src").rglob("__pycache__"),
        *ROOT.joinpath("tests").rglob("__pycache__"),
    ]:
        _remove(path)
    _remove(ROOT / ".coverage")
    print("clean: removed tool caches")


def cmd_clean_data(_: list[str]) -> None:
    for name in GENERATED_DIRS:
        _remove(ROOT / name)
    print(f"clean-data: removed {', '.join(GENERATED_DIRS)}/")


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------
def cmd_generate_demo(args: list[str]) -> None:
    run([PY, "-m", "bizpulse.cli", "generate-demo", *args])


def cmd_run_pipeline(args: list[str]) -> None:
    run([PY, "-m", "bizpulse.cli", "run-pipeline", *(args or ["--all"])])


def cmd_dashboard(args: list[str]) -> None:
    app = ROOT / "src" / "bizpulse" / "dashboard" / "app.py"
    if not app.exists():
        raise TaskError("dashboard not implemented yet (Phase 3)")
    run([PY, "-m", "streamlit", "run", str(app), *args])


def _docker() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise TaskError("docker is not installed or not on PATH")
    return docker


def cmd_docker_build(args: list[str]) -> None:
    if not (ROOT / "Dockerfile").exists():
        raise TaskError("Dockerfile not implemented yet (Phase 3)")
    run([_docker(), "build", "-t", DOCKER_IMAGE, ".", *args])


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - fixed localhost URL
            return int(response.status) == 200
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def cmd_docker_smoke(_: list[str]) -> None:
    """Build, then verify: non-root user, CLI entrypoint, pipeline run and dashboard health."""
    docker = _docker()
    cmd_docker_build([])
    uid = capture([docker, "run", "--rm", "--entrypoint", "id", DOCKER_IMAGE, "-u"]).strip()
    if uid != "10001":
        raise TaskError(f"container must run as uid 10001, got {uid!r}")
    print(f"docker-smoke: runs as non-root uid {uid}")
    run([docker, "run", "--rm", DOCKER_IMAGE, "bizpulse", "--version"])
    run([docker, "run", "--rm", DOCKER_IMAGE, "bizpulse", "smoke"])
    subprocess.run([docker, "rm", "-f", DOCKER_SMOKE_CONTAINER], capture_output=True, check=False)
    run(
        [
            docker,
            "run",
            "-d",
            "--name",
            DOCKER_SMOKE_CONTAINER,
            "-p",
            f"{DOCKER_SMOKE_PORT}:8501",
            DOCKER_IMAGE,
        ]
    )
    try:
        deadline = time.monotonic() + 180
        status = "starting"
        while time.monotonic() < deadline:
            status = capture(
                [
                    docker,
                    "inspect",
                    "--format",
                    "{{json .State.Health.Status}}",
                    DOCKER_SMOKE_CONTAINER,
                ]
            ).strip()
            if json.loads(status) == "healthy":
                break
            time.sleep(3)
        if json.loads(status) != "healthy":
            run([docker, "logs", DOCKER_SMOKE_CONTAINER], check=False)
            raise TaskError(f"container never became healthy (last status {status})")
        if not _http_ok(f"http://127.0.0.1:{DOCKER_SMOKE_PORT}/_stcore/health"):
            raise TaskError("dashboard health endpoint did not return 200 from the host")
        print("docker-smoke: container healthy and dashboard reachable")
    finally:
        subprocess.run(
            [docker, "rm", "-f", DOCKER_SMOKE_CONTAINER], capture_output=True, check=False
        )


# ---------------------------------------------------------------------------
# registry & entrypoint
# ---------------------------------------------------------------------------
COMMANDS: dict[str, tuple[Callable[[list[str]], None], str]] = {
    "setup": (cmd_setup, "Install/sync the environment with uv and verify imports"),
    "check": (cmd_check, "Run every quality gate (eol, hygiene, lint, format, mypy, size)"),
    "eol": (cmd_eol, "Verify all text files use LF line endings"),
    "hygiene": (cmd_hygiene, "Scan for secrets, personal emails and local machine paths"),
    "size": (cmd_size, "Verify generated artifacts stay under the 50 MB budget"),
    "makefile-sync": (cmd_makefile_sync, "Verify the Makefile mirrors tasks.py commands"),
    "lint": (cmd_lint, "ruff check"),
    "format": (cmd_format, "ruff autofix + format (writes files)"),
    "format-check": (cmd_format_check, "ruff format --check"),
    "typecheck": (cmd_typecheck, "mypy --strict over src/, tests/ and tasks.py"),
    "test": (cmd_test, "Run the pytest suite (extra args are forwarded)"),
    "generate-demo": (cmd_generate_demo, "Generate deterministic demo datasets (seed=42)"),
    "run-pipeline": (cmd_run_pipeline, "Ingest, validate and load datasets into DuckDB"),
    "dashboard": (cmd_dashboard, "Launch the Streamlit dashboard"),
    "docker-build": (cmd_docker_build, "Build the production Docker image"),
    "docker-smoke": (cmd_docker_smoke, "Build and smoke-test the Docker image"),
    "clean": (cmd_clean, "Remove tool caches"),
    "clean-data": (cmd_clean_data, "Remove generated data, warehouse and reports"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tasks.py",
        description="BizPulse task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n"
        + "\n".join(f"  {name:<15} {help_}" for name, (_, help_) in COMMANDS.items()),
    )
    parser.add_argument("command", choices=sorted(COMMANDS), metavar="command")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="extra arguments for the command")
    ns = parser.parse_args(argv)
    func, _ = COMMANDS[ns.command]
    try:
        func(list(ns.args))
    except TaskError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
