"""The quality gates themselves must detect violations (not just pass on a clean tree)."""

from __future__ import annotations

from pathlib import Path

import pytest

import tasks


def test_eol_detects_crlf_and_bare_cr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    good = tmp_path / "good.py"
    crlf = tmp_path / "crlf.py"
    bare = tmp_path / "bare.md"
    binary = tmp_path / "data.parquet"
    good.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    bare.write_bytes(b"line\rline\r")
    binary.write_bytes(b"\r\n\x00\x01")
    problems = tasks.find_eol_violations([good, crlf, bare, binary])
    assert problems == ["crlf.py: CRLF line endings (2 lines)", "bare.md: bare CR line endings"]


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("path = 'D" + ":" + "\\" + "Users\\someone\\file.txt'", "windows_abs_path"),
        ("open('C" + ":" + "/temp/out.csv')", "windows_abs_path"),
        ("cd /c/Us" + "ers/someone/project", "posix_user_home"),
        ("key = 'AKIA" + "ABCDEFGHIJKLMNOP'", "aws_access_key"),
        ("-----BEGIN RSA " + "PRIVATE KEY-----", "private_key"),
        ("api_key = '" + "s3cr3tvalue1234'", "credential_assignment"),
        ("token: 'gh" + "p_" + "a" * 36 + "'", "github_token"),
        ("contact " + "jane.doe" + "@" + "gmail.com", "personal_email"),
    ],
)
def test_hygiene_flags_violations(text: str, rule: str) -> None:
    rules = {finding[1] for finding in tasks.scan_text(text)}
    assert rule in rules


@pytest.mark.parametrize(
    "text",
    [
        "author = 'OmarHazem123" + "@" + "users.noreply.github.com'",
        "Co-Authored-By: Bot <noreply" + "@" + "anthropic.com>",
        "email: someone" + "@" + "example.com",
        "the C: drive is nearly full",
        "ratio: 3:1 and http://localhost:8501/_stcore/health",
        "@dataclass(frozen=True)",
    ],
)
def test_hygiene_allows_benign_text(text: str) -> None:
    assert tasks.scan_text(text) == []


def test_makefile_parity_detects_drift() -> None:
    makefile = "RUN := uv run python tasks.py\n\nhelp:\n\t@$(RUN) --help\n\nlint:\n\t$(RUN) lint\n"
    assert tasks.makefile_problems(makefile, ["lint"]) == []
    problems = tasks.makefile_problems(makefile, ["lint", "test"])
    assert problems == ["Makefile is missing target 'test'"]
    spaces = "lint:\n    $(RUN) lint\n"
    assert tasks.makefile_problems(spaces, ["lint"]) == [
        "target 'lint': recipe must start with a TAB"
    ]
    wrong = "lint:\n\t$(RUN) format\n"
    assert tasks.makefile_problems(wrong, ["lint"]) == [
        "target 'lint': recipe must delegate to tasks.py lint"
    ]


def test_repository_makefile_mirrors_every_command() -> None:
    text = (tasks.ROOT / "Makefile").read_text(encoding="utf-8")
    assert tasks.makefile_problems(text, tasks.COMMANDS) == []


def test_repository_is_clean() -> None:
    files = tasks.project_files()
    assert tasks.find_eol_violations(files) == []
