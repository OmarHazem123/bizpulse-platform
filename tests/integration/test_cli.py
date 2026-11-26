from __future__ import annotations

from pathlib import Path

import pytest

from bizpulse import __version__
from bizpulse.cli import main

pytestmark = pytest.mark.integration


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_generate_run_and_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = str(tmp_path / "data")
    assert main(["--data-dir", data, "runs"]) == 0
    assert "no warehouse yet" in capsys.readouterr().out

    assert main(["--data-dir", data, "generate-demo", "--scale", "0.05", "--datasets", "saas"]) == 0
    assert "saas" in capsys.readouterr().out

    source = tmp_path / "data" / "raw" / "saas_invoices.xlsx"
    mapping = (
        Path(__file__).parents[2] / "src" / "bizpulse" / "resources" / "mappings" / "saas.yaml"
    )
    args = ["--data-dir", data, "run-pipeline", "--file", str(source), "--mapping", str(mapping)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "success_with_warnings" in out
    assert "reports/saas/" in out

    assert main(args) == 0
    assert "skipped" in capsys.readouterr().out

    assert main(["--data-dir", data, "runs", "--dataset", "saas"]) == 0
    history = capsys.readouterr().out
    assert history.count("saas") == 2


def test_run_pipeline_argument_errors(tmp_path: Path) -> None:
    data = str(tmp_path / "data")
    assert main(["--data-dir", data, "run-pipeline"]) == 2
    assert main(["--data-dir", data, "run-pipeline", "--file", "x.csv"]) == 2
    assert main(["--data-dir", data, "run-pipeline", "--all"]) == 1
