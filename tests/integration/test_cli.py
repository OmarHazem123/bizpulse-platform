from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from bizpulse import __version__
from bizpulse.cli import main
from bizpulse.config import ENV_DATA_DIR, Settings

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


def test_help_documents_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for command in (
        "generate-demo",
        "run-pipeline",
        "runs",
        "status",
        "report",
        "forecast",
        "whatif",
        "smoke",
        "dashboard",
    ):
        assert command in out


def test_analysis_commands_need_data(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = str(tmp_path / "empty")
    assert main(["--data-dir", data, "report"]) == 1
    assert main(["--data-dir", data, "forecast", "--dataset", "x"]) == 1
    assert "no data yet" in capsys.readouterr().err


def test_report_text_and_json(
    demo_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = str(demo_settings.data_dir)
    target = tmp_path / "out" / "saas.json"
    assert main(["--data-dir", data, "report", "--dataset", "saas", "--json", str(target)]) == 0
    out = capsys.readouterr().out
    assert "RFM segments" in out and "Forecast (next 30 days)" in out
    assert "weekly monitoring" in out  # saas is monitored weekly (amendment A2)
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["dataset"]["name"] == "saas"
    assert report["anomalies"]["granularity"] == "W"
    assert len(report["forecast"]["daily"]) == 30
    assert report["kpis"]["all_time"]["net_revenue"] > 0

    assert main(["--data-dir", data, "report", "--dataset", "nope"]) == 1
    assert "unknown dataset" in capsys.readouterr().err


def test_report_all_datasets_without_ml(
    demo_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--data-dir", str(demo_settings.data_dir), "report", "--no-forecast", "--no-anomalies"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.count("KPIs (all time)") == 3
    assert "Forecast" not in out and "Anomalies" not in out


def test_forecast_command_writes_csv(
    demo_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "fc.csv"
    args = ["--data-dir", str(demo_settings.data_dir), "forecast", "--dataset", "retail_fb"]
    assert main([*args, "--csv", str(target)]) == 0
    assert "MAE ratio vs seasonal naive" in capsys.readouterr().out
    frame = pd.read_csv(target, index_col=0)
    assert list(frame.columns) == ["yhat", "lo80", "hi80", "lo95", "hi95", "baseline"]
    assert len(frame) == 30
    assert b"\r\n" not in target.read_bytes()


def test_whatif_command(demo_settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--data-dir", str(demo_settings.data_dir), "whatif", "--dataset", "ecommerce"]
    assert main(args) == 0
    zero = capsys.readouterr().out
    assert "(+0.0%)" in zero and "pass --cac" in zero
    assert main([*args, "--retention-pp", "5", "--aov-pct", "10", "--cac", "40"]) == 0
    scenario = capsys.readouterr().out
    assert "LTV:CAC" in scenario and "pass --cac" not in scenario
    assert main([*args, "--retention-pp", "90"]) == 2  # outside the lever bounds
    assert "retention" in capsys.readouterr().err


@pytest.mark.slow
def test_smoke_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["smoke", "--scale", "0.05", "--work-dir", str(tmp_path / "smoke")]) == 0
    out = capsys.readouterr().out
    assert "smoke: OK" in out
    assert "quarantine matches the manifest exactly" in out


def test_dashboard_command_launches_streamlit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from streamlit.web import cli as streamlit_cli

    from bizpulse.cli import DASHBOARD_APP

    calls: list[list[str]] = []
    monkeypatch.setattr(streamlit_cli, "main", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["bizpulse"])
    monkeypatch.setenv(ENV_DATA_DIR, "unused")  # restored after the test
    data = tmp_path / "data"
    assert main(["--data-dir", str(data), "dashboard", "--port", "8600", "--headless"]) == 0
    argv = calls[0]
    assert argv[:3] == ["streamlit", "run", str(DASHBOARD_APP)]
    assert DASHBOARD_APP.is_file()
    assert argv[argv.index("--server.port") + 1] == "8600"
    assert argv[argv.index("--server.headless") + 1] == "true"
    assert os.environ[ENV_DATA_DIR] == str(data.resolve())
    assert not (data / "warehouse").exists()  # no --demo-if-empty: nothing generated


def test_status_and_markdown_export(
    demo_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = str(demo_settings.data_dir)
    assert main(["--data-dir", data, "status"]) == 0
    out = capsys.readouterr().out
    assert "3 datasets" in out
    for name in ("ecommerce", "saas", "retail_fb"):
        assert name in out
    assert "success_with_warnings" in out

    target = tmp_path / "summary.md"
    args = ["--data-dir", data, "report", "--dataset", "ecommerce", "--markdown", str(target)]
    assert main([*args, "--no-forecast"]) == 0
    summary = target.read_text(encoding="utf-8")
    assert summary.startswith("# Executive summary:")
    for heading in ("## Key figures", "## Revenue concentration", "## Customers", "## Incidents"):
        assert heading in summary
    assert "| Gross revenue |" in summary
    assert "\r" not in summary

    assert main(["--data-dir", str(tmp_path / "none"), "status"]) == 0
    assert "no warehouse yet" in capsys.readouterr().out
