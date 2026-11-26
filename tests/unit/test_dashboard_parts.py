"""Pure dashboard layers: palette, chart builders and upload helpers (no Streamlit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from bizpulse.config import Settings
from bizpulse.dashboard import charts
from bizpulse.dashboard.ingest import (
    NOT_MAPPED,
    build_config,
    dataset_name_error,
    explicit_columns,
    ingest,
    mapping_table,
    preview,
    safe_filename,
    staged_upload,
    suggest_dataset_name,
)
from bizpulse.dashboard.palette import DARK, LIGHT, palette_for, severity_label
from bizpulse.ingestion.mapping import MappingConfig
from bizpulse.ml.anomaly import Alert, RootCause
from bizpulse.schema import CANONICAL_FIELDS
from tests.conftest import make_rows


# --------------------------------------------------------------------------- palette
def test_palette_follows_theme_and_reserves_status_colors() -> None:
    assert palette_for("dark") is DARK
    assert palette_for("light") is LIGHT
    assert palette_for(None) is LIGHT
    for pal in (LIGHT, DARK):
        assert pal.critical not in pal.series and pal.warning not in pal.series
        assert len(set(pal.series)) == len(pal.series)
    # Sequential ramps: one hue; in dark mode the low end recedes toward the surface.
    assert DARK.sequential == tuple(reversed(LIGHT.sequential))
    assert severity_label("CRITICAL").endswith("CRITICAL")
    assert severity_label("WARNING") != "WARNING"  # an icon always accompanies the label


# --------------------------------------------------------------------------- charts
def _alert(severity: str, *, scope_value: str | None = None) -> Alert:
    return Alert(
        date=pd.Timestamp("2026-03-10"),
        metric="net_revenue",
        value=50.0,
        baseline=200.0,
        deviation=-0.75,
        z=-6.0,
        methods=("zscore", "iqr"),
        severity=severity,
        direction="drop",
        root_causes=(
            RootCause("region", "North", 10.0, 120.0, -110.0, 0.73),
            RootCause("channel", "web", 20.0, 90.0, -70.0, 0.47),
        ),
        scope_dimension="region" if scope_value else None,
        scope_value=scope_value,
    )


def test_fold_other_keeps_top_and_sums_the_tail() -> None:
    values = pd.Series({"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0})
    folded = charts.fold_other(values, top=2)
    assert list(folded.index) == ["a", "b", charts.OTHER]
    assert folded[charts.OTHER] == 6.0
    assert list(charts.fold_other(values, top=4).index) == ["a", "b", "c", "d", "e"]


def test_ranked_bars_fold_and_no_fold() -> None:
    values = pd.Series({f"f{i}": float(i) for i in range(12)})
    folded = charts.ranked_bars(values, LIGHT, top=3)
    assert list(folded.data[0].y) == ["f11", "f10", "f9", charts.OTHER]
    ranking = charts.ranked_bars(values, LIGHT, top=3, fold=False)
    assert list(ranking.data[0].y) == ["f11", "f10", "f9"]
    assert folded.data[0].marker.color == LIGHT.series[0]  # one hue for nominal bars


def test_forecast_figure_draws_bands_actuals_forecast_and_baseline() -> None:
    history = pd.Series(
        range(60), index=pd.date_range("2026-01-01", periods=60, freq="D"), dtype="float64"
    )
    index = pd.date_range("2026-03-02", periods=5, freq="D")
    forecast = pd.DataFrame(
        {
            "yhat": 60.0,
            "lo80": 55.0,
            "hi80": 65.0,
            "lo95": 50.0,
            "hi95": 70.0,
            "baseline": 58.0,
        },
        index=index,
    )
    fig = charts.forecast_figure(history, forecast, DARK, days=30, prefix="$")
    names = [trace.name for trace in fig.data if trace.name]
    assert names == [
        "95% interval",
        "80% interval",
        "Actual",
        "Forecast (LightGBM)",
        "Seasonal naive",
    ]
    assert len(fig.data[4].x) == 30  # only the recent window of actuals
    forecast_line = fig.data[5]
    assert forecast_line.line.dash == "dash"
    assert len(forecast_line.x) == len(index) + 1  # bridged to the last actual point


def test_anomaly_timeline_marks_severities_with_status_colors() -> None:
    values = pd.Series([100.0, 120.0, 50.0], index=pd.date_range("2026-03-08", periods=3, freq="D"))
    fig = charts.anomaly_timeline(
        values, [_alert("CRITICAL"), _alert("WARNING")], LIGHT, name="Net revenue"
    )
    markers = {trace.name: trace for trace in fig.data[1:]}
    assert markers[severity_label("CRITICAL")].marker.color == LIGHT.critical
    assert markers[severity_label("WARNING")].marker.color == LIGHT.warning
    assert markers[severity_label("CRITICAL")].marker.symbol != (
        markers[severity_label("WARNING")].marker.symbol
    )


def test_root_cause_bars_group_by_dimension() -> None:
    fig = charts.root_cause_bars(_alert("CRITICAL"), LIGHT, "$")
    assert list(fig.data[0].y) == ["channel: web", "region: North"]
    assert list(fig.data[0].x) == [-70.0, -110.0]


def test_heatmap_and_whatif_figures() -> None:
    grid = pd.DataFrame([[1.0, 0.5], [1.0, None]], index=["2026-01", "2026-02"], columns=[0, 1])
    heat = charts.heatmap(grid, DARK, x_title="m", y_title="c", fmt=".0%", value_name="v")
    assert heat.data[0].colorscale[0][1] == DARK.sequential[0]
    months = pd.Series([1.0, 2.0], index=[1, 2])
    lines = charts.whatif_figure(months, months * 2, LIGHT, prefix="$")
    assert [t.name for t in lines.data] == ["Status quo", "Scenario"]
    assert lines.data[0].line.color == LIGHT.muted


# --------------------------------------------------------------------------- upload
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("orders.csv", "orders.csv"),
        ("..\\..\\evil name.xlsx", "evil_name.xlsx"),
        ("/srv/../x y?.parquet", "x_y_.parquet"),
        ("...", "upload"),
    ],
)
def test_safe_filename(raw: str, expected: str) -> None:
    assert safe_filename(raw) == expected


def test_dataset_name_helpers() -> None:
    assert suggest_dataset_name("Q3 Orders (final).csv") == "q3_orders_final"
    assert suggest_dataset_name("2026 sales.xlsx") == "data_2026_sales"
    assert dataset_name_error(suggest_dataset_name("x.csv")) is None
    assert dataset_name_error("acme") is None
    assert dataset_name_error("Acme") is not None
    assert dataset_name_error("1acme") is not None


def test_staged_upload_is_removed_afterwards(settings: Settings) -> None:
    with staged_upload(b"a,b\n1,2\n", "my file.csv", settings) as path:
        assert path.name == "my_file.csv"
        assert path.read_bytes() == b"a,b\n1,2\n"
        folder = path.parent
    assert not folder.exists()


def _write_upload(tmp_path: Path) -> bytes:
    frame = make_rows({}, {"customer_id": "C2"}, {"order_date": "not a date"})
    frame = frame.rename(columns={"order_date": "Order Date", "customer_id": "Client"})
    target = tmp_path / "upload.csv"
    frame.to_csv(target, index=False, lineterminator="\n")
    return target.read_bytes()


def test_preview_mapping_table_and_explicit_pins(tmp_path: Path, settings: Settings) -> None:
    data = _write_upload(tmp_path)
    config = MappingConfig(dataset="acme")
    with staged_upload(data, "acme.csv", settings) as path:
        found = preview(path, config)
    assert len(found.sample) == 3
    table = mapping_table(found.mapping)
    assert list(table["field"]) == list(CANONICAL_FIELDS)
    by_field = dict(zip(table["field"], table["source"], strict=True))
    assert by_field["order_date"] == "Order Date"
    assert by_field["customer_id"] == NOT_MAPPED  # "Client" is not a known synonym
    assert "customer_id" in found.mapping.missing_required

    edited = table.copy()
    edited.loc[edited["field"] == "customer_id", "source"] = "Client"
    pinned = explicit_columns(edited, table)
    assert pinned == {"customer_id": "Client"}  # unchanged auto matches stay automatic


def test_build_config_validates(tmp_path: Path) -> None:
    config = build_config(
        dataset="acme",
        display_name=None,
        domain="custom",
        reporting_currency="usd",
        timezone="Europe/Berlin",
        dayfirst=True,
        columns={"customer_id": "Client"},
    )
    assert config.reporting_currency == "USD"
    assert config.columns == {"customer_id": "Client"}
    with pytest.raises(ValidationError):
        build_config(
            dataset="Bad Name",
            display_name=None,
            domain="custom",
            reporting_currency="USD",
            timezone="UTC",
            dayfirst=False,
        )
    with pytest.raises(ValidationError):
        build_config(
            dataset="acme",
            display_name=None,
            domain="custom",
            reporting_currency="USD",
            timezone="Mars/Olympus",
            dayfirst=False,
        )


def test_ingest_loads_quarantines_and_compacts(tmp_path: Path, settings: Settings) -> None:
    data = _write_upload(tmp_path)
    config = build_config(
        dataset="acme",
        display_name="Acme",
        domain="custom",
        reporting_currency="USD",
        timezone="UTC",
        dayfirst=False,
        columns={"customer_id": "Client"},
    )
    with staged_upload(data, "acme orders.csv", settings) as path:
        outcome = ingest(path, config, settings)
    result = outcome.result
    assert result.status == "success_with_warnings"
    assert result.report.source_file == "acme_orders.csv"  # original name, not a temp name
    assert (result.report.rows.valid, result.report.rows.quarantined) == (2, 1)
    assert outcome.compacted is not None and outcome.compact_error is None
    assert not (settings.data_dir / "uploads").exists() or not any(
        (settings.data_dir / "uploads").iterdir()
    )
