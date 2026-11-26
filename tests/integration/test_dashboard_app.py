"""The Streamlit dashboard, driven headlessly with ``streamlit.testing.v1.AppTest``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from bizpulse.app.app import CUSTOMERS, FORECASTS, OVERVIEW, TABS, UPLOAD, WHATIF
from bizpulse.app.views.upload import LAST_RUN
from bizpulse.config import ENV_DATA_DIR, Settings
from bizpulse.smoke import HOSTILE_ROWS, VALID_ROWS
from tests.conftest import make_rows

pytestmark = [pytest.mark.integration, pytest.mark.slow]

APP = Path(__file__).resolve().parents[2] / "src" / "bizpulse" / "app" / "app.py"
TIMEOUT = 240


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> AppTest:
    monkeypatch.setenv(ENV_DATA_DIR, str(data_dir))
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT)


def _show(at: AppTest, tab: str) -> AppTest:
    at.session_state["nav"] = tab
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


def _texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [i.value for i in at.info] + [w.value for w in at.warning]
    return "\n".join(str(p) for p in parts)


@pytest.mark.parametrize("dataset", ["ecommerce", "saas", "retail_fb"])
def test_every_tab_renders_for_every_dataset(
    dataset: str, demo_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(monkeypatch, demo_settings.data_dir).run()
    assert not at.exception
    at.session_state["dataset"] = dataset

    _show(at, OVERVIEW)
    assert len(at.metric) == 10  # 7 KPI tiles + MoM + WoW growth + forecast preview
    assert len(at.get("download_button")) == 1  # executive summary
    assert len(at.get("plotly_chart")) >= 5

    _show(at, CUSTOMERS)
    assert {m.label for m in at.metric} >= {"LTV after 1 month", "LTV after 12 months"}

    _show(at, FORECASTS)
    labels = {m.label for m in at.metric}
    assert "Backtest MAE vs seasonal naive" in labels
    assert "Incidents" in labels and "Company-wide incidents" in labels
    weekly = "Weekly monitoring" in _texts(at)
    assert weekly == (dataset == "saas")  # amendment A2: subscriptions monitored weekly

    _show(at, WHATIF)
    assert len(at.slider) == 5
    # Zero levers: the scenario equals the status quo, so no delta is shown.
    assert all(m.delta in (None, "") for m in at.metric[:2])

    _show(at, UPLOAD)
    assert len(at.get("file_uploader")) == 1


def test_filters_and_simulator_levers(
    demo_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(monkeypatch, demo_settings.data_dir).run()
    at.session_state["dataset"] = "ecommerce"
    at.radio(key="freq").set_value("M")
    _show(at, OVERVIEW)
    assert "monthly trends" in _texts(at)

    _show(at, WHATIF)
    at.slider(key="wi_ret_ecommerce").set_value(5.0)
    at.number_input(key="wi_cac_ecommerce").set_value(40.0)
    at.run()
    assert not at.exception and not at.error
    revenue = at.metric[0]
    assert revenue.delta is not None and str(revenue.delta).startswith("+")


def test_empty_warehouse_shows_call_to_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(monkeypatch, tmp_path / "empty").run()
    assert not at.exception and not at.error
    for tab in (OVERVIEW, CUSTOMERS, FORECASTS, WHATIF):
        _show(at, tab)
        assert "No data yet" in _texts(at)
        assert len(at.button) >= 1  # "Load demo data"
    _show(at, UPLOAD)
    assert len(at.get("file_uploader")) == 1
    assert set(TABS) == {OVERVIEW, CUSTOMERS, FORECASTS, WHATIF, UPLOAD}


def test_upload_flow_loads_a_new_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = make_rows({}, {"customer_id": "C2", "quantity": "3"}, {"order_date": "31/31/2026"})
    source = tmp_path / "Acme Orders.csv"
    frame.to_csv(source, index=False, lineterminator="\n")

    data_dir = tmp_path / "data"
    at = _app(monkeypatch, data_dir).run()
    _show(at, UPLOAD)
    at.file_uploader(key="up_file").upload(source.name, source.read_bytes(), "text/csv")
    at.run()
    assert not at.exception and not at.error
    assert at.text_input(key="up_dataset").value == "acme_orders"
    assert len(at.dataframe) >= 1  # preview

    at.button(key="up_run").click()
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    outcome = at.session_state[LAST_RUN]
    assert outcome["status"] == "success_with_warnings"
    report = outcome["report"]
    assert (report["rows"]["valid"], report["rows"]["quarantined"]) == (2, 1)
    assert report["reason_counts"] == {"order_date.unparseable": 1}
    assert b"Data Quality Audit" in outcome["markdown"]

    # The new dataset is selectable and every analytics tab copes with tiny data.
    assert "acme_orders" in at.selectbox(key="dataset").options
    assert not (data_dir / "uploads").exists() or not any((data_dir / "uploads").iterdir())
    for tab in (OVERVIEW, CUSTOMERS, FORECASTS, WHATIF):
        _show(at, tab)


def test_hostile_upload_without_optional_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NaN / inf / bad dates and no optional columns: valid rows load, nothing crashes."""
    header = "transaction_id,order_date,customer_id,product_id,quantity,unit_price"
    source = tmp_path / "hostile.csv"
    lines = [header, *VALID_ROWS, *HOSTILE_ROWS]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    at = _app(monkeypatch, tmp_path / "data").run()
    _show(at, UPLOAD)
    at.file_uploader(key="up_file").upload(source.name, source.read_bytes(), "text/csv")
    at.run()
    assert not at.exception and not at.error
    at.button(key="up_run").click()
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    outcome = at.session_state[LAST_RUN]
    assert outcome["status"] == "success_with_warnings"
    rows = outcome["report"]["rows"]
    assert (rows["valid"], rows["quarantined"]) == (len(VALID_ROWS), len(HOSTILE_ROWS))
    # Every analytics tab copes with a tiny dataset that has no optional columns.
    for tab in (OVERVIEW, CUSTOMERS, FORECASTS, WHATIF, UPLOAD):
        _show(at, tab)
