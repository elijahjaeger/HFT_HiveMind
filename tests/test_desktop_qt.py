"""Qt-layer smoke tests, run against the offscreen platform plugin.

These exercise the table model, the chart builders and construction of the main
window itself, so a broken widget wiring fails in CI rather than on launch.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# pandas before PySide6: see app/__init__.py for the import-order note.
pytest.importorskip("pandas")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from app.qt_charts import time_series_chart, update_chart  # noqa: E402
from app.qt_models import RowTableModel, format_cell  # noqa: E402

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)


@pytest.fixture(scope="module")
def qt_app():
    """A single QApplication for the module."""
    app = QApplication.instance() or QApplication([])
    yield app


# -- table model -----------------------------------------------------------


def test_empty_values_render_as_blank_cells():
    assert format_cell(None) == ""
    assert format_cell(float("nan")) == ""


def test_numbers_and_timestamps_are_formatted_for_display():
    assert format_cell(0.4200) == "0.42"
    assert format_cell(12345.678) == "12,345.68"
    assert format_cell(True) == "yes"
    assert format_cell(T0).endswith("UTC")


def test_the_table_model_exposes_rows_and_columns(qt_app):
    model = RowTableModel([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    assert model.rowCount() == 2
    assert model.columnCount() == 2
    assert model.columns == ["a", "b"]
    index = model.index(1, 1)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "y"
    assert model.data(index, Qt.ItemDataRole.UserRole) == "y"


def test_column_headers_are_humanised(qt_app):
    model = RowTableModel([{"market_id": "M1"}])
    header = model.headerData(0, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
    assert header == "market id"


def test_replacing_rows_resets_the_model(qt_app):
    model = RowTableModel([{"a": 1}])
    model.set_rows([{"b": 1}, {"b": 2}])
    assert model.columns == ["b"]
    assert model.rowCount() == 2


def test_an_empty_model_is_valid(qt_app):
    model = RowTableModel([])
    assert model.rowCount() == 0
    assert model.columnCount() == 0
    assert model.data(model.index(0, 0)) is None


# -- charts ----------------------------------------------------------------


def points(n: int) -> list[tuple[datetime, float]]:
    return [(T0 + timedelta(hours=i), 0.4 + 0.01 * i) for i in range(n)]


def test_a_chart_is_built_with_one_series_per_entry(qt_app):
    view = time_series_chart(
        "Price", {"Bid": points(3), "Ask": points(3)}, {"Trades": points(2)},
        y_title="p", y_range=(0.0, 1.0),
    )
    assert len(view.chart().series()) == 3
    assert view.chart().title() == "Price"


def test_an_empty_chart_is_still_valid(qt_app):
    view = time_series_chart("Equity", {})
    assert view.chart().series() == []


def test_updating_a_chart_replaces_its_series(qt_app):
    view = time_series_chart("Equity", {"Equity": points(3)})
    update_chart(view, {"Equity": points(5), "Peak": points(5)})
    assert len(view.chart().series()) == 2


def test_none_values_are_skipped_rather_than_plotted_as_zero(qt_app):
    view = time_series_chart("Price", {"Bid": [(T0, None), (T0, 0.4)]})
    assert view.chart().series()[0].count() == 1


# -- main window -----------------------------------------------------------


def test_the_main_window_builds_and_loads_demo_data(qt_app):
    from app.desktop import MainWindow

    window = MainWindow()
    try:
        # The data worker runs on a thread; wait for it to land.
        deadline = datetime.now(UTC) + timedelta(seconds=60)
        while window.dataset is None and datetime.now(UTC) < deadline:
            qt_app.processEvents()

        assert window.dataset is not None, "demo data never loaded"
        assert window.dataset.is_demo
        # Simple mode is the default: only the surfaces needed to run an
        # experiment and read the evidence.
        simple = {window.tabs.tabText(i) for i in range(window.tabs.count())}
        assert simple == {
            "Backtest",
            "Research",
            "Corpus",
            "Market explorer",
        }, f"unexpected simple-mode tabs: {simple}"
        # isHidden() rather than isVisible(): the window itself is never
        # shown in this test, so every child reports invisible regardless.
        assert window.controls_dock.isHidden() is True
        assert window.single_run_box.isHidden() is True
        assert window.simple_mode is True

        # Advanced reveals everything; nothing is removed, only hidden.
        window.advanced_toggle.setChecked(True)
        advanced = {window.tabs.tabText(i) for i in range(window.tabs.count())}
        assert simple < advanced, "advanced mode must be a superset"
        assert {
            "Calibration",
            "Paper mode",
            "Audit log",
            "Data source",
            "Agents && risk",
        } <= advanced, f"missing tab(s): {advanced}"
        assert window.controls_dock.isHidden() is False
        assert window.single_run_box.isHidden() is False
        assert window.simple_mode is False
        window.advanced_toggle.setChecked(False)
        labels = [w.text() for w in window.findChildren(QLabel)]
        assert any("LIVE TRADING DISABLED" in text for text in labels)
        assert any("DEMO DATA" in text for text in labels)
        assert window.market_model.rowCount() == len(window.dataset.markets)
        assert window.decision_market.count() == len(window.dataset.markets)
        assert window.as_of is not None
    finally:
        window.close()


def test_collect_settings_reads_the_controls_back(qt_app):
    from app.desktop import MainWindow

    window = MainWindow()
    try:
        window.kill_switch.setChecked(True)
        window.max_position.setValue(123)
        settings = window.collect_settings()
        assert settings["risk"]["kill_switch"] is True
        assert settings["risk"]["max_position_size"] == pytest.approx(123.0)
        assert settings["app"]["live_trading_enabled"] is False
    finally:
        window.close()
