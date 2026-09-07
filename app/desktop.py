"""Native desktop application for the prediction-market research sandbox.

Run it with::

    python -m app.desktop

A plain Qt application: no browser, no web server, no network listener. The
only outbound traffic the app can make is the read-only Kalshi market-data
fetch on the Data tab, and it is always HTTP GET.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python app/desktop.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pandas MUST be imported before PySide6; see app/__init__.py for why.
import pandas  # noqa: E402, F401
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QAction, QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from adapters.kalshi import KalshiFetchSpec  # noqa: E402
from app import presenter  # noqa: E402
from app.qt_charts import (  # noqa: E402
    reliability_chart,
    time_series_chart,
    update_chart,
    update_reliability_chart,
)
from app.qt_models import RowTableModel  # noqa: E402
from app.workers import (  # noqa: E402
    BacktestWorker,
    DemoDataWorker,
    KalshiFetchWorker,
    WorkerRunner,
)
from backtest.metrics import MetricsSummary, metrics_rows  # noqa: E402
from backtest.skips import skip_summary  # noqa: E402
from core.config import get_settings, resolve_path  # noqa: E402
from core.dataset import MarketDataset  # noqa: E402
from core.models import RiskAction  # noqa: E402
from paper import PAPER_MODE_BANNER  # noqa: E402
from paper.session import PaperConfig, PaperSession  # noqa: E402
from paper.sources import KalshiQuoteSource, ReplayQuoteSource  # noqa: E402
from paper.store import PaperStore  # noqa: E402
from research.corpus import build_research_corpus  # noqa: E402
from research.evaluate import (  # noqa: E402
    calibration_comparison,
    calibration_rows,
    fold_rows,
    reliability_points,
)
from research.features import feature_dictionary_rows  # noqa: E402
from research.pipeline import (  # noqa: E402
    check_drift,
    prepare_data,
    run_model_backtest,
    train_and_evaluate,
)
from research.registry import ResearchRegistry  # noqa: E402

ACTION_COLOR = {
    RiskAction.APPROVE: "#1a7f37",
    RiskAction.REDUCE: "#9a6700",
    RiskAction.REJECT: "#cf222e",
}
AUDIT_KINDS = ["PROPOSAL", "RISK_DECISION", "FILL", "REJECTION", "SETTLEMENT",
               "FORECAST", "LIQUIDITY"]
DEFAULT_AUDIT_KINDS = {"PROPOSAL", "RISK_DECISION", "FILL", "REJECTION", "SETTLEMENT"}


def banner(text: str, background: str, foreground: str = "#ffffff") -> QLabel:
    """A full-width coloured banner label."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setStyleSheet(
        f"background:{background};color:{foreground};padding:8px 12px;"
        "border-radius:4px;font-weight:600;"
    )
    return label


def table_view(model: RowTableModel, stretch_last: bool = True) -> QTableView:
    """A configured read-only table view."""
    view = QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    view.setSortingEnabled(False)
    view.verticalHeader().setVisible(False)
    view.horizontalHeader().setStretchLastSection(stretch_last)
    view.setWordWrap(False)
    return view


def metric_tile(caption: str, value: str) -> QFrame:
    """A small caption-over-value tile."""
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 6, 10, 6)
    caption_label = QLabel(caption)
    caption_label.setStyleSheet("color:#57606a;font-size:11px;")
    value_label = QLabel(value)
    font = QFont()
    font.setPointSize(15)
    font.setBold(True)
    value_label.setFont(font)
    layout.addWidget(caption_label)
    layout.addWidget(value_label)
    return frame


class MainWindow(QMainWindow):
    """The application window."""

    def __init__(self) -> None:
        super().__init__()
        self.settings: dict[str, Any] = get_settings()
        self.dataset: MarketDataset | None = None
        self.result: Any = None
        self.calibration: Any = None
        self.paper_session: Any = None
        self.paper_store: Any = None
        self.paper_source_is_live = False
        self._paper_replay_clock: datetime | None = None
        self.paper_timer = QTimer(self)
        self.paper_timer.timeout.connect(self.poll_paper_session)
        self.prepared: Any = None
        self.training: Any = None
        self.research_backtest: Any = None
        self.research_corpus_data: Any = None
        self.registry = ResearchRegistry(
            self.settings.get('research', {}).get(
                'registry_path', 'data/research/registry.duckdb'
            )
        )
        self._runner: WorkerRunner | None = None

        self.setWindowTitle("HFT Hivemind - prediction-market research sandbox")
        self.resize(1400, 900)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(banner(presenter.LIVE_TRADING_BANNER, "#8b1a1a"))
        self.provenance = banner("Loading data...", "#7a5c00")
        layout.addWidget(self.provenance)

        self.as_of_label = QLabel("As-of: -")
        self.as_of_slider = QSlider(Qt.Orientation.Horizontal)
        self.as_of_slider.setEnabled(False)
        self.as_of_slider.valueChanged.connect(self._on_as_of_changed)
        as_of_row = QHBoxLayout()
        as_of_row.addWidget(QLabel("As-of (agents only see data at or before this):"))
        as_of_row.addWidget(self.as_of_slider, 1)
        as_of_row.addWidget(self.as_of_label)
        layout.addLayout(as_of_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_explorer_tab(), "Market explorer")
        self.tabs.addTab(self._build_decision_tab(), "Agents && risk")
        self.tabs.addTab(self._build_backtest_tab(), "Backtest")
        self.tabs.addTab(self._build_calibration_tab(), "Calibration")
        self.tabs.addTab(self._build_research_tab(), "Research")
        self.tabs.addTab(self._build_paper_tab(), "Paper mode")
        self.tabs.addTab(self._build_audit_tab(), "Audit log")
        self.tabs.addTab(self._build_data_tab(), "Data source")
        layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._build_controls_dock()
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(160)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Simulation only - no orders are ever placed.")

        self._build_menu()
        self.load_demo_data()

    # -- construction ------------------------------------------------------

    def _build_menu(self) -> None:
        data_menu = self.menuBar().addMenu("&Data")
        reload_demo = QAction("Reload DEMO DATA", self)
        reload_demo.triggered.connect(lambda: self.load_demo_data(refresh=True))
        data_menu.addAction(reload_demo)
        fetch = QAction("Fetch Kalshi history (read-only)...", self)
        fetch.triggered.connect(lambda: self.tabs.setCurrentIndex(7))
        data_menu.addAction(fetch)

        run_menu = self.menuBar().addMenu("&Run")
        run_action = QAction("Run backtest", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.run_backtest)
        run_menu.addAction(run_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _build_controls_dock(self):
        from PySide6.QtWidgets import QDockWidget

        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)

        data_cfg = self.settings["data"]
        demo_box = QGroupBox("Demo data")
        demo_form = QFormLayout(demo_box)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2_000_000_000)
        self.seed_input.setValue(int(data_cfg["seed"]))
        self.markets_input = QSpinBox()
        self.markets_input.setRange(3, 12)
        self.markets_input.setValue(int(data_cfg["n_markets"]))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(60, 400)
        self.steps_input.setSingleStep(20)
        self.steps_input.setValue(int(data_cfg["n_steps"]))
        regenerate = QPushButton("Regenerate demo data")
        regenerate.clicked.connect(lambda: self.load_demo_data(refresh=True))
        demo_form.addRow("Seed", self.seed_input)
        demo_form.addRow("Markets", self.markets_input)
        demo_form.addRow("Snapshots", self.steps_input)
        demo_form.addRow(regenerate)
        layout.addWidget(demo_box)

        strategy = self.settings["strategy"]
        strategy_box = QGroupBox("Strategy")
        strategy_form = QFormLayout(strategy_box)
        self.min_edge = self._spin(strategy["min_edge"], 0.005, 0.10, 0.005, 3)
        self.base_size = QSpinBox()
        self.base_size.setRange(10, 500)
        self.base_size.setValue(int(strategy["base_size"]))
        self.max_order = QSpinBox()
        self.max_order.setRange(50, 1000)
        self.max_order.setSingleStep(50)
        self.max_order.setValue(int(strategy["max_order_size"]))
        self.require_ev = QCheckBox("Require positive EV after costs")
        self.require_ev.setChecked(bool(strategy["require_positive_ev"]))
        strategy_form.addRow("Minimum edge", self.min_edge)
        strategy_form.addRow("Base size", self.base_size)
        strategy_form.addRow("Max order size", self.max_order)
        strategy_form.addRow(self.require_ev)
        layout.addWidget(strategy_box)

        risk = self.settings["risk"]
        risk_box = QGroupBox("Risk limits")
        risk_form = QFormLayout(risk_box)
        self.kill_switch = QCheckBox("Kill switch (reject everything)")
        self.kill_switch.setChecked(bool(risk["kill_switch"]))
        self.max_position = QSpinBox()
        self.max_position.setRange(10, 5000)
        self.max_position.setSingleStep(10)
        self.max_position.setValue(int(risk["max_position_size"]))
        self.max_event_exposure = QSpinBox()
        self.max_event_exposure.setRange(50, 20_000)
        self.max_event_exposure.setSingleStep(50)
        self.max_event_exposure.setValue(int(risk["max_event_exposure"]))
        self.max_total_exposure = QSpinBox()
        self.max_total_exposure.setRange(100, 50_000)
        self.max_total_exposure.setSingleStep(100)
        self.max_total_exposure.setValue(int(risk["max_total_exposure"]))
        self.max_spread = self._spin(risk["max_spread"], 0.01, 0.50, 0.005, 3)
        self.max_slippage = self._spin(risk["max_slippage"], 0.001, 0.05, 0.001, 4)
        self.max_data_age = QSpinBox()
        self.max_data_age.setRange(60, 86_400)
        self.max_data_age.setSingleStep(60)
        self.max_data_age.setValue(int(risk["max_data_age_seconds"]))
        self.max_drawdown = self._spin(risk["max_drawdown"], 0.01, 1.0, 0.01, 2)
        risk_form.addRow(self.kill_switch)
        risk_form.addRow("Max position", self.max_position)
        risk_form.addRow("Max event exposure $", self.max_event_exposure)
        risk_form.addRow("Max total exposure $", self.max_total_exposure)
        risk_form.addRow("Max spread", self.max_spread)
        risk_form.addRow("Max slippage", self.max_slippage)
        risk_form.addRow("Max data age (s)", self.max_data_age)
        risk_form.addRow("Max drawdown", self.max_drawdown)
        layout.addWidget(risk_box)

        fills = self.settings["fills"]
        exec_box = QGroupBox("Execution && capital")
        exec_form = QFormLayout(exec_box)
        self.fee_rate = self._spin(fills["fee_rate"], 0.0, 0.20, 0.005, 3)
        self.fee_model = QComboBox()
        self.fee_model.addItems(["kalshi", "notional"])
        self.fee_model.setCurrentText(str(fills["fee_model"]))
        self.base_slippage = self._spin(fills["base_slippage"], 0.0, 0.05, 0.001, 4)
        self.impact = self._spin(fills["impact_coefficient"], 0.0, 0.05, 0.001, 4)
        self.allow_partial = QCheckBox("Allow partial fills")
        self.allow_partial.setChecked(bool(fills["allow_partial_fills"]))
        self.capital = QDoubleSpinBox()
        self.capital.setRange(100.0, 10_000_000.0)
        self.capital.setSingleStep(1000.0)
        self.capital.setValue(float(self.settings["backtest"]["initial_capital"]))
        exec_form.addRow("Fee rate", self.fee_rate)
        exec_form.addRow("Fee model", self.fee_model)
        exec_form.addRow("Base slippage", self.base_slippage)
        exec_form.addRow("Impact coefficient", self.impact)
        exec_form.addRow(self.allow_partial)
        exec_form.addRow("Initial capital $", self.capital)
        layout.addWidget(exec_box)

        for widget in (
            self.min_edge, self.base_size, self.max_order, self.max_position,
            self.max_event_exposure, self.max_total_exposure, self.max_spread,
            self.max_slippage, self.max_data_age, self.max_drawdown,
            self.fee_rate, self.base_slippage, self.impact, self.capital,
        ):
            widget.valueChanged.connect(self._refresh_decision_panel)
        for toggle in (self.kill_switch, self.require_ev, self.allow_partial):
            toggle.toggled.connect(self._refresh_decision_panel)
        self.fee_model.currentTextChanged.connect(self._refresh_decision_panel)

        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        dock.setWidget(scroll)
        return dock

    @staticmethod
    def _spin(value: float, low: float, high: float, step: float, decimals: int):
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        return spin

    def _build_explorer_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.market_model = RowTableModel()
        self.market_table = table_view(self.market_model)
        self.price_chart = time_series_chart("Price", {}, y_title="Implied probability")
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.market_table)
        splitter.addWidget(self.price_chart)
        splitter.setSizes([260, 420])
        layout.addWidget(splitter, 1)
        self.explorer_note = QLabel(
            "Gaps in the bid/ask lines are stale periods or one-sided books - both are "
            "real gaps in the data, never interpolated."
        )
        self.explorer_note.setStyleSheet("color:#57606a;")
        self.explorer_note.setWordWrap(True)
        layout.addWidget(self.explorer_note)
        return page

    def _build_decision_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Market:"))
        self.decision_market = QComboBox()
        self.decision_market.currentTextChanged.connect(self._refresh_decision_panel)
        picker_row.addWidget(self.decision_market, 1)
        layout.addLayout(picker_row)

        self.question_label = QLabel("")
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("color:#57606a;")
        layout.addWidget(self.question_label)

        top = QSplitter(Qt.Orientation.Horizontal)
        forecast_panel = QWidget()
        forecast_layout = QVBoxLayout(forecast_panel)
        forecast_layout.addWidget(QLabel("<b>Agent forecasts</b>"))
        self.forecast_model = RowTableModel()
        forecast_layout.addWidget(table_view(self.forecast_model))
        forecast_layout.addWidget(
            QLabel("<b>Declared inputs and assumptions</b>")
        )
        self.provenance_model = RowTableModel()
        provenance_table = table_view(self.provenance_model)
        provenance_table.setMaximumHeight(190)
        forecast_layout.addWidget(provenance_table)
        self.consensus_row = QHBoxLayout()
        forecast_layout.addLayout(self.consensus_row)
        self.excluded_label = QLabel("")
        self.excluded_label.setStyleSheet("color:#57606a;")
        self.excluded_label.setWordWrap(True)
        forecast_layout.addWidget(self.excluded_label)
        top.addWidget(forecast_panel)

        liquidity_panel = QWidget()
        liquidity_layout = QVBoxLayout(liquidity_panel)
        liquidity_layout.addWidget(QLabel("<b>Liquidity</b>"))
        self.liquidity_grid = QGridLayout()
        liquidity_layout.addLayout(self.liquidity_grid)
        self.liquidity_warnings = QPlainTextEdit()
        self.liquidity_warnings.setReadOnly(True)
        self.liquidity_warnings.setMaximumHeight(120)
        liquidity_layout.addWidget(self.liquidity_warnings)
        liquidity_layout.addStretch(1)
        top.addWidget(liquidity_panel)
        top.setSizes([760, 420])
        layout.addWidget(top, 1)

        layout.addWidget(QLabel("<b>Proposed trade and risk decision</b>"))
        self.proposal_row = QHBoxLayout()
        layout.addLayout(self.proposal_row)
        self.proposal_note = QLabel("")
        self.proposal_note.setWordWrap(True)
        self.proposal_note.setStyleSheet("color:#57606a;")
        layout.addWidget(self.proposal_note)
        self.decision_banner = QLabel("")
        self.decision_banner.setWordWrap(True)
        layout.addWidget(self.decision_banner)
        self.reason_model = RowTableModel()
        self.reason_table = table_view(self.reason_model)
        self.reason_table.setMaximumHeight(180)
        self.reason_table.setVisible(False)
        layout.addWidget(self.reason_table)
        footer = QLabel(
            "The risk agent only returns this decision object. It has no order-placement path."
        )
        footer.setStyleSheet("color:#57606a;")
        layout.addWidget(footer)
        return page

    def _build_backtest_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        self.run_button = QPushButton("Run backtest")
        self.run_button.clicked.connect(self.run_backtest)
        controls.addWidget(self.run_button)
        note = QLabel(
            "Replays the loaded data chronologically through the agent stack. Results "
            "describe one simulation - not evidence of profitability."
        )
        note.setStyleSheet("color:#57606a;")
        note.setWordWrap(True)
        controls.addWidget(note, 1)
        layout.addLayout(controls)

        self.metric_row = QHBoxLayout()
        layout.addLayout(self.metric_row)

        charts = QSplitter(Qt.Orientation.Vertical)
        self.equity_chart = time_series_chart("Equity curve (simulated)", {}, y_title="Equity $")
        self.drawdown_chart = time_series_chart(
            "Drawdown (%)", {}, y_title="DD %", show_legend=False
        )
        self.equity_chart.setMinimumHeight(260)
        self.drawdown_chart.setMinimumHeight(240)
        charts.addWidget(self.equity_chart)
        charts.addWidget(self.drawdown_chart)
        charts.setSizes([340, 260])
        charts.setMinimumHeight(510)
        layout.addWidget(charts, 3)

        tables = QSplitter(Qt.Orientation.Horizontal)
        metrics_panel = QWidget()
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.addWidget(QLabel("<b>Metrics</b>"))
        self.metrics_model = RowTableModel()
        metrics_layout.addWidget(table_view(self.metrics_model))
        tables.addWidget(metrics_panel)

        calibration_panel = QWidget()
        calibration_layout = QVBoxLayout(calibration_panel)
        calibration_layout.addWidget(QLabel("<b>Calibration by probability bucket</b>"))
        self.calibration_model = RowTableModel()
        calibration_layout.addWidget(table_view(self.calibration_model))
        tables.addWidget(calibration_panel)

        decisions_panel = QWidget()
        decisions_layout = QVBoxLayout(decisions_panel)
        decisions_layout.addWidget(QLabel("<b>Risk decisions</b>"))
        self.decisions_model = RowTableModel()
        decisions_layout.addWidget(table_view(self.decisions_model))
        tables.addWidget(decisions_panel)
        tables.setSizes([420, 420, 460])
        layout.addWidget(tables, 2)
        return page

    def _build_calibration_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Forecast source:"))
        self.calib_source = QComboBox()
        self.calib_source.addItem("consensus")
        controls.addWidget(self.calib_source)
        controls.addWidget(QLabel("Train:"))
        self.calib_train = self._spin(0.5, 0.1, 0.8, 0.05, 2)
        controls.addWidget(self.calib_train)
        controls.addWidget(QLabel("Validation:"))
        self.calib_validation = self._spin(0.25, 0.05, 0.5, 0.05, 2)
        controls.addWidget(self.calib_validation)
        controls.addWidget(QLabel("Buckets:"))
        self.calib_buckets = QSpinBox()
        self.calib_buckets.setRange(4, 20)
        self.calib_buckets.setValue(10)
        controls.addWidget(self.calib_buckets)
        controls.addWidget(QLabel("Select on:"))
        self.calib_metric = QComboBox()
        self.calib_metric.addItems(["brier", "log_loss", "ece"])
        controls.addWidget(self.calib_metric)
        self.calib_button = QPushButton("Fit and compare")
        self.calib_button.clicked.connect(self.run_calibration)
        controls.addWidget(self.calib_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.calib_status = QLabel(
            "Run a backtest, then fit calibrators on its forecasts. Splits are "
            "chronological and grouped by market, so no market appears in two folds."
        )
        self.calib_status.setStyleSheet("color:#57606a;")
        self.calib_status.setWordWrap(True)
        layout.addWidget(self.calib_status)

        self.calib_warning = QLabel("")
        self.calib_warning.setWordWrap(True)
        self.calib_warning.setVisible(False)
        layout.addWidget(self.calib_warning)

        top = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Held-out test fold</b>"))
        self.calib_test_model = RowTableModel()
        test_table = table_view(self.calib_test_model)
        test_table.setMaximumHeight(150)
        left_layout.addWidget(test_table)
        left_layout.addWidget(QLabel("<b>Chronological split</b>"))
        self.calib_split_model = RowTableModel()
        left_layout.addWidget(table_view(self.calib_split_model))
        top.addWidget(left)

        self.reliability_view = reliability_chart({})
        self.reliability_view.setMinimumHeight(340)
        top.addWidget(self.reliability_view)
        top.setSizes([780, 620])
        layout.addWidget(top, 3)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        comparison = QWidget()
        comparison_layout = QVBoxLayout(comparison)
        comparison_layout.addWidget(
            QLabel("<b>All folds</b> (train scores are in-sample by construction)")
        )
        self.calib_compare_model = RowTableModel()
        comparison_layout.addWidget(table_view(self.calib_compare_model))
        bottom.addWidget(comparison)

        buckets = QWidget()
        buckets_layout = QVBoxLayout(buckets)
        picker = QHBoxLayout()
        picker.addWidget(QLabel("<b>Calibration table</b>"))
        self.calib_table_method = QComboBox()
        self.calib_table_method.addItems(["uncalibrated", "isotonic", "logistic"])
        self.calib_table_method.currentTextChanged.connect(self._refresh_calibration_table)
        picker.addWidget(self.calib_table_method)
        self.calib_table_fold = QComboBox()
        self.calib_table_fold.addItems(["test", "validation", "train"])
        self.calib_table_fold.currentTextChanged.connect(self._refresh_calibration_table)
        picker.addWidget(self.calib_table_fold)
        picker.addStretch(1)
        buckets_layout.addLayout(picker)
        self.calib_bucket_model = RowTableModel()
        buckets_layout.addWidget(table_view(self.calib_bucket_model))
        bottom.addWidget(buckets)
        bottom.setSizes([700, 700])
        layout.addWidget(bottom, 2)
        return page

    def run_calibration(self) -> None:
        """Fit and compare calibrators on the current backtest result."""
        if self.result is None:
            QMessageBox.information(
                self,
                "No backtest",
                "Run a backtest first: calibration is fitted on the forecasts it recorded.",
            )
            return
        report, message = presenter.run_calibration(
            self.result,
            source=self.calib_source.currentText(),
            train_fraction=float(self.calib_train.value()),
            validation_fraction=float(self.calib_validation.value()),
            n_buckets=int(self.calib_buckets.value()),
            selection_metric=self.calib_metric.currentText(),
        )
        self.calibration = report
        self.calib_status.setText(message)

        if report is None:
            self.calib_warning.setVisible(False)
            for model in (
                self.calib_test_model,
                self.calib_split_model,
                self.calib_compare_model,
                self.calib_bucket_model,
            ):
                model.set_rows([])
            update_reliability_chart(self.reliability_view, {})
            return

        self.calib_test_model.set_rows(report.test_rows())
        self.calib_split_model.set_rows(presenter.split_rows(report))
        self.calib_compare_model.set_rows(report.comparison_rows())
        update_reliability_chart(self.reliability_view, report.reliability_series("test"))
        self._refresh_calibration_table()

        if report.warnings:
            bullets = chr(10).join("- " + warning for warning in report.warnings)
            self.calib_warning.setText(
                "These results are not statistically trustworthy:" + chr(10) + bullets
            )
            self.calib_warning.setStyleSheet(
                "background:#7a5c00;color:#ffffff;padding:8px 12px;border-radius:4px;"
            )
            self.calib_warning.setVisible(True)
        else:
            self.calib_warning.setVisible(False)

    def _refresh_calibration_table(self) -> None:
        """Repopulate the bucket table for the selected method and fold."""
        report = getattr(self, "calibration", None)
        if report is None:
            return
        self.calib_bucket_model.set_rows(
            report.calibration_rows(
                self.calib_table_method.currentText(),
                self.calib_table_fold.currentText(),
            )
        )

    def _build_paper_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(banner(PAPER_MODE_BANNER, "#7a2f00"))

        controls = QGroupBox("Paper session")
        form = QFormLayout(controls)
        paper_cfg = self.settings.get("paper", {})
        self.paper_tickers = QLineEdit(", ".join(paper_cfg.get("tickers") or []))
        self.paper_tickers.setPlaceholderText("KXHIGHNY-25AUG31-B70, KXHIGHNY-25AUG31-B72")
        self.paper_source = QComboBox()
        self.paper_source.addItems(["Replay loaded dataset (offline)", "Kalshi live (read-only)"])
        self.paper_interval = QSpinBox()
        self.paper_interval.setRange(5, 3600)
        self.paper_interval.setValue(int(paper_cfg.get("poll_interval_seconds", 60)))
        self.paper_delay = QSpinBox()
        self.paper_delay.setRange(0, 3600)
        self.paper_delay.setValue(int(paper_cfg.get("execution_delay_seconds", 60)))
        self.paper_capital = QDoubleSpinBox()
        self.paper_capital.setRange(100.0, 10_000_000.0)
        self.paper_capital.setSingleStep(1000.0)
        self.paper_capital.setValue(float(paper_cfg.get("initial_capital", 10_000.0)))
        form.addRow("Market tickers", self.paper_tickers)
        form.addRow("Data source", self.paper_source)
        form.addRow("Poll interval (s)", self.paper_interval)
        form.addRow("Execution delay (s)", self.paper_delay)
        form.addRow("Hypothetical capital $", self.paper_capital)

        buttons = QHBoxLayout()
        self.paper_start = QPushButton("Start paper session")
        self.paper_start.clicked.connect(self.start_paper_session)
        self.paper_poll_once = QPushButton("Poll once")
        self.paper_poll_once.clicked.connect(self.poll_paper_session)
        self.paper_poll_once.setEnabled(False)
        self.paper_stop = QPushButton("Stop")
        self.paper_stop.clicked.connect(self.stop_paper_session)
        self.paper_stop.setEnabled(False)
        buttons.addWidget(self.paper_start)
        buttons.addWidget(self.paper_poll_once)
        buttons.addWidget(self.paper_stop)
        buttons.addStretch(1)
        form.addRow(buttons)
        layout.addWidget(controls)

        self.paper_status_label = QLabel("No paper session running.")
        self.paper_status_label.setWordWrap(True)
        self.paper_status_label.setStyleSheet("color:#57606a;")
        layout.addWidget(self.paper_status_label)

        top = QSplitter(Qt.Orientation.Horizontal)
        status_panel = QWidget()
        status_layout = QVBoxLayout(status_panel)
        status_layout.addWidget(QLabel("<b>Session</b>"))
        self.paper_status_model = RowTableModel()
        status_layout.addWidget(table_view(self.paper_status_model))
        top.addWidget(status_panel)

        orders_panel = QWidget()
        orders_layout = QVBoxLayout(orders_panel)
        orders_layout.addWidget(QLabel("<b>Hypothetical orders</b> (never submitted)"))
        self.paper_orders_model = RowTableModel()
        orders_layout.addWidget(table_view(self.paper_orders_model))
        top.addWidget(orders_panel)
        top.setSizes([420, 980])
        layout.addWidget(top, 2)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        fills_panel = QWidget()
        fills_layout = QVBoxLayout(fills_panel)
        fills_layout.addWidget(QLabel("<b>Hypothetical fills</b> (simulated locally)"))
        self.paper_fills_model = RowTableModel()
        fills_layout.addWidget(table_view(self.paper_fills_model))
        bottom.addWidget(fills_panel)

        events_panel = QWidget()
        events_layout = QVBoxLayout(events_panel)
        events_layout.addWidget(QLabel("<b>Session log</b>"))
        self.paper_events_model = RowTableModel()
        events_layout.addWidget(table_view(self.paper_events_model))
        bottom.addWidget(events_panel)
        bottom.setSizes([760, 640])
        layout.addWidget(bottom, 2)

        self.paper_store_label = QLabel("")
        self.paper_store_label.setStyleSheet("color:#57606a;")
        self.paper_store_label.setWordWrap(True)
        layout.addWidget(self.paper_store_label)
        return page

    def start_paper_session(self) -> None:
        """Start a paper session against replayed or live read-only data."""
        settings = self.collect_settings()
        tickers = [t.strip() for t in self.paper_tickers.text().split(",") if t.strip()]
        live = self.paper_source.currentIndex() == 1

        if live:
            if not tickers:
                QMessageBox.warning(
                    self, "No tickers", "Enter at least one Kalshi market ticker."
                )
                return
            source = KalshiQuoteSource(settings.get("venues", {}).get("kalshi", {}))
        else:
            if self.dataset is None:
                QMessageBox.information(self, "No data", "Load a dataset first.")
                return
            source = ReplayQuoteSource(
                self.dataset.markets, self.dataset.quotes, self.dataset.trades
            )
            tickers = tickers or list(self.dataset.market_ids)
            self.paper_tickers.setText(", ".join(tickers))
            self._paper_replay_clock = self.dataset.quotes[0].ts if self.dataset.quotes else None

        store_path = settings.get("paper", {}).get(
            "store_path", "data/paper/paper_trading.duckdb"
        )
        config = PaperConfig(
            tickers=tickers,
            poll_interval_seconds=float(self.paper_interval.value()),
            execution_delay_seconds=float(self.paper_delay.value()),
            initial_capital=float(self.paper_capital.value()),
        )
        try:
            self.paper_store = PaperStore(store_path)
            self.paper_session = PaperSession(
                config, settings, source, store=self.paper_store
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            QMessageBox.critical(self, "Could not start", f"{type(error).__name__}: {error}")
            return

        self.paper_source_is_live = live
        self.paper_start.setEnabled(False)
        self.paper_poll_once.setEnabled(True)
        self.paper_stop.setEnabled(True)
        self.paper_timer.start(int(config.poll_interval_seconds * 1000))
        self.paper_store_label.setText(
            f"Hypothetical orders and fills are written to {resolve_path(store_path)}"
        )
        self.statusBar().showMessage(
            f"PAPER MODE: session {self.paper_session.session_id} started. "
            "No orders are submitted."
        )
        self.poll_paper_session()

    def poll_paper_session(self) -> None:
        """Run one poll of the paper session and refresh the tables."""
        if self.paper_session is None:
            return
        source = self.paper_session.quote_source
        if not self.paper_source_is_live and hasattr(source, "advance_to"):
            # Replay mode walks the loaded dataset forward one poll at a time.
            step = timedelta(seconds=self.paper_session.config.poll_interval_seconds)
            self._paper_replay_clock = (self._paper_replay_clock or datetime.now(UTC)) + step
            source.advance_to(self._paper_replay_clock)
            self.paper_session.clock = lambda: self._paper_replay_clock

        result = self.paper_session.poll()
        self.paper_status_label.setText(presenter.paper_poll_summary(result))
        self._refresh_paper_tables()

    def stop_paper_session(self) -> None:
        """Stop the paper session and leave its records in place."""
        self.paper_timer.stop()
        if self.paper_session is not None:
            self.paper_session.stop()
            self._refresh_paper_tables()
            self.statusBar().showMessage(
                f"PAPER MODE: session {self.paper_session.session_id} stopped."
            )
        self.paper_session = None
        self.paper_start.setEnabled(True)
        self.paper_poll_once.setEnabled(False)
        self.paper_stop.setEnabled(False)

    def _refresh_paper_tables(self) -> None:
        """Reload the persisted paper records into the tables."""
        if self.paper_session is None or self.paper_store is None:
            return
        session_id = self.paper_session.session_id
        self.paper_status_model.set_rows(presenter.paper_status_rows(self.paper_session))
        self.paper_orders_model.set_rows(
            presenter.paper_frame_rows(self.paper_store.orders(session_id))
        )
        self.paper_fills_model.set_rows(
            presenter.paper_frame_rows(self.paper_store.fills(session_id))
        )
        self.paper_events_model.set_rows(
            presenter.paper_frame_rows(self.paper_store.events(session_id))
        )

    def _build_research_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(banner(presenter.RESEARCH_BANNER, "#1f3d63"))

        steps = QGroupBox("Run the pipeline, one step at a time")
        steps_layout = QVBoxLayout(steps)
        row = QHBoxLayout()
        self.research_corpus_choice = QComboBox()
        self.research_corpus_choice.addItems(
            ["Synthetic research corpus (demo)", "Loaded dataset"]
        )
        row.addWidget(QLabel("Data:"))
        row.addWidget(self.research_corpus_choice, 1)
        self.research_model_choice = QComboBox()
        self.research_model_choice.addItems(["logistic_regression"])
        row.addWidget(QLabel("Model:"))
        row.addWidget(self.research_model_choice, 1)
        steps_layout.addLayout(row)

        buttons = QHBoxLayout()
        self.btn_prepare = QPushButton("1. Prepare data")
        self.btn_prepare.clicked.connect(self.run_prepare_data)
        self.btn_train = QPushButton("2. Train && evaluate")
        self.btn_train.clicked.connect(self.run_train_evaluate)
        self.btn_research_backtest = QPushButton("3. Backtest forecasts")
        self.btn_research_backtest.clicked.connect(self.run_research_backtest)
        self.btn_drift = QPushButton("4. Check drift")
        self.btn_drift.clicked.connect(self.run_drift_check)
        for button in (
            self.btn_prepare, self.btn_train,
            self.btn_research_backtest, self.btn_drift,
        ):
            buttons.addWidget(button)
        for button in (self.btn_train, self.btn_research_backtest, self.btn_drift):
            button.setEnabled(False)
        buttons.addStretch(1)
        steps_layout.addLayout(buttons)

        self.research_status = QLabel("Press \u201cPrepare data\u201d to begin.")
        self.research_status.setWordWrap(True)
        self.research_status.setStyleSheet("color:#57606a;")
        steps_layout.addWidget(self.research_status)
        outer.addWidget(steps)

        self.research_tabs = QTabWidget()

        # --- Data -------------------------------------------------------
        data_page = QWidget()
        data_layout = QHBoxLayout(data_page)
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Where this data came from</b>"))
        self.research_corpus_model = RowTableModel()
        left.addWidget(table_view(self.research_corpus_model))
        left.addWidget(QLabel("<b>Can it support training?</b>"))
        self.research_readiness_model = RowTableModel()
        left.addWidget(table_view(self.research_readiness_model))
        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Dataset built</b>"))
        self.research_dataset_model = RowTableModel()
        right.addWidget(table_view(self.research_dataset_model))
        right.addWidget(QLabel("<b>Feature dictionary</b>"))
        self.research_features_model = RowTableModel()
        self.research_features_model.set_rows(feature_dictionary_rows())
        right.addWidget(table_view(self.research_features_model))
        data_layout.addLayout(left, 1)
        data_layout.addLayout(right, 1)
        self.research_tabs.addTab(data_page, "1. Data")

        # --- Model ------------------------------------------------------
        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        self.research_verdict = QLabel("")
        self.research_verdict.setWordWrap(True)
        self.research_verdict.setFont(QFont("", 11, QFont.Weight.Bold))
        model_layout.addWidget(self.research_verdict)
        model_layout.addWidget(
            QLabel("<b>Candidate vs frozen baselines</b> (identical out-of-sample rows)")
        )
        self.research_comparison_model = RowTableModel()
        model_layout.addWidget(table_view(self.research_comparison_model))
        model_layout.addWidget(QLabel("<b>Per fold</b> (variation matters more than the average)"))
        self.research_folds_model = RowTableModel()
        model_layout.addWidget(table_view(self.research_folds_model))
        self.research_tabs.addTab(model_page, "2. Model")

        # --- Calibration ------------------------------------------------
        calib_page = QWidget()
        calib_layout = QHBoxLayout(calib_page)
        chart_side = QVBoxLayout()
        chart_side.addWidget(QLabel("<b>Reliability diagram</b>"))
        self.research_reliability_view = reliability_chart(
            {}, "Predicted probability vs observed YES rate"
        )
        chart_side.addWidget(self.research_reliability_view)
        table_side = QVBoxLayout()
        table_side.addWidget(QLabel("<b>Uncalibrated vs calibrated vs market</b>"))
        self.research_calib_compare_model = RowTableModel()
        table_side.addWidget(table_view(self.research_calib_compare_model))
        table_side.addWidget(QLabel("<b>Calibration table by probability bin</b>"))
        self.research_calib_table_model = RowTableModel()
        table_side.addWidget(table_view(self.research_calib_table_model))
        calib_layout.addLayout(chart_side, 1)
        calib_layout.addLayout(table_side, 1)
        self.research_tabs.addTab(calib_page, "3. Calibration")

        # --- Backtest ---------------------------------------------------
        bt_page = QWidget()
        bt_layout = QVBoxLayout(bt_page)
        bt_layout.addWidget(
            QLabel("<b>Simulated on out-of-sample forecasts only</b>, against a flat baseline")
        )
        self.research_bt_metrics_model = RowTableModel()
        bt_layout.addWidget(table_view(self.research_bt_metrics_model))
        bt_layout.addWidget(
            QLabel("<b>Why decisions did not become trades</b>")
        )
        self.research_skips_model = RowTableModel()
        bt_layout.addWidget(table_view(self.research_skips_model))
        self.research_tabs.addTab(bt_page, "4. Backtest")

        # --- Drift ------------------------------------------------------
        drift_page = QWidget()
        drift_layout = QVBoxLayout(drift_page)
        self.research_drift_summary = QLabel("Not checked yet.")
        self.research_drift_summary.setWordWrap(True)
        drift_layout.addWidget(self.research_drift_summary)
        self.research_drift_model = RowTableModel()
        drift_layout.addWidget(table_view(self.research_drift_model))
        self.research_tabs.addTab(drift_page, "5. Drift")

        # --- Approval ---------------------------------------------------
        approve_page = QWidget()
        approve_layout = QVBoxLayout(approve_page)
        approve_layout.addWidget(
            banner(
                "Approval marks a candidate \u201capproved for future paper-trading "
                "evaluation\u201d. It deploys nothing, starts nothing, and places no "
                "order. Every check below must pass first.",
                "#5a4200",
            )
        )
        approve_layout.addWidget(QLabel("<b>Promotion gate</b>"))
        self.research_gate_model = RowTableModel()
        approve_layout.addWidget(table_view(self.research_gate_model))
        form = QFormLayout()
        self.approve_name = QLineEdit()
        self.approve_name.setPlaceholderText("Your name, recorded with the decision")
        self.approve_reason = QLineEdit()
        self.approve_reason.setPlaceholderText(
            "Why this candidate is worth evaluating (at least 10 characters)"
        )
        form.addRow("Approved by", self.approve_name)
        form.addRow("Rationale", self.approve_reason)
        approve_layout.addLayout(form)
        approve_buttons = QHBoxLayout()
        self.btn_approve = QPushButton("Approve for future paper-trading evaluation")
        self.btn_approve.clicked.connect(self.approve_candidate)
        self.btn_approve.setEnabled(False)
        self.btn_reject = QPushButton("Record a rejection")
        self.btn_reject.clicked.connect(self.reject_candidate)
        self.btn_reject.setEnabled(False)
        approve_buttons.addWidget(self.btn_approve)
        approve_buttons.addWidget(self.btn_reject)
        approve_buttons.addStretch(1)
        approve_layout.addLayout(approve_buttons)
        approve_layout.addWidget(QLabel("<b>Recorded runs</b>"))
        self.research_runs_model = RowTableModel()
        approve_layout.addWidget(table_view(self.research_runs_model))
        self.research_tabs.addTab(approve_page, "6. Approval")

        outer.addWidget(self.research_tabs, 1)
        return page

    # -- research actions --------------------------------------------------

    def _research_corpus(self):
        """The dataset the pipeline should run on."""
        if self.research_corpus_choice.currentIndex() == 1 and self.dataset is not None:
            return self.dataset
        return build_research_corpus()

    def run_prepare_data(self) -> None:
        """Stage 1: build causal features and settlement labels."""
        self.research_status.setText("Building causal features and settlement labels...")
        QApplication.processEvents()
        try:
            self.research_corpus_data = self._research_corpus()
            self.prepared = prepare_data(
                self.research_corpus_data, self.collect_settings(), self.registry
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            QMessageBox.critical(self, "Data preparation failed", str(error))
            self.research_status.setText(f"Failed: {error}")
            return

        self.research_corpus_model.set_rows(presenter.corpus_rows(self.prepared))
        self.research_readiness_model.set_rows(presenter.readiness_rows(self.prepared))
        self.research_dataset_model.set_rows(presenter.dataset_summary_rows(self.prepared))
        self._refresh_research_runs()
        ready = self.prepared.readiness
        self.btn_train.setEnabled(ready.ok)
        self._set_research_status(
            f"Built {len(self.prepared.dataset.examples)} rows across "
            f"{self.prepared.dataset.effective_n} resolved market(s). "
            + ("Ready to train." if ready.ok
               else "TRAINING BLOCKED: " + " ".join(ready.blockers))
        )
        self.research_tabs.setCurrentIndex(0)

    def run_train_evaluate(self) -> None:
        """Stage 2: walk-forward training, calibration and baseline comparison."""
        if self.prepared is None:
            return
        self.research_status.setText("Running walk-forward training and calibration...")
        QApplication.processEvents()
        try:
            self.training = train_and_evaluate(
                self.prepared, self.collect_settings(),
                self.research_model_choice.currentText(), self.registry,
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            QMessageBox.critical(self, "Training failed", str(error))
            self.research_status.setText(f"Failed: {error}")
            return

        outcome = self.training
        self.research_comparison_model.set_rows(presenter.comparison_rows(outcome))
        self.research_folds_model.set_rows(fold_rows(outcome.report))
        self.research_verdict.setText(outcome.comparison.verdict())

        probabilities, outcomes, _ = outcome.report.pooled()
        self.research_calib_table_model.set_rows(calibration_rows(probabilities, outcomes))
        self.research_calib_compare_model.set_rows(calibration_comparison(outcome.report))
        points = [
            (p["mean_predicted"], p["observed_frequency"])
            for p in reliability_points(probabilities, outcomes)
            if p["mean_predicted"] is not None and p["observed_frequency"] is not None
        ]
        update_reliability_chart(
            self.research_reliability_view, {outcome.report.model_name: points}
        )
        self.research_gate_model.set_rows(presenter.gate_rows(outcome))
        self._refresh_research_runs()

        gate_ok = outcome.gate.passed
        self.btn_approve.setEnabled(gate_ok)
        self.btn_reject.setEnabled(True)
        self.btn_research_backtest.setEnabled(True)
        self.btn_drift.setEnabled(True)
        warnings = "; ".join(outcome.candidate.warnings[:2])
        self._set_research_status(
            outcome.comparison.verdict() + (f" Note: {warnings}" if warnings else "")
        )
        self.research_tabs.setCurrentIndex(1)

    def run_research_backtest(self) -> None:
        """Stage 3: simulate the out-of-sample forecasts."""
        if self.prepared is None or self.training is None:
            return
        self.research_status.setText("Simulating out-of-sample forecasts...")
        QApplication.processEvents()
        try:
            self.research_backtest = run_model_backtest(
                self.prepared, self.training, self.research_corpus_data,
                self.collect_settings(), self.registry,
            )
        except Exception as error:  # noqa: BLE001 - surfaced in the UI
            QMessageBox.critical(self, "Backtest failed", str(error))
            self.research_status.setText(f"Failed: {error}")
            return

        result = self.research_backtest
        self.research_bt_metrics_model.set_rows(presenter.backtest_metric_rows(result))
        self.research_skips_model.set_rows(skip_summary(result.skips))
        self._refresh_research_runs()
        self._set_research_status(
            f"Simulated {result.metrics.get('n_trades', 0)} trade(s) from "
            f"{result.n_forecasts} out-of-sample forecast(s); "
            f"{len(result.skips)} decision(s) declined. Simulation only."
        )
        self.research_tabs.setCurrentIndex(3)

    def run_drift_check(self) -> None:
        """Stage 4: compare the recent window against the training reference."""
        if self.prepared is None or self.training is None:
            return
        self.research_status.setText("Measuring drift...")
        QApplication.processEvents()
        report = check_drift(
            self.prepared, self.training, self.collect_settings(), self.registry
        )
        self.research_drift_model.set_rows(presenter.drift_rows(report))
        self.research_drift_summary.setText(report.summary())
        self._refresh_research_runs()
        self._set_research_status(report.summary())
        self.research_tabs.setCurrentIndex(4)

    def approve_candidate(self) -> None:
        """Record an explicit human approval. Nothing is deployed or started."""
        if self.training is None or not self.training.run_id:
            return
        try:
            self.registry.request_promotion(
                self.training.run_id, self.training.gate,
                self.approve_name.text(), self.approve_reason.text(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Approval refused", str(error))
            return
        QMessageBox.information(
            self, "Recorded",
            "Recorded as approved for future paper-trading evaluation.\n\n"
            "Nothing has been deployed or started. This is a label on a stored "
            "run; no order can be placed by this application.",
        )
        self._refresh_research_runs()

    def reject_candidate(self) -> None:
        """Record an explicit decision not to promote."""
        if self.training is None or not self.training.run_id:
            return
        self.registry.record_rejection(
            self.training.run_id,
            self.approve_name.text() or "unspecified",
            self.approve_reason.text() or "No rationale given.",
        )
        self._refresh_research_runs()

    def _set_research_status(self, text: str) -> None:
        """Show a status line, keeping the full text available on hover."""
        self.research_status.setText(text)
        self.research_status.setToolTip(text)

    def _refresh_research_runs(self) -> None:
        """Reload the recorded runs table."""
        frame = self.registry.runs(limit=50)
        if frame.empty:
            self.research_runs_model.set_rows([])
            return
        columns = [
            "run_id", "created_at", "kind", "model_name", "effective_n",
            "n_forecasts", "data_version", "is_synthetic",
        ]
        self.research_runs_model.set_rows(
            frame[[c for c in columns if c in frame.columns]].to_dict("records")
        )

    def _build_audit_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Event types:"))
        self.audit_checks: dict[str, QCheckBox] = {}
        for kind in AUDIT_KINDS:
            box = QCheckBox(kind)
            box.setChecked(kind in DEFAULT_AUDIT_KINDS)
            box.toggled.connect(self._refresh_audit)
            filter_row.addWidget(box)
            self.audit_checks[kind] = box
        filter_row.addStretch(1)
        layout.addLayout(filter_row)
        self.audit_model = RowTableModel()
        layout.addWidget(table_view(self.audit_model), 1)
        self.audit_count = QLabel("Run a backtest to populate the audit log.")
        self.audit_count.setStyleSheet("color:#57606a;")
        layout.addWidget(self.audit_count)
        return page

    def _build_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(banner(
            "Read-only historical data. This tab issues HTTP GET requests to Kalshi's "
            "public market-data endpoints. It sends no credentials and cannot place, "
            "amend or cancel an order.", "#1f3d5c",
        ))

        box = QGroupBox("Fetch Kalshi history")
        form = QFormLayout(box)
        self.kalshi_tickers = QLineEdit()
        self.kalshi_tickers.setPlaceholderText("KXHIGHNY-25AUG31-B70, KXHIGHNY-25AUG31-B72")
        now = datetime.now(UTC)
        self.kalshi_start = QDateTimeEdit()
        self.kalshi_start.setDateTime(now - timedelta(days=7))
        self.kalshi_start.setCalendarPopup(True)
        self.kalshi_end = QDateTimeEdit()
        self.kalshi_end.setDateTime(now)
        self.kalshi_end.setCalendarPopup(True)
        self.kalshi_interval = QComboBox()
        self.kalshi_interval.addItems(["1 minute", "1 hour", "1 day"])
        self.kalshi_interval.setCurrentIndex(1)
        self.kalshi_historical = QCheckBox(
            "Use the /historical tier (markets settled more than ~3 months ago)"
        )
        self.kalshi_historical.setChecked(True)
        self.kalshi_trades = QCheckBox("Also fetch the public trade tape")
        self.kalshi_trades.setChecked(True)
        self.kalshi_depth_proxy = QCheckBox(
            "Use candle volume as a depth PROXY (candlesticks carry no book depth)"
        )
        self.kalshi_depth_proxy.setChecked(
            bool(self.settings.get("venues", {}).get("kalshi", {}).get("depth_from_volume", True))
        )
        fetch_button = QPushButton("Fetch (read-only)")
        fetch_button.clicked.connect(self.fetch_kalshi)
        form.addRow("Market tickers", self.kalshi_tickers)
        form.addRow("Start (UTC)", self.kalshi_start)
        form.addRow("End (UTC)", self.kalshi_end)
        form.addRow("Candle interval", self.kalshi_interval)
        form.addRow(self.kalshi_historical)
        form.addRow(self.kalshi_trades)
        form.addRow(self.kalshi_depth_proxy)
        form.addRow(fetch_button)
        layout.addWidget(box)

        back_to_demo = QPushButton("Switch back to DEMO DATA")
        back_to_demo.clicked.connect(lambda: self.load_demo_data())
        layout.addWidget(back_to_demo)

        layout.addWidget(QLabel("<b>Dataset provenance</b>"))
        self.provenance_view = QPlainTextEdit()
        self.provenance_view.setReadOnly(True)
        layout.addWidget(self.provenance_view, 1)
        return page

    # -- data loading ------------------------------------------------------

    def collect_settings(self) -> dict[str, Any]:
        """Read every control back into a settings dictionary."""
        settings = get_settings()
        settings["data"].update({
            "seed": int(self.seed_input.value()),
            "n_markets": int(self.markets_input.value()),
            "n_steps": int(self.steps_input.value()),
            "duckdb_path": (
                f"data/processed/demo_{self.seed_input.value()}_"
                f"{self.markets_input.value()}_{self.steps_input.value()}.duckdb"
            ),
        })
        settings["strategy"].update({
            "min_edge": float(self.min_edge.value()),
            "base_size": float(self.base_size.value()),
            "max_order_size": float(self.max_order.value()),
            "require_positive_ev": bool(self.require_ev.isChecked()),
        })
        settings["risk"].update({
            "kill_switch": bool(self.kill_switch.isChecked()),
            "max_position_size": float(self.max_position.value()),
            "max_event_exposure": float(self.max_event_exposure.value()),
            "max_total_exposure": float(self.max_total_exposure.value()),
            "max_spread": float(self.max_spread.value()),
            "max_slippage": float(self.max_slippage.value()),
            "max_data_age_seconds": float(self.max_data_age.value()),
            "max_drawdown": float(self.max_drawdown.value()),
        })
        settings["fills"].update({
            "fee_rate": float(self.fee_rate.value()),
            "fee_model": self.fee_model.currentText(),
            "base_slippage": float(self.base_slippage.value()),
            "impact_coefficient": float(self.impact.value()),
            "allow_partial_fills": bool(self.allow_partial.isChecked()),
        })
        settings["backtest"]["initial_capital"] = float(self.capital.value())
        self.settings = settings
        return settings

    def load_demo_data(self, refresh: bool = False) -> None:
        """Generate or load the synthetic demo dataset in the background."""
        settings = self.collect_settings()
        self._start(DemoDataWorker(settings, refresh=refresh), self._on_dataset_ready)

    def fetch_kalshi(self) -> None:
        """Fetch read-only Kalshi history for the tickers in the Data tab."""
        tickers = [t.strip() for t in self.kalshi_tickers.text().split(",") if t.strip()]
        if not tickers:
            QMessageBox.warning(
                self, "No tickers",
                "Enter at least one Kalshi market ticker, for example "
                "KXHIGHNY-25AUG31-B70.",
            )
            return
        start = self.kalshi_start.dateTime().toPython().replace(tzinfo=UTC)
        end = self.kalshi_end.dateTime().toPython().replace(tzinfo=UTC)
        if end <= start:
            QMessageBox.warning(self, "Invalid range", "The end time must be after the start.")
            return

        config = dict(self.settings.get("venues", {}).get("kalshi", {}))
        config["depth_from_volume"] = bool(self.kalshi_depth_proxy.isChecked())
        spec = KalshiFetchSpec(
            tickers=tickers,
            start_ts=int(start.timestamp()),
            end_ts=int(end.timestamp()),
            period_interval={0: 1, 1: 60, 2: 1440}[self.kalshi_interval.currentIndex()],
            include_trades=bool(self.kalshi_trades.isChecked()),
            historical=bool(self.kalshi_historical.isChecked()),
        )
        self._start(KalshiFetchWorker(config, spec), self._on_dataset_ready)

    def run_backtest(self) -> None:
        """Run a backtest over the loaded dataset."""
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Load a dataset first.")
            return
        settings = self.collect_settings()
        self.run_button.setEnabled(False)
        self._start(BacktestWorker(self.dataset, settings), self._on_result_ready)

    def _start(self, worker, on_finished) -> None:
        self.progress.setVisible(True)
        worker.progress.connect(self.statusBar().showMessage)
        worker.finished.connect(on_finished)
        worker.failed.connect(self._on_worker_failed)
        self._runner = WorkerRunner(worker, self)
        self._runner.start()

    def _on_worker_failed(self, message: str) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.statusBar().showMessage("Failed.")
        QMessageBox.critical(self, "Operation failed", message)

    # -- refresh -----------------------------------------------------------

    def _on_dataset_ready(self, dataset: MarketDataset) -> None:
        self.progress.setVisible(False)
        self.dataset = dataset
        self.result = None
        colour = "#7a5c00" if dataset.is_demo else "#1f3d5c"
        self.provenance.setText(presenter.provenance_banner(dataset))
        self.provenance.setStyleSheet(
            f"background:{colour};color:#ffffff;padding:8px 12px;"
            "border-radius:4px;font-weight:600;"
        )

        summary = dataset.summary()
        lines = [f"{key}: {value}" for key, value in summary.items() if key != "notes"]
        if dataset.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in dataset.notes)
        self.provenance_view.setPlainText("\n".join(lines))

        self._timestamps = sorted({q.ts for q in dataset.quotes})
        self.as_of_slider.setEnabled(bool(self._timestamps))
        self.as_of_slider.setRange(0, max(0, len(self._timestamps) - 1))
        self.as_of_slider.setValue(max(0, len(self._timestamps) - 1))

        current = self.decision_market.currentText()
        self.decision_market.blockSignals(True)
        self.decision_market.clear()
        self.decision_market.addItems(dataset.market_ids)
        if current in dataset.market_ids:
            self.decision_market.setCurrentText(current)
        self.decision_market.blockSignals(False)

        self.statusBar().showMessage(
            f"Loaded {len(dataset.markets)} markets, {len(dataset.quotes):,} quotes, "
            f"{len(dataset.trades):,} trades. Simulation only."
        )
        self._refresh_all()

    def _on_as_of_changed(self, _value: int) -> None:
        self._refresh_all()

    @property
    def as_of(self) -> datetime | None:
        """Currently selected as-of timestamp."""
        if not getattr(self, "_timestamps", None):
            return None
        index = min(self.as_of_slider.value(), len(self._timestamps) - 1)
        return self._timestamps[index]

    def _refresh_all(self) -> None:
        as_of = self.as_of
        if self.dataset is None or as_of is None:
            return
        self.as_of_label.setText(as_of.strftime("%Y-%m-%d %H:%M UTC"))
        self.market_model.set_rows(presenter.market_rows(self.dataset, as_of))
        self.market_table.resizeColumnsToContents()
        self._refresh_price_chart()
        self._refresh_decision_panel()

    def _refresh_price_chart(self) -> None:
        as_of = self.as_of
        if self.dataset is None or as_of is None or not self.dataset.market_ids:
            return
        market_id = self.decision_market.currentText() or self.dataset.market_ids[0]
        series = presenter.price_series(self.dataset, market_id, as_of)
        update_chart(
            self.price_chart,
            {"Bid": series["bid"], "Ask": series["ask"], "Mid": series["mid"]},
            {"Trades": series["trades"]},
            y_title="Implied probability",
            y_range=(0.0, 1.0),
        )

    def _refresh_decision_panel(self) -> None:
        as_of = self.as_of
        market_id = self.decision_market.currentText()
        if self.dataset is None or as_of is None or not market_id:
            return
        settings = self.collect_settings()
        market = self.dataset.market(market_id)
        preview = presenter.decision_preview(self.dataset, market, as_of, settings)

        self.question_label.setText(market.question)
        self.forecast_model.set_rows(presenter.forecast_rows(preview))
        self.provenance_model.set_rows(
            presenter.forecast_provenance_rows(preview.forecasts)
        )

        _clear_layout(self.consensus_row)
        consensus = preview.consensus
        probability = "n/a" if not consensus or consensus.probability is None else (
            f"{consensus.probability:.3f}"
        )
        self.consensus_row.addWidget(metric_tile("Blended probability", probability))
        self.consensus_row.addWidget(
            metric_tile("Disagreement", f"{consensus.disagreement:.3f}" if consensus else "-")
        )
        self.consensus_row.addWidget(
            metric_tile("Contributors", str(consensus.n_contributors) if consensus else "0")
        )
        self.consensus_row.addStretch(1)
        self.excluded_label.setText(
            "Excluded from the blend: "
            + ", ".join(f"{k} ({v})" for k, v in consensus.excluded.items())
            if consensus and consensus.excluded else ""
        )

        report = preview.liquidity
        _clear_layout(self.liquidity_grid)
        if report is not None:
            age = report.data_age_seconds
            tiles = [
                ("Spread", "n/a" if report.spread is None else f"{report.spread:.3f}"),
                ("Depth", f"{report.depth:.0f}"),
                ("Quote age (s)", "n/a" if age is None else f"{age:.0f}"),
                ("Stale", "YES" if report.is_stale else "no"),
            ]
            for column, (caption, value) in enumerate(tiles):
                self.liquidity_grid.addWidget(metric_tile(caption, value), 0, column)
            self.liquidity_warnings.setPlainText("\n".join(report.warnings) or "No warnings.")

        _clear_layout(self.proposal_row)
        proposal = preview.proposal
        if proposal is None:
            self.proposal_note.setText(preview.no_proposal_reason)
            self.decision_banner.setText("")
            self.decision_banner.setStyleSheet("")
            self.reason_model.set_rows([])
            self.reason_table.setVisible(False)
            self._refresh_price_chart()
            return

        for caption, value in [
            ("Side", proposal.side.value),
            ("Size", f"{proposal.size:.0f}"),
            ("Reference price", f"{proposal.reference_price:.3f}"),
            ("Edge", f"{proposal.edge:+.3f}"),
            ("EV after costs", f"${proposal.expected_value_total:+,.2f}"),
        ]:
            self.proposal_row.addWidget(metric_tile(caption, value))
        self.proposal_row.addStretch(1)
        self.proposal_note.setText(
            f"Estimated slippage {proposal.expected_slippage:.4f}; fee "
            f"{proposal.expected_fee_per_contract:.4f}/contract; EV "
            f"{proposal.expected_value_per_contract:+.4f}/contract. {proposal.rationale}"
        )

        decision = preview.decision
        if decision is not None:
            colour = ACTION_COLOR[decision.action]
            self.decision_banner.setText(
                f"Risk decision: {decision.action.value} - approved size "
                f"{decision.approved_size:.0f} of {decision.original_size:.0f}"
            )
            self.decision_banner.setStyleSheet(
                f"border-left:5px solid {colour};padding:8px 12px;font-weight:600;"
                "background:rgba(128,128,128,0.10);"
            )
            self.reason_model.set_rows(presenter.risk_reason_rows(decision))
            self.reason_table.setVisible(True)
            self.reason_table.resizeColumnsToContents()
        self._refresh_price_chart()

    def _on_result_ready(self, result: Any) -> None:
        self.progress.setVisible(False)
        self.run_button.setEnabled(True)
        self.result = result

        _clear_layout(self.metric_row)
        for caption, value in presenter.headline_metrics(result):
            self.metric_row.addWidget(metric_tile(caption, value))
        self.metric_row.addStretch(1)

        series = presenter.equity_series(result)
        update_chart(
            self.equity_chart,
            {"Equity": series["equity"], "Peak": series["peak"]},
            y_title="Equity $",
        )
        update_chart(
            self.drawdown_chart,
            {"Drawdown": series["drawdown"]},
            y_title="DD %",
            show_legend=False,
        )

        summary = MetricsSummary.model_validate(result.metrics)
        self.metrics_model.set_rows(metrics_rows(summary))
        self.calibration_model.set_rows(presenter.calibration_rows(result))
        self.decisions_model.set_rows(result.decisions_frame().to_dict("records"))
        self._refresh_audit()
        sources = presenter.calibration_sources(result)
        current = self.calib_source.currentText()
        self.calib_source.blockSignals(True)
        self.calib_source.clear()
        self.calib_source.addItems(sources)
        if current in sources:
            self.calib_source.setCurrentText(current)
        self.calib_source.blockSignals(False)
        self.statusBar().showMessage(
            f"Backtest complete: {summary.n_trades} trades, "
            f"{summary.n_rejected} rejected by risk. Simulation only."
        )

    def _refresh_audit(self) -> None:
        if self.result is None:
            return
        kinds = {kind for kind, box in self.audit_checks.items() if box.isChecked()}
        rows = presenter.audit_rows(self.result, kinds)
        self.audit_model.set_rows(rows)
        self.audit_count.setText(
            f"{len(rows):,} of {len(self.result.audit_log):,} audit rows. Every proposal, "
            "risk decision, fill, rejection and settlement is logged."
        )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "HFT Hivemind - prediction-market research sandbox\n\n"
            "Simulation only. No orders are placed, no credentials are used, and no "
            "claim of profitability is made.\n\n"
            "Data is either fabricated demo data or read-only historical data fetched "
            "from public venue endpoints.",
        )


def _clear_layout(layout) -> None:
    """Remove and delete every widget in a layout."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def main() -> int:
    """Start the desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("HFT Hivemind")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
