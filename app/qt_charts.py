"""QtCharts helpers: time-series and scatter plots drawn with native widgets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from PySide6.QtCharts import (
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtGui import QColor, QPainter

Point = tuple[datetime, float]

COLORS = {
    "uncalibrated": "#8c959f",
    "isotonic": "#0969da",
    "logistic": "#bc4c00",
    "perfect calibration": "#57606a",
    "bid": "#1a7f37",
    "ask": "#cf222e",
    "mid": "#0969da",
    "trades": "#8c959f",
    "equity": "#0969da",
    "peak": "#8250df",
    "drawdown": "#cf222e",
}


def _msecs(moment: datetime) -> float:
    """Convert a datetime to milliseconds since the epoch for Qt axes."""
    return float(moment.timestamp() * 1000.0)


def _apply_points(series: QLineSeries | QScatterSeries, points: Sequence[Point]) -> None:
    for moment, value in points:
        if value is None:
            continue
        series.append(_msecs(moment), float(value))


def time_series_chart(
    title: str,
    line_points: dict[str, Sequence[Point]],
    scatter_points: dict[str, Sequence[Point]] | None = None,
    y_title: str = "",
    y_range: tuple[float, float] | None = None,
    show_legend: bool = True,
) -> QChartView:
    """Build a chart view with one line per entry in ``line_points``.

    Args:
        title: Chart title.
        line_points: Series name to ``(timestamp, value)`` points.
        scatter_points: Optional point series drawn as markers.
        y_title: Label for the value axis.
        y_range: Fixed value-axis range; auto-scaled when omitted.
        show_legend: Hide the legend to give a short chart more plot area;
            Qt elides axis labels to "..." when the plot area is too small.

    Returns:
        A ready-to-embed :class:`QChartView`.
    """
    chart = QChart()
    chart.setTitle(title)
    chart.legend().setVisible(show_legend)
    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

    x_axis = QDateTimeAxis()
    x_axis.setFormat("MMM d HH:mm")
    x_axis.setTitleText("Time (UTC)")
    x_axis.setTickCount(6)
    chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)

    y_axis = QValueAxis()
    y_axis.setTitleText(y_title)
    if y_range is not None:
        y_axis.setRange(*y_range)
    chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

    lows: list[float] = []
    highs: list[float] = []

    for name, points in line_points.items():
        series = QLineSeries()
        series.setName(name)
        color = QColor(COLORS.get(name.lower(), "#0969da"))
        pen = series.pen()
        pen.setColor(color)
        pen.setWidth(2 if name.lower() in {"mid", "equity"} else 1)
        series.setPen(pen)
        _apply_points(series, points)
        chart.addSeries(series)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
        values = [v for _, v in points if v is not None]
        if values:
            lows.append(min(values))
            highs.append(max(values))

    for name, points in (scatter_points or {}).items():
        series = QScatterSeries()
        series.setName(name)
        series.setMarkerSize(5.0)
        series.setColor(QColor(COLORS.get(name.lower(), "#8c959f")))
        series.setBorderColor(QColor(COLORS.get(name.lower(), "#8c959f")))
        _apply_points(series, points)
        chart.addSeries(series)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)

    if y_range is None and lows and highs:
        low, high = min(lows), max(highs)
        pad = max((high - low) * 0.08, 1e-6)
        y_axis.setRange(low - pad, high + pad)
        y_range = (low - pad, high + pad)

    # Pin the label format, otherwise a narrow range produces very long labels
    # that Qt elides to "..." instead of shrinking.
    if y_range is not None:
        span = abs(y_range[1] - y_range[0])
        y_axis.setLabelFormat("%.0f" if span >= 50 else "%.2f")
        y_axis.setTickCount(6)

    all_points = [p for points in line_points.values() for p in points]
    all_points += [p for points in (scatter_points or {}).values() for p in points]
    if all_points:
        first = min(ts for ts, _ in all_points)
        last = max(ts for ts, _ in all_points)
        x_axis.setRange(
            QDateTime.fromMSecsSinceEpoch(int(_msecs(first))),
            QDateTime.fromMSecsSinceEpoch(int(_msecs(last))),
        )

    view = QChartView(chart)
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    return view


def update_chart(
    view: QChartView,
    line_points: dict[str, Sequence[Point]],
    scatter_points: dict[str, Sequence[Point]] | None = None,
    y_title: str = "",
    y_range: tuple[float, float] | None = None,
    show_legend: bool = True,
) -> None:
    """Replace the contents of an existing chart view in place."""
    title = view.chart().title()
    fresh = time_series_chart(
        title, line_points, scatter_points, y_title, y_range, show_legend
    )
    old = view.chart()
    view.setChart(fresh.chart())
    old.deleteLater()


def reliability_chart(
    series: dict[str, Sequence[tuple[float, float]]],
    title: str = "Reliability diagram",
) -> QChartView:
    """Plot predicted probability against realised frequency.

    A perfectly calibrated method lies on the diagonal. Points above it are
    under-confident (things happened more often than forecast); points below are
    over-confident. Only populated buckets are plotted, so the curve never
    implies data where there is none.

    Args:
        series: Method name to ``(mean predicted, observed frequency)`` points.
        title: Chart title.

    Returns:
        A ready-to-embed :class:`QChartView`.
    """
    chart = QChart()
    chart.setTitle(title)
    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

    x_axis = QValueAxis()
    x_axis.setTitleText("Mean predicted probability")
    x_axis.setRange(0.0, 1.0)
    x_axis.setTickCount(6)
    x_axis.setLabelFormat("%.1f")
    chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)

    y_axis = QValueAxis()
    y_axis.setTitleText("Observed frequency")
    y_axis.setRange(0.0, 1.0)
    y_axis.setTickCount(6)
    y_axis.setLabelFormat("%.1f")
    chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

    diagonal = QLineSeries()
    diagonal.setName("Perfect calibration")
    diagonal.append(0.0, 0.0)
    diagonal.append(1.0, 1.0)
    pen = diagonal.pen()
    pen.setColor(QColor(COLORS["perfect calibration"]))
    pen.setStyle(Qt.PenStyle.DashLine)
    pen.setWidth(1)
    diagonal.setPen(pen)
    chart.addSeries(diagonal)
    diagonal.attachAxis(x_axis)
    diagonal.attachAxis(y_axis)

    for name, points in series.items():
        ordered = sorted(points)
        color = QColor(COLORS.get(name.lower(), "#0969da"))
        line = QLineSeries()
        line.setName(name)
        line_pen = line.pen()
        line_pen.setColor(color)
        line_pen.setWidth(2)
        line.setPen(line_pen)
        for x, y in ordered:
            line.append(float(x), float(y))
        chart.addSeries(line)
        line.attachAxis(x_axis)
        line.attachAxis(y_axis)

        markers = QScatterSeries()
        markers.setName(f"{name} buckets")
        markers.setMarkerSize(8.0)
        markers.setColor(color)
        markers.setBorderColor(color)
        for x, y in ordered:
            markers.append(float(x), float(y))
        chart.addSeries(markers)
        markers.attachAxis(x_axis)
        markers.attachAxis(y_axis)
        chart.legend().markers(markers)[0].setVisible(False)

    view = QChartView(chart)
    view.setRenderHint(QPainter.RenderHint.Antialiasing)
    return view


def update_reliability_chart(
    view: QChartView, series: dict[str, Sequence[tuple[float, float]]]
) -> None:
    """Replace the contents of an existing reliability chart in place."""
    fresh = reliability_chart(series, view.chart().title())
    old = view.chart()
    view.setChart(fresh.chart())
    old.deleteLater()
