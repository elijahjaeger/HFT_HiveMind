"""Qt table model for the row dictionaries produced by ``app.presenter``."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


def format_cell(value: Any) -> str:
    """Render one cell value for display."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


class RowTableModel(QAbstractTableModel):
    """A read-only table over a list of uniform row dictionaries."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        columns: list[str] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = []
        self.set_rows(rows or [], columns)

    def set_rows(
        self, rows: list[dict[str, Any]], columns: list[str] | None = None
    ) -> None:
        """Replace the contents of the table."""
        self.beginResetModel()
        self._rows = list(rows)
        if columns is not None:
            self._columns = list(columns)
        elif self._rows:
            self._columns = list(self._rows[0].keys())
        else:
            self._columns = []
        self.endResetModel()

    @property
    def columns(self) -> list[str]:
        """Column keys currently displayed."""
        return list(self._columns)

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        """Number of rows (Qt API)."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:  # noqa: N802
        """Number of columns (Qt API)."""
        if parent is not None and parent.isValid():
            return 0
        return len(self._columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Cell contents for a given role (Qt API)."""
        if not index.isValid():
            return None
        value = self._rows[index.row()].get(self._columns[index.column()])
        if role == Qt.ItemDataRole.DisplayRole:
            return format_cell(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value) if value is not None else ""
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.UserRole:
            return value
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """Column and row headers (Qt API)."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._columns):
                return self._columns[section].replace("_", " ")
            return None
        return section + 1

    def row(self, index: int) -> dict[str, Any]:
        """Return the underlying dictionary for one row."""
        return dict(self._rows[index])
