"""
Table widgets for AI-ATS application.

Provides styled table components for displaying data.
"""

from typing import Any, Callable, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QLineEdit,
    QAbstractItemView,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.utils.constants import COLORS


class StyledTable(QTableWidget):
    """
    Styled table widget with consistent appearance.

    Provides a clean, modern table design with hover effects
    and selection highlighting.
    """

    row_clicked = pyqtSignal(int)  # Emits row index when clicked
    row_double_clicked = pyqtSignal(int)  # Emits row index on double-click
    selection_count_changed = pyqtSignal(int)  # Emits number of selected rows

    def __init__(self, multi_select: bool = False, parent=None):
        """Initialize the styled table."""
        super().__init__(parent)
        self._multi_select = multi_select
        self._setup_style()
        self._connect_signals()

    def _setup_style(self):
        """Apply table styling."""
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        if self._multi_select:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Header styling
        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        header.setHighlightSections(False)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        vertical_header = self.verticalHeader()
        vertical_header.setVisible(False)
        vertical_header.setDefaultSectionSize(48)

        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 8px 12px;
                color: {COLORS['text_primary']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['primary_glow']};
                color: {COLORS['text_primary']};
            }}
            QTableWidget::item:hover {{
                background-color: {COLORS['surface_overlay']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_secondary']};
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 12px;
                border: none;
                border-bottom: 1px solid {COLORS['border_muted']};
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: {COLORS['surface_overlay']};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['surface_elevated']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['border_muted']};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['text_tertiary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background-color: {COLORS['surface_elevated']};
                height: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {COLORS['border_muted']};
                border-radius: 4px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {COLORS['text_tertiary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

    def _connect_signals(self):
        """Connect table signals."""
        self.cellClicked.connect(lambda row, _: self.row_clicked.emit(row))
        self.cellDoubleClicked.connect(lambda row, _: self.row_double_clicked.emit(row))
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self):
        """Emit the number of uniquely selected rows."""
        count = len(set(item.row() for item in self.selectedItems()))
        self.selection_count_changed.emit(count)

    def set_columns(self, columns: list[str]):
        """Set table column headers."""
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)

    def add_row(self, values: list[Any], data: Any = None):
        """
        Add a row to the table.

        Args:
            values: List of values for each column.
            data: Optional data to associate with the row.
        """
        row = self.rowCount()
        self.insertRow(row)

        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, data)
            self.setItem(row, col, item)

    def clear_rows(self):
        """Clear all rows from the table."""
        self.setRowCount(0)

    def get_row_data(self, row: int) -> Any:
        """Get the data associated with a row."""
        item = self.item(row, 0)
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def get_selected_row_data(self) -> Any:
        """Get data from the currently selected row."""
        selected = self.selectedItems()
        if selected:
            return selected[0].data(Qt.ItemDataRole.UserRole)
        return None

    def get_all_selected_rows_data(self) -> list[Any]:
        """Get data from ALL currently selected rows (one entry per row)."""
        rows = sorted(set(item.row() for item in self.selectedItems()))
        result = []
        for row in rows:
            data = self.get_row_data(row)
            if data is not None:
                result.append(data)
        return result


class DataTable(QWidget):
    """
    Complete data table with search and pagination.

    Wraps StyledTable with additional functionality.
    """

    row_selected = pyqtSignal(object)  # Emits row data when selected
    selection_changed = pyqtSignal(int)  # Emits number of selected rows

    def __init__(
        self,
        columns: list[str],
        searchable: bool = True,
        multi_select: bool = False,
        parent=None,
    ):
        """
        Initialize the data table.

        Args:
            columns: List of column headers.
            searchable: Whether to show search box.
            multi_select: Allow selecting multiple rows (Ctrl/Shift+click).
            parent: Parent widget.
        """
        super().__init__(parent)
        self.columns = columns
        self.searchable = searchable
        self._multi_select = multi_select
        self._all_data = []  # Store all data for filtering

        self._setup_ui()

    def _setup_ui(self):
        """Set up the table UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Search bar (if enabled)
        if self.searchable:
            search_layout = QHBoxLayout()

            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search...")
            self.search_input.setMinimumHeight(36)
            self.search_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {COLORS['surface_elevated']};
                    color: {COLORS['text_primary']};
                    border: 1px solid {COLORS['border_muted']};
                    border-radius: 6px;
                    padding: 8px 12px;
                    font-size: 13px;
                }}
                QLineEdit:focus {{
                    border-color: {COLORS['primary']};
                }}
            """)
            self.search_input.textChanged.connect(self._filter_data)
            search_layout.addWidget(self.search_input)

            layout.addLayout(search_layout)

        # Table
        self.table = StyledTable(multi_select=self._multi_select)
        self.table.set_columns(self.columns)
        self.table.row_clicked.connect(self._on_row_clicked)
        self.table.selection_count_changed.connect(self.selection_changed)
        layout.addWidget(self.table)

        # Row count label
        self.count_label = QLabel("0 items")
        self.count_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
        """)
        layout.addWidget(self.count_label)

    def set_data(
        self,
        data: list[dict],
        columns_map: Optional[dict[str, str]] = None,
    ):
        """
        Set table data.

        Args:
            data: List of dictionaries with row data.
            columns_map: Optional mapping of column headers to dict keys.
        """
        self._all_data = data
        self._display_data(data, columns_map)

    def _display_data(
        self,
        data: list[dict],
        columns_map: Optional[dict[str, str]] = None,
    ):
        """Display data in the table."""
        self.table.clear_rows()

        # Default mapping uses column headers as keys
        if columns_map is None:
            columns_map = {col: col.lower().replace(" ", "_") for col in self.columns}

        for row_data in data:
            values = []
            for col in self.columns:
                key = columns_map.get(col, col)
                value = row_data.get(key, "")
                values.append(value)
            self.table.add_row(values, row_data)

        self._update_count()

    def _filter_data(self, search_text: str):
        """Filter data based on search text."""
        if not search_text:
            self._display_data(self._all_data)
            return

        search_lower = search_text.lower()
        filtered = []

        for row_data in self._all_data:
            # Search in all string values
            for value in row_data.values():
                if isinstance(value, str) and search_lower in value.lower():
                    filtered.append(row_data)
                    break

        self._display_data(filtered)

    def _on_row_clicked(self, row: int):
        """Handle row click."""
        data = self.table.get_row_data(row)
        if data:
            self.row_selected.emit(data)

    def _update_count(self):
        """Update the row count label."""
        count = self.table.rowCount()
        self.count_label.setText(f"{count} item{'s' if count != 1 else ''}")

    def clear(self):
        """Clear all data."""
        self._all_data = []
        self.table.clear_rows()
        self._update_count()

    def get_selected_data(self) -> Any:
        """Get currently selected row data."""
        return self.table.get_selected_row_data()

    def get_all_selected_data(self) -> list[Any]:
        """Get data from all currently selected rows."""
        return self.table.get_all_selected_rows_data()
