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

    def __init__(self, parent=None):
        """Initialize the styled table."""
        super().__init__(parent)
        self._setup_style()
        self._connect_signals()

    def _setup_style(self):
        """Apply table styling."""
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
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
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: transparent;
            }}
            QTableWidget::item {{
                padding: 8px 12px;
                border-bottom: 1px solid #f1f5f9;
            }}
            QTableWidget::item:selected {{
                background-color: #eff6ff;
                color: {COLORS['text_primary']};
            }}
            QTableWidget::item:hover {{
                background-color: #f8fafc;
            }}
            QHeaderView::section {{
                background-color: #f8fafc;
                color: {COLORS['text_secondary']};
                font-weight: 600;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 12px;
                border: none;
                border-bottom: 1px solid #e2e8f0;
            }}
            QTableWidget QTableCornerButton::section {{
                background-color: #f8fafc;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: #f1f5f9;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #cbd5e1;
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #94a3b8;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _connect_signals(self):
        """Connect table signals."""
        self.cellClicked.connect(lambda row, _: self.row_clicked.emit(row))
        self.cellDoubleClicked.connect(lambda row, _: self.row_double_clicked.emit(row))

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


class DataTable(QWidget):
    """
    Complete data table with search and pagination.

    Wraps StyledTable with additional functionality.
    """

    row_selected = pyqtSignal(object)  # Emits row data when selected

    def __init__(
        self,
        columns: list[str],
        searchable: bool = True,
        parent=None,
    ):
        """
        Initialize the data table.

        Args:
            columns: List of column headers.
            searchable: Whether to show search box.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.columns = columns
        self.searchable = searchable
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
                    background-color: {COLORS['surface']};
                    border: 1px solid #e2e8f0;
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
        self.table = StyledTable()
        self.table.set_columns(self.columns)
        self.table.row_clicked.connect(self._on_row_clicked)
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
