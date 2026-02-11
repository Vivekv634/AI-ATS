"""
Base view class for AI-ATS application views.

Provides common functionality and styling for all views.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS


class BaseView(QWidget):
    """
    Base class for all application views.

    Provides common layout structure, styling, and methods
    that all views can inherit and extend.
    """

    # Signal emitted when view needs refresh
    refresh_requested = pyqtSignal()

    def __init__(self, title: str, description: str = "", parent=None):
        """
        Initialize the base view.

        Args:
            title: View title displayed in header.
            description: Optional description text.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.title = title
        self.description = description

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        """Set up the base UI structure."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 24, 24, 24)
        self.main_layout.setSpacing(16)

        # Header section
        self.header = self._create_header()
        self.main_layout.addWidget(self.header)

        # Content area (scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(16)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

    def _create_header(self) -> QWidget:
        """Create the view header with title and description."""
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 8)
        header_layout.setSpacing(4)

        # Title
        self.title_label = QLabel(self.title)
        title_font = QFont("Segoe UI", 18)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(self.title_label)

        # Description
        if self.description:
            self.description_label = QLabel(self.description)
            self.description_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 13px;"
            )
            self.description_label.setWordWrap(True)
            header_layout.addWidget(self.description_label)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: #e2e8f0;")
        separator.setFixedHeight(1)
        header_layout.addWidget(separator)

        return header

    def _apply_styles(self):
        """Apply base styles to the view."""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['background']};
            }}
            QScrollArea {{
                background-color: transparent;
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

    def add_widget(self, widget: QWidget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def add_layout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)

    def add_stretch(self):
        """Add stretch to push content to top."""
        self.content_layout.addStretch()

    def clear_content(self):
        """Clear all widgets from content area."""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_title(self, title: str):
        """Update the view title."""
        self.title = title
        self.title_label.setText(title)

    def refresh(self):
        """
        Refresh the view data.

        Override this method in subclasses to implement
        view-specific refresh logic.
        """
        self.refresh_requested.emit()
