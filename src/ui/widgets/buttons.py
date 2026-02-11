"""
Button widgets for AI-ATS application.

Provides styled button components with consistent appearance.
"""

from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon

from src.utils.constants import COLORS


class PrimaryButton(QPushButton):
    """
    Primary action button with filled background.

    Used for main actions like "Save", "Submit", "Run Matching".
    """

    def __init__(self, text: str, parent=None):
        """Initialize the primary button."""
        super().__init__(text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply primary button styling."""
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:pressed {{
                background-color: #1e40af;
            }}
            QPushButton:disabled {{
                background-color: #94a3b8;
                color: #e2e8f0;
            }}
        """)


class SecondaryButton(QPushButton):
    """
    Secondary action button with outline style.

    Used for secondary actions like "Cancel", "Back".
    """

    def __init__(self, text: str, parent=None):
        """Initialize the secondary button."""
        super().__init__(text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply secondary button styling."""
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: 1px solid {COLORS['primary']};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #eff6ff;
            }}
            QPushButton:pressed {{
                background-color: #dbeafe;
            }}
            QPushButton:disabled {{
                border-color: #94a3b8;
                color: #94a3b8;
            }}
        """)


class DangerButton(QPushButton):
    """
    Danger/destructive action button.

    Used for destructive actions like "Delete", "Remove".
    """

    def __init__(self, text: str, parent=None):
        """Initialize the danger button."""
        super().__init__(text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply danger button styling."""
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['error']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
            QPushButton:pressed {{
                background-color: #b91c1c;
            }}
            QPushButton:disabled {{
                background-color: #94a3b8;
                color: #e2e8f0;
            }}
        """)


class SuccessButton(QPushButton):
    """
    Success action button.

    Used for positive actions like "Approve", "Accept".
    """

    def __init__(self, text: str, parent=None):
        """Initialize the success button."""
        super().__init__(text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply success button styling."""
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #16a34a;
            }}
            QPushButton:pressed {{
                background-color: #15803d;
            }}
            QPushButton:disabled {{
                background-color: #94a3b8;
                color: #e2e8f0;
            }}
        """)


class IconButton(QPushButton):
    """
    Icon-only button for toolbar actions.

    Displays an icon without text.
    """

    def __init__(self, icon_text: str = "", parent=None):
        """
        Initialize the icon button.

        Args:
            icon_text: Unicode icon character or text.
            parent: Parent widget.
        """
        super().__init__(icon_text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply icon button styling."""
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 12))
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #f1f5f9;
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: #e2e8f0;
            }}
        """)


class TextButton(QPushButton):
    """
    Text-only button without background.

    Used for links and subtle actions.
    """

    def __init__(self, text: str, parent=None):
        """Initialize the text button."""
        super().__init__(text, parent)
        self._setup_style()

    def _setup_style(self):
        """Apply text button styling."""
        self.setFont(QFont("Segoe UI", 10))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: none;
                padding: 4px 8px;
                text-decoration: none;
            }}
            QPushButton:hover {{
                text-decoration: underline;
            }}
            QPushButton:pressed {{
                color: {COLORS['primary_dark']};
            }}
        """)


class ButtonGroup(QWidget):
    """
    Container for grouping multiple buttons.

    Arranges buttons horizontally with consistent spacing.
    """

    def __init__(self, parent=None):
        """Initialize the button group."""
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)

    def add_button(self, button: QPushButton):
        """Add a button to the group."""
        self.layout.addWidget(button)

    def add_stretch(self):
        """Add stretch to push buttons."""
        self.layout.addStretch()

    def add_spacing(self, width: int = 16):
        """Add spacing between buttons."""
        self.layout.addSpacing(width)
