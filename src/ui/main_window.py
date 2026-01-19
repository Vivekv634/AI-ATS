"""
Main Window for AI-ATS Application

This module contains the main application window and entry point
for the PyQt6 graphical user interface.
"""

import sys
from pathlib import Path

# Ensure src is in path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QIcon

from src.utils.config import get_settings
from src.utils.constants import APP_DISPLAY_NAME, VERSION


class SidebarButton(QPushButton):
    """Custom styled button for the sidebar navigation."""

    def __init__(self, text: str, icon_name: str = "", parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setMinimumHeight(45)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                text-align: left;
                font-size: 14px;
                color: #64748b;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
                color: #1e293b;
            }
            QPushButton:checked {
                background-color: #e0e7ff;
                color: #2563eb;
                font-weight: bold;
            }
            """
        )


class Sidebar(QFrame):
    """Navigation sidebar for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(5)

        # App title
        title_label = QLabel(APP_DISPLAY_NAME)
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1e293b; padding: 10px;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        layout.addSpacing(20)

        # Navigation buttons
        self.nav_buttons = []

        nav_items = [
            ("Dashboard", "dashboard"),
            ("Candidates", "candidates"),
            ("Job Postings", "jobs"),
            ("Matching", "matching"),
            ("Analytics", "analytics"),
            ("Settings", "settings"),
        ]

        for text, name in nav_items:
            btn = SidebarButton(text)
            btn.setObjectName(name)
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        # Set first button as checked
        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

        layout.addStretch()

        # Version label at bottom
        version_label = QLabel(f"v{VERSION}")
        version_label.setStyleSheet("color: #94a3b8; font-size: 11px; padding: 10px;")
        layout.addWidget(version_label)


class PlaceholderView(QWidget):
    """Placeholder view for sections under development."""

    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #1e293b;")
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        desc_label = QLabel(description)
        desc_label.setFont(QFont("Segoe UI", 14))
        desc_label.setStyleSheet("color: #64748b;")
        layout.addWidget(desc_label, alignment=Qt.AlignmentFlag.AlignCenter)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface."""
        # Window properties
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(1200, 700)
        self.resize(
            self.settings.ui.window_width,
            self.settings.ui.window_height,
        )

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Content area
        content_container = QWidget()
        content_container.setStyleSheet("background-color: #f8fafc;")
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Stacked widget for different views
        self.content_stack = QStackedWidget()
        content_layout.addWidget(self.content_stack)

        # Add placeholder views
        views = [
            ("Dashboard", "Overview of recruitment activities and metrics"),
            ("Candidates", "Manage and view candidate profiles"),
            ("Job Postings", "Create and manage job postings"),
            ("Matching", "AI-powered candidate-job matching"),
            ("Analytics", "Reports and insights"),
            ("Settings", "Application configuration"),
        ]

        for title, desc in views:
            view = PlaceholderView(title, desc)
            self.content_stack.addWidget(view)

        main_layout.addWidget(content_container)

        # Connect navigation buttons
        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))

        # Apply stylesheet
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8fafc;
            }
            """
        )

    def switch_view(self, index: int):
        """Switch to a different view."""
        self.content_stack.setCurrentIndex(index)

        # Update button states
        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.setChecked(i == index)


def run_application() -> int:
    """
    Initialize and run the PyQt6 application.

    Returns:
        Application exit code
    """
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationVersion(VERSION)

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    return app.exec()


def main():
    """Entry point for GUI script."""
    sys.exit(run_application())


if __name__ == "__main__":
    main()
