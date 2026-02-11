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

# Import actual views
from src.ui.views.dashboard_view import DashboardView
from src.ui.views.jobs_view import JobsView
from src.ui.views.matching_view import MatchingView
from src.ui.views.settings_view import SettingsView
from src.ui.views.candidates_view import CandidatesView
from src.ui.views.analytics_view import AnalyticsView


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

        # Per-view refresh tracking: True means the view needs a refresh
        self._view_needs_refresh: dict[int, bool] = {}

        # Add actual views
        # Dashboard
        self.dashboard_view = DashboardView()
        self.content_stack.addWidget(self.dashboard_view)

        # Candidates
        self.candidates_view = CandidatesView()
        self.content_stack.addWidget(self.candidates_view)

        # Jobs
        self.jobs_view = JobsView()
        self.content_stack.addWidget(self.jobs_view)

        # Matching
        self.matching_view = MatchingView()
        self.content_stack.addWidget(self.matching_view)

        # Analytics
        self.analytics_view = AnalyticsView()
        self.content_stack.addWidget(self.analytics_view)

        # Settings
        try:
            self.settings_view = SettingsView()
        except Exception:
            self.settings_view = PlaceholderView("Settings", "Application configuration")
        self.content_stack.addWidget(self.settings_view)

        # Mark all views except dashboard as needing first refresh
        for i in range(self.content_stack.count()):
            self._view_needs_refresh[i] = True
        self._view_needs_refresh[0] = False  # Dashboard refreshed at startup

        main_layout.addWidget(content_container)

        # Connect navigation buttons
        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))

        # Connect dashboard quick actions to navigation
        self.dashboard_view.navigate_to_view.connect(self.switch_view)

        # Apply stylesheet
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f8fafc;
            }
            """
        )

        # Load initial data for dashboard
        self.dashboard_view.refresh()

    def switch_view(self, index: int):
        """Switch to a different view."""
        self.content_stack.setCurrentIndex(index)

        # Update sidebar button states
        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.setChecked(i == index)

        # Only refresh if the view is marked as needing it (first visit or dirty)
        if self._view_needs_refresh.get(index, False):
            current_widget = self.content_stack.widget(index)
            if hasattr(current_widget, "refresh"):
                current_widget.refresh()
            self._view_needs_refresh[index] = False

    def mark_view_dirty(self, index: int):
        """Mark a view as needing refresh on next visit."""
        self._view_needs_refresh[index] = True


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
