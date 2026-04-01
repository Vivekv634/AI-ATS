"""
Main Window for AI-ATS Application

Dark Midnight Navy theme — dimensional layering, left-border active indicator,
Inter/Segoe UI typography hierarchy.
"""

import sys
from pathlib import Path

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
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QIcon, QColor

from src.utils.config import get_settings
from src.utils.constants import APP_DISPLAY_NAME, VERSION, COLORS

from src.ui.views.dashboard_view import DashboardView
from src.ui.views.jobs_view import JobsView
from src.ui.views.matching_view import MatchingView
from src.ui.views.settings_view import SettingsView
from src.ui.views.candidates_view import CandidatesView
from src.ui.views.analytics_view import AnalyticsView


# ── Nav item definition ────────────────────────────────────────────────────────

_NAV_ITEMS: list[tuple[str, str, str]] = [
    # (display name, object name, unicode glyph)
    ("Dashboard",   "dashboard",  "⊟"),
    ("Candidates",  "candidates", "◈"),
    ("Job Postings","jobs",       "⊡"),
    ("AI Matching", "matching",   "⊕"),
    ("Analytics",   "analytics",  "⊘"),
    ("Settings",    "settings",   "⊗"),
]


# ── Sidebar nav button ─────────────────────────────────────────────────────────

class SidebarButton(QWidget):
    """
    Sidebar navigation item with left-border active indicator.

    Layout (horizontal):
      [3px accent bar] [icon glyph] [label text]

    The accent bar is a thin QFrame that becomes visible only when the
    button is in the checked/active state — a pattern from Linear/VS Code.
    """

    def __init__(self, text: str, glyph: str = "", parent=None) -> None:
        super().__init__(parent)
        self._checked: bool = False
        self._text: str = text
        self._glyph: str = glyph
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 2, 8, 2)
        outer.setSpacing(0)

        # Left accent bar — 3 px, only visible when active
        self._accent_bar = QFrame()
        self._accent_bar.setFixedWidth(3)
        self._accent_bar.setStyleSheet("background-color: transparent; border-radius: 2px;")
        outer.addWidget(self._accent_bar)

        # Glyph icon
        self._icon_label = QLabel(self._glyph)
        self._icon_label.setFixedWidth(32)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setFont(QFont("Segoe UI Symbol", 13))
        self._icon_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        outer.addWidget(self._icon_label)

        # Nav text label
        self._text_label = QLabel(self._text)
        self._text_label.setFont(QFont("Segoe UI", 10))
        self._text_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        outer.addWidget(self._text_label)
        outer.addStretch()

        self._apply_style()

    def _apply_style(self) -> None:
        if self._checked:
            self.setStyleSheet(
                f"QWidget {{ background-color: {COLORS['surface_overlay']};"
                f" border-radius: 6px; }}"
            )
            self._accent_bar.setStyleSheet(
                f"background-color: {COLORS['primary']}; border-radius: 2px;"
            )
            self._icon_label.setStyleSheet(f"color: {COLORS['primary']};")
            self._text_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-weight: bold;"
            )
        else:
            self.setStyleSheet(
                "QWidget { background-color: transparent; border-radius: 6px; }"
            )
            self._accent_bar.setStyleSheet(
                "background-color: transparent; border-radius: 2px;"
            )
            self._icon_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
            self._text_label.setStyleSheet(f"color: {COLORS['text_secondary']};")

    def setChecked(self, checked: bool) -> None:
        self._checked = checked
        self._apply_style()

    def isChecked(self) -> bool:
        return self._checked

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._clicked_callback()

    def enterEvent(self, event) -> None:
        if not self._checked:
            self.setStyleSheet(
                f"QWidget {{ background-color: {COLORS['surface_elevated']};"
                f" border-radius: 6px; }}"
            )
            self._text_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            self._icon_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if not self._checked:
            self._apply_style()
        super().leaveEvent(event)

    def set_click_callback(self, cb) -> None:
        self._clicked_callback = cb


# ── Sidebar ────────────────────────────────────────────────────────────────────

class Sidebar(QFrame):
    """
    Application navigation sidebar.

    Structure:
      ┌────────────────────────────────┐
      │  ⬡ AI-ATS   [AI]             │  ← brand row
      │  ─────────────────────────    │  ← divider
      │  ⊟  Dashboard                 │  ← nav items
      │  ◈  Candidates                │
      │  ⊡  Job Postings              │
      │  ⊕  AI Matching               │
      │  ⊘  Analytics                 │
      │  ─────────────────────────    │
      │  ⊗  Settings                  │
      │                               │
      │  v0.1.0                       │  ← version
      └────────────────────────────────┘
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(210)
        self.nav_buttons: list[SidebarButton] = []
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-right: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(2)

        # ── Brand row ──────────────────────────────────────────────────────────
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(8, 4, 8, 4)

        brand_icon = QLabel("⬡")
        brand_icon.setFont(QFont("Segoe UI Symbol", 16))
        brand_icon.setStyleSheet(f"color: {COLORS['primary']};")
        brand_row.addWidget(brand_icon)

        brand_name = QLabel("AI-ATS")
        brand_name.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        brand_name.setStyleSheet(f"color: {COLORS['text_primary']};")
        brand_row.addWidget(brand_name)
        brand_row.addStretch()

        # "AI" pill badge
        ai_badge = QLabel("AI")
        ai_badge.setFixedSize(26, 16)
        ai_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_badge.setStyleSheet(
            f"""
            background-color: {COLORS['accent_dim']};
            color: {COLORS['accent']};
            border: 1px solid {COLORS['accent']};
            border-radius: 4px;
            font-size: 9px;
            font-weight: bold;
            """
        )
        brand_row.addWidget(ai_badge)

        layout.addLayout(brand_row)

        # ── Top divider ────────────────────────────────────────────────────────
        layout.addSpacing(10)
        divider_top = self._make_divider()
        layout.addWidget(divider_top)
        layout.addSpacing(6)

        # ── Main nav items (all except Settings) ──────────────────────────────
        for text, name, glyph in _NAV_ITEMS[:-1]:
            btn = SidebarButton(text, glyph)
            btn.setObjectName(name)
            btn.set_click_callback(lambda b=btn: None)  # placeholder; wired in MainWindow
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # ── Bottom divider + Settings ─────────────────────────────────────────
        divider_bot = self._make_divider()
        layout.addWidget(divider_bot)
        layout.addSpacing(6)

        text, name, glyph = _NAV_ITEMS[-1]
        settings_btn = SidebarButton(text, glyph)
        settings_btn.setObjectName(name)
        settings_btn.set_click_callback(lambda: None)
        self.nav_buttons.append(settings_btn)
        layout.addWidget(settings_btn)

        layout.addSpacing(12)

        # ── Version footer ─────────────────────────────────────────────────────
        ver_label = QLabel(f"v{VERSION}")
        ver_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 10px; padding-left: 12px;"
        )
        layout.addWidget(ver_label)

        # Set first button active by default
        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

    @staticmethod
    def _make_divider() -> QFrame:
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet(f"background-color: {COLORS['border_subtle']}; border: none;")
        return div


# ── Content header bar ─────────────────────────────────────────────────────────

class ContentHeader(QFrame):
    """
    Thin header bar above the content area.

    Shows the current view name and a subtle separator.
    Used as a breadcrumb/context anchor.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(50)
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)

        self._title = QLabel("Dashboard")
        self._title.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(self._title)
        layout.addStretch()

        # Subtle app name tag on the right
        tag = QLabel(APP_DISPLAY_NAME)
        tag.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
        layout.addWidget(tag)

    def set_title(self, title: str) -> None:
        self._title.setText(title)


# ── Placeholder view ───────────────────────────────────────────────────────────

class PlaceholderView(QWidget):
    """Placeholder for views under development."""

    def __init__(self, title: str, description: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)

        d = QLabel(description)
        d.setFont(QFont("Segoe UI", 14))
        d.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(d, alignment=Qt.AlignmentFlag.AlignCenter)


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window — dark midnight navy theme."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(1200, 700)
        self.resize(
            self.settings.ui.window_width,
            self.settings.ui.window_height,
        )

        # Global stylesheet — sets baseline for all QWidget descendants
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
                font-family: "Segoe UI", "Inter", "SF Pro Display", sans-serif;
            }}
            QScrollBar:vertical {{
                background: {COLORS['surface']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_muted']};
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLORS['text_tertiary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QScrollBar:horizontal {{
                background: {COLORS['surface']};
                height: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:horizontal {{
                background: {COLORS['border_muted']};
                border-radius: 4px;
                min-width: 24px;
            }}
            QToolTip {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }}
            QLineEdit, QTextEdit, QPlainTextEdit {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 6px;
                padding: 6px 10px;
                selection-background-color: {COLORS['primary_glow']};
            }}
            QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {COLORS['primary']};
            }}
            QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 6px;
                padding: 6px 10px;
                min-height: 32px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 6px;
                selection-background-color: {COLORS['primary_glow']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QSplitter::handle {{
                background-color: {COLORS['border_subtle']};
            }}
            QSplitter::handle:horizontal {{
                width: 1px;
            }}
            QSplitter::handle:vertical {{
                height: 1px;
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                background-color: {COLORS['surface']};
            }}
            QTabBar::tab {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                padding: 8px 16px;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {COLORS['primary']};
                border-bottom: 2px solid {COLORS['primary']};
            }}
            QTabBar::tab:hover:!selected {{
                color: {COLORS['text_primary']};
            }}
            QCheckBox {{
                color: {COLORS['text_primary']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {COLORS['border_muted']};
                background-color: {COLORS['surface_elevated']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QGroupBox {{
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: {COLORS['text_secondary']};
            }}
            QProgressBar {{
                background-color: {COLORS['surface_elevated']};
                border: none;
                border-radius: 4px;
                height: 8px;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
            QMessageBox {{
                background-color: {COLORS['surface']};
            }}
            QMessageBox QLabel {{
                color: {COLORS['text_primary']};
            }}
            QDialog {{
                background-color: {COLORS['surface']};
            }}
            """
        )

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ────────────────────────────────────────────────────────────
        self.sidebar = Sidebar()
        root.addWidget(self.sidebar)

        # ── Right column: header + content stack ───────────────────────────────
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)

        self.header = ContentHeader()
        right_col.addWidget(self.header)

        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet(
            f"background-color: {COLORS['background']};"
        )
        right_col.addWidget(self.content_stack)

        right_wrapper = QWidget()
        right_wrapper.setLayout(right_col)
        root.addWidget(right_wrapper)

        # ── Lazy refresh tracking ──────────────────────────────────────────────
        self._view_needs_refresh: dict[int, bool] = {}

        # ── Instantiate views ──────────────────────────────────────────────────
        self.dashboard_view = DashboardView()
        self.content_stack.addWidget(self.dashboard_view)

        self.candidates_view = CandidatesView()
        self.content_stack.addWidget(self.candidates_view)

        self.jobs_view = JobsView()
        self.content_stack.addWidget(self.jobs_view)

        self.matching_view = MatchingView()
        self.content_stack.addWidget(self.matching_view)

        self.analytics_view = AnalyticsView()
        self.content_stack.addWidget(self.analytics_view)

        try:
            self.settings_view = SettingsView()
        except Exception:
            self.settings_view = PlaceholderView("Settings", "Application configuration")
        self.content_stack.addWidget(self.settings_view)

        for i in range(self.content_stack.count()):
            self._view_needs_refresh[i] = True
        self._view_needs_refresh[0] = False

        # ── Wire sidebar click callbacks ───────────────────────────────────────
        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.set_click_callback(lambda idx=i: self.switch_view(idx))

        # ── Cross-view signals ─────────────────────────────────────────────────
        self.dashboard_view.navigate_to_view.connect(self.switch_view)
        self.jobs_view.job_created.connect(lambda: self.mark_view_dirty(3))

        # ── Load dashboard ─────────────────────────────────────────────────────
        self.dashboard_view.refresh()

    def switch_view(self, index: int) -> None:
        """Switch to a different view, updating sidebar state and header."""
        self.content_stack.setCurrentIndex(index)

        for i, btn in enumerate(self.sidebar.nav_buttons):
            btn.setChecked(i == index)

        # Update header title
        if 0 <= index < len(_NAV_ITEMS):
            self.header.set_title(_NAV_ITEMS[index][0])

        if self._view_needs_refresh.get(index, False):
            widget = self.content_stack.widget(index)
            if hasattr(widget, "refresh"):
                widget.refresh()
            self._view_needs_refresh[index] = False

    def mark_view_dirty(self, index: int) -> None:
        """Mark a view as needing refresh on next visit."""
        self._view_needs_refresh[index] = True


# ── Entry point ────────────────────────────────────────────────────────────────

def run_application() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationVersion(VERSION)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    return app.exec()


def main() -> None:
    sys.exit(run_application())


if __name__ == "__main__":
    main()
