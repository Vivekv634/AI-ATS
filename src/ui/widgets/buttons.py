"""
Button widgets for AI-ATS application — dark theme.

Button hierarchy:
  PrimaryButton   — filled blue, primary CTA
  SecondaryButton — ghost/outline, secondary actions
  DangerButton    — filled red, destructive actions
  SuccessButton   — filled green, positive confirmations
  IconButton      — square icon-only, toolbar actions
  TextButton      — link-style, inline actions
  ButtonGroup     — horizontal container with spacing helpers
"""

from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS


# ── Shared style helper ────────────────────────────────────────────────────────

def _base_font() -> QFont:
    f = QFont("Segoe UI", 10)
    f.setWeight(QFont.Weight.Medium)
    return f


# ── Primary ────────────────────────────────────────────────────────────────────

class PrimaryButton(QPushButton):
    """
    Main CTA button — filled primary blue with subtle glow on hover.

    Use for: Save, Run Matching, Import, Confirm.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 7px;
                padding: 8px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #5AABFF;
                border: 1px solid {COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border_subtle']};
            }}
            """
        )


# ── Secondary ──────────────────────────────────────────────────────────────────

class SecondaryButton(QPushButton):
    """
    Ghost/outline button — transparent fill, primary border.

    Use for: Cancel, Back, secondary navigation.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 7px;
                padding: 8px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_glow']};
                border-color: {COLORS['primary']};
                color: {COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['surface_elevated']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )


# ── Danger ─────────────────────────────────────────────────────────────────────

class DangerButton(QPushButton):
    """
    Destructive action button — filled error red.

    Use for: Delete, Remove, Reject.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['error_dim']};
                color: {COLORS['error']};
                border: 1px solid {COLORS['error']};
                border-radius: 7px;
                padding: 8px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['error']};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #CC3333;
                color: white;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )


# ── Success ────────────────────────────────────────────────────────────────────

class SuccessButton(QPushButton):
    """
    Positive action button — filled success green.

    Use for: Approve, Hire, Accept.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['success_dim']};
                color: {COLORS['success']};
                border: 1px solid {COLORS['success']};
                border-radius: 7px;
                padding: 8px 18px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success']};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #228A36;
                color: white;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )


# ── Icon button ────────────────────────────────────────────────────────────────

class IconButton(QPushButton):
    """
    Square icon-only button for toolbars.

    44×44 px hit target (WCAG 2.5.5). Shows a subtle background on hover.
    """

    def __init__(self, icon_text: str = "", parent=None) -> None:
        super().__init__(icon_text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI Symbol", 13))
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 7px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['surface_overlay']};
            }}
            """
        )


# ── Text / link button ─────────────────────────────────────────────────────────

class TextButton(QPushButton):
    """
    Inline text-link button with no background.

    Use for: "View all", "Details", subtle navigation.
    """

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: none;
                padding: 4px 6px;
            }}
            QPushButton:hover {{
                color: #5AABFF;
                text-decoration: underline;
            }}
            QPushButton:pressed {{
                color: {COLORS['primary_dark']};
            }}
            """
        )


# ── Button group ───────────────────────────────────────────────────────────────

class ButtonGroup(QWidget):
    """
    Horizontal container for grouping buttons with consistent spacing.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)

    def add_button(self, button: QPushButton) -> None:
        self.layout.addWidget(button)

    def add_stretch(self) -> None:
        self.layout.addStretch()

    def add_spacing(self, width: int = 16) -> None:
        self.layout.addSpacing(width)
