"""
Card widgets for AI-ATS application — dark elevation system.

Four elevation levels mirror the surface tokens in COLORS:
  Card        → surface (L1)
  InfoCard    → surface (L1) with left-accent stripe
  StatCard    → surface_elevated (L2) — slightly lifted
  ScoreCard   → surface_elevated (L2) with progress bar
  CandidateCard → surface_overlay (L3) — most prominent
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from src.utils.constants import COLORS, SHADOWS


def _make_shadow(level: str = "md") -> QGraphicsDropShadowEffect:
    """Create a drop-shadow effect using the SHADOWS token for the given level."""
    blur, x, y, alpha = SHADOWS[level]
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(blur)
    fx.setXOffset(x)
    fx.setYOffset(y)
    fx.setColor(QColor(0, 0, 0, alpha))
    return fx


class Card(QFrame):
    """
    Base card — surface (L1) elevation.

    Provides rounded corners, a subtle border, and a soft drop-shadow.
    All other card types inherit from this.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            Card {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 10px;
            }}
            """
        )
        self.setGraphicsEffect(_make_shadow("sm"))

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)


class StatCard(Card):
    """
    KPI / statistics card — surface_elevated (L2).

    Shows a metric value with title, optional subtitle/trend,
    and a coloured left-accent stripe tied to the metric's status colour.
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.accent_color = color or COLORS["primary"]
        self._setup_content()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Use surface_elevated for stat cards — one step above base cards
        self.setStyleSheet(
            f"""
            StatCard {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 10px;
                border-left: 3px solid {COLORS['primary']};
            }}
            """
        )
        self.setGraphicsEffect(_make_shadow("md"))
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(8)

    def _setup_content(self) -> None:
        # Title row
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 500;"
            f" letter-spacing: 0.5px; text-transform: uppercase;"
        )
        self.layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(self.value)
        val_font = QFont("Segoe UI", 30, QFont.Weight.Bold)
        self.value_label.setFont(val_font)
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.layout.addWidget(self.value_label)

        # Subtitle / trend
        if self.subtitle:
            self.subtitle_label = QLabel(self.subtitle)
            self.subtitle_label.setStyleSheet(
                f"color: {self.accent_color}; font-size: 12px;"
            )
            self.layout.addWidget(self.subtitle_label)

        self.layout.addStretch()

    def set_value(self, value: str) -> None:
        self.value = value
        self.value_label.setText(value)

    def set_subtitle(self, subtitle: str) -> None:
        self.subtitle = subtitle
        if hasattr(self, "subtitle_label"):
            self.subtitle_label.setText(subtitle)


class InfoCard(Card):
    """
    Content card with title, optional description, and a content slot.

    Left-accent stripe uses primary colour for a subtle directional cue.
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.description = description
        self._setup_content()

    def _setup_content(self) -> None:
        self.title_label = QLabel(self.title)
        title_font = QFont("Segoe UI", 13, QFont.Weight.DemiBold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.layout.addWidget(self.title_label)

        if self.description:
            self.description_label = QLabel(self.description)
            self.description_label.setWordWrap(True)
            self.description_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 13px; line-height: 1.6;"
            )
            self.layout.addWidget(self.description_label)

        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(8)
        self.layout.addLayout(self.content_area)

    def add_content(self, widget: QWidget) -> None:
        self.content_area.addWidget(widget)


class ScoreCard(Card):
    """
    Score display card with animated progress bar.

    Colour-codes the score: green ≥ 85%, blue ≥ 70%, amber ≥ 50%, red < 50%.
    """

    def __init__(
        self,
        title: str,
        score: float,
        max_score: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.score = score
        self.max_score = max_score
        self._setup_content()

    def _setup_content(self) -> None:
        header = QHBoxLayout()

        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        header.addWidget(self.title_label)
        header.addStretch()

        self.score_label = QLabel(f"{self.score * 100:.0f}%")
        score_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        self.score_label.setFont(score_font)
        self.score_label.setStyleSheet(f"color: {self._score_color()};")
        header.addWidget(self.score_label)
        self.layout.addLayout(header)

        # Progress track
        self.progress_bar = QFrame()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border-radius: 3px;"
        )
        self.progress_fill = QFrame(self.progress_bar)
        self._update_progress()
        self.layout.addWidget(self.progress_bar)

    def _score_color(self) -> str:
        if self.max_score == 0:
            return COLORS["text_secondary"]
        pct = self.score / self.max_score
        if pct >= 0.85:
            return COLORS["success"]
        if pct >= 0.70:
            return COLORS["primary"]
        if pct >= 0.50:
            return COLORS["warning"]
        return COLORS["error"]

    def _update_progress(self) -> None:
        pct = min(self.score / self.max_score, 1.0) if self.max_score else 0.0
        width = int(self.progress_bar.width() * pct) if self.progress_bar.width() > 0 else 0
        self.progress_fill.setGeometry(0, 0, max(width, 4), 6)
        self.progress_fill.setStyleSheet(
            f"background-color: {self._score_color()}; border-radius: 3px;"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_progress()

    def set_score(self, score: float) -> None:
        self.score = score
        self.score_label.setText(f"{score * 100:.0f}%")
        self.score_label.setStyleSheet(f"color: {self._score_color()};")
        self._update_progress()


class CandidateCard(Card):
    """
    Candidate result card — surface_overlay (L3, most elevated).

    Shows name, AI match score badge, experience, and skill chips.
    """

    def __init__(
        self,
        name: str,
        score: float,
        skills: list[str] | None = None,
        experience: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.name = name
        self.score = score
        self.skills = skills or []
        self.experience = experience
        self._setup_content()

    def _setup_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"""
            CandidateCard {{
                background-color: {COLORS['surface_overlay']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 10px;
            }}
            CandidateCard:hover {{
                border-color: {COLORS['primary']};
            }}
            """
        )
        self.setGraphicsEffect(_make_shadow("lg"))
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)

    def _setup_content(self) -> None:
        # Header row: name + score badge
        header = QHBoxLayout()

        self.name_label = QLabel(self.name)
        name_font = QFont("Segoe UI", 13, QFont.Weight.DemiBold)
        self.name_label.setFont(name_font)
        self.name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(self.name_label)
        header.addStretch()

        score_pct = int((self.score or 0) * 100)
        score_color = self._score_color()
        self.score_badge = QLabel(f"{score_pct}%")
        self.score_badge.setFixedSize(46, 24)
        self.score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_badge.setStyleSheet(
            f"""
            background-color: {score_color}22;
            color: {score_color};
            border: 1px solid {score_color};
            border-radius: 6px;
            font-weight: bold;
            font-size: 11px;
            """
        )
        header.addWidget(self.score_badge)
        self.layout.addLayout(header)

        # Experience
        if self.experience:
            exp_label = QLabel(self.experience)
            exp_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px;"
            )
            self.layout.addWidget(exp_label)

        # Skill chips
        if self.skills:
            chips_row = QHBoxLayout()
            chips_row.setSpacing(6)
            for skill in self.skills[:4]:
                chip = QLabel(skill)
                chip.setStyleSheet(
                    f"""
                    background-color: {COLORS['primary_glow']};
                    color: {COLORS['primary']};
                    border: 1px solid {COLORS['primary_glow']};
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    """
                )
                chips_row.addWidget(chip)
            if len(self.skills) > 4:
                more = QLabel(f"+{len(self.skills) - 4}")
                more.setStyleSheet(
                    f"color: {COLORS['text_tertiary']}; font-size: 11px;"
                )
                chips_row.addWidget(more)
            chips_row.addStretch()
            self.layout.addLayout(chips_row)

    def _score_color(self) -> str:
        if self.score is None:
            return COLORS["text_secondary"]
        if self.score >= 0.85:
            return COLORS["success"]
        if self.score >= 0.70:
            return COLORS["primary"]
        if self.score >= 0.50:
            return COLORS["warning"]
        return COLORS["error"]
