"""
Card widgets for AI-ATS application.

Provides reusable card components for displaying statistics,
information, and grouped content.
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

from src.utils.constants import COLORS


class Card(QFrame):
    """
    Base card widget with shadow and rounded corners.

    A container for grouping related content with a clean,
    elevated appearance.
    """

    def __init__(self, parent=None):
        """Initialize the card."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the card UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            Card {{
                background-color: {COLORS['surface']};
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }}
        """)

        # Add subtle shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 25))
        self.setGraphicsEffect(shadow)

        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 16, 16, 16)
        self.layout.setSpacing(12)


class StatCard(Card):
    """
    Statistics card for displaying a single metric.

    Shows a value, label, and optional trend indicator.
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color: str = None,
        parent=None,
    ):
        """
        Initialize the stat card.

        Args:
            title: Card title/label.
            value: Main value to display.
            subtitle: Optional subtitle or trend text.
            color: Optional accent color.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.accent_color = color or COLORS['primary']

        self._setup_content()

    def _setup_content(self):
        """Set up the card content."""
        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
            font-weight: 500;
        """)
        self.layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(self.value)
        value_font = QFont("Segoe UI", 28)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.layout.addWidget(self.value_label)

        # Subtitle/trend
        if self.subtitle:
            self.subtitle_label = QLabel(self.subtitle)
            self.subtitle_label.setStyleSheet(f"""
                color: {self.accent_color};
                font-size: 12px;
            """)
            self.layout.addWidget(self.subtitle_label)

        self.layout.addStretch()

    def set_value(self, value: str):
        """Update the displayed value."""
        self.value = value
        self.value_label.setText(value)

    def set_subtitle(self, subtitle: str):
        """Update the subtitle text."""
        self.subtitle = subtitle
        if hasattr(self, 'subtitle_label'):
            self.subtitle_label.setText(subtitle)


class InfoCard(Card):
    """
    Information card for displaying detailed content.

    Shows a title, description, and optional action area.
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        parent=None,
    ):
        """
        Initialize the info card.

        Args:
            title: Card title.
            description: Card description text.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.title = title
        self.description = description

        self._setup_content()

    def _setup_content(self):
        """Set up the card content."""
        # Title
        self.title_label = QLabel(self.title)
        title_font = QFont("Segoe UI", 14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.layout.addWidget(self.title_label)

        # Description
        if self.description:
            self.description_label = QLabel(self.description)
            self.description_label.setWordWrap(True)
            self.description_label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 13px;
                line-height: 1.5;
            """)
            self.layout.addWidget(self.description_label)

        # Content area for additional widgets
        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(8)
        self.layout.addLayout(self.content_area)

    def add_content(self, widget: QWidget):
        """Add a widget to the card content area."""
        self.content_area.addWidget(widget)


class ScoreCard(Card):
    """
    Score display card with visual indicator.

    Shows a score value with a colored progress-style indicator.
    """

    def __init__(
        self,
        title: str,
        score: float,
        max_score: float = 1.0,
        parent=None,
    ):
        """
        Initialize the score card.

        Args:
            title: Score label.
            score: Current score value.
            max_score: Maximum possible score.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.title = title
        self.score = score
        self.max_score = max_score

        self._setup_content()

    def _setup_content(self):
        """Set up the card content."""
        # Title and score value row
        header_layout = QHBoxLayout()

        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 13px;
        """)
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.score_label = QLabel(f"{self.score * 100:.0f}%")
        score_font = QFont("Segoe UI", 14)
        score_font.setBold(True)
        self.score_label.setFont(score_font)
        self.score_label.setStyleSheet(f"color: {self._get_score_color()};")
        header_layout.addWidget(self.score_label)

        self.layout.addLayout(header_layout)

        # Progress bar
        self.progress_bar = QFrame()
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(f"""
            background-color: #e2e8f0;
            border-radius: 4px;
        """)

        # Progress fill
        self.progress_fill = QFrame(self.progress_bar)
        self._update_progress()

        self.layout.addWidget(self.progress_bar)

    def _get_score_color(self) -> str:
        """Get color based on score value."""
        if self.max_score == 0:
            return COLORS['text_secondary']
        percentage = self.score / self.max_score
        if percentage >= 0.85:
            return COLORS['success']
        elif percentage >= 0.70:
            return COLORS['primary']
        elif percentage >= 0.50:
            return COLORS['warning']
        return COLORS['error']

    def _update_progress(self):
        """Update the progress bar fill."""
        if self.max_score == 0:
            percentage = 0.0
        else:
            percentage = min(self.score / self.max_score, 1.0)
        width = int(self.progress_bar.width() * percentage) if self.progress_bar.width() > 0 else 0

        self.progress_fill.setGeometry(0, 0, max(width, 4), 8)
        self.progress_fill.setStyleSheet(f"""
            background-color: {self._get_score_color()};
            border-radius: 4px;
        """)

    def resizeEvent(self, event):
        """Handle resize to update progress bar."""
        super().resizeEvent(event)
        self._update_progress()

    def set_score(self, score: float):
        """Update the displayed score."""
        self.score = score
        self.score_label.setText(f"{score * 100:.0f}%")
        self.score_label.setStyleSheet(f"color: {self._get_score_color()};")
        self._update_progress()


class CandidateCard(Card):
    """
    Card for displaying candidate information.

    Shows candidate name, match score, and key details.
    """

    def __init__(
        self,
        name: str,
        score: float,
        skills: list[str] = None,
        experience: str = "",
        parent=None,
    ):
        """
        Initialize the candidate card.

        Args:
            name: Candidate name.
            score: Match score (0-1).
            skills: List of matched skills.
            experience: Experience summary.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.name = name
        self.score = score
        self.skills = skills or []
        self.experience = experience

        self._setup_content()

    def _setup_content(self):
        """Set up the card content."""
        # Header with name and score
        header_layout = QHBoxLayout()

        # Name
        self.name_label = QLabel(self.name)
        name_font = QFont("Segoe UI", 14)
        name_font.setBold(True)
        self.name_label.setFont(name_font)
        self.name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(self.name_label)

        header_layout.addStretch()

        # Score badge
        score_pct = int((self.score or 0) * 100)
        self.score_badge = QLabel(f"{score_pct}%")
        self.score_badge.setFixedSize(50, 28)
        self.score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_color = self._get_score_color()
        self.score_badge.setStyleSheet(f"""
            background-color: {score_color};
            color: white;
            border-radius: 14px;
            font-weight: bold;
            font-size: 12px;
        """)
        header_layout.addWidget(self.score_badge)

        self.layout.addLayout(header_layout)

        # Experience
        if self.experience:
            exp_label = QLabel(self.experience)
            exp_label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 12px;
            """)
            self.layout.addWidget(exp_label)

        # Skills
        if self.skills:
            skills_layout = QHBoxLayout()
            skills_layout.setSpacing(6)

            for skill in self.skills[:4]:  # Show max 4 skills
                skill_badge = QLabel(skill)
                skill_badge.setStyleSheet(f"""
                    background-color: #e0e7ff;
                    color: {COLORS['primary']};
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                """)
                skills_layout.addWidget(skill_badge)

            if len(self.skills) > 4:
                more_label = QLabel(f"+{len(self.skills) - 4}")
                more_label.setStyleSheet(f"""
                    color: {COLORS['text_secondary']};
                    font-size: 11px;
                """)
                skills_layout.addWidget(more_label)

            skills_layout.addStretch()
            self.layout.addLayout(skills_layout)

    def _get_score_color(self) -> str:
        """Get color based on score value."""
        if self.score is None:
            return COLORS['text_secondary']
        if self.score >= 0.85:
            return COLORS['success']
        elif self.score >= 0.70:
            return COLORS['primary']
        elif self.score >= 0.50:
            return COLORS['warning']
        return COLORS['error']
