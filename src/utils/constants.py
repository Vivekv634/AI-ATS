"""
Application-wide constants for AI-ATS.

This module contains all constant values used throughout the application.
Modify these values to customize behavior without changing code logic.
"""

from enum import Enum, auto
from typing import Final


# =============================================================================
# Application Constants
# =============================================================================

APP_NAME: Final[str] = "AI-ATS"
APP_DISPLAY_NAME: Final[str] = "AI-Powered Applicant Tracking System"
VERSION: Final[str] = "0.1.0"


# =============================================================================
# File Types
# =============================================================================

SUPPORTED_RESUME_FORMATS: Final[tuple[str, ...]] = (
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
)

SUPPORTED_JOB_FORMATS: Final[tuple[str, ...]] = (
    ".pdf",
    ".docx",
    ".txt",
    ".json",
)


# =============================================================================
# NLP Constants
# =============================================================================

# Common skill categories for extraction
SKILL_CATEGORIES: Final[dict[str, list[str]]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    ],
    "frameworks": [
        "react", "angular", "vue", "django", "flask", "fastapi", "spring",
        "node.js", "express", ".net", "rails", "laravel", "tensorflow",
        "pytorch", "keras", "scikit-learn",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "oracle", "sql server", "sqlite", "dynamodb", "firebase",
    ],
    "cloud_platforms": [
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
        "kubernetes", "docker", "terraform", "ansible",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem-solving",
        "analytical", "creative", "adaptable", "organized", "detail-oriented",
    ],
}

# Education degree levels (ordered by level)
EDUCATION_LEVELS: Final[dict[str, int]] = {
    "high school": 1,
    "diploma": 2,
    "associate": 3,
    "bachelor": 4,
    "master": 5,
    "mba": 5,
    "phd": 6,
    "doctorate": 6,
}


# =============================================================================
# Scoring Constants
# =============================================================================

# Default weights for candidate scoring
DEFAULT_SCORING_WEIGHTS: Final[dict[str, float]] = {
    "skills_match": 0.35,
    "experience_match": 0.25,
    "education_match": 0.15,
    "semantic_similarity": 0.20,
    "keyword_match": 0.05,
}

# Score thresholds
SCORE_THRESHOLDS: Final[dict[str, float]] = {
    "excellent": 0.85,
    "good": 0.70,
    "fair": 0.50,
    "poor": 0.30,
}

# Skill embedding similarity thresholds
SKILL_EXACT_THRESHOLD: Final[float] = 0.92
"""Cosine similarity >= this value -> treat as exact match (score = 1.0)."""

SKILL_STRONG_PARTIAL_THRESHOLD: Final[float] = 0.75
"""Cosine similarity >= this value -> strong partial match (score = 0.8)."""

SKILL_WEAK_PARTIAL_THRESHOLD: Final[float] = 0.60
"""Cosine similarity >= this value -> weak partial match (score = 0.5)."""

EXP_RELEVANCE_TITLE_THRESHOLD: Final[float] = 0.40
"""Cosine similarity >= this value -> experience entry title added to relevant_titles_matched."""

EDU_FIELD_MATCH_THRESHOLD: Final[float] = 0.50
"""Cosine similarity >= this value -> EducationMatch.field_match = True."""

EDU_FIELD_WEIGHT: Final[float] = 0.40
"""Weight of field-of-study similarity in the combined education score (0-1).
The remaining (1 - EDU_FIELD_WEIGHT) is the degree-level score weight."""


# =============================================================================
# Ethical AI Constants
# =============================================================================

# Protected attributes for bias detection
PROTECTED_ATTRIBUTES: Final[list[str]] = [
    "gender",
    "age",
    "race",
    "ethnicity",
    "religion",
    "disability",
    "nationality",
    "marital_status",
]

# Fairness thresholds
FAIRNESS_THRESHOLDS: Final[dict[str, float]] = {
    "demographic_parity_difference": 0.1,
    "equalized_odds_difference": 0.1,
    "disparate_impact_ratio": 0.8,
}

# Minimum number of candidates required in a demographic group before that
# group is included in fairness metric calculations.  Groups smaller than this
# produce statistically unreliable rates (e.g. one extra selection/rejection
# swings the positive rate by 20–50 %), which could mask or falsely trigger
# fairness violations.  Configurable via FairnessCalculator(min_group_size=N).
FAIRNESS_MIN_GROUP_SIZE: Final[int] = 5


# =============================================================================
# Enums
# =============================================================================


class CandidateStatus(str, Enum):
    """Status of a candidate in the pipeline."""

    NEW = "new"
    SCREENING = "screening"
    SHORTLISTED = "shortlisted"
    INTERVIEWING = "interview"
    OFFERED = "offer"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class JobStatus(str, Enum):
    """Status of a job posting."""

    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    FILLED = "filled"


class MatchScoreLevel(Enum):
    """Categorical levels for match scores."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    @classmethod
    def from_score(cls, score: float) -> "MatchScoreLevel":
        """Convert a numeric score to a level."""
        if score >= SCORE_THRESHOLDS["excellent"]:
            return cls.EXCELLENT
        elif score >= SCORE_THRESHOLDS["good"]:
            return cls.GOOD
        elif score >= SCORE_THRESHOLDS["fair"]:
            return cls.FAIR
        return cls.POOR


class AuditAction(str, Enum):
    """Types of actions that can be audited."""

    CANDIDATE_ADDED = "candidate_added"
    CANDIDATE_SCORED = "candidate_scored"
    CANDIDATE_RANKED = "candidate_ranked"
    CANDIDATE_STATUS_CHANGED = "candidate_status_changed"
    JOB_CREATED = "job_created"
    JOB_CLOSED = "job_closed"
    BIAS_DETECTED = "bias_detected"
    MANUAL_OVERRIDE = "manual_override"
    REPORT_GENERATED = "report_generated"


# =============================================================================
# UI Constants
# =============================================================================

# Color palette for the application
COLORS: Final[dict[str, str]] = {
    # ── Brand ──────────────────────────────────────────────────────────────────
    "primary":          "#4B9EFF",   # Accessible bright blue (4.7:1 on bg_base)
    "primary_dark":     "#1D6FCC",   # Pressed/deep state
    "primary_glow":     "#1A3A6E",   # Subtle glow bg behind primary elements
    "accent":           "#7C6AF7",   # Indigo accent — used on AI/ML badges
    "accent_dim":       "#2A2456",   # Dimmed accent background
    "secondary":        "#8B949E",   # Neutral label colour

    # ── Status ─────────────────────────────────────────────────────────────────
    "success":          "#2EA043",   # GitHub-green — pass, hired, high score
    "success_dim":      "#1A3A28",   # Dim success background
    "warning":          "#D29922",   # Amber — partial match, caution
    "warning_dim":      "#3A2E10",   # Dim warning background
    "error":            "#F85149",   # Red — fail, rejected, error
    "error_dim":        "#3A1A1A",   # Dim error background
    "info":             "#58A6FF",   # Info blue

    # ── Surface elevation (L0 → L3) ───────────────────────────────────────────
    # L0: absolute base (window / wallpaper)
    "background":       "#080C14",
    # L1: default panel / card surface
    "surface":          "#0D1117",
    # L2: elevated — hover backgrounds, active rows, inner panels
    "surface_elevated": "#161B22",
    # L3: overlay — dropdowns, tooltips, modals, active sidebar item
    "surface_overlay":  "#1C2333",

    # ── Borders ────────────────────────────────────────────────────────────────
    "border_subtle":    "#21262D",   # Near-invisible divider
    "border_muted":     "#30363D",   # Visible border

    # ── Typography ─────────────────────────────────────────────────────────────
    "text_primary":     "#E6EDF3",   # Off-white body/heading text
    "text_secondary":   "#8B949E",   # Muted labels, hints
    "text_tertiary":    "#484F58",   # Ghost text, disabled, version labels
    "text_on_primary":  "#FFFFFF",   # Text on filled primary buttons
}

# ── Shadow definitions ─────────────────────────────────────────────────────────
# Each entry: (blur_radius, x_offset, y_offset, alpha) for QGraphicsDropShadowEffect
SHADOWS: Final[dict[str, tuple[int, int, int, int]]] = {
    "sm":  (8,  0, 1, 40),   # Subtle card lift
    "md":  (16, 0, 4, 60),   # Standard card elevation
    "lg":  (28, 0, 8, 80),   # Modal / floating panel
    "xl":  (48, 0, 16, 100), # Full overlay shadow
}

# Dashboard refresh intervals (in milliseconds)
REFRESH_INTERVALS: Final[dict[str, int]] = {
    "statistics": 30000,  # 30 seconds
    "candidate_list": 60000,  # 1 minute
    "notifications": 10000,  # 10 seconds
}
