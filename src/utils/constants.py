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
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "secondary": "#64748b",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "background": "#f8fafc",
    "surface": "#ffffff",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
}

# Dashboard refresh intervals (in milliseconds)
REFRESH_INTERVALS: Final[dict[str, int]] = {
    "statistics": 30000,  # 30 seconds
    "candidate_list": 60000,  # 1 minute
    "notifications": 10000,  # 10 seconds
}
