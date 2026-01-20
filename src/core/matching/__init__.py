"""Candidate-job matching engine module."""

from .matching_engine import (
    MatchingEngine,
    MatchResult,
    get_matching_engine,
)

__all__ = [
    "MatchingEngine",
    "MatchResult",
    "get_matching_engine",
]
