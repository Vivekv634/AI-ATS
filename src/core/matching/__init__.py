"""Candidate-job matching engine module."""

from .matching_engine import (
    MatchingEngine,
    MatchResult,
    get_matching_engine,
)
from .skill_scorer import (
    EmbeddingSkillScorer,
    get_embedding_skill_scorer,
)

__all__ = [
    "MatchingEngine",
    "MatchResult",
    "get_matching_engine",
    "EmbeddingSkillScorer",
    "get_embedding_skill_scorer",
]
