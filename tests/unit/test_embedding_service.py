"""
Unit tests for EmbeddingService.
EmbeddingModel and VectorStore are injected as fakes — no GPU, no ChromaDB needed.
"""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.ml.embeddings.embedding_service import EmbeddingService
from src.ml.nlp.accurate_resume_parser import (
    AccurateResumeParser,
    ParsedResume,
    ContactInfo,
    SkillCategory,
    ExperienceEntry,
    EducationEntry,
    ProjectEntry,
)


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

class FakeEmbeddingModel:
    """Returns a deterministic 4-dim vector without loading any model."""
    def encode_resume(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)


class FakeVectorStore:
    """Records upsert calls for assertion."""
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.calls.append({"ids": ids, "documents": documents, "metadatas": metadatas})


def _make_parsed(
    *,
    name: str = "Vivek Vaish",
    email: str = "v@test.com",
    summary: str = "Backend engineer",
) -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name=name, email=email),
        skills=[SkillCategory(category="Languages", skills=["python", "javascript"])],
        experience=[ExperienceEntry(title="SWE Intern", company="Acme", bullets=["Built API"])],
        education=[EducationEntry(degree="B.Tech", institution="Galgotias University")],
        projects=[ProjectEntry(name="AutoBook", bullets=["Automated booking"])],
        summary=summary,
        raw_text="raw resume text",
    )


def _make_service(repo: Optional[Any] = None) -> EmbeddingService:
    return EmbeddingService(
        model=FakeEmbeddingModel(),
        store=FakeVectorStore(),
        repo=repo,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildText:
    def test_includes_summary(self) -> None:
        svc: EmbeddingService = _make_service()
        parsed: ParsedResume = _make_parsed(summary="Experienced backend developer")
        text: str = svc._build_text(parsed)
        assert "experienced backend developer" in text.lower()

    def test_includes_skills(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "python" in text.lower()

    def test_includes_experience_title(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "swe intern" in text.lower()

    def test_includes_education(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "galgotias" in text.lower()

    def test_includes_project_name(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "autobook" in text.lower()

    def test_returns_non_empty_string(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert len(text) > 30


class TestEmbedCandidate:
    def test_returns_embedding_id_string(self) -> None:
        svc: EmbeddingService = _make_service()
        eid: str = svc.embed_candidate("abc123", _make_parsed())
        assert eid == "candidate_abc123"

    def test_upsert_called_with_correct_id(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        assert store.calls[0]["ids"] == ["candidate_xyz"]

    def test_upsert_called_with_document(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        assert store.calls[0]["documents"] is not None
        assert len(store.calls[0]["documents"][0]) > 0

    def test_metadata_contains_candidate_id(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        meta: dict[str, Any] = store.calls[0]["metadatas"][0]
        assert meta["candidate_id"] == "xyz"

    def test_repo_set_embedding_id_called(self) -> None:
        repo: MagicMock = MagicMock()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=FakeVectorStore(), repo=repo)
        svc.embed_candidate("xyz", _make_parsed())
        repo.set_embedding_id.assert_called_once_with("xyz", "candidate_xyz")

    def test_repo_none_does_not_raise(self) -> None:
        svc: EmbeddingService = _make_service(repo=None)
        eid: str = svc.embed_candidate("xyz", _make_parsed())
        assert eid == "candidate_xyz"
