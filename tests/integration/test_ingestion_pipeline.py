"""
End-to-end ingestion pipeline tests using real PDFs from data/raw/resumes/.
MongoDB is replaced with an in-memory FakeRepo — no live DB required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from src.services.ingestion_service import IngestionService, IngestionResult
from src.ml.nlp.accurate_resume_parser import ParsedResume

RESUMES_DIR: Path = Path("data/raw/resumes")
ALL_PDFS: list[Path] = list(RESUMES_DIR.glob("*.pdf"))


# ---------------------------------------------------------------------------
# In-memory repository stub
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    id: str


class FakeRepo:
    """In-memory candidate store satisfying CandidateRepoProtocol."""

    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self._candidates: dict[str, _FakeCandidate] = {}   # email → candidate
        self._counter: int = 0

    def hash_exists(self, file_hash: str) -> bool:
        return file_hash in self._hashes

    def upsert_by_email(
        self,
        parsed: ParsedResume,
        file_hash: str,
        filename: str,
    ) -> _FakeCandidate:
        self._hashes.add(file_hash)
        email: str = (
            parsed.contact.email.lower()
            if parsed.contact.email
            else f"_no_email_{self._counter}"
        )
        self._counter += 1
        candidate = _FakeCandidate(id=str(self._counter))
        self._candidates[email] = candidate
        return candidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo() -> FakeRepo:
    return FakeRepo()


@pytest.fixture
def svc(repo: FakeRepo) -> IngestionService:
    from src.services.file_validator import FileValidator
    from src.ml.nlp.accurate_resume_parser import AccurateResumeParser

    s: IngestionService = IngestionService.__new__(IngestionService)
    s._validator = FileValidator()
    s._parser = AccurateResumeParser()
    s._repo = repo
    return s


# ---------------------------------------------------------------------------
# Single-resume tests
# ---------------------------------------------------------------------------

class TestSingleResume:
    def test_vivek_resume_ingested_successfully(self, svc: IngestionService) -> None:
        result: IngestionResult = svc.ingest_file(RESUMES_DIR / "vivek_resume.pdf")
        assert result.status == "success"
        assert result.candidate_name
        assert result.candidate_email == "vaishvivek634@gmail.com"
        assert len(result.file_hash) == 64
        assert result.processing_time_ms > 0

    def test_same_file_twice_is_duplicate(self, svc: IngestionService) -> None:
        path: Path = RESUMES_DIR / "vivek_resume.pdf"
        first: IngestionResult = svc.ingest_file(path)
        second: IngestionResult = svc.ingest_file(path)
        assert first.status == "success"
        assert second.status == "duplicate"

    def test_nonexistent_file_returns_error(self, svc: IngestionService) -> None:
        result: IngestionResult = svc.ingest_file(Path("does_not_exist.pdf"))
        assert result.status == "error"
        assert result.error_message


# ---------------------------------------------------------------------------
# Batch tests across all PDFs
# ---------------------------------------------------------------------------

class TestBatchIngestion:
    def test_all_real_pdfs_ingest_without_exception(
        self, svc: IngestionService
    ) -> None:
        assert len(ALL_PDFS) > 0, "No PDFs found in data/raw/resumes"
        statuses: list[str] = []
        for pdf in ALL_PDFS:
            result: IngestionResult = svc.ingest_file(pdf)
            assert result.status in ("success", "duplicate", "error"), (
                f"{pdf.name} returned unexpected status: {result.status}"
            )
            statuses.append(result.status)

        success_count: int = statuses.count("success")
        assert success_count >= len(ALL_PDFS) * 0.8, (
            f"Only {success_count}/{len(ALL_PDFS)} resumes ingested successfully"
        )

    def test_no_duplicate_hashes_in_batch(self, svc: IngestionService) -> None:
        """Each unique file must yield a success exactly once."""
        seen_hashes: set[str] = set()
        for pdf in ALL_PDFS:
            result: IngestionResult = svc.ingest_file(pdf)
            if result.status == "success":
                assert result.file_hash not in seen_hashes, (
                    f"Hash collision on: {pdf.name}"
                )
                seen_hashes.add(result.file_hash)


# ---------------------------------------------------------------------------
# Embedding text-building integration
# ---------------------------------------------------------------------------

class TestEmbeddingTextBuilding:
    """Verifies _build_text against real parsed PDFs. No model needed."""

    def test_vivek_text_contains_python(self) -> None:
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")
        svc: EmbeddingService = EmbeddingService(model=None, store=None, repo=None)  # type: ignore[arg-type]
        text: str = svc._build_text(parsed)
        assert "python" in text.lower()
        assert len(text) > 100

    def test_all_pdfs_produce_non_empty_text(self) -> None:
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume

        parser: AccurateResumeParser = AccurateResumeParser()
        svc: EmbeddingService = EmbeddingService(model=None, store=None, repo=None)  # type: ignore[arg-type]
        for pdf in ALL_PDFS:
            parsed: ParsedResume = parser.parse(pdf)
            text: str = svc._build_text(parsed)
            assert len(text) > 0, f"{pdf.name}: _build_text returned empty string"


# ---------------------------------------------------------------------------
# Job embedding text-building integration
# ---------------------------------------------------------------------------

class TestJobEmbeddingIntegration:
    """Verifies job-side text building and end-to-end SemanticMatcher path."""

    def test_build_jd_text_non_empty_for_real_job(self) -> None:
        import numpy as np
        from unittest.mock import MagicMock
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.data.models.job import Job, SkillRequirement

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build high-performance REST APIs using Python and FastAPI.",
            responsibilities=["Design scalable services", "Write unit tests"],
            company_name="Acme Corp",
            skill_requirements=[
                SkillRequirement(name="python", is_required=True),
                SkillRequirement(name="fastapi", is_required=True),
                SkillRequirement(name="docker", is_required=False),
            ],
        )
        svc: EmbeddingService = EmbeddingService(model=MagicMock(), store=MagicMock(), repo=None)  # type: ignore[arg-type]
        text: str = svc._build_jd_text(job)
        assert "python" in text.lower()
        assert "fastapi" in text.lower()
        assert len(text) > 50

    def test_vivek_resume_vs_python_job_semantic_similarity(self) -> None:
        """compute_similarity_from_parsed returns SemanticMatch for a real parsed resume."""
        import numpy as np
        from src.ml.embeddings.semantic_similarity import SemanticMatcher
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, SkillRequirement
        from src.data.models import SemanticMatch

        class _FakeModel:
            model_name: str = "fake"

            def encode(self, text: Any, **kw: Any) -> np.ndarray:
                if isinstance(text, list):
                    return np.ones((len(text), 4), dtype=np.float32)
                return np.ones(4, dtype=np.float32)

            def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
                return 0.85

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build REST APIs with Python and FastAPI.",
            responsibilities=["Design APIs", "Write tests"],
            company_name="Acme",
            skill_requirements=[SkillRequirement(name="python", is_required=True)],
        )

        matcher: SemanticMatcher = SemanticMatcher(
            embedding_model=_FakeModel()  # type: ignore[arg-type]
        )
        result: SemanticMatch = matcher.compute_similarity_from_parsed(parsed, job)
        assert result.overall_similarity == pytest.approx(0.85, abs=1e-4)
        assert 0.0 <= result.skills_similarity <= 1.0

    def test_match_from_parsed_pipeline_end_to_end(self) -> None:
        """MatchingEngine.match_from_parsed returns populated MatchResult for real PDF."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, SkillRequirement

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build REST APIs with Python and FastAPI.",
            responsibilities=["Design APIs", "Write unit tests"],
            company_name="Acme",
            skill_requirements=[SkillRequirement(name="python", is_required=True)],
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(parsed, job)

        assert result.overall_score > 0.0
        assert "vivek" in result.candidate_name.lower()
        assert result.score_breakdown is not None
