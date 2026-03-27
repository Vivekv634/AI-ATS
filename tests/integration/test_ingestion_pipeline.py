"""
End-to-end ingestion pipeline tests using real PDFs from data/raw/resumes/.
MongoDB is replaced with an in-memory FakeRepo — no live DB required.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
