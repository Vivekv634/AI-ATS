"""
EmbeddingService — generates and stores resume embeddings after ingestion.

Flow:
  ParsedResume → build text → EmbeddingModel.encode_resume() →
  VectorStore.upsert() → CandidateRepository.set_embedding_id()
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from src.ml.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from src.ml.embeddings.vector_store import VectorStore, get_resume_store
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ParsedResume

logger = get_logger(__name__)


class EmbeddingService:
    """
    Thin orchestrator: build resume text → encode → upsert to VectorStore → link ID.

    Inject fakes for model and store in tests; uses real singletons in production.
    """

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        store: Optional[VectorStore] = None,
        repo: Optional[Any] = None,
    ) -> None:
        self._model: EmbeddingModel = model or get_embedding_model()
        self._store: VectorStore = store or get_resume_store()
        self._repo: Optional[Any] = repo

    def embed_candidate(
        self,
        candidate_id: str,
        parsed: "ParsedResume",
    ) -> str:
        """
        Generate, store, and link an embedding for one candidate.

        Args:
            candidate_id: MongoDB ID string of the stored Candidate document.
            parsed: Fully parsed resume from AccurateResumeParser.

        Returns:
            embedding_id — the ChromaDB/FAISS document ID used to store the vector.
        """
        text: str = self._build_text(parsed)
        embedding: np.ndarray = self._model.encode_resume(text)
        embedding_id: str = f"candidate_{candidate_id}"

        self._store.upsert(
            ids=[embedding_id],
            embeddings=np.array([embedding]),
            documents=[text],
            metadatas=[{
                "candidate_id": candidate_id,
                "candidate_name": parsed.contact.name,
                "email": parsed.contact.email,
            }],
        )

        if self._repo is not None:
            self._repo.set_embedding_id(candidate_id, embedding_id)

        logger.info(f"Embedded candidate {candidate_id} → {embedding_id}")
        return embedding_id

    def _build_text(self, parsed: "ParsedResume") -> str:
        """
        Build a plain-text representation of a ParsedResume for embedding.

        Concatenates: summary, skill categories, experience titles + bullets,
        education entries, and project names + bullets.
        """
        parts: list[str] = []

        if parsed.summary:
            parts.append(parsed.summary)

        for cat in parsed.skills:
            line: str = f"{cat.category}: {', '.join(cat.skills)}"
            parts.append(line)

        for exp in parsed.experience:
            header: str = f"{exp.title} at {exp.company}".strip(" at")
            parts.append(header)
            parts.extend(exp.bullets[:3])

        for edu in parsed.education:
            parts.append(f"{edu.degree} {edu.institution}".strip())

        for proj in parsed.projects:
            parts.append(proj.name)
            parts.extend(proj.bullets[:2])

        text: str = " ".join(p for p in parts if p.strip())

        # Fall back to raw_text when no structured sections produced content
        if not text and parsed.raw_text:
            text = parsed.raw_text

        return text
