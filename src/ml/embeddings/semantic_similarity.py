"""
Semantic similarity computation for resume-job matching.

Provides high-level functions for computing semantic similarity
between resumes and job descriptions using embeddings.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data.models import SemanticMatch
from src.ml.nlp import ResumeParseResult, JDParseResult
from src.utils.config import get_settings
from src.utils.logger import get_logger

from .embedding_model import EmbeddingModel, get_embedding_model
from .vector_store import (
    VectorStore,
    SearchResult,
    get_resume_store,
    get_job_store,
)

logger = get_logger(__name__)


@dataclass
class CandidateMatch:
    """Result of matching a candidate to a job using semantic similarity."""

    candidate_id: str
    candidate_name: str
    similarity_score: float
    semantic_match: SemanticMatch
    resume_text: Optional[str] = None


class SemanticMatcher:
    """
    Computes semantic similarity between resumes and job descriptions.

    Uses sentence embeddings to capture semantic meaning beyond
    keyword matching.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        resume_store: Optional[VectorStore] = None,
        job_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the semantic matcher.

        Args:
            embedding_model: Optional custom embedding model.
            resume_store: Optional custom vector store for resumes.
            job_store: Optional custom vector store for jobs.
        """
        self._embedding_model = embedding_model
        self._resume_store = resume_store
        self._job_store = job_store
        self._model_name = get_settings().ml.embedding_model

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model (lazy initialization)."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def resume_store(self) -> VectorStore:
        """Get the resume vector store (lazy initialization)."""
        if self._resume_store is None:
            self._resume_store = get_resume_store()
        return self._resume_store

    @property
    def job_store(self) -> VectorStore:
        """Get the job vector store (lazy initialization)."""
        if self._job_store is None:
            self._job_store = get_job_store()
        return self._job_store

    def compute_similarity(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> SemanticMatch:
        """
        Compute semantic similarity between a resume and job description.

        Args:
            resume_result: Parsed resume data.
            jd_result: Parsed job description data.

        Returns:
            SemanticMatch with similarity scores.
        """
        # Extract text content
        resume_text = self._get_resume_text(resume_result)
        jd_text = jd_result.raw_text

        if not resume_text or not jd_text:
            return SemanticMatch(model_used=self._model_name)

        # Compute overall similarity
        resume_embedding = self.embedding_model.encode(resume_text)
        jd_embedding = self.embedding_model.encode(jd_text)
        overall_sim = self.embedding_model.similarity(resume_embedding, jd_embedding)

        # Compute section-specific similarities
        skills_sim = self._compute_skills_similarity(resume_result, jd_result)
        exp_sim = self._compute_experience_similarity(resume_result, jd_result)
        summary_sim = self._compute_summary_similarity(resume_result, jd_result)

        return SemanticMatch(
            overall_similarity=round(overall_sim, 4),
            summary_similarity=round(summary_sim, 4),
            skills_similarity=round(skills_sim, 4),
            experience_similarity=round(exp_sim, 4),
            model_used=self._model_name,
        )

    def _get_resume_text(self, resume_result: ResumeParseResult) -> str:
        """Extract text content from parsed resume."""
        if resume_result.extraction_result:
            return resume_result.extraction_result.text
        elif resume_result.preprocessed:
            return resume_result.preprocessed.cleaned_text
        return ""

    def _compute_skills_similarity(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> float:
        """Compute similarity between candidate skills and job requirements."""
        # Get candidate skills text
        candidate_skills = [s.get("name", "") for s in resume_result.skills if s.get("name")]
        if not candidate_skills:
            return 0.0

        # Get required skills
        required_skills = jd_result.required_skills + jd_result.preferred_skills
        if not required_skills:
            return 0.5

        # Create skill text representations
        candidate_skills_text = ", ".join(candidate_skills)
        required_skills_text = ", ".join(required_skills)

        # Compute similarity
        candidate_emb = self.embedding_model.encode(candidate_skills_text)
        required_emb = self.embedding_model.encode(required_skills_text)

        return self.embedding_model.similarity(candidate_emb, required_emb)

    def _compute_experience_similarity(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> float:
        """Compute similarity between candidate experience and job requirements."""
        # Extract experience text from resume
        exp_text_parts = []
        if resume_result.preprocessed and resume_result.preprocessed.sections:
            for section in resume_result.preprocessed.sections:
                if section.section_type in ["experience", "work_history"]:
                    exp_text_parts.append(section.content)

        if not exp_text_parts:
            return 0.0

        experience_text = " ".join(exp_text_parts)

        # Get job responsibilities text
        responsibilities_text = " ".join(jd_result.responsibilities)
        if not responsibilities_text:
            return 0.5

        # Compute similarity
        exp_emb = self.embedding_model.encode(experience_text)
        resp_emb = self.embedding_model.encode(responsibilities_text)

        return self.embedding_model.similarity(exp_emb, resp_emb)

    def _compute_summary_similarity(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> float:
        """Compute similarity between candidate summary and job description."""
        # Extract summary from resume
        summary_text = ""
        if resume_result.preprocessed and resume_result.preprocessed.sections:
            for section in resume_result.preprocessed.sections:
                if section.section_type in ["summary", "objective", "profile"]:
                    summary_text = section.content
                    break

        if not summary_text:
            # Use first part of resume as summary
            full_text = self._get_resume_text(resume_result)
            summary_text = full_text[:500] if full_text else ""

        if not summary_text:
            return 0.0

        # Get job summary (title + first part of description)
        jd_summary = f"{jd_result.title}. {jd_result.raw_text[:500]}"

        # Compute similarity
        summary_emb = self.embedding_model.encode(summary_text)
        jd_emb = self.embedding_model.encode(jd_summary)

        return self.embedding_model.similarity(summary_emb, jd_emb)

    def index_resume(
        self,
        resume_id: str,
        resume_result: ResumeParseResult,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Index a resume in the vector store for later search.

        Args:
            resume_id: Unique identifier for the resume.
            resume_result: Parsed resume data.
            metadata: Optional additional metadata.
        """
        resume_text = self._get_resume_text(resume_result)
        if not resume_text:
            logger.warning(f"Cannot index resume {resume_id}: no text content")
            return

        embedding = self.embedding_model.encode(resume_text)

        # Prepare metadata
        meta = {
            "type": "resume",
            "candidate_name": self._get_candidate_name(resume_result),
            "skills": ",".join(s.get("name", "") for s in resume_result.skills[:20]),
            "experience_years": resume_result.total_experience_years,
            "education": resume_result.highest_education or "",
        }
        if metadata:
            meta.update(metadata)

        self.resume_store.upsert(
            ids=[resume_id],
            embeddings=embedding.reshape(1, -1),
            documents=[resume_text[:2000]],  # Truncate for storage
            metadatas=[meta],
        )

        logger.debug(f"Indexed resume: {resume_id}")

    def index_job(
        self,
        job_id: str,
        jd_result: JDParseResult,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Index a job description in the vector store for later search.

        Args:
            job_id: Unique identifier for the job.
            jd_result: Parsed job description data.
            metadata: Optional additional metadata.
        """
        jd_text = jd_result.raw_text
        if not jd_text:
            logger.warning(f"Cannot index job {job_id}: no text content")
            return

        embedding = self.embedding_model.encode(jd_text)

        # Prepare metadata
        meta = {
            "type": "job",
            "title": jd_result.title or "",
            "company": jd_result.company_name or "",
            "required_skills": ",".join(jd_result.required_skills[:20]),
            "experience_level": jd_result.experience_level or "",
        }
        if metadata:
            meta.update(metadata)

        self.job_store.upsert(
            ids=[job_id],
            embeddings=embedding.reshape(1, -1),
            documents=[jd_text[:2000]],
            metadatas=[meta],
        )

        logger.debug(f"Indexed job: {job_id}")

    def find_matching_candidates(
        self,
        jd_result: JDParseResult,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find candidates that match a job description.

        Args:
            jd_result: Parsed job description.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching candidates sorted by similarity.
        """
        jd_embedding = self.embedding_model.encode(jd_result.raw_text)

        results = self.resume_store.search(
            query_embedding=jd_embedding,
            top_k=top_k,
        )

        # Filter by minimum score
        return [r for r in results if r.score >= min_score]

    def find_matching_jobs(
        self,
        resume_result: ResumeParseResult,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find jobs that match a candidate's resume.

        Args:
            resume_result: Parsed resume.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching jobs sorted by similarity.
        """
        resume_text = self._get_resume_text(resume_result)
        if not resume_text:
            return []

        resume_embedding = self.embedding_model.encode(resume_text)

        results = self.job_store.search(
            query_embedding=resume_embedding,
            top_k=top_k,
        )

        return [r for r in results if r.score >= min_score]

    def _get_candidate_name(self, resume_result: ResumeParseResult) -> str:
        """Extract candidate name from resume."""
        if resume_result.contact:
            name = resume_result.contact.get("full_name")
            if not name:
                first = resume_result.contact.get("first_name", "")
                last = resume_result.contact.get("last_name", "")
                name = f"{first} {last}".strip()
            return name or "Unknown"
        return "Unknown"

    def batch_compute_similarity(
        self,
        resume_results: list[ResumeParseResult],
        jd_result: JDParseResult,
        show_progress: bool = False,
    ) -> list[tuple[int, SemanticMatch]]:
        """
        Compute similarity for multiple resumes against a single job.

        Args:
            resume_results: List of parsed resumes.
            jd_result: Parsed job description.
            show_progress: Whether to show progress bar.

        Returns:
            List of (index, SemanticMatch) tuples.
        """
        results = []

        # Get JD embedding once
        jd_embedding = self.embedding_model.encode(jd_result.raw_text)

        # Get all resume texts
        resume_texts = [self._get_resume_text(r) for r in resume_results]

        # Batch encode resumes
        valid_indices = [i for i, t in enumerate(resume_texts) if t]
        valid_texts = [resume_texts[i] for i in valid_indices]

        if not valid_texts:
            return []

        resume_embeddings = self.embedding_model.encode(
            valid_texts,
            show_progress=show_progress,
        )

        # Compute similarities
        for idx, (i, emb) in enumerate(zip(valid_indices, resume_embeddings)):
            overall_sim = self.embedding_model.similarity(emb, jd_embedding)

            # For batch processing, use overall similarity for all components
            # (detailed component analysis would be too slow)
            match = SemanticMatch(
                overall_similarity=round(overall_sim, 4),
                summary_similarity=round(overall_sim, 4),
                skills_similarity=round(overall_sim, 4),
                experience_similarity=round(overall_sim, 4),
                model_used=self._model_name,
            )

            results.append((i, match))

        return results


# Singleton instance
_semantic_matcher: Optional[SemanticMatcher] = None


def get_semantic_matcher() -> SemanticMatcher:
    """Get the semantic matcher singleton instance."""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = SemanticMatcher()
    return _semantic_matcher
