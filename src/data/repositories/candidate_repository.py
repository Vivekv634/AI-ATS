"""
Candidate repository for AI-ATS.

Provides data access operations for candidate documents,
including search and filtering capabilities.
"""

from typing import Any, Optional

from bson import ObjectId

from src.data.models.candidate import (
    Candidate,
    CandidateCreate,
    CandidateUpdate,
)
from src.utils.constants import CandidateStatus
from src.utils.logger import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class CandidateRepository(BaseRepository[Candidate]):
    """Repository for candidate document operations."""

    @property
    def collection_name(self) -> str:
        return "candidates"

    @property
    def model_class(self) -> type[Candidate]:
        return Candidate

    # -------------------------------------------------------------------------
    # Create Operations
    # -------------------------------------------------------------------------

    def create_from_schema(self, data: CandidateCreate) -> Candidate:
        """Create a candidate from a create schema."""
        candidate = Candidate(
            first_name=data.first_name,
            last_name=data.last_name,
            contact=data.contact,
            headline=data.headline,
            summary=data.summary,
            skills=data.skills,
            work_experience=data.work_experience,
            education=data.education,
            certifications=data.certifications,
            languages=data.languages,
            metadata=data.metadata if data.metadata else None,
        )
        return self.create(candidate)

    async def create_from_schema_async(self, data: CandidateCreate) -> Candidate:
        """Create a candidate from a create schema asynchronously."""
        candidate = Candidate(
            first_name=data.first_name,
            last_name=data.last_name,
            contact=data.contact,
            headline=data.headline,
            summary=data.summary,
            skills=data.skills,
            work_experience=data.work_experience,
            education=data.education,
            certifications=data.certifications,
            languages=data.languages,
            metadata=data.metadata if data.metadata else None,
        )
        return await self.create_async(candidate)

    # -------------------------------------------------------------------------
    # Update Operations
    # -------------------------------------------------------------------------

    def update_from_schema(
        self, id_value: str | ObjectId, data: CandidateUpdate
    ) -> Optional[Candidate]:
        """Update a candidate from an update schema."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return self.get_by_id(id_value)
        return self.update(id_value, update_data)

    async def update_from_schema_async(
        self, id_value: str | ObjectId, data: CandidateUpdate
    ) -> Optional[Candidate]:
        """Update a candidate from an update schema asynchronously."""
        update_data = data.model_dump(exclude_unset=True, exclude_none=True)
        if not update_data:
            return await self.get_by_id_async(id_value)
        return await self.update_async(id_value, update_data)

    def update_status(
        self, id_value: str | ObjectId, status: CandidateStatus
    ) -> Optional[Candidate]:
        """Update candidate status."""
        return self.update(id_value, {"status": status.value})

    async def update_status_async(
        self, id_value: str | ObjectId, status: CandidateStatus
    ) -> Optional[Candidate]:
        """Update candidate status asynchronously."""
        return await self.update_async(id_value, {"status": status.value})

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get_by_email(self, email: str) -> Optional[Candidate]:
        """Get a candidate by email address."""
        return self.find_one({"contact.email": email.lower()})

    async def get_by_email_async(self, email: str) -> Optional[Candidate]:
        """Get a candidate by email address asynchronously."""
        return await self.find_one_async({"contact.email": email.lower()})

    def email_exists(self, email: str) -> bool:
        """Check if a candidate with the given email exists."""
        return self.exists({"contact.email": email.lower()})

    async def email_exists_async(self, email: str) -> bool:
        """Check if a candidate with the given email exists asynchronously."""
        return await self.exists_async({"contact.email": email.lower()})

    def get_by_status(
        self,
        status: CandidateStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by status."""
        return self.find({"status": status.value}, skip=skip, limit=limit)

    async def get_by_status_async(
        self,
        status: CandidateStatus,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by status asynchronously."""
        return await self.find_async({"status": status.value}, skip=skip, limit=limit)

    def get_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by metadata tags."""
        if match_all:
            query = {"metadata.tags": {"$all": tags}}
        else:
            query = {"metadata.tags": {"$in": tags}}
        return self.find(query, skip=skip, limit=limit)

    async def get_by_tags_async(
        self,
        tags: list[str],
        match_all: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Get candidates by metadata tags asynchronously."""
        if match_all:
            query = {"metadata.tags": {"$all": tags}}
        else:
            query = {"metadata.tags": {"$in": tags}}
        return await self.find_async(query, skip=skip, limit=limit)

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(
        self,
        query_text: Optional[str] = None,
        status: Optional[CandidateStatus] = None,
        skills: Optional[list[str]] = None,
        min_experience_years: Optional[float] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """
        Search candidates with multiple filters.

        Args:
            query_text: Text to search in name, headline, summary
            status: Filter by candidate status
            skills: Filter by skill names (matches any)
            min_experience_years: Minimum years of experience
            tags: Filter by metadata tags (matches any)
            skip: Number of documents to skip
            limit: Maximum documents to return

        Returns:
            List of matching candidates
        """
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"first_name": {"$regex": query_text, "$options": "i"}},
                {"last_name": {"$regex": query_text, "$options": "i"}},
                {"headline": {"$regex": query_text, "$options": "i"}},
                {"summary": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if skills:
            # Normalize skill names for matching
            normalized_skills = [s.lower() for s in skills]
            query["skills.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        candidates = self.find(query, skip=skip, limit=limit)

        # Filter by experience in Python (calculated field)
        if min_experience_years is not None:
            candidates = [
                c for c in candidates
                if c.total_experience_years >= min_experience_years
            ]

        return candidates

    async def search_async(
        self,
        query_text: Optional[str] = None,
        status: Optional[CandidateStatus] = None,
        skills: Optional[list[str]] = None,
        min_experience_years: Optional[float] = None,
        tags: Optional[list[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Candidate]:
        """Search candidates with multiple filters asynchronously."""
        query: dict[str, Any] = {}

        if query_text:
            query["$or"] = [
                {"first_name": {"$regex": query_text, "$options": "i"}},
                {"last_name": {"$regex": query_text, "$options": "i"}},
                {"headline": {"$regex": query_text, "$options": "i"}},
                {"summary": {"$regex": query_text, "$options": "i"}},
            ]

        if status:
            query["status"] = status.value

        if skills:
            normalized_skills = [s.lower() for s in skills]
            query["skills.name"] = {"$in": normalized_skills}

        if tags:
            query["metadata.tags"] = {"$in": tags}

        candidates = await self.find_async(query, skip=skip, limit=limit)

        if min_experience_years is not None:
            candidates = [
                c for c in candidates
                if c.total_experience_years >= min_experience_years
            ]

        return candidates

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def get_status_counts(self) -> dict[str, int]:
        """Get count of candidates by status."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = list(collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in results}

    async def get_status_counts_async(self) -> dict[str, int]:
        """Get count of candidates by status asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = await collection.aggregate(pipeline).to_list(length=None)
        return {r["_id"]: r["count"] for r in results}

    def get_skill_distribution(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most common skills across all candidates."""
        collection = self._get_sync_collection()
        pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return list(collection.aggregate(pipeline))

    async def get_skill_distribution_async(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most common skills across all candidates asynchronously."""
        collection = self._get_async_collection()
        pipeline = [
            {"$unwind": "$skills"},
            {"$group": {"_id": "$skills.name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit},
        ]
        return await collection.aggregate(pipeline).to_list(length=limit)

    # -------------------------------------------------------------------------
    # Resume Linking
    # -------------------------------------------------------------------------

    def link_resume(
        self, candidate_id: str | ObjectId, resume_id: str | ObjectId
    ) -> Optional[Candidate]:
        """Link a resume to a candidate."""
        return self.update(candidate_id, {"resume_id": self._to_object_id(resume_id)})

    async def link_resume_async(
        self, candidate_id: str | ObjectId, resume_id: str | ObjectId
    ) -> Optional[Candidate]:
        """Link a resume to a candidate asynchronously."""
        return await self.update_async(
            candidate_id, {"resume_id": self._to_object_id(resume_id)}
        )

    def set_embedding_id(
        self, candidate_id: str | ObjectId, embedding_id: str
    ) -> Optional[Candidate]:
        """Set the vector embedding ID for a candidate."""
        return self.update(candidate_id, {"embedding_id": embedding_id})

    async def set_embedding_id_async(
        self, candidate_id: str | ObjectId, embedding_id: str
    ) -> Optional[Candidate]:
        """Set the vector embedding ID for a candidate asynchronously."""
        return await self.update_async(candidate_id, {"embedding_id": embedding_id})


# Singleton instance
_candidate_repository: Optional[CandidateRepository] = None


def get_candidate_repository() -> CandidateRepository:
    """Get the candidate repository singleton instance."""
    global _candidate_repository
    if _candidate_repository is None:
        _candidate_repository = CandidateRepository()
    return _candidate_repository
