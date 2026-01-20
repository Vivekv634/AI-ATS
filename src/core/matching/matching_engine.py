"""
Candidate-Job matching engine.

Scores and ranks candidates against job requirements using multiple
matching strategies including skills matching, experience matching,
education matching, and keyword matching.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.data.models import (
    EducationMatch,
    ExperienceMatch,
    Explanation,
    ExplanationFactor,
    KeywordMatch,
    Match,
    MatchStatus,
    ScoreBreakdown,
    SkillMatch,
)
from src.ml.nlp import ResumeParseResult, JDParseResult
from src.utils.constants import (
    DEFAULT_SCORING_WEIGHTS,
    EDUCATION_LEVELS,
    MatchScoreLevel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Complete result of matching a candidate to a job."""

    # Basic info
    candidate_name: str = ""
    job_title: str = ""

    # Scores
    overall_score: float = 0.0
    score_level: MatchScoreLevel = MatchScoreLevel.POOR

    # Component scores
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    keyword_score: float = 0.0

    # Detailed matches
    skill_matches: list[SkillMatch] = field(default_factory=list)
    experience_match: Optional[ExperienceMatch] = None
    education_match: Optional[EducationMatch] = None
    keyword_match: Optional[KeywordMatch] = None

    # Breakdown
    score_breakdown: Optional[ScoreBreakdown] = None

    # Explanation
    explanation: Optional[Explanation] = None

    # Raw data references
    resume_result: Optional[ResumeParseResult] = None
    jd_result: Optional[JDParseResult] = None

    @property
    def matched_skills(self) -> list[str]:
        """Get list of matched skills."""
        return [s.skill_name for s in self.skill_matches if s.candidate_has_skill]

    @property
    def missing_skills(self) -> list[str]:
        """Get list of missing required skills."""
        return [
            s.skill_name for s in self.skill_matches
            if s.required and not s.candidate_has_skill
        ]


class MatchingEngine:
    """
    Engine for scoring candidates against job requirements.

    Uses a multi-factor approach:
    - Skills matching (required vs preferred)
    - Experience years matching
    - Education level matching
    - Keyword/terminology matching
    """

    def __init__(self, weights: Optional[dict[str, float]] = None):
        """
        Initialize the matching engine.

        Args:
            weights: Optional custom scoring weights
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS

    def match(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> MatchResult:
        """
        Match a candidate resume against a job description.

        Args:
            resume_result: Parsed resume data
            jd_result: Parsed job description data

        Returns:
            MatchResult with scores and explanations
        """
        result = MatchResult(
            resume_result=resume_result,
            jd_result=jd_result,
        )

        # Extract candidate name
        if resume_result.contact:
            name = resume_result.contact.get("full_name")
            if not name:
                first = resume_result.contact.get("first_name", "")
                last = resume_result.contact.get("last_name", "")
                name = f"{first} {last}".strip()
            result.candidate_name = name or "Unknown Candidate"
        else:
            result.candidate_name = "Unknown Candidate"

        result.job_title = jd_result.title or "Unknown Position"

        # Calculate individual scores
        result.skill_matches, result.skills_score = self._match_skills(
            resume_result, jd_result
        )

        result.experience_match, result.experience_score = self._match_experience(
            resume_result, jd_result
        )

        result.education_match, result.education_score = self._match_education(
            resume_result, jd_result
        )

        result.keyword_match, result.keyword_score = self._match_keywords(
            resume_result, jd_result
        )

        # Calculate weighted overall score
        result.score_breakdown = self._calculate_breakdown(result)
        result.overall_score = result.score_breakdown.total_score
        result.score_level = MatchScoreLevel.from_score(result.overall_score)

        # Generate explanation
        result.explanation = self._generate_explanation(result)

        return result

    def _match_skills(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[list[SkillMatch], float]:
        """Match candidate skills against job requirements."""
        skill_matches = []

        # Get candidate skills (lowercase for comparison)
        candidate_skills = {
            s["name"].lower(): s for s in resume.skills
        }

        # Get all required skills
        required_skills = set(s.lower() for s in jd.required_skills)
        preferred_skills = set(s.lower() for s in jd.preferred_skills)

        # Score required skills
        required_matched = 0
        for skill in required_skills:
            has_skill = skill in candidate_skills
            match = SkillMatch(
                skill_name=skill,
                required=True,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            )

            # Check for partial match (related skills)
            if not has_skill:
                related = self._find_related_skill(skill, candidate_skills.keys())
                if related:
                    match.partial_match = True
                    match.related_skill = related
                    match.match_score = 0.5

            if has_skill:
                required_matched += 1
            elif match.partial_match:
                required_matched += 0.5

            skill_matches.append(match)

        # Score preferred skills
        preferred_matched = 0
        for skill in preferred_skills:
            has_skill = skill in candidate_skills
            match = SkillMatch(
                skill_name=skill,
                required=False,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            )

            if has_skill:
                preferred_matched += 1

            skill_matches.append(match)

        # Calculate overall skills score
        score = 0.0
        total_weight = 0.0

        if required_skills:
            # Required skills are worth more
            required_weight = 0.7
            score += required_weight * (required_matched / len(required_skills))
            total_weight += required_weight

        if preferred_skills:
            # Preferred skills contribute less
            preferred_weight = 0.3
            score += preferred_weight * (preferred_matched / len(preferred_skills))
            total_weight += preferred_weight

        if total_weight > 0:
            score = score / total_weight
        elif candidate_skills:
            # If no requirements specified but candidate has skills
            score = 0.5

        return skill_matches, round(score, 3)

    def _find_related_skill(
        self, target: str, candidate_skills: set[str]
    ) -> Optional[str]:
        """Find a related skill in candidate's skill set."""
        # Simple related skill mappings
        related_groups = [
            {"python", "django", "flask", "fastapi"},
            {"javascript", "typescript", "node.js", "react", "angular", "vue"},
            {"java", "spring", "hibernate", "kotlin"},
            {"c#", ".net", "asp.net"},
            {"sql", "mysql", "postgresql", "oracle"},
            {"aws", "gcp", "azure", "cloud"},
            {"docker", "kubernetes", "containerization"},
            {"machine learning", "deep learning", "tensorflow", "pytorch"},
        ]

        target_lower = target.lower()
        for group in related_groups:
            if target_lower in group:
                for skill in candidate_skills:
                    if skill.lower() in group and skill.lower() != target_lower:
                        return skill
        return None

    def _match_experience(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[ExperienceMatch], float]:
        """Match candidate experience against job requirements."""
        candidate_years = resume.total_experience_years
        required_years = jd.experience_years_min or 0

        if required_years == 0:
            # No experience requirement
            return None, 1.0 if candidate_years > 0 else 0.5

        years_diff = candidate_years - required_years

        match = ExperienceMatch(
            required_years=required_years,
            candidate_years=candidate_years,
            years_difference=years_diff,
            meets_minimum=candidate_years >= required_years,
            score=0.0,
        )

        # Calculate score
        if candidate_years >= required_years:
            # Meets or exceeds requirement
            match.score = 1.0
        elif candidate_years >= required_years * 0.7:
            # Close to requirement (within 30%)
            match.score = 0.7 + 0.3 * (candidate_years / required_years)
        elif candidate_years > 0:
            # Has some experience
            match.score = 0.5 * (candidate_years / required_years)
        else:
            # No experience
            match.score = 0.0

        return match, round(match.score, 3)

    def _match_education(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[EducationMatch], float]:
        """Match candidate education against job requirements."""
        required_degree = jd.education_requirement
        candidate_degree = resume.highest_education

        if not required_degree:
            # No education requirement
            return None, 1.0 if candidate_degree else 0.7

        match = EducationMatch(
            required_degree=required_degree,
            candidate_degree=candidate_degree,
            meets_requirement=False,
            score=0.0,
        )

        if not candidate_degree:
            # No education info found
            match.score = 0.3
            return match, match.score

        # Get education levels
        required_level = EDUCATION_LEVELS.get(required_degree.lower(), 0)
        candidate_level = EDUCATION_LEVELS.get(candidate_degree.lower(), 0)

        if candidate_level >= required_level:
            match.meets_requirement = True
            match.score = 1.0
        elif candidate_level == required_level - 1:
            # One level below
            match.score = 0.7
        else:
            # More than one level below
            match.score = max(0.3, candidate_level / required_level) if required_level > 0 else 0.3

        return match, round(match.score, 3)

    def _match_keywords(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[KeywordMatch], float]:
        """Match keywords from JD in resume."""
        # Extract important keywords from JD
        jd_text = jd.raw_text.lower()
        resume_text = ""

        if resume.extraction_result:
            resume_text = resume.extraction_result.text.lower()
        elif resume.preprocessed:
            resume_text = resume.preprocessed.cleaned_text.lower()

        if not resume_text:
            return None, 0.0

        # Extract keywords from responsibilities and qualifications
        keywords = set()
        for resp in jd.responsibilities:
            words = resp.lower().split()
            keywords.update(w for w in words if len(w) > 3)
        for qual in jd.qualifications:
            words = qual.lower().split()
            keywords.update(w for w in words if len(w) > 3)

        # Common stopwords to exclude
        stopwords = {
            "with", "have", "will", "that", "this", "from", "your", "they",
            "their", "what", "when", "where", "which", "would", "could",
            "should", "must", "able", "about", "experience", "work", "team",
        }
        keywords -= stopwords

        if not keywords:
            return None, 0.5

        # Check matches
        matched = [kw for kw in keywords if kw in resume_text]
        missing = [kw for kw in keywords if kw not in resume_text]

        match = KeywordMatch(
            total_keywords=len(keywords),
            matched_keywords=len(matched),
            match_percentage=len(matched) / len(keywords) if keywords else 0,
            matched_terms=matched[:20],  # Limit for display
            missing_terms=missing[:10],
        )

        return match, round(match.match_percentage, 3)

    def _calculate_breakdown(self, result: MatchResult) -> ScoreBreakdown:
        """Calculate weighted score breakdown."""
        breakdown = ScoreBreakdown(
            skills_score=result.skills_score,
            skills_weight=self.weights["skills_match"],
            skills_weighted=result.skills_score * self.weights["skills_match"],

            experience_score=result.experience_score,
            experience_weight=self.weights["experience_match"],
            experience_weighted=result.experience_score * self.weights["experience_match"],

            education_score=result.education_score,
            education_weight=self.weights["education_match"],
            education_weighted=result.education_score * self.weights["education_match"],

            # Using keyword score for both semantic and keyword
            # (semantic requires embeddings which we're not implementing here)
            semantic_score=result.keyword_score,
            semantic_weight=self.weights["semantic_similarity"],
            semantic_weighted=result.keyword_score * self.weights["semantic_similarity"],

            keyword_score=result.keyword_score,
            keyword_weight=self.weights["keyword_match"],
            keyword_weighted=result.keyword_score * self.weights["keyword_match"],
        )

        return breakdown

    def _generate_explanation(self, result: MatchResult) -> Explanation:
        """Generate human-readable explanation of the match."""
        factors = []
        strengths = []
        gaps = []
        recommendations = []

        # Skills analysis
        matched_skills = result.matched_skills
        missing_skills = result.missing_skills

        if matched_skills:
            strengths.append(f"Matches {len(matched_skills)} required skills: {', '.join(matched_skills[:5])}")
            factors.append(ExplanationFactor(
                factor_name="Skills Match",
                factor_type="positive" if result.skills_score > 0.5 else "neutral",
                description=f"Candidate has {len(matched_skills)} of the required skills",
                impact_score=result.skills_score,
            ))

        if missing_skills:
            gaps.append(f"Missing skills: {', '.join(missing_skills[:5])}")
            recommendations.append(f"Consider candidates who also have: {', '.join(missing_skills[:3])}")

        # Experience analysis
        if result.experience_match:
            exp = result.experience_match
            if exp.meets_minimum:
                strengths.append(f"Has {exp.candidate_years:.1f} years experience (required: {exp.required_years:.1f})")
                factors.append(ExplanationFactor(
                    factor_name="Experience",
                    factor_type="positive",
                    description=f"Meets experience requirement with {exp.candidate_years:.1f} years",
                    impact_score=result.experience_score,
                ))
            else:
                gaps.append(f"Has {exp.candidate_years:.1f} years experience (required: {exp.required_years:.1f})")
                factors.append(ExplanationFactor(
                    factor_name="Experience",
                    factor_type="negative",
                    description=f"Below required experience ({exp.years_difference:.1f} years short)",
                    impact_score=result.experience_score,
                ))

        # Education analysis
        if result.education_match:
            edu = result.education_match
            if edu.meets_requirement:
                strengths.append(f"Has {edu.candidate_degree} degree (required: {edu.required_degree})")
            else:
                gaps.append(f"Has {edu.candidate_degree or 'no degree listed'} (required: {edu.required_degree})")

        # Generate summary
        if result.overall_score >= 0.85:
            summary = f"{result.candidate_name} is an excellent match for {result.job_title}"
        elif result.overall_score >= 0.70:
            summary = f"{result.candidate_name} is a good match for {result.job_title}"
        elif result.overall_score >= 0.50:
            summary = f"{result.candidate_name} is a fair match for {result.job_title}"
        else:
            summary = f"{result.candidate_name} may not be the best fit for {result.job_title}"

        return Explanation(
            summary=summary,
            factors=factors,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
        )

    def rank_candidates(
        self,
        candidates: list[MatchResult],
    ) -> list[MatchResult]:
        """
        Rank candidates by their overall match score.

        Args:
            candidates: List of match results

        Returns:
            Sorted list with highest scores first
        """
        return sorted(candidates, key=lambda x: x.overall_score, reverse=True)


# Singleton instance
_matching_engine: Optional[MatchingEngine] = None


def get_matching_engine() -> MatchingEngine:
    """Get the matching engine singleton instance."""
    global _matching_engine
    if _matching_engine is None:
        _matching_engine = MatchingEngine()
    return _matching_engine
