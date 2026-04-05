import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
import numpy as np
from unittest.mock import Mock

from src.data.models.match import SemanticMatch
from src.ml.nlp.accurate_resume_parser import (
    ParsedResume, ExperienceEntry, SkillCategory, ContactInfo,
)
from src.data.models.job import Job, SkillRequirement


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_job(
    title: str = "Software Engineer",
    description: str = "Build reliable software.",
    responsibilities: list[str] | None = None,
    skill_names: list[str] | None = None,
) -> Job:
    return Job(
        title=title,
        description=description,
        responsibilities=responsibilities or ["Design systems", "Write code"],
        company_name="Acme Corp",
        skill_requirements=[
            SkillRequirement(name=s, is_required=True)
            for s in (skill_names or ["Python", "SQL"])
        ],
    )


def _make_parsed(
    summary: str = "Experienced developer.",
    skill_names: list[str] | None = None,
    exp_bullets: list[str] | None = None,
    raw_text: str = "Developer with Python experience.",
) -> ParsedResume:
    return ParsedResume(
        summary=summary,
        skills=[SkillCategory(category="Technical", skills=skill_names or ["Python", "Django"])],
        experience=[ExperienceEntry(
            title="Developer",
            company="Tech Co",
            bullets=exp_bullets or ["Built REST APIs", "Led code reviews"],
        )],
        raw_text=raw_text,
    )


def _mock_model(similarity_values: list[float] | None = None):
    """
    Mock EmbeddingModel.

    encode(str)  → 1-D unit array (JD side; in production served from LRU cache)
    encode(list) → 2-D array, one distinct unit vector per text (resume batch)
    similarity   → values consumed from similarity_values in call order
    """
    from src.ml.embeddings.embedding_model import EmbeddingModel

    mock = Mock(spec=EmbeddingModel)
    DIM: int = 8

    def _encode(texts, **kwargs):
        if isinstance(texts, list):
            n: int = len(texts)
            embs: np.ndarray = np.zeros((n, DIM))
            for i in range(n):
                embs[i, i % DIM] = 1.0
            return embs
        # Single string (JD-side): derive a distinct unit vector from the text
        # so that per-section routing bugs (wrong emb compared to wrong JD emb)
        # are detectable — all-identical vectors would mask them.
        dim_idx: int = abs(hash(texts)) % DIM
        vec: np.ndarray = np.zeros(DIM)
        vec[dim_idx] = 1.0
        return vec

    mock.encode.side_effect = _encode

    if similarity_values is not None:
        mock.similarity.side_effect = list(similarity_values)
    else:
        mock.similarity.return_value = 0.6

    return mock


# ── SemanticMatch model ───────────────────────────────────────────────────────

def test_semantic_match_has_weighted_similarity_field() -> None:
    sm: SemanticMatch = SemanticMatch()
    assert sm.weighted_similarity == 0.0


def test_semantic_match_weighted_similarity_clamped_above() -> None:
    sm: SemanticMatch = SemanticMatch(weighted_similarity=1.5)
    assert sm.weighted_similarity == pytest.approx(1.0)


def test_semantic_match_weighted_similarity_clamped_below() -> None:
    sm: SemanticMatch = SemanticMatch(weighted_similarity=-0.2)
    assert sm.weighted_similarity == pytest.approx(0.0)


# ── _build_resume_text ────────────────────────────────────────────────────────

def test_build_resume_text_includes_all_bullets() -> None:
    """Removing the [:3] cap means all bullets appear in the embedding text."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = [f"task_{i}" for i in range(6)]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)
    matcher: SemanticMatcher = SemanticMatcher(embedding_model=_mock_model())

    text: str = matcher._build_resume_text(parsed)

    for bullet in bullets:
        assert bullet in text, f"Expected bullet '{bullet}' in resume text"


def test_build_resume_text_three_bullets_unaffected() -> None:
    """Regression: resumes with ≤3 bullets still work correctly."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = ["alpha", "beta", "gamma"]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)
    matcher: SemanticMatcher = SemanticMatcher(embedding_model=_mock_model())

    text: str = matcher._build_resume_text(parsed)

    for bullet in bullets:
        assert bullet in text


# ── compute_similarity_from_parsed ───────────────────────────────────────────

def test_weighted_similarity_formula() -> None:
    """weighted_similarity = 0.35*overall + 0.30*skills + 0.25*exp + 0.10*summary."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher, _SECTION_WEIGHTS

    overall, skills, exp, summary = 0.80, 0.60, 0.70, 0.50
    expected: float = (
        _SECTION_WEIGHTS["overall"] * overall
        + _SECTION_WEIGHTS["skills"] * skills
        + _SECTION_WEIGHTS["experience"] * exp
        + _SECTION_WEIGHTS["summary"] * summary
    )

    result: SemanticMatch = SemanticMatcher(
        embedding_model=_mock_model([overall, skills, exp, summary])
    ).compute_similarity_from_parsed(_make_parsed(), _make_job())

    assert result.weighted_similarity == pytest.approx(expected, abs=1e-4)


def test_all_four_section_scores_are_populated() -> None:
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    overall, skills, exp, summary = 0.80, 0.60, 0.70, 0.50
    result: SemanticMatch = SemanticMatcher(
        embedding_model=_mock_model([overall, skills, exp, summary])
    ).compute_similarity_from_parsed(_make_parsed(), _make_job())

    assert result.overall_similarity == pytest.approx(overall, abs=1e-4)
    assert result.skills_similarity == pytest.approx(skills, abs=1e-4)
    assert result.experience_similarity == pytest.approx(exp, abs=1e-4)
    assert result.summary_similarity == pytest.approx(summary, abs=1e-4)


def test_resume_side_uses_one_batch_encode_call() -> None:
    """All 4 resume-side texts are batched into exactly one encode(list) call."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock: Mock = _mock_model()
    SemanticMatcher(embedding_model=mock).compute_similarity_from_parsed(
        _make_parsed(), _make_job()
    )

    list_calls: list = [
        c for c in mock.encode.call_args_list if isinstance(c.args[0], list)
    ]
    assert len(list_calls) == 1, (
        f"Expected exactly 1 batch encode call, got {len(list_calls)}"
    )


def test_batch_encode_contains_all_bullet_text() -> None:
    """The experience text passed in the batch includes all bullets, not just 3."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = [f"bullet_{i}" for i in range(6)]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)

    mock: Mock = _mock_model()
    SemanticMatcher(embedding_model=mock).compute_similarity_from_parsed(
        parsed, _make_job()
    )

    list_calls: list = [
        c for c in mock.encode.call_args_list if isinstance(c.args[0], list)
    ]
    all_batch_text: str = " ".join(list_calls[0].args[0])
    for bullet in bullets:
        assert bullet in all_batch_text, f"'{bullet}' missing from batch texts"


# ── matching_engine integration ───────────────────────────────────────────────

def test_matching_engine_uses_weighted_similarity_as_semantic_score() -> None:
    """semantic_score in MatchResult equals weighted_similarity, not overall_similarity."""
    from src.core.matching.matching_engine import MatchingEngine

    engine: MatchingEngine = MatchingEngine(
        use_semantic=True, use_bias_detection=False, use_explainability=False
    )

    known_weighted: float = 0.73
    known_overall: float = 0.55

    mock_sm: SemanticMatch = SemanticMatch(
        overall_similarity=known_overall,
        weighted_similarity=known_weighted,
        skills_similarity=0.80,
        experience_similarity=0.70,
        summary_similarity=0.50,
    )
    mock_matcher: Mock = Mock()
    mock_matcher.compute_similarity_from_parsed.return_value = mock_sm
    engine._semantic_matcher = mock_matcher

    result = engine.match_from_parsed(_make_parsed(), _make_job())

    assert result.semantic_score == pytest.approx(known_weighted, abs=1e-3), (
        f"Expected semantic_score={known_weighted}, got {result.semantic_score}. "
        "matching_engine should use weighted_similarity, not overall_similarity."
    )


# ── compute_similarity() helpers and fixture ─────────────────────────────────

def _make_resume_result(
    text: str = "Python developer with 5 years of experience building web apps.",
    skills: list[dict] | None = None,
) -> Mock:
    """Minimal ResumeParseResult mock for compute_similarity() tests."""
    result = Mock()
    result.extraction_result.text = text
    result.skills = skills or [{"name": "Python"}, {"name": "Django"}]
    result.preprocessed = None  # forces fallback to extraction_result.text
    return result


def _make_jd_result(
    raw_text: str = "Software engineer role requiring Python and SQL skills.",
    required_skills: list[str] | None = None,
    responsibilities: list[str] | None = None,
    title: str = "Software Engineer",
) -> Mock:
    """Minimal JDParseResult mock for compute_similarity() tests."""
    result = Mock()
    result.raw_text = raw_text
    result.required_skills = required_skills or ["Python", "SQL"]
    result.preferred_skills = []
    result.responsibilities = responsibilities or ["Build APIs", "Write tests"]
    result.title = title
    return result


@pytest.fixture
def mock_embedding_model() -> Mock:
    """
    Deterministic embedding model stub for compute_similarity() tests.

    encode(str)  → content-hashed unit vector (distinct per text, catches
                   per-section routing bugs where wrong resume emb is compared
                   against wrong JD emb — aligned with _mock_model() behaviour)
    encode(list) → one distinct unit vector per text (resume batch side)
    similarity   → dot product of the two unit vectors
    """
    DIM: int = 4
    model = Mock()

    def _encode(texts, normalize=True, show_progress=False):
        if isinstance(texts, str):
            dim_idx: int = abs(hash(texts)) % DIM
            vec = np.zeros(DIM)
            vec[dim_idx] = 1.0
            return vec
        n = len(texts)
        embs = np.zeros((n, DIM))
        for i in range(n):
            embs[i, i % DIM] = 1.0
        return embs

    model.encode.side_effect = _encode
    model.similarity.side_effect = lambda a, b: float(np.dot(a, b))
    return model


def test_compute_similarity_returns_nonzero_weighted_similarity() -> None:
    """compute_similarity() must return weighted_similarity > 0 when both sides have content."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    # _mock_model() with no args returns similarity=0.6 for all section pairs,
    # ensuring weighted_similarity > 0 without depending on vector alignment.
    matcher = SemanticMatcher(embedding_model=_mock_model())
    resume = _make_resume_result()
    jd = _make_jd_result()

    result = matcher.compute_similarity(resume, jd)

    assert result.weighted_similarity > 0.0, (
        "compute_similarity() returned weighted_similarity=0.0; "
        "it must compute the weighted combination of section scores."
    )


def test_compute_similarity_empty_inputs_return_zero_weighted_similarity(
    mock_embedding_model: Mock,
) -> None:
    """compute_similarity() must return weighted_similarity=0.0 on empty resume or JD."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    matcher = SemanticMatcher(embedding_model=mock_embedding_model)

    empty_resume = Mock()
    empty_resume.extraction_result = None
    empty_resume.preprocessed = None
    empty_resume.skills = []
    jd = _make_jd_result()

    result = matcher.compute_similarity(empty_resume, jd)
    assert result.weighted_similarity == 0.0


def test_compute_similarity_empty_jd_returns_zero_weighted_similarity(
    mock_embedding_model: Mock,
) -> None:
    """compute_similarity() must return weighted_similarity=0.0 when JD has no text."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    matcher = SemanticMatcher(embedding_model=mock_embedding_model)

    resume = _make_resume_result()
    empty_jd = _make_jd_result(raw_text="")

    result = matcher.compute_similarity(resume, empty_jd)
    assert result.weighted_similarity == 0.0
    # encode must never be called when the guard triggers
    mock_embedding_model.encode.assert_not_called()
    # _model_name lazy property must resolve to a non-empty string even on early exit
    assert result.model_used is not None and result.model_used != "", (
        "model_used must be a non-empty string even when returning a zero-score match"
    )
