import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
from src.core.matching.matching_engine import MatchResult
from src.core.ranking.ranking_config import RankingConfig, RankedResult
from src.core.ranking.fairness_reranker import FairnessReranker
from src.data.models import BiasCheckResult


def _mr(
    overall_score: float = 0.5,
    protected_attrs: list[str] | None = None,
    name: str = "Candidate",
) -> MatchResult:
    return MatchResult(
        candidate_name=name,
        overall_score=overall_score,
        skills_score=overall_score,
        experience_score=overall_score,
        education_score=overall_score,
        semantic_score=overall_score,
        keyword_score=overall_score,
        bias_check=BiasCheckResult(
            protected_attributes_found=protected_attrs or []
        ),
    )


def _ranked(mr: MatchResult, rank: int, score: float) -> RankedResult:
    return RankedResult(match_result=mr, rank=rank, effective_score=score)


def test_off_mode_returns_unchanged() -> None:
    rr1 = _ranked(_mr(0.9), 1, 0.9)
    rr2 = _ranked(_mr(0.7), 2, 0.7)
    result = FairnessReranker(RankingConfig(fairness_mode="off")).apply([rr1, rr2])
    assert result[0] is rr1
    assert result[1] is rr2
    assert result[0].fairness_flags == []


def test_flag_mode_does_not_change_order() -> None:
    # Build 2 groups of 5 with extreme score separation to guarantee a violation
    group_a = [_ranked(_mr(0.9, ["group_a"], f"A{i}"), i + 1, 0.9) for i in range(5)]
    group_b = [_ranked(_mr(0.2, ["group_b"], f"B{i}"), i + 6, 0.2) for i in range(5)]
    ranked_input = group_a + group_b

    result = FairnessReranker(
        RankingConfig(fairness_mode="flag", fairness_min_group_size=5)
    ).apply(ranked_input)

    # Order unchanged
    for original, returned in zip(ranked_input, result):
        assert original is returned


def test_flag_mode_attaches_violations_when_bias_present() -> None:
    # Group A (scores ~0.9) all selected; Group B (scores ~0.2) none selected
    # → demographic parity difference >> 0.1 threshold → violation
    group_a = [_ranked(_mr(0.9, ["group_a"], f"A{i}"), i + 1, 0.9) for i in range(5)]
    group_b = [_ranked(_mr(0.2, ["group_b"], f"B{i}"), i + 6, 0.2) for i in range(5)]

    result = FairnessReranker(
        RankingConfig(fairness_mode="flag", fairness_min_group_size=5)
    ).apply(group_a + group_b)

    # At least one result should carry a violation flag
    all_flags = [flag for rr in result for flag in rr.fairness_flags]
    assert any("parity" in f.lower() or "disparate" in f.lower() for f in all_flags)


def test_flag_mode_small_groups_produce_warnings_not_violations() -> None:
    # Both groups have 2 members < min_group_size=5 → skipped → warnings only
    group_a = [_ranked(_mr(0.9, ["group_a"], f"A{i}"), i + 1, 0.9) for i in range(2)]
    group_b = [_ranked(_mr(0.2, ["group_b"], f"B{i}"), i + 3, 0.2) for i in range(2)]

    result = FairnessReranker(
        RankingConfig(fairness_mode="flag", fairness_min_group_size=5)
    ).apply(group_a + group_b)

    all_flags = [flag for rr in result for flag in rr.fairness_flags]
    # Warnings present (group too small), no violations triggered
    assert any("minimum" in f.lower() or "excluded" in f.lower() for f in all_flags)
    assert not any("exceeds threshold" in f or "below threshold" in f for f in all_flags)


def test_rerank_does_not_promote_outside_tolerance_band() -> None:
    # top_score=0.90, tolerance=0.05 → band threshold=0.85
    # in_band: scores 0.90, 0.88, 0.86 (all >= 0.85)
    # out_of_band: score 0.60 — must stay after in_band regardless of group
    in_band = [
        _ranked(_mr(0.90, ["group_a"], "A1"), 1, 0.90),
        _ranked(_mr(0.88, ["group_a"], "A2"), 2, 0.88),
        _ranked(_mr(0.86, ["group_a"], "A3"), 3, 0.86),
    ]
    out_of_band = [
        _ranked(_mr(0.60, ["group_b"], "B1"), 4, 0.60),
    ]
    all_ranked = in_band + out_of_band

    result = FairnessReranker(
        RankingConfig(fairness_mode="rerank", rerank_tolerance=0.05)
    ).apply(all_ranked)

    # B1 must never appear in the first 3 positions
    top3_names = [r.match_result.candidate_name for r in result[:3]]
    assert "B1" not in top3_names


def test_rerank_sets_reranked_true_when_position_changes() -> None:
    # 2 groups of 3 in band — B1 is under-represented and should be promoted
    in_band = [
        _ranked(_mr(0.90, ["group_a"], "A1"), 1, 0.90),
        _ranked(_mr(0.89, ["group_a"], "A2"), 2, 0.89),
        _ranked(_mr(0.88, ["group_a"], "A3"), 3, 0.88),
        _ranked(_mr(0.87, ["group_b"], "B1"), 4, 0.87),
        _ranked(_mr(0.86, ["group_b"], "B2"), 5, 0.86),
        _ranked(_mr(0.85, ["group_b"], "B3"), 6, 0.85),
    ]

    result = FairnessReranker(
        RankingConfig(fairness_mode="rerank", rerank_tolerance=0.10)
    ).apply(in_band)

    # At least one candidate should have been reranked
    assert any(r.reranked for r in result)
    # A group_b candidate should appear in top 3 and be marked reranked
    top3: list[RankedResult] = result[:3]
    group_b_in_top3: list[RankedResult] = [
        r for r in top3 if r.match_result.candidate_name.startswith("B")
    ]
    assert any(r.reranked for r in group_b_in_top3)


def test_rerank_off_mode_reranked_is_always_false() -> None:
    rr1 = _ranked(_mr(0.9, ["group_a"]), 1, 0.9)
    rr2 = _ranked(_mr(0.7, ["group_b"]), 2, 0.7)
    result = FairnessReranker(RankingConfig(fairness_mode="off")).apply([rr1, rr2])
    assert all(not r.reranked for r in result)


def test_get_group_composite_key_is_sorted() -> None:
    # Attributes in different order must produce the same group key
    mr_ab = _mr(0.8, ["attr_b", "attr_a"])
    mr_ba = _mr(0.8, ["attr_a", "attr_b"])
    rr_ab = _ranked(mr_ab, 1, 0.8)
    rr_ba = _ranked(mr_ba, 2, 0.8)
    assert FairnessReranker._get_group(rr_ab) == FairnessReranker._get_group(rr_ba)
    assert FairnessReranker._get_group(rr_ab) == "attr_a|attr_b"


def test_get_group_single_attribute_unchanged() -> None:
    rr = _ranked(_mr(0.9, ["group_a"]), 1, 0.9)
    assert FairnessReranker._get_group(rr) == "group_a"


def test_get_group_returns_unknown_when_bias_check_is_none() -> None:
    # match_result.bias_check = None must not raise; returns "unknown"
    mr = MatchResult(candidate_name="X", overall_score=0.5, bias_check=None)
    rr = RankedResult(match_result=mr, rank=1, effective_score=0.5)
    assert FairnessReranker._get_group(rr) == "unknown"


def test_rerank_large_in_band_selects_all_candidates() -> None:
    # 10 group_a + 10 group_b, all in band — every candidate must appear exactly once
    in_band = (
        [_ranked(_mr(0.90, ["group_a"], f"A{i}"), i + 1, 0.90 - i * 0.001) for i in range(10)]
        + [_ranked(_mr(0.89, ["group_b"], f"B{i}"), i + 11, 0.89 - i * 0.001) for i in range(10)]
    )
    result = FairnessReranker(
        RankingConfig(fairness_mode="rerank", rerank_tolerance=0.10)
    ).apply(in_band)

    assert len(result) == 20
    names_in = {r.match_result.candidate_name for r in in_band}
    names_out = {r.match_result.candidate_name for r in result}
    assert names_in == names_out  # same candidates, possibly reordered
    ranks = [r.rank for r in result]
    assert ranks == list(range(1, 21))  # ranks reassigned 1..20
