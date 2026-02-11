"""
Fairness metrics calculation for bias detection.

Implements various fairness metrics to measure and monitor
bias in the candidate matching and ranking process.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.constants import FAIRNESS_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GroupMetrics:
    """Metrics for a specific demographic group."""

    group_name: str
    group_size: int
    positive_count: int  # Number selected/shortlisted
    positive_rate: float  # Selection rate
    average_score: float
    score_std: float


@dataclass
class FairnessMetrics:
    """Complete fairness metrics result."""

    # Demographic Parity
    demographic_parity_difference: float = 0.0
    demographic_parity_ratio: float = 1.0

    # Equalized Odds
    equalized_odds_difference: float = 0.0
    true_positive_rate_difference: float = 0.0
    false_positive_rate_difference: float = 0.0

    # Disparate Impact
    disparate_impact_ratio: float = 1.0

    # Score Distribution
    score_gap: float = 0.0  # Gap between highest and lowest group avg scores
    score_variance_ratio: float = 1.0

    # Group-level metrics
    group_metrics: list[GroupMetrics] = field(default_factory=list)

    # Fairness assessment
    is_fair: bool = True
    violations: list[str] = field(default_factory=list)
    thresholds_used: dict[str, float] = field(default_factory=dict)


class FairnessCalculator:
    """
    Calculates fairness metrics for candidate evaluation.

    Implements standard fairness metrics including:
    - Demographic Parity
    - Equalized Odds
    - Disparate Impact
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the fairness calculator.

        Args:
            thresholds: Custom fairness thresholds. Defaults to config values.
        """
        self.thresholds = thresholds or FAIRNESS_THRESHOLDS

    def calculate(
        self,
        scores: list[float],
        group_labels: list[str],
        outcomes: Optional[list[bool]] = None,
        selection_threshold: float = 0.7,
    ) -> FairnessMetrics:
        """
        Calculate fairness metrics for a set of candidates.

        Args:
            scores: List of match scores for each candidate.
            group_labels: List of group labels (e.g., demographic group).
            outcomes: Optional list of actual outcomes (selected/not selected).
                     If not provided, uses selection_threshold on scores.
            selection_threshold: Score threshold for considering a candidate
                                as "selected" (if outcomes not provided).

        Returns:
            FairnessMetrics with all calculated metrics.
        """
        if len(scores) != len(group_labels):
            raise ValueError("Scores and group labels must have same length")

        scores = np.array(scores)
        group_labels = np.array(group_labels)

        # Determine outcomes
        if outcomes is None:
            outcomes = scores >= selection_threshold
        else:
            outcomes = np.array(outcomes)

        # Get unique groups
        unique_groups = np.unique(group_labels)

        if len(unique_groups) < 2:
            logger.warning("Less than 2 groups found, cannot calculate fairness metrics")
            return FairnessMetrics(is_fair=True)

        # Calculate group-level metrics
        group_metrics = []
        group_positive_rates = {}
        group_avg_scores = {}

        for group in unique_groups:
            mask = group_labels == group
            group_scores = scores[mask]
            group_outcomes = outcomes[mask]

            positive_count = int(np.sum(group_outcomes))
            positive_rate = positive_count / len(group_outcomes) if len(group_outcomes) > 0 else 0

            metrics = GroupMetrics(
                group_name=str(group),
                group_size=int(np.sum(mask)),
                positive_count=positive_count,
                positive_rate=positive_rate,
                average_score=float(np.mean(group_scores)),
                score_std=float(np.std(group_scores)) if len(group_scores) > 1 else 0,
            )
            group_metrics.append(metrics)
            group_positive_rates[group] = positive_rate
            group_avg_scores[group] = metrics.average_score

        # Calculate Demographic Parity
        dp_diff, dp_ratio = self._calculate_demographic_parity(group_positive_rates)

        # Calculate Disparate Impact
        di_ratio = self._calculate_disparate_impact(group_positive_rates)

        # Calculate Equalized Odds (simplified - using positive rates)
        eo_diff, tpr_diff, fpr_diff = self._calculate_equalized_odds(
            scores, group_labels, outcomes, unique_groups
        )

        # Calculate score distribution metrics
        score_gap = max(group_avg_scores.values()) - min(group_avg_scores.values())
        variances = [m.score_std ** 2 for m in group_metrics if m.score_std > 0]
        score_var_ratio = max(variances) / min(variances) if len(variances) >= 2 and min(variances) > 0 else 1.0

        # Check for violations
        violations = []
        if abs(dp_diff) > self.thresholds["demographic_parity_difference"]:
            violations.append(
                f"Demographic parity difference ({dp_diff:.3f}) exceeds threshold "
                f"({self.thresholds['demographic_parity_difference']})"
            )

        if abs(eo_diff) > self.thresholds["equalized_odds_difference"]:
            violations.append(
                f"Equalized odds difference ({eo_diff:.3f}) exceeds threshold "
                f"({self.thresholds['equalized_odds_difference']})"
            )

        if di_ratio < self.thresholds["disparate_impact_ratio"]:
            violations.append(
                f"Disparate impact ratio ({di_ratio:.3f}) below threshold "
                f"({self.thresholds['disparate_impact_ratio']})"
            )

        return FairnessMetrics(
            demographic_parity_difference=dp_diff,
            demographic_parity_ratio=dp_ratio,
            equalized_odds_difference=eo_diff,
            true_positive_rate_difference=tpr_diff,
            false_positive_rate_difference=fpr_diff,
            disparate_impact_ratio=di_ratio,
            score_gap=score_gap,
            score_variance_ratio=score_var_ratio,
            group_metrics=group_metrics,
            is_fair=len(violations) == 0,
            violations=violations,
            thresholds_used=self.thresholds.copy(),
        )

    def _calculate_demographic_parity(
        self,
        group_rates: dict[str, float],
    ) -> tuple[float, float]:
        """
        Calculate demographic parity metrics.

        Demographic parity requires that the selection rate is equal
        across all groups.

        Returns:
            (difference, ratio) where:
            - difference: max_rate - min_rate
            - ratio: min_rate / max_rate
        """
        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        difference = max_rate - min_rate
        ratio = min_rate / max_rate if max_rate > 0 else 1.0

        return round(difference, 4), round(ratio, 4)

    def _calculate_disparate_impact(
        self,
        group_rates: dict[str, float],
    ) -> float:
        """
        Calculate disparate impact ratio (Four-Fifths Rule).

        The four-fifths rule states that selection rate for any group
        should be at least 80% of the highest group's rate.

        Returns:
            Ratio of lowest to highest selection rate.
        """
        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        if max_rate == 0:
            return 1.0

        return round(min_rate / max_rate, 4)

    def _calculate_equalized_odds(
        self,
        scores: np.ndarray,
        group_labels: np.ndarray,
        outcomes: np.ndarray,
        groups: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calculate equalized odds metrics.

        Equalized odds requires that both TPR and FPR are equal
        across groups. This is a simplified version that uses
        score-based predictions.

        Returns:
            (equalized_odds_diff, tpr_diff, fpr_diff)
        """
        # For a proper equalized odds calculation, we need ground truth labels
        # Here we use a simplified approach based on score distributions

        group_tprs = {}
        group_fprs = {}

        for group in groups:
            mask = group_labels == group
            group_scores = scores[mask]
            group_outcomes = outcomes[mask]

            # Positive cases: high scores
            high_score_mask = group_scores >= 0.7
            # Negative cases: low scores
            low_score_mask = group_scores < 0.5

            # TPR approximation: rate of high-scorers who are selected
            if np.sum(high_score_mask) > 0:
                tpr = np.sum(group_outcomes[high_score_mask]) / np.sum(high_score_mask)
            else:
                tpr = 0

            # FPR approximation: rate of low-scorers who are selected
            if np.sum(low_score_mask) > 0:
                fpr = np.sum(group_outcomes[low_score_mask]) / np.sum(low_score_mask)
            else:
                fpr = 0

            group_tprs[group] = tpr
            group_fprs[group] = fpr

        # Calculate differences
        tpr_values = list(group_tprs.values())
        fpr_values = list(group_fprs.values())

        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0

        # Equalized odds difference is max of TPR and FPR differences
        eo_diff = max(tpr_diff, fpr_diff)

        return round(eo_diff, 4), round(tpr_diff, 4), round(fpr_diff, 4)

    def calculate_individual_fairness(
        self,
        candidate_scores: list[tuple[str, float, dict]],
        similarity_threshold: float = 0.1,
    ) -> dict[str, float]:
        """
        Calculate individual fairness metrics.

        Individual fairness requires that similar individuals
        receive similar scores.

        Args:
            candidate_scores: List of (candidate_id, score, features) tuples
            similarity_threshold: Max allowed score difference for similar candidates

        Returns:
            Dictionary with individual fairness metrics.
        """
        if len(candidate_scores) < 2:
            return {"individual_fairness_score": 1.0, "violations": 0}

        violations = 0
        total_pairs = 0

        for i in range(len(candidate_scores)):
            for j in range(i + 1, len(candidate_scores)):
                _, score_i, features_i = candidate_scores[i]
                _, score_j, features_j = candidate_scores[j]

                # Calculate feature similarity (simplified)
                feature_sim = self._calculate_feature_similarity(features_i, features_j)

                if feature_sim > 0.8:  # Similar candidates
                    total_pairs += 1
                    score_diff = abs(score_i - score_j)
                    if score_diff > similarity_threshold:
                        violations += 1

        if total_pairs == 0:
            return {"individual_fairness_score": 1.0, "violations": 0}

        fairness_score = 1.0 - (violations / total_pairs)

        return {
            "individual_fairness_score": round(fairness_score, 4),
            "violations": violations,
            "total_similar_pairs": total_pairs,
        }

    def _calculate_feature_similarity(
        self,
        features1: dict,
        features2: dict,
    ) -> float:
        """Calculate similarity between two feature dictionaries."""
        if not features1 or not features2:
            return 0.0

        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        matches = 0
        for key in common_keys:
            if features1[key] == features2[key]:
                matches += 1
            elif isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                # For numeric values, check if within 10%
                max_val = max(abs(features1[key]), abs(features2[key]))
                if max_val > 0 and abs(features1[key] - features2[key]) / max_val < 0.1:
                    matches += 1

        return matches / len(common_keys)


# Singleton instance
_calculator: Optional[FairnessCalculator] = None


def get_fairness_calculator() -> FairnessCalculator:
    """Get the fairness calculator singleton instance."""
    global _calculator
    if _calculator is None:
        _calculator = FairnessCalculator()
    return _calculator
