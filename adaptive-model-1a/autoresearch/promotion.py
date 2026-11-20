"""Promotion-gate helpers for experiments and selflearn cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PromotionDecision:
    """Result of comparing baseline and candidate metrics."""

    passed: bool
    status: str
    reasons: List[str] = field(default_factory=list)


def evaluate_promotion_gate(
    baseline_score: Optional[float],
    candidate_score: float,
    baseline_buckets: Optional[Dict[str, float]] = None,
    candidate_buckets: Optional[Dict[str, float]] = None,
    baseline_val_loss: Optional[float] = None,
    candidate_val_loss: Optional[float] = None,
    min_score_improvement: float = 2.0,
    max_bucket_regression: float = 1.0,
    max_val_loss_worsening: float = 0.05,
    guardrail_fired: bool = False,
) -> PromotionDecision:
    """Apply the default benchmark/val-loss promotion rules."""
    reasons: List[str] = []

    if guardrail_fired:
        reasons.append("guardrail fired")

    if baseline_score is not None:
        if candidate_score - baseline_score < min_score_improvement:
            reasons.append(
                f"benchmark improvement {candidate_score - baseline_score:.2f} < {min_score_improvement:.2f}"
            )

    if baseline_val_loss is not None and candidate_val_loss is not None:
        if candidate_val_loss - baseline_val_loss > max_val_loss_worsening:
            reasons.append(
                f"val loss worsened by {candidate_val_loss - baseline_val_loss:.4f} > {max_val_loss_worsening:.4f}"
            )

    if baseline_buckets and candidate_buckets:
        for bucket, base_score in baseline_buckets.items():
            cand_score = candidate_buckets.get(bucket, 0.0)
            if base_score - cand_score > max_bucket_regression:
                reasons.append(
                    f"bucket {bucket} regressed by {base_score - cand_score:.2f} > {max_bucket_regression:.2f}"
                )

    if reasons:
        return PromotionDecision(False, "discard", reasons)
    return PromotionDecision(True, "keep", [])
