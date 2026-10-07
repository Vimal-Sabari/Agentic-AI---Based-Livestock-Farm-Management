"""
Verdict synthesis and transparent weighted scoring.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from feeding_agent.src.feeding_agent.config import VerdictConfig
from feeding_agent.src.feeding_agent.schemas import (
    AvailabilityResult,
    BaselineStats,
    DeviationResult,
    HerdResult,
    RuminationResult,
    VerdictBreakdown,
)

logger = logging.getLogger(__name__)


def compute_verdict(
    baseline: BaselineStats,
    deviation: DeviationResult,
    rumination_res: RuminationResult,
    avail_res: AvailabilityResult,
    herd_res: HerdResult,
    coverage_pct: float,
    cfg: Optional[VerdictConfig] = None,
    min_coverage_pct: float = 40.0,
    min_days_required: int = 2,
) -> VerdictBreakdown:
    """
    Computes transparent weighted score, verdict finding, evidence strength, confidence, and explanation.
    """
    cfg = cfg or VerdictConfig()
    explanation: List[str] = []

    # Check hard inconclusive conditions
    hard_inconclusive = False
    inconclusive_reason = ""

    if coverage_pct < min_coverage_pct:
        hard_inconclusive = True
        inconclusive_reason = f"Anomaly window temporal coverage ({coverage_pct:.1f}%) is below minimum threshold ({min_coverage_pct:.1f}%)."
    elif baseline.n_days < min_days_required:
        hard_inconclusive = True
        inconclusive_reason = f"Baseline length ({baseline.n_days} days) is below minimum required ({min_days_required} days)."

    # 1. Feeding signal score
    if deviation.feeding_significant:
        feeding_score = 1.0 if abs(deviation.feeding_delta_pct) >= 25.0 else 0.7
        explanation.append(f"Significant feeding {deviation.feeding_direction} detected ({deviation.feeding_delta_pct:+.1f}%, z={deviation.feeding_zscore:+.2f}).")
    else:
        feeding_score = 0.0
        explanation.append(f"Feeding duration remained within normal range ({deviation.feeding_delta_pct:+.1f}%, z={deviation.feeding_zscore:+.2f}).")

    # 2. Rumination signal score
    if rumination_res.available and deviation.rumination_daily_val is not None:
        if deviation.rumination_significant and deviation.rumination_direction == deviation.feeding_direction:
            rumination_score = 1.0
            explanation.append(f"Rumination aligned with feeding ({deviation.rumination_delta_pct:+.1f}%).")
        elif deviation.rumination_significant:
            rumination_score = 0.3
            explanation.append(f"Rumination shifted ({deviation.rumination_delta_pct:+.1f}%) in opposite direction of feeding.")
        else:
            rumination_score = 0.5
            explanation.append("Rumination remained stable.")
    else:
        rumination_score = 0.0
        explanation.append("Rumination telemetry was unavailable.")

    # 3. Intake signal score
    if deviation.intake_daily_val is not None:
        if deviation.intake_significant and deviation.intake_direction == deviation.feeding_direction:
            intake_score = 1.0
            explanation.append(f"Intake aligned with feeding ({deviation.intake_delta_pct:+.1f}%).")
        elif deviation.intake_significant:
            intake_score = 0.3
            explanation.append("Intake shifted in opposite direction.")
        else:
            intake_score = 0.5
            explanation.append("Intake remained stable.")
    else:
        intake_score = 0.0
        explanation.append("Estimated intake telemetry was unavailable.")

    # 4. Consistency score
    available_streams = []
    if deviation.feeding_significant or deviation.feeding_direction != "stable":
        available_streams.append(deviation.feeding_direction)
    if rumination_res.available and deviation.rumination_direction and deviation.rumination_direction != "stable":
        available_streams.append(deviation.rumination_direction)
    if deviation.intake_direction and deviation.intake_direction != "stable":
        available_streams.append(deviation.intake_direction)

    if not available_streams:
        consistency_score = 1.0
    else:
        target_dir = deviation.feeding_direction
        agree_count = sum(1 for s in available_streams if s == target_dir)
        consistency_score = float(agree_count / len(available_streams))

    # 5. Coverage score
    coverage_score = min(1.0, max(0.0, coverage_pct / 100.0))

    # Weighted combination
    w = cfg.weights
    weighted_score = (
        w.feeding * feeding_score
        + w.rumination * rumination_score
        + w.intake * intake_score
        + w.consistency * consistency_score
        + w.coverage * coverage_score
    )

    # Determine finding
    if hard_inconclusive:
        finding = "inconclusive"
        explanation.insert(0, f"INCONCLUSIVE: {inconclusive_reason}")
    elif deviation.feeding_significant and weighted_score >= cfg.supports_min_score:
        finding = "supports"
    elif not deviation.feeding_significant and weighted_score < cfg.inconclusive_min_score:
        finding = "does_not_support"
    elif not deviation.feeding_significant:
        finding = "does_not_support"
    else:
        finding = "inconclusive"
        explanation.append("Contradictory or weak signal across metrics.")

    # Determine evidence strength
    agreeing_streams_count = sum(
        1 for d in [deviation.feeding_direction, deviation.rumination_direction, deviation.intake_direction]
        if d is not None and d != "stable" and d == deviation.feeding_direction
    )

    if finding == "inconclusive" and hard_inconclusive:
        evidence_strength = "none"
    elif weighted_score >= 0.75 and agreeing_streams_count >= 2:
        evidence_strength = "strong"
    elif weighted_score >= cfg.supports_min_score:
        evidence_strength = "moderate"
    elif weighted_score >= cfg.inconclusive_min_score:
        evidence_strength = "weak"
    else:
        evidence_strength = "none"

    # Compute raw confidence
    conf = weighted_score

    # Apply penalties
    if baseline.is_short:
        conf -= cfg.short_baseline_penalty
        explanation.append(f"Short baseline penalty (-{cfg.short_baseline_penalty:.2f}).")
    if avail_res.status == "unknown":
        conf -= cfg.unknown_availability_penalty
        explanation.append(f"Unknown feed availability penalty (-{cfg.unknown_availability_penalty:.2f}).")
    if not rumination_res.available:
        conf -= cfg.missing_rumination_penalty
        explanation.append(f"Missing rumination penalty (-{cfg.missing_rumination_penalty:.2f}).")
    if rumination_res.proxy_used:
        conf -= cfg.proxy_penalty
        explanation.append(f"Proxy rumination penalty (-{cfg.proxy_penalty:.2f}).")

    confidence = max(0.0, min(1.0, float(conf)))

    if hard_inconclusive:
        confidence = min(0.25, confidence)

    return VerdictBreakdown(
        feeding_score=feeding_score,
        rumination_score=rumination_score,
        intake_score=intake_score,
        consistency_score=consistency_score,
        coverage_score=coverage_score,
        weighted_score=weighted_score,
        finding=finding,
        evidence_strength=evidence_strength,
        confidence=confidence,
        explanation=explanation,
    )
