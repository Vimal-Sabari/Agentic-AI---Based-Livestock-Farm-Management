"""
Verdict logic – transparent rule-based scoring.

Each dimension is scored 0-1 and combined via configurable weights.
Score breakdown is logged and returned for explainability.

Rules are in config (no magic numbers here).  The function is pure and
independently testable.  No Streamlit or LangGraph imports.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from env_welfare_agent.src.env_agent.config import AgentConfig, THIThresholds, VerdictConfig
from env_welfare_agent.src.env_agent.schemas import (
    AlignmentResult,
    BaselineStats,
    ExposureMetrics,
    HerdDeviationResult,
    VerdictBreakdown,
)

logger = logging.getLogger(__name__)


def compute_verdict(
    exposure: ExposureMetrics,
    baseline: BaselineStats,
    herd: HerdDeviationResult,
    alignment: AlignmentResult,
    cfg: Optional[AgentConfig] = None,
) -> VerdictBreakdown:
    """
    Compute the verdict by scoring four evidence dimensions.

    Dimensions
    ----------
    1. env_signal      – Is there a meaningful environmental change vs baseline?
    2. temporal_align  – Does the environmental change temporally precede/coincide?
    3. herd_wide       – Do other cows show similar behavioral deviations?
    4. data_coverage   – Is the data complete enough to be reliable?

    Returns
    -------
    VerdictBreakdown with transparent per-dimension scores and explanation strings.
    """
    cfg = cfg or AgentConfig()
    vc = cfg.verdict
    th = cfg.thi_thresholds
    explanation: List[str] = []

    # ------------------------------------------------------------------
    # 1. Environmental signal score
    # ------------------------------------------------------------------
    env_score, env_exp = _score_env_signal(exposure, baseline, th, vc)
    explanation.extend(env_exp)

    # ------------------------------------------------------------------
    # 2. Temporal alignment score
    # ------------------------------------------------------------------
    align_score, align_exp = _score_temporal_alignment(alignment, vc)
    explanation.extend(align_exp)

    # ------------------------------------------------------------------
    # 3. Herd-wide score
    # ------------------------------------------------------------------
    herd_score, herd_exp = _score_herd_wide(herd, vc)
    explanation.extend(herd_exp)

    # ------------------------------------------------------------------
    # 4. Data coverage score
    # ------------------------------------------------------------------
    cov_score, cov_exp = _score_coverage(exposure, baseline, vc)
    explanation.extend(cov_exp)

    # ------------------------------------------------------------------
    # Weighted total
    # ------------------------------------------------------------------
    weighted = (
        vc.weight_env_signal * env_score
        + vc.weight_temporal_alignment * align_score
        + vc.weight_herd_wide * herd_score
        + vc.weight_data_coverage * cov_score
    )

    # ------------------------------------------------------------------
    # Verdict & evidence strength
    # ------------------------------------------------------------------
    # Override: if data coverage is extremely low → inconclusive regardless
    low_data = exposure.coverage_pct < cfg.data.min_coverage_pct_for_analysis

    if low_data:
        finding = "inconclusive"
        explanation.append(
            f"Data coverage {exposure.coverage_pct:.0f}% below minimum "
            f"{cfg.data.min_coverage_pct_for_analysis:.0f}% → inconclusive override."
        )
        evidence_strength = "none"
        confidence = round(max(0.1, exposure.coverage_pct / 100.0 * 0.4), 2)
    elif weighted >= vc.supports_min_score:
        finding = "supports"
        evidence_strength = (
            "strong" if weighted >= 0.75
            else "moderate" if weighted >= 0.55
            else "weak"
        )
        confidence = round(min(0.97, weighted * 1.05), 2)
    elif weighted >= vc.inconclusive_min_score:
        finding = "inconclusive"
        evidence_strength = "weak" if weighted >= 0.40 else "none"
        confidence = round(weighted * 0.8, 2)
    else:
        finding = "does_not_support"
        evidence_strength = "strong" if env_score <= 0.1 else "moderate"
        confidence = round(min(0.92, (1.0 - weighted) * 0.85), 2)

    breakdown = VerdictBreakdown(
        env_signal_score=round(env_score, 3),
        temporal_alignment_score=round(align_score, 3),
        herd_wide_score=round(herd_score, 3),
        data_coverage_score=round(cov_score, 3),
        weighted_total=round(weighted, 3),
        finding=finding,
        evidence_strength=evidence_strength,
        confidence=confidence,
        explanation=explanation,
    )

    logger.info(
        "Verdict: %s (strength=%s, confidence=%.2f, weighted=%.3f) "
        "[env=%.2f, align=%.2f, herd=%.2f, cov=%.2f]",
        finding, evidence_strength, confidence, weighted,
        env_score, align_score, herd_score, cov_score,
    )

    return breakdown


# ---------------------------------------------------------------------------
# Private scorer helpers
# ---------------------------------------------------------------------------

def _score_env_signal(
    exposure: ExposureMetrics,
    baseline: BaselineStats,
    th: THIThresholds,
    vc: VerdictConfig,
) -> tuple[float, List[str]]:
    """Score environmental signal strength (0-1)."""
    exp: List[str] = []

    if math.isnan(exposure.mean_thi) or math.isnan(baseline.mean_thi):
        exp.append("ENV: Missing THI data → signal score = 0.")
        return 0.0, exp

    delta_thi = exposure.mean_thi - baseline.mean_thi
    total_stress_hours = exposure.hours_mild + exposure.hours_moderate + exposure.hours_severe + exposure.hours_emergency
    is_above_threshold = exposure.mean_thi >= th.mild_min

    # Sub-scores
    # (a) Delta vs baseline (capped at 1.0 for delta ≥ 15 THI units)
    delta_score = min(1.0, max(0.0, delta_thi / 15.0))
    # (b) Absolute exposure level (max score when in severe+)
    level_score = 0.0
    if exposure.mean_thi >= th.emergency_min:
        level_score = 1.0
    elif exposure.mean_thi >= th.severe_min:
        level_score = 0.85
    elif exposure.mean_thi >= th.moderate_min:
        level_score = 0.65
    elif exposure.mean_thi >= th.mild_min:
        level_score = 0.40
    # (c) Hours in stress
    window_hours = max(1, total_stress_hours + 1)
    hours_score = min(1.0, total_stress_hours / 12.0)

    score = (delta_score * 0.4 + level_score * 0.4 + hours_score * 0.2)

    exp.append(
        f"ENV: mean_thi={exposure.mean_thi:.1f} (baseline={baseline.mean_thi:.1f}, "
        f"delta={delta_thi:+.1f}), stress_hours={total_stress_hours:.0f} → score={score:.2f}"
    )
    if exposure.poor_night_recovery:
        exp.append(f"ENV: Poor night-time recovery (night min THI={exposure.night_min_thi:.1f}).")

    return min(1.0, score), exp


def _score_temporal_alignment(
    alignment: AlignmentResult,
    vc: VerdictConfig,
) -> tuple[float, List[str]]:
    """Score temporal alignment quality (0-1)."""
    exp: List[str] = []

    if not alignment.sufficient_data or math.isnan(alignment.best_r):
        exp.append(
            f"ALIGN: Insufficient data ({alignment.n_points} points) → alignment score = 0.4 (neutral)."
        )
        return 0.4, exp  # neutral – don't penalise or reward

    r = alignment.best_r
    lag = alignment.best_lag_hours

    # Strong negative correlation at small lag → high score
    # r ranges −1 to +1; we want negative r (activity drops when THI rises)
    r_score = min(1.0, max(0.0, (-r + 1.0) / 2.0))  # maps [-1,1] → [1,0]
    # Slight bonus for lag in plausible range (0-6 h)
    lag_bonus = 0.1 if lag <= 6 else 0.0

    score = min(1.0, r_score + lag_bonus)
    exp.append(
        f"ALIGN: best_r={r:.3f} at lag={lag}h (n={alignment.n_points}) → score={score:.2f}."
    )
    if alignment.n_points < 12:
        exp.append("ALIGN: Small sample; correlation estimate may be unreliable.")

    return score, exp


def _score_herd_wide(
    herd: HerdDeviationResult,
    vc: VerdictConfig,
) -> tuple[float, List[str]]:
    """Score herd-wide behavioural evidence (0-1)."""
    exp: List[str] = []

    if herd.cows_analyzed == 0:
        exp.append("HERD: No herd data → herd score = 0.")
        return 0.0, exp

    frac = herd.fraction_deviating
    herd_score = min(1.0, frac / vc.weight_herd_wide)  # normalise by threshold region
    herd_score = min(1.0, frac * 1.5)  # simpler: 100% at frac ≥ 0.67

    # Penalty if target cow deviates MUCH more than herd peers (weakens env explanation)
    target_penalty = 0.15 if herd.target_exceeds_herd_median else 0.0
    score = max(0.0, herd_score - target_penalty)

    exp.append(
        f"HERD: {herd.cows_deviating}/{herd.cows_analyzed} cows deviating "
        f"({frac*100:.0f}%), is_herd_wide={herd.is_herd_wide}, "
        f"target_z={herd.target_cow_zscore}, target_exceeds_herd_median={herd.target_exceeds_herd_median} "
        f"→ score={score:.2f}."
    )

    return score, exp


def _score_coverage(
    exposure: ExposureMetrics,
    baseline: BaselineStats,
    vc: VerdictConfig,
) -> tuple[float, List[str]]:
    """Score data coverage quality (0-1)."""
    exp: List[str] = []
    # Average of anomaly window and baseline coverage
    avg_cov = (exposure.coverage_pct + baseline.coverage_pct) / 2.0
    score = min(1.0, avg_cov / 100.0)
    exp.append(
        f"COVERAGE: anomaly={exposure.coverage_pct:.0f}%, "
        f"baseline={baseline.coverage_pct:.0f}% → score={score:.2f}."
    )
    return score, exp
