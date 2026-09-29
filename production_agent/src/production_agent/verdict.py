"""
Verdict scoring logic.
"""
from __future__ import annotations
import math
from typing import List, Optional

from production_agent.src.production_agent.config import VerdictConfig, AgentConfig
from production_agent.src.production_agent.schemas import (
    BaselineStats, DeviationResult, TrendResult, HerdDeviationResult, VerdictBreakdown
)

def compute_verdict(
    baseline: BaselineStats,
    deviation: DeviationResult,
    trend: TrendResult,
    herd: HerdDeviationResult,
    cfg: Optional[AgentConfig] = None
) -> VerdictBreakdown:
    cfg = cfg or AgentConfig()
    vc = cfg.verdict
    explanation: List[str] = []
    
    if baseline.n_days < cfg.baseline.min_days_required or math.isnan(deviation.delta_pct):
        explanation.append("Insufficient baseline days or missing data to compute a robust deviation.")
        return VerdictBreakdown("inconclusive", "none", 0.0, 0.0, explanation)
        
    # 1. Deviation Score
    dev_score = min(1.0, max(0.0, abs(deviation.delta_pct) / (cfg.deviation.pct_drop_threshold * 2)))
    if deviation.delta_pct > 0:
        dev_score = 0.0 # Not a drop
    explanation.append(f"Deviation: {deviation.delta_pct:+.1f}% vs baseline (score: {dev_score:.2f})")
    
    # 2. Persistence Score
    pers_score = min(1.0, deviation.days_persisted / max(1, cfg.deviation.min_persistence_days))
    explanation.append(f"Persistence: {deviation.days_persisted} days (score: {pers_score:.2f})")
    
    # 3. Trend Score
    trend_score = 1.0 if trend.direction == "falling" else 0.0
    explanation.append(f"Trend: {trend.direction} (score: {trend_score:.2f})")
    
    # 4. Herd isolation (If herd is dropping, it explains the drop as non-cow-specific)
    # The prompt says: "to separate cow-specific decline from herd-wide effects"
    herd_score = 1.0 if not herd.is_herd_wide else 0.0
    explanation.append(f"Herd isolated: {not herd.is_herd_wide} (score: {herd_score:.2f})")
    
    weighted = (
        vc.weight_deviation * dev_score +
        vc.weight_persistence * pers_score +
        vc.weight_trend * trend_score +
        vc.weight_herd_context * herd_score
    )
    
    if baseline.coverage_pct < cfg.data.min_coverage_pct_for_analysis:
        finding = "inconclusive"
        evidence = "none"
        confidence = 0.2
        explanation.append("Inconclusive due to poor data coverage.")
    elif weighted >= vc.supports_min_score and deviation.delta_pct <= -cfg.deviation.pct_drop_threshold:
        finding = "supports"
        evidence = "strong" if weighted >= 0.75 else "moderate"
        confidence = round(min(0.95, weighted * (baseline.coverage_pct/100)), 2)
    else:
        finding = "does_not_support"
        evidence = "strong" if weighted < 0.2 else "moderate"
        confidence = round(min(0.9, (1.0 - weighted) * (baseline.coverage_pct/100)), 2)
        
    return VerdictBreakdown(
        finding=finding,
        evidence_strength=evidence,
        confidence=confidence,
        weighted_total=round(weighted, 3),
        explanation=explanation
    )
