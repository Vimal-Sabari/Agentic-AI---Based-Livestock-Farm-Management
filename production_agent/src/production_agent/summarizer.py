"""
Summarizer module for the Production Agent.
"""
from __future__ import annotations
from typing import Any

from production_agent.src.production_agent.schemas import (
    BaselineStats, DeviationResult, HerdDeviationResult, VerdictBreakdown
)

def generate_summary(
    cow_id: str,
    baseline: BaselineStats,
    deviation: DeviationResult,
    herd: HerdDeviationResult,
    verdict: VerdictBreakdown,
    use_llm: bool = False
) -> str:
    """Generates the natural language summary based on metrics."""
    
    if verdict.finding == "inconclusive":
        return f"Analysis for {cow_id} is inconclusive due to insufficient baseline data or sparse telemetry."
        
    cow_fmt = cow_id.replace("cow_", "Cow #")
    
    if deviation.delta_pct <= -5.0:
        base_str = f"a decline that has persisted for {deviation.days_persisted} day(s)"
        trend_str = "exceeds its normal day-to-day variation" if verdict.finding == "supports" else "is within expected day-to-day variation"
        
        summary = (
            f"{cow_fmt}'s milk yield is {abs(deviation.delta_pct):.1f}% below its recent baseline "
            f"({deviation.mean_yield_daily:.1f} vs {baseline.mean_yield_daily:.1f} kg/day), {base_str} and {trend_str}. "
            f"{herd.cows_deviating} of {herd.cows_analyzed - 1} other cows show a comparable decline."
        )
    else:
        summary = (
            f"{cow_fmt}'s milk yield ({deviation.mean_yield_daily:.1f} kg/day) is stable compared to its "
            f"recent baseline ({baseline.mean_yield_daily:.1f} kg/day, delta: {deviation.delta_pct:+.1f}%). "
            f"No significant cow-specific drop detected."
        )
        
    if use_llm:
        # Placeholder for Langchain/LLM integration as requested in section A.3.
        # Fallback to template if disabled.
        pass
        
    return summary
