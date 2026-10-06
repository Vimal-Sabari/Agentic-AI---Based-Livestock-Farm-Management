"""
Core agent logic orchestrating the Production modules.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import pandas as pd

from src.core.models import AgentRequest, AgentResponse, MetricItem, HerdContext, DataQuality, SuggestedFollowup
from production_agent.src.production_agent.config import AgentConfig
from production_agent.src.production_agent.schemas import ProductionTimeseries
from production_agent.src.production_agent.baseline import derive_baseline
from production_agent.src.production_agent.deviation import compute_deviation
from production_agent.src.production_agent.trend import compute_trend
from production_agent.src.production_agent.sessions import compute_session_stats
from production_agent.src.production_agent.composition import compute_composition
from production_agent.src.production_agent.herd import compute_herd_deviation
from production_agent.src.production_agent.verdict import compute_verdict
from production_agent.src.production_agent.summarizer import generate_summary

logger = logging.getLogger(__name__)

class ProductionAgent:
    def __init__(self, ts: ProductionTimeseries, cfg: Optional[AgentConfig] = None):
        self.ts = ts
        self.cfg = cfg or AgentConfig.load()
        self.agent_id = "production"
        
    def handle_query(self, request: AgentRequest) -> AgentResponse:
        try:
            if request.question_type not in ["production_check", "trend_check", "composition_check", "general"]:
                raise ValueError(f"Unsupported question_type: {request.question_type}")
                
            astart = datetime.fromisoformat(request.anomaly_window.start.replace("Z", "+00:00"))
            aend = datetime.fromisoformat(request.anomaly_window.end.replace("Z", "+00:00"))
            bstart = None
            bend = None
            if request.baseline_window:
                bstart = datetime.fromisoformat(request.baseline_window.start.replace("Z", "+00:00"))
                bend = datetime.fromisoformat(request.baseline_window.end.replace("Z", "+00:00"))
                
            baseline = derive_baseline(self.ts, request.cow_id, astart, bstart, bend, self.cfg.baseline)
            deviation = compute_deviation(self.ts, request.cow_id, astart, aend, baseline, self.cfg.deviation)
            trend = compute_trend(self.ts, request.cow_id, astart, aend, self.cfg.trend)
            session = compute_session_stats(self.ts, request.cow_id, astart, aend)
            composition = compute_composition(self.ts, request.cow_id, astart, aend)
            herd = compute_herd_deviation(self.ts, request.cow_id, astart, aend, self.cfg.deviation.pct_drop_threshold, self.cfg.herd)
            
            verdict = compute_verdict(baseline, deviation, trend, herd, self.cfg)
            summary = generate_summary(request.cow_id, baseline, deviation, herd, verdict, self.cfg.summarizer.use_llm)
            
            # Followups
            followups = []
            if verdict.finding == "supports":
                if herd.is_herd_wide:
                    followups.append(SuggestedFollowup(to_agent="environment_welfare", reason="Herd-wide production drop detected. Check for heat stress or sweeping environmental changes."))
                else:
                    followups.append(SuggestedFollowup(to_agent="health_behavior", reason="Cow-specific production drop. Check activity and core temperature for clinical signs."))
                    followups.append(SuggestedFollowup(to_agent="feeding_nutrition", reason="Check feed intake and rumination corresponding to this yield drop."))
                    
            limitations = []
            if not session.has_sessions:
                limitations.append("Per-milking session data not available.")
            if not composition.has_composition:
                limitations.append("Milk composition (fat, protein, SCC) not available in telemetry.")
            if baseline.n_days < self.cfg.baseline.default_days_before_anomaly:
                limitations.append(f"Baseline is short ({baseline.n_days} days).")
                
            metrics = {
                "mean_yield_daily": MetricItem(value=deviation.mean_yield_daily, unit="kg/day", baseline=baseline.mean_yield_daily, delta_pct=deviation.delta_pct),
                "trend_slope": MetricItem(value=trend.slope_kg_per_day, unit="kg/day²", baseline=0.0, delta_pct=0.0),
                "days_persisted": MetricItem(value=deviation.days_persisted, unit="days", baseline=0.0, delta_pct=0.0),
                "herd_fraction_deviating": MetricItem(value=herd.fraction_deviating, unit="fraction", baseline=0.0, delta_pct=0.0)
            }
            if session.has_sessions:
                metrics["morning_yield"] = MetricItem(value=session.morning_avg, unit="kg", baseline=session.morning_avg/(1+session.morning_delta_pct/100) if session.morning_delta_pct else session.morning_avg, delta_pct=session.morning_delta_pct)
                metrics["evening_yield"] = MetricItem(value=session.evening_avg, unit="kg", baseline=session.evening_avg/(1+session.evening_delta_pct/100) if session.evening_delta_pct else session.evening_avg, delta_pct=session.evening_delta_pct)
            if composition.has_composition:
                metrics["fat_pct_delta"] = MetricItem(value=composition.fat_delta_pct, unit="%", baseline=0.0, delta_pct=composition.fat_delta_pct)
                
            return AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding=verdict.finding,
                summary=summary,
                metrics=metrics,
                herd_context=HerdContext(cows_analyzed=herd.cows_analyzed, cows_deviating=herd.cows_deviating, fraction_deviating=herd.fraction_deviating, is_herd_wide=herd.is_herd_wide),
                evidence_strength=verdict.evidence_strength,
                confidence=verdict.confidence,
                data_quality=DataQuality(coverage_pct=baseline.coverage_pct, missing_streams=[], notes="\n".join(verdict.explanation)),
                limitations=limitations,
                suggested_followups=followups
            )
            
        except Exception as e:
            logger.exception("Error in ProductionAgent.handle_query")
            return AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding="inconclusive",
                summary=f"Analysis failed: {str(e)}",
                metrics={}, herd_context=HerdContext(), evidence_strength="none", confidence=0.0,
                data_quality=DataQuality(coverage_pct=0.0, missing_streams=["all"], notes=str(e)),
                limitations=[str(e)], suggested_followups=[]
            )
            
    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            req = state.get("pending_request")
            if not req: return state
            if isinstance(req, dict): req = AgentRequest(**req)
            if req.to_agent != self.agent_id: return state
            
            resp = self.handle_query(req)
            responses = state.get("specialist_responses", [])
            responses.append(resp.model_dump())
            return {**state, "specialist_responses": responses, "latest_response": resp.model_dump(), "pending_request": None}
        return node
