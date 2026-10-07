"""
Feeding & Nutrition Specialist Agent.
Implements the shared agent contract (section A.3) and LangGraph node wrapper.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.core.models import (
    AgentRequest,
    AgentResponse,
    DataQuality,
    HerdContext,
    MetricItem,
    SuggestedFollowup,
    TimeWindow,
)
from feeding_agent.src.feeding_agent.availability import compute_availability
from feeding_agent.src.feeding_agent.baseline import derive_baseline
from feeding_agent.src.feeding_agent.config import AgentConfig
from feeding_agent.src.feeding_agent.deviation import compute_deviation
from feeding_agent.src.feeding_agent.herd import compute_herd_comparison
from feeding_agent.src.feeding_agent.meals import compute_meal_pattern
from feeding_agent.src.feeding_agent.rumination import compute_rumination
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries, VerdictBreakdown
from feeding_agent.src.feeding_agent.summarizer import build_summary
from feeding_agent.src.feeding_agent.verdict import compute_verdict

logger = logging.getLogger(__name__)

SUPPORTED_QUESTION_TYPES = {"feeding_drop_check", "feeding_check", "rumination_check", "intake_check", "general"}
AGENT_ID = "feeding_nutrition"


class FeedingNutritionAgent:
    """
    Feeding & Nutrition Specialist Agent.
    """

    def __init__(self, ts: FeedingTimeseries, cfg: Optional[AgentConfig] = None):
        self.ts = ts
        self.cfg = cfg or AgentConfig.load()
        self.agent_id = AGENT_ID
        self.message_log: List[Tuple[AgentRequest, AgentResponse, str]] = []
        self.latest_verdict: Optional[VerdictBreakdown] = None

    def handle_query(self, request: AgentRequest) -> AgentResponse:
        """
        Main query entry point. Validates request, executes analysis, and returns AgentResponse.
        Never raises exceptions; top-level catch returns structured inconclusive response.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            # 1. Validate question type
            if request.question_type not in SUPPORTED_QUESTION_TYPES:
                resp = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"Unsupported question_type '{request.question_type}'. Supported types: {sorted(list(SUPPORTED_QUESTION_TYPES))}.",
                    metrics={},
                    herd_context=HerdContext(),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes="Unsupported question type."),
                    limitations=[f"Unsupported question_type: {request.question_type}"],
                    suggested_followups=[],
                )
                self.message_log.append((request, resp, now_iso))
                return resp

            # 2. Parse timestamps
            try:
                astart = datetime.fromisoformat(request.anomaly_window.start.replace("Z", "+00:00"))
                aend = datetime.fromisoformat(request.anomaly_window.end.replace("Z", "+00:00"))
            except Exception as parse_err:
                resp = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"Invalid anomaly window timestamp format: {parse_err}",
                    metrics={},
                    herd_context=HerdContext(),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes="Invalid timestamp format."),
                    limitations=["Invalid timestamp format in anomaly window."],
                    suggested_followups=[],
                )
                self.message_log.append((request, resp, now_iso))
                return resp

            if aend <= astart:
                resp = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"Inverted or zero-length anomaly window (start={astart}, end={aend}).",
                    metrics={},
                    herd_context=HerdContext(),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes="Inverted window."),
                    limitations=["Inverted or zero-length anomaly window."],
                    suggested_followups=[],
                )
                self.message_log.append((request, resp, now_iso))
                return resp

            bstart, bend = None, None
            if request.baseline_window:
                try:
                    bstart = datetime.fromisoformat(request.baseline_window.start.replace("Z", "+00:00"))
                    bend = datetime.fromisoformat(request.baseline_window.end.replace("Z", "+00:00"))
                except Exception:
                    pass

            # 3. Check cow existence
            known_cows = self.ts.df["cow_id"].unique() if "cow_id" in self.ts.df.columns else []
            if request.cow_id not in known_cows:
                resp = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"Cow '{request.cow_id}' not found in telemetry dataset.",
                    metrics={},
                    herd_context=HerdContext(),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=["all"], notes="Target cow not present."),
                    limitations=[f"Cow '{request.cow_id}' not found in dataset."],
                    suggested_followups=[],
                )
                self.message_log.append((request, resp, now_iso))
                return resp

            # 4. Clip windows to data bounds if necessary
            min_dt = self.ts.df["timestamp"].min()
            max_dt = self.ts.df["timestamp"].max()

            limitations: List[str] = []

            if astart < min_dt or aend > max_dt:
                limitations.append(f"Anomaly window [{astart}, {aend}] extends beyond telemetry bounds [{min_dt}, {max_dt}].")
                astart = max(astart, min_dt)
                aend = min(aend, max_dt)

            if aend <= astart:
                resp = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary="Anomaly window falls entirely outside available dataset range.",
                    metrics={},
                    herd_context=HerdContext(),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes="Window outside dataset bounds."),
                    limitations=limitations,
                    suggested_followups=[],
                )
                self.message_log.append((request, resp, now_iso))
                return resp

            # 5. Baseline analysis
            baseline = derive_baseline(
                self.ts,
                request.cow_id,
                astart,
                bstart,
                bend,
                cfg=self.cfg.baseline,
                data_cfg=self.cfg.data,
                meals_cfg=self.cfg.meals,
            )

            # 6. Deviation analysis
            dev = compute_deviation(
                self.ts,
                request.cow_id,
                astart,
                aend,
                baseline,
                cfg=self.cfg.deviation,
                data_cfg=self.cfg.data,
            )

            # 7. Meal pattern analysis
            meal_stats = compute_meal_pattern(
                self.ts,
                request.cow_id,
                astart,
                aend,
                baseline_profile=baseline.hourly_feeding_profile,
                cfg=self.cfg.meals,
            )

            # 8. Rumination analysis
            rum_res = compute_rumination(
                self.ts,
                request.cow_id,
                astart,
                aend,
                baseline,
                dev,
                cfg=self.cfg.rumination,
            )

            # 9. Feed Availability analysis
            avail_res = compute_availability(
                self.ts,
                request.cow_id,
                astart,
                aend,
                cfg=self.cfg.availability,
            )

            # 10. Herd comparison
            herd_res = compute_herd_comparison(
                self.ts,
                request.cow_id,
                astart,
                aend,
                target_direction=dev.feeding_direction,
                cfg=self.cfg.herd,
                baseline_cfg=self.cfg.baseline,
                dev_cfg=self.cfg.deviation,
                data_cfg=self.cfg.data,
            )

            # 11. Anomaly window coverage pct
            cow_anom_df = self.ts.df[(self.ts.df["cow_id"] == request.cow_id) & (self.ts.df["timestamp"] >= astart) & (self.ts.df["timestamp"] < aend)]
            total_window_hours = max(1.0, (aend - astart).total_seconds() / 3600.0)
            valid_hours = len(cow_anom_df["feeding_minutes"].dropna()) if not cow_anom_df.empty else 0
            anom_coverage_pct = min(100.0, (valid_hours / total_window_hours) * 100.0)

            # 12. Verdict synthesis
            verdict = compute_verdict(
                baseline=baseline,
                deviation=dev,
                rumination_res=rum_res,
                avail_res=avail_res,
                herd_res=herd_res,
                coverage_pct=anom_coverage_pct,
                cfg=self.cfg.verdict,
                min_coverage_pct=self.cfg.data.min_coverage_pct,
                min_days_required=self.cfg.baseline.min_days_required,
            )
            self.latest_verdict = verdict

            # Collect explicit limitations
            if baseline.is_short:
                limitations.append(f"Baseline period is short ({baseline.n_days} days; reliable threshold is {self.cfg.baseline.min_days_reliable} days).")
            if not rum_res.available:
                limitations.append("Rumination sensor telemetry is absent.")
            elif rum_res.proxy_used:
                limitations.append("Rumination was derived via quiet-lying proxy.")
            if avail_res.status == "unknown":
                limitations.append("Feed bunk availability telemetry is absent.")
            elif avail_res.restricted:
                limitations.append("Feed bunk availability was restricted (<50%), suggesting a feed delivery or management cause.")
            if herd_res.notes and "Insufficient" in herd_res.notes:
                limitations.append(herd_res.notes)

            # Construct Metrics dict
            # Metric keys: feeding_duration, rumination_duration, estimated_intake, bunk_visits, feed_availability, mean_bout_length, feeding_rumination_ratio
            metrics: Dict[str, MetricItem] = {}

            if dev.feeding_daily_val is not None and dev.feeding_baseline_val is not None:
                metrics["feeding_duration"] = MetricItem(
                    value=round(dev.feeding_daily_val, 1),
                    unit="min/day",
                    baseline=round(dev.feeding_baseline_val, 1),
                    delta_pct=round(dev.feeding_delta_pct, 1),
                )

            if meal_stats.mean_bout_length is not None and baseline.baseline_mean_bout_length is not None:
                bout_delta = (
                    ((meal_stats.mean_bout_length - baseline.baseline_mean_bout_length) / max(baseline.baseline_mean_bout_length, 1.0)) * 100.0
                    if baseline.baseline_mean_bout_length > 0
                    else 0.0
                )
                metrics["mean_bout_length"] = MetricItem(
                    value=round(meal_stats.mean_bout_length, 1),
                    unit="min",
                    baseline=round(baseline.baseline_mean_bout_length, 1),
                    delta_pct=round(bout_delta, 1),
                )

            if rum_res.available and dev.rumination_daily_val is not None and dev.rumination_baseline_val is not None:
                metrics["rumination_duration"] = MetricItem(
                    value=round(dev.rumination_daily_val, 1),
                    unit="min/day",
                    baseline=round(dev.rumination_baseline_val, 1),
                    delta_pct=round(dev.rumination_delta_pct or 0.0, 1),
                )

            if dev.intake_daily_val is not None and dev.intake_baseline_val is not None:
                metrics["estimated_intake"] = MetricItem(
                    value=round(dev.intake_daily_val, 1),
                    unit="kg/day",
                    baseline=round(dev.intake_baseline_val, 1),
                    delta_pct=round(dev.intake_delta_pct or 0.0, 1),
                )

            if dev.visits_daily_val is not None and dev.visits_baseline_val is not None:
                metrics["bunk_visits"] = MetricItem(
                    value=round(dev.visits_daily_val, 1),
                    unit="visits/day",
                    baseline=round(dev.visits_baseline_val, 1),
                    delta_pct=round(dev.visits_delta_pct or 0.0, 1),
                )

            if avail_res.stream_exists and avail_res.mean_availability is not None:
                metrics["feed_availability"] = MetricItem(
                    value=round(avail_res.mean_availability, 1),
                    unit="%",
                    baseline=round(avail_res.baseline_availability or 85.0, 1),
                    delta_pct=round(dev.feed_avail_delta_pct or 0.0, 1),
                )

            if dev.ratio_val is not None and dev.ratio_baseline is not None:
                metrics["feeding_rumination_ratio"] = MetricItem(
                    value=round(dev.ratio_val, 2),
                    unit="ratio",
                    baseline=round(dev.ratio_baseline, 2),
                    delta_pct=round(dev.ratio_delta_pct or 0.0, 1),
                )

            # 13. Summary text
            summary = build_summary(
                cow_id=request.cow_id,
                metrics=metrics,
                finding=verdict.finding,
                is_herd_wide=herd_res.is_herd_wide,
                avail_status=avail_res.status,
                use_llm=self.cfg.summarizer.use_llm,
            )

            # 14. Follow-up suggestions
            followups: List[SuggestedFollowup] = []
            if verdict.finding == "supports":
                if herd_res.is_herd_wide:
                    followups.append(
                        SuggestedFollowup(
                            to_agent="environment_welfare",
                            reason="Herd-wide feeding change detected. Check microclimate, THI, or feed delivery schedule.",
                        )
                    )
                else:
                    followups.append(
                        SuggestedFollowup(
                            to_agent="production",
                            reason="Check whether daily milk yield dropped alongside intake reduction.",
                        )
                    )
                    # If availability is restricted -> note management cause instead of clinical health issue (no health follow-up)
                    if not avail_res.restricted:
                        followups.append(
                            SuggestedFollowup(
                                to_agent="health_behavior",
                                reason="Cow-specific feeding/rumination change, recommend veterinary review.",
                            )
                        )

            if avail_res.status == "unknown" and verdict.finding == "supports":
                limitations.append("Feed bunk availability telemetry missing; verify physical feed delivery records.")

            # Missing streams list
            missing_streams = []
            for stream_name, is_present in self.ts.stream_availability.items():
                if not is_present:
                    missing_streams.append(stream_name)

            data_qual = DataQuality(
                coverage_pct=anom_coverage_pct,
                missing_streams=missing_streams,
                notes=f"Baseline days: {baseline.n_days}. Anomaly window coverage: {anom_coverage_pct:.1f}%.",
            )

            herd_ctx = HerdContext(
                cows_analyzed=herd_res.cows_analyzed,
                cows_deviating=herd_res.cows_deviating,
                fraction_deviating=herd_res.fraction_deviating,
                is_herd_wide=herd_res.is_herd_wide,
            )

            response = AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding=verdict.finding,
                summary=summary,
                metrics=metrics,
                herd_context=herd_ctx,
                evidence_strength=verdict.evidence_strength,
                confidence=verdict.confidence,
                data_quality=data_qual,
                limitations=limitations,
                suggested_followups=followups,
            )

            self.message_log.append((request, response, now_iso))
            return response

        except Exception as e:
            logger.exception("Error during FeedingNutritionAgent query processing for cow '%s'", request.cow_id)
            err_resp = AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding="inconclusive",
                summary=f"Analysis encountered an unexpected failure: {str(e)}",
                metrics={},
                herd_context=HerdContext(),
                evidence_strength="none",
                confidence=0.0,
                data_quality=DataQuality(coverage_pct=0.0, missing_streams=["all"], notes=f"Execution exception: {str(e)}"),
                limitations=[f"Unexpected failure: {str(e)}"],
                suggested_followups=[],
            )
            self.message_log.append((request, err_resp, now_iso))
            return err_resp

    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns a plain callable node compatible with LangGraph multi-agent blackboard state.
        Does not import LangGraph.
        """
        def node(state: Dict[str, Any]) -> Dict[str, Any]:
            req_data = state.get("pending_request")
            if not req_data:
                return state

            if isinstance(req_data, dict):
                req = AgentRequest(**req_data)
            elif isinstance(req_data, AgentRequest):
                req = req_data
            else:
                return state

            if req.to_agent != self.agent_id:
                return state

            resp = self.handle_query(req)
            responses = list(state.get("specialist_responses", []))
            responses.append(resp.model_dump())

            return {
                **state,
                "specialist_responses": responses,
                "latest_response": resp.model_dump(),
                "pending_request": None,
            }

        return node
