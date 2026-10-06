"""
Environment & Welfare Agent — core implementation.

Implements the shared interface contract (section A.3):
    handle_query(AgentRequest) -> AgentResponse

Supports question_types:
    "environment_check"  – primary analysis
    "herd_wide_check"    – focus on herd-wide deviation
    "heat_stress_check"  – focus on heat stress thresholds

No Streamlit or LangGraph imports.  All I/O is via the AgentRequest / AgentResponse models.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from env_welfare_agent.src.env_agent.alignment import compute_temporal_alignment
from env_welfare_agent.src.env_agent.baseline import derive_baseline
from env_welfare_agent.src.env_agent.config import AgentConfig
from env_welfare_agent.src.env_agent.exposure import compute_exposure
from env_welfare_agent.src.env_agent.herd import compute_herd_deviation
from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    AlignmentResult,
    BaselineStats,
    CowActivityTimeseries,
    EnvTimeseries,
    ExposureMetrics,
    HerdDeviationResult,
    VerdictBreakdown,
)
from env_welfare_agent.src.env_agent.summarizer import build_summary
from env_welfare_agent.src.env_agent.verdict import compute_verdict

# Import shared contract models from the project root
import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.models import (  # noqa: E402
    AgentRequest,
    AgentResponse,
    DataQuality,
    HerdContext,
    MetricItem,
    SuggestedFollowup,
    TimeWindow,
)

logger = logging.getLogger(__name__)

SUPPORTED_QUESTION_TYPES = {"environment_check", "herd_wide_check", "heat_stress_check"}
AGENT_ID = "environment_welfare"


class EnvironmentWelfareAgent:
    """
    Production-quality Environment & Welfare Specialist Agent.

    Parameters
    ----------
    env_ts : EnvTimeseries
        Pre-loaded canonical environmental time series.
    cow_act_ts : CowActivityTimeseries
        Pre-loaded canonical per-cow activity time series.
    config : AgentConfig, optional
        Loaded from default.yaml if not provided.

    Usage
    -----
    >>> agent = EnvironmentWelfareAgent(env_ts, cow_act_ts)
    >>> response = agent.handle_query(request)
    """

    def __init__(
        self,
        env_ts: EnvTimeseries,
        cow_act_ts: CowActivityTimeseries,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._env_ts = env_ts
        self._cow_act_ts = cow_act_ts
        self._cfg = config or AgentConfig.load()
        self.message_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Primary interface entry point (section A.3)
    # ------------------------------------------------------------------

    def handle_query(self, request: AgentRequest) -> AgentResponse:
        """
        Main entry point.  Validates input, runs analysis, returns AgentResponse.
        Never raises; structured errors are returned as inconclusive responses.
        """
        try:
            return self._dispatch(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in handle_query: %s", exc)
            resp = self._error_response(request, str(exc))
            self._record(request, resp)
            return resp

    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Return a callable LangGraph node.  Reads 'pending_request' from state dict
        and writes back 'specialist_responses' and 'latest_response'.
        """
        def _node(state: Dict[str, Any]) -> Dict[str, Any]:
            req_data = state.get("pending_request")
            if not req_data:
                return state
            req = AgentRequest(**req_data) if isinstance(req_data, dict) else req_data
            if req.to_agent != AGENT_ID:
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
        return _node

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, request: AgentRequest) -> AgentResponse:
        # --- Validate question type ---
        if request.question_type not in SUPPORTED_QUESTION_TYPES:
            return self._unsupported_type_response(request)

        # --- Parse windows ---
        try:
            anomaly_start, anomaly_end = _parse_window(request.anomaly_window)
        except Exception as exc:
            return self._validation_error(request, f"Invalid anomaly_window: {exc}")

        if anomaly_start >= anomaly_end:
            return self._validation_error(request, "anomaly_window start must be before end.")

        baseline_start = baseline_end = None
        if request.baseline_window:
            try:
                baseline_start, baseline_end = _parse_window(request.baseline_window)
            except Exception as exc:
                return self._validation_error(request, f"Invalid baseline_window: {exc}")
            if baseline_start >= baseline_end:
                return self._validation_error(request, "baseline_window start must be before end.")

        # --- Check cow exists in activity data ---
        cow_col = COW_ACT_COLS["COW_ID"]
        known_cows = self._cow_act_ts.df[cow_col].unique().tolist() if not self._cow_act_ts.df.empty else []
        if request.cow_id not in known_cows:
            logger.warning("Cow '%s' not found in dataset. Known: %s", request.cow_id, known_cows[:5])
            return self._unknown_cow_response(request, known_cows)

        # --- Run analysis pipeline ---
        baseline_stats = derive_baseline(
            self._env_ts, anomaly_start, baseline_start, baseline_end, self._cfg.baseline
        )
        exposure = compute_exposure(
            self._env_ts, anomaly_start, anomaly_end, self._cfg.thi_thresholds, self._cfg.exposure
        )
        herd_result = compute_herd_deviation(
            self._cow_act_ts, request.cow_id,
            anomaly_start, anomaly_end, baseline_start, baseline_end,
            self._cfg.baseline.default_days_before_anomaly, self._cfg.herd,
        )
        alignment = compute_temporal_alignment(
            self._env_ts, self._cow_act_ts, anomaly_start, anomaly_end, self._cfg.alignment
        )
        verdict = compute_verdict(exposure, baseline_stats, herd_result, alignment, self._cfg)
        summary = build_summary(
            exposure, baseline_stats, herd_result, alignment, verdict,
            request.cow_id, self._cfg.summarizer,
        )

        # --- Build response ---
        response = self._build_response(
            request, exposure, baseline_stats, herd_result, alignment, verdict, summary
        )
        self._record(request, response)
        return response

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def _build_response(
        self,
        request: AgentRequest,
        exposure: ExposureMetrics,
        baseline: BaselineStats,
        herd: HerdDeviationResult,
        alignment: AlignmentResult,
        verdict: VerdictBreakdown,
        summary: str,
    ) -> AgentResponse:
        nan = float("nan")

        def _mi(val, unit, base, delta=None):
            if math.isnan(val):
                return MetricItem(value=nan, unit=unit, baseline=base if not math.isnan(base) else nan, delta_pct=nan)
            if delta is None:
                delta = ((val - base) / max(abs(base), 1e-6)) * 100.0 if not math.isnan(base) else nan
            return MetricItem(value=round(val, 2), unit=unit, baseline=round(base, 2), delta_pct=round(delta, 2))

        metrics = {
            "mean_thi": _mi(exposure.mean_thi, "", baseline.mean_thi),
            "max_thi": _mi(exposure.max_thi, "", baseline.p90_thi),
            "mean_barn_temp_c": _mi(exposure.mean_temp_c, "°C", baseline.mean_temp_c),
            "mean_rh_pct": _mi(exposure.mean_rh_pct, "%", baseline.mean_rh_pct),
            "hours_in_stress": MetricItem(
                value=round(exposure.hours_mild + exposure.hours_moderate + exposure.hours_severe + exposure.hours_emergency, 1),
                unit="h", baseline=0.0, delta_pct=nan,
            ),
            "cumulative_heat_load": MetricItem(
                value=round(exposure.cumulative_heat_load, 2), unit="THI·h", baseline=0.0, delta_pct=nan,
            ),
            "night_min_thi": MetricItem(
                value=round(exposure.night_min_thi, 2) if not math.isnan(exposure.night_min_thi) else nan,
                unit="", baseline=68.0, delta_pct=nan,
            ),
            "fraction_herd_deviating": MetricItem(
                value=round(herd.fraction_deviating, 3),
                unit="fraction", baseline=0.0,
                delta_pct=round(herd.fraction_deviating * 100, 1),
            ),
        }

        # Suggested follow-ups
        followups: List[SuggestedFollowup] = []
        if verdict.finding == "supports":
            followups.append(SuggestedFollowup(
                to_agent="feeding_nutrition",
                reason="Heat stress commonly suppresses dry matter intake and rumination; recommend checking feeding metrics.",
            ))
            followups.append(SuggestedFollowup(
                to_agent="production",
                reason="Elevated THI is associated with reduced milk yield; recommend checking lactation performance.",
            ))
        elif verdict.finding == "does_not_support":
            followups.append(SuggestedFollowup(
                to_agent="feeding_nutrition",
                reason="Environment does not explain the anomaly; inspect individual feeding and rumination for metabolic causes.",
            ))

        # Limitations
        limitations: List[str] = []
        if exposure.coverage_pct < 80:
            limitations.append(f"Anomaly window data coverage is {exposure.coverage_pct:.0f}%.")
        if baseline.coverage_pct < 70:
            limitations.append(f"Baseline coverage is {baseline.coverage_pct:.0f}% (some hours missing).")
        if not alignment.sufficient_data:
            limitations.append("Insufficient data for reliable temporal cross-correlation.")
        if herd.cows_analyzed < 3:
            limitations.append(f"Only {herd.cows_analyzed} cow(s) available for herd check; fraction may be unreliable.")
        limitations.append(
            "THI thresholds are reference values that vary by breed, acclimatization, and literature source."
        )
        limitations.append("This is decision support, not a clinical diagnosis.")

        # Welfare notes added to limitations list for transparency
        if exposure.poor_night_recovery:
            limitations.append("Sustained heat load with poor nocturnal cooling detected.")
        if not math.isnan(exposure.max_thi_rate_per_hour) and exposure.max_thi_rate_per_hour > 5.0:
            limitations.append(
                f"Rapid THI rise detected ({exposure.max_thi_rate_per_hour:.1f} THI/h); "
                "sudden environmental changes may cause acute stress."
            )

        # Missing streams
        missing_streams: List[str] = []
        if math.isnan(exposure.mean_thi):
            missing_streams.extend(["thi", "barn_temp_c"])
        if math.isnan(exposure.mean_rh_pct):
            missing_streams.append("relative_humidity")
        missing_streams.extend(self._env_ts.notes)  # adapter-level quality notes

        return AgentResponse(
            query_id=request.query_id,
            from_agent=AGENT_ID,
            to_agent=request.from_agent,
            cow_id=request.cow_id,
            finding=verdict.finding,
            summary=summary,
            metrics=metrics,
            herd_context=HerdContext(
                cows_analyzed=herd.cows_analyzed,
                cows_deviating=herd.cows_deviating,
                fraction_deviating=round(herd.fraction_deviating, 3),
                is_herd_wide=herd.is_herd_wide,
            ),
            evidence_strength=verdict.evidence_strength,
            confidence=verdict.confidence,
            data_quality=DataQuality(
                coverage_pct=round(exposure.coverage_pct, 1),
                missing_streams=[s for s in missing_streams if s],
                notes="; ".join(self._env_ts.notes[:3]),
            ),
            limitations=limitations,
            suggested_followups=followups,
        )

    # ------------------------------------------------------------------
    # Error / validation response helpers
    # ------------------------------------------------------------------

    def _error_response(self, request: AgentRequest, detail: str) -> AgentResponse:
        return AgentResponse(
            query_id=request.query_id,
            from_agent=AGENT_ID,
            to_agent=request.from_agent,
            cow_id=request.cow_id,
            finding="inconclusive",
            summary=f"Analysis could not be completed: {detail[:200]}. Recommend manual inspection.",
            metrics={},
            herd_context=HerdContext(),
            evidence_strength="none",
            confidence=0.0,
            data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes=f"Error: {detail}"),
            limitations=["Processing error; results unavailable."],
            suggested_followups=[],
        )

    def _validation_error(self, request: AgentRequest, msg: str) -> AgentResponse:
        resp = self._error_response(request, msg)
        self._record(request, resp)
        return resp

    def _unsupported_type_response(self, request: AgentRequest) -> AgentResponse:
        msg = (
            f"Unsupported question_type '{request.question_type}'. "
            f"Supported: {sorted(SUPPORTED_QUESTION_TYPES)}."
        )
        resp = self._error_response(request, msg)
        self._record(request, resp)
        return resp

    def _unknown_cow_response(self, request: AgentRequest, known_cows: List[str]) -> AgentResponse:
        msg = (
            f"Cow '{request.cow_id}' not found in activity dataset. "
            f"Known cows (first 5): {known_cows[:5]}."
        )
        resp = self._error_response(request, msg)
        self._record(request, resp)
        return resp

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _record(self, request: AgentRequest, response: AgentResponse) -> None:
        self.message_history.append({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "query_id": request.query_id,
            "cow_id": request.cow_id,
            "question_type": request.question_type,
            "finding": response.finding,
            "confidence": response.confidence,
            "evidence_strength": response.evidence_strength,
            "request_json": request.model_dump_json(indent=2),
            "response_json": response.model_dump_json(indent=2),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_window(tw: TimeWindow) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(tw.start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(tw.end.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return start, end
