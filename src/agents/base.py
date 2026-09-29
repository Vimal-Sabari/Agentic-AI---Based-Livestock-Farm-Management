"""
Base Specialist Agent Abstract Interface.
Implements section A.3 shared contract:
- Entry point: handle_query(request: AgentRequest) -> AgentResponse
- LangGraph node wrapper: as_langgraph_node()
- Statistical analysis with optional template or LLM natural language rephrasing
- Strict error isolation, baseline derivation, and data quality tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from src.core.models import (
    AgentRequest,
    AgentResponse,
    MetricItem,
    HerdContext,
    DataQuality,
    SuggestedFollowup,
)
from src.data.repository import LivestockRepository, get_repository


class BaseSpecialistAgent(ABC):
    """
    Abstract base class for all specialist agents.
    Enforces deterministic calculations, strict metric-to-summary binding,
    and fallback natural-language generation.
    """

    def __init__(
        self,
        agent_id: str,
        accent_color: str,
        repo: Optional[LivestockRepository] = None,
        use_llm: bool = False
    ):
        self.agent_id = agent_id
        self.accent_color = accent_color
        self.repo = repo or get_repository()
        self.use_llm = use_llm
        self.message_history: List[Dict[str, Any]] = []

    def handle_query(self, request: AgentRequest) -> AgentResponse:
        """
        Public entry point complying with section A.3 interface contract.
        Wraps domain-specific analysis with robust error boundaries and audit logging.
        """
        try:
            # Parse anomaly window
            start_dt = datetime.fromisoformat(request.anomaly_window.start.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(request.anomaly_window.end.replace("Z", "+00:00"))

            # Parse optional baseline window
            base_start_dt = None
            base_end_dt = None
            if request.baseline_window:
                base_start_dt = datetime.fromisoformat(request.baseline_window.start.replace("Z", "+00:00"))
                base_end_dt = datetime.fromisoformat(request.baseline_window.end.replace("Z", "+00:00"))

            # Check cow existence
            all_cows = self.repo.get_cows()
            if request.cow_id not in all_cows:
                response = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"Cow '{request.cow_id}' was not found in the herd repository. No records available.",
                    metrics={},
                    herd_context=HerdContext(cows_analyzed=len(all_cows)),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=["all"], notes="Unknown cow ID"),
                    limitations=[f"Cow ID '{request.cow_id}' does not exist in dataset."],
                    suggested_followups=[]
                )
                self._record_interaction(request, response)
                return response

            # Slice anomaly window data
            cow_anomaly_df = self.repo.get_cow_data(request.cow_id, start_time=start_dt, end_time=end_dt)
            if cow_anomaly_df.empty:
                response = AgentResponse(
                    query_id=request.query_id,
                    from_agent=self.agent_id,
                    to_agent=request.from_agent,
                    cow_id=request.cow_id,
                    finding="inconclusive",
                    summary=f"No telemetry records available for {request.cow_id} in the requested anomaly window.",
                    metrics={},
                    herd_context=HerdContext(cows_analyzed=len(all_cows)),
                    evidence_strength="none",
                    confidence=0.0,
                    data_quality=DataQuality(coverage_pct=0.0, missing_streams=["all"], notes="Empty time window"),
                    limitations=["Time window has no sensor data for this animal."],
                    suggested_followups=[]
                )
                self._record_interaction(request, response)
                return response

            # Derive individual cow baseline
            cow_baseline_df = self.repo.compute_individual_baseline(
                cow_id=request.cow_id,
                anomaly_start=start_dt,
                baseline_start=base_start_dt,
                baseline_end=base_end_dt
            )

            # Delegate to specialized analysis
            response = self._analyze(
                request=request,
                anomaly_df=cow_anomaly_df,
                baseline_df=cow_baseline_df,
                start_dt=start_dt,
                end_dt=end_dt
            )

            self._record_interaction(request, response)
            return response

        except Exception as e:
            # Fallback structured error response ensuring zero system crashes
            error_response = AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding="inconclusive",
                summary=f"Analysis encountered an unexpected condition: {str(e)[:150]}. Recommend manual inspection.",
                metrics={},
                herd_context=HerdContext(),
                evidence_strength="none",
                confidence=0.0,
                data_quality=DataQuality(coverage_pct=0.0, missing_streams=[], notes=f"Processing error: {str(e)}"),
                limitations=["Analysis interrupted due to internal processing error."],
                suggested_followups=[]
            )
            self._record_interaction(request, error_response)
            return error_response

    @abstractmethod
    def _analyze(
        self,
        request: AgentRequest,
        anomaly_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        start_dt: datetime,
        end_dt: datetime
    ) -> AgentResponse:
        """Domain-specific statistical evaluation implemented by each specialist."""
        pass

    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Wraps the specialist agent as a callable LangGraph node.
        Reads 'current_request' or specialist-targeted request from state and writes back 'agent_responses'.
        """
        def node_callable(state: Dict[str, Any]) -> Dict[str, Any]:
            req_data = state.get("pending_request")
            if not req_data:
                # If no direct pending request, look in state context
                return state

            if isinstance(req_data, dict):
                request = AgentRequest(**req_data)
            else:
                request = req_data

            if request.to_agent != self.agent_id:
                # Not addressed to this agent
                return state

            response = self.handle_query(request)

            responses = state.get("specialist_responses", [])
            responses.append(response.model_dump())

            # Return updated state
            return {
                **state,
                "specialist_responses": responses,
                "latest_response": response.model_dump(),
                "pending_request": None
            }

        return node_callable

    def _record_interaction(self, request: AgentRequest, response: AgentResponse) -> None:
        """Audit logging for message history and UI session log."""
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "query_id": request.query_id,
            "cow_id": request.cow_id,
            "question": request.question,
            "question_type": request.question_type,
            "finding": response.finding,
            "confidence": response.confidence,
            "evidence_strength": response.evidence_strength,
            "request_json": request.model_dump_json(indent=2),
            "response_json": response.model_dump_json(indent=2),
        })
