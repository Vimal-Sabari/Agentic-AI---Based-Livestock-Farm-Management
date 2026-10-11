"""
Health & Behavior Lead Investigator Agent Adapter.
Connects the dedicated health_agent modular package to LangGraph and Streamlit.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from health_agent.src.health_agent.agent import HealthBehaviorAgent as ImplAgent
from health_agent.src.health_agent.config import HealthAgentConfig
from src.core.models import AgentRequest, AgentResponse, TimeWindow
from src.data.repository import LivestockRepository, get_repository


class HealthBehaviorAgent:
    """
    Lead Investigator in the multi-agent system.
    Delegates to the dedicated health_agent modular package.
    """

    def __init__(
        self,
        repo: Optional[LivestockRepository] = None,
        use_llm: bool = False,
    ) -> None:
        self.agent_id = "health_behavior"
        self.accent_color = "#e11d48"  # Rose / Crimson
        self.repo = repo or get_repository()
        self.use_llm = use_llm
        self._impl = ImplAgent(repo=self.repo, use_llm=use_llm)

    @property
    def message_history(self) -> List[Dict[str, Any]]:
        return self._impl.message_history

    def detect_anomalies(
        self,
        cow_id: str,
        start_dt: datetime,
        end_dt: datetime,
        baseline_start_dt: Optional[datetime] = None,
        baseline_end_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Detects anomalies against individual baseline."""
        return self._impl.detect_anomalies(
            cow_id=cow_id,
            start_dt=start_dt,
            end_dt=end_dt,
            baseline_start_dt=baseline_start_dt,
            baseline_end_dt=baseline_end_dt,
        )

    def plan_queries(
        self,
        anomaly_data: Dict[str, Any],
        baseline_window: Optional[TimeWindow] = None,
    ) -> List[AgentRequest]:
        """Plans goal-directed queries to specialists."""
        return self._impl.plan_queries(
            anomaly_data=anomaly_data,
            baseline_window=baseline_window,
        )

    def create_query(
        self,
        to_agent: str,
        cow_id: str,
        start_iso: str,
        end_iso: str,
        question: str,
        question_type: str,
        anomaly_summary: str,
        baseline_start_iso: Optional[str] = None,
        baseline_end_iso: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> AgentRequest:
        """Constructs an AgentRequest complying strictly with section A.3."""
        baseline_window = None
        if baseline_start_iso and baseline_end_iso:
            baseline_window = TimeWindow(start=baseline_start_iso, end=baseline_end_iso)

        return AgentRequest(
            from_agent=self.agent_id,
            to_agent=to_agent,
            cow_id=cow_id,
            question=question,
            question_type=question_type,
            anomaly_window=TimeWindow(start=start_iso, end=end_iso),
            baseline_window=baseline_window,
            context={
                "anomaly_summary": anomaly_summary,
                "extra": extra or {}
            }
        )

    def synthesize_assessment(
        self,
        cow_id: str,
        anomaly_data: Dict[str, Any],
        specialist_responses: List[AgentResponse],
    ) -> Dict[str, Any]:
        """Synthesizes specialist findings into veterinary guidance."""
        return self._impl.synthesize_assessment(
            cow_id=cow_id,
            anomaly_data=anomaly_data,
            specialist_responses=specialist_responses,
        )

    def build_followup(
        self,
        parent_response: AgentResponse,
        anomaly_data: Dict[str, Any],
        existing_queries: List[AgentRequest],
        max_followups: int = 2,
    ) -> Optional[AgentRequest]:
        """Builds a follow-up inquiry based on specialist referrals."""
        return self._impl.build_followup(
            parent_response=parent_response,
            anomaly_data=anomaly_data,
            existing_queries=existing_queries,
            max_followups=max_followups,
        )

