"""
Health & Behavior Lead Investigator Agent.
Coordinates individualized baseline derivation, anomaly detection with Z-scores,
goal-directed specialist inquiries, and multi-agent clinical synthesis.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.models import AgentRequest, AgentResponse, TimeWindow
from src.data.repository import LivestockRepository, get_repository
from health_agent.src.health_agent.config import HealthAgentConfig
from health_agent.src.health_agent.schemas import (
    HealthBaseline,
    AnomalyDetectionResult,
)
from health_agent.src.health_agent.baseline import derive_health_baseline
from health_agent.src.health_agent.anomaly import detect_health_anomalies
from health_agent.src.health_agent.query_builder import (
    plan_investigation_queries,
    build_followup_query,
)
from health_agent.src.health_agent.synthesis import synthesize_clinical_assessment

logger = logging.getLogger(__name__)


class HealthBehaviorAgent:
    """
    Lead Investigator in the multi-agent system.
    Detects sensor-level behavioral and physiological anomalies, orchestrates specialist consultations,
    and synthesizes veterinary early-warning decision support.
    """

    def __init__(
        self,
        repo: Optional[LivestockRepository] = None,
        config: Optional[HealthAgentConfig] = None,
        use_llm: bool = False,
    ) -> None:
        self.agent_id = "health_behavior"
        self.accent_color = "#e11d48"  # Rose / Crimson
        self.repo = repo or get_repository()
        self.config = config or HealthAgentConfig.load()
        self.use_llm = use_llm
        self.message_history: List[Dict[str, Any]] = []

    def derive_baseline(
        self,
        cow_id: str,
        anomaly_start: datetime,
        baseline_start: Optional[datetime] = None,
        baseline_end: Optional[datetime] = None,
    ) -> HealthBaseline:
        """Computes individual cow moving baseline for vital signs and posture."""
        return derive_health_baseline(
            cow_id=cow_id,
            anomaly_start=anomaly_start,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            repo=self.repo,
            cfg=self.config.baseline,
        )

    def detect_anomalies(
        self,
        cow_id: str,
        start_dt: datetime,
        end_dt: datetime,
        baseline_start_dt: Optional[datetime] = None,
        baseline_end_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Scans cow timeline to detect anomalies against the cow's own baseline.
        Returns detailed anomaly metrics, clinical flags, and severity score.
        """
        result: AnomalyDetectionResult = detect_health_anomalies(
            cow_id=cow_id,
            start_dt=start_dt,
            end_dt=end_dt,
            baseline_start_dt=baseline_start_dt,
            baseline_end_dt=baseline_end_dt,
            repo=self.repo,
            cfg=self.config,
        )
        return result.model_dump()

    def plan_queries(
        self,
        anomaly_data: Dict[str, Any] | AnomalyDetectionResult,
        baseline_window: Optional[TimeWindow] = None,
    ) -> List[AgentRequest]:
        """Generates tailored, goal-directed queries based on anomaly signature."""
        if isinstance(anomaly_data, dict):
            anomaly = AnomalyDetectionResult(**anomaly_data)
        else:
            anomaly = anomaly_data

        return plan_investigation_queries(
            anomaly=anomaly,
            baseline_window=baseline_window,
        )

    def synthesize_assessment(
        self,
        cow_id: str,
        anomaly_data: Dict[str, Any] | AnomalyDetectionResult,
        specialist_responses: List[AgentResponse],
    ) -> Dict[str, Any]:
        """
        Integrates all specialist evidence and computes the final risk assessment
        and prioritized veterinary decision-support guidance.
        """
        if isinstance(anomaly_data, dict):
            anomaly = AnomalyDetectionResult(**anomaly_data)
        else:
            anomaly = anomaly_data

        assessment = synthesize_clinical_assessment(
            cow_id=cow_id,
            anomaly=anomaly,
            specialist_responses=specialist_responses,
        )

        # Record assessment interaction in history
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "cow_id": cow_id,
            "risk_level": assessment["risk_level"],
            "summary_statement": assessment["summary_statement"],
            "specialist_count": len(specialist_responses),
        })

        return assessment

    def build_followup(
        self,
        parent_response: AgentResponse,
        anomaly_data: Dict[str, Any] | AnomalyDetectionResult,
        existing_queries: List[AgentRequest],
        max_followups: int = 2,
    ) -> Optional[AgentRequest]:
        """Builds a follow-up inquiry based on specialist recommendations."""
        if isinstance(anomaly_data, dict):
            anomaly = AnomalyDetectionResult(**anomaly_data)
        else:
            anomaly = anomaly_data

        return build_followup_query(
            parent_response=parent_response,
            anomaly=anomaly,
            existing_queries=existing_queries,
            max_followups=max_followups,
        )

