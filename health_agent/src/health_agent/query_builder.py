"""
Goal-Directed Query Builder for Health & Behavior Lead Agent.
Dynamically tailors inquiries and sets of target specialists based on detected anomaly signatures.
Phrases hypotheses as balanced pairs (evidence for and against) and supports bounded follow-up chaining.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.core.models import (
    AgentRequest,
    AgentResponse,
    QuestionType,
    TimeWindow,
)
from health_agent.src.health_agent.schemas import AnomalyDetectionResult

logger = logging.getLogger(__name__)


def _determine_priority(severity_score: float) -> str:
    """Maps continuous severity score to discrete request priority."""
    if severity_score >= 0.75:
        return "urgent"
    elif severity_score >= 0.45:
        return "high"
    elif severity_score >= 0.20:
        return "normal"
    return "low"


def _build_structured_context(anomaly: AnomalyDetectionResult) -> Dict[str, Any]:
    """Extracts compact, structured metric comparison and clinical flags for specialist context."""
    metric_context: Dict[str, Dict[str, Any]] = {}
    for name, m in anomaly.metrics.items():
        metric_context[name] = {
            "value": m.value,
            "baseline_mean": m.baseline_mean,
            "baseline_std": m.baseline_std,
            "delta_pct": m.delta_pct,
            "z_score": m.z_score,
            "unit": m.unit
        }

    return {
        "anomaly_summary": anomaly.anomaly_summary,
        "severity_score": anomaly.severity_score,
        "is_fever": anomaly.is_fever,
        "is_lethargic": anomaly.is_lethargic,
        "is_restless": anomaly.is_restless,
        "anomaly_metrics": metric_context,
        "start_dt": anomaly.start_dt,
        "end_dt": anomaly.end_dt,
    }


def plan_investigation_queries(
    anomaly: AnomalyDetectionResult,
    baseline_window: Optional[TimeWindow] = None,
) -> List[AgentRequest]:
    """
    Formulates a targeted, goal-directed list of inquiries based on the cow's specific anomaly signature.
    Phrases hypotheses as balanced pairs (evidence for and against) to minimize confirmation bias.
    """
    if not anomaly.has_anomaly:
        return []

    priority = _determine_priority(anomaly.severity_score)
    context = _build_structured_context(anomaly)
    window = TimeWindow(start=anomaly.start_dt, end=anomaly.end_dt)
    cow_id = anomaly.cow_id

    requests: List[AgentRequest] = []

    # CASE 1: Fever + Lethargy (Acute systemic illness, e.g. Mastitis, Metritis, or extreme Heat Stress)
    if anomaly.is_fever and anomaly.is_lethargic:
        # Priority 1: Environment - Rule in/out thermal microclimate vs individual infection
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="environment_welfare",
            cow_id=cow_id,
            question="Evaluate evidence for and against microclimate thermal stress vs individual systemic fever (THI levels, herd-wide fraction, nighttime cooling).",
            question_type=QuestionType.HEAT_STRESS_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

        # Priority 2: Feeding - Check appetite suppression vs bunk delivery issue
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=cow_id,
            question="Evaluate evidence for and against pathological anorexia/digestive shutdown vs feed bunk delivery delay (rumination time, dry matter intake, bunk fill %).",
            question_type=QuestionType.FEEDING_DROP_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

        # Priority 3: Production - Check mastitic yield collapse and milk conductivity
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="production",
            cow_id=cow_id,
            question="Evaluate evidence for and against acute mastitic lactation collapse vs normal daily fluctuation (daily milk yield delta, slope kg/day², electrical conductivity).",
            question_type=QuestionType.YIELD_IMPACT_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

    # CASE 2: Fever Alone (Early febrile onset without pronounced activity drop yet)
    elif anomaly.is_fever:
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="environment_welfare",
            cow_id=cow_id,
            question="Evaluate evidence for and against ambient heat stress as the primary driver of the detected febrile temperature shift.",
            question_type=QuestionType.HEAT_STRESS_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=cow_id,
            question="Evaluate evidence for early rumination depression corresponding with the observed core temperature spike.",
            question_type=QuestionType.FEEDING_DROP_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

    # CASE 3: Lethargy Alone without Fever (Subclinical Ketosis, Lameness, Indigestion, Displaced Abomasum)
    elif anomaly.is_lethargic:
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=cow_id,
            question="Evaluate evidence for and against metabolic/digestive indigestion vs feed management delay (rumination drop, intake decline, bunk visits).",
            question_type=QuestionType.FEEDING_DROP_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="environment_welfare",
            cow_id=cow_id,
            question="Evaluate evidence for and against barn environment factors (high humidity, poor ventilation) contributing to behavioral lethargy.",
            question_type=QuestionType.ENVIRONMENT_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

    # CASE 4: Restlessness / Surge in Activity (Estrus, Acute Discomfort, Insect Agitation)
    elif anomaly.is_restless:
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=cow_id,
            question="Evaluate evidence for feeding disruption or irregular bunk visitation during the observed restlessness window.",
            question_type=QuestionType.INTAKE_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))
        requests.append(AgentRequest(
            query_id=str(uuid.uuid4()),
            from_agent="health_behavior",
            to_agent="production",
            cow_id=cow_id,
            question="Evaluate evidence for milk yield fluctuations or parlor entry timing irregularities consistent with estrus or restlessness.",
            question_type=QuestionType.YIELD_IMPACT_CHECK,
            anomaly_window=window,
            baseline_window=baseline_window,
            priority=priority,
            timeout_s=5.0,
            context=context
        ))

    return requests


def build_followup_query(
    parent_response: AgentResponse,
    anomaly: AnomalyDetectionResult,
    existing_queries: List[AgentRequest],
    max_followups: int = 2,
) -> Optional[AgentRequest]:
    """
    Consumes suggested_followups from specialist responses or refines inconclusive inquiries.
    Enforces a strict max_followups bound to prevent cyclic recursion.
    """
    # Count how many follow-up queries have already been created
    followup_count = sum(1 for q in existing_queries if q.context.get("is_followup", False))
    if followup_count >= max_followups:
        logger.debug(f"Reached max follow-up limit ({max_followups}) for investigation.")
        return None

    consulted_agents = {q.to_agent for q in existing_queries}

    # 1. Inspect suggested_followups from specialist
    for followup in parent_response.suggested_followups:
        target_agent = followup.to_agent
        if target_agent not in consulted_agents and target_agent != "health_behavior":
            # Map target agent to appropriate controlled question type
            q_type = QuestionType.GENERAL
            if target_agent == "environment_welfare":
                q_type = QuestionType.HEAT_STRESS_CHECK
            elif target_agent == "feeding_nutrition":
                q_type = QuestionType.FEEDING_DROP_CHECK
            elif target_agent == "production":
                q_type = QuestionType.YIELD_IMPACT_CHECK

            context = _build_structured_context(anomaly)
            context["is_followup"] = True
            context["followup_source_agent"] = parent_response.from_agent
            context["followup_reason"] = followup.reason

            return AgentRequest(
                query_id=str(uuid.uuid4()),
                from_agent="health_behavior",
                to_agent=target_agent,
                cow_id=parent_response.cow_id,
                question=f"Follow-up inquiry referred by {parent_response.from_agent}: {followup.reason}",
                question_type=q_type,
                anomaly_window=TimeWindow(start=anomaly.start_dt, end=anomaly.end_dt),
                priority=_determine_priority(anomaly.severity_score),
                timeout_s=5.0,
                context=context
            )

    return None
