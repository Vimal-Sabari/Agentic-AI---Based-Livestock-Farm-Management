"""
Veterinary Clinical Synthesis Engine for Health & Behavior Lead Agent.
Aggregates specialist responses, cross-correlates clinical evidence, assigns risk ratings,
evaluates differential diagnostic hypotheses, and builds actionable veterinary checklists.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from src.core.models import AgentResponse, MetricItem
from health_agent.src.health_agent.schemas import AnomalyDetectionResult

logger = logging.getLogger(__name__)


def synthesize_clinical_assessment(
    cow_id: str,
    anomaly: AnomalyDetectionResult,
    specialist_responses: List[AgentResponse],
) -> Dict[str, Any]:
    """
    Integrates individual anomaly metrics with all specialist evidence
    to produce early-warning veterinary decision-support guidance.
    """
    resp_by_agent: Dict[str, AgentResponse] = {r.from_agent: r for r in specialist_responses}

    env_resp = resp_by_agent.get("environment_welfare")
    feed_resp = resp_by_agent.get("feeding_nutrition")
    prod_resp = resp_by_agent.get("production")

    is_fever = anomaly.is_fever
    is_lethargic = anomaly.is_lethargic

    # 1. Evaluate Environment Evidence (Rule out herd-wide thermal stress)
    env_supports = bool(env_resp and env_resp.finding == "supports")
    is_herd_heat_stress = bool(
        env_supports and env_resp is not None and env_resp.herd_context.is_herd_wide
    )

    # 2. Evaluate Feeding & Production Evidence
    feed_supports = bool(feed_resp and feed_resp.finding == "supports")
    prod_supports = bool(prod_resp and prod_resp.finding == "supports")

    # 3. Calculate Aggregate Confidence
    supporting_confidences = [r.confidence for r in specialist_responses if r.finding == "supports"]
    if supporting_confidences:
        mean_confidence = round(float(np.mean(supporting_confidences)), 2)
    else:
        confidence_scores = [r.confidence for r in specialist_responses if r.confidence > 0]
        mean_confidence = round(float(np.mean(confidence_scores)), 2) if confidence_scores else 0.85

    possible_conditions: List[str] = []
    recommended_actions: List[str] = []

    # ==========================================================================
    # CLINICAL REASONING MATRIX
    # ==========================================================================

    # SCENARIO A: Herd-wide Heat Stress / Hyperthermia
    if is_herd_heat_stress and env_resp is not None:
        risk_level = "High" if (feed_supports or prod_supports) else "Moderate"
        thi_val = env_resp.metrics.get("thi", MetricItem(value=0, baseline=0, delta_pct=0)).value
        pct_herd = env_resp.herd_context.fraction_deviating * 100.0

        summary_statement = (
            f"Evaluation indicates herd-wide microclimate thermal strain (THI {thi_val:.1f}) "
            f"affecting {pct_herd:.0f}% of cows analyzed. "
            f"Findings are consistent with acute environmental heat stress rather than isolated infectious disease."
        )
        possible_conditions.append("Herd-wide Heat Stress / Ambient Hyperthermia")
        possible_conditions.append("Microclimate Ventilation Deficit")

        recommended_actions.append("Activate high-volume barn ventilation fans and evaporative soaker cycles.")
        recommended_actions.append("Ensure unobstructed access to chilled, clean drinking water troughs.")
        recommended_actions.append("Shift TMR feeding timing to cooler evening hours to mitigate ruminal heat load.")

    # SCENARIO B: Acute Individual Febrile Illness / Mastitis / Metritis
    elif is_fever and (feed_supports or prod_supports):
        risk_level = "Critical" if (feed_supports and prod_supports) else "High"
        
        feed_note = "rumination collapse " if feed_supports else ""
        prod_note = "and milk production drop " if prod_supports else ""
        
        summary_statement = (
            f"Isolated acute physiological anomaly in {cow_id} with core temperature elevation "
            f"unsupported by microclimate (THI normal). Concurrently supported by significant "
            f"{feed_note}{prod_note}. "
            f"Findings are consistent with acute individual systemic or localized inflammation."
        )

        if prod_supports:
            possible_conditions.append("Clinical Mastitis (supported by elevated conductivity and yield collapse)")
        possible_conditions.append("Acute Systemic Infection / Metritis / Foot Rot")
        possible_conditions.append("Severe Infectious Pododermatitis (Lameness)")

        recommended_actions.append("Immediate physical veterinary examination (rectal temp, heart rate, rumen auscultation).")
        recommended_actions.append("Perform California Mastitis Test (CMT) on all four quarters and inspect milk secretions.")
        recommended_actions.append("Isolate animal in hospital pen with clean deep straw bedding.")
        if feed_supports:
            recommended_actions.append("Administer veterinarian-approved anti-inflammatory / supportive hydration therapy.")

    # SCENARIO C: Nutritional / Metabolic Depression without Fever (Ketosis / LDA)
    elif feed_supports and not is_fever:
        risk_level = "High" if prod_supports else "Moderate"
        summary_statement = (
            f"Individual nutritional depression in {cow_id} with significant rumination drop without febrile response. "
            f"Findings are consistent with early metabolic disturbance or digestive discomfort."
        )
        possible_conditions.append("Subclinical Ketosis / Off-Feed Indigestion")
        possible_conditions.append("Left Displaced Abomasum (LDA)")
        possible_conditions.append("Subacute Ruminal Acidosis (SARA)")

        recommended_actions.append("Check blood or milk beta-hydroxybutyrate (BHB) levels for ketosis.")
        recommended_actions.append("Perform left flank percussion/auscultation for abomasal ping.")
        recommended_actions.append("Assess oral cud chewing and provide oral propylene glycol if indicated.")

    # SCENARIO D: Normal Telemetry (Within Individual Baselines)
    elif not anomaly.has_anomaly:
        risk_level = "Low"
        summary_statement = (
            f"Cow {cow_id} displays behavioral, nutritional, and production metrics consistent with her established individual baseline. "
            f"No clinical intervention indicated."
        )
        possible_conditions.append("Normal Physiological State")
        recommended_actions.append("Continue routine automated monitoring.")

    # SCENARIO E: Mild / Transient Deviation
    else:
        risk_level = "Moderate"
        summary_statement = (
            f"Mild behavioral deviation observed in {cow_id}. Specialist telemetry does not demonstrate acute systemic failure. "
            f"Findings suggest transient stress or early onset requiring close observation."
        )
        possible_conditions.append("Transient Environmental/Social Stress")
        possible_conditions.append("Early Onset Malaise / Subclinical Lameness")

        recommended_actions.append("Re-evaluate sensor telemetry in 12 hours.")
        recommended_actions.append("Visual check by pen rider during next feed push-up.")

    # Convert metric comparisons to dictionary
    anomaly_metrics_dict = {
        name: {
            "value": m.value,
            "baseline_mean": m.baseline_mean,
            "baseline_std": m.baseline_std,
            "delta_pct": m.delta_pct,
            "z_score": m.z_score,
            "unit": m.unit,
        }
        for name, m in anomaly.metrics.items()
    }

    return {
        "cow_id": cow_id,
        "generated_at": datetime.now().isoformat(),
        "risk_level": risk_level,
        "confidence": mean_confidence,
        "summary_statement": summary_statement,
        "possible_conditions": possible_conditions,
        "recommended_actions": recommended_actions,
        "anomaly_severity_score": anomaly.severity_score,
        "anomaly_metrics": anomaly_metrics_dict,
        "specialist_findings": [r.model_dump() for r in specialist_responses],
    }
