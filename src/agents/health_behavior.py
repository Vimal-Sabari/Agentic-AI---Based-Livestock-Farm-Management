"""
Health & Behavior Lead Investigator Agent.
Role: Lead investigator. Builds per-cow baselines, detects anomalies (activity, lying,
acceleration, temperature), determines investigation goals, dispatches specialist queries,
integrates evidence, and produces the final decision-support assessment for veterinary review.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.models import (
    AgentRequest,
    AgentResponse,
    MetricItem,
    HerdContext,
    DataQuality,
    SuggestedFollowup,
    TimeWindow,
)
from src.data.schema import CANONICAL_COLUMNS
from src.data.repository import LivestockRepository, get_repository


class HealthBehaviorAgent:
    """
    Lead Investigator in the multi-agent system.
    Detects sensor-level behavioral and physiological anomalies, orchestrates specialist consultations,
    and synthesizes veterinary early-warning decision support.
    """

    def __init__(
        self,
        repo: Optional[LivestockRepository] = None,
        use_llm: bool = False
    ):
        self.agent_id = "health_behavior"
        self.accent_color = "#e11d48"  # Rose / Crimson
        self.repo = repo or get_repository()
        self.use_llm = use_llm
        self.message_history: List[Dict[str, Any]] = []

    def detect_anomalies(
        self,
        cow_id: str,
        start_dt: datetime,
        end_dt: datetime,
        baseline_start_dt: Optional[datetime] = None,
        baseline_end_dt: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Scans cow timeline to detect anomalies against the cow's own baseline.
        Returns detailed anomaly metrics and flags.
        """
        cow_df = self.repo.get_cow_data(cow_id, start_time=start_dt, end_time=end_dt)
        if cow_df.empty:
            return {
                "has_anomaly": False,
                "reason": f"No telemetry for cow '{cow_id}' in specified window.",
                "metrics": {},
                "anomaly_summary": "No data",
                "severity_score": 0.0,
            }

        base_df = self.repo.compute_individual_baseline(
            cow_id=cow_id,
            anomaly_start=start_dt,
            baseline_start=baseline_start_dt,
            baseline_end=baseline_end_dt
        )

        temp_col = CANONICAL_COLUMNS["CORE_BODY_TEMP_C"]
        act_col = CANONICAL_COLUMNS["ACTIVITY_INDEX"]
        lying_col = CANONICAL_COLUMNS["LYING_TIME_MIN"]
        accel_col = CANONICAL_COLUMNS["ACCELERATION_RMS"]
        steps_col = CANONICAL_COLUMNS["STEP_COUNT"]

        # Compute metrics
        val_temp = round(float(cow_df[temp_col].mean()), 2) if temp_col in cow_df else 38.6
        base_temp = round(float(base_df[temp_col].mean()), 2) if not base_df.empty and temp_col in base_df else 38.6
        delta_temp = round(((val_temp - base_temp) / max(0.1, base_temp)) * 100.0, 1)

        val_act = round(float(cow_df[act_col].mean()), 1) if act_col in cow_df else 50.0
        base_act = round(float(base_df[act_col].mean()), 1) if not base_df.empty and act_col in base_df else 50.0
        delta_act = round(((val_act - base_act) / max(1.0, base_act)) * 100.0, 1)

        # Scale hourly lying min to daily equivalent
        scale_day = 24.0 / max(1, len(cow_df))
        val_lying = round(float(cow_df[lying_col].sum() * scale_day), 1) if lying_col in cow_df else 600.0
        base_lying = round(float(base_df[lying_col].mean() * 24.0), 1) if not base_df.empty and lying_col in base_df else 600.0
        delta_lying = round(((val_lying - base_lying) / max(1.0, base_lying)) * 100.0, 1)

        val_accel = round(float(cow_df[accel_col].mean()), 4) if accel_col in cow_df else 0.12
        base_accel = round(float(base_df[accel_col].mean()), 4) if not base_df.empty and accel_col in base_df else 0.12
        delta_accel = round(((val_accel - base_accel) / max(0.01, base_accel)) * 100.0, 1)

        metrics: Dict[str, MetricItem] = {
            "core_body_temperature": MetricItem(value=val_temp, unit="°C", baseline=base_temp, delta_pct=delta_temp),
            "activity_index": MetricItem(value=val_act, unit="pts", baseline=base_act, delta_pct=delta_act),
            "lying_duration": MetricItem(value=val_lying, unit="min/day", baseline=base_lying, delta_pct=delta_lying),
            "acceleration_rms": MetricItem(value=val_accel, unit="g", baseline=base_accel, delta_pct=delta_accel),
        }

        # Anomaly logic
        is_fever = (val_temp >= 39.2) or (delta_temp >= 2.0)
        is_lethargic = (delta_act <= -20.0) or (delta_lying >= 20.0)
        is_restless = (delta_act >= 35.0) or (delta_lying <= -25.0)

        has_anomaly = is_fever or is_lethargic or is_restless

        summary_parts = []
        if is_fever:
            summary_parts.append(f"Core body temp elevated to {val_temp:.2f}°C (baseline {base_temp:.2f}°C, delta {delta_temp:+.1f}%)")
        if delta_act <= -20.0:
            summary_parts.append(f"Activity index suppressed by {delta_act:.1f}% to {val_act:.1f}")
        elif delta_act >= 35.0:
            summary_parts.append(f"Activity index increased by {delta_act:+.1f}% to {val_act:.1f}")
        if abs(delta_lying) >= 18.0:
            summary_parts.append(f"Lying time altered by {delta_lying:+.1f}% ({val_lying:.1f} min/day)")

        anomaly_summary = "; ".join(summary_parts) if summary_parts else "All behavioral and physiological metrics within baseline bounds."

        # Severity score 0.0 - 1.0
        severity = 0.0
        if is_fever:
            severity += 0.45
        if abs(delta_act) >= 25.0:
            severity += 0.30
        if abs(delta_lying) >= 25.0:
            severity += 0.25
        severity = min(1.0, round(severity, 2))

        return {
            "has_anomaly": has_anomaly,
            "is_fever": is_fever,
            "is_lethargic": is_lethargic,
            "is_restless": is_restless,
            "severity_score": severity,
            "metrics": metrics,
            "anomaly_summary": anomaly_summary,
            "cow_id": cow_id,
            "start_dt": start_dt.isoformat(),
            "end_dt": end_dt.isoformat(),
        }

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
        extra: Optional[Dict[str, Any]] = None
    ) -> AgentRequest:
        """Constructs an AgentRequest complying strictly with section A.3."""
        baseline_window = None
        if baseline_start_iso and baseline_end_iso:
            baseline_window = TimeWindow(start=baseline_start_iso, end=baseline_end_iso)

        return AgentRequest(
            query_id=str(uuid.uuid4()),
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
        specialist_responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """
        Integrates all specialist evidence and computes the final risk assessment
        and prioritized veterinary decision-support guidance.
        """
        # Group responses by agent
        resp_by_agent: Dict[str, AgentResponse] = {r.from_agent: r for r in specialist_responses}

        env_resp = resp_by_agent.get("environment_welfare")
        feed_resp = resp_by_agent.get("feeding_nutrition")
        prod_resp = resp_by_agent.get("production")

        is_fever = anomaly_data.get("is_fever", False)
        is_lethargic = anomaly_data.get("is_lethargic", False)

        env_supports = env_resp and env_resp.finding == "supports"
        is_herd_heat_stress = bool(env_supports and env_resp.herd_context.is_herd_wide)

        feed_supports = feed_resp and feed_resp.finding == "supports"
        prod_supports = prod_resp and prod_resp.finding == "supports"

        # Risk level determination
        risk_level = "Low"
        confidence_scores = [r.confidence for r in specialist_responses if r.confidence > 0]
        mean_confidence = round(float(np.mean(confidence_scores)), 2) if confidence_scores else 0.80

        actions: List[str] = []
        possible_conditions: List[str] = []

        if is_herd_heat_stress:
            risk_level = "High" if (feed_supports or prod_supports) else "Moderate"
            summary_statement = (
                f"Evaluation indicates herd-wide microclimate thermal strain (THI {env_resp.metrics.get('thi', MetricItem(value=0, baseline=0, delta_pct=0)).value:.1f}) "
                f"affecting {env_resp.herd_context.fraction_deviating*100:.0f}% of cows analyzed. "
                f"Findings are consistent with acute heat stress rather than isolated infectious disease."
            )
            possible_conditions.append("Herd-wide Heat Stress / Hyperthermia")
            actions.append("Activate high-volume barn ventilation fans and evaporative soaker cycles.")
            actions.append("Ensure unobstructed access to chilled, clean drinking water troughs.")
            actions.append("Shift TMR feeding timing to cooler evening hours to mitigate ruminal heat load.")

        elif is_fever and (feed_supports or prod_supports):
            risk_level = "Critical" if (feed_supports and prod_supports) else "High"
            summary_statement = (
                f"Isolated acute physiological anomaly in {cow_id} with core temperature elevation "
                f"unsupported by microclimate (THI normal). Concurrently supported by significant "
                f"{'rumination depression ' if feed_supports else ''}{'and production drop ' if prod_supports else ''}. "
                f"Findings are consistent with acute individual systemic or localized inflammation."
            )
            possible_conditions.append("Clinical Mastitis (supported by elevated conductivity & yield collapse)")
            possible_conditions.append("Acute Systemic Infection / Metritis / Foot Rot")
            actions.append("Immediate physical veterinary examination (rectal temp, heart rate, rumen auscultation).")
            actions.append("Perform California Mastitis Test (CMT) on all four quarters and inspect milk secretions.")
            actions.append("Isolate animal in hospital pen with clean deep straw bedding.")

        elif feed_supports and not is_fever:
            risk_level = "Moderate"
            summary_statement = (
                f"Individual nutritional depression in {cow_id} with rumination drop without febrile response. "
                f"Findings are consistent with early metabolic disturbance or digestive discomfort."
            )
            possible_conditions.append("Subclinical Ketosis / Off-Feed Indigestion")
            possible_conditions.append("Left Displaced Abomasum (LDA)")
            actions.append("Check blood/milk beta-hydroxybutyrate (BHB) for ketosis.")
            actions.append("Perform left flank percussion/auscultation for abomasal ping.")
            actions.append("Assess oral cud chewing and provide oral propylene glycol if indicated.")

        elif not anomaly_data.get("has_anomaly", False):
            risk_level = "Low"
            summary_statement = (
                f"Cow {cow_id} displays behavioral, nutritional, and production metrics consistent with its established individual baseline. "
                f"No veterinary intervention indicated."
            )
            actions.append("Continue routine automated monitoring.")

        else:
            risk_level = "Moderate"
            summary_statement = (
                f"Mild behavioral deviation observed in {cow_id}. Specialist telemetry does not demonstrate acute systemic failure. "
                f"Findings suggest transient stress or early onset requiring close watch."
            )
            actions.append("Re-evaluate telemetry in 12 hours.")
            actions.append("Visual check by pen rider during next feed push-up.")

        assessment = {
            "cow_id": cow_id,
            "generated_at": datetime.now().isoformat(),
            "risk_level": risk_level,
            "confidence": mean_confidence,
            "summary_statement": summary_statement,
            "possible_conditions": possible_conditions,
            "recommended_actions": actions,
            "anomaly_metrics": anomaly_data.get("metrics", {}),
            "specialist_findings": [r.model_dump() for r in specialist_responses],
        }

        return assessment
