"""
Feeding & Nutrition Specialist Agent.
Accent: Amber (#d97706)
Interprets feeding duration, frequency, rumination, intake, and bunk availability relative to individual baseline.
Adheres strictly to section A.3 shared contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.agents.base import BaseSpecialistAgent
from src.core.models import (
    AgentRequest,
    AgentResponse,
    MetricItem,
    HerdContext,
    DataQuality,
    SuggestedFollowup,
)
from src.data.schema import CANONICAL_COLUMNS


class FeedingNutritionAgent(BaseSpecialistAgent):
    """
    Evaluates feeding behavior, rumination stability, and dry matter intake against
    the cow's historical baseline, separating metabolic/health anorexia from bunk management delays.
    """

    def __init__(self, repo=None, use_llm: bool = False):
        super().__init__(
            agent_id="feeding_nutrition",
            accent_color="#d97706",
            repo=repo,
            use_llm=use_llm
        )

    def _analyze(
        self,
        request: AgentRequest,
        anomaly_df: pd.DataFrame,
        baseline_df: pd.DataFrame,
        start_dt: datetime,
        end_dt: datetime
    ) -> AgentResponse:
        feed_col = CANONICAL_COLUMNS["FEEDING_DURATION_MIN"]
        rum_col = CANONICAL_COLUMNS["RUMINATION_MIN"]
        intake_col = CANONICAL_COLUMNS["ESTIMATED_INTAKE_KG"]
        visits_col = CANONICAL_COLUMNS["FEED_BUNK_VISITS"]
        avail_col = CANONICAL_COLUMNS["FEED_AVAILABILITY_PCT"]

        # Data quality checks
        total_expected_records = max(1, int((end_dt - start_dt).total_seconds() / 3600))
        actual_records = len(anomaly_df)
        coverage_pct = round(min(100.0, (actual_records / total_expected_records) * 100.0), 1)

        missing_streams = []
        for col_name, label in [(feed_col, "feeding_duration"), (rum_col, "rumination"), (intake_col, "intake")]:
            if col_name not in anomaly_df.columns or anomaly_df[col_name].isna().all():
                missing_streams.append(label)

        if coverage_pct < 40.0 or len(missing_streams) >= 2:
            return AgentResponse(
                query_id=request.query_id,
                from_agent=self.agent_id,
                to_agent=request.from_agent,
                cow_id=request.cow_id,
                finding="inconclusive",
                summary=f"Feeding and rumination telemetry coverage is insufficient ({coverage_pct}%) to evaluate nutritional patterns.",
                metrics={},
                herd_context=HerdContext(),
                evidence_strength="none",
                confidence=0.20,
                data_quality=DataQuality(
                    coverage_pct=coverage_pct,
                    missing_streams=missing_streams,
                    notes="Sensors offline or missing streams."
                ),
                limitations=["Sparse rumination/feeding sensor data."],
                suggested_followups=[]
            )

        # Scale hourly data to 24h equivalent for intuitive dairy metrics
        hours_in_window = max(1, len(anomaly_df))
        scale_to_day = 24.0 / hours_in_window

        val_feed = round(float(anomaly_df[feed_col].sum() * scale_to_day), 1) if feed_col in anomaly_df else 0.0
        base_feed = round(float(baseline_df[feed_col].mean() * 24.0), 1) if not baseline_df.empty and feed_col in baseline_df else val_feed
        delta_feed = round(((val_feed - base_feed) / max(1.0, base_feed)) * 100.0, 1)

        val_rum = round(float(anomaly_df[rum_col].sum() * scale_to_day), 1) if rum_col in anomaly_df else 0.0
        base_rum = round(float(baseline_df[rum_col].mean() * 24.0), 1) if not baseline_df.empty and rum_col in baseline_df else val_rum
        delta_rum = round(((val_rum - base_rum) / max(1.0, base_rum)) * 100.0, 1)

        val_intake = round(float(anomaly_df[intake_col].sum() * scale_to_day), 1) if intake_col in anomaly_df else 0.0
        base_intake = round(float(baseline_df[intake_col].mean() * 24.0), 1) if not baseline_df.empty and intake_col in baseline_df else val_intake
        delta_intake = round(((val_intake - base_intake) / max(1.0, base_intake)) * 100.0, 1)

        val_visits = round(float(anomaly_df[visits_col].sum() * scale_to_day), 0) if visits_col in anomaly_df else 0.0
        base_visits = round(float(baseline_df[visits_col].mean() * 24.0), 0) if not baseline_df.empty and visits_col in baseline_df else val_visits
        delta_visits = round(((val_visits - base_visits) / max(1.0, base_visits)) * 100.0, 1)

        feed_avail_mean = round(float(anomaly_df[avail_col].mean()), 1) if avail_col in anomaly_df else 85.0

        metrics: Dict[str, MetricItem] = {
            "feeding_duration": MetricItem(value=val_feed, unit="min/day", baseline=base_feed, delta_pct=delta_feed),
            "rumination_duration": MetricItem(value=val_rum, unit="min/day", baseline=base_rum, delta_pct=delta_rum),
            "estimated_intake": MetricItem(value=val_intake, unit="kg/day", baseline=base_intake, delta_pct=delta_intake),
            "bunk_visits": MetricItem(value=val_visits, unit="visits/day", baseline=base_visits, delta_pct=delta_visits),
            "feed_availability": MetricItem(value=feed_avail_mean, unit="%", baseline=85.0, delta_pct=round(((feed_avail_mean - 85.0) / 85.0) * 100.0, 1)),
        }

        # Herd context evaluation
        herd_df = self.repo.get_herd_data(start_time=start_dt, end_time=end_dt)
        cow_col = CANONICAL_COLUMNS["COW_ID"]
        all_cows = herd_df[cow_col].unique().tolist()
        cows_analyzed = len(all_cows)

        cows_deviating = 0
        for c_id in all_cows:
            c_data = herd_df[herd_df[cow_col] == c_id]
            if not c_data.empty and rum_col in c_data:
                c_rum_equiv = c_data[rum_col].sum() * scale_to_day
                # Derive cow rough baseline
                if c_rum_equiv < 350.0:  # significant depression
                    cows_deviating += 1

        fraction_deviating = round(float(cows_deviating / max(1, cows_analyzed)), 2)
        is_herd_wide = bool(fraction_deviating >= 0.35)

        suggested_followups: List[SuggestedFollowup] = []
        limitations: List[str] = []

        # Decision logic
        if delta_rum <= -20.0 or delta_intake <= -25.0:
            finding = "supports"
            evidence_strength = "strong" if delta_rum <= -30.0 else "moderate"
            confidence = 0.89

            if feed_avail_mean < 40.0:
                summary = (
                    f"Feeding duration dropped by {delta_feed:.1f}% to {val_feed:.1f} min/day and intake dropped by {delta_intake:.1f}% to {val_intake:.1f} kg/day, "
                    f"correlated with restricted feed bunk availability ({feed_avail_mean:.1f}%); plausible management delivery delay."
                )
            else:
                summary = (
                    f"Rumination decreased by {delta_rum:.1f}% to {val_rum:.1f} min/day (baseline {base_rum:.1f} min/day) "
                    f"and intake fell by {delta_intake:.1f}% to {val_intake:.1f} kg/day with normal bunk availability ({feed_avail_mean:.1f}%), "
                    f"consistent with acute metabolic depression or systemic disease."
                )
                suggested_followups.append(
                    SuggestedFollowup(to_agent="production", reason="Check if milk yield has dropped concurrently with rumination collapse.")
                )
        elif delta_rum >= 15.0 or delta_feed >= 15.0:
            finding = "does_not_support"
            evidence_strength = "moderate"
            confidence = 0.82
            summary = (
                f"Rumination ({val_rum:.1f} min/day, delta {delta_rum:+.1f}%) and feeding ({val_feed:.1f} min/day, delta {delta_feed:+.1f}%) "
                f"are elevated or stable; nutritional depression does not support the suspected malaise."
            )
        else:
            finding = "does_not_support"
            evidence_strength = "strong"
            confidence = 0.86
            summary = (
                f"Rumination time ({val_rum:.1f} min/day, baseline {base_rum:.1f} min/day, delta {delta_rum:+.1f}%) "
                f"and intake ({val_intake:.1f} kg/day, baseline {base_intake:.1f} kg/day, delta {delta_intake:+.1f}%) "
                f"are consistent with normal physiological range."
            )

        if is_herd_wide:
            limitations.append(f"Similar nutritional depressions observed in {fraction_deviating*100:.0f}% of herd cows.")
        limitations.append("Intake is estimated via bunk visit duration and standard intake rate equations.")

        return AgentResponse(
            query_id=request.query_id,
            from_agent=self.agent_id,
            to_agent=request.from_agent,
            cow_id=request.cow_id,
            finding=finding,
            summary=summary,
            metrics=metrics,
            herd_context=HerdContext(
                cows_analyzed=cows_analyzed,
                cows_deviating=cows_deviating,
                fraction_deviating=fraction_deviating,
                is_herd_wide=is_herd_wide
            ),
            evidence_strength=evidence_strength,
            confidence=confidence,
            data_quality=DataQuality(
                coverage_pct=coverage_pct,
                missing_streams=missing_streams,
                notes="Feeding and rumination streams synchronized."
            ),
            limitations=limitations,
            suggested_followups=suggested_followups
        )
