"""
Comprehensive Unit Tests for Health & Behavior Lead Agent.
Validates baseline derivation, Z-score anomaly detection, query planning, and veterinary synthesis.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone

from health_agent.src.health_agent import (
    HealthAgentConfig,
    derive_health_baseline,
    detect_health_anomalies,
    plan_investigation_queries,
)
from health_agent.src.health_agent.synthesis import synthesize_clinical_assessment
from src.core.models import AgentResponse, HerdContext, MetricItem, DataQuality
from src.data.repository import get_repository


@pytest.fixture
def repo():
    return get_repository("data/mmcows_synthetic.csv")


def test_baseline_derivation_with_spread(repo):
    """Verifies that baseline derives both mean and spread (std/MAD) with valid days."""
    anomaly_start = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
    baseline = derive_health_baseline(cow_id="cow_07", anomaly_start=anomaly_start, repo=repo)

    assert baseline.cow_id == "cow_07"
    assert 38.0 <= baseline.core_temp_c.mean <= 39.5
    assert baseline.core_temp_c.std > 0.0
    assert baseline.core_temp_c.mad >= 0.0
    assert baseline.daily_lying_min.mean > 300.0
    assert baseline.valid_baseline_days >= 3
    assert baseline.coverage_pct > 80.0


def test_anomaly_detection_mastitis_scenario(repo):
    """Tests fever and lethargy detection on cow_07 on day 11 (mastitis)."""
    start_dt = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)

    result = detect_health_anomalies(cow_id="cow_07", start_dt=start_dt, end_dt=end_dt, repo=repo)

    assert result.has_anomaly is True
    assert result.is_fever is True
    assert result.is_lethargic is True
    assert result.severity_score >= 0.50
    assert "core_body_temperature" in result.metrics
    assert result.metrics["core_body_temperature"].z_score > 1.5


def test_query_planning_tailoring(repo):
    """Tests that severe fever + lethargy dispatches all 3 specialist queries."""
    start_dt = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)
    anomaly = detect_health_anomalies(cow_id="cow_07", start_dt=start_dt, end_dt=end_dt, repo=repo)

    queries = plan_investigation_queries(anomaly)
    assert len(queries) == 3

    target_agents = {q.to_agent for q in queries}
    assert target_agents == {"environment_welfare", "feeding_nutrition", "production"}

    for q in queries:
        assert q.priority in ["high", "urgent"]
        assert q.timeout_s == 5.0
        assert "evidence for and against" in q.question
        assert q.cow_id == "cow_07"


def test_clinical_synthesis_mastitis(repo):
    """Tests that fever + rumination collapse + yield drop synthesizes into Critical Mastitis."""
    start_dt = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)
    anomaly = detect_health_anomalies(cow_id="cow_07", start_dt=start_dt, end_dt=end_dt, repo=repo)

    # Mock specialist responses for mastitis
    env_resp = AgentResponse(
        query_id="q1",
        from_agent="environment_welfare",
        to_agent="health_behavior",
        cow_id="cow_07",
        finding="does_not_support",
        summary="THI 64.2 within comfort zone; microclimate does not explain core temperature elevation.",
        metrics={"thi": MetricItem(value=64.2, unit="", baseline=63.8, delta_pct=0.6)},
        herd_context=HerdContext(cows_analyzed=16, cows_deviating=1, is_herd_wide=False),
        confidence=0.90
    )

    feed_resp = AgentResponse(
        query_id="q2",
        from_agent="feeding_nutrition",
        to_agent="health_behavior",
        cow_id="cow_07",
        finding="supports",
        summary="Rumination decreased by -41.7% with normal bunk availability.",
        metrics={"rumination_duration": MetricItem(value=280.0, unit="min/day", baseline=480.0, delta_pct=-41.7)},
        confidence=0.89
    )

    prod_resp = AgentResponse(
        query_id="q3",
        from_agent="production",
        to_agent="health_behavior",
        cow_id="cow_07",
        finding="supports",
        summary="Daily milk yield dropped -42.2% with elevated electrical conductivity.",
        metrics={"daily_milk_yield": MetricItem(value=18.5, unit="kg/day", baseline=32.0, delta_pct=-42.2)},
        confidence=0.92
    )

    assessment = synthesize_clinical_assessment(
        cow_id="cow_07",
        anomaly=anomaly,
        specialist_responses=[env_resp, feed_resp, prod_resp]
    )

    assert assessment["risk_level"] == "Critical"
    assert any("Mastitis" in cond for cond in assessment["possible_conditions"])
    assert any("California Mastitis Test" in act for act in assessment["recommended_actions"])
    assert assessment["confidence"] >= 0.85
