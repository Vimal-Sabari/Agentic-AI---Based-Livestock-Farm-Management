"""
Unit tests validating section A.3 shared interface contract,
Pydantic v2 models, error handling, and deterministic baseline calculations.
"""

import pytest
from datetime import datetime
from src.core.models import AgentRequest, AgentResponse, TimeWindow, MetricItem
from src.agents.environment_welfare import EnvironmentWelfareAgent
from src.agents.feeding_nutrition import FeedingNutritionAgent
from src.agents.production import ProductionAgent
from src.agents.health_behavior import HealthBehaviorAgent
from src.data.repository import get_repository


@pytest.fixture
def repo():
    return get_repository("data/mmcows_synthetic.csv")


def test_agent_request_model():
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="environment_welfare",
        cow_id="cow_07",
        question="Could environmental conditions explain the change?",
        question_type="environment_check",
        anomaly_window=TimeWindow(start="2026-09-11T00:00:00Z", end="2026-09-12T00:00:00Z"),
        context={"anomaly_summary": "Activity -27%, lying time up", "extra": {}}
    )
    assert req.query_id is not None
    assert req.cow_id == "cow_07"
    assert req.anomaly_window.start == "2026-09-11T00:00:00Z"


def test_environment_agent_contract(repo):
    agent = EnvironmentWelfareAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="environment_welfare",
        cow_id="cow_07",
        question="Could environmental conditions explain the change?",
        question_type="environment_check",
        anomaly_window=TimeWindow(start="2026-09-11T00:00:00Z", end="2026-09-12T00:00:00Z"),
        context={"anomaly_summary": "Activity -27%, lying time up"}
    )
    resp = agent.handle_query(req)
    assert isinstance(resp, AgentResponse)
    assert resp.query_id == req.query_id
    assert resp.cow_id == "cow_07"
    assert resp.from_agent == "environment_welfare"
    assert resp.to_agent == "health_behavior"
    assert resp.finding in ["supports", "does_not_support", "inconclusive"]
    assert resp.evidence_strength in ["strong", "moderate", "weak", "none"]
    assert 0.0 <= resp.confidence <= 1.0
    assert "thi" in resp.metrics
    # Numbers in summary must be present in metrics
    thi_val = resp.metrics["thi"].value
    assert f"{thi_val:.1f}" in resp.summary


def test_feeding_agent_contract(repo):
    agent = FeedingNutritionAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_07",
        question="Has rumination or intake dropped?",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-11T00:00:00Z", end="2026-09-12T00:00:00Z"),
        context={"anomaly_summary": "Activity -27%, lying time up"}
    )
    resp = agent.handle_query(req)
    assert isinstance(resp, AgentResponse)
    assert resp.cow_id == "cow_07"
    assert "rumination_duration" in resp.metrics
    assert "feeding_duration" in resp.metrics


def test_production_agent_contract(repo):
    agent = ProductionAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="production",
        cow_id="cow_07",
        question="Has milk yield dropped?",
        question_type="yield_impact_check",
        anomaly_window=TimeWindow(start="2026-09-11T00:00:00Z", end="2026-09-12T00:00:00Z"),
        context={"anomaly_summary": "Activity -27%, lying time up"}
    )
    resp = agent.handle_query(req)
    assert isinstance(resp, AgentResponse)
    assert resp.cow_id == "cow_07"
    assert "daily_milk_yield" in resp.metrics


def test_unknown_cow_handling(repo):
    agent = EnvironmentWelfareAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="environment_welfare",
        cow_id="cow_unknown_999",
        question="Check environment",
        question_type="environment_check",
        anomaly_window=TimeWindow(start="2026-09-11T00:00:00Z", end="2026-09-12T00:00:00Z"),
    )
    resp = agent.handle_query(req)
    assert resp.finding == "inconclusive"
    assert resp.confidence == 0.0
    assert "not found" in resp.summary.lower()


def test_empty_window_handling(repo):
    agent = FeedingNutritionAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_07",
        question="Check feeding",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2010-01-01T00:00:00Z", end="2010-01-02T00:00:00Z"),
    )
    resp = agent.handle_query(req)
    assert resp.finding == "inconclusive"
    assert resp.confidence == 0.0
    assert "no telemetry" in resp.summary.lower()


def test_heat_stress_herd_wide_detection(repo):
    # Days 5-6 (2026-09-06 to 2026-09-07) was the synthetic heatwave
    agent = EnvironmentWelfareAgent(repo=repo)
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="environment_welfare",
        cow_id="cow_03",
        question="Could heat stress explain behavioral shift?",
        question_type="heat_stress_check",
        anomaly_window=TimeWindow(start="2026-09-06T10:00:00Z", end="2026-09-06T18:00:00Z"),
    )
    resp = agent.handle_query(req)
    assert resp.finding == "supports"
    assert resp.herd_context.is_herd_wide is True
    assert resp.metrics["thi"].value >= 72.0
