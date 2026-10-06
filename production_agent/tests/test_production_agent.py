"""
Tests for the Production Agent.
"""
from datetime import datetime, timezone
import pytest

from src.core.models import AgentRequest, TimeWindow
from production_agent.src.production_agent.adapters.synthetic import generate_synthetic
from production_agent.src.production_agent.agent import ProductionAgent

@pytest.fixture
def stable_data():
    return generate_synthetic(scenario="stable")

@pytest.fixture
def specific_drop_data():
    return generate_synthetic(scenario="cow_specific_drop")

@pytest.fixture
def herd_drop_data():
    return generate_synthetic(scenario="herd_wide_drop")

def _make_request(cow_id: str, start: str, end: str) -> AgentRequest:
    return AgentRequest(
        from_agent="health_behavior",
        to_agent="production",
        cow_id=cow_id,
        question="Check production",
        question_type="production_check",
        anomaly_window=TimeWindow(start=start, end=end)
    )

def test_baseline_and_deviation(specific_drop_data):
    agent = ProductionAgent(specific_drop_data)
    # Cow 1 has a drop in the last 2 days. 14 days total, so days 12-14
    req = _make_request("cow_01", "2026-09-13T00:00:00Z", "2026-09-15T00:00:00Z")
    resp = agent.handle_query(req)
    
    assert resp.finding == "supports"
    assert resp.metrics["mean_yield_daily"].delta_pct <= -10.0
    assert not resp.herd_context.is_herd_wide
    assert len(resp.suggested_followups) == 2 # health and feeding

def test_herd_wide_drop(herd_drop_data):
    agent = ProductionAgent(herd_drop_data)
    # Assuming cow_03 (odd index) dropped
    req = _make_request("cow_03", "2026-09-13T00:00:00Z", "2026-09-15T00:00:00Z")
    resp = agent.handle_query(req)
    
    assert resp.finding == "supports"
    assert resp.herd_context.is_herd_wide
    assert "environment_welfare" in [f.to_agent for f in resp.suggested_followups]

def test_stable_cow(stable_data):
    agent = ProductionAgent(stable_data)
    req = _make_request("cow_01", "2026-09-13T00:00:00Z", "2026-09-15T00:00:00Z")
    resp = agent.handle_query(req)
    
    assert resp.finding == "does_not_support"
    assert resp.metrics["days_persisted"].value == 0

def test_unsupported_question_type(stable_data):
    agent = ProductionAgent(stable_data)
    req = _make_request("cow_01", "2026-09-13T00:00:00Z", "2026-09-15T00:00:00Z")
    req.question_type = "invalid_type"
    resp = agent.handle_query(req)
    
    assert resp.finding == "inconclusive"
    assert "Unsupported question_type" in resp.summary
