"""
Unit tests for LangGraph multi-agent orchestration workflow.
"""

import pytest
from src.orchestration.graph import build_investigation_graph
from src.data.repository import get_repository


@pytest.fixture
def repo():
    return get_repository("data/mmcows_synthetic.csv")


def test_full_mastitis_investigation_flow(repo):
    graph = build_investigation_graph(repo=repo)
    state_input = {
        "cow_id": "cow_07",
        "start_iso": "2026-09-11T00:00:00Z",
        "end_iso": "2026-09-12T00:00:00Z",
    }
    result = graph.invoke(state_input)

    assert result["investigation_complete"] is True
    assert "anomaly_data" in result
    assert result["anomaly_data"]["has_anomaly"] is True

    # Check that specialists were consulted
    consulted = result["consulted_agents"]
    assert "environment_welfare" in consulted
    assert "feeding_nutrition" in consulted
    assert "production" in consulted

    # Check final assessment structure
    assessment = result["final_assessment"]
    assert assessment["cow_id"] == "cow_07"
    assert assessment["risk_level"] in ["High", "Critical"]
    assert assessment["confidence"] > 0.70
    assert len(assessment["recommended_actions"]) >= 2
    assert "consistent with" in assessment["summary_statement"].lower()


def test_healthy_cow_investigation_flow(repo):
    graph = build_investigation_graph(repo=repo)
    state_input = {
        "cow_id": "cow_01",
        "start_iso": "2026-09-03T00:00:00Z",
        "end_iso": "2026-09-04T00:00:00Z",
    }
    result = graph.invoke(state_input)

    assert result["investigation_complete"] is True
    assessment = result["final_assessment"]
    assert assessment["risk_level"] == "Low"
