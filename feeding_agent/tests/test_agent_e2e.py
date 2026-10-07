"""
End-to-end integration and scenario tests for Feeding & Nutrition Agent.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd
import pytest

from src.core.models import AgentRequest, AgentResponse, TimeWindow
from feeding_agent.src.feeding_agent.adapters.mmcows import load_mmcows
from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.agent import FeedingNutritionAgent


@pytest.mark.parametrize(
    "scenario",
    [
        "stable",
        "cow_specific_drop",
        "herd_wide_drop",
        "rumination_only_drop",
        "restricted_availability",
        "missing_rumination",
        "short_baseline",
        "sparse_coverage",
        "abnormal_increase",
    ],
)
def test_e2e_all_synthetic_scenarios(scenario):
    ts = generate_synthetic(scenario=scenario, seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Analyze feeding and rumination changes.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert isinstance(resp, AgentResponse)
    assert resp.finding in ["supports", "does_not_support", "inconclusive"]
    assert 0.0 <= resp.confidence <= 1.0

    # Verify every number appearing in summary exists in metrics (same rounding)
    allowed_numbers = set()
    for m in resp.metrics.values():
        allowed_numbers.add(f"{m.value:.0f}")
        allowed_numbers.add(f"{m.baseline:.0f}")
        allowed_numbers.add(f"{abs(m.delta_pct):.0f}")
        allowed_numbers.add(f"{m.value:.1f}")
        allowed_numbers.add(f"{m.baseline:.1f}")
        allowed_numbers.add(f"{abs(m.delta_pct):.1f}")

    # Remove cow ID (e.g. cow_01) before number extraction
    clean_summary = re.sub(r"cow_\d+", "", resp.summary)
    summary_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clean_summary)

    for num_str in summary_numbers:
        assert any(
            num_str in [
                f"{m.value:.0f}",
                f"{m.baseline:.0f}",
                f"{abs(m.delta_pct):.0f}",
                f"{m.value:.1f}",
                f"{m.baseline:.1f}",
                f"{abs(m.delta_pct):.1f}",
            ]
            for m in resp.metrics.values()
        )


def test_scenario_cow_specific_drop():
    ts = generate_synthetic(scenario="cow_specific_drop", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check for feeding drop.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert resp.finding == "supports"
    assert not resp.herd_context.is_herd_wide

    target_agents = [sf.to_agent for sf in resp.suggested_followups]
    assert "production" in target_agents
    assert "health_behavior" in target_agents


def test_scenario_herd_wide_drop():
    ts = generate_synthetic(scenario="herd_wide_drop", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check for herd drop.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert resp.finding == "supports"
    assert resp.herd_context.is_herd_wide

    target_agents = [sf.to_agent for sf in resp.suggested_followups]
    assert "environment_welfare" in target_agents


def test_scenario_stable():
    ts = generate_synthetic(scenario="stable", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check for feeding drop.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert resp.finding == "does_not_support"


def test_scenario_abnormal_increase():
    ts = generate_synthetic(scenario="abnormal_increase", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check for feeding increase.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert resp.finding == "supports"
    assert resp.metrics["feeding_duration"].delta_pct > 0


def test_short_baseline_lower_confidence():
    # Compare confidence on short_baseline vs stable
    ts_short = generate_synthetic(scenario="short_baseline", seed=42)
    agent_short = FeedingNutritionAgent(ts=ts_short)

    ts_stable = generate_synthetic(scenario="stable", seed=42)
    agent_stable = FeedingNutritionAgent(ts=ts_stable)

    anom_start_short = ts_short.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end_short = ts_short.df["timestamp"].max()

    anom_start_stable = ts_stable.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end_stable = ts_stable.df["timestamp"].max()

    req_short = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check feeding.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start_short.isoformat(), end=anom_end_short.isoformat()),
    )

    req_stable = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check feeding.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start_stable.isoformat(), end=anom_end_stable.isoformat()),
    )

    resp_short = agent_short.handle_query(req_short)
    resp_stable = agent_stable.handle_query(req_stable)

    # Short baseline applies penalty, reducing confidence
    assert resp_short.confidence < resp_stable.confidence
    assert any("short" in lim.lower() for lim in resp_short.limitations)


def test_sparse_coverage_honest_data_quality():
    ts = generate_synthetic(scenario="sparse_coverage", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check feeding with sparse coverage.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert resp.data_quality.coverage_pct <= 60.0
    if resp.data_quality.coverage_pct < 40.0:
        assert resp.finding == "inconclusive"
        assert resp.confidence <= 0.25


def test_missing_rumination_honest_data_quality():
    ts = generate_synthetic(scenario="missing_rumination", seed=42)
    agent = FeedingNutritionAgent(ts=ts)

    anom_start = ts.df["timestamp"].max() - pd.Timedelta(days=2)
    anom_end = ts.df["timestamp"].max()

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Check feeding with missing rumination.",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start=anom_start.isoformat(), end=anom_end.isoformat()),
    )

    resp = agent.handle_query(req)
    assert "rumination" in resp.data_quality.missing_streams
    assert "rumination_duration" not in resp.metrics
    assert any("rumination" in lim.lower() for lim in resp.limitations)


def test_edge_cases_no_crash(stable_ts):
    agent = FeedingNutritionAgent(ts=stable_ts)

    # 1. Unknown cow
    req_unknown = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_9999",
        question="Unknown cow test",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-14T00:00:00Z"),
    )
    resp1 = agent.handle_query(req_unknown)
    assert resp1.finding == "inconclusive"
    assert resp1.confidence == 0.0

    # 2. Empty window (start equals end)
    req_empty = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Empty window test",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-13T00:00:00Z"),
    )
    resp_empty = agent.handle_query(req_empty)
    assert resp_empty.finding == "inconclusive"

    # 3. Inverted window
    req_inv = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Inverted window test",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-14T00:00:00Z", end="2026-09-13T00:00:00Z"),
    )
    resp2 = agent.handle_query(req_inv)
    assert resp2.finding == "inconclusive"

    # 4. Unsupported question type
    req_unsupported = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Unsupported type test",
        question_type="random_unsupported_type",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-14T00:00:00Z"),
    )
    resp3 = agent.handle_query(req_unsupported)
    assert resp3.finding == "inconclusive"

    # 5. Window entirely outside data range
    req_outside = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Outside range test",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2030-01-01T00:00:00Z", end="2030-01-02T00:00:00Z"),
    )
    resp4 = agent.handle_query(req_outside)
    assert resp4.finding == "inconclusive"


def test_langgraph_node_wrapper(stable_ts):
    agent = FeedingNutritionAgent(ts=stable_ts)
    node_fn = agent.as_langgraph_node()

    # Case A: Request addressed to feeding_nutrition
    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="LangGraph node test",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-14T23:00:00Z"),
    )

    state = {
        "cow_id": "cow_01",
        "pending_request": req.model_dump(),
        "specialist_responses": [],
    }

    new_state = node_fn(state)
    assert new_state["pending_request"] is None
    assert len(new_state["specialist_responses"]) == 1
    assert new_state["latest_response"]["from_agent"] == "feeding_nutrition"

    # Case B: Request addressed to another specialist (no-op)
    other_req = AgentRequest(
        from_agent="health_behavior",
        to_agent="production",
        cow_id="cow_01",
        question="Other agent test",
        question_type="yield_impact_check",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-14T23:00:00Z"),
    )

    state2 = {
        "pending_request": other_req.model_dump(),
        "specialist_responses": [],
    }
    new_state2 = node_fn(state2)
    assert new_state2["pending_request"] == other_req.model_dump()
    assert len(new_state2["specialist_responses"]) == 0

    # Case C: Missing pending_request (no-op)
    state3 = {"cow_id": "cow_01"}
    new_state3 = node_fn(state3)
    assert new_state3 == state3


def test_real_dataset_contract_execution():
    csv_path = Path("data/mmcows_synthetic.csv")
    if not csv_path.exists():
        pytest.skip("data/mmcows_synthetic.csv absent")

    ts = load_mmcows(csv_path)
    agent = FeedingNutritionAgent(ts=ts)

    req = AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Has rumination time or dry matter intake dropped relative to baseline?",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(start="2026-09-13T00:00:00Z", end="2026-09-14T23:00:00Z"),
        baseline_window=None,
    )

    resp = agent.handle_query(req)
    assert isinstance(resp, AgentResponse)
    assert resp.from_agent == "feeding_nutrition"
    assert resp.to_agent == "health_behavior"
    assert resp.cow_id == "cow_01"
    assert resp.finding in ["supports", "does_not_support", "inconclusive"]
