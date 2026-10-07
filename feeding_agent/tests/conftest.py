"""
Pytest configuration and shared fixtures for feeding_agent test suite.
"""
import sys
from pathlib import Path
import pytest

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries
from src.core.models import AgentRequest, TimeWindow


@pytest.fixture
def stable_ts() -> FeedingTimeseries:
    return generate_synthetic(scenario="stable", seed=42)


@pytest.fixture
def cow_drop_ts() -> FeedingTimeseries:
    return generate_synthetic(scenario="cow_specific_drop", seed=42)


@pytest.fixture
def herd_drop_ts() -> FeedingTimeseries:
    return generate_synthetic(scenario="herd_wide_drop", seed=42)


@pytest.fixture
def sample_request() -> AgentRequest:
    return AgentRequest(
        from_agent="health_behavior",
        to_agent="feeding_nutrition",
        cow_id="cow_01",
        question="Has feeding or rumination dropped relative to baseline?",
        question_type="feeding_drop_check",
        anomaly_window=TimeWindow(
            start="2026-09-13T00:00:00Z",
            end="2026-09-14T23:00:00Z",
        ),
        baseline_window=None,
    )
