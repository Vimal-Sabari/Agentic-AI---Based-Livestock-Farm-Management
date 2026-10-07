"""
Health & Behavior Lead Investigator Agent Package.
"""
from __future__ import annotations

from health_agent.src.health_agent.config import HealthAgentConfig
from health_agent.src.health_agent.schemas import (
    BaselineMetric,
    HealthBaseline,
    HealthMetricComparison,
    AnomalyDetectionResult,
)
from health_agent.src.health_agent.baseline import derive_health_baseline
from health_agent.src.health_agent.anomaly import detect_health_anomalies
from health_agent.src.health_agent.query_builder import (
    plan_investigation_queries,
    build_followup_query,
)

# Backward compatibility alias
HealthAnomalyMetrics = HealthMetricComparison

__all__ = [
    "HealthAgentConfig",
    "BaselineMetric",
    "HealthBaseline",
    "HealthMetricComparison",
    "HealthAnomalyMetrics",
    "AnomalyDetectionResult",
    "derive_health_baseline",
    "detect_health_anomalies",
    "plan_investigation_queries",
    "build_followup_query",
]
