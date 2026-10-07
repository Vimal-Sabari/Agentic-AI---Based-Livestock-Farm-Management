"""
Data models and schemas for Health & Behavior Agent.
Includes statistical spread metrics (mean, std, median, MAD) and Z-score models.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BaselineMetric(BaseModel):
    """Statistical summary of an individual cow's metric during healthy baseline."""
    mean: float = Field(description="Mean baseline value")
    std: float = Field(default=0.0, description="Sample standard deviation (spread)")
    median: float = Field(default=0.0, description="Median baseline value")
    mad: float = Field(default=0.0, description="Median Absolute Deviation (robust spread)")
    n_samples: int = Field(default=0, description="Number of valid baseline observation points")


class HealthBaseline(BaseModel):
    """Comprehensive individualized baseline container for a specific cow."""
    cow_id: str
    core_temp_c: BaselineMetric
    activity_index: BaselineMetric
    daily_lying_min: BaselineMetric
    accel_rms: BaselineMetric
    valid_baseline_days: int = Field(default=0, description="Number of calendar days with >= 80% coverage")
    total_hours_analyzed: int = Field(default=0)
    coverage_pct: float = Field(default=100.0)


class HealthMetricComparison(BaseModel):
    """Detailed observation vs individual baseline comparison including Z-score."""
    value: float = Field(description="Observed value during anomaly window")
    baseline_mean: float = Field(description="Individual cow baseline mean")
    baseline_std: float = Field(default=0.0, description="Individual cow baseline standard deviation")
    delta_pct: float = Field(description="Percentage change relative to baseline mean")
    z_score: float = Field(default=0.0, description="Number of standard deviations away from baseline mean")
    unit: str = Field(default="")


class AnomalyDetectionResult(BaseModel):
    """Structured output of the anomaly detection phase."""
    cow_id: str
    has_anomaly: bool
    is_fever: bool
    is_lethargic: bool
    is_restless: bool
    severity_score: float = Field(ge=0.0, le=1.0, description="Composite anomaly severity between 0.0 and 1.0")
    metrics: Dict[str, HealthMetricComparison] = Field(default_factory=dict)
    anomaly_summary: str
    start_dt: str
    end_dt: str
    limitations: List[str] = Field(default_factory=list)
