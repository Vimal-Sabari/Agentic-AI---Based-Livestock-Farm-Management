"""
Anomaly Detection Engine for Health & Behavior Lead Agent.
Compares observed window against individual cow baseline using both percentage deltas
and standardized Z-scores (accounting for individual physiological spread).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from health_agent.src.health_agent.config import HealthAgentConfig, HealthThresholdsConfig, SeverityWeightsConfig
from health_agent.src.health_agent.schemas import (
    HealthBaseline,
    HealthMetricComparison,
    AnomalyDetectionResult,
)
from health_agent.src.health_agent.baseline import derive_health_baseline
from src.data.schema import CANONICAL_COLUMNS
from src.data.repository import LivestockRepository, get_repository

logger = logging.getLogger(__name__)


def detect_health_anomalies(
    cow_id: str,
    start_dt: datetime,
    end_dt: datetime,
    baseline: Optional[HealthBaseline] = None,
    baseline_start_dt: Optional[datetime] = None,
    baseline_end_dt: Optional[datetime] = None,
    repo: Optional[LivestockRepository] = None,
    cfg: Optional[HealthAgentConfig] = None,
) -> AnomalyDetectionResult:
    """
    Evaluates individual cow telemetry within an anomaly window against her historical baseline.
    Computes both percentage deltas and individualized Z-scores to determine clinical flags.
    """
    repository = repo or get_repository()
    config = cfg or HealthAgentConfig.load()
    thresholds = config.thresholds
    weights = config.severity_weights

    # 1. Check if cow exists
    all_cows = repository.get_cows()
    if cow_id not in all_cows:
        return AnomalyDetectionResult(
            cow_id=cow_id,
            has_anomaly=False,
            is_fever=False,
            is_lethargic=False,
            is_restless=False,
            severity_score=0.0,
            metrics={},
            anomaly_summary=f"Cow '{cow_id}' was not found in the herd repository.",
            start_dt=start_dt.isoformat(),
            end_dt=end_dt.isoformat(),
            limitations=[f"Cow ID '{cow_id}' does not exist in dataset."]
        )

    # 2. Derive baseline if not provided
    if baseline is None:
        baseline = derive_health_baseline(
            cow_id=cow_id,
            anomaly_start=start_dt,
            baseline_start=baseline_start_dt,
            baseline_end=baseline_end_dt,
            repo=repository,
            cfg=config.baseline
        )

    # 3. Fetch anomaly window telemetry
    cow_df = repository.get_cow_data(cow_id, start_time=start_dt, end_time=end_dt)
    if cow_df.empty:
        return AnomalyDetectionResult(
            cow_id=cow_id,
            has_anomaly=False,
            is_fever=False,
            is_lethargic=False,
            is_restless=False,
            severity_score=0.0,
            metrics={},
            anomaly_summary=f"No telemetry records available for {cow_id} in specified window.",
            start_dt=start_dt.isoformat(),
            end_dt=end_dt.isoformat(),
            limitations=["Empty time window with zero sensor records."]
        )

    temp_col = CANONICAL_COLUMNS["CORE_BODY_TEMP_C"]
    act_col = CANONICAL_COLUMNS["ACTIVITY_INDEX"]
    lying_col = CANONICAL_COLUMNS["LYING_TIME_MIN"]
    accel_col = CANONICAL_COLUMNS["ACCELERATION_RMS"]

    # 4. Extract observed values
    val_temp = float(cow_df[temp_col].mean()) if temp_col in cow_df and not cow_df[temp_col].dropna().empty else baseline.core_temp_c.mean
    val_act = float(cow_df[act_col].mean()) if act_col in cow_df and not cow_df[act_col].dropna().empty else baseline.activity_index.mean
    val_accel = float(cow_df[accel_col].mean()) if accel_col in cow_df and not cow_df[accel_col].dropna().empty else baseline.accel_rms.mean

    # For lying duration: scale hourly sum to daily equivalent based on window length
    hours_in_window = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0)
    if lying_col in cow_df and not cow_df[lying_col].dropna().empty:
        hourly_sum = float(cow_df[lying_col].sum())
        val_lying = (hourly_sum / hours_in_window) * 24.0
    else:
        val_lying = baseline.daily_lying_min.mean

    # 5. Helper function to compute comparison metrics (delta % and Z-score)
    def make_comparison(val: float, base_mean: float, base_std: float, unit: str) -> HealthMetricComparison:
        eps = 0.001
        delta_pct = ((val - base_mean) / max(eps, abs(base_mean))) * 100.0
        # Biological noise floor: daily lying variation is naturally +/- 15 min; temp is 0.1°C; act is 1.0 pts
        min_std = 15.0 if "min" in unit else (1.0 if "pts" in unit else 0.1)
        effective_std = base_std if base_std >= min_std else min_std
        z_score = (val - base_mean) / effective_std
        return HealthMetricComparison(
            value=round(val, 2),
            baseline_mean=round(base_mean, 2),
            baseline_std=round(base_std, 3),
            delta_pct=round(delta_pct, 1),
            z_score=round(z_score, 2),
            unit=unit
        )

    temp_cmp = make_comparison(val_temp, baseline.core_temp_c.mean, baseline.core_temp_c.std, "°C")
    act_cmp = make_comparison(val_act, baseline.activity_index.mean, baseline.activity_index.std, "pts")
    lying_cmp = make_comparison(val_lying, baseline.daily_lying_min.mean, baseline.daily_lying_min.std, "min/day")
    accel_cmp = make_comparison(val_accel, baseline.accel_rms.mean, baseline.accel_rms.std, "g")

    metrics: Dict[str, HealthMetricComparison] = {
        "core_body_temperature": temp_cmp,
        "activity_index": act_cmp,
        "lying_duration": lying_cmp,
        "acceleration_rms": accel_cmp,
    }

    # 6. Evaluate clinical anomaly rules (using both delta % and Z-score)
    is_fever = bool(
        (val_temp >= thresholds.fever_temp_threshold_c) or
        (temp_cmp.delta_pct >= thresholds.fever_delta_pct) or
        (temp_cmp.z_score >= thresholds.fever_zscore_threshold)
    )

    is_lethargic = bool(
        (act_cmp.delta_pct <= thresholds.lethargy_act_delta_pct or act_cmp.z_score <= thresholds.lethargy_zscore_threshold) or
        (lying_cmp.delta_pct >= thresholds.lethargy_lying_delta_pct or lying_cmp.z_score >= thresholds.lying_zscore_threshold)
    )

    is_restless = bool(
        (act_cmp.delta_pct >= thresholds.restlessness_act_delta_pct or act_cmp.z_score >= thresholds.restlessness_zscore_threshold) or
        (lying_cmp.delta_pct <= thresholds.restlessness_lying_delta_pct or lying_cmp.z_score <= -thresholds.lying_zscore_threshold)
    )

    has_anomaly = is_fever or is_lethargic or is_restless

    # 7. Composite Severity Score Calculation (0.0 to 1.0)
    severity = 0.0
    if is_fever:
        temp_factor = min(1.0, max(0.5, temp_cmp.z_score / 3.0)) if temp_cmp.z_score > 0 else 0.8
        severity += weights.temp_weight * temp_factor
    if is_lethargic or is_restless:
        act_factor = min(1.0, abs(act_cmp.z_score) / 3.0) if abs(act_cmp.z_score) > 0 else 0.6
        severity += weights.activity_weight * act_factor
        lying_factor = min(1.0, abs(lying_cmp.z_score) / 3.0) if abs(lying_cmp.z_score) > 0 else 0.5
        severity += weights.lying_weight * lying_factor

    severity_score = min(1.0, round(severity, 2))

    # 8. Human-readable summary generation
    summary_parts: List[str] = []
    if is_fever:
        summary_parts.append(
            f"Core body temp elevated to {temp_cmp.value:.2f}°C (baseline {temp_cmp.baseline_mean:.2f}°C, delta {temp_cmp.delta_pct:+.1f}%, z = {temp_cmp.z_score:+.1f} SD)"
        )
    if act_cmp.delta_pct <= thresholds.lethargy_act_delta_pct or act_cmp.z_score <= thresholds.lethargy_zscore_threshold:
        summary_parts.append(
            f"Activity index suppressed by {act_cmp.delta_pct:.1f}% to {act_cmp.value:.1f} (baseline {act_cmp.baseline_mean:.1f}, z = {act_cmp.z_score:+.1f} SD)"
        )
    elif act_cmp.delta_pct >= thresholds.restlessness_act_delta_pct or act_cmp.z_score >= thresholds.restlessness_zscore_threshold:
        summary_parts.append(
            f"Activity index surged by {act_cmp.delta_pct:+.1f}% to {act_cmp.value:.1f} (z = {act_cmp.z_score:+.1f} SD)"
        )

    if abs(lying_cmp.delta_pct) >= 15.0 or abs(lying_cmp.z_score) >= 2.0:
        summary_parts.append(
            f"Lying time altered by {lying_cmp.delta_pct:+.1f}% ({lying_cmp.value:.1f} min/day vs baseline {lying_cmp.baseline_mean:.1f} min/day, z = {lying_cmp.z_score:+.1f} SD)"
        )

    anomaly_summary = "; ".join(summary_parts) if summary_parts else "All behavioral and physiological metrics within normal baseline bounds."

    limitations: List[str] = []
    if baseline.valid_baseline_days < 3:
        limitations.append(f"Baseline window is short ({baseline.valid_baseline_days} valid days).")

    return AnomalyDetectionResult(
        cow_id=cow_id,
        has_anomaly=has_anomaly,
        is_fever=is_fever,
        is_lethargic=is_lethargic,
        is_restless=is_restless,
        severity_score=severity_score,
        metrics=metrics,
        anomaly_summary=anomaly_summary,
        start_dt=start_dt.isoformat(),
        end_dt=end_dt.isoformat(),
        limitations=limitations
    )
