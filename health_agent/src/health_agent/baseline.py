"""
Individual Cow Baseline Computation Module.
Computes personalized central tendency (mean/median) and spread (standard deviation / MAD)
with calendar-day aggregation and coverage gating for lying behavior.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from health_agent.src.health_agent.config import HealthAgentConfig, BaselineConfig
from health_agent.src.health_agent.schemas import BaselineMetric, HealthBaseline
from src.data.schema import CANONICAL_COLUMNS
from src.data.repository import LivestockRepository, get_repository

logger = logging.getLogger(__name__)


def _compute_metric_stats(series: pd.Series) -> BaselineMetric:
    """Computes mean, sample standard deviation, median, and MAD for a numeric series."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return BaselineMetric(mean=0.0, std=0.0, median=0.0, mad=0.0, n_samples=0)

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    med_val = float(clean.median())

    # MAD (Median Absolute Deviation scaled for normal distribution consistency)
    abs_dev = np.abs(clean - med_val)
    mad_val = float(np.median(abs_dev) * 1.4826) if len(clean) > 0 else 0.0

    return BaselineMetric(
        mean=round(mean_val, 4),
        std=round(std_val, 4) if not np.isnan(std_val) else 0.0,
        median=round(med_val, 4),
        mad=round(mad_val, 4) if not np.isnan(mad_val) else 0.0,
        n_samples=len(clean)
    )


def derive_health_baseline(
    cow_id: str,
    anomaly_start: datetime,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
    repo: Optional[LivestockRepository] = None,
    cfg: Optional[BaselineConfig] = None,
) -> HealthBaseline:
    """
    Extracts historical telemetry for an individual cow and derives her baseline
    mean and spread (variance/MAD) across temperature, activity, lying time, and acceleration.
    """
    repository = repo or get_repository()
    config = cfg or HealthAgentConfig.load().baseline

    # Determine baseline time window
    if baseline_start and baseline_end:
        b_start = baseline_start
        b_end = baseline_end
    else:
        b_end = anomaly_start
        b_start = anomaly_start - timedelta(days=config.default_days_preceding)

    # Fetch cow telemetry
    raw_df = repository.get_cow_data(cow_id=cow_id, start_time=b_start, end_time=b_end)

    if raw_df.empty:
        # Graceful empty baseline fallback
        return HealthBaseline(
            cow_id=cow_id,
            core_temp_c=BaselineMetric(mean=38.6, std=0.25, median=38.6, mad=0.20, n_samples=0),
            activity_index=BaselineMetric(mean=50.0, std=8.0, median=50.0, mad=6.5, n_samples=0),
            daily_lying_min=BaselineMetric(mean=600.0, std=60.0, median=600.0, mad=50.0, n_samples=0),
            accel_rms=BaselineMetric(mean=0.12, std=0.03, median=0.12, mad=0.02, n_samples=0),
            valid_baseline_days=0,
            total_hours_analyzed=0,
            coverage_pct=0.0
        )

    df = raw_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    temp_col = CANONICAL_COLUMNS["CORE_BODY_TEMP_C"]
    act_col = CANONICAL_COLUMNS["ACTIVITY_INDEX"]
    lying_col = CANONICAL_COLUMNS["LYING_TIME_MIN"]
    accel_col = CANONICAL_COLUMNS["ACCELERATION_RMS"]

    # 1. Compute stats for continuous physiological/behavioral series
    temp_stats = _compute_metric_stats(df[temp_col]) if temp_col in df else BaselineMetric(mean=38.6, std=0.25, median=38.6, mad=0.2, n_samples=0)
    act_stats = _compute_metric_stats(df[act_col]) if act_col in df else BaselineMetric(mean=50.0, std=8.0, median=50.0, mad=6.5, n_samples=0)
    accel_stats = _compute_metric_stats(df[accel_col]) if accel_col in df else BaselineMetric(mean=0.12, std=0.03, median=0.12, mad=0.02, n_samples=0)

    # 2. Compute daily lying stats with coverage filtering (prevent circadian bias)
    valid_daily_totals: list[float] = []
    if lying_col in df:
        df["_date"] = df["timestamp"].dt.date
        grouped = df.groupby("_date")

        min_hours_per_day = int(24 * (config.min_daily_coverage_pct / 100.0))  # e.g., 24 * 0.8 = 19 hours

        for _, day_slice in grouped:
            valid_hours = day_slice[lying_col].dropna()
            if len(valid_hours) >= min_hours_per_day:
                # Scale day sum proportionally to 24h if missing 1-4 hours
                day_total = float(valid_hours.sum()) * (24.0 / len(valid_hours))
                valid_daily_totals.append(day_total)

    if valid_daily_totals:
        daily_series = pd.Series(valid_daily_totals)
        lying_stats = _compute_metric_stats(daily_series)
    else:
        # Fallback if baseline window has < 1 full valid day
        hourly_lying = df[lying_col].dropna() if lying_col in df else pd.Series([])
        if not hourly_lying.empty:
            mean_daily = float(hourly_lying.mean() * 24.0)
            std_daily = float(hourly_lying.std() * np.sqrt(24.0)) if len(hourly_lying) > 1 else 50.0
            lying_stats = BaselineMetric(
                mean=round(mean_daily, 1),
                std=round(std_daily, 1),
                median=round(mean_daily, 1),
                mad=round(std_daily * 0.6745, 1),
                n_samples=len(hourly_lying)
            )
        else:
            lying_stats = BaselineMetric(mean=600.0, std=60.0, median=600.0, mad=50.0, n_samples=0)

    # Calculate overall baseline window coverage
    total_expected_hours = max(1, int((b_end - b_start).total_seconds() / 3600))
    coverage_pct = round(min(100.0, (len(df) / total_expected_hours) * 100.0), 1)

    return HealthBaseline(
        cow_id=cow_id,
        core_temp_c=temp_stats,
        activity_index=act_stats,
        daily_lying_min=lying_stats,
        accel_rms=accel_stats,
        valid_baseline_days=len(valid_daily_totals),
        total_hours_analyzed=len(df),
        coverage_pct=coverage_pct
    )
