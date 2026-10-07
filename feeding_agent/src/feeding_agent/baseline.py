"""
Baseline derivation logic for individual cow feeding time series.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from feeding_agent.src.feeding_agent.config import BaselineConfig, DataConfig, MealsConfig
from feeding_agent.src.feeding_agent.meals import detect_bouts
from feeding_agent.src.feeding_agent.schemas import BaselineStats, FeedingTimeseries

logger = logging.getLogger(__name__)


def _robust_median_mad(series: pd.Series, trim_fraction: float = 0.1) -> Tuple[float, float]:
    clean = series.dropna()
    if len(clean) == 0:
        return 0.0, 0.0
    if len(clean) < 4:
        # Fallback to trimmed mean / SD
        if len(clean) > 2 and trim_fraction > 0:
            low = clean.quantile(trim_fraction)
            high = clean.quantile(1.0 - trim_fraction)
            trimmed = clean[(clean >= low) & (clean <= high)]
            if len(trimmed) > 0:
                clean = trimmed
        mean_val = float(clean.mean())
        std_val = float(clean.std(ddof=0)) if len(clean) > 1 else 0.0
        return mean_val, std_val

    med = float(clean.median())
    mad = float((clean - med).abs().median())
    scaled_mad = float(1.4826 * mad)
    return med, scaled_mad


def derive_baseline(
    ts: FeedingTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
    cfg: Optional[BaselineConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    meals_cfg: Optional[MealsConfig] = None,
) -> BaselineStats:
    """
    Derives baseline statistics for an individual cow over complete historical days.
    Guarantees no data leakage from the anomaly window.
    """
    cfg = cfg or BaselineConfig()
    data_cfg = data_cfg or DataConfig()
    meals_cfg = meals_cfg or MealsConfig()

    df = ts.df
    if df.empty or "cow_id" not in df.columns:
        return BaselineStats(
            n_days=0,
            coverage_pct=0.0,
            is_short=True,
            feeding_median=0.0,
            feeding_mad=0.0,
        )

    # Ensure timezone aware UTC
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)

    if baseline_start is None:
        baseline_start = anomaly_start - timedelta(days=cfg.default_days_before_anomaly)
    elif baseline_start.tzinfo is None:
        baseline_start = baseline_start.replace(tzinfo=timezone.utc)

    if baseline_end is None:
        baseline_end = anomaly_start
    elif baseline_end.tzinfo is None:
        baseline_end = baseline_end.replace(tzinfo=timezone.utc)

    # Ensure baseline_end does not leak into anomaly window
    if baseline_end > anomaly_start:
        baseline_end = anomaly_start

    cow_df = df[(df["cow_id"] == cow_id) & (df["timestamp"] >= baseline_start) & (df["timestamp"] < baseline_end)].copy()

    if cow_df.empty:
        return BaselineStats(
            n_days=0,
            coverage_pct=0.0,
            is_short=True,
            feeding_median=0.0,
            feeding_mad=0.0,
        )

    # Hourly diurnal profile (median feeding minutes by hour of day 0..23)
    cow_df["hour"] = cow_df["timestamp"].dt.hour
    hourly_profile_series = cow_df.groupby("hour")["feeding_minutes"].median()
    hourly_profile: Dict[int, float] = {h: float(hourly_profile_series.get(h, 0.0)) for h in range(24)}

    # Group by date for daily totals
    cow_df["date"] = cow_df["timestamp"].dt.date
    daily_groups = cow_df.groupby("date")

    valid_days_feeding = []
    valid_days_rumination = []
    valid_days_intake = []
    valid_days_visits = []

    for _, group in daily_groups:
        hours_count = len(group["feeding_minutes"].dropna())
        coverage = (hours_count / 24.0) * 100.0
        if coverage >= data_cfg.min_day_coverage_pct:
            # Scale sum if day is partially complete
            scaling = 24.0 / max(hours_count, 1)
            valid_days_feeding.append(group["feeding_minutes"].sum() * scaling)

            if ts.stream_availability.get("rumination", False) and "rumination_minutes" in group.columns:
                rum_non_null = group["rumination_minutes"].dropna()
                if len(rum_non_null) > 0:
                    valid_days_rumination.append(rum_non_null.sum() * scaling)

            if ts.stream_availability.get("intake", False) and "intake_kg" in group.columns:
                int_non_null = group["intake_kg"].dropna()
                if len(int_non_null) > 0:
                    valid_days_intake.append(int_non_null.sum() * scaling)

            if ts.stream_availability.get("visits", False) and "feeding_visits" in group.columns:
                vis_non_null = group["feeding_visits"].dropna()
                if len(vis_non_null) > 0:
                    valid_days_visits.append(vis_non_null.sum() * scaling)

    n_days = len(valid_days_feeding)
    is_short = n_days < cfg.min_days_reliable

    total_expected_hours = max(1.0, (baseline_end - baseline_start).total_seconds() / 3600.0)
    total_actual_hours = len(cow_df["feeding_minutes"].dropna())
    coverage_pct = min(100.0, (total_actual_hours / total_expected_hours) * 100.0)

    f_med, f_mad = _robust_median_mad(pd.Series(valid_days_feeding), cfg.trim_fraction)
    r_med, r_mad = _robust_median_mad(pd.Series(valid_days_rumination), cfg.trim_fraction) if valid_days_rumination else (None, None)
    i_med, i_mad = _robust_median_mad(pd.Series(valid_days_intake), cfg.trim_fraction) if valid_days_intake else (None, None)
    v_med, v_mad = _robust_median_mad(pd.Series(valid_days_visits), cfg.trim_fraction) if valid_days_visits else (None, None)

    # Baseline meal stats
    b_bouts_df = detect_bouts(
        cow_df.set_index("timestamp")["feeding_minutes"].sort_index(),
        gap_minutes=meals_cfg.gap_minutes,
    )
    b_days = max(1.0, (baseline_end - baseline_start).total_seconds() / 86400.0)
    baseline_bouts_per_day = float(len(b_bouts_df) / b_days) if not b_bouts_df.empty else 0.0
    baseline_mean_bout_length = float(b_bouts_df["duration_minutes"].mean()) if not b_bouts_df.empty else 0.0

    return BaselineStats(
        n_days=n_days,
        coverage_pct=coverage_pct,
        is_short=is_short,
        feeding_median=f_med,
        feeding_mad=f_mad,
        rumination_median=r_med,
        rumination_mad=r_mad,
        intake_median=i_med,
        intake_mad=i_mad,
        visits_median=v_med,
        visits_mad=v_mad,
        hourly_feeding_profile=hourly_profile,
        baseline_mean_bout_length=baseline_mean_bout_length,
        baseline_bouts_per_day=baseline_bouts_per_day,
    )
