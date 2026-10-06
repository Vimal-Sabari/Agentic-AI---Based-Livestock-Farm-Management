"""
Feeding bout detection and meal pattern analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional
import numpy as np
import pandas as pd

from feeding_agent.src.feeding_agent.config import MealsConfig
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries, MealStats

logger = logging.getLogger(__name__)


def detect_bouts(feeding_series: pd.Series, gap_minutes: int = 30) -> pd.DataFrame:
    """
    Detects contiguous feeding bouts separated by gaps > gap_minutes.
    Expects feeding_series indexed by DatetimeIndex with values representing feeding minutes in each period.
    """
    if feeding_series.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "duration_minutes"])

    active = feeding_series[feeding_series > 0].sort_index()
    if active.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "duration_minutes"])

    timestamps = active.index
    bouts = []

    current_start = timestamps[0]
    current_last_ts = timestamps[0]
    current_dur = float(active.iloc[0])

    for i in range(1, len(active)):
        ts = timestamps[i]
        val = float(active.iloc[i])

        # At hourly frequency, if ts is consecutive to current_last_ts (<= 1 hour), un-fed gap is 0 min.
        # Otherwise, un-fed gap is the time difference minus 1 hour.
        time_diff_min = (ts - current_last_ts).total_seconds() / 60.0
        gap_between_periods = max(0.0, time_diff_min - 60.0)

        if gap_between_periods <= gap_minutes:
            # Merge into current bout
            current_last_ts = ts
            current_dur += val
        else:
            # Finalize current bout
            current_end = current_last_ts + pd.Timedelta(minutes=60.0)
            bouts.append({
                "start_time": current_start,
                "end_time": current_end,
                "duration_minutes": current_dur,
            })
            current_start = ts
            current_last_ts = ts
            current_dur = val

    # Append final bout
    current_end = current_last_ts + pd.Timedelta(minutes=60.0)
    bouts.append({
        "start_time": current_start,
        "end_time": current_end,
        "duration_minutes": current_dur,
    })

    return pd.DataFrame(bouts)


def compute_meal_pattern(
    ts: FeedingTimeseries,
    cow_id: str,
    start: datetime,
    end: datetime,
    baseline_profile: Optional[Dict[int, float]] = None,
    cfg: Optional[MealsConfig] = None,
) -> MealStats:
    """
    Computes meal pattern statistics over a given window.
    """
    cfg = cfg or MealsConfig()
    df = ts.df

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cow_df = df[(df["cow_id"] == cow_id) & (df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
    if cow_df.empty or "feeding_minutes" not in cow_df.columns:
        return MealStats(
            bouts_df=pd.DataFrame(columns=["start_time", "end_time", "duration_minutes"]),
            bouts_per_day=0.0,
            mean_bout_length=0.0,
            longest_bout=0.0,
            peak_hour=0,
            peak_hour_shift=0.0,
        )

    cow_df = cow_df.set_index("timestamp")["feeding_minutes"].sort_index()
    bouts_df = detect_bouts(cow_df, gap_minutes=cfg.gap_minutes)

    total_days = max(1.0, (end - start).total_seconds() / 86400.0)
    bouts_per_day = float(len(bouts_df) / total_days)
    mean_bout_length = float(bouts_df["duration_minutes"].mean()) if not bouts_df.empty else 0.0
    longest_bout = float(bouts_df["duration_minutes"].max()) if not bouts_df.empty else 0.0

    # Diurnal peak hour in anomaly window
    hourly_mean = cow_df.groupby(cow_df.index.hour).mean()
    peak_hour = int(hourly_mean.idxmax()) if not hourly_mean.empty and hourly_mean.max() > 0 else 0

    # Baseline peak hour
    baseline_peak = 0
    if baseline_profile:
        baseline_peak = max(baseline_profile.items(), key=lambda x: x[1])[0]

    # Circular distance in hours between peak hours
    peak_shift = float(abs((peak_hour - baseline_peak + 12) % 24 - 12))

    return MealStats(
        bouts_df=bouts_df,
        bouts_per_day=bouts_per_day,
        mean_bout_length=mean_bout_length,
        longest_bout=longest_bout,
        peak_hour=peak_hour,
        peak_hour_shift=peak_shift,
    )
