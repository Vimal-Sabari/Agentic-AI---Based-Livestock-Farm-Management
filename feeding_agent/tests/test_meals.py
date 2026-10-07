"""
Unit tests for feeding bout detection and meal pattern analysis.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest

from feeding_agent.src.feeding_agent.meals import compute_meal_pattern, detect_bouts
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries


def test_detect_bouts_gap_merging():
    # 4 consecutive hours: 06:00 (20 min), 07:00 (30 min), 08:00 (0 min), 09:00 (25 min)
    idx = pd.date_range("2026-09-01T06:00:00Z", periods=4, freq="1h")
    series = pd.Series([20.0, 30.0, 0.0, 25.0], index=idx)

    bouts = detect_bouts(series, gap_minutes=30)
    # Hours 06:00 and 07:00 merge (gap = 0 <= 30 min)
    # Hour 08:00 has 0 min -> gap between 07:00 and 09:00 is 1 hour (60 min > 30 min), starting a new bout
    assert len(bouts) == 2
    assert bouts.iloc[0]["duration_minutes"] == 50.0
    assert bouts.iloc[1]["duration_minutes"] == 25.0


def test_peak_hour_circular_shift():
    # Construct 24 hours of data with peak at 23:00 (hour 23)
    idx = pd.date_range("2026-09-01T00:00:00Z", periods=24, freq="1h")
    feeding_vals = [5.0] * 24
    feeding_vals[23] = 45.0  # Anomaly peak at 23:00

    df = pd.DataFrame({
        "timestamp": idx,
        "cow_id": "cow_01",
        "feeding_minutes": feeding_vals,
    })
    ts = FeedingTimeseries(df=df, resample_freq="1h", source_file="test")

    # Baseline profile with peak at 01:00 (hour 1)
    baseline_profile = {h: 5.0 for h in range(24)}
    baseline_profile[1] = 40.0

    start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 9, 1, 23, 59, tzinfo=timezone.utc)

    stats = compute_meal_pattern(ts, "cow_01", start, end, baseline_profile=baseline_profile)
    assert stats.peak_hour == 23
    # Circular distance between hour 23 and hour 1 is 2 hours (not 22 hours)
    assert stats.peak_hour_shift == pytest.approx(2.0)
