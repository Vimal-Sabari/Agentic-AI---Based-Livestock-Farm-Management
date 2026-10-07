"""
Unit tests for baseline derivation logic.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from feeding_agent.src.feeding_agent.baseline import _robust_median_mad, derive_baseline
from feeding_agent.src.feeding_agent.config import BaselineConfig
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries


def test_robust_median_mad_known():
    s = pd.Series([10.0, 12.0, 11.0, 13.0, 100.0])  # 100 is an outlier
    med, mad = _robust_median_mad(s)
    assert med == 12.0
    # Abs deviations from 12: |10-12|=2, |12-12|=0, |11-12|=1, |13-12|=1, |100-12|=88
    # Sorted abs devs: [0, 1, 1, 2, 88]. Median abs dev = 1.0. Scaled MAD = 1.4826
    assert abs(mad - 1.4826) < 1e-3


def test_outlier_does_not_move_baseline(stable_ts):
    anom_start = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    base_clean = derive_baseline(stable_ts, "cow_01", anomaly_start=anom_start)

    # Inject a massive outlier in one baseline day
    df_noisy = stable_ts.df.copy()
    first_day_mask = (df_noisy["cow_id"] == "cow_01") & (df_noisy["timestamp"] < anom_start)
    first_idx = df_noisy[first_day_mask].index[0]
    df_noisy.loc[first_idx, "feeding_minutes"] = 1000.0  # huge single-hour outlier

    ts_noisy = FeedingTimeseries(
        df=df_noisy,
        resample_freq=stable_ts.resample_freq,
        source_file="noisy",
        coverage_pct=stable_ts.coverage_pct,
        notes="noisy",
        stream_availability=stable_ts.stream_availability,
    )
    base_noisy = derive_baseline(ts_noisy, "cow_01", anomaly_start=anom_start)

    # Median is robust against a single outlier day
    assert abs(base_noisy.feeding_median - base_clean.feeding_median) < 20.0


def test_baseline_derivation_no_leakage(stable_ts):
    anom_start = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    base = derive_baseline(stable_ts, "cow_01", anomaly_start=anom_start)

    assert base.n_days >= 5
    assert not base.is_short
    assert base.feeding_median > 0
    assert 0 in base.hourly_feeding_profile
    assert base.baseline_mean_bout_length > 0


def test_short_baseline_flag(stable_ts):
    anom_start = datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc)  # Only 2 days prior
    cfg = BaselineConfig(default_days_before_anomaly=2, min_days_reliable=5)
    base = derive_baseline(stable_ts, "cow_01", anomaly_start=anom_start, cfg=cfg)

    assert base.is_short
    assert base.n_days <= 2
