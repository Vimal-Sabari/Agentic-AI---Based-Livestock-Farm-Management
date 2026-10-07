"""
Unit tests for herd comparison analysis.
"""
from datetime import datetime, timezone
import pytest

from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.config import HerdConfig
from feeding_agent.src.feeding_agent.herd import compute_herd_comparison
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries


def test_herd_comparison_target_excluded():
    ts = generate_synthetic(scenario="herd_wide_drop", seed=42)
    astart = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    aend = datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc)

    res = compute_herd_comparison(ts, target_cow_id="cow_01", anomaly_start=astart, anomaly_end=aend, target_direction="decline")

    # 16 total cows - 1 target cow = 15 cows analyzed
    assert res.cows_analyzed == 15
    assert res.fraction_deviating >= 0.35
    assert res.is_herd_wide is True


def test_herd_comparison_too_few_cows():
    ts = generate_synthetic(scenario="herd_wide_drop", seed=42)
    # Filter dataset down to only 3 cows
    df_small = ts.df[ts.df["cow_id"].isin(["cow_01", "cow_02", "cow_03"])].copy()
    ts_small = FeedingTimeseries(df=df_small, resample_freq="1h", source_file="small")

    astart = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    aend = datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc)

    cfg = HerdConfig(min_cows=4)
    res = compute_herd_comparison(ts_small, target_cow_id="cow_01", anomaly_start=astart, anomaly_end=aend, cfg=cfg)

    # 3 cows - 1 target = 2 cows analyzed < min_cows (4)
    assert res.cows_analyzed == 2
    assert res.is_herd_wide is False
    assert "Insufficient" in res.notes
