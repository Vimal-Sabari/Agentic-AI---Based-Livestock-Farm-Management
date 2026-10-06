"""
Unit tests for verdict scoring, synthesis logic, and feed availability.
"""
from datetime import datetime, timezone
import pandas as pd
import pytest

from feeding_agent.src.feeding_agent.availability import compute_availability
from feeding_agent.src.feeding_agent.config import AgentConfig, AvailabilityConfig, VerdictConfig, VerdictWeights
from feeding_agent.src.feeding_agent.schemas import (
    AvailabilityResult,
    BaselineStats,
    DeviationResult,
    FeedingTimeseries,
    HerdResult,
    RuminationResult,
)
from feeding_agent.src.feeding_agent.verdict import compute_verdict


def test_verdict_supports():
    base = BaselineStats(n_days=7, coverage_pct=100.0, is_short=False, feeding_median=300.0, feeding_mad=15.0)
    dev = DeviationResult(
        feeding_daily_val=210.0,
        feeding_baseline_val=300.0,
        feeding_delta_pct=-30.0,
        feeding_zscore=-6.0,
        feeding_direction="decline",
        feeding_significant=True,
        rumination_daily_val=350.0,
        rumination_baseline_val=500.0,
        rumination_delta_pct=-30.0,
        rumination_zscore=-5.0,
        rumination_direction="decline",
        rumination_significant=True,
        intake_daily_val=15.0,
        intake_baseline_val=20.0,
        intake_delta_pct=-25.0,
        intake_zscore=-4.0,
        intake_direction="decline",
        intake_significant=True,
    )
    rum = RuminationResult(available=True, proxy_used=False)
    avail = AvailabilityResult(stream_exists=True, status="normal")
    herd = HerdResult(cows_analyzed=15, cows_deviating=1, fraction_deviating=0.06, is_herd_wide=False)

    verdict = compute_verdict(base, dev, rum, avail, herd, coverage_pct=100.0)

    assert verdict.finding == "supports"
    assert verdict.evidence_strength == "strong"
    assert 0.0 <= verdict.confidence <= 1.0


def test_verdict_does_not_support():
    base = BaselineStats(n_days=7, coverage_pct=100.0, is_short=False, feeding_median=300.0, feeding_mad=15.0)
    dev = DeviationResult(
        feeding_daily_val=295.0,
        feeding_baseline_val=300.0,
        feeding_delta_pct=-1.7,
        feeding_zscore=-0.3,
        feeding_direction="stable",
        feeding_significant=False,
    )
    rum = RuminationResult(available=True, proxy_used=False)
    avail = AvailabilityResult(stream_exists=True, status="normal")
    herd = HerdResult(cows_analyzed=15, cows_deviating=0, fraction_deviating=0.0, is_herd_wide=False)

    verdict = compute_verdict(base, dev, rum, avail, herd, coverage_pct=100.0)

    assert verdict.finding == "does_not_support"
    assert verdict.evidence_strength in ["weak", "none"]


def test_verdict_inconclusive_low_coverage():
    base = BaselineStats(n_days=7, coverage_pct=100.0, is_short=False, feeding_median=300.0, feeding_mad=15.0)
    dev = DeviationResult(
        feeding_daily_val=210.0,
        feeding_baseline_val=300.0,
        feeding_delta_pct=-30.0,
        feeding_zscore=-6.0,
        feeding_direction="decline",
        feeding_significant=True,
    )
    rum = RuminationResult(available=True)
    avail = AvailabilityResult(stream_exists=True, status="normal")
    herd = HerdResult(cows_analyzed=15, cows_deviating=1, fraction_deviating=0.06, is_herd_wide=False)

    verdict = compute_verdict(base, dev, rum, avail, herd, coverage_pct=20.0)  # Below 40% threshold

    assert verdict.finding == "inconclusive"
    assert verdict.evidence_strength == "none"
    assert verdict.confidence <= 0.25


def test_verdict_weights_sum_validation():
    # Pass weights that do not sum to 1.0 -> AgentConfig normalizes them
    d = {
        "verdict": {
            "weights": {
                "feeding": 1.0,
                "rumination": 1.0,
                "intake": 1.0,
                "consistency": 1.0,
                "coverage": 1.0,
            }
        }
    }
    cfg = AgentConfig.from_dict(d)
    w = cfg.verdict.weights
    total = w.feeding + w.rumination + w.intake + w.consistency + w.coverage
    assert total == pytest.approx(1.0)


def test_confidence_clamping():
    base = BaselineStats(n_days=1, coverage_pct=10.0, is_short=True, feeding_median=0.0, feeding_mad=0.0)
    dev = DeviationResult(
        feeding_daily_val=0.0,
        feeding_baseline_val=0.0,
        feeding_delta_pct=0.0,
        feeding_zscore=0.0,
        feeding_direction="stable",
        feeding_significant=False,
    )
    rum = RuminationResult(available=False, proxy_used=True)
    avail = AvailabilityResult(stream_exists=False, status="unknown")
    herd = HerdResult(cows_analyzed=0, cows_deviating=0, fraction_deviating=0.0, is_herd_wide=False)

    verdict = compute_verdict(base, dev, rum, avail, herd, coverage_pct=10.0)
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.confidence <= 0.25


def test_availability_analysis():
    # 1. Unknown availability (stream missing)
    df_no_avail = pd.DataFrame({"timestamp": [pd.Timestamp("2026-09-01T00:00:00Z")], "cow_id": ["cow_01"]})
    ts_no_avail = FeedingTimeseries(df=df_no_avail, stream_availability={"availability": False})
    res_unk = compute_availability(ts_no_avail, "cow_01", datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc), datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc))
    assert res_unk.status == "unknown"
    assert res_unk.stream_exists is False

    # 2. Restricted availability (<50%)
    df_restr = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01T00:00:00Z", periods=48, freq="1h"),
        "cow_id": "cow_01",
        "feed_availability_pct": [35.0] * 48,
    })
    ts_restr = FeedingTimeseries(df=df_restr, stream_availability={"availability": True})
    res_restr = compute_availability(ts_restr, "cow_01", datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc), datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc))
    assert res_restr.status == "restricted"
    assert res_restr.restricted is True

    # 3. Normal availability (>=50%)
    df_norm = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01T00:00:00Z", periods=48, freq="1h"),
        "cow_id": "cow_01",
        "feed_availability_pct": [85.0] * 48,
    })
    ts_norm = FeedingTimeseries(df=df_norm, stream_availability={"availability": True})
    res_norm = compute_availability(ts_norm, "cow_01", datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc), datetime(2026, 9, 3, 0, 0, tzinfo=timezone.utc))
    assert res_norm.status == "normal"
    assert res_norm.restricted is False
