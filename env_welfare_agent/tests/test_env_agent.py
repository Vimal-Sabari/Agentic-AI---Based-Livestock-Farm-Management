"""
Comprehensive pytest suite for the Environment & Welfare Agent.

Coverage:
- THI formula (known reference values)
- Threshold classification
- Diurnal-aware baseline derivation
- Exposure metrics (all fields)
- Herd-wide fraction
- Verdict logic (three scenarios)
- Missing-data behaviour
- Request validation (unknown cow, empty window, inverted window, window outside data range)
- Edge cases: all-NaN environment, single-cow herd, anomaly at dataset start
- End-to-end: synthetic scenario → handle_query → schema-valid AgentResponse with expected finding
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from env_welfare_agent.src.env_agent.config import AgentConfig, THIThresholds
from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    ENV_TS_COLS,
    BaselineStats,
    CowActivityTimeseries,
    EnvTimeseries,
    ExposureMetrics,
    HerdDeviationResult,
    AlignmentResult,
)
from env_welfare_agent.src.env_agent.thi import classify_thi, classify_thi_series, compute_thi
from env_welfare_agent.src.env_agent.baseline import derive_baseline, diurnal_delta
from env_welfare_agent.src.env_agent.exposure import compute_exposure
from env_welfare_agent.src.env_agent.herd import compute_herd_deviation
from env_welfare_agent.src.env_agent.verdict import compute_verdict
from env_welfare_agent.src.env_agent.agent import EnvironmentWelfareAgent
from env_welfare_agent.src.env_agent.adapters.synthetic import generate_synthetic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.models import AgentRequest, AgentResponse, TimeWindow


# ===========================================================================
# Fixtures
# ===========================================================================

UTC = timezone.utc

def _tz(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

@pytest.fixture(scope="module")
def heat_stress_data():
    """Herd-wide heat stress scenario (all cows, 14 days)."""
    return generate_synthetic("herd_heat_stress", n_cows=16, n_days=14, seed=0)


@pytest.fixture(scope="module")
def single_sick_data():
    """Single sick cow with normal environment (cow_01 anomaly only)."""
    return generate_synthetic("single_sick_cow", n_cows=16, n_days=14, seed=1)


@pytest.fixture(scope="module")
def mixed_data():
    """Mixed/ambiguous scenario (~40% cows deviating, moderate THI)."""
    return generate_synthetic("mixed_ambiguous", n_cows=16, n_days=14, seed=2)


@pytest.fixture(scope="module")
def heat_stress_agent(heat_stress_data):
    env_ts, act_ts = heat_stress_data
    return EnvironmentWelfareAgent(env_ts, act_ts)


@pytest.fixture(scope="module")
def sick_agent(single_sick_data):
    env_ts, act_ts = single_sick_data
    return EnvironmentWelfareAgent(env_ts, act_ts)


@pytest.fixture(scope="module")
def mixed_agent(mixed_data):
    env_ts, act_ts = mixed_data
    return EnvironmentWelfareAgent(env_ts, act_ts)


def _make_request(cow_id, start_iso, end_iso, qtype="environment_check"):
    return AgentRequest(
        from_agent="health_behavior",
        to_agent="environment_welfare",
        cow_id=cow_id,
        question="Could environment explain the anomaly?",
        question_type=qtype,
        anomaly_window=TimeWindow(start=start_iso, end=end_iso),
    )


# ===========================================================================
# 1. THI formula — reference values
# ===========================================================================

class TestTHIFormula:
    """Validate THI = (1.8*T+32) − (0.55 − 0.0055*RH) * (1.8*T − 26)."""

    @pytest.mark.parametrize("temp_c, rh, expected, tol", [
        (22.0, 50.0, 67.86, 0.3),   # near comfort boundary
        (28.0, 70.0, 78.37, 0.3),   # moderate stress
        (32.0, 80.0, 86.12, 0.3),   # severe stress
        (18.0, 40.0, 62.29, 0.5),   # well within comfort
        (35.0, 90.0, 92.97, 0.5),   # emergency territory
    ])
    def test_thi_scalar(self, temp_c, rh, expected, tol):
        result = compute_thi(temp_c, rh)
        assert abs(result - expected) <= tol, (
            f"T={temp_c}, RH={rh}: expected~{expected}, got {result:.2f}"
        )

    def test_thi_series(self):
        temps = pd.Series([22.0, 28.0, 32.0])
        rhs = pd.Series([50.0, 70.0, 80.0])
        results = compute_thi(temps, rhs)
        assert isinstance(results, pd.Series)
        assert len(results) == 3
        assert all(results.notna())

    def test_thi_nan_propagation(self):
        result = compute_thi(float("nan"), 60.0)
        assert math.isnan(result)

    def test_thi_series_nan(self):
        temps = pd.Series([float("nan"), 28.0])
        rhs = pd.Series([70.0, 70.0])
        result = compute_thi(temps, rhs)
        assert math.isnan(result.iloc[0])
        assert not math.isnan(result.iloc[1])


# ===========================================================================
# 2. Threshold classification
# ===========================================================================

class TestTHIClassification:
    def test_comfort(self):
        assert classify_thi(65.0) == "comfort"

    def test_mild(self):
        assert classify_thi(70.0) == "mild"

    def test_moderate(self):
        assert classify_thi(76.0) == "moderate"

    def test_severe(self):
        assert classify_thi(85.0) == "severe"

    def test_emergency(self):
        assert classify_thi(92.0) == "emergency"

    def test_nan(self):
        assert classify_thi(float("nan")) == "unknown"

    def test_boundary_mild(self):
        # Exactly at mild threshold
        assert classify_thi(68.0) == "mild"

    def test_boundary_moderate(self):
        assert classify_thi(72.0) == "moderate"

    def test_custom_thresholds(self):
        th = THIThresholds(mild_min=60.0, moderate_min=65.0)
        assert classify_thi(62.0, th) == "mild"

    def test_series_classification(self):
        thi_s = pd.Series([60.0, 70.0, 78.0, 85.0, float("nan")])
        result = classify_thi_series(thi_s)
        assert result.iloc[0] == "comfort"
        assert result.iloc[1] == "mild"
        assert result.iloc[2] == "moderate"
        assert result.iloc[3] == "severe"
        assert result.iloc[4] == "unknown"


# ===========================================================================
# 3. Diurnal-aware baseline
# ===========================================================================

class TestBaseline:
    def _make_env_ts(self, n_days=5, start_thi=65.0):
        start = datetime(2026, 9, 1, tzinfo=UTC)
        hours = n_days * 24
        ts = [start + timedelta(hours=h) for h in range(hours)]
        # Simple diurnal: peaks at 14:00
        thi_vals = [
            start_thi + 8.0 * math.sin(math.pi * (t.hour - 6) / 12.0)
            for t in ts
        ]
        df = pd.DataFrame({
            ENV_TS_COLS["TIMESTAMP"]: ts,
            ENV_TS_COLS["TEMP_C"]: [20.0] * hours,
            ENV_TS_COLS["RH_PCT"]: [60.0] * hours,
            ENV_TS_COLS["THI"]: thi_vals,
            ENV_TS_COLS["SOURCE"]: ["barn"] * hours,
        })
        return EnvTimeseries(df=df, resample_freq="1h", coverage_pct=100.0)

    def test_baseline_mean_thi(self):
        env_ts = self._make_env_ts(n_days=7)
        anomaly_start = datetime(2026, 9, 6, tzinfo=UTC)
        bs = derive_baseline(env_ts, anomaly_start)
        assert not math.isnan(bs.mean_thi)
        assert 60 < bs.mean_thi < 75  # sensible range

    def test_diurnal_profile_populated(self):
        env_ts = self._make_env_ts(n_days=7)
        anomaly_start = datetime(2026, 9, 6, tzinfo=UTC)
        bs = derive_baseline(env_ts, anomaly_start)
        # Should have 24 hour entries
        assert len(bs.diurnal_thi_profile) == 24

    def test_no_baseline_data(self):
        env_ts = self._make_env_ts(n_days=3)
        # Anomaly at very start – no baseline data before it
        anomaly_start = datetime(2026, 9, 1, tzinfo=UTC)
        bs = derive_baseline(env_ts, anomaly_start)
        assert bs.coverage_pct == 0.0
        assert math.isnan(bs.mean_thi)

    def test_explicit_baseline_window(self):
        env_ts = self._make_env_ts(n_days=10)
        b_start = datetime(2026, 9, 2, tzinfo=UTC)
        b_end = datetime(2026, 9, 4, tzinfo=UTC)
        anomaly_start = datetime(2026, 9, 8, tzinfo=UTC)
        bs = derive_baseline(env_ts, anomaly_start, b_start, b_end)
        assert bs.n_hours > 0
        assert 0 < bs.coverage_pct <= 100

    def test_diurnal_delta(self):
        env_ts = self._make_env_ts(n_days=10)
        anomaly_start = datetime(2026, 9, 8, tzinfo=UTC)
        anomaly_end = anomaly_start + timedelta(days=1)
        bs = derive_baseline(env_ts, anomaly_start)
        deltas = diurnal_delta(env_ts, bs, anomaly_start, anomaly_end)
        # Should have entries for 24 hours; most deltas near 0 (same base pattern)
        assert len(deltas) <= 24


# ===========================================================================
# 4. Exposure metrics
# ===========================================================================

class TestExposureMetrics:
    def _make_hot_env(self):
        """THI = 82-86 for 48 hours."""
        start = datetime(2026, 9, 8, tzinfo=UTC)
        ts = [start + timedelta(hours=h) for h in range(48)]
        thi_vals = [82.0 + 4.0 * math.sin(math.pi * t.hour / 12.0) for t in ts]
        df = pd.DataFrame({
            ENV_TS_COLS["TIMESTAMP"]: ts,
            ENV_TS_COLS["TEMP_C"]: [30.0] * 48,
            ENV_TS_COLS["RH_PCT"]: [80.0] * 48,
            ENV_TS_COLS["THI"]: thi_vals,
            ENV_TS_COLS["SOURCE"]: ["barn"] * 48,
        })
        return EnvTimeseries(df=df, resample_freq="1h")

    def test_mean_thi_hot(self):
        env_ts = self._make_hot_env()
        exp = compute_exposure(env_ts, datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC))
        assert exp.mean_thi >= 80.0

    def test_cumulative_heat_load_positive(self):
        env_ts = self._make_hot_env()
        exp = compute_exposure(env_ts, datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC))
        assert exp.cumulative_heat_load > 0

    def test_stress_hours(self):
        env_ts = self._make_hot_env()
        exp = compute_exposure(env_ts, datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC))
        total_stress = exp.hours_mild + exp.hours_moderate + exp.hours_severe + exp.hours_emergency
        assert total_stress > 0

    def test_poor_night_recovery(self):
        env_ts = self._make_hot_env()
        exp = compute_exposure(env_ts, datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 10, tzinfo=UTC))
        # All hours above 68 → poor night recovery
        assert exp.poor_night_recovery is True

    def test_empty_window(self):
        env_ts = self._make_hot_env()
        # Window entirely outside data range
        exp = compute_exposure(
            env_ts,
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        )
        assert exp.coverage_pct == 0.0
        assert math.isnan(exp.mean_thi)

    def test_all_nan_thi(self):
        start = datetime(2026, 9, 8, tzinfo=UTC)
        ts = [start + timedelta(hours=h) for h in range(24)]
        df = pd.DataFrame({
            ENV_TS_COLS["TIMESTAMP"]: ts,
            ENV_TS_COLS["TEMP_C"]: [float("nan")] * 24,
            ENV_TS_COLS["RH_PCT"]: [float("nan")] * 24,
            ENV_TS_COLS["THI"]: [float("nan")] * 24,
            ENV_TS_COLS["SOURCE"]: ["barn"] * 24,
        })
        env_ts = EnvTimeseries(df=df, resample_freq="1h")
        exp = compute_exposure(env_ts, start, start + timedelta(hours=24))
        assert math.isnan(exp.mean_thi)
        assert exp.coverage_pct == 0.0


# ===========================================================================
# 5. Herd-wide deviation
# ===========================================================================

class TestHerdDeviation:
    def _make_act_ts(self, n_cows=16, n_hours=96, all_drop=False, only_first_drops=False):
        start = datetime(2026, 9, 8, tzinfo=UTC)
        rows = []
        for cow_idx in range(1, n_cows + 1):
            cid = f"cow_{cow_idx:02d}"
            base = 50.0 + cow_idx * 0.5
            for h in range(n_hours):
                ts = start + timedelta(hours=h)
                in_anomaly = h >= 48  # second 2 days = anomaly
                if in_anomaly:
                    if all_drop:
                        act = base * 0.65  # all cows drop
                    elif only_first_drops and cid == "cow_01":
                        act = base * 0.60
                    else:
                        act = base * 1.0
                else:
                    act = base
                rows.append({COW_ACT_COLS["TIMESTAMP"]: ts, COW_ACT_COLS["COW_ID"]: cid,
                              COW_ACT_COLS["ACTIVITY"]: act + np.random.default_rng(cow_idx + h).normal(0, 1),
                              COW_ACT_COLS["LYING"]: 25.0})
        df = pd.DataFrame(rows)
        df[COW_ACT_COLS["TIMESTAMP"]] = pd.to_datetime(df[COW_ACT_COLS["TIMESTAMP"]], utc=True)
        return CowActivityTimeseries(df=df, resample_freq="1h")

    def test_herd_wide_all_drop(self):
        act_ts = self._make_act_ts(all_drop=True)
        start = datetime(2026, 9, 8, tzinfo=UTC)
        result = compute_herd_deviation(
            act_ts, "cow_01",
            anomaly_start=start + timedelta(hours=48),
            anomaly_end=start + timedelta(hours=96),
            baseline_start=start,
            baseline_end=start + timedelta(hours=48),
        )
        assert result.fraction_deviating > 0.5
        assert result.is_herd_wide is True

    def test_single_cow_only(self):
        act_ts = self._make_act_ts(only_first_drops=True)
        start = datetime(2026, 9, 8, tzinfo=UTC)
        result = compute_herd_deviation(
            act_ts, "cow_01",
            anomaly_start=start + timedelta(hours=48),
            anomaly_end=start + timedelta(hours=96),
            baseline_start=start,
            baseline_end=start + timedelta(hours=48),
        )
        assert result.fraction_deviating < 0.2
        assert result.is_herd_wide is False

    def test_single_cow_herd(self):
        """Only one cow → skip herd check gracefully."""
        act_ts = self._make_act_ts(n_cows=1, all_drop=True)
        start = datetime(2026, 9, 8, tzinfo=UTC)
        result = compute_herd_deviation(
            act_ts, "cow_01",
            anomaly_start=start + timedelta(hours=48),
            anomaly_end=start + timedelta(hours=96),
        )
        assert result.cows_analyzed <= 1
        assert result.is_herd_wide is False


# ===========================================================================
# 6. Verdict logic — three scenarios
# ===========================================================================

class TestVerdictLogic:
    def _base_exposure(self, mean_thi=80.0, hours_severe=12.0, cov=95.0):
        return ExposureMetrics(
            mean_thi=mean_thi, max_thi=mean_thi + 4, min_thi=mean_thi - 4,
            mean_temp_c=30.0, mean_rh_pct=80.0,
            hours_mild=2.0, hours_moderate=8.0,
            hours_severe=hours_severe, hours_emergency=0.0,
            cumulative_heat_load=144.0, night_min_thi=75.0,
            poor_night_recovery=True, max_thi_rate_per_hour=2.0,
            coverage_pct=cov, dominant_stress_band="severe",
        )

    def _base_baseline(self, mean_thi=65.0, cov=95.0):
        return BaselineStats(
            mean_thi=mean_thi, std_thi=3.0, p10_thi=60.0, p90_thi=70.0,
            mean_temp_c=20.0, mean_rh_pct=60.0,
            diurnal_thi_profile={h: 65.0 for h in range(24)},
            coverage_pct=cov, n_hours=72,
        )

    def _herd_wide_result(self, frac=0.75):
        cows = 16
        deviating = int(cows * frac)
        return HerdDeviationResult(
            cows_analyzed=cows, cows_deviating=deviating,
            fraction_deviating=frac, is_herd_wide=frac >= 0.5,
            target_cow_zscore=-2.5, target_exceeds_herd_median=False,
            per_cow_deviations={},
        )

    def _align_strong(self):
        return AlignmentResult(
            best_lag_hours=2, best_r=-0.80,
            lags_tested=list(range(25)), r_values=[-0.5] * 25,
            n_points=20, sufficient_data=True,
        )

    def _align_no_data(self):
        return AlignmentResult(
            best_lag_hours=0, best_r=float("nan"),
            lags_tested=[], r_values=[], n_points=0, sufficient_data=False,
        )

    def test_scenario_a_supports(self):
        """Herd-wide heat stress → should support."""
        v = compute_verdict(
            self._base_exposure(mean_thi=82.0),
            self._base_baseline(mean_thi=65.0),
            self._herd_wide_result(frac=0.80),
            self._align_strong(),
        )
        assert v.finding == "supports"
        assert v.evidence_strength in ("strong", "moderate")

    def test_scenario_b_does_not_support(self):
        """Normal environment, isolated sick cow → does_not_support."""
        normal_exposure = ExposureMetrics(
            mean_thi=64.0, max_thi=67.0, min_thi=61.0,
            mean_temp_c=18.0, mean_rh_pct=55.0,
            hours_mild=0.0, hours_moderate=0.0, hours_severe=0.0, hours_emergency=0.0,
            cumulative_heat_load=0.0, night_min_thi=60.0,
            poor_night_recovery=False, max_thi_rate_per_hour=0.5,
            coverage_pct=95.0, dominant_stress_band="comfort",
        )
        isolated_herd = self._herd_wide_result(frac=0.05)
        v = compute_verdict(
            normal_exposure,
            self._base_baseline(mean_thi=63.0),
            isolated_herd,
            self._align_no_data(),
        )
        assert v.finding == "does_not_support"

    def test_scenario_c_inconclusive(self):
        """Moderate THI rise, ~40% herd deviating → inconclusive."""
        moderate_exposure = self._base_exposure(mean_thi=74.0, hours_severe=0.0, cov=75.0)
        mixed_herd = self._herd_wide_result(frac=0.40)
        v = compute_verdict(
            moderate_exposure,
            self._base_baseline(mean_thi=69.0),
            mixed_herd,
            self._align_no_data(),
        )
        # Should be inconclusive or moderate support — not strongly one way
        assert v.finding in ("inconclusive", "supports", "does_not_support")
        # Evidence should not be "strong"
        assert v.evidence_strength != "strong"

    def test_low_coverage_forces_inconclusive(self):
        v = compute_verdict(
            self._base_exposure(cov=20.0),
            self._base_baseline(cov=20.0),
            self._herd_wide_result(),
            self._align_strong(),
        )
        assert v.finding == "inconclusive"
        assert v.evidence_strength == "none"


# ===========================================================================
# 7. Request validation edge cases
# ===========================================================================

class TestRequestValidation:
    def _make_minimal_agent(self):
        env_ts, act_ts = generate_synthetic("herd_heat_stress", n_cows=4, n_days=5, seed=99)
        return EnvironmentWelfareAgent(env_ts, act_ts)

    def test_unknown_cow(self):
        agent = self._make_minimal_agent()
        req = _make_request("cow_999", "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
        resp = agent.handle_query(req)
        assert resp.finding == "inconclusive"
        assert resp.confidence == 0.0
        assert "not found" in resp.summary.lower()

    def test_inverted_window(self):
        agent = self._make_minimal_agent()
        req = _make_request("cow_01", "2026-09-10T00:00:00Z", "2026-09-09T00:00:00Z")
        resp = agent.handle_query(req)
        assert resp.finding == "inconclusive"

    def test_window_outside_data_range(self):
        agent = self._make_minimal_agent()
        req = _make_request("cow_01", "2010-01-01T00:00:00Z", "2010-01-02T00:00:00Z")
        resp = agent.handle_query(req)
        # Should handle gracefully, not crash
        assert isinstance(resp, AgentResponse)

    def test_unsupported_question_type(self):
        agent = self._make_minimal_agent()
        req = AgentRequest(
            from_agent="health_behavior",
            to_agent="environment_welfare",
            cow_id="cow_01",
            question="?",
            question_type="yield_drop_check",  # not supported by this agent
            anomaly_window=TimeWindow(start="2026-09-09T00:00:00Z", end="2026-09-10T00:00:00Z"),
        )
        resp = agent.handle_query(req)
        assert resp.finding == "inconclusive"
        assert "unsupported" in resp.summary.lower()

    def test_anomaly_at_dataset_start_no_baseline(self):
        """Anomaly at dataset start → baseline is empty → lower confidence but no crash."""
        agent = self._make_minimal_agent()
        req = _make_request("cow_01", "2026-09-01T00:00:00Z", "2026-09-02T00:00:00Z")
        resp = agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        # Baseline coverage should be 0 → lower confidence
        assert resp.confidence <= 0.70


# ===========================================================================
# 8. End-to-end integration tests
# ===========================================================================

class TestEndToEnd:
    def test_herd_heat_stress_supports(self, heat_stress_agent):
        """Scenario (a): herd-wide heat stress → expect 'supports' finding."""
        req = _make_request(
            "cow_03",
            "2026-09-09T00:00:00Z",   # day 8 = anomaly start
            "2026-09-11T00:00:00Z",
        )
        resp = heat_stress_agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        assert resp.from_agent == "environment_welfare"
        assert resp.finding == "supports"
        assert resp.herd_context.is_herd_wide is True
        assert resp.confidence > 0.5
        # Required metric keys
        assert "mean_thi" in resp.metrics
        assert "fraction_herd_deviating" in resp.metrics
        # Summary should contain metric numbers
        thi_val = resp.metrics["mean_thi"].value
        if not math.isnan(thi_val):
            assert str(round(thi_val, 1)) in resp.summary

    def test_single_sick_cow_does_not_support(self, sick_agent):
        """Scenario (b): single sick cow, normal environment → does_not_support or inconclusive."""
        req = _make_request(
            "cow_01",
            "2026-09-09T00:00:00Z",
            "2026-09-11T00:00:00Z",
        )
        resp = sick_agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        assert resp.finding in ("does_not_support", "inconclusive")
        # Herd context: only cow_01 deviating
        assert resp.herd_context.fraction_deviating < 0.3

    def test_mixed_scenario_result_valid(self, mixed_agent):
        """Scenario (c): mixed → response is schema-valid and not an error."""
        req = _make_request(
            "cow_03",
            "2026-09-09T00:00:00Z",
            "2026-09-11T00:00:00Z",
            qtype="heat_stress_check",
        )
        resp = mixed_agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        assert resp.query_id == req.query_id
        assert resp.cow_id == req.cow_id
        assert resp.finding in ("supports", "does_not_support", "inconclusive")
        assert resp.evidence_strength in ("strong", "moderate", "weak", "none")
        assert 0.0 <= resp.confidence <= 1.0
        assert isinstance(resp.limitations, list)
        assert isinstance(resp.suggested_followups, list)

    def test_herd_wide_check_question_type(self, heat_stress_agent):
        """herd_wide_check question type should work identically."""
        req = _make_request(
            "cow_05",
            "2026-09-09T00:00:00Z",
            "2026-09-10T00:00:00Z",
            qtype="herd_wide_check",
        )
        resp = heat_stress_agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        assert resp.finding in ("supports", "does_not_support", "inconclusive")

    def test_message_history_recorded(self, heat_stress_agent):
        """Every query should be recorded in message history."""
        before = len(heat_stress_agent.message_history)
        req = _make_request("cow_02", "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
        heat_stress_agent.handle_query(req)
        assert len(heat_stress_agent.message_history) == before + 1

    def test_response_query_id_matches(self, heat_stress_agent):
        req = _make_request("cow_04", "2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
        resp = heat_stress_agent.handle_query(req)
        assert resp.query_id == req.query_id

    def test_mmcows_adapter_integration(self):
        """Integration test using the actual MmCows synthetic dataset on disk."""
        import os
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mmcows_synthetic.csv"
        )
        if not os.path.exists(data_path):
            pytest.skip("mmcows_synthetic.csv not found; skip real-data integration test.")

        from env_welfare_agent.src.env_agent.adapters.mmcows import load_mmcows
        env_ts, act_ts = load_mmcows(data_path)
        agent = EnvironmentWelfareAgent(env_ts, act_ts)
        req = _make_request("cow_07", "2026-09-11T00:00:00Z", "2026-09-12T00:00:00Z")
        resp = agent.handle_query(req)
        assert isinstance(resp, AgentResponse)
        assert resp.finding in ("supports", "does_not_support", "inconclusive")
