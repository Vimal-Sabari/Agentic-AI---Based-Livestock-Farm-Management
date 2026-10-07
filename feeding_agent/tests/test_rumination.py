"""
Unit tests for rumination analysis and proxy handling.
"""
from datetime import datetime, timezone
import pytest

from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.baseline import derive_baseline
from feeding_agent.src.feeding_agent.config import RuminationConfig
from feeding_agent.src.feeding_agent.deviation import compute_deviation
from feeding_agent.src.feeding_agent.rumination import compute_rumination


def test_missing_rumination_no_proxy():
    ts = generate_synthetic(scenario="missing_rumination", seed=42)
    astart = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    aend = datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc)

    base = derive_baseline(ts, "cow_01", astart)
    dev = compute_deviation(ts, "cow_01", astart, aend, base)

    cfg = RuminationConfig(allow_proxy=False)
    res = compute_rumination(ts, "cow_01", astart, aend, base, dev, cfg)

    assert res.available is False
    assert res.proxy_used is False
    assert res.rumination_daily_val is None
    assert res.baseline_val is None
    assert res.delta_pct is None


def test_missing_rumination_with_proxy():
    ts = generate_synthetic(scenario="missing_rumination", seed=42)
    astart = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
    aend = datetime(2026, 9, 14, 23, 0, tzinfo=timezone.utc)

    base = derive_baseline(ts, "cow_01", astart)
    dev = compute_deviation(ts, "cow_01", astart, aend, base)

    cfg = RuminationConfig(allow_proxy=True)
    res = compute_rumination(ts, "cow_01", astart, aend, base, dev, cfg)

    assert res.available is True
    assert res.proxy_used is True
    assert res.rumination_daily_val is not None
    assert "proxy" in res.notes.lower()
