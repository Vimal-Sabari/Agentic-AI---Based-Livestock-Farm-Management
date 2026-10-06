"""
Unit tests for deviation metric calculation.
"""
from datetime import datetime, timezone
import pytest

from feeding_agent.src.feeding_agent.config import DeviationConfig
from feeding_agent.src.feeding_agent.deviation import _calc_stream_dev, compute_deviation
from feeding_agent.src.feeding_agent.schemas import BaselineStats


def test_calc_stream_dev_exact_values():
    cfg = DeviationConfig(decline_pct_threshold=15.0, increase_pct_threshold=25.0, zscore_threshold=1.5, relative_spread_floor=0.05)
    # Hand-made data: observed=200, median=250, mad=20
    # delta_pct = (200 - 250) / 250 * 100 = -20.0%
    # spread_floor = 0.05 * 250 = 12.5. max(20, 12.5) = 20.
    # zscore = (200 - 250) / 20 = -2.5
    delta, zscore, direction, sig = _calc_stream_dev(observed_val=200.0, baseline_median=250.0, baseline_mad=20.0, cfg=cfg)

    assert delta == pytest.approx(-20.0)
    assert zscore == pytest.approx(-2.5)
    assert direction == "decline"
    assert sig is True


def test_calc_stream_dev_flat_baseline_no_inf():
    cfg = DeviationConfig(decline_pct_threshold=15.0, zscore_threshold=1.5, relative_spread_floor=0.05)
    # Flat baseline: median = 300.0, MAD = 0.0
    delta, zscore, direction, sig = _calc_stream_dev(observed_val=210.0, baseline_median=300.0, baseline_mad=0.0, cfg=cfg)

    assert delta == pytest.approx(-30.0)
    # Denom = max(0.0, 0.05 * 300) = 15.0
    # zscore = (210 - 300) / 15 = -6.0
    assert zscore == pytest.approx(-6.0)
    assert not float("-inf") == zscore
    assert direction == "decline"
    assert sig is True


def test_direction_and_significance_rules():
    cfg = DeviationConfig(decline_pct_threshold=15.0, increase_pct_threshold=25.0, zscore_threshold=1.5, relative_spread_floor=0.05)

    # 1. Delta exceeds threshold (-20%), but zscore does not (-1.0) -> not significant, stable
    d1, z1, dir1, sig1 = _calc_stream_dev(observed_val=240.0, baseline_median=300.0, baseline_mad=60.0, cfg=cfg)
    assert d1 == pytest.approx(-20.0)
    assert z1 == pytest.approx(-1.0)
    assert sig1 is False
    assert dir1 == "stable"

    # 2. Zscore exceeds threshold (-2.0), but delta does not (-10%) -> not significant, stable
    d2, z2, dir2, sig2 = _calc_stream_dev(observed_val=270.0, baseline_median=300.0, baseline_mad=15.0, cfg=cfg)
    assert d2 == pytest.approx(-10.0)
    assert z2 == pytest.approx(-2.0)
    assert sig2 is False
    assert dir2 == "stable"

    # 3. Abnormal increase: +30% delta, zscore = +3.0 -> significant increase
    d3, z3, dir3, sig3 = _calc_stream_dev(observed_val=390.0, baseline_median=300.0, baseline_mad=30.0, cfg=cfg)
    assert d3 == pytest.approx(30.0)
    assert z3 == pytest.approx(3.0)
    assert sig3 is True
    assert dir3 == "increase"
