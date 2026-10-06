"""
Canonical schema definitions and analytical dataclasses for Feeding & Nutrition Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

FEED_TS_COLS: Dict[str, str] = {
    "timestamp": "timestamp",
    "cow_id": "cow_id",
    "feeding_minutes": "feeding_minutes",
    "feeding_visits": "feeding_visits",
    "rumination_minutes": "rumination_minutes",
    "intake_kg": "intake_kg",
    "feed_availability_pct": "feed_availability_pct",
    "group": "group",
}


@dataclass
class FeedingTimeseries:
    """Canonical representation of processed feeding time series."""
    df: pd.DataFrame
    resample_freq: str = "1h"
    source_file: str = "unknown"
    coverage_pct: float = 100.0
    notes: str = ""
    stream_availability: Dict[str, bool] = field(default_factory=lambda: {
        "rumination": True,
        "intake": True,
        "visits": True,
        "availability": True,
    })


@dataclass
class BaselineStats:
    """Summary statistics for an individual cow's baseline period."""
    n_days: int
    coverage_pct: float
    is_short: bool
    feeding_median: float
    feeding_mad: float
    rumination_median: Optional[float] = None
    rumination_mad: Optional[float] = None
    intake_median: Optional[float] = None
    intake_mad: Optional[float] = None
    visits_median: Optional[float] = None
    visits_mad: Optional[float] = None
    hourly_feeding_profile: Dict[int, float] = field(default_factory=dict)
    baseline_mean_bout_length: float = 0.0
    baseline_bouts_per_day: float = 0.0


@dataclass
class DeviationResult:
    """Deviation metrics comparing anomaly window to baseline."""
    feeding_daily_val: float
    feeding_baseline_val: float
    feeding_delta_pct: float
    feeding_zscore: float
    feeding_direction: str  # "decline", "increase", "stable"
    feeding_significant: bool

    rumination_daily_val: Optional[float] = None
    rumination_baseline_val: Optional[float] = None
    rumination_delta_pct: Optional[float] = None
    rumination_zscore: Optional[float] = None
    rumination_direction: Optional[str] = None
    rumination_significant: Optional[bool] = None

    intake_daily_val: Optional[float] = None
    intake_baseline_val: Optional[float] = None
    intake_delta_pct: Optional[float] = None
    intake_zscore: Optional[float] = None
    intake_direction: Optional[str] = None
    intake_significant: Optional[bool] = None

    visits_daily_val: Optional[float] = None
    visits_baseline_val: Optional[float] = None
    visits_delta_pct: Optional[float] = None
    visits_zscore: Optional[float] = None
    visits_direction: Optional[str] = None
    visits_significant: Optional[bool] = None

    feed_avail_val: Optional[float] = None
    feed_avail_baseline: Optional[float] = None
    feed_avail_delta_pct: Optional[float] = None

    ratio_val: Optional[float] = None
    ratio_baseline: Optional[float] = None
    ratio_delta_pct: Optional[float] = None


@dataclass
class MealStats:
    """Feeding bout and meal pattern analysis."""
    bouts_df: pd.DataFrame
    bouts_per_day: float
    mean_bout_length: float
    longest_bout: float
    peak_hour: int
    peak_hour_shift: float


@dataclass
class RuminationResult:
    """Rumination analysis results."""
    available: bool
    proxy_used: bool = False
    rumination_daily_val: Optional[float] = None
    baseline_val: Optional[float] = None
    delta_pct: Optional[float] = None
    ratio_val: Optional[float] = None
    ratio_baseline: Optional[float] = None
    ratio_delta_pct: Optional[float] = None
    notes: str = ""


@dataclass
class AvailabilityResult:
    """Feed bunk availability analysis results."""
    stream_exists: bool
    mean_availability: Optional[float] = None
    baseline_availability: Optional[float] = None
    restricted: bool = False
    status: str = "unknown"  # "normal", "restricted", "unknown"


@dataclass
class HerdResult:
    """Herd-wide comparison result (excluding target cow)."""
    cows_analyzed: int
    cows_deviating: int
    fraction_deviating: float
    is_herd_wide: bool
    notes: str = ""


@dataclass
class VerdictBreakdown:
    """Detailed score breakdown and verdict synthesis."""
    feeding_score: float
    rumination_score: float
    intake_score: float
    consistency_score: float
    coverage_score: float
    weighted_score: float
    finding: str  # "supports", "does_not_support", "inconclusive"
    evidence_strength: str  # "strong", "moderate", "weak", "none"
    confidence: float
    explanation: List[str] = field(default_factory=list)
