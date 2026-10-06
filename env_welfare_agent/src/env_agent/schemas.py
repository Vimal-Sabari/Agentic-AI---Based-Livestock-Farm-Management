"""
Canonical schemas for the Environment & Welfare Agent.
These dataclasses represent the normalised intermediate data that all
analytics modules exchange — fully decoupled from any raw dataset format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical column names used across the agent (never change without updating
# all adapters).
# ---------------------------------------------------------------------------

ENV_TS_COLS = {
    "TIMESTAMP": "timestamp",
    "TEMP_C": "temp_c",
    "RH_PCT": "rh_pct",
    "THI": "thi",
    "SOURCE": "source",   # "barn" | "weather"
}

COW_ACT_COLS = {
    "TIMESTAMP": "timestamp",
    "COW_ID": "cow_id",
    "ACTIVITY": "activity_metric",
    "LYING": "lying_minutes",  # nullable
}

STRESS_BAND = Literal["comfort", "mild", "moderate", "severe", "emergency", "unknown"]


@dataclass
class EnvTimeseries:
    """
    Normalised environmental time series.

    Fields
    ------
    df : pd.DataFrame
        Columns: timestamp (tz-aware), temp_c, rh_pct, thi (float, may be NaN),
        source ("barn" | "weather"), plus any extra columns from the raw data.
        Resampled to the configured frequency; gaps filled with NaN, not forward-filled.
    resample_freq : str
        Pandas offset alias, e.g. "1h".
    source_file : str
        Origin path or synthetic label for provenance.
    coverage_pct : float
        Fraction of expected rows that actually have data (0-100).
    notes : list[str]
        Adapter-level quality notes (e.g. column-name mappings, missing columns).
    """

    df: pd.DataFrame
    resample_freq: str
    source_file: str = "unknown"
    coverage_pct: float = 100.0
    notes: List[str] = field(default_factory=list)


@dataclass
class CowActivityTimeseries:
    """
    Normalised per-cow activity time series for herd-wide comparison.

    Fields
    ------
    df : pd.DataFrame
        Columns: timestamp (tz-aware), cow_id (str), activity_metric (float),
        lying_minutes (float, nullable).
    resample_freq : str
        Pandas offset alias.
    source_file : str
    coverage_pct : float
    notes : list[str]
    """

    df: pd.DataFrame
    resample_freq: str
    source_file: str = "unknown"
    coverage_pct: float = 100.0
    notes: List[str] = field(default_factory=list)


@dataclass
class ExposureMetrics:
    """
    Computed environmental exposure metrics for a given time window.
    All metrics computed relative to the canonical THI column.
    """

    mean_thi: float
    max_thi: float
    min_thi: float
    mean_temp_c: float
    mean_rh_pct: float
    # Heat-stress band hour counts
    hours_mild: float
    hours_moderate: float
    hours_severe: float
    hours_emergency: float
    # Cumulative load above mild threshold (sum of THI - threshold)
    cumulative_heat_load: float
    # Night-time recovery
    night_min_thi: float      # NaN if no night rows
    poor_night_recovery: bool  # True when night THI stays above mild threshold
    # Rate of change (°THI/hr), max over rolling 1-h windows
    max_thi_rate_per_hour: float
    # Data completeness for this window
    coverage_pct: float
    # Dominant stress band (most common non-comfort band)
    dominant_stress_band: str


@dataclass
class BaselineStats:
    """Statistical summary of baseline period metrics, diurnal-aware."""

    mean_thi: float
    std_thi: float
    p10_thi: float
    p90_thi: float
    mean_temp_c: float
    mean_rh_pct: float
    # Diurnal profile: mapping hour_of_day (0-23) -> mean_thi
    diurnal_thi_profile: dict  # {int: float}
    coverage_pct: float
    n_hours: int


@dataclass
class HerdDeviationResult:
    """Result of the herd-wide activity deviation check."""

    cows_analyzed: int
    cows_deviating: int
    fraction_deviating: float
    is_herd_wide: bool
    # Target cow's own z-score vs its baseline
    target_cow_zscore: Optional[float]
    # Whether the target cow deviates MORE than the herd median (weakens env explanation)
    target_exceeds_herd_median: bool
    # Per-cow deviation detail (cow_id -> zscore or pct_change)
    per_cow_deviations: dict  # {str: float}


@dataclass
class AlignmentResult:
    """Result of temporal alignment (lagged cross-correlation THI ↔ activity)."""

    best_lag_hours: int
    best_r: float          # Pearson r at best lag
    lags_tested: List[int]
    r_values: List[float]
    n_points: int           # number of overlapping points used
    sufficient_data: bool   # False when n_points < min_points_for_correlation


@dataclass
class VerdictBreakdown:
    """Transparent, auditable verdict scoring breakdown."""

    env_signal_score: float         # 0-1
    temporal_alignment_score: float # 0-1
    herd_wide_score: float          # 0-1
    data_coverage_score: float      # 0-1
    weighted_total: float           # final score 0-1
    finding: Literal["supports", "does_not_support", "inconclusive"]
    evidence_strength: Literal["strong", "moderate", "weak", "none"]
    confidence: float               # 0-1
    explanation: List[str]          # human-readable reasoning per dimension
