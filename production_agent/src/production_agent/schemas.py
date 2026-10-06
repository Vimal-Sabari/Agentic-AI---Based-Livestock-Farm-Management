"""
Canonical schemas for the Production Agent.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict
import pandas as pd


PROD_TS_COLS = {
    "TIMESTAMP": "timestamp",
    "COW_ID": "cow_id",
    "MILK_YIELD": "milk_yield_kg",
    "SESSION": "session",             # "morning" | "evening" | "unknown"
    "FAT_PCT": "fat_pct",             # nullable
    "PROTEIN_PCT": "protein_pct",     # nullable
    "SCC": "scc",                     # nullable
    "DURATION": "milking_duration_min"
}


@dataclass
class ProductionTimeseries:
    """Normalised per-milking production time series."""
    df: pd.DataFrame
    source_file: str = "unknown"
    coverage_pct: float = 100.0
    notes: List[str] = field(default_factory=list)


@dataclass
class BaselineStats:
    """Robust baseline summary."""
    mean_yield_daily: float
    std_yield_daily: float
    median_yield_daily: float
    n_days: int
    coverage_pct: float
    has_composition: bool


@dataclass
class DeviationResult:
    """Result of comparing anomaly window to baseline."""
    mean_yield_daily: float
    delta_kg: float
    delta_pct: float
    z_score: float
    days_persisted: int
    is_significant_drop: bool


@dataclass
class TrendResult:
    """Trend analysis within the anomaly window."""
    slope_kg_per_day: float
    direction: Literal["rising", "stable", "falling"]
    change_point_detected: bool
    change_point_date: Optional[str]


@dataclass
class SessionResult:
    """Analysis of morning vs evening milk yield."""
    has_sessions: bool
    morning_avg: float
    evening_avg: float
    morning_delta_pct: float
    evening_delta_pct: float


@dataclass
class CompositionResult:
    has_composition: bool
    fat_delta_pct: float
    protein_delta_pct: float
    scc_delta_pct: float


@dataclass
class HerdDeviationResult:
    cows_analyzed: int
    cows_deviating: int
    fraction_deviating: float
    is_herd_wide: bool


@dataclass
class VerdictBreakdown:
    finding: Literal["supports", "does_not_support", "inconclusive"]
    evidence_strength: Literal["strong", "moderate", "weak", "none"]
    confidence: float
    weighted_total: float
    explanation: List[str]
