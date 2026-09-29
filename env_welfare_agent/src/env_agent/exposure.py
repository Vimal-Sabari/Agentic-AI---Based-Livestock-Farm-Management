"""
Environmental exposure metrics for a given anomaly time window.

All functions are pure (no side-effects, no I/O, no UI imports).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.config import ExposureConfig, THIThresholds
from env_welfare_agent.src.env_agent.schemas import (
    ENV_TS_COLS,
    EnvTimeseries,
    ExposureMetrics,
)
from env_welfare_agent.src.env_agent.thi import classify_thi_series

logger = logging.getLogger(__name__)


def compute_exposure(
    env_ts: EnvTimeseries,
    anomaly_start: datetime,
    anomaly_end: datetime,
    thresholds: Optional[THIThresholds] = None,
    cfg: Optional[ExposureConfig] = None,
) -> ExposureMetrics:
    """
    Compute comprehensive exposure metrics for the anomaly window.

    Parameters
    ----------
    env_ts : EnvTimeseries
        Full canonical environmental time series.
    anomaly_start / anomaly_end : datetime
        Bounds of the anomaly window (tz-aware).
    thresholds : THIThresholds, optional
    cfg : ExposureConfig, optional

    Returns
    -------
    ExposureMetrics
        All metrics are NaN-safe; coverage_pct reflects data completeness.
    """
    th = thresholds or THIThresholds()
    ecfg = cfg or ExposureConfig()

    df = env_ts.df.copy()
    ts_col = ENV_TS_COLS["TIMESTAMP"]
    thi_col = ENV_TS_COLS["THI"]
    temp_col = ENV_TS_COLS["TEMP_C"]
    rh_col = ENV_TS_COLS["RH_PCT"]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)

    mask = (df[ts_col] >= anomaly_start) & (df[ts_col] < anomaly_end)
    window = df[mask].copy()

    expected_hours = max(1, int((anomaly_end - anomaly_start).total_seconds() / 3600))
    n_valid = window[thi_col].notna().sum() if thi_col in window else 0
    coverage = round(100.0 * n_valid / expected_hours, 1)

    if window.empty or n_valid == 0:
        logger.warning("No valid THI data in anomaly window [%s, %s].", anomaly_start, anomaly_end)
        return _empty_metrics(coverage)

    thi = window[thi_col].dropna()
    temp = window[temp_col].dropna() if temp_col in window else pd.Series(dtype=float)
    rh = window[rh_col].dropna() if rh_col in window else pd.Series(dtype=float)

    # Basic statistics
    mean_thi = float(thi.mean())
    max_thi = float(thi.max())
    min_thi = float(thi.min())
    mean_temp = float(temp.mean()) if len(temp) else float("nan")
    mean_rh = float(rh.mean()) if len(rh) else float("nan")

    # Stress-band hour counts
    bands = classify_thi_series(window[thi_col], th)
    band_counts = bands.value_counts()
    hours_mild = float(band_counts.get("mild", 0))
    hours_moderate = float(band_counts.get("moderate", 0))
    hours_severe = float(band_counts.get("severe", 0))
    hours_emergency = float(band_counts.get("emergency", 0))

    # Dominant stress band (most common non-comfort band)
    stress_bands = band_counts.drop(labels=["comfort", "unknown"], errors="ignore")
    dominant_stress_band = str(stress_bands.idxmax()) if not stress_bands.empty else "comfort"

    # Cumulative heat load: sum of (THI - base_thi) for rows above threshold
    base_thi = ecfg.heat_load_base_thi
    load_series = (window[thi_col] - base_thi).clip(lower=0.0)
    cumulative_heat_load = float(load_series.sum())

    # Night-time recovery
    night_hours_set = set(ecfg.night_hours)
    night_mask = window[ts_col].dt.hour.isin(night_hours_set)
    night_rows = window[night_mask]
    if not night_rows.empty and night_rows[thi_col].notna().any():
        night_min_thi = float(night_rows[thi_col].min())
        poor_night_recovery = bool(night_min_thi >= th.mild_min)
    else:
        night_min_thi = float("nan")
        poor_night_recovery = False

    # Rate of change: max absolute hourly change in THI
    thi_full = window.set_index(ts_col)[thi_col].sort_index()
    thi_diff = thi_full.diff().abs()
    max_thi_rate = float(thi_diff.max()) if not thi_diff.empty else float("nan")

    logger.info(
        "Exposure computed: mean_thi=%.1f, max=%.1f, hours_stress=%.0f, load=%.1f, coverage=%.0f%%",
        mean_thi, max_thi, hours_mild + hours_moderate + hours_severe + hours_emergency,
        cumulative_heat_load, coverage,
    )

    return ExposureMetrics(
        mean_thi=round(mean_thi, 2),
        max_thi=round(max_thi, 2),
        min_thi=round(min_thi, 2),
        mean_temp_c=round(mean_temp, 2) if not np.isnan(mean_temp) else float("nan"),
        mean_rh_pct=round(mean_rh, 2) if not np.isnan(mean_rh) else float("nan"),
        hours_mild=hours_mild,
        hours_moderate=hours_moderate,
        hours_severe=hours_severe,
        hours_emergency=hours_emergency,
        cumulative_heat_load=round(cumulative_heat_load, 2),
        night_min_thi=round(night_min_thi, 2) if not np.isnan(night_min_thi) else float("nan"),
        poor_night_recovery=poor_night_recovery,
        max_thi_rate_per_hour=round(max_thi_rate, 2) if not np.isnan(max_thi_rate) else float("nan"),
        coverage_pct=coverage,
        dominant_stress_band=dominant_stress_band,
    )


def _empty_metrics(coverage: float) -> ExposureMetrics:
    nan = float("nan")
    return ExposureMetrics(
        mean_thi=nan, max_thi=nan, min_thi=nan,
        mean_temp_c=nan, mean_rh_pct=nan,
        hours_mild=0.0, hours_moderate=0.0, hours_severe=0.0, hours_emergency=0.0,
        cumulative_heat_load=0.0,
        night_min_thi=nan, poor_night_recovery=False,
        max_thi_rate_per_hour=nan,
        coverage_pct=coverage,
        dominant_stress_band="unknown",
    )
