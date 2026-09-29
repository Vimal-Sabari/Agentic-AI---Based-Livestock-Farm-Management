"""
Baseline statistics derivation – diurnal-aware.

Given an EnvTimeseries and a baseline window (start/end or N preceding days),
this module computes distributional statistics of THI/temp/humidity
while accounting for the diurnal (hour-of-day) profile.

A normal afternoon peak should NOT be flagged as anomalous.  The approach:
  1. Compute hour-of-day mean THI over the baseline period → diurnal_profile.
  2. For anomaly-vs-baseline comparison, match each anomaly hour to the same
     hour-of-day in the baseline period (diurnal-paired comparison).

No Streamlit or LangGraph imports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.config import BaselineConfig
from env_welfare_agent.src.env_agent.schemas import ENV_TS_COLS, BaselineStats, EnvTimeseries

logger = logging.getLogger(__name__)


def derive_baseline(
    env_ts: EnvTimeseries,
    anomaly_start: datetime,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
    cfg: Optional[BaselineConfig] = None,
) -> BaselineStats:
    """
    Compute baseline statistics from the baseline window.

    If ``baseline_start``/``baseline_end`` are None, falls back to the N days
    immediately preceding ``anomaly_start`` (configurable via ``cfg.default_days_before_anomaly``).

    Parameters
    ----------
    env_ts : EnvTimeseries
        Full canonical time series (entire dataset).
    anomaly_start : datetime
        Start of the anomaly window (tz-aware).
    baseline_start / baseline_end : datetime, optional
        Explicit baseline window bounds (tz-aware).
    cfg : BaselineConfig, optional

    Returns
    -------
    BaselineStats
        If no data in the baseline window, returns a stats object with NaN means
        and coverage_pct=0.
    """
    cfg = cfg or BaselineConfig()
    df = env_ts.df.copy()
    thi_col = ENV_TS_COLS["THI"]
    temp_col = ENV_TS_COLS["TEMP_C"]
    rh_col = ENV_TS_COLS["RH_PCT"]
    ts_col = ENV_TS_COLS["TIMESTAMP"]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.sort_values(ts_col)

    # Ensure anomaly_start is tz-aware
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)

    # Determine baseline window
    if baseline_start is None or baseline_end is None:
        b_end = anomaly_start
        b_start = anomaly_start - timedelta(days=cfg.default_days_before_anomaly)
        logger.info(
            "No explicit baseline window; using %d days before anomaly_start: [%s, %s]",
            cfg.default_days_before_anomaly, b_start, b_end,
        )
    else:
        b_start = baseline_start if baseline_start.tzinfo else baseline_start.replace(tzinfo=timezone.utc)
        b_end = baseline_end if baseline_end.tzinfo else baseline_end.replace(tzinfo=timezone.utc)

    mask = (df[ts_col] >= b_start) & (df[ts_col] < b_end)
    base_df = df[mask].copy()

    if base_df.empty:
        logger.warning("Baseline window [%s, %s] contains no data.", b_start, b_end)
        # Return NaN stats with coverage=0
        return BaselineStats(
            mean_thi=float("nan"),
            std_thi=float("nan"),
            p10_thi=float("nan"),
            p90_thi=float("nan"),
            mean_temp_c=float("nan"),
            mean_rh_pct=float("nan"),
            diurnal_thi_profile={},
            coverage_pct=0.0,
            n_hours=0,
        )

    # Coverage
    expected_hours = max(1, int((b_end - b_start).total_seconds() / 3600))
    n_hours = base_df[thi_col].notna().sum()
    coverage = round(100.0 * n_hours / expected_hours, 1)

    # Aggregate statistics
    thi_vals = base_df[thi_col].dropna()
    temp_vals = base_df[temp_col].dropna() if temp_col in base_df else pd.Series(dtype=float)
    rh_vals = base_df[rh_col].dropna() if rh_col in base_df else pd.Series(dtype=float)

    mean_thi = float(thi_vals.mean()) if len(thi_vals) else float("nan")
    std_thi = float(thi_vals.std(ddof=1)) if len(thi_vals) > 1 else float("nan")
    p10_thi = float(thi_vals.quantile(0.10)) if len(thi_vals) else float("nan")
    p90_thi = float(thi_vals.quantile(0.90)) if len(thi_vals) else float("nan")
    mean_temp_c = float(temp_vals.mean()) if len(temp_vals) else float("nan")
    mean_rh_pct = float(rh_vals.mean()) if len(rh_vals) else float("nan")

    # Diurnal profile: hour-of-day (0-23) → mean THI in baseline period
    base_df = base_df.copy()
    base_df["_hour"] = base_df[ts_col].dt.hour
    diurnal_raw = base_df.groupby("_hour")[thi_col].mean()
    diurnal_profile = {int(h): round(float(v), 2) for h, v in diurnal_raw.items() if not np.isnan(v)}

    logger.info(
        "Baseline derived: mean_thi=%.1f ± %.1f, coverage=%.0f%%, n_hours=%d",
        mean_thi, std_thi if not np.isnan(std_thi) else 0, coverage, n_hours,
    )

    return BaselineStats(
        mean_thi=round(mean_thi, 2),
        std_thi=round(std_thi, 2) if not np.isnan(std_thi) else float("nan"),
        p10_thi=round(p10_thi, 2),
        p90_thi=round(p90_thi, 2),
        mean_temp_c=round(mean_temp_c, 2),
        mean_rh_pct=round(mean_rh_pct, 2),
        diurnal_thi_profile=diurnal_profile,
        coverage_pct=coverage,
        n_hours=int(n_hours),
    )


def diurnal_delta(
    env_ts: EnvTimeseries,
    baseline_stats: BaselineStats,
    anomaly_start: datetime,
    anomaly_end: datetime,
) -> dict:
    """
    Compute hour-matched anomaly vs. baseline THI deltas.
    For each hour in the anomaly window, subtract the baseline mean THI at the
    same hour of day.  Returns {hour_of_day: delta} dict.

    This avoids flagging a normal 14:00 temperature peak as anomalous.
    """
    df = env_ts.df.copy()
    ts_col = ENV_TS_COLS["TIMESTAMP"]
    thi_col = ENV_TS_COLS["THI"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)

    mask = (df[ts_col] >= anomaly_start) & (df[ts_col] < anomaly_end)
    anom_df = df[mask].copy()
    if anom_df.empty or not baseline_stats.diurnal_thi_profile:
        return {}

    anom_df["_hour"] = anom_df[ts_col].dt.hour
    anom_diurnal = anom_df.groupby("_hour")[thi_col].mean()

    deltas = {}
    for hour, anom_val in anom_diurnal.items():
        base_val = baseline_stats.diurnal_thi_profile.get(int(hour))
        if base_val is not None and not np.isnan(anom_val):
            deltas[int(hour)] = round(float(anom_val) - float(base_val), 2)
    return deltas
