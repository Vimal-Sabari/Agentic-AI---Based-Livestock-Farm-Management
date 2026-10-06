"""
Deviation analysis comparing anomaly window against baseline stats.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from feeding_agent.src.feeding_agent.config import DataConfig, DeviationConfig
from feeding_agent.src.feeding_agent.schemas import BaselineStats, DeviationResult, FeedingTimeseries

logger = logging.getLogger(__name__)


def _calc_stream_dev(
    observed_val: Optional[float],
    baseline_median: Optional[float],
    baseline_mad: Optional[float],
    cfg: DeviationConfig,
    is_decline_metric: bool = True,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[bool]]:
    if observed_val is None or baseline_median is None:
        return None, None, None, None

    if baseline_median == 0.0:
        delta_pct = 0.0 if observed_val == 0.0 else (100.0 if observed_val > 0 else -100.0)
    else:
        delta_pct = ((observed_val - baseline_median) / abs(baseline_median)) * 100.0

    # Robust z-score floor
    spread = baseline_mad if (baseline_mad is not None and baseline_mad > 0) else 0.0
    floor = cfg.relative_spread_floor * abs(baseline_median)
    denom = max(spread, floor, 1e-4)

    zscore = (observed_val - baseline_median) / denom

    if delta_pct <= -cfg.decline_pct_threshold and zscore <= -cfg.zscore_threshold:
        direction = "decline"
        significant = True
    elif delta_pct >= cfg.increase_pct_threshold and zscore >= cfg.zscore_threshold:
        direction = "increase"
        significant = True
    else:
        direction = "stable"
        significant = False

    return delta_pct, zscore, direction, significant


def compute_deviation(
    ts: FeedingTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    baseline: BaselineStats,
    cfg: Optional[DeviationConfig] = None,
    data_cfg: Optional[DataConfig] = None,
) -> DeviationResult:
    """
    Computes per-day equivalent metrics in anomaly window and compares with baseline.
    """
    cfg = cfg or DeviationConfig()
    data_cfg = data_cfg or DataConfig()

    df = ts.df
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)

    cow_anom = df[(df["cow_id"] == cow_id) & (df["timestamp"] >= anomaly_start) & (df["timestamp"] < anomaly_end)].copy()

    total_hours = max(1.0, (anomaly_end - anomaly_start).total_seconds() / 3600.0)
    valid_hours = len(cow_anom["feeding_minutes"].dropna()) if not cow_anom.empty else 0
    scaling = 24.0 / max(valid_hours, 1) if valid_hours > 0 else 1.0

    # Feeding
    if not cow_anom.empty and valid_hours > 0:
        f_val = float(cow_anom["feeding_minutes"].sum() * scaling)
    else:
        f_val = 0.0

    f_delta, f_zscore, f_dir, f_sig = _calc_stream_dev(f_val, baseline.feeding_median, baseline.feeding_mad, cfg)
    f_delta = f_delta if f_delta is not None else 0.0
    f_zscore = f_zscore if f_zscore is not None else 0.0
    f_dir = f_dir if f_dir is not None else "stable"
    f_sig = f_sig if f_sig is not None else False

    # Rumination
    r_val, r_delta, r_zscore, r_dir, r_sig = None, None, None, None, None
    if ts.stream_availability.get("rumination", False) and not cow_anom.empty and "rumination_minutes" in cow_anom.columns:
        rum_non_null = cow_anom["rumination_minutes"].dropna()
        if len(rum_non_null) > 0:
            r_val = float(rum_non_null.sum() * scaling)
            r_delta, r_zscore, r_dir, r_sig = _calc_stream_dev(r_val, baseline.rumination_median, baseline.rumination_mad, cfg)

    # Intake
    i_val, i_delta, i_zscore, i_dir, i_sig = None, None, None, None, None
    if ts.stream_availability.get("intake", False) and not cow_anom.empty and "intake_kg" in cow_anom.columns:
        int_non_null = cow_anom["intake_kg"].dropna()
        if len(int_non_null) > 0:
            i_val = float(int_non_null.sum() * scaling)
            i_delta, i_zscore, i_dir, i_sig = _calc_stream_dev(i_val, baseline.intake_median, baseline.intake_mad, cfg)

    # Visits
    v_val, v_delta, v_zscore, v_dir, v_sig = None, None, None, None, None
    if ts.stream_availability.get("visits", False) and not cow_anom.empty and "feeding_visits" in cow_anom.columns:
        vis_non_null = cow_anom["feeding_visits"].dropna()
        if len(vis_non_null) > 0:
            v_val = float(vis_non_null.sum() * scaling)
            v_delta, v_zscore, v_dir, v_sig = _calc_stream_dev(v_val, baseline.visits_median, baseline.visits_mad, cfg)

    # Feed availability
    avail_val, avail_baseline, avail_delta = None, None, None
    if ts.stream_availability.get("availability", False) and not cow_anom.empty and "feed_availability_pct" in cow_anom.columns:
        av_non_null = cow_anom["feed_availability_pct"].dropna()
        if len(av_non_null) > 0:
            avail_val = float(av_non_null.mean())
            # Baseline mean availability
            base_cow_df = df[(df["cow_id"] == cow_id) & (df["timestamp"] < anomaly_start)]
            if not base_cow_df.empty and "feed_availability_pct" in base_cow_df.columns:
                b_av = base_cow_df["feed_availability_pct"].dropna()
                if len(b_av) > 0:
                    avail_baseline = float(b_av.mean())
                    avail_delta = ((avail_val - avail_baseline) / max(avail_baseline, 1e-4)) * 100.0

    # Feeding-to-rumination ratio
    ratio_val, ratio_base, ratio_delta = None, None, None
    if f_val is not None and r_val is not None and r_val > 0:
        ratio_val = f_val / r_val
    if baseline.feeding_median is not None and baseline.rumination_median is not None and baseline.rumination_median > 0:
        ratio_base = baseline.feeding_median / baseline.rumination_median
    if ratio_val is not None and ratio_base is not None and ratio_base > 0:
        ratio_delta = ((ratio_val - ratio_base) / ratio_base) * 100.0

    return DeviationResult(
        feeding_daily_val=f_val,
        feeding_baseline_val=baseline.feeding_median,
        feeding_delta_pct=f_delta,
        feeding_zscore=f_zscore,
        feeding_direction=f_dir,
        feeding_significant=f_sig,
        rumination_daily_val=r_val,
        rumination_baseline_val=baseline.rumination_median,
        rumination_delta_pct=r_delta,
        rumination_zscore=r_zscore,
        rumination_direction=r_dir,
        rumination_significant=r_sig,
        intake_daily_val=i_val,
        intake_baseline_val=baseline.intake_median,
        intake_delta_pct=i_delta,
        intake_zscore=i_zscore,
        intake_direction=i_dir,
        intake_significant=i_sig,
        visits_daily_val=v_val,
        visits_baseline_val=baseline.visits_median,
        visits_delta_pct=v_delta,
        visits_zscore=v_zscore,
        visits_direction=v_dir,
        visits_significant=v_sig,
        feed_avail_val=avail_val,
        feed_avail_baseline=avail_baseline,
        feed_avail_delta_pct=avail_delta,
        ratio_val=ratio_val,
        ratio_baseline=ratio_base,
        ratio_delta_pct=ratio_delta,
    )
