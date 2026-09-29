"""
Herd-wide behavioral deviation analysis.

For each cow, compare its activity/lying in the anomaly window vs its own
baseline.  Uses z-score (or percent-change fallback when std ≈ 0).
Reports fraction deviating and whether the target cow is an outlier vs peers.

No Streamlit or LangGraph imports.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.config import HerdConfig
from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    CowActivityTimeseries,
    HerdDeviationResult,
)

logger = logging.getLogger(__name__)


def compute_herd_deviation(
    cow_act_ts: CowActivityTimeseries,
    target_cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
    baseline_days: int = 3,
    cfg: Optional[HerdConfig] = None,
) -> HerdDeviationResult:
    """
    Compute herd-wide activity deviation relative to each cow's own baseline.

    Parameters
    ----------
    cow_act_ts : CowActivityTimeseries
    target_cow_id : str
    anomaly_start / anomaly_end : datetime (tz-aware)
    baseline_start / baseline_end : datetime, optional
        If None, uses ``baseline_days`` preceding anomaly_start.
    baseline_days : int
        Fallback baseline duration in days.
    cfg : HerdConfig, optional

    Returns
    -------
    HerdDeviationResult
    """
    from datetime import timedelta

    hcfg = cfg or HerdConfig()
    df = cow_act_ts.df.copy()
    ts_col = COW_ACT_COLS["TIMESTAMP"]
    cow_col = COW_ACT_COLS["COW_ID"]
    act_col = COW_ACT_COLS["ACTIVITY"]

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    # Ensure tz-awareness
    def _tz(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    anomaly_start = _tz(anomaly_start)
    anomaly_end = _tz(anomaly_end)
    b_start = _tz(baseline_start) if baseline_start else anomaly_start - timedelta(days=baseline_days)
    b_end = _tz(baseline_end) if baseline_end else anomaly_start

    all_cows = df[cow_col].unique().tolist()
    n_cows = len(all_cows)

    if n_cows < hcfg.min_cows_for_herd_check:
        logger.warning(
            "Only %d cow(s) available; minimum %d required for herd check.",
            n_cows, hcfg.min_cows_for_herd_check,
        )
        # Try to compute at least the target cow z-score
        target_z = _cow_zscore(df, target_cow_id, anomaly_start, anomaly_end, b_start, b_end, act_col, ts_col, cow_col, hcfg)
        return HerdDeviationResult(
            cows_analyzed=n_cows,
            cows_deviating=0,
            fraction_deviating=0.0,
            is_herd_wide=False,
            target_cow_zscore=target_z,
            target_exceeds_herd_median=False,
            per_cow_deviations={},
        )

    per_cow: dict[str, float] = {}
    for cow in all_cows:
        z = _cow_zscore(df, cow, anomaly_start, anomaly_end, b_start, b_end, act_col, ts_col, cow_col, hcfg)
        if z is not None:
            per_cow[cow] = z

    if not per_cow:
        return HerdDeviationResult(
            cows_analyzed=n_cows, cows_deviating=0, fraction_deviating=0.0,
            is_herd_wide=False, target_cow_zscore=None,
            target_exceeds_herd_median=False, per_cow_deviations={},
        )

    # Count cows with significant negative deviation (activity suppression)
    z_thresh = hcfg.deviation_zscore_threshold
    cows_deviating = sum(1 for z in per_cow.values() if z <= -z_thresh)
    fraction_deviating = cows_deviating / len(per_cow)
    is_herd_wide = fraction_deviating >= hcfg.is_herd_wide_threshold

    # Target cow stats
    target_z = per_cow.get(target_cow_id)
    other_zscores = [z for c, z in per_cow.items() if c != target_cow_id]
    if other_zscores and target_z is not None:
        herd_median_z = float(np.median(other_zscores))
        target_exceeds_herd_median = bool(target_z < herd_median_z - 0.5)
    else:
        target_exceeds_herd_median = False

    logger.info(
        "Herd check: %d/%d cows deviating (%.0f%%), is_herd_wide=%s, target_z=%.2f",
        cows_deviating, len(per_cow), fraction_deviating * 100,
        is_herd_wide, target_z if target_z is not None else float("nan"),
    )

    return HerdDeviationResult(
        cows_analyzed=len(per_cow),
        cows_deviating=cows_deviating,
        fraction_deviating=round(fraction_deviating, 3),
        is_herd_wide=is_herd_wide,
        target_cow_zscore=round(target_z, 2) if target_z is not None else None,
        target_exceeds_herd_median=target_exceeds_herd_median,
        per_cow_deviations={k: round(v, 2) for k, v in per_cow.items()},
    )


def _cow_zscore(
    df: pd.DataFrame,
    cow_id: str,
    anom_start: datetime,
    anom_end: datetime,
    b_start: datetime,
    b_end: datetime,
    act_col: str,
    ts_col: str,
    cow_col: str,
    cfg: HerdConfig,
) -> Optional[float]:
    """Compute z-score of anomaly-window activity vs baseline for one cow."""
    cow_df = df[df[cow_col] == cow_id]
    base = cow_df[(cow_df[ts_col] >= b_start) & (cow_df[ts_col] < b_end)][act_col].dropna()
    anom = cow_df[(cow_df[ts_col] >= anom_start) & (cow_df[ts_col] < anom_end)][act_col].dropna()

    if base.empty or anom.empty:
        return None

    base_mean = float(base.mean())
    base_std = float(base.std(ddof=1)) if len(base) > 1 else 0.0
    anom_mean = float(anom.mean())

    if base_std < 0.5:
        # std ≈ 0: use percent-change as proxy for z-score magnitude
        if base_mean > 0:
            pct = (anom_mean - base_mean) / base_mean * 100.0
            # Normalise to z-score-like scale: treat cfg.deviation_pct_threshold as 1 sigma
            return pct / cfg.deviation_pct_threshold
        return 0.0

    return (anom_mean - base_mean) / base_std
