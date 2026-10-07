"""
Rumination metrics calculation and proxy handling.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from feeding_agent.src.feeding_agent.config import RuminationConfig
from feeding_agent.src.feeding_agent.schemas import BaselineStats, DeviationResult, FeedingTimeseries, RuminationResult

logger = logging.getLogger(__name__)


def compute_rumination(
    ts: FeedingTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    baseline: BaselineStats,
    deviation: DeviationResult,
    cfg: Optional[RuminationConfig] = None,
) -> RuminationResult:
    """
    Evaluates rumination metrics and handles absent telemetry or proxy mode.
    """
    cfg = cfg or RuminationConfig()

    has_rum_stream = ts.stream_availability.get("rumination", False)

    if not has_rum_stream:
        if not cfg.allow_proxy:
            return RuminationResult(
                available=False,
                proxy_used=False,
                notes="Rumination telemetry absent.",
            )
        else:
            # Proxy calculation: low activity after feeding estimate (if activity/standing streams exist)
            df = ts.df
            if anomaly_start.tzinfo is None:
                anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
            if anomaly_end.tzinfo is None:
                anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)

            cow_df = df[(df["cow_id"] == cow_id) & (df["timestamp"] >= anomaly_start) & (df["timestamp"] < anomaly_end)]
            proxy_val = 0.0
            if not cow_df.empty and "lying_time_min" in cow_df.columns and "feeding_minutes" in cow_df.columns:
                # Approximate rumination as 80% of quiet lying time outside feeding
                quiet_lying = (cow_df["lying_time_min"] - cow_df["feeding_minutes"].clip(lower=0)).clip(lower=0)
                proxy_val = float(quiet_lying.sum() * 0.8 * (24.0 / max(len(cow_df), 1)))

            return RuminationResult(
                available=True,
                proxy_used=True,
                rumination_daily_val=proxy_val,
                baseline_val=baseline.rumination_median or 450.0,
                delta_pct=((proxy_val - (baseline.rumination_median or 450.0)) / (baseline.rumination_median or 450.0)) * 100.0,
                notes="Rumination derived from quiet lying proxy.",
            )

    return RuminationResult(
        available=True,
        proxy_used=False,
        rumination_daily_val=deviation.rumination_daily_val,
        baseline_val=deviation.rumination_baseline_val,
        delta_pct=deviation.rumination_delta_pct,
        ratio_val=deviation.ratio_val,
        ratio_baseline=deviation.ratio_baseline,
        ratio_delta_pct=deviation.ratio_delta_pct,
        notes="Direct sensor telemetry.",
    )
