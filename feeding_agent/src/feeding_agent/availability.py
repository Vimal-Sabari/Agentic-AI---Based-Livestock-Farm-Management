"""
Feed availability analysis and bunk fill restriction checking.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from feeding_agent.src.feeding_agent.config import AvailabilityConfig
from feeding_agent.src.feeding_agent.schemas import AvailabilityResult, FeedingTimeseries

logger = logging.getLogger(__name__)


def compute_availability(
    ts: FeedingTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    cfg: Optional[AvailabilityConfig] = None,
) -> AvailabilityResult:
    """
    Evaluates feed bunk fill/availability during the anomaly window.
    """
    cfg = cfg or AvailabilityConfig()

    has_avail_stream = ts.stream_availability.get("availability", False)
    if not has_avail_stream or "feed_availability_pct" not in ts.df.columns:
        return AvailabilityResult(
            stream_exists=False,
            mean_availability=None,
            baseline_availability=None,
            restricted=False,
            status="unknown",
        )

    df = ts.df
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)

    cow_anom = df[(df["cow_id"] == cow_id) & (df["timestamp"] >= anomaly_start) & (df["timestamp"] < anomaly_end)]
    if cow_anom.empty:
        return AvailabilityResult(
            stream_exists=True,
            mean_availability=None,
            baseline_availability=None,
            restricted=False,
            status="unknown",
        )

    avail_vals = cow_anom["feed_availability_pct"].dropna()
    if avail_vals.empty:
        return AvailabilityResult(
            stream_exists=True,
            mean_availability=None,
            baseline_availability=None,
            restricted=False,
            status="unknown",
        )

    mean_avail = float(avail_vals.mean())

    # Baseline availability
    base_df = df[(df["cow_id"] == cow_id) & (df["timestamp"] < anomaly_start)]
    base_avail_val = None
    if not base_df.empty and "feed_availability_pct" in base_df.columns:
        b_vals = base_df["feed_availability_pct"].dropna()
        if not b_vals.empty:
            base_avail_val = float(b_vals.mean())

    restricted = mean_avail < cfg.restricted_below_pct
    status = "restricted" if restricted else "normal"

    return AvailabilityResult(
        stream_exists=True,
        mean_availability=mean_avail,
        baseline_availability=base_avail_val,
        restricted=restricted,
        status=status,
    )
