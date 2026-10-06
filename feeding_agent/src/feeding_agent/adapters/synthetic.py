"""
Deterministic synthetic dataset generator supporting multiple evaluation scenarios.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries

logger = logging.getLogger(__name__)


def generate_synthetic(
    scenario: str = "stable",
    n_cows: int = 16,
    n_days: int = 14,
    seed: int = 42,
    start: str = "2026-09-01",
) -> FeedingTimeseries:
    """
    Generates a realistic synthetic time series dataset for feeding and nutrition monitoring.
    Deterministic given the random seed.
    Anomaly changes are applied strictly to the last 2 days (48 hours).
    """
    if scenario == "short_baseline":
        # 3 days of history before the 2-day anomaly window -> total 5 days
        n_days = 5

    rng = np.random.default_rng(seed)
    cow_ids = [f"cow_{i+1:02d}" for i in range(n_cows)]

    timestamps = pd.date_range(start=f"{start}T00:00:00Z", periods=n_days * 24, freq="1h", tz="UTC")

    records = []

    for ts in timestamps:
        hour = ts.hour
        is_anomaly_period = (ts >= timestamps[-48])  # Last 2 days (48 hours)

        # Diurnal shapes
        # Fresh feed delivery peaks around 06:00 and 16:00 (feeding 3 to 6 hours/day total)
        feed_diurnal = 10.0 + 20.0 * np.exp(-0.5 * ((hour - 6) / 1.5) ** 2) + 15.0 * np.exp(-0.5 * ((hour - 16) / 1.5) ** 2)
        # Rumination peaks at night (22:00 - 04:00, roughly 7 to 9 hours/day total)
        rum_diurnal = 15.0 + 20.0 * (1.0 if (hour >= 22 or hour <= 4) else 0.2)

        for cow_id in cow_ids:
            cow_num = int(cow_id.split("_")[1])

            # Individual baseline noise
            base_feed_noise = rng.normal(0, 2.0)
            base_rum_noise = rng.normal(0, 3.0)

            feeding_min = max(0.0, feed_diurnal + base_feed_noise + rng.normal(0, 1.5))
            rumination_min = max(0.0, rum_diurnal + base_rum_noise + rng.normal(0, 2.0))
            intake_kg = max(0.0, (feeding_min / 15.0) * 2.5 + rng.normal(0, 0.2))
            visits = max(0.0, float(rng.poisson(1.0 if feeding_min > 5 else 0.2)))
            availability_pct = float(np.clip(85.0 + rng.normal(0, 5.0), 0.0, 100.0))

            # Apply scenario alterations strictly to the anomaly period (last 48 hours)
            if is_anomaly_period:
                if scenario == "cow_specific_drop" and cow_id == "cow_01":
                    feeding_min *= 0.75
                    rumination_min *= 0.70
                    intake_kg *= 0.75
                    visits *= 0.75
                elif scenario == "herd_wide_drop":
                    # ~60% of cows experience feeding drop (cause: environment)
                    if cow_num <= int(round(n_cows * 0.6)):
                        feeding_min *= 0.80
                        intake_kg *= 0.80
                elif scenario == "rumination_only_drop" and cow_id == "cow_01":
                    rumination_min *= 0.70
                elif scenario == "restricted_availability":
                    availability_pct = float(np.clip(35.0 + rng.normal(0, 5.0), 0.0, 100.0))
                    if cow_id == "cow_01":
                        feeding_min *= 0.75
                        intake_kg *= 0.75
                elif scenario == "abnormal_increase" and cow_id == "cow_01":
                    feeding_min *= 1.30
                    intake_kg *= 1.30
                    visits *= 1.30

            records.append({
                "timestamp": ts,
                "cow_id": cow_id,
                "feeding_minutes": min(60.0, feeding_min),
                "rumination_minutes": min(60.0, rumination_min),
                "intake_kg": intake_kg,
                "feeding_visits": visits,
                "feed_availability_pct": availability_pct,
            })

    df = pd.DataFrame(records)

    stream_avail = {
        "rumination": True,
        "intake": True,
        "visits": True,
        "availability": True,
    }

    if scenario == "missing_rumination":
        df["rumination_minutes"] = np.nan
        stream_avail["rumination"] = False

    if scenario == "sparse_coverage":
        # Drop ~60% of hours randomly in anomaly window for cow_01
        anom_mask = (df["timestamp"] >= timestamps[-48]) & (df["cow_id"] == "cow_01")
        anom_indices = df[anom_mask].index
        drop_indices = rng.choice(anom_indices, size=int(len(anom_indices) * 0.6), replace=False)
        df.loc[drop_indices, "feeding_minutes"] = np.nan

    total_valid = df["feeding_minutes"].notna().sum()
    coverage_pct = (total_valid / float(len(df))) * 100.0

    return FeedingTimeseries(
        df=df,
        resample_freq="1h",
        source_file=f"synthetic://{scenario}",
        coverage_pct=coverage_pct,
        notes=f"Generated synthetic scenario '{scenario}'.",
        stream_availability=stream_avail,
    )
