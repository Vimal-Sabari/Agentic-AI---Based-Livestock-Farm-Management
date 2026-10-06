"""
Synthetic data generator for offline demo and testing.
Produces 16 cows × 14 days × hourly records (5376 env rows, 86016 cow-activity rows).

Three scripted scenarios (controllable via parameters):
  (a) herd_heat_stress  – barn THI surges to 82-88 for 48 h; all cows show activity drop
  (b) single_sick_cow   – normal environment; only cow_01 shows a 35 % activity drop
  (c) mixed_ambiguous   – moderate THI rise (72-78); ~40 % of cows show partial drop

The generator uses numpy random seeds for reproducibility.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    ENV_TS_COLS,
    CowActivityTimeseries,
    EnvTimeseries,
)
from env_welfare_agent.src.env_agent.thi import compute_thi

logger = logging.getLogger(__name__)

Scenario = Literal["herd_heat_stress", "single_sick_cow", "mixed_ambiguous"]


def generate_synthetic(
    scenario: Scenario = "herd_heat_stress",
    n_cows: int = 16,
    n_days: int = 14,
    start_date: str = "2026-09-01",
    resample_freq: str = "1h",
    seed: int = 42,
    anomaly_day_start: int = 8,   # 0-indexed day number when anomaly begins
    anomaly_duration_days: int = 2,
) -> tuple[EnvTimeseries, CowActivityTimeseries]:
    """
    Generate synthetic env and activity data with a controllable anomaly event.

    Returns
    -------
    (EnvTimeseries, CowActivityTimeseries)
        Ready for analysis without any file I/O.
    """
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    total_hours = n_days * 24
    timestamps = [start + timedelta(hours=h) for h in range(total_hours)]
    ts_index = pd.DatetimeIndex(timestamps, tz="UTC")
    cow_ids = [f"cow_{i:02d}" for i in range(1, n_cows + 1)]
    anomaly_hours = set(range(anomaly_day_start * 24, (anomaly_day_start + anomaly_duration_days) * 24))

    # ------------------------------------------------------------------
    # Barn environment
    # ------------------------------------------------------------------
    hour_of_day = np.array([t.hour for t in timestamps], dtype=float)
    # Diurnal cycle: temp peaks ~14:00
    base_temp = 17.0 + 5.0 * np.sin(np.pi * (hour_of_day - 6) / 12.0)
    base_rh = 62.0 - 4.0 * np.sin(np.pi * (hour_of_day - 6) / 12.0)
    noise_t = rng.normal(0, 0.5, total_hours)
    noise_rh = rng.normal(0, 1.5, total_hours)
    temp_arr = base_temp + noise_t
    rh_arr = np.clip(base_rh + noise_rh, 30, 98)

    # Inject environmental anomaly
    if scenario in ("herd_heat_stress", "mixed_ambiguous"):
        for h in anomaly_hours:
            if h < total_hours:
                if scenario == "herd_heat_stress":
                    temp_arr[h] += 12.0 + rng.normal(0, 0.8)
                    rh_arr[h] = np.clip(rh_arr[h] + 15.0, 30, 98)
                else:  # mixed_ambiguous – moderate rise
                    temp_arr[h] += 6.0 + rng.normal(0, 0.5)
                    rh_arr[h] = np.clip(rh_arr[h] + 8.0, 30, 98)

    thi_arr = compute_thi(pd.Series(temp_arr), pd.Series(rh_arr)).values

    env_df = pd.DataFrame(
        {
            ENV_TS_COLS["TIMESTAMP"]: ts_index,
            ENV_TS_COLS["TEMP_C"]: temp_arr,
            ENV_TS_COLS["RH_PCT"]: rh_arr,
            ENV_TS_COLS["THI"]: thi_arr,
            ENV_TS_COLS["SOURCE"]: "barn",
            "ambient_temp_c": temp_arr - 2.0 + rng.normal(0, 0.3, total_hours),
        }
    )

    # ------------------------------------------------------------------
    # Per-cow activity (individual baselines)
    # ------------------------------------------------------------------
    cow_baselines = {cid: rng.uniform(42, 58) for cid in cow_ids}
    cow_lying_bases = {cid: rng.uniform(22, 28) for cid in cow_ids}

    act_rows = []
    for h, ts in enumerate(timestamps):
        hour = ts.hour
        in_anomaly = h in anomaly_hours
        # Diurnal: lower activity at night
        diurnal_factor = 0.7 if (hour < 5 or hour >= 22) else 1.1

        for cid in cow_ids:
            base_act = cow_baselines[cid] * diurnal_factor
            base_lying = cow_lying_bases[cid] / 24.0 * 60.0  # per-hour minutes

            # Apply scenario-specific activity modification
            act_mod = 1.0
            if in_anomaly:
                if scenario == "herd_heat_stress":
                    act_mod = rng.uniform(0.60, 0.75)  # all cows drop 25-40%
                elif scenario == "single_sick_cow" and cid == "cow_01":
                    act_mod = rng.uniform(0.60, 0.70)  # only cow_01 sick
                elif scenario == "mixed_ambiguous":
                    # ~40 % of cows deviate (deterministic based on cow index)
                    cow_idx = int(cid.split("_")[1])
                    if cow_idx % 3 == 0:  # cows 03, 06, 09, 12, 15 → deviate
                        act_mod = rng.uniform(0.72, 0.82)
                    else:
                        act_mod = rng.uniform(0.92, 1.05)  # mostly normal

            act = base_act * act_mod + rng.normal(0, 2.0)
            lying = base_lying * (1.0 / max(act_mod, 0.1)) + rng.normal(0, 1.5)
            act_rows.append(
                {
                    COW_ACT_COLS["TIMESTAMP"]: ts,
                    COW_ACT_COLS["COW_ID"]: cid,
                    COW_ACT_COLS["ACTIVITY"]: float(np.clip(act, 0, 100)),
                    COW_ACT_COLS["LYING"]: float(np.clip(lying, 0, 60)),
                }
            )

    act_df = pd.DataFrame(act_rows)
    act_df[COW_ACT_COLS["TIMESTAMP"]] = pd.to_datetime(act_df[COW_ACT_COLS["TIMESTAMP"]], utc=True)

    n_env = len(env_df)
    n_act_total = len(act_df)

    logger.info(
        "Synthetic data generated: scenario=%s, %d env rows, %d cow-activity rows.",
        scenario, n_env, n_act_total,
    )

    return (
        EnvTimeseries(
            df=env_df,
            resample_freq=resample_freq,
            source_file=f"synthetic:{scenario}",
            coverage_pct=100.0,
            notes=[f"Synthetic scenario={scenario}, anomaly_days={anomaly_day_start}-{anomaly_day_start+anomaly_duration_days}"],
        ),
        CowActivityTimeseries(
            df=act_df,
            resample_freq=resample_freq,
            source_file=f"synthetic:{scenario}",
            coverage_pct=100.0,
            notes=[f"Synthetic scenario={scenario}"],
        ),
    )
