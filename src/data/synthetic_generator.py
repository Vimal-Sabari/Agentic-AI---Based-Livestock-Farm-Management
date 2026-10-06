"""
High-Fidelity Synthetic Dairy Cow Data Generator.
Replicates the 16-cow, 14-day MmCows dataset structure with all required modalities
and realistic physiological/behavioral correlations and injected clinical/environmental scenarios.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, calculate_thi


def generate_mmcows_synthetic_dataset(
    num_cows: int = 16,
    num_days: int = 14,
    start_date: str = "2026-09-01T00:00:00Z",
    seed: int = 42,
    output_path: str = "data/mmcows_synthetic.csv"
) -> pd.DataFrame:
    """
    Generates a full 16-cow, 14-day synthetic dataset adhering to the canonical schema.
    Embeds realistic circadian rhythms, individualized baselines, and targeted anomaly scenarios.
    """
    np.random.seed(seed)
    
    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    total_hours = num_days * 24
    timestamps = [start_dt + timedelta(hours=h) for h in range(total_hours)]
    
    # 16 cows (e.g. cow_01 through cow_16)
    cow_ids = [f"cow_{i:02d}" for i in range(1, num_cows + 1)]
    
    # Individual baseline parameter distributions
    cow_profiles = {}
    for cow_id in cow_ids:
        cow_profiles[cow_id] = {
            "base_activity": np.random.uniform(42.0, 58.0),
            "base_lying_per_hr": np.random.uniform(22.0, 27.0),  # ~550-650 min/day
            "base_temp": np.random.uniform(38.45, 38.75),       # healthy core temp °C
            "base_feeding_per_hr": np.random.uniform(9.0, 13.0), # ~240 min/day
            "base_rumination_per_hr": np.random.uniform(18.0, 23.0), # ~480 min/day
            "base_intake_per_hr": np.random.uniform(0.85, 1.25),     # ~24 kg DMI / day
            "base_yield_daily": np.random.uniform(28.0, 36.0),       # ~32 kg/day
        }
    
    rows = []
    
    for t_idx, ts in enumerate(timestamps):
        day_idx = t_idx // 24
        hour_of_day = ts.hour
        
        # Diurnal environment pattern (ambient & barn)
        # Days 6 and 7 represent an extreme Heat Wave across the farm
        is_heatwave_day = (day_idx in [5, 6])
        
        if is_heatwave_day:
            barn_temp = 29.5 + 4.5 * np.sin((hour_of_day - 8) * np.pi / 12) + np.random.normal(0, 0.4)
            barn_rh = 72.0 + 8.0 * np.cos((hour_of_day - 8) * np.pi / 12) + np.random.normal(0, 1.0)
            ambient_temp = barn_temp + 2.5
        else:
            barn_temp = 18.0 + 3.5 * np.sin((hour_of_day - 8) * np.pi / 12) + np.random.normal(0, 0.4)
            barn_rh = 58.0 + 6.0 * np.cos((hour_of_day - 8) * np.pi / 12) + np.random.normal(0, 1.0)
            ambient_temp = barn_temp - 1.0
            
        barn_rh = float(np.clip(barn_rh, 30.0, 95.0))
        thi = calculate_thi(barn_temp, barn_rh)
        
        # Feed availability (Drop on Day 8 afternoon across barn)
        bunk_empty_event = (day_idx == 8 and 10 <= hour_of_day <= 15)
        feed_avail = 20.0 + np.random.uniform(0, 5) if bunk_empty_event else 90.0 + np.random.uniform(-5, 5)
        feed_avail = float(np.clip(feed_avail, 10.0, 100.0))
        
        for cow_id in cow_ids:
            prof = cow_profiles[cow_id]
            
            # Circadian cycle multipliers
            # Cows rest more at night (22:00 - 05:00), feed more in morning (06:00 - 09:00) and evening (16:00 - 19:00)
            is_night = (hour_of_day < 5 or hour_of_day >= 22)
            is_feeding_peak = (6 <= hour_of_day <= 9) or (16 <= hour_of_day <= 19)
            
            act_mult = 0.55 if is_night else (1.35 if is_feeding_peak else 1.0)
            lying_mult = 1.45 if is_night else (0.65 if is_feeding_peak else 0.95)
            feed_mult = 0.25 if is_night else (1.9 if is_feeding_peak else 0.8)
            
            # Base values for this hour
            activity = prof["base_activity"] * act_mult + np.random.normal(0, 2.5)
            lying_min = prof["base_lying_per_hr"] * lying_mult + np.random.normal(0, 1.5)
            lying_min = float(np.clip(lying_min, 0.0, 58.0))
            standing_min = 60.0 - lying_min
            
            accel_rms = 0.08 + (activity / 100.0) * 0.18 + np.random.normal(0, 0.01)
            steps = int(max(0, activity * 8 + np.random.normal(0, 15)))
            
            body_temp = prof["base_temp"] + 0.15 * np.sin((hour_of_day - 14) * np.pi / 12) + np.random.normal(0, 0.05)
            
            feeding_min = prof["base_feeding_per_hr"] * feed_mult * (feed_avail / 100.0) + np.random.normal(0, 1.0)
            feeding_min = float(np.clip(feeding_min, 0.0, 50.0))
            
            rumination_min = prof["base_rumination_per_hr"] * (1.2 if is_night else 0.85) + np.random.normal(0, 1.2)
            rumination_min = float(np.clip(rumination_min, 0.0, 55.0))
            
            intake_kg = prof["base_intake_per_hr"] * (feeding_min / max(1.0, prof["base_feeding_per_hr"])) + np.random.normal(0, 0.05)
            intake_kg = float(max(0.0, intake_kg))
            
            visits = int(max(0, np.round(feeding_min / 4.0 + np.random.normal(0, 0.5))))
            
            # Daily milk yield distributed at 2 milking sessions: 06:00 and 17:00
            milk_yield = 0.0
            if hour_of_day in [6, 17]:
                milk_yield = (prof["base_yield_daily"] / 2.0) + np.random.normal(0, 0.7)
                
            milk_conduct = 5.2 + np.random.normal(0, 0.08) # normal 5.0 - 5.5 mS/cm
            
            # --- INJECT ANOMALY SCENARIOS ---
            
            # Scenario A: COW_07 - Isolated Acute Health Anomaly (Clinical Mastitis / Systemic Fever)
            # Days 10 to 12
            if cow_id == "cow_07" and (10 <= day_idx <= 12):
                severity = 1.0 if day_idx == 11 else 0.65
                body_temp += 1.25 * severity  # reaches ~39.9°C (fever)
                activity *= (1.0 - 0.32 * severity)  # activity drops -32%
                lying_min = float(np.clip(lying_min * (1.0 + 0.28 * severity), 0.0, 58.0)) # lethargy
                standing_min = 60.0 - lying_min
                rumination_min *= (1.0 - 0.38 * severity) # rumination drops -38%
                feeding_min *= (1.0 - 0.30 * severity)
                intake_kg *= (1.0 - 0.35 * severity)
                if milk_yield > 0.0:
                    milk_yield *= (1.0 - 0.36 * severity) # yield drop -36%
                milk_conduct += 1.4 * severity # high conductivity indicative of mastitis
            
            # Scenario B: COW_12 - Metabolic Drop (Subclinical Ketosis / Off-feed)
            # Days 11 to 13
            if cow_id == "cow_12" and (11 <= day_idx <= 13):
                feeding_min *= 0.65 # -35% feeding
                intake_kg *= 0.62   # -38% intake
                rumination_min *= 0.72 # -28% rumination
                activity *= 0.82
                if milk_yield > 0.0:
                    milk_yield *= 0.80 # -20% yield drop
                # body temp stays normal!
            
            # Scenario C: Herd-Wide Heat Stress Response during Days 5 & 6
            if is_heatwave_day:
                # Moderate heat stress effect on body temp & intake across herd
                body_temp += 0.45 + np.random.normal(0, 0.05)
                feeding_min *= 0.82
                rumination_min *= 0.85
                intake_kg *= 0.80
                activity *= 0.88
                if milk_yield > 0.0:
                    milk_yield *= 0.88
            
            rows.append({
                CANONICAL_COLUMNS["TIMESTAMP"]: ts.isoformat(),
                CANONICAL_COLUMNS["COW_ID"]: cow_id,
                CANONICAL_COLUMNS["ACTIVITY_INDEX"]: round(float(activity), 2),
                CANONICAL_COLUMNS["LYING_TIME_MIN"]: round(float(lying_min), 2),
                CANONICAL_COLUMNS["STANDING_TIME_MIN"]: round(float(standing_min), 2),
                CANONICAL_COLUMNS["ACCELERATION_RMS"]: round(float(accel_rms), 4),
                CANONICAL_COLUMNS["STEP_COUNT"]: int(steps),
                CANONICAL_COLUMNS["CORE_BODY_TEMP_C"]: round(float(body_temp), 2),
                CANONICAL_COLUMNS["FEEDING_DURATION_MIN"]: round(float(feeding_min), 2),
                CANONICAL_COLUMNS["RUMINATION_MIN"]: round(float(rumination_min), 2),
                CANONICAL_COLUMNS["ESTIMATED_INTAKE_KG"]: round(float(intake_kg), 2),
                CANONICAL_COLUMNS["FEED_BUNK_VISITS"]: int(visits),
                CANONICAL_COLUMNS["FEED_AVAILABILITY_PCT"]: round(float(feed_avail), 1),
                CANONICAL_COLUMNS["BARN_TEMP_C"]: round(float(barn_temp), 2),
                CANONICAL_COLUMNS["RELATIVE_HUMIDITY"]: round(float(barn_rh), 1),
                CANONICAL_COLUMNS["THI"]: round(float(thi), 2),
                CANONICAL_COLUMNS["AMBIENT_TEMP_C"]: round(float(ambient_temp), 2),
                CANONICAL_COLUMNS["VENTILATION_ACTIVE"]: 1 if is_heatwave_day else 0,
                CANONICAL_COLUMNS["MILK_YIELD_KG"]: round(float(milk_yield), 2),
                CANONICAL_COLUMNS["MILKING_DURATION_MIN"]: round(float(7.5 + np.random.normal(0, 0.5)), 2) if milk_yield > 0 else 0.0,
                CANONICAL_COLUMNS["MILK_CONDUCTIVITY"]: round(float(milk_conduct), 2),
            })
            
    df = pd.DataFrame(rows)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated synthetic dataset with {len(df)} records saved to {output_path}")
        
    return df


if __name__ == "__main__":
    generate_mmcows_synthetic_dataset()
