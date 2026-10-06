"""
Canonical Schema Definition for Multi-Agent Dairy Cow Intelligent Monitoring.
Decouples raw sensor formats (MmCows, IceTag, SMARTBOW, CSVs) from agent logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field


# Canonical column names used across all agents
CANONICAL_COLUMNS = {
    # Timestamps & IDs
    "TIMESTAMP": "timestamp",
    "COW_ID": "cow_id",
    
    # Health & Behavior Stream
    "ACTIVITY_INDEX": "activity_index",            # standardized activity score (0-100 or sensor counts)
    "LYING_TIME_MIN": "lying_time_min",            # minutes per window/day spent lying
    "STANDING_TIME_MIN": "standing_time_min",      # minutes spent standing
    "ACCELERATION_RMS": "acceleration_rms",        # root mean square acceleration (g or m/s^2)
    "STEP_COUNT": "step_count",                    # steps recorded
    "CORE_BODY_TEMP_C": "core_body_temp_c",        # core temperature in degrees Celsius (reticular/ear sensor)
    
    # Feeding & Nutrition Stream
    "FEEDING_DURATION_MIN": "feeding_duration_min",  # minutes feeding at feed bunk
    "RUMINATION_MIN": "rumination_min",              # minutes spent ruminating
    "ESTIMATED_INTAKE_KG": "estimated_intake_kg",    # dry matter / fresh feed intake in kg
    "FEED_BUNK_VISITS": "feed_bunk_visits",          # frequency of visits to feed lane
    "FEED_AVAILABILITY_PCT": "feed_availability_pct",# bunk fill percentage / push-up score
    
    # Barn Environment & Weather Stream
    "BARN_TEMP_C": "barn_temp_c",                  # barn interior temperature in °C
    "RELATIVE_HUMIDITY": "relative_humidity_pct",   # relative humidity in %
    "THI": "thi",                                  # Temperature-Humidity Index
    "AMBIENT_TEMP_C": "ambient_temp_c",            # outdoor temperature in °C
    "VENTILATION_ACTIVE": "ventilation_active",    # fan/mister status (0 or 1)
    
    # Production Stream
    "MILK_YIELD_KG": "milk_yield_kg",              # daily or per-milking yield in kg
    "MILKING_DURATION_MIN": "milking_duration_min",# duration in parlor/robot
    "MILK_CONDUCTIVITY": "milk_conductivity",      # electrical conductivity (mS/cm)
}


def calculate_thi(temp_c: float, humidity_pct: float) -> float:
    """
    Standard dairy cattle Temperature-Humidity Index (THI) formula (NRC / Thom):
    THI = (1.8 * T + 32) - (0.55 - 0.0055 * RH) * (1.8 * T - 26)
    """
    if pd.isna(temp_c) or pd.isna(humidity_pct):
        return float("nan")
    t = float(temp_c)
    rh = float(humidity_pct)
    thi = (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)
    return round(thi, 2)


def get_thi_category(thi: float) -> str:
    """Classifies THI into heat stress bands."""
    if pd.isna(thi):
        return "Unknown"
    if thi < 68.0:
        return "Comfort (No Stress)"
    if thi < 72.0:
        return "Mild Stress"
    if thi < 80.0:
        return "Moderate Stress"
    if thi < 90.0:
        return "Severe Stress"
    return "Emergency Stress"
