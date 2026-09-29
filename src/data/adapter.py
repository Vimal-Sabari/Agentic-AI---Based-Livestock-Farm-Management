"""
Data Adapter Layer.
Inspects raw files, maps arbitrary column headers to the canonical schema,
computes derived metrics (such as THI), and isolates agent logic from dataset-specific variations.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.data.schema import CANONICAL_COLUMNS, calculate_thi


class DatasetAdapter:
    """
    Adapter that normalizes raw dataset schemas into the standardized canonical schema.
    Supports MmCows, IceTag/SMARTBOW, and generic CSV formats.
    """

    # Default heuristic mappings for known datasets
    DEFAULT_COLUMN_MAPPINGS: Dict[str, List[str]] = {
        CANONICAL_COLUMNS["TIMESTAMP"]: ["timestamp", "time", "date", "datetime", "record_time", "DateTime"],
        CANONICAL_COLUMNS["COW_ID"]: ["cow_id", "cow", "animal_id", "id", "CowID", "EarTag"],
        CANONICAL_COLUMNS["ACTIVITY_INDEX"]: ["activity_index", "activity", "motion", "act_score", "IceTag_Activity"],
        CANONICAL_COLUMNS["LYING_TIME_MIN"]: ["lying_time_min", "lying_time", "lying_duration", "lying_minutes", "LyingTime"],
        CANONICAL_COLUMNS["STANDING_TIME_MIN"]: ["standing_time_min", "standing_time", "standing_minutes"],
        CANONICAL_COLUMNS["ACCELERATION_RMS"]: ["acceleration_rms", "acc_rms", "accel", "rms_acc", "AccX"],
        CANONICAL_COLUMNS["STEP_COUNT"]: ["step_count", "steps", "num_steps", "StepCount"],
        CANONICAL_COLUMNS["CORE_BODY_TEMP_C"]: ["core_body_temp_c", "body_temp", "temp_c", "reticular_temp", "CoreTemp"],
        CANONICAL_COLUMNS["FEEDING_DURATION_MIN"]: ["feeding_duration_min", "feeding_duration", "feed_minutes", "bunk_time_min"],
        CANONICAL_COLUMNS["RUMINATION_MIN"]: ["rumination_min", "rumination_duration", "rumination_time", "ruminate_min"],
        CANONICAL_COLUMNS["ESTIMATED_INTAKE_KG"]: ["estimated_intake_kg", "feed_intake", "intake_kg", "dmi_kg"],
        CANONICAL_COLUMNS["FEED_BUNK_VISITS"]: ["feed_bunk_visits", "bunk_visits", "visit_count", "feeding_bouts"],
        CANONICAL_COLUMNS["FEED_AVAILABILITY_PCT"]: ["feed_availability_pct", "bunk_fill_pct", "feed_availability"],
        CANONICAL_COLUMNS["BARN_TEMP_C"]: ["barn_temp_c", "barn_temp", "indoor_temp", "t_barn"],
        CANONICAL_COLUMNS["RELATIVE_HUMIDITY"]: ["relative_humidity_pct", "humidity", "rh", "rh_pct", "barn_rh"],
        CANONICAL_COLUMNS["THI"]: ["thi", "temperature_humidity_index", "THI"],
        CANONICAL_COLUMNS["AMBIENT_TEMP_C"]: ["ambient_temp_c", "outdoor_temp", "weather_temp"],
        CANONICAL_COLUMNS["VENTILATION_ACTIVE"]: ["ventilation_active", "fans_on", "misters_on"],
        CANONICAL_COLUMNS["MILK_YIELD_KG"]: ["milk_yield_kg", "milk_yield", "yield_kg", "daily_yield", "MilkYield"],
        CANONICAL_COLUMNS["MILKING_DURATION_MIN"]: ["milking_duration_min", "milking_time", "milking_duration"],
        CANONICAL_COLUMNS["MILK_CONDUCTIVITY"]: ["milk_conductivity", "conductivity", "elec_conductivity"],
    }

    @classmethod
    def inspect_file(cls, filepath: str) -> Dict[str, any]:
        """Inspects a dataset file to extract column names, row counts, and detected mappings."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Try reading header
        if filepath.endswith(".parquet"):
            df_sample = pd.read_parquet(filepath)
        else:
            df_sample = pd.read_csv(filepath, nrows=100)

        raw_columns = list(df_sample.columns)
        detected_mapping = cls.detect_mapping(raw_columns)

        return {
            "filepath": filepath,
            "raw_columns": raw_columns,
            "detected_mapping": detected_mapping,
            "sample_rows": len(df_sample),
        }

    @classmethod
    def detect_mapping(cls, raw_columns: List[str]) -> Dict[str, str]:
        """Maps raw column names to canonical schema columns using heuristics."""
        raw_cols_lower = {col.lower().strip(): col for col in raw_columns}
        canonical_map: Dict[str, str] = {}

        for canonical_name, candidate_names in cls.DEFAULT_COLUMN_MAPPINGS.items():
            for candidate in candidate_names:
                candidate_lower = candidate.lower()
                if candidate_lower in raw_cols_lower:
                    canonical_map[raw_cols_lower[candidate_lower]] = canonical_name
                    break

        return canonical_map

    @classmethod
    def adapt_dataframe(
        cls,
        df: pd.DataFrame,
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Transforms raw DataFrame into canonical DataFrame.
        """
        mapping = custom_mapping or cls.detect_mapping(list(df.columns))
        adapted = df.rename(columns=mapping)

        # Standardize timestamp column if present
        ts_col = CANONICAL_COLUMNS["TIMESTAMP"]
        if ts_col in adapted.columns:
            adapted[ts_col] = pd.to_datetime(adapted[ts_col], utc=True)
            adapted = adapted.sort_values(by=ts_col)

        # Standardize cow_id as string
        cow_col = CANONICAL_COLUMNS["COW_ID"]
        if cow_col in adapted.columns:
            adapted[cow_col] = adapted[cow_col].astype(str)

        # Calculate THI if missing but temperature & humidity exist
        thi_col = CANONICAL_COLUMNS["THI"]
        t_col = CANONICAL_COLUMNS["BARN_TEMP_C"]
        rh_col = CANONICAL_COLUMNS["RELATIVE_HUMIDITY"]
        if (thi_col not in adapted.columns or adapted[thi_col].isna().all()) and (t_col in adapted.columns and rh_col in adapted.columns):
            adapted[thi_col] = [
                calculate_thi(t, rh) for t, rh in zip(adapted[t_col], adapted[rh_col])
            ]

        return adapted
