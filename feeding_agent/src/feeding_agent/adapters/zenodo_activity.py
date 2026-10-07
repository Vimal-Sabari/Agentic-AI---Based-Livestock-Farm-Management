"""
Best-effort adapter for the secondary Zenodo IceTag/SMARTBOW activity dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries

logger = logging.getLogger(__name__)


def load_zenodo(path: Path | str) -> FeedingTimeseries:
    """
    Best-effort loader for Zenodo activity dataset.
    Raises FileNotFoundError if dataset file is absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Zenodo IceTag dataset file not found at '{path}'. Download from https://zenodo.org/records/15005885"
        )

    # If present, load and adapt
    df_raw = pd.read_csv(path)
    # Perform standard adaptation if columns match expected schema
    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df_raw.get("timestamp", df_raw.get("datetime")), utc=True)
    df["cow_id"] = df_raw.get("cow_id", df_raw.get("animal_id")).astype(str)
    df["feeding_minutes"] = pd.to_numeric(df_raw.get("feeding_duration_min", 0.0), errors="coerce")

    return FeedingTimeseries(
        df=df,
        resample_freq="1h",
        source_file=str(path),
        coverage_pct=100.0,
        notes="Loaded Zenodo activity dataset.",
        stream_availability={
            "rumination": "rumination_min" in df_raw.columns,
            "intake": "estimated_intake_kg" in df_raw.columns,
            "visits": "visits" in df_raw.columns,
            "availability": False,
        },
    )
