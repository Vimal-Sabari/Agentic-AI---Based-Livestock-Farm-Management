"""
Loader and adapter for MmCows dataset and local CSV telemetry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from feeding_agent.src.feeding_agent.schemas import FEED_TS_COLS, FeedingTimeseries

logger = logging.getLogger(__name__)

CANDIDATES = {
    "timestamp": ["timestamp", "time", "datetime", "date_time"],
    "cow_id": ["cow_id", "cow", "animal_id", "tag_id"],
    "feeding_minutes": ["feeding_duration_min", "feeding_min", "feed_min", "feeding_duration", "time_at_feeder_min"],
    "rumination_minutes": ["rumination_min", "rumination", "rum_min", "rumination_duration"],
    "intake_kg": ["estimated_intake_kg", "intake_kg", "feed_intake", "dmi_kg"],
    "feeding_visits": ["feed_bunk_visits", "feeder_visits", "feeding_visits", "visits"],
    "feed_availability_pct": ["feed_availability_pct", "feed_availability", "bunk_fill_pct"],
    "group": ["group", "pen", "lot"],
}


def _match_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_mmcows(path: Path | str, resample_freq: str = "1h") -> FeedingTimeseries:
    """
    Loads local CSV / MmCows telemetry file and maps columns to canonical FeedingTimeseries.
    Case-insensitive candidate list matching. Preserves sensor gaps as NaN.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MmCows telemetry dataset file not found at: {path}")

    raw_df = pd.read_csv(path)
    cols = list(raw_df.columns)

    ts_col = _match_column(cols, CANDIDATES["timestamp"])
    cow_col = _match_column(cols, CANDIDATES["cow_id"])
    feed_col = _match_column(cols, CANDIDATES["feeding_minutes"])

    notes_list: List[str] = []

    # Best-effort fallback if feeding minutes missing but zone column present
    if not feed_col:
        zone_col = _match_column(cols, ["zone", "location", "area"])
        if zone_col:
            raw_df["derived_feeding_min"] = (raw_df[zone_col].astype(str).str.lower().str.contains("feed")).astype(float) * 60.0
            feed_col = "derived_feeding_min"
            notes_list.append("Derived feeding minutes from location/zone stream.")

    if not ts_col or not cow_col or not feed_col:
        raise ValueError(
            f"Dataset missing required primary columns (timestamp={ts_col}, cow_id={cow_col}, feeding_minutes={feed_col})."
        )

    rum_col = _match_column(cols, CANDIDATES["rumination_minutes"])
    if not rum_col:
        # Check if behaviour label column exists
        behav_col = _match_column(cols, ["behaviour", "behavior", "activity_label", "state"])
        if behav_col:
            raw_df["derived_rum_min"] = (raw_df[behav_col].astype(str).str.lower().str.contains("ruminat")).astype(float) * 60.0
            rum_col = "derived_rum_min"
            notes_list.append("Derived rumination minutes from behaviour label stream.")

    int_col = _match_column(cols, CANDIDATES["intake_kg"])
    vis_col = _match_column(cols, CANDIDATES["feeding_visits"])
    av_col = _match_column(cols, CANDIDATES["feed_availability_pct"])
    grp_col = _match_column(cols, CANDIDATES["group"])

    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(raw_df[ts_col], utc=True)
    df["cow_id"] = raw_df[cow_col].astype(str)
    df["feeding_minutes"] = pd.to_numeric(raw_df[feed_col], errors="coerce")

    if rum_col:
        df["rumination_minutes"] = pd.to_numeric(raw_df[rum_col], errors="coerce")
    else:
        df["rumination_minutes"] = np.nan

    if int_col:
        df["intake_kg"] = pd.to_numeric(raw_df[int_col], errors="coerce")
    else:
        df["intake_kg"] = np.nan

    if vis_col:
        df["feeding_visits"] = pd.to_numeric(raw_df[vis_col], errors="coerce")
    else:
        df["feeding_visits"] = np.nan

    if av_col:
        df["feed_availability_pct"] = pd.to_numeric(raw_df[av_col], errors="coerce")
    else:
        df["feed_availability_pct"] = np.nan

    if grp_col:
        df["group"] = raw_df[grp_col].astype(str)

    # Resample per cow to resample_freq preserving NaNs (min_count=1)
    resampled_dfs = []
    for cow_id, group in df.groupby("cow_id"):
        group = group.set_index("timestamp").sort_index()

        # Aggregation using min_count=1 to keep gaps as NaN
        agg_map = {
            "feeding_minutes": lambda s: s.sum(min_count=1),
            "rumination_minutes": lambda s: s.sum(min_count=1),
            "intake_kg": lambda s: s.sum(min_count=1),
            "feeding_visits": lambda s: s.sum(min_count=1),
            "feed_availability_pct": "mean",
        }
        if "group" in group.columns:
            agg_map["group"] = "first"

        r_grp = group.resample(resample_freq).agg(agg_map)
        r_grp["cow_id"] = cow_id
        r_grp = r_grp.reset_index()
        resampled_dfs.append(r_grp)

    final_df = pd.concat(resampled_dfs, ignore_index=True) if resampled_dfs else df

    # Stream availability: True only if at least one non-NaN value exists in the source
    stream_avail = {
        "rumination": bool(final_df["rumination_minutes"].notna().any()),
        "intake": bool(final_df["intake_kg"].notna().any()),
        "visits": bool(final_df["feeding_visits"].notna().any()),
        "availability": bool(final_df["feed_availability_pct"].notna().any()),
    }

    # Calculate overall coverage pct: share of expected cow-hours with non-NaN feeding_minutes
    total_valid = int(final_df["feeding_minutes"].notna().sum())
    total_rows = len(final_df)
    coverage_pct = float((total_valid / float(total_rows)) * 100.0) if total_rows > 0 else 0.0

    notes_str = "; ".join(notes_list) if notes_list else "MmCows CSV file loaded successfully."

    return FeedingTimeseries(
        df=final_df,
        resample_freq=resample_freq,
        source_file=str(path),
        coverage_pct=coverage_pct,
        notes=notes_str,
        stream_availability=stream_avail,
    )
