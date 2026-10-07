"""
Unit tests for data adapters.
"""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from feeding_agent.src.feeding_agent.adapters.mmcows import load_mmcows
from feeding_agent.src.feeding_agent.adapters.zenodo_activity import load_zenodo


def test_load_mmcows_existing_synthetic():
    csv_path = Path("data/mmcows_synthetic.csv")
    if not csv_path.exists():
        pytest.skip("data/mmcows_synthetic.csv file absent; skipping load test.")

    ts = load_mmcows(csv_path)
    assert not ts.df.empty
    assert "feeding_minutes" in ts.df.columns
    assert "cow_id" in ts.df.columns
    assert ts.df["timestamp"].dt.tz is not None


def test_candidate_column_matching_with_renamed_columns():
    # Test candidate column matching on custom column names
    data = {
        "time": ["2026-09-01 00:00:00", "2026-09-01 01:00:00"],
        "animal_id": ["cow_01", "cow_01"],
        "feed_min": [15.5, 20.0],
        "rum_min": [30.0, 25.0],
        "dmi_kg": [2.1, 2.5],
        "visits": [2, 3],
        "bunk_fill_pct": [88.0, 90.0],
    }
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        pd.DataFrame(data).to_csv(f.name, index=False)
        temp_path = f.name

    try:
        ts = load_mmcows(temp_path)
        assert "feeding_minutes" in ts.df.columns
        assert "rumination_minutes" in ts.df.columns
        assert "intake_kg" in ts.df.columns
        assert "feeding_visits" in ts.df.columns
        assert "feed_availability_pct" in ts.df.columns
        assert ts.stream_availability["rumination"] is True
        assert ts.stream_availability["intake"] is True
        assert ts.stream_availability["visits"] is True
        assert ts.stream_availability["availability"] is True
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_utc_conversion():
    data = {
        "timestamp": ["2026-09-01 05:00:00+02:00", "2026-09-01 06:00:00+02:00"],
        "cow_id": ["cow_01", "cow_01"],
        "feeding_duration_min": [10.0, 15.0],
    }
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        pd.DataFrame(data).to_csv(f.name, index=False)
        temp_path = f.name

    try:
        ts = load_mmcows(temp_path)
        assert ts.df["timestamp"].dt.tz is not None
        # Verified UTC
        assert str(ts.df["timestamp"].dt.tz) in ["UTC", "datetime.timezone.utc"]
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_gaps_stay_nan():
    # Gap in feeding duration (NaN) must not become 0.0 during resampling
    data = {
        "timestamp": ["2026-09-01 00:00:00Z", "2026-09-01 01:00:00Z"],
        "cow_id": ["cow_01", "cow_01"],
        "feeding_duration_min": [10.0, np.nan],
    }
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        pd.DataFrame(data).to_csv(f.name, index=False)
        temp_path = f.name

    try:
        ts = load_mmcows(temp_path)
        # Hour 01:00:00 must remain NaN
        hour1_val = ts.df[ts.df["timestamp"] == "2026-09-01 01:00:00+00:00"]["feeding_minutes"].iloc[0]
        assert np.isnan(hour1_val)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_zenodo_missing_raises():
    with pytest.raises(FileNotFoundError) as exc_info:
        load_zenodo("non_existent_zenodo_file.csv")
    assert "Zenodo" in str(exc_info.value)


def test_zenodo_skipped_without_file():
    zenodo_path = Path("data/zenodo_activity.csv")
    if not zenodo_path.exists():
        pytest.skip(f"Zenodo file not present at {zenodo_path}; test skipped.")
    ts = load_zenodo(zenodo_path)
    assert not ts.df.empty
