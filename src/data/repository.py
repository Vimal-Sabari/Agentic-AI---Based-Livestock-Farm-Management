"""
Data Repository & Baseline Derivation Engine.
Provides high-performance querying for cow timelines, automated baseline derivation,
and herd-wide context evaluation.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.data.schema import CANONICAL_COLUMNS
from src.data.adapter import DatasetAdapter
from src.data.synthetic_generator import generate_mmcows_synthetic_dataset


class LivestockRepository:
    """
    In-memory and file-backed repository for cow sensor streams.
    Provides window slicing, statistical baselines, and herd-wide comparative context.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or "data/mmcows_synthetic.csv"
        self._df: Optional[pd.DataFrame] = None
        self._ensure_data_loaded()

    def _ensure_data_loaded(self) -> None:
        """Loads or generates dataset if not already present."""
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Generating synthetic MmCows dataset...")
            self._df = generate_mmcows_synthetic_dataset(output_path=self.data_path)
        else:
            raw_df = pd.read_csv(self.data_path)
            self._df = DatasetAdapter.adapt_dataframe(raw_df)

        # Ensure timestamp is datetime and sorted
        ts_col = CANONICAL_COLUMNS["TIMESTAMP"]
        self._df[ts_col] = pd.to_datetime(self._df[ts_col], utc=True)
        self._df = self._df.sort_values(by=ts_col).reset_index(drop=True)

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            self._ensure_data_loaded()
        return self._df

    def reload(self, new_path: Optional[str] = None) -> None:
        if new_path:
            self.data_path = new_path
        self._df = None
        self._ensure_data_loaded()

    def get_cows(self) -> List[str]:
        """Returns sorted list of unique cow IDs."""
        cow_col = CANONICAL_COLUMNS["COW_ID"]
        return sorted(self.dataframe[cow_col].unique().tolist())

    def get_time_range(self) -> Tuple[datetime, datetime]:
        """Returns (min_timestamp, max_timestamp) in UTC."""
        ts_col = CANONICAL_COLUMNS["TIMESTAMP"]
        return self.dataframe[ts_col].min().to_pydatetime(), self.dataframe[ts_col].max().to_pydatetime()

    def get_cow_data(
        self,
        cow_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Returns filtered DataFrame for a specific cow within [start_time, end_time]."""
        cow_col = CANONICAL_COLUMNS["COW_ID"]
        ts_col = CANONICAL_COLUMNS["TIMESTAMP"]
        sub = self.dataframe[self.dataframe[cow_col] == cow_id]

        if start_time is not None:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            sub = sub[sub[ts_col] >= start_time]

        if end_time is not None:
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            sub = sub[sub[ts_col] <= end_time]

        return sub.copy()

    def get_herd_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Returns herd-wide DataFrame sliced to time window."""
        ts_col = CANONICAL_COLUMNS["TIMESTAMP"]
        sub = self.dataframe.copy()
        if start_time is not None:
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            sub = sub[sub[ts_col] >= start_time]
        if end_time is not None:
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            sub = sub[sub[ts_col] <= end_time]
        return sub

    def compute_individual_baseline(
        self,
        cow_id: str,
        anomaly_start: datetime,
        baseline_start: Optional[datetime] = None,
        baseline_end: Optional[datetime] = None,
        default_baseline_days: int = 5
    ) -> pd.DataFrame:
        """
        Retrieves baseline DataFrame for a specific cow.
        If baseline window is not specified, uses `default_baseline_days` immediately prior to anomaly_start.
        """
        if anomaly_start.tzinfo is None:
            anomaly_start = anomaly_start.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)

        if baseline_start is None or baseline_end is None:
            end_dt = anomaly_start
            start_dt = anomaly_start - timedelta(days=default_baseline_days)
        else:
            start_dt = baseline_start
            end_dt = baseline_end
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=pd.Timestamp.utcnow().tzinfo)

        baseline_df = self.get_cow_data(cow_id, start_time=start_dt, end_time=end_dt)

        # Fallback if baseline window has no data: take earliest available historical data before anomaly_start
        if baseline_df.empty:
            cow_all = self.get_cow_data(cow_id, end_time=anomaly_start)
            if not cow_all.empty:
                baseline_df = cow_all.tail(default_baseline_days * 24)

        return baseline_df


# Global singleton instance for easy access across agent views & LangGraph
_GLOBAL_REPO: Optional[LivestockRepository] = None

def get_repository(data_path: Optional[str] = None) -> LivestockRepository:
    global _GLOBAL_REPO
    if _GLOBAL_REPO is None or (data_path and _GLOBAL_REPO.data_path != data_path):
        _GLOBAL_REPO = LivestockRepository(data_path=data_path)
    return _GLOBAL_REPO
