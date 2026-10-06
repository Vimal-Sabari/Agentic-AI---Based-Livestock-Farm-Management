"""
Baseline statistics for milk production.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
import numpy as np

from production_agent.src.production_agent.config import BaselineConfig
from production_agent.src.production_agent.schemas import PROD_TS_COLS, BaselineStats, ProductionTimeseries

logger = logging.getLogger(__name__)

def derive_baseline(
    ts: ProductionTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    baseline_start: Optional[datetime] = None,
    baseline_end: Optional[datetime] = None,
    cfg: Optional[BaselineConfig] = None,
) -> BaselineStats:
    cfg = cfg or BaselineConfig()
    df = ts.df.copy()
    
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
        
    if baseline_start is None or baseline_end is None:
        b_end = anomaly_start
        b_start = anomaly_start - timedelta(days=cfg.default_days_before_anomaly)
    else:
        b_start = baseline_start if baseline_start.tzinfo else baseline_start.replace(tzinfo=timezone.utc)
        b_end = baseline_end if baseline_end.tzinfo else baseline_end.replace(tzinfo=timezone.utc)
        
    cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cow_id]
    mask = (cow_df[PROD_TS_COLS["TIMESTAMP"]] >= b_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < b_end)
    base_df = cow_df[mask].copy()
    
    # Aggregate to daily yield for baseline stability
    base_df["date"] = base_df[PROD_TS_COLS["TIMESTAMP"]].dt.date
    daily = base_df.groupby("date")[PROD_TS_COLS["MILK_YIELD"]].sum()
    
    expected_days = max(1, (b_end - b_start).days)
    n_days = len(daily)
    coverage = min(100.0, round(100.0 * n_days / expected_days, 1))
    
    has_comp = PROD_TS_COLS["FAT_PCT"] in base_df.columns and base_df[PROD_TS_COLS["FAT_PCT"]].notna().any()
    
    if n_days < cfg.min_days_required:
        return BaselineStats(
            mean_yield_daily=float("nan"),
            std_yield_daily=float("nan"),
            median_yield_daily=float("nan"),
            n_days=n_days,
            coverage_pct=coverage,
            has_composition=has_comp
        )
        
    from scipy.stats import trim_mean
    
    mean_val = float(trim_mean(daily, cfg.trim_fraction)) if n_days >= 3 else float(daily.mean())
    median_val = float(daily.median())
    std_val = float(daily.std(ddof=1)) if n_days > 1 else 0.0
    
    # If std is extremely small or zero, set a minimum 5% variability floor
    if std_val < 0.05 * mean_val:
        std_val = 0.05 * mean_val
        
    return BaselineStats(
        mean_yield_daily=round(mean_val, 2),
        std_yield_daily=round(std_val, 2),
        median_yield_daily=round(median_val, 2),
        n_days=n_days,
        coverage_pct=coverage,
        has_composition=has_comp
    )
