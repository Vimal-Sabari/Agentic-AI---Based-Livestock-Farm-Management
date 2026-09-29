"""
Deviation logic for milk production.
"""
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from production_agent.src.production_agent.config import DeviationConfig
from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries, BaselineStats, DeviationResult

def compute_deviation(
    ts: ProductionTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    baseline: BaselineStats,
    cfg: Optional[DeviationConfig] = None
) -> DeviationResult:
    cfg = cfg or DeviationConfig()
    df = ts.df.copy()
    
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)
        
    cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cow_id]
    mask = (cow_df[PROD_TS_COLS["TIMESTAMP"]] >= anomaly_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_end)
    anom_df = cow_df[mask].copy()
    
    anom_df["date"] = anom_df[PROD_TS_COLS["TIMESTAMP"]].dt.date
    daily = anom_df.groupby("date")[PROD_TS_COLS["MILK_YIELD"]].sum()
    
    if daily.empty or math.isnan(baseline.mean_yield_daily):
        return DeviationResult(
            mean_yield_daily=float("nan"), delta_kg=float("nan"),
            delta_pct=float("nan"), z_score=float("nan"),
            days_persisted=0, is_significant_drop=False
        )
        
    mean_anom = float(daily.mean())
    delta_kg = mean_anom - baseline.mean_yield_daily
    delta_pct = (delta_kg / baseline.mean_yield_daily) * 100.0 if baseline.mean_yield_daily > 0 else 0.0
    
    z_score = delta_kg / baseline.std_yield_daily if baseline.std_yield_daily > 0 else 0.0
    
    # Persistence: number of consecutive days in anomaly window where yield is below mean - threshold
    drop_days = sum(1 for y in daily if y < baseline.mean_yield_daily - (baseline.mean_yield_daily * cfg.pct_drop_threshold/100))
    
    is_sig = (
        delta_pct <= -cfg.pct_drop_threshold and 
        z_score <= -cfg.zscore_threshold and
        drop_days >= cfg.min_persistence_days
    )
    
    return DeviationResult(
        mean_yield_daily=round(mean_anom, 2),
        delta_kg=round(delta_kg, 2),
        delta_pct=round(delta_pct, 2),
        z_score=round(z_score, 2),
        days_persisted=drop_days,
        is_significant_drop=is_sig
    )
