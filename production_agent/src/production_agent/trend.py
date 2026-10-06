"""
Trend and change-point analysis.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats

from production_agent.src.production_agent.config import TrendConfig
from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries, TrendResult

def compute_trend(
    ts: ProductionTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    cfg: Optional[TrendConfig] = None
) -> TrendResult:
    cfg = cfg or TrendConfig()
    df = ts.df.copy()
    
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)
        
    cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cow_id]
    mask = (cow_df[PROD_TS_COLS["TIMESTAMP"]] >= anomaly_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_end)
    anom_df = cow_df[mask].copy()
    
    if anom_df.empty:
        return TrendResult(0.0, "stable", False, None)
        
    anom_df["date"] = anom_df[PROD_TS_COLS["TIMESTAMP"]].dt.date
    daily = anom_df.groupby("date")[PROD_TS_COLS["MILK_YIELD"]].sum().reset_index()
    
    if len(daily) < cfg.min_days_for_trend:
        # Not enough points for a robust slope
        if len(daily) == 2:
            diff = float(daily.iloc[1][PROD_TS_COLS["MILK_YIELD"]] - daily.iloc[0][PROD_TS_COLS["MILK_YIELD"]])
            direction = "falling" if diff < -1.0 else ("rising" if diff > 1.0 else "stable")
            return TrendResult(round(diff, 2), direction, False, None)
        return TrendResult(0.0, "stable", False, None)
        
    # Theil-Sen robust slope
    x = np.arange(len(daily))
    y = daily[PROD_TS_COLS["MILK_YIELD"]].values
    slope, intercept, lo_slope, up_slope = stats.mstats.theilslopes(y, x, 0.95)
    
    slope = float(slope)
    direction = "falling" if slope < -0.5 else ("rising" if slope > 0.5 else "stable")
    
    # Simple change-point detection (rolling diff)
    daily["diff"] = daily[PROD_TS_COLS["MILK_YIELD"]].diff()
    max_drop_idx = daily["diff"].idxmin()
    max_drop_val = daily["diff"].min()
    
    change_point = False
    cp_date = None
    if max_drop_val < -3.0: # arbitrary threshold for sudden drop
        change_point = True
        cp_date = daily.iloc[max_drop_idx]["date"].isoformat()
        
    return TrendResult(
        slope_kg_per_day=round(slope, 2),
        direction=direction,
        change_point_detected=change_point,
        change_point_date=cp_date
    )
