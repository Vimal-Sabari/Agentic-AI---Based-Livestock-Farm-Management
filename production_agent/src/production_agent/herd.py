"""
Herd-wide deviation logic.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from production_agent.src.production_agent.config import HerdConfig
from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries, HerdDeviationResult

def compute_herd_deviation(
    ts: ProductionTimeseries,
    target_cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    pct_drop_threshold: float = 5.0,
    cfg: Optional[HerdConfig] = None
) -> HerdDeviationResult:
    cfg = cfg or HerdConfig()
    df = ts.df.copy()
    
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)
        
    all_cows = df[PROD_TS_COLS["COW_ID"]].unique().tolist()
    
    if len(all_cows) < cfg.min_cows_for_herd_check:
        return HerdDeviationResult(len(all_cows), 0, 0.0, False)
        
    deviating = 0
    for cid in all_cows:
        cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cid]
        anom = cow_df[(cow_df[PROD_TS_COLS["TIMESTAMP"]] >= anomaly_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_end)]
        base = cow_df[cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_start]
        
        if not anom.empty and not base.empty:
            a_mean = anom.groupby(anom[PROD_TS_COLS["TIMESTAMP"]].dt.date)[PROD_TS_COLS["MILK_YIELD"]].sum().mean()
            b_mean = base.groupby(base[PROD_TS_COLS["TIMESTAMP"]].dt.date)[PROD_TS_COLS["MILK_YIELD"]].sum().mean()
            if b_mean > 0:
                pct = (a_mean - b_mean) / b_mean * 100.0
                if pct <= -pct_drop_threshold:
                    deviating += 1
                    
    frac = deviating / len(all_cows)
    
    return HerdDeviationResult(
        cows_analyzed=len(all_cows),
        cows_deviating=deviating,
        fraction_deviating=round(frac, 3),
        is_herd_wide=frac >= cfg.is_herd_wide_threshold
    )
