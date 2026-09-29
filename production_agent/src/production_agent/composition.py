"""
Milk composition analysis (fat, protein, SCC).
"""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries, CompositionResult

def compute_composition(
    ts: ProductionTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
) -> CompositionResult:
    df = ts.df.copy()
    
    if PROD_TS_COLS["FAT_PCT"] not in df.columns or df[PROD_TS_COLS["FAT_PCT"]].isna().all():
        return CompositionResult(False, 0.0, 0.0, 0.0)
        
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)
        
    cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cow_id]
    mask = (cow_df[PROD_TS_COLS["TIMESTAMP"]] >= anomaly_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_end)
    anom_df = cow_df[mask]
    base_df = cow_df[cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_start]
    
    if anom_df.empty or base_df.empty:
        return CompositionResult(True, 0.0, 0.0, 0.0)
        
    f_base = base_df[PROD_TS_COLS["FAT_PCT"]].mean()
    p_base = base_df[PROD_TS_COLS["PROTEIN_PCT"]].mean()
    s_base = base_df[PROD_TS_COLS["SCC"]].mean()
    
    f_anom = anom_df[PROD_TS_COLS["FAT_PCT"]].mean()
    p_anom = anom_df[PROD_TS_COLS["PROTEIN_PCT"]].mean()
    s_anom = anom_df[PROD_TS_COLS["SCC"]].mean()
    
    f_del = (f_anom - f_base) / f_base * 100.0 if pd.notna(f_base) and f_base > 0 else 0.0
    p_del = (p_anom - p_base) / p_base * 100.0 if pd.notna(p_base) and p_base > 0 else 0.0
    s_del = (s_anom - s_base) / s_base * 100.0 if pd.notna(s_base) and s_base > 0 else 0.0
    
    return CompositionResult(
        has_composition=True,
        fat_delta_pct=round(float(f_del), 2) if pd.notna(f_del) else 0.0,
        protein_delta_pct=round(float(p_del), 2) if pd.notna(p_del) else 0.0,
        scc_delta_pct=round(float(s_del), 2) if pd.notna(s_del) else 0.0
    )
