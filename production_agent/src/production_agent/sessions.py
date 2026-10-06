"""
Session analysis (morning vs evening).
"""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd

from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries, SessionResult

def compute_session_stats(
    ts: ProductionTimeseries,
    cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
) -> SessionResult:
    df = ts.df.copy()
    
    if anomaly_start.tzinfo is None:
        anomaly_start = anomaly_start.replace(tzinfo=timezone.utc)
    if anomaly_end.tzinfo is None:
        anomaly_end = anomaly_end.replace(tzinfo=timezone.utc)
        
    cow_df = df[df[PROD_TS_COLS["COW_ID"]] == cow_id]
    mask = (cow_df[PROD_TS_COLS["TIMESTAMP"]] >= anomaly_start) & (cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_end)
    anom_df = cow_df[mask].copy()
    
    if anom_df.empty or "morning" not in anom_df[PROD_TS_COLS["SESSION"]].values or "evening" not in anom_df[PROD_TS_COLS["SESSION"]].values:
        return SessionResult(False, 0.0, 0.0, 0.0, 0.0)
        
    # Baseline for reference
    base_df = cow_df[cow_df[PROD_TS_COLS["TIMESTAMP"]] < anomaly_start]
    
    m_base = base_df[base_df[PROD_TS_COLS["SESSION"]] == "morning"][PROD_TS_COLS["MILK_YIELD"]].mean() if not base_df.empty else float("nan")
    e_base = base_df[base_df[PROD_TS_COLS["SESSION"]] == "evening"][PROD_TS_COLS["MILK_YIELD"]].mean() if not base_df.empty else float("nan")
    
    m_anom = anom_df[anom_df[PROD_TS_COLS["SESSION"]] == "morning"][PROD_TS_COLS["MILK_YIELD"]].mean()
    e_anom = anom_df[anom_df[PROD_TS_COLS["SESSION"]] == "evening"][PROD_TS_COLS["MILK_YIELD"]].mean()
    
    m_delta = (m_anom - m_base) / m_base * 100.0 if pd.notna(m_base) and m_base > 0 else 0.0
    e_delta = (e_anom - e_base) / e_base * 100.0 if pd.notna(e_base) and e_base > 0 else 0.0
    
    return SessionResult(
        has_sessions=True,
        morning_avg=round(float(m_anom), 2),
        evening_avg=round(float(e_anom), 2),
        morning_delta_pct=round(float(m_delta), 2),
        evening_delta_pct=round(float(e_delta), 2)
    )
