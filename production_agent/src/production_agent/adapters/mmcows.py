"""
MmCows dataset adapter for the Production Agent.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries

logger = logging.getLogger(__name__)

_TS_CANDIDATES = ["timestamp", "time", "datetime"]
_COW_CANDIDATES = ["cow_id", "cow", "animal_id"]
_YIELD_CANDIDATES = ["milk_yield_kg", "yield_kg", "milk_production"]
_DURATION_CANDIDATES = ["milking_duration_min", "milking_duration"]
_FAT_CANDIDATES = ["fat_pct", "milk_fat"]
_PROT_CANDIDATES = ["protein_pct", "milk_protein"]
_SCC_CANDIDATES = ["scc", "somatic_cell_count"]

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def load_mmcows(
    path: str | Path,
    nrows: Optional[int] = None,
) -> ProductionTimeseries:
    path = Path(path)
    notes: list[str] = []

    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
        if nrows: raw = raw.head(nrows)
    else:
        raw = pd.read_csv(path, nrows=nrows)

    notes.append(f"Source: {path.name}. Raw columns: {list(raw.columns)}")

    ts_col = _find_col(raw, _TS_CANDIDATES)
    cow_col = _find_col(raw, _COW_CANDIDATES)
    yield_col = _find_col(raw, _YIELD_CANDIDATES)
    dur_col = _find_col(raw, _DURATION_CANDIDATES)
    fat_col = _find_col(raw, _FAT_CANDIDATES)
    prot_col = _find_col(raw, _PROT_CANDIDATES)
    scc_col = _find_col(raw, _SCC_CANDIDATES)

    if ts_col is None or cow_col is None or yield_col is None:
        raise ValueError("Missing essential columns: timestamp, cow_id, or milk_yield_kg")

    df = pd.DataFrame()
    df[PROD_TS_COLS["TIMESTAMP"]] = pd.to_datetime(raw[ts_col], utc=True)
    df[PROD_TS_COLS["COW_ID"]] = raw[cow_col].astype(str)
    df[PROD_TS_COLS["MILK_YIELD"]] = raw[yield_col].astype(float)
    
    # Filter only rows where milk yield > 0 (milking sessions)
    df = df[df[PROD_TS_COLS["MILK_YIELD"]] > 0].copy()

    # Determine session (morning vs evening)
    hour = df[PROD_TS_COLS["TIMESTAMP"]].dt.hour
    df[PROD_TS_COLS["SESSION"]] = "unknown"
    df.loc[hour.between(4, 11), PROD_TS_COLS["SESSION"]] = "morning"
    df.loc[hour.between(14, 21), PROD_TS_COLS["SESSION"]] = "evening"

    df[PROD_TS_COLS["DURATION"]] = raw[dur_col].astype(float) if dur_col else float("nan")
    df[PROD_TS_COLS["FAT_PCT"]] = raw[fat_col].astype(float) if fat_col else float("nan")
    df[PROD_TS_COLS["PROTEIN_PCT"]] = raw[prot_col].astype(float) if prot_col else float("nan")
    df[PROD_TS_COLS["SCC"]] = raw[scc_col].astype(float) if scc_col else float("nan")

    if fat_col is None and prot_col is None:
        notes.append("No milk composition columns (fat, protein, scc) found.")

    return ProductionTimeseries(
        df=df,
        source_file=str(path),
        coverage_pct=100.0,
        notes=notes
    )
