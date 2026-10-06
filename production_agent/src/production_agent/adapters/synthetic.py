"""
Synthetic data generator for Production Agent testing.
Produces 16 cows × 14 days × 2 milkings/day (448 rows).
Scenarios:
  (a) cow_specific_drop: cow_01 drops yield by 15% in last 2 days
  (b) herd_wide_drop: 60% of herd drops yield by 10%
  (c) stable: normal variation
  (d) short_baseline: only 2 days of data
  (e) missing_days: gaps in telemetry
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal
import numpy as np
import pandas as pd

from production_agent.src.production_agent.schemas import PROD_TS_COLS, ProductionTimeseries

logger = logging.getLogger(__name__)

Scenario = Literal["cow_specific_drop", "herd_wide_drop", "stable", "short_baseline", "missing_days"]

def generate_synthetic(
    scenario: Scenario = "cow_specific_drop",
    n_cows: int = 16,
    n_days: int = 14,
    start_date: str = "2026-09-01",
    seed: int = 42,
) -> ProductionTimeseries:
    rng = np.random.default_rng(seed)
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    
    if scenario == "short_baseline":
        n_days = 4 # 2 days baseline, 2 days anomaly
        
    timestamps = []
    for d in range(n_days):
        timestamps.append(start + timedelta(days=d, hours=6))  # Morning
        timestamps.append(start + timedelta(days=d, hours=17)) # Evening
        
    cow_ids = [f"cow_{i:02d}" for i in range(1, n_cows + 1)]
    base_yields = {cid: rng.uniform(28, 38) for cid in cow_ids}
    
    anomaly_day_start = n_days - 2
    
    rows = []
    for ts in timestamps:
        day_idx = (ts - start).days
        is_morning = ts.hour == 6
        in_anomaly = day_idx >= anomaly_day_start
        
        # Missing days scenario
        if scenario == "missing_days" and day_idx in (2, 3, 5):
            continue
            
        for cid in cow_ids:
            base = base_yields[cid]
            # Evening yield is typically slightly lower
            session_yield = base * 0.52 if is_morning else base * 0.48
            
            # Apply scenarios
            mod = 1.0
            if in_anomaly:
                if scenario == "cow_specific_drop" and cid == "cow_01":
                    mod = 0.82 # 18% drop
                elif scenario == "herd_wide_drop":
                    cow_idx = int(cid.split("_")[1])
                    if cow_idx % 2 != 0: # 50% of herd drops
                        mod = rng.uniform(0.85, 0.90)
                        
            final_yield = session_yield * mod + rng.normal(0, 0.5)
            
            # Composition (random stable values with noise)
            fat = 3.8 + rng.normal(0, 0.2)
            prot = 3.2 + rng.normal(0, 0.1)
            scc = rng.uniform(50, 150)
            
            rows.append({
                PROD_TS_COLS["TIMESTAMP"]: ts,
                PROD_TS_COLS["COW_ID"]: cid,
                PROD_TS_COLS["MILK_YIELD"]: float(np.clip(final_yield, 0, 50)),
                PROD_TS_COLS["SESSION"]: "morning" if is_morning else "evening",
                PROD_TS_COLS["FAT_PCT"]: float(np.clip(fat, 2, 6)),
                PROD_TS_COLS["PROTEIN_PCT"]: float(np.clip(prot, 2, 5)),
                PROD_TS_COLS["SCC"]: float(scc),
                PROD_TS_COLS["DURATION"]: float(rng.uniform(5, 8))
            })
            
    df = pd.DataFrame(rows)
    df[PROD_TS_COLS["TIMESTAMP"]] = pd.to_datetime(df[PROD_TS_COLS["TIMESTAMP"]], utc=True)
    
    return ProductionTimeseries(
        df=df,
        source_file=f"synthetic:{scenario}",
        coverage_pct=100.0,
        notes=[f"Synthetic scenario={scenario}"]
    )
