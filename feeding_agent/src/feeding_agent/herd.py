"""
Herd comparison analysis evaluating feeding changes across the rest of the herd.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from feeding_agent.src.feeding_agent.baseline import derive_baseline
from feeding_agent.src.feeding_agent.config import BaselineConfig, DataConfig, DeviationConfig, HerdConfig
from feeding_agent.src.feeding_agent.deviation import compute_deviation
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries, HerdResult

logger = logging.getLogger(__name__)


def compute_herd_comparison(
    ts: FeedingTimeseries,
    target_cow_id: str,
    anomaly_start: datetime,
    anomaly_end: datetime,
    target_direction: str = "decline",
    cfg: Optional[HerdConfig] = None,
    baseline_cfg: Optional[BaselineConfig] = None,
    dev_cfg: Optional[DeviationConfig] = None,
    data_cfg: Optional[DataConfig] = None,
) -> HerdResult:
    """
    Evaluates deviation across all cows in the dataset except the target cow.
    """
    cfg = cfg or HerdConfig()
    baseline_cfg = baseline_cfg or BaselineConfig()
    dev_cfg = dev_cfg or DeviationConfig()
    data_cfg = data_cfg or DataConfig()

    df = ts.df
    if df.empty or "cow_id" not in df.columns:
        return HerdResult(
            cows_analyzed=0,
            cows_deviating=0,
            fraction_deviating=0.0,
            is_herd_wide=False,
            notes="No herd data available.",
        )

    all_cows = [c for c in df["cow_id"].unique() if c != target_cow_id]
    if not all_cows:
        return HerdResult(
            cows_analyzed=0,
            cows_deviating=0,
            fraction_deviating=0.0,
            is_herd_wide=False,
            notes="Target cow is the only cow in dataset.",
        )

    cows_analyzed = 0
    cows_deviating = 0

    for cow in all_cows:
        base = derive_baseline(ts, cow, anomaly_start, cfg=baseline_cfg, data_cfg=data_cfg)
        if base.n_days < baseline_cfg.min_days_required:
            continue

        cows_analyzed += 1
        dev = compute_deviation(ts, cow, anomaly_start, anomaly_end, base, cfg=dev_cfg, data_cfg=data_cfg)

        if dev.feeding_significant and dev.feeding_direction == target_direction:
            cows_deviating += 1

    if cows_analyzed < cfg.min_cows:
        return HerdResult(
            cows_analyzed=cows_analyzed,
            cows_deviating=cows_deviating,
            fraction_deviating=0.0,
            is_herd_wide=False,
            notes=f"Insufficient herd cows analyzed ({cows_analyzed} < min {cfg.min_cows}).",
        )

    frac = cows_deviating / float(cows_analyzed)
    is_herd = frac >= cfg.herd_wide_fraction

    return HerdResult(
        cows_analyzed=cows_analyzed,
        cows_deviating=cows_deviating,
        fraction_deviating=frac,
        is_herd_wide=is_herd,
        notes=f"{cows_deviating}/{cows_analyzed} cows showed matching feeding {target_direction}.",
    )
