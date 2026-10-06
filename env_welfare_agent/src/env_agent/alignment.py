"""
Temporal alignment analysis.

Computes lagged cross-correlation between barn THI and herd-median activity
over lags 0 to max_lag_hours.  Reports best lag and Pearson r with caveats
about small sample sizes.

No Streamlit or LangGraph imports.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.config import AlignmentConfig
from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    ENV_TS_COLS,
    AlignmentResult,
    CowActivityTimeseries,
    EnvTimeseries,
)

logger = logging.getLogger(__name__)


def compute_temporal_alignment(
    env_ts: EnvTimeseries,
    cow_act_ts: CowActivityTimeseries,
    anomaly_start: datetime,
    anomaly_end: datetime,
    cfg: Optional[AlignmentConfig] = None,
) -> AlignmentResult:
    """
    Compute lagged cross-correlation between THI and herd-median activity.

    For each lag h in [0, max_lag_hours]:
        r = pearson_corr(THI(t), median_activity(t + h))

    A *negative* r at lag h > 0 means activity drops h hours AFTER THI rises,
    which is consistent with environmental causation.

    Parameters
    ----------
    env_ts : EnvTimeseries
    cow_act_ts : CowActivityTimeseries
    anomaly_start / anomaly_end : datetime (tz-aware)
    cfg : AlignmentConfig, optional

    Returns
    -------
    AlignmentResult
    """
    acfg = cfg or AlignmentConfig()
    ts_col_env = ENV_TS_COLS["TIMESTAMP"]
    thi_col = ENV_TS_COLS["THI"]
    ts_col_act = COW_ACT_COLS["TIMESTAMP"]
    act_col = COW_ACT_COLS["ACTIVITY"]
    cow_col = COW_ACT_COLS["COW_ID"]

    def _tz(dt: datetime) -> datetime:
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    anomaly_start = _tz(anomaly_start)
    anomaly_end = _tz(anomaly_end)

    # --- Slice env ---
    env_df = env_ts.df.copy()
    env_df[ts_col_env] = pd.to_datetime(env_df[ts_col_env], utc=True)
    env_win = env_df[(env_df[ts_col_env] >= anomaly_start) & (env_df[ts_col_env] < anomaly_end)]
    env_series = env_win.set_index(ts_col_env)[thi_col].sort_index().dropna()

    # --- Herd-median activity ---
    act_df = cow_act_ts.df.copy()
    act_df[ts_col_act] = pd.to_datetime(act_df[ts_col_act], utc=True)
    act_win = act_df[(act_df[ts_col_act] >= anomaly_start) & (act_df[ts_col_act] < anomaly_end)]
    if act_win.empty:
        return _no_data_result(acfg.max_lag_hours)

    herd_median = (
        act_win.groupby(ts_col_act)[act_col].median().sort_index().dropna()
    )

    if len(env_series) < acfg.min_points_for_correlation or len(herd_median) < acfg.min_points_for_correlation:
        logger.warning(
            "Insufficient points for cross-correlation: env=%d, activity=%d (min=%d).",
            len(env_series), len(herd_median), acfg.min_points_for_correlation,
        )
        return AlignmentResult(
            best_lag_hours=0,
            best_r=float("nan"),
            lags_tested=[],
            r_values=[],
            n_points=min(len(env_series), len(herd_median)),
            sufficient_data=False,
        )

    # Align on common timestamps
    combined = pd.DataFrame({"thi": env_series, "activity": herd_median}).dropna()

    if len(combined) < acfg.min_points_for_correlation:
        return AlignmentResult(
            best_lag_hours=0, best_r=float("nan"),
            lags_tested=[], r_values=[],
            n_points=len(combined), sufficient_data=False,
        )

    lags = list(range(0, acfg.max_lag_hours + 1))
    r_values: list[float] = []

    thi_arr = combined["thi"].values

    for lag in lags:
        if lag == 0:
            act_arr = combined["activity"].values
        else:
            # Shift activity backward by lag (so thi[t] paired with activity[t+lag])
            shifted_activity = herd_median.shift(-lag, freq=env_series.index.freq or "1h")
            comb_lag = pd.DataFrame({"thi": env_series, "act": shifted_activity}).dropna()
            if len(comb_lag) < acfg.min_points_for_correlation:
                r_values.append(float("nan"))
                continue
            act_arr = comb_lag["act"].values
            thi_arr_lag = comb_lag["thi"].values
            r = _pearson_r(thi_arr_lag, act_arr)
            r_values.append(r)
            continue
        r = _pearson_r(thi_arr, act_arr)
        r_values.append(r)

    # Best lag: most negative correlation (strongest inverse relationship)
    valid_pairs = [(lag, r) for lag, r in zip(lags, r_values) if not math.isnan(r)]
    if not valid_pairs:
        return _no_data_result(acfg.max_lag_hours)

    # Prefer most negative r (activity falls after THI rises)
    best_lag, best_r = min(valid_pairs, key=lambda x: x[1])
    n_points = len(combined)

    logger.info(
        "Temporal alignment: best_lag=%d h, best_r=%.3f, n_points=%d",
        best_lag, best_r, n_points,
    )

    return AlignmentResult(
        best_lag_hours=int(best_lag),
        best_r=round(float(best_r), 3),
        lags_tested=lags,
        r_values=[round(r, 3) if not math.isnan(r) else float("nan") for r in r_values],
        n_points=n_points,
        sufficient_data=True,
    )


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returns NaN if degenerate."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    if x_std < 1e-10 or y_std < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _no_data_result(max_lag: int) -> AlignmentResult:
    return AlignmentResult(
        best_lag_hours=0, best_r=float("nan"),
        lags_tested=list(range(max_lag + 1)),
        r_values=[float("nan")] * (max_lag + 1),
        n_points=0, sufficient_data=False,
    )
