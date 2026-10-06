"""
THI computation and heat-stress classification.

Pure functions with no external dependencies beyond numpy/pandas.
Every function is independently unit-testable.

THI formula (NRC 2001 / Thom 1958):
    THI = (1.8 * T + 32) - (0.55 - 0.0055 * RH) * (1.8 * T - 26)
where T is dry-bulb temperature in °C and RH is relative humidity in %.

Known validation values (USDA Agricultural Handbook):
    T=22°C, RH=50%  → THI ≈ 66.9  (comfort)
    T=28°C, RH=70%  → THI ≈ 78.7  (moderate stress)
    T=32°C, RH=80%  → THI ≈ 87.5  (severe stress)
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

from env_welfare_agent.src.env_agent.config import THIThresholds

logger = logging.getLogger(__name__)


def compute_thi(
    temp_c: Union[pd.Series, float, np.ndarray],
    rh_pct: Union[pd.Series, float, np.ndarray],
) -> Union[pd.Series, float]:
    """
    Compute Temperature-Humidity Index using the Thom / NRC formula.

    Parameters
    ----------
    temp_c : float, np.ndarray, or pd.Series
        Dry-bulb temperature in °C.  NaN → NaN output.
    rh_pct : float, np.ndarray, or pd.Series
        Relative humidity in %.  NaN → NaN output.

    Returns
    -------
    THI as same type as input (scalar or Series).
    """
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    thi = (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)

    if isinstance(temp_c, pd.Series):
        return pd.Series(thi, index=temp_c.index, name="thi")
    if np.isscalar(temp_c) and np.ndim(thi) == 0:
        return float(thi)
    return thi


def classify_thi(
    thi_value: float,
    thresholds: THIThresholds | None = None,
) -> str:
    """
    Classify a single THI value into a heat-stress band.

    Parameters
    ----------
    thi_value : float
        THI value.  NaN → "unknown".
    thresholds : THIThresholds, optional
        Uses default thresholds if None.

    Returns
    -------
    str: "comfort" | "mild" | "moderate" | "severe" | "emergency" | "unknown"
    """
    th = thresholds or THIThresholds()
    if np.isnan(thi_value):
        return "unknown"
    if thi_value >= th.emergency_min:
        return "emergency"
    if thi_value >= th.severe_min:
        return "severe"
    if thi_value >= th.moderate_min:
        return "moderate"
    if thi_value >= th.mild_min:
        return "mild"
    return "comfort"


def classify_thi_series(
    thi_series: pd.Series,
    thresholds: THIThresholds | None = None,
) -> pd.Series:
    """
    Vectorised THI classification for a Series.

    Returns
    -------
    pd.Series of str labels, same index as input.
    """
    th = thresholds or THIThresholds()
    conditions = [
        thi_series.isna(),
        thi_series >= th.emergency_min,
        thi_series >= th.severe_min,
        thi_series >= th.moderate_min,
        thi_series >= th.mild_min,
    ]
    choices = ["unknown", "emergency", "severe", "moderate", "mild"]
    return pd.Series(
        np.select(conditions, choices, default="comfort"),
        index=thi_series.index,
        name="stress_band",
    )
