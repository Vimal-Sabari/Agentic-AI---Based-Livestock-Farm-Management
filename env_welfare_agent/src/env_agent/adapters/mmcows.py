"""
MmCows dataset adapter for the Environment & Welfare Agent.

Inspects real column names from the CSV/parquet, maps them to the canonical
schema, and returns typed EnvTimeseries and CowActivityTimeseries objects.

Column mapping (documented from actual dataset inspection, 2026-09-28):
  Raw column            → Canonical field
  -----------------------------------------
  timestamp             → timestamp
  barn_temp_c           → temp_c          (source = "barn")
  relative_humidity_pct → rh_pct
  thi                   → thi             (pre-computed in dataset)
  ambient_temp_c        → [extra col]     (retained as-is)
  cow_id                → cow_id
  activity_index        → activity_metric
  lying_time_min        → lying_minutes

If the dataset has different column names they can be passed via
``column_overrides`` (raw_name → canonical_name dict).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from env_welfare_agent.src.env_agent.schemas import (
    COW_ACT_COLS,
    ENV_TS_COLS,
    CowActivityTimeseries,
    EnvTimeseries,
)
from env_welfare_agent.src.env_agent.thi import compute_thi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default candidate column lists (first match wins, case-insensitive)
# ---------------------------------------------------------------------------
_BARN_TEMP_CANDIDATES = ["barn_temp_c", "barn_temp", "indoor_temp_c", "t_barn"]
_AMBIENT_TEMP_CANDIDATES = ["ambient_temp_c", "outdoor_temp_c", "weather_temp"]
_RH_CANDIDATES = ["relative_humidity_pct", "rh_pct", "rh", "humidity", "barn_rh"]
_THI_CANDIDATES = ["thi", "temperature_humidity_index", "thi_index"]
_ACTIVITY_CANDIDATES = ["activity_index", "activity_score", "activity", "motion", "act"]
_LYING_CANDIDATES = ["lying_time_min", "lying_minutes", "lying_duration", "lying_min"]
_TS_CANDIDATES = ["timestamp", "time", "datetime", "date_time", "record_time"]
_COW_CANDIDATES = ["cow_id", "cow", "animal_id", "ear_tag", "tag_id"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return first candidate column that exists in df (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_mmcows(
    path: str | Path,
    resample_freq: str = "1h",
    column_overrides: Optional[Dict[str, str]] = None,
    nrows: Optional[int] = None,
) -> tuple[EnvTimeseries, CowActivityTimeseries]:
    """
    Load a MmCows-format CSV (or parquet), detect columns, resample,
    and return canonical schema objects.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the CSV/parquet file.
    resample_freq : str
        Pandas offset alias for resampling (e.g. "1h", "30min").
    column_overrides : dict, optional
        ``{raw_col_name: canonical_role}`` where canonical_role is one of
        "temp_c", "rh_pct", "thi", "cow_id", "activity_metric", "lying_minutes",
        "timestamp".  Overrides auto-detection for that column.
    nrows : int, optional
        Read only first N rows (useful for testing).

    Returns
    -------
    (EnvTimeseries, CowActivityTimeseries)
    """
    path = Path(path)
    notes: list[str] = []

    # --- Load ---
    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
        if nrows:
            raw = raw.head(nrows)
    else:
        raw = pd.read_csv(path, nrows=nrows)

    logger.info("Loaded %d rows from %s with columns: %s", len(raw), path.name, list(raw.columns))
    notes.append(f"Source: {path.name}. Raw columns: {list(raw.columns)}")

    # Apply user overrides to a local copy of candidates
    overrides = column_overrides or {}
    rev_overrides: Dict[str, str] = {v: k for k, v in overrides.items()}  # canonical → raw

    def _resolve(canonical: str, candidates: list[str]) -> Optional[str]:
        if canonical in rev_overrides and rev_overrides[canonical] in raw.columns:
            return rev_overrides[canonical]
        return _find_col(raw, candidates)

    # --- Detect columns ---
    ts_col = _resolve("timestamp", _BARN_TEMP_CANDIDATES)  # actually TS
    ts_col = _resolve("timestamp", _TS_CANDIDATES)
    barn_col = _resolve("temp_c", _BARN_TEMP_CANDIDATES)
    ambient_col = _resolve("ambient_temp_c", _AMBIENT_TEMP_CANDIDATES)
    rh_col = _resolve("rh_pct", _RH_CANDIDATES)
    thi_col = _resolve("thi", _THI_CANDIDATES)
    activity_col = _resolve("activity_metric", _ACTIVITY_CANDIDATES)
    lying_col = _resolve("lying_minutes", _LYING_CANDIDATES)
    cow_col = _resolve("cow_id", _COW_CANDIDATES)

    # Log mapping
    col_map = {
        "timestamp": ts_col, "temp_c": barn_col, "rh_pct": rh_col,
        "thi": thi_col, "activity_metric": activity_col,
        "lying_minutes": lying_col, "cow_id": cow_col,
    }
    notes.append(f"Column mapping: {col_map}")
    logger.info("Column mapping resolved: %s", col_map)

    # --- Validate required columns ---
    if ts_col is None:
        raise ValueError("Cannot find timestamp column in dataset. Check column names or use column_overrides.")
    if barn_col is None and thi_col is None:
        raise ValueError("Neither barn_temp nor THI column found – cannot proceed.")

    # --- Parse timestamps ---
    raw = raw.copy()
    raw["_ts"] = pd.to_datetime(raw[ts_col], utc=True)

    # =========================================================================
    # Build EnvTimeseries (one row per timestamp – barn-level, not per-cow)
    # =========================================================================
    # Take unique env rows (barn env is the same for all cows at a given time)
    env_raw = raw.drop_duplicates(subset=["_ts"]).set_index("_ts").sort_index()

    env_df = pd.DataFrame(index=env_raw.index)
    env_df.index.name = ENV_TS_COLS["TIMESTAMP"]

    # temp_c
    if barn_col:
        env_df[ENV_TS_COLS["TEMP_C"]] = env_raw[barn_col].astype(float)
    else:
        env_df[ENV_TS_COLS["TEMP_C"]] = float("nan")
        notes.append("barn_temp_c not found; temp_c set to NaN.")

    # rh_pct
    if rh_col:
        env_df[ENV_TS_COLS["RH_PCT"]] = env_raw[rh_col].astype(float)
    else:
        env_df[ENV_TS_COLS["RH_PCT"]] = float("nan")
        notes.append("relative_humidity_pct not found; rh_pct set to NaN.")

    # thi – use dataset value if present, otherwise compute
    if thi_col:
        env_df[ENV_TS_COLS["THI"]] = env_raw[thi_col].astype(float)
        # Cross-check with computed values when both columns exist
        if barn_col and rh_col:
            computed = compute_thi(
                env_df[ENV_TS_COLS["TEMP_C"]], env_df[ENV_TS_COLS["RH_PCT"]]
            )
            discrepancy = (env_df[ENV_TS_COLS["THI"]] - computed).abs()
            large_disc = discrepancy[discrepancy > 2.0]
            if not large_disc.empty:
                msg = (
                    f"{len(large_disc)} timestamps with THI discrepancy > 2.0 "
                    f"between dataset value and computed value (max {discrepancy.max():.2f}). "
                    "Using dataset-provided THI."
                )
                notes.append(msg)
                logger.warning(msg)
    else:
        if barn_col and rh_col:
            env_df[ENV_TS_COLS["THI"]] = compute_thi(
                env_df[ENV_TS_COLS["TEMP_C"]], env_df[ENV_TS_COLS["RH_PCT"]]
            )
            notes.append("THI column absent; computed from barn_temp_c and relative_humidity_pct.")
        else:
            env_df[ENV_TS_COLS["THI"]] = float("nan")
            notes.append("THI absent and cannot compute (missing temp or rh).")

    # Ambient temp as extra column (retained for UI)
    if ambient_col:
        env_df["ambient_temp_c"] = env_raw[ambient_col].astype(float)

    env_df[ENV_TS_COLS["SOURCE"]] = "barn"

    # Resample to uniform frequency
    env_df = _resample_env(env_df, resample_freq)

    n_expected = len(env_df)
    n_actual = env_df[ENV_TS_COLS["TEMP_C"]].notna().sum()
    env_coverage = round(100.0 * n_actual / max(n_expected, 1), 1)

    # =========================================================================
    # Build CowActivityTimeseries (one row per cow per timestamp)
    # =========================================================================
    if activity_col is None and lying_col is None:
        raise ValueError("Neither activity nor lying column found. Cannot build CowActivityTimeseries.")
    if cow_col is None:
        raise ValueError("Cannot find cow_id column.")

    act_raw = raw[[ts_col, cow_col] + [c for c in [activity_col, lying_col] if c is not None]].copy()
    act_raw = act_raw.rename(columns={"_ts": ENV_TS_COLS["TIMESTAMP"]})
    act_raw["_ts_parsed"] = pd.to_datetime(act_raw[ts_col], utc=True)
    act_raw[COW_ACT_COLS["COW_ID"]] = act_raw[cow_col].astype(str)

    if activity_col:
        act_raw[COW_ACT_COLS["ACTIVITY"]] = act_raw[activity_col].astype(float)
    else:
        act_raw[COW_ACT_COLS["ACTIVITY"]] = float("nan")
        notes.append("activity_metric not found; set to NaN.")

    if lying_col:
        act_raw[COW_ACT_COLS["LYING"]] = act_raw[lying_col].astype(float)
    else:
        act_raw[COW_ACT_COLS["LYING"]] = float("nan")

    act_df_parts = []
    for cow_id, grp in act_raw.groupby(COW_ACT_COLS["COW_ID"]):
        grp2 = grp.set_index("_ts_parsed").sort_index()
        grp2 = grp2[[COW_ACT_COLS["ACTIVITY"], COW_ACT_COLS["LYING"]]]
        grp2 = grp2.resample(resample_freq).mean()
        grp2[COW_ACT_COLS["COW_ID"]] = cow_id
        grp2.index.name = COW_ACT_COLS["TIMESTAMP"]
        act_df_parts.append(grp2.reset_index())

    act_df = pd.concat(act_df_parts, ignore_index=True) if act_df_parts else pd.DataFrame()

    n_exp_act = len(act_df)
    n_act_act = act_df[COW_ACT_COLS["ACTIVITY"]].notna().sum() if not act_df.empty else 0
    act_coverage = round(100.0 * n_act_act / max(n_exp_act, 1), 1)

    return (
        EnvTimeseries(
            df=env_df.reset_index(),
            resample_freq=resample_freq,
            source_file=str(path),
            coverage_pct=env_coverage,
            notes=notes,
        ),
        CowActivityTimeseries(
            df=act_df,
            resample_freq=resample_freq,
            source_file=str(path),
            coverage_pct=act_coverage,
            notes=notes,
        ),
    )


def _resample_env(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample env DataFrame (indexed by timestamp) to uniform freq; gaps → NaN."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    str_cols = df.select_dtypes(exclude="number").columns.tolist()
    resampled_num = df[numeric_cols].resample(freq).mean()
    resampled_str = df[str_cols].resample(freq).first() if str_cols else pd.DataFrame(index=resampled_num.index)
    return resampled_num.join(resampled_str)
