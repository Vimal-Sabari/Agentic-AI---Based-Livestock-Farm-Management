"""
Configuration dataclass loaded from YAML.
All analytics modules import AgentConfig only; no Streamlit or LangGraph imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


@dataclass
class THIThresholds:
    comfort_max: float = 68.0
    mild_min: float = 68.0
    moderate_min: float = 72.0
    severe_min: float = 80.0
    emergency_min: float = 90.0


@dataclass
class BaselineConfig:
    default_days_before_anomaly: int = 3
    resample_freq: str = "1h"
    min_coverage_pct: float = 50.0


@dataclass
class HerdConfig:
    is_herd_wide_threshold: float = 0.50
    deviation_zscore_threshold: float = 1.5
    deviation_pct_threshold: float = 20.0
    min_cows_for_herd_check: int = 2


@dataclass
class AlignmentConfig:
    max_lag_hours: int = 24
    min_points_for_correlation: int = 6


@dataclass
class VerdictConfig:
    weight_env_signal: float = 0.35
    weight_temporal_alignment: float = 0.20
    weight_herd_wide: float = 0.30
    weight_data_coverage: float = 0.15
    supports_min_score: float = 0.55
    inconclusive_min_score: float = 0.30


@dataclass
class ExposureConfig:
    heat_load_base_thi: float = 68.0
    night_hours: List[int] = field(
        default_factory=lambda: [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
    )


@dataclass
class SummarizerConfig:
    use_llm: bool = False
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.1
    max_summary_sentences: int = 3


@dataclass
class DataConfig:
    min_coverage_pct_for_analysis: float = 40.0


@dataclass
class AgentConfig:
    """Top-level configuration for the Environment & Welfare Agent."""

    thi_thresholds: THIThresholds = field(default_factory=THIThresholds)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    herd: HerdConfig = field(default_factory=HerdConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AgentConfig":
        """Load config from YAML file; fall back to defaults if file missing."""
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not config_path.exists():
            logger.warning("Config file not found at %s – using defaults.", config_path)
            return cls()
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        try:
            return cls(
                thi_thresholds=THIThresholds(**raw.get("thi_thresholds", {})),
                baseline=BaselineConfig(**raw.get("baseline", {})),
                herd=HerdConfig(**raw.get("herd", {})),
                alignment=AlignmentConfig(**raw.get("alignment", {})),
                verdict=VerdictConfig(**raw.get("verdict", {})),
                exposure=ExposureConfig(**raw.get("exposure", {})),
                summarizer=SummarizerConfig(**raw.get("summarizer", {})),
                data=DataConfig(**raw.get("data", {})),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse config from %s: %s – using defaults.", config_path, exc)
            return cls()
