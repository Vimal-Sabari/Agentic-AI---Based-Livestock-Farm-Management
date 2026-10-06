"""
Configuration dataclass loaded from YAML for the Production Agent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"


@dataclass
class BaselineConfig:
    default_days_before_anomaly: int = 7
    min_days_required: int = 1
    trim_fraction: float = 0.1


@dataclass
class DeviationConfig:
    pct_drop_threshold: float = 5.0
    zscore_threshold: float = 1.5
    min_persistence_days: int = 1


@dataclass
class TrendConfig:
    min_days_for_trend: int = 3


@dataclass
class SessionsConfig:
    morning_hour: int = 6
    evening_hour: int = 17
    difference_threshold_pct: float = 15.0


@dataclass
class HerdConfig:
    is_herd_wide_threshold: float = 0.35
    min_cows_for_herd_check: int = 2


@dataclass
class VerdictConfig:
    weight_deviation: float = 0.40
    weight_persistence: float = 0.20
    weight_trend: float = 0.15
    weight_herd_context: float = 0.25
    supports_min_score: float = 0.55
    inconclusive_min_score: float = 0.30


@dataclass
class SummarizerConfig:
    use_llm: bool = False
    llm_model: str = "gemini-1.5-flash"
    llm_temperature: float = 0.1


@dataclass
class DataConfig:
    min_coverage_pct_for_analysis: float = 40.0


@dataclass
class AgentConfig:
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    deviation: DeviationConfig = field(default_factory=DeviationConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    herd: HerdConfig = field(default_factory=HerdConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "AgentConfig":
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not config_path.exists():
            logger.warning("Config file not found at %s – using defaults.", config_path)
            return cls()
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        try:
            return cls(
                baseline=BaselineConfig(**raw.get("baseline", {})),
                deviation=DeviationConfig(**raw.get("deviation", {})),
                trend=TrendConfig(**raw.get("trend", {})),
                sessions=SessionsConfig(**raw.get("sessions", {})),
                herd=HerdConfig(**raw.get("herd", {})),
                verdict=VerdictConfig(**raw.get("verdict", {})),
                summarizer=SummarizerConfig(**raw.get("summarizer", {})),
                data=DataConfig(**raw.get("data", {})),
            )
        except Exception as exc:
            logger.error("Failed to parse config from %s: %s – using defaults.", config_path, exc)
            return cls()
