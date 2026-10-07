"""
Configuration dataclasses loaded from YAML for Feeding & Nutrition Agent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    resample_freq: str = "1h"
    min_coverage_pct: float = 40.0
    min_day_coverage_pct: float = 50.0


@dataclass
class BaselineConfig:
    default_days_before_anomaly: int = 7
    min_days_required: int = 2
    min_days_reliable: int = 5
    trim_fraction: float = 0.1


@dataclass
class DeviationConfig:
    decline_pct_threshold: float = 15.0
    increase_pct_threshold: float = 25.0
    zscore_threshold: float = 1.5
    relative_spread_floor: float = 0.05


@dataclass
class MealsConfig:
    gap_minutes: int = 30


@dataclass
class RuminationConfig:
    allow_proxy: bool = False


@dataclass
class AvailabilityConfig:
    restricted_below_pct: float = 50.0
    unknown_confidence_penalty: float = 0.10


@dataclass
class HerdConfig:
    herd_wide_fraction: float = 0.35
    min_cows: int = 4


@dataclass
class VerdictWeights:
    feeding: float = 0.35
    rumination: float = 0.25
    intake: float = 0.15
    consistency: float = 0.15
    coverage: float = 0.10


@dataclass
class VerdictConfig:
    weights: VerdictWeights = field(default_factory=VerdictWeights)
    supports_min_score: float = 0.55
    inconclusive_min_score: float = 0.30
    short_baseline_penalty: float = 0.15
    unknown_availability_penalty: float = 0.10
    missing_rumination_penalty: float = 0.15
    proxy_penalty: float = 0.15


@dataclass
class SummarizerConfig:
    use_llm: bool = False


@dataclass
class AgentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    deviation: DeviationConfig = field(default_factory=DeviationConfig)
    meals: MealsConfig = field(default_factory=MealsConfig)
    rumination: RuminationConfig = field(default_factory=RuminationConfig)
    availability: AvailabilityConfig = field(default_factory=AvailabilityConfig)
    herd: HerdConfig = field(default_factory=HerdConfig)
    verdict: VerdictConfig = field(default_factory=VerdictConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)

    @classmethod
    def load(cls, config_path: Optional[Path | str] = None) -> AgentConfig:
        if config_path is None:
            # Default location: feeding_agent/config/default.yaml relative to repository root or module
            base_dir = Path(__file__).resolve().parents[2]
            config_path = base_dir / "config" / "default.yaml"
        else:
            config_path = Path(config_path)

        data_dict: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data_dict = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to load config from %s: %s. Using default config.", config_path, e)
        else:
            logger.warning("Config file not found at %s. Using default config.", config_path)

        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AgentConfig:
        data_cfg = DataConfig(**d.get("data", {}))
        baseline_cfg = BaselineConfig(**d.get("baseline", {}))
        dev_cfg = DeviationConfig(**d.get("deviation", {}))
        meals_cfg = MealsConfig(**d.get("meals", {}))
        rum_cfg = RuminationConfig(**d.get("rumination", {}))
        avail_cfg = AvailabilityConfig(**d.get("availability", {}))
        herd_cfg = HerdConfig(**d.get("herd", {}))

        v_dict = d.get("verdict", {})
        w_dict = v_dict.get("weights", {})
        weights = VerdictWeights(**w_dict)

        # Validate weights sum to 1.0
        total_w = weights.feeding + weights.rumination + weights.intake + weights.consistency + weights.coverage
        if not abs(total_w - 1.0) < 1e-4:
            logger.warning("Verdict weights sum to %.4f != 1.0. Normalizing.", total_w)
            if total_w > 0:
                weights.feeding /= total_w
                weights.rumination /= total_w
                weights.intake /= total_w
                weights.consistency /= total_w
                weights.coverage /= total_w

        verdict_cfg = VerdictConfig(
            weights=weights,
            supports_min_score=v_dict.get("supports_min_score", 0.55),
            inconclusive_min_score=v_dict.get("inconclusive_min_score", 0.30),
            short_baseline_penalty=v_dict.get("short_baseline_penalty", 0.15),
            unknown_availability_penalty=v_dict.get("unknown_availability_penalty", 0.10),
            missing_rumination_penalty=v_dict.get("missing_rumination_penalty", 0.15),
            proxy_penalty=v_dict.get("proxy_penalty", 0.15),
        )
        summarizer_cfg = SummarizerConfig(**d.get("summarizer", {}))

        return cls(
            data=data_cfg,
            baseline=baseline_cfg,
            deviation=dev_cfg,
            meals=meals_cfg,
            rumination=rum_cfg,
            availability=avail_cfg,
            herd=herd_cfg,
            verdict=verdict_cfg,
            summarizer=summarizer_cfg,
        )
