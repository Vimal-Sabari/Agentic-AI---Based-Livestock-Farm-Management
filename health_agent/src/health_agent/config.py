"""
Configuration loader for Health & Behavior Lead Agent.
Loads YAML settings with strongly-typed fallback dataclasses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class HealthThresholdsConfig:
    fever_temp_threshold_c: float = 39.2
    fever_delta_pct: float = 2.0
    fever_zscore_threshold: float = 2.5
    lethargy_act_delta_pct: float = -20.0
    lethargy_zscore_threshold: float = -2.0
    restlessness_act_delta_pct: float = 35.0
    restlessness_zscore_threshold: float = 2.5
    lethargy_lying_delta_pct: float = 20.0
    restlessness_lying_delta_pct: float = -25.0
    lying_zscore_threshold: float = 2.0
    accel_drop_delta_pct: float = -25.0
    accel_zscore_threshold: float = -2.0

@dataclass
class SeverityWeightsConfig:
    temp_weight: float = 0.45
    activity_weight: float = 0.30
    lying_weight: float = 0.25


@dataclass
class BaselineConfig:
    default_days_preceding: int = 5
    min_baseline_hours: int = 24
    min_daily_coverage_pct: float = 80.0


@dataclass
class HealthAgentConfig:
    thresholds: HealthThresholdsConfig = field(default_factory=HealthThresholdsConfig)
    severity_weights: SeverityWeightsConfig = field(default_factory=SeverityWeightsConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> HealthAgentConfig:
        """
        Loads configuration from YAML file.
        Searches standard paths or falls back to robust defaults.
        """
        if config_path is None:
            # Look relative to package root
            package_root = Path(__file__).resolve().parent.parent.parent
            candidates = [
                package_root / "config" / "default.yaml",
                Path("config/default.yaml"),
            ]
            for c in candidates:
                if c.exists():
                    config_path = c
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}

                thresh_raw = raw.get("health_thresholds", {})
                weights_raw = raw.get("severity_weights", {})
                base_raw = raw.get("baseline", {})

                return cls(
                    thresholds=HealthThresholdsConfig(**{k: v for k, v in thresh_raw.items() if hasattr(HealthThresholdsConfig, k)}),
                    severity_weights=SeverityWeightsConfig(**{k: v for k, v in weights_raw.items() if hasattr(SeverityWeightsConfig, k)}),
                    baseline=BaselineConfig(**{k: v for k, v in base_raw.items() if hasattr(BaselineConfig, k)}),
                )
            except Exception as err:
                logger.warning(f"Failed to parse config at {config_path}: {err}. Using default settings.")

        return cls()
