"""
Adapters package for Feeding & Nutrition Agent.
"""
from feeding_agent.src.feeding_agent.adapters.mmcows import load_mmcows
from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.adapters.zenodo_activity import load_zenodo

__all__ = ["load_mmcows", "generate_synthetic", "load_zenodo"]
