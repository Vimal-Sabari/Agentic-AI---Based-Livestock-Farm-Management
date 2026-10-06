"""
Summary generation for AgentResponse.

Template-based by default.  If config.summarizer.use_llm is True and a valid
API key is available, the Google Gemini API rewrites the template output into
natural language.  The LLM is NEVER allowed to alter numbers; it only rewrites
the template text.  If the LLM fails, the template string is used as fallback.

No Streamlit or LangGraph imports.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from env_welfare_agent.src.env_agent.config import SummarizerConfig
from env_welfare_agent.src.env_agent.schemas import (
    AlignmentResult,
    BaselineStats,
    ExposureMetrics,
    HerdDeviationResult,
    VerdictBreakdown,
)

logger = logging.getLogger(__name__)


def build_summary(
    exposure: ExposureMetrics,
    baseline: BaselineStats,
    herd: HerdDeviationResult,
    alignment: AlignmentResult,
    verdict: VerdictBreakdown,
    cow_id: str,
    cfg: Optional[SummarizerConfig] = None,
) -> str:
    """
    Generate a 1-3 sentence human-readable summary.

    All numbers in the output are taken directly from the computed metrics;
    no values are invented.

    Parameters
    ----------
    exposure, baseline, herd, alignment, verdict : computed analysis objects
    cow_id : str
    cfg : SummarizerConfig, optional

    Returns
    -------
    str — the final summary text (template or LLM-rephrased).
    """
    scfg = cfg or SummarizerConfig()
    template = _build_template(exposure, baseline, herd, alignment, verdict, cow_id)

    if scfg.use_llm:
        try:
            rephrased = _llm_rephrase(template, scfg)
            if rephrased:
                logger.info("LLM-generated summary used.")
                return rephrased
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM summarisation failed (%s); falling back to template.", exc)

    return template


def _build_template(
    exposure: ExposureMetrics,
    baseline: BaselineStats,
    herd: HerdDeviationResult,
    alignment: AlignmentResult,
    verdict: VerdictBreakdown,
    cow_id: str,
) -> str:
    """
    Build template string from computed metrics.
    Every number in the output MUST appear in the metric dicts passed to the caller.
    """
    parts: list[str] = []

    # ---- Environmental summary ----
    if not math.isnan(exposure.mean_thi):
        delta_pct = 0.0
        if not math.isnan(baseline.mean_thi) and baseline.mean_thi > 0:
            delta_pct = (exposure.mean_thi - baseline.mean_thi) / baseline.mean_thi * 100.0

        stress_hours = (
            exposure.hours_mild + exposure.hours_moderate
            + exposure.hours_severe + exposure.hours_emergency
        )
        # Select dominant non-comfort band for description
        band_label = exposure.dominant_stress_band
        band_map = {
            "mild": "mild", "moderate": "moderate",
            "severe": "severe", "emergency": "emergency",
            "comfort": "comfort", "unknown": "unknown",
        }
        band_str = band_map.get(band_label, band_label)

        env_sent = (
            f"THI averaged {exposure.mean_thi:.1f} in the anomaly window versus "
            f"{baseline.mean_thi:.1f} at baseline ({delta_pct:+.1f}%), "
            f"with {stress_hours:.0f} hour(s) in the {band_str} stress range."
        )
        parts.append(env_sent)
    else:
        parts.append("Environmental temperature/humidity data was insufficient for THI analysis.")

    # ---- Herd context ----
    if herd.cows_analyzed > 0:
        herd_frac_pct = herd.fraction_deviating * 100.0
        other = herd.cows_analyzed - 1  # excluding target
        herd_sentence = (
            f"{herd.cows_deviating} of {herd.cows_analyzed} cows "
            f"({herd_frac_pct:.0f}%) show a similar activity reduction "
            f"relative to their own baselines"
        )
        if herd.is_herd_wide:
            herd_sentence += f", so an environmental explanation is plausible for {cow_id}."
        elif herd.fraction_deviating < 0.2:
            herd_sentence += (
                f"; the change appears isolated to {cow_id}, "
                "making a purely environmental cause less likely."
            )
        else:
            herd_sentence += "."
        parts.append(herd_sentence)

    # ---- Welfare / night recovery note ----
    if exposure.poor_night_recovery and not math.isnan(exposure.night_min_thi):
        parts.append(
            f"Night-time THI remained at {exposure.night_min_thi:.1f}, "
            "indicating poor nocturnal recovery which may compound daytime heat stress."
        )

    return " ".join(parts)


def _llm_rephrase(template: str, cfg: SummarizerConfig) -> Optional[str]:
    """
    Optional LLM rephrasing via Google Gemini API.
    The template is provided as context; the model is instructed NOT to alter numbers.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        logger.warning("google-generativeai package not installed; LLM fallback skipped.")
        return None

    system_prompt = (
        "You are a dairy farm veterinary assistant. Rewrite the following machine-generated "
        "environmental assessment in clear, professional language for a herd manager. "
        "IMPORTANT: Do NOT change any numbers, percentages, or factual claims. "
        "Keep it under 3 sentences. Do not add a diagnosis.\n\n"
        f"Input: {template}"
    )
    model = genai.GenerativeModel(cfg.llm_model)
    response = model.generate_content(
        system_prompt,
        generation_config={"temperature": cfg.llm_temperature},
    )
    return response.text.strip() if response.text else None
