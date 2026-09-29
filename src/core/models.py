from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class TimeWindow(BaseModel):
    """Time window defined by ISO-8601 start and end timestamps."""
    start: str = Field(description="ISO-8601 start timestamp")
    end: str = Field(description="ISO-8601 end timestamp")

    @field_validator("start", "end")
    @classmethod
    def validate_iso_format(cls, v: str) -> str:
        try:
            # Check ISO format compatibility
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as err:
            raise ValueError(f"Timestamp must be ISO-8601 compatible string: {v}") from err
        return v


class AgentRequest(BaseModel):
    """
    Standard request payload sent to specialist agents.
    Complies with section A.3 shared interface contract.
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = Field(description="ID/role of calling agent (e.g. 'health_behavior')")
    to_agent: str = Field(description="ID/role of target agent (e.g. 'environment_welfare', 'feeding_nutrition', 'production')")
    cow_id: str = Field(description="Target individual cow ID (e.g. 'cow_07')")
    question: str = Field(description="Natural language question")
    question_type: str = Field(description="Type/category of question (e.g. 'environment_check', 'feeding_drop_check', 'yield_impact_check')")
    anomaly_window: TimeWindow = Field(description="Time window where anomaly was detected")
    baseline_window: Optional[TimeWindow] = Field(
        default=None,
        description="Optional baseline window. If absent, agent computes baseline from historical period preceding anomaly."
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context information, e.g. {'anomaly_summary': 'Activity -27%, lying time up', 'extra': {}}"
    )


class MetricItem(BaseModel):
    """Single metric comparison relative to individual cow baseline."""
    value: float = Field(description="Observed value during anomaly window")
    unit: str = Field(default="", description="Unit of measurement (e.g. '°C', 'min/day', 'kg/milking')")
    baseline: float = Field(description="Cow's individual baseline value")
    delta_pct: float = Field(description="Percentage deviation from baseline (positive or negative)")


class HerdContext(BaseModel):
    """Context across the herd to determine if a change is isolated or herd-wide."""
    cows_analyzed: int = Field(default=0, description="Total number of cows analyzed in herd")
    cows_deviating: int = Field(default=0, description="Number of cows showing similar deviation")
    fraction_deviating: float = Field(default=0.0, description="Fraction of herd deviating (0.0 to 1.0)")
    is_herd_wide: bool = Field(default=False, description="True if herd-wide effect (typically > 35-50% deviating)")


class DataQuality(BaseModel):
    """Honest data stream quality and completeness report."""
    coverage_pct: float = Field(default=100.0, description="Temporal data coverage percentage (0.0 to 100.0)")
    missing_streams: List[str] = Field(default_factory=list, description="List of missing sensor modalities")
    notes: str = Field(default="", description="Explanatory notes on sensor noise or gaps")


class SuggestedFollowup(BaseModel):
    """Recommended followup query to another specialist."""
    to_agent: str = Field(description="Target specialist agent ID (e.g. 'production', 'feeding_nutrition')")
    reason: str = Field(description="Reason for follow-up query")


class AgentResponse(BaseModel):
    """
    Standard response payload returned by all specialist agents.
    Complies with section A.3 shared interface contract.
    """
    query_id: str = Field(description="Same UUID as matching AgentRequest")
    from_agent: str = Field(description="ID/role of responding agent")
    to_agent: str = Field(description="ID/role of requesting agent")
    cow_id: str = Field(description="Cow ID analyzed")
    finding: Literal["supports", "does_not_support", "inconclusive"] = Field(
        description="Verdict: supports hypothesis, does not support, or inconclusive"
    )
    summary: str = Field(description="Human-readable 1-3 sentence interpretation. Numbers must match metrics.")
    metrics: Dict[str, MetricItem] = Field(
        default_factory=dict,
        description="Dictionary of individual metrics compared to baseline"
    )
    herd_context: HerdContext = Field(default_factory=HerdContext)
    evidence_strength: Literal["strong", "moderate", "weak", "none"] = Field(
        default="moderate",
        description="Strength of evidence"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    data_quality: DataQuality = Field(default_factory=DataQuality)
    limitations: List[str] = Field(default_factory=list, description="Explicit caveats and limitations")
    suggested_followups: List[SuggestedFollowup] = Field(
        default_factory=list,
        description="Suggested follow-up agent queries"
    )
