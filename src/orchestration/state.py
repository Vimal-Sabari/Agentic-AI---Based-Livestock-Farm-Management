"""
LangGraph Multi-Agent Orchestration State Definition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class MultiAgentInvestigationState(TypedDict, total=False):
    """
    Shared blackboard state for LangGraph event-driven investigation.
    """
    cow_id: str
    start_iso: str
    end_iso: str
    baseline_start_iso: Optional[str]
    baseline_end_iso: Optional[str]
    
    # Anomaly detection results from Health Agent
    anomaly_data: Dict[str, Any]
    
    # Active query dispatched to specialist
    pending_request: Optional[Dict[str, Any]]
    
    # Accumulated specialist responses
    specialist_responses: List[Dict[str, Any]]
    latest_response: Optional[Dict[str, Any]]
    
    # Execution history and trace
    consulted_agents: List[str]
    investigation_complete: bool
    final_assessment: Optional[Dict[str, Any]]
    trace_logs: List[Dict[str, Any]]
