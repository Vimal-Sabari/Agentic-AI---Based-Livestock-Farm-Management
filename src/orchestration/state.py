"""
LangGraph Multi-Agent Orchestration State Definition.
Exercise: Define the shared blackboard state used by LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class MultiAgentInvestigationState(TypedDict, total=False):
    """
    Shared blackboard state for LangGraph event-driven investigation.
    
    TODO:
    - Define input fields:
      - cow_id: str
      - start_iso: str
      - end_iso: str
      - baseline_start_iso: Optional[str]
      - baseline_end_iso: Optional[str]
      
    - Define investigation fields:
      - anomaly_data: Dict[str, Any]             # Output from detect_anomalies node
      - planned_queries: List[Dict[str, Any]]    # Queue of requests created by health_planner
      - pending_request: Optional[Dict[str, Any]] # Active AgentRequest currently dispatched
      - specialist_responses: List[Dict[str, Any]] # Accumulated responses from specialists
      - latest_response: Optional[Dict[str, Any]]  # Most recent response received
      - consulted_agents: List[str]              # List of specialist IDs already queried
      - investigation_complete: bool             # True when planning loop finishes
      - final_assessment: Optional[Dict[str, Any]]# Synthesized veterinary decision support
      - trace_logs: List[Dict[str, Any]]         # Step-by-step audit trail for UI visualization
    """
    # TODO: Define your state fields here
    cow_id: str
    start_iso: str
    end_iso: str
    baseline_start_iso: Optional[str]
    baseline_end_iso: Optional[str]

    anomaly_data: Dict[str,Any]

    pending_request: Optional[Dict[str, Any]]
    specialist_responses: List[Dict[str, Any]]
    latest_response: Optional[Dict[str, Any]]
    consulted_agents: List[str]

    investigation_complete: bool
    final_assessment: Optional[Dict[str, Any]]
    trace_logs:List[Dict[str,Any]]

