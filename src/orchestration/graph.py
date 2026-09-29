"""
LangGraph Multi-Agent Orchestration Graph.
Event-driven, goal-directed routing between Health & Behavior Lead Agent
and Specialist Agents (Environment, Feeding, Production).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import StateGraph, END

from src.core.models import AgentRequest, AgentResponse
from src.data.repository import get_repository
from src.agents.environment_welfare import EnvironmentWelfareAgent
from src.agents.feeding_nutrition import FeedingNutritionAgent
from src.agents.production import ProductionAgent
from src.agents.health_behavior import HealthBehaviorAgent
from src.orchestration.state import MultiAgentInvestigationState


def build_investigation_graph(repo=None) -> Any:
    """
    Constructs the event-driven LangGraph workflow.
    """
    repository = repo or get_repository()
    
    # Initialize agents
    health_agent = HealthBehaviorAgent(repo=repository)
    env_agent = EnvironmentWelfareAgent(repo=repository)
    feeding_agent = FeedingNutritionAgent(repo=repository)
    production_agent = ProductionAgent(repo=repository)
    
    # --- Node Implementations ---
    
    def detect_anomalies_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        cow_id = state["cow_id"]
        start_dt = datetime.fromisoformat(state["start_iso"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(state["end_iso"].replace("Z", "+00:00"))
        
        base_start = datetime.fromisoformat(state["baseline_start_iso"].replace("Z", "+00:00")) if state.get("baseline_start_iso") else None
        base_end = datetime.fromisoformat(state["baseline_end_iso"].replace("Z", "+00:00")) if state.get("baseline_end_iso") else None
        
        anomaly_data = health_agent.detect_anomalies(
            cow_id=cow_id,
            start_dt=start_dt,
            end_dt=end_dt,
            baseline_start_dt=base_start,
            baseline_end_dt=base_end
        )
        
        trace = list(state.get("trace_logs", []))
        trace.append({
            "step": "Anomaly Detection",
            "agent": "health_behavior",
            "timestamp": datetime.now().isoformat(),
            "details": f"Anomaly detected: {anomaly_data['has_anomaly']}. Severity: {anomaly_data['severity_score']}. Summary: {anomaly_data['anomaly_summary']}"
        })
        
        return {
            **state,
            "anomaly_data": anomaly_data,
            "specialist_responses": state.get("specialist_responses", []),
            "consulted_agents": state.get("consulted_agents", []),
            "trace_logs": trace,
            "investigation_complete": False
        }
    
    def health_planner_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        Health Agent evaluates accumulated evidence and decides the next goal-directed query.
        """
        cow_id = state["cow_id"]
        anomaly = state["anomaly_data"]
        consulted = set(state.get("consulted_agents", []))
        responses = state.get("specialist_responses", [])
        trace = list(state.get("trace_logs", []))
        
        # If no anomaly was detected in baseline check, complete investigation immediately
        if not anomaly.get("has_anomaly", False):
            trace.append({
                "step": "Planning Decision",
                "agent": "health_behavior",
                "timestamp": datetime.now().isoformat(),
                "details": "No physiological or behavioral anomaly detected. Moving directly to synthesis."
            })
            return {
                **state,
                "pending_request": None,
                "investigation_complete": True,
                "trace_logs": trace
            }
        
        # Priority 1: Check environment first to see if herd-wide microclimate explains the change
        if "environment_welfare" not in consulted:
            req = health_agent.create_query(
                to_agent="environment_welfare",
                cow_id=cow_id,
                start_iso=state["start_iso"],
                end_iso=state["end_iso"],
                question="Could environmental microclimate or ambient heat stress explain the observed behavioral shifts?",
                question_type="environment_check",
                anomaly_summary=anomaly["anomaly_summary"],
                baseline_start_iso=state.get("baseline_start_iso"),
                baseline_end_iso=state.get("baseline_end_iso")
            )
            trace.append({
                "step": "Query Dispatched",
                "agent": "health_behavior",
                "timestamp": datetime.now().isoformat(),
                "details": f"Querying Environment Agent (THI / microclimate check): '{req.question}'"
            })
            return {
                **state,
                "pending_request": req.model_dump(),
                "trace_logs": trace
            }
            
        # Priority 2: Check Feeding & Nutrition
        if "feeding_nutrition" not in consulted:
            req = health_agent.create_query(
                to_agent="feeding_nutrition",
                cow_id=cow_id,
                start_iso=state["start_iso"],
                end_iso=state["end_iso"],
                question="Has rumination time or dry matter intake dropped relative to the cow's own baseline?",
                question_type="feeding_drop_check",
                anomaly_summary=anomaly["anomaly_summary"],
                baseline_start_iso=state.get("baseline_start_iso"),
                baseline_end_iso=state.get("baseline_end_iso")
            )
            trace.append({
                "step": "Query Dispatched",
                "agent": "health_behavior",
                "timestamp": datetime.now().isoformat(),
                "details": f"Querying Feeding & Nutrition Agent: '{req.question}'"
            })
            return {
                **state,
                "pending_request": req.model_dump(),
                "trace_logs": trace
            }
            
        # Priority 3: Check Production if yield impact needs verification
        if "production" not in consulted:
            req = health_agent.create_query(
                to_agent="production",
                cow_id=cow_id,
                start_iso=state["start_iso"],
                end_iso=state["end_iso"],
                question="Has daily milk yield dropped or electrical conductivity spiked compared to baseline?",
                question_type="yield_impact_check",
                anomaly_summary=anomaly["anomaly_summary"],
                baseline_start_iso=state.get("baseline_start_iso"),
                baseline_end_iso=state.get("baseline_end_iso")
            )
            trace.append({
                "step": "Query Dispatched",
                "agent": "health_behavior",
                "timestamp": datetime.now().isoformat(),
                "details": f"Querying Production Agent: '{req.question}'"
            })
            return {
                **state,
                "pending_request": req.model_dump(),
                "trace_logs": trace
            }
            
        # All relevant specialists consulted
        trace.append({
            "step": "Planning Decision",
            "agent": "health_behavior",
            "timestamp": datetime.now().isoformat(),
            "details": "All specialist inquiries complete. Moving to final multi-agent synthesis."
        })
        return {
            **state,
            "pending_request": None,
            "investigation_complete": True,
            "trace_logs": trace
        }
        
    def env_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        res_state = env_agent.as_langgraph_node()(state)
        consulted = list(res_state.get("consulted_agents", []))
        if "environment_welfare" not in consulted:
            consulted.append("environment_welfare")
        trace = list(res_state.get("trace_logs", []))
        latest = res_state.get("latest_response", {})
        trace.append({
            "step": "Specialist Response",
            "agent": "environment_welfare",
            "timestamp": datetime.now().isoformat(),
            "details": f"Finding: {latest.get('finding')}. Summary: {latest.get('summary')}"
        })
        return {**res_state, "consulted_agents": consulted, "trace_logs": trace}
        
    def feeding_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        res_state = feeding_agent.as_langgraph_node()(state)
        consulted = list(res_state.get("consulted_agents", []))
        if "feeding_nutrition" not in consulted:
            consulted.append("feeding_nutrition")
        trace = list(res_state.get("trace_logs", []))
        latest = res_state.get("latest_response", {})
        trace.append({
            "step": "Specialist Response",
            "agent": "feeding_nutrition",
            "timestamp": datetime.now().isoformat(),
            "details": f"Finding: {latest.get('finding')}. Summary: {latest.get('summary')}"
        })
        return {**res_state, "consulted_agents": consulted, "trace_logs": trace}
        
    def prod_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        res_state = production_agent.as_langgraph_node()(state)
        consulted = list(res_state.get("consulted_agents", []))
        if "production" not in consulted:
            consulted.append("production")
        trace = list(res_state.get("trace_logs", []))
        latest = res_state.get("latest_response", {})
        trace.append({
            "step": "Specialist Response",
            "agent": "production",
            "timestamp": datetime.now().isoformat(),
            "details": f"Finding: {latest.get('finding')}. Summary: {latest.get('summary')}"
        })
        return {**res_state, "consulted_agents": consulted, "trace_logs": trace}
        
    def synthesize_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        cow_id = state["cow_id"]
        anomaly = state["anomaly_data"]
        raw_responses = state.get("specialist_responses", [])
        
        specialist_responses = [AgentResponse(**r) for r in raw_responses]
        assessment = health_agent.synthesize_assessment(
            cow_id=cow_id,
            anomaly_data=anomaly,
            specialist_responses=specialist_responses
        )
        
        trace = list(state.get("trace_logs", []))
        trace.append({
            "step": "Synthesis & Decision Support",
            "agent": "health_behavior",
            "timestamp": datetime.now().isoformat(),
            "details": f"Assessment complete. Risk: {assessment['risk_level']}. Confidence: {assessment['confidence']}. Actions recommended: {len(assessment['recommended_actions'])}"
        })
        
        return {
            **state,
            "final_assessment": assessment,
            "investigation_complete": True,
            "trace_logs": trace
        }
        
    # Router logic
    def route_from_planner(state: MultiAgentInvestigationState) -> str:
        if state.get("investigation_complete", False) or not state.get("pending_request"):
            return "synthesize"
        to_agent = state["pending_request"].get("to_agent")
        if to_agent == "environment_welfare":
            return "environment_welfare"
        elif to_agent == "feeding_nutrition":
            return "feeding_nutrition"
        elif to_agent == "production":
            return "production"
        return "synthesize"

    # Build Graph
    builder = StateGraph(MultiAgentInvestigationState)
    builder.add_node("detect_anomalies", detect_anomalies_node)
    builder.add_node("health_planner", health_planner_node)
    builder.add_node("environment_welfare", env_node)
    builder.add_node("feeding_nutrition", feeding_node)
    builder.add_node("production", prod_node)
    builder.add_node("synthesize", synthesize_node)
    
    builder.set_entry_point("detect_anomalies")
    builder.add_edge("detect_anomalies", "health_planner")
    
    builder.add_conditional_edges(
        "health_planner",
        route_from_planner,
        {
            "environment_welfare": "environment_welfare",
            "feeding_nutrition": "feeding_nutrition",
            "production": "production",
            "synthesize": "synthesize"
        }
    )
    
    # Return to health planner after specialist handles query
    builder.add_edge("environment_welfare", "health_planner")
    builder.add_edge("feeding_nutrition", "health_planner")
    builder.add_edge("production", "health_planner")
    
    builder.add_edge("synthesize", END)
    
    return builder.compile()
