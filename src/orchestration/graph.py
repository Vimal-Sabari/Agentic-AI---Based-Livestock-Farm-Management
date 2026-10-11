"""
LangGraph Multi-Agent Orchestration Graph Template.
Exercise: Build an event-driven, goal-directed LangGraph StateGraph connecting
the Health Lead Agent with Specialist Agents (Environment, Feeding, Production).
Follow the TODOs in each node and routing function.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END

from src.core.models import AgentRequest, AgentResponse
from src.data.repository import get_repository
from src.agents.health_behavior import HealthBehaviorAgent
from src.agents.environment_welfare import EnvironmentWelfareAgent
from src.agents.feeding_nutrition import FeedingNutritionAgent
from src.agents.production import ProductionAgent
from src.orchestration.state import MultiAgentInvestigationState


def build_investigation_graph(repo=None) -> Any:
    """
    Constructs the event-driven LangGraph workflow.
    """
    repository = repo or get_repository()
    
    # 1. Initialize all 4 agents
    health_agent = HealthBehaviorAgent(repo=repository)
    env_agent = EnvironmentWelfareAgent(repo=repository)
    feeding_agent = FeedingNutritionAgent(repo=repository)
    production_agent = ProductionAgent(repo=repository)
    
    # ==========================================================================
    # NODE 1: ANOMALY DETECTION
    # ==========================================================================
    def detect_anomalies_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Extract cow_id, start_iso, end_iso, baseline_start_iso, baseline_end_iso from state.
        - Call health_agent.detect_anomalies(...)
        - Append a trace log entry to trace_logs.
        - Return updated state dict with 'anomaly_data', 'specialist_responses': [], 'consulted_agents': [], 'investigation_complete': False.
        """
        cow_id = state["cow_id"]
        start_dt = datetime.fromisoformat(state["start_iso"].replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(state["end_iso"].replace("Z", "+00:00"))
        base_start = (
            datetime.fromisoformat(state["baseline_start_iso"].replace("Z", "+00:00"))
            if state.get("baseline_start_iso")
            else None
        )
        base_end = (
            datetime.fromisoformat(state["baseline_end_iso"].replace("Z", "+00:00"))
            if state.get("baseline_end_iso")
            else None
        )

        anomaly_dict = health_agent.detect_anomalies(
            cow_id=cow_id,
            start_dt=start_dt,
            end_dt=end_dt,
            baseline_start_dt=base_start,
            baseline_end_dt=base_end,
        )

        trace = list(state.get("trace_logs", []))
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "detect_anomalies",
            "message": f"Anomaly check for {cow_id}: has_anomaly={anomaly_dict.get('has_anomaly')} (Severity: {anomaly_dict.get('severity_score', 0):.2f})",
        })
        return {
            "anomaly_data": anomaly_dict,
            "specialist_responses": [],
            "consulted_agents": [],
            "investigation_complete": False,
            "trace_logs": trace,
        }


    # ==========================================================================
    # NODE 2: HEALTH PLANNER (GOAL-DIRECTED ROUTER)
    # ==========================================================================
    def health_planner_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Check if anomaly_data has_anomaly is False. If False, set investigation_complete = True and return.
        - Use health_agent.plan_queries(anomaly_data) to get the list of planned requests.
        - Find the next planned request whose to_agent is NOT in consulted_agents.
        - If no unconsulted queries remain, set investigation_complete = True and pending_request = None.
        - Otherwise, set pending_request = request.model_dump() and append a trace log entry.
        """
        anomaly_data = state.get("anomaly_data", {})
        consulted = list(state.get("consulted_agents", []))
        responses = list(state.get("specialist_responses", []))
        trace = list(state.get("trace_logs", []))
        # 1. If cow is healthy, exit early!
        if not anomaly_data.get("has_anomaly", False):
            trace.append({
                "timestamp": datetime.now().isoformat(),
                "node": "health_planner",
                "message": "Cow vitals within normal individual baseline. No specialist inquiry needed.",
            })
            return {
                "investigation_complete": True,
                "pending_request": None,
                "trace_logs": trace,
            }

         # 2. Track all query IDs that have already been answered
        answered_query_ids = {r.get("query_id") for r in responses if r.get("query_id")}

        # 3. Check if the latest response suggested a dynamic follow-up query
        latest = state.get("latest_response")
        if latest:
            try:
                latest_resp_obj = AgentResponse(**latest)
                consulted_queries = [
                    AgentRequest(
                        query_id=r.get("query_id", ""),
                        from_agent="health_behavior",
                        to_agent=r.get("from_agent", ""),
                        cow_id=state["cow_id"],
                        question="investigation",
                        anomaly_window=latest_resp_obj.time_window,
                        context={"is_followup": True} if r.get("from_agent") in consulted else {},
                    )
                    for r in responses
                ]
                followup_req = health_agent.build_followup(
                    parent_response=latest_resp_obj,
                    anomaly_data=anomaly_data,
                    existing_queries=consulted_queries,
                    max_followups=2,
                )
                if followup_req and followup_req.query_id not in answered_query_ids:
                    trace.append({
                        "timestamp": datetime.now().isoformat(),
                        "node": "health_planner",
                        "message": f"Dispatching follow-up query to {followup_req.to_agent}: {followup_req.question}",
                    })
                    return {
                        "pending_request": followup_req.model_dump(),
                        "trace_logs": trace,
                    }
            except Exception:
                pass
        # 4. Check the standard planned queries queue
        planned_queries = health_agent.plan_queries(anomaly_data)
        next_query = None
        for q in planned_queries:
            if q.to_agent not in consulted and q.query_id not in answered_query_ids:
                next_query = q
                break

        # 5. If an unconsulted query exists, dispatch it!
        if next_query:
            trace.append({
                "timestamp": datetime.now().isoformat(),
                "node": "health_planner",
                "message": f"Dispatching goal-directed query to {next_query.to_agent} ({next_query.question_type})",
            })
            return {
                "pending_request": next_query.model_dump(),
                "trace_logs": trace,
            }
        # 6. All queries answered -> finish investigation!
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "health_planner",
            "message": f"All targeted specialists consulted ({len(consulted)}). Transitioning to clinical synthesis.",
        })
        return {
            "investigation_complete": True,
            "pending_request": None,
            "trace_logs": trace,
        }

    # ==========================================================================
    # NODES 3, 4, 5: SPECIALIST AGENT WRAPPERS
    # ==========================================================================
    def env_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Execute env_agent.as_langgraph_node()(state) or pass pending_request to env_agent.handle_query().
        - Mark 'environment_welfare' in consulted_agents.
        - Append trace log with environment finding & summary.
        """
        pending_dict = state["pending_request"]
        request = AgentRequest(**pending_dict)
        response = env_agent.handle_query(request)
        resp_dict = response.model_dump()
        responses = list(state.get("specialist_responses", []))
        responses.append(resp_dict)
        consulted = list(state.get("consulted_agents", []))
        if "environment_welfare" not in consulted:
            consulted.append("environment_welfare")
        trace = list(state.get("trace_logs", []))
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "environment_welfare",
            "message": f"Environment Agent replied: finding='{response.finding}', confidence={response.confidence*100:.0f}%",
        })
        return {
            "latest_response": resp_dict,
            "specialist_responses": responses,
            "consulted_agents": consulted,
            "trace_logs": trace,
        }

    def feeding_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Execute feeding_agent.as_langgraph_node()(state) or pass pending_request to feeding_agent.handle_query().
        - Mark 'feeding_nutrition' in consulted_agents.
        - Append trace log with feeding finding & summary.
        """
        pending_dict = state["pending_request"]
        request = AgentRequest(**pending_dict)
        response = feeding_agent.handle_query(request)
        resp_dict = response.model_dump()
        responses = list(state.get("specialist_responses", []))
        responses.append(resp_dict)
        consulted = list(state.get("consulted_agents", []))
        if "feeding_nutrition" not in consulted:
            consulted.append("feeding_nutrition")
        trace = list(state.get("trace_logs", []))
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "feeding_nutrition",
            "message": f"Feeding Agent replied: finding='{response.finding}', confidence={response.confidence*100:.0f}%",
        })
        return {
            "latest_response": resp_dict,
            "specialist_responses": responses,
            "consulted_agents": consulted,
            "trace_logs": trace,
        }

    def prod_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Execute production_agent.as_langgraph_node()(state) or pass pending_request to production_agent.handle_query().
        - Mark 'production' in consulted_agents.
        - Append trace log with production finding & summary.
        """
        pending_dict = state["pending_request"]
        request = AgentRequest(**pending_dict)
        response = production_agent.handle_query(request)
        resp_dict = response.model_dump()
        responses = list(state.get("specialist_responses", []))
        responses.append(resp_dict)
        consulted = list(state.get("consulted_agents", []))
        if "production" not in consulted:
            consulted.append("production")
        trace = list(state.get("trace_logs", []))
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "production",
            "message": f"Production Agent replied: finding='{response.finding}', confidence={response.confidence*100:.0f}%",
        })
        return {
            "latest_response": resp_dict,
            "specialist_responses": responses,
            "consulted_agents": consulted,
            "trace_logs": trace,
        }

    # ==========================================================================
    # NODE 6: MULTI-EVIDENCE SYNTHESIS
    # ==========================================================================
    def synthesize_node(state: MultiAgentInvestigationState) -> MultiAgentInvestigationState:
        """
        TODO:
        - Parse specialist_responses into AgentResponse objects.
        - Call health_agent.synthesize_assessment(cow_id, anomaly_data, specialist_responses).
        - Set final_assessment = assessment, investigation_complete = True.
        - Append trace log entry.
        """
        cow_id = state["cow_id"]
        anomaly_data = state.get("anomaly_data", {})
        raw_responses = state.get("specialist_responses", [])
        # Parse collected responses into AgentResponse objects
        responses = [AgentResponse(**r) for r in raw_responses]
        # Produce veterinary clinical decision support
        assessment = health_agent.synthesize_assessment(
            cow_id=cow_id,
            anomaly_data=anomaly_data,
            specialist_responses=responses,
        )
        trace = list(state.get("trace_logs", []))
        trace.append({
            "timestamp": datetime.now().isoformat(),
            "node": "synthesize",
            "message": f"Clinical assessment synthesized: Risk='{assessment.get('risk_level')}', Confidence={assessment.get('confidence', 0)*100:.0f}%",
        })
        return {
            "final_assessment": assessment,
            "investigation_complete": True,
            "trace_logs": trace,
        }

    # ==========================================================================
    # ROUTER FUNCTION (CONDITIONAL EDGES)
    # ==========================================================================
    def route_from_planner(state: MultiAgentInvestigationState) -> str:
        """
        TODO:
        - If investigation_complete is True or pending_request is None -> return 'synthesize'.
        - Look up pending_request['to_agent']:
            - 'environment_welfare' -> return 'environment_welfare'
            - 'feeding_nutrition' -> return 'feeding_nutrition'
            - 'production' -> return 'production'
        - Default fallback -> return 'synthesize'
        """
        if state.get("investigation_complete") or not state.get("pending_request"):
            return "synthesize"
        target = state["pending_request"].get("to_agent")
        if target in ["environment_welfare", "feeding_nutrition", "production"]:
            return target
        return "synthesize" 
    # ==========================================================================
    # GRAPH ASSEMBLY & COMPILATION
    # ==========================================================================
    # TODO:
    # builder = StateGraph(MultiAgentInvestigationState)
    # Add nodes: detect_anomalies, health_planner, environment_welfare, feeding_nutrition, production, synthesize
    # Set entry point: detect_anomalies
    # Add edge: detect_anomalies -> health_planner
    # Add conditional edges: health_planner -> route_from_planner -> {environment_welfare, feeding_nutrition, production, synthesize}
    # Add edges back from specialists: environment_welfare -> health_planner, feeding_nutrition -> health_planner, production -> health_planner
    # Add edge: synthesize -> END
    # Return builder.compile()

    builder = StateGraph(MultiAgentInvestigationState)
    # 1. Register all nodes
    builder.add_node("detect_anomalies", detect_anomalies_node)
    builder.add_node("health_planner", health_planner_node)
    builder.add_node("environment_welfare", env_node)
    builder.add_node("feeding_nutrition", feeding_node)
    builder.add_node("production", prod_node)
    builder.add_node("synthesize", synthesize_node)
    # 2. Set entry point
    builder.set_entry_point("detect_anomalies")
    # 3. Add initial edge
    builder.add_edge("detect_anomalies", "health_planner")
    # 4. Add conditional routing from planner
    builder.add_conditional_edges(
        "health_planner",
        route_from_planner,
        {
            "environment_welfare": "environment_welfare",
            "feeding_nutrition": "feeding_nutrition",
            "production": "production",
            "synthesize": "synthesize",
        },
    )
    # 5. Add loop-back edges from specialists to planner
    builder.add_edge("environment_welfare", "health_planner")
    builder.add_edge("feeding_nutrition", "health_planner")
    builder.add_edge("production", "health_planner")
    # 6. Add terminal edge
    builder.add_edge("synthesize", END)
    # 7. Compile into runnable state machine
    return builder.compile()
