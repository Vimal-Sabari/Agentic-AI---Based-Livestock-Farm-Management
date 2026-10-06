"""
Production Specialist Agent Adapter.
"""
from __future__ import annotations
import logging
import os
from typing import Dict, Any, Callable

from src.agents.base import BaseSpecialistAgent
from src.core.models import AgentRequest, AgentResponse
from production_agent.src.production_agent.agent import ProductionAgent as ImplAgent
from production_agent.src.production_agent.adapters.mmcows import load_mmcows
from production_agent.src.production_agent.adapters.synthetic import generate_synthetic

logger = logging.getLogger(__name__)

class ProductionAgent(BaseSpecialistAgent):
    """
    Evaluates milk production deviation.
    Delegates to the robust production_agent package.
    """
    def __init__(self, repo=None, use_llm: bool = False):
        super().__init__(agent_id="production", accent_color="#4f46e5", repo=repo, use_llm=use_llm)
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mmcows_synthetic.csv")
        try:
            if os.path.exists(data_path):
                ts = load_mmcows(data_path)
            else:
                ts = generate_synthetic()
            self._impl = ImplAgent(ts)
        except Exception as e:
            logger.error(f"Failed to load data for ProductionAgent: {e}")
            self._impl = None

    def handle_query(self, request: AgentRequest) -> AgentResponse:
        if self._impl:
            response = self._impl.handle_query(request)
            self._record_interaction(request, response)
            return response
        return super().handle_query(request)

    def _analyze(self, request, anomaly_df, baseline_df, start_dt, end_dt) -> AgentResponse:
        if self._impl: return self._impl.handle_query(request)
        raise NotImplementedError("Fallback not implemented.")

    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        if self._impl: return self._impl.as_langgraph_node()
        return super().as_langgraph_node()
