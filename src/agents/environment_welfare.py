"""
Environment & Welfare Specialist Agent.
Accent: Teal (#0d9488)
Interprets barn temperature, relative humidity, THI, ventilation, and checks herd-wide exposure.
Adheres strictly to section A.3 shared contract.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import os

from src.agents.base import BaseSpecialistAgent
from src.core.models import AgentRequest, AgentResponse
from env_welfare_agent.src.env_agent.agent import EnvironmentWelfareAgent as ImplAgent
from env_welfare_agent.src.env_agent.adapters.mmcows import load_mmcows
from env_welfare_agent.src.env_agent.adapters.synthetic import generate_synthetic

logger = logging.getLogger(__name__)


class EnvironmentWelfareAgent(BaseSpecialistAgent):
    """
    Evaluates microclimate and ambient thermal stress on individual cow behavior
    and contextualizes against herd-wide impact.
    Delegates to the robust env_welfare_agent package.
    """

    def __init__(self, repo=None, use_llm: bool = False):
        super().__init__(
            agent_id="environment_welfare",
            accent_color="#0d9488",
            repo=repo,
            use_llm=use_llm
        )
        # Initialize the core robust implementation.
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mmcows_synthetic.csv")
        try:
            if os.path.exists(data_path):
                env_ts, act_ts = load_mmcows(data_path)
            else:
                env_ts, act_ts = generate_synthetic()
            self._impl = ImplAgent(env_ts, act_ts)
        except Exception as e:
            logger.error(f"Failed to load data for EnvironmentWelfareAgent: {e}")
            self._impl = None

    def handle_query(self, request: AgentRequest) -> AgentResponse:
        """Delegate directly to the robust implementation, recording interaction."""
        if self._impl:
            response = self._impl.handle_query(request)
            self._record_interaction(request, response)
            return response
        else:
            return super().handle_query(request)

    def _analyze(
        self,
        request: AgentRequest,
        anomaly_df: Any,
        baseline_df: Any,
        start_dt: datetime,
        end_dt: datetime
    ) -> AgentResponse:
        """Fallback if base handle_query is somehow bypassed."""
        if self._impl:
            return self._impl.handle_query(request)
        raise NotImplementedError("Fallback not implemented.")

    def as_langgraph_node(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Delegate to robust implementation for LangGraph Node."""
        if self._impl:
            return self._impl.as_langgraph_node()
        return super().as_langgraph_node()
