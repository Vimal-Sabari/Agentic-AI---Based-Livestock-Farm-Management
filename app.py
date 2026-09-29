"""
Main Application Launcher for Multi-Agent Dairy Cow Intelligent Monitoring System.
Provides unified dashboard navigation allowing individual inspection of each specialist agent UI
as well as the full LangGraph-orchestrated multi-agent investigation system.
"""

from __future__ import annotations

import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Dairy Cow Multi-Agent Intelligent Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ui.views.environment_view import render_environment_view
from src.ui.views.feeding_view import render_feeding_view
from src.ui.views.production_view import render_production_view
from src.ui.views.health_view import render_health_view
from src.ui.views.multi_agent_view import render_multi_agent_view


def main():
    st.sidebar.title("🐄 Dairy Agentic AI")
    st.sidebar.caption("Individual Cow Behavioral & Production Monitoring")

    view_mode = st.sidebar.radio(
        "Agent View Selection:",
        [
            "🚀 Multi-Agent System (LangGraph)",
            "🌡️ Environment & Welfare Specialist",
            "🌾 Feeding & Nutrition Specialist",
            "🥛 Production Specialist",
            "🩺 Health & Behavior Lead Agent",
        ],
        index=0
    )

    st.sidebar.markdown("---")

    if view_mode == "🚀 Multi-Agent System (LangGraph)":
        render_multi_agent_view()
    elif view_mode == "🌡️ Environment & Welfare Specialist":
        render_environment_view()
    elif view_mode == "🌾 Feeding & Nutrition Specialist":
        render_feeding_view()
    elif view_mode == "🥛 Production Specialist":
        render_production_view()
    elif view_mode == "🩺 Health & Behavior Lead Agent":
        render_health_view()


if __name__ == "__main__":
    main()
