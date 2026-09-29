"""
Feeding & Nutrition Specialist Agent View.
Accent: Amber (#d97706)
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.agents.feeding_nutrition import FeedingNutritionAgent
from src.core.models import AgentRequest, AgentResponse, TimeWindow
from src.data.repository import get_repository
from src.ui.styles import get_custom_css
from src.ui.components import (
    render_header,
    render_verdict_card,
    render_metrics_row,
    render_evidence_limitations,
    render_simulate_query_panel,
    render_message_log,
    plot_metric_timeline,
)


def render_feeding_view():
    st.markdown(get_custom_css("feeding_nutrition"), unsafe_allow_html=True)
    repo = get_repository()

    if "feeding_agent" not in st.session_state:
        st.session_state.feeding_agent = FeedingNutritionAgent(repo=repo)
    if "feeding_last_response" not in st.session_state:
        st.session_state.feeding_last_response = None
    if "feeding_last_run" not in st.session_state:
        st.session_state.feeding_last_run = None

    agent = st.session_state.feeding_agent
    cows = repo.get_cows()
    min_dt, max_dt = repo.get_time_range()

    st.sidebar.markdown("### ⚙️ Feeding Evaluation Controls")
    st.sidebar.markdown(f"**Data Source:** `MmCows Synthetic` ({len(cows)} cows, 14 days)")

    st.sidebar.markdown("#### Quick Scenario Presets")
    preset = st.sidebar.selectbox(
        "Load Preset Scenario:",
        [
            "Custom Window",
            "Acute Mastitis / Anorexia (cow_07, Days 10-12)",
            "Metabolic Ketosis / Off-Feed (cow_12, Days 11-13)",
            "Feed Delivery Delay / Bunk Empty (Day 8)",
            "Normal Feeding Baseline (cow_02)"
        ]
    )

    if preset == "Acute Mastitis / Anorexia (cow_07, Days 10-12)":
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        default_q = "Has rumination duration or dry matter intake dropped relative to baseline?"
    elif preset == "Metabolic Ketosis / Off-Feed (cow_12, Days 11-13)":
        default_cow = "cow_12"
        default_start = (min_dt + timedelta(days=11)).date()
        default_end = (min_dt + timedelta(days=12)).date()
        default_q = "Is cow_12 showing clinical rumination depression and reduced bunk visits?"
    elif preset == "Feed Delivery Delay / Bunk Empty (Day 8)":
        default_cow = "cow_04"
        default_start = (min_dt + timedelta(days=8, hours=8)).date()
        default_end = (min_dt + timedelta(days=8, hours=16)).date()
        default_q = "Did bunk availability explain feeding duration drop across the pen?"
    else:
        default_cow = "cow_02"
        default_start = (min_dt + timedelta(days=3)).date()
        default_end = (min_dt + timedelta(days=4)).date()
        default_q = "Are feeding metrics normal relative to baseline history?"

    selected_cow = st.sidebar.selectbox("Target Cow ID:", cows, index=cows.index(default_cow) if default_cow in cows else 0)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_dt.date(), max_value=max_dt.date())
    with c2:
        end_date = st.date_input("End Date", value=default_end, min_value=min_dt.date(), max_value=max_dt.date())

    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"

    run_btn = st.sidebar.button("🔍 Run Feeding Analysis", use_container_width=True)

    tab_analysis, tab_simulate, tab_logs = st.tabs([
        "📊 Specialist Dashboard",
        "🧪 Simulate Health Agent Query",
        "📜 Message Audit Log"
    ])

    if run_btn or st.session_state.feeding_last_response is None:
        req = AgentRequest(
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=selected_cow,
            question=default_q,
            question_type="feeding_drop_check",
            anomaly_window=TimeWindow(start=start_iso, end=end_iso),
            context={"anomaly_summary": f"Nutritional evaluation for {selected_cow}"}
        )
        st.session_state.feeding_last_response = agent.handle_query(req)
        st.session_state.feeding_last_run = datetime.now()

    response: AgentResponse = st.session_state.feeding_last_response

    with tab_analysis:
        render_header(
            agent_key="feeding_nutrition",
            finding=response.finding if response else None,
            last_run_dt=st.session_state.feeding_last_run
        )

        if response:
            render_verdict_card(response)
            render_metrics_row(response.metrics)

            st.markdown("#### 📈 Rumination & Feeding Telemetry")
            cow_timeline = repo.get_cow_data(selected_cow)
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

            rum_base = response.metrics.get("rumination_duration").baseline / 24.0 if "rumination_duration" in response.metrics else None
            feed_base = response.metrics.get("feeding_duration").baseline / 24.0 if "feeding_duration" in response.metrics else None

            ch1, ch2 = st.columns(2)
            with ch1:
                fig_rum = plot_metric_timeline(
                    df=cow_timeline,
                    col_name="rumination_min",
                    title=f"Rumination Activity (min/hr) — {selected_cow}",
                    unit="min/hr",
                    color="#d97706",
                    baseline_val=rum_base,
                    anomaly_window=(start_dt, end_dt)
                )
                st.plotly_chart(fig_rum, use_container_width=True)

            with ch2:
                fig_feed = plot_metric_timeline(
                    df=cow_timeline,
                    col_name="feeding_duration_min",
                    title=f"Feeding Duration at Bunk (min/hr) — {selected_cow}",
                    unit="min/hr",
                    color="#b45309",
                    baseline_val=feed_base,
                    anomaly_window=(start_dt, end_dt)
                )
                st.plotly_chart(fig_feed, use_container_width=True)

            render_evidence_limitations(response)

    with tab_simulate:
        sim_response = render_simulate_query_panel(
            agent=agent,
            cow_id=selected_cow,
            default_question=default_q,
            default_question_type="feeding_drop_check",
            anomaly_summary=f"Behavioral shift recorded for {selected_cow}",
            start_iso=start_iso,
            end_iso=end_iso
        )
        if sim_response:
            st.session_state.feeding_last_response = sim_response
            st.session_state.feeding_last_run = datetime.now()
            st.success("Query processed! Switch to 'Specialist Dashboard' tab to inspect the interactive verdict and charts.")

    with tab_logs:
        render_message_log(agent)
