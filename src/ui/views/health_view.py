"""
Health & Behavior Lead Investigator View.
Accent: Rose (#e11d48)
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.agents.health_behavior import HealthBehaviorAgent
from src.core.models import AgentRequest, AgentResponse, TimeWindow
from src.data.repository import get_repository
from src.ui.styles import get_custom_css
from src.ui.components import (
    render_header,
    render_metrics_row,
    plot_metric_timeline,
)


def render_health_view():
    st.markdown(get_custom_css("health_behavior"), unsafe_allow_html=True)
    repo = get_repository()

    if "health_agent" not in st.session_state:
        st.session_state.health_agent = HealthBehaviorAgent(repo=repo)

    agent = st.session_state.health_agent
    cows = repo.get_cows()
    min_dt, max_dt = repo.get_time_range()

    st.sidebar.markdown("### ⚙️ Cow Baseline & Anomaly Controls")
    st.sidebar.markdown(f"**Data Source:** `MmCows Synthetic` ({len(cows)} cows, 14 days)")

    st.sidebar.markdown("#### Quick Scenario Presets")
    preset = st.sidebar.selectbox(
        "Load Preset Scenario:",
        [
            "Custom Window",
            "Day 10-12 Clinical Mastitis Anomaly (cow_07)",
            "Day 6-7 Herd Heat Stress (cow_03)",
            "Day 11-13 Ketosis Off-Feed (cow_12)",
            "Normal Baseline (cow_01)"
        ]
    )

    if preset == "Day 10-12 Clinical Mastitis Anomaly (cow_07)":
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
    elif preset == "Day 6-7 Herd Heat Stress (cow_03)":
        default_cow = "cow_03"
        default_start = (min_dt + timedelta(days=5, hours=10)).date()
        default_end = (min_dt + timedelta(days=5, hours=20)).date()
    elif preset == "Day 11-13 Ketosis Off-Feed (cow_12)":
        default_cow = "cow_12"
        default_start = (min_dt + timedelta(days=11)).date()
        default_end = (min_dt + timedelta(days=12)).date()
    else:
        default_cow = "cow_01"
        default_start = (min_dt + timedelta(days=2)).date()
        default_end = (min_dt + timedelta(days=3)).date()

    selected_cow = st.sidebar.selectbox("Target Cow ID:", cows, index=cows.index(default_cow) if default_cow in cows else 0)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_dt.date(), max_value=max_dt.date())
    with c2:
        end_date = st.date_input("End Date", value=default_end, min_value=min_dt.date(), max_value=max_dt.date())

    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"

    start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

    anomaly = agent.detect_anomalies(
        cow_id=selected_cow,
        start_dt=start_dt,
        end_dt=end_dt
    )

    render_header(
        agent_key="health_behavior",
        finding="supports" if anomaly["has_anomaly"] else "does_not_support",
        last_run_dt=datetime.now()
    )

    # Lead summary card
    status_label = "Anomaly Detected — Needs Specialist Investigation" if anomaly["has_anomaly"] else "Normal Baseline Confirmed"
    badge_bg = "#fee2e2" if anomaly["has_anomaly"] else "#dcfce7"
    badge_text = "#b91c1c" if anomaly["has_anomaly"] else "#15803d"

    st.markdown(
        f"""
        <div class="verdict-card {'supports' if anomaly['has_anomaly'] else 'does_not_support'}">
            <div class="verdict-headline">{anomaly['anomaly_summary']}</div>
            <div class="verdict-badges">
                <span class="badge" style="background:{badge_bg}; color:{badge_text}; font-weight:600;">{status_label}</span>
                <span class="badge">Severity Score: {anomaly['severity_score'] * 100:.0f}%</span>
                <span class="badge">Fever Flag: {'YES (Core Temp Elevated)' if anomaly.get('is_fever') else 'Normal'}</span>
                <span class="badge">Lethargy Flag: {'YES (Suppressed Activity)' if anomaly.get('is_lethargic') else 'Normal'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_metrics_row(anomaly.get("metrics", {}))

    st.markdown("#### 📈 Individual Cow Vitals & Behavioral Timeline")
    cow_timeline = repo.get_cow_data(selected_cow)
    temp_base = anomaly.get("metrics", {}).get("core_body_temperature", {}).baseline if "core_body_temperature" in anomaly.get("metrics", {}) else 38.6
    act_base = anomaly.get("metrics", {}).get("activity_index", {}).baseline if "activity_index" in anomaly.get("metrics", {}) else 50.0

    ch1, ch2 = st.columns(2)
    with ch1:
        fig_temp = plot_metric_timeline(
            df=cow_timeline,
            col_name="core_body_temp_c",
            title=f"Core Body Temperature (°C) — {selected_cow}",
            unit="°C",
            color="#e11d48",
            baseline_val=temp_base,
            anomaly_window=(start_dt, end_dt)
        )
        fig_temp.add_hline(y=39.2, line_dash="dot", line_color="#ef4444", annotation_text="Fever Threshold (39.2°C)")
        st.plotly_chart(fig_temp, use_container_width=True)

    with ch2:
        fig_act = plot_metric_timeline(
            df=cow_timeline,
            col_name="activity_index",
            title=f"Activity Index — {selected_cow}",
            unit="pts",
            color="#f43f5e",
            baseline_val=act_base,
            anomaly_window=(start_dt, end_dt)
        )
        st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("---")
    st.info("💡 To run the full dynamic LangGraph multi-agent investigation and receive the synthesized veterinary decision-support report, switch to the **'Multi-Agent System (LangGraph)'** view in the sidebar.")
