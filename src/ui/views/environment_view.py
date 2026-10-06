"""
Environment & Welfare Specialist Agent View.
Accent: Teal (#0d9488)
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.environment_welfare import EnvironmentWelfareAgent
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


def render_environment_view():
    st.markdown(get_custom_css("environment_welfare"), unsafe_allow_html=True)
    repo = get_repository()

    # Session state caching for agent instance
    if "env_agent" not in st.session_state:
        st.session_state.env_agent = EnvironmentWelfareAgent(repo=repo)
    if "env_last_response" not in st.session_state:
        st.session_state.env_last_response = None
    if "env_last_run" not in st.session_state:
        st.session_state.env_last_run = None

    agent = st.session_state.env_agent
    cows = repo.get_cows()
    min_dt, max_dt = repo.get_time_range()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("### ⚙️ Evaluation Controls")
    
    # Dataset status indicator
    st.sidebar.markdown(f"**Data Source:** `MmCows Synthetic` ({len(cows)} cows, 14 days)")
    
    # Preset scenarios for quick demo
    st.sidebar.markdown("#### Quick Scenario Presets")
    preset = st.sidebar.selectbox(
        "Load Preset Scenario:",
        [
            "Custom Window",
            "Day 6-7 Herd Heat Stress (THI 82+)",
            "Day 11 Mastitis Anomaly (cow_07, Normal THI)",
            "Normal Baseline (cow_01)"
        ]
    )

    if preset == "Day 6-7 Herd Heat Stress (THI 82+)":
        default_cow = "cow_03"
        default_start = (min_dt + timedelta(days=5, hours=10)).date()
        default_end = (min_dt + timedelta(days=5, hours=20)).date()
        default_q = "Could ambient thermal stress explain the behavioral and core temperature elevations?"
    elif preset == "Day 11 Mastitis Anomaly (cow_07, Normal THI)":
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        default_q = "Could microclimate conditions explain the fever and activity drop in cow_07?"
    elif preset == "Normal Baseline (cow_01)":
        default_cow = "cow_01"
        default_start = (min_dt + timedelta(days=2)).date()
        default_end = (min_dt + timedelta(days=3)).date()
        default_q = "Is environment normal for cow_01?"
    else:
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        default_q = "Could environmental conditions explain the observed change?"

    selected_cow = st.sidebar.selectbox("Target Cow ID:", cows, index=cows.index(default_cow) if default_cow in cows else 0)

    st.sidebar.markdown("#### Anomaly Time Window")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_dt.date(), max_value=max_dt.date())
    with c2:
        end_date = st.date_input("End Date", value=default_end, min_value=min_dt.date(), max_value=max_dt.date())

    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"

    run_btn = st.sidebar.button("🔍 Run Environment Analysis", use_container_width=True)

    # Tabs for Standalone Specialist View vs Simulate Panel vs Message Log
    tab_analysis, tab_simulate, tab_logs = st.tabs([
        "📊 Specialist Dashboard",
        "🧪 Simulate Health Agent Query",
        "📜 Message Audit Log"
    ])

    # If run clicked from sidebar
    if run_btn or st.session_state.env_last_response is None:
        req = AgentRequest(
            from_agent="health_behavior",
            to_agent="environment_welfare",
            cow_id=selected_cow,
            question=default_q,
            question_type="environment_check",
            anomaly_window=TimeWindow(start=start_iso, end=end_iso),
            context={"anomaly_summary": f"Routine check for {selected_cow} in window"}
        )
        st.session_state.env_last_response = agent.handle_query(req)
        st.session_state.env_last_run = datetime.now()

    response: AgentResponse = st.session_state.env_last_response

    # --- TAB 1: DASHBOARD ---
    with tab_analysis:
        render_header(
            agent_key="environment_welfare",
            finding=response.finding if response else None,
            last_run_dt=st.session_state.env_last_run
        )

        if response:
            render_verdict_card(response)
            render_metrics_row(response.metrics)

            # Display Charts
            st.markdown("#### 📈 Barn Microclimate & THI Trend")
            cow_timeline = repo.get_cow_data(selected_cow)
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

            thi_base = response.metrics.get("thi").baseline if "thi" in response.metrics else None

            ch1, ch2 = st.columns(2)
            with ch1:
                fig_thi = plot_metric_timeline(
                    df=cow_timeline,
                    col_name="thi",
                    title=f"THI Index Timeline — {selected_cow}",
                    unit="THI",
                    color="#0d9488",
                    baseline_val=thi_base,
                    anomaly_window=(start_dt, end_dt)
                )
                # THI threshold line
                fig_thi.add_hline(y=72.0, line_dash="dot", line_color="#ef4444", annotation_text="Heat Stress Threshold (72)")
                st.plotly_chart(fig_thi, use_container_width=True)

            with ch2:
                fig_temp = plot_metric_timeline(
                    df=cow_timeline,
                    col_name="barn_temp_c",
                    title="Barn Ambient Temperature (°C)",
                    unit="°C",
                    color="#0284c7",
                    anomaly_window=(start_dt, end_dt)
                )
                st.plotly_chart(fig_temp, use_container_width=True)
                
            st.markdown("#### 🔬 Advanced Environmental Analytics")
            
            adv_c1, adv_c2 = st.columns(2)
            
            with adv_c1:
                # 1. Diurnal Profile Chart
                st.markdown("**Diurnal THI Profile (Anomaly vs Baseline)**")
                try:
                    # We can use the agent's internal data for detailed charts
                    impl = agent._impl
                    if impl:
                        env_df = impl._env_ts.df.copy()
                        env_df["hour"] = pd.to_datetime(env_df["timestamp"]).dt.hour
                        
                        base_df = env_df[pd.to_datetime(env_df["timestamp"]) < start_dt]
                        anom_df = env_df[(pd.to_datetime(env_df["timestamp"]) >= start_dt) & (pd.to_datetime(env_df["timestamp"]) <= end_dt)]
                        
                        base_diurnal = base_df.groupby("hour")["thi"].mean().reset_index()
                        anom_diurnal = anom_df.groupby("hour")["thi"].mean().reset_index()
                        
                        fig_diurnal = go.Figure()
                        fig_diurnal.add_trace(go.Scatter(x=base_diurnal["hour"], y=base_diurnal["thi"], mode='lines+markers', name='Baseline (Avg)', line=dict(color='gray', dash='dash')))
                        fig_diurnal.add_trace(go.Scatter(x=anom_diurnal["hour"], y=anom_diurnal["thi"], mode='lines+markers', name='Anomaly Window', line=dict(color='#ef4444', width=3)))
                        fig_diurnal.update_layout(xaxis_title="Hour of Day", yaxis_title="THI", margin=dict(l=20, r=20, t=30, b=20), height=300)
                        st.plotly_chart(fig_diurnal, use_container_width=True)
                    else:
                        st.info("Advanced analytics not available.")
                except Exception as e:
                    st.error(f"Could not render diurnal profile: {e}")
                    
            with adv_c2:
                # 2. Lagged Cross-Correlation
                st.markdown("**Lagged Cross-Correlation (THI vs Activity)**")
                try:
                    if impl:
                        act_df = impl._cow_act_ts.df.copy()
                        act_win = act_df[(pd.to_datetime(act_df["timestamp"]) >= start_dt) & (pd.to_datetime(act_df["timestamp"]) <= end_dt)]
                        herd_median = act_win.groupby("timestamp")["activity_metric"].median().sort_index().dropna()
                        env_series = anom_df.set_index("timestamp")["thi"].sort_index().dropna()
                        
                        # Align and compute correlation
                        combined = pd.DataFrame({"thi": env_series, "activity": herd_median}).dropna()
                        lags = list(range(0, 13))
                        r_vals = []
                        for lag in lags:
                            if lag == 0:
                                r_vals.append(combined["thi"].corr(combined["activity"]))
                            else:
                                r_vals.append(combined["thi"].corr(combined["activity"].shift(-lag)))
                                
                        fig_lag = go.Figure()
                        fig_lag.add_trace(go.Bar(x=lags, y=r_vals, marker_color=['#0d9488' if r < 0 else '#ef4444' for r in r_vals]))
                        fig_lag.update_layout(xaxis_title="Lag (Hours)", yaxis_title="Pearson Correlation (r)", margin=dict(l=20, r=20, t=30, b=20), height=300)
                        fig_lag.add_hline(y=0, line_width=1, line_color="black")
                        st.plotly_chart(fig_lag, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render cross-correlation: {e}")
                    
            # 3. Activity Heatmap
            st.markdown("**Herd-Wide Activity Heatmap (Anomaly Window)**")
            try:
                if impl:
                    act_df = impl._cow_act_ts.df.copy()
                    act_win = act_df[(pd.to_datetime(act_df["timestamp"]) >= start_dt) & (pd.to_datetime(act_df["timestamp"]) <= end_dt)]
                    heatmap_data = act_win.pivot_table(index="cow_id", columns="timestamp", values="activity_metric")
                    # simplify timestamps for x-axis
                    heatmap_data.columns = [ts.strftime("%H:%M \n%d/%m") for ts in heatmap_data.columns]
                    
                    fig_heat = px.imshow(
                        heatmap_data, 
                        color_continuous_scale="RdYlBu", 
                        aspect="auto",
                        labels=dict(x="Time", y="Cow ID", color="Activity")
                    )
                    fig_heat.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=400)
                    st.plotly_chart(fig_heat, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render activity heatmap: {e}")

            render_evidence_limitations(response)

    # --- TAB 2: SIMULATE QUERY ---
    with tab_simulate:
        sim_response = render_simulate_query_panel(
            agent=agent,
            cow_id=selected_cow,
            default_question=default_q,
            default_question_type="environment_check",
            anomaly_summary=f"Activity drop detected in {selected_cow}",
            start_iso=start_iso,
            end_iso=end_iso
        )
        if sim_response:
            st.session_state.env_last_response = sim_response
            st.session_state.env_last_run = datetime.now()
            st.success("Query processed! Switch to 'Specialist Dashboard' tab to inspect the interactive verdict and charts.")

    # --- TAB 3: MESSAGE LOG ---
    with tab_logs:
        render_message_log(agent)
