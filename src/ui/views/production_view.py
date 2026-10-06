"""
Production Specialist Agent View.
Accent: Indigo (#4f46e5)
"""

from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agents.production import ProductionAgent
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


def render_production_view():
    st.markdown(get_custom_css("production"), unsafe_allow_html=True)
    repo = get_repository()

    if "production_agent" not in st.session_state:
        st.session_state.production_agent = ProductionAgent(repo=repo)
    if "production_last_response" not in st.session_state:
        st.session_state.production_last_response = None
    if "production_last_run" not in st.session_state:
        st.session_state.production_last_run = None

    agent = st.session_state.production_agent
    cows = repo.get_cows()
    min_dt, max_dt = repo.get_time_range()

    st.sidebar.markdown("### ⚙️ Production Evaluation Controls")
    st.sidebar.markdown(f"**Data Source:** `MmCows Synthetic` ({len(cows)} cows, 14 days)")

    st.sidebar.markdown("#### Quick Scenario Presets")
    preset = st.sidebar.selectbox(
        "Load Preset Scenario:",
        [
            "Custom Window",
            "Mastitis Milk Yield Collapse & Conductivity Spike (cow_07, Days 10-12)",
            "Ketosis Yield Reduction (cow_12, Days 11-13)",
            "Normal Lactation Baseline (cow_05)"
        ]
    )

    if preset == "Mastitis Milk Yield Collapse & Conductivity Spike (cow_07, Days 10-12)":
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        default_q = "Has daily milk yield dropped or electrical conductivity spiked compared to baseline?"
    elif preset == "Ketosis Yield Reduction (cow_12, Days 11-13)":
        default_cow = "cow_12"
        default_start = (min_dt + timedelta(days=11)).date()
        default_end = (min_dt + timedelta(days=12)).date()
        default_q = "Has milk yield decreased concurrently with off-feed behavior?"
    else:
        default_cow = "cow_05"
        default_start = (min_dt + timedelta(days=3)).date()
        default_end = (min_dt + timedelta(days=4)).date()
        default_q = "Is milk yield consistent with historical baseline?"

    selected_cow = st.sidebar.selectbox("Target Cow ID:", cows, index=cows.index(default_cow) if default_cow in cows else 0)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=default_start, min_value=min_dt.date(), max_value=max_dt.date())
    with c2:
        end_date = st.date_input("End Date", value=default_end, min_value=min_dt.date(), max_value=max_dt.date())

    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"

    run_btn = st.sidebar.button("🔍 Run Production Analysis", use_container_width=True)

    tab_analysis, tab_simulate, tab_logs = st.tabs([
        "📊 Specialist Dashboard",
        "🧪 Simulate Health Agent Query",
        "📜 Message Audit Log"
    ])

    if run_btn or st.session_state.production_last_response is None:
        req = AgentRequest(
            from_agent="health_behavior",
            to_agent="production",
            cow_id=selected_cow,
            question=default_q,
            question_type="production_check",
            anomaly_window=TimeWindow(start=start_iso, end=end_iso),
            context={"anomaly_summary": f"Parlor assessment for {selected_cow}"}
        )
        st.session_state.production_last_response = agent.handle_query(req)
        st.session_state.production_last_run = datetime.now()

    response: AgentResponse = st.session_state.production_last_response

    with tab_analysis:
        render_header(
            agent_key="production",
            finding=response.finding if response else None,
            last_run_dt=st.session_state.production_last_run
        )

        if response:
            render_verdict_card(response)
            render_metrics_row(response.metrics)

            st.markdown("#### 📈 Parlor Milk Yield & Udder Conductivity Trends")
            cow_timeline = repo.get_cow_data(selected_cow)
            milking_sessions = cow_timeline[cow_timeline["milk_yield_kg"] > 0]
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))

            base_daily_metric = response.metrics.get("mean_yield_daily")
            base_daily = base_daily_metric.baseline if base_daily_metric else None
            base_milking_session = (base_daily / 2.0) if base_daily else None

            ch1, ch2 = st.columns(2)
            with ch1:
                fig_yield = plot_metric_timeline(
                    df=milking_sessions,
                    col_name="milk_yield_kg",
                    title=f"Session Milk Yield (kg) — {selected_cow}",
                    unit="kg",
                    color="#4f46e5",
                    baseline_val=base_milking_session,
                    anomaly_window=(start_dt, end_dt)
                )
                st.plotly_chart(fig_yield, use_container_width=True)

            with ch2:
                fig_cond = plot_metric_timeline(
                    df=cow_timeline,
                    col_name="milk_conductivity",
                    title=f"Milk Electrical Conductivity (mS/cm) — {selected_cow}",
                    unit="mS/cm",
                    color="#6366f1",
                    baseline_val=5.2,
                    anomaly_window=(start_dt, end_dt)
                )
                fig_cond.add_hline(y=6.2, line_dash="dot", line_color="#ef4444", annotation_text="Mastitis Risk Threshold (6.2)")
                st.plotly_chart(fig_cond, use_container_width=True)

            st.markdown("#### 🔬 Advanced Production Analytics")
            adv_c1, adv_c2 = st.columns(2)
            
            impl = agent._impl
            if impl:
                with adv_c1:
                    st.markdown("**Session Comparison (Morning vs Evening)**")
                    has_sess = "morning_yield" in response.metrics and "evening_yield" in response.metrics
                    if has_sess:
                        m_val = response.metrics["morning_yield"].value
                        e_val = response.metrics["evening_yield"].value
                        fig_sess = go.Figure(data=[
                            go.Bar(name='Morning', x=['Morning'], y=[m_val], marker_color='#4f46e5'),
                            go.Bar(name='Evening', x=['Evening'], y=[e_val], marker_color='#6366f1')
                        ])
                        fig_sess.update_layout(yaxis_title="Milk Yield (kg)", margin=dict(l=20, r=20, t=30, b=20), height=300)
                        st.plotly_chart(fig_sess, use_container_width=True)
                    else:
                        st.info("Per-milking session data not available.")
                        
                with adv_c2:
                    st.markdown("**Herd Yield Deviation Distribution**")
                    try:
                        df = impl.ts.df.copy()
                        all_cows = df["cow_id"].unique()
                        drops = []
                        for cid in all_cows:
                            cdf = df[df["cow_id"] == cid]
                            a_df = cdf[(cdf["timestamp"] >= start_dt) & (cdf["timestamp"] <= end_dt)]
                            b_df = cdf[cdf["timestamp"] < start_dt]
                            if not a_df.empty and not b_df.empty:
                                a_m = a_df.groupby(a_df["timestamp"].dt.date)["milk_yield_kg"].sum().mean()
                                b_m = b_df.groupby(b_df["timestamp"].dt.date)["milk_yield_kg"].sum().mean()
                                if b_m > 0:
                                    drops.append({"cow_id": cid, "pct_change": (a_m - b_m) / b_m * 100.0})
                        
                        if drops:
                            drops_df = pd.DataFrame(drops)
                            fig_dist = px.histogram(drops_df, x="pct_change", nbins=15, title="Herd % Change from Baseline")
                            # Add vertical line for target cow
                            target_drop = drops_df[drops_df["cow_id"] == selected_cow]
                            if not target_drop.empty:
                                val = target_drop.iloc[0]["pct_change"]
                                fig_dist.add_vline(x=val, line_dash="dash", line_color="red", annotation_text=f"Target: {val:.1f}%")
                            fig_dist.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300)
                            st.plotly_chart(fig_dist, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not render herd distribution: {e}")
                        
                st.markdown("**Milk Composition Trends**")
                if "fat_pct_delta" in response.metrics:
                    fat_delta = response.metrics["fat_pct_delta"].value
                    st.metric("Fat % Change from Baseline", f"{fat_delta:+.2f}%")
                    # Could add more complex composition charts here if needed
                else:
                    st.info("Milk composition data (Fat, Protein, SCC) not available in current dataset.")

            render_evidence_limitations(response)

    with tab_simulate:
        sim_response = render_simulate_query_panel(
            agent=agent,
            cow_id=selected_cow,
            default_question=default_q,
            default_question_type="production_check",
            anomaly_summary=f"Production inquiry for {selected_cow}",
            start_iso=start_iso,
            end_iso=end_iso
        )
        if sim_response:
            st.session_state.production_last_response = sim_response
            st.session_state.production_last_run = datetime.now()
            st.success("Query processed! Switch to 'Specialist Dashboard' tab to inspect the interactive verdict and charts.")

    with tab_logs:
        render_message_log(agent)
