"""
Multi-Agent Orchestration & Decision Support Dashboard.
Executes the event-driven LangGraph workflow connecting all four agents:
1. Health & Behavior (Lead Investigator)
2. Environment & Welfare (Specialist)
3. Feeding & Nutrition (Specialist)
4. Production (Specialist)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from src.orchestration.graph import build_investigation_graph
from src.data.repository import get_repository
from src.ui.styles import get_custom_css, STATUS_COLORS, THEME_CONFIG


def render_multi_agent_view():
    st.markdown(get_custom_css("multi_agent"), unsafe_allow_html=True)
    repo = get_repository()

    if "langgraph_engine" not in st.session_state:
        st.session_state.langgraph_engine = build_investigation_graph(repo=repo)
    if "investigation_result" not in st.session_state:
        st.session_state.investigation_result = None

    cows = repo.get_cows()
    min_dt, max_dt = repo.get_time_range()

    st.sidebar.markdown("### 🐄 Multi-Agent Investigation Controls")
    st.sidebar.markdown(f"**Data Source:** `MmCows Synthetic` ({len(cows)} cows, 14 days)")

    st.sidebar.markdown("#### Select Clinical / Operational Scenario")
    scenario = st.sidebar.selectbox(
        "Demonstration Scenarios:",
        [
            "Scenario 1: Clinical Mastitis / Fever (cow_07, Days 10-12)",
            "Scenario 2: Herd-Wide Severe Heat Stress (cow_03, Days 6-7)",
            "Scenario 3: Subclinical Ketosis / Off-Feed (cow_12, Days 11-13)",
            "Scenario 4: Healthy Baseline Animal (cow_01, Days 2-3)",
            "Custom Time Window"
        ]
    )

    if scenario.startswith("Scenario 1"):
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        scenario_desc = "Isolated cow condition: High fever, suppressed activity, normal barn THI, rumination collapse, sharp milk yield drop."
    elif scenario.startswith("Scenario 2"):
        default_cow = "cow_03"
        default_start = (min_dt + timedelta(days=5, hours=10)).date()
        default_end = (min_dt + timedelta(days=5, hours=20)).date()
        scenario_desc = "Herd-wide condition: Extreme THI (82.5), 85%+ herd deviating, moderate intake drop. Microclimate explains behavioral changes."
    elif scenario.startswith("Scenario 3"):
        default_cow = "cow_12"
        default_start = (min_dt + timedelta(days=11)).date()
        default_end = (min_dt + timedelta(days=12)).date()
        scenario_desc = "Metabolic off-feed: Normal body temperature, severe feeding & rumination drop, moderate yield decrease."
    elif scenario.startswith("Scenario 4"):
        default_cow = "cow_01"
        default_start = (min_dt + timedelta(days=2)).date()
        default_end = (min_dt + timedelta(days=3)).date()
        scenario_desc = "Healthy control: All vitals, intake, microclimate, and milk yield remain inside individual baseline bounds."
    else:
        default_cow = "cow_07"
        default_start = (min_dt + timedelta(days=10)).date()
        default_end = (min_dt + timedelta(days=11)).date()
        scenario_desc = "Custom user-selected time window and cow ID."

    selected_cow = st.sidebar.selectbox("Subject Cow ID:", cows, index=cows.index(default_cow) if default_cow in cows else 0)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Investigation Start", value=default_start, min_value=min_dt.date(), max_value=max_dt.date())
    with c2:
        end_date = st.date_input("Investigation End", value=default_end, min_value=min_dt.date(), max_value=max_dt.date())

    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"

    st.sidebar.caption(f"ℹ️ {scenario_desc}")

    run_inv_btn = st.sidebar.button("🚀 Run LangGraph Multi-Agent Investigation", use_container_width=True)

    # Main Area Header
    st.markdown(
        """
        <div class="agent-header">
            <div class="agent-title">
                <span class="agent-icon">🐄</span>
                <span>Dairy Cow Multi-Agent Decision-Support System</span>
                <span class="status-pill" style="background:#e0f2fe; color:#0369a1; border:1px solid #bae6fd;">LangGraph Orchestrated</span>
            </div>
            <div class="agent-meta">
                <span>Architecture: <strong>Event-Driven / Goal-Directed</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Overview Cards of the 4 Agents
    st.markdown("##### 👥 Active Agent Network")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(
            """
            <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; padding:0.75rem;">
                <div style="font-weight:700; color:#9f1239; font-size:0.875rem;">🩺 Health & Behavior</div>
                <div style="font-size:0.75rem; color:#4c0519;">Lead Investigator. Baseline deviation, routing & synthesis.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with a2:
        st.markdown(
            """
            <div style="background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px; padding:0.75rem;">
                <div style="font-weight:700; color:#115e59; font-size:0.875rem;">🌡️ Environment & Welfare</div>
                <div style="font-size:0.75rem; color:#134e4a;">Specialist. Barn THI, microclimate & herd-wide exposure.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with a3:
        st.markdown(
            """
            <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:0.75rem;">
                <div style="font-weight:700; color:#92400e; font-size:0.875rem;">🌾 Feeding & Nutrition</div>
                <div style="font-size:0.75rem; color:#78350f;">Specialist. Rumination, intake, bunk visits & delivery.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with a4:
        st.markdown(
            """
            <div style="background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:0.75rem;">
                <div style="font-weight:700; color:#3730a3; font-size:0.875rem;">🥛 Production Agent</div>
                <div style="font-size:0.75rem; color:#312e81;">Specialist. Milk yield trends & conductivity markers.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Run investigation
    if run_inv_btn or st.session_state.investigation_result is None:
        with st.spinner("Health & Behavior Agent orchestrating specialist inquiries via LangGraph..."):
            graph_input = {
                "cow_id": selected_cow,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }
            res = st.session_state.langgraph_engine.invoke(graph_input)
            st.session_state.investigation_result = res

    result = st.session_state.investigation_result
    if not result:
        st.warning("Please click 'Run LangGraph Multi-Agent Investigation' to begin.")
        return

    assessment = result.get("final_assessment", {})
    risk_level = assessment.get("risk_level", "Low")

    # Risk badge colors
    risk_style = {
        "Critical": {"bg": "#fee2e2", "border": "#dc2626", "text": "#991b1b"},
        "High": {"bg": "#ffedd5", "border": "#ea580c", "text": "#9a3412"},
        "Moderate": {"bg": "#fef9c3", "border": "#ca8a04", "text": "#854d0e"},
        "Low": {"bg": "#dcfce7", "border": "#16a34a", "text": "#166534"},
    }.get(risk_level, {"bg": "#f1f5f9", "border": "#94a3b8", "text": "#334155"})

    # --- VERDICT & DECISION SUPPORT PANEL ---
    st.markdown("### 📋 Veterinary Decision-Support Early Warning")
    st.markdown(
        f"""
        <div style="background:{risk_style['bg']}; border-left:6px solid {risk_style['border']}; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <span style="font-size:1.15rem; font-weight:700; color:{risk_style['text']};">
                    Synthesized Assessment for {selected_cow}
                </span>
                <span style="background:{risk_style['border']}; color:white; font-size:0.8rem; font-weight:700; padding:0.3rem 0.8rem; border-radius:9999px; text-transform:uppercase;">
                    Risk Level: {risk_level}
                </span>
            </div>
            <div style="font-size:1.0rem; color:#1e293b; line-height:1.55; margin-bottom:1rem;">
                {assessment.get('summary_statement', 'Baseline telemetry verified.')}
            </div>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; font-size:0.8rem;">
                <span class="badge" style="background:white; color:#334155; border:1px solid #cbd5e1;">Confidence: {assessment.get('confidence', 0)*100:.0f}%</span>
                <span class="badge" style="background:white; color:#334155; border:1px solid #cbd5e1;">Specialists Consulted: {len(result.get('specialist_responses', []))}</span>
                <span class="badge" style="background:white; color:#334155; border:1px solid #cbd5e1;">Evaluation Time: {assessment.get('generated_at', '')[:19]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Actions & Possible Conditions columns
    col_cond, col_act = st.columns(2)
    with col_cond:
        st.markdown("#### 🔍 Consistent Clinical / Operational Hypotheses")
        st.caption("Statistical alignment based on multi-sensor evidence (never an autonomous diagnosis).")
        conds = assessment.get("possible_conditions", [])
        if conds:
            for cond in conds:
                st.markdown(f"- 📌 **{cond}**")
        else:
            st.markdown("- ✅ *No clinical disorder patterns identified.*")

    with col_act:
        st.markdown("#### 🩺 Recommended Veterinary / Management Actions")
        st.caption("Decision-support protocol for herd managers and attending veterinarians.")
        acts = assessment.get("recommended_actions", [])
        if acts:
            for act in acts:
                st.markdown(f"- 📋 {act}")
        else:
            st.markdown("- *Continue regular monitoring protocol.*")

    # Specialist Responses Matrix
    st.markdown("---")
    st.markdown("### 📊 Specialist Findings Matrix")
    raw_responses = result.get("specialist_responses", [])

    if raw_responses:
        m_cols = st.columns(len(raw_responses))
        for col, resp_dict in zip(m_cols, raw_responses):
            agent_key = resp_dict["from_agent"]
            cfg = THEME_CONFIG.get(agent_key, {})
            sc = STATUS_COLORS.get(resp_dict["finding"], STATUS_COLORS["no_data"])
            
            with col:
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-top:4px solid {cfg.get('accent', '#64748b')}; border-radius:8px; padding:1rem; height:100%;">
                        <div style="font-weight:700; color:#0f172a; font-size:0.9rem; margin-bottom:0.25rem;">
                            {cfg.get('icon', '')} {cfg.get('name', agent_key)}
                        </div>
                        <span class="status-pill" style="background:{sc['bg']}; color:{sc['text']}; font-size:0.7rem; margin-bottom:0.5rem; display:inline-block;">
                            {sc['label']}
                        </span>
                        <div style="font-size:0.8rem; color:#475569; line-height:1.4; margin-top:0.5rem;">
                            {resp_dict['summary']}
                        </div>
                        <div style="margin-top:0.75rem; font-size:0.75rem; color:#64748b;">
                            Confidence: <strong>{resp_dict['confidence']*100:.0f}%</strong> | Evidence: <strong>{resp_dict['evidence_strength'].title()}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No specialist consultation was required (healthy baseline cow).")

    # LangGraph Execution Trace
    st.markdown("---")
    with st.expander("🔄 Inspect LangGraph Execution Trace & Multi-Agent Event Log", expanded=False):
        traces = result.get("trace_logs", [])
        for i, tr in enumerate(traces):
            ag_cfg = THEME_CONFIG.get(tr.get("agent"), {})
            st.markdown(f"**Step {i+1} — {tr.get('step')}** (`{ag_cfg.get('name', tr.get('agent'))}` at `{tr.get('timestamp')[11:19]}`):")
            st.code(tr.get("details"), language="text")

    # Complete State Inspection
    with st.expander("📦 Complete LangGraph State Payload (JSON)", expanded=False):
        st.json(result)
