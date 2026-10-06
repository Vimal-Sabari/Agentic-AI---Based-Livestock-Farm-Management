"""
Reusable UI Components for Specialist and Multi-Agent Views.
Complies with section A.4 specifications:
- Header row with status pill and timestamp
- Verdict card with finding, summary, and confidence
- Metrics row (st.metric tiles with deltas)
- Plotly charts
- Expandable Evidence & limitations
- "Simulate Health Agent query" panel with raw JSON request/response
- "Message log" session history table
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.models import AgentRequest, AgentResponse, MetricItem, TimeWindow
from src.ui.styles import THEME_CONFIG, STATUS_COLORS


def render_header(
    agent_key: str,
    finding: Optional[str] = None,
    last_run_dt: Optional[datetime] = None
) -> None:
    """Renders agent header row with status pill and run time."""
    cfg = THEME_CONFIG.get(agent_key, THEME_CONFIG["environment_welfare"])
    
    if finding and finding in STATUS_COLORS:
        sc = STATUS_COLORS[finding]
        pill_html = f'<span class="status-pill" style="background:{sc["bg"]}; color:{sc["text"]}; border:1px solid {sc["border"]};">{sc["label"]}</span>'
    else:
        pill_html = '<span class="status-pill" style="background:#f1f5f9; color:#475569; border:1px solid #cbd5e1;">Ready / Awaiting Query</span>'
        
    time_str = last_run_dt.strftime("%H:%M:%S UTC") if last_run_dt else "Not run yet"
    
    st.markdown(
        f"""
        <div class="agent-header">
            <div class="agent-title">
                <span class="agent-icon">{cfg['icon']}</span>
                <span>{cfg['name']}</span>
                {pill_html}
            </div>
            <div class="agent-meta">
                <span>Last query: <strong>{time_str}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_verdict_card(response: AgentResponse) -> None:
    """Renders the primary finding verdict card."""
    finding = response.finding
    sc = STATUS_COLORS.get(finding, STATUS_COLORS["no_data"])
    
    # Evidence badge
    ev_color = {
        "strong": "#dc2626",
        "moderate": "#d97706",
        "weak": "#64748b",
        "none": "#94a3b8"
    }.get(response.evidence_strength, "#64748b")
    
    is_hw = response.herd_context.is_herd_wide
    hw_badge = f'<span class="badge" style="background:#fef3c7; color:#92400e;">Herd-wide ({response.herd_context.fraction_deviating*100:.0f}%)</span>' if is_hw else '<span class="badge" style="background:#e2e8f0; color:#334155;">Isolated Cow Deviation</span>'

    st.markdown(
        f"""
        <div class="verdict-card {finding}">
            <div class="verdict-headline">{response.summary}</div>
            <div class="verdict-badges">
                <span class="badge" style="background:{sc['bg']}; color:{sc['text']}; font-weight:600; border:1px solid {sc['border']};">
                    Finding: {sc['label']}
                </span>
                <span class="badge">Confidence: {response.confidence * 100:.0f}%</span>
                <span class="badge" style="color:{ev_color};">Evidence: {response.evidence_strength.title()}</span>
                {hw_badge}
                <span class="badge">Data Coverage: {response.data_quality.coverage_pct:.0f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metrics_row(metrics: Dict[str, MetricItem]) -> None:
    """Renders st.metric tiles with value, baseline, and delta."""
    if not metrics:
        st.info("No comparative metrics available for this query window.")
        return
        
    cols = st.columns(len(metrics))
    for col, (name, item) in zip(cols, metrics.items()):
        clean_name = name.replace("_", " ").title()
        val_str = f"{item.value:g} {item.unit}".strip()
        delta_str = f"{item.delta_pct:+.1f}% vs base ({item.baseline:g} {item.unit})".strip()
        
        # Color inversion: For metrics where positive is bad vs good
        # Delta color normal in Streamlit: positive is green, negative is red
        col.metric(
            label=clean_name,
            value=val_str,
            delta=f"{item.delta_pct:+.1f}%",
            help=f"Cow Baseline: {item.baseline:g} {item.unit}"
        )


def render_evidence_limitations(response: AgentResponse) -> None:
    """Renders expandable section for limitations and follow-ups."""
    with st.expander("📋 Evidence, Data Quality & Follow-ups", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🔬 Data Quality & Limitations")
            st.markdown(f"- **Coverage:** `{response.data_quality.coverage_pct:.1f}%`")
            if response.data_quality.missing_streams:
                st.markdown(f"- **Missing Streams:** `{', '.join(response.data_quality.missing_streams)}`")
            if response.data_quality.notes:
                st.markdown(f"- **Telemetry Notes:** {response.data_quality.notes}")
            if response.limitations:
                for lim in response.limitations:
                    st.markdown(f"- ⚠️ *{lim}*")
            else:
                st.markdown("- *No known limitations reported for this evaluation.*")
                
        with c2:
            st.markdown("##### 🔗 Suggested Follow-up Inquiries")
            if response.suggested_followups:
                for fu in response.suggested_followups:
                    agent_name = THEME_CONFIG.get(fu.to_agent, {}).get("name", fu.to_agent)
                    st.markdown(f"- **Query `{agent_name}`**: {fu.reason}")
            else:
                st.markdown("- *No additional specialist inquiries required.*")


def render_simulate_query_panel(
    agent,
    cow_id: str,
    default_question: str,
    default_question_type: str,
    anomaly_summary: str,
    start_iso: str,
    end_iso: str,
    baseline_start_iso: Optional[str] = None,
    baseline_end_iso: Optional[str] = None
) -> Optional[AgentResponse]:
    """
    Renders section A.4 interactive test bench:
    - Builds AgentRequest
    - Shows raw JSON request
    - Calls handle_query
    - Shows raw JSON AgentResponse in collapsible block
    """
    st.markdown("### 🧪 Simulate Health Agent Query")
    st.caption("Standalone evaluation panel for validating the A.3 interface contract and JSON payloads.")
    
    with st.form(key=f"sim_form_{agent.agent_id}"):
        c1, c2 = st.columns([3, 1])
        with c1:
            question_input = st.text_input("Lead Agent Question:", value=default_question)
        with c2:
            q_type = st.text_input("Question Type:", value=default_question_type)
            
        custom_summary = st.text_input("Anomaly Context Summary:", value=anomaly_summary)
        
        # Build proposed request object
        base_win = TimeWindow(start=baseline_start_iso, end=baseline_end_iso) if (baseline_start_iso and baseline_end_iso) else None
        proposed_request = AgentRequest(
            from_agent="health_behavior",
            to_agent=agent.agent_id,
            cow_id=cow_id,
            question=question_input,
            question_type=q_type,
            anomaly_window=TimeWindow(start=start_iso, end=end_iso),
            baseline_window=base_win,
            context={"anomaly_summary": custom_summary, "extra": {"source": "Streamlit Simulator"}}
        )
        
        # Display outgoing JSON
        with st.expander("📤 Outgoing `AgentRequest` JSON Payload", expanded=False):
            st.code(proposed_request.model_dump_json(indent=2), language="json")
            
        submitted = st.form_submit_button("🚀 Execute handle_query(request)")
        
    if submitted:
        with st.spinner(f"Querying {agent.agent_id}..."):
            response = agent.handle_query(proposed_request)
            
        with st.expander("📥 Incoming `AgentResponse` JSON Payload", expanded=True):
            st.code(response.model_dump_json(indent=2), language="json")
            
        return response
        
    return None


def render_message_log(agent) -> None:
    """Renders the message log tab with all session queries and responses."""
    st.markdown("### 📜 Session Message Log")
    if not agent.message_history:
        st.info("No queries have been executed in this session yet.")
        return
        
    rows = []
    for msg in agent.message_history:
        rows.append({
            "Timestamp": msg["timestamp"][:19],
            "Query ID": msg["query_id"][:8] + "...",
            "Cow ID": msg["cow_id"],
            "Question Type": msg["question_type"],
            "Finding": msg["finding"].upper(),
            "Confidence": f"{msg['confidence']*100:.0f}%",
            "Evidence": msg["evidence_strength"].title(),
        })
        
    df_log = pd.DataFrame(rows)
    st.dataframe(df_log, use_container_width=True, hide_index=True)
    
    with st.expander("Inspect Raw JSON Message Payloads"):
        idx = st.selectbox("Select interaction:", range(len(agent.message_history)), format_func=lambda i: f"#{i+1}: {rows[i]['Cow ID']} ({rows[i]['Question Type']})")
        selected = agent.message_history[idx]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Request JSON:**")
            st.code(selected["request_json"], language="json")
        with c2:
            st.markdown("**Response JSON:**")
            st.code(selected["response_json"], language="json")


def plot_metric_timeline(
    df: pd.DataFrame,
    col_name: str,
    title: str,
    unit: str,
    color: str,
    baseline_val: Optional[float] = None,
    anomaly_window: Optional[Tuple[datetime, datetime]] = None
) -> go.Figure:
    """Generates clean Plotly line chart with baseline and anomaly window shading."""
    fig = go.Figure()
    
    if col_name in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df[col_name],
            mode="lines",
            name=title,
            line=dict(color=color, width=2.2),
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Value: %{y:.2f} " + unit + "<extra></extra>"
        ))
        
    if baseline_val is not None:
        fig.add_hline(
            y=baseline_val,
            line_dash="dash",
            line_color="#64748b",
            annotation_text=f"Baseline ({baseline_val:.1f} {unit})",
            annotation_position="bottom right"
        )
        
    if anomaly_window:
        start_w, end_w = anomaly_window
        fig.add_vrect(
            x0=start_w,
            x1=end_w,
            fillcolor="#f87171",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="Anomaly Window",
            annotation_position="top left"
        )
        
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#000000")),
        margin=dict(l=40, r=20, t=40, b=30),
        height=280,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=False, linecolor="#000000"),
        yaxis=dict(showgrid=False, linecolor="#000000", title=unit),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
