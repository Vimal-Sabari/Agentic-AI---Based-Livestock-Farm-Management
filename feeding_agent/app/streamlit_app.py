"""
Standalone Streamlit Dashboard for Feeding & Nutrition Specialist Agent.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from src.core.models import AgentRequest, TimeWindow
from feeding_agent.app.ui_theme import (
    COLOR_AMBER,
    COLOR_GREEN,
    COLOR_GREY,
    COLOR_PRIMARY,
    COLOR_PRIMARY_LIGHT,
    COLOR_RED,
    CUSTOM_CSS,
)
from feeding_agent.src.feeding_agent.adapters.mmcows import load_mmcows
from feeding_agent.src.feeding_agent.adapters.synthetic import generate_synthetic
from feeding_agent.src.feeding_agent.adapters.zenodo_activity import load_zenodo
from feeding_agent.src.feeding_agent.agent import SUPPORTED_QUESTION_TYPES, FeedingNutritionAgent
from feeding_agent.src.feeding_agent.config import AgentConfig
from feeding_agent.src.feeding_agent.schemas import FeedingTimeseries

st.set_page_config(
    page_title="Feeding & Nutrition Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- DATA & AGENT CACHING ---
@st.cache_data(show_spinner=False)
def get_synthetic_data(scenario: str) -> FeedingTimeseries:
    return generate_synthetic(scenario=scenario)


@st.cache_data(show_spinner=False)
def get_mmcows_data(path_str: str) -> FeedingTimeseries:
    return load_mmcows(path_str)


# --- SIDEBAR CONTROLS ---
st.sidebar.title("🌾 Feeding Agent Controls")

data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic Scenario", "Local CSV Path (MmCows)", "Zenodo Activity File"],
)

ts: FeedingTimeseries | None = None

if data_source == "Synthetic Scenario":
    scenario = st.sidebar.selectbox(
        "Scenario",
        [
            "stable",
            "cow_specific_drop",
            "herd_wide_drop",
            "rumination_only_drop",
            "restricted_availability",
            "missing_rumination",
            "short_baseline",
            "sparse_coverage",
            "abnormal_increase",
        ],
        index=1,
    )
    ts = get_synthetic_data(scenario)

elif data_source == "Local CSV Path (MmCows)":
    csv_path = st.sidebar.text_input("CSV Path", value="data/mmcows_synthetic.csv")
    if Path(csv_path).exists():
        try:
            ts = get_mmcows_data(csv_path)
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    else:
        st.sidebar.warning("CSV path not found.")

else:  # Zenodo
    z_path = st.sidebar.text_input("Zenodo File Path", value="data/zenodo_activity.csv")
    if Path(z_path).exists():
        try:
            ts = load_zenodo(z_path)
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("Zenodo file not found. Download from https://zenodo.org/records/15005885")

if ts is None:
    st.info("Please select or load a valid dataset in the sidebar to proceed.")
    st.stop()

# Cow Selector
known_cows = list(ts.df["cow_id"].unique()) if "cow_id" in ts.df.columns else []
selected_cow = st.sidebar.selectbox("Target Cow ID", known_cows, index=0)

# Window Pickers
min_dt = ts.df["timestamp"].min()
max_dt = ts.df["timestamp"].max()

default_anom_start = max_dt - timedelta(days=2)
default_anom_end = max_dt

st.sidebar.subheader("Anomaly Window")
anom_start_date = st.sidebar.date_input("Anomaly Start Date", default_anom_start.date())
anom_start_time = st.sidebar.time_input("Anomaly Start Time", default_anom_start.time())
anom_end_date = st.sidebar.date_input("Anomaly End Date", default_anom_end.date())
anom_end_time = st.sidebar.time_input("Anomaly End Time", default_anom_end.time())

astart_dt = datetime.combine(anom_start_date, anom_start_time).replace(tzinfo=timezone.utc)
aend_dt = datetime.combine(anom_end_date, anom_end_time).replace(tzinfo=timezone.utc)

st.sidebar.subheader("Baseline Window")
auto_baseline = st.sidebar.checkbox("Derive automatically (7 days prior)", value=True)

if not auto_baseline:
    base_start_date = st.sidebar.date_input("Baseline Start Date", (astart_dt - timedelta(days=7)).date())
    base_end_date = st.sidebar.date_input("Baseline End Date", astart_dt.date())
    bstart_dt = datetime.combine(base_start_date, time(0, 0)).replace(tzinfo=timezone.utc)
    bend_dt = datetime.combine(base_end_date, time(0, 0)).replace(tzinfo=timezone.utc)
else:
    bstart_dt = None
    bend_dt = None

# Collapsible Threshold Overrides (Session Only)
with st.sidebar.expander("⚙️ Threshold Overrides (Session Only)"):
    override_decline = st.slider("Decline Threshold (%)", 5.0, 50.0, 15.0)
    override_zscore = st.slider("z-Score Threshold", 0.5, 3.0, 1.5)
    override_restricted = st.slider("Restricted Availability (%)", 20.0, 80.0, 50.0)

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# Session state for agent instance
cfg = AgentConfig.load()
cfg.deviation.decline_pct_threshold = override_decline
cfg.deviation.zscore_threshold = override_zscore
cfg.availability.restricted_below_pct = override_restricted

if "agent" not in st.session_state or st.session_state.get("ts_file") != ts.source_file:
    st.session_state.agent = FeedingNutritionAgent(ts=ts, cfg=cfg)
    st.session_state.ts_file = ts.source_file
else:
    # Update config in existing session agent
    st.session_state.agent.cfg = cfg
    st.session_state.agent.ts = ts

agent: FeedingNutritionAgent = st.session_state.agent

# Construct Request
req_payload = AgentRequest(
    from_agent="health_behavior",
    to_agent="feeding_nutrition",
    cow_id=selected_cow,
    question=f"Has dry matter intake or rumination dropped for {selected_cow}?",
    question_type="feeding_drop_check",
    anomaly_window=TimeWindow(start=astart_dt.isoformat(), end=aend_dt.isoformat()),
    baseline_window=TimeWindow(start=bstart_dt.isoformat(), end=bend_dt.isoformat()) if (bstart_dt and bend_dt) else None,
)

# Execute query
response = agent.handle_query(req_payload)
last_run_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

# --- HEADER ROW ---
st.markdown(
    f"""
    <div class="agent-header">
        <div>
            <h2 style="margin: 0; color: #111827;">🌾 Feeding & Nutrition Agent</h2>
            <p style="margin: 4px 0 0 0; color: #6b7280; font-size: 0.9rem;">
                Target Cow: <strong>{selected_cow}</strong> | Dataset Range: {min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')} | Last Run: {last_run_str}
            </p>
        </div>
        <div style="text-align: right;">
            <span class="status-badge badge-{response.finding}">{response.finding.replace('_', ' ')}</span>
            <div style="margin-top: 6px; font-size: 0.85rem; color: #4b5563;">
                Confidence: <strong>{response.confidence * 100:.0f}%</strong> ({response.evidence_strength} evidence)
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "🤖 Simulate Query", "📜 Message Log"])

# === TAB 1: DASHBOARD ===
with tab1:
    # Verdict Card
    finding_class = f"verdict-{response.finding}"
    st.markdown(
        f"""
        <div class="verdict-card {finding_class}">
            <h4 style="margin: 0 0 0.5rem 0; color: #111827;">Summary Finding</h4>
            <p style="margin: 0; font-size: 1.05rem; line-height: 1.5; color: #1f2937;">
                {response.summary}
            </p>
            {"<span style='display:inline-block; margin-top:8px; background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:4px; font-weight:600; font-size:0.8rem;'>Herd-Wide Phenomenon Detected</span>" if response.herd_context.is_herd_wide else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    def render_metric(col, key, title):
        item = response.metrics.get(key)
        if item:
            delta_str = f"{item.delta_pct:+.1f}% vs base"
            col.metric(title, f"{item.value:.1f} {item.unit}".strip(), delta_str)
        else:
            col.metric(title, "N/A", "No Stream")

    render_metric(m1, "feeding_duration", "Feeding Time")
    render_metric(m2, "rumination_duration", "Rumination")
    render_metric(m3, "estimated_intake", "Est. Intake")
    render_metric(m4, "bunk_visits", "Bunk Visits")
    render_metric(m5, "mean_bout_length", "Bout Length")
    render_metric(m6, "feed_availability", "Feed Avail.")

    st.markdown("---")

    # Charts Row
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Daily Feeding & Rumination Time")
        cow_df = ts.df[ts.df["cow_id"] == selected_cow].copy()
        if not cow_df.empty:
            cow_df["date"] = cow_df["timestamp"].dt.date
            daily = cow_df.groupby("date")[["feeding_minutes", "rumination_minutes"]].sum().reset_index()

            fig1 = go.Figure()
            fig1.add_trace(go.Bar(x=daily["date"], y=daily["feeding_minutes"], name="Feeding (min/day)", marker_color=COLOR_PRIMARY))
            if "rumination_minutes" in daily.columns and daily["rumination_minutes"].notna().any():
                fig1.add_trace(go.Bar(x=daily["date"], y=daily["rumination_minutes"], name="Rumination (min/day)", marker_color="#3b82f6"))

            # Baseline Band
            f_item = response.metrics.get("feeding_duration")
            if f_item and f_item.baseline > 0:
                base_val = f_item.baseline
                fig1.add_hline(y=base_val, line_dash="dash", line_color=COLOR_GREY, annotation_text=f"Baseline ({base_val:.0f} min)")
                fig1.add_hrect(
                    y0=base_val * 0.85,
                    y1=base_val * 1.15,
                    fillcolor=COLOR_PRIMARY_LIGHT,
                    opacity=0.35,
                    line_width=0,
                    annotation_text="Normal Band (±15%)",
                    annotation_position="bottom right",
                )

            # Highlight Anomaly Window
            fig1.add_vrect(
                x0=astart_dt.date(),
                x1=aend_dt.date(),
                fillcolor=COLOR_RED,
                opacity=0.15,
                line_width=0,
                annotation_text="Anomaly Window",
                annotation_position="top left",
            )
            fig1.update_layout(barmode="group", margin=dict(l=20, r=20, t=30, b=20), height=320)
            st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Diurnal Feeding Profile (Hourly)")
        if not cow_df.empty:
            cow_df["hour"] = cow_df["timestamp"].dt.hour
            anom_hours = cow_df[(cow_df["timestamp"] >= astart_dt) & (cow_df["timestamp"] < aend_dt)].groupby("hour")["feeding_minutes"].mean()
            base_hours = cow_df[cow_df["timestamp"] < astart_dt].groupby("hour")["feeding_minutes"].median()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(24)), y=[base_hours.get(h, 0) for h in range(24)], name="Baseline Median", line=dict(color=COLOR_GREY, dash="dash")))
            fig2.add_trace(go.Scatter(x=list(range(24)), y=[anom_hours.get(h, 0) for h in range(24)], name="Anomaly Window", line=dict(color=COLOR_PRIMARY, width=3)))
            fig2.update_layout(xaxis_title="Hour of Day", yaxis_title="Feeding (min/hr)", margin=dict(l=20, r=20, t=30, b=20), height=320)
            st.plotly_chart(fig2, use_container_width=True)

    # Herd Comparison Bar Chart
    st.subheader("Herd Feeding Deviation Comparison (% change vs own baseline)")
    if "cow_id" in ts.df.columns:
        cow_list = list(ts.df["cow_id"].unique())
        cow_deltas = []
        for c in cow_list:
            c_df = ts.df[ts.df["cow_id"] == c]
            base_val = c_df[c_df["timestamp"] < astart_dt]["feeding_minutes"].sum() / max(1.0, len(c_df[c_df["timestamp"] < astart_dt]) / 24.0)
            anom_val = c_df[(c_df["timestamp"] >= astart_dt) & (c_df["timestamp"] < aend_dt)]["feeding_minutes"].sum() * (24.0 / max(1.0, len(c_df[(c_df["timestamp"] >= astart_dt) & (c_df["timestamp"] < aend_dt)])))
            d_pct = ((anom_val - base_val) / max(base_val, 1e-4)) * 100.0 if base_val > 0 else 0.0
            cow_deltas.append({"cow_id": c, "delta_pct": d_pct, "is_target": (c == selected_cow)})

        h_df = pd.DataFrame(cow_deltas)
        colors = [COLOR_RED if row["is_target"] else COLOR_PRIMARY for _, row in h_df.iterrows()]

        fig3 = go.Figure(go.Bar(x=h_df["cow_id"], y=h_df["delta_pct"], marker_color=colors))
        fig3.add_hline(y=-15.0, line_dash="dot", line_color=COLOR_AMBER, annotation_text="Significance Threshold (-15%)")
        fig3.update_layout(yaxis_title="Feeding Change (%)", margin=dict(l=20, r=20, t=30, b=20), height=280)
        st.plotly_chart(fig3, use_container_width=True)

    # Expander: Evidence & Limitations
    with st.expander("🔍 Evidence Breakdown, Limitations & Follow-ups", expanded=True):
        if agent.latest_verdict:
            vb = agent.latest_verdict
            w = cfg.verdict.weights
            score_data = [
                {"Dimension": "Feeding Signal", "Raw Score (0-1)": f"{vb.feeding_score:.2f}", "Weight": f"{w.feeding:.2f}", "Contribution": f"{vb.feeding_score * w.feeding:.3f}"},
                {"Dimension": "Rumination Signal", "Raw Score (0-1)": f"{vb.rumination_score:.2f}", "Weight": f"{w.rumination:.2f}", "Contribution": f"{vb.rumination_score * w.rumination:.3f}"},
                {"Dimension": "Intake Signal", "Raw Score (0-1)": f"{vb.intake_score:.2f}", "Weight": f"{w.intake:.2f}", "Contribution": f"{vb.intake_score * w.intake:.3f}"},
                {"Dimension": "Consistency", "Raw Score (0-1)": f"{vb.consistency_score:.2f}", "Weight": f"{w.consistency:.2f}", "Contribution": f"{vb.consistency_score * w.consistency:.3f}"},
                {"Dimension": "Coverage", "Raw Score (0-1)": f"{vb.coverage_score:.2f}", "Weight": f"{w.coverage:.2f}", "Contribution": f"{vb.coverage_score * w.coverage:.3f}"},
                {"Dimension": "Total Weighted Score", "Raw Score (0-1)": "-", "Weight": "1.00", "Contribution": f"<strong>{vb.weighted_score:.3f}</strong>"},
            ]
            st.markdown("#### Transparent Score Breakdown")
            st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("#### Data Quality & Limitations")
            st.write(f"**Window Coverage:** {response.data_quality.coverage_pct:.1f}%")
            if response.data_quality.missing_streams:
                st.write(f"**Missing Sensor Modalities:** {', '.join(response.data_quality.missing_streams)}")
            if response.limitations:
                st.warning("\n".join([f"• {lim}" for lim in response.limitations]))
            else:
                st.success("No critical caveats noted.")

        with e2:
            st.markdown("#### Suggested Follow-up Queries")
            if response.suggested_followups:
                for sf in response.suggested_followups:
                    st.info(f"**Target Specialist:** `{sf.to_agent}`\n\n**Reason:** {sf.reason}")
            else:
                st.write("No follow-up specialist queries recommended.")

# === TAB 2: SIMULATE QUERY ===
with tab2:
    st.subheader("Simulate Health & Behavior Lead Agent Request")

    with st.form("sim_form"):
        sim_cow = st.selectbox("Target Cow", known_cows, index=0)
        sim_q_type = st.selectbox("Question Type", sorted(list(SUPPORTED_QUESTION_TYPES)))
        sim_q_text = st.text_input("Question Text", f"Check feeding and rumination for {sim_cow}")
        submit_sim = st.form_submit_button("Dispatch Agent Request", type="primary")

    if submit_sim:
        sim_req = AgentRequest(
            from_agent="health_behavior",
            to_agent="feeding_nutrition",
            cow_id=sim_cow,
            question=sim_q_text,
            question_type=sim_q_type,
            anomaly_window=TimeWindow(start=astart_dt.isoformat(), end=aend_dt.isoformat()),
        )
        sim_resp = agent.handle_query(sim_req)

        st.markdown("### Raw Request JSON")
        st.json(sim_req.model_dump())

        with st.expander("### Raw Agent Response JSON", expanded=True):
            st.json(sim_resp.model_dump())

# === TAB 3: MESSAGE LOG ===
with tab3:
    st.subheader("Session Interaction Log")
    if agent.message_log:
        log_rows = []
        for req, resp, ts_str in agent.message_log:
            log_rows.append({
                "Timestamp": ts_str,
                "Cow ID": req.cow_id,
                "Question Type": req.question_type,
                "Finding": resp.finding,
                "Confidence": f"{resp.confidence * 100:.0f}%",
                "Summary": resp.summary,
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)
    else:
        st.info("No queries handled in this session yet.")
