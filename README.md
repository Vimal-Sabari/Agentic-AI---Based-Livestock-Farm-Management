# Multi-Agent Intelligent Monitoring System for Dairy Cows

An event-driven, goal-directed multi-agent system for dairy cow health, welfare, feeding, and production monitoring. It maintains an individualized behavioral and production baseline per cow and uses specialized AI agents to investigate deviations from that baseline.

Orchestrated with **LangGraph**, built with **Pydantic v2**, and visualized with **Streamlit + Plotly**.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |   Health & Behavior Agent (Lead)      |
                      |   - Individual Baselines & Anomalies  |
                      |   - Dynamic Specialist Querying       |
                      |   - Evidence Synthesis & Vet Warning  |
                      +-------------------+-------------------+
                                          |
                        Goal-directed LangGraph Inquiries
                                          |
         +--------------------------------+-------------------------------+
         |                                |                               |
+--------v------------------+  +----------v-----------------+  +----------v------------------+
| Environment & Welfare     |  | Feeding & Nutrition        |  | Production Agent            |
| Specialist (Teal #0d9488) |  | Specialist (Amber #d97706) |  | Specialist (Indigo #4f46e5) |
| - Microclimate & THI      |  | - Rumination & Intake      |  | - Milk Yield Trends         |
| - Herd-wide vs Isolated   |  | - Feed Bunk Availability   |  | - Electrical Conductivity   |
+---------------------------+  +----------------------------+  +-----------------------------+
```

### The 4 Agents
| Agent | Role | Accent | Key Metrics |
|---|---|---|---|
| **Health & Behavior** | **Lead Investigator**. Establishes cow baselines, detects anomalies (fever, lethargy, restlessness), queries specialists, and synthesizes veterinary early warning. | Rose (`#e11d48`) | Core Body Temp, Activity Index, Lying Minutes, Acceleration RMS |
| **Environment & Welfare** | **Specialist**. Interprets barn temperature, relative humidity, THI, ventilation, and checks if changes are herd-wide. | Teal (`#0d9488`) | THI, Barn Temp (°C), Humidity (%), Ventilation Status |
| **Feeding & Nutrition** | **Specialist**. Interprets feeding duration/frequency, rumination, dry matter intake, and bunk availability relative to cow history. | Amber (`#d97706`) | Feeding Duration, Rumination Minutes, Intake (kg), Bunk Visits, Feed Availability (%) |
| **Production** | **Specialist**. Interprets daily/session milk yield trends and electrical conductivity against cow's individual baseline. | Indigo (`#4f46e5`) | Daily Milk Yield (kg), Conductivity (mS/cm), Milking Duration |

---

## Shared Interface Contract (Section A.3)

All specialist agents strictly implement:
```python
def handle_query(request: AgentRequest) -> AgentResponse
```
And expose a LangGraph wrapper:
```python
agent.as_langgraph_node()
```

### Request Schema (`AgentRequest`)
```json
{
  "query_id": "uuid",
  "from_agent": "health_behavior",
  "to_agent": "environment_welfare",
  "cow_id": "cow_07",
  "question": "Could environmental conditions explain the change?",
  "question_type": "environment_check",
  "anomaly_window": {
    "start": "2026-09-11T00:00:00Z",
    "end": "2026-09-12T00:00:00Z"
  },
  "baseline_window": null,
  "context": {
    "anomaly_summary": "Activity -27%, lying time up",
    "extra": {}
  }
}
```

### Response Schema (`AgentResponse`)
```json
{
  "query_id": "same uuid",
  "from_agent": "environment_welfare",
  "to_agent": "health_behavior",
  "cow_id": "cow_07",
  "finding": "supports | does_not_support | inconclusive",
  "summary": "Human-readable 1-3 sentence interpretation.",
  "metrics": {
    "thi": {
      "value": 64.2,
      "unit": "",
      "baseline": 63.8,
      "delta_pct": 0.6
    }
  },
  "herd_context": {
    "cows_analyzed": 16,
    "cows_deviating": 1,
    "fraction_deviating": 0.06,
    "is_herd_wide": false
  },
  "evidence_strength": "strong | moderate | weak | none",
  "confidence": 0.90,
  "data_quality": {
    "coverage_pct": 100.0,
    "missing_streams": [],
    "notes": "Microclimate telemetry intact."
  },
  "limitations": ["..."],
  "suggested_followups": [
    {
      "to_agent": "feeding_nutrition",
      "reason": "Examine individual feed intake for digestive or clinical signs."
    }
  ]
}
```

---

## Data Pipeline & Adapter (Section A.2)

- **Primary Dataset Support**: Compatible with [MmCows](https://github.com/NEIS-lab/MmCows) (16 dairy cows, 14 days, UWB, IMU, core temp, lying, THI, yield) and [IceTag/SMARTBOW](https://zenodo.org/records/15005885) datasets.
- **Data Adapter Layer** (`src/data/adapter.py`): Dynamically inspects raw dataset files, maps arbitrary column names to canonical schema, computes derived indices (e.g. Thom's THI formula).
- **Synthetic Data Generator** (`src/data/synthetic_generator.py`): Automatically ships a 16-cow, 14-day dataset (5,376 records) with circadian rhythms and embedded clinical scenarios (heat stress, mastitis, off-feed) so the entire platform runs completely offline without external downloads or API keys.

---

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Tests
```bash
python -m pytest tests/ -v
```
Verifies contract conformance, Pydantic v2 schemas, edge cases (unknown cow IDs, empty windows), and LangGraph execution.

### 3. Launch Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## UI Features (Section A.4)

1. **Multi-Agent System View**:
   - Live LangGraph execution trace.
   - Synthesized veterinary decision support (Risk level, confidence, clinical hypotheses, recommended action checklist).
   - Side-by-side specialist findings matrix.
2. **Individual Specialist Views** (Environment, Feeding, Production, Health):
   - Curated theme colors (Teal, Amber, Indigo, Rose).
   - Header row with status pills and last query timestamps.
   - Key metric cards with individual baseline deltas.
   - Interactive Plotly timeline charts with baseline thresholds and anomaly shading.
   - **Simulate Health Agent Query** panel: interactive workbench displaying raw JSON `AgentRequest`, executing `handle_query`, and rendering raw JSON `AgentResponse`.
   - **Message Audit Log** tab: session history table with raw payload inspection.
