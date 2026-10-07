# Feeding & Nutrition Specialist Agent (`feeding_agent`)

The **Feeding & Nutrition Agent** is an autonomous specialist within a four-agent dairy cow monitoring system. It analyzes time-series telemetry (feeding duration, feeder bunk visits, rumination time, estimated dry matter intake, and feed bunk fill availability) to detect behavioral deviations relative to individual cow baselines.

---

## 1. Quick Start

### Installation
Ensure dependencies from `feeding_agent/requirements.txt` are installed:
```bash
pip install -r feeding_agent/requirements.txt
```

### Running Tests
Execute the isolated test suite from the repository root:
```bash
python -m pytest feeding_agent/tests -q
```

### Running the Interactive Dashboard
Launch the standalone Streamlit dashboard from the repository root:
```bash
streamlit run feeding_agent/app/streamlit_app.py
```

---

## 2. Shared Interface Contract

The agent strictly implements the shared Pydantic contract model from `src.core.models`:

- **Agent ID:** `"feeding_nutrition"`
- **Input:** `AgentRequest(query_id, from_agent, to_agent, cow_id, question, question_type, anomaly_window, baseline_window, context)`
- **Output:** `AgentResponse(query_id, from_agent, to_agent, cow_id, finding, summary, metrics, herd_context, evidence_strength, confidence, data_quality, limitations, suggested_followups)`

### Supported Question Types
- `"feeding_drop_check"` (primary orchestrator query)
- `"feeding_check"`
- `"rumination_check"`
- `"intake_check"`
- `"general"`

Unrecognized question types return a structured `inconclusive` response without crashing.

---

## 3. Synthetic Evaluation Scenarios

The agent includes a deterministic synthetic telemetry generator (`generate_synthetic`) supporting 9 scenarios:

1. `stable`: Normal baseline diurnal feeding and rumination patterns.
2. `cow_specific_drop`: Isolated drop in feeding (-25%), rumination (-30%), and intake (-25%) for `cow_01`.
3. `herd_wide_drop`: ~60% of herd cows exhibit feeding reductions (suggesting environmental or feed delivery causes).
4. `rumination_only_drop`: `cow_01` exhibits a rumination drop (-30%) while feeding duration remains stable.
5. `restricted_availability`: Feed bunk fill drops to ~35% in the anomaly window.
6. `missing_rumination`: Rumination sensor stream is completely absent (all NaN).
7. `short_baseline`: Only 3 days of historical telemetry available prior to anomaly window.
8. `sparse_coverage`: ~60% of telemetry hours are missing/NaN during the anomaly window.
9. `abnormal_increase`: `cow_01` exhibits an abnormal +30% increase in feeding duration.

---

## 4. Configuration

Configurable thresholds are loaded from `feeding_agent/config/default.yaml`:
- **Data:** Resampling frequency (`1h`), minimum anomaly coverage (`40.0%`), minimum day coverage (`50.0%`).
- **Baseline:** Default days before anomaly (`7`), minimum required days (`2`), reliable days (`5`).
- **Deviation:** Significance thresholds (decline `-15%`, z-score `1.5`, spread floor `0.05`).
- **Verdict Scoring:** Weighted combination (feeding `0.35`, rumination `0.25`, intake `0.15`, consistency `0.15`, coverage `0.10`).

---

## 5. Known Limitations

1. **Short Dataset History:** The MmCows synthetic dataset contains 14 days of telemetry, resulting in 5 to 7-day baseline windows. Baseline confidence penalties are automatically applied when baseline length is short.
2. **Hourly Sampling Frequency:** Sensor data is resampled to hourly intervals (`1h`), which merges adjacent feeding hours into single bouts and limits sub-hourly meal bout granularity.
3. **Estimated Intake:** `estimated_intake_kg` telemetry represents model-estimated intake based on feeder duration and visit frequency rather than direct load-cell DMI weigh scales.
