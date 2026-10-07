# Feeding & Nutrition Agent Plan

## 1. Overview
The Feeding & Nutrition Agent is a standalone specialist in a 4-agent dairy cow monitoring system. It analyzes time-series telemetry (feeding duration, visits, rumination, intake, feed availability) to detect deviations relative to individual cow baselines and evaluate herd context.

## 2. Directory & Package Layout
`feeding_agent/`
- `__init__.py` (Exports FeedingNutritionAgent, AgentConfig, FeedingTimeseries)
- `README.md`
- `INTEGRATION_NOTES.md` (Integration instructions for graph.py & app.py)
- `PLAN.md`
- `requirements.txt`
- `config/default.yaml`
- `src/feeding_agent/`
  - `__init__.py`
  - `schemas.py` (Canonical FEED_TS_COLS, FeedingTimeseries dataclass, analysis output dataclasses)
  - `config.py` (Dataclass config loaded from default.yaml)
  - `baseline.py` (Median/MAD per-day baseline and hourly diurnal profile)
  - `deviation.py` (Per-day equivalent % change and z-scores)
  - `meals.py` (Feeding bout detection and diurnal meal pattern peak shift)
  - `rumination.py` (Rumination stats and optional proxy handling)
  - `availability.py` (Feed bunk availability and restriction check)
  - `herd.py` (Herd comparison excluding target cow)
  - `verdict.py` (Scoring, finding, evidence strength, confidence calculation)
  - `summarizer.py` (Template-based summary generation with optional LLM fallback)
  - `agent.py` (FeedingNutritionAgent class, handle_query, as_langgraph_node)
  - `adapters/`
    - `__init__.py`
    - `mmcows.py` (Flexible candidate column matcher for local CSV / MmCows)
    - `zenodo_activity.py` (Best-effort IceTag/SMARTBOW dataset loader)
    - `synthetic.py` (Deterministic scenario data generator)
- `app/`
  - `streamlit_app.py` (Self-contained dashboard with 3 tabs)
  - `ui_theme.py` (CSS styling and color constants with amber accent #d97706)
- `tests/`
  - `conftest.py`
  - `test_baseline.py`
  - `test_deviation.py`
  - `test_meals.py`
  - `test_rumination.py`
  - `test_herd.py`
  - `test_verdict.py`
  - `test_adapters.py`
  - `test_agent_e2e.py`

## 3. Analysis Logic & Algorithms
- **Baseline**: Calculate per-day baseline using median and MAD (scaled 1.4826*MAD). Minimum 4 complete days required for median/MAD, fallback to trimmed mean.
- **Deviation**: Daily scaled values compared vs baseline. Robust z-score = (value - baseline_median) / max(MAD, spread_floor). Require BOTH % change >= threshold AND z-score >= threshold for significance.
- **Bout Detection**: Group contiguous feeding hours with gap <= gap_minutes (e.g., 30 min). Calculate bout count, mean bout length, and peak-hour circular shift.
- **Herd Analysis**: Target cow excluded. Evaluate % change across other cows against their own baselines. Herd-wide threshold = 35% of analyzed cows.
- **Verdict**: Weighted scoring: feeding (0.35), rumination (0.25), intake (0.15), consistency (0.15), coverage (0.10). Output finding ("supports", "does_not_support", "inconclusive"), evidence strength, and confidence.
- **Summary Text**: Deterministic template generated from metrics. Numbers in summary must strictly match metrics.

## 4. Test Strategy
- `test_baseline.py`: Test robust median/MAD calculations, outlier resilience, short baseline flag, and time window boundary enforcement.
- `test_deviation.py`: Test exact percentage deltas, z-scores, zero-variance floor, direction rules.
- `test_meals.py`: Test bout aggregation, gap merging, peak shift calculation.
- `test_rumination.py`: Test handling when rumination stream is present vs missing, proxy flags.
- `test_herd.py`: Test herd-wide detection, target cow exclusion, min cow count constraints.
- `test_verdict.py`: Test score breakdown, thresholds, inconclusive conditions, confidence penalties.
- `test_adapters.py`: Test column name fuzzy matching, UTC timestamp parsing, resampling, synthetic generator scenarios.
- `test_agent_e2e.py`: Test handle_query and as_langgraph_node across synthetic scenarios (stable, cow_specific_drop, herd_wide_drop, etc.), error resilience, contract validation, and exact summary metric matching.
