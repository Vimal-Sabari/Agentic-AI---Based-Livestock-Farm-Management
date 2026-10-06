# Integration Notes for Repository Owner

> **STATUS: NOT APPLIED**
> The following code snippet documents how to register the Feeding & Nutrition Agent in `src/orchestration/graph.py` and `app.py`. Do not apply this directly in this package; it is reserved for the repo owner during full system integration.

```python
# 1. Import the Feeding & Nutrition Agent and adapter
from feeding_agent import FeedingNutritionAgent
from feeding_agent.src.feeding_agent.adapters.mmcows import load_mmcows

# 2. Initialize timeseries telemetry and agent instance
feeding_ts = load_mmcows("data/mmcows_synthetic.csv")
feeding_agent = FeedingNutritionAgent(ts=feeding_ts)

# 3. Add node to LangGraph StateGraph builder in src/orchestration/graph.py
builder.add_node("feeding_nutrition", feeding_agent.as_langgraph_node())
```
