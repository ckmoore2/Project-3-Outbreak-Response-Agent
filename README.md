# Project 3 — Autonomous Outbreak Response Planning Agent

An LLM-powered agentic system that accepts natural language outbreak planning
queries and autonomously orchestrates SZR surrogate model inference, county
profile lookups, multi-scenario comparisons, and structured report generation
— all without human intervention mid-task.

**Framework:** LangGraph ReAct  
**LLM:** Claude Sonnet (claude-sonnet-4-20250514)  
**Surrogate model:** SZRPredictor Experiment D — hidden_dims=[128,256,128], R²=0.937  
**Data:** NC county ACS/CDC PLACES HSI profiles  

---

## Setup

### 1. Copy Project 2 artifacts

```bash
cp ../Project-2-SZR-Outbreak-Outcome-Predictor/outputs/best_model.pt  model/
cp ../Project-2-SZR-Outbreak-Outcome-Predictor/outputs/scaler.pkl     model/
cp ../Project-2-SZR-Outbreak-Outcome-Predictor/model/szr_predictor.py model/
```

> **Important:** Replace `model/szr_predictor.py` with the copy from Project 2.
> The `FEATURE_COLUMNS` list in that file must exactly match what the scaler
> was fitted on. The placeholder in this repo shows the expected shape but
> the real column names come from Project 2.

Optionally copy processed county CSVs for full real-data coverage:
```bash
mkdir -p data
cp ../Project-2-SZR-Outbreak-Outcome-Predictor/data/processed/merged_county_df.csv  data/
cp ../Project-2-SZR-Outbreak-Outcome-Predictor/data/processed/nc_county_hsi_real.csv data/
```
Without the CSVs the agent falls back to `data/nc_counties.json` which covers 15
representative NC counties.

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
# Optionally set LANGCHAIN_API_KEY for LangSmith tracing
```

### 4. Run

```bash
streamlit run app.py
```

---

## Project structure

```
Project-3-Outbreak-Agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph ReAct agent (create_react_agent)
│   ├── prompts.py        # System prompt + tool documentation
│   └── tools.py          # Four LangChain @tool functions
├── model/
│   ├── szr_predictor.py  # SZRPredictor class (copy from Project 2)
│   ├── best_model.pt     # Trained weights (copy from Project 2)
│   └── scaler.pkl        # Fitted StandardScaler (copy from Project 2)
├── data/
│   ├── nc_counties.json          # Bundled fallback — 15 NC county profiles
│   ├── merged_county_df.csv      # (optional) full Project 2 county data
│   └── nc_county_hsi_real.csv    # (optional) real HSI scores
├── app.py                # Streamlit UI — live reasoning trace + report
├── requirements.txt
└── .env.example
```

---

## Agent tools

| Tool | Description |
|---|---|
| `lookup_county_profile` | Returns HSI, population, density, healthcare access for any NC county |
| `predict_outbreak` | Runs SZRPredictor for a county + intervention level, returns 5 metrics |
| `compare_interventions` | Runs all 4 intervention levels for a county, ranks by survival fraction |
| `synthesize_report` | Produces a structured markdown recommendation report |

### Intervention levels

| Level | Description | κ multiplier |
|---|---|---|
| none | No organized response | 1.0× |
| low | Informal neighborhood groups | 1.5× |
| medium | Structured local emergency response | 2.5× |
| high | Full military/EMD deployment | 4.0× |

---

## Example queries

- *"What intervention minimizes casualties in rural NC counties with low healthcare access?"*
- *"Compare outbreak outcomes for Robeson and Wake counties under a medium intervention."*
- *"Which NC county is most at risk and what response level achieves containment?"*
- *"Analyze Pitt County — no intervention vs. high military deployment."*

---

## LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable
full agent trace logging at [smith.langchain.com](https://smith.langchain.com).
Each tool call, reasoning step, and token usage is captured per run.
The Streamlit UI also shows the trace inline without LangSmith.

---

## How it works

The agent runs a **ReAct** (Reason + Act) loop via LangGraph:

```
User query
    │
    ▼
┌─────────────┐     tool call      ┌─────────────┐
│  Agent node │ ──────────────────▶│  Tool node  │
│  (LLM)      │ ◀────────────────── │  (executor) │
└─────────────┘     tool result    └─────────────┘
    │
    │  no more tool calls
    ▼
Final response (synthesize_report output)
```

1. LLM reads the query + system prompt → decides which tool to call first
2. Tool executes locally (no network calls — model runs in-process)
3. Result returned to LLM as a tool message
4. LLM reasons over result → calls next tool or produces final answer
5. Loop ends when LLM calls `synthesize_report` and emits the markdown report

---

## Techniques learned

- **LangGraph ReAct agents** — stateful tool-use loop, graph compilation, streaming
- **LangChain `@tool` decorator** — typed tool definitions with docstring-based descriptions
- **Local surrogate model inference** — loading `.pt` weights + sklearn scaler in-process
- **Streamlit streaming UI** — rendering agent events as they arrive via `agent.stream()`
- **LangSmith observability** — tracing tool calls, latency, and token usage per agent run
