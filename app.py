"""
app.py — Streamlit UI for the Autonomous Outbreak Response Planning Agent.

Run:
    streamlit run app.py
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SZR Outbreak Response Agent",
    page_icon="🧟",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧟 SZR Agent")
    st.caption("Autonomous Outbreak Response Planner")

    st.markdown("---")
    st.markdown("**Model**")
    st.code("SZRPredictor Exp D\nhidden=[128,256,128]\nR²=0.937", language=None)

    st.markdown("**Framework**")
    st.code("LangGraph ReAct\nclaude-sonnet-4-20250514", language=None)

    st.markdown("---")
    st.markdown("**Example queries**")
    examples = [
        "What intervention minimizes casualties in rural NC counties with low healthcare access?",
        "Compare outbreak outcomes for Robeson and Wake counties under a medium intervention.",
        "Which NC county is most at risk and what response level achieves containment?",
        "Analyze Pitt County — what happens with no intervention vs. high military deployment?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex[:30]):
            st.session_state["query_input"] = ex

    st.markdown("---")
    langsmith_on = bool(os.getenv("LANGCHAIN_TRACING_V2"))
    st.markdown(
        f"**LangSmith tracing:** {'🟢 on' if langsmith_on else '⚪ off'}"
    )
    if langsmith_on:
        project = os.getenv("LANGCHAIN_PROJECT", "outbreak-response-agent")
        st.caption(f"Project: {project}")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Autonomous Outbreak Response Planning Agent")
st.markdown(
    "Enter a planning query below. The agent will autonomously look up county "
    "profiles, run the trained SZR surrogate model, compare intervention "
    "scenarios, and produce a structured recommendation."
)

# ── Query input ────────────────────────────────────────────────────────────────
default_query = st.session_state.get("query_input", "")
query = st.text_area(
    "Planning query",
    value=default_query,
    height=80,
    placeholder="e.g. What intervention minimizes casualties in rural NC counties with low healthcare access?",
)

run_btn = st.button("▶ Run Agent", type="primary", disabled=not query.strip())

# ── Session state ──────────────────────────────────────────────────────────────
if "trace_steps" not in st.session_state:
    st.session_state["trace_steps"] = []
if "final_report" not in st.session_state:
    st.session_state["final_report"] = ""

# ── Agent run ──────────────────────────────────────────────────────────────────
if run_btn and query.strip():
    st.session_state["trace_steps"] = []
    st.session_state["final_report"] = ""

    from agent.graph import create_agent

    agent = create_agent()

    col_trace, col_report = st.columns([1, 1], gap="large")

    with col_trace:
        st.subheader("🔍 Agent Reasoning Trace")
        trace_box = st.empty()

    with col_report:
        st.subheader("📋 Final Report")
        report_box = st.empty()

    steps: list[str] = []

    def _render_trace():
        trace_box.markdown("\n\n---\n\n".join(steps) if steps else "_Waiting for agent..._")

    _render_trace()

    try:
        for event in agent.stream(
            {"messages": [("user", query)]},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    # ── Agent thinking / tool call ─────────────────────────
                    if node_name == "agent":
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                args_str = json.dumps(tc["args"], indent=2)
                                steps.append(
                                    f"🔧 **Tool call:** `{tc['name']}`\n"
                                    f"```json\n{args_str}\n```"
                                )
                        elif hasattr(msg, "content") and msg.content:
                            # Final agent response (no more tool calls)
                            content = msg.content
                            if isinstance(content, list):
                                content = " ".join(
                                    c.get("text", "") if isinstance(c, dict) else str(c)
                                    for c in content
                                )
                            steps.append(f"🤖 **Agent:** {content}")
                            st.session_state["final_report"] = content

                    # ── Tool result ────────────────────────────────────────
                    elif node_name == "tools":
                        raw = msg.content if hasattr(msg, "content") else str(msg)
                        # Try to pretty-print JSON results
                        try:
                            parsed = json.loads(raw)
                            display = json.dumps(parsed, indent=2)
                            # If it looks like a markdown report, show as-is
                            if "# " in display:
                                steps.append(f"📊 **Tool result:**\n{raw}")
                            else:
                                steps.append(f"📊 **Tool result:**\n```json\n{display}\n```")
                        except (json.JSONDecodeError, TypeError):
                            # Markdown report or plain text
                            steps.append(f"📊 **Tool result:**\n{raw}")

                _render_trace()

    except Exception as e:
        steps.append(f"❌ **Error:** {e}")
        _render_trace()
        st.error(f"Agent error: {e}")

    # ── Render final report ────────────────────────────────────────────────
    report = st.session_state.get("final_report", "")

    # synthesize_report returns markdown directly; pull it from the last tool result
    # if the agent relayed it without modification
    if report:
        with col_report:
            report_box.markdown(report)
            st.download_button(
                "⬇ Download report (.md)",
                data=report,
                file_name="outbreak_response_report.md",
                mime="text/markdown",
            )
    else:
        with col_report:
            report_box.info("Report will appear here once the agent finishes.")

# ── Show previous results if re-rendered without new run ──────────────────────
elif st.session_state.get("final_report"):
    col_trace, col_report = st.columns([1, 1], gap="large")

    with col_trace:
        st.subheader("🔍 Agent Reasoning Trace")
        steps = st.session_state.get("trace_steps", [])
        st.markdown("\n\n---\n\n".join(steps) if steps else "_No trace recorded._")

    with col_report:
        st.subheader("📋 Final Report")
        st.markdown(st.session_state["final_report"])
        st.download_button(
            "⬇ Download report (.md)",
            data=st.session_state["final_report"],
            file_name="outbreak_response_report.md",
            mime="text/markdown",
        )
