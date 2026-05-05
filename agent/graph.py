"""
agent/graph.py — LangGraph ReAct agent for the Outbreak Response Planner.

Uses langgraph.prebuilt.create_react_agent which builds:
  START → agent_node ⇄ tool_node → END

The agent node calls the LLM; tool_node dispatches to whichever tool the
LLM selected. The loop continues until the LLM emits a final text response
with no further tool calls.

LangSmith tracing is enabled automatically when LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set in .env.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    lookup_county_profile,
    predict_outbreak,
    compare_interventions,
    synthesize_report,
)

load_dotenv()

TOOLS = [
    lookup_county_profile,
    predict_outbreak,
    compare_interventions,
    synthesize_report,
]


def create_agent():
    """
    Build and return a compiled LangGraph ReAct agent.

    The agent is stateless — create a new instance per Streamlit session or
    call. State is managed inside the graph across turns.
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.0,   # deterministic — important for reproducible planning
        max_tokens=4096,
    )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    return agent
