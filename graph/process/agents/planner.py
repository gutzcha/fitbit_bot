# graph/process/agents/planner.py
"""
PROCESS AGENT, PLANNER
======================

Builds a structured-output LLM runnable for the PROCESS planner.

Input:
- planner_input (string)

Output:
- ProcessPlan (validated Pydantic object)
"""

from graph.helpers import make_llm
from graph.process.prompts.planner import PLANNER_PROMPT
from graph.process.schemas import ProcessPlan


def build_planner(provider: str = "ollama", model_type: str = "slow"):
    llm = make_llm(provider, model_type)
    structured_llm = llm.with_structured_output(ProcessPlan)
    return PLANNER_PROMPT | structured_llm
