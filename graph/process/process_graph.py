"""
graph/process/graph.py
"""

from typing import Literal, Dict, Any
from langgraph.graph import END, StateGraph
from graph.state import AssistantState

# Node Factories
from graph.process.nodes.planner import make_planner
from graph.process.nodes.execution import make_execution_agent
# Validator is removed as requested
from graph.process.nodes.suggestor import make_suggestor_node


def build_process_graph(config: dict):
    # 1. Initialize Node Functions
    planner_node = make_planner(config)
    executor_node = make_execution_agent(config)
    suggestor_node = make_suggestor_node(config)

    workflow = StateGraph(AssistantState)

    # 2. Add Nodes
    workflow.add_node("PLANNER", planner_node)
    workflow.add_node("EXECUTION_AGENT", executor_node)
    workflow.add_node("SUGGESTOR", suggestor_node)

    workflow.set_entry_point("PLANNER")

    # 3. Routers

    def plan_router(state: AssistantState) -> Literal["clarification", "execute"]:
        """
        Check if the Planner needs more info before starting execution.
        """
        plan = state.get("process_plan")
        if plan and plan.needs_clarification:
            return "clarification"
        return "execute"

    def execution_router(state: AssistantState) -> Literal["clarification", "continue"]:
        """
        Check if the Execution Agent failed to find data and needs to ask the user.
        If yes, skip the Suggestor.
        """
        if state.get("needs_clarification"):
            return "clarification"
        return "continue"

    # 4. Edges

    # PLANNER -> (Check)
    workflow.add_conditional_edges(
        "PLANNER",
        plan_router,
        {
            "clarification": END,      # Exit to ask user
            "execute": "EXECUTION_AGENT"
        }
    )

    # EXECUTION_AGENT -> (Check)
    workflow.add_conditional_edges(
        "EXECUTION_AGENT",
        execution_router,
        {
            "clarification": END,      # Exit if data was missing
            "continue": "SUGGESTOR"    # Proceed to add coaching advice
        }
    )

    # SUGGESTOR -> END
    workflow.add_edge("SUGGESTOR", END)

    return workflow.compile()