"""
graph/graph.py
"""

from typing import Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.helpers import get_fast_slow_llm, make_llm
from graph.nodes import (make_clarification_node, make_intent_node,
                         make_static_response_node)
from graph.process.process_graph import build_process_graph
from graph.state import AssistantState

load_dotenv()


def build_graph(config: dict):
    """
    Builds the StateGraph with the specified configuration.
    """
    provider = config.get("provider", "ollama")
    max_history = config.get("max_history_limit", 10)

    # 1. Setup LLMs
    intent_fast_llm, intent_slow_llm = get_fast_slow_llm(provider)
    clarification_llm = make_llm(provider, "slow")

    # 2. Instantiate Nodes
    intent_node = make_intent_node(
        fast_llm=intent_fast_llm,
        slow_llm=intent_slow_llm if config.get("slow_fallback_enabled", True) else None,
        slow_fallback_enabled=config.get("slow_fallback_enabled", True),
        slow_fallback_min_confidence=config.get("slow_fallback_min_confidence", 0.9),
        max_history_limit=max_history,
    )

    clarification_node = make_clarification_node(clarification_llm)
    static_response_node = make_static_response_node()
    process_node = build_process_graph(config)

    # 3. Router Logic: Intent Node -> Action
    def intent_router(
        state: AssistantState,
    ) -> Literal["clarification", "process", "static_respond"]:
        metadata = state.get("intent_metadata")

        if not metadata:
            return "clarification"

        # Check for low confidence or explicit clarification flag
        if metadata.needs_clarification or metadata.confidence < 0.75:
            return "clarification"

        # Route to static response for non-data intents
        if metadata.intent in ["GREETING", "OUT_OF_SCOPE", "DATA_AVAILABILITY"]:
            return "static_respond"

        # Default to data processing
        return "process"

    # # 4. Process Router Logic: Data Agent -> End or Clarification Loop
    def process_router(state: AssistantState) -> Literal["clarification", "end"]:
        # If the Data Agent failed or requests clarification, loop back
        if state.get("needs_clarification"):
            return "clarification"
        return "end"

    # 5. Build Graph
    workflow = StateGraph(AssistantState)

    workflow.add_node("INTENT", intent_node)
    workflow.add_node("CLARIFICATION", clarification_node)
    workflow.add_node("STATIC_RESPOND", static_response_node)
    workflow.add_node("PROCESS", process_node)

    workflow.set_entry_point("INTENT")

    # Edge: Intent -> Next Step
    workflow.add_conditional_edges(
        "INTENT",
        intent_router,
        {
            "clarification": "CLARIFICATION",
            "static_respond": "STATIC_RESPOND",
            "process": "PROCESS",
        },
    )

    # # Edge: Process -> End or Clarification (Loop Closure)
    workflow.add_conditional_edges(
        "PROCESS", process_router, {"clarification": "CLARIFICATION", "end": END}
    )
    # workflow.add_edge("PROCESS", END)
    workflow.add_edge("CLARIFICATION", END)
    workflow.add_edge("STATIC_RESPOND", END)

    # Initialize Memory
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)
