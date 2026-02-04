"""
graph/graph.py

Main runtime graph.

Flow:
INTENT -> (CLARIFICATION | STATIC_RESPOND | PROCESS)
PROCESS -> (CLARIFICATION | SUGGESTOR)
SUGGESTOR -> END
CLARIFICATION -> END
STATIC_RESPOND -> END

Notes:
- Only one clarification node exists, in the main graph
- Suggestor is in the main graph
- PROCESS is a single node (not a subgraph)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.config_loader import load_graph_config
from graph.nodes import (
    make_clarification_node,
    make_intent_node,
    make_static_response_node,
    make_data_availability_node
)
from graph.nodes.suggestor import make_suggestor_node
from graph.process.process import make_process_node
from graph.state import AssistantState

load_dotenv()


def build_graph(config: Optional[dict] = None):
    config_dict: Dict = config if isinstance(config, dict) else load_graph_config()
    runtime_dict: Dict = config_dict.get("runtime_nodes", {})

    intent_node = make_intent_node(runtime_dict.get("graph.nodes.intent", {}))
    clarification_node = make_clarification_node(runtime_dict.get("graph.nodes.request_clarification", {}))
    data_availability_node = make_data_availability_node(runtime_dict.get("graph.nodes.data_availability", {}))
    static_response_node = make_static_response_node(runtime_dict.get("graph.nodes.static_response", {}))

    process_node = make_process_node(config_dict)

    suggestor_cfg = runtime_dict.get("graph.process.nodes.suggestor", {})
    suggestor_node = make_suggestor_node(suggestor_cfg)
    suggestor_enabled = bool(suggestor_cfg.get("enabled", True))

    def intent_router(state: AssistantState) -> Literal["clarification", "process", "static_respond", "data_availability"]:
        metadata = state.get("intent_metadata")
        if not metadata:
            return "clarification"

        needs_clar = getattr(metadata, "needs_clarification", False)
        confidence = getattr(metadata, "confidence", 0.0)
        intent = getattr(metadata, "intent", None)

        if isinstance(metadata, dict):
            needs_clar = metadata.get("needs_clarification", needs_clar)
            confidence = metadata.get("confidence", confidence)
            intent = metadata.get("intent", intent)

        if needs_clar or float(confidence) < 0.75:
            return "clarification"

        if intent in ["GREETING", "OUT_OF_SCOPE"]:
            return "static_respond"

        if intent == "DATA_AVAILABILITY":
            return "data_availability"

        return "process"

    def after_process_router(state: AssistantState) -> Literal["clarification", "suggest", "end"]:
        if state.get("needs_clarification"):
            return "clarification"
        if suggestor_enabled:
            return "suggest"
        return "end"

    workflow = StateGraph(AssistantState)

    workflow.add_node("INTENT", intent_node)
    workflow.add_node("CLARIFICATION", clarification_node)
    workflow.add_node("STATIC_RESPOND", static_response_node)
    workflow.add_node("PROCESS", process_node)
    workflow.add_node("DATA_AVAILABILITY", data_availability_node)

    if suggestor_enabled:
        workflow.add_node("SUGGESTOR", suggestor_node)

    workflow.set_entry_point("INTENT")

    workflow.add_conditional_edges(
        "INTENT",
        intent_router,
        {
            "clarification": "CLARIFICATION",
            "static_respond": "STATIC_RESPOND",
            "data_availability": "DATA_AVAILABILITY",
            "process": "PROCESS",
        },
    )

    if suggestor_enabled:
        workflow.add_conditional_edges(
            "PROCESS",
            after_process_router,
            {
                "clarification": "CLARIFICATION",
                "suggest": "SUGGESTOR",
                "end": END,
            },
        )
        workflow.add_edge("SUGGESTOR", END)
    else:
        workflow.add_conditional_edges(
            "PROCESS",
            after_process_router,
            {
                "clarification": "CLARIFICATION",
                "end": END,
                "suggest": END,
            },
        )

    workflow.add_edge("CLARIFICATION", END)
    workflow.add_edge("STATIC_RESPOND", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
