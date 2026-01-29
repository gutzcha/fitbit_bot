# graph/process/rag_retrievaer/rag_graph.py
"""
RAG SUBGRAPH APP
================

Builds and returns the compiled RAG retriever subgraph.
"""
from typing import Dict
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.process.rag_retrievaer.const import (
    RETRIEVE,
    GRADE_DOCUMENTS,
    GENERATE,
    NO_DATA_AVAILABLE,
)
from graph.process.rag_retrievaer.nodes import generate, grade_documents, retrieve, no_available_data
from graph.process.rag_retrievaer.rag_state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    if state["no_data_available"]:
        return NO_DATA_AVAILABLE
    return GENERATE


def build_rag_app():
    workflow = StateGraph(GraphState)

    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(NO_DATA_AVAILABLE, no_available_data)

    workflow.set_entry_point(RETRIEVE)
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {NO_DATA_AVAILABLE: NO_DATA_AVAILABLE, GENERATE: GENERATE},
    )
    workflow.add_edge(NO_DATA_AVAILABLE, END)
    workflow.add_edge(GENERATE, END)

    return workflow.compile()
