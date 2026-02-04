"""
graph/process/rag_retriever/rag_graph.py

RAG SUBGRAPH APP
================
Builds and returns the compiled RAG retriever subgraph using provided config.
"""

from typing import Any, Dict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.process.rag_retriever.const import (
    GENERATE,
    GRADE_DOCUMENTS,
    NO_DATA_AVAILABLE,
    RETRIEVE,
)
from graph.process.rag_retriever.nodes import (
    make_generate_node,
    make_grade_documents_node,
    make_no_available_data_node,
    make_retrieve_node,
)
from graph.process.rag_retriever.rag_state import GraphState

load_dotenv()


def decide_to_generate(state: GraphState) -> str:
    if state.get("no_data_available"):
        return NO_DATA_AVAILABLE
    return GENERATE


def build_rag_app(rag_config: Dict[str, Any]):
    """
    Builds the RAG Subgraph.

    Args:
        rag_config: Dictionary containing configs for 'retriever', 'grade_documents', and 'generate'.
    """

    # 1. Initialize Nodes with specific configs
    retrieve_node = make_retrieve_node(rag_config.get("retriever", {}))
    grade_node = make_grade_documents_node(rag_config.get("grade_documents", {}))
    # Assuming 'rewriter' config or similar is used for generation, or a dedicated 'generator' key
    generate_node = make_generate_node(rag_config.get("generate", {}))
    no_data_node = make_no_available_data_node()

    # 2. Build Graph
    workflow = StateGraph(GraphState)

    workflow.add_node(RETRIEVE, retrieve_node)
    workflow.add_node(GRADE_DOCUMENTS, grade_node)
    workflow.add_node(GENERATE, generate_node)
    workflow.add_node(NO_DATA_AVAILABLE, no_data_node)

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