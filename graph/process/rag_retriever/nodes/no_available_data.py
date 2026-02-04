# graph/process/rag_retriever/nodes/no_data.py

from typing import Any, Callable, Dict

from graph.process.rag_retriever.rag_state import GraphState


def make_no_available_data_node() -> Callable[[GraphState], Dict[str, Any]]:
    """
    Factory for the NO_DATA_AVAILABLE node.
    """

    def no_available_data(state: GraphState) -> Dict[str, Any]:
        return {
            "generation": "Sorry, I could not find any relevant information in the knowledge base to answer your specific question.",
            "no_data_available": True,
        }

    return no_available_data
