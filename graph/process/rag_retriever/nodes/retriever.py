"""
graph.process.rag_retriever.nodes.retriever
"""
from typing import Any, Dict, Callable

from graph.process.rag_retriever.rag_state import GraphState
from graph.process.rag_retriever.chains.retriever import make_retriever


def make_retrieve_node(config: Dict[str, Any]) -> Callable[[GraphState], Dict[str, Any]]:
    """
    Factory for the RETRIEVE node.
    """
    # Initialize retriever using config (e.g., k, score_threshold)
    retriever = make_retriever(config)

    def retriever_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        documents = retriever.invoke(question)

        # Check if retrieval failed entirely
        if not documents:
            return {"documents": [], "no_data_available": True}

        return {"documents": documents, "question": question}

    return retriever_node