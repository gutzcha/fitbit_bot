from typing import Any, Dict

from graph.process.rag_retrievaer.rag_state import GraphState
from graph.process.rag_retrievaer.retriever import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}