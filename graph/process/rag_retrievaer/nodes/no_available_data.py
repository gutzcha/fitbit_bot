from typing import Any, Dict

from graph.process.rag_retrievaer.rag_state import GraphState


def no_available_data(state: GraphState) -> Dict[str, Any]:
    print("---NO DATA WAS AVAILABLE---")
    question = state["question"]
    static_response = """
    No data was available regarding your query, please try to rephrase and ask again
    """
    return {"documents": [], "question": question, "generation": static_response}