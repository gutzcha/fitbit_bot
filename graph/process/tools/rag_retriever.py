# graph/process/tools/rag_retriever.py

from typing import Tuple, Dict, Any
from langchain_core.tools import tool, convert_runnable_to_tool
from graph.process.rag_retrievaer.rag_graph import build_rag_app
from graph.process.rag_retrievaer.rag_state import GraphState

# Build the graph once to avoid overhead
rag_app = build_rag_app()

@tool(response_format="content_and_artifact")
def fetch_knowledge_base(question: str) -> Tuple[str, Dict[str, Any]]:
    """
    Search the Knowledge Base for general health, science, and behavioral information.

    Args:
        question (str): The scientific or health-related question to ask.

    Returns:
        Tuple[str, Dict]:
            - The natural language summary (visible to Agent).
            - The dictionary containing source documents (visible to Validator).
    """
    # 1. Invoke RAG
    result = rag_app.invoke({
        "question": question,
        "generation": "",
        "documents": [],
        "no_data_available": False
    })

    # 2. Prepare the "Artifact"
    artifact = {
        "sources": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in result.get("documents", [])
        ],
        "full_generation": result.get("generation"),
        "raw_status": "no_data" if result.get("no_data_available") else "success"
    }

    # 3. Prepare the "Content" (Simple Text for Agent)
    if result.get("no_data_available"):
        content = "No relevant scientific data found in the knowledge base."
    else:
        content = result.get("generation") or "Data retrieved successfully."

    # Optional: Print to verify during debugging
    # print(f"--artifact--: {artifact}")

    # 4. Return Tuple (Content, Artifact)
    return content, artifact