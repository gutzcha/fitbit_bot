# graph/process/tools/rag_retriever.py

from typing import Any, Dict, Tuple

from langchain_core.tools import Tool, tool

from graph.process.rag_retriever.rag_graph import build_rag_app


def make_rag_tool(config: Dict[str, Any]):
    """
    Factory that creates a configured RAG tool.

    Args:
        config (Dict): The configuration dictionary containing 'retriever', 'grade_documents', etc.
                       This is passed down from the Execution Agent.

    Returns:
        Tool: A LangChain Tool instance ready for the agent to use.
    """

    # 1. Build the specific RAG app instance for this tool
    # This ensures the tool uses the specific models/settings defined in config.json
    rag_app = build_rag_app(rag_config=config)

    # 2. Define the tool function (Closure)
    # The @tool decorator is applied here so it captures the specific 'rag_app' instance
    @tool(response_format="content_and_artifact")
    def fetch_knowledge_base(question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Search the Knowledge Base for general health, science, and behavioral information.

        Args:
            question (str): The scientific or health-related question to ask.

        Returns:
            Tuple[str, Dict]:
                - The natural language summary (visible to Agent).
                - The dictionary containing source documents (visible to Validator/Artifacts).
        """
        # A. Invoke RAG
        # We initialize the state with defaults
        result = rag_app.invoke(
            {
                "question": question,
                "generation": "",
                "documents": [],
                "no_data_available": False,
            }
        )

        # B. Prepare the "Artifact" (Hidden from LLM context, used for citation/validation)
        artifact = {
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("documents", [])
            ],
            "full_generation": result.get("generation"),
            "raw_status": "no_data" if result.get("no_data_available") else "success",
        }

        # C. Prepare the "Content" (Visible to the Agent)
        if result.get("no_data_available"):
            content = "No relevant scientific data found in the knowledge base."
        else:
            # Return the generated answer from the RAG pipeline
            content = result.get("generation") or "Data retrieved successfully."

        return content, artifact

    # 3. Return the configured tool
    return fetch_knowledge_base