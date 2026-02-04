"""
graph/process/rag_retriever/chains/retriever.py
Factory for creating the specific Vector Store Retriever.
"""

from typing import Any, Dict

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from graph.consts import KB_NAME, KB_PATH


def make_retriever(config: Dict[str, Any]):
    """
    Factory to build the vector store retriever based on configuration.

    Args:
        config (Dict): Configuration dictionary containing:
            - embeddings: { "provider": str, "model": str, "base_url": str }
            - retriever_k: int
            - score_threshold: float (optional)

    Returns:
        VectorStoreRetriever: A runnable retriever object.
    """

    # ---------------------------------------------------------
    # 1. Configure Embeddings
    # ---------------------------------------------------------
    embed_cfg = config.get("embeddings", {})
    provider = embed_cfg.get("provider", "ollama")
    model_name = embed_cfg.get("model", "mxbai-embed-large")

    if provider == "openai":
        embeddings = OpenAIEmbeddings(model=model_name)
    else:
        # Default to Ollama
        base_url = embed_cfg.get("base_url", "http://localhost:11434")
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)

    # ---------------------------------------------------------
    # 2. Initialize Vector Store
    # ---------------------------------------------------------
    # You can allow the config to override the DB path if needed,
    # but defaulting to the constant is usually safe for this app.
    persist_dir = config.get("db_path", str(KB_PATH))

    vectorstore = Chroma(
        collection_name=KB_NAME,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # ---------------------------------------------------------
    # 3. Create Retriever
    # ---------------------------------------------------------
    k = config.get("retriever_k", 5)
    score_threshold = config.get("score_threshold", 0.0)

    # Determine search type based on threshold availability
    if score_threshold > 0:
        search_type = "similarity_score_threshold"
        search_kwargs = {"k": k, "score_threshold": score_threshold}
    else:
        search_type = "similarity"
        search_kwargs = {"k": k}

    return vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
