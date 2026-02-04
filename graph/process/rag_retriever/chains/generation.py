# graph/process/rag_retrievaer/chains/generation.py

from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client


def make_generation_chain(config: Dict[str, Any]):
    client = Client(cache=True)

    """
    Factory to build the RAG Generation Chain.

    Args:
        config (Dict): Configuration dictionary. Expects an 'llm' key.
                       Example: { "llm": { "model": "ollama:llama3", "temperature": 0 } }
    """

    # 1. Initialize LLM
    # Extract the 'llm' block from the config passed by the node factory
    llm_config = config.get("llm", {})

    # Initialize the specific model for generation
    # (e.g., you might want a more capable/slower model here)
    llm = init_chat_model(**llm_config)

    # 2. Get Prompt
    # Default to the standard RAG prompt, but allow override via config
    prompt_repo = config.get("prompt_repo", "rlm/rag-prompt")
    prompt = client.pull_prompt(prompt_repo)

    # 3. Build Chain
    generation_chain = prompt | llm | StrOutputParser()

    return generation_chain
