# graph/process/rag_retrievaer/nodes/generation.py

from typing import Any, Callable, Dict

from graph.process.rag_retriever.chains.generation import make_generation_chain
from graph.process.rag_retriever.rag_state import GraphState


def make_generate_node(
    config: Dict[str, Any],
) -> Callable[[GraphState], Dict[str, Any]]:
    """
    Factory for the GENERATE node.
    """
    # 1. Initialize the chain using the factory
    # This injects the specific LLM/Prompt config
    generation_chain = make_generation_chain(config)

    def generate(state: GraphState) -> Dict[str, Any]:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # 2. Format Context
        # Standard RAG prompts expect a single string for context, not a list of objects.
        # We join the page_content of the valid docs.
        context_str = "\n\n".join([doc.page_content for doc in documents])

        # 3. Invoke Chain
        generation = generation_chain.invoke(
            {"context": context_str, "question": question}
        )

        return {"documents": documents, "question": question, "generation": generation}

    return generate
