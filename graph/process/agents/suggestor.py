"""
graph/process/agents/suggestor.py

Factory for building the Suggestor (Coaching) Agent.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.helpers import make_llm
from graph.process.prompts.suggestor import SUGGESTOR_SYSTEM_PROMPT


def build_suggestor_agent(provider: str = "anthropic", model_type: str = "fast"):
    """
    Builds the Suggestor Agent chain.

    Args:
        provider: LLM provider (e.g., 'anthropic', 'ollama').
        model_type: 'fast' is recommended for this lightweight task.

    Returns:
        Runnable: A Chain (Prompt -> LLM -> StrParser).
    """
    # 1. Setup LLM
    llm = make_llm(provider=provider, model_type=model_type)

    # 2. Setup Prompt
    # We expect 'tone' in the system vars, and 'input_context' from the human
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUGGESTOR_SYSTEM_PROMPT),
        ("human", "{input_context}"),
    ])

    # 3. Create Chain
    # We use StrOutputParser because we just want the raw suggestion text
    chain = prompt | llm | StrOutputParser()

    return chain