"""
graph/chains/request_clarification.py

Chain responsible for generating clarifying questions.
Uses the pre-compiled prompt defined in graph/prompts/clarification.py
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ✅ Import the ready-to-use Template object
from graph.prompts.request_clarification import CLARIFICATION_PROMPT


def build_clarification_chain(llm: BaseChatModel) -> RunnableLambda:
    """
    Builds a chain that takes 'user_input' and returns a clarification question.
    """

    # 1. Construct Chain
    # The prompt already has the schema injected, so we just pipe it.
    chain = CLARIFICATION_PROMPT | llm | StrOutputParser()

    return chain
