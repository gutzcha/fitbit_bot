# graph/process/rag_retriever/chains/grader.py

from typing import Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    # User requested 'true' or 'false' (string) instead of 'yes'/'no'
    binary_score: str = Field(
        description="Documents are relevant to the question, 'true' or 'false'"
    )


def make_grader_chain(config: Dict[str, Any]):
    """
    Factory to build the Retrieval Grader Chain.

    Args:
        config (Dict): Configuration dictionary. Expects an 'llm' key.
                       Example: { "llm": { "model": "ollama:llama3", "temperature": 0 } }
    """

    # 1. Initialize LLM
    llm_config = config.get("llm", {})
    llm = init_chat_model(**llm_config)

    # 2. Configure Structured Output
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # 3. Create Prompt
    # Updated to ask for 'true' or 'false'
    system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'true' or 'false' to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    # 4. Build Chain
    grader_chain = grade_prompt | structured_llm_grader

    return grader_chain
