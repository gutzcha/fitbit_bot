"""
graph/process/agents/execution.py

Refactored execution agent with proper context injection using LangChain v1.0+ create_agent.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from graph.process.prompts.execution import EXECUTION_SYSTEM_PROMPT
from graph.process.schemas import ExecutionResponse
from graph.process.tools import make_rag_tool, make_sql_tool


def build_execution_agent(
    manager_llm: BaseChatModel,
    sql_config: Dict[str, Any],
    sql_validation_config: Dict[str, Any],
    rag_config: Dict[str, Any],
) -> Runnable:
    """
    Build an execution agent using LangChain v1.0+ create_agent.

    The agent receives messages enriched with context system messages.
    Returns ExecutionResponse via structured output.

    Key changes from deprecated create_react_agent:
    - Uses create_agent (LangChain v1.0+)
    - Context injected as system messages in the message list
    - Structured output via response_format parameter
    - No need for state_modifier or prompt templates
    """

    # Build tools
    sql_tool = make_sql_tool(
        agent_config=sql_config,
        validation_config=sql_validation_config,
    )

    rag_tool = make_rag_tool(
        config=rag_config,
    )

    tools = [sql_tool, rag_tool]

    # create_agent handles:
    # - Tool binding to model
    # - ReAct loop execution
    # - Structured output via response_format
    # - Message history management
    agent = create_agent(
        model=manager_llm,
        tools=tools,
        system_prompt=EXECUTION_SYSTEM_PROMPT,
        response_format=ExecutionResponse,
    )

    return agent
