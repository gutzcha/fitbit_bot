# graph/process/process.py
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from graph.memory import trim_conversation_history
from graph.process.agents.execution import build_execution_agent
from graph.process.schemas import ExecutionResponse
from graph.state import AssistantState
from graph.helpers import extract_json_from_markdown, build_context_messages


def extract_execution_response_from_messages(messages: list[BaseMessage]) -> ExecutionResponse:
    """
    Extract ExecutionResponse from the message list.

    This handles cases where models output JSON text instead of using tool calling.

    Args:
        messages: List of messages from the agent

    Returns:
        ExecutionResponse object

    Raises:
        ValueError: If no valid ExecutionResponse found
    """
    # Look through messages in reverse (most recent first)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content or ""

            # Skip empty messages
            if not content.strip():
                continue

            # Try to extract JSON from the content
            try:
                json_data = extract_json_from_markdown(content)

                # Validate it has the required fields
                if "answer" in json_data and "confidence" in json_data:
                    return ExecutionResponse.model_validate(json_data)


            except (ValueError, Exception):
                return ExecutionResponse(
                    answer=json.dumps(content),
                    confidence=0.0,
                    needs_clarification=True,
                    clarification_question="Could you rephrase your question?"
                )

                continue

    raise ValueError("Could not find valid ExecutionResponse in messages")


def make_process_node(full_config: Dict[str, Any]) -> Callable[[AssistantState], Dict[str, Any]]:
    """
    PROCESS is a single node.
    It owns LLM + agent construction, runs the execution agent, and writes the final answer
    back into AssistantState so downstream nodes can see it.

    Key improvements:
    - Context (user_profile, intent_metadata, conversation_state) injected as SystemMessages
    - Handles both proper structured output AND JSON text fallback
    - Agent receives proper message list with all context visible
    """

    runtime_nodes = full_config.get("runtime_nodes", {})
    exec_cfg = runtime_nodes.get("graph.process.nodes.execution", {}) or {}

    llm_manager_cfg = exec_cfg.get("llm_manager", {}) or {}
    manager_llm = init_chat_model(**llm_manager_cfg)

    max_iterations = int(exec_cfg.get("max_iterations", 5))
    max_history_limit = int(exec_cfg.get("max_history_limit", 20))

    sql_ref = exec_cfg.get("sql_config_ref", "graph.process.nodes.sql_agent")
    sql_validation_ref = exec_cfg.get(
        "sql_validation_config_ref", "graph.process.nodes.sql_validation"
    )
    rag_ref = exec_cfg.get("rag_config_ref", "graph.process.rag_retriever")

    sql_cfg = runtime_nodes.get(sql_ref, {}) or {}
    sql_validation_cfg = runtime_nodes.get(sql_validation_ref, {}) or {}
    rag_cfg = runtime_nodes.get(rag_ref, {}) or {}

    agent = build_execution_agent(
        manager_llm=manager_llm,
        sql_config=sql_cfg,
        sql_validation_config=sql_validation_cfg,
        rag_config=rag_cfg,
    )

    def process_node(state: AssistantState) -> Dict[str, Any]:
        messages = state.get("messages", []) or []
        if not messages:
            return {}

        # Trim conversation history to stay within context limits
        trimmed_messages = trim_conversation_history(
            messages, max_messages=max_history_limit
        )

        # Build complete message list with context injected as system messages
        messages_with_context = build_context_messages(
            trimmed_messages=trimmed_messages,
            conversation_state=state.get("conversation_state"),
            intent_metadata=state.get("intent_metadata"),
            user_profile=state.get("user_profile"),
        )

        # Invoke agent with the enriched message list
        # create_agent expects: {"messages": [...]}
        agent_input = {"messages": messages_with_context}

        # Agent may return different formats depending on model capabilities:
        # 1. Proper tool calling: {'messages': [...], 'structured_response': ExecutionResponse(...)}
        # 2. JSON text fallback: {'messages': [...AIMessage(content='```json\n{...}\n```')...]}
        raw_result = agent.invoke(
            agent_input,
            config={"recursion_limit": max_iterations},
        )
        messages_list = raw_result.get("messages", []) if isinstance(raw_result, dict) else []

        # Extract ExecutionResponse - try multiple methods
        exec_result = None

        # Method 1: Check for 'structured_response' key (proper tool calling)
        if isinstance(raw_result, dict) and "structured_response" in raw_result:
            exec_result = raw_result.get("structured_response")

            if not isinstance(exec_result, ExecutionResponse):
                # Try to validate from dict
                if isinstance(exec_result, dict):
                    try:
                        exec_result = ExecutionResponse.model_validate(exec_result)
                    except Exception as e:
                        print(f"Warning: Could not validate structured_response: {e}")
                        exec_result = None

        # Method 2: Direct ExecutionResponse return (rare)
        if exec_result is None and isinstance(raw_result, ExecutionResponse):
            exec_result = raw_result

        # Method 3: Parse JSON from message content (fallback for models without tool calling)
        if exec_result is None:
            if messages_list:
                try:
                    exec_result = extract_execution_response_from_messages(messages_list)
                except ValueError as e:
                    print(f"Warning: Could not extract ExecutionResponse from messages: {e}")

        # Final fallback: construct a basic response
        if exec_result is None:
            print("Warning: Using fallback ExecutionResponse")

            exec_result = ExecutionResponse(
                answer="I processed your request but encountered an issue generating a structured response.",
                confidence=0.0,
                needs_clarification=True,
                clarification_question="Could you rephrase your question?"
            )

        # Extract the answer
        answer_text = (exec_result.answer or "").strip()
        if not answer_text:
            answer_text = "I ran the process but did not receive an answer."

        # Update messages with the assistant's response
        # Note: Only append to the original trimmed_messages, not the context-enriched ones
        updated_messages = list(trimmed_messages) + [AIMessage(content=answer_text)]

        # Build the update dict for AssistantState
        update: Dict[str, Any] = {
            "messages": updated_messages,
            "response": answer_text,
            "execution_result": exec_result,
            "needs_clarification": bool(exec_result.needs_clarification),
        }

        if exec_result.needs_clarification and exec_result.clarification_question:
            update["clarification_question"] = exec_result.clarification_question

        return update

    return process_node