# graph/process/tools/sql_metrics.py

import json
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool, tool

from graph.helpers import get_current_date
# Assuming you have refactored this builder to accept configs as well
from graph.process.agents.sql_agent import build_sql_agent


def make_sql_tool(agent_config: Dict[str, Any], validation_config: Dict[str, Any]):
    """
    Factory that creates a configured SQL Query tool.

    Args:
        agent_config (Dict): Config for the SQL Generation Agent (LLM, temperature, etc).
                             Also contains 'user_id' if defined in config.json.
        validation_config (Dict): Config for the SQL Validation step (retries, fixer LLM).

    Returns:
        Tool: A LangChain Tool instance ready for the agent to use.
    """

    # 1. Extract Context from Config
    # We allow the user_id to be configured in the JSON, or fallback to the known ID.
    # This allows you to switch users just by changing config.json.
    user_id = agent_config.get("user_id", 1503960366)

    # 2. Build the Agent Instance (ONCE)
    # We build the agent here so we don't re-initialize the LLM on every tool call.
    agent = build_sql_agent(
        agent_config=agent_config, validation_config=validation_config
    )

    # 3. Define the tool function (Closure)
    @tool
    def fetch_user_metrics_sql(query: str) -> str:
        """
        Query the user's SQL database for personal health metrics and profile data.

        Use this tool when the user asks about their specific data, such as:
        - "How many steps did I take yesterday?"
        - "What is my average heart rate?"
        - "Did I lose weight last week?"

        Args:
            query (str): A natural language question specifically about user data.

        Returns:
            str: A natural language answer derived from the database, or a failure message.
        """
        current_date = get_current_date()

        # 4. Construct the Contextualized Prompt
        # We inject the User ID and Date into the prompt for the SQL Agent
        contextualized_prompt = (
            f"--- SYSTEM CONTEXT ---\n"
            f"User ID: {user_id}\n"
            f"Current Date: {current_date}\n\n"
            f"--- TOOL QUERY ---\n"
            f"{query}\n\n"
            f"--- INSTRUCTIONS ---\n"
            f"1. Query the database for the specific metrics requested.\n"
            f"2. Return the final answer in natural language based on the SQL results.\n"
            f"3. If no data is found, explicitly state that."
        )

        # 5. Invoke the Agent
        try:
            # The agent expects a dictionary with 'messages'
            result = agent.invoke(
                {"messages": [HumanMessage(content=contextualized_prompt)]}
            )

            messages = result.get("messages", [])
            last_ai = next(
                (m for m in reversed(messages) if isinstance(m, AIMessage)), None
            )

            if last_ai and isinstance(last_ai.content, str) and last_ai.content.strip():
                text = last_ai.content.strip()

                # Try parse as JSON and extract "answer"
                try:
                    payload = json.loads(text)
                    if isinstance(payload, dict):
                        answer = payload.get("answer")
                        if isinstance(answer, str) and answer.strip():
                            return answer.strip()
                except Exception:
                    pass

                # Fallback, return raw text
                return text

            return "I queried the database but could not retrieve a structured answer."

        except Exception as e:
            return f"Error executing SQL tool: {str(e)}"

    return fetch_user_metrics_sql
