"""
graph/process/tools/sql_metrics.py
"""
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from graph.consts import CHAT_CONFIG_PATH
from graph.helpers import get_current_date, load_config
from graph.process.agents.sql_agent import build_sql_agent

# Load configuration to get user_id and provider
# In a real deployment, you might want to inject this context dynamically,
# but for a standalone tool, we load the global config.
try:
    config = load_config(CHAT_CONFIG_PATH)
except Exception:
    logging.warning("Could not load config from CHAT_CONFIG_PATH, using defaults.")
    config = {}

USER_ID = config.get("user_id", 1503960366)
PROVIDER = config.get("provider", "ollama")


@tool
def fetch_user_metrics_sql(query: str) -> str:
    """
    Query the user's SQL database for personal health metrics and profile data.

    Use this tool when the user asks about their specific data, such as:
    - "How many steps did I take yesterday?"
    - "What is my average heart rate?"
    - "What is my daily calorie goal?"
    - "Did I lose weight last week?"

    Args:
        query (str): A natural language question specifically about user data.

    Returns:
        str: A natural language answer derived from the database, or a failure message.
    """
    current_date = get_current_date()

    # 1. Build the Agent
    # We use 'slow' model_type to ensure high-quality SQL generation
    agent = build_sql_agent(provider=PROVIDER, model_type="slow")

    # 2. Construct the Contextualized Prompt
    # This enforces the agent to look at the specific User ID and Date
    contextualized_prompt = (
        f"--- SYSTEM CONTEXT ---\n"
        f"User ID: {USER_ID}\n"
        f"Current Date: {current_date}\n\n"
        f"--- TOOL QUERY ---\n"
        f"{query}\n\n"
        f"--- INSTRUCTIONS ---\n"
        f"1. Query the database for the specific metrics requested.\n"
        f"2. Return the final answer in natural language based on the SQL results.\n"
        f"3. If no data is found, explicitly state that."
    )

    # 3. Invoke the Agent
    try:
        # The agent expects a list of messages
        result = agent.invoke({"messages": [HumanMessage(content=contextualized_prompt)]})

        # 4. Extract Response
        # The agent returns a DataAgentResponse object in the 'structured_response' field
        structured_data = result.get("structured_response")

        if structured_data and structured_data.answer:
            return structured_data.answer

        # Fallback if the structured output parsing failed but text exists
        if "output" in result:
            # Some agents might return the raw string in 'output'
            return str(result["output"])

        return "I queried the database but could not retrieve a structured answer."

    except Exception as e:
        return f"Error executing SQL tool: {str(e)}"