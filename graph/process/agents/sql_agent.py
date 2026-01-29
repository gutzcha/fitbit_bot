"""
graph/process/agents/sql_agent.py

Factory for building the SQL Data Agent using SQLDatabaseToolkit.
"""

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from graph.consts import DB_URI
from graph.data_config import DB_SCHEMA_CONTEXT
from graph.helpers import make_llm
from graph.process.prompts.sql_agent import SQL_AGENT_PROMPT
from graph.process.schemas import SQLAgentResponse

# Singleton DB connection
_DB_INSTANCE = None


def get_db():
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = SQLDatabase.from_uri(DB_URI, view_support=False)
    return _DB_INSTANCE


def build_sql_agent(provider: str = "ollama", model_type: str = "slow"):
    """
    Constructs a Data Agent.
    User Profile is NO LONGER injected here; it must be passed in the context.
    """
    # 1. Setup Resources
    db = get_db()
    llm = make_llm(provider, model_type)

    # 2. Prepare Tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # 3. Prepare System Prompt
    # We only inject the Schema Context here.
    # The Profile will be provided by the Execution Node in the message history.
    final_system_prompt = SQL_AGENT_PROMPT.format(
        schema_context=DB_SCHEMA_CONTEXT,
    )

    # 4. Create Agent
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=final_system_prompt,
        response_format=SQLAgentResponse,
    )