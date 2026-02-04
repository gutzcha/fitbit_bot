"""
graph/process/agents/sql_agent.py

Factory for building the SQL Data Agent using SQLDatabaseToolkit.
"""

from typing import Any, Dict

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model

from dataset.dataset_config import SQL_SCHEMA
from graph.process.schemas import SQLAgentResponse
from graph.consts import DB_URI
from graph.helpers import get_current_date
# Singleton DB connection
_DB_INSTANCE = None



def get_db(db_uri: str):
    """
    Singleton accessor for the SQL Database.
    Initializes the connection only once using the provided URI.
    """
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = SQLDatabase.from_uri(db_uri, view_support=False)
    return _DB_INSTANCE


def build_sql_agent(agent_config: Dict[str, Any], validation_config: Dict[str, Any]):
    """
    Constructs the SQL Agent using the modern 'create_agent' factory.

    Args:
        agent_config: Configuration for the generation model and DB connection.
        validation_config: Configuration for validation (optional).
    """

    # 1. Setup Resources
    llm_config = agent_config.get("llm", {})
    llm = init_chat_model(**llm_config)


    db_uri = agent_config.get("db_uri", DB_URI)
    db = get_db(db_uri)

    # 2. Prepare Tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # 3. System Prompt
    system_prompt = f"""You are an expert SQLite data analyst. 
    Your role is to query the database to answer user questions strictly based on the provided data.

    ### 1. Database Schema Context
    Use this knowledge to understand table relationships and column meanings:
    {SQL_SCHEMA}

    ### 2. Operational Rules
    - **READ ONLY**: Never execute INSERT, UPDATE, DELETE, or DROP. Only use SELECT.
    - **Limit Results**: Always LIMIT your results to the top 5 or 10 rows unless explicitly asked for all.
    - **Current Date**: If the query involves "today", "yesterday", or "recent", use date({get_current_date()}) function in SQLite. Never use data('now')
    
    ### 3. Response Format
    You must output your final answer strictly in the structured JSON format provided (SQLAgentResponse).
    """

    # 4. Create Agent
    # uses langchain.agents.create_agent
    agent_runnable = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        response_format=SQLAgentResponse,
    )

    return agent_runnable