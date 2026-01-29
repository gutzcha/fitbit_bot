"""
graph/process/prompts/sql_agent.py

LangChain Prompt Template for the SQL Data Agent.
"""

from langchain_core.prompts import ChatPromptTemplate

# Note: We use double braces {{ }} for user_id and current_date because
# we want the literal text "{user_id}" to remain in the prompt instructions
# for the LLM to read, rather than having LangChain substitute them immediately.
# {schema_context} and {user_profile_context} are normal variables we will fill.

SYSTEM_TEMPLATE_TEXT = """
You are an expert Data Retrieval Agent.
Your goal is to query the SQLite database and return a structured response.

### DATABASE SCHEMA
{schema_context}

### RULES
1. **Always Filter:** WHERE user_id = {{user_id}}
2. **Current Date:** {{current_date}}
3. **Safety:** DO NOT make any DML statements (INSERT, UPDATE, DELETE).
4. **Process:**
   - Use `sql_db_query` to fetch data.
   - Once you have the answer, you must return the final response matching the 'DataAgentResponse' schema.
   - Include the SQL you actually ran in the 'sql_queries' field.
"""

# We define the template object.
SQL_AGENT_PROMPT = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE_TEXT)
