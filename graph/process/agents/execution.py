# graph/process/agents/execution.py

from langchain.agents import create_agent  # Your custom factory
from graph.helpers import make_llm
from graph.process.prompts.execution import EXECUTION_SYSTEM_PROMPT
from graph.process.schemas import ExecutionResponse
from graph.process.tools import fetch_knowledge_base, fetch_user_metrics_sql



def build_execution_agent(provider: str = "ollama", model_type: str = "slow"):
    """
    Builds the Execution Agent Runnable with Structured Output enforcement.
    """
    # 1. Setup LLM
    llm = make_llm(provider=provider, model_type=model_type)

    # 2. Setup Tools
    tools = [fetch_knowledge_base, fetch_user_metrics_sql]

    # 3. Create Agent
    # We pass 'ExecutionResponse' to response_format to force JSON output
    agent_runnable = create_agent(
        model=llm,
        tools=tools,
        system_prompt=EXECUTION_SYSTEM_PROMPT,
        response_format=ExecutionResponse
    )

    return agent_runnable